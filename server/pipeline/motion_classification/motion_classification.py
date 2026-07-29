from logging import Logger
from typing import Any, Optional

import numpy as np

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import VEGETATION_CATEGORIES
from pipeline.scene_generation.projection import unproject_bbox, unproject_bbox_equirect


class ObjectMotionClassificationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        stationary_drift_ratio: float = 0.6,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # An object whose centroid drifts less than this fraction of its own
        # (video-pixel) bbox diagonal over the whole clip is treated as stationary
        # (oscillating in place -- wind, ripples) rather than a moving rigid body.
        self.stationary_drift_ratio = stationary_drift_ratio


class ObjectMotionClassificationStage(PipelineStage):
    """
    Classifies every object with an extracted per-object video (VideoObjectExtractionStage)
    as stationary (attached to the ground/scene, oscillating in place -- gentle wind
    sway on foliage, ripples) or moving (a rigid body translating across the frame --
    a bird, car, walking person). VideoGenerationStage's camera is explicitly locked
    off, so 2D centroid drift reflects real object motion, not camera motion.

    Stationary objects get simple sway parameters (amplitude/frequency) from their own
    horizontal centroid signal. Moving objects get their world-space trajectory
    unprojected -- reusing the same unproject_bbox/unproject_bbox_equirect utilities
    SceneGenerationStage already uses, holding depth fixed at the reference-frame
    estimate since no per-frame depth exists -- and fit with a smoothed curve to
    estimate linear velocity/acceleration. Angular velocity is out of scope: not
    reliably recoverable from a monocular silhouette.

    Reads:  object_motion_{i} (VideoObjectExtractionStage), metadata_{i},
            ContextKey.DEPTH / PANORAMA_OBJECT_DEPTH, INTRINSICS, EXTRINSICS,
            ContextKey.INPUT / PANORAMA
    Writes: metadata_{i}["stationary"] (bool), plus one of:
              metadata_{i}["sway"]   = {"amplitude", "frequencyHz"}              (stationary)
              metadata_{i}["motion"] = {"velocity": [x,y,z], "acceleration": [x,y,z]} (moving)
    Output key (SemanticKey.OUTPUT) -> ContextKey.OBJECT_MOTION_COUNT (int)
    """

    @classmethod
    def config_class(cls) -> type[ObjectMotionClassificationConfiguration]:
        return ObjectMotionClassificationConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects, skipping")
            return context

        depth = context.input_depth(ContextKey.DEPTH)
        pano_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        intrinsics = context.input_intrinsics(ContextKey.INTRINSICS)
        extrinsics = context.input_extrinsics(ContextKey.EXTRINSICS)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        input_image = context.input_image(ContextKey.INPUT)

        processed = 0
        task = self.create_progress(object_count, "Classifying object motion…")
        for idx in range(object_count):
            motion = context.input_object(f"object_motion_{idx}")
            metadata = context.input_object(f"metadata_{idx}") or {}
            if metadata.get("synthetic"):
                # Synthetic distribution points (DistributionSynthesisStage) are
                # procedurally scattered decorative fill, never individually
                # tracked in the generated video -- there's no per-instance
                # drift signal to classify from. They're inherently non-moving,
                # so mark them stationary with no measured "sway": when this
                # instance renders as a MESH sharing a bucket some real sibling
                # got rigged for, SceneAnimationStage already falls back to
                # generic amplitude/frequency defaults for exactly this case
                # (absent "sway"), rather than leaving it un-animated while its
                # rigged mesh siblings sway.
                context.add_object(f"metadata_{idx}", {**metadata, "stationary": True})
                processed += 1
                self.advance_progress(task)
                continue
            # No usable track for this object: either Video Object Extraction never
            # produced one (occluded at frame 0, mask too small, video generation
            # off) or it produced fewer than two frames where the mask was non-empty.
            # Fall back to stationary with no measured sway, rather than leaving
            # "stationary" absent entirely.
            #
            # An absent flag isn't neutral -- SceneAnimationStage skips any object
            # without it outright, so an untracked instance got no video, no sway and
            # no physics, and sat frozen next to tracked siblings that swayed. Marking
            # it stationary lets those paths engage: a mesh picks up the generic sway
            # defaults, and a billboard can still animate from the clip belonging to
            # the crop it actually displays (its pooled sibling's -- see
            # Object3D.texture_source_index), which frequently exists even when this
            # instance's own track doesn't. Same trade already made below when world
            # unprojection fails: a genuinely moving object that we failed to track
            # loses its physics, which is the better error than freezing every plant
            # the tracker happened to miss.
            usable = motion is not None
            if usable:
                fps = motion["fps"]
                centroids = np.asarray(motion["centroids"], dtype=float)
                bboxes = np.asarray(motion["bboxes"], dtype=float)
                valid = bboxes[:, 2] > 0  # frames where the mask wasn't empty
                usable = bool(valid.sum() >= 2)
            if not usable:
                context.add_object(f"metadata_{idx}", {**metadata, "stationary": True})
                processed += 1
                self.advance_progress(task)
                continue

            valid_centroids = centroids[valid]
            valid_bboxes = bboxes[valid]
            diagonal = float(np.median(np.hypot(valid_bboxes[:, 2], valid_bboxes[:, 3]))) or 1.0

            # Total drift: twice the max distance from the trajectory's own mean
            # position -- robust to a single outlier frame in a way peak-to-peak
            # bbox extent isn't, and cheap.
            centered = valid_centroids - valid_centroids.mean(axis=0)
            drift = float(np.max(np.linalg.norm(centered, axis=1))) * 2.0
            stationary = (drift / diagonal) < self.config.stationary_drift_ratio

            # Vegetation is rooted. Whatever drift its centroid shows is the
            # tracker following the plant's own sway (or the mask breathing as
            # leaves move), never the plant translating through the scene, so the
            # drift test has no valid "moving" answer to give here and its false
            # positives are pure cost: the moving branch emits `motion` and no
            # `sway`, and SceneAnimationStage turns that into a PhysicsHandoff --
            # a Rigidbody and a collider bolted onto a flower, which then sits
            # frozen next to its swaying siblings because nothing gave it sway.
            # Observed on the Rainier capture: a flower classified moving at a
            # measured velocity of 0.0014 m/s, i.e. indistinguishable from
            # stationary except in which branch it took.
            if not stationary and metadata.get("class") in VEGETATION_CATEGORIES:
                self.log_info(
                    f"  {idx} ({metadata.get('class')}): drift {drift / diagonal:.2f} treated as sway, not translation"
                )
                stationary = True

            updates: dict[str, Any] = {"stationary": stationary}
            if stationary:
                updates["sway"] = self._sway_params(valid_centroids, diagonal, fps)
            else:
                motion_estimate = self._world_motion(
                    valid_centroids, fps, motion["width"], motion["height"],
                    depth, pano_depth, intrinsics, extrinsics, panorama,
                )
                if motion_estimate is not None:
                    updates["motion"] = motion_estimate
                else:
                    # Couldn't unproject (no depth/intrinsics available this run) --
                    # fall back to stationary rather than silently dropping the object.
                    updates["stationary"] = True
                    updates["sway"] = self._sway_params(valid_centroids, diagonal, fps)

            context.add_object(f"metadata_{idx}", {**metadata, **updates})
            processed += 1
            self.advance_progress(task)
        self.finish_progress(task)

        context.add_object(ContextKey.OBJECT_MOTION_COUNT, processed)
        self.log_info(f"Classified {processed} object(s)")
        return context

    def _sway_params(self, centroids: np.ndarray, diagonal: float, fps: float) -> dict:
        x = centroids[:, 0]
        detrended = x - x.mean()
        amplitude = float((detrended.max() - detrended.min()) / diagonal) if diagonal else 0.0

        crossings = np.where(np.diff(np.signbit(detrended)))[0]
        duration_s = len(x) / fps if fps else 0.0
        frequency_hz = (len(crossings) / 2.0) / duration_s if duration_s > 0 else 0.0

        return {
            "amplitude": round(min(amplitude, 1.0), 4),
            "frequencyHz": round(min(frequency_hz, 2.0), 3),
        }

    def _world_motion(
        self,
        centroids: np.ndarray,
        fps: float,
        video_w: int,
        video_h: int,
        depth,
        pano_depth,
        intrinsics,
        extrinsics,
        panorama,
    ) -> Optional[dict]:
        positions = []
        for cx, cy in centroids:
            point_bbox = [float(cx), float(cy), 1.0, 1.0]
            if pano_depth is not None and panorama is not None:
                result = unproject_bbox_equirect(point_bbox, video_w, video_h, pano_depth=pano_depth, extrinsics=extrinsics)
            elif depth is not None and intrinsics is not None:
                result = unproject_bbox(point_bbox, video_w, video_h, depth_map=depth, intrinsics=intrinsics, extrinsics=extrinsics)
            else:
                return None
            if result is None:
                continue
            position, _, _ = result
            positions.append(position)

        if len(positions) < 3:
            return None

        positions_arr = np.asarray(positions, dtype=float)
        t = np.arange(len(positions_arr)) / fps if fps else np.arange(len(positions_arr), dtype=float)

        # Fit each axis with a degree-2 polynomial (position = p0 + v*t + 0.5*a*t^2):
        # smooths frame-to-frame segmentation noise out of a plain finite-difference,
        # and its derivatives give the clip's own constant-acceleration estimate.
        mid_t = float(t[len(t) // 2])
        velocity = np.zeros(3)
        acceleration = np.zeros(3)
        for axis in range(3):
            a2, a1, _ = np.polyfit(t, positions_arr[:, axis], deg=2)
            velocity[axis] = 2.0 * a2 * mid_t + a1
            acceleration[axis] = 2.0 * a2

        return {
            "velocity": [round(float(v), 4) for v in velocity],
            "acceleration": [round(float(a), 4) for a in acceleration],
        }

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(ContextKey.OBJECT_MOTION_COUNT) is not None

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        count = context.object(ContextKey.OBJECT_MOTION_COUNT)
        if count is None:
            return None
        return ReportSection(
            stage_name=self.name,
            title="Object Motion Classification",
            body=(
                "Each object with an extracted per-object video was classified as "
                "stationary (gentle in-place motion, e.g. wind on foliage) or moving "
                "(a rigid body translating through the scene), using that object's own "
                "2D centroid trajectory against the locked-off generated video."
            ),
            stats={"Objects classified": str(count)},
        )
