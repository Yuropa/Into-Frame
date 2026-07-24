import av
import numpy as np
import torch
from logging import Logger
from pathlib import Path
from typing import Any, Optional
from PIL import Image as PILImage

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import ANIMATABLE_CATEGORIES, VEGETATION_CATEGORIES
from pipeline.segmentation.video_segmentation import VideoSeg
from util.device_utils import DeviceStrategy, preferred_device
from util.video_utils import Video

# Fallback frame rate when neither the source Video nor its own container header
# report one (shouldn't happen for a real encoded file, but av.average_rate can be
# None for some containers).
DEFAULT_FRAME_RATE = 24.0


class VideoObjectExtractionConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        reference_frame_idx: int = 0,
        max_tracked_per_bucket: int = 3,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.reference_frame_idx = reference_frame_idx
        # Vegetation instances sharing a (class, bucket) visual variant (see
        # ObjectCategoryClusteringStage) are visually interchangeable and their sway
        # is generic wind physics, not something that needs individual ground truth --
        # a scene can have dozens of real detected trees/bushes in one bucket, and
        # tracking all of them would be paying SAM2's most expensive step once per
        # near-duplicate instead of once per visual variant. Only the first
        # max_tracked_per_bucket instances encountered per (class, bucket) get tracked;
        # the rest render with no sway animation. Does not apply to non-vegetation
        # ANIMATABLE_CATEGORIES (person/vehicle/animal/...): those are genuinely
        # distinct subjects, not decorative fill, so each real detection keeps its
        # own trajectory regardless of bucket.
        self.max_tracked_per_bucket = max_tracked_per_bucket


class VideoObjectExtractionStage(PipelineStage):
    """
    For each object SegmentationStage found on the reference frame, tracks its mask
    through the whole video with VideoSeg (SAM2's video predictor) and writes out one
    video per object: same frame size and frame count as the source, with everything
    outside that object's per-frame mask blacked out.

    Input key      (SemanticKey.INPUT)         -> ContextKey.INPUT            (Image, default)
                                                   or a Panorama at the same key
      The frame crop_{i}/metadata_{i}'s boxes are expressed against, and which
      reference_frame_idx (below) corresponds to in the video — normally the same
      image/panorama VideoGenerationStage conditions frame 0 on, so the defaults
      line up without extra config (see config.yaml's `keys: input: panorama`
      override, matching the same override on Video Generation and Object
      Segmentation/Recognition/Detection once those also run on the panorama).
    Video key      (SemanticKey.VIDEO)         -> ContextKey.GENERATED_VIDEO   (Video)
    Object count   (SemanticKey.OBJECT_COUNT)  -> ContextKey.OBJECT_COUNT     (int)
    Output key     (SemanticKey.OUTPUT)        -> ContextKey.OBJECT_VIDEO_COUNT (int)

    Scene generation already ran by this point (see config.yaml stage order) and
    decides -- per category/bucket curation, LOD distance, size limits, etc. -- which
    detections actually become a placed Object3D; everything else (environment/
    indeterminate classes, filtered categories, oversized/unprojectable detections,
    a synthetic distribution point with no crop of its own) never appears in the
    scene at all. Extraction only tracks/encodes objects whose index shows up as
    some Object3D.source_index in ContextKey.SCENE, since SAM2 video tracking is by
    far the most expensive step here and there's no point paying it for a detection
    nothing will ever render.

    Placed objects are further filtered by ANIMATABLE_CATEGORIES: only categories
    that can plausibly show visible motion -- vegetation (wind sway), people/
    vehicles/animals (rigid-body movement), flowing water -- get tracked. Everything
    else (street furniture, signage, fixed infrastructure, statues, buildings, ...)
    is permanently static, so SceneAnimationStage's video-billboard/sway/physics
    annotation has nothing to attach for them regardless -- tracking them would
    only produce mask-tracking noise on an object that never actually moves.

    Within VEGETATION_CATEGORIES, instances are further capped at
    max_tracked_per_bucket per (class, bucket) visual variant (see
    ObjectCategoryClusteringStage) -- a scene can have dozens of real detected
    trees/bushes clustered into a handful of visual buckets, and each instance's
    sway is generic wind physics rather than something needing individual ground
    truth, so tracking every one of them would pay SAM2's most expensive step once
    per near-duplicate instead of once per visual variant. Instances beyond the cap
    still get placed (via the shared bucket mesh or billboard pool) but render with
    no sway animation. Non-vegetation ANIMATABLE_CATEGORIES are never capped this
    way -- a person or vehicle is a genuinely distinct subject, not decorative fill,
    so every real detection keeps its own trajectory.

    Reads:  ContextKey.SCENE (Scene, to know which detections were actually placed),
            crop_{i} (Image, RGBA masked crop), metadata_{i} ({"box": [x, y, w, h], ...})
    Writes: object_video_{i} (Video) for every object whose mask survives tracking —
              color frames, background blacked out, for use as an animated billboard.
            object_video_alpha_{i} (Video) — a matching grayscale matte video (mask
              value repeated across RGB), so a billboard shader can composite
              transparency without keying on black (which clips dark object pixels
              and bleeds at compressed edges).
            object_motion_{i} (object) — {"fps", "width", "height", "centroids":
              [[x,y], ...], "bboxes": [[x,y,w,h], ...]} per-frame stats in
              video-pixel space (width/height are that space's frame size, needed
              to unproject centroids with the same depth/intrinsics utilities
              SceneGenerationStage uses), derived from the same per-frame masks
              used to write the videos above.
              A zero bbox/centroid marks a frame where the object's mask was empty
              (occluded/off-screen) — downstream consumers should skip those frames
              rather than treat them as a real position sample.

    Config: reference_frame_idx — video frame that lines up with the reference image's
      masks (default 0, matching VideoGenerationStage's frame-0 conditioning).
    """

    @classmethod
    def config_class(cls) -> type[VideoObjectExtractionConfiguration]:
        return VideoObjectExtractionConfiguration

    def __init__(self, config: VideoObjectExtractionConfiguration) -> None:
        super().__init__(config)
        self._video_seg = None
        # AUTO (the default device), not MEMORY: this stage is compute-bound, not
        # memory-bound (VRAM footprint stays low even at 100% GPU util -- SAM2 video
        # tracking works one frame at a time), so it doesn't need MEMORY's
        # highest-total-capacity GPU. Landing it on the default device instead keeps
        # it off whichever GPU the memory-hungry diffusion stages contend for.
        self.preferred_device, _ = preferred_device(DeviceStrategy.AUTO)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT,
            SemanticKey.VIDEO: ContextKey.GENERATED_VIDEO,
            SemanticKey.OBJECT_COUNT: ContextKey.OBJECT_COUNT,
            SemanticKey.OUTPUT: ContextKey.OBJECT_VIDEO_COUNT,
        })

    def _resolve_source(self, context: PipelineContext, input_key):
        """Image and Panorama both expose .width/.height/.size, so either can define
        ref_size below — try Image (the common case) before Panorama."""
        image = context.input_image(input_key)
        if image is not None:
            return image
        return context.input_panorama(input_key)

    def _reference_mask(
        self,
        idx: int,
        context: PipelineContext,
        ref_size: tuple[int, int],
        video_size: tuple[int, int],
    ) -> Optional[np.ndarray]:
        """Reconstruct a full-frame boolean mask for object `idx`, resized to match
        the video's own resolution (which can differ from the reference image's, e.g.
        after VideoGenerationStage's aspect-preserving resize)."""
        crop = context.input_image(f"crop_{idx}")
        metadata = context.input_object(f"metadata_{idx}")
        if crop is None or metadata is None:
            return None

        box = metadata.get("box")
        if not box:
            return None
        x, y, w, h = box

        alpha = crop.rgba().split()[-1]
        target_size = (max(1, round(w)), max(1, round(h)))
        if alpha.size != target_size:
            alpha = alpha.resize(target_size)

        # Columns wrapped modulo ref_size's width: a box straddling the
        # panorama's own left/right seam (rare -- see util/instance_merge.py)
        # still pastes into the right place instead of being silently clipped
        # by a plain PIL paste.
        ref_w, ref_h = ref_size
        alpha_arr = np.array(alpha) > 127
        canvas = np.zeros((ref_h, ref_w), dtype=bool)
        ix, iy = round(x), round(y)
        ch, cw = alpha_arr.shape
        y0, y1 = max(0, iy), min(ref_h, iy + ch)
        if y1 > y0:
            cols = (ix + np.arange(cw)) % ref_w
            rows = np.arange(y0, y1)
            canvas[np.ix_(rows, cols)] |= alpha_arr[y0 - iy: y1 - iy, :]

        canvas_img = PILImage.fromarray((canvas * 255).astype(np.uint8), mode="L")
        if ref_size != video_size:
            canvas_img = canvas_img.resize(video_size, PILImage.NEAREST)
        return np.array(canvas_img) > 127

    def _read_frames(self, path: Path) -> tuple[list[np.ndarray], float]:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate else DEFAULT_FRAME_RATE
            # stream.frames (container-declared frame count) is usually reliable for an
            # MP4 we encoded ourselves -- when it isn't (0/unknown, e.g. some muxers
            # leave it unset), fall back to a single-step task rather than guessing a
            # total, same coarse behaviour this stage already had.
            total = int(stream.frames) if stream.frames else 0
            decode_task = self.create_progress(max(total, 1), "Decoding video…")
            frames = []
            for frames_decoded, frame in enumerate(container.decode(stream), start=1):
                frames.append(frame.to_ndarray(format="rgb24"))
                if total > 0:
                    self.update_progress(decode_task, frames_decoded / total)
            self.finish_progress(decode_task)
        return frames, fps

    def _encode_h264(
        self, path: Path, width: int, height: int, fps: float, rgb_frames, total: int, on_progress=None,
    ):
        with av.open(str(path), mode="w") as out_container:
            stream = out_container.add_stream("h264", rate=max(1, round(fps)))
            stream.width = width
            stream.height = height
            stream.pix_fmt = "yuv420p"
            for i, frame in enumerate(rgb_frames, start=1):
                video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
                for packet in stream.encode(video_frame):
                    out_container.mux(packet)
                if on_progress is not None and total > 0:
                    on_progress(i / total, "Encoding…")
            for packet in stream.encode():
                out_container.mux(packet)

    def _write_masked_video(
        self, path: Path, frames: list[np.ndarray], masks: np.ndarray, fps: float, on_progress=None,
    ):
        height, width = frames[0].shape[:2]
        rgb_frames = ((frame * mask[..., None]).astype(np.uint8) for frame, mask in zip(frames, masks))
        self._encode_h264(path, width, height, fps, rgb_frames, total=len(frames), on_progress=on_progress)

    def _write_alpha_video(self, path: Path, masks: np.ndarray, fps: float, on_progress=None):
        """Grayscale matte video (mask value repeated across RGB) matching a
        _write_masked_video color video frame-for-frame — see class docstring's
        object_video_alpha_{i} entry for why a separate matte beats black-keying."""
        height, width = masks[0].shape
        rgb_frames = (np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2) for mask in masks)
        self._encode_h264(path, width, height, fps, rgb_frames, total=len(masks), on_progress=on_progress)

    def _frame_stats(self, masks: np.ndarray) -> dict:
        """Per-frame centroid + tight bbox (video-pixel space) from each frame's mask.
        A frame with an empty mask gets an all-zero entry -- callers must treat
        that as "no sample" rather than a real position at the origin."""
        centroids: list[list[float]] = []
        bboxes: list[list[float]] = []
        for mask in masks:
            ys, xs = np.nonzero(mask)
            if len(xs) == 0:
                centroids.append([0.0, 0.0])
                bboxes.append([0.0, 0.0, 0.0, 0.0])
                continue
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            centroids.append([float(xs.mean()), float(ys.mean())])
            bboxes.append([x0, y0, x1 - x0, y1 - y0])
        return {"centroids": centroids, "bboxes": bboxes}

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, video_key, count_key, output_key = self._resolved_keys()

        input_image = self._resolve_source(context, input_key)
        video = context.input_video(video_key)
        object_count = context.input_object(count_key) or 0

        if video is None or not object_count:
            self.log_info("No video or no detected objects, skipping")
            return context

        scene = context.input_scene(ContextKey.SCENE)
        if scene is None:
            self.log_info("No scene yet, skipping (nothing placed = nothing valid to extract)")
            return context
        placed_indices = {obj.source_index for obj in scene.objects if obj.source_index is not None}
        if not placed_indices:
            self.log_info("Scene has no detections placed, skipping")
            return context

        frames, fps = self._read_frames(video.path)
        if not frames:
            self.log_info(f"  {video_key}: no decodable frames, skipping")
            return context

        video_h, video_w = frames[0].shape[:2]
        video_size = (video_w, video_h)
        ref_size = input_image.size if input_image is not None else video_size

        if self._video_seg is None:
            self._video_seg = VideoSeg(self.preferred_device)

        extracted = 0
        tracked_per_bucket: dict[tuple[str, int], int] = {}
        extraction_task = self.create_progress(object_count, "Extracting per-object videos…")
        for idx in range(object_count):
            if idx not in placed_indices:
                self.advance_progress(extraction_task)
                continue

            metadata = context.input_object(f"metadata_{idx}") or {}
            obj_class = metadata.get("class")
            if obj_class not in ANIMATABLE_CATEGORIES:
                self.advance_progress(extraction_task)
                continue

            # Synthetic points (DistributionSynthesisStage) are procedural placements,
            # not real detections -- they never appeared in the panorama the video was
            # generated from, so there's no real footage of them to track. They have
            # no "box"/crop_{idx} either, so _reference_mask below would always fail
            # for them anyway -- skip explicitly and early instead of silently burning
            # a max_tracked_per_bucket slot (they'd all collapse onto bucket 0, since
            # synthetic points never have a "bucket") on a guaranteed no-op.
            if metadata.get("synthetic"):
                self.advance_progress(extraction_task)
                continue

            if obj_class in VEGETATION_CATEGORIES:
                bucket_key = (obj_class, int(metadata.get("bucket") or 0))
                tracked = tracked_per_bucket.get(bucket_key, 0)
                if tracked >= self.config.max_tracked_per_bucket:
                    self.advance_progress(extraction_task)
                    continue
                tracked_per_bucket[bucket_key] = tracked + 1

            out_key = f"object_video_{idx}"
            alpha_key = f"object_video_alpha_{idx}"
            motion_key = f"object_motion_{idx}"

            if (context.video(out_key) is not None
                    and context.video(alpha_key) is not None
                    and context.object(motion_key) is not None):
                self.log_info(f"  {out_key}: cached")
                extracted += 1
                self.advance_progress(extraction_task)
                continue

            mask = self._reference_mask(idx, context, ref_size, video_size)
            if mask is None or not mask.any():
                self.advance_progress(extraction_task)
                continue

            temp_path = self.temp / out_key if self.temp is not None else None
            if temp_path is not None:
                temp_path.mkdir(parents=True, exist_ok=True)

            result = self._video_seg.segment_video(
                video,
                reference_mask=mask,
                temp_path=temp_path,
                reference_frame_idx=self.config.reference_frame_idx,
                on_progress=self.make_progress_callback(extraction_task),
            )

            num_frames = min(len(frames), result.num_frames)
            frame_masks = result.masks[:num_frames]

            # Fresh callback per phase (not one reused across both): make_progress_callback
            # reads the sub-task's *current* completed position each time it's called, so
            # each of these two encodes gets whatever headroom segment_video's own
            # progress calls (and the other encode) left in this object's one step,
            # instead of both writing into a stale, already-consumed span.
            out_path = (temp_path or self.temp) / "object.mp4"
            self._write_masked_video(
                out_path, frames[:num_frames], frame_masks, fps,
                on_progress=self.make_progress_callback(extraction_task),
            )
            context.add_video(out_key, Video(out_path, fps=fps, num_frames=num_frames))

            alpha_path = (temp_path or self.temp) / "object_alpha.mp4"
            self._write_alpha_video(
                alpha_path, frame_masks, fps,
                on_progress=self.make_progress_callback(extraction_task),
            )
            context.add_video(alpha_key, Video(alpha_path, fps=fps, num_frames=num_frames))

            context.add_object(motion_key, {
                "fps": fps,
                "width": video_w,
                "height": video_h,
                **self._frame_stats(frame_masks),
            })

            self.log_info(f"  {out_key}: {num_frames} frames @ {fps:.1f}fps")
            extracted += 1
            self.advance_progress(extraction_task)
        self.finish_progress(extraction_task)

        context.add_object(output_key, extracted)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, _, output_key = self._resolved_keys()
        return context.object(output_key) is not None

    def model_names(self) -> list[str]:
        return VideoSeg.model_names()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        _, _, _, output_key = self._resolved_keys()
        count = context.object(output_key)
        if count is None:
            return None
        return ReportSection(
            stage_name=self.name,
            title="Video Object Extraction",
            body=(
                "Each detected object's mask was tracked across the whole video with "
                "SAM2's video predictor, then used to isolate that object into its own "
                "video with the rest of each frame blacked out."
            ),
            stats={"Videos extracted": str(count)},
        )

    def clean_up(self):
        if self._video_seg is not None:
            self._video_seg.close()
            self._video_seg = None
        super().clean_up()
