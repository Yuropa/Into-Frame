import threading

import av
import numpy as np
import torch
from logging import Logger
from pathlib import Path
from typing import Any, Optional
from PIL import Image as PILImage

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import ANIMATABLE_CATEGORIES
from pipeline.segmentation.video_segmentation import VideoSeg
from util.device_utils import DeviceStrategy
from util.gpu_task_pool import GpuTaskPool
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
        max_tracked_per_category: int = 3,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.reference_frame_idx = reference_frame_idx
        # SAM2 video tracking is by far the most expensive step in this stage. A
        # class routinely splits into dozens of small ObjectCategoryClusteringStage
        # visual-similarity buckets (e.g. a "tree" class with 65 buckets and only
        # 1-2 instances each is typical), so a per-bucket cap lets nearly everything
        # through uncapped -- and non-vegetation ANIMATABLE_CATEGORIES (person,
        # vehicle, animal, ...) previously had no cap at all, on the reasoning that
        # each is a genuinely distinct subject. In practice a real scene can have
        # well over a hundred detected trees or a couple dozen people, and tracking
        # all of them is far more than needed to drive that class's animation look --
        # a handful of tracked clips per class is enough. Capped per class overall
        # instead, uniformly across every ANIMATABLE_CATEGORIES class. Instances
        # beyond the cap still get placed (mesh/billboard) via Scene Generation,
        # just with no sway/motion animation.
        self.max_tracked_per_category = max_tracked_per_category


class VideoObjectExtractionStage(PipelineStage):
    """
    For each object SegmentationStage found on the reference frame, tracks its mask
    through the whole video with VideoSeg (SAM2's video predictor) and writes out one
    video per object: same frame size and frame count as the source, with everything
    outside that object's per-frame mask blacked out.

    SAM2 tracking itself runs via GpuTaskPool: every eligible object is enqueued once
    (cheap category/cache/cap filtering and mask reconstruction happen serially first,
    see run() -- there's no benefit to doing plain CPU work inside the pool), then
    the pool runs across every available device in parallel (one VideoSeg subprocess
    per device, created lazily and reused for that device's whole share of the work --
    spinning up a fresh SAM2 subprocess per object would be far too slow). An object
    whose tracking fails on its first (parallel) attempt is retried once, serially, on
    the pool's fallback device -- see GpuTaskPool's own docstring. A fallback failure
    is not caught here either; it propagates out of run() and aborts the pipeline run,
    same terminal severity a single failure already had before this stage used the
    pool (there was no retry at all previously).

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

    Every ANIMATABLE_CATEGORIES class is further capped at max_tracked_per_category
    instances overall (not per ObjectCategoryClusteringStage visual-similarity
    bucket -- a class routinely splits into dozens of small buckets, which would
    let a per-bucket cap through almost entirely uncapped). A couple of tracked
    clips is already enough to drive a class's animation look, whether that's
    generic wind sway on foliage or a person/vehicle's rigid-body trajectory --
    tracking every one of what can be well over a hundred real detections just
    pays SAM2's most expensive step for redundant samples. Instances beyond the
    cap still get placed (via the shared bucket mesh, individual mesh, or
    billboard pool) but render with no sway/motion animation.

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
      max_tracked_per_category — cap on tracked instances per class overall
      (default 3), across every ANIMATABLE_CATEGORIES class.
    """

    @classmethod
    def config_class(cls) -> type[VideoObjectExtractionConfiguration]:
        return VideoObjectExtractionConfiguration

    def __init__(self, config: VideoObjectExtractionConfiguration) -> None:
        super().__init__(config)
        # Keyed by device (not a single instance) -- GpuTaskPool runs one worker
        # per available device, and each needs its own VideoSeg subprocess (a
        # RemoteClient pinned to that device via CUDA_VISIBLE_DEVICES). Created
        # lazily in _get_video_seg the first time a given device is actually used,
        # and reused for every other task that lands on that same device.
        self._video_segs: dict[torch.device, VideoSeg] = {}
        self._video_segs_lock = threading.Lock()
        # Guards self.advance_progress(extraction_task) when called from a
        # GpuTaskPool worker thread -- it does a read-then-write on the task's
        # shared `completed` counter that isn't safe under concurrent callers,
        # unlike a plain context.add_video/add_object (see _track_one).
        self._progress_lock = threading.Lock()

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

    def _get_video_seg(self, device: torch.device) -> VideoSeg:
        with self._video_segs_lock:
            video_seg = self._video_segs.get(device)
            if video_seg is None:
                video_seg = VideoSeg(device)
                self._video_segs[device] = video_seg
            return video_seg

    def _track_one(self, device: torch.device, data: tuple[int, np.ndarray]) -> bool:
        """GpuTaskPool work_fn -- runs on a worker thread pinned to `device`. Writes
        its results straight to context (safe: every task touches distinct keys, so
        this is just non-overlapping dict/set writes on the shared per-stage state)
        rather than batching them until after pool.run() returns, so a task that
        fails unrecoverably on the fallback phase doesn't discard every other
        object's already-completed video."""
        idx, mask = data
        video_seg = self._get_video_seg(device)

        out_key = f"object_video_{idx}"
        alpha_key = f"object_video_alpha_{idx}"
        motion_key = f"object_motion_{idx}"
        temp_path = self.temp / out_key if self.temp is not None else None
        if temp_path is not None:
            temp_path.mkdir(parents=True, exist_ok=True)

        result = video_seg.segment_video(
            self._video,
            reference_mask=mask,
            temp_path=temp_path,
            reference_frame_idx=self.config.reference_frame_idx,
        )

        num_frames = min(len(self._frames), result.num_frames)
        frame_masks = result.masks[:num_frames]

        out_path = (temp_path or self.temp) / "object.mp4"
        self._write_masked_video(out_path, self._frames[:num_frames], frame_masks, self._fps)
        self._context.add_video(out_key, Video(out_path, fps=self._fps, num_frames=num_frames))

        alpha_path = (temp_path or self.temp) / "object_alpha.mp4"
        self._write_alpha_video(alpha_path, frame_masks, self._fps)
        self._context.add_video(alpha_key, Video(alpha_path, fps=self._fps, num_frames=num_frames))

        self._context.add_object(motion_key, {
            "fps": self._fps,
            "width": self._video_w,
            "height": self._video_h,
            **self._frame_stats(frame_masks),
        })

        self.log_info(f"  {out_key}: {num_frames} frames @ {self._fps:.1f}fps on {device}")
        with self._progress_lock:
            self.advance_progress(self._extraction_task)
        return True

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

        # Shared, read-only state _track_one needs -- GpuTaskPool's work_fn signature
        # is fixed to (device, data), so anything beyond the per-task mask has to be
        # stage instance state rather than an extra closure argument.
        self._context = context
        self._video = video
        self._frames = frames
        self._fps = fps
        self._video_w = video_w
        self._video_h = video_h

        extracted = 0
        tracked_per_category: dict[str, int] = {}
        eligible: list[tuple[int, np.ndarray]] = []
        self._extraction_task = extraction_task = self.create_progress(object_count, "Extracting per-object videos…")

        # Pass 1: cheap, serial filtering + mask reconstruction. Category/cache/cap
        # checks and mask reconstruction are all plain CPU work with no reason to
        # run inside the parallel pool -- deciding what's worth SAM2 tracking before
        # touching any GPU keeps GpuTaskPool's work_fn (_track_one) doing nothing but
        # the actually-parallelizable part.
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
            # a max_tracked_per_category slot on a guaranteed no-op.
            if metadata.get("synthetic"):
                self.advance_progress(extraction_task)
                continue

            tracked = tracked_per_category.get(obj_class, 0)
            if tracked >= self.config.max_tracked_per_category:
                self.advance_progress(extraction_task)
                continue

            out_key = f"object_video_{idx}"
            alpha_key = f"object_video_alpha_{idx}"
            motion_key = f"object_motion_{idx}"

            if (context.video(out_key) is not None
                    and context.video(alpha_key) is not None
                    and context.object(motion_key) is not None):
                self.log_info(f"  {out_key}: cached")
                tracked_per_category[obj_class] = tracked + 1
                extracted += 1
                self.advance_progress(extraction_task)
                continue

            mask = self._reference_mask(idx, context, ref_size, video_size)
            if mask is None or not mask.any():
                self.advance_progress(extraction_task)
                continue

            tracked_per_category[obj_class] = tracked + 1
            eligible.append((idx, mask))

        # Pass 2: SAM2 tracking + encoding, parallel across every available device
        # (GpuTaskPool), with anything that fails its first attempt retried once,
        # serially, on the fallback device. _track_one writes context/log/progress
        # itself as each task completes -- see its docstring for why.
        if eligible:
            pool = GpuTaskPool(self._track_one, device_strategy=DeviceStrategy.AUTO)
            for task_data in eligible:
                pool.enqueue(task_data)
            outcomes = pool.run()
            extracted += sum(1 for ok in outcomes if ok)

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
        with self._video_segs_lock:
            for video_seg in self._video_segs.values():
                video_seg.close()
            self._video_segs.clear()
        super().clean_up()
