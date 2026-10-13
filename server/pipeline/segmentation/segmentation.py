from logging import Logger
from typing import Any

import torch
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.foreground_segmentation import ForegroundSeg
from pipeline.segmentation.segmentation_result import SegmentationResult
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.mask_inpainting import MaskInPainting
from pipeline.segmentation.hires_crops import build_hires_crops, CAPTURED_FOV_RESTRICTED_KEY
from pipeline.segmentation.photo_detection import detection_photo, store_photo_masks
from util.captured_cone import captured_cone_from_context
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

# Must match image_segmentation_imp.py's own _TILE_SIZE -- see its use in
# SegmentationStage.run() below for why.
_TILE_SIZE = 1024


class SegmentationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        # SAM2's automatic mask generator (points_per_side=64, crop_n_layers=1 in
        # image_segmentation_imp.py) finds a lot of texture fragments -- individual
        # leaves, pebbles, bark cracks -- that aren't meaningful discrete objects for
        # this pipeline's purposes. Reject a crop whose bounding box covers less than
        # this fraction of the detection source's area. Calibrated against a real
        # 1200x807 test photo (309 raw SAM2 detections): 0.0003 (~290px² there) removed
        # ~15% of them, all confirmed by eye to be unrecognisable specks.
        min_area_fraction: float = 0.0003,
        # Reject a crop covering MORE than this fraction -- SAM2 sometimes segments
        # almost the entire scene (a whole meadow, a whole mountain backdrop) as a
        # single candidate "object" instead of a discrete one. Calibrated against the
        # same test photo: the largest genuine single object (a full tree) was ~5% of
        # image area; the smallest whole-scene false positive was ~33% -- 0.20 sits
        # with wide margin in that gap.
        max_area_fraction: float = 0.20,
        # Keep only detections that lie (mostly) inside the photographed cone of
        # the panorama. Everything outside it is the outpainter's guess, and the
        # outpainter repeats the hero content around the sphere -- on the Shark
        # Fin capture the same sea stack is detected three times, and across the
        # five sample captures 65-85% of "real" detections were of generated
        # pixels. Distribution Synthesis exists to populate the unobserved part
        # from the observed statistics; it should not be fed hallucinated
        # exemplars. See util.captured_cone.
        restrict_to_captured_fov: bool = True,
        # Degrees of slack on every side of the cone: an object straddling the
        # photo's edge is still real (the outpainter continues it rather than
        # inventing it), and the embedding is not pixel-exact at the border.
        captured_fov_margin_deg: float = 3.0,
        # Re-cut in-cone crops from the original photograph at native resolution
        # (crop_hires_{i}); see pipeline/segmentation/hires_crops.py.
        hires_crops_from_input: bool = True,
        # The super-resolution factor the panorama went through before this stage
        # (Panorama Supersampling is Swin2SR x2). Pixel counts alone say the
        # panorama matches the photo's resolution inside the cone; it does not --
        # it is the photo downsampled into a 2048-wide panorama and then
        # upscaled 2x, so its real information content is the pre-SR width. Used
        # only to decide whether the re-cut is worth doing. 1.0 if that stage is off.
        panorama_sr_factor: float = 2.0,
        # Inside the captured cone, detect on the PHOTOGRAPH (at native resolution,
        # optionally super-resolved) instead of on the panorama's downsampled copy of
        # it, and store each detection in both spaces -- see
        # pipeline/segmentation/photo_detection.py. Panorama detections that fall
        # inside the cone are then dropped as duplicates of the photo pass; the
        # panorama still supplies the out-of-cone detections foreground removal
        # needs. Requires a cone (intrinsics + input photo).
        detect_on_photo: bool = True,
        # Swin2SR factor applied to the photo before detecting on it (1 = none,
        # 2 or 4). Small blooms that are 20 px tall in a 1200 px photo clear SAM2's
        # mask-area floor at 2x; 4x costs four passes of SAM2 for little more.
        photo_detect_sr_factor: float = 2.0,
        # Resolution the PANORAMA pass runs at, as a fraction of the panorama's
        # size. With detect_on_photo the panorama pass only has to find occluders
        # outside the cone -- things Panorama Foreground Inpainting must cut out of
        # the terrain plate and that are never placed -- and a 4096-wide panorama is
        # twelve SAM2 tiles (21 minutes on the 2026-10-13 run) where 2048 is three.
        # Masks are upscaled back to full resolution for the crops. 1.0 restores the
        # full-resolution pass.
        panorama_detect_scale: float = 0.5,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.panorama_detect_scale = float(panorama_detect_scale)
        self.detect_on_photo = bool(detect_on_photo)
        self.photo_detect_sr_factor = float(photo_detect_sr_factor)
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.restrict_to_captured_fov = bool(restrict_to_captured_fov)
        self.captured_fov_margin_deg = float(captured_fov_margin_deg)
        self.hires_crops_from_input = bool(hires_crops_from_input)
        self.panorama_sr_factor = float(panorama_sr_factor)


class SegmentationStage(PipelineStage):
    """
    Detects and segments foreground objects in the input image, storing each
    object's masked crop and bounding-box metadata for downstream stages.

    Input key  (SemanticKey.INPUT)  → ContextKey.INPUT          (Image, default)
                                       or a Panorama at the same key — e.g.
                                       point it at ContextKey.PANORAMA to detect
                                       objects across the full 360° panorama
                                       instead of the narrow input photo.
    Output key (SemanticKey.OUTPUT) → ContextKey.OBJECT_COUNT    (int)

    Dynamic context keys per detected object (index i):
      crop_{i}      → Image  (masked object crop)
      metadata_{i}  → object ({"box": [...], "score": float})

    Config: min_area_fraction/max_area_fraction (see SegmentationConfiguration) --
    raw SAM2 detections outside this bounding-box-area range are dropped before
    ever being stored, so Classification/Typing/Correlation/Distribution never
    see them.
    """

    @classmethod
    def config_class(cls) -> type[SegmentationConfiguration]:
        return SegmentationConfiguration

    def __init__(self, config: SegmentationConfiguration) -> None:
        super().__init__(config)
        self._seg = None
        self._foreground_seg = None
        self._mask_inpainting = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT,
            SemanticKey.OUTPUT: ContextKey.OBJECT_COUNT
        })

    def _resolve_source(self, context: PipelineContext, input_key) -> "Image | None":
        """Image and Panorama both expose .width/.height/.rgb()/.image, so either
        can be the detection source — try Image (the common case) before Panorama,
        wrapping a Panorama into a plain Image since Panorama has no .rgba(), which
        masked_images() below needs."""
        image = context.input_image(input_key)
        if image is not None:
            return image
        panorama = context.input_panorama(input_key)
        if panorama is not None:
            return Image(panorama.image)
        return None

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, output_key = self._resolved_keys()

        source = self._resolve_source(context, input_key)
        if source is None:
            self.log_info("No input image or panorama, skipping")
            return context
        input_image = source.copy()
        total_crops = 0

        cfg: SegmentationConfiguration = self.config
        # min_area_fraction/max_area_fraction (see their own comments above)
        # were calibrated against a single ~1200x807 hero photo -- i.e.
        # roughly one _TILE_SIZE tile's worth of view (image_segmentation_imp.py
        # tiles anything panorama-scale into 1024x1024 crops before detecting
        # anything). Using the actual source image's area as the fraction's
        # denominator silently breaks that calibration once this stage is
        # pointed at a full 360° panorama instead of a hero photo (see
        # config.yaml's "Object Segmentation": keys.input: panorama) -- 20% of
        # an entire panorama is a much larger absolute area, and a much less
        # object-shaped extent, than 20% of one tile, so a whole cliff face or
        # hillside comfortably survives a filter meant to catch exactly that.
        # Capping the reference area at one tile's worth restores the original
        # calibration regardless of how large the source image actually is; a
        # source smaller than one tile (the non-panoramic case this was
        # calibrated against) is unaffected since the cap never binds there.
        img_area = min(input_image.size[0] * input_image.size[1], _TILE_SIZE * _TILE_SIZE)
        min_area = cfg.min_area_fraction * img_area
        max_area = cfg.max_area_fraction * img_area

        # The photographed cone, when the detection source is the panorama. None
        # when the source is the photo itself (every pixel is real) or when there
        # are no intrinsics to locate the photo with -- the cone filter and the
        # hi-res re-cut are both skipped in that case, and the stage behaves as
        # it did before either existed.
        cone = None
        if context.input_panorama(input_key) is not None:
            cone = captured_cone_from_context(context, input_image.width, input_image.height)
            if cone is None:
                self.log_warning(
                    "Detecting on the panorama but no intrinsics/input photo to locate "
                    "the captured cone -- keeping detections from generated regions too"
                )
            else:
                u0, u1 = cone.u_range
                self.log_info(
                    f"Captured cone: {cone.hfov_deg:.1f}° x {cone.vfov_deg:.1f}° "
                    f"(panorama columns {u0:.0f}-{u1:.0f}), photo is "
                    f"{cone.scale_gain() * cfg.panorama_sr_factor:.2f}x the panorama's native resolution there"
                )
        # (index, panorama box) of every stored crop that lies in the cone, for
        # the hi-res re-cut after the loop.
        in_cone: list[tuple[int, list[float]]] = []
        photo_pass = cone is not None and cfg.detect_on_photo
        skipped_inside = 0

        def store_segmentation_result(result: SegmentationResult):
            nonlocal total_crops, skipped_inside

            # Cropping
            cropping_task = self.create_progress(result.length, "Cropping…")
            filtered = 0
            filtered_fov = 0
            for crop in result.masked_images(input_image):
                _, _, w, h = crop.box
                area = w * h
                if area < min_area or area > max_area:
                    filtered += 1
                    self.advance_progress(cropping_task)
                    continue

                # Flagged, not dropped: an out-of-cone detection is still useful
                # as an occluder for Panorama Foreground Inpainting (a generated
                # tree still has to come out of the terrain plate) and as a
                # tracking target. What it must not do is get PLACED or taught to
                # Distribution Synthesis as an exemplar -- Scene Generation, Object
                # Distribution and Panorama Asset Generation read the flag.
                inside = cone.contains_box(crop.box, cfg.captured_fov_margin_deg) if cone is not None else None
                if inside is False:
                    filtered_fov += 1
                elif inside and photo_pass:
                    # The photo pass already covered this ground at full resolution;
                    # the panorama's copy of the same object is a duplicate.
                    skipped_inside += 1
                    self.advance_progress(cropping_task)
                    continue

                # Indices must stay contiguous (no holes for filtered-out crops) --
                # downstream stages iterate range(OBJECT_COUNT) and assume every
                # index has a crop_{i}/metadata_{i} pair.
                i = total_crops
                context.add_image(f"crop_{i}", crop.image)
                self.log_info(f"  crop_{i}: box={[round(x, 1) for x in crop.box]} score={crop.score:.3f}")

                metadata = {
                    "box": [float(x) for x in crop.box],
                    "score": float(crop.score)
                }
                if inside is not None:
                    # Recorded either way, so a run with the filter off can still
                    # tell photographed detections from generated ones downstream.
                    metadata["in_captured_fov"] = bool(inside)
                    if inside:
                        in_cone.append((i, metadata["box"]))
                context.add_object(f"metadata_{i}", metadata)
                total_crops += 1
                self.advance_progress(cropping_task)
            self.finish_progress(cropping_task)
            if filtered:
                self.log_info(f"  Filtered {filtered} crop(s) outside size bounds ({result.length - filtered} kept)")
            if filtered_fov:
                self.log_info(
                    f"  {filtered_fov} crop(s) lie outside the photographed cone, {len(in_cone)} inside it"
                    + (" -- outside ones will not be placed or used as exemplars"
                       if cfg.restrict_to_captured_fov else "")
                )

        # Foreground Segmentation
        # foreground_segmenting_task = self.create_progress(2, "Foreground Segmenting…")
        # self.advance_progress(foreground_segmenting_task)

        # infill_count = 0
        # while True:
        #     if self._foreground_seg is None:
        #         self._foreground_seg = ForegroundSeg(self.device)
        #     result = self._foreground_seg.segment(input_image)

        #     if result.is_empty():
        #         break

        #     store_segmentation_result(result)

        #     if self._mask_inpainting is None:
        #         self._mask_inpainting = MaskInPainting(
        #             self.device,
        #             self.torch_dtype
        #         )

        #     for idx in range(result.length):
        #         full_mask = Image(result.masks[idx]) # self._prepare_mask_and_image(input_image, result.masks[idx], result.boxes[idx])
        #         input_image = self._mask_inpainting.inpaint_crop(input_image, mask_image=full_mask, box=result.boxes[idx])
        #         context.add_image(f"infill_img_{infill_count}", input_image)
        #         infill_count += 1

        #         print(f"Ran in-fill {idx}")
        #         self.log_memory_usage()

        #     break

        # self.advance_progress(foreground_segmenting_task)
        # self.finish_progress(foreground_segmenting_task)

        #Segmentation
        segmenting_task = self.create_progress(2, "Segmenting…")
        if self._seg is None:
            self._seg = ImageSeg(self.preferred_device)
        self.advance_progress(segmenting_task)

        if photo_pass:
            photo_img, factor = detection_photo(
                context, self.preferred_device, cfg.photo_detect_sr_factor, self.temp, self.log_info,
            )
            if photo_img is None:
                photo_pass = False
                self.log_warning("detect_on_photo is on but there is no input photo; detecting on the panorama only")
            else:
                photo_task = self.create_progress(1, "Segmenting the photograph…")
                photo_result = self._seg.segment(
                    photo_img, self.temp, on_progress=self.make_progress_callback(photo_task),
                )
                native_area = min(cone.photo_w * cone.photo_h, _TILE_SIZE * _TILE_SIZE)
                written, filtered_photo = store_photo_masks(
                    context, cone, photo_img.rgba(), factor,
                    photo_result.masks, photo_result.scores, total_crops,
                    min_area=cfg.min_area_fraction * native_area,
                    max_area=cfg.max_area_fraction * native_area,
                    source="sam2_photo", log=self.log_info,
                )
                total_crops += written
                self.finish_progress(photo_task)
                self.log_info(
                    f"Photo pass: {written} detection(s) from the photograph at {factor:.0f}x"
                    f" ({filtered_photo} outside size bounds)"
                )

        # The remote SAM2 call already reports real per-tile progress on
        # panorama-scale input (image_segmentation_imp.py's tiled path can be
        # a dozen+ full SAM2 passes) -- without on_progress here, that signal
        # was computed server-side and simply dropped, leaving the bar frozen
        # on this one step for the entire multi-minute call.
        scale = cfg.panorama_detect_scale if photo_pass else 1.0
        if 0.0 < scale < 1.0:
            small_size = (max(16, int(input_image.width * scale)), max(16, int(input_image.height * scale)))
            small = Image(input_image.rgb().resize(small_size, PILImage.LANCZOS))
            self.log_info(
                f"Panorama pass at {scale:.2f}x ({small_size[0]}x{small_size[1]} px): "
                "it only has to find occluders outside the cone"
            )
            small_result = self._seg.segment(
                small, self.temp, on_progress=self.make_progress_callback(segmenting_task),
            )
            full_w, full_h = input_image.width, input_image.height
            masks = [
                np.asarray(PILImage.fromarray(np.asarray(m, dtype=np.uint8) * 255).resize(
                    (full_w, full_h), PILImage.NEAREST,
                )) > 127
                for m in small_result.masks
            ]
            boxes = [[v / scale for v in b] for b in small_result.boxes]
            result = SegmentationResult(masks=masks, boxes=boxes, scores=small_result.scores)
        else:
            result = self._seg.segment(
                input_image, self.temp, on_progress=self.make_progress_callback(segmenting_task),
            )
        store_segmentation_result(result)
        if skipped_inside:
            self.log_info(
                f"  {skipped_inside} panorama detection(s) inside the cone dropped as duplicates of the photo pass"
            )

        self.advance_progress(segmenting_task)
        self.finish_progress(segmenting_task)

        if cone is not None and cfg.hires_crops_from_input and in_cone:
            photo = context.input_image(ContextKey.INPUT)
            hires_task = self.create_progress(1, "Re-cutting crops from the photo…")
            build_hires_crops(
                self._seg, context, photo, cone, in_cone, self.temp, self.log_info,
                sr_factor=cfg.panorama_sr_factor,
                on_progress=self.make_progress_callback(hires_task),
            )
            self.finish_progress(hires_task)

        if cone is not None:
            # Read by Scene Generation, Object Distribution and Panorama Asset
            # Generation: when True, a detection whose metadata says
            # in_captured_fov == False is neither placed nor used as an exemplar.
            context.add_object(CAPTURED_FOV_RESTRICTED_KEY, bool(cfg.restrict_to_captured_fov))

        context.add_object(output_key, total_crops)
        return context
    
    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return context.object(output_key) is not None

    def _prepare_mask_and_image(self, original_image: Image, small_mask: np.ndarray, box, radius: float = 5):
        x, y, w, h = box

        full_mask = PILImage.new("L", original_image.size, 0)
        small_mask_pil = Image(small_mask).L(copy=True).resize((w, h))
        full_mask.paste(small_mask, (x, y))    
        full_mask = full_mask.filter(ImageFilter.GaussianBlur(radius=radius))
    
        return Image(full_mask)

    def model_names(self) -> list[str]:
        return ImageSeg.model_names() + ForegroundSeg.model_names() + MaskInPainting.model_names()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        _, output_key = self._resolved_keys()
        count = context.object(output_key)
        if count is None:
            return None
        images = []
        for i in range(min(count, 8)):
            crop_img = context.image(f"crop_{i}")
            if crop_img is not None:
                meta = context.object(f"metadata_{i}") or {}
                label = meta.get("class") or f"Object {i + 1}"
                score = meta.get("score")
                cap = label if score is None else f"{label} ({score:.2f})"
                images.append((crop_img.image, cap))
        return ReportSection(
            stage_name=self.name,
            title="Object Segmentation",
            body=(
                "Foreground objects were detected and segmented from the input image. "
                "Each detected object is isolated as a masked crop for downstream 3D "
                "reconstruction. The segmentation uses SAM-based instance segmentation "
                "to produce clean per-object masks with associated bounding boxes and "
                "confidence scores."
            ),
            images=images,
            stats={"Objects detected": str(count)},
        )

    def clean_up(self):
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        self._foreground_seg = None
        self._mask_inpainting = None
        super().clean_up()