from logging import Logger
from typing import Any

import torch
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.foreground_segmentation import ForegroundSeg
from pipeline.segmentation.segmentation_result import SegmentationResult
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.mask_inpainting import MaskInPainting
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps


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
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction


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
        img_area = input_image.size[0] * input_image.size[1]
        min_area = cfg.min_area_fraction * img_area
        max_area = cfg.max_area_fraction * img_area

        def store_segmentation_result(result: SegmentationResult):
            nonlocal total_crops

            # Cropping
            cropping_task = self.create_progress(result.length, "Cropping…")
            filtered = 0
            for crop in result.masked_images(input_image):
                _, _, w, h = crop.box
                area = w * h
                if area < min_area or area > max_area:
                    filtered += 1
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
                context.add_object(f"metadata_{i}", metadata)
                total_crops += 1
                self.advance_progress(cropping_task)
            self.finish_progress(cropping_task)
            if filtered:
                self.log_info(f"  Filtered {filtered} crop(s) outside size bounds ({result.length - filtered} kept)")

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

        result = self._seg.segment(input_image, self.temp)
        store_segmentation_result(result)

        self.advance_progress(segmenting_task)
        self.finish_progress(segmenting_task)

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