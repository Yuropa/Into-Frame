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

class SegmentationStage(PipelineStage):
    """
    Detects and segments foreground objects in the input image, storing each
    object's masked crop and bounding-box metadata for downstream stages.

    Input key  (SemanticKey.INPUT)  → ContextKey.INPUT          (Image)
    Output key (SemanticKey.OUTPUT) → ContextKey.OBJECT_COUNT    (int)

    Dynamic context keys per detected object (index i):
      crop_{i}      → Image  (masked object crop)
      metadata_{i}  → object ({"box": [...], "score": float})
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
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

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, output_key = self._resolved_keys()

        input_image = context.input_image(input_key).copy()
        total_crops = 0

        def store_segmentation_result(result: SegmentationResult):
            nonlocal total_crops

            # Cropping
            cropping_task = self.create_progress(result.length, "Cropping…")
            for idx, crop in enumerate(result.masked_images(input_image)):
                i = total_crops + idx
                context.add_image(f"crop_{i}", crop.image)
                self.log_info(f"  crop_{i}: box={[round(x, 1) for x in crop.box]} score={crop.score:.3f}")

                metadata = {
                    "box": [float(x) for x in crop.box],
                    "score": float(crop.score)
                }
                context.add_object(f"metadata_{i}", metadata)
                self.advance_progress(cropping_task)
            self.finish_progress(cropping_task)
            total_crops += result.length

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

    def clean_up(self):
        if self._seg is not None:
            self._seg.close()
            self._seg = None
        self._foreground_seg = None
        self._mask_inpainting = None
        super().clean_up()