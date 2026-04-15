from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.segmentation.foreground_segmentation import ForegroundSeg
from pipeline.segmentation.segmentation_result import SegmentationResult
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.mask_inpainting import MaskInPainting
from util.image_utils import Image
import numpy as np
from PIL import Image as PILImage
from PIL import ImageOps

class SegmentationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._seg = None
        self._foreground_seg = None
        self._mask_inpainting = None

    def run(self, context: PipelineContext) -> PipelineContext:
        input_image = context.input_image(ContextKey.INPUT).copy()
        total_crops = 0

        def store_segmentation_result(result: SegmentationResult):
            nonlocal total_crops

            # Cropping
            cropping_task = self.create_progress(result.length, "Cropping...")
            for idx, crop in enumerate(result.masked_images(input_image)):
                i = total_crops + idx
                context.add_image(f"crop_{i}", crop.image)

                print(f"Crop {crop.box}")
                metadata = {
                    "box": [float(x) for x in crop.box],
                    "score": float(crop.score)
                }
                context.add_object(f"metadata_{i}", metadata)
                self.advance_progress(cropping_task)
            self.finish_progress(cropping_task)
            total_crops += result.length

        # Foreground Segmentation
        foreground_segmenting_task = self.create_progress(2, "Foreground Segmenting...")
        if self._foreground_seg is None:
            self._foreground_seg = ForegroundSeg(self.device)
        self.advance_progress(foreground_segmenting_task)

        infill_count = 0
        while True:
            result = self._foreground_seg.segment(input_image)
            if result.is_empty():
                break

            store_segmentation_result(result)

            if self._mask_inpainting is None:
                self._mask_inpainting = MaskInPainting(
                    self.device,
                    self.torch_dtype
                )

            for idx in range(result.length):
                full_mask = self._prepare_mask_and_image(input_image, result.masks[idx], result.boxes[idx])
                input_image = self._mask_inpainting.inpaint(input_image, mask_image=full_mask)
                context.add_image(f"infill_img_{infill_count}", input_image)
                infill_count += 1

        self.advance_progress(foreground_segmenting_task)
        self.finish_progress(foreground_segmenting_task)

        #Segmentation
        segmenting_task = self.create_progress(2, "Segmenting...")
        if self._seg is None:
            self._seg = ImageSeg(self.device)
        self.advance_progress(segmenting_task)

        result = self._seg.segment(input_image)
        store_segmentation_result(result)

        self.advance_progress(segmenting_task)
        self.finish_progress(segmenting_task)

        context.add_object("count", total_crops)
        return context
    
    def _prepare_mask_and_image(self, original_image: Image, small_mask: Image, box, radius: float = 5):
        x, y, w, h = box
        
        full_mask = PILImage.new("L", original_image.size, 0)
        small_mask = small_mask.image.convert("L")
        full_mask.paste(small_mask, (x, y))    
        full_mask = full_mask.filter(ImageFilter.GaussianBlur(radius=radius))
    
        return Image(full_mask)

    def model_names(self) -> list[str]:
        return ImageSeg.model_names() + ForegroundSeg.model_names() + MaskInPainting.model_names()

    def clean_up(self):
        super().clean_up()
        self._seg = None
        self._foreground_seg = None
        self._mask_inpainting = None