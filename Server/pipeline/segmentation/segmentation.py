from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext
from util.image_utils import Image
import numpy as np
import PIL

class SegmentationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._seg = None

    def run(self, context: PipelineContext) -> PipelineContext:
        segmenting_task = self.create_progress(2, "Segmenting...")
        if self._seg is None:
            self._seg = ImageSeg(self.device)
        self.advance_progress(segmenting_task)

        result = self._seg.segment(context.input_image)

        self._captioning = ImageCaptioning(self.device)
        self.advance_progress(segmenting_task)
        self.finish_progress(segmenting_task)

        captioning_task = self.create_progress(len(result.masks), "Captioning...")
        for i, crop in enumerate(result.masked_images(context.input_image)):
            context.add_image(f"crop_{i}", crop.image)

            captioning_image = self._create_captioning_image(context.input_image, crop.mask, crop.box)
            label = self._captioning.caption(captioning_image)

            metadata = {
                "box": [float(x) for x in crop.box],
                "score": float(crop.score),
                "label": label
            }
            context.add_object(f"metadata_{i}", metadata)
            context.add_image(f"masked_{i}", captioning_image)
            self.advance_progress(captioning_task)

        context.add_object("count", len(result.masks))
        self.finish_progress(captioning_task)
        return context

    def model_names(self) -> list[str]:
        return ImageSeg.model_names() + ImageCaptioning.model_names()

    def clean_up(self):
        super().clean_up()
        self._seg = None
        self._captioning = None

    def _create_captioning_image(self, original, mask, box):
        x1, y1, x2, y2 = [int(x) for x in box]
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        original = original.image
        background = original.copy().point(lambda p: p * 0.3)
        result = background.copy()
        result.paste(original, mask=mask)
        result = result.crop((x1, y1, x2, y2))
        return Image(result)