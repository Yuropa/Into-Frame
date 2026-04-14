from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext, ContextKey
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

        input_image = context.input_image(ContextKey.INPUT)
        result = self._seg.segment(input_image)

        self._captioning = ImageCaptioning(self.device)
        self.advance_progress(segmenting_task)
        self.finish_progress(segmenting_task)

        captioning_task = self.create_progress(len(result.masks), "Captioning...")
        for i, crop in enumerate(result.masked_images(input_image)):
            context.add_image(f"crop_{i}", crop.image)

            metadata = {
                "box": [float(x) for x in crop.box],
                "score": float(crop.score)
            }
            context.add_object(f"metadata_{i}", metadata)
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