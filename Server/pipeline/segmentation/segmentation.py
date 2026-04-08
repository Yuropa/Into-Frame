from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.segmentation.image_segmentation import ImageSeg
from pipeline.pipeline_context import PipelineContext

class SegementationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._seg = None

    def run(self, context: PipelineContext) -> PipelineContext:
        if self._seg is None:
            self._seg = ImageSeg(self.device)

        result = self._seg.segment(context.input_image)

        for i, crop in enumerate(result.masked_images(context.input_image)):
            context.add_image(f"crop_{i}", crop)

        return context

    def model_names(self) -> list[str]:
        return ImageSeg.model_names()

    def clean_up(self):
        super().clean_up()
        self._seg = None