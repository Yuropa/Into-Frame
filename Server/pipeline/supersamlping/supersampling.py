from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.supersamlping.image_supersampling import SuperSample
from pipeline.pipeline_context import PipelineContext
from util.image_utils import Image
import numpy as np
import PIL

class SupersamplingStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._samp = None

    def run(self, context: PipelineContext) -> PipelineContext:
        segmenting_task = self.create_progress(2, "Supersampling...")
        if self._samp is None:
            self._samp = SuperSample(self.device)
        self.advance_progress(segmenting_task)

        result = self._samp.supersample(context.input_image)

        context.add_image("image", result)
        self.finish_progress(segmenting_task)
        return context

    def model_names(self) -> list[str]:
        return SuperSample.model_names()

    def clean_up(self):
        super().clean_up()
        self._samp = None