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
        count = context.input_object("count")

        segmenting_task = self.create_progress(count + 1, "Supersampling...")
        if self._samp is None:
            self._samp = SuperSample(self.device)
        self.advance_progress(segmenting_task)

        for i in range(count):
            input_image = context.input_image(f"crop_{i}")
            result = self._samp.supersample(input_image)

            context.add_image(f"crop_{i}", result)
            self.advance_progress(segmenting_task)
        self.finish_progress(segmenting_task)
        return context

    def model_names(self) -> list[str]:
        return SuperSample.model_names()

    def clean_up(self):
        super().clean_up()
        self._samp = None