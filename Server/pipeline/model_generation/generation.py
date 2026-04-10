from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator
from pipeline.pipeline_context import PipelineContext
from util.image_utils import Image

class ModelGenerationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._gen = None

    def run(self, context: PipelineContext) -> PipelineContext:
        generation_task = self.create_progress(2, "Meshifying...")
        if self._gen is None:
            self._gen = ModelGenerator(self.device)
        self.advance_progress(generation_task)

        input_image = context.input_image("crop_3")
        mesh = self._gen.meshify(input_image)

        self.advance_progress(generation_task)
        self.finish_progress(generation_task)

        context.add_mesh("mesh", mesh)
        return context

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names()

    def clean_up(self):
        super().clean_up()
        self._gen = None