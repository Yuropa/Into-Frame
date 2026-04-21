from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator
from pipeline.pipeline_context import PipelineContext
from util.image_utils import Image

class ModelGenerationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        generation_task = self.create_progress(2, "Meshifying...")
        self.advance_progress(generation_task)

        image_name = "crop_3"
        input_image = context.input_image(image_name)
        gen = ModelGenerator(self.device, self.config.temp / image_name)
        mesh = gen.meshify(input_image)

        self.advance_progress(generation_task)
        self.finish_progress(generation_task)

        context.add_mesh("mesh", mesh)
        return context

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names()

    def clean_up(self):
        super().clean_up()