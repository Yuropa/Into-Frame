from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator
from pipeline.pipeline_context import PipelineContext
from util.image_utils import Image

class ModelGenerationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        count = context.input_object("count")

        generation_task = self.create_progress(count, "Meshifying...")
        for idx in range(count):
            image_name = f"crop_{idx}"
            input_image = context.input_image(image_name)
            gen = ModelGenerator(self.device, self.config.temp / image_name)
            mesh = gen.meshify(input_image)

            self.advance_progress(generation_task)
            context.add_mesh(f"mesh_{idx}", mesh)
            gen.close()

            print(f"Generated mesh for {image_name} vertices={mesh.vertex_count} faces={mesh.face_count}")
        self.finish_progress(generation_task)

        return context

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names()

    def clean_up(self):
        super().clean_up()