from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator
from pipeline.pipeline_context import PipelineContext
from util.image_utils import Image

class ModelGenerationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        count = context.input_object("count")

        super().clean_up()
        gen = ModelGenerator(self.device)
        generation_task = self.create_progress(count, "Meshifying...")
        for idx in range(count):
            super().clean_up()
            image_name = f"crop_{idx}"
            input_image = context.input_image(image_name)
            mesh = gen.meshify(input_image, self.config.temp / image_name)

            self.advance_progress(generation_task)
            context.add_mesh(f"mesh_{idx}", mesh)

            print(f"Generated mesh for {image_name} vertices={mesh.vertex_count} faces={mesh.face_count}")
        
        gen.close()
        self.finish_progress(generation_task)

        return context

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names()

    def clean_up(self):
        super().clean_up()