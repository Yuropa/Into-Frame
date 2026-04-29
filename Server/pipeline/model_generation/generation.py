from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator
from pipeline.pipeline_context import PipelineContext
from util.device_utils import DeviceStrategy, preferred_device

class ModelGenerationStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)

        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def run(self, context: PipelineContext) -> PipelineContext:
        count = context.input_object("count")

        super().clean_up()
        gen = ModelGenerator(self.preferred_device)
        generation_task = self.create_progress(count, "Meshifying...")
        for idx in range(count):
            mesh_name = f"mesh_{idx}"   
            image_name = f"crop_{idx}"

            cached_mesh = context.mesh(mesh_name)
            if cached_mesh is not None:
                # Already cached
                print(f"Using cached mesh for {image_name} vertices={cached_mesh.vertex_count} faces={cached_mesh.face_count}")
                self.advance_progress(generation_task)
                continue

            super().clean_up()
            input_image = context.input_image(image_name)
            mesh = gen.meshify(input_image, self.config.temp / image_name)

            self.advance_progress(generation_task)
            context.add_mesh(mesh_name, mesh)

            print(f"Generated mesh for {image_name} vertices={mesh.vertex_count} faces={mesh.face_count}")
        
        gen.close()
        self.finish_progress(generation_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object("count")
        if count is None:
            return False
        return all(context.mesh(f"mesh_{idx}") is not None for idx in range(count))

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names()

    def clean_up(self):
        super().clean_up()