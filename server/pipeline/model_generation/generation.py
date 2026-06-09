from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator, ModelGeneratorType
from pipeline.pipeline_context import PipelineContext
from util.device_utils import DeviceStrategy, preferred_device


class ModelGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        generator_type: str = "SAM3D",
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.generator_type = ModelGeneratorType[generator_type.upper()]


class ModelGenerationStage(PipelineStage):
    """
    Generates a 3D mesh for each segmented object crop using a reconstruction model.

    Reads dynamic context keys per object (index i):
      crop_{i}   → Image  (textured object crop from SegmentationStage)

    Writes dynamic context keys per object (index i):
      mesh_{i}   → Mesh   (generated GLB mesh)

    Also reads: count (object) → number of crops to process
    """

    @classmethod
    def config_class(cls):
        return ModelGenerationConfiguration

    def __init__(self, config: ModelGenerationConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def run(self, context: PipelineContext) -> PipelineContext:
        count = context.input_object("count")

        gen_type = getattr(self.config, "generator_type", ModelGeneratorType.default())
        super().clean_up()
        gen = ModelGenerator(self.preferred_device, type=gen_type)
        generation_task = self.create_progress(count, "Meshifying...")
        for idx in range(count):
            mesh_name = f"mesh_{idx}"   
            image_name = f"crop_{idx}"

            cached_mesh = context.mesh(mesh_name)
            if cached_mesh is not None:
                self.log_info(f"  {image_name}: cached ({cached_mesh.vertex_count}v {cached_mesh.face_count}f)")
                self.advance_progress(generation_task)
                continue

            super().clean_up()
            input_image = context.input_image(image_name)
            mesh = gen.meshify(input_image, self.temp / image_name, seed=self.seed)

            self.advance_progress(generation_task)
            context.add_mesh(mesh_name, mesh)
            self.log_info(f"  {image_name}: {mesh.vertex_count}v {mesh.face_count}f")
        
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