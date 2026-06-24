from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator, ModelGeneratorType
from pipeline.object_typing.categories import CategoryFilter
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
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.generator_type = ModelGeneratorType[generator_type.upper()]
        self.category_filter = CategoryFilter(include_categories, exclude_categories)


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
        generation_task = self.create_progress(count, "Meshifying…")
        for idx in range(count):
            mesh_name = f"mesh_{idx}"
            image_name = f"crop_{idx}"

            metadata = context.input_object(f"metadata_{idx}") or {}
            obj_class = metadata.get("class", "")
            if not self.config.category_filter.allows(obj_class):
                self.log_info(f"  {image_name}: '{obj_class}' — excluded by category filter")
                self.advance_progress(generation_task)
                continue

            cached_mesh = context.mesh(mesh_name)
            if cached_mesh is not None:
                self.log_info(f"  {image_name}: cached ({cached_mesh.vertex_count}v {cached_mesh.face_count}f)")
                self.advance_progress(generation_task)
                continue

            super().clean_up()
            input_image = context.input_image(image_name)
            self.log_info(f"  {image_name}: generating mesh…")
            mesh = gen.meshify(input_image, self.temp / image_name, seed=self.seed)
            mesh = mesh.repair()

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
        for idx in range(count):
            metadata = context.object(f"metadata_{idx}")
            obj_class = (metadata or {}).get("class", "")
            if not self.config.category_filter.allows(obj_class):
                continue
            if context.mesh(f"mesh_{idx}") is None:
                return False
        return True

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        count = context.object("count")
        if count is None or count == 0:
            return None
        images = []
        total_verts = 0
        total_faces = 0
        reconstructed = 0
        for i in range(count):
            mesh = context.mesh(f"mesh_{i}")
            crop = context.image(f"crop_{i}")
            meta = context.object(f"metadata_{i}") or {}
            obj_class = meta.get("class", f"Object {i + 1}")
            if mesh is not None:
                total_verts += mesh.vertex_count
                total_faces += mesh.face_count
                reconstructed += 1
            if crop is not None and len(images) < 6:
                label = obj_class
                if mesh is not None:
                    label += f" ({mesh.vertex_count:,}v)"
                images.append((crop.image, label))
        stats = {"Objects reconstructed": str(reconstructed)}
        if reconstructed > 0:
            stats["Total vertices"] = f"{total_verts:,}"
            stats["Total triangles"] = f"{total_faces:,}"
            stats["Avg vertices / object"] = f"{total_verts // reconstructed:,}"
            stats["Generator"] = self.config.generator_type.name
        return ReportSection(
            stage_name=self.name,
            title="3D Object Reconstruction",
            body=(
                "Each segmented foreground object was individually reconstructed as a "
                "textured 3D mesh using a single-image 3D reconstruction model. The "
                f"{self.config.generator_type.name} model lifts each 2D crop into a "
                "full 3D asset with UV-mapped texture, then applies mesh repair to "
                "produce watertight geometry suitable for real-time rendering. Meshes "
                "are placed in the scene using depth-derived position estimates."
            ),
            images=images,
            stats=stats,
        )

    def clean_up(self):
        super().clean_up()