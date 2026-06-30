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
    Generates one 3D mesh per object category using a reconstruction model.

    For each distinct category present in the scene, one representative crop is
    chosen and meshified. The result is stored as category_mesh_{class} and
    normalised to a 1×1 canonical box so SceneGenerationStage can scale each
    instance independently.

    Reads dynamic context keys per object (index i):
      crop_{i}      → Image  (textured object crop from SegmentationStage)
      metadata_{i}  → object ({"class": str, ...})

    Writes one key per category:
      category_mesh_{class} → Mesh

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

        # Group object indices by category, respecting the category filter.
        category_to_indices: dict[str, list[int]] = {}
        for idx in range(count):
            metadata = context.input_object(f"metadata_{idx}") or {}
            obj_class = metadata.get("class", "")
            if not self.config.category_filter.allows(obj_class):
                continue
            category_to_indices.setdefault(obj_class, []).append(idx)

        gen_type = getattr(self.config, "generator_type", ModelGeneratorType.default())
        super().clean_up()
        gen = ModelGenerator(self.preferred_device, type=gen_type)
        generation_task = self.create_progress(len(category_to_indices), "Meshifying…")

        try:
            for obj_class, indices in category_to_indices.items():
                mesh_key = f"category_mesh_{obj_class}"

                cached_mesh = context.mesh(mesh_key)
                if cached_mesh is not None:
                    self.log_info(f"  {mesh_key}: cached ({cached_mesh.vertex_count}v {cached_mesh.face_count}f)")
                    self.advance_progress(generation_task)
                    continue

                representative_idx = indices[0]
                image_name = f"crop_{representative_idx}"

                super().clean_up()
                input_image = context.input_image(image_name)
                self.log_info(f"  {mesh_key}: generating mesh from {image_name}…")
                mesh = gen.meshify(input_image, self.temp / mesh_key, seed=self.seed)
                mesh = mesh.repair()
                mesh.fit_to_box(1.0, 1.0)

                context.add_mesh(mesh_key, mesh)
                self.advance_progress(generation_task)
                self.log_info(f"  {mesh_key}: {mesh.vertex_count}v {mesh.face_count}f")
        finally:
            gen.close()

        self.finish_progress(generation_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object("count")
        if count is None:
            return False
        seen_classes: set[str] = set()
        for idx in range(count):
            metadata = context.object(f"metadata_{idx}")
            obj_class = (metadata or {}).get("class", "")
            if not self.config.category_filter.allows(obj_class):
                continue
            if obj_class not in seen_classes:
                seen_classes.add(obj_class)
                if context.mesh(f"category_mesh_{obj_class}") is None:
                    return False
        return True

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        count = context.object("count")
        if count is None or count == 0:
            return None

        # Collect one representative crop per category for display.
        seen_classes: set[str] = set()
        images = []
        total_verts = 0
        total_faces = 0

        for i in range(count):
            meta = context.object(f"metadata_{i}") or {}
            obj_class = meta.get("class", f"Object {i + 1}")
            mesh_key = f"category_mesh_{obj_class}"
            mesh = context.mesh(mesh_key)
            crop = context.image(f"crop_{i}")

            if obj_class not in seen_classes:
                seen_classes.add(obj_class)
                if mesh is not None:
                    total_verts += mesh.vertex_count
                    total_faces += mesh.face_count
                if crop is not None and len(images) < 6:
                    label = obj_class
                    if mesh is not None:
                        label += f" ({mesh.vertex_count:,}v)"
                    images.append((crop.image, label))

        reconstructed = len(seen_classes)
        stats = {"Categories reconstructed": str(reconstructed)}
        if reconstructed > 0:
            stats["Total vertices"] = f"{total_verts:,}"
            stats["Total triangles"] = f"{total_faces:,}"
            stats["Generator"] = self.config.generator_type.name
        return ReportSection(
            stage_name=self.name,
            title="3D Object Reconstruction",
            body=(
                "One 3D mesh is generated per object category using a single-image "
                "reconstruction model. The "
                f"{self.config.generator_type.name} model lifts a representative 2D "
                "crop into a full 3D asset with UV-mapped texture, then applies mesh "
                "repair and normalises to a canonical 1 m box. At scene placement each "
                "instance randomly uses the category mesh (with a random rotation) or "
                "a billboard drawn from the category's crop pool."
            ),
            images=images,
            stats=stats,
        )

    def clean_up(self):
        super().clean_up()