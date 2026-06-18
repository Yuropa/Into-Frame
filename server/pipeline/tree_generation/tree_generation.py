import torch
from logging import Logger
from typing import Any

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import CategoryFilter, VEGETATION_CATEGORIES
from pipeline.tree_generation.model_generation_treed import ModelGeneratorTreeD
from util.device_utils import DeviceStrategy, preferred_device


class TreeMeshGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Default to vegetation only when no filter is specified.
        if include_categories is None and exclude_categories is None:
            include_categories = list(VEGETATION_CATEGORIES)
        self.category_filter = CategoryFilter(include_categories, exclude_categories)


class TreeMeshGenerationStage(PipelineStage):
    """
    Generates 3D meshes for vegetation objects (trees, plants, etc.) using
    Tree-D Fusion (TreeDFusion_ECCV24). Crops from SegmentationStage are already
    RGBA-masked, so no additional segmentation is needed.

    By default processes only vegetation categories (tree, forest, bush, flower, plant).
    Override include_categories or exclude_categories in config to change the filter.

    Reads:  ContextKey.OBJECT_COUNT, crop_{i}, metadata_{i}
    Writes: mesh_{i} for each object that passes the category filter
    """

    @classmethod
    def config_class(cls):
        return TreeMeshGenerationConfiguration

    def __init__(self, config: TreeMeshGenerationConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def run(self, context: PipelineContext) -> PipelineContext:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if not count:
            self.log_info("No objects to process, skipping")
            return context

        to_generate = []
        for idx in range(count):
            metadata = context.input_object(f"metadata_{idx}") or {}
            obj_class = metadata.get("class", "")
            if not self.config.category_filter.allows(obj_class):
                self.log_info(f"  crop_{idx}: '{obj_class}' — excluded by category filter")
                continue
            if context.input_mesh(f"mesh_{idx}") is not None:
                self.log_info(f"  crop_{idx}: mesh already cached")
                continue
            to_generate.append((idx, obj_class))

        if not to_generate:
            self.log_info("No matching objects to meshify")
            return context

        gen_task = self.create_progress(len(to_generate), "Meshifying trees…")
        super().clean_up()
        gen = ModelGeneratorTreeD(self.preferred_device)

        for idx, obj_class in to_generate:
            crop = context.input_image(f"crop_{idx}")
            temp_path = self.temp / f"crop_{idx}" if self.temp is not None else None
            self.log_info(f"  crop_{idx}: '{obj_class}' → Tree-D Fusion")
            if temp_path is not None:
                self.log_info(f"  crop_{idx}: Magic123 log → {temp_path / 'magic123.log'}")
            super().clean_up()
            mesh = gen.meshify(crop, temp_path, seed=self.seed, species=obj_class)
            mesh = mesh.repair()
            context.add_mesh(f"mesh_{idx}", mesh)
            self.log_info(f"  crop_{idx}: {mesh.vertex_count}v {mesh.face_count}f")
            self.advance_progress(gen_task)

        gen.close()
        self.finish_progress(gen_task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            return False
        for idx in range(count):
            metadata = context.object(f"metadata_{idx}")
            if metadata is None:
                continue
            obj_class = metadata.get("class", "")
            if not self.config.category_filter.allows(obj_class):
                continue
            if context.mesh(f"mesh_{idx}") is None:
                return False
        return True

    def clean_up(self):
        super().clean_up()
