from logging import Logger
from typing import Any

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import VEGETATION_CATEGORIES

_BONE_NAMES = ["SwayBone_0", "SwayBone_1", "SwayBone_2"]


class CategoryMeshRiggingConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        rig_categories: frozenset[str] = VEGETATION_CATEGORIES,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Only categories where wind sway actually makes sense -- rigging e.g. a
        # bench or fire hydrant would just be a skeleton nothing ever moves.
        self.rig_categories = rig_categories


class CategoryMeshRiggingStage(PipelineStage):
    """
    Bakes a 3-bone vertical sway skeleton into each shared category mesh
    (PanoramaAssetGenerationStage's category_mesh_{cls}) that has at least one
    stationary placed instance -- once per category, since every instance of
    that category renders the same shared mesh asset. Per-instance sway timing
    (amplitude/frequency/phase) is attached separately, to each instance's own
    Object3D, by SceneAnimationStage; this stage only bakes the shared geometry's
    skin (joints/weights/inverseBindMatrices), not any animation values.

    See util.gltf_skin.inject_skin for the actual GLB skin injection -- it runs
    lazily, inside Mesh.save(), whenever Mesh.skin_bone_heights is set (mirroring
    how Mesh.extra_uv/inject_texcoord1 already works), not here directly: the
    GLB file for a category mesh doesn't get exported until something actually
    asks to serve or cache that asset, long after this stage returns.

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i} (needs "class" and, from
            ObjectMotionClassificationStage, "stationary"), category_mesh_{cls}
    Writes: category_mesh_{cls} (Mesh, with skin_bone_heights/skin_bone_names set)
            for every qualifying category
    Output key (SemanticKey.OUTPUT) -> ContextKey.CATEGORY_MESH_RIGGING_COUNT (int)
    """

    @classmethod
    def config_class(cls) -> type[CategoryMeshRiggingConfiguration]:
        return CategoryMeshRiggingConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects, skipping")
            return context

        categories_to_rig: set[str] = set()
        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}
            cls = metadata.get("class")
            if cls in self.config.rig_categories and metadata.get("stationary"):
                categories_to_rig.add(cls)

        if not categories_to_rig:
            self.log_info("No stationary vegetation instances to rig")
            context.add_object(ContextKey.CATEGORY_MESH_RIGGING_COUNT, 0)
            return context

        rigged = 0
        task = self.create_progress(len(categories_to_rig), "Rigging category meshes…")
        for cls in sorted(categories_to_rig):
            mesh_key = f"category_mesh_{cls}"
            mesh = context.input_mesh(mesh_key)
            if mesh is None:
                self.advance_progress(task)
                continue

            y_min = float(mesh.mesh.bounds[0][1])
            y_max = float(mesh.mesh.bounds[1][1])
            if y_max <= y_min:
                self.advance_progress(task)
                continue

            bone_heights = [y_min, y_min + (y_max - y_min) * 0.5, y_max]
            mesh.skin_bone_heights = bone_heights
            mesh.skin_bone_names = _BONE_NAMES
            context.add_mesh(mesh_key, mesh)

            self.log_info(f"  {mesh_key}: rigged 3-bone sway skeleton")
            rigged += 1
            self.advance_progress(task)
        self.finish_progress(task)

        context.add_object(ContextKey.CATEGORY_MESH_RIGGING_COUNT, rigged)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(ContextKey.CATEGORY_MESH_RIGGING_COUNT) is not None

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        count = context.object(ContextKey.CATEGORY_MESH_RIGGING_COUNT)
        if count is None:
            return None
        return ReportSection(
            stage_name=self.name,
            title="Category Mesh Rigging",
            body=(
                "Shared category meshes with at least one stationary placed instance "
                "got a 3-bone vertical sway skeleton baked into their GLB, driven "
                "procedurally at runtime rather than a baked animation clip -- the "
                "same mesh asset is reused by every instance of that category, each "
                "with its own sway timing."
            ),
            stats={"Category meshes rigged": str(count)},
        )
