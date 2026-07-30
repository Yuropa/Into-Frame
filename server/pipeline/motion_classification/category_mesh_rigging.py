from logging import Logger
from typing import Any

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import VEGETATION_CATEGORIES
from scene.object import ObjectType

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
        rig_categories=VEGETATION_CATEGORIES,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Only categories where wind sway actually makes sense -- rigging e.g. a
        # bench or fire hydrant would just be a skeleton nothing ever moves.
        #
        # It is also a COST control, which is the less obvious half. A rigged mesh
        # becomes a SkinnedMeshRenderer on the client: skinned every frame, no GPU
        # instancing, no static batching, plus a WindSway.LateUpdate writing three bone
        # transforms per instance. That is fine for the few dozen trees and flowers it
        # was designed around and ruinous for ground cover -- GrassCoverStage alone
        # places 6,661 instances, so including grass_tuft here turns the whole scene
        # into ~6,700 skinned meshes. Drop it from this list and grass renders as plain
        # (static) MeshRenderers instead.
        #
        # Coerced from whatever config.yaml supplies, which is a list.
        self.rig_categories = frozenset(rig_categories)


class CategoryMeshRiggingStage(PipelineStage):
    """
    Bakes a 3-bone vertical sway skeleton into each shared bucket mesh
    (PanoramaAssetGenerationStage's category_mesh_{class}_{bucket}, one visual-
    similarity variant within a class -- see ObjectCategoryClusteringStage) that
    has at least one stationary placed instance rendering it as a mesh -- once per
    mesh asset, since every instance sharing that bucket renders the same GLB.
    Per-instance sway timing (amplitude/frequency/phase) is attached separately, to
    each instance's own Object3D, by SceneAnimationStage; this stage only bakes the
    shared geometry's skin (joints/weights/inverseBindMatrices), not any animation
    values.

    Reads the already-built Scene rather than re-deriving which (class, bucket)
    groups exist and re-guessing their mesh key -- Object3D.mesh already names
    exactly the asset a MESH-type placement uses, however that key is built
    upstream, and Object3D.source_index already says which detection it is. This
    also naturally skips a bucket that clustering/LOD decided to render as a
    billboard everywhere in this scene (its mesh, if one even exists, never shows
    up as any placed Object3D.mesh here, so nothing rigs it).

    See util.gltf_skin.inject_skin for the actual GLB skin injection -- it runs
    lazily, inside Mesh.save(), whenever Mesh.skin_bone_heights is set (mirroring
    how Mesh.extra_uv/inject_texcoord1 already works), not here directly: the
    GLB file for a mesh asset doesn't get exported until something actually asks
    to serve or cache it, long after this stage returns.

    Reads:  ContextKey.SCENE (to find which mesh assets are actually placed and by
            which detection), metadata_{i} (needs "class" and, from
            ObjectMotionClassificationStage, "stationary")
    Writes: <mesh asset> (Mesh, with skin_bone_heights/skin_bone_names set) for
            every qualifying mesh asset
    Output key (SemanticKey.OUTPUT) -> ContextKey.CATEGORY_MESH_RIGGING_COUNT (int)
    """

    @classmethod
    def config_class(cls) -> type[CategoryMeshRiggingConfiguration]:
        return CategoryMeshRiggingConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        scene = context.input_scene(ContextKey.SCENE)
        if scene is None or not scene.objects:
            self.log_info("No scene, skipping")
            return context

        mesh_keys_to_rig: set[str] = set()
        for obj in scene.objects:
            if obj.type != ObjectType.MESH or obj.source_index is None:
                continue
            metadata = context.input_object(f"metadata_{obj.source_index}") or {}
            if metadata.get("class") in self.config.rig_categories and metadata.get("stationary"):
                mesh_keys_to_rig.add(obj.mesh)

        if not mesh_keys_to_rig:
            self.log_info("No stationary vegetation instances to rig")
            context.add_object(ContextKey.CATEGORY_MESH_RIGGING_COUNT, 0)
            return context

        rigged = 0
        task = self.create_progress(len(mesh_keys_to_rig), "Rigging category meshes…")
        for mesh_key in sorted(mesh_keys_to_rig):
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
                "Shared bucket meshes with at least one stationary placed instance "
                "got a 3-bone vertical sway skeleton baked into their GLB, driven "
                "procedurally at runtime rather than a baked animation clip -- the "
                "same mesh asset is reused by every instance of that bucket, each "
                "with its own sway timing."
            ),
            stats={"Meshes rigged": str(count)},
        )
