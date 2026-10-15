"""
Material Maps: PBR textures for every mesh asset the scene places.

Asset generation leaves each mesh with a base colour and nothing else, or, from
TRELLIS, with a metallic-roughness texture that makes meadow flowers polished
metal (metallic 0.74, roughness 0.19 on the Rainier lupine). glTFast builds a
glTF PBR material for every GLB and reads normal, occlusion and metallic-
roughness textures natively, so writing those into the GLB is all the client
needs.

For each mesh asset a placed object uses (its own mesh and every LOD mesh),
this stage derives, with util.material_maps:

- a tangent-space normal map from the base colour's detail;
- ambient occlusion: cavity occlusion from the same detail times geometric
  occlusion baked from depth renders in many directions and rasterised into
  the mesh's UV layout (geometry only on per-face atlases, image only on cards);
- roughness from the object's class, rougher in crevices, and metallic zero for
  natural classes (kept, never raised, from the source otherwise).

Placed after Scene Generation, which decides which mesh keys are used, and
before Category Mesh Rigging, which injects its skin into the glTF JSON and
leaves materials untouched.

Reads:  ContextKey.SCENE, metadata_{i} ("class"), the mesh assets it names
Writes: every processed mesh asset under its own key; material_maps_summary
"""
from logging import Logger
from typing import Any

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from scene.object import ObjectType

SUMMARY_KEY = "material_maps_summary"


class MaterialMapsConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        normal_strength: float = 2.0,
        cavity_strength: float = 0.5,
        ao_directions: int = 64,
        ao_resolution: int = 512,
        # Meshes with fewer faces than this are cards (alpha-cut quads): no
        # geometric occlusion, only the image-derived maps.
        min_faces_for_geometric_ao: int = 100,
        max_texture_size: int = 1024,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.normal_strength = float(normal_strength)
        self.cavity_strength = float(cavity_strength)
        self.ao_directions = int(ao_directions)
        self.ao_resolution = int(ao_resolution)
        self.min_faces_for_geometric_ao = int(min_faces_for_geometric_ao)
        self.max_texture_size = int(max_texture_size)


class MaterialMapsStage(PipelineStage):
    @classmethod
    def config_class(cls) -> type[MaterialMapsConfiguration]:
        return MaterialMapsConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        from trimesh.visual.material import PBRMaterial
        from util.material_maps import mesh_material_maps, vertex_occlusion

        cfg: MaterialMapsConfiguration = self.config
        scene = context.input_scene(ContextKey.SCENE)
        if scene is None or not scene.objects:
            self.log_info("No scene, skipping")
            context.add_object(SUMMARY_KEY, {"meshes": 0})
            return context

        # mesh key -> object class (first placement that names it wins)
        targets: dict[str, str] = {}
        for obj in scene.objects:
            if obj.type != ObjectType.MESH:
                continue
            metadata = context.input_object(f"metadata_{obj.source_index}") or {} if obj.source_index is not None else {}
            cls = str(metadata.get("class") or "")
            keys = [obj.mesh] + [lod.get("mesh") for lod in (obj.lods or []) if lod.get("mesh")]
            for key in keys:
                if key and key.startswith("category_mesh_"):
                    targets.setdefault(key, cls)

        summary: dict[str, dict] = {}
        task = self.create_progress(max(len(targets), 1), "Material maps…")
        for key in sorted(targets):
            mesh = context.input_mesh(key)
            self.advance_progress(task)
            if mesh is None:
                continue
            tm = mesh.mesh
            material = getattr(tm.visual, "material", None)
            base = getattr(material, "baseColorTexture", None) if material is not None else None
            uv = getattr(tm.visual, "uv", None)
            if base is None or uv is None:
                self.log_info(f"  {key}: no base-colour texture, skipped")
                continue
            cls = targets[key]
            ao = None
            if len(tm.faces) >= cfg.min_faces_for_geometric_ao:
                ao = vertex_occlusion(tm.vertices, tm.faces, tm.vertex_normals,
                                      directions=cfg.ao_directions, resolution=cfg.ao_resolution)
                if ao is None:
                    self.log_warning(f"  {key}: no OpenGL context for geometric occlusion; image-derived only")
            maps = mesh_material_maps(
                base, cls, uv=uv, faces=tm.faces, vertex_ao=ao,
                source_mr=getattr(material, "metallicRoughnessTexture", None),
                normal_strength=cfg.normal_strength, cavity_strength=cfg.cavity_strength,
                max_size=cfg.max_texture_size,
            )
            new_material = PBRMaterial(
                name=getattr(material, "name", None),
                baseColorTexture=base,
                baseColorFactor=getattr(material, "baseColorFactor", None),
                metallicRoughnessTexture=maps["metallic_roughness"],
                metallicFactor=1.0,            # the texture's B channel is the metallic value
                roughnessFactor=1.0,           # and its G channel the roughness
                normalTexture=maps["normal"],
                occlusionTexture=maps["occlusion"],
                alphaMode=getattr(material, "alphaMode", None),
                alphaCutoff=getattr(material, "alphaCutoff", None),
                doubleSided=getattr(material, "doubleSided", None),
            )
            tm.visual.material = new_material
            context.add_mesh(key, mesh)
            stats = dict(maps["stats"], cls=cls, faces=int(len(tm.faces)),
                         geometric_ao=None if ao is None else round(float(ao.mean()), 3))
            summary[key] = stats
            self.log_info(
                f"  {key} ({cls or 'unknown class'}, {len(tm.faces)} faces): occlusion "
                f"{stats['occlusion_mean']:.2f}{'' if ao is None else f' (geometric {ao.mean():.2f})'}, "
                f"roughness {stats['roughness_mean']:.2f}, metallic {stats['metallic_mean']:.2f}"
                f"{'' if stats['image_detail'] else ', per-face atlas: geometry only'}"
            )
        self.finish_progress(task)
        context.add_object(SUMMARY_KEY, {"meshes": len(summary), "assets": summary})
        self.log_info(f"Material maps written for {len(summary)} mesh asset(s)")
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(SUMMARY_KEY) is not None

    def model_names(self) -> list[str]:
        return []

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        summary = context.object(SUMMARY_KEY)
        if not summary:
            return None
        return ReportSection(
            stage_name=self.name,
            title="Material Maps",
            body=(
                "Every placed mesh asset got glTF normal, occlusion and metallic-roughness "
                "textures: normals and cavity occlusion from its base colour's detail, "
                "geometric occlusion baked from depth renders, roughness from its class, "
                "metallic zero for natural classes."
            ),
            stats={"Mesh assets": str(summary.get("meshes", 0))},
        )
