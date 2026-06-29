from typing import Any, Optional
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.terrain.terrain_generator import TerrainMeshGenerator


class TerrainMeshConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 42,
        inner_min_dist: float = 1.5,
        outer_min_dist: float = 6.0,
        n_boundary: int = 12,
        z_far: Optional[float] = None,
        noise_amplitude: float = 0.05,
        texture_tile_factor: float = 8.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.inner_min_dist = inner_min_dist
        self.outer_min_dist = outer_min_dist
        self.n_boundary = n_boundary
        self.z_far = z_far
        self.noise_amplitude = noise_amplitude
        self.texture_tile_factor = texture_tile_factor


class TerrainMeshStage(PipelineStage):
    """
    Converts the ground-plane height map into a variable-density terrain mesh.

    Texture source priority:
      1. SplatMaterial (TERRAIN_MATERIAL) — UVs are set 0→1 so Unity can sample
         the blend maps directly; the first layer tile is embedded as a GLB preview.
      2. Pre-baked single texture (TERRAIN_TEXTURE) — tiled at texture_tile_factor.
      3. Equirectangular panorama — inline bake at UV scale 1.
      4. Original image via pinhole projection.
      5. Geometry-only (no UVs).

    Input key       (SemanticKey.INPUT)      → ContextKey.HEIGHT_MAP
    Panorama key    (SemanticKey.PANORAMA)   → ContextKey.PANORAMA        (optional)
    Intrinsics key  (SemanticKey.INTRINSICS) → ContextKey.INTRINSICS      (optional)
    Output key      (SemanticKey.OUTPUT)     → ContextKey.TERRAIN_MESH
    """

    @classmethod
    def config_class(cls) -> type[TerrainMeshConfiguration]:
        return TerrainMeshConfiguration

    def __init__(self, config: TerrainMeshConfiguration) -> None:
        super().__init__(config)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.HEIGHT_MAP,
            SemanticKey.PANORAMA: ContextKey.PANORAMA,
            SemanticKey.INTRINSICS: ContextKey.INTRINSICS,
            SemanticKey.OUTPUT: ContextKey.TERRAIN_MESH,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        from scene.splat_material import SplatMaterial

        input_key, panorama_key, intrinsics_key, output_key = self._resolved_keys()
        cfg: TerrainMeshConfiguration = self.config

        task = self.create_progress(3, "Terrain Mesh…")

        height_map = context.input_depth(input_key)
        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)
        self.advance_progress(task)

        if height_map is None:
            self.log_warning("No height map found — skipping terrain mesh generation")
            self.finish_progress(task)
            return context

        grid_size = (
            cfg.z_far
            or (params.get("grid_size_meters") if params else None)
            or 100.0
        )

        # ── Resolve texture source ─────────────────────────────────────────────
        precomputed_image = None
        panorama_tex = None
        pinhole_texture = None
        intrinsics = None
        tile_factor = 1.0

        splat: Optional[SplatMaterial] = context.input_object(ContextKey.TERRAIN_MATERIAL)
        if splat is not None and splat.layers:
            # UVs 0→1 so Unity samples blend maps directly at vertex UV coords.
            # Embed the first layer tile as a preview texture so the GLB is not blank.
            precomputed_image = splat.layers[0].tile
            tile_factor = 1.0
            self.log_info(
                f"Terrain texture: SplatMaterial ({splat.layer_count} layer(s), "
                f"{len(splat.blend_maps)} blend map(s)) — UVs 0→1"
            )
        else:
            texture_img = context.input_image(ContextKey.TERRAIN_TEXTURE)
            if texture_img is not None:
                precomputed_image = texture_img.image
                tile_factor = cfg.texture_tile_factor
                self.log_info("Terrain texture: pre-baked single texture")
            else:
                panorama_tex = context.input_panorama(panorama_key)
                if panorama_tex is not None:
                    tile_factor = 1.0
                    self.log_info("Terrain texture: equirectangular panorama (inline bake)")
                else:
                    original = context.input_image(ContextKey.INPUT)
                    intrinsics = context.input_intrinsics(intrinsics_key)
                    if original is not None and intrinsics is not None:
                        pinhole_texture = original.rgb()
                        self.log_info("Terrain texture: original image (pinhole projection)")
                    else:
                        self.log_warning("No texture source — geometry-only terrain mesh")

        # ── Generate mesh ──────────────────────────────────────────────────────
        mesh = TerrainMeshGenerator.generate(
            height_map=height_map,
            grid_size_meters=grid_size,
            inner_min_dist=cfg.inner_min_dist,
            outer_min_dist=cfg.outer_min_dist,
            n_boundary=cfg.n_boundary,
            z_far=grid_size / 2.0,
            noise_amplitude=cfg.noise_amplitude,
            noise_seed=cfg.seed,
            panorama=panorama_tex,
            texture=pinhole_texture,
            intrinsics=intrinsics,
            precomputed_texture=precomputed_image,
            texture_tile_factor=tile_factor,
        )
        self.advance_progress(task)

        context.add_mesh(output_key, mesh)

        self.log_info(
            f"Terrain mesh: {mesh.vertex_count} vertices, {mesh.face_count} triangles, "
            f"{grid_size:.0f} m grid, UV scale ×{tile_factor:.0f}"
        )

        if self.temp is not None:
            mesh.save(self.temp / "terrain.glb")

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, _, output_key = self._resolved_keys()
        return context.mesh(output_key) is not None

    def model_names(self) -> list[str]:
        return []

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        from scene.splat_material import SplatMaterial

        _, _, _, output_key = self._resolved_keys()
        mesh = context.mesh(output_key)
        if mesh is None:
            return None

        params = context.object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        cfg: TerrainMeshConfiguration = self.config
        grid_size = cfg.z_far or params.get("grid_size_meters") or 100.0

        splat: Optional[SplatMaterial] = context.object(ContextKey.TERRAIN_MATERIAL)
        texture_desc = (
            f"SplatMaterial ({splat.layer_count} layers)" if splat
            else "single texture" if context.image(ContextKey.TERRAIN_TEXTURE)
            else "panorama / pinhole / none"
        )

        return ReportSection(
            stage_name=self.name,
            title="Terrain Mesh Generation",
            body=(
                "The height map was converted into a variable-density terrain mesh. "
                "Vertex density is highest near the camera using logarithmic spacing. "
                "Multi-octave smooth noise is blended in with distance."
            ),
            stats={
                "Vertices": f"{mesh.vertex_count:,}",
                "Triangles": f"{mesh.face_count:,}",
                "Grid extent": f"{grid_size:.0f} m",
                "Inner min dist": f"{cfg.inner_min_dist:.1f} m",
                "Outer min dist": f"{cfg.outer_min_dist:.1f} m",
                "Texture": texture_desc,
            },
        )
