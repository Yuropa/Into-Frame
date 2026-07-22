from typing import Any, Optional
from logging import Logger

import numpy as np
import PIL.Image
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.terrain.terrain_generator import TerrainMeshGenerator


def _uv_debug_texture(size: int = 512, divisions: int = 10) -> PIL.Image.Image:
    """
    R = U (0 -> 1, left to right), G = V (0 -> 1, top to bottom), B = 0, with
    thin white gridlines every 1/divisions so wraps, seams, and the scale of a
    UV region are all visible at a glance instead of just a smooth gradient.

    Sampling this through a mesh's UV lets you read a vertex's own UV
    coordinate straight off its rendered colour: (R, G) / 255 ≈ (U, V).
    """
    coords = (np.arange(size, dtype=np.float32) + 0.5) / size
    uu, vv = np.meshgrid(coords, coords)
    img = np.stack([
        (uu * 255.0).clip(0, 255).astype(np.uint8),
        (vv * 255.0).clip(0, 255).astype(np.uint8),
        np.zeros((size, size), dtype=np.uint8),
    ], axis=-1)

    line_px = max(1, size // 256)
    cell = size / divisions
    u_px = np.arange(size)
    on_grid = ((u_px[None, :] % cell) < line_px) | ((u_px[:, None] % cell) < line_px)
    img[on_grid] = 255
    return PIL.Image.fromarray(img, "RGB")


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
        noise_blend_floor: float = 0.15,
        texture_tile_factor: float = 8.0,
        water_depression_m: float = 0.5,
        # Non-primary connected ground components (see HEIGHT_MAP_COMPONENT_ID /
        # HeightMapGenerator._label_ground_components) each get their own separate
        # mesh -- a real, physically disconnected landmass or rock formation,
        # rather than being folded into (or discarded from) the base terrain.
        formation_depression_m: float = 0.5,
        formation_min_dist: float = 0.5,
        formation_n_boundary: int = 8,
        # Debug aid: also export terrain_uv_debug.glb alongside terrain.glb,
        # with its embedded preview material swapped for a synthetic R=U/G=V
        # gradient (see _uv_debug_texture) instead of the real panorama,
        # sampled through the exact same corrected panorama UV (TEXCOORD_1)
        # everything else in this stage produces. Lets you read a vertex's UV
        # coordinate straight off its colour in a generic viewer -- e.g. spot
        # wraps, seams, or the observed/computed boundary at a glance --
        # without needing a real photo or Unity. terrain.glb itself always
        # keeps the real panorama preview; the mesh's real UV data (and
        # Unity's live render, which never reads either embedded material) are
        # unchanged either way.
        debug_uv_texture: bool = False,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.inner_min_dist = inner_min_dist
        self.outer_min_dist = outer_min_dist
        self.n_boundary = n_boundary
        self.z_far = z_far
        self.noise_amplitude = noise_amplitude
        self.noise_blend_floor = noise_blend_floor
        self.texture_tile_factor = texture_tile_factor
        self.water_depression_m = water_depression_m
        self.formation_depression_m = formation_depression_m
        self.formation_min_dist = formation_min_dist
        self.formation_n_boundary = formation_n_boundary
        self.debug_uv_texture = debug_uv_texture


class TerrainMeshStage(PipelineStage):
    """
    Converts the ground-plane height map into a variable-density terrain mesh.

    Texture source priority:
      1. SplatMaterial (TERRAIN_MATERIAL) — UVs are set 0→1 (a world-space
         top-down grid) so Unity can sample the blend maps, and every non-
         equirect layer, directly at those UV coords. This convention is
         load-bearing for Unity's real-time multi-layer blending and stays
         fixed regardless of what layer 0 is. The first layer tile is
         embedded as a GLB preview: if it's the live panorama layer
         (equirect=True), a top-down bake of it is embedded instead of the
         raw equirect image (which would be mismatched under top-down UVs)
         — Unity's own renderer never reads this embedded preview texture,
         it recomputes true equirect UVs live from world position for that
         layer, so this only matters for a generic GLB viewer/debug export.
      2. Pre-baked single texture (TERRAIN_TEXTURE) — tiled at texture_tile_factor.
      3. Equirectangular panorama — each vertex gets its own UV via a direct
         projection onto the panorama's own pixel grid (Panorama.mesh_uvs),
         and the panorama image is used as the texture with no bake step —
         no world-space<->equirectangular resampling distortion.
      4. Original image via pinhole projection.
      5. Geometry-only (no UVs).

    Where ContextKey.REGION_MAP identifies WATER cells, those vertices are
    depressed water_depression_m below the water surface in the terrain mesh
    (so an eventually-animated water plane never clips through the lakebed),
    and a separate flat water Mesh is written to ContextKey.WATER_MESH.

    Where ContextKey.HEIGHT_MAP_COMPONENT_ID identifies non-primary connected
    ground components (see HeightMapGenerator._label_ground_components) --
    real geometry genuinely disconnected from the base terrain, e.g. a
    separate landmass or an isolated rock formation across water -- each gets
    its own geometry-only mesh (TerrainMeshGenerator.generate_component_mesh),
    written under a dynamic "terrain_formation_{id}" key, with the base
    terrain's own copy of that footprint depressed formation_depression_m
    below it (same reasoning as the water depression). A manifest of what was
    created is written to ContextKey.TERRAIN_FORMATIONS for
    TerrainTextureGenerationStage (bakes+applies each formation's own
    texture) and SceneGenerationStage (places each as its own object) to
    consume.

    Input key       (SemanticKey.INPUT)      → ContextKey.HEIGHT_MAP
    Panorama key    (SemanticKey.PANORAMA)   → ContextKey.PANORAMA_TERRAIN (optional)
    Intrinsics key  (SemanticKey.INTRINSICS) → ContextKey.INTRINSICS      (optional)
    Output key      (SemanticKey.OUTPUT)     → ContextKey.TERRAIN_MESH
                                              → ContextKey.WATER_MESH     (optional)
                                              → ContextKey.TERRAIN_FORMATIONS (optional)
    """

    @classmethod
    def config_class(cls) -> type[TerrainMeshConfiguration]:
        return TerrainMeshConfiguration

    def __init__(self, config: TerrainMeshConfiguration) -> None:
        super().__init__(config)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.HEIGHT_MAP,
            SemanticKey.PANORAMA: ContextKey.PANORAMA_TERRAIN,
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

        sky_mask = context.input_object(ContextKey.PANORAMA_SKY_MASK)
        if isinstance(sky_mask, list):
            sky_mask = np.array(sky_mask, dtype=bool)

        # ── Resolve texture source ─────────────────────────────────────────────
        precomputed_image = None
        pinhole_texture = None
        intrinsics = None
        tile_factor = 1.0

        splat: Optional[SplatMaterial] = context.input_splat_material(ContextKey.TERRAIN_MATERIAL)
        # Fetched unconditionally -- needed below for the panorama UV (TEXCOORD_1)
        # regardless of which branch wins for UV0/the embedded preview texture.
        # Unity's live shader and the GLB preview material both depend on it
        # whenever a panorama exists, not just in the no-SplatMaterial fallback
        # branch further down that historically was the only place this got read.
        panorama_tex = context.input_panorama(panorama_key)
        if splat is not None and splat.layers:
            # UVs 0→1 (a world-space top-down grid) so Unity samples blend maps,
            # and every non-equirect layer, directly at those UV coords — this
            # stays fixed regardless of what layer 0 is.
            layer0 = splat.layers[0]
            if layer0.equirect:
                # layer0.tile is the raw, unbaked equirect panorama — embedding it
                # directly under top-down UVs would be mismatched (wrong pixel
                # layout for that UV convention), and Unity's own renderer never
                # reads this embedded texture anyway (it samples the panorama
                # layer via TEXCOORD_1/panoUV -- see TerrainSplat.shader). The
                # actual GLB preview gets swapped in further down (mesh.preview_image,
                # sampled via that same corrected UV) whenever a panorama is
                # available, so this only needs to be a cheap placeholder that
                # keeps UV0 pinned to the world-space top-down grid Unity's blend
                # maps need -- not a real bake of the panorama into that grid
                # (used to be Panorama.bake_topdown; removed since nothing ever
                # looks at its output anymore).
                precomputed_image = PIL.Image.new("RGB", (2, 2))
                tile_factor = 1.0
            else:
                precomputed_image = layer0.tile
                tile_factor = layer0.tile_factor
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
                if panorama_tex is not None:
                    tile_factor = 1.0
                    self.log_info("Terrain texture: equirectangular panorama (direct mesh→panorama UV projection)")
                else:
                    original = context.input_image(ContextKey.INPUT)
                    intrinsics = context.input_intrinsics(intrinsics_key)
                    if original is not None and intrinsics is not None:
                        pinhole_texture = original.rgb()
                        self.log_info("Terrain texture: original image (pinhole projection)")
                    else:
                        self.log_warning("No texture source — geometry-only terrain mesh")

        region_map = context.input_depth(ContextKey.REGION_MAP)
        # Ridge-override-aware, not the plain HEIGHT_MAP_OBSERVED_MASK: a cell
        # the mountain-ridge envelope overrode in TerrainReconstructionStage
        # has moved to a corrected position its cached panorama UV no longer
        # matches, so it must NOT be treated as "trust the cached UV" here even
        # though it's still perfectly good real data to every other consumer of
        # HEIGHT_MAP_OBSERVED_MASK. Falls back to the plain mask if the solve
        # never ran (e.g. TerrainReconstructionStage disabled).
        observed_mask = (
            context.input_depth(ContextKey.HEIGHT_MAP_PANO_UV_TRUST_MASK)
            or context.input_depth(ContextKey.HEIGHT_MAP_OBSERVED_MASK)
        )
        component_id = context.input_depth(ContextKey.HEIGHT_MAP_COMPONENT_ID)
        pano_uv_u = context.input_depth(ContextKey.HEIGHT_MAP_PANO_U)
        pano_uv_v = context.input_depth(ContextKey.HEIGHT_MAP_PANO_V)

        # ── Generate mesh ──────────────────────────────────────────────────────
        mesh, water_mesh, pano_uv = TerrainMeshGenerator.generate(
            height_map=height_map,
            grid_size_meters=grid_size,
            inner_min_dist=cfg.inner_min_dist,
            outer_min_dist=cfg.outer_min_dist,
            n_boundary=cfg.n_boundary,
            z_far=grid_size / 2.0,
            noise_amplitude=cfg.noise_amplitude,
            noise_blend_floor=cfg.noise_blend_floor,
            noise_seed=cfg.seed,
            panorama=panorama_tex,
            texture=pinhole_texture,
            intrinsics=intrinsics,
            precomputed_texture=precomputed_image,
            texture_tile_factor=tile_factor,
            region_map=region_map,
            water_depression_m=cfg.water_depression_m,
            observed_mask=observed_mask,
            component_id=component_id,
            formation_depression_m=cfg.formation_depression_m,
            sky_mask=sky_mask,
            pano_uv_u=pano_uv_u,
            pano_uv_v=pano_uv_v,
        )
        self.advance_progress(task)

        # Carried through to Mesh.save() as glTF TEXCOORD_1 -- Unity's terrain
        # shader samples it directly for the panorama layer instead of
        # recomputing an equirect projection from world position every frame
        # (see TerrainSplat.shader). None whenever no panorama was available.
        mesh.extra_uv = pano_uv
        # Also swap in as the GLB's own embedded preview material (sampled via
        # that same TEXCOORD_1) whenever there's a real panorama to show --
        # regardless of which branch above won for the embedded texture/UV0
        # (e.g. the SplatMaterial branch's cheap placeholder above). Unity
        # itself never reads this embedded material for the panorama layer,
        # so overwriting it here has no effect on the live render.
        if panorama_tex is not None and pano_uv is not None:
            mesh.preview_image = panorama_tex.image

        context.add_mesh(output_key, mesh)

        self.log_info(
            f"Terrain mesh: {mesh.vertex_count} vertices, {mesh.face_count} triangles, "
            f"{grid_size:.0f} m grid, UV scale ×{tile_factor:.0f}"
        )

        # ── Formation meshes (non-primary ground components) ─────────────────────
        formations: list[dict] = []
        if component_id is not None:
            n_components = int(component_id.depth.max())
            for target_id in range(2, n_components + 1):
                result = TerrainMeshGenerator.generate_component_mesh(
                    height_map=height_map,
                    component_id=component_id,
                    target_id=target_id,
                    grid_size_meters=grid_size,
                    min_dist=cfg.formation_min_dist,
                    n_boundary=cfg.formation_n_boundary,
                    seed=cfg.seed + target_id,
                )
                if result is None:
                    continue
                formation_mesh, x_center, z_center, x_half, z_half = result
                mesh_key = f"terrain_formation_{target_id}"
                context.add_mesh(mesh_key, formation_mesh)
                formations.append({
                    "id": target_id,
                    "mesh_key": mesh_key,
                    "x_center": x_center,
                    "z_center": z_center,
                    "x_half": x_half,
                    "z_half": z_half,
                })
                self.log_info(
                    f"Formation mesh {target_id}: {formation_mesh.vertex_count} vertices, "
                    f"{formation_mesh.face_count} triangles, centered ({x_center:.1f}, {z_center:.1f}), "
                    f"±({x_half:.1f}, {z_half:.1f}) m"
                )
                if self.temp is not None:
                    formation_mesh.save(self.temp / f"terrain_formation_{target_id}.glb")

        if formations:
            context.add_object(ContextKey.TERRAIN_FORMATIONS, formations)

        if self.temp is not None:
            mesh.save(self.temp / "terrain.glb")
            if cfg.debug_uv_texture and pano_uv is not None:
                # Swap the embedded preview to the synthetic UV grid and save
                # under its own filename -- terrain.glb (and the real panorama
                # texture Unity/other viewers expect from it) is left untouched.
                real_preview = mesh.preview_image
                mesh.preview_image = _uv_debug_texture()
                mesh.save(self.temp / "terrain_uv_debug.glb")
                mesh.preview_image = real_preview

        if water_mesh is not None:
            context.add_mesh(ContextKey.WATER_MESH, water_mesh)
            self.log_info(
                f"Water mesh: {water_mesh.vertex_count} vertices, {water_mesh.face_count} triangles"
            )
            if self.temp is not None:
                water_mesh.save(self.temp / "water.glb")
        elif region_map is None:
            self.log_info("No region map in context — skipping water mesh")

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

        splat: Optional[SplatMaterial] = context.splat_material(ContextKey.TERRAIN_MATERIAL)
        texture_desc = (
            f"SplatMaterial ({splat.layer_count} layers)" if splat
            else "single texture" if context.image(ContextKey.TERRAIN_TEXTURE)
            else "panorama / pinhole / none"
        )

        stats = {
            "Vertices": f"{mesh.vertex_count:,}",
            "Triangles": f"{mesh.face_count:,}",
            "Grid extent": f"{grid_size:.0f} m",
            "Inner min dist": f"{cfg.inner_min_dist:.1f} m",
            "Outer min dist": f"{cfg.outer_min_dist:.1f} m",
            "Texture": texture_desc,
        }

        water_mesh = context.mesh(ContextKey.WATER_MESH)
        if water_mesh is not None:
            stats["Water mesh"] = f"{water_mesh.vertex_count:,} vertices, {water_mesh.face_count:,} triangles"
            stats["Water depression"] = f"{cfg.water_depression_m:.2f} m"

        return ReportSection(
            stage_name=self.name,
            title="Terrain Mesh Generation",
            body=(
                "The height map was converted into a variable-density terrain mesh. "
                "Vertex density is highest near the camera using logarithmic spacing. "
                "Multi-octave smooth noise is blended in with distance."
            ),
            stats=stats,
        )
