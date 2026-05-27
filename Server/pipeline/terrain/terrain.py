from typing import Any, Optional
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.terrain.terrain_generator import TerrainMeshGenerator, UVMode


class TerrainMeshConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 42,
        inner_grid_n: int = 30,
        n_rings: int = 12,
        ring_base_points: int = 64,
        z_far: Optional[float] = None,
        noise_amplitude: float = 0.05,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.inner_grid_n = inner_grid_n
        self.n_rings = n_rings
        self.ring_base_points = ring_base_points
        self.z_far = z_far
        self.noise_amplitude = noise_amplitude


class TerrainMeshStage(PipelineStage):
    """
    Converts the ground-plane height map into a variable-density terrain mesh
    with an embedded texture sampled from the panorama or original image.

    Vertex density is highest near the camera (origin) in both X and Z — driven by
    logarithmic spacing — so the ground within 1–2 m can be inspected closely while
    the mesh still extends to the full grid extents (~100 m).  Multi-octave smooth
    noise is blended in with distance for a natural look.

    Texture source (tried in order):
      1. Panorama image  → equirectangular UV projection (full 360° coverage)
      2. Original image  → pinhole UV projection via camera intrinsics (FOV-limited)
      3. No texture      → geometry-only mesh

    Input key       (SemanticKey.INPUT)      → ContextKey.HEIGHT_MAP      (Depth)
    Panorama key    (SemanticKey.PANORAMA)   → ContextKey.PANORAMA        (Image, optional)
    Intrinsics key  (SemanticKey.INTRINSICS) → ContextKey.INTRINSICS      (CameraIntrinsics, optional)
    Output key      (SemanticKey.OUTPUT)     → ContextKey.TERRAIN_MESH    (Mesh, GLB with texture)

    Also reads ContextKey.HEIGHT_MAP_PARAMS for grid_size_meters / z_far when
    not overridden in TerrainMeshConfiguration.
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
        input_key, panorama_key, intrinsics_key, output_key = self._resolved_keys()
        cfg: TerrainMeshConfiguration = self.config

        task = self.create_progress(3, "Terrain Mesh...")

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

        # ── Resolve texture source ────────────────────────────────────────────
        texture_pil = None
        uv_mode: UVMode = "panorama"
        intrinsics = None

        panorama = context.input_image(panorama_key)
        if panorama is not None:
            texture_pil = panorama.rgb()
            uv_mode = "panorama"
            self.log_info("Terrain texture: equirectangular panorama")
        else:
            original = context.input_image(ContextKey.INPUT)
            intrinsics = context.input_intrinsics(intrinsics_key)
            if original is not None and intrinsics is not None:
                texture_pil = original.rgb()
                uv_mode = "pinhole"
                self.log_info("Terrain texture: original image (pinhole projection)")
            else:
                self.log_warning(
                    "No texture source found — generating geometry-only terrain mesh"
                )

        # ── Generate mesh ─────────────────────────────────────────────────────
        mesh = TerrainMeshGenerator.generate(
            height_map=height_map,
            grid_size_meters=grid_size,
            inner_grid_n=cfg.inner_grid_n,
            n_rings=cfg.n_rings,
            ring_base_points=cfg.ring_base_points,
            z_far=grid_size / 2.0,  # half-extent: terrain spans ±z_far in Z
            noise_amplitude=cfg.noise_amplitude,
            noise_seed=cfg.seed,
            texture=texture_pil,
            uv_mode=uv_mode,
            intrinsics=intrinsics,
        )
        self.advance_progress(task)

        context.add_mesh(output_key, mesh)

        self.log_info(
            f"Terrain mesh: {mesh.vertex_count} vertices, {mesh.face_count} triangles, "
            f"{grid_size:.0f} m grid"
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
