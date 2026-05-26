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
        n_z_vertices: int = 150,
        n_x_half_vertices: int = 50,
        z_far: Optional[float] = None,
        noise_amplitude: float = 0.05,
        noise_seed: int = 42,
    ):
        super().__init__(name, device, torch_dtype, log, keys)
        # Row count along the Z (depth) axis — log-spaced, dense near camera.
        self.n_z_vertices = n_z_vertices
        # Column count on *each side* of X=0 — log-spaced, dense near centre.
        # Total X columns = 2 * n_x_half_vertices + 1.
        self.n_x_half_vertices = n_x_half_vertices
        # Far edge in metres. None → read from HEIGHT_MAP_PARAMS, fallback 100 m.
        self.z_far = z_far
        # Peak height displacement from noise in metres.  Scales in with
        # sqrt(Z/z_far) so the ground at the viewer's feet matches the raw map.
        self.noise_amplitude = noise_amplitude
        self.noise_seed = noise_seed


class TerrainMeshStage(PipelineStage):
    """
    Converts the ground-plane height map into a variable-density terrain mesh.

    Vertex density is highest near the camera (origin) in both X and Z — driven by
    logarithmic spacing — so the ground within 1–2 m can be inspected closely while
    the mesh still extends to the full grid extents (~100 m).  Multi-octave smooth
    noise is blended in with distance to give far terrain a natural appearance.

    Input key  (SemanticKey.INPUT)  → ContextKey.HEIGHT_MAP       (Depth, height grid)
    Output key (SemanticKey.OUTPUT) → ContextKey.TERRAIN_MESH     (Mesh, GLB)

    Also reads ContextKey.HEIGHT_MAP_PARAMS (object) for grid_size_meters / z_far when
    those are not overridden in TerrainMeshConfiguration.
    """

    def __init__(self, config: TerrainMeshConfiguration) -> None:
        super().__init__(config)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.HEIGHT_MAP,
            SemanticKey.OUTPUT: ContextKey.TERRAIN_MESH,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, output_key = self._resolved_keys()
        cfg: TerrainMeshConfiguration = self.config

        task = self.create_progress(3, "Terrain Mesh...")

        height_map = context.input_depth(input_key)
        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS)
        self.advance_progress(task)

        if height_map is None:
            self.log_warning("No height map found — skipping terrain mesh generation")
            self.finish_progress(task)
            return context

        # Resolve grid size: config override → height map params → 100 m fallback
        grid_size = (
            cfg.z_far
            or (params.get("grid_size_meters") if params else None)
            or 100.0
        )

        mesh = TerrainMeshGenerator.generate(
            height_map=height_map,
            grid_size_meters=grid_size,
            n_z_vertices=cfg.n_z_vertices,
            n_x_half_vertices=cfg.n_x_half_vertices,
            z_far=grid_size,
            noise_amplitude=cfg.noise_amplitude,
            noise_seed=cfg.noise_seed,
        )
        self.advance_progress(task)

        context.add_mesh(output_key, mesh)

        self.log_info(
            f"Terrain mesh: {mesh.vertex_count} vertices, {mesh.face_count} triangles "
            f"({grid_size:.0f} m grid)"
        )

        if self.temp is not None:
            mesh.save(self.temp / "terrain.glb")

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return context.mesh(output_key) is not None

    def model_names(self) -> list[str]:
        return []
