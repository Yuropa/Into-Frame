from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.heightmap.heightmap_generator import HeightMapGenerator
from util.depth_utils import Depth


class HeightMapConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 512,
        ground_y_max: float = -0.5,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.grid_size_meters = grid_size_meters
        self.grid_resolution = grid_resolution
        # Y threshold in camera space; points with Y <= this are treated as ground.
        # -0.5 means at least 0.5 m below the camera origin.
        self.ground_y_max = ground_y_max


class HeightMapStage(PipelineStage):
    """
    Projects ground-plane points from a rectilinear depth map into a top-down
    height grid, then interpolates any missing cells from their neighbours.

    Input key      (SemanticKey.DEPTH)      → ContextKey.DEPTH          (Depth, metric metres)
    Intrinsics key (SemanticKey.INTRINSICS) → ContextKey.INTRINSICS     (CameraIntrinsics)
    Output key     (SemanticKey.OUTPUT)     → ContextKey.HEIGHT_MAP     (Depth, grid_resolution²)
                                              ContextKey.HEIGHT_MAP_PARAMS (object, grid metadata)

    Grid layout: rows = Z near→far, cols = X left→right, values = Y in camera space (metres).
    Configure grid_size_meters, grid_resolution, and ground_y_max via HeightMapConfiguration.
    """

    @classmethod
    def config_class(cls) -> type[HeightMapConfiguration]:
        return HeightMapConfiguration

    def __init__(self, config: HeightMapConfiguration) -> None:
        super().__init__(config)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.DEPTH: ContextKey.DEPTH,
            SemanticKey.INTRINSICS: ContextKey.INTRINSICS,
            SemanticKey.OUTPUT: ContextKey.HEIGHT_MAP,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        depth_key, intrinsics_key, output_key = self._resolved_keys()
        cfg: HeightMapConfiguration = self.config

        task = self.create_progress(3, "Height Map...")

        depth = context.input_depth(depth_key)
        intrinsics = context.input_intrinsics(intrinsics_key)
        self.advance_progress(task)

        if depth is None:
            self.log_warning("No depth map found — skipping height map generation")
            self.finish_progress(task)
            return context

        if intrinsics is None:
            self.log_warning("No camera intrinsics found — skipping height map generation")
            self.finish_progress(task)
            return context

        height_array = HeightMapGenerator.generate(
            depth=depth,
            intrinsics=intrinsics,
            grid_size_meters=cfg.grid_size_meters,
            grid_resolution=cfg.grid_resolution,
            ground_y_max=cfg.ground_y_max,
        )
        self.advance_progress(task)

        height_map = Depth(height_array)
        context.add_depth(output_key, height_map)

        context.add_object(ContextKey.HEIGHT_MAP_PARAMS, {
            "grid_size_meters": cfg.grid_size_meters,
            "grid_resolution": cfg.grid_resolution,
            "ground_y_max": cfg.ground_y_max,
        })

        if self.temp is not None:
            height_map.save_debug_image(self.temp / "heightmap.png")

        self.log_info(
            f"Height map {height_array.shape}, "
            f"Y range {height_array.min():.2f} → {height_array.max():.2f} m"
        )

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, output_key = self._resolved_keys()
        return context.depth(output_key) is not None

    def model_names(self) -> list[str]:
        return []
