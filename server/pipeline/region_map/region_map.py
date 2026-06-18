from typing import Any
from logging import Logger

import numpy as np
from PIL import Image as PILImage
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.region_map.region_map_generator import RegionMapGenerator
from pipeline.panorama_segmentation.panorama_region_result import (
    ALL_REGION_TYPES,
    colorize_region_type_map,
)
from util.depth_utils import Depth


class RegionMapConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 1024,
        ground_y_max: float = -0.5,
        camera_height_meters: float = 1.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.grid_size_meters = grid_size_meters
        self.grid_resolution = grid_resolution
        self.ground_y_max = ground_y_max
        self.camera_height_meters = camera_height_meters


class RegionMapStage(PipelineStage):
    """
    Projects per-pixel semantic region labels from the equirectangular panorama onto a
    top-down grid, producing a region map analogous to the height map.

    Each grid cell is assigned the dominant coarse region type (sky, water, terrain,
    ground, vegetation, built, other) among all ground-plane panorama pixels that project
    into it.  Empty cells are filled by nearest-neighbour propagation.

    Reads:  ContextKey.PANORAMA_DEPTH            (equirectangular depth, metres)
            ContextKey.PANORAMA_REGION_TYPE_MAP   (per-pixel uint8 type indices from
                                                   PanoramaRegionStage)
            ContextKey.PANORAMA_SKY_MASK          (optional bool sky mask)
    Writes: ContextKey.REGION_MAP                (Depth wrapping a float32 type-index
                                                   grid, grid_resolution²)
    Debug:  self.temp / "region_map.png"         (colorized top-down view)
    """

    @classmethod
    def config_class(cls) -> type[RegionMapConfiguration]:
        return RegionMapConfiguration

    def __init__(self, config: RegionMapConfiguration) -> None:
        super().__init__(config)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.DEPTH: ContextKey.PANORAMA_DEPTH,
            SemanticKey.OUTPUT: ContextKey.REGION_MAP,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        depth_key, output_key = self._resolved_keys()
        cfg: RegionMapConfiguration = self.config

        task = self.create_progress(3, "Region Map…")

        panorama_depth = context.input_depth(depth_key)
        type_idx_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        sky_mask = context.input_object(ContextKey.PANORAMA_SKY_MASK)
        self.advance_progress(task)

        if panorama_depth is None:
            self.log_warning("No panorama depth found — skipping region map generation")
            self.finish_progress(task)
            return context

        if type_idx_depth is None:
            self.log_warning("No region type map found — skipping region map generation")
            self.finish_progress(task)
            return context

        type_idx_map = type_idx_depth.depth.astype(np.uint8)

        region_map = RegionMapGenerator.generate(
            panorama_depth=panorama_depth,
            type_idx_map=type_idx_map,
            grid_size_meters=cfg.grid_size_meters,
            grid_resolution=cfg.grid_resolution,
            ground_y_max=cfg.ground_y_max,
            camera_height_meters=cfg.camera_height_meters,
            sky_mask=sky_mask,
        )
        self.advance_progress(task)

        context.add_depth(output_key, region_map.astype(np.float32))

        unique_types = [
            ALL_REGION_TYPES[i]
            for i in np.unique(region_map)
            if i < len(ALL_REGION_TYPES)
        ]
        self.log_info(f"Region map {region_map.shape}, types present: {unique_types}")

        if self.temp is not None:
            PILImage.fromarray(colorize_region_type_map(region_map)).save(
                self.temp / "region_map.png"
            )

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return context.depth(output_key) is not None

    def model_names(self) -> list[str]:
        return []
