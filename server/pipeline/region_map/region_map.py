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
    REGION_TYPE_SKY,
    REGION_TYPE_TERRAIN,
    REGION_TYPE_WATER,
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
        water_skeleton_smooth_radius: int = 40,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.grid_size_meters = grid_size_meters
        self.grid_resolution = grid_resolution
        self.ground_y_max = ground_y_max
        self.camera_height_meters = camera_height_meters
        self.water_skeleton_smooth_radius = water_skeleton_smooth_radius


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

        task = self.create_progress(5, "Region Map…")

        panorama_depth = context.input_depth(depth_key)
        type_idx_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        sky_mask = context.input_object(ContextKey.PANORAMA_SKY_MASK)
        if isinstance(sky_mask, list):
            sky_mask = np.array(sky_mask, dtype=bool)
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
        self.advance_progress(task)

        # Mountain silhouette — detect ridgeline in panorama, then project to top-down grid.
        sky_idx = ALL_REGION_TYPES.index(REGION_TYPE_SKY)
        terrain_idx = ALL_REGION_TYPES.index(REGION_TYPE_TERRAIN)
        pano_silhouette = RegionMapGenerator.extract_mountain_silhouette(
            type_idx_map=type_idx_map,
            sky_idx=sky_idx,
            terrain_idx=terrain_idx,
        )
        silhouette_grid = RegionMapGenerator.project_silhouette_to_grid(
            panorama_depth=panorama_depth,
            silhouette_mask=pano_silhouette,
            grid_size_meters=cfg.grid_size_meters,
            grid_resolution=cfg.grid_resolution,
        )
        context.add_depth(ContextKey.MOUNTAIN_SILHOUETTE, silhouette_grid)
        silhouette_px = int(silhouette_grid.sum())
        self.log_info(f"Mountain silhouette: {silhouette_px} grid cells")
        if self.temp is not None:
            rgb = np.zeros((*silhouette_grid.shape, 3), dtype=np.uint8)
            rgb[silhouette_grid > 0] = (255, 255, 255)
            PILImage.fromarray(rgb).save(self.temp / "mountain_silhouette.png")
        self.advance_progress(task)

        # Water skeleton — medial axis of water cells in the top-down region map.
        water_idx = ALL_REGION_TYPES.index(REGION_TYPE_WATER)
        water_skeleton = RegionMapGenerator.extract_water_skeleton(
            region_map=region_map,
            water_idx=water_idx,
            smooth_radius=cfg.water_skeleton_smooth_radius,
        )
        context.add_depth(ContextKey.WATER_SKELETON, water_skeleton)
        skeleton_px = int(water_skeleton.sum())
        self.log_info(f"Water skeleton: {skeleton_px} skeleton pixels")
        if self.temp is not None:
            rgb = np.zeros((*water_skeleton.shape, 3), dtype=np.uint8)
            rgb[water_skeleton > 0] = (30, 144, 255)
            PILImage.fromarray(rgb).save(self.temp / "water_skeleton.png")

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return (
            context.depth(output_key) is not None
            and context.depth(ContextKey.MOUNTAIN_SILHOUETTE) is not None
            and context.depth(ContextKey.WATER_SKELETON) is not None
        )

    def model_names(self) -> list[str]:
        return []
