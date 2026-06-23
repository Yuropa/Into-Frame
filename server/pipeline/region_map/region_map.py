from typing import Any
from logging import Logger

import numpy as np
from PIL import Image as PILImage
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.region_map.region_map_generator import RegionMapGenerator
from pipeline.panorama_segmentation.panorama_region_result import (
    RegionType,
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
        road_skeleton_smooth_radius: int = 8,
        trail_skeleton_smooth_radius: int = 4,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.grid_size_meters = grid_size_meters
        self.grid_resolution = grid_resolution
        self.ground_y_max = ground_y_max
        self.camera_height_meters = camera_height_meters
        self.water_skeleton_smooth_radius = water_skeleton_smooth_radius
        self.road_skeleton_smooth_radius = road_skeleton_smooth_radius
        self.trail_skeleton_smooth_radius = trail_skeleton_smooth_radius


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

        region_map, certainty_array = RegionMapGenerator.generate(
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
        context.add_depth(ContextKey.REGION_MAP_CERTAINTY, Depth(certainty_array))

        unique_types = [RegionType(i).label for i in np.unique(region_map) if i < len(RegionType)]
        self.log_info(
            f"Region map {region_map.shape}, types present: {unique_types}, "
            f"certainty mean {certainty_array[certainty_array > 0].mean():.2f}"
        )

        if self.temp is not None:
            PILImage.fromarray(colorize_region_type_map(region_map)).save(
                self.temp / "region_map.png"
            )
            Depth(certainty_array).save_debug_image(self.temp / "region_map_certainty.png")
        self.advance_progress(task)

        # Mountain ridgeline — detect sky-terrain boundary per column, sample depth
        # just below it, and project to actual XZ positions on the top-down grid.
        sky_idx = RegionType.SKY
        terrain_idx = RegionType.TERRAIN
        sky_px = int((type_idx_map == sky_idx).sum())
        terrain_px = int((type_idx_map == terrain_idx).sum())
        unique_types_pano = [RegionType(i).label for i in np.unique(type_idx_map) if i < len(RegionType)]
        self.log_info(f"Ridgeline inputs: {sky_px} sky px, {terrain_px} terrain px — types in panorama: {unique_types_pano}")
        silhouette_grid = RegionMapGenerator.extract_mountain_ridgeline(
            type_idx_map=type_idx_map,
            panorama_depth=panorama_depth,
            sky_idx=sky_idx,
            terrain_idx=terrain_idx,
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

        for type_idx, ctx_key, smooth_radius, color, filename in [
            (RegionType.WATER, ContextKey.WATER_SKELETON,  cfg.water_skeleton_smooth_radius, (30, 144, 255),  "water_skeleton.png"),
            (RegionType.ROAD,  ContextKey.ROAD_SKELETON,   cfg.road_skeleton_smooth_radius,  (80, 80, 80),    "road_skeleton.png"),
            (RegionType.TRAIL, ContextKey.TRAIL_SKELETON,  cfg.trail_skeleton_smooth_radius, (180, 140, 100), "trail_skeleton.png"),
        ]:
            skeleton = RegionMapGenerator.extract_region_skeleton(
                region_map=region_map,
                type_idx=int(type_idx),
                smooth_radius=smooth_radius,
            )
            context.add_depth(ctx_key, skeleton)
            self.log_info(f"{type_idx.label} skeleton: {int(skeleton.sum())} pixels")
            if self.temp is not None:
                rgb = np.zeros((*skeleton.shape, 3), dtype=np.uint8)
                rgb[skeleton > 0] = color
                PILImage.fromarray(rgb).save(self.temp / filename)

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return (
            context.depth(output_key) is not None
            and context.depth(ContextKey.MOUNTAIN_SILHOUETTE) is not None
            and context.depth(ContextKey.WATER_SKELETON) is not None
            and context.depth(ContextKey.ROAD_SKELETON) is not None
            and context.depth(ContextKey.TRAIL_SKELETON) is not None
        )

    def model_names(self) -> list[str]:
        return []
