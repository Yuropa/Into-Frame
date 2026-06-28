"""
RegionMapRefinementStage — post-geometry region map correction.

Runs after TerrainReconstructionStage, when the final height map is available.
Two passes:

  Pass 1 — Polyline painting.
    Rivers, roads, trails and mountain ridges may have been extended by the
    terrain solver beyond what the panorama segmentation covered.  Every cell
    that lies on one of these polylines is stamped with the corresponding
    region type, regardless of its current classification or certainty.

  Pass 2 — Height-based reclassification of uncertain cells.
    Cells far from the camera have low REGION_MAP_CERTAINTY (sin² of the
    depression angle approaches zero at the horizon).  For those cells the
    segmentation is a poor guess, so we replace it with geometry-derived
    evidence:
      - Flat cells within `water_elevation_tolerance` metres of the known
        water surface → WATER.
      - Cells whose local gradient exceeds `terrain_gradient_threshold`
        (steep slopes) → TERRAIN.

Pipeline position: after TerrainReconstructionStage, before any generation
                  stage that consumes REGION_MAP (object distribution, etc.).

Reads:
  ContextKey.REGION_MAP             (Depth, uint8 type-index grid)
  ContextKey.REGION_MAP_CERTAINTY   (Depth, float32 [0, 1])
  ContextKey.HEIGHT_MAP             (Depth, float32 metres, post-reconstruction)
  ContextKey.HEIGHT_MAP_PARAMS      (dict, grid_size_meters etc.)
  ContextKey.LINEAR_GRAPH           (LinearGraph, world-space polylines)
  ContextKey.WATER_CHAINS           (list[ndarray], (M,3) XYZ)
  ContextKey.MOUNTAIN_RIDGE_CHAINS  (list[ndarray], (M,3) XYZ)

Writes:
  ContextKey.REGION_MAP             (Depth, refined uint8 type-index grid)
"""
from __future__ import annotations

from typing import Any
from logging import Logger

import numpy as np
import torch
from PIL import Image as PILImage

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.panorama_segmentation.panorama_region_result import (
    RegionType,
    colorize_region_type_map,
)
from util.depth_utils import Depth


class RegionMapRefinementConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        certainty_threshold: float = 0.2,
        water_elevation_tolerance: float = 0.5,
        water_gradient_threshold: float = 0.04,
        terrain_gradient_threshold: float = 0.15,
        paint_width: int = 2,
    ):
        """
        certainty_threshold        — cells with certainty below this are candidates
                                     for height-based reclassification.
        water_elevation_tolerance  — metres; a low-certainty cell is reclassified
                                     as WATER if its elevation is within this range
                                     of the median known-water surface elevation.
        water_gradient_threshold   — height change per cell (normalised); flat
                                     cells below this are eligible for WATER.
        terrain_gradient_threshold — height change per cell; steep cells above
                                     this are eligible for TERRAIN.
        paint_width                — half-width in grid cells to dilate each
                                     painted polyline (0 = single-pixel).
        """
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.certainty_threshold = certainty_threshold
        self.water_elevation_tolerance = water_elevation_tolerance
        self.water_gradient_threshold = water_gradient_threshold
        self.terrain_gradient_threshold = terrain_gradient_threshold
        self.paint_width = paint_width


class RegionMapRefinementStage(PipelineStage):

    @classmethod
    def config_class(cls) -> type[RegionMapRefinementConfiguration]:
        return RegionMapRefinementConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: RegionMapRefinementConfiguration = self.config
        task = self.create_progress(4, "Refining region map…")

        region_map_depth = context.input_depth(ContextKey.REGION_MAP)
        if region_map_depth is None:
            self.log_warning("No region map — skipping refinement")
            self.finish_progress(task)
            return context

        certainty_depth = context.input_depth(ContextKey.REGION_MAP_CERTAINTY)
        height_depth    = context.input_depth(ContextKey.HEIGHT_MAP)
        params          = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_size       = float(params.get("grid_size_meters", 100.0))

        region_map = region_map_depth.depth.astype(np.uint8).copy()
        H, W = region_map.shape

        certainty = (
            certainty_depth.depth.astype(np.float32)
            if certainty_depth is not None
            else np.zeros((H, W), dtype=np.float32)
        )
        height_map = (
            height_depth.depth.astype(np.float32)
            if height_depth is not None
            else None
        )

        # Align height map to region map resolution if they differ.
        if height_map is not None and height_map.shape != (H, W):
            from scipy.ndimage import zoom as _zoom
            height_map = _zoom(
                height_map,
                (H / height_map.shape[0], W / height_map.shape[1]),
                order=1,
            ).astype(np.float32)

        self.advance_progress(task)

        # ── Pass 1: stamp polylines ───────────────────────────────────────────
        half = grid_size / 2.0

        def _xz_to_rc(points_xyz: np.ndarray):
            """World XZ → (row, col) grid indices, clipped to bounds."""
            X = points_xyz[:, 0].astype(np.float64)
            Z = points_xyz[:, 2].astype(np.float64)
            col = np.clip(((X + half) / grid_size * W).astype(int), 0, W - 1)
            row = np.clip(((Z + half) / grid_size * H).astype(int), 0, H - 1)
            return row, col

        def _paint_polyline(
            region_map: np.ndarray,
            chains: list[np.ndarray],
            region_type: RegionType,
            width: int,
        ) -> np.ndarray:
            if not chains:
                return region_map
            mask = np.zeros((H, W), dtype=bool)
            for chain in chains:
                chain = np.asarray(chain, dtype=np.float32)
                if len(chain) < 1:
                    continue
                rows, cols = _xz_to_rc(chain)
                # Rasterise segments so there are no gaps between sparse points.
                for k in range(len(rows) - 1):
                    r0, c0 = rows[k], cols[k]
                    r1, c1 = rows[k + 1], cols[k + 1]
                    n = max(abs(r1 - r0), abs(c1 - c0), 1) + 1
                    rs = np.round(np.linspace(r0, r1, n)).astype(int)
                    cs = np.round(np.linspace(c0, c1, n)).astype(int)
                    rs = np.clip(rs, 0, H - 1)
                    cs = np.clip(cs, 0, W - 1)
                    mask[rs, cs] = True
                # Last point
                mask[rows[-1], cols[-1]] = True

            if width > 0:
                from scipy.ndimage import binary_dilation
                from skimage.morphology import disk
                mask = binary_dilation(mask, structure=disk(width))

            region_map = region_map.copy()
            region_map[mask] = int(region_type)
            return region_map

        # Water / river chains from the region map stage.
        water_chains = context.input_object(ContextKey.WATER_CHAINS) or []
        region_map = _paint_polyline(region_map, water_chains, RegionType.WATER, cfg.paint_width)

        # Rivers, roads, trails from the linear structure graph.
        graph = context.input_object(ContextKey.LINEAR_GRAPH)
        if graph is not None:
            type_map = {"river": RegionType.WATER, "road": RegionType.ROAD, "trail": RegionType.TRAIL}
            for structure in graph.structures:
                rt = type_map.get(structure.type)
                if rt is not None and len(structure.path) >= 1:
                    region_map = _paint_polyline(region_map, [structure.path], rt, cfg.paint_width)

        # Mountain ridge chains → TERRAIN.
        ridge_chains = context.input_object(ContextKey.MOUNTAIN_RIDGE_CHAINS) or []
        region_map = _paint_polyline(region_map, ridge_chains, RegionType.TERRAIN, cfg.paint_width)

        n_painted = int(
            (region_map != region_map_depth.depth.astype(np.uint8)).sum()
        )
        self.log_info(f"Polyline painting changed {n_painted} cells")
        self.advance_progress(task)

        # ── Pass 2: height-based reclassification of uncertain distant cells ──
        if height_map is not None:
            low_cert = certainty < cfg.certainty_threshold

            # Gradient in height-map units per cell.  Normalise by cell size so the
            # threshold is in metres/metre rather than metres/pixel.
            cell_size_m = grid_size / max(H, W)
            gy, gx = np.gradient(height_map)
            grad_mag = np.hypot(gx, gy) / cell_size_m  # m/m

            # Derive water surface elevation from cells currently classified as WATER.
            water_mask_known = (region_map == int(RegionType.WATER)) & ~low_cert
            if water_mask_known.any():
                water_elev = float(np.median(height_map[water_mask_known]))
                in_water_band = np.abs(height_map - water_elev) <= cfg.water_elevation_tolerance
                flat_enough   = grad_mag < cfg.water_gradient_threshold
                reclassify_water = low_cert & in_water_band & flat_enough
                region_map[reclassify_water] = int(RegionType.WATER)
                self.log_info(
                    f"Water surface at Y={water_elev:.2f} m; "
                    f"reclassified {reclassify_water.sum()} cells as WATER"
                )
            else:
                self.log_info("No known-water cells to derive surface elevation from")

            steep = grad_mag > cfg.terrain_gradient_threshold
            reclassify_terrain = (
                low_cert & steep
                & (region_map != int(RegionType.WATER))
                & (region_map != int(RegionType.SKY))
            )
            region_map[reclassify_terrain] = int(RegionType.TERRAIN)
            self.log_info(
                f"Reclassified {reclassify_terrain.sum()} steep cells as TERRAIN"
            )

        self.advance_progress(task)

        context.add_depth(ContextKey.REGION_MAP, region_map.astype(np.float32))

        if self.temp is not None:
            PILImage.fromarray(colorize_region_type_map(region_map)).save(
                self.temp / "region_map_refined.png"
            )

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.depth(ContextKey.REGION_MAP) is not None

    def model_names(self) -> list[str]:
        return []
