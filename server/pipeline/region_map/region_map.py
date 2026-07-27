from typing import Any
from logging import Logger

import cv2
import numpy as np
from PIL import Image as PILImage
import torch
from scipy.ndimage import binary_dilation

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.region_map.region_map_generator import RegionMapGenerator
from pipeline.panorama_segmentation.panorama_region_result import (
    RegionType,
    colorize_region_type_map,
)
from util.depth_utils import Depth
from util.panorama_utils import Panorama
from util.projection_utils import grid_cell_panorama_uv


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
        grid_resolution: int = 4096,
        ground_y_max: float = -0.5,
        camera_height_meters: float = 1.0,
        nadir_exclusion_radius: float = 3.0,
        nadir_ramp_width: float = 5.0,
        certainty_falloff_meters: float = 20.0,
        water_skeleton_smooth_radius: int = 40,
        road_skeleton_smooth_radius: int = 8,
        trail_skeleton_smooth_radius: int = 4,
        interior_peak_max_range_factor: float = 2.0,
        interior_peak_min_depth_m: float = 50.0,
        ridge_prominence_min_m: float = 0.0,
        ridge_prominence_shoulder_m: float = 25.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.grid_size_meters = grid_size_meters
        self.grid_resolution = grid_resolution
        self.ground_y_max = ground_y_max
        self.camera_height_meters = camera_height_meters
        self.nadir_exclusion_radius = nadir_exclusion_radius
        self.nadir_ramp_width = nadir_ramp_width
        # Depth-model-trust radius (metres) at which certainty decays to 0.5 — must
        # match Height Map's value so REGION_MAP_CERTAINTY and HEIGHT_MAP_CERTAINTY
        # stay on the same scale for downstream thresholds (see HeightMapConfiguration).
        self.certainty_falloff_meters = certainty_falloff_meters
        self.water_skeleton_smooth_radius = water_skeleton_smooth_radius
        self.road_skeleton_smooth_radius = road_skeleton_smooth_radius
        self.trail_skeleton_smooth_radius = trail_skeleton_smooth_radius
        # Interior-peak candidates farther than grid_size_meters/2 * this factor are
        # dropped entirely rather than clamped onto the grid boundary — real distant
        # content (a mountain summit kilometres out) isn't representable in a local
        # terrain grid this small, and clamping it there just produces a meaningless,
        # flattened position once reprojected using local terrain elevation. That
        # content is already covered by extract_mountain_ridgeline's silhouette. See
        # RegionMapGenerator.extract_interior_peaks.
        self.interior_peak_max_range_factor = interior_peak_max_range_factor
        # Interior-peak candidates nearer than this are excluded — a relative depth
        # jump this close to camera is routine ground undulation, not a meaningfully
        # elevated feature, and near_side's foreground-bias selection would otherwise
        # let near-camera noise dominate the output. See
        # RegionMapGenerator.extract_interior_peaks.
        self.interior_peak_min_depth_m = interior_peak_min_depth_m
        # Prominence pruning of the ridge silhouette before it becomes solve
        # anchors. Defaults to 0 (keep the whole horizon) -- see
        # RegionMapGenerator.extract_mountain_ridgeline for why the near-field
        # flooding this used to guard against is now fixed at its actual source,
        # and why prominence discards most of a real horizon.
        self.ridge_prominence_min_m = ridge_prominence_min_m
        self.ridge_prominence_shoulder_m = ridge_prominence_shoulder_m


class RegionMapStage(PipelineStage):
    """
    Projects per-pixel semantic region labels from the equirectangular panorama onto a
    top-down grid, producing a region map analogous to the height map.

    Each grid cell is assigned the dominant coarse region type (sky, water, terrain,
    ground, vegetation, built, other) among all ground-plane panorama pixels that project
    into it.  Empty cells are filled by nearest-neighbour propagation.

    Reads:  ContextKey.PANORAMA_DEPTH                     (equirectangular depth on the
                                                           object-removed + LoRA-corrected
                                                           panorama_terrain, metres — not
                                                           PANORAMA_OBJECT_DEPTH, which is
                                                           computed on the ORIGINAL panorama
                                                           and is for object detection/
                                                           distribution instead, which need
                                                           to know what was really there)
            ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN   (per-pixel uint8 type indices from
                                                           the terrain-scoped PanoramaRegionStage
                                                           pass, i.e. also on panorama_terrain --
                                                           an object PanoramaForegroundInpainting
                                                           removed must not still show up as
                                                           VEGETATION/BUILT here and get read as
                                                           part of the ridgeline/water/interior
                                                           peaks)
            ContextKey.PANORAMA_SKY_MASK                  (optional bool sky mask; already
                                                           panorama_terrain-derived by this point
                                                           -- PanoramaDepthStage's terrain-scoped
                                                           instance runs after the object-scoped
                                                           one and overwrites this same key)
    Writes: ContextKey.REGION_MAP                (Depth wrapping a float32 type-index
                                                   grid, grid_resolution²)
    Debug:  self.temp / "region_map.png" etc.    (colorized top-down views of each
                                                   extracted feature)
            self.temp / "panorama_debug_*.png"   (each extracted feature projected back
                                                   onto the source panorama photo in a
                                                   bold colour, for visual sanity-checking)
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

        task = self.create_progress(6, "Region Map…")

        panorama_depth = context.input_depth(depth_key)
        type_idx_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN)
        sky_mask = context.input_sky_mask()

        # Source photo for the panorama-space debug overlays below (see "Debug:
        # project extracted features back onto the source panorama"). Matches
        # what panorama_depth/type_idx_depth were themselves computed from
        # (ContextKey.PANORAMA_TERRAIN — object-removed + LoRA-corrected, not
        # the original ContextKey.PANORAMA; see this stage's docstring), so
        # features stay correctly aligned when projected back. The
        # grid→panorama pixel map is the same for every grid-space feature
        # this stage extracts, so it's computed once here rather than per
        # debug image.
        panorama_source = context.input_panorama(ContextKey.PANORAMA_TERRAIN)
        panorama_source_rgb = None
        _grid_pano_u = _grid_pano_v = None
        if self.temp is not None and panorama_source is not None:
            panorama_source_rgb = np.array(panorama_source.rgb())
            # X/Z only here -- grid_cell_panorama_uv's own pano_u/pano_v assume every
            # cell sits on a flat plane camera_height_meters below the camera. That's
            # the right bootstrap assumption for HeightMapStage (elevation isn't known
            # yet there), but wrong for debug-projecting cells that are actually on
            # real terrain far from flat (a mountain slope, easily metres above camera
            # height) -- it squashed virtually the whole grid into a thin band near the
            # mathematical flat-ground horizon row, regardless of where that terrain
            # actually appears in the photo. Use HEIGHT_MAP's real per-cell elevation
            # (already computed by this point in the pipeline) with the unrestricted
            # Panorama.project_3d instead.
            _, _, X_grid, Z_grid = grid_cell_panorama_uv(
                cfg.grid_size_meters, cfg.grid_resolution, cfg.camera_height_meters,
                panorama_source_rgb.shape[0], panorama_source_rgb.shape[1],
            )
            height_map_depth = context.input_depth(ContextKey.HEIGHT_MAP)
            if height_map_depth is not None and height_map_depth.depth.shape == X_grid.shape:
                Y_grid = height_map_depth.depth.astype(np.float32)
            else:
                Y_grid = np.full_like(X_grid, -cfg.camera_height_meters)
            vertices = np.stack(
                [X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], axis=1
            ).astype(np.float64)
            pu, pv = panorama_source.project_3d(vertices)
            _grid_pano_u = pu.reshape(X_grid.shape).astype(np.float32)
            _grid_pano_v = pv.reshape(X_grid.shape).astype(np.float32)
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
            nadir_exclusion_radius=cfg.nadir_exclusion_radius,
            nadir_ramp_width=cfg.nadir_ramp_width,
            certainty_falloff_meters=cfg.certainty_falloff_meters,
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
            if panorama_source_rgb is not None:
                overlay = self._panorama_debug_scatter_colors(
                    panorama_source_rgb, colorize_region_type_map(region_map),
                    _grid_pano_u, _grid_pano_v,
                )
                PILImage.fromarray(overlay).save(self.temp / "panorama_debug_regions.png")
        self.advance_progress(task)

        # Mountain ridgeline — detect sky-terrain boundary per column, sample depth
        # just below it, and project to actual XZ positions on the top-down grid.
        sky_idx = RegionType.SKY
        terrain_idx = RegionType.TERRAIN
        water_idx = RegionType.WATER
        sky_px = int((type_idx_map == sky_idx).sum())
        terrain_px = int((type_idx_map == terrain_idx).sum())
        unique_types_pano = [RegionType(i).label for i in np.unique(type_idx_map) if i < len(RegionType)]
        self.log_info(f"Ridgeline inputs: {sky_px} sky px, {terrain_px} terrain px — types in panorama: {unique_types_pano}")
        silhouette_grid, ridge_chains = RegionMapGenerator.extract_mountain_ridgeline(
            type_idx_map=type_idx_map,
            sky_idx=sky_idx,
            water_idx=water_idx,
            grid_size_meters=cfg.grid_size_meters,
            grid_resolution=cfg.grid_resolution,
            panorama_depth=panorama_depth,
            prominence_min_m=cfg.ridge_prominence_min_m,
            prominence_shoulder_m=cfg.ridge_prominence_shoulder_m,
        )
        context.add_depth(ContextKey.MOUNTAIN_SILHOUETTE, silhouette_grid)
        context.add_object(ContextKey.MOUNTAIN_RIDGE_CHAINS, ridge_chains)
        silhouette_px = int(silhouette_grid.sum())
        # Azimuth coverage, not just chain count: the failure mode worth catching
        # here is a horizon that survives extraction but only anchors a few
        # degrees of it, which leaves the rest of the scene flat (and, downstream,
        # smears the panorama across it at grazing incidence).
        azimuth_cov = 0.0
        if ridge_chains:
            az = np.concatenate([
                np.degrees(np.arctan2(np.asarray(c)[:, 0], np.asarray(c)[:, 2]))
                for c in ridge_chains
            ])
            azimuth_cov = float((np.histogram(az, np.arange(-180, 181, 5))[0] > 0).mean())
        self.log_info(
            f"Mountain silhouette: {silhouette_px} grid cells, {len(ridge_chains)} ridge chain(s), "
            f"{azimuth_cov * 100:.0f}% azimuth coverage"
            + (f" (prominence ≥ {cfg.ridge_prominence_min_m:.0f} m)"
               if cfg.ridge_prominence_min_m > 0 else "")
        )
        if self.temp is not None:
            rgb = np.zeros((*silhouette_grid.shape, 3), dtype=np.uint8)
            rgb[silhouette_grid > 0] = (255, 255, 255)
            PILImage.fromarray(rgb).save(self.temp / "mountain_silhouette.png")
            if panorama_source_rgb is not None and panorama_source is not None:
                overlay = self._panorama_debug_chains(
                    panorama_source_rgb, ridge_chains, panorama_source, color=(255, 0, 0),
                )
                PILImage.fromarray(overlay).save(self.temp / "panorama_debug_ridge.png")
        self.advance_progress(task)

        # Panorama-space horizon silhouette — sky/terrain boundary in equirectangular
        # pixel coordinates, stored at the segmentation model's native resolution.
        _sky_above = np.zeros(type_idx_map.shape, dtype=bool)
        _sky_above[1:] = (type_idx_map == sky_idx)[:-1]
        horizon_mask = ((type_idx_map != sky_idx) & _sky_above).astype(np.float32)
        context.add_depth(ContextKey.PANORAMA_HORIZON, Depth(horizon_mask))
        self.log_info(f"Panorama horizon: {int(horizon_mask.sum())} pixels")
        if self.temp is not None:
            PILImage.fromarray((horizon_mask * 255).astype(np.uint8)).save(
                self.temp / "panorama_horizon.png"
            )

        # Interior peaks — depth jumps + Canny/corners on RGB to find elevated
        # terrain features that appear against background terrain, not against sky.
        # PANORAMA_TERRAIN (object-removed + LoRA-corrected), matching panorama_depth/
        # type_idx_map above -- an un-removed object's own edges/texture would
        # otherwise trip the Canny/corner heuristic same as it would the ridgeline.
        panorama_img = context.input_image(ContextKey.PANORAMA_TERRAIN)
        panorama_rgb = np.array(panorama_img.rgb()) if panorama_img is not None else None
        if self.temp is not None and panorama_rgb is not None:
            h_pano, w_pano = panorama_rgb.shape[:2]
            h_seg, w_seg = horizon_mask.shape
            if (h_seg, w_seg) != (h_pano, w_pano):
                hm_up = cv2.resize(
                    horizon_mask, (w_pano, h_pano), interpolation=cv2.INTER_NEAREST
                )
            else:
                hm_up = horizon_mask
            overlay = panorama_rgb.copy()
            overlay[hm_up > 0] = [255, 60, 60]
            PILImage.fromarray(overlay).save(self.temp / "panorama_horizon_overlay.png")

        interior_peaks = RegionMapGenerator.extract_interior_peaks(
            type_idx_map=type_idx_map,
            panorama_depth=panorama_depth,
            sky_idx=sky_idx,
            panorama_rgb=panorama_rgb,
            grid_size_meters=cfg.grid_size_meters,
            grid_resolution=cfg.grid_resolution,
            max_range_factor=cfg.interior_peak_max_range_factor,
            min_depth_m=cfg.interior_peak_min_depth_m,
        )
        context.add_depth(ContextKey.INTERIOR_PEAKS, interior_peaks)
        peak_cells = int((interior_peaks > 0).sum())
        self.log_info(f"Interior peaks: {peak_cells} grid cells")
        if self.temp is not None:
            peak_img = (interior_peaks * 255).clip(0, 255).astype(np.uint8)
            PILImage.fromarray(peak_img).save(self.temp / "interior_peaks.png")
            if panorama_source_rgb is not None:
                overlay = self._panorama_debug_scatter(
                    panorama_source_rgb, interior_peaks, _grid_pano_u, _grid_pano_v,
                    color=(255, 140, 0),
                )
                PILImage.fromarray(overlay).save(self.temp / "panorama_debug_peaks.png")

        # Bright, bold debug colours for the panorama-space overlays below --
        # deliberately different from the top-down debug palette above (e.g.
        # ROAD's grey (80, 80, 80) would nearly vanish against a real photo).
        _panorama_debug_colors = {
            RegionType.WATER: (0, 220, 255),
            RegionType.ROAD:  (255, 0, 255),
            RegionType.TRAIL: (255, 255, 0),
        }

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
                if panorama_source_rgb is not None:
                    overlay = self._panorama_debug_scatter(
                        panorama_source_rgb, skeleton, _grid_pano_u, _grid_pano_v,
                        color=_panorama_debug_colors[type_idx],
                    )
                    debug_name = "panorama_debug_" + filename.replace(".png", "") + ".png"
                    PILImage.fromarray(overlay).save(self.temp / debug_name)

        water_chains = RegionMapGenerator.extract_water_chains(
            type_idx_map=type_idx_map,
            panorama_depth=panorama_depth,
            water_idx=int(RegionType.WATER),
            grid_size_meters=cfg.grid_size_meters,
            grid_resolution=cfg.grid_resolution,
            ground_y_max=cfg.ground_y_max,
            camera_height_meters=cfg.camera_height_meters,
        )
        context.add_object(ContextKey.WATER_CHAINS, water_chains)
        self.log_info(f"Water chains: {len(water_chains)}")
        if self.temp is not None and panorama_source_rgb is not None and panorama_source is not None:
            overlay = self._panorama_debug_chains(
                panorama_source_rgb, water_chains, panorama_source, color=(0, 120, 255),
            )
            PILImage.fromarray(overlay).save(self.temp / "panorama_debug_water_chains.png")

        self.finish_progress(task)
        return context

    # ── Debug: project extracted features back onto the source panorama ────────
    #
    # Every debug image here starts from the real panorama photo and paints one
    # extracted feature over it in a single bold colour, so the feature can be
    # visually sanity-checked against what's actually in the photo (as opposed
    # to the top-down grid debug images above, which have no photographic
    # context at all). Grid-space features (region types, skeletons, peaks) use
    # _grid_pano_u/_grid_pano_v, computed once above from each cell's real
    # HEIGHT_MAP elevation via Panorama.project_3d — not grid_cell_panorama_uv's
    # own flat-ground-plane pano_u/pano_v, which is right for
    # inverse_map_panorama_to_grid's bootstrap use (elevation isn't known yet
    # there) but wrong for debug-projecting cells on real, non-flat terrain.
    # World-space features (ridge/water chains, which already carry real
    # elevation) also go through Panorama.project_3d directly, in
    # _panorama_debug_chains below.

    def _panorama_debug_scatter(
        self,
        panorama_rgb: np.ndarray,
        grid_mask: np.ndarray,
        pano_u: np.ndarray,
        pano_v: np.ndarray,
        color: tuple[int, int, int],
        dilate_px: int = 3,
    ) -> np.ndarray:
        """
        Scatter a top-down grid mask (e.g. a skeleton or interior-peaks grid) onto
        a copy of panorama_rgb in a flat debug colour, using a precomputed
        grid_cell_panorama_uv map (pano_u, pano_v — same shape as grid_mask).
        dilate_px thickens the projected points since distant grid cells compress
        into just a few panorama pixels near the horizon and would otherwise be
        nearly invisible.
        """
        h, w = panorama_rgb.shape[:2]
        overlay = panorama_rgb.copy()
        mask = grid_mask > 0
        if not np.any(mask):
            return overlay

        ui = np.mod(np.round(pano_u[mask]).astype(np.int64), w)
        vi = np.clip(np.round(pano_v[mask]).astype(np.int64), 0, h - 1)

        hit = np.zeros((h, w), dtype=bool)
        hit[vi, ui] = True
        if dilate_px > 0:
            hit = binary_dilation(hit, iterations=dilate_px)
        overlay[hit] = color
        return overlay

    def _panorama_debug_scatter_colors(
        self,
        panorama_rgb: np.ndarray,
        grid_colors: np.ndarray,
        pano_u: np.ndarray,
        pano_v: np.ndarray,
        alpha: float = 0.6,
        max_fill_px: int = 3,
    ) -> np.ndarray:
        """
        Like _panorama_debug_scatter, but scatters grid_colors (G, G, 3) — one
        colour per grid cell, e.g. colorize_region_type_map's output — alpha-
        blended over the source photo so the underlying photo detail stays
        visible for comparison. pano_u/pano_v: precomputed grid_cell_panorama_uv
        map, same shape as grid_colors[..., 0].

        Small gaps (mostly near the nadir, where many grid cells compress into
        few panorama pixels) are nearest-neighbour filled up to max_fill_px away,
        rather than a per-channel dilation/max filter, which would fringe
        different regions' colours together right at their boundaries.
        """
        from scipy.ndimage import distance_transform_edt

        h, w = panorama_rgb.shape[:2]
        ui = np.mod(np.round(pano_u).astype(np.int64), w).ravel()
        vi = np.clip(np.round(pano_v).astype(np.int64), 0, h - 1).ravel()

        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        hit = np.zeros((h, w), dtype=bool)
        canvas[vi, ui] = grid_colors.reshape(-1, 3)
        hit[vi, ui] = True

        if max_fill_px > 0:
            dist, (near_v, near_u) = distance_transform_edt(~hit, return_indices=True)
            fillable = (~hit) & (dist <= max_fill_px)
            canvas[fillable] = canvas[near_v[fillable], near_u[fillable]]
            hit |= fillable

        overlay = panorama_rgb.astype(np.float32)
        blended = overlay.copy()
        blended[hit] = overlay[hit] * (1.0 - alpha) + canvas[hit].astype(np.float32) * alpha
        return blended.clip(0, 255).astype(np.uint8)

    def _panorama_debug_chains(
        self,
        panorama_rgb: np.ndarray,
        chains: list[np.ndarray],
        panorama: Panorama,
        color: tuple[int, int, int],
        thickness: int = 4,
    ) -> np.ndarray:
        """
        Draw world-space XYZ polylines (MOUNTAIN_RIDGE_CHAINS, WATER_CHAINS) onto
        a copy of panorama_rgb via Panorama.project_3d.

        Deliberately not Panorama.sample_3d's horizon/sky gating (there was a
        prior version of this that reused a since-removed uv_for_3d, which
        excluded anything above the horizon): a mountain ridgeline silhouette
        sits above the horizon by definition, so gating on that basis meant
        every ridge-chain segment was silently dropped and nothing was ever
        drawn. This function only needs pixel coordinates to draw a line at,
        not a real-colour sample, so no sky/horizon gate applies here at all.
        """
        overlay = panorama_rgb.copy()
        w = overlay.shape[1]
        for chain in chains:
            if len(chain) < 2:
                continue
            pu, pv = panorama.project_3d(chain)
            for i in range(len(chain) - 1):
                u0, u1 = float(pu[i]), float(pu[i + 1])
                if abs(u1 - u0) > w / 2:
                    # Consecutive chain points straddle the panorama's 360° seam --
                    # skip the connecting segment rather than draw a false line
                    # clear across the image.
                    continue
                cv2.line(
                    overlay,
                    (int(round(u0)), int(round(pv[i]))),
                    (int(round(u1)), int(round(pv[i + 1]))),
                    color, thickness, cv2.LINE_AA,
                )
        return overlay

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return (
            context.depth(output_key) is not None
            and context.depth(ContextKey.MOUNTAIN_SILHOUETTE) is not None
            and context.object(ContextKey.MOUNTAIN_RIDGE_CHAINS) is not None
            and context.object(ContextKey.WATER_CHAINS) is not None
            and context.depth(ContextKey.INTERIOR_PEAKS) is not None
            and context.depth(ContextKey.WATER_SKELETON) is not None
            and context.depth(ContextKey.ROAD_SKELETON) is not None
            and context.depth(ContextKey.TRAIL_SKELETON) is not None
        )

    def model_names(self) -> list[str]:
        return []
