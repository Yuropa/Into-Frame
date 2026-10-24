from pathlib import Path
from typing import Any, Optional
from logging import Logger

import numpy as np
import torch

from pipeline.panorama_depth.calibration import CAMERA_HEIGHT_KEY
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.heightmap.heightmap_generator import HeightMapGenerator
from util.depth_utils import Depth
from util.panorama_utils import Panorama


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
        grid_resolution: int = 4096,
        ground_y_max: float = -0.5,
        use_equirectangular: bool = False,
        smooth_sigma: float = 0.0,
        camera_height_meters: float = 1.0,
        flood_fill: bool = True,
        flood_fill_max_step: float = 1.5,
        label_smooth_sigma: float = 1.5,
        nadir_exclusion_radius: float = 1.0,
        nadir_ramp_width: float = 5.0,
        far_exclusion_radius: Optional[float] = None,
        flat_zone_certainty: float = 0.15,
        certainty_falloff_meters: float = 20.0,
        elevation_distortion_power: float = 1.0,
        save_point_cloud: bool = True,
        point_cloud_stride: int = 4,
        min_forward_samples: int = 4,
        fill_boundary_falloff_cells: float = 6.0,
        min_component_area_fraction: float = 0.001,
        despike_threshold_m: float = 0.3,
        despike_window: int = 5,
        despike_reference_distance_m: float = 10.0,
        despike_dense_threshold_scale: float = 2.5,
        despike_dense_min_real_support: float = 0.6,
        pinnacle_threshold_m: float = 1.0,
        pinnacle_window: int = 31,
        wall_relief_m: float = 4.0,
        wall_rock_fraction: float = 0.5,
        wall_min_distance_m: float = 10.0,
        wall_shoulder_factor: float = 2.0,
        wall_shoulder_min_m: float = 6.0,
        wall_shoulder_max_m: float = 30.0,
        wall_bin_m: float = 2.0,
        densify_radius_m: float = 0.6,
        densify_min_samples: int = 3,
        densify_min_distance_m: float = 20.0,
        densify_far_fraction: float = 0.5,
        wall_far_fraction: float = 0.33,
        wall_max_distance_m: float = 0.0,
        region_closing_iterations: int = 2,
        single_sample_blur_sigma: float = 1.5,
        reclaimed_certainty_factor: float = 0.5,
        level_ground_to_camera: bool = True,
        standing_viewer_prior: bool = True,
        viewer_eye_height_m: float = 1.7,
        standing_min_drop_m: float = 3.0,
        standing_pedestal_radius_m: float = 3.0,
        standing_ring_m: tuple[float, float] = (7.0, 10.0),
        anchor_water_plane: bool = True,
        anchor_water_min_fraction: float = 0.02,
        bank_reach_m: float = 15.0,
        bank_max_rise_m: float = 6.0,
        level_ground_ring_m: tuple[float, float] = (6.0, 12.0),
        level_ground_tolerance_m: float = 0.15,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.grid_size_meters = grid_size_meters
        self.grid_resolution = grid_resolution
        # Y threshold in camera space; points with Y <= this are treated as ground.
        # -0.5 means at least 0.5 m below the camera origin.
        self.ground_y_max = ground_y_max
        self.use_equirectangular = use_equirectangular
        # Max Gaussian sigma (in grid pixels) for distance-weighted smoothing.
        # 0 disables smoothing; sigma is 0 at the centre, smooth_sigma at the corners.
        self.smooth_sigma = smooth_sigma
        # Assumed camera height above the ground plane (metres). Used to derive the Y
        # floor filter that rejects sky-pixel artefacts and as the flood-fill seed height.
        self.camera_height_meters = camera_height_meters
        # Flood-fill from the grid centre outward; stops at height discontinuities and
        # empty cells (sky gaps), yielding a connected ground region rather than a noisy
        # global threshold selection.
        self.flood_fill = flood_fill
        # Maximum Y change (metres) between adjacent grid cells during flood-fill.
        self.flood_fill_max_step = flood_fill_max_step
        # Gaussian sigma (grid cells) for a NaN-aware smoothed copy used only to
        # decide flood-fill connectivity -- absorbs depth-model noise that would
        # otherwise spuriously sever a small patch from the larger formation it
        # should belong to, without altering the actual returned height values.
        # 0 disables (raw height_map decides connectivity, as before). See
        # HeightMapGenerator._label_ground_components.
        self.label_smooth_sigma = label_smooth_sigma
        # Cells within nadir_exclusion_radius are pinned to -camera_height_meters (flat
        # ground prior) with certainty flat_zone_certainty. Certainty then ramps smoothly
        # up to full geometric certainty over nadir_ramp_width metres beyond that radius.
        self.nadir_exclusion_radius = nadir_exclusion_radius
        self.nadir_ramp_width = nadir_ramp_width
        # Mirror image of nadir_exclusion_radius at the far end of the range: beyond
        # this radius (metres), a cell backed only by the flat-ground single-ray guess
        # (no genuine forward-projected sample) stops counting as observed ground
        # truth for hard-pinning purposes, since sin(phi_grid) -> 0 out there crushes
        # real elevation signal the same way phi_grid -> -90 deg near the nadir
        # amplifies depth-model noise into it. Certainty and the computed height are
        # left alone -- only true_observed is affected. None disables (no far
        # exclusion). See HeightMapGenerator.generate's far_exclusion_radius docstring.
        self.far_exclusion_radius = far_exclusion_radius
        self.flat_zone_certainty = flat_zone_certainty
        # Distance (metres) at which observed-ground certainty decays to 0.5. This is a
        # depth-model-trust radius, independent of camera_height_meters -- it must be
        # large enough that real mid-range terrain can cross TerrainReconstructionStage's
        # confidence_threshold, or nothing beyond the nadir ramp ever gets Dirichlet-pinned
        # and the whole map beyond a few metres is left to pure interpolation/noise.
        self.certainty_falloff_meters = certainty_falloff_meters
        # Exponent on cos(phi) (phi = true camera-relative elevation angle to each
        # observed cell), multiplied into certainty -- the standard equirectangular
        # areal-distortion factor: 1 straight ahead at the horizon, falling to 0 at
        # the top/bottom of the source panorama (zenith/nadir), where equirectangular
        # stretching makes the raw depth estimate itself least reliable, independent
        # of how far away the ground point is. 0 disables. See
        # HeightMapGenerator._build_certainty.
        self.elevation_distortion_power = elevation_distortion_power
        # Debug/inspection artefact: the raw sky-masked panorama depth unprojected to a
        # dense world-space point cloud (every pixel, no ground-plane Y filter, no grid
        # binning) — the full-resolution geometry HeightMapGenerator's single-sample-per-
        # cell grid necessarily throws away. Saved as ground_point_cloud.glb next to the
        # other debug images when a build/output directory is configured.
        self.save_point_cloud = save_point_cloud
        # Take every Nth pixel in each axis before unprojecting (1 = full resolution).
        # Equirectangular panoramas are large enough that a full dump is both slow to
        # write and awkward to load in a mesh viewer; 4 (1/16 of the pixels) keeps the
        # file a manageable size while still far denser than the height-map grid.
        self.point_cloud_stride = point_cloud_stride
        # Equirectangular mode only: minimum independent forward-projected panorama
        # pixels that must land in one grid cell before its single inverse-mapped
        # sample is overridden by real per-cell mean/relief/slope statistics. Below
        # this, either a plane fit is impossible (<3 points) or too noisy to trust.
        self.min_forward_samples = min_forward_samples
        # How many cells (at each octave's own resolution) beyond a real
        # observation the synthetic fill's injected noise needs before reaching
        # full amplitude -- unobserved cells near real data (e.g. the occluded
        # top of a cliff whose base is directly observed) hug that data's
        # diffused trend instead of drifting via unconstrained randomness; only
        # cells with no nearby observation get real freedom to wander. See
        # HeightMapGenerator._interpolate.
        self.fill_boundary_falloff_cells = fill_boundary_falloff_cells
        # Connected walkable-ground components smaller than this fraction of the
        # grid are folded into the interpolated-gap fallback rather than kept as
        # their own component -- avoids spurious few-cell noise clusters becoming
        # a "formation" downstream. See HeightMapGenerator._label_ground_components.
        self.min_component_area_fraction = min_component_area_fraction
        # Equirectangular mode only: despike any cell (including dense_mask/
        # any_forward_mask, backed by min_forward_samples' multi-sample stats --
        # a mixed-pixel cell straddling a real depth discontinuity can still
        # disagree sharply with its neighbours even after averaging) that
        # diverges from its own despike_window-sized neighbourhood median by
        # more than this many metres -- source depth-map noise (a "flying
        # pixel" edge flickering between two competing surfaces) otherwise
        # lands as an alternating checkerboard of wildly different heights
        # between adjacent grid cells. 0 disables. See
        # HeightMapGenerator._despike_cells.
        self.despike_threshold_m = despike_threshold_m
        self.despike_window = despike_window
        # Per-pixel depth-model noise grows with sampled range (and is further
        # amplified for elevated/steep rays via Y = depth * sin(phi)), so the
        # effective despike threshold shrinks below despike_threshold_m -- never
        # above it -- for cells sampled beyond this distance, scaled by
        # despike_reference_distance_m / sampled_depth.
        self.despike_reference_distance_m = despike_reference_distance_m
        # Extra multiplier (>= 1.0) on the effective despike threshold for
        # dense_mask/any_forward_mask cells only -- already averaged over
        # multiple forward-projected samples, so genuine sensor noise should
        # mostly have cancelled; a bigger deviation is required before one is
        # flagged than a single-ray cell needs.
        self.despike_dense_threshold_scale = despike_dense_threshold_scale
        # Minimum fraction (0-1) of real (non-interpolated) cells required in a
        # dense_mask/any_forward_mask cell's own despike_window before it can be
        # flagged at all -- protects a real, sparsely-surrounded sample (e.g. an
        # isolated distant ridge point) from being judged against a mostly-
        # fabricated local median and wrongly flattened. See
        # HeightMapGenerator._despike_cells.
        self.despike_dense_min_real_support = despike_dense_min_real_support
        # Tall narrow measured features (a tree base at 15 m) are opened away -- see
        # HeightMapGenerator._flatten_pinnacles. 0 disables.
        self.pinnacle_threshold_m = float(pinnacle_threshold_m)
        self.pinnacle_window = int(pinnacle_window)
        # A cliff face seen side-on (a headland across a cove, a sea stack) is
        # taken at its crest and the land behind it raised to match -- see the
        # wall rule in HeightMapGenerator.generate. wall_relief_m is the span of
        # raw samples inside one cell that makes it a wall (0 disables); nearer
        # than wall_min_distance_m the nadir's depth noise spans metres too.
        self.wall_relief_m = float(wall_relief_m)
        self.wall_rock_fraction = float(wall_rock_fraction)
        self.wall_min_distance_m = float(wall_min_distance_m)
        self.wall_shoulder_factor = float(wall_shoulder_factor)
        self.wall_shoulder_min_m = float(wall_shoulder_min_m)
        self.wall_shoulder_max_m = float(wall_shoulder_max_m)
        self.wall_bin_m = float(wall_bin_m)
        # Far samples are one cell in ten; cells within densify_radius_m of
        # densify_min_samples real samples take their mean instead of the flat-
        # ground guess, beyond densify_min_distance_m. See the generator.
        self.densify_radius_m = float(densify_radius_m)
        self.densify_min_samples = int(densify_min_samples)
        self.densify_min_distance_m = float(densify_min_distance_m)
        self.densify_far_fraction = float(densify_far_fraction)
        self.wall_far_fraction = float(wall_far_fraction)
        self.wall_max_distance_m = float(wall_max_distance_m)
        # Binary closing iterations applied to the ground-valid classification
        # before it gates which pixels count as ground (see
        # HeightMapGenerator._ground_valid_mask). Per-pixel semantic segmentation
        # is noisy right at class boundaries; closing removes single-pixel
        # holes/islands that would otherwise show up as salt-and-pepper speckle
        # in both the observed mask and the height values sampled from it.
        # 0 disables.
        self.region_closing_iterations = region_closing_iterations
        # Gaussian sigma (grid cells) for a partial blur applied only to single-
        # inverse-mapped-sample cells (no intra-cell averaging), blended in
        # per-cell by (1 - certainty) -- dense (multi-sample) cells are never
        # touched. Targets per-pixel depth-model dither that despiking's outlier
        # test can't catch. 0 disables. See
        # HeightMapGenerator._despike_cells and _build_certainty.
        self.single_sample_blur_sigma = single_sample_blur_sigma
        # When the camera height is the configured prior rather than measured from
        # the photograph, it means "the viewer's eye is this far above the ground
        # they stand on". The nadir disc is diffused from the ground just outside
        # it, so the map is shifted so that ring (level_ground_ring_m, the nearest
        # measured ground past the disc) sits at -camera_height. On Rainier the
        # ring sat at -0.4 m under a 1.0 m camera: the eye was 0.4 m off the
        # ground and knee-high flowers 1.5 m away reached the horizon.
        self.level_ground_to_camera = bool(level_ground_to_camera)
        # The viewer stands on something. With a photo-measured camera height the
        # nadir is scaled to the plane the photographed people stand on, and on a
        # clifftop capture that plane is the beach far below (Shark Fin: ground
        # 12.4 m under a camera the client leaves at the photographer's height, so
        # the viewer hovers over sand with no cliff under them). When the ground in
        # standing_ring_m is not water and lies more than standing_min_drop_m below
        # eye height, a pedestal at -viewer_eye_height_m is raised under the camera:
        # flat to standing_pedestal_radius_m and blended into the measured terrain
        # over 1.5 x the drop (at least 8 m) -- a clifftop knoll above the beach.
        # Over water (Paris: a bridge) nothing changes; the viewer is on the deck.
        self.standing_viewer_prior = bool(standing_viewer_prior)
        self.viewer_eye_height_m = float(viewer_eye_height_m)
        self.standing_min_drop_m = float(standing_min_drop_m)
        self.standing_pedestal_radius_m = float(standing_pedestal_radius_m)
        self.standing_ring_m = (float(standing_ring_m[0]), float(standing_ring_m[1]))
        # Water scenes with a photo-measured camera height: the height was measured
        # to the water plane, so the water IS at -camera_height. The depth-derived
        # map put Paris's river at -8.1 m under a 12.1 m camera and its quays at
        # -8.4 m (below the water, no relief), and the solved terrain sat under the
        # water sheet. Water cells are pinned to the plane, the whole map shifted
        # with them, and the banks given the rise the panorama shows: from each
        # column's waterline up to where the depth starts receding, at the
        # waterline's flat-water distance -- a wall of that height.
        self.anchor_water_plane = bool(anchor_water_plane)
        self.anchor_water_min_fraction = float(anchor_water_min_fraction)
        self.bank_reach_m = float(bank_reach_m)
        self.bank_max_rise_m = float(bank_max_rise_m)
        self.level_ground_ring_m = (float(level_ground_ring_m[0]), float(level_ground_ring_m[1]))
        self.level_ground_tolerance_m = float(level_ground_tolerance_m)
        # Equirectangular mode only: certainty multiplier for cells whose
        # ground validity came from PANORAMA_REGION_AMBIGUOUS_MASK -- pixels
        # PanoramaRegionStage resolved from an uncorroborated ambiguous label
        # (e.g. "wall") to a ground-valid runner-up type, rather than a
        # direct classification. A fixed, conservative factor rather than one
        # scaled by the (distrusted) original label's own confidence, since a
        # confidently-wrong original call should not suppress the reclaim.
        # See HeightMapGenerator._build_certainty.
        self.reclaimed_certainty_factor = reclaimed_certainty_factor

class HeightMapStage(PipelineStage):
    """
    Projects ground-plane points from a rectilinear depth map into a top-down
    height grid, then interpolates any missing cells from their neighbours.

    Input key      (SemanticKey.DEPTH)      → ContextKey.DEPTH          (Depth, metric metres)
    Intrinsics key (SemanticKey.INTRINSICS) → ContextKey.INTRINSICS     (CameraIntrinsics)
    Output key     (SemanticKey.OUTPUT)     → ContextKey.HEIGHT_MAP     (Depth, grid_resolution²)
                                              ContextKey.HEIGHT_MAP_PARAMS (object, grid metadata)
                                              ContextKey.HEIGHT_MAP_PANO_U/_V (Depth, each cell's
                                              own panorama UV as measured -- see
                                              HeightMapGenerator._panorama_uv_from_height)

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

    def _anchor_to_water_plane(self, height, pano_u, pano_v, sampled, type_map, panorama_depth, camera_height, cfg):
        """See anchor_water_plane. Returns the adjusted height array."""
        from scipy.ndimage import distance_transform_edt
        from pipeline.panorama_segmentation.panorama_region_result import RegionType
        water_idx = int(RegionType.WATER)
        # Only a man-made shore stands up as a wall. Ground, sand or rock above a
        # waterline is a slope the depth already describes (the same rule on
        # Shark Fin's beach made a 5.5 m "wall" of sand), and vegetation above a
        # quay is the trees on it, not more quay (Paris: banks ran to the 8 m cap).
        land_types = [int(t) for t in (RegionType.BUILT, RegionType.ROAD, RegionType.OTHER)]
        H, W = type_map.shape[:2]
        n = height.shape[0]
        # Which cells are water: each cell looked up in the panorama along the ray
        # to a point on the water plane at its (x, z). The map's own heights cannot
        # be used for this lookup -- they are what is being corrected (Paris's
        # -8 m map looked everything up within 12 deg of the horizon and saw banks
        # and buildings where the river is).
        axis = (np.arange(n, dtype=np.float64) + 0.5) / n * self._grid_size_m - self._grid_size_m / 2.0
        zz, xx = np.meshgrid(axis, axis, indexing="ij")
        r_cell = np.hypot(xx, zz)
        lon = np.arctan2(xx, zz)
        lat = -np.arctan2(float(camera_height), np.maximum(r_cell, 1e-6))
        row = np.clip(((0.5 - lat / np.pi) * (H - 1)).astype(np.intp), 0, H - 1)
        col = np.clip(((lon / (2.0 * np.pi) + 0.5) * (W - 1)).astype(np.intp), 0, W - 1)
        valid = np.isfinite(height) & (r_cell >= 3.0)
        cell_type = np.rint(type_map[row, col]).astype(np.int32)
        water = valid & (cell_type == water_idx)
        frac = float(water.mean())
        if frac < cfg.anchor_water_min_fraction:
            self.log_info(f"Water plane anchor: {frac:.1%} of the grid is water — nothing to anchor to")
            return height
        y_w = -float(camera_height)
        shift = y_w - float(np.median(height[water]))
        out = np.where(np.isfinite(height), height + shift, height).astype(height.dtype)
        out[water] = y_w

        # Bank rise per panorama column: waterline -> top of the wall.
        depth = None
        if panorama_depth is not None:
            d = np.asarray(panorama_depth.depth, dtype=np.float32)
            if d.shape != (H, W):
                d = d[(np.arange(H) * d.shape[0]) // H][:, (np.arange(W) * d.shape[1]) // W]
            depth = d
        t = np.rint(np.asarray(type_map)).astype(np.int32)
        phi_down = (np.arange(H) / H - 0.5) * np.pi  # radians below the horizon, + = down
        az_of_col = (np.arange(W) / W - 0.5) * 2.0 * np.pi
        sectors = 72
        rises = [[] for _ in range(sectors)]
        is_water = t == water_idx
        is_land = np.isin(t, land_types)
        for c in range(0, W, 2):
            wc = np.flatnonzero(is_water[:, c])
            if wc.size == 0:
                continue
            # The highest water run's top: scan up from the water's lowest row.
            r_wl = int(wc.min())
            if phi_down[r_wl] <= np.radians(1.0):
                continue
            r_edge = camera_height / np.tan(phi_down[r_wl])
            r_top = None
            for rr in range(r_wl - 1, -1, -1):
                if not is_land[rr, c]:
                    r_top = rr + 1
                    break
                if depth is not None and np.isfinite(depth[rr, c]) and depth[rr, c] > 1.25 * r_edge / max(np.cos(phi_down[rr]), 1e-3):
                    r_top = rr + 1
                    break
            if r_top is None or r_top >= r_wl - 1:
                continue
            rise = r_edge * (np.tan(phi_down[r_wl]) - np.tan(phi_down[r_top]))
            if not np.isfinite(rise) or rise <= 0.0:
                continue
            k = int(((az_of_col[c] + np.pi) / (2.0 * np.pi)) * sectors) % sectors
            rises[k].append(min(float(rise), cfg.bank_max_rise_m))
        rise_by_sector = np.array([np.median(r) if len(r) >= 3 else np.nan for r in rises])
        if np.isfinite(rise_by_sector).sum() >= 4:
            # Fill empty sectors from the nearest measured one (circular), then smooth.
            idx = np.arange(sectors)
            good = np.isfinite(rise_by_sector)
            filled = rise_by_sector.copy()
            for k in idx[~good]:
                dist = np.minimum(np.abs(idx[good] - k), sectors - np.abs(idx[good] - k))
                filled[k] = rise_by_sector[good][np.argmin(dist)]
            filled = np.median(np.stack([np.roll(filled, k) for k in (-2, -1, 0, 1, 2)]), axis=0)  # circular 5-sector median
            az = lon
            cell_m = self._grid_size_m / n
            dist_w = distance_transform_edt(~water) * cell_m
            land = ~water & np.isfinite(out)
            sector_of = (((az + np.pi) / (2.0 * np.pi)) * sectors).astype(int) % sectors
            target = y_w + filled[sector_of]
            near = land & (dist_w <= cfg.bank_reach_m)
            out[near] = np.maximum(out[near], target[near])
            self.log_info(
                f"Water plane anchor: {frac:.0%} of the grid is water, pinned to {y_w:.2f} m (map shifted {shift:+.2f} m); "
                f"banks within {cfg.bank_reach_m:.0f} m of the water rise {np.nanpercentile(rise_by_sector, 10):.1f}/"
                f"{np.nanmedian(rise_by_sector):.1f}/{np.nanpercentile(rise_by_sector, 90):.1f} m (p10/50/90 over "
                f"{int(good.sum())} of {sectors} azimuth sectors) above it, from the photographed waterline"
            )
        else:
            self.log_info(
                f"Water plane anchor: {frac:.0%} of the grid is water, pinned to {y_w:.2f} m (map shifted {shift:+.2f} m); "
                "no measurable bank rise in the panorama"
            )
        # Land never sits under the water surface.
        land = ~water & np.isfinite(out)
        out[land] = np.maximum(out[land], y_w + 0.3)
        return out

    def run(self, context: PipelineContext) -> PipelineContext:
        depth_key, intrinsics_key, output_key = self._resolved_keys()
        cfg: HeightMapConfiguration = self.config
        # Measured from the photo by Panorama Depth Calibration when it could be
        # (see CAMERA_HEIGHT_KEY); the configured height otherwise.
        camera_estimate = context.input_object(CAMERA_HEIGHT_KEY) or {}
        camera_height = float(camera_estimate.get("meters") or cfg.camera_height_meters)
        # The grid's extent: chosen per capture by Panorama Depth Calibration from
        # the depth's reliable range (see its adaptive_grid), the configured value
        # when it did not.
        grid_size = float(context.input_object(ContextKey.TERRAIN_GRID_SIZE_M) or cfg.grid_size_meters)
        self._grid_size_m = grid_size
        if grid_size != cfg.grid_size_meters:
            self.log_info(f"Grid: {grid_size:.0f} m across (Panorama Depth Calibration), not the configured {cfg.grid_size_meters:.0f} m")

        task = self.create_progress(3, "Height Map…")

        depth = context.input_depth(depth_key)
        intrinsics = context.input_intrinsics(intrinsics_key)
        sky_mask = context.input_sky_mask()
        panorama_depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
        # _TERRAIN: classified from panorama_terrain (object-removed + LoRA-
        # corrected), matching panorama_depth above -- not the original-panorama
        # PANORAMA_REGION_TYPE_MAP, which would still show a removed object as
        # VEGETATION/BUILT and let it gate ground-plane validity here.
        region_type_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN)
        region_type_mask = region_type_depth.depth if region_type_depth is not None else None
        region_ambiguous_depth = context.input_depth(ContextKey.PANORAMA_REGION_AMBIGUOUS_MASK_TERRAIN)
        region_ambiguous_mask = (
            region_ambiguous_depth.depth.astype(bool) if region_ambiguous_depth is not None else None
        )
        self.advance_progress(task)

        if depth is None:
            self.log_warning("No depth map found — skipping height map generation")
            self.finish_progress(task)
            return context

        if not cfg.use_equirectangular and intrinsics is None:
            self.log_warning("No camera intrinsics found — skipping height map generation")
            self.finish_progress(task)
            return context

        if sky_mask is not None:
            self.log_info("Using sky mask to exclude horizon artefacts")

        # When the primary depth is already equirectangular (depth_key == PANORAMA_DEPTH),
        # using the same map as the panorama fill source would silently undo the flood-fill:
        # cells rejected for having bad depth values would be re-projected with those same
        # bad values. Skip the fill in that case; interpolation handles the gaps instead.
        fill_panorama_depth = panorama_depth if depth_key != ContextKey.PANORAMA_DEPTH else None

        if fill_panorama_depth is not None:
            self.log_info("Panorama depth available — will fill unseen terrain regions")

        if region_type_mask is not None:
            self.log_info("Region type map available — restricting height to water/terrain/ground")

        if cfg.use_equirectangular and cfg.save_point_cloud and self.temp is not None:
            panorama_terrain = context.input_panorama(ContextKey.PANORAMA_TERRAIN)
            colors = (
                np.array(panorama_terrain.rgb()) if panorama_terrain is not None else None
            )
            n_points = self._save_ground_point_cloud(
                depth, sky_mask, colors, cfg.point_cloud_stride, self.temp / "ground_point_cloud.glb",
            )
            if n_points is not None:
                self.log_info(
                    f"Ground point cloud: {n_points:,} points "
                    f"(stride {cfg.point_cloud_stride}) → ground_point_cloud.glb"
                )

        # The far field is not terrain. Panorama pixels deeper than the far-field
        # start (FAR_FIELD_START_M, where the depth models saturate -- see Panorama
        # Depth Calibration's far_fraction) belong to FarFieldBackdropStage's shell,
        # not to this grid; observing them here is what modelled the mountain as a
        # ring of rock at 30 m. Tightens far_exclusion_radius to that start.
        far_exclusion_radius = cfg.far_exclusion_radius
        far_start = context.input_object(ContextKey.FAR_FIELD_START_M)
        if far_start:
            if far_exclusion_radius is None or far_exclusion_radius > float(far_start):
                self.log_info(
                    f"Far exclusion tightened from {far_exclusion_radius} m to the far-field start "
                    f"{float(far_start):.1f} m"
                )
                far_exclusion_radius = float(far_start)
        height_array, certainty_array, cell_relief_array, cell_slope_array, true_observed_array, component_id_array, pano_uv_u_array, pano_uv_v_array, real_sample_mask_array = HeightMapGenerator.generate(
            depth=depth,
            intrinsics=intrinsics,
            grid_size_meters=grid_size,
            grid_resolution=cfg.grid_resolution,
            ground_y_max=cfg.ground_y_max,
            use_equirectangular=cfg.use_equirectangular,
            smooth_sigma=cfg.smooth_sigma,
            camera_height_meters=camera_height,
            sky_mask=sky_mask,
            flood_fill=cfg.flood_fill,
            flood_fill_max_step=cfg.flood_fill_max_step,
            label_smooth_sigma=cfg.label_smooth_sigma,
            panorama_depth=fill_panorama_depth,
            region_type_mask=region_type_mask,
            region_ambiguous_mask=region_ambiguous_mask,
            reclaimed_certainty_factor=cfg.reclaimed_certainty_factor,
            nadir_exclusion_radius=cfg.nadir_exclusion_radius,
            nadir_ramp_width=cfg.nadir_ramp_width,
            far_exclusion_radius=far_exclusion_radius,
            flat_zone_certainty=cfg.flat_zone_certainty,
            certainty_falloff_meters=cfg.certainty_falloff_meters,
            elevation_distortion_power=cfg.elevation_distortion_power,
            min_forward_samples=cfg.min_forward_samples,
            fill_boundary_falloff_cells=cfg.fill_boundary_falloff_cells,
            min_component_area_fraction=cfg.min_component_area_fraction,
            despike_threshold_m=cfg.despike_threshold_m,
            despike_window=cfg.despike_window,
            despike_reference_distance_m=cfg.despike_reference_distance_m,
            despike_dense_threshold_scale=cfg.despike_dense_threshold_scale,
            despike_dense_min_real_support=cfg.despike_dense_min_real_support,
            pinnacle_threshold_m=cfg.pinnacle_threshold_m,
            pinnacle_window=cfg.pinnacle_window,
            wall_relief_m=cfg.wall_relief_m, wall_rock_fraction=cfg.wall_rock_fraction,
            wall_min_distance_m=cfg.wall_min_distance_m,
            wall_shoulder_factor=cfg.wall_shoulder_factor,
            wall_shoulder_min_m=cfg.wall_shoulder_min_m,
            wall_shoulder_max_m=cfg.wall_shoulder_max_m,
            wall_bin_m=cfg.wall_bin_m,
            densify_radius_m=cfg.densify_radius_m,
            densify_min_samples=cfg.densify_min_samples,
            densify_min_distance_m=cfg.densify_min_distance_m, densify_far_fraction=cfg.densify_far_fraction, wall_far_fraction=cfg.wall_far_fraction, wall_max_distance_m=cfg.wall_max_distance_m, far_start_m=(float(far_start) if far_start else None),
            region_closing_iterations=cfg.region_closing_iterations,
            single_sample_blur_sigma=cfg.single_sample_blur_sigma,
            debug_dir=self.temp,
        )
        self.advance_progress(task)

        dens_report = getattr(HeightMapGenerator, "_last_densify_report", None)
        HeightMapGenerator._last_densify_report = None
        if dens_report:
            n_dens, n_fwd, win = dens_report
            self.log_info(
                f"Densified: {n_dens:,} cell(s) beyond {cfg.densify_min_distance_m:.0f} m took the mean of the real "
                f"samples within {win:.2f} m ({n_fwd:,} forward-projected cell(s) in all)"
            )
        wall_report = getattr(HeightMapGenerator, "_last_wall_report", None)
        HeightMapGenerator._last_wall_report = None
        if wall_report:
            n_wall, n_shoulder, n_facing, rel50, rel90, cr50, cr90, r0, r1 = wall_report
            self.log_info(
                f"Walls: {n_wall:,} cell(s) hold a vertical face of {rel50:.1f}/{rel90:.1f} m (p50/p90) "
                f"at {r0:.0f}-{r1:.0f} m, raised to their crest ({cr50:+.1f}/{cr90:+.1f} m); {n_facing:,} face the "
                f"camera and carry {n_shoulder:,} shoulder cell(s) behind them"
            )

        if cfg.level_ground_to_camera and camera_estimate.get("source") != "photo":
            r0, r1 = cfg.level_ground_ring_m
            n = cfg.grid_resolution
            axis = (np.arange(n, dtype=np.float32) + 0.5) / n * grid_size - grid_size / 2.0
            zz, xx = np.meshgrid(axis, axis, indexing="ij")
            rr = np.hypot(xx, zz)
            ring = (rr >= r0) & (rr <= r1) & np.isfinite(height_array)
            if int(ring.sum()) >= 100:
                ring_y = float(np.median(height_array[ring]))
                shift = -camera_height - ring_y
                if abs(shift) > cfg.level_ground_tolerance_m:
                    height_array = np.where(np.isfinite(height_array), height_array + shift, height_array).astype(height_array.dtype)
                    self.log_info(
                        f"Ground {r0:.0f}-{r1:.0f} m from the camera sat {ring_y:+.2f} m under a "
                        f"{camera_height:.2f} m camera ({camera_estimate.get('source', 'config')}); "
                        f"whole map shifted {shift:+.2f} m so the viewer stands at eye height above it"
                    )
                else:
                    self.log_info(f"Ground {r0:.0f}-{r1:.0f} m from the camera sits {ring_y:+.2f} m, within "
                                  f"{cfg.level_ground_tolerance_m:.2f} m of the {camera_height:.2f} m camera")

        if cfg.anchor_water_plane and camera_estimate.get("source") == "photo" and region_type_mask is not None:
            height_array = self._anchor_to_water_plane(
                height_array, pano_uv_u_array, pano_uv_v_array, real_sample_mask_array,
                np.asarray(region_type_mask), panorama_depth, camera_height, cfg,
            )

        if cfg.standing_viewer_prior and camera_estimate.get("source") == "photo":
            from pipeline.panorama_segmentation.panorama_region_result import RegionType as _RT
            n = cfg.grid_resolution
            axis = (np.arange(n, dtype=np.float32) + 0.5) / n * grid_size - grid_size / 2.0
            zz, xx = np.meshgrid(axis, axis, indexing="ij")
            rr = np.hypot(xx, zz)
            r0, r1 = cfg.standing_ring_m
            ring = (rr >= r0) & (rr <= r1) & np.isfinite(height_array)
            water_frac = None
            if region_type_mask is not None and ring.any():
                tm = np.asarray(region_type_mask)
                th, tw = tm.shape
                u = pano_uv_u_array[ring]
                v = pano_uv_v_array[ring]
                okuv = np.isfinite(u) & np.isfinite(v)
                if okuv.sum() >= 50:
                    tu = np.clip((u[okuv] * (tw - 1)).astype(np.int64), 0, tw - 1)
                    tv = np.clip((v[okuv] * (th - 1)).astype(np.int64), 0, th - 1)
                    water_frac = float((np.rint(tm[tv, tu]).astype(np.int32) == int(_RT.WATER)).mean())
            target = -float(cfg.viewer_eye_height_m)
            ring_y = float(np.median(height_array[ring])) if ring.any() else np.nan
            if water_frac is None and region_type_mask is not None and ring.any() and np.isfinite(ring_y):
                # The ring was filled rather than sampled (Paris: the water plane
                # anchor wrote it, so no panorama UVs). Type it the way the anchor
                # does: look each cell up along the ray to its point on the ring's
                # own plane. A bridge or a boat deck must never get a pedestal.
                tm = np.asarray(region_type_mask)
                th, tw = tm.shape
                lon = np.arctan2(xx[ring], zz[ring])
                lat = -np.arctan2(-ring_y, np.maximum(rr[ring], 1e-6))
                row = np.clip(((0.5 - lat / np.pi) * (th - 1)).astype(np.intp), 0, th - 1)
                col = np.clip(((lon / (2.0 * np.pi) + 0.5) * (tw - 1)).astype(np.intp), 0, tw - 1)
                water_frac = float((np.rint(tm[row, col]).astype(np.int32) == int(_RT.WATER)).mean())
            drop = target - ring_y if np.isfinite(ring_y) else 0.0
            if water_frac is not None and water_frac >= 0.5:
                self.log_info(
                    f"Standing viewer: the ground {r0:.0f}-{r1:.0f} m out is {water_frac:.0%} water "
                    f"({ring_y:+.2f} m) -- the viewer is on a deck over it, nothing raised"
                )
            elif np.isfinite(ring_y) and drop >= cfg.standing_min_drop_m:
                R0 = float(cfg.standing_pedestal_radius_m)
                R1 = R0 + max(8.0, 1.5 * drop)
                t = np.clip((rr - R0) / max(R1 - R0, 1e-6), 0.0, 1.0)
                sm = t * t * (3.0 - 2.0 * t)
                measured = np.where(np.isfinite(height_array), height_array, ring_y)
                raised = target * (1.0 - sm) + measured * sm
                within = rr <= R1
                lifted = within & (raised > measured)
                height_array = np.where(lifted, raised, height_array).astype(height_array.dtype)
                true_observed_array[within] = True
                self.log_info(
                    f"Standing viewer: the ground {r0:.0f}-{r1:.0f} m out lies {ring_y:+.2f} m, {drop:.1f} m below eye "
                    f"height ({'%.0f%% water' % (100 * water_frac) if water_frac is not None else 'type unknown'}); "
                    f"a pedestal at {target:+.1f} m raised under the camera, flat to {R0:.0f} m and blended out by {R1:.0f} m "
                    f"({int(lifted.sum()):,} cell(s) lifted)"
                )
            elif np.isfinite(ring_y):
                self.log_info(f"Standing viewer: the ground {r0:.0f}-{r1:.0f} m out lies {ring_y:+.2f} m, within {cfg.standing_min_drop_m:.0f} m of eye height")

        height_map = Depth(height_array)
        context.add_depth(output_key, height_map)
        context.add_depth(ContextKey.HEIGHT_MAP_CERTAINTY, Depth(certainty_array))
        context.add_depth(ContextKey.HEIGHT_MAP_CELL_RELIEF, Depth(cell_relief_array))
        context.add_depth(ContextKey.HEIGHT_MAP_CELL_SLOPE, Depth(cell_slope_array))
        context.add_depth(
            ContextKey.HEIGHT_MAP_OBSERVED_MASK, Depth(true_observed_array.astype(np.float32))
        )
        context.add_depth(
            ContextKey.HEIGHT_MAP_REAL_SAMPLE_MASK, Depth(real_sample_mask_array.astype(np.float32))
        )
        context.add_depth(
            ContextKey.HEIGHT_MAP_COMPONENT_ID, Depth(component_id_array.astype(np.float32))
        )
        context.add_depth(ContextKey.HEIGHT_MAP_PANO_U, Depth(pano_uv_u_array))
        context.add_depth(ContextKey.HEIGHT_MAP_PANO_V, Depth(pano_uv_v_array))

        context.add_object(ContextKey.HEIGHT_MAP_PARAMS, {
            "grid_size_meters":    grid_size,
            "grid_resolution":     cfg.grid_resolution,
            "ground_y_max":        cfg.ground_y_max,
            "camera_height_meters": camera_height,
        })

        if self.temp is not None:
            height_map.save_debug_image(self.temp / "heightmap.png")
            Depth(certainty_array).save_debug_image(self.temp / "heightmap_certainty.png")
            if cell_relief_array.any():
                Depth(cell_relief_array).save_debug_image(self.temp / "heightmap_cell_relief.png")
            if cell_slope_array.any():
                Depth(cell_slope_array).save_debug_image(self.temp / "heightmap_cell_slope.png")

        n_relief = int((cell_relief_array > 0).sum())
        n_slope = int((cell_slope_array > 0).sum())
        self.log_info(
            f"Height map {height_array.shape}, "
            f"Y range {height_array.min():.2f} → {height_array.max():.2f} m, "
            f"certainty mean {certainty_array[certainty_array > 0].mean():.2f}, "
            f"{n_relief} cell(s) with measured intra-cell relief "
            f"(max {cell_relief_array.max():.2f} m), "
            f"{n_slope} cell(s) with measured plane-fit slope "
            f"(max {cell_slope_array.max():.1f}°)"
        )

        n_components = int(component_id_array.max())
        if n_components > 1:
            sizes = [int((component_id_array == i).sum()) for i in range(1, n_components + 1)]
            self.log_info(
                f"Ground components: {n_components} ({sizes[0]:,} base + "
                f"{', '.join(f'{s:,}' for s in sizes[1:])} cell(s) in {n_components - 1} "
                f"separate component(s), kept as real geometry instead of discarded)"
            )

        self.finish_progress(task)
        return context

    @staticmethod
    def _save_ground_point_cloud(
        depth: Depth,
        sky_mask: Optional[np.ndarray],
        colors: Optional[np.ndarray],
        stride: int,
        out_path: Path,
    ) -> Optional[int]:
        """
        Unprojects every (strided) panorama pixel to a world-space XYZ point, masking
        only the sky, and writes the result as a .glb point cloud (a POINTS-mode mesh
        with per-vertex color, loadable in Blender, macOS Quick Look, online glTF
        viewers, etc.). Unlike the height map itself, this applies no ground-plane Y
        filter and no grid binning — it is the raw scene geometry (cliffs, overhangs,
        distant peaks included) that the single-sample-per-cell grid lookup in
        HeightMapGenerator cannot represent.
        """
        d = depth.depth.astype(np.float32)
        if sky_mask is not None and sky_mask.shape == d.shape:
            d = d.copy()
            d[sky_mask] = np.nan

        s = max(1, stride)
        d = d[::s, ::s]

        X, Y, Z = Panorama.equirectangular_unproject(Depth(d))
        valid = np.isfinite(d) & (d > 0)
        if not valid.any():
            return None

        points = np.stack([X[valid], Y[valid], Z[valid]], axis=-1)

        point_colors = None
        if colors is not None and colors.shape[:2] == depth.depth.shape:
            point_colors = colors[::s, ::s][valid]

        import trimesh
        trimesh.PointCloud(points, colors=point_colors).export(str(out_path))
        return int(points.shape[0])

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, _, output_key = self._resolved_keys()
        return context.depth(output_key) is not None

    def model_names(self) -> list[str]:
        return []

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        from pipeline.report.report_utils import colorize_depth
        _, _, output_key = self._resolved_keys()
        height_map = context.depth(output_key)
        if height_map is None:
            return None
        params = context.object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        cfg: HeightMapConfiguration = self.config
        stats = {
            "Grid resolution": f"{height_map.width} × {height_map.height} cells",
            "Grid size": f"{params.get('grid_size_meters', cfg.grid_size_meters):.0f} m",
            "Height range": f"{height_map.min():.2f} – {height_map.max():.2f} m",
        }
        return ReportSection(
            stage_name=self.name,
            title="Terrain Height Map",
            body=(
                "Ground-plane points from the depth map were projected into a top-down "
                "height grid using the estimated camera intrinsics. The grid is flood-filled "
                "from the camera position outward, stopping at height discontinuities, to "
                "produce a connected ground surface free of sky-pixel artefacts. Cells "
                "lacking direct observations are interpolated from their neighbours. "
                "The resulting height map drives terrain mesh generation."
            ),
            images=[(colorize_depth(height_map), "Ground-plane height map (bright = high elevation)")],
            stats=stats,
        )
