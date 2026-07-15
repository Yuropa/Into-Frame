from pathlib import Path
from typing import Any, Optional
from logging import Logger

import numpy as np
import torch

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
        nadir_exclusion_radius: float = 1.0,
        nadir_ramp_width: float = 5.0,
        flat_zone_certainty: float = 0.15,
        certainty_falloff_meters: float = 20.0,
        save_point_cloud: bool = True,
        point_cloud_stride: int = 4,
        min_forward_samples: int = 4,
        fill_boundary_falloff_cells: float = 6.0,
        min_component_area_fraction: float = 0.001,
        despike_threshold_m: float = 0.3,
        despike_window: int = 5,
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
        # Cells within nadir_exclusion_radius are pinned to -camera_height_meters (flat
        # ground prior) with certainty flat_zone_certainty. Certainty then ramps smoothly
        # up to full geometric certainty over nadir_ramp_width metres beyond that radius.
        self.nadir_exclusion_radius = nadir_exclusion_radius
        self.nadir_ramp_width = nadir_ramp_width
        self.flat_zone_certainty = flat_zone_certainty
        # Distance (metres) at which observed-ground certainty decays to 0.5. This is a
        # depth-model-trust radius, independent of camera_height_meters -- it must be
        # large enough that real mid-range terrain can cross TerrainReconstructionStage's
        # confidence_threshold, or nothing beyond the nadir ramp ever gets Dirichlet-pinned
        # and the whole map beyond a few metres is left to pure interpolation/noise.
        self.certainty_falloff_meters = certainty_falloff_meters
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
        # Equirectangular mode only: despike single-inverse-mapped-sample cells
        # (not backed by min_forward_samples' dense statistics) that diverge from
        # their own despike_window-sized neighbourhood median by more than this
        # many metres -- source depth-map noise (a "flying pixel" edge flickering
        # between two competing surfaces) otherwise lands as an alternating
        # checkerboard of wildly different heights between adjacent grid cells.
        # 0 disables. See HeightMapGenerator._despike_single_sample_cells.
        self.despike_threshold_m = despike_threshold_m
        self.despike_window = despike_window


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

        task = self.create_progress(3, "Height Map…")

        depth = context.input_depth(depth_key)
        intrinsics = context.input_intrinsics(intrinsics_key)
        sky_mask = context.input_object(ContextKey.PANORAMA_SKY_MASK)
        if isinstance(sky_mask, list):
            sky_mask = np.array(sky_mask, dtype=bool)
        panorama_depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
        region_type_depth = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        region_type_mask = region_type_depth.depth if region_type_depth is not None else None
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

        height_array, certainty_array, cell_relief_array, cell_slope_array, true_observed_array, component_id_array = HeightMapGenerator.generate(
            depth=depth,
            intrinsics=intrinsics,
            grid_size_meters=cfg.grid_size_meters,
            grid_resolution=cfg.grid_resolution,
            ground_y_max=cfg.ground_y_max,
            use_equirectangular=cfg.use_equirectangular,
            smooth_sigma=cfg.smooth_sigma,
            camera_height_meters=cfg.camera_height_meters,
            sky_mask=sky_mask,
            flood_fill=cfg.flood_fill,
            flood_fill_max_step=cfg.flood_fill_max_step,
            panorama_depth=fill_panorama_depth,
            region_type_mask=region_type_mask,
            nadir_exclusion_radius=cfg.nadir_exclusion_radius,
            nadir_ramp_width=cfg.nadir_ramp_width,
            flat_zone_certainty=cfg.flat_zone_certainty,
            certainty_falloff_meters=cfg.certainty_falloff_meters,
            min_forward_samples=cfg.min_forward_samples,
            fill_boundary_falloff_cells=cfg.fill_boundary_falloff_cells,
            min_component_area_fraction=cfg.min_component_area_fraction,
            despike_threshold_m=cfg.despike_threshold_m,
            despike_window=cfg.despike_window,
            debug_dir=self.temp,
        )
        self.advance_progress(task)

        height_map = Depth(height_array)
        context.add_depth(output_key, height_map)
        context.add_depth(ContextKey.HEIGHT_MAP_CERTAINTY, Depth(certainty_array))
        context.add_depth(ContextKey.HEIGHT_MAP_CELL_RELIEF, Depth(cell_relief_array))
        context.add_depth(ContextKey.HEIGHT_MAP_CELL_SLOPE, Depth(cell_slope_array))
        context.add_depth(
            ContextKey.HEIGHT_MAP_OBSERVED_MASK, Depth(true_observed_array.astype(np.float32))
        )
        context.add_depth(
            ContextKey.HEIGHT_MAP_COMPONENT_ID, Depth(component_id_array.astype(np.float32))
        )

        context.add_object(ContextKey.HEIGHT_MAP_PARAMS, {
            "grid_size_meters":    cfg.grid_size_meters,
            "grid_resolution":     cfg.grid_resolution,
            "ground_y_max":        cfg.ground_y_max,
            "camera_height_meters": cfg.camera_height_meters,
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
