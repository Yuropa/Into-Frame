from typing import Any
from logging import Logger

import numpy as np
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

        height_array, certainty_array = HeightMapGenerator.generate(
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
            debug_dir=self.temp,
        )
        self.advance_progress(task)

        height_map = Depth(height_array)
        context.add_depth(output_key, height_map)
        context.add_depth(ContextKey.HEIGHT_MAP_CERTAINTY, Depth(certainty_array))

        context.add_object(ContextKey.HEIGHT_MAP_PARAMS, {
            "grid_size_meters": cfg.grid_size_meters,
            "grid_resolution": cfg.grid_resolution,
            "ground_y_max": cfg.ground_y_max,
        })

        if self.temp is not None:
            height_map.save_debug_image(self.temp / "heightmap.png")
            Depth(certainty_array).save_debug_image(self.temp / "heightmap_certainty.png")

        self.log_info(
            f"Height map {height_array.shape}, "
            f"Y range {height_array.min():.2f} → {height_array.max():.2f} m, "
            f"certainty mean {certainty_array[certainty_array > 0].mean():.2f}"
        )

        self.finish_progress(task)
        return context

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
