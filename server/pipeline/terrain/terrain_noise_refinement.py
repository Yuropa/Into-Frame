"""
TerrainNoiseRefinementStage — post-reconstruction heightmap enhancement.

Applies three passes after TerrainReconstructionStage:

  1. Road grading     : blends terrain with a locally smooth proxy along the
                        detected road skeleton, simulating cut-and-fill.
  2. fBm noise        : multi-octave Perlin (fractal Brownian motion) applied
                        uniformly, creating rolling hills at the meso-scale
                        (noise_scale controls the base wavelength).
  3. Landlab diffusion: Landlab LinearDiffuser applied at a reduced resolution
                        for speed, giving hillslope creep that rounds sharp
                        solver artefacts and creates geomorphically realistic
                        slopes without the uniform blurring of Laplacian
                        diffusion or the staircase artefacts of thermal erosion.

Pipeline position: after TerrainReconstructionStage, before TerrainMeshStage.

Reads:
  ContextKey.HEIGHT_MAP       (Depth)          — reconstructed DEM
  ContextKey.HEIGHT_MAP_PARAMS (dict, optional) — grid_size_meters
  ContextKey.ROAD_SKELETON    (Depth, optional) — binary road/trail mask

Writes:
  ContextKey.HEIGHT_MAP       (Depth)          — refined DEM
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as nd_zoom
from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth


class TerrainNoiseRefinementConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        # Road grading
        road_blend_weight: float = 0.8,
        road_blur_sigma: float = 2.0,
        road_terrain_smooth_sigma: float = 3.0,
        # fBm noise
        noise_scale: float = 40.0,
        noise_octaves: int = 4,
        noise_amplitude: float = 0.4,
        # Landlab hillslope diffusion
        linear_diffusivity: float = 1e-3,
        diffusion_dt: float = 200.0,
        landlab_resolution: int = 1024,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.road_blend_weight = road_blend_weight
        self.road_blur_sigma = road_blur_sigma
        self.road_terrain_smooth_sigma = road_terrain_smooth_sigma
        self.noise_scale = noise_scale
        self.noise_octaves = noise_octaves
        self.noise_amplitude = noise_amplitude
        self.linear_diffusivity = linear_diffusivity
        self.diffusion_dt = diffusion_dt
        self.landlab_resolution = landlab_resolution


class TerrainNoiseRefinementStage(PipelineStage):
    """
    Adds meso-scale terrain variation and realistic hillslope morphology to the
    reconstructed heightmap via fBm noise and Landlab hillslope diffusion.
    """

    @classmethod
    def config_class(cls) -> type[TerrainNoiseRefinementConfiguration]:
        return TerrainNoiseRefinementConfiguration

    def __init__(self, config: TerrainNoiseRefinementConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TerrainNoiseRefinementConfiguration = self.config
        task = self.create_progress(3, "Terrain Noise Refinement…")

        hm_depth = context.input_depth(ContextKey.HEIGHT_MAP)
        if hm_depth is None:
            self.log_warning("No height map — skipping terrain noise refinement")
            self.finish_progress(task)
            return context

        terrain = hm_depth.depth.copy().astype(np.float64)
        H, W = terrain.shape

        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_size = float(params.get("grid_size_meters", 100.0))

        # ── Pass 1: Road Grading ──────────────────────────────────────────────
        road_depth = context.input_depth(ContextKey.ROAD_SKELETON)
        if road_depth is not None:
            road_raw = road_depth.depth.astype(np.float64)
            if road_raw.shape != (H, W):
                from PIL import Image as PILImage
                peak = road_raw.max()
                road_img = PILImage.fromarray(
                    ((road_raw / (peak + 1e-9)) * 255).clip(0, 255).astype(np.uint8)
                )
                road_img = road_img.resize((W, H), resample=PILImage.BILINEAR)
                road_raw = np.asarray(road_img).astype(np.float64) / 255.0
            else:
                peak = road_raw.max()
                if peak > 0:
                    road_raw = road_raw / peak

            road_influence = gaussian_filter(road_raw, sigma=cfg.road_blur_sigma)
            graded = gaussian_filter(terrain, sigma=cfg.road_terrain_smooth_sigma)
            blend = road_influence * cfg.road_blend_weight
            terrain = (1.0 - blend) * terrain + blend * graded
            self.log_info("Terrain noise refinement: road grading applied")
        else:
            self.log_info("Terrain noise refinement: no road skeleton, skipping road grading")
        self.advance_progress(task)

        # ── Pass 2: fBm Noise ─────────────────────────────────────────────────
        noise_layer = self._make_noise(H, W, cfg.noise_scale, cfg.noise_octaves, cfg.seed)
        terrain += noise_layer * cfg.noise_amplitude
        self.advance_progress(task)

        # ── Pass 3: Landlab Hillslope Diffusion ───────────────────────────────
        # Run LinearDiffuser at a reduced resolution (landlab_resolution) for
        # speed, then bicubic-upsample the result back to original size.
        # Edge nodes are held at FIXED_VALUE so ridge/boundary heights are
        # preserved; only interior (CORE) nodes are shaped by diffusion.
        solve_res = min(cfg.landlab_resolution, H, W)
        if solve_res < H:
            terrain_small = nd_zoom(terrain, solve_res / H, order=1)
            cell_size_m = grid_size / solve_res
            terrain_small = TerrainNoiseRefinementStage._landlab_diffuse(
                terrain_small, cell_size_m,
                cfg.linear_diffusivity, cfg.diffusion_dt,
            )
            terrain = nd_zoom(terrain_small, H / solve_res, order=3)
        else:
            cell_size_m = grid_size / H
            terrain = TerrainNoiseRefinementStage._landlab_diffuse(
                terrain, cell_size_m,
                cfg.linear_diffusivity, cfg.diffusion_dt,
            )

        context.add_depth(ContextKey.HEIGHT_MAP, Depth(terrain.astype(np.float32)))

        y_min, y_max = float(terrain.min()), float(terrain.max())
        self.log_info(
            f"Terrain noise refinement: {H}×{W}, "
            f"noise_amplitude={cfg.noise_amplitude:.3f}, "
            f"landlab_resolution={solve_res}, "
            f"Y=[{y_min:.3f}, {y_max:.3f}]"
        )

        if self.temp is not None:
            Depth(terrain.astype(np.float32)).save_debug_image(
                self.temp / "heightmap_noise_refined.png"
            )

        self.finish_progress(task)
        return context

    @staticmethod
    def _landlab_diffuse(
        terrain: np.ndarray,
        cell_size_m: float,
        linear_diffusivity: float,
        dt: float,
    ) -> np.ndarray:
        """
        Apply Landlab LinearDiffuser (hillslope creep) to the terrain.

        Edge nodes are FIXED_VALUE: their elevations don't change, and the
        diffusion only reshapes interior (CORE) nodes. This preserves ridge
        heights and boundary anchors set by the solver while rounding solver
        artefacts and creating smooth, geomorphically realistic slopes.

        diffusion_length ≈ sqrt(4 * linear_diffusivity * dt). At the default
        K=1e-3 and dt=200, that's ~0.89 m at 1024-cell resolution — enough to
        round multi-cell artefacts without blurring large-scale terrain shape.
        """
        from landlab import RasterModelGrid
        from landlab.components import LinearDiffuser

        H, W = terrain.shape
        mg = RasterModelGrid((H, W), xy_spacing=cell_size_m)

        # FIXED_VALUE at all edges: boundary elevations stay pinned.
        mg.set_closed_boundaries_at_grid_edges(
            right_is_closed=False,
            top_is_closed=False,
            left_is_closed=False,
            bottom_is_closed=False,
        )

        mg.add_field("topographic__elevation", terrain.ravel().copy(), at="node")

        ld = LinearDiffuser(mg, linear_diffusivity=linear_diffusivity)
        ld.run_one_step(dt)

        return mg.at_node["topographic__elevation"].reshape(H, W).astype(np.float64)

    @staticmethod
    def _make_noise(H: int, W: int, scale: float, octaves: int, seed: int) -> np.ndarray:
        """
        fBm-like terrain noise via per-octave Gaussian-smoothed white noise.

        For each octave, white noise is generated at a coarse grid sized so that
        the Gaussian smoothing radius is always ~2 px, then bicubic-upsampled to
        (H, W).  This is fully vectorized and runs in under a second at 4096²,
        unlike the previous pnoise2 Python loop which iterated 16 M times.
        """
        import PIL.Image as _PIL

        rng = np.random.default_rng(seed)
        noise = np.zeros((H, W), dtype=np.float64)
        amplitude = 1.0
        total_amplitude = 0.0

        for octave in range(octaves):
            sigma_px = scale / (2.0 ** octave)
            if sigma_px < 1.0:
                break

            factor = max(1, int(sigma_px / 2.0))
            oh = max(H // factor, 4)
            ow = max(W // factor, 4)

            raw = rng.standard_normal((oh, ow))
            smoothed_small = gaussian_filter(raw, sigma=2.0, truncate=3.0)

            if oh != H or ow != W:
                img = _PIL.fromarray(smoothed_small.astype(np.float32))
                img = img.resize((W, H), _PIL.BICUBIC)
                smoothed = np.asarray(img).astype(np.float64)
            else:
                smoothed = smoothed_small.astype(np.float64)

            s = float(smoothed.std())
            if s > 1e-9:
                smoothed /= s

            noise += amplitude * smoothed
            total_amplitude += amplitude
            amplitude *= 0.5

        if total_amplitude > 0:
            noise /= total_amplitude

        lo, hi = noise.min(), noise.max()
        if hi > lo:
            noise = 2.0 * (noise - lo) / (hi - lo) - 1.0
        return noise

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.HEIGHT_MAP)

    def model_names(self) -> list[str]:
        return []
