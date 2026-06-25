"""
TerrainNoiseRefinementStage — post-reconstruction heightmap enhancement.

Applies two passes after TerrainReconstructionStage:

  1. Road grading        : blends terrain with a locally smooth proxy along the
                           detected road skeleton, simulating cut-and-fill.
  2. Anisotropic noise   : Perlin noise modulated by the inverse gradient magnitude,
     + thermal weathering  so cragginess builds on natural slopes but is muted on
                           roads and river channels.  A Laplacian weathering pass
                           then redistributes material from unstable slopes.

Pipeline position: after TerrainReconstructionStage, before TerrainMeshStage.

Reads:
  ContextKey.HEIGHT_MAP       (Depth)          — reconstructed DEM
  ContextKey.ROAD_SKELETON    (Depth, optional) — binary road/trail mask

Writes:
  ContextKey.HEIGHT_MAP       (Depth)          — refined DEM
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, laplace
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
        # Anisotropic Perlin noise
        noise_scale: float = 40.0,
        noise_octaves: int = 4,
        noise_amplitude: float = 0.4,
        # Controls how aggressively steep slopes suppress noise.
        # Units: height-map value per pixel. Increase if noise bleeds onto
        # roads/rivers; decrease if natural slopes look too smooth.
        noise_gradient_k: float = 0.3,
        # Thermal weathering
        weathering_iterations: int = 5,
        weathering_rate: float = 0.15,
        talus_threshold: float = 0.05,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.road_blend_weight = road_blend_weight
        self.road_blur_sigma = road_blur_sigma
        self.road_terrain_smooth_sigma = road_terrain_smooth_sigma
        self.noise_scale = noise_scale
        self.noise_octaves = noise_octaves
        self.noise_amplitude = noise_amplitude
        self.noise_gradient_k = noise_gradient_k
        self.weathering_iterations = weathering_iterations
        self.weathering_rate = weathering_rate
        self.talus_threshold = talus_threshold


class TerrainNoiseRefinementStage(PipelineStage):
    """
    Adds fine-grained surface detail and road grading to the reconstructed
    heightmap without disturbing the solver's ridge/river constraints.
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

        # ── Pass 2: Gradient-Weighted Perlin Noise ────────────────────────────
        noise_layer = self._make_noise(H, W, cfg.noise_scale, cfg.noise_octaves, cfg.seed)

        gy, gx = np.gradient(terrain)
        slope_mag = np.sqrt(gx ** 2 + gy ** 2)
        diffusion_coeff = np.exp(-(slope_mag / cfg.noise_gradient_k) ** 2)

        terrain += noise_layer * diffusion_coeff * cfg.noise_amplitude
        self.advance_progress(task)

        # ── Pass 3: Thermal Weathering ────────────────────────────────────────
        for _ in range(cfg.weathering_iterations):
            deltas = laplace(terrain)
            erode_mask = np.abs(deltas) > cfg.talus_threshold
            terrain[erode_mask] += deltas[erode_mask] * cfg.weathering_rate

        context.add_depth(ContextKey.HEIGHT_MAP, Depth(terrain.astype(np.float32)))

        y_min, y_max = float(terrain.min()), float(terrain.max())
        self.log_info(
            f"Terrain noise refinement: {H}×{W}, "
            f"noise_amplitude={cfg.noise_amplitude:.3f}, "
            f"weathering_iters={cfg.weathering_iterations}, "
            f"Y=[{y_min:.3f}, {y_max:.3f}]"
        )

        if self.temp is not None:
            Depth(terrain.astype(np.float32)).save_debug_image(
                self.temp / "heightmap_noise_refined.png"
            )

        self.finish_progress(task)
        return context

    @staticmethod
    def _make_noise(H: int, W: int, scale: float, octaves: int, seed: int) -> np.ndarray:
        """Vectorized Perlin noise, offset by seed so each run is uncorrelated."""
        from noise import pnoise2

        offset_x = (seed * 0.3713) % 1024.0
        offset_y = (seed * 0.1931) % 1024.0

        ii = (np.arange(H, dtype=np.float64) + offset_y) / scale
        jj = (np.arange(W, dtype=np.float64) + offset_x) / scale
        ig, jg = np.meshgrid(ii, jj, indexing="ij")

        fn = np.frompyfunc(lambda x, y: pnoise2(x, y, octaves=octaves), 2, 1)
        noise = fn(ig.ravel(), jg.ravel()).astype(np.float64).reshape(H, W)

        lo, hi = noise.min(), noise.max()
        if hi > lo:
            noise = 2.0 * (noise - lo) / (hi - lo) - 1.0
        return noise

    def has_expected_output(self, context: PipelineContext) -> bool:
        return False

    def model_names(self) -> list[str]:
        return []
