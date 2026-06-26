"""
TerrainNoiseRefinementStage — post-reconstruction heightmap enhancement.

Applies three passes after TerrainReconstructionStage:

  1. Road grading     : blends terrain with a locally smooth proxy along the
                        detected road skeleton, simulating cut-and-fill.
  2. fBm noise        : multi-octave Perlin (fractal Brownian motion) applied
                        uniformly, creating rolling hills at the meso-scale
                        (noise_scale controls the base wavelength).
  3. Thermal erosion  : angle-of-repose erosion that moves material from slopes
                        steeper than talus_threshold toward lower neighbours,
                        rounding sharp peaks and depositing detritus on valley
                        floors.

Pipeline position: after TerrainReconstructionStage, before TerrainMeshStage.

Reads:
  ContextKey.HEIGHT_MAP       (Depth)          — reconstructed DEM
  ContextKey.ROAD_SKELETON    (Depth, optional) — binary road/trail mask

Writes:
  ContextKey.HEIGHT_MAP       (Depth)          — refined DEM
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
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
    Adds meso-scale terrain variation and realistic slopes to the reconstructed
    heightmap via fBm noise and angle-of-repose thermal erosion.
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

        # ── Pass 2: fBm Noise ─────────────────────────────────────────────────
        # Add Perlin fBm uniformly — no gradient suppression. Gradient suppression
        # was muting noise on mountain slopes, making them look like flat mesas, and
        # the previous noise_scale (80 px ≈ 4 m at 4.9 cm/cell) produced features
        # far too small to be visible in a 200 m terrain.  Rolling hills need 50–100 m
        # wavelengths; fine texture is added by higher octaves automatically.
        noise_layer = self._make_noise(H, W, cfg.noise_scale, cfg.noise_octaves, cfg.seed)
        terrain += noise_layer * cfg.noise_amplitude
        self.advance_progress(task)

        # ── Pass 3: Thermal Erosion ───────────────────────────────────────────
        # Real angle-of-repose erosion: move material from any slope that exceeds
        # `talus_threshold` (m/cell) toward the lower neighbour.  This rounds off
        # sharp peaks and deposits material on valley floors, unlike the previous
        # Laplacian-diffusion approach which just smoothed everything uniformly.
        terrain = TerrainNoiseRefinementStage._thermal_erode(
            terrain,
            n_iters=cfg.weathering_iterations,
            rate=cfg.weathering_rate,
            talus=cfg.talus_threshold,
        )

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
    def _thermal_erode(
        terrain: np.ndarray,
        n_iters: int,
        rate: float,
        talus: float,
    ) -> np.ndarray:
        """
        Angle-of-repose thermal erosion.

        For every pair of adjacent cells whose height difference exceeds `talus`
        (in heightmap units per grid cell), a fraction `rate` of the excess is
        moved from the higher cell to the lower one.  Repeated many times this
        rounds sharp peaks into smooth slopes and deposits detritus on valley
        floors, without the uniform blurring produced by Laplacian diffusion.

        talus: maximum stable slope in metres per grid cell.
               At 4.9 cm/cell, talus=0.025 ≈ 27° (natural loose-rock slope).
        rate:  fraction of excess moved per iteration (keep < 0.25 for stability).
        """
        terrain = terrain.copy()
        for _ in range(n_iters):
            north = np.vstack([terrain[:1, :],  terrain[:-1, :]])
            south = np.vstack([terrain[1:, :],  terrain[-1:, :]])
            west  = np.hstack([terrain[:, :1],  terrain[:, :-1]])
            east  = np.hstack([terrain[:, 1:],  terrain[:, -1:]])

            dn = np.maximum(0.0, terrain - north - talus)
            ds = np.maximum(0.0, terrain - south - talus)
            dw = np.maximum(0.0, terrain - west  - talus)
            de = np.maximum(0.0, terrain - east  - talus)

            # Remove from high cells and deposit on their lower neighbours.
            terrain -= (dn + ds + dw + de) * rate
            terrain[:-1, :] += dn[1:, :]  * rate   # north neighbour receives
            terrain[1:,  :] += ds[:-1, :] * rate   # south neighbour receives
            terrain[:, :-1] += dw[:, 1:]  * rate   # west  neighbour receives
            terrain[:, 1:]  += de[:, :-1] * rate   # east  neighbour receives

        return terrain

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

            # Downsample so the target wavelength maps to ~2 px, smooth, upsample.
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
