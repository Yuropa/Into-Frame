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
        # Peak sharpening: exponent applied to normalised heights.
        # None = auto-derive spatially from ridge chain silhouette.
        # Explicit float = uniform exponent across the whole terrain.
        peak_sharpness: float | None = None,
        peak_sharpness_min: float = 1.0,
        peak_sharpness_max: float = 2.5,
        peak_sharpness_window_m: float = 20.0,
        peak_sharpness_spread_m: float = 15.0,
        # Hydrological erosion (stream power incision).
        # Derives the complete drainage network from terrain topology — including
        # mountain channels not observed in the water chains — then carves valleys
        # proportional to drainage area × slope.
        hydro_enabled: bool = True,
        hydro_erodibility: float = 1e-5,   # K_sp stream power erodibility (m^(1-2m)/yr)
        hydro_dt: float = 1000.0,          # timestep per erosion step (conceptual years)
        hydro_n_steps: int = 10,           # number of erosion steps
        hydro_resolution: int = 256,       # downsample to this for flow routing (speed)
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
        self.peak_sharpness = peak_sharpness
        self.peak_sharpness_min = peak_sharpness_min
        self.peak_sharpness_max = peak_sharpness_max
        self.peak_sharpness_window_m = peak_sharpness_window_m
        self.peak_sharpness_spread_m = peak_sharpness_spread_m
        self.hydro_enabled = hydro_enabled
        self.hydro_erodibility = hydro_erodibility
        self.hydro_dt = hydro_dt
        self.hydro_n_steps = hydro_n_steps
        self.hydro_resolution = hydro_resolution


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

        # ── Pass 4: Peak Sharpening ───────────────────────────────────────────────
        # Normalise to [0,1], apply h^sharpness_map element-wise, rescale back.
        # sharpness_map is (H, W): varies spatially so jagged ridgeline zones get
        # pointier peaks while smoother areas of the terrain stay rounded.
        fixed_sharpness = cfg.peak_sharpness
        if fixed_sharpness is None:
            chains = context.input_object(ContextKey.MOUNTAIN_RIDGE_CHAINS) or []
            sharpness_map = TerrainNoiseRefinementStage._sharpness_map_from_chains(
                chains, H, W, grid_size,
                cfg.peak_sharpness_min,
                cfg.peak_sharpness_max,
                window_m=cfg.peak_sharpness_window_m,
                spread_sigma_m=cfg.peak_sharpness_spread_m,
            )
            self.log_info(
                f"Terrain noise refinement: auto sharpness map "
                f"[{sharpness_map.min():.2f}, {sharpness_map.max():.2f}]"
            )
        else:
            sharpness_map = np.full((H, W), fixed_sharpness, dtype=np.float32)

        lo, hi = terrain.min(), terrain.max()
        if hi > lo:
            t = ((terrain - lo) / (hi - lo)).clip(0.0, 1.0)
            terrain = lo + (hi - lo) * np.power(t, sharpness_map)

        # ── Pass 5: Hydrological Erosion ─────────────────────────────────────
        # Derives the complete drainage network from terrain topology via flow
        # routing — capturing mountain channels that are absent from water_chains
        # (anything beyond camera range won't have been segmented). FastscapeEroder
        # then carves channels proportional to drainage_area^m × slope^n, giving
        # realistic valley incision at all scales: deep trunk valleys where many
        # headwaters merge, shallow rills near divides.
        if cfg.hydro_enabled and cfg.hydro_n_steps > 0:
            solve_res = min(cfg.hydro_resolution, H, W)
            if solve_res < H:
                t_small = nd_zoom(terrain, solve_res / H, order=1)
                cell_size_m = grid_size / solve_res
                t_small = TerrainNoiseRefinementStage._hydro_erode(
                    t_small, cell_size_m,
                    cfg.hydro_erodibility, cfg.hydro_dt, cfg.hydro_n_steps,
                )
                terrain = nd_zoom(t_small, H / solve_res, order=3)
            else:
                cell_size_m = grid_size / H
                terrain = TerrainNoiseRefinementStage._hydro_erode(
                    terrain, cell_size_m,
                    cfg.hydro_erodibility, cfg.hydro_dt, cfg.hydro_n_steps,
                )
            self.log_info(
                f"Terrain noise refinement: hydrological erosion "
                f"(K={cfg.hydro_erodibility:.1e}, dt={cfg.hydro_dt:.0f}×{cfg.hydro_n_steps})"
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
    def _sharpness_map_from_chains(
        chains: list,
        H: int,
        W: int,
        grid_size_meters: float,
        sharpness_min: float,
        sharpness_max: float,
        window_m: float = 20.0,
        spread_sigma_m: float = 15.0,
        ref_low: float = 0.05,
        ref_high: float = 0.20,
    ) -> np.ndarray:
        """
        Build a (H, W) sharpness exponent map from ridge chain silhouettes.

        For each chain:
          1. Arc-length-parameterise the XZ path and resample Y to 1-sample/metre.
          2. Slide a window of width window_m metres along the resampled Y profile;
             at each step score local jaggedness as std(diff(Y_win)) / range(Y_win).
          3. Project each sample point (X, Z) to its heightmap pixel and splat the
             jaggedness score there weighted by 1.

        All splatted values are then Gaussian-spread by spread_sigma_m metres so
        the influence diffuses smoothly away from the ridgeline.  Pixels with no
        chain coverage fall back to sharpness_min.

        Jaggedness is mapped linearly to [sharpness_min, sharpness_max]:
          ref_low  → sharpness_min (smooth/rounded)
          ref_high → sharpness_max (sharp alpine)
        """
        half = grid_size_meters / 2.0
        score_acc  = np.zeros((H, W), dtype=np.float64)
        weight_acc = np.zeros((H, W), dtype=np.float64)

        for chain in chains:
            if len(chain) < 4:
                continue
            pts = np.asarray(chain, dtype=np.float64)
            X_c, Y_c, Z_c = pts[:, 0], pts[:, 1], pts[:, 2]

            # Arc-length parameterisation in XZ.
            dxz    = np.sqrt(np.diff(X_c) ** 2 + np.diff(Z_c) ** 2)
            arc    = np.concatenate([[0.0], np.cumsum(dxz)])
            total  = arc[-1]
            if total < 1e-6:
                continue

            # Resample to ~1 sample/metre for uniform derivative computation.
            n_samples  = max(4, int(total))
            s_uni      = np.linspace(0.0, total, n_samples)
            Y_uni      = np.interp(s_uni, arc, Y_c)
            X_uni      = np.interp(s_uni, arc, X_c)
            Z_uni      = np.interp(s_uni, arc, Z_c)

            half_w = max(2, int(window_m / 2))
            local_scores = np.empty(n_samples)
            for i in range(n_samples):
                lo_i, hi_i = max(0, i - half_w), min(n_samples, i + half_w + 1)
                Y_win  = Y_uni[lo_i:hi_i]
                y_rng  = Y_win.max() - Y_win.min()
                local_scores[i] = (
                    float(np.std(np.diff(Y_win)) / y_rng)
                    if y_rng > 1e-6 and len(Y_win) >= 3
                    else 0.0
                )

            # Map jaggedness → sharpness.
            t_scores  = np.clip((local_scores - ref_low) / (ref_high - ref_low), 0.0, 1.0)
            sharpness = sharpness_min + (sharpness_max - sharpness_min) * t_scores

            # Project to heightmap pixels and accumulate.
            cols = ((X_uni + half) / grid_size_meters * W).clip(0, W - 1).astype(int)
            rows = ((Z_uni + half) / grid_size_meters * H).clip(0, H - 1).astype(int)
            np.add.at(score_acc,  (rows, cols), sharpness)
            np.add.at(weight_acc, (rows, cols), 1.0)

        # Gaussian spread so influence diffuses smoothly away from ridgelines.
        spread_px   = max(1.0, spread_sigma_m / grid_size_meters * max(H, W))
        score_blur  = gaussian_filter(score_acc,  sigma=spread_px)
        weight_blur = gaussian_filter(weight_acc, sigma=spread_px)

        covered      = weight_blur > 1e-6
        sharpness_map = np.where(
            covered,
            (score_blur / weight_blur.clip(1e-9)).clip(sharpness_min, sharpness_max),
            sharpness_min,
        )
        return sharpness_map.astype(np.float32)

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

    @staticmethod
    def _hydro_erode(
        terrain: np.ndarray,
        cell_size_m: float,
        K_sp: float,
        dt: float,
        n_steps: int,
    ) -> np.ndarray:
        """
        Carve fluvial channels into the terrain via stream power erosion.

        Fills closed depressions so flow can route freely, then runs
        FlowAccumulator + FastscapeEroder for n_steps × dt. Flow routing derives
        the complete drainage network from terrain topology — including mountain
        headwater streams that are absent from observed water chains. Erosion
        depth scales with drainage_area^0.5 × slope, so high-order trunk valleys
        cut deep while headwater rills remain subtle.
        """
        from landlab import RasterModelGrid
        from landlab.components import SinkFillerBarnes, FlowAccumulator, FastscapeEroder

        H, W = terrain.shape
        mg = RasterModelGrid((H, W), xy_spacing=cell_size_m)
        # All four edges are open outlets so the drainage network can exit the grid.
        mg.set_closed_boundaries_at_grid_edges(
            right_is_closed=False, top_is_closed=False,
            left_is_closed=False, bottom_is_closed=False,
        )
        mg.add_field("topographic__elevation", terrain.ravel().copy(), at="node")

        # Fill closed sinks once so every interior cell can route to an outlet.
        # Re-filling each step would be correct but prohibitively slow.
        sf = SinkFillerBarnes(mg, method="Steepest", fill_flat=False)
        sf.run_one_step()

        fa = FlowAccumulator(mg, flow_director="FlowDirectorSteepest")
        fsc = FastscapeEroder(mg, K_sp=K_sp, m_sp=0.5, n_sp=1.0)
        for _ in range(n_steps):
            fa.run_one_step()
            fsc.run_one_step(dt)

        return mg.at_node["topographic__elevation"].reshape(H, W).astype(np.float64)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.HEIGHT_MAP)

    def model_names(self) -> list[str]:
        return []
