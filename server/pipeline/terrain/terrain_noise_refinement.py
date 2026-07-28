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

Noise, diffusion, and erosion strength are additionally scaled down continuously by
HEIGHT_MAP_CERTAINTY (see observed_trust_strength) on top of the existing hard
CLIFF_MASK exclusion -- certainty already decays smoothly with distance from real
point-cloud data (HeightMapStage's certainty_falloff_meters), so this "nudges" cells
near real data instead of applying the exact same full-strength synthesis there as
on terrain that was never observed at all. The literal true_observed boolean is still
restored exactly at the end of this stage (undoing whatever residual each pass left
even after gating), so this only changes how strongly those passes reshape cells in
the transition zone around real data, not the real cells themselves.

Pipeline position: after TerrainReconstructionStage, before TerrainMeshStage.

Reads:
  ContextKey.HEIGHT_MAP       (Depth)          — reconstructed DEM
  ContextKey.HEIGHT_MAP_PARAMS (dict, optional) — grid_size_meters
  ContextKey.HEIGHT_MAP_OBSERVED_MASK (Depth, optional) — true direct-observation
                                mask; real cells are restored after all passes run
  ContextKey.HEIGHT_MAP_CERTAINTY (Depth, optional) — continuous observed-data trust,
                                used to taper noise/diffusion/erosion near real data
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
from util.terrain_noise_utils import ridge_chain_jaggedness_map


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
        noise_amplitude: float = 0.2,
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
        hydro_erodibility: float = 5e-6,   # K_sp stream power erodibility (m^(1-2m)/yr)
        hydro_dt: float = 1000.0,          # timestep per erosion step (conceptual years)
        hydro_n_steps: int = 10,           # number of erosion steps
        hydro_resolution: int = 256,       # downsample to this for flow routing (speed)
        # Cliff protection: CLIFF_MASK cells (from TerrainReconstructionStage) are
        # excluded from hillslope diffusion and hydro erosion, and get suppressed
        # fBm noise, so measured/inferred steep faces survive this stage's smoothing.
        cliff_noise_suppression: float = 0.85,
        # How strongly HEIGHT_MAP_CERTAINTY (real point-cloud trust, already decaying
        # smoothly with distance from observed data -- see HeightMapStage's
        # certainty_falloff_meters) additionally protects cells from noise/diffusion/
        # erosion, on top of cliff protection. 0 disables (old behaviour: only the
        # literal true_observed boolean, restored at the very end, is protected --
        # everything else gets full-strength synthesis regardless of how close it is
        # to real data). 1.0 = a cell at certainty=1 is fully protected, same as a
        # cliff cell; certainty fades this out gradually rather than the hard on/off
        # edge the final true_observed-only restore leaves behind.
        observed_trust_strength: float = 1.0,
        # Post-diffusion texture for UNOBSERVED terrain. Hillslope diffusion
        # (Pass 3) rounds off the fBm from Pass 2 wherever nothing protects it --
        # worst at the nadir disc, which has no real data and no cliff/certainty
        # protection, so it diffuses into a smooth dome ringed by the rough,
        # restore-protected observed terrain around it. That smooth disc under
        # the camera, with a hard rim where it meets textured ground, is the
        # "plateau". This re-injects meso-scale fBm AFTER diffusion, scaled by
        # (1 - protect) so it lands only on unobserved/untrusted cells and fades
        # to zero into real data -- giving the nadir terrain-like variation
        # comparable to its neighbours and softening the rim, without touching
        # measured geometry. Metres of amplitude; 0 disables.
        unobserved_texture_amplitude: float = 0.6,
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
        self.cliff_noise_suppression = cliff_noise_suppression
        self.observed_trust_strength = observed_trust_strength
        self.unobserved_texture_amplitude = unobserved_texture_amplitude


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

        # True observed mask: cells with a genuine direct point-cloud measurement
        # (see HeightMapGenerator.generate docstring / HEIGHT_MAP_OBSERVED_MASK).
        # By this point TerrainReconstructionStage has already spliced real values
        # back into `terrain` at every such cell, so this snapshot is exactly what
        # gets restored at the end of this stage, undoing whatever any of the
        # passes below did to real data.
        observed_depth = context.input_depth(ContextKey.HEIGHT_MAP_OBSERVED_MASK)
        true_observed = (
            observed_depth.depth.astype(bool) if observed_depth is not None and observed_depth.depth.shape == (H, W)
            else np.zeros((H, W), dtype=bool)
        )
        original_terrain = terrain.copy()

        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_size = float(params.get("grid_size_meters", 100.0))

        # Cliff mask (from TerrainReconstructionStage): [0,1] strength combining
        # measured near-field slope and inferred distant ridge-chain jaggedness.
        # Cells with non-zero strength are protected from this stage's smoothing
        # passes — diffusion and erosion are built to round terrain, which is
        # exactly wrong for a cliff face.
        cliff_depth = context.input_depth(ContextKey.CLIFF_MASK)
        cliff_mask = (
            gaussian_filter(cliff_depth.depth.astype(np.float64), sigma=2.0)
            if cliff_depth is not None and cliff_depth.depth.shape == (H, W)
            else np.zeros((H, W), dtype=np.float64)
        )

        # Combined protection field: a cliff cell OR a cell HEIGHT_MAP_CERTAINTY says
        # we can trust (real point-cloud data nearby, continuously decaying with
        # distance -- see HeightMapStage's certainty_falloff_meters) both resist this
        # stage's noise/diffusion/erosion passes. This is what makes the true_observed
        # restore at the end of this stage a final correction rather than the *only*
        # thing protecting real data: without it, a cell one pixel outside the literal
        # observed boolean got the exact same full-strength synthetic reshaping as
        # terrain 90 m away that was never observed at all -- a hard "trust cliff"
        # right at the boundary of real coverage instead of a gradual taper.
        certainty_depth = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
        certainty_available = certainty_depth is not None and certainty_depth.depth.shape == (H, W)
        certainty = (
            certainty_depth.depth.astype(np.float64) if certainty_available
            else np.zeros((H, W), dtype=np.float64)
        )
        protect = np.maximum(cliff_mask, certainty * cfg.observed_trust_strength)
        has_protection = bool(protect.any())
        # Separate fallback for the final restore step below: certainty absent
        # there should mean "trust true_observed fully, as if HEIGHT_MAP_CERTAINTY
        # never existed" (restore weight 1), not "zero protection" (protect's own
        # fallback, correct for its purpose of gating how much noise/diffusion/
        # erosion apply -- 0 there means "no extra protection beyond cliff_mask",
        # the opposite of what the restore step needs when certainty is missing).
        restore_certainty = certainty if certainty_available else np.ones((H, W), dtype=np.float64)

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
        # Rolling-hill noise looks wrong splattered across a rock face (or over
        # real point-cloud data), so it's suppressed (not zeroed — a little texture
        # keeps cliff faces from looking perfectly planar) in proportion to combined
        # cliff + observed-trust protection.
        noise_layer = self._make_noise(H, W, cfg.noise_scale, cfg.noise_octaves, cfg.seed)
        noise_gate = (1.0 - cfg.cliff_noise_suppression * protect) if has_protection else 1.0
        terrain += noise_layer * cfg.noise_amplitude * noise_gate
        self.advance_progress(task)

        # ── Pass 3: Landlab Hillslope Diffusion ───────────────────────────────
        # Run LinearDiffuser at a reduced resolution (landlab_resolution) for
        # speed, then bicubic-upsample the result back to original size.
        # Edge nodes are held at FIXED_VALUE so ridge/boundary heights are
        # preserved; only interior (CORE) nodes are shaped by diffusion.
        pre_diffusion = terrain.copy()
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

        # Hillslope creep is a rounding process — undo it on protected cells
        # (cliffs, and cells near trusted real data) by restoring their
        # pre-diffusion elevation, feathered by protection strength so the
        # transition blends into the diffused terrain around it.
        if has_protection:
            terrain = terrain * (1.0 - protect) + pre_diffusion * protect

        # ── Pass 4: Peak Sharpening ───────────────────────────────────────────────
        # Apply h^sharpness_map to each cell's height normalised against its own
        # LOCAL base and summit, not the grid-wide min/max.
        #
        # t^s (s > 1) is the right shape operator for making a peak pointier: it
        # fixes t = 1 (the summit) and pulls mid-slope values down, so relief
        # concentrates toward the top. But that only holds when t is measured
        # against the feature's own base and its own summit. Normalising against
        # the grid-wide [min, max] instead -- which this pass used to do -- makes
        # t a measure of "how tall is this cell compared to the single highest
        # point anywhere on the grid," so every feature that isn't the tallest
        # one gets crushed toward the global minimum rather than sharpened.
        # Measured on one capture (global range [-0.38, 54.31] m, s = 2.05): 25 m
        # -> 10.95 m, 16 m -> 4.24 m, 10 m -> 1.43 m, 5 m -> 0.09 m. Only terrain
        # within a few metres of the global summit survived at all.
        #
        # That interacted badly with sharpness_map's own spatial variation: two
        # cells with near-identical heights but different exponents got wildly
        # different outputs, so any edge in sharpness_map became a cliff in the
        # terrain. On the same capture, 13.79 m at s = 1.0 and 10.22 m two metres
        # away at s = 2.05 came out as 13.79 m and 1.43 m -- a 3.6 m step turned
        # into a 10 m wall with nothing physically real behind it. (The edge
        # itself was ridge_chain_jaggedness_map's missing distance falloff, fixed
        # separately, but a local normalisation is what makes this pass robust to
        # a sharpness gradient of any steepness rather than merely to that one.)
        #
        # Local base/summit come from min/max filters over peak_sharpness_window_m
        # -- the same window the jaggedness score itself is measured over, so the
        # normalisation scope matches the scope the exponent was derived at.
        # Computed at reduced resolution (the fields are smooth by construction
        # and a metre-scale window is tens of cells wide at 4096), then blurred
        # to take the staircase edges off the filters' own plateaus.
        pre_sharpen = terrain.copy()
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

        local_lo, local_hi = TerrainNoiseRefinementStage._local_extent(
            terrain, grid_size, cfg.peak_sharpness_window_m,
        )
        span = local_hi - local_lo
        # Below ~1 cm of local relief there's no peak to sharpen and t is pure
        # numerical noise; leave those cells exactly as they are.
        active = span > 0.01
        if active.any():
            t = np.zeros_like(terrain)
            np.divide(terrain - local_lo, span, out=t, where=active)
            np.clip(t, 0.0, 1.0, out=t)
            sharpened = local_lo + span * np.power(t, sharpness_map)
            terrain = np.where(active, sharpened, terrain)

        # Same protection every other pass in this stage honours: sharpening is a
        # synthetic reshaping, and a cliff face or a directly-measured cell has no
        # business being reshaped by it. This pass was previously the only one
        # that skipped `protect` entirely, relying solely on the true_observed
        # restore at the very end -- which meant every cell in the taper zone
        # around real data got full-strength sharpening no matter how close to a
        # real measurement it sat.
        if has_protection:
            terrain = terrain * (1.0 - protect) + pre_sharpen * protect

        # ── Pass 5: Hydrological Erosion ─────────────────────────────────────
        # Derives the complete drainage network from terrain topology via flow
        # routing — capturing mountain channels that are absent from water_chains
        # (anything beyond camera range won't have been segmented). FastscapeEroder
        # then carves channels proportional to drainage_area^m × slope^n, giving
        # realistic valley incision at all scales: deep trunk valleys where many
        # headwaters merge, shallow rills near divides.
        if cfg.hydro_enabled and cfg.hydro_n_steps > 0:
            pre_erosion = terrain.copy()
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
            # Stream-power erosion carves V-shaped valleys — the wrong shape for
            # a cliff face, and not something that should carve into terrain we
            # actually measured. Restore pre-erosion elevation on protected cells.
            if has_protection:
                terrain = terrain * (1.0 - protect) + pre_erosion * protect
            self.log_info(
                f"Terrain noise refinement: hydrological erosion "
                f"(K={cfg.hydro_erodibility:.1e}, dt={cfg.hydro_dt:.0f}×{cfg.hydro_n_steps})"
            )

        # ── Pass 6: Unobserved-region texture ─────────────────────────────────
        # See unobserved_texture_amplitude. Every synthetic pass above that adds
        # fine detail (fBm, before diffusion) is either smoothed off unprotected
        # cells by diffusion or destroyed by hydro erosion's downsample-to-
        # hydro_resolution round-trip; observed cells get their detail back from
        # the restore below, but unobserved terrain -- the nadir disc above all,
        # which has no real data to restore -- is left as the smooth erosion
        # output: a flat dome under the camera ringed by textured ground (the
        # "plateau"). Re-inject meso-scale fBm HERE, after every smoothing pass
        # and before the restore, scaled by (1 - protect) so it only lands on
        # unobserved/untrusted cells and fades to nothing into real data. Runs at
        # full resolution, so nothing downstream in this stage dilutes it. A seed
        # distinct from Pass 2 so it isn't a copy of noise the earlier passes
        # already reshaped.
        if cfg.unobserved_texture_amplitude > 0.0 and has_protection:
            fill_texture = self._make_noise(H, W, cfg.noise_scale, cfg.noise_octaves, cfg.seed + 101)
            terrain = terrain + fill_texture * cfg.unobserved_texture_amplitude * (1.0 - protect)

        # ── Restore observed data ─────────────────────────────────────────────
        # A single restore covering every pass above (road grading, fBm noise,
        # diffusion, peak sharpening, hydro erosion) uniformly, rather than adding
        # per-pass exclusion logic to each one -- simpler, and avoids a pass like
        # peak sharpening (a global tone-curve remap) producing local kinks from
        # mid-algorithm exclusion.
        #
        # Scaled by `certainty` (already loaded above for `protect`), not just the
        # true_observed boolean: a hard binary restore here would silently discard
        # everything `protect`'s certainty-based tapering did earlier in this
        # stage for exactly the cells it was meant to help -- the low-certainty
        # end of true_observed (near-nadir real samples, noisiest by construction
        # since Y = depth * sin(phi) there amplifies ordinary depth noise almost
        # directly into height noise) would still get spliced back to its raw,
        # unsmoothed value at the very end regardless of how gently the earlier
        # passes treated it. High-certainty real cells still end up at ~zero net
        # synthetic influence, same as before `protect` existed; only the
        # low-certainty band is now allowed to keep some of this stage's
        # smoothing instead of being unconditionally overwritten by its own noise.
        if true_observed.any():
            restore_weight = gaussian_filter(
                true_observed.astype(np.float64) * restore_certainty, sigma=1.0
            )
            terrain = terrain * (1.0 - restore_weight) + original_terrain * restore_weight
            self.log_info(
                f"Terrain noise refinement: {int(true_observed.sum())} px eligible "
                f"(mean certainty {float(restore_certainty[true_observed].mean()):.2f}), reverted "
                f"toward measured elevation"
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
        jaggedness = ridge_chain_jaggedness_map(
            chains, H, W, grid_size_meters,
            window_m=window_m, spread_sigma_m=spread_sigma_m,
            ref_low=ref_low, ref_high=ref_high,
        )
        return (sharpness_min + (sharpness_max - sharpness_min) * jaggedness).astype(np.float32)

    @staticmethod
    def _local_extent(
        terrain: np.ndarray,
        grid_size_meters: float,
        window_m: float,
        solve_res: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-cell local minimum and maximum elevation over a window_m window.

        Used by peak sharpening to normalise each cell against its own feature's
        base and summit instead of the grid-wide extremes -- see that pass for
        why the global version is wrong.

        Computed at solve_res and upsampled: both fields are smooth by
        construction, a metres-wide window is tens of cells across at native
        resolution, and min/max filters are the expensive part. The Gaussian
        after each filter removes the flat plateaus min/max filters leave behind
        (a max filter holds the same value across the whole window around each
        local peak), which would otherwise show up as faceting in the sharpened
        output.

        Returns (local_lo, local_hi), both native-resolution float64.
        """
        from scipy.ndimage import maximum_filter, minimum_filter

        H, W = terrain.shape
        res = min(solve_res, H, W)
        small = nd_zoom(terrain, res / H, order=1) if res < H else terrain

        cell_m = grid_size_meters / res
        size = max(3, int(round(window_m / max(cell_m, 1e-6))))
        smooth_sigma = max(1.0, size / 4.0)

        lo = gaussian_filter(minimum_filter(small, size=size, mode="nearest"), sigma=smooth_sigma)
        hi = gaussian_filter(maximum_filter(small, size=size, mode="nearest"), sigma=smooth_sigma)

        if res < H:
            lo = nd_zoom(lo, H / res, order=1)
            hi = nd_zoom(hi, H / res, order=1)

        # Clamped against the NATIVE terrain, after upsampling: the blur (and the
        # upsample itself) can leave lo above / hi below a cell's own value at a
        # sharp extremum, and an un-clamped lo > terrain would push that cell UP
        # once t clips to 0. Clamping here keeps t inside [0, 1] with the summit
        # still landing exactly at 1, and guarantees sharpening can never raise a
        # cell above its own local maximum.
        lo = np.minimum(lo, terrain)
        hi = np.maximum(hi, terrain)
        return lo.astype(np.float64), hi.astype(np.float64)

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
        K=1e-3 and dt=2250, that's ~3 m — enough to round the kink where the
        ridge slope envelope snaps onto the foreground, without blurring
        large-scale terrain shape. LinearDiffuser's implicit solve has no CFL
        stability limit, so dt can be raised freely to widen this radius.
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
