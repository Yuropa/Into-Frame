"""
TerrainReconstructionStage — Landlab harmonic heightmap reconstruction.

Replaces the weighted sparse least-squares solver with steady-state diffusion:

  1. Mark high-confidence observed terrain as Dirichlet (fixed) boundary conditions.
  2. Mark distant mountain ridge anchor points as fixed at their extracted elevation.
  3. Mark river path nodes as fixed at a lowered elevation to carve valleys.
  4. Mark water chain nodes as fixed at their water surface elevation.
  5. Solve ∇²h = 0 on all remaining (free) nodes.

The harmonic solution is the unique smooth surface through all fixed boundary
values — the exact steady state that Landlab's LinearDiffuser converges to,
reached here in one direct sparse solve rather than hundreds of thousands of
explicit timesteps.

Landlab's RasterModelGrid manages node statuses (CORE vs FIXED_VALUE).
scipy assembles and solves the resulting Laplacian system.

Each mountain chain operates independently: it only influences its local
neighbourhood through diffusion spreading from its fixed nodes. There is no
global coupling between distant chains. The noise refinement stage's
Landlab LinearDiffuser then handles hillslope shaping and smoothing.
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Any
from logging import Logger
from scipy.ndimage import zoom as nd_zoom, gaussian_filter, distance_transform_edt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth


class TerrainReconstructionConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        solve_resolution: int = 512,
        confidence_threshold: float = 0.3,
        ridge_min_anchor_distance: float = 0.5,
        ridge_profile_sigma_m: float = 20.0,
        river_valley_depth: float = 0.5,
        river_drop_per_segment: float = 0.05,
        lake_y_range_threshold: float = 0.3,
        upsample_noise_amplitude: float = 0.02,
        upsample_noise_octaves: int = 3,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.solve_resolution = solve_resolution
        self.confidence_threshold = confidence_threshold
        self.ridge_min_anchor_distance = ridge_min_anchor_distance
        self.ridge_profile_sigma_m = ridge_profile_sigma_m
        self.river_valley_depth = river_valley_depth
        self.river_drop_per_segment = river_drop_per_segment
        self.lake_y_range_threshold = lake_y_range_threshold
        self.upsample_noise_amplitude = upsample_noise_amplitude
        self.upsample_noise_octaves = upsample_noise_octaves


class TerrainReconstructionStage(PipelineStage):
    """
    Builds a globally coherent heightmap via harmonic interpolation.

    Observed terrain, ridge anchors, rivers, and water bodies become Dirichlet
    boundary conditions; the solver finds the unique smooth surface through all
    of them. Mountains reach their correct elevation because their nodes are
    pinned directly — no weight tuning required.
    """

    @classmethod
    def config_class(cls) -> type[TerrainReconstructionConfiguration]:
        return TerrainReconstructionConfiguration

    def __init__(self, config: TerrainReconstructionConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TerrainReconstructionConfiguration = self.config
        task = self.create_progress(4, "Terrain Reconstruction…")

        # ── Load inputs ───────────────────────────────────────────────────────
        hm_depth = context.input_depth(ContextKey.HEIGHT_MAP)
        if hm_depth is None:
            self.log_warning("No height map — skipping terrain reconstruction")
            self.finish_progress(task)
            return context

        heightmap = hm_depth.depth.copy()   # (H, W) float32, Y in metres
        H, W = heightmap.shape

        cert_depth = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
        confidence = (
            cert_depth.depth.copy() if cert_depth is not None
            else np.ones((H, W), dtype=np.float32)
        )

        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_size = float(params.get("grid_size_meters", 100.0))

        # ── Downsample to solve_resolution ────────────────────────────────────
        solve_res = min(cfg.solve_resolution, H, W)
        if solve_res < H:
            scale = solve_res / H
            hm_s    = nd_zoom(heightmap,   (scale, scale), order=1).astype(np.float64)
            conf_s  = nd_zoom(confidence,  (scale, scale), order=1).astype(np.float64)
        else:
            hm_s   = heightmap.astype(np.float64)
            conf_s = confidence.astype(np.float64)
        H_s, W_s = hm_s.shape
        cell_size_m = grid_size / H_s

        self.log_info(f"Terrain reconstruction: solving at {H_s}×{W_s} (original {H}×{W})")
        self.advance_progress(task)

        # ── Initialise fixed-elevation grid and mask ──────────────────────────
        # fixed_elev: what each fixed node is held at (starts from observed HM)
        # fixed_mask: True → hold that cell at fixed_elev in the solve
        fixed_elev = hm_s.copy()
        fixed_mask = conf_s >= cfg.confidence_threshold    # (H_s, W_s) bool

        x_half = z_half = grid_size / 2.0

        def world_to_grid(xyz: np.ndarray):
            """Project world XYZ array (N,3) → (rows, cols) on the solve grid."""
            col = np.clip((xyz[:, 0] + x_half) / grid_size * (W_s - 1), 0, W_s - 1)
            row = np.clip((xyz[:, 2] + z_half) / grid_size * (H_s - 1), 0, H_s - 1)
            return row.round().astype(np.int32), col.round().astype(np.int32)

        # ── Ridge anchors: distant mountain chains ────────────────────────────
        # Foreground chains (close to camera) are already captured by the high-
        # confidence observed terrain; anchoring them would conflict with the
        # rising slope toward the mountains.
        ridge_chains = context.input_object(ContextKey.MOUNTAIN_RIDGE_CHAINS) or []
        min_anchor_m = cfg.ridge_min_anchor_distance * z_half
        n_anchored, n_skipped = 0, 0

        for raw_chain in ridge_chains:
            chain = np.asarray(raw_chain, dtype=np.float32)
            if len(chain) < 2:
                continue
            horiz = np.sqrt(chain[:, 0] ** 2 + chain[:, 2] ** 2)
            if float(np.median(horiz)) < min_anchor_m:
                n_skipped += 1
                continue
            rows_c, cols_c = world_to_grid(chain)
            fixed_mask[rows_c, cols_c] = True
            fixed_elev[rows_c, cols_c] = chain[:, 1].astype(np.float64)  # Y = elevation
            n_anchored += 1

        self.log_info(
            f"Ridge chains: {n_anchored} anchored (≥{min_anchor_m:.0f} m), "
            f"{n_skipped} foreground skipped"
        )

        # ── Gaussian mountain profile halo ────────────────────────────────────
        # The harmonic solve with only crest nodes produces a linear ramp from
        # flat terrain to peak. Adding a Gaussian skirt of fixed nodes gives the
        # solver the slope shape: steep near the top, tapering at the base.
        if cfg.ridge_profile_sigma_m > 0.0 and n_anchored > 0:
            sigma_px = cfg.ridge_profile_sigma_m / cell_size_m
            profile_best = np.full((H_s, W_s), -np.inf)

            for raw_chain in ridge_chains:
                chain = np.asarray(raw_chain, dtype=np.float32)
                if len(chain) < 2:
                    continue
                horiz = np.sqrt(chain[:, 0] ** 2 + chain[:, 2] ** 2)
                if float(np.median(horiz)) < min_anchor_m:
                    continue
                rows_c, cols_c = world_to_grid(chain)

                # Per-cell crest elevation (max where multiple chain points overlap)
                crest_elev_map = np.zeros((H_s, W_s), dtype=np.float64)
                np.maximum.at(crest_elev_map, (rows_c, cols_c), chain[:, 1].astype(np.float64))
                crest_mask = np.zeros((H_s, W_s), dtype=bool)
                crest_mask[rows_c, cols_c] = True

                dist_px, src = distance_transform_edt(~crest_mask, return_indices=True)
                nearest_elev = crest_elev_map[src[0], src[1]]
                profile = nearest_elev * np.exp(-0.5 * (dist_px / sigma_px) ** 2)
                profile_best = np.maximum(profile_best, profile)

            # Pin slope nodes that aren't already fixed by high-confidence observations.
            apply = (
                ~fixed_mask
                & (profile_best > fixed_elev + 0.1)
                & (profile_best > 0.5)
            )
            fixed_mask[apply] = True
            fixed_elev[apply] = profile_best[apply]
            self.log_info(f"Ridge profile halo: {int(apply.sum())} slope nodes pinned (σ={cfg.ridge_profile_sigma_m:.0f} m)")

        self.advance_progress(task)

        # ── River constraints ─────────────────────────────────────────────────
        graph = context.input_object(ContextKey.LINEAR_GRAPH)
        n_rivers = 0
        if graph is not None:
            for structure in graph.structures:
                if structure.type != "river":
                    continue
                path = np.asarray(structure.path, dtype=np.float32)
                rows_r, cols_r = world_to_grid(path)
                base = hm_s[rows_r, cols_r]  # original observed elevation as base
                drop = np.arange(len(path), dtype=np.float64) * cfg.river_drop_per_segment
                fixed_mask[rows_r, cols_r] = True
                fixed_elev[rows_r, cols_r] = base - cfg.river_valley_depth - drop
                n_rivers += 1

        # ── Water chains ──────────────────────────────────────────────────────
        water_chains = context.input_object(ContextKey.WATER_CHAINS) or []
        n_water = 0
        for raw_chain in water_chains:
            chain = np.asarray(raw_chain, dtype=np.float32)
            if len(chain) < 2:
                continue
            y_range = float(chain[:, 1].max() - chain[:, 1].min())
            water_y: Any = (
                float(np.median(chain[:, 1])) if y_range < cfg.lake_y_range_threshold
                else chain[:, 1].astype(np.float64)
            )
            rows_w, cols_w = world_to_grid(chain)
            fixed_mask[rows_w, cols_w] = True
            fixed_elev[rows_w, cols_w] = water_y
            n_water += 1

        self.log_info(
            f"Terrain reconstruction: {n_rivers} river(s), {n_water} water chain(s); "
            f"{int(fixed_mask.sum())} / {H_s * W_s} nodes fixed"
        )
        self.advance_progress(task)

        # ── Solve: harmonic interpolation ─────────────────────────────────────
        new_hm_s = TerrainReconstructionStage._landlab_harmonic_solve(
            fixed_elev, fixed_mask, cell_size_m,
        )

        # ── Upsample back to original resolution ──────────────────────────────
        if H_s < H or W_s < W:
            new_hm = nd_zoom(new_hm_s, (H / H_s, W / W_s), order=3).astype(np.float32)
        else:
            new_hm = new_hm_s.astype(np.float32)

        # ── Fine-grain noise on the upsampled grid ────────────────────────────
        if cfg.upsample_noise_amplitude > 0.0:
            rng = np.random.default_rng(cfg.seed)
            noise = np.zeros((H, W), dtype=np.float32)
            for octave in range(cfg.upsample_noise_octaves):
                amplitude = 0.5 ** octave
                raw = rng.standard_normal((H, W)).astype(np.float32) * amplitude
                sigma = max(1.0, min(H, W) / (4.0 * (2 ** octave)))
                noise += gaussian_filter(raw, sigma=sigma)
            peak = float(np.abs(noise).max())
            if peak > 0.0:
                noise /= peak
            new_hm += noise * cfg.upsample_noise_amplitude

        context.add_depth(ContextKey.HEIGHT_MAP, Depth(new_hm))

        if self.temp is not None:
            Depth(new_hm).save_debug_image(self.temp / "heightmap_reconstructed.png")
            diff = np.abs(new_hm - heightmap)
            Depth(diff).save_debug_image(self.temp / "heightmap_reconstruction_diff.png")
            from pipeline.heightmap.heightmap_generator import HeightMapGenerator
            cert = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
            cert_arr = cert.depth if cert is not None else np.zeros_like(new_hm)
            HeightMapGenerator._save_radial_profile(
                new_hm, cert_arr, grid_size,
                self.temp / "heightmap_reconstructed_radial_profile.json",
            )

        y_before = (heightmap.min(), heightmap.max())
        y_after  = (new_hm.min(),   new_hm.max())
        self.log_info(
            f"Y range: [{y_before[0]:.2f}, {y_before[1]:.2f}] → "
            f"[{y_after[0]:.2f}, {y_after[1]:.2f}]"
        )

        self.finish_progress(task)
        return context

    @staticmethod
    def _landlab_harmonic_solve(
        fixed_elev: np.ndarray,
        fixed_mask: np.ndarray,
        cell_size_m: float,
    ) -> np.ndarray:
        """
        Solve ∇²h = 0 on free (CORE) nodes with Dirichlet BCs at fixed nodes.

        Landlab's RasterModelGrid sets up node boundary statuses.
        scipy assembles and solves the sparse Laplacian system in one direct pass.

        This is the exact steady-state solution that Landlab's LinearDiffuser
        converges to — solved here without the O((L/dx)²) explicit timesteps.
        Each fixed-node cluster (mountain chain, flat terrain, river) operates
        independently; the harmonic solution smoothly bridges between them.
        """
        from landlab import RasterModelGrid

        H, W = fixed_elev.shape
        mg = RasterModelGrid((H, W), xy_spacing=cell_size_m)
        z = mg.add_field("topographic__elevation", fixed_elev.ravel().copy(), at="node")

        # Mark only explicitly-specified nodes as fixed — do not pin the grid
        # perimeter. Boundary free nodes get natural Neumann (zero-flux) BCs:
        # adjacent_nodes_at_node returns -1 for out-of-grid directions, so those
        # nodes simply have lower degree in the Laplacian (2 or 3 instead of 4).
        mg.status_at_node[:] = mg.BC_NODE_IS_CORE
        mg.status_at_node[fixed_mask.ravel()] = mg.BC_NODE_IS_FIXED_VALUE

        core = mg.core_nodes   # all non-fixed nodes
        n_core = len(core)
        if n_core == 0:
            return fixed_elev.copy()

        core_map = np.full(mg.number_of_nodes, -1, dtype=np.int64)
        core_map[core] = np.arange(n_core)

        # adjacent_nodes_at_node: (N, 4) — -1 where no grid neighbour exists.
        # Handles perimeter nodes correctly without special-casing.
        all_nb = mg.adjacent_nodes_at_node[core]  # (n_core, 4)
        valid = all_nb >= 0
        degree = valid.sum(axis=1).astype(np.float64)

        is_fixed = mg.status_at_node != mg.BC_NODE_IS_CORE

        ki, di = np.where(valid)
        nb = all_nb[ki, di]
        is_fixed_nb = is_fixed[nb]

        # RHS: sum fixed-neighbour elevations for each core node.
        rhs = np.zeros(n_core, dtype=np.float64)
        np.add.at(rhs, ki[is_fixed_nb], z[nb[is_fixed_nb]])

        # Off-diagonal: -1 coupling to each free neighbour.
        ki_off = ki[~is_fixed_nb]
        j_off = core_map[nb[~is_fixed_nb]]

        rows = np.concatenate([np.arange(n_core), ki_off])
        cols = np.concatenate([np.arange(n_core), j_off])
        vals = np.concatenate([degree, np.full(len(ki_off), -1.0)])

        A = csr_matrix((vals, (rows, cols)), shape=(n_core, n_core))
        x = spsolve(A, rhs)

        result = z.copy()
        result[core] = x
        return result.reshape(H, W)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.HEIGHT_MAP)

    def model_names(self) -> list[str]:
        return []
