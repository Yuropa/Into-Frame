"""
TerrainReconstructionStage — feature-primitive heightmap refinement.

Reads the existing heightmap + confidence map, then incorporates detected
linear structures (rivers) and mountain silhouette (ridges) as sparse
least-squares constraints, producing a globally coherent, modified DEM.

Pipeline position: after HeightMapStage + RegionMapStage + LinearStructureStage,
                   before TerrainMeshStage.

Reads:
  ContextKey.HEIGHT_MAP              (Depth)         — existing DEM
  ContextKey.HEIGHT_MAP_CERTAINTY    (Depth)         — per-pixel confidence [0, 1]
  ContextKey.HEIGHT_MAP_PARAMS       (dict)          — grid_size_meters, etc.
  ContextKey.LINEAR_GRAPH            (LinearGraph)   — world-space polylines
  ContextKey.MOUNTAIN_SILHOUETTE     (Depth)         — binary ridge grid (fallback)
  ContextKey.MOUNTAIN_RIDGE_CHAINS   (list[ndarray]) — (M,3) XYZ ridge polylines
  ContextKey.WATER_CHAINS            (list[ndarray]) — (M,3) XYZ water polylines

Writes:
  ContextKey.HEIGHT_MAP              (Depth)         — refined DEM
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Any
from logging import Logger
from scipy.ndimage import zoom as nd_zoom, gaussian_filter

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.terrain.terrain_solver import TerrainSolver
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
        laplacian_weight: float = 0.01,
        heightmap_data_weight: float = 0.3,
        ridge_weight: float = 2.0,
        ridge_crest_height: float = 0.5,
        ridge_sigma: float = 15.0,
        ridge_anchor_weight: float = 1.0,
        ridge_anchor_stride: int = 5,
        river_weight: float = 2.0,
        river_valley_depth: float = 0.5,
        river_drop_per_segment: float = 0.05,
        river_sigma: float = 10.0,
        river_anchor_weight: float = 2.0,
        river_anchor_stride: int = 5,
        lake_y_range_threshold: float = 0.3,
        solve_resolution: int = 512,
        upsample_noise_amplitude: float = 0.02,
        upsample_noise_octaves: int = 3,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.laplacian_weight = laplacian_weight
        self.heightmap_data_weight = heightmap_data_weight
        self.ridge_weight = ridge_weight
        self.ridge_crest_height = ridge_crest_height
        self.ridge_sigma = ridge_sigma
        self.ridge_anchor_weight = ridge_anchor_weight
        self.ridge_anchor_stride = ridge_anchor_stride
        self.river_weight = river_weight
        self.river_valley_depth = river_valley_depth
        self.river_drop_per_segment = river_drop_per_segment
        self.river_sigma = river_sigma
        self.river_anchor_weight = river_anchor_weight
        self.river_anchor_stride = river_anchor_stride
        self.lake_y_range_threshold = lake_y_range_threshold
        self.solve_resolution = solve_resolution
        self.upsample_noise_amplitude = upsample_noise_amplitude
        self.upsample_noise_octaves = upsample_noise_octaves


class TerrainReconstructionStage(PipelineStage):
    """
    Refines the heightmap using sparse least-squares constrained optimization.

    Feature constraints come from:
      - Rivers  : world-space polylines from LinearGraph, weighted to carve valleys
                  with monotonically descending centrelines.
      - Ridges  : mountain silhouette grid projected onto the height-map coordinate
                  system, weighted to lift ridge cells above their neighbours.

    The resulting heightmap replaces ContextKey.HEIGHT_MAP for downstream stages
    (TerrainMeshStage, etc.) while leaving certainty and params untouched.
    """

    @classmethod
    def config_class(cls) -> type[TerrainReconstructionConfiguration]:
        return TerrainReconstructionConfiguration

    def __init__(self, config: TerrainReconstructionConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TerrainReconstructionConfiguration = self.config
        task = self.create_progress(4, "Terrain Reconstruction…")

        # ── Load height map ───────────────────────────────────────────────────
        hm_depth = context.input_depth(ContextKey.HEIGHT_MAP)
        if hm_depth is None:
            self.log_warning("No height map — skipping terrain reconstruction")
            self.finish_progress(task)
            return context

        heightmap = hm_depth.depth.copy()          # (H, W) float32
        H, W = heightmap.shape

        cert_depth = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
        confidence = cert_depth.depth.copy() if cert_depth is not None else np.ones((H, W), dtype=np.float32)
        # ground_projection_certainty → 0 as r → ∞, which makes the data term
        # negligible for most of the grid and leaves absolute elevation
        # underdetermined. A small floor keeps every cell weakly anchored to the
        # original heightmap so ridge/river shape constraints can't drift it globally.
        confidence = np.maximum(confidence, 0.05)

        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_size = params.get("grid_size_meters", 100.0)

        # ── Downsample to solve_resolution for speed ──────────────────────────
        solve_res = min(cfg.solve_resolution, H, W)
        scale_h = solve_res / H
        scale_w = solve_res / W
        if scale_h < 1.0:
            heightmap_solve = nd_zoom(heightmap, (scale_h, scale_w), order=1).astype(np.float64)
            confidence_solve = nd_zoom(confidence, (scale_h, scale_w), order=1).astype(np.float64)
        else:
            heightmap_solve = heightmap.astype(np.float64)
            confidence_solve = confidence.astype(np.float64)
        H_s, W_s = heightmap_solve.shape
        self.log_info(
            f"Terrain reconstruction: solving at {H_s}×{W_s} (original {H}×{W})"
        )
        self.advance_progress(task)

        # ── Build solver ──────────────────────────────────────────────────────
        self.log_info(f"Terrain reconstruction: heightmap_data_weight={cfg.heightmap_data_weight}")
        solver = TerrainSolver(
            heightmap=heightmap_solve,
            confidence=confidence_solve,
            laplacian_weight=cfg.laplacian_weight,
            data_weight=cfg.heightmap_data_weight,
        )

        # ── Ridge constraints ─────────────────────────────────────────────────
        # Prefer 3D chains; fall back to binary mask when absent.
        #
        # Ridge chain Y values are camera-relative elevations of the sky-terrain
        # boundary (mountains at the horizon). We use add_ridge_polyline_anchored
        # so those real elevations are applied as absolute pins on the ridge cells.
        # This is safe now that laplacian_weight is low (0.01): the data term for
        # nearby high-confidence cells (weight ≈ 0.24) dominates the Laplacian
        # (≈ 0.04), keeping foreground terrain grounded while distant ridge cells
        # are shaped by the anchor and profile constraints.
        ridge_chains = context.input_object(ContextKey.MOUNTAIN_RIDGE_CHAINS) or []
        n_ridge_chains = 0
        n_ridge_mask_cells = 0

        # Sigma proportional to solve grid height: H_s/8 px ≈ one-eighth of the
        # grid radius, giving enough reach to build genuine mountain slopes.
        effective_sigma = max(cfg.ridge_sigma, H_s / 8.0)

        if ridge_chains:
            self.log_info(
                f"Ridge constraints: {len(ridge_chains)} chain(s), "
                f"sigma={effective_sigma:.1f} px (cfg={cfg.ridge_sigma:.1f})"
            )
            for chain in ridge_chains:
                chain = np.asarray(chain, dtype=np.float32)
                if len(chain) < 2:
                    continue
                solver.add_ridge_polyline_anchored(
                    points_xyz=chain,
                    grid_size_meters=grid_size,
                    weight=cfg.ridge_weight,
                    crest_height=cfg.ridge_crest_height,
                    sigma=effective_sigma,
                    anchor_weight=cfg.ridge_anchor_weight,
                    anchor_stride=cfg.ridge_anchor_stride,
                )
                n_ridge_chains += 1
        else:
            sil_depth = context.input_depth(ContextKey.MOUNTAIN_SILHOUETTE)
            if sil_depth is not None:
                sil = sil_depth.depth
                if sil.shape != (H_s, W_s):
                    from PIL import Image as PILImage
                    sil_img = PILImage.fromarray((sil > 0).astype(np.uint8) * 255)
                    sil_img = sil_img.resize((W_s, H_s), resample=PILImage.NEAREST)
                    sil = np.asarray(sil_img).astype(np.float32) / 255.0
                ridge_mask = sil > 0
                n_ridge_mask_cells = int(ridge_mask.sum())
                if n_ridge_mask_cells > 0:
                    solver.add_ridge_mask(
                        ridge_mask,
                        weight=cfg.ridge_weight,
                        crest_height=cfg.ridge_crest_height,
                        sigma=effective_sigma,
                    )
        self.advance_progress(task)

        # ── River constraints from LinearGraph ────────────────────────────────
        graph = context.input_object(ContextKey.LINEAR_GRAPH)
        n_rivers = 0
        if graph is not None:
            x_half = z_far = grid_size / 2.0
            for structure in graph.structures:
                if structure.type != "river":
                    continue
                path = structure.path  # (K, 3) world (x, y, z)
                col = np.clip((path[:, 0] + x_half) / grid_size * (W_s - 1), 0, W_s - 1)
                row = np.clip((path[:, 2] + z_far) / (2.0 * z_far) * (H_s - 1), 0, H_s - 1)
                pts = np.stack([row, col], axis=1)
                solver.add_river_polyline(
                    pts,
                    weight=cfg.river_weight,
                    valley_depth=cfg.river_valley_depth,
                    drop_per_segment=cfg.river_drop_per_segment,
                    sigma=cfg.river_sigma,
                )
                n_rivers += 1

        # ── Water chains from panorama depth ─────────────────────────────────
        water_chains = context.input_object(ContextKey.WATER_CHAINS) or []
        n_water_chains = 0
        for chain in water_chains:
            chain = np.asarray(chain, dtype=np.float32)
            if len(chain) < 2:
                continue
            y_range = float(chain[:, 1].max() - chain[:, 1].min())
            if y_range < cfg.lake_y_range_threshold:
                # Flat water body: pin to median surface elevation uniformly.
                flat_chain = chain.copy()
                flat_chain[:, 1] = float(np.median(chain[:, 1]))
                solver.add_river_polyline_anchored(
                    flat_chain,
                    grid_size,
                    weight=cfg.river_weight,
                    valley_depth=0.0,
                    sigma=cfg.river_sigma,
                    anchor_weight=cfg.river_anchor_weight * 2.0,
                    anchor_stride=cfg.river_anchor_stride,
                )
            else:
                solver.add_river_polyline_anchored(
                    chain,
                    grid_size,
                    weight=cfg.river_weight,
                    valley_depth=cfg.river_valley_depth,
                    sigma=cfg.river_sigma,
                    anchor_weight=cfg.river_anchor_weight,
                    anchor_stride=cfg.river_anchor_stride,
                )
            n_water_chains += 1
        self.advance_progress(task)

        self.log_info(
            f"Terrain reconstruction: {H}×{W} grid, "
            f"{n_ridge_chains} ridge chain(s) / {n_ridge_mask_cells} mask cells, "
            f"{n_rivers} linear-graph river(s), {n_water_chains} water chain(s)"
        )

        # ── Solve ─────────────────────────────────────────────────────────────
        new_hm_small = solver.solve()

        # ── Upsample back to original resolution ──────────────────────────────
        if H_s < H or W_s < W:
            new_hm = nd_zoom(new_hm_small, (H / H_s, W / W_s), order=3).astype(np.float32)
        else:
            new_hm = new_hm_small.astype(np.float32)

        # ── Add fine-grain noise to restore sub-solver-resolution detail ───────
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
            # Radial profile post-solver — a ring artifact shows as oscillation here
            from pipeline.heightmap.heightmap_generator import HeightMapGenerator
            cert = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
            cert_arr = cert.depth if cert is not None else np.zeros_like(new_hm)
            HeightMapGenerator._save_radial_profile(
                new_hm, cert_arr, grid_size,
                self.temp / "heightmap_reconstructed_radial_profile.json",
            )

        y_before = (heightmap.min(), heightmap.max())
        y_after  = (new_hm.min(), new_hm.max())
        self.log_info(
            f"Y range: [{y_before[0]:.2f}, {y_before[1]:.2f}] → "
            f"[{y_after[0]:.2f}, {y_after[1]:.2f}]"
        )

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(ContextKey.HEIGHT_MAP)

    def model_names(self) -> list[str]:
        return []
