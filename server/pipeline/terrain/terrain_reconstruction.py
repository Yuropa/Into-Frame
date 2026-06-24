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
  ContextKey.MOUNTAIN_SILHOUETTE     (Depth)         — binary ridge grid

Writes:
  ContextKey.HEIGHT_MAP              (Depth)         — refined DEM
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Any
from logging import Logger

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
        laplacian_weight: float = 0.1,
        ridge_weight: float = 2.0,
        ridge_crest_height: float = 0.5,
        ridge_sigma: float = 15.0,
        river_weight: float = 2.0,
        river_valley_depth: float = 0.5,
        river_drop_per_segment: float = 0.05,
        river_sigma: float = 10.0,
        solver_iter_lim: int = 500,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.laplacian_weight = laplacian_weight
        self.ridge_weight = ridge_weight
        self.ridge_crest_height = ridge_crest_height
        self.ridge_sigma = ridge_sigma
        self.river_weight = river_weight
        self.river_valley_depth = river_valley_depth
        self.river_drop_per_segment = river_drop_per_segment
        self.river_sigma = river_sigma
        self.solver_iter_lim = solver_iter_lim


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

        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_size = params.get("grid_size_meters", 100.0)
        self.advance_progress(task)

        # ── Build solver ──────────────────────────────────────────────────────
        solver = TerrainSolver(
            heightmap=heightmap,
            confidence=confidence,
            laplacian_weight=cfg.laplacian_weight,
            iter_lim=cfg.solver_iter_lim,
        )

        # ── Ridge constraints from mountain silhouette ────────────────────────
        sil_depth = context.input_depth(ContextKey.MOUNTAIN_SILHOUETTE)
        n_ridge = 0
        if sil_depth is not None:
            sil = sil_depth.depth
            # Resize silhouette to height-map dimensions if needed
            if sil.shape != (H, W):
                from PIL import Image as PILImage
                sil_img = PILImage.fromarray((sil > 0).astype(np.uint8) * 255)
                sil_img = sil_img.resize((W, H), resample=PILImage.NEAREST)
                sil = np.asarray(sil_img).astype(np.float32) / 255.0

            ridge_mask = sil > 0
            n_ridge = int(ridge_mask.sum())
            if n_ridge > 0:
                solver.add_ridge_mask(
                    ridge_mask,
                    weight=cfg.ridge_weight,
                    crest_height=cfg.ridge_crest_height,
                    sigma=cfg.ridge_sigma,
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
                # Convert world XZ → height-map (row, col)
                col = np.clip((path[:, 0] + x_half) / grid_size * (W - 1), 0, W - 1)
                row = np.clip((path[:, 2] + z_far) / (2.0 * z_far) * (H - 1), 0, H - 1)
                pts = np.stack([row, col], axis=1)
                solver.add_river_polyline(
                    pts,
                    weight=cfg.river_weight,
                    valley_depth=cfg.river_valley_depth,
                    drop_per_segment=cfg.river_drop_per_segment,
                    sigma=cfg.river_sigma,
                )
                n_rivers += 1
        self.advance_progress(task)

        self.log_info(
            f"Terrain reconstruction: {H}×{W} grid, "
            f"{n_ridge} ridge cells, {n_rivers} river paths"
        )

        # ── Solve ─────────────────────────────────────────────────────────────
        new_hm = solver.solve()

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
        # Always re-run if upstream changed; no independent cache key yet.
        return False

    def model_names(self) -> list[str]:
        return []
