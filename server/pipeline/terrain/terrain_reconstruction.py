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
from typing import Any, Optional
from logging import Logger
from scipy.ndimage import zoom as nd_zoom, gaussian_filter, distance_transform_edt, label, binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth
from util.terrain_noise_utils import ridge_chain_jaggedness_map


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
        ridge_min_anchor_distance: float = 0.5,
        ridge_max_slope_angle_deg: float = 38.0,
        ridge_override_min_distance_m: float = 30.0,
        ridge_override_feather_m: float = 15.0,
        river_valley_depth: float = 0.5,
        river_drop_per_segment: float = 0.05,
        lake_y_range_threshold: float = 0.3,
        upsample_noise_amplitude: float = 0.02,
        upsample_noise_octaves: int = 3,
        # Cliff handling: cliff strength is a [0,1] map combining (a) measured
        # slope on directly-observed terrain (HEIGHT_MAP_OBSERVED_MASK) —
        # trustworthy ground truth, strongest close to the camera, taken as the
        # stronger of the inter-cell gradient and the intra-cell relief measured
        # before HeightMapGenerator's binning step averaged it away — and (b)
        # distant ridge-chain silhouette jaggedness, the only steepness signal
        # available beyond depth range. It relaxes the max-slope envelope locally
        # and restores measured detail the coarse solve would otherwise smooth away.
        cliff_slope_angle_low_deg: float = 50.0,
        cliff_slope_angle_high_deg: float = 75.0,
        cliff_max_slope_angle_deg: float = 82.0,
        # Cliff-top shelf: a directly-observed cliff face (measured_cliff) whose top
        # edge borders occluded terrain (the camera never saw over the edge) would
        # otherwise leave that plateau as free nodes with no local anchor, letting
        # the harmonic solve ramp it down toward whatever low ground happens to be
        # fixed nearby. Instead, project a shallow "shelf" outward from each cliff
        # blob's crest and pin any free cell the shelf would place above its current
        # value -- same "push up only" mechanism as the ridge slope envelope above,
        # just with a near-flat angle instead of a steep talus descent.
        cliff_shelf_angle_deg: float = 8.0,
        cliff_shelf_crest_percentile: float = 85.0,
        cliff_shelf_min_blob_cells: int = 20,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.solve_resolution = solve_resolution
        self.ridge_min_anchor_distance = ridge_min_anchor_distance
        self.ridge_max_slope_angle_deg = ridge_max_slope_angle_deg
        self.ridge_override_min_distance_m = ridge_override_min_distance_m
        self.ridge_override_feather_m = ridge_override_feather_m
        self.river_valley_depth = river_valley_depth
        self.river_drop_per_segment = river_drop_per_segment
        self.lake_y_range_threshold = lake_y_range_threshold
        self.upsample_noise_amplitude = upsample_noise_amplitude
        self.upsample_noise_octaves = upsample_noise_octaves
        self.cliff_slope_angle_low_deg = cliff_slope_angle_low_deg
        self.cliff_slope_angle_high_deg = cliff_slope_angle_high_deg
        self.cliff_max_slope_angle_deg = cliff_max_slope_angle_deg
        self.cliff_shelf_angle_deg = cliff_shelf_angle_deg
        self.cliff_shelf_crest_percentile = cliff_shelf_crest_percentile
        self.cliff_shelf_min_blob_cells = cliff_shelf_min_blob_cells


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

        relief_depth = context.input_depth(ContextKey.HEIGHT_MAP_CELL_RELIEF)
        cell_relief = (
            relief_depth.depth if relief_depth is not None and relief_depth.depth.shape == (H, W)
            else None
        )

        slope_depth = context.input_depth(ContextKey.HEIGHT_MAP_CELL_SLOPE)
        cell_slope_deg = (
            slope_depth.depth if slope_depth is not None and slope_depth.depth.shape == (H, W)
            else None
        )

        # True observed mask: cells with a genuine direct point-cloud measurement,
        # independent of HEIGHT_MAP_CERTAINTY's distance-based decay (see
        # HeightMapGenerator.generate docstring). This is what "was this cell
        # actually observed" means throughout this stage -- certainty answers "how
        # much do we trust it," not "do we have it."
        observed_depth = context.input_depth(ContextKey.HEIGHT_MAP_OBSERVED_MASK)
        true_observed = (
            observed_depth.depth.astype(bool) if observed_depth is not None and observed_depth.depth.shape == (H, W)
            else np.zeros((H, W), dtype=bool)
        )

        # Broader than true_observed: also True for the nadir disc's ramp band,
        # which true_observed excludes for elevation hard-pinning purposes (see
        # HeightMapGenerator.generate) even though those cells came from a genuine
        # depth sample with a genuinely correct cached panorama UV. Used below only
        # for the HEIGHT_MAP_PANO_UV_TRUST_MASK publish, never for fixed_mask/
        # restore_mask -- those must keep using true_observed so the ramp band still
        # isn't hard-pinned or spliced back in as elevation ground truth. Falls back
        # to true_observed if absent so an older cached context still works.
        real_sample_depth = context.input_depth(ContextKey.HEIGHT_MAP_REAL_SAMPLE_MASK)
        real_sample_mask = (
            real_sample_depth.depth.astype(bool)
            if real_sample_depth is not None and real_sample_depth.depth.shape == (H, W)
            else true_observed
        )

        # HEIGHT_MAP_CERTAINTY: continuous real-data trust (see
        # HeightMapGenerator._build_certainty). Used below to scale how hard the
        # "Restore observed data" step reverts to the raw measured value -- see
        # that section for why a purely binary true_observed restore is wrong.
        certainty_depth = context.input_depth(ContextKey.HEIGHT_MAP_CERTAINTY)
        certainty = (
            certainty_depth.depth.astype(np.float32)
            if certainty_depth is not None and certainty_depth.depth.shape == (H, W)
            else np.ones((H, W), dtype=np.float32)
        )

        params = context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}
        grid_size = float(params.get("grid_size_meters", 100.0))

        ridge_chains = context.input_object(ContextKey.MOUNTAIN_RIDGE_CHAINS) or []

        # ── Cliff mask ───────────────────────────────────────────────────────
        # measured_cliff: from directly-observed, high-certainty terrain — real
        # steep gradient in the height map itself is trustworthy ground truth,
        # and is the dominant signal close to the camera where depth is best.
        # distant_cliff: ridge-chain silhouette jaggedness, the only steepness
        # signal available for mountains beyond depth range (no ground samples
        # to measure a slope from out there).
        measured_cliff = TerrainReconstructionStage._measured_slope_cliff_strength(
            heightmap, true_observed, grid_size / H,
            angle_low_deg=cfg.cliff_slope_angle_low_deg,
            angle_high_deg=cfg.cliff_slope_angle_high_deg,
            cell_relief=cell_relief,
            cell_slope_deg=cell_slope_deg,
        )
        distant_cliff = ridge_chain_jaggedness_map(ridge_chains, H, W, grid_size)
        cliff_mask = np.maximum(measured_cliff, distant_cliff).astype(np.float32)
        context.add_depth(ContextKey.CLIFF_MASK, Depth(cliff_mask))
        self.log_info(
            f"Cliff mask: measured {int((measured_cliff > 0.1).sum())} px "
            f"(incl. intra-cell relief: {'yes' if cell_relief is not None else 'no'}, "
            f"plane-fit slope: {'yes' if cell_slope_deg is not None else 'no'}), "
            f"distant {int((distant_cliff > 0.1).sum())} px "
            f"(gated on direct observation, "
            f"{cfg.cliff_slope_angle_low_deg:.0f}–{cfg.cliff_slope_angle_high_deg:.0f}°)"
        )
        if self.temp is not None:
            Depth(cliff_mask).save_debug_image(self.temp / "cliff_mask.png")

        # ── Downsample to solve_resolution ────────────────────────────────────
        solve_res = min(cfg.solve_resolution, H, W)
        if solve_res < H:
            scale = solve_res / H
            hm_s = nd_zoom(heightmap, (scale, scale), order=1).astype(np.float64)
        else:
            hm_s = heightmap.astype(np.float64)
        H_s, W_s = hm_s.shape
        cell_size_m = grid_size / H_s

        self.log_info(f"Terrain reconstruction: solving at {H_s}×{W_s} (original {H}×{W})")
        self.advance_progress(task)

        # ── Initialise fixed-elevation grid and mask ──────────────────────────
        # fixed_elev: what each fixed node is held at (starts from observed HM)
        # fixed_mask: True → hold that cell at fixed_elev in the solve. Driven by
        # true_observed (a real point-cloud measurement), not certainty -- certainty
        # decays with distance and would leave most real terrain beyond ~43 m
        # unpinned, treated identically to an unobserved gap.
        fixed_elev = hm_s.copy()
        fixed_mask = TerrainReconstructionStage._downsample_observed_max(true_observed, H_s, W_s)
        # Snapshot of "was this cell trusted as observed ground truth" before ridge
        # anchors/overrides start mutating fixed_mask below -- needed so the ridge
        # override step (see "Ridge override of observed data") can tell an
        # originally-observed cell apart from one a ridge anchor pins directly.
        observed_fixed_mask_s = fixed_mask.copy()

        x_half = z_half = grid_size / 2.0

        def world_to_grid(xyz: np.ndarray):
            """Project world XYZ array (N,3) → (rows, cols) on the solve grid."""
            col = np.clip((xyz[:, 0] + x_half) / grid_size * (W_s - 1), 0, W_s - 1)
            row = np.clip((xyz[:, 2] + z_half) / grid_size * (H_s - 1), 0, H_s - 1)
            return row.round().astype(np.int32), col.round().astype(np.int32)

        # Per-cell horizontal distance from the camera (grid origin), used to keep
        # the ridge override below from reaching into near-camera real geometry --
        # see "Ridge override of observed data".
        _grid_x = np.linspace(-x_half, x_half, W_s)
        _grid_z = np.linspace(-z_half, z_half, H_s)
        camera_dist_m = np.hypot(_grid_x[None, :], _grid_z[:, None])

        # ── Ridge anchors: distant mountain chains ────────────────────────────
        # Foreground chains (close to camera) are already captured by the high-
        # confidence observed terrain; anchoring them would conflict with the
        # rising slope toward the mountains.
        cliff_mask_s = (
            nd_zoom(cliff_mask, (H_s / H, W_s / W), order=1) if (H_s, W_s) != (H, W)
            else cliff_mask
        ).astype(np.float64)
        # Measured-only (no distant ridge-silhouette jaggedness mixed in), since the
        # shelf mechanism below needs blobs with real elevation data behind them --
        # distant_cliff has no ground truth to anchor a crest to.
        measured_cliff_s = (
            nd_zoom(measured_cliff, (H_s / H, W_s / W), order=1) if (H_s, W_s) != (H, W)
            else measured_cliff
        ).astype(np.float64)
        min_anchor_m = cfg.ridge_min_anchor_distance * z_half
        n_anchored, n_skipped = 0, 0

        n_points_dropped = 0
        for raw_chain in ridge_chains:
            chain = np.asarray(raw_chain, dtype=np.float32)
            if len(chain) < 2:
                continue
            horiz = np.sqrt(chain[:, 0] ** 2 + chain[:, 2] ** 2)
            if float(np.median(horiz)) < min_anchor_m:
                n_skipped += 1
                continue
            # A chain can clear the median-distance qualification above while
            # still dipping past individual points close to the camera (noisy
            # silhouette tracing) -- anchoring those as hard Dirichlet BCs pins
            # near-camera terrain to a far-away chain's elevation, producing the
            # same near-camera spike failure mode ridge_override_min_distance_m
            # guards against below. Drop just the close points; the rest of the
            # chain still anchors normally.
            near_camera = horiz < cfg.ridge_override_min_distance_m
            if near_camera.any():
                n_points_dropped += int(near_camera.sum())
                chain = chain[~near_camera]
                if len(chain) == 0:
                    n_skipped += 1
                    continue
            rows_c, cols_c = world_to_grid(chain)
            fixed_mask[rows_c, cols_c] = True
            fixed_elev[rows_c, cols_c] = chain[:, 1].astype(np.float64)  # Y = elevation
            n_anchored += 1

        self.log_info(
            f"Ridge chains: {n_anchored} anchored (≥{min_anchor_m:.0f} m), "
            f"{n_skipped} foreground skipped, {n_points_dropped} near-camera "
            f"point(s) dropped (<{cfg.ridge_override_min_distance_m:.0f} m)"
        )

        # ── Critical-slope envelope ───────────────────────────────────────────
        # The harmonic solve with only crest nodes produces a smooth ramp whose
        # shape depends on whatever other fixed nodes surround it. Pinning slope
        # nodes under a physical maximum-slope constraint gives the solver the
        # correct gradient: we know the crest elevation exactly, and the steepest
        # a natural mountain can sustain is roughly the angle of repose for the
        # dominant material (~35° loose talus, ~38–45° consolidated rock).
        # From each crest pixel, elevation must drop by at least tan(θ) metres per
        # horizontal metre — this is the tightest envelope consistent with the
        # observed crest heights without assuming any particular cross-sectional
        # shape (Gaussian, parabola, etc.).
        # profile_best is shared by the ridge envelope below and the cliff-shelf
        # envelope that follows it -- both only ever raise a free cell's floor, so
        # accumulating into one array before applying means whichever mechanism
        # implies the higher floor for a given cell wins, regardless of which one
        # runs first. Applying each separately would let the first mechanism to
        # touch a cell "claim" it via ~fixed_mask, silently discarding a higher
        # floor the other mechanism would have proposed for the same cell.
        profile_best = np.full((H_s, W_s), -np.inf)

        if cfg.ridge_max_slope_angle_deg > 0.0 and n_anchored > 0:
            # Spatially-varying envelope angle: talus angle everywhere, ramped up
            # to cliff_max_slope_angle_deg wherever the cliff mask says this patch
            # of ridgeline is a steep face rather than a scree slope.
            envelope_angle = cfg.ridge_max_slope_angle_deg + (
                cfg.cliff_max_slope_angle_deg - cfg.ridge_max_slope_angle_deg
            ) * cliff_mask_s
            max_slope_tan_map = np.tan(np.radians(envelope_angle))

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
                dist_m = dist_px * cell_size_m
                # Linear cone: elevation drops at max_slope_tan m/m from the crest.
                # The tangent is taken at each destination cell (not the crest), so
                # steep-marked terrain below/beside the crest still gets the relaxed
                # cap even though the crest pixel itself may be tame.
                profile = nearest_elev - dist_m * max_slope_tan_map
                profile_best = np.maximum(profile_best, profile)

        # ── Ridge override of observed data ───────────────────────────────────
        # Distant mountain ridge chains come from the sky-terrain silhouette, a
        # completely different (and, at range, more reliable) signal than the
        # monocular/point-cloud depth that produced the height map -- depth
        # estimation is at its weakest exactly out where ridge_min_anchor_distance
        # requires a chain to be before it's used at all. true_observed's binary
        # "was there a real sample here" carries no distance/certainty decay, so a
        # noisy far-range depth reading can otherwise sit as an unmovable Dirichlet
        # BC even when the ridge envelope says this location should be much higher
        # (e.g. a false valley cut into what the silhouette confirms is a
        # mountainside). Captured as its own snapshot (pre-shelf) so only the ridge
        # envelope -- not the cliff-shelf mechanism below, which targets
        # near-camera measured cliffs and should keep protecting observed data --
        # gets this authority to override real depth samples.
        #
        # Gated on ridge_override_min_distance_m: the linear talus cone from a
        # tall/close ridge anchor is still large at short range (e.g. a crest
        # anchored right at the lowered ridge_min_anchor_distance and multiple
        # competing ridge directions around a 360° panorama can together push the
        # envelope well above real ground for tens of metres around the camera),
        # so without a floor here the override reaches in and overwrites good
        # near-camera measured terrain -- surfacing as unrealistic elevation
        # spikes and mismatched crossing seams close to the user. The override is
        # only meant to correct far terrain depth estimation is weak on, so cells
        # nearer than this stay on real observed data no matter what the envelope
        # implies.
        #
        # ridge_override_feather_m ramps the override in linearly over the band
        # [ridge_override_min_distance_m, +feather] instead of switching it on at
        # full strength right at the distance floor -- a hard on/off still leaves
        # a visible step where real ground jumps straight to the envelope's
        # value at that one radius. Cells at/beyond min_distance + feather get
        # the full override, same as before; cells inside min_distance never do.
        ridge_profile_only = profile_best.copy()
        observed_override = (
            observed_fixed_mask_s
            & (ridge_profile_only > fixed_elev + 0.1)
            & np.isfinite(ridge_profile_only)
            & (camera_dist_m >= cfg.ridge_override_min_distance_m)
        )
        if observed_override.any():
            feather_m = max(cfg.ridge_override_feather_m, 1e-6)
            blend_weight = np.clip(
                (camera_dist_m - cfg.ridge_override_min_distance_m) / feather_m, 0.0, 1.0
            )
            w = blend_weight[observed_override]
            fixed_elev[observed_override] = (
                fixed_elev[observed_override] * (1.0 - w) + ridge_profile_only[observed_override] * w
            )
        self.log_info(
            f"Ridge override: {int(observed_override.sum())} observed node(s) "
            f"blended toward the mountain ridge envelope over "
            f"{cfg.ridge_override_min_distance_m:.0f}–"
            f"{cfg.ridge_override_min_distance_m + cfg.ridge_override_feather_m:.0f} m "
            f"(trusted over raw depth beyond that)"
        )

        # ── Cliff-top shelf ────────────────────────────────────────────────────
        # A directly-observed cliff face (measured_cliff) whose top edge borders
        # occluded terrain -- the camera saw the face but never the plateau beyond
        # its top -- leaves that plateau as free nodes with no local anchor, so the
        # harmonic solve ramps it down toward whatever low ground happens to be
        # fixed nearby (usually the base of the cliff). Anchor a shallow shelf
        # outward from each cliff blob's crest so those cells settle near the
        # cliff-top elevation instead.
        if cfg.cliff_shelf_angle_deg > 0.0 and measured_cliff.any():
            cliff_blobs, n_cliff_blobs = label(
                measured_cliff_s > 0.5, structure=np.ones((3, 3))
            )
            # Fixed (observed) cells that border free (occluded) territory -- the
            # true observed/occluded boundary, cheaper and more robust than
            # computing a per-cell gradient direction.
            free_adjacent = binary_dilation(~fixed_mask, structure=np.ones((3, 3))) & fixed_mask

            shelf_crest_mask = np.zeros((H_s, W_s), dtype=bool)
            shelf_crest_elev_map = np.zeros((H_s, W_s), dtype=np.float64)
            n_shelf_blobs = 0

            for blob_id in range(1, n_cliff_blobs + 1):
                blob_mask = cliff_blobs == blob_id
                if int(blob_mask.sum()) < cfg.cliff_shelf_min_blob_cells:
                    continue
                elev_cutoff = np.percentile(fixed_elev[blob_mask], cfg.cliff_shelf_crest_percentile)
                candidate = blob_mask & (fixed_elev >= elev_cutoff) & free_adjacent
                if not candidate.any():
                    continue
                # Collapse to one median elevation per blob rather than trusting
                # individual pixels: grazing-incidence depth noise is worst right
                # at a cliff's top edge, exactly the cells being selected here.
                shelf_crest_mask[candidate] = True
                shelf_crest_elev_map[candidate] = np.median(fixed_elev[candidate])
                n_shelf_blobs += 1

            if shelf_crest_mask.any():
                dist_px, src = distance_transform_edt(~shelf_crest_mask, return_indices=True)
                nearest_elev = shelf_crest_elev_map[src[0], src[1]]
                dist_m = dist_px * cell_size_m
                shelf_profile = nearest_elev - dist_m * np.tan(np.radians(cfg.cliff_shelf_angle_deg))
                profile_best = np.maximum(profile_best, shelf_profile)
                self.log_info(
                    f"Cliff shelf: {n_shelf_blobs} crest blob(s) "
                    f"(angle={cfg.cliff_shelf_angle_deg:.0f}°, "
                    f"crest ≥ p{cfg.cliff_shelf_crest_percentile:.0f} of blob)"
                )
                if self.temp is not None:
                    Depth(shelf_crest_mask.astype(np.float32)).save_debug_image(
                        self.temp / "cliff_shelf_crest.png"
                    )
                    Depth(
                        np.where(np.isfinite(shelf_profile), shelf_profile, 0.0).astype(np.float32)
                    ).save_debug_image(self.temp / "cliff_shelf_profile.png")

        # Pin free nodes that either envelope places notably above observed terrain.
        # No absolute elevation threshold: each envelope is relative to its own crest.
        apply = (
            ~fixed_mask
            & (profile_best > fixed_elev + 0.1)
            & np.isfinite(profile_best)
        )
        fixed_mask[apply] = True
        fixed_elev[apply] = profile_best[apply]
        self.log_info(
            f"Slope/shelf envelope: {int(apply.sum())} free nodes pinned "
            f"(talus={cfg.ridge_max_slope_angle_deg:.0f}°, "
            f"cliff={cfg.cliff_max_slope_angle_deg:.0f}° where cliff_mask > 0, "
            f"shelf={cfg.cliff_shelf_angle_deg:.0f}°)"
        )

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

        # Native-resolution footprint of cells the ridge envelope overrode, so the
        # "Restore observed data" step below can exclude them -- otherwise it would
        # simply splice the raw (overridden) depth value straight back in, undoing
        # the override above unconditionally.
        observed_override_native = np.zeros((H, W), dtype=bool)
        if observed_override.any():
            observed_override_native = (
                nd_zoom(observed_override.astype(np.float64), (H / H_s, W / W_s), order=0) > 0.5
                if (H_s, W_s) != (H, W) else observed_override
            )

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
        # Runs BEFORE the observed-data restore below, not after -- this noise is
        # meant to texture the solver's smooth interpolation, not real point-cloud
        # geometry. Applying it first means the restore step (which splices real
        # elevation back in) also undoes any noise that landed on genuinely
        # observed cells, instead of permanently baking a couple of centimetres of
        # random offset onto real data with nothing downstream to revert it.
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

        # ── Restore observed data ───────────────────────────────────────────────
        # The solve_resolution downsample + bicubic upsample above necessarily loses
        # native-resolution precision, regardless of how confidently a cell was
        # observed. Splice the original native-resolution height back in wherever
        # we have a genuine direct point-cloud measurement -- that's real geometry,
        # not something the harmonic solve or the fine-grain noise above should be
        # allowed to touch. This subsumes the narrower cliff-only restoration this
        # replaced: measured_cliff is now always a subset of true_observed (same
        # observed gate, plus a steepness requirement). Feathering only softens the
        # transition right at the observed/gap boundary -- cells deep inside real
        # regions stay at ~full restore weight.
        # Cells the ridge envelope overrode above are deliberately excluded here --
        # restoring them would silently splice the raw (untrusted) depth value
        # back in and undo that override.
        #
        # Scaled by HEIGHT_MAP_CERTAINTY, not just the true_observed boolean: near
        # the nadir, equirectangular unprojection (Y = depth * sin(phi)) amplifies
        # ordinary depth-model noise almost directly into height noise, exactly
        # where certainty is already lowest (the nadir ramp). A binary restore
        # spliced that raw noisy value back in at full strength regardless,
        # undoing the harmonic solve's smoothing precisely where the real data is
        # least trustworthy -- visible as a jagged, un-smoothable halo of "real"
        # noise a few metres out from the nadir disc, sitting on top of terrain
        # that should read as smooth ground. A high-certainty true_observed cell
        # (most real data beyond the ramp) still restores at ~full weight, same
        # as before; only the low-certainty/near-nadir band is now actually
        # allowed to keep some of the solve's smoothing instead of being
        # unconditionally overwritten by its own noise.
        restore_mask = true_observed & ~observed_override_native
        if restore_mask.any():
            restore_weight = gaussian_filter(
                restore_mask.astype(np.float32) * certainty, sigma=1.0
            ).astype(np.float32)
            new_hm = (
                new_hm * (1.0 - restore_weight) + heightmap.astype(np.float32) * restore_weight
            ).astype(np.float32)
            self.log_info(
                f"Observed-data restoration: {int(restore_mask.sum())} px eligible "
                f"(mean certainty {float(certainty[restore_mask].mean()):.2f}), reverted "
                f"toward measured elevation "
                f"({int(observed_override_native.sum())} px excluded as ridge-overridden)"
            )

        context.add_depth(ContextKey.HEIGHT_MAP, Depth(new_hm))

        # Publish a ridge-override-aware trust mask, DISTINCT from
        # HEIGHT_MAP_OBSERVED_MASK, for TerrainMeshGenerator.generate()'s
        # per-vertex panorama-UV selection specifically. Deliberately not
        # overwriting HEIGHT_MAP_OBSERVED_MASK itself: TerrainNoiseRefinementStage's
        # final hard-restore and terrain_texture_generation.py's panorama-
        # visibility weighting both still want ridge-overridden cells treated as
        # trustworthy real data (the ridge envelope is a legitimate, silhouette-
        # derived correction, not a fabricated gap-fill) -- narrowing that shared
        # mask would wrongly let noise/diffusion/erosion reshape corrected peaks
        # and would wrongly cede those texels to synthetic FLUX texture.
        # TerrainMeshGenerator's cached-UV decision is the one place a
        # ridge-overridden cell must NOT be trusted: its UV was cached by
        # HeightMapGenerator._panorama_uv_from_height from the ORIGINAL
        # (pre-override, often badly wrong at range) height, while its geometry
        # has since moved to match the ridge envelope -- producing a
        # geometry/texture mismatch (stretching) confined to exactly the
        # distant-mountain cells the ridge override was introduced to correct.
        # Built from real_sample_mask, not restore_mask/true_observed: the nadir
        # disc's ramp band is deliberately excluded from true_observed (so it isn't
        # hard-pinned/restored as elevation ground truth), but its cached panorama UV
        # came from that same genuine sample and is still correct -- starving it of
        # that cached UV instead sends TerrainMeshGenerator to re-derive UV from the
        # ramp band's synthetic diffused height, which is exactly where UV-from-
        # height error is worst (near nadir), and was the actual source of "sky
        # visible on near-camera ground" after the ramp band was folded into the
        # flat-ground prior.
        pano_uv_trust_mask = real_sample_mask & ~observed_override_native
        context.add_depth(ContextKey.HEIGHT_MAP_PANO_UV_TRUST_MASK, Depth(pano_uv_trust_mask.astype(np.float32)))

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
    def _downsample_observed_max(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
        """
        Block-max downsample of a boolean observed mask: an output cell is True if
        ANY native cell within its block was truly observed. A bilinear/mean
        downsample (fine for the smooth, continuous certainty field) would dilute
        or entirely lose small clusters of real data before they reach the solve
        grid -- a single directly-observed native cell is enough to anchor a fixed
        solve node, so max pooling is the only downsample that can't silently
        discard real coverage.
        """
        in_h, in_w = mask.shape
        if in_h % out_h == 0 and in_w % out_w == 0:
            fh, fw = in_h // out_h, in_w // out_w
            return mask[: out_h * fh, : out_w * fw].reshape(out_h, fh, out_w, fw).any(axis=(1, 3))
        # Fallback for non-integer ratios: any partial coverage in a bilinear
        # resample of the boolean-as-float mask counts.
        zoomed = nd_zoom(mask.astype(np.float64), (out_h / in_h, out_w / in_w), order=1)
        return zoomed > 1e-9

    @staticmethod
    def _measured_slope_cliff_strength(
        heightmap: np.ndarray,
        observed_mask: np.ndarray,
        cell_size_m: float,
        angle_low_deg: float = 50.0,
        angle_high_deg: float = 75.0,
        cell_relief: Optional[np.ndarray] = None,
        cell_slope_deg: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Depth-measured cliff strength: (H, W) float32 in [0, 1].

        Wherever the observed height map itself carries a genuinely steep local
        gradient AND that cell was directly observed (a real point-cloud
        measurement, not solver/interpolation fill), treat the steepness as ground
        truth — no semantic label needed, and strongest exactly where depth is most
        reliable: terrain close to the camera.

        cell_relief (optional): HEIGHT_MAP_CELL_RELIEF, the raw Y range spanned by
        every raw depth sample that landed in each cell before HeightMapGenerator
        collapsed them to their mean. A cliff face narrower than one grid cell
        leaves no trace in the inter-cell gradient above — every neighbouring
        cell's mean already absorbed it — so it's converted to its own effective
        slope angle (arctan(relief / cell_size)) and combined via max() with the
        inter-cell gradient. This is a second, independent measurement of the same
        underlying thing (real steepness in the raw samples), not a fallback, so
        the stronger of the two always wins rather than one overriding the other.

        cell_slope_deg (optional): HEIGHT_MAP_CELL_SLOPE, a per-cell surface tilt
        already expressed in degrees from a local least-squares plane fit through
        the raw forward-projected points in that cell (see
        HeightMapGenerator._forward_project_cell_stats). Unlike cell_relief (a
        scalar Y range, only a proxy for slope) this is a direct 3-D orientation
        measurement, so it is combined via max() alongside the other two rather
        than converted through another arctan.

        angle_low_deg/angle_high_deg define the ramp from "not a cliff" to
        "fully a cliff"; observed_mask gates out cells whose elevation came from
        interpolation rather than a real depth sample (a sharp edge in inpainted
        noise is not evidence of a cliff).
        """
        gy, gx = np.gradient(heightmap.astype(np.float64), cell_size_m)
        slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))

        if cell_relief is not None:
            relief_slope_deg = np.degrees(np.arctan(cell_relief.astype(np.float64) / cell_size_m))
            slope_deg = np.maximum(slope_deg, relief_slope_deg)

        if cell_slope_deg is not None:
            slope_deg = np.maximum(slope_deg, cell_slope_deg.astype(np.float64))

        strength = np.clip(
            (slope_deg - angle_low_deg) / max(angle_high_deg - angle_low_deg, 1e-6),
            0.0, 1.0,
        )
        # Soften the observed gate itself so restoration doesn't leave a hard
        # seam exactly at the observed/interpolated boundary.
        observed = gaussian_filter(observed_mask.astype(np.float64), sigma=1.0)
        return (strength * observed).astype(np.float32)

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
