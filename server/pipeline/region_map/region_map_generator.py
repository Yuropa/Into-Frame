import cv2
import numpy as np
from scipy.ndimage import zoom
from typing import Optional

from util.depth_utils import Depth
from util.panorama_utils import Panorama
from util.projection_utils import (
    ground_projection_certainty,
    equirectangular_pixels_to_world,
    inverse_map_panorama_to_grid,
    nearest_sample_grid,
    project_panorama_to_ground_grid,
)
from util.grid_utils import confidence_flood_fill
from pipeline.panorama_segmentation.panorama_region_result import RegionType


class RegionMapGenerator:
    @staticmethod
    def generate(
        panorama_depth: Depth,
        type_idx_map: np.ndarray,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 4096,
        ground_y_max: float = -0.5,
        camera_height_meters: float = 1.0,
        sky_mask: Optional[np.ndarray] = None,
        nadir_exclusion_radius: float = 3.0,
        nadir_ramp_width: float = 5.0,
        certainty_falloff_meters: float = 20.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project per-pixel region type labels from an equirectangular panorama onto a
        top-down grid via inverse mapping.

        For each grid cell, the panorama pixel that corresponds to a flat ground point
        at that cell's (X, Z) centre is computed analytically and the region type is
        sampled there directly — one lookup per cell, no point-cloud scattering.
        Empty cells (sky, invalid depth) are filled by confidence-weighted BFS: cells
        close to the camera (high geometric certainty) propagate their type first.

        Returns (region_map, certainty_map):
          region_map    — (grid_resolution, grid_resolution) uint8 of RegionType indices.
          certainty_map — (grid_resolution, grid_resolution) float32 in [0, 1]:
                          distance-decayed certainty (see ground_projection_certainty,
                          falloff_m=certainty_falloff_meters) for directly observed
                          cells, 0 for filled.

        Grid layout matches the height map: rows = Z near→far, cols = X left→right.
        """
        sky_idx = RegionType.SKY
        other_idx = RegionType.OTHER

        d = panorama_depth.depth.astype(np.float32)
        # NaN sky pixels so bilinear sampling near the sky boundary doesn't blend
        # sky depth values into ground cells.
        d_masked = d.copy()
        if sky_mask is not None and sky_mask.shape == d.shape:
            d_masked[sky_mask] = np.nan

        sampled_depth, pano_u, pano_v, X_grid, Z_grid = inverse_map_panorama_to_grid(
            Depth(d_masked), grid_size_meters, grid_resolution, camera_height_meters
        )

        # Sample the discrete region type at each cell's panorama pixel.
        type_sampled = nearest_sample_grid(type_idx_map, pano_u, pano_v).astype(np.uint8)

        r_grid = np.sqrt(
            X_grid.astype(np.float64) ** 2 + Z_grid.astype(np.float64) ** 2
        ).astype(np.float32)
        phi_grid = -np.arctan2(camera_height_meters, np.maximum(r_grid, 1e-6)).astype(np.float32)
        Y_grid = (sampled_depth * np.sin(phi_grid)).astype(np.float32)

        ground_y_min = -(camera_height_meters + 5.0)
        has_data = (
            np.isfinite(sampled_depth) & (sampled_depth > 0)
            & (Y_grid <= ground_y_max) & (Y_grid >= ground_y_min)
            & (type_sampled != sky_idx)
        )
        if sky_mask is not None and sky_mask.shape == d.shape:
            has_data &= ~nearest_sample_grid(sky_mask.astype(np.uint8), pano_u, pano_v).astype(bool)

        # Exclude cells too close to the nadir: the panorama bottom is heavily distorted
        # and segmentation models trained on rectilinear images are unreliable there.
        # Without exclusion the flood fill seeds from the highest-certainty (near-center)
        # cells first, propagating wrong types radially outward in a star pattern.
        if nadir_exclusion_radius > 0:
            has_data &= r_grid >= nadir_exclusion_radius

        if not np.any(has_data):
            zero_certainty = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
            return np.full((grid_resolution, grid_resolution), other_idx, dtype=np.uint8), zero_certainty

        region_map = np.full((grid_resolution, grid_resolution), other_idx, dtype=np.uint8)
        region_map[has_data] = type_sampled[has_data]

        # Nadir ramp: smoothly reduce certainty from 0 at the exclusion boundary to full
        # geometric certainty at exclusion_radius + ramp_width.  This ensures the flood
        # fill propagates inward from the reliable ring rather than outward from the
        # edge of the exclusion zone, which is still somewhat distorted.
        nadir_ramp = np.clip(
            (r_grid - nadir_exclusion_radius) / max(float(nadir_ramp_width), 1e-6),
            0.0, 1.0,
        ).astype(np.float32) ** 2
        certainty = np.where(
            has_data,
            ground_projection_certainty(X_grid, Z_grid, certainty_falloff_meters) * nadir_ramp,
            0.0,
        ).astype(np.float32)

        return confidence_flood_fill(region_map, has_data, certainty), certainty

    @staticmethod
    def extract_mountain_ridgeline(
        type_idx_map: np.ndarray,
        sky_idx: int,
        water_idx: int = -1,
        terrain_idx: int = -1,
        min_terrain_silhouette_frac: float = 0.25,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 4096,
        ridge_radius_frac: float = 0.85,
        dilation_iters: int = 3,
        connect_radius_px: int = 30,
        chain_smooth_window: int = 15,
        max_col_gap_frac: float = 0.01,
        max_hole_cols: int = 12,
        hole_stitch_max_col_gap: int = 12,
        hole_stitch_max_dist_px: int = 90,
        panorama_depth: Optional[Depth] = None,
        near_field_support_row_offset: int = 6,
        near_field_trust_distance_m: float = 45.0,
        near_field_min_radius_m: float = 5.0,
        prominence_min_m: float = 0.0,
        prominence_shoulder_m: float = 25.0,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Extract the sky-foreground horizon per column and place it on a top-down
        grid at a fixed radius from the camera, using only the boundary's real
        angle (elevation from its row, azimuth from its column) -- not depth.

        This is a deliberate scale compromise, not a depth-accuracy shortcut: a
        real mountain kilometres out cannot be walkable local terrain in a grid
        only grid_size_meters across, so it's intentionally brought in close and
        shrunk, the same way a scale model would be. What has to survive that
        compression is the mountain's silhouette shape; its literal real-world
        distance is meaningless once compressed this far, so nothing is lost by
        not using it. Depth is doubly unsuited to supplying it anyway: DAP's raw
        prediction saturates at the same ceiling for both real distant terrain
        and sky (see PanoramaDepthCalibrationStage), so calibrated depth right
        at the ridge crest is frequently a low-confidence log-space extrapolation,
        not a measurement -- using it to place the ridge would let that noise
        warp the one thing (shape) this method actually needs to preserve.
        ridge_radius_frac (fraction of grid_size_meters / 2) is a deliberately
        chosen display distance, comfortably inside TerrainReconstructionStage's
        own ridge_override_min_distance_m + ridge_override_feather_m band so the
        solve has room to blend real near-camera terrain up to it.

        Any non-sky, non-water pixel that sits directly below a sky pixel is treated
        as part of the ridgeline — this covers bare TERRAIN, forest-covered mountains
        (VEGETATION), and built structures on hilltops (BUILT), which would all be
        missed if only the TERRAIN type were accepted. WATER pixels touching sky (an
        open ocean/lake horizon) are excluded so they don't get anchored into the
        solver as a fake mountain crest — that boundary belongs to extract_water_chains
        instead. Excluded columns simply contribute no ridge point.

        The greedy nearest-neighbour chaining below also enforces max_col_gap_frac:
        candidates are restricted to points whose source panorama column is within
        max_col_gap_frac * w of the current chain point (circular, wrapping at the
        360° seam), in addition to the existing connect_radius_px world-space
        distance check. Without this, a narrow water body (e.g. a lake or river
        mouth) that excludes a run of columns would still often get bridged — the
        mountain points on either side of a narrow lake are frequently close
        together in actual 3D space, well within connect_radius_px, so the chain
        would silently jump straight across the excluded gap and draw an
        interpolated ridge segment through the water instead of leaving a break.

        Any column containing so much as one water pixel (water_present_column) is
        permanently off-limits to the two gap-closing mechanisms below
        (max_hole_cols, hole_stitch_*) — checked directly from the panorama's own
        water mask, not inferred from anything upstream, so a real water break can
        never be closed regardless of how small it happens to look.

        For each panorama column, finds the first non-sky row below sky — that
        row/column pair is the boundary's real elevation/azimuth angle, placed
        at ridge_radius_frac * grid_size_meters / 2 from the camera along that
        angle. Columns with no non-sky pixel at all are usually a short
        classification glitch, not a real absence — real terrain is obviously
        present just past it on both sides — so they get a synthetic boundary
        row circularly interpolated from their nearest real neighbours, but
        only for runs up to max_hole_cols columns wide; wider runs are left as
        a genuine gap rather than fabricating a long stretch of invented ridge.

        Projected points are chained via greedy nearest-neighbour search within
        connect_radius_px grid cells, then smoothed with a moving average of width
        chain_smooth_window, and finally rasterised with filled line segments so the
        result is a continuous ridge rather than isolated scattered pixels.
        Disconnected ridge segments produce separate chains — except where two
        chains' nearest endpoints are within hole_stitch_max_col_gap columns and
        hole_stitch_max_dist_px grid cells of each other with no water between
        them, in which case they're stitched into one chain. This closes holes
        created by chaining noise (a depth glitch nudging one column's point just
        outside connect_radius_px) without loosening connect_radius_px/
        max_col_gap_frac themselves, which is what actually protects water breaks.

        panorama_depth (optional): calibrated depth at the same resolution as
        type_idx_map. Placing every column's Y on the same shell radius means a
        real nearby hillside and a genuinely distant peak that happen to reach
        the same elevation angle phi end up at the same synthetic height too —
        a "twin peak" in whatever direction the nearer feature sits. Sampled a
        few rows below the crest (near_field_support_row_offset — the crest
        pixel itself is the unreliable one this method's docstring already
        warns about, but real terrain a little further into the slope is not),
        this depth corroborates *how far* that column's feature actually is:
        readings within near_field_trust_distance_m replace ridge_radius_m with
        the real (smaller) distance for that column's height calculation only —
        XZ placement stays on the shell, unchanged, so the "scale model"
        compression this method relies on for layout is untouched. Columns with
        no usable reading (no panorama_depth, invalid/saturated depth, or a
        genuine reading beyond near_field_trust_distance_m) keep the full shell
        height, same as before this parameter existed.

        prominence_min_m / prominence_shoulder_m: keep only chain points within
        prominence_shoulder_m (arc length along the chain) of a local maximum
        whose topographic prominence -- the standard "key col" definition, a
        point's height above the higher of the two saddles you'd cross in each
        direction before reaching a taller point -- is at least
        prominence_min_m. Everything else is dropped from the returned chains
        entirely. Chains are tiled x3 internally so a peak straddling the
        chain's arbitrary start/end index (very often the case for a near-closed
        ring; the greedy walk's start has no topographic meaning) still sees its
        true neighbours. Only prunes the *returned* chains used for anchoring;
        the rasterised silhouette `grid` is built from the full, unpruned chains.

        DEFAULTS TO 0 (disabled). This existed to stop a near-complete ring of
        Dirichlet anchors from dominating TerrainReconstructionStage's harmonic
        solve -- on one capture, unobserved terrain within 4.5 m of the camera
        reconstructed to ~7x its own raw height. That symptom was real, but the
        ring was not its cause: the near-field flooding came from the critical-
        slope envelope projecting unbounded talus cones inward from every anchor
        (a 54 m crest at 70 m has a 69 m run-out, so its cone still carried
        15-20 m of elevation back to the camera and got hard-pinned there), and
        from ridge_chain_jaggedness_map's missing distance falloff holding the
        envelope angle at ~69 deg across half the grid. Both are fixed at
        source -- see envelope_max_reach_m in TerrainReconstructionStage and the
        falloff in ridge_chain_jaggedness_map -- so the ring no longer has a
        mechanism to reach the camera.

        Meanwhile the cost of this filter is severe, and prominence is
        structurally the wrong instrument for the job. A photographed horizon is
        a smooth, closed curve; smooth curves have almost no prominence
        anywhere, so the filter does not select "real summits," it selects the
        two or three points where the curve happens to be locally peaked and
        discards the rest of the horizon. Measured on a Mount Rainier panorama:
        every one of the 4096 columns carries a real sky/terrain boundary
        between 16 and 54 m of crest elevation -- a continuous mountainous
        horizon -- and at the old 20 m threshold only 19% of azimuth survived
        into the anchored chains, leaving a mountain in front, one behind, and
        flat ground through 290 degrees of the scene.

        That flatness is also, indirectly, a texturing bug. With no anchors, the
        terrain out to the grid edge stays a level plane, which the panorama
        then paints at grazing incidence: at 80 m each panorama row covers 12.4
        ground-metres, versus 0.51 with the horizon anchored at its real
        elevation. The radial smearing in the baked ground texture is that ratio
        made visible.

        Left as a tunable (not deleted) because the underlying idea -- an
        ordinary silhouette is a weaker elevation estimate than a real summit --
        is sound, and a capture with a genuinely flat horizon may want it back.

        Returns (grid, chains) where:
          grid   — float32 (grid_resolution, grid_resolution) binary mask (unchanged).
          chains — list of (M, 3) float32 arrays of world (X, Y, Z) per ridge chain,
                   where Y is the camera-relative elevation of the ridge crest
                   (Y = radius_for_height * sin(phi), positive = above camera;
                   radius_for_height is ridge_radius_m unless near-field
                   corroboration above scales it down for that column).
        """
        h, w = type_idx_map.shape
        sky_mask = type_idx_map == sky_idx
        water_mask = type_idx_map == water_idx
        # Any water pixel anywhere in a column, not just at the boundary itself —
        # the simplest, strictest signal for "never let a gap-closing mechanism
        # touch this column," independent of how/why its boundary is missing.
        water_present_column = water_mask.any(axis=0)                # (W,) bool

        above_is_sky = np.zeros((h, w), dtype=bool)
        above_is_sky[1:, :] = sky_mask[:-1, :]
        # Any non-sky, non-water pixel directly below sky is a ridgeline candidate.
        silhouette = (~sky_mask) & (~water_mask) & above_is_sky

        has_silhouette = silhouette.any(axis=0)                      # (W,) bool
        boundary_row = silhouette.argmax(axis=0).astype(np.float64)  # (W,) float

        cols = np.arange(w)

        # ── Is this horizon a LANDFORM at all? ────────────────────────────────
        # Everything below treats the sky boundary as the crest of ground and
        # anchors terrain elevation to it. That is right for a mountain, a cliff or
        # a forested ridge -- and catastrophic for a flat city, where the same
        # boundary is the roofline of a building forty metres away.
        #
        # The test is what the silhouette is MADE OF, not what the panorama
        # contains. Measured on the sample captures, by fraction of silhouette
        # columns whose own boundary pixel types TERRAIN:
        #
        #     Paris        0%   (71% BUILT, 29% VEGETATION) — a city skyline
        #     Iceland     87%
        #     Rainier     98%   (2% VEGETATION — a forested shoulder)
        #     Shark Fin  100%
        #
        # Paris is not near the others; it is at zero, and no column of its horizon
        # is a landform. Left ungated it reconstructed a flat river scene -- height
        # map measured -6.0 to -0.2 m -- into 71 m of relief: a 5,966 m2 hill where
        # the right-bank treeline stands and a 2,674 m2 one on the cathedral.
        #
        # Deliberately whole-scene rather than per-column. A per-column rule would
        # also drop Iceland's 5% BUILT columns, which is tempting and probably
        # right, but it would put the "built structures on hilltops" case this
        # method's own comment calls out (a hut on a summit, a ridge with a mast)
        # at risk for no measured benefit. 0.25 sits far from every capture on
        # either side; set 0 to disable the gate.
        #
        # KNOWN LIMIT: a capture that is half city and half mountain scores ~50% and
        # keeps anchoring, so its city half still becomes hills. Fixing that needs
        # the per-column rule above, and a capture that actually exhibits it.
        if terrain_idx >= 0 and min_terrain_silhouette_frac > 0 and has_silhouette.any():
            boundary_types = type_idx_map[
                boundary_row.astype(np.intp)[has_silhouette], cols[has_silhouette]
            ]
            terrain_frac = float((boundary_types == terrain_idx).mean())
            if terrain_frac < min_terrain_silhouette_frac:
                # No ridgeline, rather than a fabricated one. Every consumer of
                # both return values already handles empty (each reads them with
                # `or []`), and the resulting terrain is a level plane -- which is
                # what a flat capture should have been all along.
                return (
                    np.zeros((grid_resolution, grid_resolution), dtype=np.float32),
                    [],
                )

        def _circular_fill(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
            """Replace invalid entries of a column-indexed 1D signal with linear
            interpolation from the nearest valid neighbours, wrapping at the
            panorama's 0/w seam so a gap straddling due-north still interpolates
            from its true neighbours instead of extrapolating off one edge."""
            if valid.all() or not valid.any():
                return values
            valid_cols = cols[valid].astype(np.float64)
            valid_vals = values[valid].astype(np.float64)
            ext_cols = np.concatenate([valid_cols - w, valid_cols, valid_cols + w])
            ext_vals = np.tile(valid_vals, 3)
            interpolated = np.interp(cols.astype(np.float64), ext_cols, ext_vals)
            out = values.copy()
            out[~valid] = interpolated[~valid]
            return out

        # ── Fill small no-boundary holes caused by detection noise ────────────
        # A column with no non-sky pixel at all is either a real, deliberate water
        # break (never filled — water_present_column excludes it unconditionally)
        # or a short classification glitch with real terrain obviously on both
        # sides. Runs wider than max_hole_cols are left alone either way — past
        # that width we no longer trust that both sides are the same feature.
        noise_hole_column = (~has_silhouette) & (~water_present_column)
        if noise_hole_column.any() and has_silhouette.any() and max_hole_cols > 0:
            from scipy.ndimage import label as _label
            pad = max(max_hole_cols, 1)
            padded = np.concatenate([
                noise_hole_column[-pad:], noise_hole_column, noise_hole_column[:pad],
            ])
            labels, n_labels = _label(padded)
            fillable = np.zeros(w, dtype=bool)
            for lbl in range(1, n_labels + 1):
                idxs = np.where(labels == lbl)[0]
                if len(idxs) == 0 or len(idxs) > max_hole_cols:
                    continue
                # A run touching the padded window's own edge may be a truncated
                # copy of a larger run wrapped around — its true width isn't
                # knowable from this window, so leave it as a real gap rather
                # than risk under-counting a wide hole as fillable.
                if idxs[0] == 0 or idxs[-1] == len(padded) - 1:
                    continue
                orig_idxs = (idxs - pad) % w
                fillable[orig_idxs] = True

            if fillable.any():
                boundary_row = _circular_fill(boundary_row, has_silhouette)
                has_silhouette = has_silhouette | fillable

        half = grid_size_meters / 2.0
        ridge_radius_m = ridge_radius_frac * half

        # Place every ridge point on a shell at the fixed ridge_radius_m, using
        # only the boundary's real angle -- no depth involved. See this method's
        # own docstring for why: depth is both meaningless once compressed this
        # far, and unreliable right at the ridge crest in the first place.
        Xs, Ys, Zs = equirectangular_pixels_to_world(
            boundary_row, cols, np.full(w, ridge_radius_m, dtype=np.float64), h, w
        )

        # ── Near-field corroboration ────────────────────────────────────────
        # See this method's docstring for the "twin peak" failure mode this
        # corrects: a real, close hillside can reach the same elevation angle
        # as a genuinely distant summit and would otherwise get the exact same
        # synthetic Y. Only Y is touched -- XZ stays on the compressed shell.
        if panorama_depth is not None:
            depth_arr = panorama_depth.depth.astype(np.float64)
            if depth_arr.shape == (h, w):
                support_row = np.clip(
                    boundary_row.astype(np.int64) + near_field_support_row_offset, 0, h - 1
                )
                support_depth = depth_arr[support_row, cols]
                usable = (
                    has_silhouette
                    & np.isfinite(support_depth)
                    & (support_depth > 0)
                    & (support_depth < near_field_trust_distance_m)
                )
                if usable.any():
                    radius_for_height = np.full(w, ridge_radius_m, dtype=np.float64)
                    radius_for_height[usable] = np.clip(
                        support_depth[usable], near_field_min_radius_m, ridge_radius_m
                    )
                    phi = (0.5 - boundary_row / h) * np.pi
                    Ys = (radius_for_height * np.sin(phi)).astype(np.float32)

        # Mountains above the horizon have phi > 0, so Ys > 0 (above camera).
        in_bounds = has_silhouette
        Ys_valid = Ys[in_bounds]
        cols_valid = cols[in_bounds]
        Xs, Zs = Xs[in_bounds], Zs[in_bounds]

        if len(Xs) == 0:
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32), []

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)

        # ── Multi-chain greedy nearest-neighbour ──────────────────────────────
        # Build one chain per connected ridge segment. Each restart seeds from
        # the next unvisited point, capturing separate mountains in one pass.
        from scipy.spatial import KDTree as _KDTree
        from scipy.ndimage import uniform_filter1d as _uf1d

        cell_m = grid_size_meters / grid_resolution
        connect_m = connect_radius_px * cell_m
        max_col_gap = max(1.0, max_col_gap_frac * w)

        pts   = np.stack([Xs.astype(np.float64), Zs.astype(np.float64)], axis=-1)
        pts_y = Ys_valid.astype(np.float64)
        tree  = _KDTree(pts)

        def _col_gap(a: int, b: int) -> int:
            # Circular column distance, wrapping at the panorama's 360° seam.
            d = abs(int(a) - int(b))
            return min(d, w - d)

        all_chains_raw: list[list[int]] = []
        remaining = set(range(len(pts)))

        while remaining:
            seed = next(iter(remaining))
            remaining.discard(seed)
            chain = [seed]

            while True:
                curr = chain[-1]
                idxs = tree.query_ball_point(pts[curr], connect_m)
                candidates = [
                    i for i in idxs
                    if i in remaining and _col_gap(cols_valid[curr], cols_valid[i]) <= max_col_gap
                ]
                if not candidates:
                    break
                dists = np.linalg.norm(pts[candidates] - pts[curr], axis=1)
                nxt = candidates[int(np.argmin(dists))]
                chain.append(nxt)
                remaining.discard(nxt)

            all_chains_raw.append(chain)

        # ── Stitch small noise-scale gaps between chain fragments ─────────────
        # connect_radius_px/max_col_gap_frac above intentionally refuse to bridge
        # large gaps — most importantly water. But the same thresholds also
        # fragment a real, continuous boundary wherever depth or classification
        # noise nudges one column's projected point just outside them. Re-close
        # only genuinely small (noise-scale) gaps, on much tighter tolerances than
        # the main walk, and never across a column range containing any water.
        if hole_stitch_max_col_gap > 0 and len(all_chains_raw) > 1:
            stitch_dist_m = hole_stitch_max_dist_px * cell_m

            def _cols_between(a: int, b: int) -> np.ndarray:
                """Columns strictly between a and b along the shorter circular arc."""
                d_fwd = (b - a) % w
                d_bwd = (a - b) % w
                if d_fwd <= d_bwd:
                    return (a + np.arange(1, d_fwd)) % w
                return (b + np.arange(1, d_bwd)) % w

            def _try_stitch(chain_a: list[int], chain_b: list[int]) -> Optional[list[int]]:
                # Try all four ways two chains' endpoints can meet: a-end/b-start,
                # a-end/b-end, a-start/b-start, a-start/b-end (equivalently
                # b-end/a-start) — reversing whichever side is needed so the
                # result concatenates into one spatially continuous chain.
                for left, right in (
                    (chain_a, chain_b),
                    (chain_a, chain_b[::-1]),
                    (chain_a[::-1], chain_b),
                    (chain_b, chain_a),
                ):
                    col_l, col_r = cols_valid[left[-1]], cols_valid[right[0]]
                    if _col_gap(col_l, col_r) > hole_stitch_max_col_gap:
                        continue
                    if np.linalg.norm(pts[left[-1]] - pts[right[0]]) > stitch_dist_m:
                        continue
                    between = _cols_between(col_l, col_r)
                    if len(between) > 0 and water_present_column[between].any():
                        continue
                    return left + right
                return None

            merged = True
            while merged and len(all_chains_raw) > 1:
                merged = False
                for i in range(len(all_chains_raw)):
                    for j in range(i + 1, len(all_chains_raw)):
                        combined = _try_stitch(all_chains_raw[i], all_chains_raw[j])
                        if combined is not None:
                            all_chains_raw[i] = combined
                            del all_chains_raw[j]
                            merged = True
                            break
                    if merged:
                        break

        # Reconstruction (TerrainReconstructionStage) itself already discards any
        # chain shorter than 2 points, so that — not 3 — is the real floor; a
        # 2-point fragment that survived stitching is still a legitimate anchor.
        all_chains_data: list[tuple[np.ndarray, np.ndarray]] = []
        for chain in all_chains_raw:
            if len(chain) < 2:
                continue

            chain_xz  = pts[chain].copy()
            chain_y   = pts_y[chain].copy()

            if chain_smooth_window > 1:
                win = min(chain_smooth_window, len(chain))
                chain_xz[:, 0] = _uf1d(chain_xz[:, 0], size=win)
                chain_xz[:, 1] = _uf1d(chain_xz[:, 1], size=win)
                chain_y        = _uf1d(chain_y, size=win)

            all_chains_data.append((chain_xz, chain_y))

        # ── Rasterise all chains into binary grid ─────────────────────────────
        grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        for chain_xz, _ in all_chains_data:
            xi_c = np.clip(np.digitize(chain_xz[:, 0], x_edges) - 1, 0, grid_resolution - 1).astype(int)
            zi_c = np.clip(np.digitize(chain_xz[:, 1], z_edges) - 1, 0, grid_resolution - 1).astype(int)
            for k in range(len(xi_c) - 1):
                x0, z0 = xi_c[k],     zi_c[k]
                x1, z1 = xi_c[k + 1], zi_c[k + 1]
                n_steps = max(abs(x1 - x0), abs(z1 - z0), 1) + 1
                xs_seg = np.round(np.linspace(x0, x1, n_steps)).astype(int)
                zs_seg = np.round(np.linspace(z0, z1, n_steps)).astype(int)
                valid_seg = (xs_seg >= 0) & (xs_seg < grid_resolution) & (zs_seg >= 0) & (zs_seg < grid_resolution)
                grid[zs_seg[valid_seg], xs_seg[valid_seg]] = 1.0
            if len(xi_c) > 0:
                grid[zi_c[-1], xi_c[-1]] = 1.0

        if dilation_iters > 0:
            from scipy.ndimage import binary_dilation
            grid = binary_dilation(grid > 0, iterations=dilation_iters).astype(np.float32)

        # ── Prominence-based summit selection ───────────────────────────────────
        # See prominence_min_m/prominence_shoulder_m's own docstring above for
        # why this exists. Only affects xyz_chains (the anchoring output) --
        # `grid` above was already rasterised from the full, unpruned chains.
        anchor_chains_data = all_chains_data
        if prominence_min_m > 0:
            from scipy.signal import find_peaks, peak_prominences

            anchor_chains_data = []
            for chain_xz, chain_y in all_chains_data:
                n = len(chain_y)
                if n < 5:
                    anchor_chains_data.append((chain_xz, chain_y))
                    continue

                seg = np.hypot(np.diff(chain_xz[:, 0]), np.diff(chain_xz[:, 1]))
                arc = np.concatenate([[0.0], np.cumsum(seg)])

                # Tiled x3: see docstring -- a real summit can straddle this
                # chain's arbitrary start/end index for a near-closed ring.
                y3 = np.concatenate([chain_y, chain_y, chain_y])
                peaks3, _ = find_peaks(y3)
                if len(peaks3) == 0:
                    continue
                prominences3 = peak_prominences(y3, peaks3)[0]

                keep = np.zeros(n, dtype=bool)
                for p3, prom in zip(peaks3, prominences3):
                    if prom < prominence_min_m:
                        continue
                    p = p3 - n
                    if not (0 <= p < n):
                        continue
                    keep |= np.abs(arc - arc[p]) <= prominence_shoulder_m

                if not keep.any():
                    continue

                # Contiguous kept runs become separate chains -- a summit's
                # shoulder is a local window, not (necessarily) the whole chain.
                edges = np.flatnonzero(np.diff(np.concatenate([[False], keep, [False]])))
                for start, stop in zip(edges[0::2], edges[1::2]):
                    if stop - start >= 2:
                        anchor_chains_data.append((chain_xz[start:stop], chain_y[start:stop]))

        # ── Assemble XYZ chains ───────────────────────────────────────────────
        xyz_chains: list[np.ndarray] = []
        for chain_xz, chain_y in anchor_chains_data:
            xyz_chains.append(np.stack([
                chain_xz[:, 0].astype(np.float32),
                chain_y.astype(np.float32),
                chain_xz[:, 1].astype(np.float32),
            ], axis=1))

        return grid, xyz_chains

    @staticmethod
    def extract_water_chains(
        type_idx_map: np.ndarray,
        panorama_depth: Depth,
        water_idx: int,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 4096,
        ground_y_max: float = -0.5,
        camera_height_meters: float = 1.0,
        min_chain_len: int = 8,
        smooth_radius: int = 20,
        connect_radius_px: int = 10,
        chain_smooth_window: int = 7,
    ) -> list[np.ndarray]:
        """
        Extract water surface centerlines as ordered 3D polylines.

        Projects water-classified pixels from the panorama to the top-down grid,
        builds a skeleton of the water region, then chains skeleton cells into
        ordered polylines. Each chain is oriented upstream → downstream (higher
        camera-relative Y first: less-negative Y = higher terrain elevation).

        Returns list of (M, 3) float32 arrays of world (X, Y, Z) in camera-relative
        coordinates. Chains shorter than min_chain_len are discarded.
        """
        half = grid_size_meters / 2.0
        ground_y_min = -(camera_height_meters + 5.0)

        # Project water pixels to 3D; mask everything else.
        d_water = panorama_depth.depth.astype(np.float32).copy()
        d_water[type_idx_map != water_idx] = np.nan

        xi, zi, _, Yg, _ = project_panorama_to_ground_grid(
            Depth(d_water), grid_size_meters, grid_resolution, ground_y_max, ground_y_min
        )
        if len(xi) == 0:
            return []

        y_sum = np.zeros((grid_resolution, grid_resolution), dtype=np.float64)
        y_cnt = np.zeros((grid_resolution, grid_resolution), dtype=np.int32)
        np.add.at(y_sum, (zi, xi), Yg)
        np.add.at(y_cnt, (zi, xi), 1)

        water_grid = y_cnt > 0
        if not np.any(water_grid):
            return []

        y_mean = np.where(water_grid, y_sum / np.maximum(y_cnt, 1), np.nan)

        # Skeletonize the projected water region.
        from skimage.morphology import skeletonize as _skel, disk, closing, opening
        mask = water_grid.copy()
        if smooth_radius > 0:
            d_morph = disk(smooth_radius)
            mask = closing(mask, d_morph)
            mask = opening(mask, d_morph)
        skeleton = _skel(mask)

        skel_zi, skel_xi = np.where(skeleton)
        if len(skel_zi) == 0:
            return []

        cell_m = grid_size_meters / grid_resolution
        skel_X = (skel_xi + 0.5) * cell_m - half
        skel_Z = (skel_zi + 0.5) * cell_m - half
        skel_Y = y_mean[skel_zi, skel_xi]

        # Fill any NaN Y values via nearest-neighbor from valid skeleton cells.
        nan_mask = ~np.isfinite(skel_Y)
        if nan_mask.any() and (~nan_mask).any():
            from scipy.spatial import cKDTree as _cKDTree
            valid_xz  = np.stack([skel_X[~nan_mask], skel_Z[~nan_mask]], axis=1)
            nan_xz    = np.stack([skel_X[nan_mask],  skel_Z[nan_mask]],  axis=1)
            _, nn_idx = _cKDTree(valid_xz).query(nan_xz)
            skel_Y = skel_Y.copy()
            skel_Y[nan_mask] = skel_Y[~nan_mask][nn_idx]

        if not np.all(np.isfinite(skel_Y)):
            return []

        # Chain skeleton points using greedy nearest-neighbor, multiple chains.
        from scipy.spatial import KDTree as _KDTree
        from scipy.ndimage import uniform_filter1d as _uf1d

        pts   = np.stack([skel_X.astype(np.float64), skel_Z.astype(np.float64)], axis=1)
        pts_y = skel_Y.astype(np.float64)
        tree  = _KDTree(pts)
        connect_m = connect_radius_px * cell_m

        all_chains: list[np.ndarray] = []
        remaining = set(range(len(pts)))

        while remaining:
            seed = next(iter(remaining))
            remaining.discard(seed)
            chain = [seed]

            while True:
                curr = chain[-1]
                idxs = tree.query_ball_point(pts[curr], connect_m)
                candidates = [i for i in idxs if i in remaining]
                if not candidates:
                    break
                dists = np.linalg.norm(pts[candidates] - pts[curr], axis=1)
                nxt = candidates[int(np.argmin(dists))]
                chain.append(nxt)
                remaining.discard(nxt)

            if len(chain) < min_chain_len:
                continue

            chain_xz = pts[chain].copy()
            chain_y  = pts_y[chain].copy()

            if len(chain) >= 3 and chain_smooth_window > 1:
                win = min(chain_smooth_window, len(chain))
                chain_xz[:, 0] = _uf1d(chain_xz[:, 0], size=win)
                chain_xz[:, 1] = _uf1d(chain_xz[:, 1], size=win)
                chain_y        = _uf1d(chain_y, size=win)

            # Orient upstream → downstream: higher Y (less negative) = higher elevation.
            if chain_y[0] < chain_y[-1]:
                chain_xz = chain_xz[::-1].copy()
                chain_y  = chain_y[::-1].copy()

            all_chains.append(np.stack([
                chain_xz[:, 0].astype(np.float32),
                chain_y.astype(np.float32),
                chain_xz[:, 1].astype(np.float32),
            ], axis=1))

        return all_chains

    @staticmethod
    def extract_region_skeleton(
        region_map: np.ndarray,
        type_idx: int,
        smooth_radius: int = 40,
    ) -> np.ndarray:
        """
        Skeletonize a single region type in the top-down region map.

        Returns a float32 (H, W) binary mask of the medial axis of all matching
        cells, reducing area features to 1-pixel-wide centerlines.

        smooth_radius: closing+opening radius applied before skeletonizing. Use
        larger values for area features like water bodies; smaller values for
        inherently linear features like roads and trails.
        """
        from skimage.morphology import skeletonize as sk_skeletonize, disk, closing, opening
        mask = region_map == type_idx
        if not np.any(mask):
            return np.zeros(region_map.shape, dtype=np.float32)
        if smooth_radius > 0:
            d = disk(smooth_radius)
            mask = closing(mask, d)
            mask = opening(mask, d)
        return sk_skeletonize(mask).astype(np.float32)

    @staticmethod
    def extract_interior_peaks(
        type_idx_map: np.ndarray,
        panorama_depth: Depth,
        sky_idx: int,
        panorama_rgb: Optional[np.ndarray] = None,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 4096,
        depth_jump_rel: float = 0.20,
        canny_low: int = 50,
        canny_high: int = 150,
        corner_quality: float = 0.08,
        corner_min_dist: int = 5,
        max_range_factor: float = 2.0,
        min_depth_m: float = 50.0,
    ) -> np.ndarray:
        """
        Detect elevated terrain features visible against background terrain (rather
        than against sky) and project them onto a top-down grid.

        The sky-terrain ridgeline captures only the outermost horizon silhouette.
        This method finds interior peaks — rocky outcrops, ridges, or hilltops that
        appear in front of more distant terrain — by combining three signals:

          1. Relative depth jumps: |∇d| / d > depth_jump_rel detects boundaries
             where foreground terrain abruptly occludes more distant ground.

          2. Canny edges on the RGB panorama: visual edge boundaries at
             colour/luminance discontinuities characteristic of rock faces and
             hard terrain edges.

          3. Harris corner responses on the RGB panorama: corner features at
             convergence points of two edges, which correspond to the apices of
             pointed peaks and the vertices of rocky outcrops.

        Pixels must satisfy both (1) AND at least one of (2)/(3) to be accepted —
        this filters out pure depth-estimation noise (no visual evidence) and flat
        colour boundaries (no depth change). Sky, vegetation, and built pixels are
        excluded: foliage canopies and structures are full of depth jumps and
        Canny/corner responses of their own (leaf clusters, eaves, window
        mullions) that look just like rock edges to this heuristic, but they're
        occluding objects rather than terrain and are handled by separate object
        extraction, not the terrain mesh. Pixels nearer than min_depth_m are also
        excluded: depth_jump_rel is a *relative* threshold, so it's trivially
        satisfied by routine close-range ground undulation (a dip, a rock, a
        snow-patch edge) that isn't a meaningfully "elevated" feature — and since
        near_side below always ends up selecting whichever side of an edge is
        already closer, near-camera terrain dominates the output by construction
        unless it's excluded up front. Pixels moderately farther than half the
        grid size away in XZ (within max_range_factor) are clamped onto the grid
        boundary along their camera ray rather than dropped, same treatment as
        extract_mountain_ridgeline; pixels beyond that are dropped outright — a
        real mountain summit kilometres out isn't representable in a local
        terrain grid this small, and clamping it onto the boundary just produces
        a meaningless flattened position once reprojected using local terrain
        elevation (that content is already covered by the sky-terrain ridgeline
        silhouette this method is meant to complement, not duplicate).

        The panorama is processed at its native resolution; if depth or type maps
        are at a different resolution they are rescaled to match.

        Returns float32 (grid_resolution, grid_resolution) with each cell containing
        the count of peak-candidate pixels that projected into it, normalised to [0,1].
        """
        ref_h, ref_w = type_idx_map.shape

        # Align depth to type-map resolution if needed.
        d_raw = panorama_depth.depth.astype(np.float32)
        if d_raw.shape != (ref_h, ref_w):
            d_raw = zoom(d_raw, (ref_h / d_raw.shape[0], ref_w / d_raw.shape[1]), order=1)

        # Work at panorama_rgb resolution when provided (typically lower than depth).
        if panorama_rgb is not None:
            work_h, work_w = panorama_rgb.shape[:2]
            if (work_h, work_w) != (ref_h, ref_w):
                d_work = zoom(d_raw, (work_h / ref_h, work_w / ref_w), order=1)
                types_work = zoom(type_idx_map.astype(np.float32), (work_h / ref_h, work_w / ref_w), order=0).astype(np.uint8)
            else:
                d_work, types_work = d_raw, type_idx_map
        else:
            d_work, types_work = d_raw, type_idx_map
            work_h, work_w = ref_h, ref_w

        sky_mask = types_work == sky_idx
        occluder_mask = (types_work == RegionType.VEGETATION) | (types_work == RegionType.BUILT)
        valid = (
            (~sky_mask) & (~occluder_mask)
            & np.isfinite(d_work) & (d_work > max(0.5, min_depth_m))
        )

        # ── 1. Relative depth gradient ────────────────────────────────────────
        gy, gx = np.gradient(d_work)
        grad_mag = np.hypot(gx, gy)
        rel_grad = np.where(d_work > 0, grad_mag / d_work, 0.0)
        depth_edge = (rel_grad > depth_jump_rel) & valid

        # Exclude pixels that sit at the sky boundary (those are the ridgeline).
        above_is_sky = np.zeros((work_h, work_w), dtype=bool)
        above_is_sky[1:, :] = sky_mask[:-1, :]
        at_skyline = (~sky_mask) & above_is_sky
        depth_edge &= ~at_skyline

        if not depth_edge.any():
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        # ── 2 & 3. Visual edges and corners from RGB ─────────────────────────
        visual = np.zeros((work_h, work_w), dtype=bool)
        if panorama_rgb is not None:
            gray = cv2.cvtColor(panorama_rgb, cv2.COLOR_RGB2GRAY)
            canny = cv2.Canny(gray, canny_low, canny_high) > 0

            # Shi-Tomasi corners — peaks of pointed features appear as corners
            # where two ridgeline segments converge.
            corners_f32 = cv2.goodFeaturesToTrack(
                gray, maxCorners=0, qualityLevel=corner_quality,
                minDistance=corner_min_dist,
            )
            corner_map = np.zeros((work_h, work_w), dtype=bool)
            if corners_f32 is not None:
                pts = corners_f32[:, 0, :].astype(np.int32)
                cx = np.clip(pts[:, 0], 0, work_w - 1)
                cy = np.clip(pts[:, 1], 0, work_h - 1)
                corner_map[cy, cx] = True
                # Dilate corners by 3 px so they overlap with the depth-edge mask.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                corner_map = cv2.dilate(corner_map.astype(np.uint8), kernel).astype(bool)

            visual = canny | corner_map

        # Require depth evidence; additionally gate on visual evidence when available.
        candidates = depth_edge & (visual if panorama_rgb is not None else np.ones_like(depth_edge))

        if not candidates.any():
            # Fall back to depth-only if RGB filtering removed everything.
            candidates = depth_edge

        # ── Project to top-down grid ─────────────────────────────────────────
        # Use the near-depth side: for each edge pixel, the smaller of its own
        # depth and its immediate neighbours is the foreground (peak) distance.
        from scipy.ndimage import minimum_filter
        min_d = minimum_filter(d_work, size=3)
        # Accept pixels where the local minimum depth is within 20% of their own
        # depth — this biases toward the foreground side of each edge.
        near_side = candidates & (d_work <= min_d * 1.2)
        if not near_side.any():
            near_side = candidates

        rows, cols = np.where(near_side)
        depths = min_d[rows, cols]

        half = grid_size_meters / 2.0
        Xs, _, Zs = equirectangular_pixels_to_world(rows, cols, depths, work_h, work_w)

        in_bounds = np.isfinite(Xs) & np.isfinite(Zs)
        Xs, Zs = Xs[in_bounds], Zs[in_bounds]

        # Drop peaks far beyond what a local terrain grid can plausibly represent
        # (e.g. the mountain's own summit, kilometres out) — unlike
        # extract_mountain_ridgeline's silhouette (which this content already
        # duplicates), there's no meaningful "edge of the local terrain" position
        # for something that far away, and clamping it onto the boundary would
        # only produce a flattened artifact once reprojected using local terrain
        # elevation for debug display.
        r_inf = np.maximum(np.abs(Xs), np.abs(Zs))
        in_range = r_inf <= half * max_range_factor
        Xs, Zs, r_inf = Xs[in_range], Zs[in_range], r_inf[in_range]

        if len(Xs) == 0:
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        # Remaining moderately-out-of-range peaks (within max_range_factor) are
        # still clamped onto the grid boundary along their camera ray rather than
        # dropped — same treatment as extract_mountain_ridgeline — since a real
        # ridge just past the grid edge is still plausibly part of the local scene.
        scale = np.where(r_inf > half, half / np.maximum(r_inf, 1e-9), 1.0)
        Xs, Zs = Xs * scale, Zs * scale

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)
        xi = np.clip(np.digitize(Xs, x_edges) - 1, 0, grid_resolution - 1)
        zi = np.clip(np.digitize(Zs, z_edges) - 1, 0, grid_resolution - 1)

        grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        np.add.at(grid, (zi, xi), 1.0)

        peak_max = grid.max()
        if peak_max > 0:
            grid /= peak_max
        return grid
