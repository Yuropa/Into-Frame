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
        panorama_depth: Depth,
        sky_idx: int,
        water_idx: int = -1,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 4096,
        depth_offset_rows: int = 3,
        depth_smooth_width: int = 15,
        dilation_iters: int = 3,
        connect_radius_px: int = 30,
        chain_smooth_window: int = 15,
        max_col_gap_frac: float = 0.01,
        max_hole_cols: int = 12,
        hole_stitch_max_col_gap: int = 12,
        hole_stitch_max_dist_px: int = 90,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Extract the sky-foreground horizon per column, sample depth just below it,
        and project to a top-down grid using actual XZ positions.

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

        For each panorama column, finds the first non-sky row below sky, then
        samples depth depth_offset_rows below that boundary (where depth estimators
        are more reliable than at the exact edge). Columns with no non-sky pixel at
        all are usually a short classification glitch, not a real absence — real
        terrain is obviously present just past it on both sides — so they get a
        synthetic boundary row circularly interpolated from their nearest real
        neighbours, but only for runs up to max_hole_cols columns wide; wider runs
        are left as a genuine gap rather than fabricating a long stretch of invented
        ridge. NaN/invalid depth values (at both real and synthesised-row columns)
        are then interpolated from neighbouring columns, wrapping at the 360° seam.
        Each valid column is unprojected to XYZ and placed in the grid — close
        mountains land near the centre, distant ones near the edge.

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

        Returns (grid, chains) where:
          grid   — float32 (grid_resolution, grid_resolution) binary mask (unchanged).
          chains — list of (M, 3) float32 arrays of world (X, Y, Z) per ridge chain,
                   where Y is the camera-relative elevation of the ridge crest
                   (Y = depth * sin(phi), positive = above camera).
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

        sample_rows = np.clip(np.round(boundary_row).astype(np.int64) + depth_offset_rows, 0, h - 1)

        d = panorama_depth.depth.astype(np.float32)
        depths = np.where(has_silhouette, d[sample_rows, cols], np.nan)

        # Interpolate NaN/invalid depths (circularly) so depth gaps don't leave holes.
        nan_mask = ~np.isfinite(depths) | (depths <= 0)
        valid_for_interp = has_silhouette & ~nan_mask
        if valid_for_interp.any() and (~valid_for_interp).any():
            depths = _circular_fill(depths, valid_for_interp)

        # Suppress single-column depth spikes before projecting to XZ.
        if depth_smooth_width > 1 and np.any(np.isfinite(depths)):
            from scipy.ndimage import median_filter as _med
            _arr = np.where(np.isfinite(depths), depths, 0.0)
            depths = np.where(np.isfinite(depths), _med(_arr, size=depth_smooth_width, mode='nearest'), depths)

        Xs, Ys, Zs = equirectangular_pixels_to_world(sample_rows, cols, depths, h, w)

        half = grid_size_meters / 2.0
        in_bounds = has_silhouette & np.isfinite(depths)

        # Ridge points beyond the grid's representable extent (very often true for
        # genuinely distant mountains) are clamped radially onto the grid boundary
        # along their real camera ray, rather than silently dropped — a distant
        # peak should render at the edge of the mesh, not vanish or leave a gap in
        # the silhouette. Y is foreshortened by the same factor so the ray's real
        # elevation angle from the camera is preserved (only the distance along it
        # is compressed to fit).
        r_inf = np.maximum(np.abs(Xs), np.abs(Zs))
        scale = np.where(r_inf > half, half / np.maximum(r_inf, 1e-9), 1.0)
        Xs, Ys, Zs = Xs * scale, Ys * scale, Zs * scale

        # Mountains above the horizon have phi > 0, so Ys > 0 (above camera).
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

        # ── Assemble XYZ chains ───────────────────────────────────────────────
        xyz_chains: list[np.ndarray] = []
        for chain_xz, chain_y in all_chains_data:
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
