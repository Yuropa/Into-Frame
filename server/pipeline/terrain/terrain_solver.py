"""
Sparse least-squares terrain reconstruction inspired by feature-primitive terrain modelling.

Solves for a globally coherent heightmap that satisfies:
  - Data preservation  : high-confidence regions stay close to the original DEM.
  - Smoothness         : discrete Laplacian regulariser prevents noise artefacts.
  - Ridge constraints  : ridge pixels should be higher than their neighbours.
  - River constraints  : river pixels should be lower than their neighbours, with
                         monotonic descent along the channel centreline.
  - Peak / lake anchors: optional hard-ish elevation pins.

Polyline points are (row, col) raster coordinates in [0..H-1] × [0..W-1].
"""
from __future__ import annotations

import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from scipy.ndimage import distance_transform_edt
from typing import Optional


class TerrainSolver:
    """
    Constrained least-squares terrain reconstruction.

    Usage::
        solver = TerrainSolver(heightmap, confidence, laplacian_weight=0.1)
        solver.add_river_polyline(river_pts, weight=2.0, valley_depth=0.5)
        solver.add_ridge_mask(mountain_mask, weight=2.0, crest_height=0.5)
        new_hm = solver.solve()
    """

    def __init__(
        self,
        heightmap: np.ndarray,
        confidence: np.ndarray,
        laplacian_weight: float = 0.1,
        data_weight: float = 1.0,
        iter_lim: int = 500,
    ):
        """
        Args:
            heightmap:        (H, W) float32 — existing DEM values.
            confidence:       (H, W) float32 — per-pixel confidence in [0, 1].
                              1.0 = do not move; 0.0 = free to change.
            laplacian_weight: weight for the smoothness term.
            data_weight:      multiplier on the confidence map for the data term.
                              Reduce below 1.0 to treat the initial height map as
                              a weak prior — unobserved cells (confidence ≈ 0) get
                              no data term at all and are shaped by the Laplacian
                              and feature constraints instead. Values around 0.3
                              leave high-confidence observed cells influential while
                              letting feature constraints (ridges, rivers) dominate.
            iter_lim:         maximum LSQR iterations.
        """
        self._hm = heightmap.astype(np.float64)
        self._conf = np.clip(confidence, 0.0, 1.0).astype(np.float64)
        self._H, self._W = heightmap.shape
        self._N = self._H * self._W
        self._lap_w = laplacian_weight
        self._data_weight = data_weight
        self._iter_lim = iter_lim

        # Triplet accumulator: (local_row_indices, col_indices, values, n_rows)
        self._blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        self._rhs_blocks: list[np.ndarray] = []

    # ── Triplet accumulation ──────────────────────────────────────────────────

    def _add_block(
        self,
        local_rows: np.ndarray,
        cols: np.ndarray,
        vals: np.ndarray,
        rhs: np.ndarray,
    ) -> None:
        n = int(rhs.shape[0])
        if n == 0:
            return
        self._blocks.append((
            local_rows.astype(np.int64),
            cols.astype(np.int64),
            vals.astype(np.float64),
            n,
        ))
        self._rhs_blocks.append(rhs.astype(np.float64))

    # ── Base terms (built at solve time) ─────────────────────────────────────

    def _build_data_term(self) -> None:
        conf_flat = self._conf.ravel()
        orig_flat = self._hm.ravel()
        # Scale confidence by data_weight. Cells with negligible weight (unobserved /
        # interpolated) are skipped entirely so they are shaped by the Laplacian and
        # feature constraints rather than being anchored to noisy interpolated values.
        w = conf_flat * self._data_weight
        active = w > 1e-4
        if not active.any():
            return
        idxs = np.where(active)[0]
        n = len(idxs)
        self._add_block(np.arange(n), idxs, w[active], w[active] * orig_flat[active])

    def _build_laplacian_term(self) -> None:
        if self._lap_w <= 0.0:
            return
        H, W, lw = self._H, self._W, self._lap_w

        y_arr = np.repeat(np.arange(1, H - 1), W - 2)
        x_arr = np.tile(np.arange(1, W - 1), H - 2)
        n_int = len(y_arr)

        center = y_arr * W + x_arr
        up     = (y_arr - 1) * W + x_arr
        down   = (y_arr + 1) * W + x_arr
        left   = y_arr * W + (x_arr - 1)
        right  = y_arr * W + (x_arr + 1)

        local_rows = np.repeat(np.arange(n_int), 5)
        cols = np.stack([center, up, down, left, right], axis=1).ravel()
        vals = np.tile(np.array([4 * lw, -lw, -lw, -lw, -lw]), n_int)
        self._add_block(local_rows, cols, vals, np.zeros(n_int))

    # ── Shared helper: Gaussian cross-sectional profile constraints ──────────

    @staticmethod
    def _profile_constraints(
        mask: np.ndarray,
        H: int,
        W: int,
        weight: float,
        amplitude: float,
        sigma: float,
        sign: int,
        n_sigma: float = 2.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Emit one constraint per pixel within n_sigma*sigma of the mask.

        For each such pixel at Euclidean distance d from its nearest mask pixel,
        with G(d) = exp(-d²/(2σ²)):

          sign > 0 (ridge):  h_nearest - h_pixel = amplitude * G(d)
          sign < 0 (valley): h_pixel - h_nearest = amplitude * G(d)

        Both the target difference and the constraint weight scale with G(d),
        encoding a Gaussian cross-sectional profile: the feature centreline sits
        amplitude above (ridge) or below (valley) immediately adjacent terrain,
        with the required difference decaying smoothly with perpendicular distance.

        Returns (local_rows, cols, vals, rhs) ready for _add_block.
        """
        dist, indices = distance_transform_edt(~mask, return_indices=True)

        within = (dist > 0) & (dist < n_sigma * sigma)
        ya, xa = np.where(within)
        if len(ya) == 0:
            return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float64),) * 2

        d = dist[ya, xa]
        g = np.exp(-d ** 2 / (2.0 * sigma ** 2))

        # Nearest centreline pixel for each constrained pixel
        yr = indices[0][ya, xa]
        xr = indices[1][ya, xa]

        w           = weight * g
        profile_val = amplitude * g

        feat_idx  = yr * W + xr
        pixel_idx = ya * W + xa

        if sign > 0:
            first_col, second_col = feat_idx, pixel_idx
        else:
            first_col, second_col = pixel_idx, feat_idx

        n = len(ya)
        local_rows = np.repeat(np.arange(n), 2)
        cols = np.stack([first_col, second_col], axis=1).ravel()
        vals = np.column_stack([w, -w]).ravel()
        rhs  = w * profile_val
        return local_rows, cols, vals, rhs

    @staticmethod
    def _perpendicular_profile_constraints(
        pts: np.ndarray,
        H: int,
        W: int,
        weight: float,
        amplitude: float,
        sigma: float,
        sign: int,
        n_sigma: float = 2.5,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Like _profile_constraints but uses true perpendicular distance to the
        continuous polyline rather than Euclidean distance to the nearest
        rasterized pixel.

        For each candidate pixel q, we find the closest point p* on any segment
        of *pts*.  Because p* is the orthogonal projection of q onto that segment,
        dist(q, p*) is by definition perpendicular to the segment tangent.  This
        gives a cross-sectional profile that is faithful to the curve geometry
        even at bends and for sparsely sampled polylines.

        Strategy:
          1. Rasterize pts → binary mask; run EDT as a cheap pre-filter to
             identify candidate pixels within 1.2 × n_sigma × sigma.
          2. For each candidate, iterate over segments to find the true
             minimum perpendicular distance and the nearest point p* on the
             polyline.
          3. Apply the Gaussian profile using the true distance.
        """
        if len(pts) < 2:
            return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float64),) * 2

        # ── Step 1: fast candidate pre-filter ────────────────────────────────
        mask = np.zeros((H, W), dtype=bool)
        r_px = np.clip(np.round(pts[:, 0]).astype(int), 0, H - 1)
        c_px = np.clip(np.round(pts[:, 1]).astype(int), 0, W - 1)
        mask[r_px, c_px] = True

        edt_dist = distance_transform_edt(~mask)
        candidate_mask = (edt_dist > 0) & (edt_dist < n_sigma * sigma * 1.2)
        ya, xa = np.where(candidate_mask)
        if len(ya) == 0:
            return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float64),) * 2

        cand = np.stack([ya, xa], axis=1).astype(np.float64)  # (M, 2)

        # ── Step 2: true perpendicular distance to polyline ───────────────────
        float_pts  = pts.astype(np.float64)
        min_dist   = np.full(len(ya), np.inf)
        nearest_pt = np.zeros((len(ya), 2))

        for i in range(len(float_pts) - 1):
            p0    = float_pts[i]
            p1    = float_pts[i + 1]
            seg   = p1 - p0
            len_sq = float(np.dot(seg, seg))
            if len_sq < 1e-10:
                t = np.zeros(len(ya))
            else:
                t = np.clip(((cand - p0) @ seg) / len_sq, 0.0, 1.0)
            near = p0 + t[:, np.newaxis] * seg          # (M, 2) nearest points
            dist = np.linalg.norm(cand - near, axis=1)  # (M,)
            closer = dist < min_dist
            min_dist[closer]   = dist[closer]
            nearest_pt[closer] = near[closer]

        # ── Step 3: apply Gaussian profile using true distance ────────────────
        within = min_dist < n_sigma * sigma
        if not within.any():
            return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float64),) * 2

        ya       = ya[within]
        xa       = xa[within]
        d_final  = min_dist[within]
        near_fin = nearest_pt[within]

        g           = np.exp(-d_final ** 2 / (2.0 * sigma ** 2))
        w           = weight * g
        profile_val = amplitude * g

        yr = np.clip(np.round(near_fin[:, 0]).astype(int), 0, H - 1)
        xr = np.clip(np.round(near_fin[:, 1]).astype(int), 0, W - 1)

        feat_idx  = yr * W + xr
        pixel_idx = ya * W + xa

        if sign > 0:
            first_col, second_col = feat_idx, pixel_idx
        else:
            first_col, second_col = pixel_idx, feat_idx

        n = len(ya)
        local_rows = np.repeat(np.arange(n), 2)
        cols = np.stack([first_col, second_col], axis=1).ravel()
        vals = np.column_stack([w, -w]).ravel()
        rhs  = w * profile_val
        return local_rows, cols, vals, rhs

    # ── Public API ────────────────────────────────────────────────────────────

    def add_ridge_polyline(
        self,
        points: np.ndarray,
        weight: float = 1.0,
        crest_height: float = 0.5,
        sigma: float = 20.0,
    ) -> None:
        """
        Add ridge crest constraints from an ordered polyline.

        Ridge pixels are required to be *crest_height* above each neighbour.
        Constraints are weighted by a Gaussian influence field centred on the
        rasterised ridge.

        Args:
            points:       (N, 2) array of (row, col) coordinates.
            weight:       overall constraint weight.
            crest_height: required elevation above neighbours (same units as heightmap).
            sigma:        Gaussian falloff radius in pixels.
        """
        if len(points) < 2:
            return
        H, W = self._H, self._W
        lr, cols, vals, rhs = self._perpendicular_profile_constraints(
            points, H, W, weight, crest_height, sigma, sign=+1
        )
        self._add_block(lr, cols, vals, rhs)

    def add_ridge_mask(
        self,
        mask: np.ndarray,
        weight: float = 1.0,
        crest_height: float = 0.5,
        sigma: float = 20.0,
    ) -> None:
        """
        Add ridge crest constraints from a binary mask (same shape as heightmap).

        Ridge pixels should be *crest_height* above each of their 4-connected
        neighbours, weighted by a Gaussian influence field.
        """
        H, W = self._H, self._W
        if mask.shape != (H, W):
            raise ValueError(f"mask shape {mask.shape} != heightmap shape {(H, W)}")
        lr, cols, vals, rhs = self._profile_constraints(
            mask, H, W, weight, crest_height, sigma, sign=+1
        )
        self._add_block(lr, cols, vals, rhs)

    def add_river_polyline(
        self,
        points: np.ndarray,
        weight: float = 1.0,
        valley_depth: float = 0.5,
        drop_per_segment: float = 0.05,
        sigma: float = 10.0,
    ) -> None:
        """
        Add river valley constraints from an ordered polyline (upstream → downstream).

        Two constraint types are added:
          1. Valley cross-section: neighbours are *valley_depth* above the river.
          2. Flow monotonicity:    each successive vertex is lower by *drop_per_segment*.

        Args:
            points:           (N, 2) array of (row, col) coordinates, ordered
                              upstream → downstream.
            weight:           overall constraint weight.
            valley_depth:     required elevation of neighbours above river channel.
            drop_per_segment: required elevation drop per polyline segment.
            sigma:            Gaussian falloff radius in pixels for valley constraints.
        """
        if len(points) < 2:
            return
        H, W = self._H, self._W

        pt_rows = np.clip(np.round(points[:, 0]).astype(int), 0, H - 1)
        pt_cols = np.clip(np.round(points[:, 1]).astype(int), 0, W - 1)

        # Valley cross-section: perpendicular profile from continuous polyline
        lr, cols, vals, rhs = self._perpendicular_profile_constraints(
            points, H, W, weight, valley_depth, sigma, sign=-1
        )
        self._add_block(lr, cols, vals, rhs)

        # Flow constraints: h(p_i) - h(p_{i+1}) = drop_per_segment
        changed = (pt_rows[:-1] != pt_rows[1:]) | (pt_cols[:-1] != pt_cols[1:])
        r0 = pt_rows[:-1][changed]
        c0 = pt_cols[:-1][changed]
        r1 = pt_rows[1:][changed]
        c1 = pt_cols[1:][changed]

        if len(r0) == 0:
            return

        p0_idx = r0 * W + c0
        p1_idx = r1 * W + c1
        n_flow  = len(p0_idx)
        flow_w  = weight * 5.0  # stronger weight for monotonic flow

        local_rows = np.repeat(np.arange(n_flow), 2)
        cols = np.stack([p0_idx, p1_idx], axis=1).ravel()
        vals = np.tile(np.array([flow_w, -flow_w]), n_flow)
        rhs  = np.full(n_flow, flow_w * drop_per_segment)
        self._add_block(local_rows, cols, vals, rhs)

    def add_river_mask(
        self,
        mask: np.ndarray,
        weight: float = 1.0,
        valley_depth: float = 0.5,
        sigma: float = 10.0,
    ) -> None:
        """
        Add river valley cross-section constraints from a binary mask.

        Neighbours of river pixels are required to be *valley_depth* above the channel.
        """
        H, W = self._H, self._W
        if mask.shape != (H, W):
            raise ValueError(f"mask shape {mask.shape} != heightmap shape {(H, W)}")
        lr, cols, vals, rhs = self._profile_constraints(
            mask, H, W, weight, valley_depth, sigma, sign=-1
        )
        self._add_block(lr, cols, vals, rhs)

    def add_peak(
        self,
        x: float,
        y: float,
        elevation: float,
        weight: float = 1000.0,
    ) -> None:
        """
        Pin a single pixel to a target elevation.

        Args:
            x:         column (horizontal) in raster coordinates.
            y:         row (vertical) in raster coordinates.
            elevation: target height value (same units as heightmap).
            weight:    constraint weight (high = near-hard constraint).
        """
        row = int(np.clip(round(y), 0, self._H - 1))
        col = int(np.clip(round(x), 0, self._W - 1))
        idx = row * self._W + col
        self._add_block(
            np.array([0]),
            np.array([idx]),
            np.array([weight]),
            np.array([weight * elevation]),
        )

    def add_lake(
        self,
        polygon: np.ndarray,
        level: float,
        weight: float = 500.0,
    ) -> None:
        """
        Enforce a flat water surface inside a polygon.

        Args:
            polygon: (N, 2) array of (row, col) boundary vertices.
            level:   target elevation for all lake pixels.
            weight:  constraint weight.
        """
        from skimage.draw import polygon as sk_polygon
        H, W = self._H, self._W
        rr, cc = sk_polygon(polygon[:, 0], polygon[:, 1], shape=(H, W))
        if len(rr) == 0:
            return
        n = len(rr)
        idxs = rr * W + cc
        self._add_block(
            np.arange(n),
            idxs,
            np.full(n, weight),
            np.full(n, weight * level),
        )

    def add_ridge_polyline_anchored(
        self,
        points_xyz: np.ndarray,
        grid_size_meters: float,
        weight: float = 1.0,
        crest_height: float = 0.5,
        sigma: float = 20.0,
        anchor_weight: float = 5.0,
        anchor_stride: int = 5,
    ) -> None:
        """
        Ridge constraint with Gaussian cross-section profile + absolute elevation
        anchors derived from observed 3D depth.

        points_xyz:    (N, 3) world (X, Y, Z). Y is the camera-relative elevation
                       of the ridge crest from panorama silhouette depth.
        anchor_stride: pin every nth chain point to avoid over-constraining noisy
                       depth estimates along the silhouette.
        """
        if len(points_xyz) < 2:
            return
        H, W = self._H, self._W
        x_half = z_far = grid_size_meters / 2.0

        col = np.clip((points_xyz[:, 0] + x_half) / grid_size_meters * (W - 1), 0, W - 1)
        row = np.clip((points_xyz[:, 2] + z_far)  / (2.0 * z_far)  * (H - 1), 0, H - 1)
        pts_rc = np.stack([row, col], axis=1)

        lr, cols, vals, rhs = self._perpendicular_profile_constraints(
            pts_rc, H, W, weight, crest_height, sigma, sign=+1
        )
        self._add_block(lr, cols, vals, rhs)

        anchor_idxs = np.arange(0, len(points_xyz), anchor_stride)
        if len(anchor_idxs) == 0:
            return
        r_anch = np.clip(np.round(row[anchor_idxs]).astype(int), 0, H - 1)
        c_anch = np.clip(np.round(col[anchor_idxs]).astype(int), 0, W - 1)
        elev   = points_xyz[anchor_idxs, 1].astype(np.float64)
        idxs   = r_anch * W + c_anch
        n      = len(idxs)
        w_arr  = np.full(n, float(anchor_weight))
        self._add_block(np.arange(n), idxs, w_arr, w_arr * elev)

    def add_river_polyline_anchored(
        self,
        points_xyz: np.ndarray,
        grid_size_meters: float,
        weight: float = 1.0,
        valley_depth: float = 0.5,
        sigma: float = 10.0,
        anchor_weight: float = 2.0,
        anchor_stride: int = 5,
    ) -> None:
        """
        River valley constraint with Gaussian cross-section + absolute water-surface
        elevation anchors derived from observed panorama depth.

        points_xyz:    (N, 3) world (X, Y, Z) ordered upstream → downstream. Y is
                       the camera-relative water surface elevation.
        anchor_weight: intentionally lower than ridge anchors — water surface depth
                       estimation is less reliable than sky-terrain silhouette depth.
        """
        if len(points_xyz) < 2:
            return
        H, W = self._H, self._W
        x_half = z_far = grid_size_meters / 2.0

        col = np.clip((points_xyz[:, 0] + x_half) / grid_size_meters * (W - 1), 0, W - 1)
        row = np.clip((points_xyz[:, 2] + z_far)  / (2.0 * z_far)  * (H - 1), 0, H - 1)
        pts_rc = np.stack([row, col], axis=1)

        lr, cols_arr, vals, rhs = self._perpendicular_profile_constraints(
            pts_rc, H, W, weight, valley_depth, sigma, sign=-1
        )
        self._add_block(lr, cols_arr, vals, rhs)

        anchor_idxs = np.arange(0, len(points_xyz), anchor_stride)
        if len(anchor_idxs) == 0:
            return
        r_anch = np.clip(np.round(row[anchor_idxs]).astype(int), 0, H - 1)
        c_anch = np.clip(np.round(col[anchor_idxs]).astype(int), 0, W - 1)
        elev   = points_xyz[anchor_idxs, 1].astype(np.float64)
        idxs   = r_anch * W + c_anch
        n      = len(idxs)
        w_arr  = np.full(n, float(anchor_weight))
        self._add_block(np.arange(n), idxs, w_arr, w_arr * elev)

    # ── Solve ─────────────────────────────────────────────────────────────────

    def solve(self) -> np.ndarray:
        """
        Build and solve the full sparse least-squares system.

        Data-preservation and Laplacian smoothness terms are assembled here (once),
        then concatenated with any previously added feature constraints.

        Returns:
            (H, W) float32 — the reconstructed heightmap.
        """
        self._build_data_term()
        self._build_laplacian_term()

        # Assemble global sparse matrix with contiguous row offsets
        row_offset = 0
        all_rows, all_cols, all_vals = [], [], []
        for local_rows, cols, vals, n in self._blocks:
            all_rows.append(local_rows + row_offset)
            all_cols.append(cols)
            all_vals.append(vals)
            row_offset += n

        all_rows_arr = np.concatenate(all_rows)
        all_cols_arr = np.concatenate(all_cols)
        all_vals_arr = np.concatenate(all_vals)
        b = np.concatenate(self._rhs_blocks)

        A = scipy.sparse.coo_matrix(
            (all_vals_arr, (all_rows_arr, all_cols_arr)),
            shape=(row_offset, self._N),
        ).tocsr()

        result = scipy.sparse.linalg.lsqr(A, b, iter_lim=self._iter_lim)
        x = result[0]
        return x.reshape(self._H, self._W).astype(np.float32)
