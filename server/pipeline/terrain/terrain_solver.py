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
        iter_lim: int = 500,
    ):
        """
        Args:
            heightmap:        (H, W) float32 — existing DEM values.
            confidence:       (H, W) float32 — per-pixel confidence in [0, 1].
                              1.0 = do not move; 0.0 = free to change.
            laplacian_weight: weight for the smoothness term.
            iter_lim:         maximum LSQR iterations.
        """
        self._hm = heightmap.astype(np.float64)
        self._conf = np.clip(confidence, 0.0, 1.0).astype(np.float64)
        self._H, self._W = heightmap.shape
        self._N = self._H * self._W
        self._lap_w = laplacian_weight
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
        # Floor weight prevents rank deficiency at zero-confidence pixels
        w = np.maximum(conf_flat, 1e-4)
        n = self._N
        self._add_block(np.arange(n), np.arange(n), w, w * orig_flat)

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

    # ── Shared helper: neighbour constraints from a binary mask ───────────────

    @staticmethod
    def _neighbour_constraints(
        mask: np.ndarray,
        H: int,
        W: int,
        influence: np.ndarray,
        weight: float,
        target_diff: float,
        sign: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        For each pixel in *mask* and each of its 4-connected neighbours inside the
        grid, emit one constraint row:

          sign > 0  →  w * h_mask   - w * h_neigh = w * target_diff   (ridge)
          sign < 0  →  w * h_neigh  - w * h_mask  = w * target_diff   (river)

        Returns (local_rows, cols, vals, rhs) — all flat arrays ready for _add_block.
        """
        ry, rx = np.where(mask)
        if len(ry) == 0:
            return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float64),) * 2

        dy = np.array([-1, 1, 0, 0])
        dx = np.array([0, 0, -1, 1])

        mask_idxs, neigh_idxs, ws = [], [], []
        for d in range(4):
            ny = ry + dy[d]
            nx = rx + dx[d]
            valid = (ny >= 0) & (ny < H) & (nx >= 0) & (nx < W)
            if not valid.any():
                continue
            ny_v, nx_v = ny[valid], nx[valid]
            ry_v, rx_v = ry[valid], rx[valid]
            w = weight * influence[ny_v, nx_v]
            mask_idxs.append(ry_v * W + rx_v)
            neigh_idxs.append(ny_v * W + nx_v)
            ws.append(w)

        if not mask_idxs:
            return (np.empty(0, np.int64),) * 2 + (np.empty(0, np.float64),) * 2

        mi = np.concatenate(mask_idxs)
        ni = np.concatenate(neigh_idxs)
        w_arr = np.concatenate(ws)
        n = len(mi)

        if sign > 0:
            # h_mask - h_neigh = target_diff
            first_col, second_col = mi, ni
        else:
            # h_neigh - h_mask = target_diff
            first_col, second_col = ni, mi

        local_rows = np.repeat(np.arange(n), 2)
        cols = np.stack([first_col, second_col], axis=1).ravel()
        vals = np.column_stack([w_arr, -w_arr]).ravel()
        rhs  = w_arr * target_diff
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
        if len(points) < 1:
            return
        H, W = self._H, self._W
        mask = np.zeros((H, W), dtype=bool)
        rows = np.clip(np.round(points[:, 0]).astype(int), 0, H - 1)
        cols = np.clip(np.round(points[:, 1]).astype(int), 0, W - 1)
        mask[rows, cols] = True
        self.add_ridge_mask(mask, weight=weight, crest_height=crest_height, sigma=sigma)

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
        ridge_dist = distance_transform_edt(~mask)
        influence = np.exp(-ridge_dist ** 2 / (2.0 * sigma ** 2))
        lr, cols, vals, rhs = self._neighbour_constraints(
            mask, H, W, influence, weight, crest_height, sign=+1
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
        if len(points) < 1:
            return
        H, W = self._H, self._W

        pt_rows = np.clip(np.round(points[:, 0]).astype(int), 0, H - 1)
        pt_cols = np.clip(np.round(points[:, 1]).astype(int), 0, W - 1)

        # Valley cross-section constraints via mask
        mask = np.zeros((H, W), dtype=bool)
        mask[pt_rows, pt_cols] = True
        self.add_river_mask(mask, weight=weight, valley_depth=valley_depth, sigma=sigma)

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
        river_dist = distance_transform_edt(~mask)
        influence = np.exp(-river_dist ** 2 / (2.0 * sigma ** 2))
        lr, cols, vals, rhs = self._neighbour_constraints(
            mask, H, W, influence, weight, valley_depth, sign=-1
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
