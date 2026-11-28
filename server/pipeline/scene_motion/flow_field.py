"""Where the water goes: a flow field over the scene's water sheet.

A river runs between its banks from where it enters the terrain to where it leaves
it, faster where it narrows and still at the banks; a boat underway follows it and
the water's surface texture is carried along it. The field is the gradient of the
potential that is 0 at one open end of the channel and 1 at the other (Laplace's
equation over the water, no flow through the banks), so it follows the river's
curves without being told where they are. Which end is upstream, and how fast the
water runs, are not in the geometry: they come from the video's measured flow or
the Motion Director's prior (sign and speed_m_s).

Water that leaves the terrain along one stretch of its edge only (a cove open to
the sea, a lake) has no through-flow: its field points toward the shore, the way
the swell runs, and `kind` says "sea".

Grid in the server's frame: x right, z forward, cell `cell_m` metres; vectors are
unit directions times a speed in m/s."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FlowField:
    origin: tuple[float, float]          # world (x, z) of cell (0, 0)'s corner
    cell_m: float
    water: np.ndarray                    # bool [nz, nx]
    vx: np.ndarray                       # float32 [nz, nx], m/s
    vz: np.ndarray
    kind: str = "river"                  # "river" | "sea" | "none"
    speed_m_s: float = 0.0               # typical mid-channel speed the field is scaled to
    ends: list = field(default_factory=list)  # [(x, z) of the inflow end, (x, z) of the outflow end]

    # ── sampling ────────────────────────────────────────────────────────────

    def _cell(self, x, z):
        return ((np.asarray(z, dtype=float) - self.origin[1]) / self.cell_m - 0.5,
                (np.asarray(x, dtype=float) - self.origin[0]) / self.cell_m - 0.5)

    def sample(self, x, z) -> np.ndarray:
        """Bilinear (vx, vz) at world points; 0 off the water."""
        from scipy.ndimage import map_coordinates
        r, c = self._cell(x, z)
        coords = np.vstack([np.atleast_1d(r), np.atleast_1d(c)])
        vx = map_coordinates(self.vx, coords, order=1, mode="constant", cval=0.0)
        vz = map_coordinates(self.vz, coords, order=1, mode="constant", cval=0.0)
        return np.stack([vx, vz], axis=-1)

    def on_water(self, x, z) -> np.ndarray:
        r, c = self._cell(x, z)
        r = np.clip(np.rint(np.atleast_1d(r)).astype(int), 0, self.water.shape[0] - 1)
        c = np.clip(np.rint(np.atleast_1d(c)).astype(int), 0, self.water.shape[1] - 1)
        return self.water[r, c]

    def streamline(self, x: float, z: float, length_m: float, step_m: float = 1.0, sign: float = 1.0) -> np.ndarray:
        """Points (x, z) along the flow from (x, z) for `length_m` (against it for sign -1),
        midpoint steps of `step_m`; stops where the water ends or the flow dies."""
        pts = [(float(x), float(z))]
        p = np.array([x, z], dtype=float)
        travelled = 0.0
        while travelled < length_m:
            v = self.sample(*p)[0] * sign
            n = float(np.hypot(*v))
            if n < 1e-6:
                break
            mid = p + 0.5 * step_m * v / n
            v2 = self.sample(*mid)[0] * sign
            n2 = float(np.hypot(*v2))
            if n2 < 1e-6:
                break
            q = p + step_m * v2 / n2
            if not self.on_water(*q)[0]:
                break
            p = q
            travelled += step_m
            pts.append((float(p[0]), float(p[1])))
        return np.asarray(pts)

    # ── persistence ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        np.savez_compressed(path, origin=np.asarray(self.origin), cell_m=self.cell_m, water=self.water,
                            vx=self.vx, vz=self.vz, kind=self.kind, speed_m_s=self.speed_m_s,
                            ends=np.asarray(self.ends, dtype=float).reshape(-1, 2))

    @classmethod
    def load(cls, path: Path) -> "FlowField":
        d = np.load(path, allow_pickle=False)
        return cls(origin=tuple(float(v) for v in d["origin"]), cell_m=float(d["cell_m"]), water=d["water"].astype(bool),
                   vx=d["vx"], vz=d["vz"], kind=str(d["kind"]), speed_m_s=float(d["speed_m_s"]),
                   ends=[tuple(e) for e in d["ends"].tolist()])

    def scaled(self, speed_m_s: float, sign: float = 1.0) -> "FlowField":
        """The same field run at another mid-channel speed, or the other way."""
        k = (speed_m_s / self.speed_m_s if self.speed_m_s > 0 else 0.0) * sign
        ends = list(self.ends) if sign > 0 else list(reversed(self.ends))
        return FlowField(self.origin, self.cell_m, self.water, (self.vx * k).astype(np.float32),
                         (self.vz * k).astype(np.float32), self.kind, abs(speed_m_s), ends)


# ── building ────────────────────────────────────────────────────────────────

def rasterise_water(vertices: np.ndarray, faces: np.ndarray, half_extent_m: float, cell_m: float,
                    dry: Optional[np.ndarray] = None) -> tuple[np.ndarray, tuple[float, float]]:
    """The water sheet's triangles, seen from above, on a grid of `cell_m` cells over
    +-half_extent_m. `dry` (bool, same grid) takes cells out: ground standing above the
    water (a sea stack, a pier) that the sheet runs under."""
    from PIL import Image, ImageDraw
    n = int(round(2.0 * half_extent_m / cell_m))
    origin = (-half_extent_m, -half_extent_m)
    img = Image.new("L", (n, n), 0)
    draw = ImageDraw.Draw(img)
    uv = (np.asarray(vertices, dtype=float)[:, [0, 2]] - np.asarray(origin)) / cell_m
    for f in np.asarray(faces):
        draw.polygon([tuple(uv[i]) for i in f], fill=255)
    water = np.asarray(img) > 127          # rows = z, cols = x
    if dry is not None:
        water &= ~dry
    return water, origin


def _open_ends(water: np.ndarray, margin: int) -> list[np.ndarray]:
    """Where the water leaves the grid: water cells within `margin` cells of its edge
    (the sheet stops a few metres short of the terrain's), grouped into connected
    stretches, largest first; each as an array of (row, col)."""
    from scipy import ndimage
    nz, nx = water.shape
    band = np.zeros_like(water)
    band[:margin, :] = band[-margin:, :] = True
    band[:, :margin] = band[:, -margin:] = True
    lab, n = ndimage.label(water & band, structure=np.ones((3, 3)))
    ends = [np.argwhere(lab == k) for k in range(1, n + 1)]
    ends = [e for e in ends if len(e) >= 3 * margin]
    return sorted(ends, key=len, reverse=True)


def _ring_position(rc: np.ndarray, shape: tuple[int, int]) -> float:
    """Where a cell lies along the grid's perimeter, 0..1 clockwise from the corner (0, 0)."""
    nz, nx = shape
    r, c = float(rc[0]), float(rc[1])
    d = np.array([r, nx - 1 - c, nz - 1 - r, c])     # distance to the top, right, bottom, left edges
    side = int(np.argmin(d))
    per = 2.0 * (nx + nz)
    along = [c, nx + r, nx + nz + (nx - 1 - c), 2 * nx + nz + (nz - 1 - r)][side]
    return along / per


def build_flow_field(vertices: np.ndarray, faces: np.ndarray, half_extent_m: float = 200.0, cell_m: float = 1.0,
                     speed_m_s: float = 0.6, dry: Optional[np.ndarray] = None, bank_m: float = 3.0) -> FlowField:
    """The flow field of the water sheet (vertices in world metres). River: the potential
    flow between its two furthest-apart open ends, scaled so the median speed over the
    channel's middle is `speed_m_s`, and eased to zero over `bank_m` at the banks. The
    first end is called the inflow; flip it with `scaled(.., sign=-1)`."""
    from scipy import ndimage, sparse
    from scipy.sparse.linalg import spsolve

    water, origin = rasterise_water(vertices, faces, half_extent_m, cell_m, dry)
    nz, nx = water.shape
    empty = np.zeros_like(water, dtype=np.float32)
    if not water.any():
        return FlowField(origin, cell_m, water, empty, empty.copy(), "none", 0.0)
    # Keep the largest connected body of water.
    lab, n = ndimage.label(water)
    if n > 1:
        sizes = ndimage.sum(water, lab, range(1, n + 1))
        water = lab == (1 + int(np.argmax(sizes)))
    margin = max(2, int(round(6.0 / cell_m)))
    runs = _open_ends(water, margin)
    centres = [r.mean(axis=0) for r in runs]
    to_world = lambda rc: (origin[0] + (rc[1] + 0.5) * cell_m, origin[1] + (rc[0] + 0.5) * cell_m)
    shore_d = ndimage.distance_transform_edt(water) * cell_m

    pair = None
    if len(runs) >= 2:
        best = 0.0
        for i in range(len(runs)):
            for j in range(i + 1, len(runs)):
                d = float(np.hypot(*(centres[i] - centres[j])))
                # An end narrower than a tenth of the other is a side channel, not the river.
                if d > best and min(len(runs[i]), len(runs[j])) >= 0.1 * max(len(runs[i]), len(runs[j])):
                    best, pair = d, (i, j)
        if pair is not None and best * cell_m < 0.25 * half_extent_m:
            pair = None
        if pair is not None:
            # Two open stretches joined by a short run of the edge are one open coast
            # (Shark Fin's cove, open to the sea round the grid's corner), not a
            # river's two ends: a river's ends lie apart along the edge both ways.
            u = sorted(_ring_position(runs[i][np.argmin(np.abs(np.arange(len(runs[i])) - len(runs[i]) // 2))], water.shape)
                       for i in pair)
            gap = min(u[1] - u[0], 1.0 - (u[1] - u[0]))
            extent = sum(float(np.ptp(runs[i], axis=0).max()) for i in pair) / (2.0 * sum(water.shape))
            if gap < 0.15 + extent:
                pair = None

    if pair is None:
        # Open to the sea along one stretch (or closed): the swell runs to the shore.
        land = ~water
        if not land.any():
            return FlowField(origin, cell_m, water, empty, empty.copy(), "sea", 0.0)
        gz, gx = np.gradient(ndimage.gaussian_filter(shore_d, 4.0 / cell_m))
        norm = np.hypot(gx, gz) + 1e-9
        vx = np.where(water, -gx / norm, 0.0).astype(np.float32)
        vz = np.where(water, -gz / norm, 0.0).astype(np.float32)
        return FlowField(origin, cell_m, water, vx, vz, "sea", 1.0,
                         [to_world(centres[0])] if centres else [])

    a, b = runs[pair[0]], runs[pair[1]]
    idx = -np.ones(water.shape, dtype=np.int64)
    cells = np.argwhere(water)
    idx[water] = np.arange(len(cells))
    fixed = np.zeros(len(cells), dtype=bool); value = np.zeros(len(cells))
    fixed[idx[a[:, 0], a[:, 1]]] = True
    fixed[idx[b[:, 0], b[:, 1]]] = True; value[idx[b[:, 0], b[:, 1]]] = 1.0
    rows, cols, vals = [], [], []
    rhs = np.zeros(len(cells))
    for k, (r, c) in enumerate(cells):
        if fixed[k]:
            rows.append(k); cols.append(k); vals.append(1.0); rhs[k] = value[k]
            continue
        deg = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < nz and 0 <= cc < nx and water[rr, cc]:
                rows.append(k); cols.append(idx[rr, cc]); vals.append(-1.0); deg += 1
        rows.append(k); cols.append(k); vals.append(float(max(deg, 1)))
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(len(cells), len(cells)))
    phi_v = spsolve(A, rhs)
    phi = np.full(water.shape, np.nan); phi[water] = phi_v
    # Gradient inside the water only (a bank cell's missing neighbour is the cell itself).
    filled = np.where(water, phi, 0.0)
    def d(axis):
        fwd = np.roll(filled, -1, axis); bwd = np.roll(filled, 1, axis)
        wf = np.roll(water, -1, axis); wb = np.roll(water, 1, axis)
        fwd = np.where(wf, fwd, filled); bwd = np.where(wb, bwd, filled)
        span = np.where(wf, 1.0, 0.0) + np.where(wb, 1.0, 0.0)
        return np.where(span > 0, (fwd - bwd) / np.maximum(span, 1.0), 0.0)
    gz, gx = d(0), d(1)
    gz = ndimage.gaussian_filter(gz, 1.0); gx = ndimage.gaussian_filter(gx, 1.0)
    mag = np.hypot(gx, gz)
    ease = 1.0 - np.exp(-shore_d / max(bank_m, 1e-6))
    vx, vz = gx * ease, gz * ease
    mid = water & (shore_d >= np.percentile(shore_d[water], 60))
    ref = float(np.median(np.hypot(vx, vz)[mid])) if mid.any() else float(mag[water].max())
    k = speed_m_s / max(ref, 1e-12)
    vx, vz = vx * k, vz * k
    # Potential flow races round a sharp corner of the outline (19 m/s at a cell of
    # Paris's quay where the channel runs at 0.4): no river does.
    sp = np.hypot(vx, vz)
    cap = np.minimum(1.0, 2.5 * speed_m_s / np.maximum(sp, 1e-12))
    vx = np.where(water, vx * cap, 0.0).astype(np.float32)
    vz = np.where(water, vz * cap, 0.0).astype(np.float32)
    return FlowField(origin, cell_m, water, vx, vz, "river", float(speed_m_s),
                     [to_world(centres[pair[0]]), to_world(centres[pair[1]])])
