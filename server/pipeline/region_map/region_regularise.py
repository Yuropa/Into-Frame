"""Region regularisation: the world-space region map outside the photograph, made to
read like a map.

Outside the photographed cone the region map is the outpainted panorama projected
radially from the camera, so a class that holds a band of panorama columns becomes a
wedge running out to the grid's edge, and the map reads as a star (Paris: forest,
ground and built land in alternating spokes behind the camera). The evidence there is
weak and gets weaker with range -- a panorama pixel covers more ground the further
out it lands -- while inside the cone the map is what the photograph shows.

The steps are the ones game map generators use (Amit Patel's polygonal maps, cellular
automata cave smoothing, signed-distance boundary smoothing, domain warping):

1. Evidence. The map becomes per-class probabilities, smoothed ALONG THE ARC (the
   direction the radial projection smears) by an angle, so a wedge thins out where its
   projected footprint is coarse; confidence is 1 in the cone and near the camera and
   falls off with range outside it, and the smoothed field counts for the rest.
2. Cells. Blue-noise seeds (dart throwing, spacing growing with range: fine near, coarse
   far), Lloyd-relaxed twice, partition the ground into Voronoi cells; each takes its
   confidence-weighted majority class.
3. Cleanup. Cellular-automata majority rounds on the cell graph (a cell joins the class
   that surrounds it unless its own evidence is clearly stronger), then regions below a
   minimum area (growing with range) merge into their surroundings.
4. Boundaries. Per-class signed distance fields are blurred and re-thresholded (round,
   not polygonal), then a fractal-noise domain warp (amplitude growing with range) makes
   them read as natural.
5. Constraints. Nothing in the cone or near the camera changes, nor water (a river
   course or a coast set by the terrain plan), roads or trails; no free cell becomes one
   of those classes.

Works at a reduced resolution (cells of about 0.4 m) and writes back at the map's own,
keeping every constrained cell exactly."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


@dataclass
class RegulariseConfig:
    work_cell_m: float = 0.4            # resolution the step works at
    near_m: float = 10.0                # radius around the camera that is kept as is
    cone_margin_deg: float = 0.0        # widen the kept cone by this much each side
    confidence_falloff_m: float = 25.0  # outside the cone: confidence exp(-(r - near) / falloff)
    arc_sigma_rad: float = 0.15         # evidence smoothing along the arc (angle; Paris's spokes run 20-30 deg)
    arc_sigma_max_m: float = 25.0       # ... capped at this arc length
    seed_spacing_m: float = 1.5         # Voronoi seed spacing at the camera ...
    seed_spacing_per_m: float = 0.1     # ... growing by this per metre of range (20 m cells at 200 m)
    lloyd_iterations: int = 2
    ca_iterations: int = 3
    ca_share: float = 0.6               # surrounding share of one class that flips a cell
    ca_own_ratio: float = 1.5           # ... unless the cell's own class scores this much higher
    min_area_m2: float = 30.0           # smallest region kept near the camera ...
    min_area_per_m2: float = 0.02       # ... growing by this times range squared
    sdf_sigma_m: float = 2.5            # boundary rounding
    warp_m: float = 0.5                 # boundary noise amplitude at the camera ...
    warp_per_m: float = 0.02            # ... growing by this per metre of range
    warp_max_m: float = 5.0
    warp_scale_m: float = 18.0          # feature size of the warp noise
    # A second, broad warp bends the long straight edges that run out from the camera
    # (two classes either side of one panorama column: Shark Fin's beach and cliff top,
    # Rainier's meadow and forest), which no local rounding removes.
    bend_per_m: float = 0.15
    bend_max_m: float = 30.0
    bend_scale_m: float = 70.0
    fixed_classes: tuple = (1, 7, 8)    # WATER, ROAD, TRAIL: kept and never assigned
    linear_half_width_m: float = 4.0    # a strip no wider than twice this ...
    linear_min_area_m2: float = 60.0    # ... and at least this big is a path or a quay: kept
    linear_classes: tuple = (5, 6)      # BUILT, OTHER: the classes a path or a quay is called by
    seed: int = 0


def _polar_smooth(P: np.ndarray, cell: float, half: float, sigma_rad: float, sigma_max_m: float) -> np.ndarray:
    """Probabilities [K, N, N] smoothed along circles about the grid's centre: resample to
    (r, theta), Gaussian along theta (wrapping), resample back. The angular sigma is
    capped so far out the arc length stays at sigma_max_m."""
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    K, N, _ = P.shape
    R = half * math.sqrt(2.0)
    nr, nt = max(64, int(R / cell / 2)), 2048
    r = (np.arange(nr) + 0.5) * (R / nr)
    t = (np.arange(nt) + 0.5) * (2 * math.pi / nt) - math.pi
    rr, tt = np.meshgrid(r, t, indexing="ij")
    col = (rr * np.sin(tt) + half) / cell - 0.5          # x = r sin(theta), z = r cos(theta)
    row = (rr * np.cos(tt) + half) / cell - 0.5
    out = np.empty_like(P)
    # Rows of constant sigma: the angular sigma falls once the arc cap bites.
    sig_px = np.minimum(sigma_rad, sigma_max_m / np.maximum(r, 1e-6)) / (2 * math.pi / nt)
    bands = np.unique(np.round(sig_px, 0))
    pol = np.empty((K, nr, nt), dtype=np.float32)
    for k in range(K):
        pol[k] = map_coordinates(P[k], [row, col], order=1, mode="nearest")
    sm = np.empty_like(pol)
    for b in bands:
        rows = np.round(sig_px, 0) == b
        sm[:, rows] = gaussian_filter1d(pol[:, rows], max(float(b), 0.01), axis=2, mode="wrap")
    zz, xx = np.meshgrid((np.arange(N) + 0.5) * cell - half, (np.arange(N) + 0.5) * cell - half, indexing="ij")
    rc = np.hypot(xx, zz) / (R / nr) - 0.5
    tc = (np.arctan2(xx, zz) + math.pi) / (2 * math.pi / nt) - 0.5
    pad = np.concatenate([sm[:, :, -2:], sm, sm[:, :, :2]], axis=2)
    for k in range(K):
        out[k] = map_coordinates(pad[k], [rc, tc + 2], order=1, mode="nearest")
    return out


def _blue_noise(half: float, spacing: Callable[[np.ndarray], np.ndarray], rng, s_min: float) -> np.ndarray:
    """Dart throwing with a variable radius: candidates drawn with density ~ 1/s(r)^2, kept
    in random order unless an accepted seed lies within the candidate's own spacing (the
    spacing changes slowly with range, so its own is close to its neighbours')."""
    area = (2 * half) ** 2
    n_cand = min(int(6 * area / s_min ** 2), 600000)
    pts = rng.uniform(-half, half, size=(n_cand, 2))
    s = spacing(np.hypot(pts[:, 0], pts[:, 1]))
    keep = rng.uniform(size=n_cand) < (s_min / s) ** 2
    pts, s = pts[keep], s[keep]
    bucket = 4.0
    grid: dict = {}
    accepted = []
    for i in rng.permutation(len(pts)):
        x, z = pts[i]
        si2 = s[i] * s[i]
        reach = int(math.ceil(s[i] / bucket))
        bx, bz = int(math.floor(x / bucket)), int(math.floor(z / bucket))
        ok = True
        for dx in range(-reach, reach + 1):
            for dz in range(-reach, reach + 1):
                for (px, pz) in grid.get((bx + dx, bz + dz), ()):
                    if (px - x) ** 2 + (pz - z) ** 2 < si2:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if ok:
            grid.setdefault((bx, bz), []).append((x, z))
            accepted.append((x, z))
    return np.asarray(accepted, dtype=np.float64)


def _noise(N: int, cell: float, scale_m: float, rng) -> np.ndarray:
    """Fractal value noise in [-1, 1], three octaves."""
    from scipy.ndimage import gaussian_filter
    out = np.zeros((N, N), dtype=np.float32)
    amp, tot = 1.0, 0.0
    for o in range(3):
        sig = max(scale_m / (2 ** o) / cell, 1.0)
        n = gaussian_filter(rng.standard_normal((N, N)).astype(np.float32), sig, mode="wrap")
        n /= max(2.0 * float(n.std()), 1e-6)
        out += amp * n
        tot += amp
        amp *= 0.5
    return np.clip(out / tot, -1.0, 1.0)


def regularise(region_map: np.ndarray, grid_m: float, cone_half_deg: float,
               cfg: Optional[RegulariseConfig] = None, log: Callable[[str], None] = lambda s: None,
               pinned: Optional[np.ndarray] = None) -> tuple[np.ndarray, dict]:
    """The regularised map (same shape and dtype as `region_map`, class indices) and stats.
    `pinned` (bool, same shape) adds cells that must not change (a Scene Plan's pins)."""
    from scipy import ndimage
    from scipy.spatial import cKDTree

    cfg = cfg or RegulariseConfig()
    rng = np.random.default_rng(cfg.seed)
    full = np.rint(region_map).astype(np.int16)
    F = full.shape[0]
    half = grid_m / 2.0
    f = max(1, int(round(cfg.work_cell_m / (grid_m / F))))
    while F % f:
        f -= 1
    N = F // f
    cell = grid_m / N
    K = int(full.max()) + 1
    fixed_cls = np.array([c for c in cfg.fixed_classes if c < K], dtype=np.int64)

    def geometry(n: int):
        c = grid_m / n
        z, x = np.meshgrid((np.arange(n) + 0.5) * c - half, (np.arange(n) + 0.5) * c - half, indexing="ij")
        r = np.hypot(x, z)
        az = np.degrees(np.arctan2(x, z))
        return x, z, r, az

    def frozen_mask(cls: np.ndarray, r: np.ndarray, az: np.ndarray, pin=None) -> np.ndarray:
        m = (np.abs(az) <= cone_half_deg + cfg.cone_margin_deg) | (r <= cfg.near_m) | np.isin(cls, fixed_cls)
        return m | pin if pin is not None else m

    # ── full resolution: what may not change ──
    _, _, r_full, az_full = geometry(F)
    frozen_full = frozen_mask(full, r_full, az_full, pinned)
    del r_full, az_full
    # Thin strips of any class (a path the segmentation called "other", a quay's edge)
    # are linear features, not noise: the smoothing would erase them (Paris lost a
    # path running out behind the camera's right).
    coarse = full[::max(1, F // 1024), ::max(1, F // 1024)]
    cs = grid_m / coarse.shape[0]
    thin = np.zeros(coarse.shape, dtype=bool)
    rad = max(1, int(round(cfg.linear_half_width_m / cs)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    disk = (xx * xx + yy * yy) <= rad * rad
    for k in np.unique(coarse):
        if k in fixed_cls or k not in cfg.linear_classes:
            continue
        m = coarse == k
        # What an opening by a disk of the half width removes is the class's thin parts:
        # a path joined to a plaza is still a path.
        strip = m & ~ndimage.binary_opening(m, structure=disk)
        comp, n = ndimage.label(strip, structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        area = ndimage.sum(np.ones_like(comp), comp, idx) * cs * cs
        span = np.array([max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) for sl in ndimage.find_objects(comp)]) * cs
        keep = idx[(area >= cfg.linear_min_area_m2) & (span >= 8.0 * cfg.linear_half_width_m)]
        # A spoke of the projection is thin too, but it widens with range at a constant
        # angle; a path keeps its width in metres.
        if keep.size:
            zc, xc = (np.mgrid[0:coarse.shape[0], 0:coarse.shape[1]] + 0.5) * cs - half
            rr_c = np.hypot(xc, zc)
            width = ndimage.distance_transform_edt(m) * cs
            paths = []
            for c in keep:
                sel = comp == c
                rv, wv = rr_c[sel], width[sel]
                lo, hi = np.percentile(rv, [20, 80])
                if hi < 1.6 * max(lo, 1e-6):
                    paths.append(c)
                    continue
                w_lo = float(np.percentile(wv[rv <= lo], 90)) if (rv <= lo).any() else 0.0
                w_hi = float(np.percentile(wv[rv >= hi], 90)) if (rv >= hi).any() else 0.0
                if w_hi < 1.4 * max(w_lo, cs):
                    paths.append(c)
            if paths:
                thin |= np.isin(comp, paths)
    if thin.any():
        rep = F // coarse.shape[0]
        frozen_full |= np.repeat(np.repeat(thin, rep, axis=0), rep, axis=1)[:F, :F]
        log(f"Region regularisation: {int(thin.sum() * cs * cs)} m2 of thin strips (paths, edges) kept")

    # ── work resolution: class probabilities by area ──
    x, z, r, az = geometry(N)
    P = np.zeros((K, N, N), dtype=np.float32)
    blk = full.reshape(N, f, N, f)
    for k in range(K):
        P[k] = (blk == k).mean(axis=(1, 3))
    frozen_w = frozen_mask(np.argmax(P, axis=0), r, az) | (frozen_full.reshape(N, f, N, f).mean(axis=(1, 3)) > 0.5)
    C0 = np.argmax(P, axis=0)
    allowed = np.ones(K, dtype=bool)
    allowed[fixed_cls] = False
    present = np.array([P[k].sum() > 0 for k in range(K)])
    allowed &= present

    # 1. evidence
    conf = np.where(frozen_w, 1.0, np.exp(-np.maximum(r - cfg.near_m, 0.0) / cfg.confidence_falloff_m)).astype(np.float32)
    Ps = _polar_smooth(P, cell, half, cfg.arc_sigma_rad, cfg.arc_sigma_max_m)
    Pe = conf[None] * P + (1.0 - conf[None]) * Ps
    Pe[~allowed] = 0.0

    # 2. cells
    spacing = lambda rad: cfg.seed_spacing_m + cfg.seed_spacing_per_m * rad
    seeds = _blue_noise(half, spacing, rng, cfg.seed_spacing_m)
    pix = np.stack([x.ravel(), z.ravel()], axis=1)
    for it in range(cfg.lloyd_iterations + 1):
        _, lab = cKDTree(seeds).query(pix, workers=-1)
        cnt = np.bincount(lab, minlength=len(seeds)).astype(np.float64)
        if it < cfg.lloyd_iterations:
            sx = np.bincount(lab, weights=pix[:, 0], minlength=len(seeds))
            sz = np.bincount(lab, weights=pix[:, 1], minlength=len(seeds))
            okc = cnt > 0
            seeds = np.stack([sx[okc] / cnt[okc], sz[okc] / cnt[okc]], axis=1)
    lab = lab.reshape(N, N)
    ncell = int(lab.max()) + 1
    free = ~frozen_w
    score = np.zeros((ncell, K), dtype=np.float64)
    w = (conf * 0 + 1.0).ravel()
    for k in range(K):
        if allowed[k]:
            score[:, k] = np.bincount(lab.ravel(), weights=(Pe[k].ravel() * w), minlength=ncell)
    cell_cls = np.argmax(score, axis=1)
    C = np.where(free, cell_cls[lab], C0)

    # 3a. cellular automata on the cell graph
    for _ in range(cfg.ca_iterations):
        nb = np.zeros((ncell, K), dtype=np.float64)
        for axis in (0, 1):
            for shift in (1, -1):
                lab2 = np.roll(lab, shift, axis=axis)
                C2 = np.roll(C, shift, axis=axis)
                edge = (lab2 != lab) & free
                if axis == 0:
                    edge[0 if shift == 1 else -1, :] = False
                else:
                    edge[:, 0 if shift == 1 else -1] = False
                np.add.at(nb, (lab[edge], C2[edge]), 1.0)
        tot = nb.sum(axis=1)
        top = np.argmax(nb, axis=1)
        share = nb[np.arange(ncell), top] / np.maximum(tot, 1e-9)
        own = score[np.arange(ncell), cell_cls]
        theirs = score[np.arange(ncell), top]
        flip = (top != cell_cls) & (share >= cfg.ca_share) & allowed[top] & (own < cfg.ca_own_ratio * np.maximum(theirs, 1e-9))
        if not flip.any():
            break
        cell_cls = np.where(flip, top, cell_cls)
        C = np.where(free, cell_cls[lab], C0)

    # 3b. minimum region area, growing with range
    for k in range(K):
        if not allowed[k]:
            continue
        comp, n = ndimage.label(C == k, structure=np.ones((3, 3)))
        if n == 0:
            continue
        idx = np.arange(1, n + 1)
        area = ndimage.sum(np.ones_like(comp), comp, idx) * cell * cell
        rc = ndimage.mean(r, comp, idx)
        touches_frozen = ndimage.maximum(frozen_w.astype(np.int8), comp, idx) > 0
        small = idx[(area < cfg.min_area_m2 + cfg.min_area_per_m2 * rc ** 2) & ~touches_frozen]
        for c in small:
            m = comp == c
            ring = ndimage.binary_dilation(m, iterations=1) & ~m
            vals = C[ring]
            vals = vals[allowed[vals]]
            if vals.size:
                C[m] = np.bincount(vals, minlength=K).argmax()

    # 4a. signed distance fields, blurred and re-thresholded
    sig = cfg.sdf_sigma_m / cell
    best, best_v = C.copy(), np.full(C.shape, -np.inf, dtype=np.float32)
    for k in range(K):
        if not allowed[k] or not (C == k).any():
            continue
        inside = C == k
        sdf = (ndimage.distance_transform_edt(inside) - ndimage.distance_transform_edt(~inside)).astype(np.float32)
        sdf = ndimage.gaussian_filter(sdf, sig)
        better = sdf > best_v
        best[better], best_v[better] = k, sdf[better]
    C = np.where(free, best, C0)

    # 4b. domain warp
    amp = np.minimum(cfg.warp_m + cfg.warp_per_m * r, cfg.warp_max_m) / cell
    bend = np.minimum(cfg.bend_per_m * np.maximum(r - cfg.near_m, 0.0), cfg.bend_max_m) / cell
    wx = _noise(N, cell, cfg.warp_scale_m, rng) * amp + _noise(N, cell, cfg.bend_scale_m, rng) * bend
    wz = _noise(N, cell, cfg.warp_scale_m, rng) * amp + _noise(N, cell, cfg.bend_scale_m, rng) * bend
    rows = np.clip(np.rint(np.arange(N)[:, None] + wz), 0, N - 1).astype(np.int64)
    cols = np.clip(np.rint(np.arange(N)[None, :] + wx), 0, N - 1).astype(np.int64)
    Cw = C[rows, cols]
    C = np.where(free & allowed[Cw], Cw, C)

    # ── back to full resolution, constrained cells exactly as they were ──
    up = np.repeat(np.repeat(C, f, axis=0), f, axis=1)
    bad = ~frozen_full & ~allowed[np.clip(up, 0, K - 1)]
    if bad.any():
        # A free cell in a block the work map called water/road: nearest free class.
        ok = ~frozen_full & ~bad
        _, (ri, ci) = ndimage.distance_transform_edt(~ok, return_indices=True)
        up = np.where(bad, up[ri, ci], up)
    out = np.where(frozen_full, full, up).astype(region_map.dtype)

    def regions(m):
        return {int(k): int(ndimage.label(m == k, structure=np.ones((3, 3)))[1]) for k in np.unique(m)}
    sub = slice(None, None, f)
    _, _, r_s, az_s = geometry(N)
    cone_s = np.abs(az_s) <= cone_half_deg
    changed = full[sub, sub] != out[sub, sub]
    stats = {"cells": ncell, "work_cell_m": round(cell, 3),
             "regions_before": regions(full[sub, sub]), "regions_after": regions(out[sub, sub]),
             "changed_in_cone_pct": round(100.0 * float(changed[cone_s].mean()), 3),
             "changed_outside_cone_pct": round(100.0 * float(changed[~cone_s].mean()), 2)}
    log(f"Region regularisation: {ncell} cells (blue noise, {cfg.lloyd_iterations} Lloyd), "
        f"{stats['changed_outside_cone_pct']}% of the map outside the cone changed, "
        f"{stats['changed_in_cone_pct']}% inside; regions {sum(stats['regions_before'].values())} -> "
        f"{sum(stats['regions_after'].values())}")
    return out, stats
