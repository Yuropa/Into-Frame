"""Bands from the scene plan (docs/hybrid-scene-plan.md, "bands, not scatter").

A region the plan and the region map agree on -- a forest -- is a band of one kind
of thing, not hundreds of independent detections. The panorama's depth compresses
the far field (Rainier's treeline, 200-400 m off, landed at 7-27 m as 2 m saplings
and the painter copied it over the meadow), so the band is placed from the region
map's own shape instead.

The mask's top edge is the photographed skyline. Its local maxima are crown tips:
each tip gets a tree standing where that column's band foot meets the terrain,
exactly as tall as the tip needs it to be there, so the photograph's own crowns --
the lone spire at the left edge, the jagged treeline -- come back where they were.
Between the tips the band is filled in rows that stand behind the foot, up the
slope, with tops under the skyline; the fill is thinned by a slow random field
along the band so it reads as clumps and gaps, not a picket fence. A mask
component too narrow to be a stand (one big fir) gets its tip and no fill. Each
tree is an ordinary synthetic instance for Scene Generation's placement loop:
snapped to the ground, overlap-tested, dressed with the class's bucket assets.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

GroundHit = Callable[[float, float], Optional[tuple[float, float, float]]]
"""(theta_rad, elevation_rad) -> (world_x, world_z, horizontal distance) where the
ray from the camera meets the terrain, or None when it misses."""

GroundY = Callable[[float, float], Optional[float]]
"""(world_x, world_z) -> the terrain's height there relative to the camera, or
None off the terrain."""


def _median_columns(a: np.ndarray, smooth_px: int) -> np.ndarray:
    """Median over a window of columns, ignoring NaN."""
    if smooth_px <= 1:
        return a.copy()
    pad = smooth_px // 2
    out = a.copy()
    for i in range(len(a)):
        win = a[max(0, i - pad):i + pad + 1]
        win = win[np.isfinite(win)]
        out[i] = np.median(win) if win.size else np.nan
    return out


def _column_edges(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per column: top row, bottom row and the count of band pixels (NaN/0 where
    the column has none)."""
    h, w = mask.shape
    rows = np.arange(h)[:, None]
    has = mask.any(axis=0)
    top = np.where(has, np.where(mask, rows, h).min(axis=0), np.nan).astype(float)
    bot = np.where(has, np.where(mask, rows, -1).max(axis=0), np.nan).astype(float)
    count = mask.sum(axis=0).astype(float)
    return top, bot, count


def _clump_field(w: int, period_px: float, floor: float, rng: np.random.Generator) -> np.ndarray:
    """A slow random field along the columns in [floor, 1]: the probability that a
    fill tree at a column is kept. Low-pass noise with the given period, ranked to
    a uniform spread so the band is about half dense, with real gaps."""
    from scipy.ndimage import gaussian_filter1d
    noise = rng.standard_normal(w)
    sigma = max(1.0, period_px / 2.5)
    smooth = gaussian_filter1d(noise, sigma, mode="wrap")
    ranks = np.argsort(np.argsort(smooth)) / max(w - 1, 1)
    return floor + (1.0 - floor) * ranks ** 1.5


def band_instances(
    region_map: np.ndarray,
    region_value: int,
    ground_hit: GroundHit,
    size_m: float,
    size_band: tuple[float, float],
    far_m: float,
    cls: str,
    bucket: Optional[int],
    rows: int = 6,
    step_fraction: float = 0.3,
    min_fill: float = 0.35,
    near_fraction: float = 0.0,
    camera_xz: tuple[float, float] = (0.0, 0.0),
    ground_y: Optional[GroundY] = None,
    crown_fraction: float = 0.55,
    min_separation: float = 0.3,
    tip_prominence_deg: float = 0.25,
    clump_period_crowns: float = 5.0,
    clump_floor: float = 0.12,
    foot_fraction: float = 0.3,
    seed: int = 0,
    depth: Optional[np.ndarray] = None,
    ground_at: Optional[Callable[[float, float], tuple[float, float]]] = None,
) -> tuple[list[dict], dict]:
    """Synthetic instances following the region's shape in the region map.

    Returns (instances, summary). Tips of the mask's top edge become trees as tall
    as the skyline needs where the column's band foot meets the terrain; the band
    between them is filled with rows behind the foot, tops under the skyline,
    thinned into clumps. A column whose band is less than min_fill solid (a crown
    seen through gaps) gets no fill.

    depth (the calibrated panorama depth, the region map's shape) and ground_at
    (theta, distance -> world x, z) give a column whose rays all miss the ground
    a foot anyway: a stand on a slope behind a rise in the meadow shows no foot,
    its lowest pixel is above the horizon, and no ray from its band lands
    (Rainier band18: the forest at the right edge, 150-190 m by the depth map,
    151 of 401 columns off the terrain and the gap under it sky)."""
    from scipy import ndimage, signal

    mask = np.asarray(region_map) == int(region_value)
    h, w = mask.shape
    top_raw, bot_raw, count = _column_edges(mask)
    wide = max(3, w // 170)
    top_s = _median_columns(top_raw, max(3, w // 1024))      # the skyline, lightly cleaned
    top_wide = _median_columns(top_raw, wide)                 # the canopy's typical top
    bot = _median_columns(bot_raw, wide)                      # the band's foot
    lo, hi = float(size_band[0]), float(size_band[1])
    rng = np.random.default_rng((seed, 11))
    near_floor = near_fraction * far_m
    col_step = max(1, w // 720)

    # Which crown owns each column's skyline, and whether it is a stand or one
    # tree: a component narrower than 2.5x its angular height is a lone crown.
    labels, _n = ndimage.label(mask)
    top_label = np.zeros(w, dtype=int)
    for c in range(w):
        if np.isfinite(top_raw[c]):
            top_label[c] = labels[int(top_raw[c]), c]
    comp_cols: dict[int, list[int]] = {}
    for c in range(w):
        if top_label[c]:
            comp_cols.setdefault(int(top_label[c]), []).append(c)
    stand = np.zeros(w, dtype=bool)
    # ...or, with the depth map, wider than two of the plan's crowns at the
    # component's measured distance: a stand on a slope behind a rise shows a
    # blob of crowns little wider than tall (Rainier band18: the forest at the
    # right edge, 7 deg wide at 190-220 m, which one tree cannot be).
    have_depth = depth is not None and depth.shape == (h, w)
    comp_dist: dict[int, float] = {}
    if have_depth and comp_cols:
        labs = list(comp_cols.keys())
        finite = np.where(np.isfinite(depth), depth, np.nan)
        med = ndimage.median(finite, labels=labels, index=labs)
        comp_dist = {lab: float(m) for lab, m in zip(labs, np.atleast_1d(med))}
    for lab, cols in comp_cols.items():
        rows_of = np.where(labels == lab)[0]
        ang_h = (rows_of.max() - rows_of.min() + 1) / h * math.pi
        ang_w = (max(cols) - min(cols) + 1) / w * 2.0 * math.pi
        if ang_w >= 2.5 * ang_h:
            stand[cols] = True
        elif comp_dist.get(lab) is not None and np.isfinite(comp_dist[lab]):
            crown_ang = 0.5 * size_m / max(min(comp_dist[lab], far_m), 1.0)
            if ang_w >= 2.0 * crown_ang:
                stand[cols] = True

    def elevation(row: float) -> float:
        return (0.5 - row / h) * math.pi

    def foot(col: int):
        """(x, z, dist, el_bot, beyond) for a column: the lowest band row whose ray
        lands on far ground, capped at far_m. Forest pixels below that are
        mid-ground shrubs on the meadow (Rainier: the mask runs down to -4 deg
        and that ray lands 13 m out), not the treeline. Far ground for a stand is
        at least foot_fraction of the distance the band's angle implies for the
        plan's tree size (a treeline of 18 m firs seen 5 deg tall is not 13 m
        away); a lone crown stands where its own base lands."""
        el_t, el_b = elevation(top_s[col]), elevation(bot[col])
        theta = (col / w - 0.5) * 2.0 * math.pi
        ang = math.tan(min(max(el_t, -1.2), 1.2)) - math.tan(min(max(el_b, -1.2), 1.2))
        min_dist = near_floor
        if stand[col] and ang > 1e-4:
            min_dist = max(min_dist, foot_fraction * size_m / ang)
        n_probe = 12
        for k in range(n_probe + 1):
            el = el_b + (el_t - el_b) * (k / n_probe)
            probe = ground_hit(theta, el)
            if probe is not None and probe[2] >= min_dist:
                x, z, dist = probe
                beyond = dist > far_m
                if beyond:
                    x = camera_xz[0] + (x - camera_xz[0]) * far_m / dist
                    z = camera_xz[1] + (z - camera_xz[1]) * far_m / dist
                    dist = far_m
                return x, z, dist, el, beyond
        if have_depth and ground_at is not None:
            # No ray lands: the band's foot is hidden. Its lowest pixel's own
            # measured distance stands in, capped at far_m like a landed ray.
            # The stand's distance floor is a guard against a ray landing on the
            # meadow in front; a measured distance only has to be most of it.
            rb = int(min(max(round(float(bot[col])), 0), h - 1))
            d = float(depth[rb, col])
            if np.isfinite(d) and d >= max(1.0, near_floor, 0.5 * min_dist):
                beyond = d > far_m
                d = min(d, far_m)
                x, z = ground_at(theta, d)
                stats["depth_feet"] += 1
                return x, z, d, el_b, beyond
        return None

    def height_floor(dist: float, el_t: float, el_b: float) -> float:
        """The plan's band floor applies only where the plan size could be true
        (at or beyond the distance the band's angle implies); nearer, the band is
        a model of a far treeline and a full-size tree would cover the mountain."""
        ang = math.tan(min(max(el_t, -1.2), 1.2)) - math.tan(min(max(el_b, -1.2), 1.2))
        implied = size_m / max(ang, 1e-4)
        return (lo if dist >= 0.5 * implied else min(lo, max(0.5, 0.08 * size_m))), implied

    def tree_height(dist: float, x: float, z: float, el_t: float) -> Optional[tuple[float, float]]:
        """(height that reaches the skyline at el_t standing on the ground at (x, z), ground y)."""
        y_g = ground_y(x, z) if ground_y is not None else None
        if y_g is None:
            y_g = 0.0
        return dist * math.tan(min(max(el_t, -1.2), 1.2)) - y_g, y_g

    instances: list[dict] = []
    stats = {"columns": 0, "thin": 0, "missed": 0, "beyond": 0, "tips": 0, "fill": 0, "lone": 0,
             "spans": 0, "depth_feet": 0, "rows": [], "distance_m": [], "height_m": [], "implied_m": []}

    def emit(x: float, z: float, h_m: float, row: int, col: int, tip: bool) -> None:
        meta = {
            "class": cls, "synthetic": True, "source": "plan_band", "metric_size": True,
            "world_position": [float(x), 0.0, float(z)],
            "world_width": float(0.5 * h_m), "world_height": float(h_m),
            "score": 0.0, "band_row": int(row), "band_column": int(col), "band_tip": bool(tip),
            "overlap_min_separation": float(min_separation),
        }
        if bucket is not None:
            meta["bucket"] = int(bucket)
        instances.append(meta)

    # ── tips: the photographed crowns ────────────────────────────────────────
    el_top = np.where(np.isfinite(top_s), (0.5 - top_s / h) * math.pi, -np.inf)
    min_dist_px = max(2, int(round(w * 0.004)))
    peaks, _ = signal.find_peaks(el_top, prominence=math.radians(tip_prominence_deg), distance=min_dist_px)
    for c in peaks:
        c = int(c)
        if not np.isfinite(bot[c]) or bot[c] <= top_s[c]:
            continue
        extent_rows = bot_raw[c] - top_raw[c] + 1.0
        if count[c] / max(extent_rows, 1.0) < 0.2:
            continue
        f = foot(c)
        if f is None:
            stats["missed"] += 1
            continue
        x, z, dist, el_b, beyond = f
        el_t = elevation(top_s[c])
        floor, implied = height_floor(dist, el_t, el_b)
        hgt, _ = tree_height(dist, x, z, el_t)
        if hgt < 0.3:
            continue
        hgt = float(np.clip(hgt, floor, hi))
        # Lateral jitter of a fraction of the crown so a tip never sits on a grid line.
        jit = float(rng.uniform(-0.15, 0.15)) * hgt
        theta = (c / w - 0.5) * 2.0 * math.pi
        emit(x + jit * math.cos(theta), z - jit * math.sin(theta), hgt, 0, c, True)
        stats["tips"] += 1
        stats["beyond"] += int(beyond)
        if not stand[c]:
            stats["lone"] += 1
        stats["distance_m"].append(dist)
        stats["height_m"].append(hgt)
        stats["implied_m"].append(implied)

    # ── fill: the stands between the tips, in clumps ─────────────────────────
    # Crown angular width at the band's typical height and distance sets the clump
    # period; measured from the tips when there are any.
    if stats["height_m"]:
        crown_ang = 0.5 * float(np.median(stats["height_m"])) / max(float(np.median(stats["distance_m"])), 1e-3)
    else:
        crown_ang = 0.5 * size_m / max(far_m, 1e-3)
    period_px = max(8.0, clump_period_crowns * crown_ang / (2.0 * math.pi) * w)
    keep_p = _clump_field(w, period_px, clump_floor, rng)

    col = 0
    in_span = False
    while col < w:
        if not np.isfinite(top_wide[col]) or not np.isfinite(bot[col]) or bot[col] <= top_wide[col] or not stand[col]:
            in_span = False
            col += col_step
            continue
        if not in_span:
            stats["spans"] += 1
            in_span = True
        stats["columns"] += 1
        extent_rows = bot_raw[col] - top_raw[col] + 1.0
        fill = float(count[col]) / max(extent_rows, 1.0) if np.isfinite(bot_raw[col]) else 0.0
        if fill < min_fill:
            stats["thin"] += 1
            col += col_step
            continue
        f = foot(col)
        if f is None:
            stats["missed"] += 1
            col += col_step
            continue
        x, z, dist, el_b, beyond = f
        el_t = elevation(top_wide[col])
        el_sky = elevation(top_s[col])
        floor, implied = height_floor(dist, el_t, el_b)
        canopy, y_g = tree_height(dist, x, z, el_t)
        if canopy < 0.3:
            stats["thin"] += 1
            col += col_step
            continue
        # A fill tree is a crown's share of the canopy: a forest is several crowns
        # deep, not one row of trees as tall as the band.
        height = float(np.clip(crown_fraction * canopy, floor, hi))
        theta = (col / w - 0.5) * 2.0 * math.pi
        placed_rows = 0
        for r in range(max(1, rows)):
            if rng.uniform() > keep_p[col]:
                continue
            back = r * crown_fraction * height
            d_r = dist + back
            if d_r > 2.0 * far_m:
                break
            jitter = float(rng.uniform(-0.4, 0.4)) * height
            theta_r = theta + (jitter + (0.5 * step_fraction * height if r % 2 else 0.0)) / max(d_r, 1e-3)
            xr = x + back * math.sin(theta_r) + (jitter * math.cos(theta_r))
            zr = z + back * math.cos(theta_r) - (jitter * math.sin(theta_r))
            h_r = float(np.clip(height * float(rng.uniform(0.7, 1.2)), min(floor, lo), hi))
            y_r = ground_y(xr, zr) if ground_y is not None else y_g
            if y_r is None:
                break
            if math.atan2(y_r + h_r, d_r) > el_sky:
                # Under the skyline, or shorter: nothing stands above the mask's
                # top edge, on the mountain.
                h_r = d_r * math.tan(el_sky) - y_r
                if h_r < 0.5 * height:
                    break
            emit(xr, zr, h_r, r, col, False)
            placed_rows += 1
            stats["fill"] += 1
        stats["rows"].append(placed_rows)
        stats["beyond"] += int(beyond and placed_rows > 0)
        if placed_rows:
            stats["distance_m"].append(dist)
            stats["height_m"].append(height)
            stats["implied_m"].append(implied)
        # Next column a fraction of a crown further along the band at this distance.
        step_rad = step_fraction * height / max(dist, 1e-3)
        col += max(1, int(round(step_rad / (2.0 * math.pi) * w * float(rng.uniform(0.85, 1.15)))))

    def pct(key):
        return [round(float(v), 1) for v in np.percentile(stats[key], [5, 50, 95])] if stats[key] else None

    summary = {
        "spans": stats["spans"], "columns": stats["columns"], "thin": stats["thin"], "missed": stats["missed"],
        "beyond": stats["beyond"], "placed": len(instances), "tips": stats["tips"], "lone": stats["lone"],
        "fill": stats["fill"], "depth_feet": stats["depth_feet"],
        "rows": round(float(np.mean(stats["rows"])), 1) if stats["rows"] else None,
        "distance_m": pct("distance_m"), "height_m": pct("height_m"),
        "implied_m": [round(float(v), 0) for v in np.percentile(stats["implied_m"], [5, 50, 95])] if stats["implied_m"] else None,
    }
    return instances, summary
