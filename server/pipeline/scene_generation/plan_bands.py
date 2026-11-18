"""Bands from the scene plan (docs/hybrid-scene-plan.md, "bands, not scatter").

A region the plan and the region map agree on -- a forest -- is a band of one kind
of thing, not hundreds of independent detections. The panorama's depth compresses
the far field (Rainier's treeline, 200-400 m off, landed at 7-27 m as 2 m saplings
and the painter copied it over the meadow), so the band is placed from the region
map's own shape instead: in every column that shows forest, the band's bottom edge
is cast onto the reconstructed terrain and a tree stands where that ray lands, as
tall as the column's forest band allows at that distance. The mask's clumps become
clumps (tall columns get a second and third row behind, thin columns a short tree
or none) and its gaps stay gaps. Each tree is an ordinary synthetic instance for
Scene Generation's placement loop: snapped to the ground, overlap-tested, dressed
with the class's bucket assets.
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

GroundHit = Callable[[float, float], Optional[tuple[float, float, float]]]
"""(theta_rad, elevation_rad) -> (world_x, world_z, horizontal distance) where the
ray from the camera meets the terrain, or None when it misses."""


def _column_edges(mask: np.ndarray, smooth_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per column: top row, bottom row and the count of band pixels, with the two
    edges median-smoothed across smooth_px columns (one crown is not the edge)."""
    h, w = mask.shape
    rows = np.arange(h)[:, None]
    has = mask.any(axis=0)
    top = np.where(has, np.where(mask, rows, h).min(axis=0), np.nan).astype(float)
    bot = np.where(has, np.where(mask, rows, -1).max(axis=0), np.nan).astype(float)
    count = mask.sum(axis=0).astype(float)
    if smooth_px > 1:
        pad = smooth_px // 2

        def med(a):
            out = a.copy()
            for i in range(len(a)):
                win = a[max(0, i - pad):i + pad + 1]
                win = win[np.isfinite(win)]
                out[i] = np.median(win) if win.size else np.nan
            return out
        top, bot = med(top), med(bot)
    return top, bot, count


def band_instances(
    region_map: np.ndarray,
    region_value: int,
    ground_hit: GroundHit,
    size_m: float,
    size_band: tuple[float, float],
    far_m: float,
    cls: str,
    bucket: Optional[int],
    rows: int = 3,
    step_fraction: float = 0.4,
    min_fill: float = 0.35,
    near_fraction: float = 0.0,
    camera_xz: tuple[float, float] = (0.0, 0.0),
    seed: int = 0,
) -> tuple[list[dict], dict]:
    """Synthetic instances following the region's shape in the region map.

    Returns (instances, summary). A tree is authored where a column's band bottom
    lands on the terrain; its height is the column's band height at that distance,
    clamped to size_band; a column whose band is less than min_fill solid (a crown
    seen through gaps) is skipped; a column whose band stands more than one tree
    tall gets rows behind it, up the slope."""
    mask = np.asarray(region_map) == int(region_value)
    h, w = mask.shape
    top, bot, count = _column_edges(mask, smooth_px=max(3, w // 170))
    lo, hi = float(size_band[0]), float(size_band[1])
    rng = np.random.default_rng((seed, 11))
    near_floor = near_fraction * far_m
    instances: list[dict] = []
    stats = {"columns": 0, "thin": 0, "missed": 0, "beyond": 0, "placed": 0, "spans": 0,
             "distance_m": [], "height_m": [], "implied_m": []}
    col = 0
    in_span = False
    while col < w:
        if not np.isfinite(top[col]) or not np.isfinite(bot[col]) or bot[col] <= top[col]:
            in_span = False
            col += max(1, w // 720)
            continue
        if not in_span:
            stats["spans"] += 1
            in_span = True
        stats["columns"] += 1
        el_top = (0.5 - top[col] / h) * math.pi
        el_bot = (0.5 - bot[col] / h) * math.pi
        extent_rows = bot[col] - top[col] + 1.0
        fill = float(count[col]) / max(extent_rows, 1.0)
        theta = (col / w - 0.5) * 2.0 * math.pi
        ang = math.tan(min(max(el_top, -1.2), 1.2)) - math.tan(min(max(el_bot, -1.2), 1.2))
        if fill < min_fill or ang <= 1e-4:
            stats["thin"] += 1
            col += max(1, w // 720)
            continue
        # The band's foot: walking the column's forest rows from the top down, the
        # lowest row whose ray still lands on far ground. Forest pixels below that
        # are mid-ground shrubs on the meadow (Rainier: the mask runs down to -4 deg
        # and that ray lands 13 m out), not the treeline.
        hit = None
        n_probe = 8
        for k in range(n_probe + 1):
            el = el_bot + (el_top - el_bot) * (k / n_probe)
            probe = ground_hit(theta, el)
            if probe is not None and probe[2] >= near_floor:
                hit = probe
                el_bot = el
                ang = math.tan(min(max(el_top, -1.2), 1.2)) - math.tan(min(max(el_bot, -1.2), 1.2))
                break
        if hit is not None and hit[2] > far_m:
            # The band's foot is on the climbing far field; stand at the furthest
            # ground instead, at the height the angle allows there.
            hit = (camera_xz[0] + (hit[0] - camera_xz[0]) * far_m / hit[2],
                   camera_xz[1] + (hit[1] - camera_xz[1]) * far_m / hit[2], far_m)
            stats["beyond"] += 1
        if hit is None:
            stats["missed"] += 1
            col += max(1, w // 720)
            continue
        x, z, dist = hit
        if ang <= 1e-4:
            stats["thin"] += 1
            col += max(1, w // 720)
            continue
        implied = size_m / ang
        # A tree's height is what its angle allows where it stands: the plan's band
        # floor applies only where the plan size could be true (at or beyond the
        # implied distance); nearer, the band is a model of a far treeline and a
        # full-size tree would cover the mountain behind it.
        floor = lo if dist >= 0.5 * implied else min(lo, max(0.5, 0.15 * size_m))
        height = float(np.clip(dist * ang, floor, hi)) * float(rng.uniform(0.85, 1.15))
        height = float(np.clip(height, floor, hi))
        n_rows = int(np.clip(round(dist * ang / max(height, 1e-3)), 1, max(1, rows)))
        for r in range(n_rows):
            # Rows behind, up the slope, staggered by half a step; the loop snaps them.
            back = r * 0.6 * height
            d_r = dist + back
            jitter = float(rng.uniform(-0.3, 0.3)) * step_fraction * height
            theta_r = theta + (jitter + (0.5 * step_fraction * height if r % 2 else 0.0)) / max(d_r, 1e-3)
            xr = x + back * math.sin(theta_r) + (jitter * math.cos(theta_r))
            zr = z + back * math.cos(theta_r) - (jitter * math.sin(theta_r))
            h_r = float(np.clip(height * float(rng.uniform(0.85, 1.15)), lo, hi))
            meta = {
                "class": cls, "synthetic": True, "source": "plan_band", "metric_size": True,
                "world_position": [float(xr), 0.0, float(zr)],
                "world_width": float(0.5 * h_r), "world_height": h_r,
                "score": 0.0, "band_row": r, "band_column": int(col),
            }
            if bucket is not None:
                meta["bucket"] = int(bucket)
            instances.append(meta)
            stats["placed"] += 1
        stats["distance_m"].append(dist)
        stats["height_m"].append(height)
        stats["implied_m"].append(implied)
        # Next tree one crown further along the band at this distance.
        step_rad = step_fraction * height / max(dist, 1e-3)
        col += max(1, int(round(step_rad / (2.0 * math.pi) * w * float(rng.uniform(0.85, 1.15)))))
    summary = {
        "spans": stats["spans"], "columns": stats["columns"], "thin": stats["thin"], "missed": stats["missed"],
        "beyond": stats["beyond"], "placed": stats["placed"],
        "distance_m": [round(float(v), 1) for v in np.percentile(stats["distance_m"], [5, 50, 95])] if stats["distance_m"] else None,
        "height_m": [round(float(v), 1) for v in np.percentile(stats["height_m"], [5, 50, 95])] if stats["height_m"] else None,
        "implied_m": [round(float(v), 0) for v in np.percentile(stats["implied_m"], [5, 50, 95])] if stats["implied_m"] else None,
    }
    return instances, summary
