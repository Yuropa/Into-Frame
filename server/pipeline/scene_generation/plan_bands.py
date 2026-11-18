"""Bands from the scene plan (docs/hybrid-scene-plan.md, "bands, not scatter").

A region the plan and the region map agree on -- a forest -- is a band of one kind
of thing at one distance, not hundreds of independent detections. The panorama's
depth compresses the far field (Rainier's treeline, 200-400 m off, landed at 7-27 m
as 2 m saplings and the painter copied it over the meadow), so the band's distance
comes from the plan's size for the class over the band's angular height, capped at
the furthest ground the world has, and the trees stand at the height their angle
gives there. Each band tree is an ordinary synthetic instance for Scene Generation's
placement loop: snapped to the ground, overlap-tested, dressed with the class's
bucket assets.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


def column_spans(mask: np.ndarray, min_width_px: int, gap_px: int) -> list[tuple[int, int]]:
    """Inclusive (first, last) column ranges where `mask` has any pixel, merged
    across gaps of at most gap_px and at least min_width_px wide."""
    cols = np.where(mask.any(axis=0))[0]
    if cols.size == 0:
        return []
    spans: list[list[int]] = [[int(cols[0]), int(cols[0])]]
    for c in cols[1:]:
        c = int(c)
        if c - spans[-1][1] <= gap_px:
            spans[-1][1] = c
        else:
            spans.append([c, c])
    return [(a, b) for a, b in spans if b - a + 1 >= min_width_px]


def _column_elevations(mask: np.ndarray, c0: int, c1: int, smooth_px: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Per-column top and bottom row of the band in columns c0..c1, median-smoothed
    across smooth_px columns (a single tree crown is not the band's edge)."""
    sub = mask[:, c0:c1 + 1]
    h = sub.shape[0]
    rows = np.arange(h)[:, None]
    has = sub.any(axis=0)
    top = np.where(has, np.where(sub, rows, h).min(axis=0), np.nan).astype(float)
    bot = np.where(has, np.where(sub, rows, -1).max(axis=0), np.nan).astype(float)
    if smooth_px > 1:
        k = smooth_px
        pad = k // 2

        def med(a):
            out = a.copy()
            for i in range(len(a)):
                w = a[max(0, i - pad):i + pad + 1]
                w = w[np.isfinite(w)]
                out[i] = np.median(w) if w.size else np.nan
            return out
        top, bot = med(top), med(bot)
    return top, bot


def band_instances(
    region_map: np.ndarray,
    region_value: int,
    extrinsics,
    size_m: float,
    size_band: tuple[float, float],
    far_m: float,
    cls: str,
    bucket: Optional[int],
    rows: int = 3,
    step_fraction: float = 0.4,
    min_width_deg: float = 2.0,
    gap_deg: float = 1.5,
    min_distance_fraction: float = 0.25,
    seed: int = 0,
) -> tuple[list[dict], list[dict]]:
    """Synthetic instances for every span of `region_value` in the region map.

    Returns (instances, spans). Each span record carries its bearings, the distance
    the band stands at, the distance the plan size implies, and the tree count."""
    mask = np.asarray(region_map) == int(region_value)
    h, w = mask.shape
    px_per_deg = w / 360.0
    rng = np.random.default_rng((seed, 11))
    lo, hi = float(size_band[0]), float(size_band[1])
    instances: list[dict] = []
    spans_out: list[dict] = []
    for c0, c1 in column_spans(mask, int(min_width_deg * px_per_deg), int(gap_deg * px_per_deg)):
        top, bot = _column_elevations(mask, c0, c1)
        ok = np.isfinite(top) & np.isfinite(bot)
        if not ok.any():
            continue
        el_top = (0.5 - top / h) * math.pi
        el_bot = (0.5 - bot / h) * math.pi
        # The band's angular height where it is tallest matters for the distance (the
        # slope behind the treeline is forest too and only adds to it), so the median
        # is the honest reading of the first row's height.
        ang = np.where(ok, np.tan(np.clip(el_top, -1.2, 1.2)) - np.tan(np.clip(el_bot, -1.2, 1.2)), np.nan)
        ang_med = float(np.nanmedian(ang))
        if not np.isfinite(ang_med) or ang_med <= 1e-4:
            continue
        implied = size_m / ang_med
        distance = float(min(far_m, max(implied, min_distance_fraction * far_m)))
        a0 = (c0 / w - 0.5) * 360.0
        a1 = (c1 / w - 0.5) * 360.0
        count = 0
        heights: list[float] = []
        for r in range(max(1, int(rows))):
            # Each row a little nearer and staggered by half a step.
            d_r = max(distance - r * 0.5 * size_m, 0.5 * distance)
            step_rad = step_fraction * size_m / d_r
            theta = (c0 / w - 0.5) * 2.0 * math.pi + (0.5 * step_rad if r % 2 else 0.0)
            theta_end = (c1 / w - 0.5) * 2.0 * math.pi
            while theta <= theta_end:
                col = int(round(((theta / (2.0 * math.pi)) + 0.5) * w)) - c0
                col = int(np.clip(col, 0, c1 - c0))
                a_here = ang[col] if np.isfinite(ang[col]) else ang_med
                height = float(np.clip(d_r * a_here, lo, hi)) * float(rng.uniform(0.85, 1.15))
                height = float(np.clip(height, lo, hi))
                width = 0.5 * height
                jitter_theta = theta + float(rng.uniform(-0.3, 0.3)) * step_rad
                d_here = d_r * float(rng.uniform(0.96, 1.0))
                x, y, z = extrinsics.transform((d_here * math.sin(jitter_theta), 0.0, d_here * math.cos(jitter_theta)))
                meta = {
                    "class": cls, "synthetic": True, "source": "plan_band", "metric_size": True,
                    "world_position": [float(x), 0.0, float(z)],
                    "world_width": float(width), "world_height": float(height),
                    "score": 0.0, "band_span_deg": [round(a0, 1), round(a1, 1)], "band_row": r,
                }
                if bucket is not None:
                    meta["bucket"] = int(bucket)
                instances.append(meta)
                heights.append(height)
                count += 1
                theta += step_rad * float(rng.uniform(0.85, 1.15))
        spans_out.append({
            "from_deg": round(a0, 1), "to_deg": round(a1, 1), "distance_m": round(distance, 1),
            "implied_distance_m": round(implied, 1), "angular_height_deg": round(math.degrees(math.atan(ang_med)), 2),
            "count": count, "height_m": round(float(np.median(heights)), 2) if heights else None,
        })
    return instances, spans_out
