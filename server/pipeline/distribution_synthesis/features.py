"""
Where a painted population belongs relative to the scene's linear features.

Distribution Synthesis learns HOW objects are spaced (a pair-correlation pattern)
and paints it evenly across a region type. That cannot put Paris's trees along the
river: the pattern knows nothing about the bank, so it spreads the trees across
every paintable cell and the quay reads as a straight band along the photo's
horizon rather than a line following the water.

This adds WHERE: distance fields on the terrain grid to the features a region map
can show -- the water's edge (signed, negative on the water), roads, trails, the
forest edge and built structures -- and, per painted class, the ratio between the
distance distribution at its exemplars and at the cells it could be painted on.
The most informative feature (largest KL divergence) becomes an acceptance
probability that thins the synthesized pattern, so trees concentrate along the
bank, boats along the water near it, flowers away from the trail.

It also estimates how many of a small class the detector missed (bloom_recall):
75 flowers were detected on the Rainier meadow, which has hundreds, and the painted
density is learned from that count.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, label as nd_label

from pipeline.panorama_segmentation.panorama_region_result import RegionType

# Feature name -> (region types that make it, signed). A signed field is negative
# inside the feature (on the water, inside the forest), so "on the water near the
# bank" and "on land near the bank" are different bins.
FEATURES: dict[str, tuple[tuple[RegionType, ...], bool]] = {
    "water_edge": ((RegionType.WATER,), True),
    "road": ((RegionType.ROAD,), False),
    "trail": ((RegionType.TRAIL,), False),
    "forest_edge": ((RegionType.FOREST,), True),
    "built": ((RegionType.BUILT,), False),
}

# Signed distance bin edges in metres: fine near the feature, coarse far from it.
_EDGES = np.array([-1e9, -60, -30, -15, -8, -4, -2, 0, 2, 4, 8, 15, 30, 60, 120, 1e9], dtype=np.float64)


@dataclass
class FeatureAffinity:
    feature: str
    kl: float
    ratio: np.ndarray        # per bin, max-normalised to 1
    exemplar_hist: np.ndarray
    background_hist: np.ndarray

    def accept(self, value: float) -> float:
        b = int(np.clip(np.searchsorted(_EDGES, value, side="right") - 1, 0, len(self.ratio) - 1))
        return float(self.ratio[b])

    def describe(self) -> str:
        peak = int(np.argmax(self.ratio))
        lo, hi = _EDGES[peak], _EDGES[peak + 1]
        span = f"{lo:g}..{hi:g} m" if abs(lo) < 1e8 and abs(hi) < 1e8 else (f"< {hi:g} m" if abs(lo) >= 1e8 else f"> {lo:g} m")
        return f"{self.feature} (KL {self.kl:.2f}, densest at {span})"


def feature_fields(region_map: np.ndarray, cell_m: float, min_cells: int = 16) -> dict[str, np.ndarray]:
    """Distance (metres) from every grid cell to each feature present in the map."""
    out: dict[str, np.ndarray] = {}
    rm = np.asarray(region_map).astype(np.int64)
    for name, (types, signed) in FEATURES.items():
        mask = np.isin(rm, [int(t) for t in types])
        if int(mask.sum()) < min_cells or mask.all():
            continue
        outside = distance_transform_edt(~mask) * cell_m
        if signed:
            inside = distance_transform_edt(mask) * cell_m
            out[name] = np.where(mask, -inside, outside).astype(np.float32)
        else:
            out[name] = outside.astype(np.float32)
    return out


def world_to_cell(x: np.ndarray, z: np.ndarray, grid_size_m: float, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """Same mapping Distribution Synthesis applies per painted point."""
    cell_m = grid_size_m / resolution
    col = ((np.asarray(x) + grid_size_m / 2.0) / cell_m).astype(np.int64)
    row = ((np.asarray(z) + grid_size_m / 2.0) / cell_m).astype(np.int64)
    return np.clip(row, 0, resolution - 1), np.clip(col, 0, resolution - 1)


def learn_affinity(
    fields: dict[str, np.ndarray],
    exemplar_rows: np.ndarray,
    exemplar_cols: np.ndarray,
    background: np.ndarray,
    min_kl: float,
    floor: float,
) -> "FeatureAffinity | None":
    """The feature whose distance best separates this class's exemplars from the
    cells it could be painted on, or None when none separates them by min_kl.

    floor keeps a little density everywhere: a class learned from a handful of
    exemplars along one bank should still appear, sparsely, elsewhere."""
    best: FeatureAffinity | None = None
    for name, field in fields.items():
        bg_vals = field[background]
        if bg_vals.size == 0:
            continue
        ex_vals = field[exemplar_rows, exemplar_cols]
        ex_hist = np.histogram(ex_vals, bins=_EDGES)[0].astype(np.float64) + 0.5
        bg_hist = np.histogram(bg_vals, bins=_EDGES)[0].astype(np.float64) + 0.5
        p_ex, p_bg = ex_hist / ex_hist.sum(), bg_hist / bg_hist.sum()
        kl = float(np.sum(p_ex * np.log(p_ex / p_bg)))
        if kl < min_kl or (best is not None and kl <= best.kl):
            continue
        ratio = p_ex / p_bg
        # Bins the background barely has carry no evidence either way; don't let a
        # 0.5-pseudocount bin with one exemplar become the peak everything is scaled to.
        ratio[bg_hist < 3.0] = np.minimum(ratio[bg_hist < 3.0], 1.0)
        ratio = ratio / max(float(ratio.max()), 1e-9)
        ratio = np.maximum(ratio, floor)
        best = FeatureAffinity(name, kl, ratio.astype(np.float64), ex_hist - 0.5, bg_hist - 0.5)
    return best


def edge_tangent_yaw(field: np.ndarray, row: int, col: int) -> "float | None":
    """Yaw (degrees, Unity convention: local +X onto the tangent) of the feature edge
    through a cell -- the direction a boat moored along a bank points."""
    h, w = field.shape
    r0, r1 = max(row - 1, 0), min(row + 1, h - 1)
    c0, c1 = max(col - 1, 0), min(col + 1, w - 1)
    gz = float(field[r1, col] - field[r0, col])
    gx = float(field[row, c1] - field[row, c0])
    if abs(gx) + abs(gz) < 1e-6:
        return None
    # Tangent is perpendicular to the gradient; grid rows run along +Z, cols along +X.
    tx, tz = -gz, gx
    return float(np.degrees(np.arctan2(-tz, tx)))


def bloom_recall(
    panorama_rgb: np.ndarray,
    boxes: list,
    min_saturation: float = 0.35,
    min_value: float = 0.25,
    green_hue_deg: tuple[float, float] = (60.0, 175.0),
    max_boost: float = 6.0,
) -> tuple[float, int, int]:
    """(density multiplier, bloom blobs counted, detections counted) for a class of
    small coloured blooms, measured inside the panorama area its detections span.

    Blooms are the saturated, non-green pixels of a meadow; each connected run of
    them is one bloom (or a few touching ones, which makes this an undercount). The
    detector found `detections` of them in the same area, so blobs / detections is
    its recall, and the painted density -- learned from the detections -- is short
    by that factor. Clamped to [1, max_boost]."""
    if not boxes:
        return 1.0, 0, 0
    b = np.asarray(boxes, dtype=np.float64)          # [x, y, w, h], panorama pixels
    h_img, w_img = panorama_rgb.shape[:2]
    x0 = int(max(np.percentile(b[:, 0], 2) - 0.05 * w_img, 0))
    x1 = int(min(np.percentile(b[:, 0] + b[:, 2], 98) + 0.05 * w_img, w_img))
    y0 = int(max(np.percentile(b[:, 1], 2), 0))
    y1 = int(min(np.percentile(b[:, 1] + b[:, 3], 98), h_img))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 1.0, 0, len(b)
    rgb = np.asarray(panorama_rgb[y0:y1, x0:x1, :3], dtype=np.float32) / 255.0
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    r, g, bl = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue = (np.degrees(np.arctan2(np.sqrt(3.0) * (g - bl), 2.0 * r - g - bl)) + 360.0) % 360.0
    bloom = (sat >= min_saturation) & (mx >= min_value) & ~((hue >= green_hue_deg[0]) & (hue <= green_hue_deg[1]))
    labels, n = nd_label(bloom)
    if n == 0:
        return 1.0, 0, len(b)
    areas = np.bincount(labels.ravel())[1:]
    # A bloom's coloured part is a fraction of its box; ignore specks far smaller.
    min_area = max(4.0, 0.05 * float(np.median(b[:, 2] * b[:, 3])))
    blobs = int((areas >= min_area).sum())
    cx, cy = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    detections = int(((cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)).sum())
    if detections == 0:
        return 1.0, blobs, 0
    return float(np.clip(blobs / detections, 1.0, max_boost)), blobs, detections
