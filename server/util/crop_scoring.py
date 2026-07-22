"""Shared crop-quality scoring for picking the best/curated instances within a
group of detected objects (e.g. a category or a visual-similarity bucket
within a category). Extracted from PanoramaAssetGenerationStage so both asset
curation and any future consumer share one implementation instead of
duplicating the occlusion/composite-score math."""
from __future__ import annotations

import numpy as np


def mask_fill_ratio(crop) -> float | None:
    """crop.image is already cropped tight to its own bbox, so fill ratio is
    just alpha-nonzero-px / (crop.width * crop.height) -- a proxy for 'clean
    single-object segmentation' vs a partial/broken crop."""
    if crop is None or crop.image.mode != "RGBA":
        return None
    alpha = np.asarray(crop.image.getchannel("A"))
    return float((alpha > 0).sum()) / float(alpha.size) if alpha.size else None


def covered_fraction(box_i: list[float], box_j: list[float]) -> float:
    """Fraction of box_i's area covered by box_j -- containment-style overlap,
    not symmetric IoU, since an occluder can be much larger or smaller than
    the thing it's occluding."""
    ax1, ay1 = box_i[0], box_i[1]
    ax2, ay2 = box_i[0] + box_i[2], box_i[1] + box_i[3]
    bx1, by1 = box_j[0], box_j[1]
    bx2, by2 = box_j[0] + box_j[2], box_j[1] + box_j[3]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_i = box_i[2] * box_i[3]
    return inter / area_i if area_i > 0 else 0.0


def occlusion_score(
    idx: int, box, depth: float, depth_by_idx: dict[int, tuple[list, float]], depth_margin: float,
) -> float:
    """Max covered_fraction among all OTHER instances (any class/group -- an
    occluder that's itself excluded from curation, e.g. a bush in front of a
    building, still visually cuts off what's behind it) whose sampled depth is
    nearer than this candidate's by more than depth_margin."""
    if box is None:
        return 0.0
    best = 0.0
    for j, (jbox, jdepth) in depth_by_idx.items():
        if j == idx:
            continue
        if jdepth < depth * (1.0 - depth_margin):
            best = max(best, covered_fraction(box, jbox))
    return best


def composite_score(
    metadata,
    crop,
    depth: float,
    occlusion: float,
    threshold: float,
    weight_confidence: float,
    weight_fill_ratio: float,
    weight_depth: float,
    weight_occlusion: float,
    occlusion_covered_fraction_threshold: float,
) -> float:
    confidence = metadata.get("confidence", 0.5)
    fill_ratio = mask_fill_ratio(crop)
    fill_ratio = fill_ratio if fill_ratio is not None else 0.5
    depth_score = max(0.0, 1.0 - depth / threshold) if threshold else 0.0
    occlusion_penalty = occlusion if occlusion >= occlusion_covered_fraction_threshold else 0.0
    return (
        weight_confidence * confidence
        + weight_fill_ratio * fill_ratio
        + weight_depth * depth_score
        - weight_occlusion * occlusion_penalty
    )
