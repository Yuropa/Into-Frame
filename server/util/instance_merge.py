"""
Cross-tile deduplication for panorama object detection (SAM2 automatic
segmentation, Grounding DINO). Unlike panorama_segmentation_imp.py's
per-pixel label voting (each output pixel independently picks its
highest-confidence tile), a discrete instance -- one object's box/mask --
falling in the overlap between two tiles gets detected TWICE, once per
tile, and those two detections need to collapse into one before being
stored as a single crop_i/metadata_i entry.
"""
from __future__ import annotations

import numpy as np


def _iou_xywh(a: list[float], b: list[float]) -> float:
    ax1, ay1 = a[0], a[1]
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx1, by1 = b[0], b[1]
    bx2, by2 = b[0] + b[2], b[1] + b[3]

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def wrap_aware_box_iou(a: list[float], b: list[float], pano_w: int) -> float:
    """Box IoU between two [x, y, w, h] boxes in panorama pixel space,
    wrap-aware: also compares after shifting `b` by ±pano_w and takes the
    max -- catches the same real object's box landing on opposite sides of
    the seam (one candidate at e.g. x=4090, its wrap-seam-tile duplicate at
    x=-6 -- both really the same few pixels around column 0)."""
    return max(
        _iou_xywh(a, b),
        _iou_xywh(a, [b[0] + pano_w, b[1], b[2], b[3]]),
        _iou_xywh(a, [b[0] - pano_w, b[1], b[2], b[3]]),
    )


def _render_mask_in_frame(
    mask: np.ndarray, tile_x0: int, tile_y0: int,
    frame_x0: int, frame_y0: int, frame_w: int, frame_h: int, pano_w: int,
) -> np.ndarray:
    """Place a tile-local boolean mask into a (frame_h, frame_w) canvas whose
    origin is (frame_x0, frame_y0) in panorama pixel space, wrapping columns
    modulo pano_w."""
    canvas = np.zeros((frame_h, frame_w), dtype=bool)
    mh, mw = mask.shape
    rows_pano = tile_y0 + np.arange(mh)
    cols_pano = (tile_x0 + np.arange(mw)) % pano_w

    rows_local = rows_pano - frame_y0
    cols_local = cols_pano - frame_x0
    row_ok = (rows_local >= 0) & (rows_local < frame_h)
    col_ok = (cols_local >= 0) & (cols_local < frame_w)
    if not row_ok.any() or not col_ok.any():
        return canvas

    row_src = np.nonzero(row_ok)[0]
    col_src = np.nonzero(col_ok)[0]
    canvas[np.ix_(rows_local[row_src], cols_local[col_src])] = mask[np.ix_(row_src, col_src)]
    return canvas


def wrap_aware_mask_iou(
    mask_a: np.ndarray, tile_a_x0: int, tile_a_y0: int,
    mask_b: np.ndarray, tile_b_x0: int, tile_b_y0: int,
    pano_w: int,
) -> float:
    """Mask IoU between two tile-local boolean masks (each the same size as
    its own tile, not bbox-cropped), positioned in panorama space by their
    tile's own (x0, y0). Renders both into a small shared local frame
    covering just their union footprint, rather than a full panorama-sized
    canvas -- callers only ever use this after a cheap box-IoU pre-filter
    (wrap_aware_box_iou) has already narrowed candidates down to genuinely
    close pairs, so the union footprint stays small. Tries all three
    horizontal wrap offsets for tile_b (matching wrap_aware_box_iou) and
    takes the max.
    """
    best = 0.0
    for dx in (0, pano_w, -pano_w):
        bx0 = tile_b_x0 + dx
        x0 = min(tile_a_x0, bx0)
        y0 = min(tile_a_y0, tile_b_y0)
        x1 = max(tile_a_x0 + mask_a.shape[1], bx0 + mask_b.shape[1])
        y1 = max(tile_a_y0 + mask_a.shape[0], tile_b_y0 + mask_b.shape[0])
        w, h = x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        frame_a = _render_mask_in_frame(mask_a, tile_a_x0, tile_a_y0, x0, y0, w, h, pano_w)
        frame_b = _render_mask_in_frame(mask_b, bx0, tile_b_y0, x0, y0, w, h, pano_w)
        inter = np.logical_and(frame_a, frame_b).sum()
        union = np.logical_or(frame_a, frame_b).sum()
        if union > 0:
            best = max(best, inter / union)
    return best


def merge_tiled_instances(dets: list[dict], overlap_fn, threshold: float, pano_w: int) -> list[dict]:
    """Union-find clustering over pairs where overlap_fn(a, b) >= threshold.

    Each det is a dict with at least "box" ([x, y, w, h] in panorama space)
    and "score" (float) -- the same real object detected independently in
    two or more overlapping tiles collapses to one entry: the cluster's
    highest-scoring member, with a non-wrapping box (x + w <= pano_w)
    preferred over a wrapping one as a tie-break *ahead of* score -- keeps
    every entry a normal [0, pano_w)-bounded box, so no downstream consumer
    (unproject_bbox_equirect, ObjectCorrelationStage._iou, etc.) needs to
    learn wrap-aware box math.
    """
    n = len(dets)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        for j in range(i + 1, n):
            if overlap_fn(dets[i], dets[j]) >= threshold:
                union(i, j)

    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)

    def rank(idx: int) -> tuple[bool, float]:
        x, _, w, _ = dets[idx]["box"]
        non_wrapping = (x + w) <= pano_w
        return (non_wrapping, dets[idx]["score"])

    return [dets[max(members, key=rank)] for members in clusters.values()]
