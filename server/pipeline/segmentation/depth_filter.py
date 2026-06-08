from __future__ import annotations
import numpy as np
from util.depth_utils import Depth
from pipeline.segmentation.segmentation_result import SegmentationResult


class DepthObjectFilter:
    """
    Loose pre-filter that discards SAM masks unlikely to be foreground objects
    using the panorama depth map. No model required — pure numpy.

    Algorithm
    ---------
    1. Normalize depth to [0, 1].
    2. Build a row-wise maximum baseline: the farthest depth in each row of an
       equirectangular panorama closely tracks the background plane at that
       elevation, so objects (which are closer) fall below it.
    3. Score each mask as the median of (depth − baseline) inside the mask.
       Foreground objects have score < 0; background detections sit near 0.
    4. Keep masks whose score is below `threshold`. A value like -0.05 is a
       generous gate — it only removes detections that are clearly sitting at
       or behind the background plane.
    """

    def filter(
        self,
        result: SegmentationResult,
        depth: Depth,
        threshold: float = -0.05,
    ) -> SegmentationResult:
        if result.is_empty():
            return result

        depth_arr = depth.depth.copy()

        dmin, dmax = float(np.nanmin(depth_arr)), float(np.nanmax(depth_arr))
        if dmax - dmin < 1e-6:
            return result
        depth_norm = (depth_arr - dmin) / (dmax - dmin)

        # Row-wise maximum: farthest depth in each row = background baseline
        row_max = np.nanmax(depth_norm, axis=1)          # (H,)
        row_max = np.nan_to_num(row_max, nan=1.0)
        baseline = row_max[:, np.newaxis]                 # (H, 1) broadcasts to (H, W)

        residual = depth_norm - baseline                  # objects < 0, background ≈ 0

        kept_masks, kept_boxes, kept_scores = [], [], []
        for mask, box, score in zip(result.masks, result.boxes, result.scores):
            mask_bool = self._to_bool(mask, depth_arr.shape)
            if not mask_bool.any():
                continue
            depth_score = float(np.median(residual[mask_bool]))
            if depth_score < threshold:
                kept_masks.append(mask)
                kept_boxes.append(box)
                kept_scores.append(score)

        if not kept_masks:
            return SegmentationResult.empty()

        return SegmentationResult(
            masks=kept_masks,
            boxes=kept_boxes,
            scores=kept_scores,
        )

    @staticmethod
    def _to_bool(mask, target_shape: tuple[int, int]) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.shape != target_shape:
            from PIL import Image as PILModule
            pil = PILModule.fromarray((arr * 255).astype(np.uint8) if arr.dtype != bool else arr.astype(np.uint8) * 255, mode="L")
            pil = pil.resize((target_shape[1], target_shape[0]), PILModule.NEAREST)
            arr = np.asarray(pil)
        return arr.astype(bool)
