from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from util.depth_utils import Depth
from pipeline.segmentation.segmentation_result import SegmentationResult


class DepthObjectFilter:
    """
    Loose pre-filter that discards SAM masks unlikely to be foreground objects
    using the panorama depth map. No model required — pure numpy.

    Two complementary signals are combined with OR — a mask is kept if it passes
    either test, so genuinely foreground objects are less likely to be dropped:

    1. Row-wise baseline (original signal)
       Normalises depth, builds a per-row maximum (the farthest/sky plane at each
       elevation) and scores each mask by the median of (depth − baseline).
       Foreground objects sit clearly in front of the background → score < 0.

    2. Boundary edge gradient (research-code signal)
       Scores each mask by how consistently the pixels just *outside* the boundary
       are farther away than those just *inside*. A foreground object embedded in a
       background plane produces a strong positive depth jump at its silhouette.
       score = mean_positive_jump × fraction_positive_boundary_pixels
    """

    def filter(
        self,
        result: SegmentationResult,
        depth: Depth,
        threshold: float = -0.05,
        edge_threshold: float = 0.02,
    ) -> SegmentationResult:
        if result.is_empty():
            return result

        depth_arr = depth.depth.copy()

        dmin, dmax = float(np.nanmin(depth_arr)), float(np.nanmax(depth_arr))
        if dmax - dmin < 1e-6:
            return result
        depth_norm = (depth_arr - dmin) / (dmax - dmin)

        # Signal 1: row-wise maximum baseline
        row_max  = np.nan_to_num(np.nanmax(depth_norm, axis=1), nan=1.0)
        baseline = row_max[:, np.newaxis]
        residual = depth_norm - baseline          # objects < 0, background ≈ 0

        kept_masks, kept_boxes, kept_scores = [], [], []
        for mask, box, score in zip(result.masks, result.boxes, result.scores):
            mask_bool = self._to_bool(mask, depth_arr.shape)
            if not mask_bool.any():
                continue

            baseline_score = float(np.median(residual[mask_bool]))
            edge_score     = self._edge_gradient_score(mask_bool, depth_norm)

            if baseline_score < threshold or edge_score > edge_threshold:
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
    def _edge_gradient_score(mask_bool: np.ndarray, depth_norm: np.ndarray, iterations: int = 4) -> float:
        """
        Score = mean_positive_jump × fraction_positive, where a "jump" is how much
        farther each outer-ring pixel is compared to the mean depth of the inner ring.
        A foreground object reliably pops out of its background → high score.
        """
        if mask_bool.sum() < 25:          # too small for meaningful boundary
            return 0.0

        outer = binary_dilation(mask_bool, iterations=iterations) & ~mask_bool
        inner = mask_bool & ~binary_erosion(mask_bool, iterations=iterations)

        if not outer.any() or not inner.any():
            return 0.0

        mean_inner = float(depth_norm[inner].mean())
        jumps      = depth_norm[outer] - mean_inner   # positive = outer is farther

        positive = jumps[jumps > 0]
        if len(positive) == 0:
            return 0.0

        return float(positive.mean()) * (len(positive) / len(jumps))

    @staticmethod
    def _to_bool(mask, target_shape: tuple[int, int]) -> np.ndarray:
        arr = np.asarray(mask)
        if arr.shape != target_shape:
            from PIL import Image as PILModule
            pil = PILModule.fromarray((arr * 255).astype(np.uint8) if arr.dtype != bool else arr.astype(np.uint8) * 255, mode="L")
            pil = pil.resize((target_shape[1], target_shape[0]), PILModule.NEAREST)
            arr = np.asarray(pil)
        return arr.astype(bool)
