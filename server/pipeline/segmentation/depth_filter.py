from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, percentile_filter
from util.depth_utils import Depth
from pipeline.segmentation.segmentation_result import SegmentationResult


class DepthObjectFilter:
    """
    Loose pre-filter that discards SAM masks unlikely to be foreground objects
    using the panorama depth map. No model required — pure numpy.

    Two complementary signals are combined with OR — a mask is kept if it passes
    either test, so genuinely foreground objects are less likely to be dropped:

    1. Local windowed baseline
       Normalises depth and builds a local background reference per pixel: the
       Nth percentile depth within a horizontal window around it, wrapping at the
       panorama's 360° seam. A single *row-wide* max (the original version of this
       signal) is a poor background reference for wide equirectangular panoramas —
       one far horizon pixel anywhere in a 360° sweep would set the baseline for
       the entire row, making ordinary near terrain elsewhere in that same row
       look like it "pops out" against it. Windowing keeps the reference local to
       each object's actual surroundings.
       Scores each mask by the median of (depth − baseline). Foreground objects
       sit clearly in front of their local background → score well below 0.
       Background/terrain sits close to its own local baseline → score ≈ 0.

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
        edge_threshold: float = 0.01,
        baseline_window_frac: float = 0.12,
        baseline_percentile: float = 90.0,
    ) -> SegmentationResult:
        if result.is_empty():
            return result

        depth_arr = depth.depth.copy()

        dmin, dmax = float(np.nanmin(depth_arr)), float(np.nanmax(depth_arr))
        if dmax - dmin < 1e-6:
            return result
        depth_norm = (depth_arr - dmin) / (dmax - dmin)
        # Unknown depth treated as "far" (matches the old row-baseline behaviour
        # for fully-NaN rows), so missing data can't masquerade as foreground.
        depth_safe = np.nan_to_num(depth_norm, nan=1.0)

        # Signal 1: local windowed baseline, wrapping horizontally since the
        # panorama is a 360° cylinder. A percentile (rather than a hard max) is
        # used so a single stray near or far outlier within the window can't
        # swing the baseline.
        window = max(3, int(round(depth_safe.shape[1] * baseline_window_frac)))
        baseline = percentile_filter(
            depth_safe, percentile=baseline_percentile,
            size=(1, window), mode="wrap",
        )
        residual = depth_safe - baseline           # objects << 0, background ≈ 0

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
