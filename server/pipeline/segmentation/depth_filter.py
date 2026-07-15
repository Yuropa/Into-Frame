from __future__ import annotations
import numpy as np
from util.depth_utils import Depth
from pipeline.segmentation.segmentation_result import SegmentationResult


class DepthObjectFilter:
    """
    Filters SAM masks down to genuine foreground occluders using the panorama's
    real metric depth (Depth Anything Panorama returns metres) — a mask is kept
    if its median depth is closer than distance_threshold_m. These panoramas
    are outdoor scenes shot from ~1-2m off the ground, so anything within a few
    metres of the camera is, physically, standing between the camera and the
    ground it's trying to reconstruct.

    This replaced an earlier version that scored masks by *local relative*
    depth (how much closer a mask sits than its immediate neighbourhood, or
    how sharply depth jumps at its boundary). That measured local terrain
    roughness, not proximity to the camera, and got it backwards on ordinary
    landscape panoramas: verified on a Mt Rainier panorama where the near
    flower meadow sat at ~1.4m depth with almost no local variance (std 0.2m,
    it's a flat meadow) — so it never "popped out" of its own neighbourhood —
    while the distant treeline/ridge sat at ~22m with huge local variance
    (std 25m, from trees, gaps between them, and saddle points), so it did.
    The old signals flagged the treeline as foreground and left the meadow
    alone. Absolute distance doesn't have that failure mode: sky is already
    pinned to the depth model's far clamp (100m), so it's automatically
    excluded without any special-casing.
    """

    def filter(
        self,
        result: SegmentationResult,
        depth: Depth,
        distance_threshold_m: float = 5.0,
    ) -> SegmentationResult:
        if result.is_empty():
            return result

        depth_arr = depth.depth

        kept_masks, kept_boxes, kept_scores = [], [], []
        for mask, box, score in zip(result.masks, result.boxes, result.scores):
            mask_bool = self._to_bool(mask, depth_arr.shape)
            if not mask_bool.any():
                continue

            mask_depth = depth_arr[mask_bool]
            valid = mask_depth[np.isfinite(mask_depth)]
            if valid.size == 0:
                continue  # unknown depth — can't confirm proximity, don't remove it

            if float(np.median(valid)) < distance_threshold_m:
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
