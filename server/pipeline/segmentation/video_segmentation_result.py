from __future__ import annotations
import numpy as np


class VideoSegmentationResult:
    """Per-frame masks tracking a single object through a video, produced by
    SAM2's video predictor from one reference mask on one frame.
    """

    def __init__(self, masks: np.ndarray):
        # (num_frames, H, W) boolean array, one mask per video frame in source order.
        self.masks = np.asarray(masks, dtype=bool)

    @property
    def num_frames(self) -> int:
        return self.masks.shape[0]

    def mask_at(self, frame_idx: int) -> np.ndarray:
        return self.masks[frame_idx]
