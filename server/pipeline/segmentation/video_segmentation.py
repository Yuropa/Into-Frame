import torch
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from remote_connection.remote_client import RemoteClient
from util.video_utils import Video
from pipeline.segmentation.video_segmentation_result import VideoSegmentationResult


class VideoSeg(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "video_segmentation_imp.py"
        # Reuses the "frame" env — SAM2's video predictor ships in the same
        # `sam2` package already installed there for ImageSeg.
        super().__init__(device=device, conda_env="frame", script_path=script_path)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["facebook/sam2.1-hiera-large"]

    def segment_video(
        self,
        video: Video,
        reference_mask: np.ndarray,
        temp_path: Path,
        reference_frame_idx: int = 0,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ) -> VideoSegmentationResult:
        """Tracks `reference_mask` — a HxW boolean/0-1 mask aligned to the frame at
        `reference_frame_idx` — across every frame of `video`, propagating both
        forward and backward from that frame so the entire video is covered.
        """
        data = {
            "video_path": str(video.path),
            "mask": np.asarray(reference_mask, dtype=bool),
            "reference_frame_idx": reference_frame_idx,
        }
        result = self.send(
            action="segment_video",
            input=data,
            temp_path=temp_path,
            on_progress=on_progress,
        )
        return VideoSegmentationResult(masks=result["masks"])
