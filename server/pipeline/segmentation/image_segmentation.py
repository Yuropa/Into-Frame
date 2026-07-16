import torch
from pathlib import Path

from remote_connection.remote_client import RemoteClient
from util.image_utils import Image
from pipeline.segmentation.segmentation_result import SegmentationResult


class ImageSeg(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "image_segmentation_imp.py"
        super().__init__(device=device, conda_env="frame", script_path=script_path)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["facebook/sam2.1-hiera-large"]

    def segment(self, input: Image, temp_path: Path, on_progress=None) -> SegmentationResult:
        result = self.send(
            action="segment",
            input=input.image,
            temp_path=temp_path,
            on_progress=on_progress,
        )
        return SegmentationResult(
            masks=result["masks"],
            boxes=result["boxes"],
            scores=result["scores"],
        )

    def segment_boxes(self, input: Image, boxes: list, temp_path: Path, on_progress=None) -> list:
        """Box-prompted SAM2: one mask per [x, y, w, h] box, in `boxes` order."""
        result = self.send(
            action="segment_boxes",
            input={"image": input.image, "boxes": boxes},
            temp_path=temp_path,
            on_progress=on_progress,
        )
        return result["masks"]
