import torch
from pathlib import Path
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient


class GroundingDino(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "grounding_dino_imp.py"
        super().__init__(
            device=device,
            conda_env="frame",
            script_path=script_path,
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["IDEA-Research/grounding-dino-base"]

    def detect(self, image: PILImage.Image, tags: list[str], temp_path: Path, on_progress=None) -> list[dict]:
        result = self.send(
            action="detect",
            input={"image": image, "tags": tags},
            temp_path=temp_path,
            on_progress=on_progress,
        )
        return result["detections"]
