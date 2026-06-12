import torch
from pathlib import Path
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient


class RecognizeResult:
    def __init__(self, result: dict) -> None:
        self.tags: str = result["tags"]  # pipe-separated English tags, e.g. "dog | tree | outdoor"


class ImageRecognizeRAMPlus(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "image_recognize_ram_plus_imp.py"
        super().__init__(
            device=device,
            conda_env="frame",
            script_path=script_path,
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return []  # Downloaded via setup.sh, not a HuggingFace snapshot

    def recognize(self, input: PILImage.Image, temp_path: Path, on_progress=None) -> RecognizeResult:
        result = self.send(action="recognize", input=input, temp_path=temp_path, on_progress=on_progress)
        return RecognizeResult(result)
