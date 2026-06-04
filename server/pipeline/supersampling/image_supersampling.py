import torch
from pathlib import Path
from remote_connection.remote_client import RemoteClient
from util.image_utils import Image


class ImageSupersampling(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "image_supersampling_imp.py"
        super().__init__(device=device, conda_env="frame", script_path=script_path)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["caidas/swin2SR-classical-sr-x2-64"]

    def supersample(self, image: Image, temp_path: Path, on_progress=None) -> Image:
        result = self.send(action="supersample", input=image.image, temp_path=temp_path, on_progress=on_progress)
        return Image(result)
