import torch
from pathlib import Path
from PIL import Image
from remote_connection.remote_client import RemoteClient 
from pipeline.model_generation.model_generation_base import ModelGeneratorBase

class ImagePanorama(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "image_panorama_imp.py"

        super().__init__(
            device=device, 
            conda_env="cubediff", 
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["hlicai/cubediff-512-singlecaption"]

    def pano(self, input: Image, temp_path: Path, fov: float = 60.0, caption: str = "") -> Image:
        data = {
            "image": input,
            "fov_degrees": fov,
            "caption": caption
        }
        return self.send(action="pano", input=data, temp_path=temp_path)