import torch
from pathlib import Path
import numpy as np
from io import BytesIO
from PIL import Image
from remote_connection.remote_client import RemoteClient 
from pipeline.model_generation.model_generation_base import ModelGeneratorBase

class ImagePanorama(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "image_panorama_imp.py"

        super().__init__(
            device=device, 
            conda_env="dit360", 
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["black-forest-labs/FLUX.1-dev"]

    def pano(self, input: Image, temp_path: Path) -> DepthResult:
        return self.send(action="pano", input=self.encode_image(input), temp_path=temp_path)