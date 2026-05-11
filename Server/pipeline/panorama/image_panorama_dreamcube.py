import torch
from pathlib import Path
from PIL import Image as PILImage
from util.image_utils import Image
from util.cubemap_utils import CubeMap
from remote_connection.remote_client import RemoteClient 
from pipeline.model_generation.model_generation_base import ModelGeneratorBase
from pipeline.panorama.panorama_output import PanoramaOutput

class ImagePanoramaDreamCube(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "image_panorama_dreamcube_imp.py"

        super().__init__(
            device=device, 
            conda_env="dreamcube", 
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["KevinHuang/DreamCube"]

    def pano(self, input: PILImage, depth: PILImage, temp_path: Path, fov: float = 60.0, caption: str = "") -> PanoramaOutput:
        data = {
            "image": input,
            "fov_degrees": fov,
            "caption": caption,
            "depth": depth
        }
        values = self.send(action="pano", input=data, temp_path=temp_path)
        return PanoramaOutput(values=values)