import torch
from typing import Optional
from pathlib import Path
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient
from pipeline.panorama.panorama_output import PanoramaOutput


class ImagePanoramaWorldGen(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "image_panorama_worldgen_imp.py"
        super().__init__(device=device, conda_env="worldgen", script_path=script_path)

    @classmethod
    def model_names(cls) -> list[str]:
        # Base FLUX Fill model; WorldGen LoRA and depth checkpoints are fetched
        # lazily by the worldgen package via hf_hub_download at runtime.
        return ["black-forest-labs/FLUX.1-Fill-dev", "LeoXie/WorldGen"]

    def pano(
        self,
        input: PILImage,
        depth: Optional[PILImage],
        temp_path: Path,
        fov: float = 60.0,
        caption: str = "",
        seed: int = 0,
        on_progress=None,
    ) -> PanoramaOutput:
        data = {"image": input, "seed": seed, "fov": fov}
        values = self.send(action="pano", input=data, temp_path=temp_path, on_progress=on_progress)
        return PanoramaOutput(values=values)
