import torch
from pathlib import Path
from remote_connection.remote_client import RemoteClient
from util.intrinsic_utils import IntrinsicImages
from util.image_utils import Image

class ImageIntrinsics(RemoteClient):
    """
    Bridges to IntrinsicDiffusion (Luo et al., SIGGRAPH 2024), running in its own
    conda env, to predict per-pixel albedo (unlit reflectance) and surface normal
    maps from a single image.
    """

    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "image_intrinsics_imp.py"

        super().__init__(
            device=device,
            conda_env="intrinsicdiffusion",
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["ptx0/pseudo-journey-v2"]

    def intrinsic_images(
        self,
        input: Image,
        temp_path: Path,
        resolution: int = 1024,
        agg_num: int = 4,
        on_progress=None,
    ) -> IntrinsicImages:
        result = self.send(
            action="intrinsics",
            input={"image": input.rgb(), "resolution": resolution, "agg_num": agg_num},
            temp_path=temp_path,
            on_progress=on_progress,
        )
        return IntrinsicImages(result)
