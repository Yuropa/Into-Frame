import torch
from pathlib import Path
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient


class DepthResult:
    def __init__(self, result: dict) -> None:
        self.depth = result["depth"]        # float32 numpy array, values in metres
        self.sky_mask = result.get("sky_mask")  # bool (H, W) array, True = sky; None if unavailable


class PanoramaDepth(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "pano_depth_imp.py"
        super().__init__(
            device=device,
            conda_env="depthpano",
            script_path=script_path,
        )

    @classmethod
    def model_names(cls) -> list[str]:
        # Loaded from a local checkpoint — not a HuggingFace repo
        return []

    def depth(self, input: PILImage.Image, temp_path: Path, on_progress=None) -> DepthResult:
        result = self.send(action="depth", input=input, temp_path=temp_path, on_progress=on_progress)
        return DepthResult(result)
