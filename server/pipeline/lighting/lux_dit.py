import torch
from pathlib import Path
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient


class LightingResult:
    def __init__(self, result: dict) -> None:
        self.ldr = result["ldr"]  # PIL Image — LDR environment map
        self.log = result["log"]  # PIL Image — log-domain environment map


class LuxDiT(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "lux_dit_imp.py"
        super().__init__(
            device=device,
            conda_env="lux-dit",
            script_path=script_path,
        )

    @classmethod
    def model_names(cls) -> list[str]:
        # Downloaded via download_weights.py in setup.sh, not a HF snapshot
        return []

    def estimate(self, input: PILImage.Image, temp_path: Path, on_progress=None) -> LightingResult:
        result = self.send(action="estimate", input=input, temp_path=temp_path, on_progress=on_progress)
        return LightingResult(result)
