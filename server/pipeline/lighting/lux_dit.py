import torch
from pathlib import Path
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient


class LightingResult:
    def __init__(self, result: dict) -> None:
        self.ldr = result["ldr"]  # PIL Image — LDR environment map
        self.log = result["log"]  # PIL Image — log-domain environment map
        # (H, W, 3) float32 linear radiance from LuxDiT's own HDR merger, or None
        # if it was unavailable/rejected server-side. This is the only source of
        # real dynamic range: both images above are 8-bit, and env_log's encoding
        # is learned rather than analytic, so it cannot be inverted without the
        # merger. See LuxDiTServer._merge_hdr.
        self.hdr = result.get("hdr")
        # Server-side account of why the merge did or didn't run (missing weights,
        # a load failure, a rejected reconstruction, or success with its measured
        # range). The server's own prints go to a stdout the pipeline log doesn't
        # capture, so without this an LDR-only fallback -- which costs several
        # stops of key light -- is indistinguishable from a working one.
        self.hdr_status = result.get("hdr_status")


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
