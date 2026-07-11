import torch
from pathlib import Path
from typing import Optional
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient


class LTX2VideoGenerator(RemoteClient):
    def __init__(self, device: torch.device, offload_mode: str = "none", quantization: Optional[str] = None) -> None:
        script_path = Path(__file__).parent / "ltx2_client_imp.py"
        env_options = {"LTX2_OFFLOAD_MODE": offload_mode}
        if quantization:
            env_options["LTX2_QUANTIZATION"] = quantization

        super().__init__(
            device=device,
            conda_env="ltx2",
            script_path=script_path,
            env_options=env_options,
        )

    @classmethod
    def model_names(cls) -> list[str]:
        # Downloaded via `hf download` in setup.sh, not a HF snapshot
        return []

    def generate(
        self,
        image: PILImage.Image,
        prompt: str,
        negative_prompt: str,
        temp_path: Path,
        seed: int = 0,
        width: int = 768,
        height: int = 512,
        num_frames: int = 121,
        frame_rate: float = 24.0,
        num_inference_steps: int = 30,
        on_progress=None,
    ) -> str:
        return self.send(
            action="generate",
            input={
                "image": image,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "num_inference_steps": num_inference_steps,
            },
            temp_path=temp_path,
            on_progress=on_progress,
        )
