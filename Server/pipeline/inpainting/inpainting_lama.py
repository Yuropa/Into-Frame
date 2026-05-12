import torch
from pathlib import Path
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient 
from pipeline.model_generation.model_generation_base import ModelGeneratorBase

class InPaintingLama(RemoteClient):
    def __init__(self, device: torch.device, torch_dtype) -> None:
        script_path = Path(__file__).parent / "inpainting_lama_imp.py"

        super().__init__(
            device=device, 
            conda_env="lama", 
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return []

    def inpaint(
        self, 
        input_image: PILImage, 
        mask_image: PILImage, 
        temp_path: Path,
        prompt: str = "", 
        num_inference_steps=30, 
        guidance_scale=30.0,
        strength=1.0
    ) -> PILImage:
        data = {
            "image": input_image,
            "mask": mask_image,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "strength": strength
        }
        return self.send(action="inpaint", input=data, temp_path=temp_path)