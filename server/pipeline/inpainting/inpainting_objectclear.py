import torch
from pathlib import Path
from PIL import Image as PILImage
from remote_connection.remote_client import RemoteClient


class InPaintingObjectClear(RemoteClient):
    def __init__(self, device: torch.device, torch_dtype) -> None:
        script_path = Path(__file__).parent / "inpainting_objectclear_imp.py"
        super().__init__(device=device, conda_env="objectclear", script_path=script_path)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["jixin0101/ObjectClear"]

    def inpaint(
        self,
        input_image: PILImage,
        mask_image: PILImage,
        temp_path: Path,
        prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 2.5,
        strength: float = 1.0,  # unused — ObjectClear has no img2img strength knob
        seed: int = 0,
    ) -> PILImage:
        data = {
            "image": input_image,
            "mask": mask_image,
            # ObjectClear is instruction-prompted; fall back to its documented
            # default removal instruction when the caller doesn't supply one.
            "prompt": prompt or "remove the instance of object",
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
        }
        return self.send(action="inpaint", input=data, temp_path=temp_path)
