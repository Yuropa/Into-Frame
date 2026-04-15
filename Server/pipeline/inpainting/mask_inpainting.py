import torch
import numpy as np
from PIL import Image as PILImage
from diffusers import StableDiffusion3InpaintPipeline
from util.image_utils import Image

class MaskInPainting:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.width = 1024
        self.height = 1024

        self.pipeline = StableDiffusion3InpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch_dtype
        )
        self.pipeline = self.pipeline.to(device)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["stabilityai/stable-diffusion-3.5-medium"]

    def inpaint(
        self, 
        image: Image, 
        mask_image: Image, 
        prompt: str = "", 
        negative_prompt: str = "",
        num_inference_steps: int = 28, 
        guidance_scale: float = 3.5,
        strength: float = 0.8  # Higher strength = more "prediction" of what's behind
    ):
        original_width = image.width
        original_height = image.height
        # Ensure images are correctly sized
        image = image.image.convert("RGB").resize((self.width, self.height))
        mask_image = mask_image.image.convert("L").resize((self.width, self.height))

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            height=self.height,
            width=self.width
        ).images[0]

        result = result.resize((original_width, original_height))
        return Image(result)