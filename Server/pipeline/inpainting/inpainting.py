import torch
import numpy as np
from PIL import Image as PILImage
from diffusers import FluxInpaintPipeline
from util.image_utils import Image

class InPainting:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        # FLUX models are heavy; 'dev' is high quality, 'schnell' is faster
        self.model_id = "black-forest-labs/FLUX.1-Fill-dev"

        self.pipeline = FluxInpaintPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch_dtype
        )
        
        # self.pipeline.to(device)
        self.pipeline.enable_model_cpu_offload()

    @classmethod
    def model_names(cls) -> list[str]:
        return ["black-forest-labs/FLUX.1-Fill-dev"]

    def inpaint(self, input_image: Image, mask_image: Image, prompt: str = "", num_inference_steps=30, guidance_scale=30.0):
        """
        For FLUX.1-Fill:
        - If prompt is "", it performs logic-based background reconstruction.
        - Guidance scale for FLUX Fill typically defaults higher (around 30.0) compared to SD.
        """
        
        # Ensure images are in RGB for the pipeline
        width = (input_image.width // 16) * 16
        height = (input_image.height // 16) * 16

        init_img = input_image.resize((width, height), PILImage.LANCZOS)
        mask_img = mask_image.resize((width, height), PILImage.NEAREST)

        # FLUX handles the VAE encoding internally within the pipeline call
        # so we can bypass the manual latent processing used in your SD3 version.
        output = self.pipeline(
            prompt=prompt,
            image=init_img,
            mask_image=mask_img,
            height=init_img.height,
            width=init_img.width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512, # FLUX specific parameter
            generator=torch.Generator(device=self.device).manual_seed(42)
        ).images[0]

        return output