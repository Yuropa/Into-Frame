import torch
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
from diffusers import FluxFillPipeline
from util.image_utils import Image
from util.device_utils import offload_pipeline

class InPaintingFlux:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        # FLUX models are heavy; 'dev' is high quality, 'schnell' is faster
        self.model_id = "black-forest-labs/FLUX.1-Fill-dev"

        self.pipeline = FluxFillPipeline.from_pretrained(
            self.model_id, 
            torch_dtype=torch_dtype
        )
        
        offload_pipeline(device, self.pipeline)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["black-forest-labs/FLUX.1-Fill-dev"]

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
        """
        For FLUX.1-Fill:
        - If prompt is "", it performs logic-based background reconstruction.
        - Guidance scale for FLUX Fill typically defaults higher (around 30.0) compared to SD.
        """

        def normalize_dimension(dim):
            return (dim // 16) * 16
        
        # Ensure images are in RGB for the pipeline
        width = normalize_dimension(input_image.width)
        height = normalize_dimension(input_image.height)

        init_img = input_image.resize((width, height), PILImage.LANCZOS)
        mask_img = mask_image.resize((width, height), PILImage.NEAREST)

        # FLUX handles the VAE encoding internally within the pipeline call
        # so we can bypass the manual latent processing used in your SD3 version.
        output = self.pipeline(
            prompt=prompt,
            image=init_img,
            mask_image=mask_img,
            height=height,
            width=width,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            max_sequence_length=512, # FLUX specific parameter
        ).images[0]

        return output

    def close(self):
        pass