import torch
from PIL import Image as PILImage
from pipeline.inpainting.inpainting_flux import InPaintingFlux
from pipeline.inpainting.inpainting_lama import InPaintingLama
from enum import Enum
from pathlib import Path

class InPaintingType(Enum):
    FLUX = 1
    LAMA = 2

    @classmethod
    def default(cls):
        return cls.FLUX

class InPainting:
    def __init__(self, device, torch_dtype, type: InPaintingType = InPaintingType.default()):
        match type:
            case InPaintingType.FLUX:
                self.generator = InPaintingFlux(
                    device=device,
                    torch_dtype=torch_dtype
                )
            case InPaintingType.LAMA:
                self.generator = InPaintingLama(
                    device=device,
                    torch_dtype=torch_dtype
                )

    @classmethod
    def model_names(cls, type: InPaintingType = InPaintingType.default()) -> list[str]:
        match type:
            case InPaintingType.FLUX:
                return InPaintingFlux.model_names()
            case InPaintingType.LAMA:
                return InPaintingLama.model_names()

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
        return self.generator.inpaint(
            input_image=input_image,
            mask_image=mask_image,
            temp_path=temp_path,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        )
    
    def close(self):
        self.generator.close()