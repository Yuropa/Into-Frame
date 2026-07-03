import torch
from PIL import Image as PILImage
from pipeline.inpainting.inpainting_flux import InPaintingFlux
from pipeline.inpainting.inpainting_lama import InPaintingLama
from pipeline.panorama.panorama_lora import PanoramaLoraType
from enum import Enum
from pathlib import Path

class InPaintingType(Enum):
    FLUX = 1
    LAMA = 2

    @classmethod
    def default(cls):
        return cls.FLUX

class InPainting:
    def __init__(
        self,
        device,
        torch_dtype,
        type: InPaintingType = InPaintingType.default(),
        lora_type: PanoramaLoraType | None = None,
        lora_scale: float = 1.0,
    ):
        match type:
            case InPaintingType.FLUX:
                self.generator = InPaintingFlux(
                    device=device,
                    torch_dtype=torch_dtype,
                    lora_type=lora_type,
                    lora_scale=lora_scale,
                )
            case InPaintingType.LAMA:
                self.generator = InPaintingLama(
                    device=device,
                    torch_dtype=torch_dtype
                )

    @classmethod
    def model_names(
        cls,
        type: InPaintingType = InPaintingType.default(),
        lora_type: PanoramaLoraType | None = None,
    ) -> list[str]:
        match type:
            case InPaintingType.FLUX:
                return InPaintingFlux.model_names(lora_type)
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
        strength=1.0,
        seed: int = 0,
    ) -> PILImage:
        return self.generator.inpaint(
            input_image=input_image,
            mask_image=mask_image,
            temp_path=temp_path,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            seed=seed,
        )
    
    def set_lora_enabled(self, enabled: bool) -> None:
        """Toggle a loaded LoRA on/off (FLUX only; no-op for LaMa or a LoRA-less FLUX)."""
        set_fn = getattr(self.generator, "set_lora_enabled", None)
        if set_fn is not None:
            set_fn(enabled)

    def close(self):
        self.generator.close()