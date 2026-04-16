import torch
import numpy as np
from PIL import Image as PILImage
from PIL import ImageFilter
from diffusers import StableDiffusion3InpaintPipeline
from diffusers import FluxFillPipeline, GGUFQuantizationConfig
from util.image_utils import Image
from util.device_utils import offload_pipeline

class MaskInPainting_Stable_Diffusion:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.width = 1024
        self.height = 1024

        self.pipeline = StableDiffusion3InpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium",
            torch_dtype=torch_dtype
        )
        offload_pipeline(device, self.pipeline)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["stabilityai/stable-diffusion-3.5-medium"]

    def inpaint(
        self, 
        image: Image, 
        mask_image: Image, 
        prompt: str = "background, high quality, seamless texture and environment", 
        negative_prompt: str = "deformed, ugly, distorted, low quality, object, person",
        num_inference_steps: int = 20, 
        guidance_scale: float = 2.5,
        padding_mask_crop: int = 64,
        strength: float = 0.85  # Higher strength = more "prediction" of what's behind
    ):
        original_width = image.width
        original_height = image.height
        # Ensure images are correctly sized
        image = image.rgb(copy=True).resize((self.width, self.height))
        mask_image = mask_image.L(copy=True).resize((self.width, self.height))

        result = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            padding_mask_crop=padding_mask_crop,
            height=self.height,
            width=self.width
        ).images[0]

        result = result.resize((original_width, original_height))
        return Image(result)
    
class MaskInPainting:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype

        self.pipeline = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            # "nunchaku-ai/nunchaku-flux.1-fill-dev",
            torch_dtype=torch.bfloat16
        )
        offload_pipeline(device, self.pipeline)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["black-forest-labs/FLUX.1-Fill-dev"]

    def _prepare_dimensions(self, width, height, target_size: int = 1024):
        # Ensure dimensions are multiples of 64 for the Transformer
        # while keeping the longest side at 1024
        scale = target_size / max(width, height)
        new_w = int((width * scale) // 64) * 64
        new_h = int((height * scale) // 64) * 64
        return new_w, new_h

    def inpaint(
        self, 
        image: Image, 
        mask_image: Image,
        prompt: str = "",
        num_inference_steps: int = 20, 
        guidance_scale: float = 30.0
    ):
        original_width = image.width
        original_height = image.height

        target_width, target_height = self._prepare_dimensions(original_width, original_height)
    
        # Resize to the new proportional dimensions
        input_image = image.rgb(copy=True).resize((target_width, target_height))
        input_mask = mask_image.rgb(copy=True).resize((target_width, target_height))

        result = self.pipeline(
            prompt=prompt,
            image=input_image,
            mask_image=input_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=target_height,
            width=target_width
        ).images[0]

        result = result.resize((original_width, original_height))
        return Image(result)
    
    def _crop_box(self, box, padding, img_w, img_h):
        x, y, w, h = box
        left = max(0, x - padding)
        top = max(0, y - padding)
        right = min(img_w, x + w + padding)
        bottom = min(img_h, y + h + padding)
        return (left, top, right, bottom)

    def inpaint_crop(
        self, 
        image: Image, 
        mask_image: Image, 
        box, 
        prompt: str = "high quality background",
        num_inference_steps: int = 4, 
        guidance_scale: float = 3.5,
        padding: float = 128
    ):
        img = image.rgb(copy=True)
        mask = mask_image.L(copy=True)

        original_width = image.width
        original_height = image.height

        crop_box = self._crop_box(box=box, padding=padding, img_w=original_width, img_h=original_height)
        image_crop = img.crop(crop_box)
        mask_crop = mask.crop(crop_box)
        
        crop_width, crop_height = image_crop.size

        target_width, target_height = self._prepare_dimensions(crop_width, crop_height)
        input_image = image_crop.resize((target_width, target_height), PILImage.LANCZOS)
        input_mask = mask_crop.resize((target_width, target_height), PILImage.NEAREST)

        inpainted_crop = self.pipeline(
            prompt=prompt,
            image=input_image,
            mask_image=input_mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]

        final_crop = inpainted_crop.resize((crop_width, crop_height), PILImage.LANCZOS)
        mask_blend = mask_crop.filter(ImageFilter.GaussianBlur(radius=10))
        blended_crop = PILImage.composite(final_crop, image_crop, mask_blend)
        
        # We create a copy to avoid mutating the original image
        output_image = img.copy()
        output_image.paste(blended_crop, (crop_box[0], crop_box[1]))
        
        return Image(output_image)