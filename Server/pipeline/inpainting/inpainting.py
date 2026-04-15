from util.image_utils import Image
from pipeline.inpainting.sd3_impls import SD3LatentFormat
import torch
import numpy as np
from diffusers import StableDiffusion3Pipeline

class InPainting:
    def __init__(self, device, torch_dtype):
        self.device = device
        self.torch_dtype = torch_dtype
        self.width = 1024
        self.height = 1024

        self.pipeline = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3.5-medium", 
            torch_dtype=torch_dtype
        )
        self.pipeline = self.pipeline.to(device)

    # python sd3_infer.py --controlnet_ckpt models/sd3.5_large_controlnet_canny.safetensors 
    # --controlnet_cond_image input/canny.png --prompt "An adorable fluffy pastel creature"

    @classmethod
    def model_names(cls) -> list[str]:
        return ["stabilityai/stable-diffusion-3.5-medium"]

    def _vae_encode(
        self, image, using_2b_controlnet: bool = False, controlnet_type: int = 0
    ) -> torch.Tensor:
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images).cuda()
        if using_2b_controlnet:
            image_torch = image_torch * 2.0 - 1.0
        elif controlnet_type == 1:  # canny
            image_torch = image_torch * 255 * 0.5 + 0.5
        else:
            image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.to(self.device)
        self.vae.model = self.vae.model.to(self.device)
        latent = self.vae.model.encode(image_torch).cpu()
        self.vae.model = self.vae.model.cpu()
        return latent

    def _image_to_latent(self, image):
        image = image.resize((self.width, self.height), Image.LANCZOS)
        latent = self._vae_encode(image, using_2b_controlnet=False, controlnet_type=1)
        latent = SD3LatentFormat().process_in(latent)

    def inpaint(self, input: Image, prompt: str, num_inference_steps = 28, guidance_scale = 3.5):
        if input is None:
            return self.pipeline(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]
        else:
            return self.pipeline(
                prompt,
                image=[input.image],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            ).images[0]