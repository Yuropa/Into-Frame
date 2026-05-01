from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from util.json_utils import write_json
from remote_connection.remote_server import RemoteServer
from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel

from pa_src.pipeline import RFPanoInversionParallelFluxPipeline
from pa_src.attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor
from pa_src.utils import *

class PanoGenerator(RemoteServer):
    def setup(self):
        self.dtype = torch.float16
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
        
        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=self.dtype  # was: torch.float8_e4m3fn
        )

        self.pipeline = RFPanoInversionParallelFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=self.dtype
        ).to(device=self.device)

        self.pipeline.load_lora_weights(
            "Insta360-Research/DiT360-Panorama-Image-Generation"
        )
    
    def create_mask(self, input_image: Image.Image, packed_w: int, packed_h: int):
        """
        Creates a mask where the region corresponding to the input image's
        aspect ratio (centered) is 1 (preserved) and outpainted areas are 0.
        """
        input_w, input_h = input_image.size  # original size, before resize

        # What fraction of the 2:1 panorama width does the input occupy?
        # Clamp to (0, 1] so an ultrawide input doesn't overflow
        input_aspect = input_w / input_h
        pano_aspect = 2.0  # 2048 / 1024
        occupied_fraction = min(input_aspect / pano_aspect, 1.0)

        preserved_tokens = round(packed_w * occupied_fraction)
        margin = (packed_w - preserved_tokens) // 2

        mask = torch.zeros((packed_h, packed_w), device=self.device)
        mask[:, margin : margin + preserved_tokens] = 1.0
        return mask

    def pano(self, temp_path: Path, input_image: Image.Image) -> Any:
        timestep = 50
        seed = 42
        guidance = 2.8
        tau = 0.5
        height, width = 1024, 2048

        vae_scale_factor = 8
        latent_h = height // vae_scale_factor  # 128
        latent_w = width // vae_scale_factor   # 256

        # Flux packs latents into (h/2, w/2) patches, then DiT360 adds
        # 2 extra columns of tokens for circular padding
        packed_h = latent_h // 2  # 64
        packed_w = latent_w // 2  # 128
        img_dims = packed_h * (packed_w + 2)

        prompt = "This is a panorama image. The image depicts a village next to a snow-capped mountain, high resolution, 8k, seamless."

        # 1. Build mask from original size BEFORE resizing
        mask_2d = self.create_mask(input_image, packed_w, packed_h)

        # 2. Prepare image
        init_image = input_image.convert("RGB").resize((width, height))

        # 3. Pad mask horizontally by 1 token on each side to match circular
        #    attention (+2 columns), then flatten to [img_dims, 1]
        mask_padded = torch.cat(
            [mask_2d[:, :1], mask_2d, mask_2d[:, -1:]], dim=-1
        )  # (packed_h, packed_w + 2)
        mask_flattened = mask_padded.reshape(-1, 1)  # (img_dims, 1)

        # 4. Inversion — pass the PIL image directly; the pipeline handles
        #    VAE encoding and dtype internally
        inverted_latents, image_latents, latent_image_ids = self.pipeline.invert(
            source_prompt="",
            image=init_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=1.0,
        )

        # 5. Inject the 360-aware attention processor
        set_flux_transformer_attn_processor(
            self.pipeline.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: PersonalizeAnythingAttnProcessor(
                name=name,
                tau=tau,
                mask=mask_flattened,
                device=self.device,
                img_dims=img_dims,
            ),
        )

        # 6. Generate
        generator = torch.Generator(device=self.device).manual_seed(seed)

        result = self.pipeline(
            prompt=[prompt, prompt],
            inverted_latents=inverted_latents,
            image_latents=image_latents,
            latent_image_ids=latent_image_ids,
            height=height,
            width=width,
            guidance_scale=guidance,
            num_inference_steps=timestep,
            generator=generator,
            output_type="pil",
        ).images[1]

        return result

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            print(f"Got input image {input}")
            return self.pano(temp_path, input)
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    PanoGenerator.run()