from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from util.json_utils import write_json
from remote_connection.remote_server import RemoteServer
from diffusers import FluxTransformer2DModel
from transformers import T5EncoderModel, BitsAndBytesConfig
import torch.nn.functional as F
from quanto import freeze, qfloat8, quantize

from pa_src.pipeline import RFPanoInversionParallelFluxPipeline
from pa_src.attn_processor import PersonalizeAnythingAttnProcessor, set_flux_transformer_attn_processor
from pa_src.utils import *

class SafePersonalizeAnythingAttnProcessor(PersonalizeAnythingAttnProcessor):
    """
    Wraps PersonalizeAnythingAttnProcessor to handle None timestep.
    When timestep is None, we default t_flag=True (always do token replacement),
    which is the correct behavior during generation.
    """
    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, timestep=None):
        if timestep is None:
            timestep = 1.0  # > any tau value, so t_flag is always True
        return super().__call__(
            attn, hidden_states, encoder_hidden_states,
            attention_mask, image_rotary_emb, timestep
        )

class PanoGenerator(RemoteServer):
    def setup(self):
        if self.device.type == "cuda":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        transformer = FluxTransformer2DModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="transformer",
            torch_dtype=self.dtype
        )

        text_encoder_2 = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            subfolder="text_encoder_2",
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        )

        self.pipeline = RFPanoInversionParallelFluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            text_encoder_2=text_encoder_2,
            torch_dtype=self.dtype
        ).to(self.device)

        self.pipeline.text_encoder = self.pipeline.text_encoder.to(dtype=self.dtype)

        original_transformer_forward = self.pipeline.transformer.forward
        def transformer_forward_cast(*args, **kwargs):
            args = tuple(a.to(dtype=self.dtype) if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: v.to(dtype=self.dtype) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            return original_transformer_forward(*args, **kwargs)
        self.pipeline.transformer.forward = transformer_forward_cast

        # Load LoRA BEFORE quantizing so state dict keys are still intact
        self.pipeline.load_lora_weights(
            "Insta360-Research/DiT360-Panorama-Image-Generation",
            torch_dtype=self.dtype,
        )

        # Ensure uniform dtype before quantizing
        self.pipeline.transformer = self.pipeline.transformer.to(dtype=self.dtype)

        # Quantize and freeze AFTER LoRA is loaded
        quantize(self.pipeline.transformer, weights=qfloat8)
        freeze(self.pipeline.transformer)

        # T5 8-bit lands on CUDA but tokenizer output is always CPU;
        # hook forward to move all tensor inputs to the encoder's device
        t5 = self.pipeline.text_encoder_2
        original_t5_forward = t5.forward
        def t5_forward_on_device(*args, **kwargs):
            device = next(t5.parameters()).device
            args = tuple(a.to(device) if isinstance(a, torch.Tensor) else a for a in args)
            kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
            return original_t5_forward(*args, **kwargs)
        t5.forward = t5_forward_on_device

        original_vae_decode = self.pipeline.vae.decode
        def vae_decode_cast(z, *args, **kwargs):
            return original_vae_decode(z.to(dtype=self.dtype), *args, **kwargs)
        self.pipeline.vae.decode = vae_decode_cast

        self.pipeline.vae.enable_tiling()
        self.pipeline.vae.enable_slicing()
    
    def create_mask(self, input_image: Image.Image, packed_w: int, packed_h: int):
        """
        Creates a mask where the region corresponding to the input image's
        aspect ratio (centered) is 1 (preserved) and outpainted areas are 0.
        """
        input_w, input_h = input_image.size

        input_aspect = input_w / input_h
        pano_aspect = 2.0
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
        latent_h = height // vae_scale_factor
        latent_w = width // vae_scale_factor

        packed_h = latent_h // 2
        packed_w = latent_w // 2
        img_dims = packed_h * (packed_w + 2)

        prompt = "This is a panorama image. The image depicts a view of paris from the river with the eiffle tour in the background. high resolution, 8k, seamless."

        mask_2d = self.create_mask(input_image, packed_w, packed_h)
        init_image = input_image.convert("RGB").resize((width, height))

        mask_padded = torch.cat(
            [mask_2d[:, :1], mask_2d, mask_2d[:, -1:]], dim=-1
        )
        mask_flattened = mask_padded.reshape(-1, 1)

        # Build the latent-space mask for the pipeline's built-in blending.
        # Shape must match packed latents: (packed_h * (packed_w + 2), dim)
        # but the pipeline indexes it as latents[1] which is (seq_len, dim),
        # so we need it as (seq_len, 1) broadcastable
        latent_mask = mask_padded.reshape(-1, 1).to(
            device=self.device, dtype=self.dtype
        )  # (img_dims, 1)

        inverted_latents, image_latents, latent_image_ids = self.pipeline.invert(
            source_prompt=prompt,
            image=init_image,
            height=height,
            width=width,
            num_inversion_steps=timestep,
            gamma=0.5,
        )

        set_flux_transformer_attn_processor(
            self.pipeline.transformer,
            set_attn_proc_func=lambda name, dh, nh, ap: SafePersonalizeAnythingAttnProcessor(
                name=name,
                tau=tau,
                mask=mask_flattened,
                device=self.device,
                img_dims=img_dims,
            ),
        )

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
            mask=latent_mask,       # pipeline's built-in latent blending
            use_timestep=True,      # passes timestep to attention processor
        ).images[1]

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            print(f"Got input image {input}")
            return self.pano(temp_path, input)
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    PanoGenerator.run()