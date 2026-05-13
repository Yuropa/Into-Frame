from path_utils import add_project_paths, add_system_path, lib_path
add_project_paths()

add_system_path(lib_path() / "DreamCube")

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import py360convert
import traceback
import torch
from einops import rearrange

from remote_connection.remote_server import RemoteServer
from models.dreamcube import DreamCubeDepthPipeline
from diffusers import DiffusionPipeline, AutoencoderKL
from models.multiplane_sync_legacy import apply_custom_processors_for_vae, apply_custom_processors_for_unet
from app import prepare_inputs

class PanoGenerator(RemoteServer):
    def setup(self):
        self.USE_STABLE_DIFFUSION = False

        if self.USE_STABLE_DIFFUSION:
            self.pipeline = pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                vae=AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16),
                torch_dtype=torch.float16,
                variant="fp16",
            )

            apply_custom_processors_for_unet(
                self.pipeline.unet,
                enable_sync_self_attn=True,
                enable_sync_cross_attn=False,
                enable_sync_conv2d=True,
                enable_sync_gn=True,
                cube_padding_impl='ref',
            )
            apply_custom_processors_for_vae(
                self.pipeline.vae,
                enable_sync_attn=True,
                enable_sync_gn=True,
                enable_sync_conv2d=True,
                cube_padding_impl='ref',
            )

            self.pipeline.to(self.device)
        else:
            self.pipeline = DreamCubeDepthPipeline.from_pretrained(
                "KevinHuang/DreamCube"
            )

            apply_custom_processors_for_unet(
                self.pipeline.unet,
                enable_sync_self_attn=True,
                enable_sync_cross_attn=False,
                enable_sync_gn=True,
                enable_sync_conv2d=True,
                cube_padding_impl='ref',
            )
            apply_custom_processors_for_vae(
                self.pipeline.vae,
                mode='all',
                enable_sync_gn=True,
                enable_sync_conv2d=True,
                enable_sync_attn=True,
                cube_padding_impl='ref',
            )

            self.pipeline.to(self.device)

    def pano(self, temp_path: Path, input_image: Image.Image, depth_image: Image.Image, caption: str = "") -> dict:
        if isinstance(depth_image, np.ndarray):
            depth_image = Image.fromarray(depth_image)
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        # Resize inputs to 512x512
        image_size = 768 if self.USE_STABLE_DIFFUSION else 512
        input_image = input_image.resize((image_size, image_size))
        depth_image = depth_image.resize((image_size, image_size))

        if caption is None:
            caption = ""

        face_keys = ["F", "R", "B", "L", "U", "D"]

        if self.USE_STABLE_DIFFUSION:
            prompts = [caption] * 6
            latents = torch.randn(6, 4, image_size // 8, image_size // 8).to(self.device, dtype=torch.float16)

            with torch.inference_mode():
                with torch.autocast('cuda'):
                    images = self.pipeline(prompts, latents=latents, output_type='np').images

            images = rearrange(images, '(b m) h w c -> b m h w c', m=6)[0]  # (6, H, W, C)

            # py360convert dict format expects: F, R, B, L, U, D
            face_keys = ["F", "R", "B", "L", "U", "D"]
            cube_dict = {k: (images[i] * 255).round().astype("uint8") for i, k in enumerate(face_keys)}
        else:
            # Build per-face prompts using DreamCube's expected prefix format
            face_views = ["Front", "Right", "Back", "Left", "Up", "Down"]
            prompts = [f"{caption}" for _ in face_views]

            cube_rgbs, cube_depths, cube_masks, cube_prompts = prepare_inputs(
                image=input_image,
                depth=depth_image,
                prompts=prompts,
                device=self.device,
            )

            from torch.amp.autocast_mode import autocast
            with torch.inference_mode():
                with autocast('cuda'):
                    prediction = self.pipeline(
                        cube_rgbs=cube_rgbs,
                        cube_depths=cube_depths,
                        cube_masks=cube_masks,
                        prompt=cube_prompts,
                        height=512,
                        width=512,
                        guidance_scale=7.5,
                        num_inference_steps=50,
                        output_type='np',
                        normalize_scale=0.6,
                    )

            images_pred = prediction.images  # (B*6, H, W, C), float32
            images_pred = (images_pred * 255).round().astype("uint8")

            images_pred = rearrange(images_pred, '(b m) h w c -> b m h w c', m=6)[0]  # (6, H, W, C)

            # DreamCube face order: front, right, back, left, up, down
            # py360convert dict format expects: F, R, B, L, U, D
            cube_dict = {k: images_pred[i] for i, k in enumerate(face_keys)}

        # Save individual faces
        for i, (k, face) in enumerate(cube_dict.items()):
            Image.fromarray(face).save(str(temp_path / f"face_{k}.png"))

        # Stitch to equirectangular
        equirectangular = py360convert.c2e(cube_dict, h=1024, w=2048, cube_format='dict')
        equirectangular = np.clip(equirectangular, 0, 255).astype(np.uint8)

        return {
            "image": Image.fromarray(equirectangular),
            "faces": {k: Image.fromarray(v) for k, v in cube_dict.items()},
        }

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            try:
                print(f"Got input: {input}")
                image = input["image"]
                depth = input["depth"]
                caption = input.get("caption", "")
                result = self.pano(temp_path, image, depth, caption=caption)

                print(f"Got dream cube values {result}")
                return result
            except Exception as e:
                print(f"Unable to generate dream cube values: {e}")
                traceback.print_exc()
                raise e
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    PanoGenerator.run()