from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import py360convert

from remote_connection.remote_server import RemoteServer
from dreamcube.models.dreamcube import DreamCubeDepthPipeline
from dreamcube.models.multiplane_sync_legacy import apply_custom_processors_for_vae, apply_custom_processors_for_unet


class PanoGenerator(RemoteServer):
    def setup(self):
        self.pipeline = DreamCubeDepthPipeline.from_pretrained(
            "KevinHuang/DreamCube"
        )

        apply_custom_processors_for_unet(
            self.pipeline.unet,
            enable_sync_self_attn=True,
            enable_sync_cross_attn=False,
            enable_sync_gn=True,
            enable_sync_conv2d=True,
            cube_padding_impl='cuda',
        )
        apply_custom_processors_for_vae(
            self.pipeline.vae,
            mode='all',
            enable_sync_gn=True,
            enable_sync_conv2d=True,
            enable_sync_attn=True,
            cube_padding_impl='cuda',
        )

        self.pipeline.to(self.device)

    def pano(self, temp_path: Path, input_image: Image.Image, depth_image: Image.Image, caption: str = "") -> dict:
        # Resize inputs to 512x512
        input_image = input_image.resize((512, 512))
        depth_image = depth_image.resize((512, 512))

        # Build per-face prompts using DreamCube's expected prefix format
        face_views = ["Front", "Right", "Back", "Left", "Up", "Down"]
        prompts = [f"{caption}" for _ in face_views]

        cube_rgbs, cube_depths, cube_masks, cube_prompts = prepare_inputs(
            image=input_image,
            depth=depth_image,
            prompts=prompts,
            device=self.device,
        )

        import torch
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

        from einops import rearrange
        images_pred = rearrange(images_pred, '(b m) h w c -> b m h w c', m=6)[0]  # (6, H, W, C)

        # DreamCube face order: front, right, back, left, up, down
        # py360convert dict format expects: F, R, B, L, U, D
        face_keys = ["F", "R", "B", "L", "U", "D"]
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
            print(f"Got input: {input}")
            image = input["image"]
            depth = input["depth"]
            caption = input.get("caption", "")
            return self.pano(temp_path, image, depth, caption=caption)
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    PanoGenerator.run()