from path_utils import add_project_paths, add_system_path, lib_path, checkpoints_path
add_project_paths()

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image
import py360convert
import traceback
import torch
from einops import rearrange
from PIL import ImageFilter

from remote_connection.remote_server import RemoteServer
from diffusers import FluxInpaintPipeline
from util.device_utils import offload_pipeline

class PanoGenerator(RemoteServer):
    def setup(self):
        # Load base pipeline
        self.pipeline = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
        )

        self.pipeline.vae.enable_tiling()
        self.pipeline.vae.enable_slicing()

        self.pipeline.load_lora_weights(
            str(checkpoints_path() / "layer_pano_3d"), 
            weight_name="pano_lora_720*1440_v1.safetensors",
            adapter_name="pano"
        )
        
        offload_pipeline(self.device, self.pipeline)

    def make_outpaint_mask(
        self,
        equi_size: tuple,
        input_image: Image.Image,
        hfov_deg: float = 90.0,
        blend_width: int = 180, # How many pixels to use for the transition
    ) -> tuple[Image.Image, Image.Image, tuple]:
        equi_w, equi_h = equi_size
        target_w = int((hfov_deg / 360.0) * equi_w)
        target_h = int((hfov_deg * (input_image.height / input_image.width) / 180.0) * equi_h)

        input_resized = input_image.resize((target_w, target_h), Image.LANCZOS)
        
        # Create the background: Use a blurred version of the input to 'fill' the room colors
        canvas = input_resized.resize(equi_size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=50))
        x = (equi_w - target_w) // 2
        y = (equi_h - target_h) // 2
        canvas.paste(input_resized, (x, y))

        # --- NEW GRADIENT MASK LOGIC ---
        # 255 (White) is "Change Me", 0 (Black) is "Keep Me"
        mask = Image.new("L", equi_size, 255)
        
        # Create a mask for just the input area that is black (0)
        # But we make it slightly smaller than the actual image to ensure the edges are inpainted
        inner_w = target_w - (blend_width * 2)
        inner_h = target_h - (blend_width * 2)
        inner_x = x + blend_width
        inner_y = y + blend_width
        
        # Draw the solid 'keep' area
        mask.paste(Image.new("L", (inner_w, inner_h), 0), (inner_x, inner_y))
        
        # Use a massive blur to turn that hard rectangle into a smooth gradient
        # This creates a 'feathered' gray area where the AI blends with the original
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_width // 2))

        return canvas, mask, (x, y, target_w, target_h)

    def pano(self, temp_path: Path, input_image: Image.Image, depth_image: Image.Image, fov: float = 60.0, caption: str = "") -> dict:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        equi_size = (2048, 1024)
        prompt = caption + ", 360 degree equirectangular panorama, seamless, hyper-detailed, sharp focus, 8k resolution"

        # --- Single High-Quality Pass with Soft Masking ---
        canvas, mask, (x, y, iw, ih) = self.make_outpaint_mask(equi_size, input_image, fov)

        canvas.save(str(temp_path / "prepared_canvas.png"))
        mask.save(str(temp_path / "soft_mask.png"))

        with torch.inference_mode():
            # Using a slightly lower strength (0.85 - 0.92) allows the model to 
            # stay "grounded" in the colors of your original image while filling the rest.
            output = self.pipeline(
                prompt=prompt,
                image=canvas,
                mask_image=mask,
                height=equi_size[1],
                width=equi_size[0],
                strength=0.82, 
                guidance_scale=3.5,
                num_inference_steps=50,
                output_type='np',
            ).images[0]

        equirectangular = (output * 255).round().astype("uint8")
        equi_pil = Image.fromarray(equirectangular)
        equi_pil.save(str(temp_path / "equirectangular.png"))

        # Extract cube faces (this handles the wrapping/seams check)
        cube_dict = py360convert.e2c(equirectangular, face_w=512, cube_format='dict')

        for k, face in cube_dict.items():
            Image.fromarray(np.clip(face, 0, 255).astype("uint8")).save(str(temp_path / f"face_{k}.png"))

        return {
            "image": equi_pil,
            "faces": {k: Image.fromarray(np.clip(v, 0, 255).astype("uint8")) for k, v in cube_dict.items()},
        }

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            try:
                print(f"Got input: {input}")
                image = input["image"]
                depth = input["depth"]
                caption = input.get("caption", "")
                fov = input.get("fov_degrees", 60.0) * 2.0
                result = self.pano(temp_path, image, depth, fov=fov, caption=caption)

                print(f"Got dream cube values {result}")
                return result
            except Exception as e:
                print(f"Unable to generate dream cube values: {e}")
                traceback.print_exc()
                raise e
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    PanoGenerator.run()