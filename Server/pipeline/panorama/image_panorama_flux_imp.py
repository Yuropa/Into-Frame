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
from diffusers import FluxInpaintPipeline, FluxImg2ImgPipeline, FluxPriorReduxPipeline
from util.device_utils import offload_pipeline

class PanoGenerator(RemoteServer):
    def setup(self):
        self.inpaint_pipeline = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
        )
        self.inpaint_pipeline.vae.enable_tiling()
        self.inpaint_pipeline.vae.enable_slicing()
        
        self.inpaint_pipeline.load_lora_weights(
            str(checkpoints_path() / "layer_pano_3d"), 
            weight_name="pano_lora_720*1440_v1.safetensors",
            adapter_name="pano"
        )
        offload_pipeline(self.device, self.inpaint_pipeline)

        self.prior_pipeline = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev",
            torch_dtype=torch.float16
        )
        offload_pipeline(self.device, self.prior_pipeline)

        self.img2img_pipeline = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
            text_encoder=None,
            text_encoder_2=None 
        )
        self.img2img_pipeline.vae.enable_tiling()
        self.img2img_pipeline.vae.enable_slicing()
        offload_pipeline(self.device, self.img2img_pipeline)

    def make_outpaint_mask(
        self,
        equi_size: tuple,
        input_image: Image.Image,
        hfov_deg: float = 90.0,
        blend_width: int = 120,
    ) -> tuple[Image.Image, Image.Image, tuple]:
        equi_w, equi_h = equi_size
        target_w = int((hfov_deg / 360.0) * equi_w)
        target_h = int((hfov_deg * (input_image.height / input_image.width) / 180.0) * equi_h)

        input_resized = input_image.resize((target_w, target_h), Image.LANCZOS)
        
        canvas = input_resized.resize(equi_size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=50))
        x = (equi_w - target_w) // 2
        y = (equi_h - target_h) // 2
        canvas.paste(input_resized, (x, y))

        mask = Image.new("L", equi_size, 255)
        
        inner_w = target_w - (blend_width * 2)
        inner_h = target_h - (blend_width * 2)
        inner_x = x + blend_width
        inner_y = y + blend_width
        
        mask.paste(Image.new("L", (inner_w, inner_h), 0), (inner_x, inner_y))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blend_width // 2))

        return canvas, mask, (x, y, target_w, target_h)

    def pano(self, temp_path: Path, input_image: Image.Image, depth_image: Image.Image, fov: float = 60.0, caption: str = "") -> dict:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        equi_size = (2048, 1024)
        prompt = caption + ", 360 degree equirectangular panorama, seamless, hyper-detailed, sharp focus, 8k resolution"

        canvas, mask, _ = self.make_outpaint_mask(equi_size, input_image, fov)

        # ==========================================
        # PASS 1: Build the Geometric 360 Panorama
        # ==========================================
        print("--- [Pass 1] Generating base panoramic structure ---")
        with torch.inference_mode():
            base_pano_output = self.inpaint_pipeline(
                prompt=prompt,
                image=canvas,
                mask_image=mask,
                height=equi_size[1],
                width=equi_size[0],
                strength=0.88, 
                guidance_scale=4.0,
                num_inference_steps=40, # Accelerated slightly for pipeline staging
                output_type='pil',
            ).images[0]
            
        base_pano_output.save(str(temp_path / "pass1_structural_pano.png"))

        # ==========================================
        # PASS 2: Global Redux Style Transfer
        # ==========================================
        print("--- [Pass 2] Applying Redux style from original image ---")
        with torch.inference_mode():
            # Extract pure style embeddings from the original image token source
            redux_output = self.prior_pipeline(image=input_image)
            
            # Apply those visual weights dynamically over our freshly minted full panorama 
            stylized_pano = self.img2img_pipeline(
                **redux_output,
                image=base_pano_output,
                height=equi_size[1],
                width=equi_size[0],
                strength=0.5,          # 0.4 - 0.5 is the sweet spot. Keeps Pass 1 structures 
                                       # perfectly intact while re-painting texture/grading.
                guidance_scale=3.5,
                num_inference_steps=25,
                output_type='np'
            ).images[0]

        # Convert final stylized output array back to standard image spaces
        equirectangular = (stylized_pano * 255).round().astype("uint8")
        equi_pil = Image.fromarray(equirectangular)
        equi_pil.save(str(temp_path / "final_stylized_panorama.png"))

        # Extract cube faces for seam checking
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

                print(f"Got flux pano {result}")
                return result
            except Exception as e:
                print(f"Unable to generate flux pano: {e}")
                traceback.print_exc()
                raise e
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    PanoGenerator.run()