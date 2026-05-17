from path_utils import add_project_paths, add_system_path, lib_path, checkpoints_path
add_project_paths()

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageFilter
import py360convert
import traceback
import torch

from remote_connection.remote_server import RemoteServer
from diffusers import FluxImg2ImgPipeline, FluxPriorReduxPipeline
from util.device_utils import offload_pipeline


def _mirror_wrap_canvas(input_image: Image.Image, equi_size: tuple, hfov_deg: float) -> Image.Image:
    """
    Build a panoramic canvas by tiling the input image with horizontal mirror-wrapping.
    
    The input image is placed at the center (yaw=0), and mirror-flipped copies fill
    the remaining horizontal extent. This gives the LoRA a coherent prior for regions
    it needs to hallucinate rather than pure blur.
    """
    equi_w, equi_h = equi_size
    
    # How wide (in pixels) does the input image occupy in the equirectangular?
    tile_w = max(1, int((hfov_deg / 360.0) * equi_w))
    tile_h = max(1, int((hfov_deg * (input_image.height / input_image.width) / 180.0) * equi_h))
    tile = input_image.resize((tile_w, tile_h), Image.LANCZOS)
    tile_flip = tile.transpose(Image.FLIP_LEFT_RIGHT)

    # Start from a blurred, low-frequency version for unvisited regions
    canvas = tile.resize(equi_size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=60))

    # Paste mirror-wrapped tiles left and right of center until canvas is filled
    cx = (equi_w - tile_w) // 2
    cy = (equi_h - tile_h) // 2

    # Right side (including center)
    x = cx
    flip = False
    while x < equi_w:
        t = tile_flip if flip else tile
        canvas.paste(t, (x, cy))
        x += tile_w
        flip = not flip

    # Left side
    x = cx - tile_w
    flip = True
    while x + tile_w > 0:
        t = tile_flip if flip else tile
        canvas.paste(t, (x, cy))
        x -= tile_w
        flip = not flip

    return canvas


class PanoGenerator(RemoteServer):
    def setup(self):
        # ------------------------------------------------------------------ #
        #  Pass 1 – Panorama layout: LoRA-augmented Img2Img                  #
        # ------------------------------------------------------------------ #
        self.base_pipeline = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16,
        )
        self.base_pipeline.vae.enable_tiling()
        self.base_pipeline.vae.enable_slicing()
        self.base_pipeline.load_lora_weights(
            str(checkpoints_path() / "layer_pano_3d"),
            weight_name="pano_lora_720*1440_v1.safetensors",
            adapter_name="pano",
        )

        # ------------------------------------------------------------------ #
        #  Pass 2 – Style transfer: Redux prior + same img2img backbone       #
        #  We reuse base_pipeline's transformer/vae so we don't double VRAM. #
        # ------------------------------------------------------------------ #
        self.prior_pipeline = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev",
            torch_dtype=torch.float16,
        )

        # Redux style pipeline shares weights with base_pipeline.
        # Text encoders are None – Redux drives conditioning via image embeds alone.
        # This is the intended usage pattern for Redux (no text prompt needed).
        self.style_pipeline = FluxImg2ImgPipeline(
            scheduler=self.base_pipeline.scheduler,
            vae=self.base_pipeline.vae,
            transformer=self.base_pipeline.transformer,
            text_encoder=None,
            tokenizer=None,
            text_encoder_2=None,
            tokenizer_2=None,
        )
        self.style_pipeline.vae.enable_tiling()
        self.style_pipeline.vae.enable_slicing()

        offload_pipeline(self.device, self.base_pipeline)
        offload_pipeline(self.device, self.prior_pipeline)
        # offload_pipeline(self.device, self.style_pipeline)

    # ---------------------------------------------------------------------- #
    #  Core panorama generation                                               #
    # ---------------------------------------------------------------------- #

    def pano(
        self,
        temp_path: Path,
        input_image: Image.Image,
        fov_deg: float = 60.0,
        caption: str = "",
    ) -> dict:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        equi_size = (2048, 1024)  # width × height (2:1 equirectangular)

        prompt = (
            f"{caption}, "
            "360 degree equirectangular panorama, seamless wrap, "
            "hyper-detailed, sharp focus, 8k resolution"
        ).strip(", ")

        # ------------------------------------------------------------------ #
        #  Canvas initialisation                                              #
        # ------------------------------------------------------------------ #
        canvas = _mirror_wrap_canvas(input_image, equi_size, hfov_deg=fov_deg)
        canvas.save(str(temp_path / "00_canvas.png"))

        # ------------------------------------------------------------------ #
        #  Pass 1 – Panoramic layout expansion                               #
        #  High strength so the LoRA can invent content, but not so high     #
        #  that it ignores the seeded tiles.                                 #
        # ------------------------------------------------------------------ #
        print("--- [Pass 1] Expanding to full 360° panorama ---")
        with torch.inference_mode():
            # Activate the panorama LoRA for this pass
            self.base_pipeline.set_adapters(["pano"], adapter_weights=[1.0])

            pass1: Image.Image = self.base_pipeline(
                prompt=prompt,
                image=canvas,
                strength=0.85,
                height=equi_size[1],
                width=equi_size[0],
                guidance_scale=3.5,
                num_inference_steps=40,
                output_type="pil",
            ).images[0]

        pass1.save(str(temp_path / "01_pass1_layout.png"))

        # ------------------------------------------------------------------ #
        #  Pass 2 – Lightweight Redux style transfer                         #
        #  Low strength preserves panoramic geometry; Redux embeds           #
        #  harmonise colour/texture with the reference image.               #
        # ------------------------------------------------------------------ #
        print("--- [Pass 2] Applying style transfer via Redux ---")
        with torch.inference_mode():
            redux_embeds = self.prior_pipeline(image=input_image)

            # Deactivate pano LoRA – style pass should be LoRA-free
            self.base_pipeline.transformer.disable_adapters()

            pass2_np: np.ndarray = self.style_pipeline(
                **redux_embeds,
                image=pass1,
                strength=0.40,          # low: harmonise without destroying layout
                guidance_scale=4.0,
                num_inference_steps=30,
                output_type="np",
            ).images[0]

        equirectangular = (pass2_np * 255).round().astype("uint8")
        equi_pil = Image.fromarray(equirectangular)
        equi_pil.save(str(temp_path / "02_final_panorama.png"))

        # ------------------------------------------------------------------ #
        #  Cubemap projection                                                 #
        # ------------------------------------------------------------------ #
        cube_dict = py360convert.e2c(equirectangular, face_w=512, cube_format="dict")
        for k, face in cube_dict.items():
            Image.fromarray(np.clip(face, 0, 255).astype("uint8")).save(
                str(temp_path / f"face_{k}.png")
            )

        return {
            "image": equi_pil,
            "faces": {
                k: Image.fromarray(np.clip(v, 0, 255).astype("uint8"))
                for k, v in cube_dict.items()
            },
        }

    # ---------------------------------------------------------------------- #
    #  RemoteServer dispatch                                                  #
    # ---------------------------------------------------------------------- #

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            try:
                print(f"Got input keys: {list(input.keys())}")
                result = self.pano(
                    temp_path=temp_path,
                    input_image=input["image"],
                    fov_deg=float(input.get("fov_degrees", 60.0)),  # no doubling
                    caption=input.get("caption", ""),
                )
                print(f"Panorama complete: {result['image'].size}")
                return result
            except Exception as e:
                print(f"Unable to generate panorama: {e}")
                traceback.print_exc()
                raise
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    PanoGenerator.run()