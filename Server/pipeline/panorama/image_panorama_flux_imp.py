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
        self.base_pipeline = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
        self.base_pipeline.vae.enable_tiling()
        self.base_pipeline.vae.enable_slicing()
        self.base_pipeline.load_lora_weights(
            str(checkpoints_path() / "layer_pano_3d"),
            weight_name="pano_lora_720*1440_v1.safetensors",
            adapter_name="pano",
        )
        # model_cpu_offload moves whole components (transformer, VAE, T5) 
        # on/off GPU as needed — no meta tensors, no layer streaming
        self.base_pipeline.enable_model_cpu_offload()

        self.prior_pipeline = FluxPriorReduxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Redux-dev",
            torch_dtype=torch.bfloat16,
        )
        self.prior_pipeline.enable_model_cpu_offload()

        self.style_pipeline = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
        self.style_pipeline.vae.enable_tiling()
        self.style_pipeline.vae.enable_slicing()
        self.style_pipeline.enable_model_cpu_offload()

    # ---------------------------------------------------------------------- #
    #  Core panorama generation                                               #
    # ---------------------------------------------------------------------- #

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run text encoding in isolation so T5 is evicted before transformer runs.
        Temporarily disables model offload hooks to control device manually.
        """
        te1 = self.base_pipeline.text_encoder
        te2 = self.base_pipeline.text_encoder_2

        te1.to(self.device)
        te2.to(self.device)
        torch.cuda.empty_cache()

        with torch.inference_mode():
            result = self.base_pipeline.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=self.device,
                num_images_per_prompt=1,
            )
        prompt_embeds, pooled_prompt_embeds = result[0], result[1]

        # Evict text encoders before returning
        te1.to("cpu")
        te2.to("cpu")
        torch.cuda.empty_cache()

        return prompt_embeds, pooled_prompt_embeds

    def _tiled_redux_style(
        self,
        panorama: Image.Image,
        redux_embeds: dict,
        strength: float = 0.75,
        tile_w: int = 1024,
        tile_h: int = 1024,
        overlap: int = 256,
        num_inference_steps: int = 50,
    ) -> Image.Image:
        pano_w, pano_h = panorama.size
        accum = np.zeros((pano_h, pano_w, 3), dtype=np.float32)
        weight = np.zeros((pano_h, pano_w, 1), dtype=np.float32)

        def feather_1d(size, overlap):
            k = np.ones(size, dtype=np.float32)
            ramp = np.linspace(0, 1, overlap)
            k[:overlap] = ramp
            k[-overlap:] = ramp[::-1]
            return k

        stride_x = tile_w - overlap
        stride_y = tile_h - overlap
        xs = list(range(0, pano_w, stride_x))
        ys = list(range(0, pano_h - tile_h + 1, stride_y))
        if not ys or ys[-1] + tile_h < pano_h:
            ys.append(pano_h - tile_h)

        tile_weight = (
            feather_1d(tile_h, overlap)[:, None] *
            feather_1d(tile_w, overlap)[None, :]
        )[:, :, None]

        total = len(xs) * len(ys)
        for idx, (y, x) in enumerate([(y, x) for y in ys for x in xs]):
            print(f"    Tile {idx+1}/{total} x={x} y={y}")

            # Extract with horizontal wrap
            right = x + tile_w
            tile_pil = Image.new("RGB", (tile_w, tile_h))
            if right <= pano_w:
                tile_pil.paste(panorama.crop((x, y, right, y + tile_h)), (0, 0))
            else:
                part1_w = pano_w - x
                tile_pil.paste(panorama.crop((x, y, pano_w, y + tile_h)), (0, 0))
                tile_pil.paste(panorama.crop((0, y, right - pano_w, y + tile_h)), (part1_w, 0))

            with torch.inference_mode():
                redux_embeds_gpu = self._to_device(redux_embeds, self.device)
                styled_np = self.style_pipeline(
                    **redux_embeds_gpu,
                    image=tile_pil,
                    strength=strength,
                    guidance_scale=5.0,
                    num_inference_steps=num_inference_steps,
                    height=tile_h,
                    width=tile_w,
                    output_type="np",
                ).images[0]

            styled_f = (styled_np * 255).astype(np.float32)

            right = x + tile_w
            if right <= pano_w:
                accum[y:y+tile_h, x:right]       += styled_f * tile_weight
                weight[y:y+tile_h, x:right]       += tile_weight
            else:
                part1_w = pano_w - x
                accum[y:y+tile_h, x:pano_w]       += styled_f[:, :part1_w]  * tile_weight[:, :part1_w]
                accum[y:y+tile_h, 0:right-pano_w] += styled_f[:, part1_w:]  * tile_weight[:, part1_w:]
                weight[y:y+tile_h, x:pano_w]      += tile_weight[:, :part1_w]
                weight[y:y+tile_h, 0:right-pano_w] += tile_weight[:, part1_w:]

            torch.cuda.empty_cache()

        return Image.fromarray(
            np.clip(accum / np.maximum(weight, 1e-6), 0, 255).astype("uint8")
        )

    def _to_device(self, obj, device):
        """Recursively move all tensors in a dict/list/tuple to device."""
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        elif isinstance(obj, dict):
            return {k: self._to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            moved = [self._to_device(v, device) for v in obj]
            return type(obj)(moved)
        return obj

    def pano(
        self,
        temp_path: Path,
        input_image: Image.Image,
        fov_deg: float = 60.0,
        caption: str = "",
        style_strength=0.75
    ) -> dict:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        equi_size = (2048, 1024)
        prompt = (
            f"{caption}, 360 degree equirectangular panorama, seamless wrap, "
            "hyper-detailed, sharp focus, 8k resolution"
        ).strip(", ")

        canvas = _mirror_wrap_canvas(input_image, equi_size, hfov_deg=fov_deg)
        canvas.save(str(temp_path / "01_canvas.png"))

        # ------------------------------------------------------------------ #
        #  Pass 1 — encode text first, then transformer runs alone on GPU    #
        # ------------------------------------------------------------------ #
        print("--- [Pass 1] Encoding prompt ---")
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        print("--- [Pass 1] Expanding to 360° panorama ---")
        self.base_pipeline.set_adapters(["pano"], adapter_weights=[1.0])

        with torch.inference_mode():
            pass1: Image.Image = self.base_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                image=canvas,
                strength=0.85,
                height=equi_size[1],
                width=equi_size[0],
                guidance_scale=3.5,
                num_inference_steps=40,
                output_type="pil",
            ).images[0]

        # Fully evict base pipeline before prior runs
        self.base_pipeline.transformer.to("cpu")
        self.base_pipeline.vae.to("cpu")
        torch.cuda.empty_cache()

        if pass1.size != equi_size:
            pass1 = pass1.resize(equi_size, Image.LANCZOS)
        pass1.save(str(temp_path / "02_pass1_layout.png"))

        # ------------------------------------------------------------------ #
        #  Pass 2a — Redux embeds (prior pipeline alone on GPU)              #
        # ------------------------------------------------------------------ #
        print("--- [Pass 2a] Extracting style embeds ---")
        with torch.inference_mode():
            redux_embeds = self.prior_pipeline(image=input_image)

        self.prior_pipeline.image_encoder.to("cpu")
        torch.cuda.empty_cache()

        redux_embeds_cpu = self._to_device(redux_embeds, "cpu")

        # ------------------------------------------------------------------ #
        #  Pass 2b — Tiled style transfer                                    #
        # ------------------------------------------------------------------ #
        print("--- [Pass 2b] Tiled style transfer ---")
        final = self._tiled_redux_style(
            panorama=pass1,
            redux_embeds=redux_embeds_cpu,
            strength=style_strength,
        )

        if final.size != equi_size:
            final = final.resize(equi_size, Image.LANCZOS)
        final.save(str(temp_path / "03_final_panorama.png"))

        equirectangular = np.array(final)
        cube_dict = py360convert.e2c(equirectangular, face_w=512, cube_format="dict")
        for k, face in cube_dict.items():
            Image.fromarray(np.clip(face, 0, 255).astype("uint8")).save(
                str(temp_path / f"face_{k}.png")
            )

        return {
            "image": final,
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
                    style_strength=float(input.get("style_strength", 0.75)),
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