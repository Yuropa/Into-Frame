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
from diffusers import FluxInpaintPipeline, FluxImg2ImgPipeline, FluxPriorReduxPipeline
from util.device_utils import offload_pipeline


def _mirror_wrap_canvas(input_image: Image.Image, equi_size: tuple, hfov_deg: float) -> Image.Image:
    """
    Build a panoramic canvas by tiling the input image with horizontal mirror-wrapping.

    The input image is placed at the center (yaw=0), and mirror-flipped copies fill
    the remaining horizontal extent. This gives the LoRA a coherent prior for regions
    it needs to hallucinate rather than pure blur.
    """
    equi_w, equi_h = equi_size

    tile_w = max(1, int((hfov_deg / 360.0) * equi_w))
    tile_h = max(1, int((hfov_deg * (input_image.height / input_image.width) / 180.0) * equi_h))
    tile = input_image.resize((tile_w, tile_h), Image.LANCZOS)
    tile_flip = tile.transpose(Image.FLIP_LEFT_RIGHT)

    canvas = tile.resize(equi_size, Image.BILINEAR).filter(ImageFilter.GaussianBlur(radius=60))

    cx = (equi_w - tile_w) // 2
    cy = (equi_h - tile_h) // 2

    x = cx
    flip = False
    while x < equi_w:
        t = tile_flip if flip else tile
        canvas.paste(t, (x, cy))
        x += tile_w
        flip = not flip

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
        self.base_pipeline = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
        )
        self.base_pipeline.load_lora_weights(
            str(checkpoints_path() / "layer_pano_3d"),
            weight_name="pano_lora_720*1440_v1.safetensors",
            adapter_name="pano",
        )
        self.base_pipeline.enable_model_cpu_offload()
        self.base_pipeline.vae.enable_tiling()
        self.base_pipeline.vae.enable_slicing()

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
    #  Helpers                                                                #
    # ---------------------------------------------------------------------- #

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run text encoding in isolation so T5 is evicted before transformer runs.
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

        te1.to("cpu")
        te2.to("cpu")
        torch.cuda.empty_cache()

        return prompt_embeds, pooled_prompt_embeds

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

    def _make_canvas(
        self,
        input_image: Image.Image,
        equi_size: tuple,
        hfov_deg: float,
        minimum_strength: float = 120.0,
    ) -> tuple[Image.Image, Image.Image]:
        """
        Place input image at center of equirectangular canvas.
        Surround is a smooth outward-blurred extension of the image edges
        so the LoRA sees a neutral prior rather than tiled duplicates.
        Returns (canvas, mask) where mask=255 means 'please inpaint this region'.
        """
        equi_w, equi_h = equi_size
        hfov_deg = hfov_deg * 2.0

        tile_w = max(1, int((hfov_deg / 360.0) * equi_w))
        tile_h = max(1, int((hfov_deg / 180.0) * equi_h * (input_image.height / input_image.width)))
        tile_h = min(tile_h, equi_h)
        tile = input_image.resize((tile_w, tile_h), Image.LANCZOS)

        cx = (equi_w - tile_w) // 2
        cy = (equi_h - tile_h) // 2

        surround = Image.new("RGB", equi_size)

        left_strip = tile.crop((0, 0, 1, tile_h)).resize((cx, tile_h), Image.BILINEAR)
        surround.paste(left_strip, (0, cy))

        right_strip = tile.crop((tile_w - 1, 0, tile_w, tile_h)).resize(
            (equi_w - cx - tile_w, tile_h), Image.BILINEAR
        )
        surround.paste(right_strip, (cx + tile_w, cy))

        top_strip = tile.crop((0, 0, tile_w, 1)).resize((tile_w, cy), Image.BILINEAR)
        surround.paste(top_strip, (cx, 0))

        bot_strip = tile.crop((0, tile_h - 1, tile_w, tile_h)).resize(
            (tile_w, equi_h - cy - tile_h), Image.BILINEAR
        )
        surround.paste(bot_strip, (cx, cy + tile_h))

        avg = tuple(int(c) for c in np.array(tile).mean(axis=(0, 1)))
        canvas = Image.new("RGB", equi_size, avg)
        canvas.paste(surround, (0, 0))
        canvas = canvas.filter(ImageFilter.GaussianBlur(radius=equi_w // 20))
        canvas.paste(tile, (cx, cy))

        mask = Image.new("L", equi_size, 255)
        mask.paste(Image.new("L", (tile_w, tile_h), 0), (cx, cy))

        # Feather the mask edge so inpaint blends into the original
        feather_radius = max(minimum_strength, tile_w // 20)
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

        return canvas, mask

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

            del redux_embeds_gpu
            torch.cuda.empty_cache()

            styled_f = (styled_np * 255).astype(np.float32)

            right = x + tile_w
            if right <= pano_w:
                accum[y:y+tile_h, x:right]        += styled_f * tile_weight
                weight[y:y+tile_h, x:right]        += tile_weight
            else:
                part1_w = pano_w - x
                accum[y:y+tile_h, x:pano_w]        += styled_f[:, :part1_w]  * tile_weight[:, :part1_w]
                accum[y:y+tile_h, 0:right-pano_w]  += styled_f[:, part1_w:]  * tile_weight[:, part1_w:]
                weight[y:y+tile_h, x:pano_w]       += tile_weight[:, :part1_w]
                weight[y:y+tile_h, 0:right-pano_w] += tile_weight[:, part1_w:]

        return Image.fromarray(
            np.clip(accum / np.maximum(weight, 1e-6), 0, 255).astype("uint8")
        )

    # ---------------------------------------------------------------------- #
    #  Core panorama generation                                               #
    # ---------------------------------------------------------------------- #

    def pano(
        self,
        temp_path: Path,
        input_image: Image.Image,
        fov_deg: float = 60.0,
        caption: str = "",
        style_strength: float = 0.5,
    ) -> dict:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        equi_size = (2048, 1024)
        prompt = (
            f"{caption}, 360 degree equirectangular panorama, seamless wrap, "
            "hyper-detailed, sharp focus, 8k resolution"
        ).strip(", ")

        print("--- [Pass 0] Encoding prompt ---")
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        # ------------------------------------------------------------------ #
        #  Pass 1 — Inpaint 360° surround                                    #
        # ------------------------------------------------------------------ #
        canvas, mask = self._make_canvas(input_image, equi_size, hfov_deg=fov_deg)
        canvas.save(str(temp_path / "01_canvas.png"))
        mask.save(str(temp_path / "01_mask.png"))

        print("--- [Pass 1] Inpainting 360° surround ---")
        self.base_pipeline.set_adapters(["pano"], adapter_weights=[1.0])

        with torch.inference_mode():
            pass1: Image.Image = self.base_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                image=canvas,
                mask_image=mask,
                strength=0.99,
                height=equi_size[1],
                width=equi_size[0],
                guidance_scale=3.5,
                num_inference_steps=40,
                output_type="pil",
            ).images[0]

        self.base_pipeline.transformer.to("cpu")
        self.base_pipeline.vae.to("cpu")
        torch.cuda.empty_cache()

        if pass1.size != equi_size:
            pass1 = pass1.resize(equi_size, Image.LANCZOS)
        pass1.save(str(temp_path / "02_pass1_layout.png"))

        # ------------------------------------------------------------------ #
        #  Pass 2a — Extract Redux style embeds                              #
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
                    fov_deg=float(input.get("fov_degrees", 60.0)),
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