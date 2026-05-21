from path_utils import add_project_paths, add_system_path, lib_path, checkpoints_path
add_project_paths()

from pathlib import Path
from typing import Any
import numpy as np
import cv2
from PIL import Image, ImageFilter
import py360convert
import traceback
import torch

from remote_connection.remote_server import RemoteServer
from diffusers import FluxInpaintPipeline
from util.device_utils import offload_pipeline
from transformers import CLIPVisionModelWithProjection

# --------------------------------------------------------------------------- #
#  Canvas helpers                                                              #
# --------------------------------------------------------------------------- #

def _make_canvas(
    input_image: Image.Image,
    equi_size: tuple,
    hfov_deg: float,
    minimum_strength: float = 120.0,
) -> tuple[Image.Image, Image.Image]:
    """
    Place input image at centre of equirectangular canvas.

    The surround is a smooth outward-blurred extension of the image edges so the
    LoRA sees a neutral prior rather than tiled duplicates.

    Returns
    -------
    canvas : RGB Image  — full equirectangular initialisation
    mask   : L Image    — 255 = inpaint this region, 0 = keep as-is
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

    left_strip  = tile.crop((0, 0, 1, tile_h)).resize((cx, tile_h), Image.BILINEAR)
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

    avg    = tuple(int(c) for c in np.array(tile).mean(axis=(0, 1)))
    canvas = Image.new("RGB", equi_size, avg)
    canvas.paste(surround, (0, 0))
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=equi_w // 20))
    canvas.paste(tile, (cx, cy))

    mask = Image.new("L", equi_size, 255)
    mask.paste(Image.new("L", (tile_w, tile_h), 0), (cx, cy))

    feather_radius = max(minimum_strength, tile_w // 20)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather_radius))

    return canvas, mask


def _lab_color_transfer(
    source: Image.Image,
    target: Image.Image,
    strength: float = 0.35,
) -> Image.Image:
    """
    Transfer the LAB colour statistics of *source* onto *target*.

    Preserves all spatial structure (edges, geometry, textures) in *target*
    while nudging its palette toward *source*.

    Parameters
    ----------
    source   : reference image whose colour distribution we want to match
    target   : generated panorama whose structure we want to keep
    strength : 0.0 = no change, 1.0 = full transfer.  Keep low (0.2–0.4)
               for a light touch.
    """
    src = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = cv2.cvtColor(np.array(target.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)

    transferred = tgt.copy()
    for ch in range(3):
        tgt_mean, tgt_std = tgt[..., ch].mean(), tgt[..., ch].std() + 1e-6
        src_mean, src_std = src[..., ch].mean(), src[..., ch].std() + 1e-6
        normalised          = (tgt[..., ch] - tgt_mean) / tgt_std
        transferred[..., ch] = normalised * src_std + src_mean

    # Blend: lerp between original and fully-transferred by `strength`
    blended = tgt * (1.0 - strength) + transferred * strength
    result  = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


# --------------------------------------------------------------------------- #
#  PanoGenerator                                                               #
# --------------------------------------------------------------------------- #

class PanoGenerator(RemoteServer):

    def setup(self):
        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.bfloat16,
        )

        # Single inpaint pipeline — carries the pano LoRA + IP-Adapter
        self.base_pipeline = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            image_encoder=image_encoder,
        )

        # Pano LoRA
        self.base_pipeline.load_lora_weights(
            str(checkpoints_path() / "layer_pano_3d"),
            weight_name="pano_lora_720*1440_v1.safetensors",
            adapter_name="pano",
        )

        # IP-Adapter — injects input-image semantics into cross-attention.
        # XLabs weights work well for FLUX.1-dev; swap path if you have another.
        self.base_pipeline.load_ip_adapter(
            "XLabs-AI/flux-ip-adapter",
            weight_name="ip_adapter.safetensors",
        )

        self.base_pipeline.enable_model_cpu_offload()
        self.base_pipeline.vae.enable_tiling()
        self.base_pipeline.vae.enable_slicing()

    # ---------------------------------------------------------------------- #
    #  Helpers                                                                #
    # ---------------------------------------------------------------------- #

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Run text encoding in isolation so T5 is evicted before the transformer runs."""
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

    # ---------------------------------------------------------------------- #
    #  Core panorama generation                                               #
    # ---------------------------------------------------------------------- #

    def pano(
        self,
        temp_path: Path,
        input_image: Image.Image,
        fov_deg: float = 60.0,
        caption: str = "",
        ip_adapter_scale: float = 0.6,
        color_transfer_strength: float = 0.30,
    ) -> dict:
        """
        Generate a 360° equirectangular panorama from a single input image.

        Parameters
        ----------
        input_image            : the seed photograph / render
        fov_deg                : estimated horizontal FoV of the input image
        caption                : optional text description to guide generation
        ip_adapter_scale       : how strongly the input image guides the LoRA
                                 pass (0 = text-only, 1 = image-dominated).
                                 0.5–0.7 is a good starting range.
        color_transfer_strength: 0.0–1.0 weight for the final LAB colour
                                 transfer step.  Keep ≤ 0.4 for a light touch.
        """
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        equi_size = (2048, 1024)
        prompt = (
            f"{caption}, 360 degree equirectangular panorama, seamless wrap, "
            "hyper-detailed, sharp focus, 8k resolution"
        ).strip(", ")

        # ------------------------------------------------------------------ #
        #  Pass 0 — Encode text prompt                                       #
        # ------------------------------------------------------------------ #
        print("--- [Pass 0] Encoding prompt ---")
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        # ------------------------------------------------------------------ #
        #  Pass 1 — IP-Adapter guided inpaint with pano LoRA                #
        # ------------------------------------------------------------------ #
        print("--- [Pass 1] Building canvas & mask ---")
        canvas, mask = _make_canvas(input_image, equi_size, hfov_deg=fov_deg)
        canvas.save(str(temp_path / "01_canvas.png"))
        mask.save(str(temp_path / "01_mask.png"))

        print("--- [Pass 1] Inpainting 360° surround (IP-Adapter + pano LoRA) ---")
        self.base_pipeline.set_adapters(["pano"], adapter_weights=[1.0])

        # IP-Adapter scale: controls how tightly the generated content tracks
        # the input image appearance vs the text prompt.
        self.base_pipeline.set_ip_adapter_scale(ip_adapter_scale)

        with torch.inference_mode():
            pass1: Image.Image = self.base_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                ip_adapter_image=input_image,   # <-- image guidance injected here
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
        #  Pass 2 — LAB colour transfer (light touch, zero VRAM)            #
        # ------------------------------------------------------------------ #
        print(f"--- [Pass 2] LAB colour transfer (strength={color_transfer_strength}) ---")
        final = _lab_color_transfer(
            source=input_image,
            target=pass1,
            strength=color_transfer_strength,
        )

        if final.size != equi_size:
            final = final.resize(equi_size, Image.LANCZOS)
        final.save(str(temp_path / "03_final_panorama.png"))

        # ------------------------------------------------------------------ #
        #  Cube-map projection                                               #
        # ------------------------------------------------------------------ #
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
                    ip_adapter_scale=float(input.get("ip_adapter_scale", 0.6)),
                    color_transfer_strength=float(input.get("color_transfer_strength", 0.30)),
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