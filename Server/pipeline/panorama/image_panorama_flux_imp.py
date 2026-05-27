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

import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

# --------------------------------------------------------------------------- #
#  Canvas helpers                                                              #
# --------------------------------------------------------------------------- #

class NeuralStyleTransfer:
    """
    Arbitrary image style transfer using VGG19 gram-matrix matching.
    Fixed to handle high-resolution inputs without destroying details.
    """
    CONTENT_LAYERS = ["conv4_2"]
    STYLE_LAYERS   = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
    STYLE_WEIGHTS  = [1e3, 1e3, 1e3, 1e3, 1e3]

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._vgg = None  # Lazy loaded to conserve VRAM during main generation
        self._layer_map = {
            "conv1_1": 1,  "conv2_1": 6,  "conv3_1": 11,
            "conv4_1": 20, "conv4_2": 22, "conv5_1": 29,
        }

    @property
    def vgg(self):
        if self._vgg is None:
            self._vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(self.device).eval()
            for p in self._vgg.parameters():
                p.requires_grad_(False)
        return self._vgg

    def _get_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {}
        wanted = set(self._layer_map.values())
        name_by_idx = {v: k for k, v in self._layer_map.items()}
        for idx, layer in enumerate(self.vgg):
            x = layer(x)
            if idx in wanted:
                features[name_by_idx[idx]] = x
        return features

    @staticmethod
    def _gram(feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        f = feat.view(c, h * w)
        return f @ f.t() / (c * h * w)

    def _to_tensor(self, img: Image.Image, max_size: int = 1024) -> torch.Tensor:
        """Increased max_size to 1024 to preserve high-frequency details."""
        w, h  = img.size
        scale = max_size / max(w, h)
        if max(w, h) > max_size:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return tf(img).unsqueeze(0).to(self.device)

    def _to_image(self, tensor: torch.Tensor) -> Image.Image:
        t = tensor.squeeze(0).cpu().detach()
        t = t * torch.tensor([0.229, 0.224, 0.225])[:, None, None] \
              + torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        t = t.clamp(0, 1)
        return T.ToPILImage()(t)

    def transfer(
        self,
        content: Image.Image,
        style: Image.Image,
        strength: float = 0.5,
        steps: int = 200,
        content_weight: float = 50.0,
        style_weight: float = 5e4,
    ) -> Image.Image:
        if strength <= 0.0:
            return content

        orig_size = content.size
        ct = self._to_tensor(content, max_size=1024)
        st = self._to_tensor(style, max_size=512)   # Style reference can stay small

        content_feats = self._get_features(ct)
        style_feats   = self._get_features(st)
        style_grams   = {l: self._gram(style_feats[l]) for l in self.STYLE_LAYERS}

        canvas = ct.clone().requires_grad_(True)
        optimiser = torch.optim.Adam([canvas], lr=0.02)

        for step in range(steps):
            optimiser.zero_grad()
            feats = self._get_features(canvas)

            c_loss = nn.functional.mse_loss(feats["conv4_2"], content_feats["conv4_2"])
            s_loss = sum(
                w * nn.functional.mse_loss(self._gram(feats[l]), style_grams[l])
                for l, w in zip(self.STYLE_LAYERS, self.STYLE_WEIGHTS)
            )

            loss = content_weight * c_loss + style_weight * s_loss
            loss.backward()
            optimiser.step()

        nst_result = self._to_image(canvas).resize(orig_size, Image.LANCZOS)

        if strength < 1.0:
            nst_arr     = np.array(nst_result).astype(np.float32)
            content_arr = np.array(content).astype(np.float32)
            blended     = content_arr * (1 - strength) + nst_arr * strength
            return Image.fromarray(blended.clip(0, 255).astype(np.uint8))

        return nst_result


def _make_canvas(
    input_image: Image.Image,
    equi_size: tuple,
    hfov_deg: float,
    feather_pct: float = 0.05,  # Reduced default feather to keep the center clean
) -> tuple[Image.Image, Image.Image]:
    equi_w, equi_h = equi_size

    tile_w = max(1, int((hfov_deg / 360.0) * equi_w))
    tile_h = max(1, int((hfov_deg / 180.0) * equi_h * (input_image.height / input_image.width)))
    tile_h = min(tile_h, equi_h)
    tile = input_image.resize((tile_w, tile_h), Image.LANCZOS)

    cx = (equi_w - tile_w) // 2
    cy = (equi_h - tile_h) // 2

    surround = Image.new("RGB", equi_size)

    left_strip  = tile.crop((0, 0, 1, tile_h)).resize((cx, tile_h), Image.BILINEAR)
    surround.paste(left_strip, (0, cy))
    right_strip = tile.crop((tile_w - 1, 0, tile_w, tile_h)).resize((equi_w - cx - tile_w, tile_h), Image.BILINEAR)
    surround.paste(right_strip, (cx + tile_w, cy))
    top_strip = tile.crop((0, 0, tile_w, 1)).resize((tile_w, cy), Image.BILINEAR)
    surround.paste(top_strip, (cx, 0))
    bot_strip = tile.crop((0, tile_h - 1, tile_w, tile_h)).resize((tile_w, equi_h - cy - tile_h), Image.BILINEAR)
    surround.paste(bot_strip, (cx, cy + tile_h))

    avg    = tuple(int(c) for c in np.array(tile).mean(axis=(0, 1)))
    canvas = Image.new("RGB", equi_size, avg)
    canvas.paste(surround, (0, 0))
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=equi_w // 80))
    canvas.paste(tile, (cx, cy))

    # Strict mask generation: prevent wide gray gradients from eating the image center
    mask = Image.new("L", equi_size, 255)
    
    feather_w = max(4, int(tile_w * feather_pct))
    feather_h = max(4, int(tile_h * feather_pct))
    inner_w = max(1, tile_w - (feather_w * 2))
    inner_h = max(1, tile_h - (feather_h * 2))
    
    inner_blank = Image.new("L", (inner_w, inner_h), 0)
    mask.paste(inner_blank, (cx + feather_w, cy + feather_h))
    
    # Use a small, disciplined radius for edge softening
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(4, min(feather_w, feather_h) // 2)))

    return canvas, mask


def _make_seam_mask(equi_size: tuple, seam_width: int = 96) -> Image.Image:
    w, h = equi_size
    mask = Image.new("L", equi_size, 0)
    seam_box = Image.new("L", (seam_width, h), 255)
    mask.paste(seam_box, ((w - seam_width) // 2, 0))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=12))
    return mask


def _shift_horizon(img: Image.Image, pct: float = 0.5) -> Image.Image:
    arr = np.array(img)
    shift_amt = int(arr.shape[1] * pct)
    shifted = np.roll(arr, shift_amt, axis=1)
    return Image.fromarray(shifted)


def _lab_color_transfer(
    source: Image.Image,
    target: Image.Image,
    strength: float = 0.35,
    mask: Image.Image | None = None,
) -> Image.Image:
    if strength <= 0.0:
        return target

    def to_lab(img: Image.Image) -> np.ndarray:
        rgb = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    src = to_lab(source)
    tgt = to_lab(target)

    transferred = tgt.copy()
    for ch in range(3):
        tgt_mean = tgt[..., ch].mean()
        tgt_std  = tgt[..., ch].std() + 1e-6
        src_mean = src[..., ch].mean()
        src_std  = src[..., ch].std() + 1e-6
        transferred[..., ch] = (tgt[..., ch] - tgt_mean) / tgt_std * src_std + src_mean

    if mask is not None:
        mask_resized = mask.convert("L").resize((target.width, target.height), Image.BILINEAR)
        pixel_strength = np.array(mask_resized, dtype=np.float32)[..., None] / 255.0 * strength
    else:
        pixel_strength = strength

    blended = tgt + pixel_strength * (transferred - tgt)

    blended[..., 0] = np.clip(blended[..., 0],    0.0, 100.0)
    blended[..., 1] = np.clip(blended[..., 1], -127.0, 127.0)
    blended[..., 2] = np.clip(blended[..., 2], -127.0, 127.0)

    rgb = cv2.cvtColor(blended, cv2.COLOR_LAB2RGB)
    return Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))


# --------------------------------------------------------------------------- #
#  PanoGenerator                                                               #
# --------------------------------------------------------------------------- #

class PanoGenerator(RemoteServer):

    def setup(self):
        self.style_transfer = NeuralStyleTransfer(device=self.device)

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14",
            torch_dtype=torch.bfloat16,
        )

        self.base_pipeline = FluxInpaintPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            image_encoder=image_encoder,
        )

        self.base_pipeline.load_lora_weights(
            str(checkpoints_path() / "layer_pano_3d"),
            weight_name="pano_lora_720*1440_v1.safetensors",
            adapter_name="pano",
        )

        self.base_pipeline.load_ip_adapter(
            "XLabs-AI/flux-ip-adapter",
            weight_name="ip_adapter.safetensors",
        )

        self.base_pipeline.enable_model_cpu_offload()
        self.base_pipeline.vae.enable_tiling()
        self.base_pipeline.vae.enable_slicing()

    def _encode_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        te1 = self.base_pipeline.text_encoder
        te2 = self.base_pipeline.text_encoder_2
        te1.to(self.device)
        te2.to(self.device)
        torch.cuda.empty_cache()

        with torch.inference_mode():
            result = self.base_pipeline.encode_prompt(
                prompt=prompt, prompt_2=None, device=self.device, num_images_per_prompt=1
            )
        te1.to("cpu")
        te2.to("cpu")
        torch.cuda.empty_cache()
        return result[0], result[1]

    def pano(
        self,
        temp_path: Path,
        input_image: Image.Image,
        fov_deg: float,
        caption: str,
        ip_adapter_scale: float,
        color_transfer_strength: float,
        style_strength: float,
        nst_steps: int,
        seed: int = 0,
    ) -> dict:
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image)

        equi_size = (2048, 1024)
        prompt = (
            f"{caption}, 360 degree equirectangular panorama, seamless wrap, "
            "hyper-detailed, sharp focus, 8k resolution"
        ).strip(", ")

        # --- Pass 0 — Encode text prompt ---
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

        # --- Pass 1 — Main Generation ---
        self.report_progress(0.10, "Building blending canvas...")
        canvas, mask = _make_canvas(input_image, equi_size, hfov_deg=fov_deg, feather_pct=0.05)
        canvas.save(str(temp_path / "01_canvas.png"))
        mask.save(str(temp_path / "01_mask.png"))

        self.report_progress(0.20, "Running main 360° outpaint...")
        self.base_pipeline.set_adapters(["pano"], adapter_weights=[1.0])
        self.base_pipeline.set_ip_adapter_scale(ip_adapter_scale)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode():
            pass1: Image.Image = self.base_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                ip_adapter_image=input_image,
                image=canvas,
                mask_image=mask,
                strength=0.99,  # Force sharp latent creation over the surround
                height=equi_size[1],
                width=equi_size[0],
                guidance_scale=3.5,
                num_inference_steps=40,
                output_type="pil",
                generator=generator,
            ).images[0]
            
        pass1.save(str(temp_path / "02_pass1_initial.png"))

        # --- Pass 2 — Seam Stitch Fix ---
        self.report_progress(0.60, "Fixing panorama edge seams...")
        
        shifted_pass1 = _shift_horizon(pass1, pct=0.5)
        seam_mask = _make_seam_mask(equi_size, seam_width=96) # Kept tighter to preserve clarity
        
        with torch.inference_mode():
            fixed_shifted: Image.Image = self.base_pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                ip_adapter_image=input_image,
                image=shifted_pass1,
                mask_image=seam_mask,
                strength=0.70,  # Lower strength protects details from blurring
                height=equi_size[1],
                width=equi_size[0],
                guidance_scale=3.5,
                num_inference_steps=25,
                output_type="pil",
                generator=generator,
            ).images[0]

        pass2 = _shift_horizon(fixed_shifted, pct=-0.5)
        pass2.save(str(temp_path / "04_seams_fixed.png"))

        # Offload pipeline out of CUDA
        self.base_pipeline.transformer.to("cpu")
        self.base_pipeline.vae.to("cpu")
        torch.cuda.empty_cache()

        # --- Pass 3 — Color & Style Adjustments ---
        self.report_progress(0.85, "Post-processing color and style...")
        lab_result = _lab_color_transfer(
            source=input_image,
            target=pass2,
            strength=color_transfer_strength,
            mask=None,
        )

        # NST processing
        final = self.style_transfer.transfer(
            content=lab_result,
            style=input_image,
            strength=style_strength,
            steps=nst_steps,
        )

        # Soft-composite original back into center with feathered edges so it blends naturally
        tile_w = max(1, int((fov_deg / 360.0) * equi_size[0]))
        tile_h = max(1, int((fov_deg / 180.0) * equi_size[1] * (input_image.height / input_image.width)))
        tile_h = min(tile_h, equi_size[1])
        cx = (equi_size[0] - tile_w) // 2
        cy = (equi_size[1] - tile_h) // 2

        tile_resized = input_image.resize((tile_w, tile_h), Image.LANCZOS)
        feather_px = max(12, int(min(tile_w, tile_h) * 0.10))
        center_mask = Image.new("L", (tile_w, tile_h), 0)
        inner_w = max(1, tile_w - feather_px * 2)
        inner_h = max(1, tile_h - feather_px * 2)
        center_mask.paste(Image.new("L", (inner_w, inner_h), 255), (feather_px, feather_px))
        center_mask = center_mask.filter(ImageFilter.GaussianBlur(radius=feather_px))
        final.paste(tile_resized, (cx, cy), mask=center_mask)

        if final.size != equi_size:
            final = final.resize(equi_size, Image.LANCZOS)
        final.save(str(temp_path / "05_final_panorama.png"))

        # Cube-map projection
        self.report_progress(0.95, "Projecting cubemap...")
        equirectangular = np.array(final)
        cube_dict = py360convert.e2c(equirectangular, face_w=512, cube_format="dict")
        
        return {
            "image": final,
            "faces": {
                k: Image.fromarray(np.clip(v, 0, 255).astype("uint8"))
                for k, v in cube_dict.items()
            },
        }

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            try:
                return self.pano(
                    temp_path=temp_path,
                    input_image=input["image"],
                    fov_deg=float(input.get("fov_degrees", 60.0)),
                    caption=input.get("caption", ""),
                    ip_adapter_scale=float(input.get("ip_adapter_scale", 0.6)),
                    color_transfer_strength=float(input.get("color_transfer_strength", 0.35)),
                    style_strength=float(input.get("style_strength", 0.2)),
                    nst_steps=int(input.get("nst_steps", 150)),
                    seed=int(input.get("seed", 0)),
                )
            except Exception as e:
                traceback.print_exc()
                raise
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    PanoGenerator.run()