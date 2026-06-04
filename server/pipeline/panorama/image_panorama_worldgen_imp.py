from path_utils import add_project_paths
add_project_paths()

import torch
import cv2
import numpy as np
import py360convert
import traceback
from pathlib import Path
from typing import Any
from PIL import Image

from remote_connection.remote_server import RemoteServer
from worldgen.pano_gen import build_pano_fill_model, gen_pano_fill_image
from worldgen.pano_depth import build_depth_model, pred_depth
from pano_utils import map_image_to_pano, perspective_rays, DEFAULT_HFOV_DEG


def _lab_color_transfer(
    source: Image.Image,
    target: Image.Image,
    strength: float = 0.35,
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
        tgt_mean, tgt_std = tgt[..., ch].mean(), tgt[..., ch].std() + 1e-6
        src_mean, src_std = src[..., ch].mean(), src[..., ch].std() + 1e-6
        transferred[..., ch] = (tgt[..., ch] - tgt_mean) / tgt_std * src_std + src_mean

    blended = tgt + strength * (transferred - tgt)
    blended[..., 0] = np.clip(blended[..., 0],    0.0, 100.0)
    blended[..., 1] = np.clip(blended[..., 1], -127.0, 127.0)
    blended[..., 2] = np.clip(blended[..., 2], -127.0, 127.0)

    rgb = cv2.cvtColor(blended, cv2.COLOR_LAB2RGB)
    return Image.fromarray(np.clip(rgb * 255.0, 0, 255).astype(np.uint8))


def _enforce_wrap_continuity(img: Image.Image, blend_px: int = 48) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    w = arr.shape[1]
    t = np.linspace(1.0, 0.0, blend_px)[None, :, None]
    left_strip  = arr[:, :blend_px].copy()
    right_strip = arr[:, w - blend_px:].copy()
    avg = (left_strip + right_strip[:, ::-1]) / 2.0
    arr[:, :blend_px]     = left_strip  * (1 - t)          + avg          * t
    arr[:, w - blend_px:] = right_strip * (1 - t[:, ::-1]) + avg[:, ::-1] * t[:, ::-1]
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))

def _auto_low_vram() -> bool:
    if torch.cuda.is_available():
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return total_gb < 24
    return False


class WorldGenPanoGenerator(RemoteServer):

    def setup(self):
        device_str = str(self.device)
        self.depth_model = build_depth_model(self.device)
        self.pano_model = build_pano_fill_model(device=device_str, low_vram=_auto_low_vram())

    def _to_cubemap(self, pano_image: Image.Image) -> dict:
        cube_dict = py360convert.e2c(np.array(pano_image), face_w=512, cube_format="dict")
        return {
            k: Image.fromarray(np.clip(v, 0, 255).astype("uint8"))
            for k, v in cube_dict.items()
        }

    def _pano(self, input_image: Image.Image, seed: int, temp_path: Path, hfov_deg: float = DEFAULT_HFOV_DEG, caption: str = "") -> dict:
        self.report_progress(0.05, "Computing depth...")
        predictions = pred_depth(self.depth_model, input_image)

        # pred_depth assigns full-sphere equirectangular rays, which would
        # stretch the perspective input across the entire panorama. Replace with
        # rays matching the actual camera FOV so it occupies the right region.
        h, w = predictions["rgb"].shape[:2]
        predictions["rays"] = perspective_rays(h, w, hfov_deg, device=predictions["rgb"].device)

        self.report_progress(0.2, "Projecting image to panorama space...")
        pano_cond_img, cond_mask = map_image_to_pano(predictions, device=self.device)

        pano_w, pano_h = pano_cond_img.size

        self.report_progress(0.35, "Generating panorama fill...")
        pano_image = gen_pano_fill_image(
            self.pano_model,
            image=pano_cond_img,
            mask=cond_mask,
            prompt=caption,
            seed=seed,
            height=pano_h,
            width=pano_w,
        )

        self.report_progress(0.8, "Blending...")
        pano_image = pano_image.resize((pano_w, pano_h))
        pano_image = _lab_color_transfer(source=input_image, target=pano_image, strength=0.35)
        mask_arr = np.array(cond_mask) / 255.0
        blended = (
            np.array(pano_image) * mask_arr[:, :, None]
            + np.array(pano_cond_img) * (1 - mask_arr[:, :, None])
        )
        pano_image = _enforce_wrap_continuity(Image.fromarray(blended.astype(np.uint8)))

        self.report_progress(0.9, "Projecting cubemap...")
        return {"image": pano_image, "faces": self._to_cubemap(pano_image)}

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            try:
                input_image = input.get("image")
                if isinstance(input_image, np.ndarray):
                    input_image = Image.fromarray(input_image)
                fov = float(input.get("fov") or DEFAULT_HFOV_DEG)
                caption = input.get("caption", "")
                return self._pano(input_image, seed=int(input.get("seed", 0)), temp_path=temp_path, hfov_deg=fov, caption=caption)
            except Exception:
                traceback.print_exc()
                raise
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    WorldGenPanoGenerator.run()
