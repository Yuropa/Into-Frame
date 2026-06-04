from path_utils import add_project_paths
add_project_paths()

import torch
import numpy as np
import py360convert
import traceback
from pathlib import Path
from typing import Any
from PIL import Image

from remote_connection.remote_server import RemoteServer
from worldgen.pano_gen import build_pano_fill_model, gen_pano_fill_image
from worldgen.pano_depth import build_depth_model, pred_depth
from worldgen.utils.general_utils import map_image_to_pano


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

    def _pano(self, input_image: Image.Image, seed: int) -> dict:
        self.report_progress(0.05, "Computing depth...")
        predictions = pred_depth(self.depth_model, input_image)

        self.report_progress(0.2, "Projecting image to panorama space...")
        pano_cond_img, cond_mask = map_image_to_pano(predictions, device=self.device)

        pano_w, pano_h = pano_cond_img.size

        self.report_progress(0.35, "Generating panorama fill...")
        pano_image = gen_pano_fill_image(
            self.pano_model,
            image=pano_cond_img,
            mask=cond_mask,
            prompt="",
            seed=seed,
            height=pano_h,
            width=pano_w,
        )

        self.report_progress(0.8, "Blending...")
        pano_image = pano_image.resize((pano_w, pano_h))
        mask_arr = np.array(cond_mask) / 255.0
        blended = (
            np.array(pano_image) * mask_arr[:, :, None]
            + np.array(pano_cond_img) * (1 - mask_arr[:, :, None])
        )
        pano_image = Image.fromarray(blended.astype(np.uint8))

        self.report_progress(0.9, "Projecting cubemap...")
        return {"image": pano_image, "faces": self._to_cubemap(pano_image)}

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            try:
                input_image = input.get("image")
                if isinstance(input_image, np.ndarray):
                    input_image = Image.fromarray(input_image)
                return self._pano(input_image, seed=int(input.get("seed", 0)))
            except Exception:
                traceback.print_exc()
                raise
        raise ValueError(f"Unknown action: {action}")


if __name__ == "__main__":
    WorldGenPanoGenerator.run()
