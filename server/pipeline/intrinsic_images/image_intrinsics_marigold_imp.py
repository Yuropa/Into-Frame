"""Marigold-IID worker: albedo (and shading) from a single image.

The lighting variant of Marigold Intrinsic Image Decomposition (prs-eth/
marigold-iid-lighting-v1-1, diffusers MarigoldIntrinsicsPipeline) predicts
albedo, diffuse shading and a non-diffuse residual. It is a stronger delighter
than IntrinsicDiffusion on outdoor photographs: its albedo takes the colour of
the light out as well as its intensity (a sunset cliff comes back stone-coloured,
not orange), which the scalar shading division in util.delight cannot do.

Runs in the `frame` env (diffusers >= 0.38). Same protocol as
image_intrinsics_imp.py: action "intrinsics" -> {"albedo", "normal"}; albedo is
linear reflectance like IntrinsicDiffusion's, so every consumer is unchanged.
Marigold predicts no normals, so a flat +Z normal map is returned.
"""
from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from remote_connection.remote_server import RemoteServer

MODEL_ID = "prs-eth/marigold-iid-lighting-v1-1"


class MarigoldIntrinsicsGenerator(RemoteServer):
    def setup(self):
        from diffusers import MarigoldIntrinsicsPipeline

        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        try:
            pipe = MarigoldIntrinsicsPipeline.from_pretrained(MODEL_ID, variant="fp16", torch_dtype=dtype)
        except Exception:
            pipe = MarigoldIntrinsicsPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
        self.pipe = pipe.to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        props = getattr(self.pipe, "target_properties", None) or {}
        self.target_names = list(props.get("target_names", ["albedo", "shading", "residual"]))
        self.prediction_space = {
            n: (props.get(n, {}) or {}).get("prediction_space", "srgb") for n in self.target_names
        }

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "intrinsics":
            return self.intrinsics(
                input["image"],
                resolution=input.get("resolution", 768),
                agg_num=input.get("agg_num", 1),
            )
        raise ValueError(f"Unknown action: {action}")

    def intrinsics(self, image: PILImage.Image, resolution: int, agg_num: int) -> dict:
        rgb = image.convert("RGB")
        self.report_progress(0.1, "Running Marigold-IID inference…")
        with torch.no_grad():
            out = self.pipe(
                rgb,
                ensemble_size=max(1, int(agg_num)),
                processing_resolution=int(resolution) if resolution else None,
                match_input_resolution=True,
                output_type="np",
            )
        pred = np.asarray(out.prediction, dtype=np.float32)
        # (N, T, H, W, 3) for the multi-target models; (N, H, W, 3) if only one target.
        if pred.ndim == 4:
            pred = pred[:, None]
        idx = self.target_names.index("albedo") if "albedo" in self.target_names else 0
        albedo = np.clip(pred[0, idx], 0.0, 1.0)
        if self.prediction_space.get("albedo", "srgb") == "srgb":
            albedo = albedo ** 2.2  # IntrinsicDiffusion's convention: linear reflectance
        if albedo.shape[:2] != (rgb.height, rgb.width):
            albedo = np.asarray(
                PILImage.fromarray((albedo * 255.0 + 0.5).astype(np.uint8)).resize(rgb.size, PILImage.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        normal = np.zeros_like(albedo)
        normal[..., 2] = 1.0
        self.report_progress(1.0, "Done")
        return {"albedo": albedo.astype(np.float32), "normal": normal.astype(np.float32)}


if __name__ == "__main__":
    MarigoldIntrinsicsGenerator.run()
