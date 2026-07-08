from path_utils import add_project_paths, add_system_path, lib_path, checkpoints_path
add_project_paths()
add_system_path(lib_path() / "IntrinsicDiffusion")

import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image as PILImage
from remote_connection.remote_server import RemoteServer

class IntrinsicImagesGenerator(RemoteServer):
    def setup(self):
        from modeling.intrinsicdiffusion_pipe import load_model

        checkpoint_dir = checkpoints_path() / "intrinsic_diffusion"
        args = types.SimpleNamespace(
            pretrained_model_name_or_path="ptx0/pseudo-journey-v2",
            controlnet_model_name_or_path=str(checkpoint_dir / "controlnet"),
            prompt_adapter_path=str(checkpoint_dir / "prompt_adapter"),
            vae_scaling_factor=None,
            ddim_steps=20,
            guidance_scale=1.0,
            agg_num=4,
            enable_xformers_memory_efficient_attention=(self.device.type == "cuda"),
        )
        self.model = load_model("controlnet", self.device, args)

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "intrinsics":
            return self.intrinsics(
                input["image"],
                resolution=input.get("resolution", 1024),
                agg_num=input.get("agg_num", 4),
            )
        raise ValueError(f"Unknown action: {action}")

    def intrinsics(self, image: PILImage.Image, resolution: int, agg_num: int) -> dict:
        rgb = image.convert("RGB")
        tensor = torch.from_numpy(np.array(rgb, dtype=np.float32) / 255.0)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

        if agg_num != self.model.control_joint.agg_num:
            self.model.control_joint.update_inference_params(agg_num=agg_num)

        self.report_progress(0.1, "Running IntrinsicDiffusion inference…")
        with torch.no_grad():
            albedo, _, normal, _ = self.model.infer_intrinsic_images(
                _input_img=tensor,
                infer_img_size=resolution,
                is_linear_input=False,
                output_original_size=True,
                pred_albedo=True,
                pred_shading=False,
                pred_normal=True,
            )
        self.report_progress(1.0, "Done")

        albedo = albedo[0].to(torch.float32).cpu().permute(1, 2, 0).numpy()
        normal = normal[0].to(torch.float32).cpu().permute(1, 2, 0).numpy()
        return {"albedo": albedo, "normal": normal}

if __name__ == "__main__":
    IntrinsicImagesGenerator.run()
