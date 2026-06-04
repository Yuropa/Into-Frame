from path_utils import add_project_paths, lib_path, checkpoints_path, add_system_path, run_relative_to
add_project_paths()

add_system_path(lib_path() / "DAP")

from pathlib import Path
from typing import Any

import yaml
import torch
import torch.nn as nn
import numpy as np
from PIL import Image as PILImage

from networks.models import make
from remote_connection.remote_server import RemoteServer


class PanoDepthGenerator(RemoteServer):
    def setup(self):
        dap_root = lib_path() / "DAP"
        config_path = dap_root / "config" / "infer.yaml"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        weights_dir = checkpoints_path() / "depth_pano"
        model_path = weights_dir / "model.pth"

        state = torch.load(str(model_path), map_location=self.device)

        with run_relative_to(dap_root):
            model = make(config["model"])

            if any(k.startswith("module") for k in state.keys()):
                model = nn.DataParallel(model)

            model_state = model.state_dict()
            model.load_state_dict(
                {k: v for k, v in state.items() if k in model_state},
                strict=False,
            )

        self.model = model.to(self.device)
        self.model.eval()
        print(f"DAP model loaded from {model_path}")

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "depth":
            return self.depth(input)
        raise ValueError(f"Unknown action: {action}")

    def depth(self, input_image: PILImage.Image) -> dict:
        img = input_image.convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0
        tensor = (
            torch.from_numpy(img_np.transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )

        self.report_progress(0.1, "Running panorama depth inference...")
        with torch.inference_mode():
            outputs = self.model(tensor)

            if isinstance(outputs, dict) and "pred_depth" in outputs:
                if "pred_mask" in outputs:
                    mask = (1.0 - outputs["pred_mask"]) > 0.5
                    outputs["pred_depth"][~mask] = 1.0
                pred = outputs["pred_depth"][0].detach().cpu().squeeze().numpy()
            else:
                pred = outputs[0].detach().cpu().squeeze().numpy()

        # DAP output is 0–1 where 1.0 = 100 m; convert to metric metres
        depth_metres = pred.astype(np.float32) * 100.0
        self.report_progress(1.0, "Done")
        return {"depth": depth_metres}


if __name__ == "__main__":
    PanoDepthGenerator.run()
