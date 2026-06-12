from path_utils import add_project_paths, lib_path, checkpoints_path, add_system_path
add_project_paths()

add_system_path(lib_path() / "recognize-anything")

from pathlib import Path
from typing import Any

import torch
from PIL import Image as PILImage

from ram.models import ram_plus
from ram import inference_ram as inference
from ram import get_transform
from remote_connection.remote_server import RemoteServer


class RAMPlusServer(RemoteServer):
    def setup(self):
        checkpoint = checkpoints_path() / "recognize_anything" / "ram_plus_swin_large_14m.pth"
        if not checkpoint.exists():
            raise FileNotFoundError(f"RAM++ checkpoint not found: {checkpoint}")

        image_size = 384
        self.transform = get_transform(image_size=image_size)

        model = ram_plus(pretrained=str(checkpoint), image_size=image_size, vit="swin_l")
        model.eval()
        self.model = model.to(self.device)
        print(f"RAM++ loaded from {checkpoint}")

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "recognize":
            return self.recognize(input)
        raise ValueError(f"Unknown action: {action}")

    def recognize(self, input_image: PILImage.Image) -> dict:
        image = self.transform(input_image.convert("RGB")).unsqueeze(0).to(self.device)

        self.report_progress(0.1, "Running RAM++ inference…")
        with torch.no_grad():
            res = inference(image, self.model)

        self.report_progress(1.0, "Done")
        return {"tags": res[0]}  # English pipe-separated tags, e.g. "dog | tree | outdoor"


if __name__ == "__main__":
    RAMPlusServer.run()
