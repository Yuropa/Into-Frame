from path_utils import add_project_paths
add_project_paths()

import torch
import numpy as np
from pathlib import Path
from typing import Any
from PIL import Image

from transformers import Swin2SRForImageSuperResolution, Swin2SRImageProcessor
from remote_connection.remote_server import RemoteServer


class SupersamplingServer(RemoteServer):
    def setup(self):
        self.processor = Swin2SRImageProcessor.from_pretrained("caidas/swin2SR-classical-sr-x2-64")
        self.model = Swin2SRForImageSuperResolution.from_pretrained("caidas/swin2SR-classical-sr-x2-64").to(self.device)

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "supersample":
            return self._supersample(input)
        raise ValueError(f"Unknown action: {action}")

    def _supersample(self, image: Image.Image) -> Image.Image:
        self.report_progress(0.1, "Running supersampling…")
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs).reconstruction
        self.report_progress(1.0, "Done")
        output = output.squeeze().cpu().clamp(0, 1).numpy()
        output = (output * 255).astype(np.uint8)
        output = np.transpose(output, (1, 2, 0))
        return Image.fromarray(output)


if __name__ == "__main__":
    SupersamplingServer.run()
