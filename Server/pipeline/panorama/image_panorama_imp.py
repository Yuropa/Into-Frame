from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
import math
from PIL import Image
from util.json_utils import write_json
from torchvision import transforms
from remote_connection.remote_server import RemoteServer
from cubediff.pipelines.pipeline import CubeDiffPipeline

class PanoGenerator(RemoteServer):
    def setup(self):
        self.pipeline = CubeDiffPipeline.from_pretrained(
            "hlicai/cubediff-512-imgonly"
        )
        self.pipeline.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])

    def pano(self, temp_path: Path, input_image: Image, fov_degrees: float = 60.0, caption: str = "") -> Image.Image:
        result = self.pipeline(
            prompts="",
            conditioning_image=self.transform(input_image).unsqueeze(0).to(self.device),
            num_inference_steps=20
        )

        for i, face in enumerate(result.faces):
            Image.fromarray(face).save(str(temp_path / f"faces_{i}.png"))
            
        for i, face in enumerate(result.faces_cropped):
            Image.fromarray(face).save(str(temp_path / f"faces_cropped_{i}.png"))

        return Image.fromarray(result.equirectangular)

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            print(f"Got input image {input}")
            image = input["image"]
            fov = input.get("fov_degrees", 60.0)
            caption = input["caption"]
            return self.pano(temp_path, image, fov_degrees=fov, caption=caption)
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    PanoGenerator.run()