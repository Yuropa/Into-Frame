from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
import math
import py360convert
import numpy as np
from PIL import Image
from util.json_utils import write_json
from torchvision import transforms
from remote_connection.remote_server import RemoteServer
from cubediff.pipelines.pipeline import CubeDiffPipeline

class PanoGenerator(RemoteServer):
    def setup(self):
        self.pipeline = CubeDiffPipeline.from_pretrained(
            "hlicai/cubediff-512-singlecaption"
        )
        self.pipeline.to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor()
        ])

    def pano(self, temp_path: Path, input_image: Image, fov_degrees: float = 60.0, caption: str = "") -> dict:
        if caption is None:
            caption = ""

        result = self.pipeline(
            prompts=caption,
            conditioning_image=self.transform(input_image).unsqueeze(0).to(self.device),
            num_inference_steps=50
        )

        cube_dict = {
            "F": result.faces_cropped[0],
            "B": result.faces_cropped[1],
            "L": result.faces_cropped[2],  # swap: model's "left" is py360's "right"
            "R": result.faces_cropped[3],  # swap: model's "right" is py360's "left"
            "U": result.faces_cropped[4],
            "D": result.faces_cropped[5],
        }
        equirectangular_image = py360convert.c2e(cube_dict, h=1024, w=2048, cube_format='dict')
        equirectangular_image = np.clip(equirectangular_image, 0, 255).astype(np.uint8)

        for i, face in enumerate(result.faces):
            Image.fromarray(face).save(str(temp_path / f"faces_{i}.png"))
        
        for i, face in enumerate(result.faces_cropped):
            Image.fromarray(face).save(str(temp_path / f"faces_cropped_{i}.png"))

        result = {k: Image.fromarray(v) for k, v in cube_dict.items()}
        return {
            "image" : Image.fromarray(equirectangular_image),
            "faces": result
        }

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