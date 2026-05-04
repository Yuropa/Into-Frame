from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import torch
import math
from PIL import Image
from util.json_utils import write_json
from remote_connection.remote_server import RemoteServer
import torchvision.transforms as T
import pano_dreamer.multicondiffusion_panorama as pano_module

class FovCylindricalPanorama(pano_module.CylindricalPanorama):
    def image_to_cylindrical_panorama(self, scene, input_image, prompt,
                                       negative_prompt='', height=512, width=3912,
                                       num_inference_steps=50, guidance_scale=7.5,
                                       num_iterations=15, save_dir='output',
                                       debug=False, fov_degrees=44.701948991275390):
        # The base class calls fov2focal(input_fov * math.pi / 180, ...)
        # We patch the module-level fov2focal so our FOV is used instead
        original_fov2focal = pano_module.fov2focal
        target_fov_rad = fov_degrees * math.pi / 180

        def patched_fov2focal(fov_radians, pixels):
            # Replace whatever FOV the base class passes with ours
            return original_fov2focal(target_fov_rad, pixels)

        pano_module.fov2focal = patched_fov2focal
        try:
            return super().image_to_cylindrical_panorama(
                scene=scene,
                input_image=input_image,
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                num_iterations=num_iterations,
                save_dir=save_dir,
                debug=debug,
            )
        finally:
            pano_module.fov2focal = original_fov2focal

class PanoGenerator(RemoteServer):
    def setup(self):
        pano_module.seed_everything(42)
        self.model = FovCylindricalPanorama(self.device)

    def pano(self, temp_path: Path, input_image: Image.Image, fov_degrees: float = 60.0) -> Image.Image:
        prompt = "A seamless 360 degree panorama. High resolution, 8k, photorealistic."
        negative_prompt = "caption, subtitle, text, blur, lowres, bad anatomy, bad hands, cropped, worst quality, watermark"

        return self.model.image_to_cylindrical_panorama(
            scene="pano",
            input_image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            save_dir=str(temp_path),
            fov_degrees=fov_degrees,
        )

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "pano":
            print(f"Got input image {input}")
            image = input["image"]
            fov = input.get("fov_degrees", 60.0)
            return self.pano(temp_path, image, fov_degrees=fov)
        raise ValueError(f"Unknown action: {action}")

if __name__ == "__main__":
    PanoGenerator.run()