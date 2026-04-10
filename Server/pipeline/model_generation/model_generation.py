import torch
from transparent_background import Remover
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, remove_background

from scene.mesh import Mesh
from util.image_utils import Image

class ModelGenerator():
    def __init__(self, device) -> None:
        self.device = device
        self.image_resolution = 2024
        self.foreground_ratio = 1.3
        low_vram_mode=False
        self.model = SPAR3D.from_pretrained(
            "stabilityai/stable-point-aware-3d",
            low_vram_mode=low_vram_mode
        )
        self.model.to(device)
        self.model.eval()

        self.bg_remove = Remover(device=device)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["stabilityai/stable-point-aware-3d"]
    
    def meshify(self, image):
        cleaned_image = remove_background(
            image.image.convert("RGBA"),
            self.bg_remove
        )
        cleaned_image = foreground_crop(
            cleaned_image,
            self.foreground_ratio
        )

        mesh, glob_dict = self.model.run_image(
            image,
            bake_resolution=self.image_resolution,
            remesh="triangle",
            vertex_count=-1,
            return_points=True,
        )
        
        return Mesh(mesh)