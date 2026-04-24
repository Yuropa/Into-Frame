from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

from PIL import Image
from transparent_background import Remover
from spar3d.system import SPAR3D
from spar3d.utils import foreground_crop, remove_background
from pipeline.model_generation.model_generation_base_imp import ModelGeneratorBase

class ModelGenerator(ModelGeneratorBase):
    def __init__(self) -> None:
        super().__init__()

        self.image_resolution = 1024
        self.foreground_ratio = 1.3
        self.model = SPAR3D.from_pretrained(
            "stabilityai/stable-point-aware-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
            low_vram_mode=False
        )
        self.model.to(self.device)
        self.model.eval()
        self.bg_remove = Remover(device=self.device)
    
    def meshify(self, temp_path: Path, input: Image) -> Any:
        cleaned_image = remove_background(input, self.bg_remove)
        cleaned_image = foreground_crop(cleaned_image, self.foreground_ratio)

        cleaned_image.save(str(temp_path / "cleaned_image.png"))

        mesh, glob_dict = self.model.run_image(
            cleaned_image,
            bake_resolution=self.image_resolution,
            remesh="quad",
            vertex_count=-1,
            return_points=True
        )

        if "point_clouds" in glob_dict:
            glob_dict["point_clouds"][0].export(str(temp_path / "points.ply"))
            
        return mesh

if __name__ == "__main__":
    ModelGenerator.run()