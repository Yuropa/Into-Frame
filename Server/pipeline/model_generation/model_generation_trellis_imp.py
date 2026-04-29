from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import o_voxel
from PIL import Image
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from pipeline.model_generation.model_generation_base_imp import ModelGeneratorBase

class ModelGenerator(ModelGeneratorBase):
    def setup(self):
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(
            "microsoft/TRELLIS.2-4B"
        )
        self.pipeline.to(self.device)
    
    def pad_image(self, image: Image, padding: float = 0.15) -> Image:
        w, h = image.size
        pad_x = int(w * padding)
        pad_y = int(h * padding)
        padded = Image.new("RGBA", (w + pad_x * 2, h + pad_y * 2), (0, 0, 0, 0))
        padded.paste(image.convert("RGBA"), (pad_x, pad_y))
        return padded

    def meshify(self, temp_path: Path, input: Image) -> Any:
        padded_image = self.pad_image(input)
        mesh = self.pipeline.run(padded_image)[0]
        mesh.simplify(500000)

        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.3], [0.5, 0.5, 0.5]],
            decimation_target   =   50000,
            texture_size        =   1024,
            remesh              =   True,
            remesh_band         =   0.5,
            remesh_project      =   0,
            verbose             =   True
        )
        return glb
    
if __name__ == "__main__":
    ModelGenerator.run()
