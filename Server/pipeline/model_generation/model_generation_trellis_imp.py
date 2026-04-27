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
    
    def meshify(self, temp_path: Path, input: Image) -> Any:
        mesh = self.pipeline.run(input)[0]
        mesh.simplify(16777216)

        glb = o_voxel.postprocess.to_glb(
            vertices            =   mesh.vertices,
            faces               =   mesh.faces,
            attr_volume         =   mesh.attrs,
            coords              =   mesh.coords,
            attr_layout         =   mesh.layout,
            voxel_size          =   mesh.voxel_size,
            aabb                =   [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target   =   1000000,
            texture_size        =   4096,
            remesh              =   True,
            remesh_band         =   1,
            remesh_project      =   0,
            verbose             =   True
        )
        return glb
    
if __name__ == "__main__":
    ModelGenerator.run()
