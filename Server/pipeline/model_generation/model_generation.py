import torch
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils

from scene.mesh import Mesh
from util.image_utils import Image

class ModelGenerator():
    def __init__(self, device) -> None:
        self.device = device
        self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B").to(device)

    @classmethod
    def model_names(cls) -> list[str]:
        return ["microsoft/TRELLIS.2-4B"]
    
    def meshify(self, image):
        mesh = self.pipeline.run(image.image)[0]
        mesh.simplify(16777216) # nvdiffrast limit
        return Mesh(mesh)