import torch
from pathlib import Path
from enum import Enum

from scene.mesh import Mesh
from util.image_utils import Image
from pipeline.model_generation.model_generation_base import ModelGeneratorBase
from pipeline.model_generation.model_generation_spar3d import ModelGeneratorSpar3D
from pipeline.model_generation.model_generation_trellis import ModelGeneratorTrellis

class ModelGeneratorType(Enum):
    SPAR3D = 1
    TRELLIS = 2

    @classmethod
    def default(cls):
        return cls.TRELLIS

class ModelGenerator():
    generator: ModelGeneratorBase

    def __init__(self, device: torch.device, type: ModelGeneratorType = ModelGeneratorType.default()) -> None:
        match type:
            case ModelGeneratorType.SPAR3D:
                self.generator = ModelGeneratorSpar3D(
                    device=device
                )
            case ModelGeneratorType.TRELLIS:
                self.generator = ModelGeneratorTrellis(
                    device=device
                )

    @classmethod
    def model_names(cls, type: ModelGeneratorType = ModelGeneratorType.default()) -> list[str]:
        match type:
            case ModelGeneratorType.SPAR3D:
                return ModelGeneratorSpar3D.model_names()
            case ModelGeneratorType.TRELLIS:
                return ModelGeneratorTrellis.model_names()
    
    def meshify(self, image: Image, temp_path: Path) -> Mesh:
        return self.generator.meshify(
            image=image,
            temp_path=temp_path
        )

    def close(self):
        self.generator.close()
