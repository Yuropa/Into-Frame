from pathlib import Path
from typing import Protocol

from scene.mesh import Mesh
from util.image_utils import Image

class ModelGeneratorBase(Protocol):
    @classmethod
    def model_names(cls) -> list[str]:
        return []
    
    def meshify(self, image: Image, temp_path: Path) -> Mesh:
        pass