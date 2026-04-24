import torch
import trimesh
import base64
from io import BytesIO
from pathlib import Path
import clip

from scene.mesh import Mesh
from util.image_utils import Image
from pipeline.model_generation.model_generation_base import ModelGeneratorBase

class ModelGeneratorSpar3D(ModelGeneratorBase):
    def __init__(self, device: torch.device) -> None:
        clip.load('ViT-L/14@336px')
        script_path = Path(__file__).parent / "model_generation_spar3d_imp.py"

        super().__init__(
            device=device, 
            conda_env="stablepoint", 
            script_path=script_path
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["stabilityai/stable-point-aware-3d"]