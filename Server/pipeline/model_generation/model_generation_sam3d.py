import torch
import trimesh
import base64
from io import BytesIO
from pathlib import Path
import clip

from scene.mesh import Mesh
from util.image_utils import Image
from pipeline.model_generation.model_generation_base import ModelGeneratorBase

class ModelGeneratorSAM3D(ModelGeneratorBase):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "model_generation_sam3d_imp.py"

        super().__init__(
            device=device, 
            conda_env="sam3d", 
            script_path=script_path,
            env_options={
                "LIDRA_SKIP_INIT": 1
            }
        )

    @classmethod
    def model_names(cls) -> list[str]:
        return ["facebook/sam-3d-objects"]