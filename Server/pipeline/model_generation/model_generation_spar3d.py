import torch
import trimesh
import base64
from io import BytesIO
from pathlib import Path
import clip

# from transformers import Sam3Processor, Sam3Model

from scene.mesh import Mesh
from util.image_utils import Image
from util.json_utils import parse_json, write_json
from pipeline.model_generation.model_generation_base import ModelGeneratorBase
from remote_connection.remote_client import RemoteClient

class ModelGeneratorSpar3D(RemoteClient, ModelGeneratorBase):
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
    
    def meshify(self, image: Image, temp_path: Path) -> Mesh:
        buffer = BytesIO()
        image.image.save(buffer, format="PNG")
        image_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        response = self.send(action="meshify", input=image_b64, temp_path=temp_path)
        glb_path = Path(response)

        try:
            mesh = trimesh.load(str(glb_path))

            # trimesh.load returns a Scene for GLB — extract the geometry
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

            return Mesh(mesh)
        finally:
            glb_path.unlink(missing_ok=True)