import torch
import trimesh
from pathlib import Path

from scene.mesh import Mesh
from util.image_utils import Image
from remote_connection.remote_client import RemoteClient


class ModelGeneratorTreeD(RemoteClient):
    def __init__(self, device: torch.device) -> None:
        script_path = Path(__file__).parent / "model_generation_treed_imp.py"
        super().__init__(
            device=device,
            conda_env="treed",
            script_path=script_path,
        )

    def meshify(self, image: Image, temp_path: Path, seed: int = 0, species: str = "") -> Mesh:
        response = self.send(
            action="meshify",
            input={"image": image.image, "seed": seed, "species": species},
            temp_path=temp_path,
        )
        glb_path = Path(response)
        try:
            mesh = trimesh.load(str(glb_path))
            if isinstance(mesh, trimesh.Scene):
                geoms = list(mesh.geometry.values())
                mesh = geoms[0] if len(geoms) == 1 else trimesh.util.concatenate(geoms)
            return Mesh(mesh)
        finally:
            glb_path.unlink(missing_ok=True)
