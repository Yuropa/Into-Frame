from pathlib import Path
import trimesh

from scene.mesh import Mesh
from util.image_utils import Image
from remote_connection.remote_client import RemoteClient 

class ModelGeneratorBase(RemoteClient):
    @classmethod
    def model_names(cls) -> list[str]:
        return []
    
    def meshify(self, image: Image, temp_path: Path) -> Mesh:
        response = self.send(action="meshify", input=self.encode_image(image), temp_path=temp_path)
        glb_path = Path(response)

        try:
            mesh = trimesh.load(str(glb_path))

            # trimesh.load returns a Scene for GLB — extract the geometry
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

            return Mesh(mesh)
        finally:
            glb_path.unlink(missing_ok=True)
    