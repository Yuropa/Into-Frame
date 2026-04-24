from pathlib import Path
import trimesh
import base64
from io import BytesIO

from scene.mesh import Mesh
from util.image_utils import Image
from remote_connection.remote_client import RemoteClient 

class ModelGeneratorBase(RemoteClient):
    @classmethod
    def model_names(cls) -> list[str]:
        return []
    
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
    