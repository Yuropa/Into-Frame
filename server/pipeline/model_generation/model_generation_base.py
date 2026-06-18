from pathlib import Path
import trimesh

from scene.mesh import Mesh
from util.image_utils import Image
from remote_connection.remote_client import RemoteClient 

class ModelGeneratorBase(RemoteClient):
    @classmethod
    def model_names(cls) -> list[str]:
        return []
    
    def meshify(self, image: Image, temp_path: Path, seed: int = 0) -> Mesh:
        response = self.send(action="meshify", input={"image": image.image, "seed": seed}, temp_path=temp_path)
        glb_path = Path(response)

        try:
            mesh = trimesh.load(str(glb_path))

            if isinstance(mesh, trimesh.Scene):
                geoms = list(mesh.geometry.values())
                # For a single geometry keep it as-is so TextureVisuals are preserved.
                # concatenate() rebuilds a bare Trimesh and drops material/UV data.
                if len(geoms) == 1:
                    mesh = geoms[0]
                else:
                    mesh = trimesh.util.concatenate(geoms)

            return Mesh(mesh)
        finally:
            glb_path.unlink(missing_ok=True)
    