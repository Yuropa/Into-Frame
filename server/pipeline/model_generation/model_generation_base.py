import json
from pathlib import Path
import numpy as np
import trimesh

from scene.mesh import Mesh
from util.image_utils import Image
from remote_connection.remote_client import RemoteClient 

class ModelGeneratorBase(RemoteClient):
    # True when the generator's texture is its own albedo estimate and it wants the
    # photographed (lit) crop as input; False when it bakes the input pixels in
    # as-is and should be fed the delit crop.
    produces_albedo = False

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

            # The reconstructor hands back a conformant glTF model: right-handed,
            # +X the object's right as photographed. The scene frame is its mirror
            # (see Mesh._mirror_x), and the file is read verbatim, so until now
            # every generated mesh stood in the scene flipped left-to-right --
            # invisible on a tree or a boat, wrong on a rock that leans one way.
            out = Mesh(Mesh._mirror_x(mesh))
            out.toward_camera = read_toward_camera(temp_path)
            return out
        finally:
            glb_path.unlink(missing_ok=True)


def read_toward_camera(temp_path: "Path | None") -> "np.ndarray | None":
    """The unit direction from a generated mesh to the photographer, in the
    mesh's own (scene-frame) coordinates, from the pose the SAM3D worker wrote next
    to it (model_generation_sam3d_imp._write_pose); None when there is none (other
    generators, older runs). The worker's direction is in the GLB's right-handed
    frame, so its X flips with the mesh's (meshify above)."""
    if temp_path is None:
        return None
    pose_path = Path(temp_path) / "pose.json"
    if not pose_path.exists():
        return None
    try:
        with open(pose_path) as f:
            d = np.asarray(json.load(f).get("toward_camera_glb"), dtype=float)
    except (OSError, ValueError, TypeError):
        return None
    if d.shape != (3,) or not np.all(np.isfinite(d)) or np.linalg.norm(d) < 1e-6:
        return None
    d = d / np.linalg.norm(d)
    return np.array([-d[0], d[1], d[2]], dtype=float)
    