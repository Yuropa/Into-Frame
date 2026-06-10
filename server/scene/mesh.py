import trimesh
import numpy as np
from typing import Self, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

class Mesh:
    def __init__(self, mesh: trimesh.Trimesh) -> None:
        # trimesh.Trimesh
        self.mesh = mesh

    @property
    def vertex_count(self) -> int:
        return len(self.mesh.vertices)

    @property
    def face_count(self) -> int:
        return len(self.mesh.faces)

    @property
    def extents(self) -> np.ndarray:
        return self.mesh.extents

    @property
    def center(self) -> np.ndarray:
        return self.mesh.centroid
    
    def fit_to_box(self, width: float, height: float) -> None:
        scale = min(width / self.extents[0], height / self.extents[1])
        self.mesh.apply_scale(scale)
        self.mesh.apply_translation(-self.mesh.centroid)

    def simplify(self, face_fraction: float = 0.25) -> "Mesh":
        """Return a new Mesh with approximately face_fraction of the original faces."""
        target = max(4, int(len(self.mesh.faces) * face_fraction))
        simplified = self.mesh.simplify_quadric_decimation(target)
        return Mesh(simplified)

    def apply_crop_texture(self, image: "PILImage.Image") -> None:
        """Apply image as texture via orthographic projection along the mesh's shallowest axis.

        Projects vertices onto the two axes of greatest extent so the crop photo
        maps naturally onto the visible face of the object."""
        from trimesh.visual import TextureVisuals
        from trimesh.visual.material import PBRMaterial

        verts = self.mesh.vertices
        extent = verts.max(axis=0) - verts.min(axis=0)
        depth_axis = int(np.argmin(extent))
        plane_axes = [i for i in range(3) if i != depth_axis]

        uv = verts[:, plane_axes].copy()
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        rng = np.where((hi - lo) > 0, hi - lo, 1.0)
        uv = (uv - lo) / rng
        uv[:, 1] = 1.0 - uv[:, 1]  # flip V: image Y is top-down, 3D Y is up

        material = PBRMaterial(baseColorTexture=image)
        self.mesh.visual = TextureVisuals(uv=uv, material=material)

    def save(self, path):
        self.mesh.export(str(path), include_normals=True)

    @classmethod
    def load(cls, path) -> Self:
        loaded = trimesh.load(str(path), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Expected a single Trimesh at {path}, got {type(loaded).__name__}")
        return cls(loaded)