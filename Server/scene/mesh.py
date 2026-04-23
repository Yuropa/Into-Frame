import trimesh
import numpy as np

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

    def save(self, path):
        self.mesh.export(str(path), include_normals=True)