import trimesh

class Mesh:
    def __init__(self, mesh) -> None:
        # trimesh.Trimesh
        self.mesh = mesh

    def save(self, path):
        self.mesh.export(str(path), include_normals=True)