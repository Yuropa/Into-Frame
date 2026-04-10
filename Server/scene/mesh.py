import o_voxel

class Mesh:
    def __init__(self, mesh) -> None:
        self.mesh = mesh

    def save(self, path):
        glb = o_voxel.postprocess.to_glb(
            vertices=self.mesh.vertices,
            faces=self.mesh.faces,
            attr_volume=self.mesh.attrs,
            coords=self.mesh.coords,
            attr_layout=self.mesh.layout,
            voxel_size=self.mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1000000,
            texture_size=4096,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=True
        )
        glb.export(str(path), extension_webp=True)