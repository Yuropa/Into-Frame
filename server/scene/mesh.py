import trimesh
import numpy as np
from typing import Self, TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image as PILImage

class Mesh:
    def __init__(self, mesh: trimesh.Trimesh) -> None:
        # trimesh.Trimesh
        self.mesh = mesh
        # Optional second UV set, e.g. a panorama-projection UV distinct from
        # whatever this mesh's own primary UV0 is (see
        # TerrainMeshGenerator.generate's panorama_uv return value) -- (N, 2),
        # aligned 1:1 with self.mesh.vertices. None (the default) exports a
        # perfectly normal single-UV mesh; set it to have save() inject it as
        # glTF TEXCOORD_1 (trimesh's own TextureVisuals only supports one UV
        # set natively, so this can't just be handed to it directly).
        self.extra_uv: "np.ndarray | None" = None
        # Optional replacement for whatever texture/UV0 combination got
        # embedded as this mesh's own material -- sampled via extra_uv
        # (TEXCOORD_1) instead. See util.gltf_uv2.inject_texcoord1's own
        # docstring for why the embedded material can need this even when
        # extra_uv's live consumer (Unity) never looks at it.
        self.preview_image: "PILImage.Image | None" = None

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

    def repair(self) -> "Mesh":
        """Return a new Mesh with a clean watertight surface via Poisson reconstruction.

        Raw meshes from reconstruction models are typically non-manifold (~50% broken
        faces). Poisson reconstruction samples the surface and fits a clean closed mesh,
        eliminating holes before any downstream use (rendering, decimation, etc.).
        """
        import open3d as o3d

        pts, face_ids = trimesh.sample.sample_surface(self.mesh, 50000)
        normals = self.mesh.face_normals[face_ids]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.orient_normals_consistent_tangent_plane(30)

        poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        densities = np.asarray(densities)
        poisson.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
        poisson.remove_degenerate_triangles()
        poisson.remove_duplicated_vertices()

        return Mesh(trimesh.Trimesh(
            vertices=np.asarray(poisson.vertices),
            faces=np.asarray(poisson.triangles),
        ))

    def simplify(self, max_error_fraction: float = 0.03, min_faces: int = 50) -> "Mesh":
        """Decimate as aggressively as possible while keeping geometric error within budget.

        Tries face counts 50, 100, 200, ... (doubling) and returns the first that keeps
        the 95th-percentile surface distance below max_error_fraction * bounding_box_diagonal.
        Assumes the mesh has already been repaired (call repair() first if needed).
        """
        import open3d as o3d

        bbox_diag = float(np.linalg.norm(self.mesh.extents)) or 1.0
        error_budget = max_error_fraction * bbox_diag
        sample_pts = self.mesh.sample(2000)

        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.mesh.vertices),
            triangles=o3d.utility.Vector3iVector(self.mesh.faces),
        )

        target = min_faces
        while target < len(self.mesh.faces):
            dec = o3d_mesh.simplify_quadric_decimation(target)
            dec.remove_degenerate_triangles()
            dec.remove_duplicated_triangles()
            dec.remove_duplicated_vertices()
            candidate = trimesh.Trimesh(
                vertices=np.asarray(dec.vertices),
                faces=np.asarray(dec.triangles),
            )
            _, distances, _ = trimesh.proximity.closest_point(candidate, sample_pts)
            if np.percentile(distances, 95) <= error_budget:
                return Mesh(candidate)
            target *= 2

        return Mesh(self.mesh.copy())

    def apply_crop_texture(self, image: "PILImage.Image") -> None:
        """Apply image as texture via orthographic projection onto the front face.

        Projects vertices onto the two axes of greatest extent. In a Y-up (glTF)
        scene axis 1 (Y) is always assigned to V so the image is vertically aligned,
        with the remaining horizontal axis assigned to U."""
        from trimesh.visual import TextureVisuals
        from trimesh.visual.material import PBRMaterial

        verts = self.mesh.vertices
        extent = verts.max(axis=0) - verts.min(axis=0)
        depth_axis = int(np.argmin(extent))
        plane_axes = [i for i in range(3) if i != depth_axis]

        # Ensure Y (axis 1, glTF up) maps to V (index 1). If it's currently at
        # index 0, swap so the horizontal axis is U and vertical is V.
        if plane_axes[0] == 1:
            plane_axes = [plane_axes[1], plane_axes[0]]

        uv = verts[:, plane_axes].copy()
        lo, hi = uv.min(axis=0), uv.max(axis=0)
        rng = np.where((hi - lo) > 0, hi - lo, 1.0)
        uv = (uv - lo) / rng

        material = PBRMaterial(baseColorTexture=image)
        self.mesh.visual = TextureVisuals(uv=uv, material=material)

    def save(self, path):
        self.mesh.export(str(path), include_normals=True)
        if self.extra_uv is not None and str(path).lower().endswith(".glb"):
            from util.gltf_uv2 import inject_texcoord1
            inject_texcoord1(path, self.extra_uv, preview_image=self.preview_image)

    @classmethod
    def load(cls, path) -> Self:
        loaded = trimesh.load(str(path), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Expected a single Trimesh at {path}, got {type(loaded).__name__}")
        return cls(loaded)