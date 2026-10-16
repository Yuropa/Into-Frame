import trimesh
import numpy as np
from pathlib import Path
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
        # Optional vertical bone chain (world/mesh-local Y heights, base-to-tip)
        # for a gentle procedural sway animation -- see util.gltf_skin.inject_skin's
        # own docstring for the glTF skin (joints/weights/inverseBindMatrices) this
        # injects into the GLB at save() time. None (the default) exports a plain
        # unskinned mesh, same as today.
        self.skin_bone_heights: "list[float] | None" = None
        self.skin_bone_names: "list[str] | None" = None

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

    @property
    def has_texture(self) -> bool:
        """True when this mesh carries a real UV-mapped texture image.

        SAM3D bakes one (see ModelGenerator.meshify's with_texture_baking) and it is
        the whole difference between a boat with windows and lettering on it and a
        smear of averaged vertex colour -- so callers that are about to run a
        topology-destroying pass need to be able to ask.
        """
        visual = self.mesh.visual
        if not isinstance(visual, trimesh.visual.TextureVisuals):
            return False
        if visual.uv is None or len(visual.uv) != len(self.mesh.vertices):
            return False
        material = getattr(visual, "material", None)
        return (
            getattr(material, "baseColorTexture", None) is not None
            or getattr(material, "image", None) is not None
        )

    def _vertex_colors_01(self) -> "np.ndarray | None":
        """(N, 3) per-vertex RGB in [0, 1], baking down TextureVisuals if needed. None if unset."""
        visual = self.mesh.visual
        if getattr(visual, "kind", None) is None:
            return None
        # ColorVisuals exposes vertex_colors directly; TextureVisuals needs baking first.
        colors = visual.vertex_colors if isinstance(visual, trimesh.visual.ColorVisuals) else visual.to_color().vertex_colors
        return np.asarray(colors)[:, :3] / 255.0

    def repair(self) -> "Mesh":
        """Return a new Mesh with a clean watertight surface via Poisson reconstruction.

        Raw meshes from reconstruction models are typically non-manifold (~50% broken
        faces). Poisson reconstruction samples the surface and fits a clean closed mesh,
        eliminating holes before any downstream use (rendering, decimation, etc.).

        Poisson reconstruction discards the original topology, so any texture/UV
        material can't carry over directly. Baked colors are carried through instead:
        sampled onto the surface point cloud and reconstructed as Open3D's own
        per-vertex color output, which survives simplify()'s decimation afterward.

        Trimming Poisson's own low-density "balloon" geometry (below) is necessary
        but reintroduces holes and non-manifold edges of its own -- density-based
        vertex removal punches through an otherwise-closed surface. MeshFix (same
        repair library used for SAM3D's own vendored postprocessing, see
        project-sam3d-vendor-patches memory) closes those back up into a single
        watertight shell; it also discards vertices, so colors are re-attached by
        nearest-neighbor lookup against the pre-MeshFix positions afterward.
        """
        import open3d as o3d
        import pymeshfix
        from scipy.spatial import cKDTree

        pts, face_ids = trimesh.sample.sample_surface(self.mesh, 50000)
        normals = self.mesh.face_normals[face_ids]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd.orient_normals_consistent_tangent_plane(30)

        vertex_colors_01 = self._vertex_colors_01()
        if vertex_colors_01 is not None:
            point_colors = vertex_colors_01[self.mesh.faces[face_ids]].mean(axis=1)
            pcd.colors = o3d.utility.Vector3dVector(point_colors)

        poisson, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        densities = np.asarray(densities)
        poisson.remove_vertices_by_mask(densities < np.quantile(densities, 0.05))
        poisson.remove_degenerate_triangles()
        poisson.remove_duplicated_vertices()

        poisson_vertices = np.asarray(poisson.vertices)
        poisson_colors = np.asarray(poisson.vertex_colors) if poisson.has_vertex_colors() else None

        mf = pymeshfix.MeshFix(poisson_vertices, np.asarray(poisson.triangles))
        mf.repair()

        vertex_colors = None
        if poisson_colors is not None:
            _, nearest = cKDTree(poisson_vertices).query(mf.points)
            vertex_colors = poisson_colors[nearest]

        return Mesh(trimesh.Trimesh(
            vertices=mf.points,
            faces=mf.faces,
            vertex_colors=vertex_colors,
        ))

    def decimate(self, max_faces: int) -> "Mesh":
        """Decimate straight to a face budget, keeping per-vertex colour -- or the
        UV-mapped texture, when there is one.

        The counterpart to simplify(): that one searches for the coarsest mesh whose
        95th-percentile surface error stays within a fraction of the bounding box, which
        is the right question for a hero object seen up close. It is the wrong question
        for an asset that is one of thousands of instances -- there the budget is the
        constraint and the error is whatever it is. It is also far cheaper, since
        simplify() runs a closest_point query against a doubling sequence of candidates
        (13 rounds on a 275k-face mesh, each against 2000 sample points).

        Surface error is a poor proxy for a grass tuft anyway: what reads at a glance is
        the silhouette of the blades, not the accuracy of their surfaces.

        A textured mesh keeps its texture: open3d's quadric decimation carries no UV
        channel of its own, so the surviving UVs are re-attached by nearest-neighbour
        lookup against the pre-decimation positions -- the same transfer repair() does
        for its colours, and sound for the same reason (quadric collapse leaves most
        vertices at or very near a vertex of the input). Baking a texture down to
        per-vertex colour instead would throw away every detail finer than the
        decimated vertex spacing, which on a 4000-vertex riverboat is all of them.

        Returns a copy unchanged when already within budget, or when max_faces <= 0.
        """
        if max_faces <= 0 or len(self.mesh.faces) <= max_faces:
            return Mesh(self.mesh.copy())

        import open3d as o3d

        o3d_mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(self.mesh.vertices),
            triangles=o3d.utility.Vector3iVector(self.mesh.faces),
        )
        textured = self.has_texture
        vertex_colors_01 = None if textured else self._vertex_colors_01()
        if vertex_colors_01 is not None:
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_01)

        dec = o3d_mesh.simplify_quadric_decimation(int(max_faces))
        dec.remove_degenerate_triangles()
        dec.remove_duplicated_triangles()
        dec.remove_duplicated_vertices()

        faces = np.asarray(dec.triangles)
        if len(faces) == 0:
            # Decimation collapsed the mesh entirely (possible on a highly disconnected
            # input like separate blades of grass). Better to ship the original than
            # nothing at all -- the caller has no other asset to fall back to.
            return Mesh(self.mesh.copy())

        vertices = np.asarray(dec.vertices)
        visual = None
        if textured:
            from scipy.spatial import cKDTree

            _, nearest = cKDTree(self.mesh.vertices).query(vertices)
            visual = trimesh.visual.TextureVisuals(
                uv=np.asarray(self.mesh.visual.uv)[nearest],
                material=self.mesh.visual.material,
            )

        # Handed to the constructor rather than assigned afterwards: Trimesh's
        # processing merges duplicate vertices on construction, which renumbers them.
        # A visual attached after the fact would still be indexed by the PRE-merge
        # numbering and scramble the texture; passed in, trimesh remaps it alongside
        # the geometry (and knows not to merge across a UV seam in the first place).
        return Mesh(trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=visual,
            vertex_colors=(
                np.asarray(dec.vertex_colors)
                if visual is None and dec.has_vertex_colors() else None
            ),
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
        vertex_colors_01 = self._vertex_colors_01()
        if vertex_colors_01 is not None:
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors_01)

        target = min_faces
        while target < len(self.mesh.faces):
            dec = o3d_mesh.simplify_quadric_decimation(target)
            dec.remove_degenerate_triangles()
            dec.remove_duplicated_triangles()
            dec.remove_duplicated_vertices()
            candidate = trimesh.Trimesh(
                vertices=np.asarray(dec.vertices),
                faces=np.asarray(dec.triangles),
                vertex_colors=np.asarray(dec.vertex_colors) if dec.has_vertex_colors() else None,
            )
            _, distances, _ = trimesh.proximity.closest_point(candidate, sample_pts)
            if np.percentile(distances, 95) <= error_budget:
                return Mesh(candidate)
            target *= 2

        return Mesh(self.mesh.copy())

    def bake_vertex_colours(self, resolution: int = 1024, min_cell_px: int = 6) -> "Mesh":
        """Return a copy whose per-vertex colours are baked into a UV texture.

        The client lights meshes with URP Lit, which samples a base-colour texture
        and ignores COLOR_0, so a vertex-coloured mesh -- Tree-D Fusion's export,
        SAM3D's out-of-memory fallback -- arrives as flat grey. Nothing in the
        environment unwraps UVs (no xatlas), so this builds a per-face atlas: a
        square grid with one cell per face, the face's three vertices mapped to the
        cell's lower-right triangle (inset one pixel), and the whole cell filled
        with the Gouraud-interpolated vertex colours, extended past the triangle's
        edges by clamping the barycentrics so bilinear filtering and mip levels
        never read a neighbouring face. Vertices are unshared afterwards (three per
        face), so call it after decimation, not before.

        Returns the mesh unchanged (a copy) when it carries no colour at all.
        """
        from PIL import Image as PILImage
        from trimesh.visual import TextureVisuals
        from trimesh.visual.material import PBRMaterial

        colors = self._vertex_colors_01()
        if colors is None or self.has_texture:
            return Mesh(self.mesh.copy())
        faces = np.asarray(self.mesh.faces)
        count = len(faces)
        if count == 0:
            return Mesh(self.mesh.copy())
        grid = int(np.ceil(np.sqrt(count)))
        cell = max(int(resolution) // grid, int(min_cell_px))
        size = grid * cell
        inset = 1.0
        span = max(cell - 2.0 * inset, 1.0)

        # Barycentrics of every pixel centre in a cell relative to the canonical
        # triangle v0=(inset, inset), v1=(cell-inset, inset), v2=(inset, cell-inset)
        # in (column, row) pixel coordinates; clamped and renormalised so pixels
        # outside the triangle take the nearest edge's colour.
        px = np.arange(cell, dtype=np.float32) + 0.5
        l1 = np.clip((px[None, :] - inset) / span, 0.0, 1.0)      # varies with column
        l2 = np.clip((px[:, None] - inset) / span, 0.0, 1.0)      # varies with row
        l1 = np.broadcast_to(l1, (cell, cell)); l2 = np.broadcast_to(l2, (cell, cell))
        l0 = np.clip(1.0 - l1 - l2, 0.0, 1.0)
        total = np.maximum(l0 + l1 + l2, 1e-6)
        bary = np.stack([l0, l1, l2], axis=-1) / total[..., None]    # (cell, cell, 3)

        face_colors = colors[faces].astype(np.float32)               # (F, 3 verts, 3 rgb)
        tiles = np.einsum("rcv,fvk->frck", bary, face_colors)       # (F, cell, cell, 3)
        atlas = np.zeros((grid, cell, grid, cell, 3), dtype=np.float32)
        idx = np.arange(count)
        gy, gx = idx // grid, idx % grid
        atlas[gy, :, gx] = tiles
        image = PILImage.fromarray(
            (np.clip(atlas.reshape(size, size, 3), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8), "RGB"
        )

        # Per-face UVs: image coordinates (x right, y down) -> glTF/trimesh UV with
        # v measured from the bottom.
        x0 = (gx * cell).astype(np.float32); y0 = (gy * cell).astype(np.float32)
        corners = np.stack([
            np.stack([x0 + inset, y0 + inset], axis=-1),
            np.stack([x0 + cell - inset, y0 + inset], axis=-1),
            np.stack([x0 + inset, y0 + cell - inset], axis=-1),
        ], axis=1)                                                   # (F, 3, 2)
        uv = corners.reshape(-1, 2) / float(size)
        uv[:, 1] = 1.0 - uv[:, 1]

        vertices = np.asarray(self.mesh.vertices)[faces].reshape(-1, 3)
        new_faces = np.arange(count * 3).reshape(count, 3)
        material = PBRMaterial(baseColorTexture=image)
        baked = trimesh.Trimesh(
            vertices=vertices, faces=new_faces,
            visual=TextureVisuals(uv=uv, material=material), process=False,
        )
        return Mesh(baked)

    def match_texture_colour(self, reference_rgba: "PILImage.Image", max_gain: float = 2.5, min_gain: float = 0.85) -> "tuple[np.ndarray, np.ndarray] | None":
        """Scale the base-colour texture per channel so its mean over the surface
        (sampled at the vertex UVs, so atlas padding does not count) matches the
        mean of the reference's opaque pixels. Returns (before, after) means, or
        None when there is no texture or nothing to compare.

        SAM3D/TRELLIS often return a texture far darker or more washed out than
        the picture they were given (Rainier 2026-10-16: a tree crop averaging
        (98, 92, 40) came back (46, 46, 22) and rendered black)."""
        if not self.has_texture:
            return None
        from PIL import Image as _PIL
        ref = np.asarray(reference_rgba.convert("RGBA"), dtype=np.float32)
        opaque = ref[..., 3] > 127
        if opaque.sum() < 16:
            return None
        target = ref[..., :3][opaque].mean(axis=0)
        visual = self.mesh.visual
        material = visual.material
        attr = "baseColorTexture" if getattr(material, "baseColorTexture", None) is not None else "image"
        tex = getattr(material, attr)
        mode = tex.mode
        arr = np.asarray(tex.convert("RGBA"), dtype=np.float32)
        h, w = arr.shape[:2]
        uv = np.asarray(visual.uv, dtype=np.float64)
        cols = np.clip((uv[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(int)
        rows = np.clip((1.0 - (uv[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(int)
        current = arr[rows, cols, :3].mean(axis=0)
        gain = np.clip(target / np.maximum(current, 1.0), min_gain, max_gain)
        arr[..., :3] = np.clip(arr[..., :3] * gain, 0, 255)
        out = _PIL.fromarray(arr.astype(np.uint8), "RGBA")
        setattr(material, attr, out if mode == "RGBA" else out.convert(mode))
        return current, np.clip(current * gain, 0, 255)

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

    # The scene is built in a LEFT-handed frame: +X is the right-hand side of the
    # photograph, +Y up, +Z forward (HeightMapGenerator's grid, Panorama.mesh_uvs'
    # azimuth, every placed object's position). glTF is right-handed with +Z
    # forward and -X right, so a file that stores these coordinates verbatim is a
    # mirror image, and a conformant loader (glTFast negates X on import) hands
    # Unity the mirror image: the terrain, the water, the treeline and every
    # instanced mesh arrived flipped left-to-right while scene.json's positions
    # did not. Measured on the Rainier scene, 82% of the grass tufts were then
    # more than 15 cm off the ground they were placed on (16% floating, 66%
    # buried) and the meadow's east-shore grass stood in the mirrored lake.
    #
    # So the file stores the glTF frame: X negated and the winding reversed so the
    # faces still point outward. load() undoes it for files that carry the marker
    # below; a GLB without the marker predates this and is read as before.
    _HANDEDNESS_MARKER = "intoframe_handedness"

    @staticmethod
    def _mirror_x(tm: trimesh.Trimesh) -> trimesh.Trimesh:
        out = tm.copy()
        vertices = np.array(out.vertices, dtype=np.float64)
        vertices[:, 0] *= -1.0
        faces = np.array(out.faces, dtype=np.int64)[:, ::-1]
        # Assigning through the setters drops every cached derived quantity
        # (face/vertex normals included), so normals are rebuilt for the new
        # winding instead of being the stale mirror-image ones.
        out.vertices = vertices
        out.faces = faces
        return out

    def save(self, path):
        self._mirror_x(self.mesh).export(str(path), include_normals=True)
        if str(path).lower().endswith(".glb"):
            from util.gltf_uv2 import _read_glb, _write_glb
            if self.extra_uv is not None:
                from util.gltf_uv2 import inject_texcoord1
                inject_texcoord1(path, self.extra_uv, preview_image=self.preview_image)
            if self.skin_bone_heights is not None:
                from util.gltf_skin import inject_skin
                inject_skin(path, self.skin_bone_heights, self.skin_bone_names)
            tree, bin_data = _read_glb(Path(path).read_bytes())
            tree.setdefault("asset", {}).setdefault("extras", {})[self._HANDEDNESS_MARKER] = "gltf"
            _write_glb(Path(path), tree, bin_data)

    @classmethod
    def _stored_in_gltf_frame(cls, path) -> bool:
        if not str(path).lower().endswith(".glb"):
            return False
        try:
            from util.gltf_uv2 import _read_glb
            tree, _ = _read_glb(Path(path).read_bytes())
        except Exception:
            return False
        return (tree.get("asset", {}).get("extras", {}) or {}).get(cls._HANDEDNESS_MARKER) == "gltf"

    @classmethod
    def load(cls, path) -> Self:
        loaded = trimesh.load(str(path), force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(f"Expected a single Trimesh at {path}, got {type(loaded).__name__}")
        if cls._stored_in_gltf_frame(path):
            loaded = cls._mirror_x(loaded)
        mesh = cls(loaded)

        # trimesh models one UV set and no skinning, so loading through it drops both
        # the TEXCOORD_1 panorama UV and the sway skeleton that save() injected. Left
        # unrecovered, extra_uv/skin_bone_heights come back None and the very next
        # export writes a file missing both -- which is what the asset server does on a
        # cache miss, so every resumed run served boneless meshes (WindSway never finds
        # SwayBone_0 and nothing sways) and UV-less terrain. Read them back out of the
        # file instead; see util.gltf_attachments.
        if str(path).lower().endswith(".glb"):
            from util.gltf_attachments import read_skin, read_texcoord1

            extra_uv = read_texcoord1(path)
            if extra_uv is not None and len(extra_uv) == len(loaded.vertices):
                mesh.extra_uv = extra_uv

            skin = read_skin(path)
            if skin is not None:
                mesh.skin_bone_heights, mesh.skin_bone_names = skin

        return mesh