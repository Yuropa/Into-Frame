import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial import Delaunay
from typing import Optional
import trimesh
import trimesh.visual.material
import PIL.Image

from util.depth_utils import Depth
from util.panorama_utils import Panorama
from scene.camera import CameraIntrinsics
from scene.mesh import Mesh


class TerrainMeshGenerator:
    @staticmethod
    def generate(
        height_map: Depth,
        grid_size_meters: float,
        inner_min_dist: float = 1.5,
        outer_min_dist: float = 6.0,
        n_boundary: int = 12,
        z_far: Optional[float] = None,
        noise_amplitude: float = 0.05,
        noise_seed: int = 42,
        panorama: Optional[Panorama] = None,
        texture: Optional[PIL.Image.Image] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> Mesh:
        """
        Build a variable-density terrain mesh from a height map using Poisson
        disc sampling and Delaunay triangulation.

        A dense Poisson disc pass covers the inner region; a sparser pass covers
        the full domain.  Boundary points anchor the rectangle edges.  All points
        are triangulated with Delaunay, giving a natural, non-axis-biased LOD.

        panorama  : Panorama for equirectangular vertex-colour baking (full 360° coverage).
        texture   : PIL image for pinhole UV mapping via CameraIntrinsics (FOV-limited).
        intrinsics: required when texture is supplied.
        """
        z_far  = z_far if z_far is not None else grid_size_meters / 2.0
        x_half = grid_size_meters / 2.0
        hm     = height_map.depth  # (H, W) float32

        # ── Poisson disc sampling ─────────────────────────────────────────
        all_xz = TerrainMeshGenerator._poisson_disc_xz(
            x_half=x_half,
            z_far=z_far,
            inner_min_dist=inner_min_dist,
            outer_min_dist=outer_min_dist,
            n_boundary=n_boundary,
            seed=noise_seed,
        )
        all_xz = np.unique(np.round(all_xz, 4), axis=0)

        X_pos = all_xz[:, 0].astype(np.float32)
        Z_pos = all_xz[:, 1].astype(np.float32)

        # ── Delaunay triangulation ────────────────────────────────────────
        faces = Delaunay(all_xz).simplices[:, ::-1].astype(np.int32)

        # ── Sample height map at every vertex ─────────────────────────────
        h_hm, w_hm = hm.shape
        row_coords = ((Z_pos + z_far)  / (2.0 * z_far)        * (h_hm - 1)).clip(0, h_hm - 1)
        col_coords = ((X_pos + x_half) / grid_size_meters      * (w_hm - 1)).clip(0, w_hm - 1)

        Y_pos = map_coordinates(
            hm, [row_coords, col_coords], order=1, mode="nearest",
        ).astype(np.float32)
        Y_pos = np.nan_to_num(Y_pos, nan=0.0)

        # ── Noise, blended in with distance from origin ───────────────────
        noise_tex = TerrainMeshGenerator._smooth_noise((256, 256), seed=noise_seed)
        nr = (row_coords / (h_hm - 1) * 255).clip(0, 255)
        nc = (col_coords / (w_hm - 1) * 255).clip(0, 255)
        noise_vals = map_coordinates(noise_tex, [nr, nc], order=1, mode="wrap").astype(np.float32)

        r_end = np.hypot(x_half, z_far)
        blend = (np.hypot(X_pos, Z_pos) / r_end).clip(0.0, 1.0)
        Y_pos += noise_vals * noise_amplitude * blend

        # ── Vertex array ──────────────────────────────────────────────────
        vertices = np.stack([X_pos, Y_pos, Z_pos], axis=-1).astype(np.float32)

        # ── Colour / texture ──────────────────────────────────────────────
        if panorama is not None:
            baked = TerrainMeshGenerator._bake_topdown_texture(panorama, hm, x_half, z_far)
            u = ((X_pos + x_half) / (2.0 * x_half)).clip(0.0, 1.0).astype(np.float32)
            v = ((Z_pos + z_far)  / (2.0 * z_far )).clip(0.0, 1.0).astype(np.float32)
            uv = np.stack([u, v], axis=-1)
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=baked,
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            )
            visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
            tri_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, visual=visual, process=False,
            )
            _ = tri_mesh.vertex_normals
        elif texture is not None and intrinsics is not None:
            uv = TerrainMeshGenerator._uvs_pinhole(
                vertices[:, 0], vertices[:, 1], vertices[:, 2], intrinsics
            )
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture.convert("RGB"),
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            )
            visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
            tri_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, visual=visual, process=False,
            )
            _ = tri_mesh.vertex_normals
        else:
            tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        return Mesh(tri_mesh)

    # ── Point generation ──────────────────────────────────────────────────────

    @staticmethod
    def _poisson_disc_xz(
        x_half: float,
        z_far: float,
        inner_min_dist: float,
        outer_min_dist: float,
        n_boundary: int,
        seed: int = 42,
        k: int = 30,
    ) -> np.ndarray:
        """
        Bridson's Poisson disc sampling with linearly varying radius.
        Spacing grows from inner_min_dist at the origin to outer_min_dist
        at the domain corner, giving a smooth continuous density falloff.
        """
        rng   = np.random.default_rng(seed)
        d_max = np.hypot(x_half, z_far)

        def radius_at(x: float, z: float) -> float:
            t = min(1.0, np.hypot(x, z) / d_max)
            return inner_min_dist + (outer_min_dist - inner_min_dist) * t

        # Background grid sized to the smallest possible radius
        cell = inner_min_dist / np.sqrt(2.0)
        cols = int(np.ceil(2.0 * x_half / cell)) + 2
        rows = int(np.ceil(2.0 * z_far  / cell)) + 2
        grid = np.full((rows, cols), -1, dtype=np.int32)

        pts_x: list[float] = []
        pts_z: list[float] = []
        active: list[int]  = []

        def to_grid(x: float, z: float):
            c = int((x + x_half) / cell)
            r = int((z + z_far)  / cell)
            return np.clip(r, 0, rows - 1), np.clip(c, 0, cols - 1)

        def try_add(x: float, z: float) -> bool:
            r  = radius_at(x, z)
            gr, gc = to_grid(x, z)
            hw = int(np.ceil(r / cell)) + 1
            for dr in range(-hw, hw + 1):
                for dc in range(-hw, hw + 1):
                    nr, nc = gr + dr, gc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] >= 0:
                        i = grid[nr, nc]
                        if np.hypot(x - pts_x[i], z - pts_z[i]) < r:
                            return False
            idx = len(pts_x)
            pts_x.append(x)
            pts_z.append(z)
            active.append(idx)
            grid[gr, gc] = idx
            return True

        try_add(0.0, 0.0)

        while active:
            i      = int(rng.integers(len(active)))
            ax, az = pts_x[active[i]], pts_z[active[i]]
            r      = radius_at(ax, az)
            placed = False
            for _ in range(k):
                angle = rng.uniform(0.0, 2.0 * np.pi)
                dist  = rng.uniform(r, 2.0 * r)
                nx    = ax + dist * np.cos(angle)
                nz    = az + dist * np.sin(angle)
                if -x_half <= nx <= x_half and -z_far <= nz <= z_far:
                    if try_add(nx, nz):
                        placed = True
                        break
            if not placed:
                active.pop(i)

        pts = np.column_stack([pts_x, pts_z]).astype(np.float32)

        n  = n_boundary
        ex = np.linspace(-x_half, x_half, n, dtype=np.float32)
        ez = np.linspace(-z_far,  z_far,  n, dtype=np.float32)
        boundary_pts = np.concatenate([
            np.column_stack([ex,                                np.full(n, -z_far,  dtype=np.float32)]),
            np.column_stack([ex,                                np.full(n,  z_far,  dtype=np.float32)]),
            np.column_stack([np.full(n, -x_half, dtype=np.float32), ez]),
            np.column_stack([np.full(n,  x_half, dtype=np.float32), ez]),
        ]).astype(np.float32)

        return np.concatenate([pts, boundary_pts])

    # ── Colour helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _bake_topdown_texture(
        panorama,
        height_map: np.ndarray,
        x_half: float,
        z_far: float,
        tex_size: int = 1024,
    ) -> PIL.Image.Image:
        """
        Rasterise the panorama into a top-down orthographic texture.

        For each texel, compute its world XZ position, sample the terrain
        height there, then call panorama.sample_3d — which already handles
        near-nadir hole-filling — to get the colour.  The result is a PIL
        image that can be used as a PBR base-colour texture with simple
        orthographic UVs (u = (X+x_half)/(2*x_half), v = (Z+z_far)/(2*z_far)).
        """
        us = np.linspace(0.0, 1.0, tex_size, dtype=np.float32)
        vs = np.linspace(0.0, 1.0, tex_size, dtype=np.float32)
        ug, vg = np.meshgrid(us, vs)

        X = (ug.ravel() - 0.5) * (2.0 * x_half)
        Z = (vg.ravel() - 0.5) * (2.0 * z_far)

        h_hm, w_hm = height_map.shape
        row_coords = ((Z + z_far)  / (2.0 * z_far)  * (h_hm - 1)).clip(0, h_hm - 1)
        col_coords = ((X + x_half) / (2.0 * x_half) * (w_hm - 1)).clip(0, w_hm - 1)
        Y = map_coordinates(height_map, [row_coords, col_coords], order=1, mode="nearest").astype(np.float32)
        Y = np.nan_to_num(Y, nan=0.0)

        grid_verts = np.stack([X, Y, Z], axis=-1)
        rgba = panorama.sample_3d(grid_verts)
        return PIL.Image.fromarray(rgba[:, :3].reshape(tex_size, tex_size, 3), "RGB")

    @staticmethod
    def _uvs_pinhole(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        Z_safe = np.where(Z < 1e-3, 1e-3, Z).astype(np.float64)
        cx = X * intrinsics.fx / Z_safe + intrinsics.px
        cy = intrinsics.py - Y * intrinsics.fy / Z_safe
        u  = (cx / intrinsics.width).clip(0.0, 1.0)
        v  = (cy / intrinsics.height).clip(0.0, 1.0)
        return np.stack([u, v], axis=-1).astype(np.float32)

    # ── Noise helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _smooth_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
        """Layered Gaussian-smoothed noise (4 octaves) normalised to ±1."""
        rng   = np.random.default_rng(seed)
        noise = np.zeros(shape, dtype=np.float32)
        for octave in range(4):
            amplitude = 0.5 ** octave
            raw   = rng.standard_normal(shape).astype(np.float32) * amplitude
            sigma = max(1.0, min(shape) / (4.0 * (2 ** octave)))
            noise += gaussian_filter(raw, sigma=sigma)
        peak = np.abs(noise).max()
        if peak > 0:
            noise /= peak
        return noise
