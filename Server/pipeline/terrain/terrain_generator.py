import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial import Delaunay
from scipy.stats.qmc import PoissonDisk
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

        # Inner region half-extent: 20 % of the shorter half-dimension
        inner_half = min(x_half, z_far) * 0.20

        # ── Poisson disc sampling ─────────────────────────────────────────
        all_xz = TerrainMeshGenerator._poisson_disc_xz(
            x_half=x_half,
            z_far=z_far,
            inner_half=inner_half,
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
            vertex_colors = panorama.sample_3d(vertices)
            tri_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces,
                vertex_colors=vertex_colors, process=False,
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
        inner_half: float,
        inner_min_dist: float,
        outer_min_dist: float,
        n_boundary: int,
        seed: int = 42,
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        def sample(w: float, h: float, min_dist: float) -> np.ndarray:
            max_dim = max(w, h)
            r_norm  = min_dist / max_dim
            n_max   = max(64, int(4.0 / (np.pi * r_norm ** 2)))
            engine  = PoissonDisk(d=2, radius=r_norm, seed=int(rng.integers(2**31)))
            pts     = engine.random(n_max)
            return (pts * [w, h]).astype(np.float32)

        inner_w   = 2.0 * inner_half
        inner_pts = sample(inner_w, inner_w, inner_min_dist) - inner_half

        domain_w, domain_d = 2.0 * x_half, 2.0 * z_far
        outer_pts = sample(domain_w, domain_d, outer_min_dist) - [x_half, z_far]

        n  = n_boundary
        ex = np.linspace(-x_half, x_half, n, dtype=np.float32)
        ez = np.linspace(-z_far,  z_far,  n, dtype=np.float32)
        boundary_pts = np.concatenate([
            np.column_stack([ex,                           np.full(n, -z_far,  dtype=np.float32)]),
            np.column_stack([ex,                           np.full(n,  z_far,  dtype=np.float32)]),
            np.column_stack([np.full(n, -x_half, dtype=np.float32), ez]),
            np.column_stack([np.full(n,  x_half, dtype=np.float32), ez]),
        ]).astype(np.float32)

        return np.concatenate([inner_pts, outer_pts, boundary_pts])

    # ── Colour helpers ────────────────────────────────────────────────────────

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
