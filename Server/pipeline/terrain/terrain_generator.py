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
        inner_grid_n: int = 30,
        n_rings: int = 12,
        ring_base_points: int = 64,
        z_far: Optional[float] = None,
        noise_amplitude: float = 0.05,
        noise_seed: int = 42,
        panorama: Optional[Panorama] = None,
        texture: Optional[PIL.Image.Image] = None,
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> Mesh:
        """
        Build a variable-density terrain mesh from a height map.

        A dense Cartesian grid covers the centre, surrounded by concentric rings
        whose point count decreases logarithmically with radius.  A sparse
        rectangle of boundary points anchors the domain edges.  All points are
        triangulated with Delaunay, giving a smooth LOD falloff with no axis-bias.

        panorama  : Panorama for equirectangular vertex-colour baking (full 360° coverage).
        texture   : PIL image for pinhole UV mapping via CameraIntrinsics (FOV-limited).
        intrinsics: required when texture is supplied.
        """
        z_far  = z_far if z_far is not None else grid_size_meters / 2.0
        x_half = grid_size_meters / 2.0
        hm     = height_map.depth  # (H, W) float32

        # Inner grid half-extent: 20 % of the shorter half-dimension
        inner_half = min(x_half, z_far) * 0.20

        # ── Dense inner Cartesian grid ────────────────────────────────────
        xi = np.linspace(-inner_half, inner_half, inner_grid_n + 1, dtype=np.float32)
        XX, ZZ = np.meshgrid(xi, xi)
        inner_pts = np.stack([XX.ravel(), ZZ.ravel()], axis=-1)

        # ── Concentric rings ──────────────────────────────────────────────
        # Start just outside the inner grid's corners, end at the domain corner.
        r_start = inner_half * np.sqrt(2.0) * 1.05
        r_end   = np.hypot(x_half, z_far)
        ring_radii = np.logspace(np.log10(r_start), np.log10(r_end), n_rings)

        min_ring_pts = 8
        ring_pts_list = []
        for i, r in enumerate(ring_radii):
            t     = i / max(1, n_rings - 1)           # 0 → 1 across rings
            log_n = (1.0 - t) * np.log2(ring_base_points) + t * np.log2(min_ring_pts)
            n_pts = max(min_ring_pts, int(2 ** log_n))
            angles = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False, dtype=np.float32)
            pts = np.stack([r * np.sin(angles), r * np.cos(angles)], axis=-1)
            pts[:, 0] = pts[:, 0].clip(-x_half, x_half)
            pts[:, 1] = pts[:, 1].clip(-z_far,  z_far)
            ring_pts_list.append(pts)

        # ── Sparse boundary rectangle ─────────────────────────────────────
        # Ensures Delaunay hull fills the full rectangular domain.
        n_edge = max(4, n_rings // 2)
        ex = np.linspace(-x_half, x_half, n_edge, dtype=np.float32)
        ez = np.linspace(-z_far,  z_far,  n_edge, dtype=np.float32)
        boundary_pts = np.concatenate([
            np.stack([ex, np.full(n_edge, -z_far, dtype=np.float32)], axis=-1),
            np.stack([ex, np.full(n_edge,  z_far, dtype=np.float32)], axis=-1),
            np.stack([np.full(n_edge, -x_half, dtype=np.float32), ez], axis=-1),
            np.stack([np.full(n_edge,  x_half, dtype=np.float32), ez], axis=-1),
        ])

        # ── Combine and deduplicate ───────────────────────────────────────
        all_xz = np.concatenate([inner_pts] + ring_pts_list + [boundary_pts], axis=0)
        all_xz = np.unique(np.round(all_xz, 4), axis=0)

        X_pos = all_xz[:, 0].astype(np.float32)
        Z_pos = all_xz[:, 1].astype(np.float32)

        # ── Delaunay triangulation ────────────────────────────────────────
        faces = Delaunay(all_xz).simplices.astype(np.int32)

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
