import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Optional, Literal
import trimesh
import trimesh.visual.material
import PIL.Image

from util.depth_utils import Depth
from scene.camera import CameraIntrinsics
from scene.mesh import Mesh


UVMode = Literal["panorama", "pinhole"]


class TerrainMeshGenerator:
    @staticmethod
    def generate(
        height_map: Depth,
        grid_size_meters: float,
        n_z_vertices: int = 150,
        n_x_half_vertices: int = 50,
        z_far: Optional[float] = None,
        noise_amplitude: float = 0.05,
        noise_seed: int = 42,
        texture: Optional[PIL.Image.Image] = None,
        uv_mode: UVMode = "panorama",
        intrinsics: Optional[CameraIntrinsics] = None,
    ) -> Mesh:
        """
        Build a variable-density terrain mesh from a height map.

        Vertex density is highest near the origin (camera position) in both X and Z
        and decreases logarithmically toward the edges.  Smooth multi-octave noise is
        added to give the terrain a natural look; it is blended in gradually with
        distance so the ground at the viewer's feet matches the raw height-map data.

        When a texture image is supplied, UV coordinates are computed and embedded in
        the exported GLB so the mesh can be textured by downstream consumers.

        texture  : PIL image to embed as the base-colour texture (panorama or original).
        uv_mode  : 'panorama' — equirectangular spherical projection (use with a 360°
                               equirectangular panorama; coverage is complete).
                   'pinhole'  — standard pinhole projection via CameraIntrinsics (use
                               with the original image; vertices outside the FOV are
                               clamped to the image border).
        intrinsics: required when uv_mode='pinhole'.
        """
        z_far = z_far if z_far is not None else grid_size_meters
        x_half = grid_size_meters / 2.0
        hm = height_map.depth  # (H, W) float32

        # ── Z axis: log-spaced rows, dense near camera ──────────────────────
        z_inner = np.logspace(np.log10(0.01), np.log10(z_far), n_z_vertices - 1)
        z_positions = np.concatenate([[0.0], z_inner]).astype(np.float32)

        # ── X axis: log-spaced, symmetric around 0 ──────────────────────────
        x_inner = np.logspace(np.log10(0.01), np.log10(x_half), n_x_half_vertices)
        x_right = np.concatenate([[0.0], x_inner]).astype(np.float32)
        x_left = -x_right[:0:-1]
        x_positions = np.concatenate([x_left, x_right])

        n_z = len(z_positions)
        n_x = len(x_positions)

        # ── Sample height map at every (X, Z) grid point ────────────────────
        h, w = hm.shape
        row_coords = (z_positions / grid_size_meters * (h - 1)).clip(0, h - 1)
        col_coords = ((x_positions + x_half) / grid_size_meters * (w - 1)).clip(0, w - 1)

        Z_grid = z_positions[:, None] * np.ones((1, n_x), dtype=np.float32)
        X_grid = np.ones((n_z, 1), dtype=np.float32) * x_positions[None, :]
        R_grid = row_coords[:, None] * np.ones((1, n_x), dtype=np.float32)
        C_grid = np.ones((n_z, 1), dtype=np.float32) * col_coords[None, :]

        Y_grid = map_coordinates(
            hm,
            [R_grid.ravel(), C_grid.ravel()],
            order=1,
            mode="nearest",
        ).reshape(n_z, n_x).astype(np.float32)
        Y_grid = np.nan_to_num(Y_grid, nan=0.0)

        # ── Perlin-like noise, blended in with distance ──────────────────────
        noise = TerrainMeshGenerator._smooth_noise((n_z, n_x), seed=noise_seed)
        blend = np.sqrt(Z_grid / z_far).clip(0.0, 1.0)
        Y_grid += noise * noise_amplitude * blend

        # ── Vertex array ─────────────────────────────────────────────────────
        vertices = np.stack(
            [X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], axis=-1
        ).astype(np.float32)

        # ── Face array: 2 triangles per quad ─────────────────────────────────
        iz = np.arange(n_z - 1, dtype=np.int32)
        ix = np.arange(n_x - 1, dtype=np.int32)
        IZ, IX = np.meshgrid(iz, ix, indexing="ij")

        v00 = (IZ * n_x + IX).ravel()
        v10 = ((IZ + 1) * n_x + IX).ravel()
        v01 = (IZ * n_x + (IX + 1)).ravel()
        v11 = ((IZ + 1) * n_x + (IX + 1)).ravel()

        faces = np.concatenate([
            np.stack([v00, v10, v01], axis=-1),
            np.stack([v10, v11, v01], axis=-1),
        ], axis=0).astype(np.int32)

        # ── UV coordinates + texture ──────────────────────────────────────────
        if texture is not None:
            uv = TerrainMeshGenerator._compute_uvs(
                vertices, uv_mode, intrinsics, texture
            )
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture.convert("RGB"),
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            )
            visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
            tri_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, visual=visual, process=False
            )
            # Trigger lazy vertex-normal computation so the GLB includes them
            _ = tri_mesh.vertex_normals
        else:
            tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        return Mesh(tri_mesh)

    # ── UV helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_uvs(
        vertices: np.ndarray,
        mode: UVMode,
        intrinsics: Optional[CameraIntrinsics],
        texture: PIL.Image.Image,
    ) -> np.ndarray:
        X = vertices[:, 0]
        Y = vertices[:, 1]
        Z = vertices[:, 2]

        if mode == "panorama":
            return TerrainMeshGenerator._uvs_panorama(X, Y, Z)
        else:
            if intrinsics is None:
                raise ValueError("intrinsics required for uv_mode='pinhole'")
            return TerrainMeshGenerator._uvs_pinhole(X, Y, Z, intrinsics)

    @staticmethod
    def _uvs_panorama(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> np.ndarray:
        """
        Equirectangular projection (X right, Y up, Z forward convention).

        Assumes the panorama is centred on the forward (+Z) direction so that:
          u=0.5  → straight ahead, u=0/1 → directly behind
          v=0    → zenith (+Y), v=0.5 → horizon, v=1 → nadir (-Y)
        """
        lon = np.arctan2(X, Z)                                    # (−π, π]
        r_xz = np.sqrt(X ** 2 + Z ** 2).clip(1e-6)
        lat = np.arctan2(Y, r_xz)                                 # [−π/2, π/2]

        u = (lon + np.pi) / (2.0 * np.pi)
        v = 0.5 - lat / np.pi

        return np.stack([u, v], axis=-1).astype(np.float32)

    @staticmethod
    def _uvs_pinhole(
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
        intrinsics: CameraIntrinsics,
    ) -> np.ndarray:
        """
        Standard pinhole back-projection using camera intrinsics.

        Inverts CameraIntrinsics.unproject (which flips Y for Unity), so:
          cx = X * fx / Z + px
          cy = py − Y * fy / Z     (minus because Y is flipped in camera.py)

        Vertices that project outside the image are clamped to the border.
        """
        # Guard against Z=0 at the camera position
        Z_safe = np.where(Z < 1e-3, 1e-3, Z).astype(np.float64)

        cx = X * intrinsics.fx / Z_safe + intrinsics.px
        cy = intrinsics.py - Y * intrinsics.fy / Z_safe

        u = (cx / intrinsics.width).clip(0.0, 1.0)
        v = (cy / intrinsics.height).clip(0.0, 1.0)

        return np.stack([u, v], axis=-1).astype(np.float32)

    # ── Noise helper ──────────────────────────────────────────────────────────

    @staticmethod
    def _smooth_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
        """Layered Gaussian-smoothed noise (4 octaves) normalised to ±1."""
        rng = np.random.default_rng(seed)
        noise = np.zeros(shape, dtype=np.float32)
        for octave in range(4):
            amplitude = 0.5 ** octave
            raw = rng.standard_normal(shape).astype(np.float32) * amplitude
            sigma = max(1.0, min(shape) / (4.0 * (2 ** octave)))
            noise += gaussian_filter(raw, sigma=sigma)
        peak = np.abs(noise).max()
        if peak > 0:
            noise /= peak
        return noise
