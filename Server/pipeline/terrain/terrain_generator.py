import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from typing import Optional
import trimesh

from util.depth_utils import Depth
from scene.mesh import Mesh


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
    ) -> Mesh:
        """
        Build a variable-density terrain mesh from a height map.

        Vertex density is highest near the origin (camera position) in both X and Z
        and decreases logarithmically toward the edges.  Smooth multi-octave noise is
        added to give the terrain a natural look; it is blended in gradually with
        distance so the ground near the viewer matches the raw height map data.

        grid_size_meters : side length of the square covered by the height map (metres)
        n_z_vertices     : total row count along Z (log-spaced near → far)
        n_x_half_vertices: columns on each side of X=0 (log-spaced, symmetric)
        z_far            : far clip in metres; defaults to grid_size_meters
        noise_amplitude  : peak height displacement from noise (metres)
        noise_seed       : RNG seed for reproducibility
        """
        z_far = z_far if z_far is not None else grid_size_meters
        x_half = grid_size_meters / 2.0
        hm = height_map.depth  # (H, W) float32

        # ── Z axis: log-spaced rows, dense near camera ──────────────────────
        # logspace can't start at 0, so we generate n_z-1 points from 0.01 m
        # to z_far and prepend exactly 0 so the mesh starts at the camera.
        z_inner = np.logspace(np.log10(0.01), np.log10(z_far), n_z_vertices - 1)
        z_positions = np.concatenate([[0.0], z_inner]).astype(np.float32)

        # ── X axis: log-spaced, symmetric around 0 ──────────────────────────
        x_inner = np.logspace(np.log10(0.01), np.log10(x_half), n_x_half_vertices)
        x_right = np.concatenate([[0.0], x_inner]).astype(np.float32)
        x_left = -x_right[:0:-1]  # mirror, exclude the duplicated 0
        x_positions = np.concatenate([x_left, x_right])

        n_z = len(z_positions)
        n_x = len(x_positions)

        # ── Sample height map at every (X, Z) grid point ────────────────────
        h, w = hm.shape
        row_coords = (z_positions / grid_size_meters * (h - 1)).clip(0, h - 1)
        col_coords = ((x_positions + x_half) / grid_size_meters * (w - 1)).clip(0, w - 1)

        # Broadcast to (n_z, n_x) grids
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

        # Replace any residual NaN (shouldn't happen after interpolation, but be safe)
        Y_grid = np.nan_to_num(Y_grid, nan=0.0)

        # ── Perlin-like noise, blended in with distance ──────────────────────
        noise = TerrainMeshGenerator._smooth_noise((n_z, n_x), seed=noise_seed)
        # Scale: zero at camera (Z=0), full amplitude at z_far
        blend = np.sqrt(Z_grid / z_far).clip(0, 1)
        Y_grid += noise * noise_amplitude * blend

        # ── Build vertex array ───────────────────────────────────────────────
        vertices = np.stack(
            [X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()], axis=-1
        ).astype(np.float32)

        # ── Build face array: 2 triangles per quad ───────────────────────────
        # Vertex index: (iz, ix) → iz * n_x + ix
        iz = np.arange(n_z - 1, dtype=np.int32)
        ix = np.arange(n_x - 1, dtype=np.int32)
        IZ, IX = np.meshgrid(iz, ix, indexing="ij")  # (n_z-1, n_x-1)

        v00 = (IZ * n_x + IX).ravel()
        v10 = ((IZ + 1) * n_x + IX).ravel()
        v01 = (IZ * n_x + (IX + 1)).ravel()
        v11 = ((IZ + 1) * n_x + (IX + 1)).ravel()

        tri_a = np.stack([v00, v10, v01], axis=-1)  # lower-left triangle
        tri_b = np.stack([v10, v11, v01], axis=-1)  # upper-right triangle
        faces = np.concatenate([tri_a, tri_b], axis=0).astype(np.int32)

        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        return Mesh(tri_mesh)

    @staticmethod
    def _smooth_noise(shape: tuple[int, int], seed: int) -> np.ndarray:
        """Layered Gaussian-smoothed noise (4 octaves) normalised to ±1."""
        rng = np.random.default_rng(seed)
        noise = np.zeros(shape, dtype=np.float32)
        for octave in range(4):
            amplitude = 0.5 ** octave
            raw = rng.standard_normal(shape).astype(np.float32) * amplitude
            # Sigma decreases each octave so higher frequencies are less smooth
            sigma = max(1.0, min(shape) / (4.0 * (2 ** octave)))
            noise += gaussian_filter(raw, sigma=sigma)
        # Normalise to roughly ±1 so noise_amplitude acts as a direct metre value
        peak = np.abs(noise).max()
        if peak > 0:
            noise /= peak
        return noise
