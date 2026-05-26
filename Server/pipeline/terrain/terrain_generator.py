import numpy as np
from scipy.ndimage import map_coordinates, gaussian_filter
from scipy.spatial import KDTree
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
        # z_far is the half-extent: terrain spans [-z_far, +z_far] in Z.
        z_far = z_far if z_far is not None else grid_size_meters / 2.0
        x_half = grid_size_meters / 2.0
        hm = height_map.depth  # (H, W) float32

        # ── Z axis: log-spaced, symmetric around origin (forward + backward) ─
        n_half = max(1, n_z_vertices // 2)
        z_inner = np.logspace(np.log10(0.01), np.log10(z_far), n_half).astype(np.float32)
        z_positions = np.concatenate([-z_inner[::-1], [0.0], z_inner])

        # ── X axis: log-spaced, symmetric around 0 ──────────────────────────
        x_inner = np.logspace(np.log10(0.01), np.log10(x_half), n_x_half_vertices)
        x_right = np.concatenate([[0.0], x_inner]).astype(np.float32)
        x_left = -x_right[:0:-1]
        x_positions = np.concatenate([x_left, x_right])

        n_z = len(z_positions)
        n_x = len(x_positions)

        # ── Sample height map at every (X, Z) grid point ────────────────────
        # Height map rows run from Z=-z_far (row 0) to Z=+z_far (row h-1).
        h, w = hm.shape
        row_coords = ((z_positions + z_far) / (2.0 * z_far) * (h - 1)).clip(0, h - 1)
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

        # ── Perlin-like noise, blended in with distance from origin ─────────
        noise = TerrainMeshGenerator._smooth_noise((n_z, n_x), seed=noise_seed)
        blend = np.sqrt(np.abs(Z_grid) / z_far).clip(0.0, 1.0)
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

        # ── Colour ───────────────────────────────────────────────────────────
        if texture is not None and uv_mode == "panorama":
            # Bake vertex colours by projecting each 3D point back into the
            # panorama.  Vertices whose viewing angle is above the horizon or
            # too steeply downward (near-nadir region; poorly generated) are
            # treated as holes and filled from the nearest valid neighbour.
            vertex_colors = TerrainMeshGenerator._bake_panorama_colors(
                vertices, texture
            )
            tri_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces,
                vertex_colors=vertex_colors, process=False
            )
            _ = tri_mesh.vertex_normals
        elif texture is not None and uv_mode == "pinhole":
            # Pinhole: original image covers a limited but well-defined FOV,
            # so UV mapping is correct for every in-frame vertex.
            uv = TerrainMeshGenerator._uvs_pinhole(
                vertices[:, 0], vertices[:, 1], vertices[:, 2], intrinsics
            )
            material = trimesh.visual.material.PBRMaterial(
                baseColorTexture=texture.convert("RGB"),
                baseColorFactor=[1.0, 1.0, 1.0, 1.0],
            )
            visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
            tri_mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, visual=visual, process=False
            )
            _ = tri_mesh.vertex_normals
        else:
            tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)

        return Mesh(tri_mesh)

    # ── Colour helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _bake_panorama_colors(
        vertices: np.ndarray,
        panorama: PIL.Image.Image,
        min_lat_deg: float = -35.0,
    ) -> np.ndarray:
        """
        Sample a colour for every terrain vertex from the equirectangular panorama.

        For each vertex the direction from the camera origin to that point is computed
        and mapped to a panorama pixel.  Two classes of vertex are treated as "holes"
        and filled by interpolation from the nearest valid neighbour (in XZ):

          • above the horizon  (lat >= 0)   — would sample sky
          • steeper than min_lat_deg below  — near-nadir region, poorly generated

        Bilinear sampling is used; the panorama wraps horizontally.

        min_lat_deg: most-negative latitude (degrees) still considered valid.
                     -35° means ≈3 m horizontal distance for a 2 m camera height.
        """
        X = vertices[:, 0].astype(np.float64)
        Y = vertices[:, 1].astype(np.float64)
        Z = vertices[:, 2].astype(np.float64)

        r_xz = np.sqrt(X ** 2 + Z ** 2).clip(1e-6)
        lat = np.arctan2(Y, r_xz)            # negative = below horizon
        lon = np.arctan2(X, Z)               # 0 = forward (+Z)

        min_lat_rad = np.radians(min_lat_deg)
        valid = (lat < 0.0) & (lat >= min_lat_rad)

        pano = np.array(panorama.convert("RGB"), dtype=np.float32)
        H, W = pano.shape[:2]

        # Fractional pixel coordinates
        pu = ((lon + np.pi) / (2.0 * np.pi)) * (W - 1)   # [0, W-1], wraps
        pv = (0.5 - lat / np.pi) * (H - 1)               # [0, H-1], clamp

        # Integer corners with horizontal wrap and vertical clamp
        pu0 = np.floor(pu).astype(np.int32) % W
        pu1 = (pu0 + 1) % W
        pv0 = np.clip(np.floor(pv).astype(np.int32), 0, H - 1)
        pv1 = np.clip(pv0 + 1, 0, H - 1)

        fu = (pu - np.floor(pu))[:, None]   # (N, 1)
        fv = (pv - np.floor(pv))[:, None]

        c00 = pano[pv0, pu0]
        c10 = pano[pv0, pu1]
        c01 = pano[pv1, pu0]
        c11 = pano[pv1, pu1]

        colors_f = (c00 * (1 - fu) * (1 - fv) +
                    c10 * fu       * (1 - fv) +
                    c01 * (1 - fu) * fv       +
                    c11 * fu       * fv)

        colors = np.zeros((len(vertices), 4), dtype=np.uint8)
        colors[:, 3] = 255  # alpha
        colors[valid, :3] = np.clip(colors_f[valid], 0, 255).astype(np.uint8)

        # Fill holes: nearest valid neighbour in XZ plane
        invalid = ~valid
        if invalid.any() and valid.any():
            valid_xz = np.stack([X[valid], Z[valid]], axis=-1)
            invalid_xz = np.stack([X[invalid], Z[invalid]], axis=-1)
            _, nn = KDTree(valid_xz).query(invalid_xz)
            colors[invalid, :3] = colors[valid][nn, :3]

        return colors

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
