import numpy as np
from scipy.interpolate import griddata
from util.depth_utils import Depth
from scene.camera import CameraIntrinsics
from typing import Optional


class HeightMapGenerator:
    @staticmethod
    def generate(
        depth: Depth,
        intrinsics: Optional[CameraIntrinsics],
        grid_size_meters: float = 100.0,
        grid_resolution: int = 512,
        ground_y_max: float = -0.5,
        use_equirectangular: bool = False,
    ) -> np.ndarray:
        """
        Project ground points from a depth map onto a top-down height grid.

        The output array is (grid_resolution, grid_resolution) float32 where:
          - rows index Z (near → far), columns index X (left → right)
          - values are Y in camera space (negative = below camera)
          - missing cells are filled by interpolation from neighbours

        ground_y_max: Y threshold in camera space; points with Y <= this value are
                      treated as ground (e.g. -0.5 means at least 0.5 m below camera).
        grid_size_meters: side length of the square grid; X spans ±half, Z spans [0, full].
        use_equirectangular: when True, treat depth as an equirectangular (360°) map
                             where each pixel encodes a radial (Euclidean) distance.
                             When False, treat depth as a rectilinear (pinhole) Z-depth
                             map and use intrinsics to unproject.
        """
        d = depth.depth  # (H, W) float32, metric depth in metres
        h, w = d.shape

        cx = np.arange(w, dtype=np.float32)
        cy = np.arange(h, dtype=np.float32)
        cx, cy = np.meshgrid(cx, cy)  # both (H, W)

        if use_equirectangular:
            # Equirectangular (spherical) unprojection.
            # Each pixel maps to a ray direction; depth is radial (Euclidean) distance.
            # theta: longitude in [-π, π],  0 = forward (+Z)
            # phi:   latitude  in [-π/2, π/2], positive = above horizon
            theta = (cx / w - 0.5) * 2.0 * np.pi
            phi   = (0.5 - cy / h) * np.pi
            cos_phi = np.cos(phi)
            X =  d * cos_phi * np.sin(theta)
            Y =  d * np.sin(phi)              # Unity Y-up: positive above camera
            Z =  d * cos_phi * np.cos(theta)  # positive = forward
        else:
            # Rectilinear (pinhole) unprojection — matches CameraIntrinsics.unproject
            X = (cx - intrinsics.px) * d / intrinsics.fx
            Y = -((cy - intrinsics.py) * d / intrinsics.fy)  # flipped Y (Unity convention)
            Z = d

        ground_mask = (
            (Y <= ground_y_max)
            & (Z > 0.0)
            & np.isfinite(d)
        )

        Xg = X[ground_mask]
        Yg = Y[ground_mask]
        Zg = Z[ground_mask]

        if len(Xg) == 0:
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        half = grid_size_meters / 2.0
        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(0.0, grid_size_meters, grid_resolution + 1)

        xi = np.digitize(Xg, x_edges) - 1  # column index
        zi = np.digitize(Zg, z_edges) - 1  # row index

        in_bounds = (
            (xi >= 0) & (xi < grid_resolution)
            & (zi >= 0) & (zi < grid_resolution)
        )
        xi, zi, Yg = xi[in_bounds], zi[in_bounds], Yg[in_bounds]

        height_sum = np.zeros((grid_resolution, grid_resolution), dtype=np.float64)
        height_cnt = np.zeros((grid_resolution, grid_resolution), dtype=np.int32)
        np.add.at(height_sum, (zi, xi), Yg)
        np.add.at(height_cnt, (zi, xi), 1)

        height_map = np.full((grid_resolution, grid_resolution), np.nan, dtype=np.float32)
        has_data = height_cnt > 0
        height_map[has_data] = (height_sum[has_data] / height_cnt[has_data]).astype(np.float32)

        return HeightMapGenerator._interpolate(height_map)

    @staticmethod
    def _interpolate(height_map: np.ndarray) -> np.ndarray:
        valid = ~np.isnan(height_map)
        if not np.any(valid):
            return np.zeros_like(height_map)
        if np.all(valid):
            return height_map

        h, w = height_map.shape
        src_ys, src_xs = np.where(valid)
        values = height_map[valid]

        all_ys, all_xs = np.mgrid[0:h, 0:w]
        query = np.column_stack([all_ys.ravel(), all_xs.ravel()])

        result = griddata(
            points=np.column_stack([src_ys, src_xs]),
            values=values,
            xi=query,
            method="linear",
        ).reshape(h, w).astype(np.float32)

        # Nearest-neighbour fallback for cells outside the convex hull of known points
        still_nan = np.isnan(result)
        if np.any(still_nan):
            nearest = griddata(
                points=np.column_stack([src_ys, src_xs]),
                values=values,
                xi=query,
                method="nearest",
            ).reshape(h, w).astype(np.float32)
            result = np.where(still_nan, nearest, result)

        return result
