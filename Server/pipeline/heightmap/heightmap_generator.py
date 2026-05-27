import numpy as np
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
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
        smooth_sigma: float = 0.0,
    ) -> np.ndarray:
        """
        Project ground points from a depth map onto a top-down height grid.

        The output array is (grid_resolution, grid_resolution) float32 where:
          - rows index Z (near → far), columns index X (left → right)
          - values are Y in camera space (negative = below camera)
          - missing cells are filled by interpolation from neighbours

        ground_y_max: Y threshold in camera space; points with Y <= this value are
                      treated as ground (e.g. -0.5 means at least 0.5 m below camera).
        grid_size_meters: side length of the square grid; both X and Z span ±half (centred at origin).
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

        half = grid_size_meters / 2.0

        ground_mask = (
            (Y <= ground_y_max)
            & (np.abs(Z) <= half)
            & (np.abs(X) <= half)
            & np.isfinite(d)
        )

        Xg = X[ground_mask]
        Yg = Y[ground_mask]
        Zg = Z[ground_mask]

        if len(Xg) == 0:
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)

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

        result = HeightMapGenerator._interpolate(height_map)
        if smooth_sigma > 0:
            result = HeightMapGenerator._smooth_edge_preserving(result, max_sigma=smooth_sigma)
        return result

    @staticmethod
    def _smooth_edge_preserving(
        height_map: np.ndarray,
        max_sigma: float,
        edge_sensitivity: float = 1.0,
        n_levels: int = 5,
    ) -> np.ndarray:
        """
        Spatially-varying, edge-preserving Gaussian smooth.

        Each pixel's smoothing amount is driven by two weights multiplied together:

          distance weight  — 0 at grid centre (camera position, most reliable),
                             1 at corners (least reliable).  Same as before.

          flatness weight  — 1 where the local gradient is small (likely noise,
                             smooth freely), 0 where the gradient is large (real
                             structure, preserve it).

        The gradient threshold is set adaptively at the 85th percentile of all
        gradient magnitudes so it scales with the scene.  edge_sensitivity > 1
        preserves more structures; < 1 smooths more aggressively.

        Implementation: pre-compute n_levels+1 Gaussian blurs, then for each
        pixel pick between them according to the combined weight.
        """
        h, w = height_map.shape

        # ── Distance weight ───────────────────────────────────────────────
        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        dist = np.clip(np.sqrt(y ** 2 + x ** 2) / np.sqrt(2.0), 0.0, 1.0)

        # ── Flatness weight ───────────────────────────────────────────────
        gy, gx = np.gradient(height_map)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

        # Scale threshold by the 85th-percentile gradient so it adapts to the scene.
        # tanh gives a smooth 0→1 ramp: near zero for flat areas, near 1 for edges.
        scale = np.percentile(grad_mag, 85) / edge_sensitivity + 1e-6
        edge_weight = np.tanh(grad_mag / scale)

        # Blur the edge mask slightly to avoid halos at sharp transitions.
        edge_weight = gaussian_filter(edge_weight, sigma=2.0)
        flat_weight = (1.0 - edge_weight).clip(0.0, 1.0)

        # ── Combined smoothing amount ─────────────────────────────────────
        smooth_weight = (dist * flat_weight).astype(np.float32)

        # ── Scale-space blend ─────────────────────────────────────────────
        sigmas = np.linspace(0.0, max_sigma, n_levels + 1)
        levels = np.stack(
            [height_map if s == 0 else gaussian_filter(height_map, sigma=s)
             for s in sigmas],
            axis=0,
        )  # (n_levels+1, H, W)

        idx  = smooth_weight * n_levels
        lo   = np.floor(idx).astype(np.int32).clip(0, n_levels - 1)
        hi   = (lo + 1).clip(0, n_levels)
        frac = (idx - lo).astype(np.float32)

        rows, cols = np.mgrid[0:h, 0:w]
        return (levels[lo, rows, cols] * (1.0 - frac) +
                levels[hi, rows, cols] * frac).astype(np.float32)

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
