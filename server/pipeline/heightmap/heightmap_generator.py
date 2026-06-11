import numpy as np
from collections import deque
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from typing import Optional
from util.depth_utils import Depth
from util.panorama_utils import Panorama
from scene.camera import CameraIntrinsics


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
        camera_height_meters: float = 1.0,
        sky_mask: Optional[np.ndarray] = None,
        flood_fill: bool = True,
        flood_fill_max_step: float = 1.5,
        panorama_depth: Optional[Depth] = None,
    ) -> np.ndarray:
        """
        Project ground points from a depth map onto a top-down height grid.

        The output array is (grid_resolution, grid_resolution) float32 where:
          - rows index Z (near → far), columns index X (left → right)
          - values are Y in camera space (negative = below camera)
          - missing cells are filled by interpolation from neighbours

        camera_height_meters: assumed height of the camera above the ground plane
                              (used to derive the Y floor filter and flood-fill seed).
        ground_y_max: upper Y bound in camera space; points with Y <= this are ground
                      candidates (e.g. -0.5 = at least 0.5 m below camera).
        sky_mask: optional bool (H, W) array from the depth model where True = sky.
                  Sky pixels are excluded before projection, preventing the horizon
                  artefacts that come from sky pixels being assigned far depth values.
        flood_fill: if True, BFS from the grid centre outward to keep only connected
                    ground; stops at height discontinuities and empty cells (sky gaps).
        flood_fill_max_step: maximum Y change (metres) between adjacent cells allowed
                             during flood-fill; larger steps are treated as walls.
        grid_size_meters: side length of the square grid; both X and Z span ±half.
        use_equirectangular: treat depth as equirectangular (radial distances); otherwise
                             use pinhole unprojection via intrinsics.
        panorama_depth: optional 360° equirectangular depth map (radial metres). After
                        the primary depth is projected and flood-filled, any still-empty
                        grid cells are filled from this source before interpolation.
        """
        d = depth.depth.astype(np.float32)

        # Mask sky pixels before projection. Sky pixels assigned far depth (e.g. 100 m)
        # project near the horizon at Y ≈ -camera_height, polluting the height grid.
        if sky_mask is not None and sky_mask.shape == d.shape:
            d = d.copy()
            d[sky_mask] = np.nan

        h, w = d.shape

        if use_equirectangular:
            X, Y, Z = Panorama.equirectangular_unproject(Depth(d))
        else:
            cx = np.arange(w, dtype=np.float32)
            cy = np.arange(h, dtype=np.float32)
            cx, cy = np.meshgrid(cx, cy)
            X = (cx - intrinsics.px) * d / intrinsics.fx
            Y = -((cy - intrinsics.py) * d / intrinsics.fy)
            Z = d

        half = grid_size_meters / 2.0

        # Floor filter: sky pixels assigned ~100 m depth project to Y << -camera_height.
        # Allowing ±5 m of terrain variation around the expected ground plane catches
        # hills and slopes while excluding any sky artefacts that slipped through.
        ground_y_min = -(camera_height_meters + 5.0)

        ground_mask = (
            (Y <= ground_y_max)
            & (Y >= ground_y_min)
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

        xi = np.digitize(Xg, x_edges) - 1
        zi = np.digitize(Zg, z_edges) - 1

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

        # Flood-fill from the grid centre (camera XZ = 0,0) outward. Cells connected to
        # the starting point with small height steps are kept; everything else is set to
        # NaN and filled by nearest-neighbour extrapolation from the flood-fill boundary.
        if flood_fill:
            accepted = HeightMapGenerator._flood_fill_ground(
                height_map, camera_height_meters, flood_fill_max_step
            )
            height_map[~accepted] = np.nan

        # Fill remaining NaN cells from the panorama depth (360° coverage) before
        # falling back to pure interpolation. Only cells that are still empty after
        # the primary projection are touched, so rectilinear data always wins.
        if panorama_depth is not None:
            height_map = HeightMapGenerator._fill_from_panorama_depth(
                height_map=height_map,
                panorama_depth=panorama_depth,
                sky_mask=sky_mask,
                grid_size_meters=grid_size_meters,
                grid_resolution=grid_resolution,
                ground_y_max=ground_y_max,
                ground_y_min=ground_y_min,
            )

        result = HeightMapGenerator._interpolate(height_map)
        if smooth_sigma > 0:
            result = HeightMapGenerator._smooth_edge_preserving(result, max_sigma=smooth_sigma)
        return result

    @staticmethod
    def _flood_fill_ground(
        height_map: np.ndarray,
        camera_height_meters: float,
        max_step: float,
    ) -> np.ndarray:
        """
        BFS from the grid centre outward; returns a bool mask of accepted cells.

        A neighbour is accepted if it has data AND its height does not differ from
        the current cell by more than max_step.  Empty cells (NaN) act as barriers
        — sky-masked regions near the horizon naturally stop the fill.
        """
        grid_h, grid_w = height_map.shape
        has_data = ~np.isnan(height_map)

        start_r, start_c = grid_h // 2, grid_w // 2

        # If the centre cell is empty, find the nearest filled cell to it.
        if not has_data[start_r, start_c]:
            ys, xs = np.where(has_data)
            if len(ys) == 0:
                return has_data
            dist = np.hypot(ys - start_r, xs - start_c)
            best = int(np.argmin(dist))
            start_r, start_c = int(ys[best]), int(xs[best])

        accepted = np.zeros((grid_h, grid_w), dtype=bool)
        visited  = np.zeros((grid_h, grid_w), dtype=bool)
        queue = deque()
        queue.append((start_r, start_c))
        visited[start_r, start_c] = True
        accepted[start_r, start_c] = True

        while queue:
            r, c = queue.popleft()
            cur_h = height_map[r, c]

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_h and 0 <= nc < grid_w and not visited[nr, nc]:
                    visited[nr, nc] = True
                    if has_data[nr, nc] and abs(height_map[nr, nc] - cur_h) <= max_step:
                        accepted[nr, nc] = True
                        queue.append((nr, nc))

        return accepted

    @staticmethod
    def _fill_from_panorama_depth(
        height_map: np.ndarray,
        panorama_depth: "Depth",
        sky_mask: Optional[np.ndarray],
        grid_size_meters: float,
        grid_resolution: int,
        ground_y_max: float,
        ground_y_min: float,
    ) -> np.ndarray:
        """
        Project a 360° equirectangular depth map into empty (NaN) cells of height_map.

        Only cells that are already NaN are written; existing data is never overwritten.
        The same ground-plane filters used for the primary depth are applied so sky and
        wall pixels in the panorama don't pollute the terrain grid.
        """
        missing = np.isnan(height_map)
        if not np.any(missing):
            return height_map

        pd = panorama_depth.depth.astype(np.float32)
        if sky_mask is not None and sky_mask.shape == pd.shape:
            pd = pd.copy()
            pd[sky_mask] = np.nan

        X, Y, Z = Panorama.equirectangular_unproject(Depth(pd))

        half = grid_size_meters / 2.0
        ground_mask = (
            (Y <= ground_y_max)
            & (Y >= ground_y_min)
            & (np.abs(Z) <= half)
            & (np.abs(X) <= half)
            & np.isfinite(pd)
        )

        Xg = X[ground_mask]
        Yg = Y[ground_mask]
        Zg = Z[ground_mask]

        if len(Xg) == 0:
            return height_map

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)

        xi = np.digitize(Xg, x_edges) - 1
        zi = np.digitize(Zg, z_edges) - 1

        in_bounds = (
            (xi >= 0) & (xi < grid_resolution)
            & (zi >= 0) & (zi < grid_resolution)
        )
        xi, zi, Yg = xi[in_bounds], zi[in_bounds], Yg[in_bounds]

        pano_sum = np.zeros((grid_resolution, grid_resolution), dtype=np.float64)
        pano_cnt = np.zeros((grid_resolution, grid_resolution), dtype=np.int32)
        np.add.at(pano_sum, (zi, xi), Yg)
        np.add.at(pano_cnt, (zi, xi), 1)

        has_pano = (pano_cnt > 0) & missing
        result = height_map.copy()
        result[has_pano] = (pano_sum[has_pano] / pano_cnt[has_pano]).astype(np.float32)
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

        y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
        x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
        dist = np.clip(np.sqrt(y ** 2 + x ** 2) / np.sqrt(2.0), 0.0, 1.0)

        gy, gx = np.gradient(height_map)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2).astype(np.float32)

        scale = np.percentile(grad_mag, 85) / edge_sensitivity + 1e-6
        edge_weight = np.tanh(grad_mag / scale)
        edge_weight = gaussian_filter(edge_weight, sigma=2.0)
        flat_weight = (1.0 - edge_weight).clip(0.0, 1.0)

        smooth_weight = (dist * flat_weight).astype(np.float32)

        sigmas = np.linspace(0.0, max_sigma, n_levels + 1)
        levels = np.stack(
            [height_map if s == 0 else gaussian_filter(height_map, sigma=s)
             for s in sigmas],
            axis=0,
        )

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
