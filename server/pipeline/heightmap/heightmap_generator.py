import numpy as np
from collections import deque
from scipy.ndimage import gaussian_filter, zoom
from typing import Optional
from util.depth_utils import Depth
from util.panorama_utils import Panorama
from util.projection_utils import ground_projection_certainty, project_panorama_to_ground_grid
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
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project ground points from a depth map onto a top-down height grid.

        Returns (height_array, certainty_array), both (grid_resolution, grid_resolution)
        float32.  certainty is in [0, 1]: sin²(depression_angle) for cells with any
        direct observation (primary or panorama depth), 0 for pure interpolation.

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
            zeros = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
            return zeros, zeros.copy()

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

        # Certainty: projection-distortion based, zero for unobserved (interpolated) cells.
        # Observed cells use sin²(elevation) = h² / (r² + h²), which is the inverse of
        # the equirectangular Jacobian (ground-area per panorama pixel).
        observed = ~np.isnan(height_map)
        certainty = HeightMapGenerator._build_certainty(
            observed, grid_size_meters, grid_resolution, camera_height_meters
        )

        result = HeightMapGenerator._interpolate(height_map)
        if smooth_sigma > 0:
            result = HeightMapGenerator._smooth_edge_preserving(result, max_sigma=smooth_sigma)
        return result, certainty

    @staticmethod
    def _build_certainty(
        observed: np.ndarray,
        grid_size_meters: float,
        grid_resolution: int,
        camera_height: float,
    ) -> np.ndarray:
        """
        Build a [0, 1] certainty map over the top-down grid.

        Observed cells are scored by equirectangular projection certainty (see
        ground_projection_certainty); unobserved (interpolated) cells get 0.
        """
        half = grid_size_meters / 2.0
        x_centers = np.linspace(-half, half, grid_resolution, endpoint=False, dtype=np.float32) + half / grid_resolution
        z_centers = np.linspace(-half, half, grid_resolution, endpoint=False, dtype=np.float32) + half / grid_resolution
        X_grid, Z_grid = np.meshgrid(x_centers, z_centers)
        certainty_field = ground_projection_certainty(X_grid, Z_grid, camera_height)
        return np.where(observed, certainty_field, 0.0).astype(np.float32)

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
        """
        missing = np.isnan(height_map)
        if not np.any(missing):
            return height_map

        xi, zi, _, Yg, _ = project_panorama_to_ground_grid(
            panorama_depth=panorama_depth,
            grid_size_meters=grid_size_meters,
            grid_resolution=grid_resolution,
            ground_y_max=ground_y_max,
            ground_y_min=ground_y_min,
            sky_mask=sky_mask,
        )

        if len(xi) == 0:
            return height_map

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
    def _interpolate(height_map: np.ndarray, n_octaves: int = 4, noise_seed: int = 0) -> np.ndarray:
        """
        Multi-scale noise inpainting for unknown (NaN) cells, after Algorithm 3
        in Jain et al. 2026.

        Works coarse-to-fine: at the coarsest octave, unknown cells are seeded
        with noise around the mean height.  At each finer octave the previous
        result is upsampled, smaller-amplitude noise is injected into still-
        unknown cells, and Laplacian diffusion (iterative Gaussian relaxation)
        propagates boundary values inward.  Known cells act as fixed Dirichlet
        boundary conditions throughout, so the final output matches the input
        exactly at every measured point.
        """
        known_mask = ~np.isnan(height_map)
        if np.all(known_mask):
            return height_map
        if not np.any(known_mask):
            return np.zeros_like(height_map)

        rng = np.random.default_rng(noise_seed)
        h, w = height_map.shape
        max_factor = 2 ** (n_octaves - 1)

        # Noise amplitude calibrated to ~25% of the terrain's height variation.
        noise_scale = float(np.std(height_map[known_mask])) * 0.25

        def nan_downsample(factor: int) -> np.ndarray:
            sh, sw = max(1, h // factor), max(1, w // factor)
            crop = height_map[: sh * factor, : sw * factor]
            with np.errstate(all="ignore"):
                return np.nanmean(
                    crop.reshape(sh, factor, sw, factor), axis=(1, 3)
                ).astype(np.float32)

        def diffuse(Z: np.ndarray, mask: np.ndarray, sigma: float, n_iters: int) -> np.ndarray:
            for _ in range(n_iters):
                Z = np.where(mask, Z, gaussian_filter(Z, sigma=sigma))
            return Z

        ZI: Optional[np.ndarray] = None

        for octave in range(n_octaves - 1, -1, -1):
            factor = 2 ** octave
            ds = nan_downsample(factor) if factor > 1 else height_map.copy()
            ds_mask = ~np.isnan(ds)
            sh, sw = ds.shape
            ds_vals = np.where(ds_mask, ds, 0.0)

            if ZI is None:
                # Coarsest level: seed unknown cells with mean + full-amplitude noise.
                mean_h = float(np.nanmean(ds))
                noise = rng.standard_normal((sh, sw)).astype(np.float32) * noise_scale
                ZI = np.where(ds_mask, ds_vals, mean_h + noise)
            else:
                # Upsample and inject scale-dependent noise into still-unknown cells.
                ZI_up = zoom(ZI, (sh / ZI.shape[0], sw / ZI.shape[1]), order=1)
                amplitude = noise_scale * factor / max_factor
                noise = rng.standard_normal((sh, sw)).astype(np.float32) * amplitude
                ZI = np.where(ds_mask, ds_vals, ZI_up + noise)

            # Laplacian diffusion: larger sigma at coarser levels for long-range fill;
            # fewer iterations at fine levels where the coarse pass already set structure.
            sigma = max(1.0, sh / 32.0)
            n_iters = max(5, 40 * factor // max_factor)
            ZI = diffuse(ZI, ds_mask, sigma, n_iters)

        # Restore original known values exactly — no drift from the diffusion passes.
        return np.where(known_mask, height_map, ZI).astype(np.float32)
