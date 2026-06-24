import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
from typing import Optional

from util.depth_utils import Depth
from util.panorama_utils import Panorama
from util.projection_utils import ground_projection_certainty
from pipeline.panorama_segmentation.panorama_region_result import RegionType


class RegionMapGenerator:
    @staticmethod
    def generate(
        panorama_depth: Depth,
        type_idx_map: np.ndarray,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 512,
        ground_y_max: float = -0.5,
        camera_height_meters: float = 1.0,
        sky_mask: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project per-pixel region type labels from an equirectangular panorama onto a
        top-down grid, using the same XZ binning as the height map.

        Each grid cell is assigned the majority region type among all ground-plane
        pixels that project into it.  Empty cells are filled by nearest-neighbour
        propagation from the closest labelled cell.

        Returns (region_map, certainty_map):
          region_map   — (grid_resolution, grid_resolution) uint8 of RegionType indices.
          certainty_map — (grid_resolution, grid_resolution) float32 in [0, 1]: projection
                         certainty (sin²(depression angle)) for cells with direct observations,
                         0 for cells filled by nearest-neighbour propagation.

        Grid layout matches the height map: rows = Z near→far, cols = X left→right.
        """
        sky_idx = RegionType.SKY
        other_idx = RegionType.OTHER

        d = panorama_depth.depth.astype(np.float32)
        if sky_mask is not None and sky_mask.shape == d.shape:
            d = d.copy()
            d[sky_mask] = np.nan

        X, Y, Z = Panorama.equirectangular_unproject(Depth(d))

        ground_y_min = -(camera_height_meters + 5.0)
        half = grid_size_meters / 2.0

        ground_mask = (
            (Y <= ground_y_max)
            & (Y >= ground_y_min)
            & (np.abs(X) <= half)
            & (np.abs(Z) <= half)
            & np.isfinite(d)
            & (type_idx_map != sky_idx)
        )

        if not np.any(ground_mask):
            zero_certainty = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
            return np.full((grid_resolution, grid_resolution), other_idx, dtype=np.uint8), zero_certainty

        Xg = X[ground_mask]
        Zg = Z[ground_mask]
        Tg = type_idx_map[ground_mask]

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)

        xi = np.digitize(Xg, x_edges) - 1
        zi = np.digitize(Zg, z_edges) - 1

        in_bounds = (
            (xi >= 0) & (xi < grid_resolution)
            & (zi >= 0) & (zi < grid_resolution)
        )
        xi, zi, Tg = xi[in_bounds], zi[in_bounds], Tg[in_bounds]

        n_types = len(RegionType)
        vote_counts = np.zeros((grid_resolution, grid_resolution, n_types), dtype=np.int32)
        np.add.at(vote_counts, (zi, xi, Tg.astype(np.intp)), 1)

        has_data = vote_counts.sum(axis=2) > 0
        region_map = np.full((grid_resolution, grid_resolution), other_idx, dtype=np.uint8)
        region_map[has_data] = vote_counts[has_data].argmax(axis=1).astype(np.uint8)

        # Certainty: normalised geometric distortion × observation-density ratio.
        #
        # Geometric component — sin²(φ) = h²/(r²+h²) measures how much
        # equirectangular projection distorts each cell.  With camera_height ≈ 1 m
        # and a 100 m grid the raw values span four orders of magnitude (≈0.0001 at
        # 50 m vs 1.0 directly below the camera), making the map unreadable.  We
        # normalise to the *range of distortions actually present* in this scene so
        # the least-distorted observed cell gets 1.0 and the most-distorted gets 0.0.
        # This makes certainty relative to the scene rather than absolute.
        #
        # Observation-density component — compares the actual per-cell pixel count
        # against the expected count if every ground point had an unobstructed view
        # (expected ∝ geometric certainty).  Cells that received fewer observations
        # than predicted are (partially) occluded → their certainty is scaled down.
        # This is the "clear line of sight" factor.  NN-filled cells stay at 0.
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
        z_centers = (z_edges[:-1] + z_edges[1:]) / 2.0
        X_grid, Z_grid = np.meshgrid(x_centers, z_centers)
        certainty_field = ground_projection_certainty(X_grid, Z_grid, camera_height_meters)

        obs_count = vote_counts.sum(axis=2).astype(np.float32)
        if has_data.any():
            # Log-normalise the geometric field within the observed region.
            # sin²(φ) spans ~4 orders of magnitude across a 100 m grid, so a
            # linear rescale still clusters everything near zero.  Taking -log
            # converts the multiplicative range to an additive one and then we
            # rescale so the least-distorted observed cell → 1.0 and the most
            # distorted → 0.0.  This gives a perceptually even spread.
            log_field = -np.log(np.maximum(certainty_field, 1e-9))
            log_obs = log_field[has_data]
            log_min, log_max = float(log_obs.min()), float(log_obs.max())
            geom_normalised = 1.0 - (log_field - log_min) / max(log_max - log_min, 1e-9)
            geom_normalised = np.clip(geom_normalised, 0.0, 1.0)

            # Observation-density: actual pixel count vs. geometric expectation.
            geom_sum = float(certainty_field[has_data].sum())
            obs_sum = float(obs_count[has_data].sum())
            scale = obs_sum / geom_sum if geom_sum > 0 else 1.0
            expected = certainty_field * scale
            obs_density = np.where(
                has_data,
                np.minimum(obs_count / np.maximum(expected, 1e-6), 1.0),
                0.0,
            )
        else:
            geom_normalised = certainty_field
            obs_density = np.zeros_like(certainty_field)
        certainty = np.where(has_data, geom_normalised * obs_density, 0.0).astype(np.float32)

        return RegionMapGenerator._fill_nearest(region_map, has_data), certainty

    @staticmethod
    def _fill_nearest(region_map: np.ndarray, has_data: np.ndarray) -> np.ndarray:
        empty = ~has_data
        if not np.any(empty) or not np.any(has_data):
            return region_map
        _, indices = distance_transform_edt(empty, return_indices=True)
        result = region_map.copy()
        result[empty] = region_map[indices[0][empty], indices[1][empty]]
        return result

    @staticmethod
    def extract_mountain_ridgeline(
        type_idx_map: np.ndarray,
        panorama_depth: Depth,
        sky_idx: int,
        terrain_idx: int = -1,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 512,
        depth_offset_rows: int = 3,
        depth_smooth_width: int = 15,
        dilation_iters: int = 3,
        connect_radius_px: int = 30,
        chain_smooth_window: int = 15,
    ) -> np.ndarray:
        """
        Extract the sky-foreground horizon per column, sample depth just below it,
        and project to a top-down grid using actual XZ positions.

        Any non-sky pixel that sits directly below a sky pixel is treated as part
        of the ridgeline — this covers bare TERRAIN, forest-covered mountains
        (VEGETATION), and built structures on hilltops (BUILT), which would all be
        missed if only the TERRAIN type were accepted.

        For each panorama column, finds the first non-sky row below sky, then
        samples depth depth_offset_rows below that boundary (where depth estimators
        are more reliable than at the exact edge). NaN depth values are interpolated
        from neighboring columns. Each valid column is unprojected to XZ and placed
        in the grid — close mountains land near the centre, distant ones near the edge.

        Projected points are chained via greedy nearest-neighbour search within
        connect_radius_px grid cells, then smoothed with a moving average of width
        chain_smooth_window, and finally rasterised with filled line segments so the
        result is a continuous ridge rather than isolated scattered pixels.

        Returns float32 (grid_resolution, grid_resolution) binary mask.
        """
        h, w = type_idx_map.shape
        sky_mask = type_idx_map == sky_idx

        above_is_sky = np.zeros((h, w), dtype=bool)
        above_is_sky[1:, :] = sky_mask[:-1, :]
        # Any non-sky pixel directly below sky is a ridgeline candidate.
        silhouette = (~sky_mask) & above_is_sky

        has_silhouette = silhouette.any(axis=0)                      # (W,) bool
        boundary_row = silhouette.argmax(axis=0)                     # (W,) int
        sample_rows = np.clip(boundary_row + depth_offset_rows, 0, h - 1)

        d = panorama_depth.depth.astype(np.float32)
        cols = np.arange(w)
        depths = np.where(has_silhouette, d[sample_rows, cols], np.nan)

        # Interpolate NaN depths horizontally so depth gaps don't leave holes.
        nan_mask = ~np.isfinite(depths) | (depths <= 0)
        valid_for_interp = has_silhouette & ~nan_mask
        if valid_for_interp.any() and nan_mask.any():
            valid_cols = cols[valid_for_interp].astype(np.float64)
            valid_depths = depths[valid_for_interp]
            depths = np.where(
                has_silhouette,
                np.interp(cols.astype(np.float64), valid_cols, valid_depths),
                np.nan,
            )

        # Suppress single-column depth spikes before projecting to XZ.
        if depth_smooth_width > 1 and np.any(np.isfinite(depths)):
            from scipy.ndimage import median_filter as _med
            _arr = np.where(np.isfinite(depths), depths, 0.0)
            depths = np.where(np.isfinite(depths), _med(_arr, size=depth_smooth_width, mode='nearest'), depths)

        phi_sil = (0.5 - sample_rows / h) * np.pi   # elevation angle at ridgeline row
        cos_phi_sil = np.cos(phi_sil)               # horizontal scale factor
        theta = (cols / w - 0.5) * 2.0 * np.pi     # longitude: 0 = +Z (forward)
        Xs = depths * cos_phi_sil * np.sin(theta)
        Zs = depths * cos_phi_sil * np.cos(theta)

        half = grid_size_meters / 2.0
        in_bounds = (
            has_silhouette
            & np.isfinite(depths)
            & (np.abs(Xs) <= half)
            & (np.abs(Zs) <= half)
        )
        Xs, Zs = Xs[in_bounds], Zs[in_bounds]

        if len(Xs) == 0:
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)

        # ── Greedy nearest-neighbour chain ────────────────────────────────────
        # Depth noise projects consecutive panorama columns to scattered XZ
        # positions.  Walk the points in order, always stepping to the closest
        # unvisited neighbour within connect_radius_px grid cells, so the result
        # is a single connected path rather than isolated pixels.
        from scipy.spatial import KDTree as _KDTree
        from scipy.ndimage import uniform_filter1d as _uf1d

        cell_m = grid_size_meters / grid_resolution
        connect_m = connect_radius_px * cell_m

        pts = np.stack([Xs.astype(np.float64), Zs.astype(np.float64)], axis=-1)
        tree = _KDTree(pts)
        visited = np.zeros(len(pts), dtype=bool)
        visited[0] = True
        chain = [0]

        while True:
            curr = chain[-1]
            idxs = tree.query_ball_point(pts[curr], connect_m)
            candidates = [i for i in idxs if not visited[i]]
            if not candidates:
                break
            dists = np.linalg.norm(pts[candidates] - pts[curr], axis=1)
            nxt = candidates[int(np.argmin(dists))]
            chain.append(nxt)
            visited[nxt] = True

        chain_pts = pts[chain].copy()   # (M, 2) ordered metres

        # ── Smooth the chain to remove per-column depth jitter ────────────────
        if len(chain_pts) >= 3 and chain_smooth_window > 1:
            win = min(chain_smooth_window, len(chain_pts))
            chain_pts[:, 0] = _uf1d(chain_pts[:, 0], size=win)
            chain_pts[:, 1] = _uf1d(chain_pts[:, 1], size=win)

        # ── Rasterise as connected line segments ──────────────────────────────
        xi_c = np.clip(np.digitize(chain_pts[:, 0], x_edges) - 1, 0, grid_resolution - 1).astype(int)
        zi_c = np.clip(np.digitize(chain_pts[:, 1], z_edges) - 1, 0, grid_resolution - 1).astype(int)

        grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        for k in range(len(xi_c) - 1):
            x0, z0 = xi_c[k],     zi_c[k]
            x1, z1 = xi_c[k + 1], zi_c[k + 1]
            n_steps = max(abs(x1 - x0), abs(z1 - z0), 1) + 1
            xs_seg = np.round(np.linspace(x0, x1, n_steps)).astype(int)
            zs_seg = np.round(np.linspace(z0, z1, n_steps)).astype(int)
            valid = (xs_seg >= 0) & (xs_seg < grid_resolution) & (zs_seg >= 0) & (zs_seg < grid_resolution)
            grid[zs_seg[valid], xs_seg[valid]] = 1.0
        if len(xi_c) > 0:
            grid[zi_c[-1], xi_c[-1]] = 1.0

        # Dilate to give the ridgeline coherent thickness for the terrain solver.
        if dilation_iters > 0:
            from scipy.ndimage import binary_dilation
            grid = binary_dilation(grid > 0, iterations=dilation_iters).astype(np.float32)

        return grid

    @staticmethod
    def extract_region_skeleton(
        region_map: np.ndarray,
        type_idx: int,
        smooth_radius: int = 40,
    ) -> np.ndarray:
        """
        Skeletonize a single region type in the top-down region map.

        Returns a float32 (H, W) binary mask of the medial axis of all matching
        cells, reducing area features to 1-pixel-wide centerlines.

        smooth_radius: closing+opening radius applied before skeletonizing. Use
        larger values for area features like water bodies; smaller values for
        inherently linear features like roads and trails.
        """
        from skimage.morphology import skeletonize as sk_skeletonize, disk, binary_closing, binary_opening
        mask = region_map == type_idx
        if not np.any(mask):
            return np.zeros(region_map.shape, dtype=np.float32)
        if smooth_radius > 0:
            d = disk(smooth_radius)
            mask = binary_closing(mask, d)
            mask = binary_opening(mask, d)
        return sk_skeletonize(mask).astype(np.float32)

    @staticmethod
    def extract_interior_peaks(
        type_idx_map: np.ndarray,
        panorama_depth: Depth,
        sky_idx: int,
        panorama_rgb: Optional[np.ndarray] = None,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 512,
        depth_jump_rel: float = 0.20,
        canny_low: int = 50,
        canny_high: int = 150,
        corner_quality: float = 0.08,
        corner_min_dist: int = 5,
    ) -> np.ndarray:
        """
        Detect elevated terrain features visible against background terrain (rather
        than against sky) and project them onto a top-down grid.

        The sky-terrain ridgeline captures only the outermost horizon silhouette.
        This method finds interior peaks — rocky outcrops, ridges, or hilltops that
        appear in front of more distant terrain — by combining three signals:

          1. Relative depth jumps: |∇d| / d > depth_jump_rel detects boundaries
             where foreground terrain abruptly occludes more distant ground.

          2. Canny edges on the RGB panorama: visual edge boundaries at
             colour/luminance discontinuities characteristic of rock faces and
             hard terrain edges.

          3. Harris corner responses on the RGB panorama: corner features at
             convergence points of two edges, which correspond to the apices of
             pointed peaks and the vertices of rocky outcrops.

        Pixels must satisfy both (1) AND at least one of (2)/(3) to be accepted —
        this filters out pure depth-estimation noise (no visual evidence) and flat
        colour boundaries (no depth change).  Sky pixels and pixels more than
        half the grid size away in XZ are excluded.

        The panorama is processed at its native resolution; if depth or type maps
        are at a different resolution they are rescaled to match.

        Returns float32 (grid_resolution, grid_resolution) with each cell containing
        the count of peak-candidate pixels that projected into it, normalised to [0,1].
        """
        ref_h, ref_w = type_idx_map.shape

        # Align depth to type-map resolution if needed.
        d_raw = panorama_depth.depth.astype(np.float32)
        if d_raw.shape != (ref_h, ref_w):
            d_raw = zoom(d_raw, (ref_h / d_raw.shape[0], ref_w / d_raw.shape[1]), order=1)

        # Work at panorama_rgb resolution when provided (typically lower than depth).
        if panorama_rgb is not None:
            work_h, work_w = panorama_rgb.shape[:2]
            if (work_h, work_w) != (ref_h, ref_w):
                d_work = zoom(d_raw, (work_h / ref_h, work_w / ref_w), order=1)
                types_work = zoom(type_idx_map.astype(np.float32), (work_h / ref_h, work_w / ref_w), order=0).astype(np.uint8)
            else:
                d_work, types_work = d_raw, type_idx_map
        else:
            d_work, types_work = d_raw, type_idx_map
            work_h, work_w = ref_h, ref_w

        sky_mask = types_work == sky_idx
        valid = (~sky_mask) & np.isfinite(d_work) & (d_work > 0.5)

        # ── 1. Relative depth gradient ────────────────────────────────────────
        gy, gx = np.gradient(d_work)
        grad_mag = np.hypot(gx, gy)
        rel_grad = np.where(d_work > 0, grad_mag / d_work, 0.0)
        depth_edge = (rel_grad > depth_jump_rel) & valid

        # Exclude pixels that sit at the sky boundary (those are the ridgeline).
        above_is_sky = np.zeros((work_h, work_w), dtype=bool)
        above_is_sky[1:, :] = sky_mask[:-1, :]
        at_skyline = (~sky_mask) & above_is_sky
        depth_edge &= ~at_skyline

        if not depth_edge.any():
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        # ── 2 & 3. Visual edges and corners from RGB ─────────────────────────
        visual = np.zeros((work_h, work_w), dtype=bool)
        if panorama_rgb is not None:
            gray = cv2.cvtColor(panorama_rgb, cv2.COLOR_RGB2GRAY)
            canny = cv2.Canny(gray, canny_low, canny_high) > 0

            # Shi-Tomasi corners — peaks of pointed features appear as corners
            # where two ridgeline segments converge.
            corners_f32 = cv2.goodFeaturesToTrack(
                gray, maxCorners=0, qualityLevel=corner_quality,
                minDistance=corner_min_dist,
            )
            corner_map = np.zeros((work_h, work_w), dtype=bool)
            if corners_f32 is not None:
                pts = corners_f32[:, 0, :].astype(np.int32)
                cx = np.clip(pts[:, 0], 0, work_w - 1)
                cy = np.clip(pts[:, 1], 0, work_h - 1)
                corner_map[cy, cx] = True
                # Dilate corners by 3 px so they overlap with the depth-edge mask.
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                corner_map = cv2.dilate(corner_map.astype(np.uint8), kernel).astype(bool)

            visual = canny | corner_map

        # Require depth evidence; additionally gate on visual evidence when available.
        candidates = depth_edge & (visual if panorama_rgb is not None else np.ones_like(depth_edge))

        if not candidates.any():
            # Fall back to depth-only if RGB filtering removed everything.
            candidates = depth_edge

        # ── Project to top-down grid ─────────────────────────────────────────
        # Use the near-depth side: for each edge pixel, the smaller of its own
        # depth and its immediate neighbours is the foreground (peak) distance.
        from scipy.ndimage import minimum_filter
        min_d = minimum_filter(d_work, size=3)
        # Accept pixels where the local minimum depth is within 20% of their own
        # depth — this biases toward the foreground side of each edge.
        near_side = candidates & (d_work <= min_d * 1.2)
        if not near_side.any():
            near_side = candidates

        rows, cols = np.where(near_side)
        depths = min_d[rows, cols]

        half = grid_size_meters / 2.0
        theta = (cols / work_w - 0.5) * 2.0 * np.pi
        phi   = (0.5 - rows / work_h) * np.pi
        cos_phi = np.cos(phi)
        Xs = depths * cos_phi * np.sin(theta)
        Zs = depths * cos_phi * np.cos(theta)

        in_bounds = (
            np.isfinite(Xs) & np.isfinite(Zs)
            & (np.abs(Xs) <= half)
            & (np.abs(Zs) <= half)
        )
        Xs, Zs = Xs[in_bounds], Zs[in_bounds]

        if len(Xs) == 0:
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)
        xi = np.clip(np.digitize(Xs, x_edges) - 1, 0, grid_resolution - 1)
        zi = np.clip(np.digitize(Zs, z_edges) - 1, 0, grid_resolution - 1)

        grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        np.add.at(grid, (zi, xi), 1.0)

        peak_max = grid.max()
        if peak_max > 0:
            grid /= peak_max
        return grid
