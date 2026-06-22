import numpy as np
from scipy.ndimage import distance_transform_edt
from typing import Optional

from util.depth_utils import Depth
from util.panorama_utils import Panorama
from pipeline.panorama_segmentation.panorama_region_result import (
    ALL_REGION_TYPES,
    REGION_TYPE_SKY,
    REGION_TYPE_OTHER,
)


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
    ) -> np.ndarray:
        """
        Project per-pixel region type labels from an equirectangular panorama onto a
        top-down grid, using the same XZ binning as the height map.

        Each grid cell is assigned the majority region type among all ground-plane
        pixels that project into it.  Empty cells are filled by nearest-neighbour
        propagation from the closest labelled cell.

        Returns a (grid_resolution, grid_resolution) uint8 array of ALL_REGION_TYPES
        indices.  Grid layout matches the height map: rows = Z near→far, cols = X
        left→right.
        """
        sky_idx = ALL_REGION_TYPES.index(REGION_TYPE_SKY)
        other_idx = ALL_REGION_TYPES.index(REGION_TYPE_OTHER)

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
            return np.full((grid_resolution, grid_resolution), other_idx, dtype=np.uint8)

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

        n_types = len(ALL_REGION_TYPES)
        vote_counts = np.zeros((grid_resolution, grid_resolution, n_types), dtype=np.int32)
        np.add.at(vote_counts, (zi, xi, Tg.astype(np.intp)), 1)

        has_data = vote_counts.sum(axis=2) > 0
        region_map = np.full((grid_resolution, grid_resolution), other_idx, dtype=np.uint8)
        region_map[has_data] = vote_counts[has_data].argmax(axis=1).astype(np.uint8)

        return RegionMapGenerator._fill_nearest(region_map, has_data)

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
    def extract_mountain_silhouette(
        type_idx_map: np.ndarray,
        sky_idx: int,
        terrain_idx: int,
    ) -> np.ndarray:
        """
        Extract the mountain silhouette from an equirectangular region type map.

        Returns a float32 (H, W) binary mask where 1.0 marks terrain pixels that
        sit directly below a sky pixel — the ridgeline at the sky-terrain horizon.
        """
        h, w = type_idx_map.shape
        sky_mask = type_idx_map == sky_idx
        terrain_mask = type_idx_map == terrain_idx

        above_is_sky = np.zeros((h, w), dtype=bool)
        above_is_sky[1:, :] = sky_mask[:-1, :]

        silhouette = terrain_mask & above_is_sky
        return silhouette.astype(np.float32)

    @staticmethod
    def project_silhouette_to_grid(
        panorama_depth: Depth,
        silhouette_mask: np.ndarray,
        grid_size_meters: float = 100.0,
        grid_resolution: int = 512,
    ) -> np.ndarray:
        """
        Project panorama-space silhouette pixels onto a top-down grid.

        Mountain ridgelines are at the horizon — their literal XZ distances are
        typically far beyond the grid boundary. Instead, each silhouette pixel is
        projected to the edge of the grid in its azimuth direction, marking which
        directions have a visible ridgeline.

        Returns a float32 (grid_resolution, grid_resolution) binary mask where 1.0
        marks grid-edge cells in directions that contain a silhouette pixel.
        """
        d = panorama_depth.depth.astype(np.float32)
        X, _, Z = Panorama.equirectangular_unproject(Depth(d))

        valid = (silhouette_mask > 0) & np.isfinite(d)
        if not np.any(valid):
            return np.zeros((grid_resolution, grid_resolution), dtype=np.float32)

        Xs = X[valid]
        Zs = Z[valid]

        # Normalize to unit direction in XZ, then scale to grid edge.
        r = np.sqrt(Xs ** 2 + Zs ** 2)
        nonzero = r > 0
        Xs, Zs, r = Xs[nonzero], Zs[nonzero], r[nonzero]

        half = grid_size_meters / 2.0
        # Scale each direction so the longer axis hits the grid boundary.
        scale = half / np.maximum(np.abs(Xs), np.abs(Zs))
        Xs = Xs * scale
        Zs = Zs * scale

        x_edges = np.linspace(-half, half, grid_resolution + 1)
        z_edges = np.linspace(-half, half, grid_resolution + 1)

        xi = np.clip(np.digitize(Xs, x_edges) - 1, 0, grid_resolution - 1)
        zi = np.clip(np.digitize(Zs, z_edges) - 1, 0, grid_resolution - 1)

        grid = np.zeros((grid_resolution, grid_resolution), dtype=np.float32)
        grid[zi, xi] = 1.0
        return grid

    @staticmethod
    def extract_water_skeleton(
        region_map: np.ndarray,
        water_idx: int,
        smooth_radius: int = 40,
    ) -> np.ndarray:
        """
        Skeletonize the water region of the top-down region map.

        Returns a float32 (H, W) binary mask of the medial axis of all water cells,
        reducing water bodies to 1-pixel-wide centerlines.

        smooth_radius: closing+opening radius applied before skeletonizing. Closing
        fills concave notches; opening removes convex bumps. Together they aggressively
        simplify the boundary so the skeleton has far fewer branches.
        """
        from skimage.morphology import skeletonize as sk_skeletonize, disk, binary_closing, binary_opening
        water_mask = region_map == water_idx
        if not np.any(water_mask):
            return np.zeros(region_map.shape, dtype=np.float32)
        if smooth_radius > 0:
            d = disk(smooth_radius)
            water_mask = binary_closing(water_mask, d)
            water_mask = binary_opening(water_mask, d)
        skeleton = sk_skeletonize(water_mask)
        return skeleton.astype(np.float32)
