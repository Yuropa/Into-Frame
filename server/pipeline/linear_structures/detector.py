"""
Linear structure graph construction from pre-computed skeleton masks, plus
terrain modification (valley carving for rivers, smoothing for roads).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt, convolve
from skimage.measure import label as sk_label
from typing import Optional

from util.depth_utils import Depth
from pipeline.linear_structures.graph import LinearGraph, LinearStructure


_MIN_PATH_PX = 12
_PATH_DOWNSAMPLE = 3  # must match the downsample used in _pixel_paths_to_world

_EXTRAP_BOUNDARY_MARGIN = 0.04   # fraction of grid_size — endpoint within this of edge is "already there"
_EXTRAP_STEP_M          = 1.0    # world-space step size for extrapolation (metres)
_EXTRAP_TANGENT_PTS     = 6      # number of downsampled path points used to estimate bearing


def _neighbour_count_image(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def _skeleton_to_polylines(
    skel: np.ndarray,
    min_length: int = _MIN_PATH_PX,
) -> list[list[tuple[int, int]]]:
    """Convert a skeleton image into a list of ordered pixel paths."""
    nbr = _neighbour_count_image(skel)
    junction_mask = (nbr >= 3) & skel
    no_junctions  = skel & ~junction_mask

    frag_labels, n_frags = sk_label(no_junctions, return_num=True)
    all_pixels = set(map(tuple, np.argwhere(skel)))

    def pixel_neighbors(r, c):
        return [
            (r + dr, c + dc)
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr, dc) != (0, 0) and (r + dr, c + dc) in all_pixels
        ]

    paths: list[list[tuple[int, int]]] = []

    for frag_id in range(1, n_frags + 1):
        frag_pts = set(map(tuple, np.argwhere(frag_labels == frag_id)))
        if len(frag_pts) < min_length:
            continue

        frag_nbr = {
            pt: sum(1 for n in pixel_neighbors(*pt) if n in frag_pts)
            for pt in frag_pts
        }
        endpoints = [pt for pt, cnt in frag_nbr.items() if cnt <= 1]
        start = endpoints[0] if endpoints else next(iter(frag_pts))

        path: list[tuple[int, int]] = [start]
        prev = None
        curr = start
        while True:
            nxt = [n for n in pixel_neighbors(*curr) if n in frag_pts and n != prev]
            if not nxt:
                break
            prev, curr = curr, nxt[0]
            if curr == start:
                break
            path.append(curr)

        if len(path) >= min_length:
            paths.append(path)

    return paths


def _pixel_paths_to_world(
    paths: list[list[tuple[int, int]]],
    hm_arr: np.ndarray,
    grid_size: float,
    hm_res: int,
    downsample: int = 3,
) -> list[np.ndarray]:
    """Convert height-map pixel paths to (K, 3) world-space arrays."""
    x_half = z_far = grid_size / 2.0
    world_paths: list[np.ndarray] = []
    for path in paths:
        pts = path[::max(1, downsample)]
        rows = np.array([p[0] for p in pts], dtype=np.float32)
        cols = np.array([p[1] for p in pts], dtype=np.float32)
        z = rows / (hm_res - 1) * (2.0 * z_far) - z_far
        x = cols / (hm_res - 1) * grid_size - x_half
        y = np.array(
            [float(hm_arr[p[0], p[1]]) for p in pts], dtype=np.float32
        )
        world_paths.append(np.stack([x, y, z], axis=-1).astype(np.float32))
    return world_paths


def _at_grid_boundary(pt: np.ndarray, half: float, margin: float) -> bool:
    """True when the XZ point is within margin of any grid edge."""
    return abs(pt[0]) >= half - margin or abs(pt[2]) >= half - margin


def _extrapolate_to_boundary(
    wp: np.ndarray,
    from_start: bool,
    hm_arr: np.ndarray,
    grid_size: float,
    hm_res: int,
) -> np.ndarray:
    """
    Walk from one endpoint of a world-space path in the direction of its local
    tangent until the path reaches the grid boundary, sampling terrain height
    at each step.  Returns the new extension points (not including the anchor).

    from_start=True  → extend backward from wp[0]
    from_start=False → extend forward  from wp[-1]
    """
    half   = grid_size / 2.0
    margin = grid_size * _EXTRAP_BOUNDARY_MARGIN

    anchor = wp[0] if from_start else wp[-1]
    if _at_grid_boundary(anchor, half, margin):
        return np.empty((0, 3), dtype=np.float32)

    n = min(_EXTRAP_TANGENT_PTS, len(wp) - 1)
    if from_start:
        far = wp[n]          # point further into path body
    else:
        far = wp[-n - 1]

    dir_xz = anchor[[0, 2]] - far[[0, 2]]  # direction pointing away from body
    norm = float(np.linalg.norm(dir_xz))
    if norm < 1e-6:
        return np.empty((0, 3), dtype=np.float32)
    dir_xz = (dir_xz / norm).astype(np.float64)

    pts: list[np.ndarray] = []
    pos = anchor[[0, 2]].astype(np.float64)
    max_steps = int(grid_size * 2 / _EXTRAP_STEP_M) + 10

    for _ in range(max_steps):
        pos += dir_xz * _EXTRAP_STEP_M
        x, z = float(pos[0]), float(pos[1])

        out = abs(x) > half or abs(z) > half
        xc  = float(np.clip(x, -half, half))
        zc  = float(np.clip(z, -half, half))

        col = int(np.clip((xc + half) / grid_size * (hm_res - 1), 0, hm_res - 1))
        row = int(np.clip((zc + half) / grid_size * (hm_res - 1), 0, hm_res - 1))
        y   = float(hm_arr[row, col])
        pts.append(np.array([xc, y, zc], dtype=np.float32))
        if out:
            break

    return np.stack(pts).astype(np.float32) if pts else np.empty((0, 3), dtype=np.float32)


def _extend_world_path(
    wp: np.ndarray,
    hm_arr: np.ndarray,
    grid_size: float,
    hm_res: int,
) -> np.ndarray:
    """Extrapolate both endpoints of a world-space path to the grid boundary."""
    if len(wp) < 2:
        return wp
    prefix = _extrapolate_to_boundary(wp, from_start=True,  hm_arr=hm_arr, grid_size=grid_size, hm_res=hm_res)
    suffix = _extrapolate_to_boundary(wp, from_start=False, hm_arr=hm_arr, grid_size=grid_size, hm_res=hm_res)
    parts = []
    if len(prefix):
        parts.append(prefix[::-1])  # reverse so it flows start → existing path
    parts.append(wp)
    if len(suffix):
        parts.append(suffix)
    return np.concatenate(parts, axis=0).astype(np.float32)


def _sample_widths_from_edt(
    path: list[tuple[int, int]],
    edt: np.ndarray,
    cell_size: float,
) -> np.ndarray:
    """
    Sample per-point full width in metres along a skeleton path.

    edt[r, c] is the distance (pixels) from that pixel to the nearest
    non-region pixel — i.e. the local half-width of the region.  We sample
    every _PATH_DOWNSAMPLE steps to stay aligned with _pixel_paths_to_world.
    """
    pts = path[::_PATH_DOWNSAMPLE]
    return np.array(
        [max(0.5, float(edt[r, c]) * 2.0 * cell_size) for r, c in pts],
        dtype=np.float32,
    )


def _path_mask_from_polylines(
    paths: list[list[tuple[int, int]]], hm_res: int
) -> np.ndarray:
    mask = np.zeros((hm_res, hm_res), dtype=bool)
    for path in paths:
        for r, c in path:
            if 0 <= r < hm_res and 0 <= c < hm_res:
                mask[r, c] = True
    return mask


def _carve_valley(
    hm_arr: np.ndarray,
    path_mask: np.ndarray,
    fallback_width_m: float,
    cell_size: float,
    path_px: Optional[list[tuple[int, int]]] = None,
    widths_m: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gaussian valley carved along a river path with spatially-varying width."""
    dist_px, nn_idx = distance_transform_edt(~path_mask, return_indices=True)
    row_nn, col_nn = nn_idx

    if widths_m is not None and path_px is not None and len(widths_m) > 0:
        # Paint per-point widths onto the skeleton pixels, then propagate to
        # every grid pixel via the nearest-skeleton-point index from the EDT.
        H, W = hm_arr.shape
        width_grid = np.zeros((H, W), dtype=np.float32)
        for (r, c), w in zip(path_px, widths_m):
            if 0 <= r < H and 0 <= c < W:
                width_grid[r, c] = w
        local_w = width_grid[row_nn, col_nn]
        # Pixels not covered by a path point fall back to the median width.
        med_w = float(np.median(widths_m))
        local_w = np.where(local_w > 0, local_w, med_w)
    else:
        local_w = fallback_width_m

    sigma_px  = np.maximum(1.0, local_w / (2.0 * cell_size))
    depth_m   = np.maximum(0.2, local_w * 0.15)
    influence = np.exp(-dist_px.astype(np.float32) ** 2 / (2.0 * sigma_px ** 2))
    return hm_arr - (depth_m * influence).astype(np.float32)


def _smooth_road(hm_arr: np.ndarray, path_mask: np.ndarray,
                 width_m: float, cell_size: float) -> np.ndarray:
    """Gently flatten terrain along a road path."""
    road_y = hm_arr[path_mask]
    if road_y.size == 0:
        return hm_arr
    mean_y    = float(np.mean(road_y))
    dist_px   = distance_transform_edt(~path_mask)
    sigma_px  = max(1.0, (width_m / 2.0) / cell_size)
    influence = np.exp(-dist_px ** 2 / (2.0 * sigma_px ** 2)) * 0.5
    return hm_arr + (influence * (mean_y - hm_arr)).astype(np.float32)


class LinearStructureDetector:

    @staticmethod
    def detect(
        height_map: Depth,
        params: dict,
        water_skeleton: Optional[np.ndarray] = None,
        water_mask: Optional[np.ndarray] = None,
        road_skeleton: Optional[np.ndarray] = None,
        road_mask: Optional[np.ndarray] = None,
        trail_skeleton: Optional[np.ndarray] = None,
        trail_mask: Optional[np.ndarray] = None,
    ) -> LinearGraph:
        """
        Build a LinearGraph from pre-computed semantic skeleton masks.

        Each *_skeleton is a binary grid in height-map space (from RegionMapStage).
        Each *_mask is the full area mask for that type, used for width estimation;
        falls back to the skeleton itself when not provided.
        """
        grid_size = params.get("grid_size_meters", 100.0)
        hm_res    = int(height_map.depth.shape[0])
        cell_size = grid_size / hm_res
        hm_arr    = height_map.depth.copy()

        graph = LinearGraph()

        sources = [
            ("river", water_skeleton, water_mask),
            ("road",  road_skeleton,  road_mask),
            ("trail", trail_skeleton, trail_mask),
        ]

        for structure_type, skeleton, area_mask in sources:
            if skeleton is None or not skeleton.any():
                continue
            skel  = skeleton > 0
            paths = _skeleton_to_polylines(skel)
            if not paths:
                continue

            # EDT of the area mask gives per-pixel half-width (distance to region edge).
            edt = distance_transform_edt(area_mask) if area_mask is not None and area_mask.any() else None

            world_paths = _pixel_paths_to_world(paths, hm_arr, grid_size, hm_res, downsample=_PATH_DOWNSAMPLE)
            world_paths = [
                _extend_world_path(wp, hm_arr, grid_size, hm_res) for wp in world_paths
            ]
            for path, wp in zip(paths, world_paths):
                if len(wp) < 2:
                    continue
                if edt is not None:
                    widths_m = _sample_widths_from_edt(path, edt, cell_size)
                    mean_w   = float(widths_m.mean())
                else:
                    widths_m = None
                    mean_w   = 1.0
                graph.add(LinearStructure(type=structure_type, path=wp, width=mean_w, widths=widths_m))

        return graph

    @staticmethod
    def modify_height_map(
        height_map: Depth,
        params: dict,
        graph: LinearGraph,
        modify_rivers: bool = True,
        modify_roads: bool = True,
    ) -> Depth:
        """
        Apply terrain modifications: valley carving for rivers, smoothing for roads.
        Returns a new Depth with the modified height array.
        """
        if not graph.structures:
            return height_map

        grid_size = params.get("grid_size_meters", 100.0)
        hm_res    = int(height_map.depth.shape[0])
        cell_size = grid_size / hm_res
        hm_arr    = height_map.depth.copy()
        x_half = z_far = grid_size / 2.0

        def world_to_pixel(pts: np.ndarray):
            col = ((pts[:, 0] + x_half) / grid_size * (hm_res - 1)).astype(int)
            row = ((pts[:, 2] + z_far)  / (2.0 * z_far) * (hm_res - 1)).astype(int)
            valid = (col >= 0) & (col < hm_res) & (row >= 0) & (row < hm_res)
            return row[valid], col[valid]

        for structure in graph.structures:
            rows, cols = world_to_pixel(structure.path)
            if len(rows) == 0:
                continue
            path_px   = list(zip(rows.tolist(), cols.tolist()))
            path_mask = _path_mask_from_polylines([path_px], hm_res)

            if structure.type == "river" and modify_rivers:
                hm_arr = _carve_valley(
                    hm_arr, path_mask, structure.width, cell_size,
                    path_px=path_px, widths_m=structure.widths,
                )
            elif structure.type in ("road", "trail") and modify_roads:
                hm_arr = _smooth_road(hm_arr, path_mask, structure.width, cell_size)

        return Depth(hm_arr)
