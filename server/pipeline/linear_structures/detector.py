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


def _estimate_width(mask: np.ndarray, paths: list[list[tuple[int, int]]], cell_size: float) -> float:
    """Estimate average structure width from mask area / skeleton length."""
    total_path_px = sum(len(p) for p in paths)
    if total_path_px == 0:
        return 1.0
    area_px = float(mask.sum())
    width_px = area_px / max(1, total_path_px)
    return max(0.5, width_px * cell_size)


def _path_mask_from_polylines(
    paths: list[list[tuple[int, int]]], hm_res: int
) -> np.ndarray:
    mask = np.zeros((hm_res, hm_res), dtype=bool)
    for path in paths:
        for r, c in path:
            if 0 <= r < hm_res and 0 <= c < hm_res:
                mask[r, c] = True
    return mask


def _carve_valley(hm_arr: np.ndarray, path_mask: np.ndarray,
                  width_m: float, cell_size: float) -> np.ndarray:
    """Gaussian valley carved along a river path."""
    dist_px   = distance_transform_edt(~path_mask)
    sigma_px  = max(1.0, (width_m / 2.0) / cell_size)
    depth_m   = max(0.2, width_m * 0.15)
    influence = np.exp(-dist_px ** 2 / (2.0 * sigma_px ** 2))
    return hm_arr - depth_m * influence.astype(np.float32)


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
    ) -> LinearGraph:
        """
        Build a LinearGraph from pre-computed semantic skeleton masks.

        water_skeleton: binary skeleton of water bodies in height-map grid space
                        (from RegionMapStage via ContextKey.WATER_SKELETON).
        water_mask:     full water area mask (same grid) used for width estimation.
                        Falls back to the skeleton itself if not provided.

        Roads/trails: no reliable semantic source yet — skipped until a road
        segmentation stage is available.
        """
        grid_size = params.get("grid_size_meters", 100.0)
        hm_res    = int(height_map.depth.shape[0])
        cell_size = grid_size / hm_res
        hm_arr    = height_map.depth.copy()

        graph = LinearGraph()

        if water_skeleton is not None and water_skeleton.any():
            skel  = water_skeleton > 0
            paths = _skeleton_to_polylines(skel)
            if paths:
                area_mask   = water_mask if water_mask is not None else skel
                width_m     = _estimate_width(area_mask, paths, cell_size)
                world_paths = _pixel_paths_to_world(paths, hm_arr, grid_size, hm_res)
                for wp in world_paths:
                    if len(wp) >= 2:
                        graph.add(LinearStructure(type="river", path=wp, width=width_m))

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
                hm_arr = _carve_valley(hm_arr, path_mask, structure.width, cell_size)
            elif structure.type in ("road", "trail") and modify_roads:
                hm_arr = _smooth_road(hm_arr, path_mask, structure.width, cell_size)

        return Depth(hm_arr)
