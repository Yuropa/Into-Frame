"""
Linear structure detection from panorama colour segmentation and height map
topology.  Produces a LinearGraph of roads, rivers, and trails in world space,
and applies terrain modifications (valley carving, road smoothing) to the
height map array.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.ndimage import (
    distance_transform_edt,
    gaussian_filter,
    uniform_filter,
    binary_closing,
    convolve,
)
from skimage.morphology import skeletonize, remove_small_objects
from skimage.measure import label as sk_label
from typing import Optional

from util.depth_utils import Depth
from util.panorama_utils import Panorama
from pipeline.linear_structures.graph import LinearGraph, LinearStructure


# ── Colour ranges (OpenCV HSV: H [0-179], S [0-255], V [0-255]) ─────────────
# Each type lists one or more inclusive (lo, hi) ranges as dicts.

_COLOR_RANGES: dict[str, list[dict]] = {
    "road": [
        # asphalt / concrete — low saturation, mid brightness
        {"h_lo": 0,   "h_hi": 179, "s_lo": 0,   "s_hi": 50,  "v_lo": 40,  "v_hi": 200},
    ],
    "river": [
        # blue/teal open water
        {"h_lo": 88,  "h_hi": 135, "s_lo": 40,  "s_hi": 255, "v_lo": 20,  "v_hi": 220},
        # dark / murky water
        {"h_lo": 0,   "h_hi": 179, "s_lo": 0,   "s_hi": 80,  "v_lo": 0,   "v_hi": 55},
    ],
    "trail": [
        # bare earth / dirt paths
        {"h_lo": 5,   "h_hi": 25,  "s_lo": 30,  "s_hi": 150, "v_lo": 50,  "v_hi": 190},
    ],
}

# Minimum projected area (height-map pixels) to keep a detected region.
_MIN_AREA_PX = 80

# When skeletonising: only keep paths longer than this many pixels.
_MIN_PATH_PX = 12


def _apply_color_range(hsv: np.ndarray, ranges: list[dict]) -> np.ndarray:
    """Union of all colour ranges for one type → binary mask."""
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    for r in ranges:
        in_range = (
            (hsv[:, :, 0] >= r["h_lo"]) & (hsv[:, :, 0] <= r["h_hi"]) &
            (hsv[:, :, 1] >= r["s_lo"]) & (hsv[:, :, 1] <= r["s_hi"]) &
            (hsv[:, :, 2] >= r["v_lo"]) & (hsv[:, :, 2] <= r["v_hi"])
        )
        mask |= in_range
    return mask


def _panorama_ground_type_masks(
    panorama: Panorama,
    work_w: int = 1024,
) -> dict[str, np.ndarray]:
    """
    Colour-classify ground-facing panorama pixels.

    Returns a dict {type → bool mask} at the resized panorama resolution.
    Only pixels below the horizon (phi < 0) are considered.
    """
    work_h = work_w // 2
    pil_img = panorama.rgb().resize((work_w, work_h))
    rgb = np.array(pil_img, dtype=np.uint8)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # Ground-facing pixels: phi = (0.5 - row/H) * π < 0  →  row/H > 0.5
    row_idx = np.arange(work_h, dtype=np.float32)
    phi = (0.5 - row_idx / work_h) * np.pi
    ground_rows = phi < 0.0  # shape (H,)
    ground_mask = np.broadcast_to(ground_rows[:, None], (work_h, work_w)).copy()

    masks: dict[str, np.ndarray] = {}
    for stype, ranges in _COLOR_RANGES.items():
        m = _apply_color_range(hsv, ranges) & ground_mask
        masks[stype] = m

    return masks, work_w, work_h


def _project_masks_to_hm(
    pano_masks: dict[str, np.ndarray],
    pano_w: int,
    pano_h: int,
    pano_depth: Depth,
    grid_size: float,
    hm_res: int,
) -> dict[str, np.ndarray]:
    """
    Project panorama type masks into height-map grid coordinates.

    Returns {type → float accumulation map} in height-map space.
    """
    # Build pixel coordinate arrays for the panorama at working resolution
    depth_arr = np.array(
        pano_depth.depth  # use actual depth at native res
    )
    # Resize depth to working resolution
    depth_resized = cv2.resize(
        depth_arr,
        (pano_w, pano_h),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)

    col_idx = np.arange(pano_w, dtype=np.float32)
    row_idx = np.arange(pano_h, dtype=np.float32)
    px, py = np.meshgrid(col_idx, row_idx)

    theta = (px / pano_w - 0.5) * 2.0 * np.pi
    phi   = (0.5 - py / pano_h) * np.pi

    # Valid ground pixels with positive finite depth
    valid = (phi < 0.0) & (depth_resized > 0.1) & np.isfinite(depth_resized)

    d_v     = depth_resized[valid]
    theta_v = theta[valid]
    phi_v   = phi[valid]

    cos_phi = np.cos(phi_v)
    world_x = (d_v * cos_phi * np.sin(theta_v)).astype(np.float32)
    world_z = (d_v * cos_phi * np.cos(theta_v)).astype(np.float32)

    x_half = z_far = grid_size / 2.0
    hm_col = ((world_x + x_half) / grid_size * (hm_res - 1)).astype(np.int32)
    hm_row = ((world_z + z_far)  / (2.0 * z_far) * (hm_res - 1)).astype(np.int32)

    in_bounds = (
        (hm_col >= 0) & (hm_col < hm_res) &
        (hm_row >= 0) & (hm_row < hm_res)
    )
    hm_col = hm_col[in_bounds]
    hm_row = hm_row[in_bounds]

    projected: dict[str, np.ndarray] = {}
    for stype, mask in pano_masks.items():
        pano_flat = mask.ravel()
        valid_flat = np.flatnonzero(valid)
        type_hit = pano_flat[valid_flat][in_bounds]

        acc = np.zeros((hm_res, hm_res), dtype=np.float32)
        np.add.at(acc, (hm_row[type_hit], hm_col[type_hit]), 1.0)
        projected[stype] = acc

    return projected


def _detect_valley_mask(hm_arr: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """
    Detect valley pixels from the height map.

    A pixel is a valley if it is significantly lower than its neighbourhood
    mean — a simple proxy for drainage channels and river beds.
    """
    smooth     = gaussian_filter(hm_arr, sigma=sigma)
    hood_mean  = uniform_filter(smooth, size=15)
    return (smooth < hood_mean - 0.25).astype(bool)


def _clean_and_binarise(acc: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold + morphological close + small-object removal."""
    binary = acc > threshold
    binary = binary_closing(binary, structure=np.ones((5, 5), dtype=bool))
    return remove_small_objects(binary, min_size=_MIN_AREA_PX)


def _neighbour_count_image(skel: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    kernel[1, 1] = 0
    return convolve(skel.astype(np.uint8), kernel, mode="constant", cval=0)


def _skeleton_to_polylines(
    skel: np.ndarray,
    min_length: int = _MIN_PATH_PX,
) -> list[list[tuple[int, int]]]:
    """
    Convert a skeleton image into a list of ordered pixel paths.

    Strategy:
      1. Remove junction pixels to split the skeleton at branches.
      2. Label fragments.
      3. Trace each fragment from one endpoint to the other.
    """
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

        # Find an endpoint (≤1 fragment-internal neighbour)
        frag_nbr = {
            pt: sum(1 for n in pixel_neighbors(*pt) if n in frag_pts)
            for pt in frag_pts
        }
        endpoints = [pt for pt, cnt in frag_nbr.items() if cnt <= 1]
        start = endpoints[0] if endpoints else next(iter(frag_pts))

        # Walk the fragment
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


# ── Terrain modification ──────────────────────────────────────────────────────

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


# ── Public API ────────────────────────────────────────────────────────────────

class LinearStructureDetector:

    @staticmethod
    def detect(
        height_map: Depth,
        params: dict,
        panorama: Optional[Panorama] = None,
        panorama_depth: Optional[Depth] = None,
    ) -> LinearGraph:
        """
        Detect linear structures and return a LinearGraph.

        Uses panorama colour segmentation when available, supplemented by
        height-map valley detection for rivers.
        """
        grid_size = params.get("grid_size_meters", 100.0)
        hm_res    = int(height_map.depth.shape[0])
        cell_size = grid_size / hm_res
        hm_arr    = height_map.depth.copy()

        # ── Source masks in height-map space ─────────────────────────────────
        hm_masks: dict[str, np.ndarray] = {}

        if panorama is not None and panorama_depth is not None:
            pano_type_masks, pw, ph = _panorama_ground_type_masks(panorama)
            projected = _project_masks_to_hm(
                pano_type_masks, pw, ph, panorama_depth, grid_size, hm_res,
            )
            for stype, acc in projected.items():
                hm_masks[stype] = gaussian_filter(acc, sigma=2.0)

        # Topology-based river fallback (always computed, merged with visual)
        valley = _detect_valley_mask(hm_arr)
        river_acc = valley.astype(np.float32)
        if "river" in hm_masks:
            hm_masks["river"] = hm_masks["river"] + river_acc
        else:
            hm_masks["river"] = river_acc

        # ── Build graph ───────────────────────────────────────────────────────
        graph = LinearGraph()

        for stype, acc in hm_masks.items():
            binary = _clean_and_binarise(acc)
            if not binary.any():
                continue

            skel   = skeletonize(binary)
            paths  = _skeleton_to_polylines(skel)
            if not paths:
                continue

            width_m      = _estimate_width(binary, paths, cell_size)
            world_paths  = _pixel_paths_to_world(paths, hm_arr, grid_size, hm_res)

            for wp in world_paths:
                if len(wp) >= 2:
                    graph.add(LinearStructure(type=stype, path=wp, width=width_m))

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
            path_px  = list(zip(rows.tolist(), cols.tolist()))
            path_mask = _path_mask_from_polylines([path_px], hm_res)

            if structure.type == "river" and modify_rivers:
                hm_arr = _carve_valley(hm_arr, path_mask, structure.width, cell_size)
            elif structure.type in ("road", "trail") and modify_roads:
                hm_arr = _smooth_road(hm_arr, path_mask, structure.width, cell_size)

        return Depth(hm_arr)
