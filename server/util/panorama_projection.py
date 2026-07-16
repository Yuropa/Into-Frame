import math
import numpy as np


def project_box_to_pano(box, intrinsics, pano_w: int, pano_h: int) -> list[tuple[float, float]]:
    """Project a perspective bounding box onto equirectangular panorama pixel coordinates.

    Assumes the input camera faces lon=0, lat=0 (the panorama's forward direction).
    Returns four (u, v) corner points in panorama pixel space.
    """
    x, y, w, h = box
    corners_img = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    scale_x = intrinsics.color_width / intrinsics.width if intrinsics.width else 1.0
    scale_y = intrinsics.color_height / intrinsics.height if intrinsics.height else 1.0
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    cx = intrinsics.px * scale_x
    cy = intrinsics.py * scale_y

    result = []
    for px, py in corners_img:
        x_cam = (px - cx) / fx
        y_cam = -((py - cy) / fy)  # image +y is down; world +Y is up
        z_cam = 1.0
        n = math.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2)
        x_cam /= n; y_cam /= n; z_cam /= n
        lon = math.atan2(x_cam, z_cam)
        lat = math.atan2(y_cam, math.sqrt(x_cam ** 2 + z_cam ** 2))
        u = (lon / (2.0 * math.pi) + 0.5) * pano_w
        v = (0.5 - lat / math.pi) * pano_h
        result.append((u, v))
    return result


def project_mask_to_pano(mask: np.ndarray, box, intrinsics, pano_w: int, pano_h: int) -> np.ndarray | None:
    """Inverse-project a perspective-image object mask into equirectangular panorama space.

    For every candidate panorama pixel (restricted to the projected box's bounding
    region) this samples the perspective pixel it corresponds to, rather than
    forward-scattering source mask pixels — the latter leaves gaps whenever the
    panorama footprint is larger than the source crop's pixel count.

    box: [x, y, w, h] of the mask's origin in the full perspective photo.
    Returns a (pano_h, pano_w) bool array, or None if there's nothing to project
    (empty box) or the box straddles the panorama's longitude seam (rare given the
    source camera's limited FOV — not worth the complexity of splitting it in two).
    """
    corners = project_box_to_pano(box, intrinsics, pano_w, pano_h)
    us = [c[0] for c in corners]
    vs = [c[1] for c in corners]
    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)

    if u_max - u_min > pano_w / 2:
        return None

    pad = 2
    u0 = max(0, int(math.floor(u_min)) - pad)
    u1 = min(pano_w, int(math.ceil(u_max)) + pad)
    v0 = max(0, int(math.floor(v_min)) - pad)
    v1 = min(pano_h, int(math.ceil(v_max)) + pad)
    if u1 <= u0 or v1 <= v0:
        return None

    scale_x = intrinsics.color_width / intrinsics.width if intrinsics.width else 1.0
    scale_y = intrinsics.color_height / intrinsics.height if intrinsics.height else 1.0
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    cx = intrinsics.px * scale_x
    cy = intrinsics.py * scale_y

    uu, vv = np.meshgrid(np.arange(u0, u1), np.arange(v0, v1))
    lon = (uu.astype(np.float64) / pano_w - 0.5) * (2.0 * math.pi)
    lat = (0.5 - vv.astype(np.float64) / pano_h) * math.pi

    cos_lat = np.cos(lat)
    x_cam = cos_lat * np.sin(lon)
    y_cam = np.sin(lat)
    z_cam = cos_lat * np.cos(lon)

    valid = z_cam > 1e-6
    with np.errstate(invalid="ignore", divide="ignore"):
        px = cx + fx * (x_cam / z_cam)
        py = cy - fy * (y_cam / z_cam)

    box_x, box_y, _, _ = box
    local_x = px - box_x
    local_y = py - box_y
    mh, mw = mask.shape
    in_bounds = valid & (local_x >= 0) & (local_x < mw) & (local_y >= 0) & (local_y < mh)

    result = np.zeros((pano_h, pano_w), dtype=bool)
    if np.any(in_bounds):
        rows, cols = np.nonzero(in_bounds)
        ix = local_x[in_bounds].astype(np.int64)
        iy = local_y[in_bounds].astype(np.int64)
        result[v0 + rows, u0 + cols] = mask[iy, ix]

    return result
