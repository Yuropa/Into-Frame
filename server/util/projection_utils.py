"""
Utilities for projecting equirectangular panorama pixels onto the ground plane.

Both the height map and region map generators project a 360° panorama into a
top-down XZ grid using the same geometry, so the common parts live here.
"""

import numpy as np
from typing import Optional

from util.depth_utils import Depth
from util.panorama_utils import Panorama


def ground_projection_certainty(
    X: np.ndarray,
    Z: np.ndarray,
    camera_height: float,
) -> np.ndarray:
    """
    Certainty of back-projecting from an equirectangular panorama onto the ground plane.

    For a ground point at horizontal distance r = sqrt(X² + Z²) from the camera at
    height h, the depression angle is phi = arctan(h / r).  The equirectangular
    Jacobian (ground area covered per panorama pixel) scales as 1/sin³(phi), so we use
    sin²(phi) = h² / (r² + h²) as the certainty — it is 1 directly below the camera
    and falls toward 0 as the point approaches the horizon.

    This is purely geometric: it captures how much distortion the equirectangular
    backward-projection introduces at each ground location, independent of whether
    any observation actually landed there (callers should zero out unobserved cells).

    Returns a float32 array with the same shape as X and Z, values in (0, 1].
    """
    r_sq = np.asarray(X, dtype=np.float64) ** 2 + np.asarray(Z, dtype=np.float64) ** 2
    h_sq = float(camera_height) ** 2
    return (h_sq / (r_sq + h_sq)).astype(np.float32)


def equirectangular_pixels_to_world(
    rows: np.ndarray,
    cols: np.ndarray,
    depths: np.ndarray,
    pano_h: int,
    pano_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert sparse equirectangular pixel coordinates + radial depth to world XYZ.

    Useful when working with a selected subset of pixels (e.g. a silhouette row per
    column, or detected edge pixels) rather than the full depth image.

    rows, cols, depths: 1-D arrays of matching length.
    Returns (X, Y, Z) float32 arrays in the same camera-relative convention as
    equirectangular_unproject (theta=0 at +Z, +Y up).
    """
    theta   = (np.asarray(cols, dtype=np.float64) / pano_w - 0.5) * 2.0 * np.pi
    phi     = (0.5 - np.asarray(rows, dtype=np.float64) / pano_h) * np.pi
    cos_phi = np.cos(phi)
    d       = np.asarray(depths, dtype=np.float64)
    X = (d * cos_phi * np.sin(theta)).astype(np.float32)
    Y = (d * np.sin(phi)).astype(np.float32)
    Z = (d * cos_phi * np.cos(theta)).astype(np.float32)
    return X, Y, Z


def bilinear_sample_grid(
    img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Bilinearly sample a 2-D float array at fractional pixel coordinates (u, v).
    Wraps horizontally (equirectangular panorama convention); clamps vertically.
    img: (H, W) float32-compatible.  u, v: arbitrary matching shape.
    Returns float32 array with the same shape as u/v.
    """
    h, w = img.shape[:2]
    u0 = np.floor(u).astype(np.int32)
    v0 = np.floor(v).astype(np.int32)
    fu = (u - np.floor(u)).astype(np.float32)
    fv = (v - np.floor(v)).astype(np.float32)
    u0w = u0 % w
    u1w = (u0 + 1) % w
    v0c = np.clip(v0, 0, h - 1)
    v1c = np.clip(v0 + 1, 0, h - 1)
    return (
        img[v0c, u0w] * (1 - fu) * (1 - fv)
        + img[v0c, u1w] * fu * (1 - fv)
        + img[v1c, u0w] * (1 - fu) * fv
        + img[v1c, u1w] * fu * fv
    ).astype(np.float32)


def nearest_sample_grid(
    img: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    """
    Nearest-neighbour sample a 2-D array at pixel coordinates (u, v).
    Wraps horizontally; clamps vertically.
    """
    h, w = img.shape[:2]
    ui = np.round(u).astype(np.int32) % w
    vi = np.clip(np.round(v).astype(np.int32), 0, h - 1)
    return img[vi, ui]


def inverse_map_panorama_to_grid(
    panorama_depth: "Depth",
    grid_size_meters: float,
    grid_resolution: int,
    camera_height_meters: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Inverse mapping from terrain grid cells to equirectangular panorama pixels.

    For each cell in a (grid_resolution × grid_resolution) top-down grid, computes
    the panorama column/row (u, v) that would observe a flat ground point at that
    cell's (X, Z) centre from a camera at camera_height_meters above ground, then
    bilinearly samples the radial depth there.

    This is the inverse of forward projection: instead of scattering panorama pixels
    outward onto an unorganised point cloud, every grid cell gets exactly one clean
    depth lookup, eliminating the spatial stretching that affects near-horizon pixels.

    Grid layout: rows index Z (near → far), cols index X (left → right).

    Returns (sampled_depth, pano_u, pano_v, X_grid, Z_grid), each (G, G) float32:
      sampled_depth — bilinear depth sample at each cell's panorama pixel (metres).
      pano_u        — fractional panorama column for each cell.
      pano_v        — fractional panorama row for each cell.
      X_grid        — world X at each cell centre.
      Z_grid        — world Z at each cell centre.
    """
    half = grid_size_meters / 2.0
    cell_m = grid_size_meters / grid_resolution
    x_c = np.linspace(-half + cell_m / 2.0, half - cell_m / 2.0, grid_resolution)
    z_c = np.linspace(-half + cell_m / 2.0, half - cell_m / 2.0, grid_resolution)
    X_grid, Z_grid = np.meshgrid(x_c, z_c)  # rows = Z, cols = X

    r_safe = np.sqrt(X_grid ** 2 + Z_grid ** 2)
    np.maximum(r_safe, 1e-6, out=r_safe)

    # Azimuth: 0 at +Z (forward), matches equirectangular_unproject convention.
    theta = np.arctan2(X_grid, Z_grid)
    # Depression angle: negative = below horizon, magnitude = atan(h / r).
    phi = -np.arctan2(camera_height_meters, r_safe)

    d = panorama_depth.depth.astype(np.float32)
    pano_h, pano_w = d.shape

    pano_u = ((theta / (2.0 * np.pi)) + 0.5) * pano_w  # col in [0, W]
    pano_v = (0.5 - phi / np.pi) * pano_h               # row in [H/2, H]

    sampled = bilinear_sample_grid(d, pano_u, pano_v)
    return (
        sampled.astype(np.float32),
        pano_u.astype(np.float32),
        pano_v.astype(np.float32),
        X_grid.astype(np.float32),
        Z_grid.astype(np.float32),
    )


def project_panorama_to_ground_grid(
    panorama_depth: Depth,
    grid_size_meters: float,
    grid_resolution: int,
    ground_y_max: float,
    ground_y_min: float,
    sky_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Project an equirectangular depth map onto a top-down XZ ground grid.

    Applies sky masking, unprojection, ground-plane filtering, and digitisation
    into grid bins.  Returns only the pixels that fall within the grid bounds.

    Returns (xi, zi, Xg, Yg, Zg):
      xi, zi — integer grid column / row indices, shape (N,)
      Xg, Yg, Zg — world-space coordinates of those pixels, shape (N,)
    """
    d = panorama_depth.depth.astype(np.float32)
    if sky_mask is not None and sky_mask.shape == d.shape:
        d = d.copy()
        d[sky_mask] = np.nan

    X, Y, Z = Panorama.equirectangular_unproject(Depth(d))

    half = grid_size_meters / 2.0
    ground_mask = (
        (Y <= ground_y_max)
        & (Y >= ground_y_min)
        & (np.abs(X) <= half)
        & (np.abs(Z) <= half)
        & np.isfinite(d)
    )

    Xg = X[ground_mask]
    Yg = Y[ground_mask]
    Zg = Z[ground_mask]

    x_edges = np.linspace(-half, half, grid_resolution + 1)
    z_edges = np.linspace(-half, half, grid_resolution + 1)

    xi = np.digitize(Xg, x_edges) - 1
    zi = np.digitize(Zg, z_edges) - 1

    in_bounds = (
        (xi >= 0) & (xi < grid_resolution)
        & (zi >= 0) & (zi < grid_resolution)
    )
    return xi[in_bounds], zi[in_bounds], Xg[in_bounds], Yg[in_bounds], Zg[in_bounds]
