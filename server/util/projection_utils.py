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
