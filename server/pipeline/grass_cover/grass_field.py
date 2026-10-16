"""
The grass as a field the client grows blades from, rather than as instances.

Grass Cover used to place ~6,000 crossed-card tufts, each a GameObject, each a
photograph of a patch of meadow cut out against its neighbours. Up close they read
as flat cards with grey fans, they never sat on the ground, and the budget capped
how dense the meadow could be.

This describes the same grass as two images over a square around the camera, in
the terrain grid's own frame (+X = grid columns, +Z = grid rows, centred on the
camera):

  density_height  R = blade density 0..1 (the grass area, weighted by greenness)
                  G = blade height as a fraction of max_height_m
  colour          RGB = the photograph's colour of the ground there (delit when
                  available), which the client uses as the blade colour

and the client's GrassField grows instanced blade tufts from it around the viewer.
"""
from __future__ import annotations

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter, map_coordinates


def build_grass_field(
    mask: np.ndarray,
    greenness: "np.ndarray | None",
    pano_u: np.ndarray,
    pano_v: np.ndarray,
    sampled: np.ndarray,
    panorama_rgb: np.ndarray,
    grid_size_m: float,
    radius_m: float,
    field_px: int,
    min_greenness: float,
    full_greenness: float,
    floor: float,
    height_jitter: float,
    seed: int,
    local_colour_weight: float = 0.35,
) -> tuple[PILImage.Image, PILImage.Image, dict]:
    res = mask.shape[0]
    cell_m = grid_size_m / res
    # Field pixel centres in grid cell coordinates (row = z, col = x).
    t = (np.arange(field_px) + 0.5) / field_px * (2.0 * radius_m) - radius_m
    coords = (t + grid_size_m / 2.0) / cell_m - 0.5
    rows, cols = np.meshgrid(coords, coords, indexing="ij")
    sample = lambda arr, order=1: map_coordinates(np.asarray(arr, dtype=np.float32), [rows, cols], order=order, mode="nearest")

    density = sample(mask.astype(np.float32))
    if greenness is not None:
        g = np.where(np.isfinite(greenness), greenness, min_greenness)
        ramp = np.clip((sample(g) - min_greenness) / max(full_greenness - min_greenness, 1e-6), floor, 1.0)
        density *= ramp
    density = gaussian_filter(density, sigma=1.0).clip(0.0, 1.0)

    # Height: lower where the meadow is sparse, plus a slow random field so the
    # meadow undulates rather than being trimmed level.
    rng = np.random.default_rng(seed)
    wobble = gaussian_filter(rng.standard_normal((field_px, field_px)), sigma=field_px / 24.0)
    wobble = wobble / max(float(np.abs(wobble).max()), 1e-6)
    height = (0.55 + 0.45 * density) * (1.0 + height_jitter * wobble)
    height = height.clip(0.05, 1.0)

    # Colour: the panorama pixel each cell sees, with no-observation cells filled
    # from the nearest observed colour (blurred) so the field has no holes.
    u, v = sample(pano_u), sample(pano_v)
    ok = sample(sampled.astype(np.float32), order=0) > 0.5
    h, w = panorama_rgb.shape[:2]
    pr = np.clip(((1.0 - v) * (h - 1)).astype(np.intp), 0, h - 1)
    pc = np.clip((u * (w - 1)).astype(np.intp), 0, w - 1)
    colour = panorama_rgb[pr, pc, :3].astype(np.float32)
    if ok.any():
        mean = colour[ok].mean(axis=0)
        colour[~ok] = mean
        blurred = np.stack([gaussian_filter(colour[..., k], sigma=2.0) for k in range(3)], axis=-1)
        colour = np.where(ok[..., None], 0.7 * colour + 0.3 * blurred, blurred)

    # Anchor on the meadow's own median colour and let the local reading vary it
    # only partly: the panorama reprojected onto the ground smears radially (one
    # pixel stretched over metres), and a snow patch or a dry streak beside the
    # grass would otherwise colour whole runs of blades.
    grassy = density > 0.1
    if grassy.any():
        median = np.median(colour[grassy], axis=0)
        colour = median + local_colour_weight * (colour - median)

    density_height = np.stack([density, height, np.zeros_like(density)], axis=-1)
    dh_img = PILImage.fromarray((density_height * 255).round().astype(np.uint8), "RGB")
    colour_img = PILImage.fromarray(colour.clip(0, 255).astype(np.uint8), "RGB")
    stats = {
        "covered_fraction": round(float((density > 0.05).mean()), 3),
        "mean_density": round(float(density.mean()), 3),
    }
    return dh_img, colour_img, stats
