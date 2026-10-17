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
    anchor_u_range: "tuple[float, float] | None" = None,
    prior_mask: "np.ndarray | None" = None,
    anchor_colour: "np.ndarray | None" = None,
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
    # The median is taken over the cells the PHOTOGRAPH sees when there are
    # enough of them: the outpainted ring is murkier than the capture (Rainier:
    # field median (47, 61, 45) against the photo's meadow at about (64, 81, 54)).
    grassy = density > 0.1
    if anchor_u_range is not None:
        in_photo = grassy & ok & (u >= anchor_u_range[0]) & (u <= anchor_u_range[1])
        if in_photo.sum() >= 64:
            grassy = in_photo
            # The nearest few metres of it: the meadow the viewer stands in, not
            # the shaded one 20 m off (Rainier: (62, 81, 40) at the photo's foot,
            # (41, 56, 41) across the mid meadow in evening shade).
            fr = np.hypot(*np.meshgrid(t, t, indexing="ij"))
            r_near = float(np.percentile(fr[in_photo], 5.0))
            near = in_photo & (fr <= r_near + 4.0)
            if near.sum() >= 64:
                grassy = near
    if grassy.any():
        # The meadow's lit green, not its plain median: a flowering meadow's
        # median is pulled down and toward blue by the shadowed stems and the
        # blooms between the blades (Rainier: (48, 61, 43) against the photo's
        # green-dominant (62, 81, 40) and its sunlit blades near (74, 92, 51)).
        # Green-dominant cells only, then the 55-75th percentile brightness band.
        pick = colour[grassy]
        green = (pick[:, 1] > pick[:, 0]) & (pick[:, 1] > pick[:, 2])
        if green.sum() >= 32:
            pick = pick[green]
        lum = pick.mean(axis=1)
        lo, hi = np.percentile(lum, [55.0, 75.0])
        band = pick[(lum >= lo) & (lum <= hi)]
        median = np.median(band if len(band) else pick, axis=0)
        if anchor_colour is not None:
            median = np.asarray(anchor_colour, dtype=np.float32)
        weight = np.full(colour.shape[:2], local_colour_weight, dtype=np.float32)
        if prior_mask is not None:
            # Cells that follow the photo's near ground (grass_area) show the
            # generator's ground, not grass: they take the anchor colour alone.
            weight *= 1.0 - gaussian_filter(sample(prior_mask.astype(np.float32), order=0), sigma=2.0)
        colour = median + weight[..., None] * (colour - median)

    density_height = np.stack([density, height, np.zeros_like(density)], axis=-1)
    dh_img = PILImage.fromarray((density_height * 255).round().astype(np.uint8), "RGB")
    colour_img = PILImage.fromarray(colour.clip(0, 255).astype(np.uint8), "RGB")
    stats = {
        "covered_fraction": round(float((density > 0.05).mean()), 3),
        "mean_density": round(float(density.mean()), 3),
    }
    return dh_img, colour_img, stats


def photo_meadow_colour(photo_rgb: np.ndarray, bottom_fraction: float = 0.3) -> "np.ndarray | None":
    """The sunlit green of the meadow at the photograph's foot: green-dominant
    pixels of its bottom rows, their 55-75th percentile brightness band. The
    ground the viewer stands in, read from the one image that really shows it
    (the panorama's matching rows blend into generated ground)."""
    a = np.asarray(photo_rgb, dtype=np.float32)[..., :3]
    b = a[int(a.shape[0] * (1.0 - bottom_fraction)):].reshape(-1, 3)
    green = (b[:, 1] > b[:, 0]) & (b[:, 1] > b[:, 2])
    if green.mean() < 0.3:
        return None
    g = b[green]
    lum = g.mean(axis=1)
    lo, hi = np.percentile(lum, [55.0, 75.0])
    band = g[(lum >= lo) & (lum <= hi)]
    return np.median(band if len(band) else g, axis=0)
