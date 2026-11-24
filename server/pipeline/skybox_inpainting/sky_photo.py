"""Sky from the photograph: inside the photographed cone the sky panorama is the photograph.

Skybox Inpainting builds the whole sky again (FLUX over the gradient composite, or the
clear-sky model), matched to the photographed sky row by row, because a fill that stops
at the skyline draws the skyline back into the sky. What that leaves on the dome inside
the cone is a sky of the right colours and another structure: Shark Fin's soft cloud
bands came back as radial streaks around a sun disc the photograph does not show, and
Paris's cumulus as a cloudless gradient: the first thing a comparison with the
photograph sees, and half of every frame. (The clear-sky model is the photographed
sky's own profile and is left alone: the stage calls this for generated skies only.)

Here the photographed sky goes back over the result where the photograph shows it:
the terrain panorama's pixels that are sky, inside the cone, and unchanged from the
panorama the objects were removed from (where an object stood in front of the sky the
removal's fill is a guess, and it carries ghosts). Around those pixels -- below the
skyline, behind a removed object -- the photographed colours are carried a few degrees
on and faded into the generated sky, so the join lies behind the terrain from the
capture position. Past the cone's left, right and top edges the terrain panorama's sky
is the outpainter's continuation of the photographed one, the same clouds running on:
that is what the composite fades out over (outside_deg), so the photographed structure
ends in its own continuation and not at the edge of the frame (faded inside the cone
over 4 degrees, Shark Fin's bright bands stopped against the generated dusk as a box).
The continuation is first shifted to meet the photographed sky at the cone's edge (the
difference of the two sides' local means there, carried outward): the outpainter does
not match the frame's exposure, and Paris's cone stood in its panorama as a paler
rectangle. Thin dark things the sky mask lets through (the tip of a mast, a wire)
are left to the colours around them.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt

from pipeline.terrain.water_photo import _blur, _carried


def restore_photographed_sky(result: np.ndarray, source: np.ndarray, sky_mask: np.ndarray, cone,
                             original: Optional[np.ndarray] = None, feather_deg: float = 4.0,
                             outside_deg: float = 12.0, erode_deg: float = 0.6, changed_levels: float = 14.0,
                             edge_deg: float = 0.25, speck_levels: float = 22.0, min_fraction: float = 0.002,
                             log: Optional[Callable[[str], None]] = None) -> tuple[np.ndarray, dict]:
    """`result` (H, W, 3 uint8, the generated sky) with the photographed sky of `source`
    (the terrain panorama) composited over it inside `cone` (util.captured_cone.CapturedCone).
    `original` is the panorama before the objects were removed, when there is one of the
    same size. Returns (sky, info); the result unchanged when the cone holds too little sky."""
    H, W = sky_mask.shape
    info: dict = {"applied": False}
    if cone is None or result.shape[:2] != (H, W) or source.shape[:2] != (H, W):
        return result, info
    ppd = W / 360.0
    u0, u1 = cone.u_range
    v0, v1 = cone.v_range
    feather = max(4.0, feather_deg * ppd)
    outside = max(1.0, outside_deg * ppd)
    cols = np.arange(W, dtype=np.float32)[None, :]
    rows = np.arange(H, dtype=np.float32)[:, None]
    # Distance in from the cone's edges (negative outside it).
    edge = np.minimum(np.minimum(cols - u0, u1 - cols), np.minimum(rows - v0, v1 - rows))
    src = source.astype(np.float32)
    # Not the skyline's own pixels: they are half terrain.
    valid = binary_erosion(sky_mask.astype(bool), iterations=max(1, int(round(erode_deg * ppd)))) & (edge > -outside)
    n_sky = int(valid.sum())
    if original is not None and original.shape[:2] == (H, W):
        d = np.abs(_blur(src, 1.5) - _blur(original.astype(np.float32), 1.5)).max(axis=-1)
        changed = binary_dilation(d > changed_levels, iterations=max(1, int(round(0.4 * ppd))))
        valid &= ~changed
        info["removed_px"] = int(n_sky - valid.sum())
    in_cone = float((valid & (edge > 0)).mean())
    info["photographed_sky_fraction"] = round(in_cone, 4)
    if in_cone < min_fraction:
        if log:
            log(f"Photographed sky: {in_cone:.2%} of the panorama inside the cone; the generated sky is kept")
        return result, info
    # Thin dark things standing in the sky: darker than the sky within half a degree of them.
    m = valid.astype(np.float32)
    lum = src @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    local = _blur(lum * m, 0.5 * ppd) / np.maximum(_blur(m, 0.5 * ppd), 1e-3)
    specks = binary_dilation(valid & (local - lum > speck_levels), iterations=max(1, int(round(0.25 * ppd))))
    info["speck_px"] = int((valid & specks).sum())
    valid &= ~specks
    # The frame's own border rows and columns are a resampled edge.
    valid &= np.abs(edge) > edge_deg * ppd
    # The outpainted continuation shifted to meet the photographed sky at the cone's edge.
    photographed = valid & (edge > 0)
    continued = valid & ~photographed
    fallback = np.median(src[photographed][:: max(1, int(photographed.sum()) // 100000)], axis=0)
    if continued.any():
        sig = 1.0 * ppd
        den_p, den_c = _blur(photographed.astype(np.float32), sig), _blur(continued.astype(np.float32), sig)
        band = (den_p > 0.15) & (den_c > 0.15)
        if band.any():
            step = (_blur(src * photographed[..., None], sig) / np.maximum(den_p[..., None], 1e-3)
                    - _blur(src * continued[..., None], sig) / np.maximum(den_c[..., None], 1e-3))
            iy, ix = distance_transform_edt(~band, return_distances=False, return_indices=True)
            step = _blur(step[iy, ix], 2.0 * ppd)
            info["edge_step_levels"] = round(float(np.abs(step[band]).mean()), 1)
            src = np.where(continued[..., None], src + step, src)
    # The photographed colours carried past the pixels that have them: the nearest
    # photographed pixel's within a degree (the strip along the frame's border, a
    # speck's hole: the low frequencies alone drew the border as a faint darker line),
    # the low frequencies further off.
    carried = _carried(src, valid, fallback, 2)
    away, (iy, ix) = distance_transform_edt(~valid, return_indices=True)
    away = away.astype(np.float32)
    t = np.clip(away / ppd, 0.0, 1.0)[..., None]
    photo = np.where(valid[..., None], src, src[iy, ix] * (1.0 - t) + carried * t)
    a = np.clip(_blur(valid.astype(np.float32), 1.5), 0.0, 1.0)[..., None]
    join = 4.0 * a * (1.0 - a)
    photo = photo * (1.0 - join) + _blur(photo, 1.5) * join
    # 1 on the photographed sky, falling to 0 a feather away from it (below the skyline)
    # and over the outpainted continuation past the cone's edges.
    w = np.clip(1.0 - away / feather, 0.0, 1.0) * np.clip(1.0 + edge / outside, 0.0, 1.0)
    w = w * w * (3.0 - 2.0 * w)
    out = result.astype(np.float32) * (1.0 - w[..., None]) + photo * w[..., None]
    info.update({"applied": True, "feather_px": round(float(feather), 1), "weight_mean": round(float(w.mean()), 4),
                 "cone_cols": [int(u0), int(u1)], "cone_rows": [int(v0), int(v1)]})
    if log:
        log(f"Photographed sky back over the generated one inside the cone: {in_cone:.1%} of the panorama "
            f"(columns {int(u0)}-{int(u1)}, rows {int(max(v0, 0))}-{int(v1)}), fading out over {outside / ppd:.0f} deg of the "
            f"outpainted sky around it and {feather / ppd:.0f} deg below the skyline"
            + (f"; {info['removed_px']} px where an object was removed left to the generated sky" if info.get("removed_px") else ""))
    return np.clip(out, 0.0, 255.0).astype(np.uint8), info
