"""
Pitch rotation of an equirectangular panorama.

The panorama generator embeds the photograph with its centre on the horizon row,
whatever the camera's pitch was. A photograph taken looking down (Shark Fin: a
cliff-top view, sea horizon 5 deg above the frame's centre) therefore comes out
with the real horizon 5 deg above the panorama's, and everything built from the
panorama inherits it: sea on the far shell above eye level ("water in the sky"),
a skybox whose horizon sits too high. Rotating the panorama about the camera's
right axis by the measured pitch puts the real horizon on the panorama's and is
a true rigid rotation of the environment, so every map derived from it afterwards
is consistent; only the photo's placement inside it moves (see
util.captured_cone.CapturedCone.pitch_deg).

Conventions (pano_utils / calibration): x right, y DOWN, z forward; equirect
u = atan2(x, z) / 2pi + 0.5, v = asin(y) / pi + 0.5. pitch_deg > 0 means the camera
looked DOWN by that much.
"""
from __future__ import annotations

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import map_coordinates


def _equirect_dirs(h: int, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    lon = (u - 0.5) * 2.0 * np.pi
    lat = (v - 0.5) * np.pi
    LON, LAT = np.meshgrid(lon, lat)
    return np.cos(LAT) * np.sin(LON), np.sin(LAT), np.cos(LAT) * np.cos(LON)


def rotate_equirect_pitch(image, pitch_deg: float, order: int = 1):
    """`image` (PIL or HxWxC / HxW array) rotated so that what sat `pitch_deg`
    above the panorama's horizon now sits on it. Returns the same kind as given."""
    pil = isinstance(image, PILImage.Image)
    arr = np.asarray(image.convert("RGB") if pil else image)
    h, w = arr.shape[:2]
    th = np.radians(float(pitch_deg))
    x, y, z = _equirect_dirs(h, w)
    # Output (true) direction -> input (generated) direction: the generator's
    # frame is the true frame rotated UP by the pitch.
    y_in = y * np.cos(th) - z * np.sin(th)
    z_in = y * np.sin(th) + z * np.cos(th)
    u_in = (np.arctan2(x, z_in) / (2.0 * np.pi) + 0.5) * w - 0.5
    v_in = (np.arcsin(np.clip(y_in, -1.0, 1.0)) / np.pi + 0.5) * h - 0.5
    if arr.ndim == 2:
        out = map_coordinates(arr.astype(np.float32), [v_in, u_in], order=order, mode="wrap")
    else:
        out = np.stack([
            map_coordinates(arr[..., c].astype(np.float32), [v_in, u_in], order=order, mode="wrap")
            for c in range(arr.shape[2])
        ], axis=-1)
    if arr.dtype == np.uint8:
        out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return PILImage.fromarray(out) if pil else out.astype(arr.dtype, copy=False)
