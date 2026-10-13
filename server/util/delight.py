"""
Chroma-preserving delighting.

IntrinsicDiffusion predicts an albedo whose colour is low in chroma: the model's
reflectance for a red paintbrush bloom comes back pink-grey, a yellow lupine as
beige. Blending the photograph linearly toward that albedo (the previous
`intrinsic_delight_strength` behaviour) therefore removed most of the COLOUR along
with the shading, which is the washed-out look every delit card and mesh texture
had on the Rainier run.

What delighting is for is removing the low-frequency SHADING -- the sun side vs
the shadow side, the vignette, the cast shadow across a flower -- so the client
can light the surface again. Shading is a scalar multiplier on reflectance, and
the model is far more reliable about that scalar than about hue. So this takes
only the shading from the model:

    shading  = L(photo) / L(albedo)         (per pixel, then low-passed)
    delit    = photo / (shading / median)^strength

The ratio is normalised by its median so the overall exposure is unchanged, and
low-passed so the model's resolution (it runs at 768 px) and any hue error do not
matter: a blurred scalar field cannot change saturation, only brightness. Every
channel is divided by the same number, so hue and chroma are exactly the
photograph's.
"""
from __future__ import annotations

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import gaussian_filter

_LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def delight_preserving_chroma(
    photo: PILImage.Image,
    albedo: PILImage.Image,
    strength: float = 0.6,
    blur_fraction: float = 0.02,
    max_gain: float = 3.0,
    mask: "np.ndarray | None" = None,
) -> PILImage.Image:
    """Divide the low-frequency shading out of `photo`, keeping its colour.

    photo:         RGB or RGBA; alpha (if any) is returned untouched.
    albedo:        the model's albedo prediction (any size; resized to photo).
    strength:      0 = return the photo, 1 = divide the full estimated shading out.
    blur_fraction: low-pass sigma as a fraction of the image's longer side, so the
                   correction describes illumination, not texture.
    max_gain:      clamp on the per-pixel multiplier in either direction; a shadow
                   the model calls pitch black must not become a 20x brightening.
    mask:          optional (H, W) bool; the median is taken over these pixels only
                   (e.g. the cutout's opaque pixels), so transparent padding does
                   not set the exposure reference.
    """
    if strength <= 0.0:
        return photo
    rgba = photo.mode == "RGBA"
    src = np.asarray(photo.convert("RGBA" if rgba else "RGB"), dtype=np.float32)
    rgb = src[..., :3] / 255.0
    alb = np.asarray(albedo.convert("RGB").resize(photo.size, PILImage.BILINEAR), dtype=np.float32) / 255.0

    # Work in roughly linear light so the ratio is a physical shading estimate.
    lum_photo = (rgb ** 2.2) @ _LUMA
    lum_albedo = (alb ** 2.2) @ _LUMA
    shading = lum_photo / np.maximum(lum_albedo, 1e-3)

    h, w = shading.shape
    sigma = max(1.0, blur_fraction * max(h, w))
    if mask is not None and mask.shape == shading.shape and mask.any():
        # Normalised convolution so transparent padding does not bleed in.
        m = mask.astype(np.float32)
        num = gaussian_filter(shading * m, sigma=sigma)
        den = gaussian_filter(m, sigma=sigma)
        smooth = num / np.maximum(den, 1e-4)
        reference = float(np.median(shading[mask]))
    else:
        smooth = gaussian_filter(shading, sigma=sigma)
        reference = float(np.median(shading))
    if not np.isfinite(reference) or reference <= 1e-6:
        return photo

    gain = (reference / np.maximum(smooth, 1e-6)) ** float(strength)
    gain = np.clip(gain, 1.0 / max_gain, max_gain)[..., None]

    out = np.clip((rgb ** 2.2) * gain, 0.0, 1.0) ** (1.0 / 2.2)
    out8 = (out * 255.0 + 0.5).astype(np.uint8)
    if rgba:
        return PILImage.fromarray(np.dstack([out8, src[..., 3].astype(np.uint8)]), "RGBA")
    return PILImage.fromarray(out8, "RGB")
