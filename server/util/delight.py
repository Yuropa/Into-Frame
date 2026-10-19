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
    chroma_strength: float = 0.0,
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

    lin = (rgb ** 2.2) * gain

    # The light's own colour. Dividing out a scalar shading leaves a low sun's
    # orange on every lit face and the sky's blue in every shadow (Shark Fin's
    # cliffs stayed sunset-orange after delighting, then were lit orange again
    # by the client's sun). The illuminant chroma is estimated from the image
    # itself: the mean colour of the brightest-shaded third over the mean colour
    # of the darkest third, which is lit-over-shadow for the same materials and
    # cancels the material colour. Lit pixels are divided by it in proportion to
    # how lit they are, chroma_strength deep (0 = keep the photograph's colour).
    # Off by default: on Shark Fin's sunset cliff the shadowed face is proportionally
    # REDDER than the lit one (warm sky ambient, the camera's tone curve), so the
    # estimate came out greenish and would have tinted every lit face. A real
    # intrinsic model is the way to take the sun's colour out; this is not it.
    if chroma_strength > 0.0:
        valid = mask if (mask is not None and mask.shape == shading.shape and mask.any()) else np.ones_like(shading, bool)
        s_lo, s_hi = np.percentile(smooth[valid], [33.0, 67.0])
        lit, dark = valid & (smooth >= s_hi), valid & (smooth <= s_lo)
        if lit.sum() >= 50 and dark.sum() >= 50 and s_hi > 1.15 * max(s_lo, 1e-6):
            lin_src = rgb ** 2.2
            tint = lin_src[lit].mean(axis=0) / np.maximum(lin_src[dark].mean(axis=0), 1e-4)
            tint = np.clip(tint / max(float(tint.max()), 1e-6), 0.25, 1.0)
            lit_weight = np.clip((smooth - s_lo) / max(s_hi - s_lo, 1e-6), 0.0, 1.0)[..., None]
            lin = lin / (tint[None, None, :] ** (float(chroma_strength) * lit_weight))

    out = np.clip(lin, 0.0, 1.0) ** (1.0 / 2.2)
    out8 = (out * 255.0 + 0.5).astype(np.uint8)
    if rgba:
        return PILImage.fromarray(np.dstack([out8, src[..., 3].astype(np.uint8)]), "RGBA")
    return PILImage.fromarray(out8, "RGB")
