from __future__ import annotations
import numpy as np
import cv2
import PIL.Image
from scipy.ndimage import binary_dilation, binary_erosion


def heal_seam(
    image: PIL.Image.Image,
    mask: np.ndarray,
    band_width_px: int = 40,
    wrap_horizontal: bool = False,
    radius: int = 5,
    method: str = "telea",
) -> PIL.Image.Image:
    """
    Content-aware repair of a band straddling the boundary of `mask` — the
    automated equivalent of dragging Photoshop's Spot Healing Brush along a
    seam. Rather than leave a visible line where two independently generated
    regions meet, the boundary band is treated as a blemish and reconstructed
    from its surroundings via fast-marching inpainting (Telea, 2004) or
    Navier-Stokes diffusion, both of which propagate nearby colour/gradient
    information inward instead of copy-pasting a hard edge.

    image:           source RGB image.
    mask:            boolean array, same H×W as image, marking one side of the
                      boundary (e.g. True = sky). The healed band straddles
                      every edge where this mask transitions.
    band_width_px:   total width of the band to heal, split evenly across
                      both sides of the boundary.
    wrap_horizontal: set True for equirectangular panoramas so healing sees
                      continuous content across the left/right edge instead of
                      treating it as a hard image border.
    radius:          inpainting neighbourhood radius (cv2.inpaint's inpaintRadius).
    method:          "telea" (fast marching, default) or "ns" (Navier-Stokes).
    """
    if not mask.any() or mask.all():
        return image

    half = max(1, band_width_px // 2)
    dilated = binary_dilation(mask, iterations=half)
    eroded = binary_erosion(mask, iterations=half)
    band = dilated & ~eroded

    if not band.any():
        return image

    arr = np.array(image.convert("RGB"))
    band_u8 = (band * 255).astype(np.uint8)
    flag = cv2.INPAINT_NS if method == "ns" else cv2.INPAINT_TELEA

    if wrap_horizontal:
        w = arr.shape[1]
        pad = min(half + radius + 1, w // 2)
        arr_padded = np.concatenate([arr[:, -pad:], arr, arr[:, :pad]], axis=1)
        band_padded = np.concatenate([band_u8[:, -pad:], band_u8, band_u8[:, :pad]], axis=1)
        healed = cv2.inpaint(arr_padded, band_padded, radius, flag)[:, pad:pad + w]
    else:
        healed = cv2.inpaint(arr, band_u8, radius, flag)

    result = arr.copy()
    result[band] = healed[band]
    return PIL.Image.fromarray(result)
