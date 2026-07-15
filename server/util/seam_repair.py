from __future__ import annotations
from pathlib import Path
from typing import Callable
import numpy as np
import cv2
import PIL.Image
import PIL.ImageFilter
from scipy.ndimage import binary_dilation, binary_erosion


def heal_seam(
    image: PIL.Image.Image,
    mask: np.ndarray,
    band_width_px: int = 96,
    wrap_horizontal: bool = False,
    radius: int = 5,
    method: str = "telea",
    feather_px: int = 16,
    debug_dir: Path | None = None,
    debug_prefix: str = "seam",
    log_fn: Callable[[str], None] | None = None,
) -> PIL.Image.Image:
    """
    Content-aware repair of a band straddling the boundary of `mask` — the
    automated equivalent of dragging Photoshop's Spot Healing Brush along a
    seam. Rather than leave a visible line where two independently generated
    regions meet, the boundary band is treated as a blemish and reconstructed
    from its surroundings via fast-marching inpainting (Telea, 2004) or
    Navier-Stokes diffusion, both of which propagate nearby colour/gradient
    information inward instead of copy-pasting a hard edge.

    The inpainted band is faded in with a feathered alpha rather than swapped
    in with a hard boolean mask — cv2.inpaint leaves everything outside the
    band identical to the source, so blending with a soft-edged alpha over
    the whole image costs nothing where the mask is 0/1 and removes the ring
    artifact a hard cutoff left at the band's inner/outer edge.

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
    feather_px:      Gaussian blur radius applied to the band mask before
                      alpha-compositing, so the heal fades in/out smoothly.
    debug_dir:       when set, the feathered blend mask actually used is
                      written here for inspection.
    debug_prefix:    filename prefix for the debug mask.
    log_fn:          optional callback (e.g. a pipeline stage's log_warning)
                      invoked when the heal is skipped entirely, so a no-op
                      here shows up somewhere other than a silently
                      unblended seam in the final image.
    """
    if not mask.any() or mask.all():
        if log_fn is not None:
            coverage = mask.mean()
            log_fn(
                f"[{debug_prefix}] heal_seam: mask has no boundary to heal "
                f"(coverage={coverage:.3%}) — skipping, image returned unchanged"
            )
        return image

    half = max(1, band_width_px // 2)
    dilated = binary_dilation(mask, iterations=half)
    eroded = binary_erosion(mask, iterations=half)
    band = dilated & ~eroded

    if not band.any():
        if log_fn is not None:
            log_fn(
                f"[{debug_prefix}] heal_seam: computed band is empty "
                f"(band_width_px={band_width_px}) — skipping, image returned unchanged"
            )
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

    blend_mask_pil = PIL.Image.fromarray(band_u8, mode="L")
    if feather_px > 0:
        blend_mask_pil = blend_mask_pil.filter(PIL.ImageFilter.GaussianBlur(feather_px))
        # A wide feather relative to the (deliberately narrow) band washes the
        # whole mask out to a faint grey — the Gaussian spreads the band's 255
        # core thin rather than just softening its edges, so the healed seam
        # never gets blended in at full strength anywhere. Renormalise back up
        # so the band's centre reaches full opacity again; the taper shape
        # (and thus the "wide, soft transition" this feather exists for) is
        # unchanged, only the weak peak is restored.
        peak = np.asarray(blend_mask_pil, dtype=np.float32).max()
        if 0 < peak < 255:
            blend_mask_pil = PIL.Image.fromarray(
                np.clip(np.asarray(blend_mask_pil, dtype=np.float32) * (255.0 / peak), 0, 255).astype(np.uint8),
                mode="L",
            )

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        blend_mask_pil.save(debug_dir / f"{debug_prefix}_blend_mask.png")

    alpha = np.asarray(blend_mask_pil, dtype=np.float32)[:, :, None] / 255.0
    result = arr.astype(np.float32) * (1.0 - alpha) + healed.astype(np.float32) * alpha
    return PIL.Image.fromarray(result.clip(0, 255).astype(np.uint8))


def heal_wrap_seam(
    image: PIL.Image.Image,
    inpaint_fn: Callable[[PIL.Image.Image, PIL.Image.Image], PIL.Image.Image],
    seam_width_px: int = 96,
    feather_px: int = 12,
    eligible_mask: np.ndarray | None = None,
    debug_dir: Path | None = None,
    debug_prefix: str = "wrap_seam",
    log_fn: Callable[[str], None] | None = None,
) -> PIL.Image.Image:
    """
    Repairs the left/right wraparound discontinuity of an equirectangular
    panorama, where the generator produced column 0 and column W-1 somewhat
    independently. Column 0 and column W-1 are the same seam in the final
    360° view, so any mismatch between them shows up as a hard vertical line.

    The fix: roll the image by half its width so that seam lands in the
    middle of the frame — away from the border, where a normal masked
    generative inpaint can reach across it — run `inpaint_fn` on a feathered
    band straddling that center seam, then roll back. This is the same
    "shift to center, heal, shift back" trick used for tileable ground
    textures, minus the vertical half of the cross (equirect panoramas don't
    wrap top-to-bottom — those are the zenith/nadir poles, not a seam).

    image:          equirectangular RGB source.
    inpaint_fn:     runs the actual generative fill, e.g. a FLUX/SD inpaint
                    call. Receives (rolled_image, feathered_mask) and must
                    return an image of the same size.
    seam_width_px:  total width of the band straddling the seam that is
                    eligible to change, before feathering.
    feather_px:     Gaussian blur radius applied to the band mask so the
                    inpaint blends rather than cutting a hard edge.
    eligible_mask:  optional boolean array (H×W, in `image`'s coordinates)
                    restricting the heal to True regions — e.g. only the
                    generatively-filled sky, leaving real photographed pixels
                    untouched. None = heal everywhere the band covers.
    debug_dir:      when set, every intermediate (rolled input, raw mask,
                    feathered mask, raw inpaint output, final composite,
                    unrolled result) is written here so the process can be
                    inspected step by step.
    debug_prefix:   filename prefix for the debug images, so multiple calls
                    (e.g. one per panorama) don't clobber each other's output.
    log_fn:         optional callback (e.g. a pipeline stage's log_warning)
                    invoked when the heal is skipped entirely — e.g. because
                    eligible_mask doesn't reach the wrap column on this
                    particular panorama — so the raw unblended seam left
                    behind isn't a silent failure.
    """
    w, h = image.size
    half = w // 2

    arr = np.array(image.convert("RGB"))
    rolled = np.roll(arr, half, axis=1)
    rolled_pil = PIL.Image.fromarray(rolled)

    sw = max(1, seam_width_px // 2)
    col_lo, col_hi = half - sw, half + sw
    band = np.zeros((h, w), dtype=np.uint8)

    if eligible_mask is not None:
        # Restrict to eligible rows, but keep the band a clean solid
        # rectangle rather than tracing eligible_mask's per-pixel contour —
        # intersecting pixel-for-pixel lets the ragged shroud/horizon edge
        # cut straight across the strip, leaving FLUX a thin, jagged sliver
        # to fill instead of a simple vertical band, which is what was
        # producing noise instead of coherent cloud continuation.
        #
        # row_has_eligible from a per-row `.any()` is still just a boolean
        # per row, not necessarily contiguous — a single noisy/eroded row
        # inside the strip (segmentation noise near the shroud edge) reads
        # as ineligible while the rows above and below it don't, punching a
        # one-row hole straight through the middle of the band. Take the
        # eligible rows' min/max instead and fill solid between them, so the
        # box is a true unbroken vertical rectangle.
        rolled_eligible = np.roll(eligible_mask, half, axis=1)
        row_has_eligible = rolled_eligible[:, col_lo:col_hi].any(axis=1)
        eligible_rows = np.flatnonzero(row_has_eligible)
        if eligible_rows.size > 0:
            row_lo, row_hi = eligible_rows[0], eligible_rows[-1] + 1
            band[row_lo:row_hi, col_lo:col_hi] = 255
    else:
        band[:, col_lo:col_hi] = 255

    if debug_dir is not None:
        debug_dir = Path(debug_dir)
        rolled_pil.save(debug_dir / f"{debug_prefix}_1_rolled.png")
        PIL.Image.fromarray(band, mode="L").save(debug_dir / f"{debug_prefix}_2_mask_raw.png")

    if not band.any():
        if log_fn is not None:
            reason = "eligible_mask doesn't reach the wrap-seam column" if eligible_mask is not None else "seam_width_px produced an empty band"
            log_fn(
                f"[{debug_prefix}] heal_wrap_seam: {reason} — skipping, image returned unchanged"
            )
        return image

    mask_pil = PIL.Image.fromarray(band, mode="L")
    if feather_px > 0:
        mask_pil = mask_pil.filter(PIL.ImageFilter.GaussianBlur(feather_px))

    if debug_dir is not None:
        mask_pil.save(debug_dir / f"{debug_prefix}_3_mask_feathered.png")

    healed_rolled = inpaint_fn(rolled_pil, mask_pil)

    if debug_dir is not None:
        healed_rolled.save(debug_dir / f"{debug_prefix}_4_inpainted_rolled_raw.png")

    healed_arr = np.array(healed_rolled.convert("RGB"))
    if healed_arr.shape[:2] != (h, w):
        healed_arr = np.array(PIL.Image.fromarray(healed_arr).resize((w, h), PIL.Image.LANCZOS))

    # Composite explicitly against the feathered mask rather than trusting
    # inpaint_fn to have left everything outside it untouched — guarantees the
    # only pixels that can change are the ones we asked for, regardless of how
    # faithfully the underlying pipeline honours its own mask.
    alpha = np.asarray(mask_pil, dtype=np.float32)[:, :, None] / 255.0
    composited_rolled = (rolled.astype(np.float32) * (1.0 - alpha) + healed_arr.astype(np.float32) * alpha)
    composited_rolled = composited_rolled.clip(0, 255).astype(np.uint8)

    if debug_dir is not None:
        PIL.Image.fromarray(composited_rolled).save(debug_dir / f"{debug_prefix}_5_composited_rolled.png")

    result = np.roll(composited_rolled, -half, axis=1)
    result_pil = PIL.Image.fromarray(result)

    if debug_dir is not None:
        result_pil.save(debug_dir / f"{debug_prefix}_6_unrolled_result.png")

    return result_pil
