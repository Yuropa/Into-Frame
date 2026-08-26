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
    crop_context_px: int | None = None,
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
                    call. Receives (rolled_image, feathered_mask) — the full
                    rolled image, unless crop_context_px is set, in which case
                    just the crop around the band — and its return value is
                    resized back to whatever it was given if the sizes don't
                    match, so inpaint_fn is free to internally downscale for
                    its model's resolution limits.
    seam_width_px:  total width of the band straddling the seam that is
                    eligible to change, before feathering.
    feather_px:     Gaussian blur radius applied to the band mask so the
                    inpaint blends rather than cutting a hard edge.
    eligible_mask:  optional boolean array (H×W, in `image`'s coordinates)
                    restricting the heal to True regions — e.g. only the
                    generatively-filled sky, leaving real photographed pixels
                    untouched. None = heal everywhere the band covers.
    crop_context_px: when set, inpaint_fn is only given the band's bounding
                    box padded by this many pixels on every side, instead of
                    the full rolled image — everything outside the band has
                    alpha 0 in the composite below regardless, so pixels the
                    generator produces out there are always discarded. Handing
                    over a small crop instead of the whole panorama lets
                    inpaint_fn work at (or near) native resolution without
                    having to downscale, avoiding the patch/grid artifacts a
                    generative model produces when run far above its trained
                    resolution. None (default) sends the full rolled image,
                    matching the historical behaviour other callers rely on.
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

    if crop_context_px is not None:
        # Crop to the band's bounding box (+ context) before handing anything
        # to inpaint_fn. Only pixels inside the band can survive the alpha
        # composite below, so the rest of the panorama is wasted input —
        # worse than wasted, since forcing inpaint_fn to see the whole thing
        # is what pushes it to downscale and produce a patch/grid artifact.
        rows_nz, cols_nz = np.where(band > 0)
        crop_row_lo = max(0, int(rows_nz.min()) - crop_context_px)
        crop_row_hi = min(h, int(rows_nz.max()) + 1 + crop_context_px)
        crop_col_lo = max(0, int(cols_nz.min()) - crop_context_px)
        crop_col_hi = min(w, int(cols_nz.max()) + 1 + crop_context_px)
    else:
        crop_row_lo, crop_row_hi = 0, h
        crop_col_lo, crop_col_hi = 0, w

    rolled_crop_pil = PIL.Image.fromarray(rolled[crop_row_lo:crop_row_hi, crop_col_lo:crop_col_hi])
    mask_crop_pil = mask_pil.crop((crop_col_lo, crop_row_lo, crop_col_hi, crop_row_hi))

    if debug_dir is not None and crop_context_px is not None:
        rolled_crop_pil.save(debug_dir / f"{debug_prefix}_3b_crop.png")
        mask_crop_pil.save(debug_dir / f"{debug_prefix}_3c_mask_crop.png")

    healed_crop = inpaint_fn(rolled_crop_pil, mask_crop_pil)

    if debug_dir is not None:
        healed_crop.save(debug_dir / f"{debug_prefix}_4_inpainted_rolled_raw.png")

    crop_w, crop_h = crop_col_hi - crop_col_lo, crop_row_hi - crop_row_lo
    healed_crop_arr = np.array(healed_crop.convert("RGB"))
    if healed_crop_arr.shape[:2] != (crop_h, crop_w):
        healed_crop_arr = np.array(PIL.Image.fromarray(healed_crop_arr).resize((crop_w, crop_h), PIL.Image.LANCZOS))

    healed_full = rolled.copy()
    healed_full[crop_row_lo:crop_row_hi, crop_col_lo:crop_col_hi] = healed_crop_arr

    # Composite explicitly against the feathered mask rather than trusting
    # inpaint_fn to have left everything outside it untouched — guarantees the
    # only pixels that can change are the ones we asked for, regardless of how
    # faithfully the underlying pipeline honours its own mask.
    alpha = np.asarray(mask_pil, dtype=np.float32)[:, :, None] / 255.0
    composited_rolled = (rolled.astype(np.float32) * (1.0 - alpha) + healed_full.astype(np.float32) * alpha)
    composited_rolled = composited_rolled.clip(0, 255).astype(np.uint8)

    if debug_dir is not None:
        PIL.Image.fromarray(composited_rolled).save(debug_dir / f"{debug_prefix}_5_composited_rolled.png")

    result = np.roll(composited_rolled, -half, axis=1)
    result_pil = PIL.Image.fromarray(result)

    if debug_dir is not None:
        result_pil.save(debug_dir / f"{debug_prefix}_6_unrolled_result.png")

    return result_pil


def close_wrap_seam(
    image: PIL.Image.Image,
    band_px: int = 512,
    fine_band_px: int = 16,
    row_sigma: float = 16.0,
    ridge_px: int = 16,
    log_fn: Callable[[str], None] | None = None,
    label: str = "wrap",
) -> PIL.Image.Image:
    """
    Force an equirectangular panorama's left and right edges to match EXACTLY,
    by spreading their per-row difference back into a band at each end.

    The deterministic counterpart to heal_wrap_seam above, and meant to run
    immediately after it. That one asks a generative model to redraw a band
    across the seam, which is the only thing that can invent plausible cloud or
    terrain STRUCTURE across the join -- but a diffusion model has no continuity
    constraint, so it closes the seam only as well as it happens to, and nothing
    checks the result. Measured on the five sample captures, panorama_sky came
    out of that pass with an 11.53-level RGB step at the wrap column on the Paris
    capture, against a median adjacent-column difference of 0.145 and a 99th
    percentile of 0.523 over the whole image: a hard vertical line, 80x the
    typical column-to-column change, in an otherwise smooth sky. The equirect
    terrain layer (SplatLayer "panorama") is the same story from a different
    generator -- Iceland 66.22 against a 99th percentile of 8.58 -- and it wraps
    the viewer at the same longitude, so the two together read as one line
    running from the ground up through the sky.

    With delta = right_column - left_column (per row, per channel), adding
    +delta/2 at column 0 and -delta/2 at column W-1 lands both edges on
    (left + right) / 2, closing the seam identically on every row by
    construction. The only question is how that correction is distributed
    inland, and it is distributed at TWO scales, because delta has two parts
    that want opposite treatment:

      * The low-frequency part (delta smoothed along y by row_sigma) is the real
        defect: the two ends of the same sky at slightly different exposures,
        coherent over hundreds of rows, which is precisely what makes a LINE.
        It carries nearly all the magnitude and is spread over the full band_px
        on a raised cosine -- so thin it cannot be seen. Its steepest gradient is
        pi*|delta| / (4*band_px): at Paris's worst row (delta 43) with band_px
        512 that is 0.066 levels per column, against the 0.145 the image already
        varies by.

      * The high-frequency remainder is row-to-row noise (Paris jumps up to 8
        levels between adjacent rows). Spread over the same wide band it becomes
        512-px-wide horizontal STREAKS -- trading a vertical line for banding,
        which measurably happened: adjacent-row difference over the first 128
        columns went from 1.656 to 3.526. Confined to fine_band_px instead, it
        still closes the seam exactly and any streak it makes is 16 px wide.
        Measured, that lands at 1.659 -- the original 1.656 to within noise.

    A raised cosine rather than a linear ramp at both scales, so each correction
    meets the untouched interior with zero derivative; a linear falloff would
    trade the seam for a fainter line at the band edge, which is the mistake this
    exists to fix, moved inward.

    Replayed over all five captures' stored panorama_sky and equirect terrain
    layer, this takes the wrap step under that image's OWN 99th-percentile column
    step in all ten -- from 11.53 to 0.29 on the Paris sky, from 66.22 to 0.004 on
    the Iceland terrain -- while moving the median adjacent-column difference by
    at most 0.004, leaving the largest column step away from the seam unchanged,
    and growing the row-to-row variation inside the corrected band by 0.2%.

    Note this cannot fix a mismatch in CONTENT (a cloud on one side and not the
    other). That is heal_wrap_seam's job, and it stays in front of this.

    Matching the two edges is not always the whole job, because a generator can
    also leave a local ANOMALY at the join that is not an edge mismatch at all.
    heal_wrap_seam's own feathered composite does exactly that on the Paris sky:
    with the 11.53 step removed, the columns either side of the seam still
    disagree by 3.53 and 3.35 against a 99th percentile of 0.519 -- a narrow dip
    where FLUX's inpaint was blended in, which reads as a fainter version of the
    same line. ridge_px flattens that, by blending the columns within ridge_px of
    the seam toward a straight line between the two anchor columns just outside
    it (raised-cosine again, full strength at the seam, nothing at the anchors).
    Over so few columns any smooth image is linear to well within a level.

    That last step is SELF-GATING: it runs only when the seam neighbourhood's own
    column steps exceed the 99th percentile of the whole image's, i.e. only when
    there is still something anomalous there. On the five captures it fires on
    the Paris sky alone and no-ops on the other four and on all five terrain
    layers, which after the edge match already sit at or below their own ordinary
    variation.

    band_px:      columns at each end the low-frequency correction is spread over.
    fine_band_px: columns at each end the high-frequency remainder is confined to.
                  0 drops the remainder entirely -- the seam then closes only to
                  its low-frequency part (Paris 11.53 -> 1.265), which is still no
                  longer a coherent line, and is the conservative setting if the
                  fine band ever proves visible on some capture.
    row_sigma:    Gaussian sigma, in rows, splitting delta into those two parts.
                  0 disables the split and puts everything in the wide band.
    ridge_px:     half-width of the local flattening described above. 16 because
                  that is where the Paris sky's seam neighbourhood stops being
                  anomalous at all: peak column step within it against the image's
                  own 99th percentile runs 6.8x at 0 (i.e. untouched), 3.0x at 4,
                  1.8x at 8, 0.9x at 16 and 0.9x at 32 -- so 16 is where it crosses
                  under, and past it nothing more is bought. 0 disables, leaving
                  whatever narrow ridge the generator put at the join.
    """
    arr = np.array(image.convert("RGB")).astype(np.float32)
    w = arr.shape[1]
    out = arr.copy()

    delta = arr[:, -1, :] - arr[:, 0, :]                        # (H, 3)

    def spread(correction: np.ndarray, width: int) -> None:
        width = int(min(max(0, width), w // 2))
        if width <= 0:
            return
        ramp = 0.5 * (1.0 + np.cos(np.pi * np.arange(width, dtype=np.float32) / width))
        weight = (0.5 * ramp)[None, :, None]                    # 0.5 at the seam -> 0
        out[:, :width, :] += weight * correction[:, None, :]
        out[:, w - width:, :] -= weight[:, ::-1, :] * correction[:, None, :]

    if row_sigma > 0:
        from scipy.ndimage import gaussian_filter1d

        low = gaussian_filter1d(delta, row_sigma, axis=0, mode="nearest")
        spread(low, band_px)
        spread(delta - low, fine_band_px)
    else:
        spread(delta, band_px)

    # Local ridge flattening -- see ridge_px in the docstring. Done on the rolled
    # array so the seam is an interior column and its neighbourhood is contiguous.
    if ridge_px > 0 and w > 4 * (ridge_px + 1):
        centre = w // 2
        rolled = np.roll(out, centre, axis=1)
        profile = np.abs(rolled[:, 1:, :] - rolled[:, :-1, :]).mean(axis=(0, 2))
        lo, hi = centre - ridge_px, centre + ridge_px
        if profile[lo - 1:hi + 1].max() > np.percentile(profile, 99):
            # Interior columns lo..hi, anchored on lo-1 and hi+1, so column j sits
            # (j - lo + 1) / (n + 1) of the way between the two anchors.
            n = hi - lo + 1
            t = (np.arange(1, n + 1, dtype=np.float32) / (n + 1))[None, :, None]
            left, right = rolled[:, lo - 1, :][:, None, :], rolled[:, hi + 1, :][:, None, :]
            straight = left * (1.0 - t) + right * t
            # 1 at the seam, 0 at the anchors, so the flattening fades into
            # untouched pixels instead of substituting its own two edges.
            d = np.abs(np.arange(lo, hi + 1, dtype=np.float32) - (centre - 0.5))
            alpha = (0.5 * (1.0 + np.cos(np.pi * np.clip(d / (ridge_px + 0.5), 0, 1))))[None, :, None]
            rolled[:, lo:hi + 1, :] = rolled[:, lo:hi + 1, :] * (1.0 - alpha) + straight * alpha
            out = np.roll(rolled, -centre, axis=1)

    # rint, not a bare astype: astype truncates toward zero, which turns a
    # sub-level correction into a systematic downward bias and shows up as fresh
    # quantisation steps of its own in a smooth gradient (measured: it raised the
    # largest column step elsewhere in the Paris sky from 3.489 to 3.759, having
    # just removed an 11.53 one).
    result = np.clip(np.rint(out), 0, 255).astype(np.uint8)

    if log_fn is not None:
        residual = np.abs(result[:, -1, :].astype(np.float32) - result[:, 0, :].astype(np.float32)).mean()
        log_fn(
            f"  {label} seam: mean edge mismatch {float(np.abs(delta).mean()):.2f} -> "
            f"{float(residual):.2f} levels (spread over {band_px} px, "
            f"row detail over {fine_band_px} px)"
        )

    return PIL.Image.fromarray(result)
