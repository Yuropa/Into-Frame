"""
Detecting on the photograph instead of on the panorama, inside the captured cone.

The panorama's photographed region is a downsampled copy of the photo (the
photo's ~1200 px become ~650 panorama px at the true FOV), cut into 1024 px
tiles for SAM2 with a prompt point every 16 px and a 315 px minimum mask. A
lupine spike is 20 px tall there and does not survive; on the 2026-10-13 Rainier
run the meadow -- the subject of the photograph -- was mostly unboxed, and the
blue lupines were missed entirely, so no lupine variant existed to paint.

The photograph is a perspective image at native resolution, which is what both
SAM2 and Grounding DINO were trained on. This module runs the detectors there
(optionally on a Swin2SR-upscaled copy, so small blooms clear the mask-area
floor) and writes each detection in BOTH spaces downstream stages expect:

  crop_hires_{i}   the cutout from the photo, full resolution -- what
                   reconstruction, delighting and card textures use
  crop_{i}         the same cutout resampled into the panorama's pixel grid at
                   the box it occupies there -- Panorama Foreground Inpainting
                   pastes its alpha at that box, Instance Refinement unions
                   boxes in that space, Video Object Extraction tracks it
  metadata_{i}     box in PANORAMA pixels (depth lookup and placement are
                   panorama-space), plus photo_box in native photo pixels

The photo sits in the panorama through WorldGen's angle-linear embedding
(util.captured_cone), which is separable in u and v, so the resample from photo
to panorama space is a plain anisotropic resize of the cutout.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image as PILImage

from pipeline.segmentation.hires_crops import hires_key
from util.captured_cone import CapturedCone
from util.image_utils import Image

# The (optionally super-resolved) photograph every detector runs on, written by
# the first stage that needs it and reused by the rest.
DETECTION_PHOTO_KEY = "input_detect"
DETECTION_PHOTO_FACTOR_KEY = "input_detect_factor"


def detection_photo(
    context, device, sr_factor: float, temp: Optional[Path], log: Callable[[str], None],
) -> tuple[Optional[Image], float]:
    """The photograph at detection resolution and the factor it was scaled by.

    sr_factor 2.0 or 4.0 runs Swin2SR (2x per pass); anything under 2 returns the
    photo as is. Cached in the context under DETECTION_PHOTO_KEY."""
    cached = context.image(DETECTION_PHOTO_KEY) or context.input_image(DETECTION_PHOTO_KEY)
    if cached is not None:
        factor = context.object(DETECTION_PHOTO_FACTOR_KEY) or context.input_object(DETECTION_PHOTO_FACTOR_KEY) or 1.0
        return cached, float(factor)
    from pipeline.pipeline_context import ContextKey
    photo = context.input_image(ContextKey.INPUT)
    if photo is None:
        return None, 1.0
    passes = int(round(math.log2(sr_factor))) if sr_factor >= 2.0 else 0
    if passes <= 0:
        return photo, 1.0
    from pipeline.supersampling.image_supersampling import ImageSupersampling
    samp = ImageSupersampling(device)
    current = photo.rgb()
    done = 0
    try:
        for _ in range(passes):
            current = samp.supersample(Image(current), temp).image.convert("RGB")
            done += 1
    except Exception as exc:
        log(f"Photo super-resolution failed after {done} pass(es) ({exc}); detecting at {2 ** done}x")
    finally:
        samp.close()
    factor = float(2 ** done)
    if done == 0:
        return photo, 1.0
    context.add_image(DETECTION_PHOTO_KEY, Image(current))
    context.add_object(DETECTION_PHOTO_FACTOR_KEY, factor)
    log(f"Detection photo: {photo.width}x{photo.height} -> {current.width}x{current.height} px ({factor:.0f}x Swin2SR)")
    return context.image(DETECTION_PHOTO_KEY), factor


def store_photo_masks(
    context,
    cone: CapturedCone,
    photo_rgba: PILImage.Image,
    factor: float,
    masks: list,
    scores: list,
    start_idx: int,
    min_area: float,
    max_area: float,
    source: str,
    labels: Optional[list[str]] = None,
    extra: Optional[list[dict]] = None,
    log: Optional[Callable[[str], None]] = None,
) -> tuple[int, int]:
    """Write crop_{i}, crop_hires_{i}, metadata_{i} for every mask over the
    detection photo. Returns (written, filtered_by_size). Areas are tested in
    NATIVE photo pixels so min/max_area keep their calibration regardless of
    the detection scale."""
    written = 0
    filtered = 0
    idx = start_idx
    for k, mask in enumerate(masks):
        alpha = PILImage.fromarray((np.asarray(mask, dtype=bool) * 255).astype(np.uint8), mode="L")
        bbox = alpha.getbbox()
        if not bbox:
            continue
        x0, y0, x1, y1 = bbox
        nat_w, nat_h = (x1 - x0) / factor, (y1 - y0) / factor
        if nat_w * nat_h < min_area or nat_w * nat_h > max_area:
            filtered += 1
            continue
        cutout = photo_rgba.copy()
        cutout.putalpha(alpha)
        cutout = cutout.crop(bbox)

        pu0, pv0 = cone.photo_to_pano(x0 / factor, y0 / factor)
        pu1, pv1 = cone.photo_to_pano(x1 / factor, y1 / factor)
        px, py = int(math.floor(pu0)), int(math.floor(pv0))
        pw, ph = max(1, int(round(pu1 - pu0))), max(1, int(round(pv1 - pv0)))
        pano_crop = cutout.resize((pw, ph), PILImage.LANCZOS)

        context.add_image(f"crop_{idx}", Image(pano_crop))
        context.add_image(hires_key(idx), Image(cutout))
        metadata = {
            "box": [float(px), float(py), float(pw), float(ph)],
            "photo_box": [x0 / factor, y0 / factor, nat_w, nat_h],
            "score": float(scores[k]),
            "in_captured_fov": True,
            "source": source,
        }
        if labels is not None:
            metadata["class"] = labels[k]
        if extra is not None and extra[k]:
            metadata.update(extra[k])
        context.add_object(f"metadata_{idx}", metadata)
        if log is not None:
            label = f" '{labels[k]}'" if labels is not None else ""
            log(f"  crop_{idx}:{label} photo box={[round(v, 1) for v in metadata['photo_box']]} "
                f"-> pano box={metadata['box']} score={float(scores[k]):.3f}")
        idx += 1
        written += 1
    return written, filtered
