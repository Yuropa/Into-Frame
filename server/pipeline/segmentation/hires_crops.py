"""
Native-resolution crops for detections that lie inside the photographed cone.

Object detection runs on the generated panorama, and inside the cone that is a
downsampled copy of the photograph: on the Mount Rainier capture the 1200 px
wide photo occupies ~1300 px of a 4096 px panorama after a 2x super-resolution
pass, i.e. it was halved and then upscaled by a model that sharpens but cannot
invent detail. Every mesh SAM3D reconstructed and every card texture was cut
from that copy while the original sat untouched in the context.

This module re-cuts those crops from the photo itself: the panorama box is
mapped back to photo pixels through the same embedding WorldGen used
(util.captured_cone), SAM2 is prompted with that box on the photo, and the result
is written as crop_hires_{i} next to crop_{i}. crop_{i} itself is left alone --
Video Object Extraction matches it against a video rendered from the panorama,
and the box in metadata_{i} stays in panorama space for everything that places
objects. Consumers that want pixels (reconstruction, card textures, delighting)
prefer crop_hires_{i} when it exists.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image as PILImage

from util.captured_cone import CapturedCone
from util.image_utils import Image

HIRES_KEY = "crop_hires_{idx}"

# Context flag (bool) written by the detection stages: when True, a detection
# whose metadata carries in_captured_fov == False must not be placed in the
# scene, used as a distribution exemplar, or given an asset. See is_photographed.
CAPTURED_FOV_RESTRICTED_KEY = "captured_fov_restricted"


def hires_key(idx: int) -> str:
    return HIRES_KEY.format(idx=idx)


def is_photographed(metadata: Optional[dict], restricted: bool, exempt_classes=()) -> bool:
    """Whether a detection may be placed / learned from under the cone policy.

    True for anything without the flag (synthetic points, detections made on the
    photo itself, caches from before the flag existed), so turning the policy on
    never removes objects the cone logic has not actually judged.

    exempt_classes lists classes whose outpainted detections are kept anyway. Trees
    are the case: the outpainted panorama around the photo is what the viewer sees
    as the skybox and terrain texture, so its trees are part of the scene, and on
    the Rainier capture 72 of 74 tree detections were outpainted -- dropping them
    left the two photographed ones as the only placed trees and the only exemplars
    Distribution Synthesis had to learn from."""
    if not restricted or not metadata:
        return True
    if exempt_classes and str(metadata.get("class") or "").lower() in exempt_classes:
        return True
    return bool(metadata.get("in_captured_fov", True))


def masked_crop(source_rgba: PILImage.Image, mask: np.ndarray) -> Optional[PILImage.Image]:
    """Tight RGBA cutout of `mask` from `source_rgba`, or None for an empty mask."""
    alpha = PILImage.fromarray((np.asarray(mask).astype(np.uint8) * 255), mode="L")
    bbox = alpha.getbbox()
    if not bbox:
        return None
    rgba = source_rgba.copy()
    rgba.putalpha(alpha)
    return rgba.crop(bbox)


def build_hires_crops(
    seg,
    context,
    photo: Image,
    cone: CapturedCone,
    items: list[tuple[int, list[float]]],
    temp_path: Optional[Path],
    log: Callable[[str], None],
    min_gain: float = 1.15,
    sr_factor: float = 1.0,
    on_progress=None,
) -> int:
    """Write crop_hires_{idx} for every (idx, panorama_box) whose box maps into
    the photo. Returns how many were written.

    `seg` is an ImageSeg client (box-prompted SAM2). Skipped entirely when the
    photo is not meaningfully sharper than the panorama inside the cone
    (`min_gain`), since there would be nothing to gain and SAM2 is not free.
    `sr_factor` is how much the panorama was upscaled after generation; those
    pixels carry no extra information, so the gain is measured against the
    panorama's native width (measured: 1.84x on Rainier, 2.98x on Paris, 1.26x on
    Shark Fin Cove with the 2x placement; roughly double each without it).
    """
    if not items:
        return 0
    gain = cone.scale_gain() * max(sr_factor, 1.0)
    if gain < min_gain:
        log(f"Photo is only {gain:.2f}x the panorama's native resolution inside the cone; not re-cutting crops")
        return 0

    indices: list[int] = []
    boxes: list[list[int]] = []
    for idx, pano_box in items:
        photo_box = cone.photo_box(pano_box)
        if photo_box is None:
            continue
        indices.append(idx)
        boxes.append(photo_box)
    if not boxes:
        return 0

    masks = seg.segment_boxes(photo, boxes, temp_path, on_progress=on_progress)
    source_rgba = photo.rgba()
    written = 0
    for idx, box, mask in zip(indices, boxes, masks):
        crop = masked_crop(source_rgba, mask)
        if crop is None:
            continue
        context.add_image(hires_key(idx), crop)
        written += 1
    log(f"Re-cut {written}/{len(items)} in-cone crop(s) from the photo at {gain:.2f}x resolution")
    return written


def prefer_hires(context, idx: int, max_aspect_drift: float = 1.3):
    """crop_hires_{idx} if it still describes the same object as crop_{idx}.

    Object Instance Refinement rewrites crop_{i} when it coalesces or splits
    detections, and the hi-res cut made for the original box does not follow.
    An aspect-ratio check catches the common case (a union box is a different
    shape from one of its parts); anything that drifted more than
    `max_aspect_drift` falls back to crop_{idx}.
    """
    hires = context.input_image(hires_key(idx))
    current = context.input_image(f"crop_{idx}")
    if hires is None:
        return current
    if current is None:
        return hires
    try:
        a = hires.width / max(hires.height, 1)
        b = current.width / max(current.height, 1)
        drift = max(a / b, b / a)
    except ZeroDivisionError:
        return current
    return hires if drift <= max_aspect_drift else current
