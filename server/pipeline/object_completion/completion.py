"""
Object Completion: a complete, canonical picture of each object before it is
reconstructed.

Single-image reconstruction needs an image OF the object. What the detector
hands it is a fragment: a 47 px bloom with a leaf across it, a conifer with the
next tree in front of it, a boat cut by the edge of the frame. On the
2026-10-13 runs every flower and plant SAM3D was given came back as a flat
sheet, and 4x super-resolution did not change that, because the missing part of
the object is not in the pixels at any resolution.

This stage builds the picture the reconstructor should have had:

  1. The object's cutout (the photo-resolution crop_hires_{i}) is placed on a
     white canvas, scaled so it has room to grow.
  2. The parts of the object we KNOW are missing are marked: where a nearer
     detection sits in front of it (its occluders' masks, from the same
     depth-ordered set Asset Generation scores occlusion with), and beyond any
     edge of the photograph the box touches.
  3. FLUX Fill inpaints only those regions, prompted for the complete object on
     a plain background, so the visible pixels are kept and the hidden ones are
     continued. An optional light img2img pass over the whole canvas adds detail.
  4. BiRefNet cuts the completed object back out of the white background, and the
     result is written as crop_complete_{i}. Asset Generation prefers it for
     delighting, reconstruction and the card texture.

Only the crops Asset Generation will actually build from are completed: the
largest plausibly-shaped crop of every (class, visual-variant) bucket, the same
choice its curation makes. Crops that are neither occluded nor truncated are
left to the super-resolution path.
"""
from __future__ import annotations

import json
import math
from logging import Logger
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image as PILImage, ImageFilter

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.object_typing.categories import VEGETATION_CATEGORIES
from pipeline.object_typing.categories import (
    ENVIRONMENT_CATEGORIES, GRASS_TUFT_CATEGORY, CategoryFilter, normalize_category,
)
from pipeline.panorama_asset_generation.generation import _largest_consistent, _plausible_shape
from pipeline.segmentation.hires_crops import hires_key, is_photographed, CAPTURED_FOV_RESTRICTED_KEY
from pipeline.segmentation.photo_detection import DETECTION_PHOTO_FACTOR_KEY
from util.captured_cone import captured_cone_from_context
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image

COMPLETE_KEY = "crop_complete_{idx}"
# A clean, whole view of a rigid object, generated from its completed crop, for
# the reconstructor only (see ObjectCompletionStage._reference_views).
REFERENCE_KEY = "crop_reference_{idx}"
COMPLETION_SUMMARY_KEY = "object_completion_summary"


def complete_key(idx: int) -> str:
    return COMPLETE_KEY.format(idx=idx)


def reference_key(idx: int) -> str:
    return REFERENCE_KEY.format(idx=idx)


class ObjectCompletionConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        max_crops: int = 40,
        canvas_px: int = 1024,
        object_fraction: float = 0.6,
        occluder_depth_margin: float = 0.10,
        occluder_dilate_px: int = 8,
        min_occluded_fraction: float = 0.02,
        border_extend_fraction: float = 0.3,
        num_inference_steps: int = 28,
        guidance_scale: float = 30.0,
        refine_strength: float = 0.0,
        min_kept_visible_fraction: float = 0.5,
        occluder_reach_fraction: float = 0.25,
        truncated_edge_fraction: float = 0.3,
        context_background: bool = True,
        background_diff: float = 18.0,
        reference_view_classes: list[str] | None = None,
        reference_strength: float = 0.9,
        reference_candidates: int = 2,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Which classes to complete. Default: every placeable object class; never
        # environment labels or grass tufts (GrassCoverStage makes its own cards).
        if include_categories is None and exclude_categories is None:
            exclude_categories = [GRASS_TUFT_CATEGORY]
        self.category_filter = CategoryFilter(include_categories, exclude_categories)
        # Hard cap on FLUX calls per run (one per completed crop, ~15 s each).
        self.max_crops = int(max_crops)
        # Canvas side in pixels (FLUX Fill's native working size) and the fraction
        # of it the visible object's longer side is scaled to, leaving room for the
        # hidden part to be painted around it.
        self.canvas_px = int(canvas_px)
        self.object_fraction = float(object_fraction)
        # A detection counts as an occluder when its sampled depth is nearer by more
        # than this fraction -- the same margin Asset Generation's occlusion score
        # uses, so the two agree about who is in front.
        self.occluder_depth_margin = float(occluder_depth_margin)
        # Grow the occluder masks by this many canvas pixels so the seam between
        # kept and painted pixels sits inside the occluder, not on the object.
        self.occluder_dilate_px = int(occluder_dilate_px)
        # Crops whose occluded area (occluders plus frame truncation) is under this
        # fraction of the canvas are left alone: there is nothing to complete and the
        # super-resolution path handles detail.
        self.min_occluded_fraction = float(min_occluded_fraction)
        # When the box touches an edge of the photograph, this fraction of the
        # object's size beyond that edge is opened for the object to continue into.
        self.border_extend_fraction = float(border_extend_fraction)
        self.num_inference_steps = int(num_inference_steps)
        self.guidance_scale = float(guidance_scale)
        # Optional second pass: a full-canvas img2img at this strength to add detail
        # to the visible part too. 0 disables (the visible pixels stay the photo's).
        self.refine_strength = float(refine_strength)
        # Sanity gate on the result: BiRefNet's cutout must cover at least this
        # fraction of the originally visible pixels, or the completion is rejected
        # (the model replaced the object instead of continuing it).
        self.min_kept_visible_fraction = float(min_kept_visible_fraction)
        # See build_completion_canvas: how far from the visible silhouette an
        # occluder is painted, and how much of a box side must be opaque for the
        # object to count as cut off there.
        self.occluder_reach_fraction = float(occluder_reach_fraction)
        self.truncated_edge_fraction = float(truncated_edge_fraction)
        # Paint the completion in the object's own surroundings (the object-removed
        # terrain panorama warped into the canvas) instead of on a white card. On
        # white the model has nothing to continue but the prompt: a riverboat cut
        # by the frame was continued with a figure standing at its end.
        self.context_background = bool(context_background)
        # A painted hole whose mean absolute difference to the context canvas is
        # under this (0-255) was filled with background, not object, and is dropped.
        self.background_diff = float(background_diff)
        # Rigid classes reconstructed from a generated reference view: the object
        # redrawn whole, at eye level from three-quarters, on white, by FLUX.1-dev
        # img2img from its completed crop. Paris's boats were photographed from a
        # bridge, steeply from above and packed with people; reconstructing that
        # view gave a smeared hull. Empty disables.
        self.reference_view_classes = frozenset(
            str(c).lower() for c in (reference_view_classes if reference_view_classes is not None
                                     else ["boat", "ship", "dock", "bridge", "tower", "building",
                                           "statue", "monument", "car", "truck", "bus", "flower"])
        )
        self.reference_strength = float(reference_strength)
        self.reference_candidates = max(1, int(reference_candidates))


def build_completion_canvas(
    cutout: PILImage.Image,
    occluder_masks: list[tuple[np.ndarray, int, int]],
    touches: dict[str, bool],
    canvas_px: int,
    object_fraction: float,
    dilate_px: int,
    border_extend_fraction: float,
    reach_fraction: float = 0.25,
    truncated_edge_fraction: float = 0.3,
    background: Optional[PILImage.Image] = None,
) -> tuple[PILImage.Image, PILImage.Image, np.ndarray, float, tuple[int, int], float, dict[str, bool]]:
    """Compose the inpainting canvas and mask.

    cutout:          RGBA object cutout in its own crop space.
    occluder_masks:  [(bool mask, x_offset, y_offset)] in the cutout's crop space:
                     each occluder's own alpha with its top-left relative to the
                     cutout's box (may extend outside it).
    touches:         {"left","right","top","bottom": bool}, edges of the photograph
                     the object's box reaches.
    reach_fraction:  an occluder is only painted within this fraction of the
                     object's size of its visible silhouette. The hidden part of a
                     connected object is adjacent to its visible part; the rest of
                     the occluder is not the object and painting it only invites
                     the model to put something else there. The hole is the
                     occluder's bounding rectangle inside that band, not its
                     silhouette: a person-shaped hole on the Rainier lupine crop
                     came back filled with a person (two white mannequins, built
                     into the flower mesh and placed 36 times).
    truncated_edge_fraction: a side of the visible silhouette's box along which at
                     least this fraction of the edge is opaque is a cut, whatever
                     cut it (the photograph's frame, an undetected occluder, the
                     quay in front of a riverboat), and is opened like a frame
                     edge so the object can continue past it.
    background:      optional RGB canvas-sized image of the object's surroundings
                     with the objects removed; the completion is then painted in
                     context instead of on white. The visible object is composited
                     over it.
    Returns (canvas RGB, mask L, visible bool array, scale, (ox, oy) placement,
    masked fraction of the canvas, effective touches).
    """
    cw, ch = cutout.size
    scale = object_fraction * canvas_px / max(cw, ch, 1)
    sw, sh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
    ox, oy = (canvas_px - sw) // 2, (canvas_px - sh) // 2
    scaled = cutout.resize((sw, sh), PILImage.LANCZOS)
    alpha = np.asarray(scaled.split()[-1], dtype=np.float32) / 255.0

    if background is not None and background.size == (canvas_px, canvas_px):
        canvas = np.asarray(background.convert("RGB"), dtype=np.float32).copy()
    else:
        canvas = np.full((canvas_px, canvas_px, 3), 255.0, dtype=np.float32)
    rgb = np.asarray(scaled.convert("RGB"), dtype=np.float32)
    region = canvas[oy:oy + sh, ox:ox + sw]
    canvas[oy:oy + sh, ox:ox + sw] = rgb * alpha[..., None] + region * (1.0 - alpha[..., None])
    visible = np.zeros((canvas_px, canvas_px), dtype=bool)
    visible[oy:oy + sh, ox:ox + sw] = alpha > 0.5

    # The band around the visible silhouette an occluder may be painted in.
    reach_px = max(int(dilate_px), int(round(reach_fraction * max(sw, sh))))
    allowed = np.asarray(
        PILImage.fromarray(visible.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * reach_px + 1))
    ) > 127

    mask = np.zeros((canvas_px, canvas_px), dtype=bool)
    for m, mx, my in occluder_masks:
        mh, mw = m.shape
        # Occluder mask into canvas coordinates.
        x0, y0 = ox + mx * scale, oy + my * scale
        tw, th = max(1, int(round(mw * scale))), max(1, int(round(mh * scale)))
        tile = np.asarray(PILImage.fromarray(m.astype(np.uint8) * 255).resize((tw, th), PILImage.NEAREST)) > 127
        cx0, cy0 = int(round(x0)), int(round(y0))
        cx1, cy1 = cx0 + tw, cy0 + th
        sx0, sy0 = max(0, -cx0), max(0, -cy0)
        cx0, cy0 = max(0, cx0), max(0, cy0)
        cx1, cy1 = min(canvas_px, cx1), min(canvas_px, cy1)
        if cx1 > cx0 and cy1 > cy0:
            occ = np.zeros((canvas_px, canvas_px), dtype=bool)
            occ[cy0:cy1, cx0:cx1] = tile[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)]
            hole = occ & allowed
            if hole.any():
                ys, xs = np.nonzero(hole)
                rect = np.zeros((canvas_px, canvas_px), dtype=bool)
                rect[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = True
                mask |= rect & allowed

    # Sides along which the visible silhouette is cut straight off.
    touches = dict(touches)
    obj = visible[oy:oy + sh, ox:ox + sw]
    edge = max(1, min(2, sw, sh))
    cut = {
        "left": float(obj[:, :edge].any(axis=1).mean()),
        "right": float(obj[:, -edge:].any(axis=1).mean()),
        "top": float(obj[:edge, :].any(axis=0).mean()),
        "bottom": float(obj[-edge:, :].any(axis=0).mean()),
    }
    for side, fraction in cut.items():
        if fraction >= truncated_edge_fraction:
            touches[side] = True

    ext_w, ext_h = int(sw * border_extend_fraction), int(sh * border_extend_fraction)
    if touches.get("left"):
        mask[oy:oy + sh, max(0, ox - ext_w):ox + 2] = True
    if touches.get("right"):
        mask[oy:oy + sh, ox + sw - 2:min(canvas_px, ox + sw + ext_w)] = True
    if touches.get("top"):
        mask[max(0, oy - ext_h):oy + 2, ox:ox + sw] = True
    if touches.get("bottom"):
        mask[oy + sh - 2:min(canvas_px, oy + sh + ext_h), ox:ox + sw] = True
    # Never repaint what the photograph shows of the object itself, except in a
    # band right at each occluder's edge, where the "object" pixels are really the
    # occluder's fringe: grow the mask by dilate_px, and let only that grown band
    # eat into the visible region.
    if dilate_px > 0 and mask.any():
        core = mask
        mask = np.asarray(
            PILImage.fromarray(core.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * dilate_px + 1))
        ) > 127
        band = mask & ~core
        mask &= ~visible | band
    else:
        mask &= ~visible
    canvas_img = PILImage.fromarray(canvas.clip(0, 255).astype(np.uint8), "RGB")
    mask_img = PILImage.fromarray(mask.astype(np.uint8) * 255, "L")
    return canvas_img, mask_img, visible, scale, (ox, oy), float(mask.mean()), touches


def _object_only(
    matte: np.ndarray, visible: np.ndarray, painted: np.ndarray, grow_px: int,
    completed: Optional[np.ndarray] = None, background: Optional[np.ndarray] = None,
    background_diff: float = 18.0,
) -> tuple[np.ndarray, int]:
    """Restrict a matte of the completed canvas to the object: inside the
    photographed silhouette or the painted holes (grown a little for the matte's
    own edge), and in a connected component that contains photographed pixels.

    A painted hole whose pixels came back as its own surroundings -- the model
    continued the background, not the object -- is not part of the object: with
    the context canvas, a hole opened beside a paintbrush was filled with meadow
    and the matte took the meadow as a 1 m oval of flower. Such a hole (mean
    absolute difference to the background canvas under background_diff, per
    connected hole) is dropped. Returns (mask, holes dropped)."""
    from scipy.ndimage import label as _label
    dropped = 0
    if completed is not None and background is not None and painted.any():
        diff = np.abs(completed.astype(np.float32) - background.astype(np.float32)).mean(axis=2)
        # The refinement pass re-renders the context too, so even untouched
        # background differs from the (blurrier) context canvas; measured on
        # Rainier 9-16 outside the holes against 27 for a hole filled with meadow
        # and 35-44 for holes the model actually painted something into. The
        # threshold is therefore relative to this crop's own baseline.
        outside = ~painted & ~visible
        baseline = float(diff[outside].mean()) if outside.any() else 0.0
        threshold = max(background_diff, 2.5 * baseline)
        holes, count = _label(painted)
        keep_painted = np.zeros_like(painted)
        for h in range(1, count + 1):
            hole = holes == h
            if float(diff[hole].mean()) >= threshold:
                keep_painted |= hole
            else:
                dropped += 1
        painted = keep_painted
    allowed = visible | painted
    if grow_px > 0:
        allowed = np.asarray(
            PILImage.fromarray(allowed.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * int(grow_px) + 1))
        ) > 127
    # The photographed pixels ARE the object, whatever the matte made of a busy
    # background (with the surroundings painted in, BiRefNet picked the far bank
    # instead of the plant on three Rainier crops and the cutout kept 0% of the
    # photograph); the matte only decides what the painted holes added.
    mask = visible | (matte & allowed)
    labels, count = _label(mask)
    keep = np.unique(labels[mask & visible])
    keep = keep[keep > 0]
    return (np.isin(labels, keep) if keep.size else np.zeros_like(mask)), dropped


class ObjectCompletionStage(PipelineStage):
    @classmethod
    def config_class(cls) -> type[ObjectCompletionConfiguration]:
        return ObjectCompletionConfiguration

    # ── context background ───────────────────────────────────────────────────

    @staticmethod
    def _locate_in_photo(photo: Image, cutout: PILImage.Image, photo_box, factor: float) -> Optional[tuple[float, float]]:
        """Where the hi-res cutout sits in the photograph, in NATIVE photo pixels.
        The cutout was cut from the photograph upscaled by `factor` for detection
        (DETECTION_PHOTO_FACTOR), so it is matched downscaled, as a masked template,
        inside the detection box grown by half its size: the box is right to a few
        pixels, the match settles the rest. None when there is nothing to match."""
        import cv2
        rgba = np.asarray(cutout.convert("RGBA"))
        f = max(float(factor), 1e-6)
        tw, th = max(2, int(round(rgba.shape[1] / f))), max(2, int(round(rgba.shape[0] / f)))
        small = np.asarray(cutout.convert("RGBA").resize((tw, th), PILImage.LANCZOS))
        template = np.ascontiguousarray(small[..., :3])
        tmask = (small[..., 3] > 127).astype(np.uint8)
        if tmask.sum() < 16:
            return None
        image = np.asarray(photo.image.convert("RGB"))
        H, W = image.shape[:2]
        bx, by, bw, bh = photo_box
        x0, y0 = int(max(0, bx - bw * 0.5)), int(max(0, by - bh * 0.5))
        x1, y1 = int(min(W, bx + bw * 1.5)), int(min(H, by + bh * 1.5))
        window = np.ascontiguousarray(image[y0:y1, x0:x1])
        if window.shape[0] < th or window.shape[1] < tw:
            return None
        result = cv2.matchTemplate(window, template, cv2.TM_SQDIFF, mask=np.repeat(tmask[..., None], 3, axis=2))
        my, mx = np.unravel_index(int(np.argmin(result)), result.shape)
        return float(x0 + mx), float(y0 + my)

    def _background_canvas(
        self, photo: Image, cutout: PILImage.Image, photo_box, factor: float, cone, terrain_pano,
        canvas_px: int, object_fraction: float,
    ) -> Optional[PILImage.Image]:
        """The object's surroundings with the objects removed (PANORAMA_TERRAIN),
        warped into the completion canvas so each canvas pixel shows what stands
        behind that point of the photograph -- including beyond the photograph's
        frame, where the outpainted panorama continues the scene."""
        from scipy.ndimage import map_coordinates
        origin = self._locate_in_photo(photo, cutout, photo_box, factor)
        if origin is None:
            return None
        cw, ch = cutout.size
        scale = object_fraction * canvas_px / max(cw, ch, 1)
        sw, sh = max(1, int(round(cw * scale))), max(1, int(round(ch * scale)))
        ox, oy = (canvas_px - sw) // 2, (canvas_px - sh) // 2
        f = max(float(factor), 1e-6)
        # canvas px -> cutout px (/scale) -> native photo px (/factor) + origin
        px = origin[0] + (np.arange(canvas_px, dtype=np.float64) - ox) / scale / f
        py = origin[1] + (np.arange(canvas_px, dtype=np.float64) - oy) / scale / f
        PX, PY = np.meshgrid(px, py)
        # CapturedCone.photo_to_pano, vectorised, onto the terrain panorama's grid.
        lon = (PX / cone.photo_w - 0.5) * cone.hfov_deg
        lat = (PY / cone.photo_h - 0.5) * cone.vfov_deg
        arr = np.asarray(terrain_pano.image.convert("RGB"), dtype=np.float32)
        cols = np.mod((lon / 360.0 + 0.5) * arr.shape[1], arr.shape[1])
        rows = np.clip((lat / 180.0 + 0.5) * arr.shape[0], 0, arr.shape[0] - 1)
        out = np.stack([
            map_coordinates(arr[..., c], [rows.ravel(), cols.ravel()], order=1, mode="wrap").reshape(canvas_px, canvas_px)
            for c in range(3)
        ], axis=-1)
        return PILImage.fromarray(out.clip(0, 255).astype(np.uint8), "RGB")

    def __init__(self, config: ObjectCompletionConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)
        self._inpainter: Optional[InPainting] = None
        self._foreground = None

    # ── selection ────────────────────────────────────────────────────────────

    def _select(self, context: PipelineContext, count: int, cone_restricted: bool) -> tuple[list[int], dict[int, dict]]:
        """One crop per (class, bucket): the largest plausibly shaped in-cone crop
        with a photo-space box -- the same choice Asset Generation's curation makes
        for its mesh representative and card texture."""
        metas: dict[int, dict] = {}
        by_group: dict[tuple[str, int], list[dict]] = {}
        for idx in range(count):
            m = context.input_object(f"metadata_{idx}") or {}
            if not m.get("box") or m.get("synthetic") or m.get("position_only"):
                continue
            if not is_photographed(m, cone_restricted) or not m.get("photo_box"):
                continue
            cls = normalize_category(m.get("class"))
            if not cls or cls == "indeterminate" or cls in ENVIRONMENT_CATEGORIES:
                continue
            if not self.config.category_filter.allows(cls):
                continue
            metas[idx] = m
            by_group.setdefault((cls, int(m.get("bucket") or 0)), []).append({"idx": idx, "box": m["box"]})
        chosen = []
        for (cls, bucket), cands in sorted(by_group.items()):
            plausible = [c for c in cands if _plausible_shape(cls, c["box"])]
            pick = _largest_consistent(plausible, cands) if plausible else None
            if pick is not None:
                chosen.append(pick["idx"])
        return chosen, metas

    # ── occluders ────────────────────────────────────────────────────────────

    @staticmethod
    def _depth_at(depth, box, pano_w: int, pano_h: int) -> Optional[float]:
        if depth is None or not box:
            return None
        bx, by, bw, bh = box
        sx, sy = depth.width / pano_w, depth.height / pano_h
        cx, cy = int(round((bx + bw / 2.0) * sx)), int(round((by + bh / 2.0) * sy))
        patch = depth.depth[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        return float(np.median(valid)) if valid.size else None

    def _occluders(
        self, context: PipelineContext, idx: int, metas: dict[int, dict], depths: dict[int, float], factor: float,
    ) -> list[tuple[np.ndarray, int, int]]:
        """Masks of nearer detections overlapping crop `idx`, in its hi-res crop space."""
        me = metas[idx]
        my_depth = depths.get(idx)
        if my_depth is None:
            return []
        x, y, w, h = (v * factor for v in me["photo_box"])
        out = []
        for j, mj in metas.items():
            if j == idx:
                continue
            dj = depths.get(j)
            if dj is None or not (dj < my_depth * (1.0 - self.config.occluder_depth_margin)):
                continue
            jx, jy, jw, jh = (v * factor for v in mj["photo_box"])
            if jx + jw <= x or jx >= x + w or jy + jh <= y or jy >= y + h:
                continue
            hires = context.input_image(hires_key(j))
            if hires is None:
                continue
            alpha = np.asarray(hires.image.convert("RGBA").split()[-1]) > 127
            out.append((alpha, int(round(jx - x)), int(round(jy - y))))
        return out

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: ObjectCompletionConfiguration = self.config
        count = context.input_object(ContextKey.OBJECT_COUNT) or 0
        photo = context.input_image(ContextKey.INPUT)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        if not count or photo is None or panorama is None:
            self.log_info("Object completion: nothing to do")
            context.add_object(COMPLETION_SUMMARY_KEY, {"completed": 0})
            return context
        factor = float(context.input_object(DETECTION_PHOTO_FACTOR_KEY) or 1.0)
        cone_restricted = bool(context.input_object(CAPTURED_FOV_RESTRICTED_KEY) or False)
        depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)

        chosen, metas = self._select(context, count, cone_restricted)
        if not chosen:
            self.log_info("Object completion: no in-cone crops selected")
            context.add_object(COMPLETION_SUMMARY_KEY, {"completed": 0})
            return context
        depths = {i: self._depth_at(depth, m["box"], panorama.width, panorama.height) for i, m in metas.items()}
        cone = captured_cone_from_context(context, panorama.width, panorama.height) if cfg.context_background else None
        terrain_pano = context.input_panorama(ContextKey.PANORAMA_TERRAIN) if cfg.context_background else None
        if cfg.context_background and (cone is None or terrain_pano is None):
            self.log_warning("Object completion: no captured cone or terrain panorama; completing on white")

        # Decide which selected crops have anything to complete, before loading FLUX.
        jobs = []
        pw, ph = photo.width, photo.height
        for idx in chosen:
            m = metas[idx]
            hires = context.input_image(hires_key(idx))
            if hires is None:
                continue
            bx, by, bw, bh = m["photo_box"]
            frame = {
                "left": bx <= 1.5, "top": by <= 1.5,
                "right": bx + bw >= pw - 1.5, "bottom": by + bh >= ph - 1.5,
            }
            occluders = self._occluders(context, idx, metas, depths, factor)
            cutout = hires.image.convert("RGBA")
            background = None
            if cone is not None and terrain_pano is not None:
                background = self._background_canvas(
                    photo, cutout, m["photo_box"], factor, cone, terrain_pano, cfg.canvas_px, cfg.object_fraction,
                )
            # A plant's silhouette is straight-edged by nature (a cluster, a stem
            # against its neighbours) and a cut one is not worth continuing: the
            # opening gets meadow painted into it. Only rigid things get the test.
            cls_now = normalize_category(m.get("class"))
            edge_fraction = 2.0 if cls_now in VEGETATION_CATEGORIES else cfg.truncated_edge_fraction
            canvas, mask, visible, scale, origin, masked, touches = build_completion_canvas(
                cutout, occluders, frame, cfg.canvas_px, cfg.object_fraction,
                cfg.occluder_dilate_px, cfg.border_extend_fraction,
                reach_fraction=cfg.occluder_reach_fraction,
                truncated_edge_fraction=edge_fraction,
                background=background,
            )
            if masked < cfg.min_occluded_fraction:
                continue
            cut_sides = [side for side, on in touches.items() if on and not frame.get(side)]
            if cut_sides:
                self.log_info(f"  crop_{idx}: silhouette cut off at the {', '.join(cut_sides)}; opened to continue")
            jobs.append((idx, m, canvas, mask, visible, masked, len(occluders), touches, background))
        jobs.sort(key=lambda j: -j[5])
        jobs = jobs[: cfg.max_crops]
        self.log_info(
            f"Object completion: {len(chosen)} bucket representative(s), {len(jobs)} occluded or truncated "
            f"(≥ {cfg.min_occluded_fraction:.0%} of the canvas hidden) will be completed"
        )
        if not jobs:
            refs = self._reference_views(context, chosen, metas)
            context.add_object(COMPLETION_SUMMARY_KEY, {"completed": 0, "selected": len(chosen), "reference_views": refs})
            return context

        task = self.create_progress(len(jobs) + 1, "Completing objects…")
        super().clean_up()
        self._inpainter = InPainting(self.preferred_device, self.torch_dtype, InPaintingType.FLUX)
        self.advance_progress(task)
        results: list[tuple[int, dict, PILImage.Image, np.ndarray]] = []
        try:
            for n, (idx, m, canvas, mask, visible, masked, n_occ, touches, background) in enumerate(jobs):
                cls = normalize_category(m.get("class")).replace("_", " ")
                setting = "in its real surroundings" if background is not None else "centered on a plain white background"
                prompt = (
                    f"a complete {cls}, the entire {cls} fully visible from top to bottom, "
                    f"nothing hidden or cut off, {setting}, "
                    f"natural daylight, photorealistic, sharp detail"
                )
                debug = self.temp / f"crop_{idx}" if self.temp is not None else None
                if debug is not None:
                    debug.mkdir(parents=True, exist_ok=True)
                    canvas.save(debug / "01_canvas.png"); mask.save(debug / "02_mask.png")
                try:
                    out = self._inpainter.inpaint(
                        canvas, mask, debug or self.temp, prompt=prompt,
                        num_inference_steps=cfg.num_inference_steps, guidance_scale=cfg.guidance_scale,
                        strength=1.0, seed=self.seed + idx,
                    )
                    if cfg.refine_strength > 0.0:
                        full = PILImage.new("L", canvas.size, 255)
                        out = self._inpainter.inpaint(
                            out, full, debug or self.temp, prompt=prompt,
                            num_inference_steps=cfg.num_inference_steps, guidance_scale=cfg.guidance_scale,
                            strength=cfg.refine_strength, seed=self.seed + idx,
                        )
                except Exception as exc:
                    self.log_warning(f"  crop_{idx} ({cls}): completion failed ({exc}); keeping the crop")
                    self.advance_progress(task)
                    continue
                if debug is not None:
                    out.save(debug / "03_completed.png")
                results.append((idx, m, out.convert("RGB"), visible, np.asarray(mask) > 127, background))
                self.log_info(
                    f"  crop_{idx} ({cls}): {masked:.1%} of the canvas completed "
                    f"({n_occ} occluder(s){', touches frame' if any(touches.values()) else ''})"
                )
                self.advance_progress(task)
        finally:
            self._inpainter.close()
            self._inpainter = None
            super().clean_up()

        # Cut the completed objects back out of their white backgrounds.
        from pipeline.segmentation.foreground_segmentation import ForegroundSeg
        written, rejected = 0, 0
        self._foreground = ForegroundSeg(self.preferred_device)
        try:
            for idx, m, out, visible, painted, background in results:
                seg = self._foreground.segment(Image(out))
                matte = np.asarray(seg.masks[0], dtype=bool) if not seg.is_empty() else np.zeros_like(visible)
                # The object is what the photograph showed plus what was painted for
                # it, and nothing that is not connected to the photographed pixels:
                # a figure the model added beside the object is not the object.
                mask, dropped = _object_only(
                    matte, visible, painted, cfg.occluder_dilate_px,
                    completed=np.asarray(out.convert("RGB")),
                    background=(np.asarray(background.convert("RGB")) if background is not None else None),
                    background_diff=cfg.background_diff,
                )
                if dropped:
                    self.log_info(f"  crop_{idx}: {dropped} painted hole(s) came back as background, dropped")
                if not mask.any():
                    rejected += 1
                    continue
                kept = float((mask & visible).sum()) / max(float(visible.sum()), 1.0)
                if kept < cfg.min_kept_visible_fraction:
                    self.log_info(
                        f"  crop_{idx}: completion rejected, the cutout keeps only {kept:.0%} of the "
                        f"photographed pixels"
                    )
                    rejected += 1
                    continue
                alpha = PILImage.fromarray(mask.astype(np.uint8) * 255, "L")
                rgba = out.convert("RGBA")
                rgba.putalpha(alpha)
                bbox = alpha.getbbox()
                cutout = rgba.crop(bbox) if bbox else rgba
                context.add_image(complete_key(idx), Image(cutout))
                if self.temp is not None:
                    cutout.save(self.temp / f"crop_{idx}" / "04_cutout.png")
                written += 1
        finally:
            self._foreground.unload()
            self._foreground = None
            super().clean_up()
        self.finish_progress(task)

        summary = {"selected": len(chosen), "attempted": len(jobs), "completed": written, "rejected": rejected}
        context.add_object(COMPLETION_SUMMARY_KEY, summary)
        self.log_info(f"Object completion: {written} crop(s) completed, {rejected} rejected")
        summary["reference_views"] = self._reference_views(context, chosen, metas)
        context.add_object(COMPLETION_SUMMARY_KEY, summary)
        return context

    def _reference_views(self, context: PipelineContext, chosen: list[int], metas: dict[int, dict]) -> int:
        """crop_reference_{i} for every chosen crop of a reference_view_classes class.

        The completed crop (or the hi-res cutout) is centred on a white canvas and
        repainted by FLUX.1-dev at reference_strength with the object named and the
        view asked for. High enough that the prompt sets the viewpoint, low enough
        that colour and kind carry over; of reference_candidates seeds the one whose
        cutout's mean colour is nearest the source's is kept."""
        cfg: ObjectCompletionConfiguration = self.config
        targets = [i for i in chosen if normalize_category(metas[i].get("class")) in cfg.reference_view_classes]
        if not targets:
            return 0
        from pipeline.segmentation.foreground_segmentation import ForegroundSeg
        sources: dict[int, PILImage.Image] = {}
        for idx in targets:
            img = context.image(complete_key(idx)) or context.input_image(hires_key(idx))
            if img is not None:
                sources[idx] = img.image.convert("RGBA")
        if not sources:
            return 0
        self.log_info(f"Reference views for {len(sources)} object(s)")
        super().clean_up()
        inpainter = InPainting(self.preferred_device, self.torch_dtype, InPaintingType.FLUX_DEV)
        generated: dict[int, list[PILImage.Image]] = {}
        try:
            for idx, src in sources.items():
                m = metas[idx]
                cls = normalize_category(m.get("class")).replace("_", " ")
                name = str(m.get("name") or cls)
                subject = name if name != cls else f"a complete {cls}"
                prompt = (
                    f"{subject}, the whole {cls} in frame, seen from the side at eye level, three-quarter view, "
                    f"isolated on a plain white background, studio product photograph, even soft light, sharp"
                )
                strength = cfg.reference_strength
                if min(src.size) >= 160:
                    # A large, clear crop (Paris's tour boat, 235x187 px) is worth
                    # keeping: at 0.9 it came back a generic rowboat. Lower strength
                    # keeps its kind and build; the prompt still sets the view.
                    strength = min(strength, 0.75)
                if name != cls:
                    # A named landmark: the name is the whole description. Its crop
                    # is often a few pixels on the skyline (Paris's Eiffel Tower is
                    # 9x24 px), which at 0.9 would leave a smudge to follow, so a
                    # small one is drawn from the name alone.
                    prompt = (
                        f"the {name}, the entire structure from its base to its top in frame, "
                        f"seen from ground level, isolated on a plain white background, "
                        f"architectural photograph, even soft daylight, sharp detail"
                    )
                    if max(src.size) < 96:
                        strength = 1.0
                if cls in ("flower", "plant"):
                    # A flower crop is a few dozen pixels of a clump in a meadow; what
                    # survives is its bloom colour, so the prompt carries that and asks
                    # for one whole plant.
                    bloom = _bloom_colour_name(src)
                    prompt = (
                        f"a single {bloom + ' ' if bloom else ''}wildflower plant, one upright stem with "
                        f"leaves and its blooms, the whole plant from the base of the stem to the top in frame, "
                        f"seen from the side, isolated on a plain white background, botanical studio "
                        f"photograph, even soft light, sharp"
                    )
                size = cfg.canvas_px
                canvas = PILImage.new("RGB", (size, size), (255, 255, 255))
                fit = cfg.object_fraction * size / max(src.size)
                obj = src.resize((max(1, int(src.width * fit)), max(1, int(src.height * fit))), PILImage.LANCZOS)
                canvas.paste(obj, ((size - obj.width) // 2, (size - obj.height) // 2), obj)
                full = PILImage.new("L", (size, size), 255)
                outs = []
                for k in range(cfg.reference_candidates):
                    try:
                        outs.append(inpainter.inpaint(
                            canvas, full, self.temp, prompt=prompt,
                            num_inference_steps=cfg.num_inference_steps, guidance_scale=3.5,
                            strength=strength, seed=self.seed + 7919 * (k + 1) + idx,
                        ).convert("RGB"))
                    except Exception as exc:
                        self.log_warning(f"  crop_{idx}: reference view failed ({exc})")
                generated[idx] = outs
        finally:
            inpainter.close()
            super().clean_up()

        written = 0
        seg = ForegroundSeg(self.preferred_device)
        try:
            for idx, outs in generated.items():
                src = np.asarray(sources[idx], dtype=np.float32)
                src_rgb = src[..., :3][src[..., 3] > 127]
                target = src_rgb.mean(0) if len(src_rgb) else None
                best, best_err = None, None
                for k, out in enumerate(outs):
                    res = seg.segment(Image(out))
                    if res.is_empty():
                        continue
                    matte = np.asarray(res.masks[0], dtype=bool)
                    if matte.mean() < 0.02:
                        continue
                    rgb = np.asarray(out, dtype=np.float32)[matte]
                    err = float(np.abs(rgb.mean(0) - target).mean()) if target is not None else 0.0
                    if self.temp is not None:
                        (self.temp / f"crop_{idx}").mkdir(parents=True, exist_ok=True)
                        out.save(self.temp / f"crop_{idx}" / f"05_reference_{k}.png")
                    if best_err is None or err < best_err:
                        alpha = PILImage.fromarray(matte.astype(np.uint8) * 255, "L")
                        rgba = out.convert("RGBA"); rgba.putalpha(alpha)
                        best, best_err = rgba.crop(alpha.getbbox()), err
                if best is not None:
                    context.add_image(reference_key(idx), Image(best))
                    written += 1
                    self.log_info(f"  crop_{idx} ({metas[idx].get('name') or metas[idx].get('class')}): "
                                  f"reference view kept (colour distance {best_err:.0f})")
        finally:
            seg.unload()
            super().clean_up()
        return written

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(COMPLETION_SUMMARY_KEY) is not None

    def model_names(self) -> list[str]:
        from pipeline.segmentation.foreground_segmentation import ForegroundSeg
        refs = InPainting.model_names(InPaintingType.FLUX_DEV) if self.config.reference_view_classes else []
        return InPainting.model_names(InPaintingType.FLUX) + refs + ForegroundSeg.model_names()

    def clean_up(self):
        if self._inpainter is not None:
            self._inpainter.close()
            self._inpainter = None
        if self._foreground is not None:
            self._foreground.unload()
            self._foreground = None
        super().clean_up()


_BLOOM_HUES = (
    (0, "red"), (20, "orange"), (45, "yellow"), (75, "yellow-green"), (160, "white"),
    (200, "blue"), (250, "violet"), (290, "purple"), (320, "magenta"), (345, "red"),
)


def _bloom_colour_name(rgba: PILImage.Image) -> "str | None":
    """The colour of a flower crop's blooms: the median hue of its saturated,
    non-green opaque pixels (or "white" for bright unsaturated ones), as a word."""
    arr = np.asarray(rgba.convert("RGBA"), dtype=np.float32) / 255.0
    opaque = arr[..., 3] > 0.5
    if opaque.sum() < 16:
        return None
    hsv = np.asarray(rgba.convert("RGB").convert("HSV"), dtype=np.float32)
    h, sat, val = hsv[..., 0] * (360.0 / 255.0), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    green = (h > 70) & (h < 170)
    blooms = opaque & ~green & (sat > 0.35) & (val > 0.3)
    whites = opaque & (sat < 0.15) & (val > 0.8)
    if whites.sum() > max(blooms.sum(), 0.03 * opaque.sum()):
        return "white"
    if blooms.sum() < 0.03 * opaque.sum():
        return None
    # Circular median via the dominant 10-degree bin.
    hist, edges = np.histogram(h[blooms], bins=36, range=(0, 360))
    hue = float(edges[int(np.argmax(hist))] + 5.0)
    name = "red"
    for start, label in _BLOOM_HUES:
        if hue >= start and label != "white":
            name = label
    return name
