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
COMPLETION_SUMMARY_KEY = "object_completion_summary"


def complete_key(idx: int) -> str:
    return COMPLETE_KEY.format(idx=idx)


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


def _object_only(matte: np.ndarray, visible: np.ndarray, painted: np.ndarray, grow_px: int) -> np.ndarray:
    """Restrict a matte of the completed canvas to the object: inside the
    photographed silhouette or the painted holes (grown a little for the matte's
    own edge), and in a connected component that contains photographed pixels."""
    from scipy.ndimage import label as _label
    allowed = visible | painted
    if grow_px > 0:
        allowed = np.asarray(
            PILImage.fromarray(allowed.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * int(grow_px) + 1))
        ) > 127
    mask = matte & allowed
    if not mask.any():
        return mask
    labels, count = _label(mask)
    keep = np.unique(labels[mask & visible])
    keep = keep[keep > 0]
    return np.isin(labels, keep) if keep.size else np.zeros_like(mask)


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
            canvas, mask, visible, scale, origin, masked, touches = build_completion_canvas(
                cutout, occluders, frame, cfg.canvas_px, cfg.object_fraction,
                cfg.occluder_dilate_px, cfg.border_extend_fraction,
                reach_fraction=cfg.occluder_reach_fraction,
                truncated_edge_fraction=cfg.truncated_edge_fraction,
                background=background,
            )
            if masked < cfg.min_occluded_fraction:
                continue
            cut_sides = [side for side, on in touches.items() if on and not frame.get(side)]
            if cut_sides:
                self.log_info(f"  crop_{idx}: silhouette cut off at the {', '.join(cut_sides)}; opened to continue")
            jobs.append((idx, m, canvas, mask, visible, masked, len(occluders), touches, background is not None))
        jobs.sort(key=lambda j: -j[5])
        jobs = jobs[: cfg.max_crops]
        self.log_info(
            f"Object completion: {len(chosen)} bucket representative(s), {len(jobs)} occluded or truncated "
            f"(≥ {cfg.min_occluded_fraction:.0%} of the canvas hidden) will be completed"
        )
        if not jobs:
            context.add_object(COMPLETION_SUMMARY_KEY, {"completed": 0, "selected": len(chosen)})
            return context

        task = self.create_progress(len(jobs) + 1, "Completing objects…")
        super().clean_up()
        self._inpainter = InPainting(self.preferred_device, self.torch_dtype, InPaintingType.FLUX)
        self.advance_progress(task)
        results: list[tuple[int, dict, PILImage.Image, np.ndarray]] = []
        try:
            for n, (idx, m, canvas, mask, visible, masked, n_occ, touches, in_context) in enumerate(jobs):
                cls = normalize_category(m.get("class")).replace("_", " ")
                setting = "in its real surroundings" if in_context else "centered on a plain white background"
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
                results.append((idx, m, out.convert("RGB"), visible, np.asarray(mask) > 127))
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
            for idx, m, out, visible, painted in results:
                seg = self._foreground.segment(Image(out))
                if seg.is_empty():
                    rejected += 1
                    continue
                mask = np.asarray(seg.masks[0], dtype=bool)
                # The object is what the photograph showed plus what was painted for
                # it, and nothing that is not connected to the photographed pixels:
                # a figure the model added beside the object is not the object.
                mask = _object_only(mask, visible, painted, cfg.occluder_dilate_px)
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
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(COMPLETION_SUMMARY_KEY) is not None

    def model_names(self) -> list[str]:
        from pipeline.segmentation.foreground_segmentation import ForegroundSeg
        return InPainting.model_names(InPaintingType.FLUX) + ForegroundSeg.model_names()

    def clean_up(self):
        if self._inpainter is not None:
            self._inpainter.close()
            self._inpainter = None
        if self._foreground is not None:
            self._foreground.unload()
            self._foreground = None
        super().clean_up()
