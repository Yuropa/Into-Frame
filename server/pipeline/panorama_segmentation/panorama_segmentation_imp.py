from path_utils import add_project_paths
add_project_paths()

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, SegformerImageProcessor, SegformerForSemanticSegmentation

from remote_connection.remote_server import RemoteServer
from util.panorama_tiling import Tile as _Tile, build_tiles as _build_tiles, extract_tile as _extract_tile


_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"
# Both are ADE20K semantic models with the same 150 labels, so everything
# downstream (the label -> region-type rules) is unchanged by the choice.
# Mask2Former Swin-L is ~7 mIoU better than SegFormer-B5 on ADE20K validation
# and draws far cleaner region boundaries.
_MODEL_IDS = {
    "segformer": "nvidia/segformer-b5-finetuned-ade-640-640",
    "mask2former": "facebook/mask2former-swin-large-ade-semantic",
}

# Tile size matches the model's native trained resolution (640×640) exactly,
# so a tile needs no resizing by the processor and the model sees genuine
# full-resolution pixels — unlike segmenting the whole (multi-thousand-pixel-
# wide) panorama in one pass, where the processor's forced square resize
# squashes it down to 640×640 and the ~160×160 output logits get stretched
# back over the whole panorama, blurring anything smaller than a few dozen
# pixels (tree branches, ridgelines) into whatever's next to it.
_TILE_SIZE = 640
# Fraction of a tile's size that neighbouring tiles overlap by. Needs to be
# wide enough that the feather window (see _feather_window) has room to taper
# before the next tile takes over, or tile-grid seams show up in the result.
_TILE_OVERLAP_FRAC = 0.25
# Tiles per forward pass. Bounds peak GPU memory; a panorama with more tiles
# than this is simply processed in multiple batches.
_MAX_BATCH = 8

# _Tile/_build_tiles/_extract_tile now live in util.panorama_tiling (shared with
# SAM2/Grounding DINO panorama tiling -- see image_segmentation_imp.py,
# grounding_dino_imp.py); re-imported above under their original local names so
# nothing below this line needs to change.


def _feather_window(h: int, w: int) -> np.ndarray:
    """
    Smooth 2D weight peaking at the tile's centre and tapering toward its
    edges, so where tiles overlap, each pixel is taken from whichever tile is
    most confident about it (closest to that tile's own centre) rather than
    handed off at a hard tile-grid boundary.
    """
    wy = np.hanning(h) if h > 1 else np.ones(1)
    wx = np.hanning(w) if w > 1 else np.ones(1)
    window = np.outer(wy, wx).astype(np.float32)
    # np.hanning goes to exactly 0 at the edges — floor it so a tile can still
    # win at its own edge pixels where no neighbouring tile reaches at all
    # (the outermost rows/columns of the whole panorama).
    return np.clip(window, 0.05, 1.0)


def _paste_tiles(
    canvases: list[np.ndarray],
    tile_values: list[np.ndarray],
    weight_canvas: np.ndarray,
    tile_weight: np.ndarray,
    tile: _Tile,
) -> None:
    """
    Paste one or more per-tile value canvases (label, confidence, runner-up
    label, ...) into their matching full-resolution canvases, all under the
    same per-pixel "does this tile beat what's already there" mask.

    Every canvas for a given tile must be pasted in one call here: `better`
    is computed from weight_canvas before any of it is mutated, and
    weight_canvas itself is only updated afterwards -- so all canvases stay
    mutually consistent about which tile won each pixel. Calling this
    against the same weight_canvas a second time for a different value would
    silently drop that value wherever an earlier call already won (the
    second call's `better` would compare against weight_canvas already
    holding the winning tile's own weight).
    """
    width = weight_canvas.shape[1]
    cols = np.arange(tile.x0, tile.x0 + tile.w) % width
    rows = np.arange(tile.y0, tile.y0 + tile.h)
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

    current_weight = weight_canvas[row_grid, col_grid]
    better = tile_weight > current_weight

    for canvas, tile_value in zip(canvases, tile_values):
        canvas[row_grid, col_grid] = np.where(better, tile_value, canvas[row_grid, col_grid])

    weight_canvas[row_grid, col_grid] = np.where(better, tile_weight, current_weight)


# Blend overlapping tiles by accumulating weighted class probabilities and taking
# the argmax once, at the end, instead of letting the highest-feather tile win each
# pixel outright.
#
# Winner-take-all does not remove a tile seam, it MOVES it. The feather weight is
# highest at a tile's own centre, so the pixel where the winner flips from tile A to
# tile B is the midpoint between their two centres -- a perfectly straight vertical
# line, at which the label changes discontinuously wherever the two tiles disagree.
# Measured on the 2026-08-25 run at _TILE_SIZE 640 / _TILE_OVERLAP_FRAC 0.25 (stride
# 480, centres at 320 + 480k, midpoints at 560 + 480k): straight label edges in the
# nadir half sit at columns 560, 1040, 1520, 2000, 2480 and 3440 on Iceland, Shark
# Fin and Irises alike, and Rainier's single seam is at 3440. Every one of those is
# a feather crossover, not a region boundary.
#
# Accumulating instead makes the handover continuous: near the midpoint both tiles
# contribute about equally, so the winning class changes where the EVIDENCE changes
# rather than where the geometry does.
_BLEND_TILE_PROBABILITIES = True
# Ceiling on how many distinct classes the accumulator will track. Each one costs a
# full-resolution float32 canvas (~34 MB at 2048x4096), and only classes that reach
# some tile's top-2 are ever allocated -- measured at 21-34 distinct classes per
# capture, so this is roughly 4x headroom rather than a limit expected to bind. Past
# it, further new classes fall back to competing on their own weighted score alone,
# which is the old behaviour for those classes only.
_MAX_ACCUMULATED_CLASSES = 128


class _ProbabilityAccumulator:
    """
    Per-class weighted-probability canvases, argmaxed once when every tile is in.

    Sparse by class: SegFormer's ADE20K head has 150 classes and a dense
    (H, W, 150) float32 accumulator is 5 GB at panorama resolution, but a single
    capture only ever puts a couple of dozen of them in a tile's top-2. Canvases are
    therefore allocated lazily, on first sight of a class.

    Only each tile's top-2 classes are accumulated, not its full softmax -- that is
    what the tile loop already computes, and a class that never places in any tile's
    top two cannot win a blended argmax either. The consequence is that a class
    sitting consistently third contributes nothing, which is the intended tradeoff:
    it could not have been the answer anyway.
    """

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.scores: dict[int, np.ndarray] = {}
        # Sum of the feather weights of every tile covering each pixel. Dividing the
        # accumulated scores by it turns them back into a weighted MEAN of the
        # per-tile softmax probabilities, on the same 0..1 scale the old top-1 value
        # had. That scale is load-bearing: downstream reads confidence and runner-up
        # confidence as softmax probabilities and thresholds their difference as a
        # margin (see ambiguity_strategy_for_label), so rescaling them -- e.g.
        # normalising the pair to sum to 1 -- would silently move every one of those
        # thresholds.
        self.weight = np.zeros((height, width), dtype=np.float32)
        self.overflow = 0

    def add(self, tile: _Tile, tile_weight: np.ndarray,
            ids: list[np.ndarray], vals: list[np.ndarray]) -> None:
        cols = np.arange(tile.x0, tile.x0 + tile.w) % self.width
        rows = np.arange(tile.y0, tile.y0 + tile.h)
        row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
        np.add.at(self.weight, (row_grid, col_grid), tile_weight.astype(np.float32))

        for tile_ids, tile_vals in zip(ids, vals):
            weighted = (tile_vals * tile_weight).astype(np.float32)
            for class_id in np.unique(tile_ids):
                key = int(class_id)
                canvas = self.scores.get(key)
                if canvas is None:
                    if len(self.scores) >= _MAX_ACCUMULATED_CLASSES:
                        self.overflow += 1
                        continue
                    canvas = np.zeros((self.height, self.width), dtype=np.float32)
                    self.scores[key] = canvas
                hit = tile_ids == class_id
                # np.add.at rather than plain += : a tile straddling the wrap seam
                # has repeated column indices, and fancy-index += would apply only
                # the last write instead of summing them.
                np.add.at(canvas, (row_grid[hit], col_grid[hit]), weighted[hit])

    def add_points(self, rows: np.ndarray, cols: np.ndarray, weight: np.ndarray,
                   ids: list[np.ndarray], vals: list[np.ndarray]) -> None:
        """Like add(), for an arbitrary set of equirect pixels (a projected view)."""
        rows = np.asarray(rows, dtype=np.int64); cols = np.asarray(cols, dtype=np.int64)
        weight = np.asarray(weight, dtype=np.float32)
        np.add.at(self.weight, (rows, cols), weight)
        for point_ids, point_vals in zip(ids, vals):
            weighted = (np.asarray(point_vals, dtype=np.float32) * weight).astype(np.float32)
            for class_id in np.unique(point_ids):
                key = int(class_id)
                canvas = self.scores.get(key)
                if canvas is None:
                    if len(self.scores) >= _MAX_ACCUMULATED_CLASSES:
                        self.overflow += 1
                        continue
                    canvas = np.zeros((self.height, self.width), dtype=np.float32)
                    self.scores[key] = canvas
                hit = point_ids == class_id
                np.add.at(canvas, (rows[hit], cols[hit]), weighted[hit])

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(label, confidence, runner-up label, runner-up confidence)."""
        # Running maxima over the per-class canvases rather than np.stack + argmax:
        # stacking 34 classes at 2048x4096 is a 1.1 GB temporary, and masking the
        # winner out for the second pass copies it again. Two sequential sweeps cost
        # the same arithmetic and hold only a handful of (H, W) arrays at once --
        # measured 4.37 GB -> 1.4 GB peak RSS on a full-resolution panorama, which
        # matters because this runs in the same process as the segmentation model.
        shape = (self.height, self.width)
        class_ids = sorted(self.scores)
        if not class_ids:
            # No tile ever contributed a class. Not reachable with a real model
            # (every tile's top-2 yields two), but finalize() is the only thing
            # standing between an empty accumulator and an IndexError on lookup[best]
            # partway through a multi-hour run, so it returns an empty labelling
            # rather than raising.
            return (
                np.zeros(shape, dtype=np.int16), np.zeros(shape, dtype=np.float32),
                np.zeros(shape, dtype=np.int16), np.zeros(shape, dtype=np.float32),
            )
        lookup = np.asarray(class_ids, dtype=np.int16)

        best_score = np.full(shape, -np.inf, dtype=np.float32)
        best = np.zeros(shape, dtype=np.int16)
        for idx, class_id in enumerate(class_ids):
            canvas = self.scores[class_id]
            wins = canvas > best_score
            best_score = np.where(wins, canvas, best_score)
            best = np.where(wins, np.int16(idx), best)

        second_score = np.full(shape, -np.inf, dtype=np.float32)
        second = np.zeros(shape, dtype=np.int16)
        for idx, class_id in enumerate(class_ids):
            # Skip only the pixels this class actually won, not the whole plane --
            # a class that is the winner somewhere is still a legitimate runner-up
            # everywhere else.
            canvas = np.where(best == idx, -np.inf, self.scores[class_id])
            wins = canvas > second_score
            second_score = np.where(wins, canvas, second_score)
            second = np.where(wins, np.int16(idx), second)

        best_score[~np.isfinite(best_score)] = 0.0
        second_score[~np.isfinite(second_score)] = 0.0

        # Back to the per-tile softmax scale -- see self.weight.
        coverage = np.where(self.weight > 0, self.weight, 1.0)
        return (
            lookup[best],
            (best_score / coverage).astype(np.float32),
            lookup[second],
            (second_score / coverage).astype(np.float32),
        )



# ── Perspective views ────────────────────────────────────────────────────────
#
# ADE20K is a dataset of ordinary perspective photographs. A 640 px square cut
# straight out of an equirectangular panorama looks like one only near the
# horizon; below it the projection stretches the ground into arcs no photo ever
# contains, and on the Rainier capture the model called the entire near-field
# ground, on both sides of the camera, "water". So the panorama is rendered
# into pinhole views (a ring of yaws at a few pitches), the model runs on those,
# and every view's per-pixel class probabilities are projected back onto the
# equirect and accumulated exactly as the overlapping tiles were.
#
# Direction convention is the panorama's own (pano_utils.map_image_to_pano):
# +z forward at the panorama's centre column, +x right, +y DOWN (row 0 is the
# zenith), phi = atan2(x, z), theta = asin(y), u = (phi/pi + 1)/2, v = theta/pi + 0.5.


def _view_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """World-from-view rotation: pitch (positive = look up) then yaw (positive =
    turn right, i.e. toward +x)."""
    p, q = np.radians(pitch_deg), np.radians(yaw_deg)
    # y is down, so looking up tilts forward toward -y.
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]], dtype=np.float64)
    ry = np.array([[np.cos(q), 0, np.sin(q)], [0, 1, 0], [-np.sin(q), 0, np.cos(q)]], dtype=np.float64)
    return ry @ rx


def _equirect_uv(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi = np.arctan2(d[..., 0], d[..., 2])
    theta = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    return (phi / np.pi + 1.0) / 2.0, theta / np.pi + 0.5


def _equirect_dirs(h: int, w: int) -> np.ndarray:
    """Unit direction of every equirect pixel centre, (h, w, 3)."""
    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    uu, vv = np.meshgrid(u, v)
    phi = (uu * 2.0 - 1.0) * np.pi
    theta = (vv - 0.5) * np.pi
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta), np.cos(theta) * np.cos(phi)], axis=-1)


def _render_view(pano: np.ndarray, rot: np.ndarray, fov_deg: float, size: int) -> np.ndarray:
    """Pinhole render of the equirect `pano` (H, W, 3) uint8 along `rot`."""
    import cv2
    H, W = pano.shape[:2]
    t = np.tan(np.radians(fov_deg) / 2.0)
    xs = ((np.arange(size, dtype=np.float64) + 0.5) / size * 2.0 - 1.0) * t
    xx, yy = np.meshgrid(xs, xs)
    d = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    d = d @ rot.T
    u, v = _equirect_uv(d)
    # One wrapped column on each side so the seam samples across it.
    padded = np.concatenate([pano[:, -1:], pano, pano[:, :1]], axis=1)
    map_x = (u * W + 0.5).astype(np.float32)          # +1 for the pad, -0.5 for pixel centres
    map_y = (v * H - 0.5).astype(np.float32)
    return cv2.remap(padded, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _feather_from_normalised(nx: np.ndarray, ny: np.ndarray) -> np.ndarray:
    """Weight for a view pixel at normalised image coordinates in [-1, 1]: 1 at
    the view's centre, tapering to the floor at its edges (the tile feather's
    role, so overlapping views hand over where the evidence changes)."""
    r = np.maximum(np.abs(nx), np.abs(ny))
    return np.clip(np.cos(np.pi / 2.0 * np.clip(r, 0.0, 1.0)) ** 2, 0.05, 1.0).astype(np.float32)


class PanoramaSegmentationServer(RemoteServer):
    def setup(self):
        self.model_name = None
        self.processor = None
        self.model = None
        self._loaded: dict[str, tuple] = {}

    def _load(self, model_name: str) -> None:
        if self.model is not None and self.model_name == model_name:
            return
        loaded = getattr(self, "_loaded", None)
        if loaded is None:
            loaded = self._loaded = {}
        if model_name not in loaded:
            model_id = _MODEL_IDS.get(model_name, model_name)
            if model_name == "mask2former":
                from transformers import Mask2FormerForUniversalSegmentation
                processor = AutoImageProcessor.from_pretrained(model_id)
                model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
            else:
                processor = SegformerImageProcessor.from_pretrained(model_id)
                model = SegformerForSemanticSegmentation.from_pretrained(model_id)
            model.to(self.device)
            model.eval()
            loaded[model_name] = (processor, model)
        self.processor, self.model = loaded[model_name]
        self.model_name = model_name

    def perform(self, action: str, temp_path: Path, input: Any) -> Any:
        if action == "segment":
            if isinstance(input, dict):
                return self._segment(
                    input["panorama"], input.get("photo"), input.get("photo_hfov_deg"),
                    input.get("model", "segformer"), input.get("views") or {},
                )
            return self._segment(input, None, None, "segformer", {})
        raise ValueError(f"Unknown action: {action}")

    # ── per-image class probabilities ────────────────────────────────────────

    def _probabilities(self, images: list[Image.Image]) -> torch.Tensor:
        """(B, C, h, w) per-pixel class probabilities at each image's own size.
        All images in a batch must share one size."""
        w, h = images[0].size
        if self.model_name == "mask2former":
            inputs = self.processor(
                images=images, return_tensors="pt", do_resize=False,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            cls = torch.softmax(outputs.class_queries_logits, dim=-1)[..., :-1]      # (B, Q, C)
            masks = torch.sigmoid(outputs.masks_queries_logits)                      # (B, Q, h', w')
            seg = torch.einsum("bqc,bqhw->bchw", cls, masks)
            seg = torch.nn.functional.interpolate(seg, size=(h, w), mode="bilinear", align_corners=False)
            return seg / seg.sum(dim=1, keepdim=True).clamp_min(1e-6)
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = torch.nn.functional.interpolate(outputs.logits, size=(h, w), mode="bilinear", align_corners=False)
        return torch.softmax(logits, dim=1)

    @staticmethod
    def _top2(probs: torch.Tensor):
        """(label, conf, runner-up, runner-up conf) numpy maps for one (C, h, w)."""
        val, idx = probs.topk(2, dim=0)
        return (
            idx[0].cpu().numpy().astype(np.int16), val[0].cpu().numpy().astype(np.float32),
            idx[1].cpu().numpy().astype(np.int16), val[1].cpu().numpy().astype(np.float32),
        )

    # ── the two strategies ──────────────────────────────────────────────────

    def _segment(self, image: Image.Image, photo, photo_hfov_deg, model_name: str, views: dict) -> dict:
        self._load(model_name)
        orig_w, orig_h = image.size
        if int(views.get("count", 0)) <= 0:
            label, conf, runnerup, runnerup_conf = self._segment_tiles(image)
        else:
            label, conf, runnerup, runnerup_conf = self._segment_views(image, photo, photo_hfov_deg, views)
            check = views.get("water_check_model")
            if check and check != model_name:
                label, conf, runnerup, runnerup_conf = self._check_water(
                    image, photo, photo_hfov_deg, views, check, model_name,
                    label, conf, runnerup, runnerup_conf,
                )
        self.report_progress(0.95, "Done")
        id2label = self.model.config.id2label
        return {
            "label_map": label.tolist(),
            "confidence_map": conf,
            "runnerup_label_map": runnerup,
            "runnerup_confidence_map": runnerup_conf,
            "id2label": {str(k): v for k, v in id2label.items()},
            "width": orig_w,
            "height": orig_h,
        }

    def _segment_views(self, image: Image.Image, photo, photo_hfov_deg, views: dict):
        """Render pinhole views, segment each, accumulate back on the equirect.
        The hero photograph, when given, is one more view: the real pixels at
        the panorama's centre, weighted `photo_weight` times a rendered view."""
        W, H = image.size
        pano = np.array(image.convert("RGB"))
        count = int(views.get("count", 8))
        fov = float(views.get("fov_deg", 90.0))
        size = int(views.get("size_px", 768))
        pitches = [float(p) for p in views.get("pitches_deg", [50.0, 0.0, -50.0])]
        photo_weight = float(views.get("photo_weight", 2.0))
        batch_size = int(views.get("batch", 4))

        specs = [(360.0 * k / count, pitch) for pitch in pitches for k in range(count)]
        acc = _ProbabilityAccumulator(H, W)
        dirs = _equirect_dirs(H, W).reshape(-1, 3)
        t = np.tan(np.radians(fov) / 2.0)
        total = len(specs) + (1 if photo is not None else 0)
        done = 0
        self.report_progress(0.05, f"Segmenting {len(specs)} view(s)…")

        for b0 in range(0, len(specs), batch_size):
            batch = specs[b0: b0 + batch_size]
            rots = [_view_rotation(yaw, pitch) for yaw, pitch in batch]
            renders = [Image.fromarray(_render_view(pano, rot, fov, size)) for rot in rots]
            probs = self._probabilities(renders)
            for i, rot in enumerate(rots):
                label, conf, runnerup, runnerup_conf = self._top2(probs[i])
                # Equirect pixels inside this view, and where they land in it.
                local = dirs @ rot                       # rot^T applied to each row
                z = local[:, 2]
                front = z > 1e-6
                nx = np.zeros(len(z)); ny = np.zeros(len(z))
                nx[front] = local[front, 0] / z[front] / t
                ny[front] = local[front, 1] / z[front] / t
                inside = front & (np.abs(nx) < 1.0) & (np.abs(ny) < 1.0)
                idx = np.nonzero(inside)[0]
                col = np.clip(((nx[idx] + 1.0) / 2.0 * size).astype(np.int64), 0, size - 1)
                row = np.clip(((ny[idx] + 1.0) / 2.0 * size).astype(np.int64), 0, size - 1)
                weight = _feather_from_normalised(nx[idx], ny[idx])
                acc.add_points(
                    idx // W, idx % W, weight,
                    [label[row, col], runnerup[row, col]],
                    [conf[row, col], runnerup_conf[row, col]],
                )
            done += len(batch)
            self.report_progress(0.05 + 0.8 * done / total, f"Segmenting views… ({done}/{total})")

        if photo is not None and photo_hfov_deg:
            # The photo is embedded with WorldGen's linear-in-angle mapping
            # (util/captured_cone.py): longitude = (px/w - 0.5) * hfov, latitude =
            # (py/h - 0.5) * vfov. Rows/cols of the equirect inside that cone map
            # straight to photo pixels.
            pw, ph = photo.size
            short = min(pw, ph)
            scale = max(1.0, size / short)              # never downsample the real pixels
            tw, th = int(round(pw * scale / 32)) * 32, int(round(ph * scale / 32)) * 32
            resized = photo.convert("RGB").resize((tw, th), Image.LANCZOS)
            probs = self._probabilities([resized])
            label, conf, runnerup, runnerup_conf = self._top2(probs[0])
            hfov = float(photo_hfov_deg)
            vfov = np.degrees(2.0 * np.arctan(np.tan(np.radians(hfov) / 2.0) * ph / pw))
            lon = ((np.arange(W) + 0.5) / W - 0.5) * 360.0
            lat = ((np.arange(H) + 0.5) / H - 0.5) * 180.0
            px = (lon / hfov + 0.5)                     # normalised [0, 1] across the photo
            py = (lat / vfov + 0.5)
            cols_in = np.nonzero((px >= 0.0) & (px < 1.0))[0]
            rows_in = np.nonzero((py >= 0.0) & (py < 1.0))[0]
            if cols_in.size and rows_in.size:
                rr, cc = np.meshgrid(rows_in, cols_in, indexing="ij")
                pcol = np.clip((px[cc] * tw).astype(np.int64), 0, tw - 1)
                prow = np.clip((py[rr] * th).astype(np.int64), 0, th - 1)
                weight = photo_weight * _feather_from_normalised(px[cc] * 2.0 - 1.0, py[rr] * 2.0 - 1.0)
                acc.add_points(
                    rr.ravel(), cc.ravel(), weight.ravel(),
                    [label[prow, pcol].ravel(), runnerup[prow, pcol].ravel()],
                    [conf[prow, pcol].ravel(), runnerup_conf[prow, pcol].ravel()],
                )
            self.report_progress(0.9, "Photograph segmented")

        out = acc.finalize()
        if acc.overflow:
            self.report_progress(0.92, f"view blending: {acc.overflow} class sighting(s) past the ceiling were dropped")
        return out

    def _check_water(self, image, photo, photo_hfov_deg, views, check_model, model_name,
                     label, conf, runnerup, runnerup_conf):
        """Second opinion on water only. Mask2Former reads gravel and packed-dirt
        paths as water: on Rainier the gravel ring round the camera and the trail
        to the pond were 21% of the panorama as WATER, and reached the render as a
        murky sheet underfoot with no trail. SegFormer, over the same views, calls
        that ground "path"/"road" and the pond beside it still "water". So a pixel
        stays water only where the check model agrees; elsewhere the check model's
        own label replaces it, with water kept as the runner-up."""
        water_names = {"water", "sea", "river", "lake", "swimming pool", "waterfall", "fountain"}
        id2label = self.model.config.id2label
        water_ids = np.array([int(k) for k, v in id2label.items() if str(v).lower() in water_names], dtype=np.int16)
        is_water = np.isin(label, water_ids)
        if not is_water.any():
            return label, conf, runnerup, runnerup_conf
        self._load(check_model)
        c_label, c_conf, _, _ = self._segment_views(image, photo, photo_hfov_deg, views)
        self._load(model_name)
        veto = is_water & ~np.isin(c_label, water_ids)
        # What one model takes for water and the other for "road" is loose gravel or
        # packed dirt, not a paved road: Rainier's gravel ring would otherwise be
        # textured as asphalt in a meadow.
        name_to_id = {str(v).lower(): int(k) for k, v in id2label.items()}
        if "road" in name_to_id and "path" in name_to_id:
            c_label = np.where(veto & (c_label == name_to_id["road"]), np.int16(name_to_id["path"]), c_label)
        self.report_progress(
            0.9, f"water check ({check_model}): {int(veto.sum())} of {int(is_water.sum())} water px not water",
        )
        return (
            np.where(veto, c_label, label).astype(label.dtype),
            np.where(veto, c_conf, conf).astype(conf.dtype),
            np.where(veto, label, runnerup).astype(runnerup.dtype),
            np.where(veto, conf, runnerup_conf).astype(runnerup_conf.dtype),
        )

    def _segment_tiles(self, image: Image.Image):
        """The original strategy: overlapping equirect tiles at the model's native size."""
        orig_w, orig_h = image.size
        img_arr = np.array(image.convert("RGB"))
        tiles = _build_tiles(orig_w, orig_h, _TILE_SIZE, _TILE_OVERLAP_FRAC)
        self.report_progress(0.05, f"Segmenting {len(tiles)} tile(s)…")
        accumulator = _ProbabilityAccumulator(orig_h, orig_w)
        for batch_start in range(0, len(tiles), _MAX_BATCH):
            batch = tiles[batch_start: batch_start + _MAX_BATCH]
            crops = [Image.fromarray(_extract_tile(img_arr, t)) for t in batch]
            probs = self._probabilities(crops)
            for i, tile in enumerate(batch):
                label, conf, runnerup, runnerup_conf = self._top2(probs[i])
                accumulator.add(tile, _feather_window(tile.h, tile.w), [label, runnerup], [conf, runnerup_conf])
            done = min(len(tiles), batch_start + len(batch))
            self.report_progress(0.05 + 0.85 * done / len(tiles), f"Segmenting tiles… ({done}/{len(tiles)})")
        return accumulator.finalize()


if __name__ == "__main__":
    PanoramaSegmentationServer.run()
