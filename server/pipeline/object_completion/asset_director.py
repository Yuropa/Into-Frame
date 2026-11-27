"""Asset Director: the agent gate on what each reconstructed object is drawn as.

Object Completion redraws every bucket representative as a clean reference view (the
whole object, isolated, side on) and the reconstructor builds the mesh from that
picture, so the reference prompt decides what stands in the scene. Until now the
prompt came from a BLIP caption of the crop, a CLIP choice among a hand-written list
of kinds, and a CLIP similarity pick among the candidates. Paris, opus2: the plane
trees on the quays were drawn as a fir, the stone arch bridge as a rail viaduct with a
train on it, the crowded bateau-mouche as an empty cruiser placed 16 m long.

Two calls replace those heuristics (docs/agentic-scene-graph-plan.md §13a, first-pass
gates):

  direct()  one call per run: the photograph with the chosen crops boxed and numbered,
            a sheet of the crops, the scene plan's inventory. Per crop: what the whole
            real object is, in words an image model can draw; its real length and
            height; whether to follow the crop's pixels or the description; people.
  pick()    one call per crop after the candidates are drawn: the crop beside the
            numbered candidates; the candidate that is the photographed object, or
            none with a corrected description for one more round.

Cached across runs in cache/asset_director/ by (photo, crops, model) and by
(crop, candidates, model), so a rerun that reproduces the same crops costs nothing.
Without a key, with no_api, or past the spend cap the stage falls back to the
heuristics it had.
"""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw, ImageFont

from pipeline.scene_plan.client import ModelReply, NoApiError, PlanModel, SpendCapError, SpendLedger

_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 2

DIRECT_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["objects", "summary"],
    "properties": {
        "objects": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["idx", "what", "subject", "form", "length_m", "height_m", "follow", "people", "why"],
            "properties": {
                "idx": {"type": "integer", "description": "the crop number"},
                "what": {"type": "string", "description": "what the object is, in two to five words (open-deck tour boat, stone arch bridge, London plane tree)"},
                "subject": {"type": "string", "description": "the whole real object described for an image model in one sentence of 12 to 35 words: kind, build, materials, colours, the parts that make it this one. No setting (no river, sky, ground), no camera words."},
                "form": {"type": "string", "enum": ["conifer", "broadleaf", "palm", "shrub", "flower", "long", "tall", "compact"],
                         "description": "trees: conifer/broadleaf/palm; a bush: shrub; a flowering plant: flower; anything else by its silhouette side on: long (at least twice as long as tall: boats, bridges, trains), tall (taller than wide: towers, statues, poles), compact"},
                "length_m": {"type": ["number", "null"], "description": "real length of the whole object in metres (its longest horizontal extent), or null if unknown"},
                "height_m": {"type": ["number", "null"], "description": "real height of the whole object in metres, or null if unknown"},
                "follow": {"type": "string", "enum": ["crop", "description"],
                           "description": "crop: the crop shows the object clearly from a usable side and the drawing should keep its look; description: the crop is tiny, blurred, a fragment, or seen end-on or from above, so draw from the words"},
                "people": {"type": "string", "enum": ["none", "crowd", "few"],
                           "description": "people aboard or on it in the photograph that belong to its look (a tour boat's packed deck: crowd). none for everything else."},
                "why": {"type": "string", "description": "one sentence: what in the photograph shows it"},
            }}},
        "summary": {"type": "string"},
    },
}

DIRECT_SYSTEM = """You direct the drawing of reference pictures for a 3D scene built from one photograph.
Image 1 is the photograph with the chosen objects boxed and numbered. Image 2 is a sheet of those crops, numbered the
same. Each numbered object will be redrawn by an image model as one clean picture of the WHOLE object, isolated on white
and seen from the side, and a 3D mesh is reconstructed from that picture and stood in the scene at the object's place.
One mesh often stands for every object of its kind in the scene (one tree for the tree line, one barge for the moored
barges), so describe the typical one the box shows.

For each number say what the real object is and describe it for the image model. Look at the photograph, not only the
crop: the crop may be a fragment, a few pixels, or a view from above, and the photograph and the scene plan say what it
is (which trees line this quay, which bridge this is, what kind of boat). Name real species, real landmark names and real
vessel types when you can see them. Give the real-world length and height of the whole object in metres; these scale the
mesh, so a 45 m tour boat must not be 12 m. Do not describe the setting, the lighting or the camera.

`follow`: say `crop` only when the crop itself is a clear view the drawing should keep. `people`: a mesh with people on
it carries them forever, so say `crowd` or `few` only when they are part of what the object looks like in the photograph."""

PICK_SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["choice", "why", "revise"],
    "properties": {
        "choice": {"type": "integer", "description": "the number of the candidate to build from, or -1 if none is acceptable"},
        "why": {"type": "string", "description": "one sentence"},
        "revise": {"type": "string", "description": "when choice is -1: a corrected `subject` sentence for another round, naming what the candidates got wrong; else empty"},
    },
}

PICK_SYSTEM = """You choose the reference picture a 3D object is built from. The image is one sheet: on the left, under
"photo", the object as photographed (a crop, possibly small, partial or seen from above) and where it sits in the whole
photograph; to the right the numbered candidate drawings. The mesh reconstructed from the chosen candidate will stand in
the scene as this object, so choose the candidate that is the same kind of thing with the same build and colours, drawn
whole, alone, side on, with nothing extra (no second object, no ground slab, no water, no text). Small differences in
detail do not matter; a different kind of object does (a fir for a plane tree, a rail viaduct for a stone arch bridge, a
tug for a tour boat). If no candidate is the photographed kind of object, answer -1 and write the corrected description."""


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def _on_white(im: Image.Image) -> Image.Image:
    rgba = im.convert("RGBA")
    out = Image.new("RGB", rgba.size, (255, 255, 255))
    out.paste(rgba, (0, 0), rgba)
    return out


def boxed_photo(photo: Image.Image, rows: list[tuple[int, dict]], factor: float = 1.0) -> Image.Image:
    """The photograph with each chosen crop's box and number (photo_box is in the
    photograph's own pixels; `factor` is unused and kept for callers)."""
    im = photo.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(2, round(min(W, H) / 350))
    font = _font(max(16, round(min(W, H) / 40)))
    palette = [(255, 60, 60), (60, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 255), (255, 150, 60)]
    for n, (idx, m) in enumerate(rows):
        box = m.get("photo_box")
        if not box or len(box) != 4:
            continue
        x, y, w, h = (float(v) for v in box)
        col = palette[n % len(palette)]
        d.rectangle([x, y, x + w, y + h], outline=col, width=lw)
        label = str(idx)
        tw, th = d.textbbox((0, 0), label, font=font)[2:]
        tx = min(max(0.0, x), W - tw - 6)
        ty = y - th - 6 if y - th - 6 >= 0 else y + 2
        d.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=col)
        d.text((tx + 3, ty + 1), label, fill=(0, 0, 0), font=font)
    return im


def crop_sheet(crops: dict[int, Image.Image], cell: int = 300, cols: int = 5) -> Image.Image:
    idxs = list(crops)
    rows = (len(idxs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + 28)), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = _font(20)
    for n, idx in enumerate(idxs):
        im = _on_white(crops[idx])
        s = min((cell - 12) / im.width, (cell - 12) / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        cx, cy = (n % cols) * cell, (n // cols) * (cell + 28)
        d.text((cx + 6, cy + 2), f"{idx}  ({crops[idx].width}x{crops[idx].height} px)", fill=(0, 0, 0), font=font)
        sheet.paste(im, (cx + (cell - im.width) // 2, cy + 28 + (cell - im.height) // 2))
        d.rectangle([cx, cy, cx + cell - 1, cy + cell + 27], outline=(200, 200, 200))
    return sheet


def pick_sheet(photo: Image.Image, crop: Image.Image, box_px: Optional[tuple[float, float, float, float]],
               candidates: list[Image.Image], cell: int = 380) -> Image.Image:
    n = len(candidates)
    sheet = Image.new("RGB", (cell * (n + 1), cell + 34), (255, 255, 255))
    d = ImageDraw.Draw(sheet)
    font = _font(22)
    half = (cell - 12) // 2
    c = _on_white(crop)
    s = min((cell - 12) / c.width, half / c.height)
    c = c.resize((max(1, int(c.width * s)), max(1, int(c.height * s))), Image.LANCZOS)
    d.text((6, 4), "photo", fill=(0, 0, 0), font=font)
    sheet.paste(c, ((cell - c.width) // 2, 34 + (half - c.height) // 2))
    ph = photo.convert("RGB").copy()
    if box_px is not None:
        x, y, w, h = box_px
        ImageDraw.Draw(ph).rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=max(3, ph.width // 250))
    s = min((cell - 12) / ph.width, half / ph.height)
    ph = ph.resize((max(1, int(ph.width * s)), max(1, int(ph.height * s))), Image.LANCZOS)
    sheet.paste(ph, ((cell - ph.width) // 2, 34 + half + 6 + (half - ph.height) // 2))
    for k, cand in enumerate(candidates):
        im = _on_white(cand)
        s = min((cell - 12) / im.width, (cell - 12) / im.height)
        im = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.LANCZOS)
        x0 = cell * (k + 1)
        d.text((x0 + 6, 4), f"candidate {k}", fill=(0, 0, 0), font=font)
        sheet.paste(im, (x0 + (cell - im.width) // 2, 34 + (cell - im.height) // 2))
        d.line([x0, 0, x0, cell + 34], fill=(180, 180, 180), width=2)
    return sheet


def _image_sha(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.convert("RGBA").save(buf, "PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def plan_inventory(plan) -> str:
    if plan is None or getattr(plan, "empty", True):
        return "(no scene plan)"
    out = []
    place = plan.get("place") if hasattr(plan, "get") else None
    if isinstance(place, dict):
        out.append(f"place: {place.get('name') or place.get('kind')}")
    for lm in plan.items("landmarks"):
        out.append(f"landmark: {lm.get('name')} ({lm.get('class')}), {lm.get('height_m')} m tall, {lm.get('distance_m')} m away")
    for sp in plan.items("species"):
        out.append(f"species: {sp.get('name')} ({sp.get('class')}), about {sp.get('size_m')} m, {sp.get('form')}, {sp.get('where')}")
    for ob in plan.items("objects"):
        out.append(f"object: {ob.get('count')} x {ob.get('class')} ({ob.get('category')}), about {ob.get('size_m')} m, {ob.get('where')}")
    return "\n".join(out) or "(plan lists nothing)"


class AssetDirector:
    def __init__(self, model: str = "anthropic:claude-sonnet-5-5", cache_dir: Optional[Path] = None,
                 max_usd_per_run: float = 1.5, no_api: bool = False, refresh: bool = False, timeout_s: float = 600.0,
                 retries: int = 1, base_url: Optional[str] = None, key_service: str = "anthropic",
                 key_user: str = "3dcodeverse", log: Optional[Callable[[str], None]] = None,
                 warn: Optional[Callable[[str], None]] = None, out_dir: Optional[Path] = None):
        self.model_id = str(model)
        self.cache_dir = Path(cache_dir) if cache_dir else _ROOT / "cache" / "asset_director"
        self.ledger = SpendLedger(self.cache_dir, max_usd_per_run)
        self.refresh = bool(refresh)
        self.model = PlanModel(self.model_id, timeout_s=timeout_s, retries=retries, base_url=base_url,
                               key_service=key_service, key_user=key_user, no_api=no_api)
        self.log = log or (lambda s: None)
        self.warn = warn or self.log
        self.out_dir = out_dir
        self.dead = False     # a refused or failed call: stop asking for the rest of the run

    # ── cache ────────────────────────────────────────────────────────────────
    def _cached(self, key: str) -> Optional[dict]:
        f = self.cache_dir / f"{key}.json"
        if f.is_file() and not self.refresh:
            try:
                return json.loads(f.read_text())
            except ValueError:
                return None
        return None

    def _store(self, key: str, record: dict):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps(record, indent=2))

    def _call(self, what: str, image, system, user, tool, schema, extra=None, expected_usd: float = 0.1) -> Optional[ModelReply]:
        if self.dead:
            return None
        try:
            self.ledger.check(expected_usd, what)
            return self.model.read_photo(image, system, user, tool, schema, 6000, extra_images=extra,
                                         purpose="Record the decision.")
        except (NoApiError, SpendCapError) as exc:
            self.warn(f"Asset Director: {exc} -- falling back to the heuristics")
        except Exception as exc:
            self.warn(f"Asset Director: {what} failed ({type(exc).__name__}: {exc}) -- falling back to the heuristics")
        self.dead = True
        return None

    # ── direct ───────────────────────────────────────────────────────────────
    def direct(self, photo: Image.Image, photo_sha: str, rows: list[tuple[int, dict]], crops: dict[int, Image.Image],
               plan=None, factor: float = 1.0) -> dict[int, dict]:
        """{idx: direction} for the chosen crops; {} when no direction is available."""
        if not rows:
            return {}
        h = hashlib.sha256(f"direct:v{SCHEMA_VERSION}:{self.model_id}:{photo_sha}:".encode())
        for idx, m in rows:
            h.update(json.dumps([idx, m.get("class"), m.get("name"),
                                 [round(float(v), 1) for v in (m.get("photo_box") or [])]]).encode())
        key = "direct-" + h.hexdigest()[:40]
        record = self._cached(key)
        if record is not None:
            self.log(f"Asset Director: cached directions {key[:19]} (no call)")
        else:
            W, H = photo.size
            lines = []
            for idx, m in rows:
                box = m.get("photo_box")
                where = "outside the photograph's frame (found in the generated surround; see the crop sheet)"
                if box and len(box) == 4:
                    x, y, w, hh = (float(v) for v in box)
                    where = f"box=[{x / W:.3f}, {y / H:.3f}, {w / W:.3f}, {hh / H:.3f}] of the photograph"
                lines.append(f"{idx:4d}  detector class={m.get('class')}, name={str(m.get('name') or m.get('class'))!r}, {where}")
            user = (f"The chosen objects ({len(rows)}):\n" + "\n".join(lines)
                    + "\n\nThe scene plan's reading of this photograph:\n" + plan_inventory(plan)
                    + "\n\nGive one entry for every number.")
            annotated = boxed_photo(photo, rows, factor)
            sheet = crop_sheet({i: crops[i] for i, _ in rows if i in crops})
            if self.out_dir is not None:
                annotated.save(self.out_dir / "director_boxes.jpg", quality=90)
                sheet.save(self.out_dir / "director_crops.jpg", quality=90)
            reply = self._call("directions", annotated, DIRECT_SYSTEM, user, "asset_directions", DIRECT_SCHEMA,
                               extra=[sheet], expected_usd=0.15)
            if reply is None:
                return {}
            data = reply.data if isinstance(reply.data, dict) else {}
            record = {"schema_version": SCHEMA_VERSION, "model": f"{reply.provider}:{reply.model}",
                      "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "usage": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                                "usd": round(reply.usd, 4), "seconds": round(reply.seconds, 1)},
                      "objects": [o for o in data.get("objects", []) if isinstance(o, dict)],
                      "summary": str(data.get("summary", ""))}
            self.ledger.record("Asset Director", reply, key, "directions")
            self._store(key, record)
            self.log(f"Asset Director: {len(record['objects'])} directions from {record['model']} in {reply.seconds:.0f} s, "
                     f"~${reply.usd:.2f}; {record['summary']}")
        valid = {i for i, _ in rows}
        out: dict[int, dict] = {}
        for o in record.get("objects", []):
            try:
                idx = int(o.get("idx"))
            except (TypeError, ValueError):
                continue
            if idx in valid and str(o.get("subject") or "").strip():
                out[idx] = o
        return out

    # ── pick ─────────────────────────────────────────────────────────────────
    def pick(self, photo: Image.Image, crop: Image.Image, box_px, candidates: list[Image.Image],
             direction: Optional[dict], idx: int, round_no: int = 0) -> Optional[dict]:
        """{"choice": k or -1, "why", "revise"} or None when no decision is available."""
        if not candidates:
            return None
        h = hashlib.sha256(f"pick:v{SCHEMA_VERSION}:{self.model_id}:{_image_sha(crop)}:".encode())
        for c in candidates:
            h.update(_image_sha(c).encode())
        h.update(json.dumps((direction or {}).get("subject", "")).encode())
        if (direction or {}).get("nobody_aboard"):
            h.update(b"nobody-aboard")
        key = "pick-" + h.hexdigest()[:40]
        record = self._cached(key)
        if record is None:
            sheet = pick_sheet(photo, crop, box_px, candidates)
            if self.out_dir is not None:
                sheet.save(self.out_dir / f"director_pick_{idx}_r{round_no}.jpg", quality=90)
            d = direction or {}
            user = (f"The object is: {d.get('what') or 'see the photograph'}. The drawing was asked for as: "
                    f"{d.get('subject') or '(no description)'}\n"
                    f"Real size: length {d.get('length_m')} m, height {d.get('height_m')} m. "
                    f"There are {len(candidates)} candidates, numbered 0 to {len(candidates) - 1}.")
            if d.get("nobody_aboard"):
                # The people in the photograph are not built into the model (Paris: of three
                # redrawn tour boats the one still carrying its passengers was chosen as
                # "matching the photo", and the crowd was baked into the deck).
                user += ("\nThe scene is built without people: the drawing was asked for with nobody aboard. "
                         "A candidate that shows people is not to be chosen while another shows none.")
            reply = self._call(f"pick crop_{idx}", sheet, PICK_SYSTEM, user, "pick_reference", PICK_SCHEMA, expected_usd=0.05)
            if reply is None:
                return None
            data = reply.data if isinstance(reply.data, dict) else {}
            try:
                choice = int(data.get("choice", -1))
            except (TypeError, ValueError):
                choice = -1
            record = {"choice": choice if -1 <= choice < len(candidates) else -1, "why": str(data.get("why", "")),
                      "revise": str(data.get("revise", "")), "model": f"{reply.provider}:{reply.model}",
                      "usd": round(reply.usd, 4)}
            self.ledger.record("Asset Director", reply, key, f"pick crop_{idx}")
            self._store(key, record)
        return record
