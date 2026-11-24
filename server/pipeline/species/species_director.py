"""Species Director: the agent gate on how each plan species is built.

Species Assets builds a plant per plan species from a form (spike, raceme, seed head)
whose proportions were fixed in code: a spike's head took the top 38% of the plant and
was never narrower than half its length. Rainier: the bracted lousewort, a 0.7 m plant
with a 4 cm wide cream spike, stood in the meadow as a wall of 15 cm yellow cones that
hid the paintbrush the photograph is about.

One call per run reads the photograph with every plan species' crop hint boxed and
says, per species, what the plant measures as photographed: the flowering head's width
over its length, the share of the plant's height the head takes, the whole plant's
width over its height, and its standing height. Species Assets passes those to the
builder (docs/agentic-scene-graph-plan.md §13a, first-pass gates).

Cached in cache/species_director/ by (photo, species list, model). Without a key, with
no_api, or past the spend cap the forms keep their built-in proportions.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from PIL import Image, ImageDraw

from pipeline.object_completion.asset_director import AssetDirector, _font

_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = 1

SCHEMA: dict[str, Any] = {
    "type": "object", "additionalProperties": False, "required": ["species", "summary"],
    "properties": {
        "species": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["n", "head_aspect", "head_fraction", "plant_aspect", "height_m", "seen", "why"],
            "properties": {
                "n": {"type": "integer", "description": "the species number"},
                "head_aspect": {"type": "number", "description": "width of one flowering head (the spike, raceme, seed head or flower cluster on one stem) divided by its length along the stem, as photographed: a slender spike 0.15-0.3, a stout bracted head 0.5-0.8, a round head 1.0"},
                "head_fraction": {"type": "number", "description": "the share of the plant's standing height that the flowering head takes, 0.05-0.7"},
                "plant_aspect": {"type": "number", "description": "width of one whole plant with its leaves divided by its height, as photographed"},
                "height_m": {"type": "number", "description": "standing height of a typical plant of this species here, in metres"},
                "seen": {"type": "string", "enum": ["measured", "partly", "not_seen"],
                         "description": "measured: read off plants visible in the photograph; partly: small or occluded, read with the species' usual build; not_seen: from the species alone"},
                "why": {"type": "string", "description": "one sentence: which plants in the photograph the numbers were read from"},
            }}},
        "summary": {"type": "string"},
    },
}

SYSTEM = """You measure plants in one photograph for a procedural plant builder. The first image is the photograph with
the scene plan's species boxed and numbered where the plan saw them. For each numbered species give the proportions of
ONE typical plant as it stands in this photograph: how wide its flowering head is against the head's own length, how
much of the plant's height the head takes, how wide the whole plant is against its height, and its standing height.
Read them off the plants you can see; where a species is too small or hidden to measure, use the species' real build
and say so. The builder draws exactly these proportions, so a slender spike reported as stout fills the meadow with
clubs. Call the tool once with one entry per number."""


def boxed_species(photo: Image.Image, species: list[dict]) -> Image.Image:
    """The photograph with each species' crop hint ([x, y, w, h] as fractions) boxed and numbered."""
    im = photo.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    W, H = im.size
    lw = max(2, round(min(W, H) / 350))
    font = _font(max(16, round(min(W, H) / 40)))
    palette = [(255, 60, 60), (60, 200, 255), (255, 200, 0), (120, 255, 120), (255, 120, 255), (255, 150, 60)]
    for n, sp in enumerate(species):
        box = sp.get("crop_hint")
        if not box or len(box) != 4:
            continue
        x, y, w, h = float(box[0]) * W, float(box[1]) * H, float(box[2]) * W, float(box[3]) * H
        col = palette[n % len(palette)]
        d.rectangle([x, y, x + w, y + h], outline=col, width=lw)
        label = str(n + 1)
        tw, th = d.textbbox((0, 0), label, font=font)[2:]
        tx = min(max(0.0, x), W - tw - 6)
        ty = y - th - 6 if y - th - 6 >= 0 else y + 2
        d.rectangle([tx, ty, tx + tw + 6, ty + th + 4], fill=col)
        d.text((tx + 3, ty + 1), label, fill=(0, 0, 0), font=font)
    return im


def _clamp(v, lo, hi) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not (f == f) or f <= 0:
        return None
    return round(min(max(f, lo), hi), 3)


class SpeciesDirector(AssetDirector):
    def __init__(self, model: str = "anthropic:claude-sonnet-5-5", cache_dir: Optional[Path] = None,
                 max_usd_per_run: float = 0.5, no_api: bool = False, refresh: bool = False,
                 log: Optional[Callable[[str], None]] = None, warn: Optional[Callable[[str], None]] = None,
                 out_dir: Optional[Path] = None):
        super().__init__(model=model, cache_dir=Path(cache_dir) if cache_dir else _ROOT / "cache" / "species_director",
                         max_usd_per_run=max_usd_per_run, no_api=no_api, refresh=refresh, log=log, warn=warn,
                         out_dir=out_dir)

    def measure(self, photo: Image.Image, photo_sha: str, species: list[dict]) -> dict[str, dict]:
        """{species name: {head_aspect, head_fraction, plant_aspect, height_m, seen, why}}; {} without a reading."""
        if not species or photo is None:
            return {}
        h = hashlib.sha256(f"species:v{SCHEMA_VERSION}:{self.model_id}:{photo_sha}:".encode())
        for sp in species:
            h.update(json.dumps([sp.get("name"), sp.get("class"), sp.get("crop_hint")]).encode())
        key = "species-" + h.hexdigest()[:40]
        record = self._cached(key)
        if record is not None:
            self.log(f"Species Director: cached measurements {key[:20]} (no call)")
        else:
            lines = []
            for n, sp in enumerate(species):
                hint = sp.get("crop_hint")
                where = (f"boxed at [{', '.join(f'{float(v):.2f}' for v in hint)}]" if hint and len(hint) == 4
                         else "not boxed (the plan gave no place for it)")
                lines.append(f"{n + 1}. {sp.get('name')} ({sp.get('class')}; plan: {sp.get('form')}, {sp.get('size_m')} m, "
                             f"{sp.get('density')}; {sp.get('where')}) {where}")
            user = (f"The plan's species ({len(species)}):\n" + "\n".join(lines) + "\n\nGive one entry for every number.")
            annotated = boxed_species(photo, species)
            if self.out_dir is not None:
                try:
                    self.out_dir.mkdir(parents=True, exist_ok=True)
                    annotated.save(self.out_dir / "species_director_boxes.jpg", quality=90)
                except OSError:
                    pass
            reply = self._call("species measurements", annotated, SYSTEM, user, "species_measurements", SCHEMA,
                               expected_usd=0.1)
            if reply is None:
                return {}
            data = reply.data if isinstance(reply.data, dict) else {}
            record = {"schema_version": SCHEMA_VERSION, "model": f"{reply.provider}:{reply.model}",
                      "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                      "usage": {"input_tokens": reply.input_tokens, "output_tokens": reply.output_tokens,
                                "usd": round(reply.usd, 4), "seconds": round(reply.seconds, 1)},
                      "names": [sp.get("name") for sp in species],
                      "species": [s for s in data.get("species", []) if isinstance(s, dict)],
                      "summary": str(data.get("summary", ""))}
            self.ledger.record("Species Director", reply, key, "measurements")
            self._store(key, record)
            self.log(f"Species Director: {len(record['species'])} species measured by {record['model']} in "
                     f"{reply.seconds:.0f} s, ~${reply.usd:.2f}; {record['summary']}")
        out: dict[str, dict] = {}
        for s in record.get("species", []):
            try:
                n = int(s.get("n")) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= n < len(species):
                continue
            m = {"head_aspect": _clamp(s.get("head_aspect"), 0.08, 1.5),
                 "head_fraction": _clamp(s.get("head_fraction"), 0.05, 0.7),
                 "plant_aspect": _clamp(s.get("plant_aspect"), 0.05, 2.0),
                 "height_m": _clamp(s.get("height_m"), 0.02, 120.0),
                 "seen": str(s.get("seen") or ""), "why": str(s.get("why") or "")}
            out[str(species[n].get("name"))] = m
        return out
