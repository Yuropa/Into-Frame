"""The painted population from the scene plan (docs/hybrid-scene-plan.md).

Distribution Synthesis paints a class's instances per bucket, as many copies of
each bucket as the detections of that bucket suggest. On Rainier that made a junk
bucket (a trail crop, grass blobs, mountain strips) the meadow's most common
flower, 3,470 copies, while the magenta paintbrush that fills the photograph's
foreground had 33. The plan says the opposite -- paintbrush dense across the whole
width, lupine patchy at the lower right -- and the species layer knows which bucket
stands for which species. So the painted positions stay (they are a plausible
meadow) and each one is re-dealt a species by the plan's density and where, and
takes the species' size.
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

import numpy as np

DENSITY_WEIGHT = {"dense": 1.0, "common": 0.7, "patchy": 0.35, "scattered": 0.25, "sparse": 0.12, "rare": 0.05}


def density_weight(word: Optional[str]) -> float:
    w = str(word or "").lower()
    for key, val in DENSITY_WEIGHT.items():
        if key in w:
            return val
    return 0.3


def zone_factor(where: Optional[str], bearing_frac: float, dist_m: float, near_m: float, off_zone: float = 0.25) -> float:
    """How much of a species' weight applies at a point, from the plan's free-text
    `where`. bearing_frac is the point's bearing over the photograph's half field
    (-1 left edge, 0 centre, 1 right edge); a point outside the described side or
    depth keeps off_zone of the weight, so a patchy species still strays."""
    text = str(where or "").lower()
    first = re.split(r"[.;,]", text)[0]
    f = 1.0
    sides = []
    if re.search(r"\bleft\b", first):
        sides.append("left")
    if re.search(r"\bright\b", first):
        sides.append("right")
    if re.search(r"\bcent(re|er)\b|\bmiddle\b", first):
        sides.append("centre")
    if sides and not re.search(r"whole width|across|throughout|everywhere|left to right", first):
        in_side = (
            ("left" in sides and bearing_frac < -0.15)
            or ("right" in sides and bearing_frac > 0.15)
            or ("centre" in sides and abs(bearing_frac) < 0.4)
        )
        if not in_side:
            f *= off_zone
    near_words = re.search(r"foreground|\blower\b|near|close|front", first)
    far_words = re.search(r"distance|background|\bfar\b|mid-?ground|middle distance", first)
    if near_words and not far_words and dist_m > near_m:
        f *= off_zone
    if far_words and not near_words and dist_m < 0.5 * near_m:
        f *= off_zone
    return f


def relabel_population(
    painted: list[tuple[int, dict]],
    species: list[dict],
    species_bucket: dict[str, int],
    bearing_frac_of: Callable[[float, float], tuple[float, float]],
    near_m: float,
    rng: np.random.Generator,
    off_zone: float = 0.25,
) -> tuple[dict[int, dict], dict]:
    """Re-deal painted instances across the plan's species.

    painted: (index, metadata) of the class's painted instances.
    species: the plan's species entries of the class (name, density, where, size_m).
    species_bucket: species name -> the bucket whose mesh stands for it.
    bearing_frac_of: world (x, z) -> (bearing over half-fov, horizontal distance).
    Returns ({index: new metadata}, summary)."""
    usable = [sp for sp in species if sp.get("name") in species_bucket]
    summary = {"species": {}, "unplaced": [sp.get("name") for sp in species if sp.get("name") not in species_bucket],
               "before": {}, "after": {}}
    if not usable or not painted:
        return {}, summary
    weights = np.array([density_weight(sp.get("density")) for sp in usable], dtype=float)
    for sp, wt in zip(usable, weights):
        summary["species"][sp.get("name")] = {"weight": float(wt), "where": sp.get("where"), "bucket": species_bucket[sp["name"]]}
    out: dict[int, dict] = {}
    for idx, meta in painted:
        summary["before"][meta.get("bucket")] = summary["before"].get(meta.get("bucket"), 0) + 1
        wp = meta.get("world_position") or [0.0, 0.0, 0.0]
        bf, dist = bearing_frac_of(float(wp[0]), float(wp[2]))
        p = weights * np.array([zone_factor(sp.get("where"), bf, dist, near_m, off_zone) for sp in usable])
        if p.sum() <= 0:
            continue
        k = int(rng.choice(len(usable), p=p / p.sum()))
        sp = usable[k]
        size = float(sp.get("size_m") or meta.get("world_height") or 0.4)
        old_h = float(meta.get("world_height") or size)
        old_w = float(meta.get("world_width") or 0.5 * old_h)
        aspect = float(np.clip(old_w / max(old_h, 1e-3), 0.25, 1.0))
        new_h = size * float(rng.uniform(0.75, 1.2))
        new = dict(meta)
        new.update({
            "bucket": int(species_bucket[sp["name"]]), "world_height": new_h, "world_width": new_h * aspect,
            "plan_species": sp.get("name"), "source": "plan_population", "metric_size": True,
        })
        out[idx] = new
        summary["after"][sp.get("name")] = summary["after"].get(sp.get("name"), 0) + 1
    return out, summary
