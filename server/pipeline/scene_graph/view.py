"""The whole representation of a built scene as one explorable page.

`build_view` reads a cached run on the CPU and gathers everything the pipeline holds
about the scene into one document: the scene graph (export.py) with an Asset node for
every model the scene places, every detection with its cut-out and the reviewer's
decision, every model with its measured geometry and estimated physical properties
(physical.py), every placement, the photograph, the panorama and the ground's map, the
Scene Plan, and the agent's edit log. `render_html` puts that document into viewer.html,
a page with no dependencies: an index, the node-link graph, a top-down map, the
photograph with its detections, tables, and the raw data.

    view = build_view(run_dir)
    Path("scene_graph.html").write_text(render_html(view))

Nothing here writes into the run. The Scene Graph stage and scripts/scene-graph.py html
are the two callers.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from pipeline.scene_graph import physical
from pipeline.scene_graph.export import DEFAULT_CONFIG, REGION_CLASSES, REGION_TYPES, ROOT, WATER_CLASSES, _Exporter, _Run
from pipeline.scene_graph.graph import LOG_FILE, Node, SceneGraph

TEMPLATE = Path(__file__).with_name("viewer.html")
VIEW_VERSION = 1
THUMB_PX = 112
MODEL_IMAGE_PX = 220
BASIC_OBJECT_KEYS = {"id", "position", "rotation", "scale", "type", "texture", "mesh", "name", "sourceIndex", "lods"}
# Region map colours (RegionType id -> RGB): a map sheet's, readable on both themes.
REGION_RGB = {0: (150, 160, 170), 1: (92, 142, 176), 2: (140, 128, 116), 3: (201, 186, 150), 4: (128, 160, 104),
              5: (176, 152, 150), 6: (150, 150, 146), 7: (104, 104, 108), 8: (188, 160, 120), 9: (76, 112, 80),
              10: (232, 236, 240)}
WATER_WORDS = {"water", "lake", "ocean", "sea", "river"}
GROUND_WORDS = {"sand", "beach", "ground", "grass", "road", "trail", "path", "snow", "field", "meadow", "terrain",
                "cliff", "glacier", "ice", "mountain", "mountains", "hill", "forest"}


def _data_uri(img, fmt: str, **kw) -> str:
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    mime = {"WEBP": "image/webp", "JPEG": "image/jpeg", "PNG": "image/png"}[fmt]
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _thumb(path: Optional[Path], px: int) -> str:
    """A small WebP of an image file (alpha kept); '' when there is none."""
    if path is None:
        return ""
    try:
        from PIL import Image
        img = Image.open(path)
        img = img.convert("RGBA" if img.mode in ("RGBA", "LA", "P") else "RGB")
        img.thumbnail((px, px), Image.LANCZOS)
        return _data_uri(img, "WEBP", quality=72, method=4)
    except Exception:
        return ""


def _photo(path: Optional[Path], width: int, quality: int) -> Optional[dict]:
    if path is None or not Path(path).is_file():
        return None
    try:
        from PIL import Image
        img = Image.open(path).convert("RGB")
        full = img.size
        if img.width > width:
            img = img.resize((width, max(1, round(img.height * width / img.width))), Image.LANCZOS)
        return {"src": _data_uri(img, "JPEG", quality=quality), "w": full[0], "h": full[1]}
    except Exception:
        return None


def _scene_name(run_dir: Path) -> str:
    """The sample photograph this run was built from, by thumbnail (as scripts/hybrid-run.sh)."""
    try:
        from PIL import Image
        def thumb(p):
            return np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BOX), dtype=np.float32)
        want = thumb(run_dir / "input.png")
        best = None
        for p in sorted((ROOT / "server" / "samples").glob("*.jpg")):
            err = float(np.abs(thumb(p) - want).mean())
            if err < 6 and (best is None or err < best[0]):
                best = (err, p.stem)
        return best[1] if best else ""
    except Exception:
        return ""


def _stats(values: list[float]) -> Optional[list[float]]:
    if not values:
        return None
    a = np.asarray(values, dtype=np.float64)
    return [float(f"{a.min():.4g}"), float(f"{np.median(a):.4g}"), float(f"{a.max():.4g}")]


def _lod_chain(o: dict) -> str:
    """'mesh 0.06 > card 0.025 > billboard 0.003' for one object's levels of detail."""
    parts = []
    for lod in o.get("lods") or []:
        what = "mesh" if lod.get("mesh") and not str(lod.get("mesh")).endswith("_card") else (
            "card" if lod.get("mesh") or lod.get("texture") else "nothing")
        if what == "card" and parts and parts[-1].startswith("card"):
            what = "billboard"
        parts.append(f"{what} {float(lod.get('screenHeight') or 0):g}")
    return " > ".join(parts)


def _region_map_image(region: Optional[np.ndarray], height: Optional[np.ndarray], px: int = 640) -> str:
    """The region map, hill-shaded by the height map, +z up, as a JPEG data URI."""
    if region is None:
        return ""
    try:
        from PIL import Image
        ids = np.rint(region).astype(np.int32)
        step = max(1, ids.shape[0] // px)
        ids = ids[::step, ::step]
        rgb = np.zeros(ids.shape + (3,), dtype=np.float32)
        for tid, colour in REGION_RGB.items():
            rgb[ids == tid] = colour
        if height is not None and height.shape[:2] == region.shape[:2]:
            h = np.asarray(height, dtype=np.float32)[::step, ::step]
            h = np.where(np.isfinite(h), h, np.nanmedian(h))
            cell = 1.0 * step
            dz, dx = np.gradient(h, cell)
            shade = np.clip(1.0 + 1.6 * (dx * 0.6 + dz * -0.6), 0.55, 1.35)   # lit from the upper left of the sheet
            rgb *= shade[..., None]
        img = Image.fromarray(np.clip(rgb[::-1], 0, 255).astype(np.uint8))   # row 0 is -z: flip so +z is up
        return _data_uri(img, "JPEG", quality=78)
    except Exception:
        return ""


def _trim(v: Any, limit: int = 1500) -> Any:
    """The value with its long strings and long number lists cut short (embedded textures,
    spherical-harmonic tables): the page shows what is there, not every byte of it."""
    if isinstance(v, str) and len(v) > limit:
        return f"{v[:120]}... ({len(v)} characters)"
    if isinstance(v, dict):
        return {k: _trim(x, limit) for k, x in v.items()}
    if isinstance(v, list):
        if len(v) > 64 and all(isinstance(x, (int, float)) for x in v):
            return v[:16] + [f"... ({len(v)} numbers)"]
        return [_trim(x, limit) for x in v]
    return v


def _detection_node(g: SceneGraph, i: int, cls: str) -> str:
    if f"thing_{i}" in g.nodes:
        return f"thing_{i}"
    if f"pop_{cls}" in g.nodes:
        return f"pop_{cls}"
    if cls in WATER_WORDS and "water" in g.nodes:
        return "water"
    if cls in ("sky", "cloud", "clouds", "fog", "haze") and "atmosphere" in g.nodes:
        return "atmosphere"
    if cls in GROUND_WORDS and "ground" in g.nodes:
        return "ground"
    return ""


def _owner(g: SceneGraph, key: str, cls: str, kind: str, sources: list[int]) -> str:
    """The node a model belongs to: the population of its class, the thing it was built
    from, or the surface / backdrop / landmark that names its mesh."""
    for n in g.nodes.values():
        if n.type != "Asset" and (n.geometry or {}).get("mesh") == key:
            return n.id
    if kind in ("model", "card"):
        if f"pop_{cls}" in g.nodes:
            return f"pop_{cls}"
        for i in sources:
            if f"thing_{i}" in g.nodes:
                return f"thing_{i}"
        return ""
    fixed = {"terrain": "ground", "water": "water", "formation": "ground"}
    if kind in fixed and fixed[kind] in g.nodes:
        return fixed[kind]
    if key == "far_field_backdrop" and "far_field" in g.nodes:
        return "far_field"
    if key == "treeline_backdrop" and "treeline" in g.nodes:
        return "treeline"
    if kind in ("structure", "landmark"):
        slug = key.split("_", 1)[1] if "_" in key else key
        for n in g.nodes.values():
            if n.id.startswith("landmark_") and (slug in n.id or n.id[len("landmark_"):] in slug):
                return n.id
        for i in sources:
            if f"thing_{i}" in g.nodes:
                return f"thing_{i}"
    return ""


def build_view(run_dir: Path, config_path: Path = DEFAULT_CONFIG, title: Optional[str] = None,
               sample_points: int = physical.SAMPLE_POINTS) -> dict:
    """Everything the run holds about its scene, as one JSON-able document."""
    run_dir = Path(run_dir).resolve()
    run = _Run(run_dir, Path(config_path))
    ex = _Exporter(run)
    g = ex.build()
    scene = ex.scene or {}
    objects = [o for o in scene.get("objects") or [] if isinstance(o, dict)]

    # detections
    detections: list[dict] = []
    meta_by_i: dict[int, dict] = {}
    photo_path = run_dir / "input.png"
    pano_path = None
    for stage in ("Panorama Supersampling", "Panorama"):
        if (run_dir / stage / "panorama.png").is_file():
            pano_path = run_dir / stage / "panorama.png"
            break
    pano_path = pano_path or run.latest_file("panorama.png")
    photo = _photo(photo_path, 1400, 80)
    pano = _photo(pano_path, 2048, 70)
    for i, m in ex._instances():
        meta_by_i[i] = m
        cls = str(m.get("class") or "none").lower()
        d: dict[str, Any] = {"i": i, "cls": cls, "name": str(m.get("name") or ""),
                             "node": _detection_node(g, i, cls), "thumb": _thumb(run.latest_file(f"crop_{i}.png"), THUMB_PX)}
        for k_in, k_out in (("confidence", "conf"), ("score", "score"), ("source", "src"), ("label_source", "label_src"),
                            ("in_captured_fov", "in_photo"), ("bucket", "bucket"), ("stationary", "stationary"),
                            ("vetoed_class", "vetoed"), ("veto_reason", "veto"), ("erase", "erase"),
                            ("part_of", "part_of"), ("synthetic", "synthetic"), ("review", "review"),
                            ("world_height", "h"), ("world_width", "w"), ("motion", "motion"), ("sway", "sway")):
            if m.get(k_in) is not None:
                v = m[k_in]
                d[k_out] = round(v, 4) if isinstance(v, float) else v
        if cls in REGION_CLASSES:
            d["region"] = 1
        p = m.get("world_position")
        if isinstance(p, (list, tuple)) and len(p) == 3:
            d["pos"] = [round(float(v), 3) for v in p]
        if pano and isinstance(m.get("box"), (list, tuple)) and len(m["box"]) == 4:
            x, y, w, h = (float(v) for v in m["box"])
            d["box"] = [round(x / pano["w"], 5), round(y / pano["h"], 5), round(w / pano["w"], 5), round(h / pano["h"], 5)]
        if photo and isinstance(m.get("photo_box"), (list, tuple)) and len(m["photo_box"]) == 4:
            x, y, w, h = (float(v) for v in m["photo_box"])
            d["pbox"] = [round(x / photo["w"], 5), round(y / photo["h"], 5), round(w / photo["w"], 5), round(h / photo["h"], 5)]
        detections.append(d)

    # models and placements
    by_key: dict[str, list[dict]] = {}
    for o in objects:
        key = str(o.get("mesh") or (f"image:{o.get('texture')}" if o.get("texture") else "unnamed"))
        by_key.setdefault(key, []).append(o)
    models: list[dict] = []
    model_index: dict[str, int] = {}
    rows: list[list] = []
    for key in sorted(by_key, key=lambda k: (-len(by_key[k]), k)):
        insts = by_key[key]
        cls, kind = physical.model_class(key)
        sources: dict[int, int] = {}
        for o in insts:
            if isinstance(o.get("sourceIndex"), int):
                sources[o["sourceIndex"]] = sources.get(o["sourceIndex"], 0) + 1
        top_sources = [i for i, _ in sorted(sources.items(), key=lambda kv: -kv[1])]
        if cls == "other" and top_sources and top_sources[0] in meta_by_i:
            cls = str(meta_by_i[top_sources[0]].get("class") or "other").lower()
        path = run.latest_file(f"{key}.glb")
        geom = physical.mesh_properties(path, sample_points) if path is not None else {}
        points = geom.pop("points", "")
        if path is not None:
            geom["stage"] = path.parent.name
        scales = [abs(float((o.get("scale") or {}).get("y") or 1.0)) for o in insts]
        dists = [math.hypot(float(o["position"]["x"]), float(o["position"]["z"])) for o in insts if o.get("position")]
        first = {"mesh": 0, "card": 0}
        chains: dict[str, int] = {}
        extra: dict[str, int] = {}
        for o in insts:
            chain = _lod_chain(o)
            chains[chain or "single level"] = chains.get(chain or "single level", 0) + 1
            is_card = kind in ("card", "landmark") or o.get("type") == "billboard" or chain.startswith("card")
            first["card" if is_card else "mesh"] += 1
            for k in o:
                if k not in BASIC_OBJECT_KEYS:
                    extra[k] = extra.get(k, 0) + 1
        med = float(np.median(scales)) if scales else 1.0
        phys = physical.estimate(cls, kind, geom, med) if geom else {}
        if phys.get("mass_kg") is not None and len(set(round(s, 3) for s in scales)) > 1:
            lo = physical.estimate(cls, kind, geom, min(scales)).get("mass_kg")
            hi = physical.estimate(cls, kind, geom, max(scales)).get("mass_kg")
            if lo is not None and hi is not None:
                phys["mass_range_kg"] = [lo, hi]
        # A collider the scene carries (Scene Animation's static proxies, the terrain's
        # physics mesh) is a fact, not the class's prior.
        proxies = sorted({str(o["physicsMesh"]) for o in insts if o.get("physicsMesh")})
        if proxies and phys:
            proxy_path = run.latest_file(f"{proxies[0]}.glb")
            proxy_geom = physical.mesh_properties(proxy_path, 0) if proxy_path is not None else {}
            triangles = proxy_geom.get("triangles")
            phys["collider"] = f"mesh ({proxies[0]}" + (f", {triangles} triangles" if triangles else "") + ")"
            phys["collider_in_scene"] = True
            if phys.get("body") in ("dynamic", "floating"):
                phys["body"] = "static"
        images = []
        if kind in ("model", "card", "landmark", "structure"):
            for i in top_sources[:1]:
                for label, name in (("cut-out", f"crop_{i}.png"), ("completed", f"crop_complete_{i}.png"),
                                    ("reference view", f"crop_reference_{i}.png"), ("delit", f"crop_delit_{i}.png")):
                    src = _thumb(run.latest_file(name), MODEL_IMAGE_PX)
                    if src:
                        images.append({"label": label, "src": src})
        model = {"key": key, "cls": cls, "kind": kind, "base": physical.base_key(key), "n": len(insts),
                 "sources": top_sources[:12], "source_count": len(sources), "dist": _stats(dists), "scale": _stats(scales),
                 "first": first, "lods": sorted(chains.items(), key=lambda kv: -kv[1])[:6], "annotations": extra,
                 "geom": geom, "phys": phys, "points": points, "images": images, "node": "", "owner": ""}
        model_index[key] = len(models)
        models.append(model)
        for o in insts:
            pos, rot = o.get("position") or {}, o.get("rotation") or {}
            flags = (1 if "sway" in o else 0) | (2 if "physics" in o else 0) | (4 if o.get("videoColor") or o.get("video_color") else 0)
            chain = _lod_chain(o)
            card = 8 if (kind in ("card", "landmark") or chain.startswith("card")) else 0
            rows.append([round(float(pos.get("x") or 0), 2), round(float(pos.get("z") or 0), 2),
                         round(float(pos.get("y") or 0), 2), round(float(rot.get("y") or 0), 1),
                         round(abs(float((o.get("scale") or {}).get("y") or 1.0)), 3), model_index[key],
                         flags | card, o["sourceIndex"] if isinstance(o.get("sourceIndex"), int) else -1])

    # an Asset node per model (its variants and card share one), with what was measured and estimated
    groups: dict[str, list[dict]] = {}
    for m in models:
        groups.setdefault(m["base"], []).append(m)
    for base, members in groups.items():
        lead = next((m for m in members if m["key"] == base), members[0])
        nid = f"asset_{base}"
        if nid in g.nodes:
            n = g.nodes[nid]
        else:
            n = g.add(Node(nid, "Asset", lead["kind"], base))
            n.geometry = {"mesh": base}
        stage = lead["geom"].get("stage") or "Scene Generation"
        n.set("instances", sum(m["n"] for m in members), "measured", "", 1.0, "Scene Generation")
        n.set("placed_as", {k: sum(m["first"][k] for m in members) for k in ("mesh", "card")}, "measured", "", 1.0, "Scene Generation")
        if len(members) > 1:
            n.set("model_files", [m["key"] for m in members], "measured", "", 1.0, "Scene Generation")
        geo, phys = lead["geom"], lead["phys"]
        if geo.get("triangles") is not None:
            n.set("triangles", geo["triangles"], "measured", "", 1.0, stage)
        if geo.get("watertight") is not None:
            n.set("watertight", geo["watertight"], "measured", "", 1.0, stage)
        if phys.get("size_m"):
            n.set("size_m", phys["size_m"], "measured", "m", 1.0, stage)
        for k, unit in (("volume_m3", "m3"), ("material", ""), ("density_kg_m3", "kg/m3"), ("mass_kg", "kg"),
                        ("friction_static", ""), ("friction_dynamic", ""), ("restitution", ""), ("body", ""), ("collider", "")):
            if phys.get(k) is not None:
                n.set(k, phys[k], "default", unit, 0.5, "physical priors")
        owner = _owner(g, base, lead["cls"], lead["kind"], lead["sources"])
        if owner and not any(r.subject == owner and r.predicate == "made_of" and r.object == nid for r in g.relations):
            g.relate(owner, "made_of", nid, "measured", "Scene Generation")
        for m in members:
            m["node"], m["owner"] = nid, owner

    placed_by_source: dict[int, dict[str, int]] = {}
    for r in rows:
        if r[7] >= 0:
            d = placed_by_source.setdefault(r[7], {})
            d[models[r[5]]["key"]] = d.get(models[r[5]]["key"], 0) + 1
    for d in detections:
        if d["i"] in placed_by_source:
            d["placed"] = placed_by_source[d["i"]]

    # the ground's map
    params = run.obj("height_map_params", warn=False) or {}
    grid_m = float(params.get("grid_size_meters") or 0) or (ex.cell_m * ex.region_map.shape[0] if ex.cell_m and ex.region_map is not None else 0)
    height = None
    try:
        height = run.array("height_map")
    except Exception:
        pass
    ground_map = {"src": _region_map_image(ex.region_map, height), "size_m": grid_m,
                  "legend": [[REGION_TYPES[t], "#%02x%02x%02x" % REGION_RGB[t]] for t in sorted(REGION_RGB) if t != 0]}

    # edit log, accepted scene edits, review record
    log = []
    log_path = run_dir / LOG_FILE
    if log_path.is_file():
        for line in log_path.read_text().splitlines():
            try:
                log.append(json.loads(line))
            except Exception:
                continue
    edits = []
    try:
        edits = json.loads((run_dir / "scene_edits.json").read_text()).get("edits") or []
    except Exception:
        pass
    review = run.obj("inventory_review", warn=False) or {}
    camera = g.nodes.get("camera")
    fov = camera.attrs["fov_deg"].value if camera is not None and "fov_deg" in camera.attrs else None
    tp = run.obj("terrain_plan", warn=False) or {}
    name = title or _scene_name(run_dir) or run_dir.name[:8]
    g.meta["warnings"] = list(run.warnings)
    globals_ = _trim({k: v for k, v in scene.items() if k != "objects"})
    return {"view_version": VIEW_VERSION, "title": name, "run_id": run_dir.name,
            "generated": time.strftime("%Y-%m-%d %H:%M"),
            "graph": g.to_json(), "log": log, "edits": edits, "review": review if isinstance(review, dict) else {},
            "detections": detections, "models": models,
            "placements": {"cols": ["x", "z", "y", "yaw", "scale", "model", "flags", "source"], "rows": rows},
            "images": {"photo": photo, "panorama": pano, "map": ground_map},
            "camera": {"fov_deg": fov or tp.get("hfov_deg"), "position": [0.0, 0.0]},
            "plan": ex.plan, "scene": globals_, "stages": list(run.saved_order), "warnings": list(run.warnings)}


def physical_table(view: dict) -> dict:
    """Per model key: geometry and the physical estimate (the stage's physical_properties.json)."""
    return {m["key"]: {"class": m["cls"], "kind": m["kind"], "instances": m["n"], "scale": m["scale"],
                       "geometry": m["geom"], "physical": m["phys"]} for m in view["models"]}


def render_html(view: dict, standalone: bool = True) -> str:
    """The page. standalone=False leaves out the document skeleton (for a host that adds its own)."""
    body = TEMPLATE.read_text()
    data = json.dumps(view, separators=(",", ":"), ensure_ascii=False, default=str).replace("</", "<\\/").replace("<!--", "<\\u0021--")
    title = f"{view['title']} scene graph"
    body = body.replace("__TITLE__", title.replace("<", "&lt;")).replace("__DATA__", data)
    if not standalone:
        return body
    return ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '</head>\n<body>\n' + body + '\n</body>\n</html>\n')
