"""Size, volume and a first estimate of the physical properties of a placed model.

Two halves. `mesh_properties` measures a GLB on the CPU: triangles, the local bounding
box, surface area, the enclosed volume when the mesh closes and the convex hull's
volume when it does not, and a small coloured point sample a page can draw. `estimate`
turns that, the model's class and the scale it is placed at into what a physics engine
asks for: material, density, mass, friction, restitution, body type and collider shape.

The measurements are measured. The material table below is a prior per class, not a
reading of the object: a rock is stone, a tree is green wood in a thin share of its
crown's hull, a boat is a hull that displaces a sixth of its volume. Every number it
yields is evidence `default` in the scene graph until something reads the object itself.
"""
from __future__ import annotations

import base64
import json
import re
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np

HANDEDNESS_MARKER = "intoframe_handedness"   # scene/mesh.py: the file is X-negated
MAX_BYTES = 96 * 1024 * 1024                 # a larger GLB is described from its header alone
SAMPLE_POINTS = 5000

# class -> material, density (kg/m3), hull_fill (share of the convex hull the shape
# occupies, used when the mesh does not close), solid (share of that volume that is
# material), static and dynamic friction, restitution, body, collider.
# body: static (never moves), rooted (static, bends in wind), floating (rides the water
# surface), dynamic (a rigid body), kinematic (moved by animation), none (no collision).
PRIORS: dict[str, dict[str, Any]] = {
    "rock":       dict(material="stone", density=2600, hull_fill=0.75, solid=1.0, friction=(0.80, 0.60), restitution=0.10, body="dynamic", collider="convex hull", static_above_kg=400.0),
    "boat":       dict(material="hull (displaces water)", density=170, hull_fill=0.55, solid=1.0, friction=(0.40, 0.30), restitution=0.20, body="floating", collider="convex hull"),
    "ship":       dict(material="hull (displaces water)", density=250, hull_fill=0.60, solid=1.0, friction=(0.40, 0.30), restitution=0.15, body="floating", collider="convex hull"),
    "tree":       dict(material="green wood", density=900, hull_fill=0.035, solid=1.0, friction=(0.60, 0.50), restitution=0.10, body="rooted", collider="capsule (trunk)"),
    "bush":       dict(material="green wood and leaf", density=700, hull_fill=0.02, solid=1.0, friction=(0.50, 0.40), restitution=0.05, body="rooted", collider="none (trigger)"),
    "flower":     dict(material="plant tissue", density=700, hull_fill=0.01, solid=1.0, friction=(0.40, 0.30), restitution=0.0, body="rooted", collider="none"),
    "plant":      dict(material="plant tissue", density=700, hull_fill=0.01, solid=1.0, friction=(0.40, 0.30), restitution=0.0, body="rooted", collider="none"),
    "grass_tuft": dict(material="plant tissue", density=700, hull_fill=0.005, solid=1.0, friction=(0.40, 0.30), restitution=0.0, body="rooted", collider="none"),
    "reed":       dict(material="plant tissue", density=700, hull_fill=0.01, solid=1.0, friction=(0.40, 0.30), restitution=0.0, body="rooted", collider="none"),
    "building":   dict(material="masonry (hollow)", density=2200, hull_fill=0.85, solid=0.22, friction=(0.70, 0.60), restitution=0.05, body="static", collider="mesh"),
    "tower":      dict(material="steel lattice", density=7850, hull_fill=0.25, solid=0.004, friction=(0.60, 0.45), restitution=0.10, body="static", collider="mesh"),
    "bridge":     dict(material="masonry", density=2300, hull_fill=0.55, solid=0.8, friction=(0.70, 0.60), restitution=0.05, body="static", collider="mesh"),
    "dock":       dict(material="timber", density=650, hull_fill=0.50, solid=0.5, friction=(0.55, 0.45), restitution=0.10, body="static", collider="mesh"),
    "car":        dict(material="steel shell", density=135, hull_fill=0.70, solid=1.0, friction=(0.90, 0.70), restitution=0.10, body="dynamic", collider="box"),
    "truck":      dict(material="steel shell", density=190, hull_fill=0.75, solid=1.0, friction=(0.90, 0.70), restitution=0.10, body="dynamic", collider="box"),
    "bus":        dict(material="steel shell", density=170, hull_fill=0.85, solid=1.0, friction=(0.90, 0.70), restitution=0.10, body="dynamic", collider="box"),
    "motorcycle": dict(material="steel frame", density=420, hull_fill=0.35, solid=1.0, friction=(0.85, 0.65), restitution=0.10, body="dynamic", collider="box"),
    "bicycle":    dict(material="steel frame", density=110, hull_fill=0.12, solid=1.0, friction=(0.80, 0.60), restitution=0.10, body="dynamic", collider="box"),
    "bench":      dict(material="timber and iron", density=700, hull_fill=0.18, solid=1.0, friction=(0.55, 0.45), restitution=0.10, body="dynamic", collider="box", static_above_kg=25.0),
    "statue":     dict(material="stone", density=2600, hull_fill=0.55, solid=1.0, friction=(0.75, 0.60), restitution=0.05, body="static", collider="convex hull"),
    "person":     dict(material="body", density=985, hull_fill=0.55, solid=1.0, friction=(0.70, 0.50), restitution=0.05, body="kinematic", collider="capsule"),
    "animal":     dict(material="body", density=1000, hull_fill=0.50, solid=1.0, friction=(0.70, 0.50), restitution=0.05, body="kinematic", collider="capsule"),
    "bird":       dict(material="body", density=750, hull_fill=0.40, solid=1.0, friction=(0.50, 0.40), restitution=0.05, body="kinematic", collider="sphere"),
}
# Models that are the ground, the water or a painted backdrop: no mass, and a collider
# only where something can stand on or run into it.
FIXED: dict[str, dict[str, Any]] = {
    "terrain":   dict(material="ground", body="static", collider="mesh (terrain_physics_mesh)", friction=(0.80, 0.65), restitution=0.05),
    "formation": dict(material="rock face", body="static", collider="mesh", friction=(0.80, 0.60), restitution=0.05),
    "water":     dict(material="water", density=1000, body="fluid surface", collider="none (buoyancy plane)", friction=(0.0, 0.0), restitution=0.0),
    "backdrop":  dict(material="painted image", body="static", collider="none", friction=None, restitution=None),
    "landmark":  dict(material="painted image", body="static", collider="none", friction=None, restitution=None),
    "structure": dict(material="masonry (placeholder volume)", density=2200, hull_fill=0.9, solid=0.25, body="static", collider="mesh", friction=(0.70, 0.60), restitution=0.05),
}
DEFAULT = dict(material="unknown", density=600, hull_fill=0.5, solid=1.0, friction=(0.60, 0.45), restitution=0.10, body="dynamic", collider="convex hull")


def model_class(key: str) -> tuple[str, str]:
    """(class, kind) of a scene mesh key. kind: reconstruction, card, terrain, formation,
    water, backdrop, landmark, structure, other."""
    m = re.match(r"category_mesh_([a-z_]+?)_(\d+)(?:_v\d+)?(_card)?$", key)
    if m:
        return m.group(1), "card" if m.group(3) else "model"
    if key in ("terrain_mesh", "terrain_physics_mesh"):
        return "terrain", "terrain"
    if key.startswith("terrain_formation"):
        return "formation", "formation"
    if key.startswith("water"):
        return "water", "water"
    if key in ("far_field_backdrop", "treeline_backdrop") or key.endswith("_backdrop"):
        return "backdrop", "backdrop"
    if key.startswith("landmark_card"):
        return "landmark", "landmark"
    if key.startswith("structure_"):
        return "structure", "structure"
    return "other", "other"


def base_key(key: str) -> str:
    """category_mesh_tree_0_v1 / _card -> category_mesh_tree_0 (the model its variants share)."""
    return re.sub(r"(_v\d+)?(_card)?$", "", key) if key.startswith("category_mesh_") else key


def glb_json(path: Path) -> dict:
    """The JSON chunk of a GLB (no buffers read)."""
    with open(path, "rb") as f:
        magic, _version, _length = struct.unpack("<4sII", f.read(12))
        if magic != b"glTF":
            return {}
        size, kind = struct.unpack("<II", f.read(8))
        if kind != 0x4E4F534A:
            return {}
        return json.loads(f.read(size).decode("utf-8"))


def _header_properties(tree: dict) -> dict:
    """Triangles and the bounding box from accessor counts and POSITION min/max."""
    acc = tree.get("accessors") or []
    tris, verts = 0, 0
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for mesh in tree.get("meshes") or []:
        for prim in mesh.get("primitives") or []:
            pos = (prim.get("attributes") or {}).get("POSITION")
            if pos is None or pos >= len(acc):
                continue
            a = acc[pos]
            verts += int(a.get("count") or 0)
            if a.get("min") and a.get("max"):
                lo, hi = np.minimum(lo, a["min"][:3]), np.maximum(hi, a["max"][:3])
            idx = prim.get("indices")
            n = int(acc[idx].get("count") or 0) if idx is not None and idx < len(acc) else int(a.get("count") or 0)
            tris += n // 3 if prim.get("mode", 4) == 4 else 0
    out: dict[str, Any] = {"triangles": tris, "vertices": verts}
    if np.all(np.isfinite(lo)):
        out["extents"] = [float(v) for v in (hi - lo)]
        out["bounds"] = [[float(v) for v in lo], [float(v) for v in hi]]
    return out


def _vertex_colours(geom) -> Optional[np.ndarray]:
    """Per-vertex RGB (0..255) from vertex colours or the base colour texture; None when neither."""
    try:
        vis = geom.visual
        if getattr(vis, "kind", None) == "vertex":
            return np.asarray(vis.vertex_colors)[:, :3].astype(np.float32)
        if getattr(vis, "kind", None) == "texture":
            mat = getattr(vis, "material", None)
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            uv = getattr(vis, "uv", None)
            if img is not None and uv is not None and len(uv) == len(geom.vertices):
                tex = np.asarray(img.convert("RGB"), dtype=np.float32)
                h, w = tex.shape[:2]
                u = np.clip((np.asarray(uv)[:, 0] % 1.0) * (w - 1), 0, w - 1).astype(np.int64)
                v = np.clip((1.0 - (np.asarray(uv)[:, 1] % 1.0)) * (h - 1), 0, h - 1).astype(np.int64)
                return tex[v, u]
            base = getattr(mat, "baseColorFactor", None)
            if base is not None:
                c = np.asarray(base, dtype=np.float32)[:3]
                c = c * 255.0 if c.max() <= 1.0 else c
                return np.tile(c, (len(geom.vertices), 1))
    except Exception:
        return None
    return None


def _point_sample(parts: list, lo: np.ndarray, hi: np.ndarray, count: int, flip_x: bool, seed: int = 7) -> str:
    """base64 of count x (x, y, z, r, g, b) bytes: points on the surface, positions in
    0..255 of the longest side of the bounding box, colours shaded by a light from above."""
    rng = np.random.default_rng(seed)
    areas = np.array([float(p.area) for p in parts], dtype=np.float64)
    if not np.isfinite(areas).all() or areas.sum() <= 0:
        return ""
    light = np.array([0.35, 0.85, 0.40]); light /= np.linalg.norm(light)
    pts, cols = [], []
    for part, share in zip(parts, areas / areas.sum()):
        n = max(1, int(round(count * share)))
        tri = np.asarray(part.triangles, dtype=np.float64)
        w = np.asarray(part.area_faces, dtype=np.float64)
        if len(tri) == 0 or w.sum() <= 0:
            continue
        face = rng.choice(len(tri), size=n, p=w / w.sum())
        r1, r2 = np.sqrt(rng.random(n)), rng.random(n)
        b = np.stack([1 - r1, r1 * (1 - r2), r1 * r2], axis=1)
        pts.append(np.einsum("nk,nkd->nd", b, tri[face]))
        vc = _vertex_colours(part)
        if vc is None:
            c = np.full((n, 3), 168.0)
        else:
            c = np.einsum("nk,nkd->nd", b, vc[np.asarray(part.faces)[face]])
        shade = 0.55 + 0.45 * np.abs(np.asarray(part.face_normals)[face] @ light)
        cols.append(c * shade[:, None])
    if not pts:
        return ""
    p = np.concatenate(pts); c = np.concatenate(cols)
    side = float(max((hi - lo).max(), 1e-9))
    q = (p - lo) / side
    if flip_x:
        q[:, 0] = (hi[0] - lo[0]) / side - q[:, 0]
    out = np.concatenate([np.clip(np.rint(q * 255), 0, 255), np.clip(np.rint(c), 0, 255)], axis=1).astype(np.uint8)
    return base64.b64encode(out.tobytes()).decode("ascii")


def mesh_properties(path: Path, sample: int = SAMPLE_POINTS) -> dict:
    """Measured properties of a GLB in its own units (the scene scales it when placing)."""
    path = Path(path)
    out: dict[str, Any] = {"file": path.name, "bytes": path.stat().st_size}
    try:
        tree = glb_json(path)
    except Exception as e:
        return {**out, "error": f"{type(e).__name__}: {e}"}
    flip = ((tree.get("asset") or {}).get("extras") or {}).get(HANDEDNESS_MARKER) == "gltf"

    def server_frame(lo, hi) -> list:
        # scene/mesh.py writes the file X-negated: bounds are reported in the scene's own frame
        lo, hi = [float(v) for v in lo], [float(v) for v in hi]
        if flip:
            lo[0], hi[0] = -hi[0], -lo[0]
        return [lo, hi]

    out.update(_header_properties(tree))
    if out.get("bounds"):
        out["bounds"] = server_frame(*out["bounds"])
    out["skinned"] = bool(tree.get("skins"))
    out["textures"] = len(tree.get("images") or [])
    if out["bytes"] > MAX_BYTES:
        out["note"] = "described from the file header (too large to load)"
        return out
    try:
        import trimesh
        scene = trimesh.load(path, force="scene", process=False)
        parts = [g for g in scene.dump(concatenate=False) if isinstance(g, trimesh.Trimesh) and len(g.faces)]
        if not parts:
            return out
        whole = trimesh.util.concatenate([trimesh.Trimesh(vertices=p.vertices, faces=p.faces, process=False) for p in parts])
        lo, hi = whole.bounds
        out["extents"] = [float(v) for v in (hi - lo)]
        out["bounds"] = server_frame(lo, hi)
        out["triangles"], out["vertices"], out["parts"] = int(len(whole.faces)), int(len(whole.vertices)), len(parts)
        out["area"] = float(whole.area)
        welded = whole.copy(); welded.merge_vertices()
        out["watertight"] = bool(welded.is_watertight)
        if welded.is_watertight and welded.is_winding_consistent:
            out["volume"] = abs(float(welded.volume))
        try:
            out["hull_volume"] = float(whole.convex_hull.volume)
        except Exception:
            pass
        if sample:
            out["points"] = _point_sample(parts, np.asarray(lo), np.asarray(hi), sample, flip)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def estimate(cls: str, kind: str, geom: dict, scale: float, name: str = "") -> dict:
    """Physical properties of one model placed at `scale` (uniform). Sizes in metres."""
    table = FIXED.get(kind) or FIXED.get(cls) or PRIORS.get(cls) or DEFAULT
    out: dict[str, Any] = {"material": table["material"], "body": table["body"], "collider": table["collider"],
                           "prior": cls if cls in PRIORS else (kind if kind in FIXED else "default")}
    if table.get("friction"):
        out["friction_static"], out["friction_dynamic"] = table["friction"]
    if table.get("restitution") is not None:
        out["restitution"] = table["restitution"]
    ext = geom.get("extents")
    if ext:
        out["size_m"] = [round(float(e) * scale, 3) for e in ext]
    if kind == "card":
        out.update(body="none", collider="none", note="a flat card: no volume of its own")
        return out
    if "density" not in table or "hull_fill" not in table:
        if table.get("density"):
            out["density_kg_m3"] = table["density"]
        return out
    if geom.get("volume"):
        volume, how = geom["volume"] * scale ** 3, "enclosed volume of the mesh"
        if cls in ("tree", "bush", "flower", "plant", "grass_tuft", "reed"):
            # a closed crown shell is not solid wood
            volume, how = (geom.get("hull_volume") or geom["volume"]) * scale ** 3 * table["hull_fill"], \
                f"{table['hull_fill']:g} of the convex hull"
    elif geom.get("hull_volume"):
        volume, how = geom["hull_volume"] * scale ** 3 * table["hull_fill"], f"{table['hull_fill']:g} of the convex hull (mesh is open)"
    elif ext:
        volume, how = float(np.prod(ext)) * scale ** 3 * table["hull_fill"] * 0.6, "from the bounding box"
    else:
        return out
    mass = volume * table["solid"] * table["density"]
    out.update(volume_m3=float(f"{volume:.4g}"), volume_from=how, density_kg_m3=table["density"],
               solid_fraction=table["solid"], mass_kg=float(f"{mass:.4g}"))
    limit = table.get("static_above_kg")
    if limit is not None and mass > limit:
        out["body"] = "static"
        out["note"] = f"heavier than {limit:g} kg: nothing in the scene moves it"
    return out
