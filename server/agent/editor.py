"""Scene editor: typed actions on the scene the renderers draw (docs/agentic-scene-graph-plan.md).

The first pass writes that scene as scene.json: placed objects (mesh key, position,
rotation, scale), lighting, atmosphere, sky. An agent's decision about the scene is an
action on that file, not a stage parameter, so applying one costs a JSON edit and a
render (scripts/render-scene.py --scene), never a pipeline rerun:

    place         a mesh the run already holds (category_mesh_boat_0, category_mesh_tree_1_card,
                  landmark_card_6 ...) at a position, with a height in metres and a yaw
    move          an object to a position, or by a delta
    scale         an object to a height in metres, or by a factor
    rotate        an object to a yaw, or by a delta
    remove        an object
    swap_mesh     an object's mesh key
    set_atmosphere  haze_near_m / haze_far_m / haze_color
    set_sun       azimuth_deg / elevation_deg / intensity / color
    set_light     sun_scale / ambient_scale: multipliers on the rendered light (1.5 = half again
                  as bright); the fix for a scene whose exposure fit left the mid-ground black

A position (the `at` field) is one of
    {"world": [x, y, z]}
    {"bearing_deg": b, "distance_m": d}          from the capture camera, b > 0 to the right
    {"photo_box": [x, y, w, h]}                   fractions of the photograph; the editor casts the
                                                  ray through the box's bottom centre against the
                                                  terrain and water meshes (the photo depth saturates
                                                  past ~100 m, so it is only the fallback)
    {"delta": [dx, dy, dz]}                       move only
y snaps to the surface the object stands on (`on`: ground | water | none; water classes
default to water) unless the position gives it explicitly with `"snap": false`. The
uniform scale is height_m over the mesh's own height, and the mesh's base is set on
the surface whatever its origin.

A target is an object id, a unique object name, thing_<sourceIndex> (the graph's node
id), or sourceIndex:<n>. Environment meshes (terrain, water, treeline, far_field,
formations) are refused unless the action says `"force": true`.

Editor.apply returns a record of what happened; nothing is written until save().
Accepted actions are persisted in <run>/scene_edits.json and replayed by the Scene
Edits stage (pipeline/scene_edits), so a rerun of the later stages and the .frame
archive keep them.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

ACTIONS = ("place", "move", "scale", "rotate", "remove", "swap_mesh", "set_atmosphere", "set_sun", "set_light")
ENV_NAMES = {"terrain", "water", "water_apron", "treeline", "far_field", "backdrop"}
WATER_CLASSES = {"boat", "ship", "barge", "sailboat", "kayak", "canoe", "yacht", "ferry", "buoy", "swan", "duck"}
EDITS_FILE = "scene_edits.json"


class EditError(ValueError):
    pass


def _v(d: Any, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if isinstance(d, dict):
        return np.array([float(d.get("x", default[0])), float(d.get("y", default[1])), float(d.get("z", default[2]))])
    if isinstance(d, (list, tuple)) and len(d) == 3:
        return np.array([float(x) for x in d])
    return np.array(default, dtype=float)


def _xyz(v) -> dict:
    return {"x": round(float(v[0]), 4), "y": round(float(v[1]), 4), "z": round(float(v[2]), 4)}


def edits_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16] if path.is_file() else ""


def load_edits(run_dir: Path) -> dict:
    p = run_dir / EDITS_FILE
    if p.is_file():
        d = json.loads(p.read_text())
        d.setdefault("edits", [])
        return d
    return {"version": 1, "edits": []}


def save_edits(run_dir: Path, data: dict) -> Path:
    p = run_dir / EDITS_FILE
    p.write_text(json.dumps(data, indent=1) + "\n")
    return p


class Editor:
    """Actions on one scene dict, with the run's context for meshes, terrain and the camera."""

    def __init__(self, scene: dict, context=None, run_dir: Optional[Path] = None, log=print):
        self.scene = scene
        self.context = context
        self.run_dir = Path(run_dir) if run_dir else None
        self.log = log
        self.applied: list[dict] = []
        self._mesh_bounds: dict[str, Optional[tuple[np.ndarray, np.ndarray]]] = {}
        self._terrain = None
        self._water = None
        self._intr = None
        ext = scene.get("extrinsics") or {}
        self.R = np.array(ext.get("rotation") or np.eye(3).ravel(), dtype=float).reshape(3, 3)
        self.t = np.array(ext.get("translation") or [0, 0, 0], dtype=float)
        terrain = next((o for o in scene.get("objects", []) if o.get("name") == "terrain"), None)
        self.terrain_yaw = float(_v(terrain.get("rotation"))[1]) if terrain else float(scene.get("skyboxRotation") or 0.0)

    # ── lookups ──────────────────────────────────────────────────────────────
    @property
    def objects(self) -> list[dict]:
        return self.scene.setdefault("objects", [])

    def find(self, target: str) -> dict:
        t = str(target)
        if t.startswith("thing_") or t.startswith("sourceIndex:"):
            idx = t.split("_", 1)[1] if t.startswith("thing_") else t.split(":", 1)[1]
            hits = [o for o in self.objects if str(o.get("sourceIndex")) == idx]
        else:
            hits = [o for o in self.objects if str(o.get("id")) == t]
            if not hits:
                hits = [o for o in self.objects if o.get("name") == t]
        if not hits:
            raise EditError(f"no object matches '{target}'")
        if len(hits) > 1:
            ids = ", ".join(f"{o.get('name')}#{str(o.get('id'))[:8]} (sourceIndex {o.get('sourceIndex')})" for o in hits[:6])
            raise EditError(f"'{target}' matches {len(hits)} objects; use an id or thing_<sourceIndex>: {ids}")
        return hits[0]

    @staticmethod
    def is_env(o: dict) -> bool:
        n = str(o.get("name") or "")
        return n in ENV_NAMES or n.startswith("formation") or n.startswith("terrain_formation")

    def mesh_bounds(self, key: str) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if key not in self._mesh_bounds:
            b = None
            if self.context is not None:
                m = self.context.input_mesh(key)
                if m is not None and getattr(m, "mesh", None) is not None:
                    bb = m.mesh.bounds
                    b = (np.asarray(bb[0], dtype=float), np.asarray(bb[1], dtype=float))
            self._mesh_bounds[key] = b
        return self._mesh_bounds[key]

    def mesh_keys(self) -> list[str]:
        """Mesh keys the scene already uses (a safe vocabulary for place/swap_mesh)."""
        return sorted({str(o.get("mesh")) for o in self.objects if o.get("mesh") and not self.is_env(o)})

    def _surfaces(self):
        if self._terrain is None and self.context is not None:
            from pipeline.pipeline_context import ContextKey
            self._terrain = self.context.input_mesh(ContextKey.TERRAIN_MESH) or False
            self._water = self.context.input_mesh(ContextKey.WATER_MESH) or False
        return (self._terrain or None), (self._water or None)

    # ── geometry ─────────────────────────────────────────────────────────────
    def world_from_bearing(self, bearing_deg: float, distance_m: float) -> np.ndarray:
        b = math.radians(float(bearing_deg))
        local = np.array([math.sin(b) * float(distance_m), 0.0, math.cos(b) * float(distance_m)])
        return self.t + self.R @ local

    def bearing_distance(self, world) -> tuple[float, float]:
        local = self.R.T @ (np.asarray(world, dtype=float) - self.t)
        return math.degrees(math.atan2(local[0], local[2])), float(math.hypot(local[0], local[2]))

    def _to_terrain_local(self, p: np.ndarray) -> np.ndarray:
        from pipeline.scene_generation.projection import terrain_local_xz
        lx, lz = terrain_local_xz(float(p[0]), float(p[2]), self.terrain_yaw, float(self.t[0]), float(self.t[2]))
        return np.array([lx, p[1], lz])

    def _local_dir(self, d: np.ndarray) -> np.ndarray:
        th = math.radians(self.terrain_yaw)
        return np.array([d[0] * math.cos(th) - d[2] * math.sin(th), d[1], d[0] * math.sin(th) + d[2] * math.cos(th)])

    def surface_y(self, world_xz: np.ndarray, on: str) -> Optional[float]:
        """Height of the ground (or the water) under a world (x, z)."""
        from pipeline.scene_generation.projection import height_map_y_at, mesh_y_at
        terrain, water = self._surfaces()
        p = self._to_terrain_local(np.array([world_xz[0], 0.0, world_xz[2]]))
        if on == "water" and water is not None:
            y = mesh_y_at(float(p[0]), float(p[2]), water)
            if y is not None:
                return y
        if terrain is not None:
            y = mesh_y_at(float(p[0]), float(p[2]), terrain)
            if y is not None:
                return y
        if self.context is not None:
            from pipeline.pipeline_context import ContextKey
            hm = self.context.input_depth(ContextKey.HEIGHT_MAP)
            grid = float((self.context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}).get("grid_size_meters", 0.0))
            if hm is not None and grid > 0:
                return height_map_y_at(float(p[0]), float(p[2]), hm, grid)
        return None

    def ray_hit(self, u: float, v: float, on: str = "ground") -> Optional[tuple[np.ndarray, str]]:
        """The nearest terrain or water point on the ray through photo fraction (u, v)."""
        if self.context is None:
            return None
        from pipeline.pipeline_context import ContextKey
        intr = self.context.input_intrinsics(ContextKey.INTRINSICS)
        if intr is None:
            return None
        cx, cy = float(u) * intr.width, float(v) * intr.height
        d = np.array(intr.unproject(cx, cy, 1.0), dtype=float)
        d = self.R @ (d / np.linalg.norm(d))
        terrain, water = self._surfaces()
        o_local = self._to_terrain_local(self.t.copy())
        d_local = self._local_dir(d)
        hits: dict[str, float] = {}
        for name, m in (("water", water), ("ground", terrain)):
            if m is None:
                continue
            locs, _, _ = m.mesh.ray.intersects_location(o_local[None, :], d_local[None, :], multiple_hits=True)
            if len(locs) == 0:
                continue
            dist = np.linalg.norm(locs - o_local, axis=1)
            dist = dist[dist > 0.5]
            if dist.size:
                hits[name] = float(dist.min())
        if not hits:
            return None
        # The surface the object stands on wins when the ray reaches it within 2 m of
        # the nearest hit (a boat's box bottom sits on the water behind the quay edge).
        name = min(hits, key=hits.get)
        if on in hits and hits[on] <= hits[name] + 2.0:
            name = on
        return self.t + d * hits[name], name

    def depth_hit(self, u: float, v: float) -> Optional[np.ndarray]:
        """The photo depth's point for fraction (u, v); None past the model's far clamp."""
        if self.context is None:
            return None
        from pipeline.pipeline_context import ContextKey
        intr = self.context.input_intrinsics(ContextKey.INTRINSICS)
        depth = self.context.input_depth(ContextKey.DEPTH)
        if intr is None or depth is None:
            return None
        arr = np.asarray(depth.depth, dtype=float)
        h, w = arr.shape[:2]
        x, y = min(w - 1, max(0, int(u * w))), min(h - 1, max(0, int(v * h)))
        patch = arr[max(0, y - 3):y + 4, max(0, x - 3):x + 4]
        valid = patch[np.isfinite(patch) & (patch > 0)]
        if valid.size == 0:
            return None
        z = float(np.median(valid))
        if z > 0.9 * float(np.nanmax(arr)):
            return None
        return self.t + self.R @ np.array(intr.unproject(u * intr.width, v * intr.height, z), dtype=float)

    def resolve_at(self, at: dict, on: str, current: Optional[np.ndarray] = None) -> tuple[np.ndarray, str]:
        """A position from an `at` spec: (world xyz with y snapped, how it was found)."""
        if not isinstance(at, dict):
            raise EditError("`at` must be an object: world, bearing_deg+distance_m, photo_box or delta")
        snap = at.get("snap", True)
        how = ""
        if at.get("world") is not None:
            p = _v(at["world"])
            how = "world"
        elif at.get("bearing_deg") is not None and at.get("distance_m") is not None:
            p = self.world_from_bearing(at["bearing_deg"], at["distance_m"])
            how = f"bearing {float(at['bearing_deg']):.1f} deg, {float(at['distance_m']):.1f} m"
        elif at.get("photo_box") is not None:
            x, y, w, h = [float(v) for v in at["photo_box"]]
            if max(x, y, w, h) > 1.5:
                raise EditError("photo_box is in fractions of the photograph (0-1), not pixels")
            u, v = x + w / 2.0, y + h
            hit = self.ray_hit(u, v, on)
            if hit is not None:
                p, surface = hit
                b, dist = self.bearing_distance(p)
                how = f"photo box ray hits the {surface} at {dist:.1f} m, bearing {b:.1f} deg"
                if on == "ground" and surface == "water":
                    on = "water"
            else:
                p = self.depth_hit(u, v)
                if p is None:
                    raise EditError("photo_box: the ray misses the terrain and the depth is at its far clamp; give bearing_deg and distance_m")
                b, dist = self.bearing_distance(p)
                how = f"photo depth at {dist:.1f} m, bearing {b:.1f} deg"
        elif at.get("delta") is not None:
            if current is None:
                raise EditError("delta needs an existing object")
            p = np.asarray(current, dtype=float) + _v(at["delta"])
            how = f"delta {at['delta']}"
            snap = at.get("snap", False)
        else:
            raise EditError("`at` needs world, bearing_deg+distance_m, photo_box or delta")
        if snap and on != "none":
            y = self.surface_y(p, on)
            if y is None:
                how += "; no surface under it, y kept"
            else:
                p = np.array([p[0], y, p[2]])
                how += f"; on the {on} at y {y:.2f}"
        return p, how

    def _place_y(self, mesh: str, scale: float, surface_y: float) -> float:
        b = self.mesh_bounds(mesh)
        return surface_y - (float(b[0][1]) * scale if b is not None else 0.0)

    def _scale_for_height(self, mesh: str, height_m: float, fallback: float) -> float:
        b = self.mesh_bounds(mesh)
        if b is None:
            return fallback
        ext = float(b[1][1] - b[0][1])
        return float(height_m) / ext if ext > 1e-6 else fallback

    # ── actions ──────────────────────────────────────────────────────────────
    def apply(self, action: dict, who: str = "agent") -> dict:
        kind = str(action.get("action") or "")
        if kind not in ACTIONS:
            raise EditError(f"unknown action '{kind}'; one of {', '.join(ACTIONS)}")
        rec = {"action": kind, "who": who, "why": str(action.get("why") or ""), "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        fn = getattr(self, f"_do_{kind}")
        rec.update(fn(action))
        rec["input"] = {k: v for k, v in action.items() if k not in ("why",)}
        self.applied.append(rec)
        self.log(f"  {kind}: {rec.get('note', '')}")
        return rec

    def _guard(self, o: dict, action: dict) -> None:
        if self.is_env(o) and not action.get("force"):
            raise EditError(f"'{o.get('name')}' is an environment mesh; set force: true to touch it")

    def _do_place(self, a: dict) -> dict:
        mesh = str(a.get("mesh") or "")
        if not mesh:
            raise EditError("place needs a mesh key")
        if self.context is not None and self.mesh_bounds(mesh) is None:
            raise EditError(f"the run has no mesh '{mesh}'; known keys: {', '.join(self.mesh_keys()[:12])}")
        cls = str(a.get("class") or mesh.replace("category_mesh_", "").split("_")[0])
        on = str(a.get("on") or ("water" if cls in WATER_CLASSES else "ground"))
        p, how = self.resolve_at(a.get("at") or {}, on)
        height = a.get("height_m")
        if height is None:
            like = [o for o in self.objects if o.get("mesh") == mesh]
            scale = float(_v(like[0].get("scale"), (1, 1, 1))[1]) if like else 1.0
        else:
            scale = self._scale_for_height(mesh, float(height), 1.0)
        yaw = float(a.get("yaw_deg") or 0.0)
        y = self._place_y(mesh, scale, float(p[1])) if a.get("at", {}).get("snap", True) and on != "none" else float(p[1])
        o = {"id": str(uuid.uuid4()), "position": _xyz([p[0], y, p[2]]), "rotation": _xyz([0.0, yaw, 0.0]),
             "scale": _xyz([scale, scale, scale]), "type": "mesh", "texture": "", "mesh": mesh,
             "name": str(a.get("name") or f"agent_{cls}"), "agent": {"who": "agent", "class": cls}}
        self.objects.append(o)
        b, dist = self.bearing_distance([p[0], y, p[2]])
        far = float(self.scene.get("farClipPlane") or 0.0)
        warn = f"; beyond the far clip ({far:.0f} m), it will not draw" if far and dist > far else ""
        return {"id": o["id"], "name": o["name"], "mesh": mesh, "position": o["position"], "scale": round(scale, 4),
                "note": f"{o['name']} ({mesh}) at {dist:.1f} m bearing {b:.1f} deg, height {height if height is not None else 'as others'} m; {how}{warn}"}

    def _do_move(self, a: dict) -> dict:
        o = self.find(a.get("target"))
        self._guard(o, a)
        cls = str(a.get("class") or (o.get("agent") or {}).get("class") or str(o.get("mesh") or "").replace("category_mesh_", "").split("_")[0])
        on = str(a.get("on") or ("water" if cls in WATER_CLASSES else "ground"))
        cur = _v(o.get("position"))
        p, how = self.resolve_at(a.get("at") or {}, on, current=cur)
        scale = float(_v(o.get("scale"), (1, 1, 1))[1])
        snapped = (a.get("at") or {}).get("snap", (a.get("at") or {}).get("delta") is None)
        y = self._place_y(str(o.get("mesh") or ""), scale, float(p[1])) if snapped and on != "none" else float(p[1])
        old = o["position"]
        o["position"] = _xyz([p[0], y, p[2]])
        b, dist = self.bearing_distance([p[0], y, p[2]])
        return {"id": o["id"], "name": o.get("name"), "old": old, "position": o["position"],
                "note": f"{o.get('name')} to {dist:.1f} m bearing {b:.1f} deg; {how}"}

    def _do_scale(self, a: dict) -> dict:
        o = self.find(a.get("target"))
        self._guard(o, a)
        old = _v(o.get("scale"), (1, 1, 1))
        if a.get("height_m") is not None:
            s = self._scale_for_height(str(o.get("mesh") or ""), float(a["height_m"]), float(old[1]))
        elif a.get("factor") is not None:
            s = float(old[1]) * float(a["factor"])
        else:
            raise EditError("scale needs height_m or factor")
        # Keep the base where it was.
        b = self.mesh_bounds(str(o.get("mesh") or ""))
        base_y = float(_v(o.get("position"))[1]) + (float(b[0][1]) * float(old[1]) if b is not None else 0.0)
        o["scale"] = _xyz([s, s, s])
        pos = _v(o.get("position"))
        o["position"] = _xyz([pos[0], base_y - (float(b[0][1]) * s if b is not None else 0.0), pos[2]])
        h = (float(b[1][1] - b[0][1]) * s) if b is not None else None
        return {"id": o["id"], "name": o.get("name"), "old_scale": round(float(old[1]), 4), "scale": round(s, 4),
                "note": f"{o.get('name')} scale {float(old[1]):.3f} -> {s:.3f}" + (f" ({h:.1f} m tall)" if h else "")}

    def _do_rotate(self, a: dict) -> dict:
        o = self.find(a.get("target"))
        self._guard(o, a)
        r = _v(o.get("rotation"))
        yaw = float(a["yaw_deg"]) if a.get("yaw_deg") is not None else (r[1] + float(a.get("delta_deg") or 0.0))
        o["rotation"] = _xyz([r[0], yaw % 360.0, r[2]])
        return {"id": o["id"], "name": o.get("name"), "note": f"{o.get('name')} yaw {r[1]:.1f} -> {yaw % 360.0:.1f} deg"}

    def _do_remove(self, a: dict) -> dict:
        o = self.find(a.get("target"))
        self._guard(o, a)
        self.objects.remove(o)
        b, dist = self.bearing_distance(_v(o.get("position")))
        return {"id": o["id"], "name": o.get("name"), "removed": o,
                "note": f"removed {o.get('name')} ({o.get('mesh')}) at {dist:.1f} m bearing {b:.1f} deg"}

    def _do_swap_mesh(self, a: dict) -> dict:
        o = self.find(a.get("target"))
        self._guard(o, a)
        mesh = str(a.get("mesh") or "")
        if self.context is not None and self.mesh_bounds(mesh) is None:
            raise EditError(f"the run has no mesh '{mesh}'")
        old = o.get("mesh")
        oldb, newb = self.mesh_bounds(str(old or "")), self.mesh_bounds(mesh)
        s = float(_v(o.get("scale"), (1, 1, 1))[1])
        if oldb is not None and newb is not None and a.get("keep_height", True):
            h = float(oldb[1][1] - oldb[0][1]) * s
            s = self._scale_for_height(mesh, h, s)
            base_y = float(_v(o.get("position"))[1]) + float(oldb[0][1]) * float(_v(o.get("scale"), (1, 1, 1))[1])
            pos = _v(o.get("position"))
            o["position"] = _xyz([pos[0], base_y - float(newb[0][1]) * s, pos[2]])
            o["scale"] = _xyz([s, s, s])
        o["mesh"] = mesh
        if o.get("lods"):
            o["lods"] = []
        return {"id": o["id"], "name": o.get("name"), "old_mesh": old, "mesh": mesh, "note": f"{o.get('name')} mesh {old} -> {mesh}"}

    def _do_set_atmosphere(self, a: dict) -> dict:
        atm = dict(self.scene.get("atmosphere") or {})
        old = dict(atm)
        for k in ("haze_near_m", "haze_far_m", "haze_color", "tint_rgb"):
            if a.get(k) is not None:
                atm[k] = a[k]
        atm["evidence"] = "agent"
        self.scene["atmosphere"] = atm
        return {"old": old, "atmosphere": atm, "note": f"atmosphere {old} -> {atm}"}

    def _do_set_sun(self, a: dict) -> dict:
        L = self.scene.get("lighting")
        if not isinstance(L, dict) or not isinstance(L.get("sun"), dict):
            raise EditError("the scene has no lighting.sun to edit")
        sun = L["sun"]
        old = {k: sun.get(k) for k in ("direction", "elevation_deg", "intensity", "color")}
        d = np.array(sun.get("direction") or [0, 1, 0], dtype=float)
        az, el = math.degrees(math.atan2(d[0], d[2])), math.degrees(math.asin(max(-1.0, min(1.0, d[1]))))
        if a.get("azimuth_deg") is not None:
            az = float(a["azimuth_deg"])
        if a.get("elevation_deg") is not None:
            el = float(a["elevation_deg"])
        r_az, r_el = math.radians(az), math.radians(el)
        sun["direction"] = [round(math.cos(r_el) * math.sin(r_az), 5), round(math.sin(r_el), 5), round(math.cos(r_el) * math.cos(r_az), 5)]
        sun["elevation_deg"] = round(el, 2)
        sun["direction_source"] = "agent"
        if a.get("intensity") is not None:
            sun["intensity"] = float(a["intensity"])
        if a.get("color"):
            sun["color"] = str(a["color"])
        new = {k: sun.get(k) for k in ("direction", "elevation_deg", "intensity", "color")}
        return {"old": old, "sun": new, "note": f"sun az {az:.1f} el {el:.1f} intensity {sun.get('intensity')} colour {sun.get('color')}"}

    def _do_set_light(self, a: dict) -> dict:
        """Multiply the rendered sun and ambient light. The renderer draws sun.intensity as
        written (Exposure Fit already folded its sun_scale in) and ambient as the sky mean x
        lighting.fit.ambient_scale, so sun_scale edits sun.intensity and ambient_scale edits
        the fit's ambient_scale; both are multipliers on what is there."""
        L = self.scene.get("lighting")
        if not isinstance(L, dict) or not isinstance(L.get("sun"), dict):
            raise EditError("the scene has no lighting to edit")
        ss, as_ = a.get("sun_scale"), a.get("ambient_scale")
        if ss is None and as_ is None:
            raise EditError("set_light needs sun_scale and/or ambient_scale")
        for v, n in ((ss, "sun_scale"), (as_, "ambient_scale")):
            if v is not None and not (0.05 <= float(v) <= 20.0):
                raise EditError(f"{n} {v} is outside 0.05-20")
        sun = L["sun"]
        fit = L.setdefault("fit", {})
        old = {"sun_intensity": sun.get("intensity"), "ambient_scale": fit.get("ambient_scale")}
        if ss is not None:
            sun["intensity"] = round(float(sun.get("intensity") or 1.0) * float(ss), 4)
            fit["agent_sun_scale"] = round(float(fit.get("agent_sun_scale") or 1.0) * float(ss), 4)
        if as_ is not None:
            fit["ambient_scale"] = round(float(fit.get("ambient_scale") or 1.0) * float(as_), 4)
            fit["agent_ambient_scale"] = round(float(fit.get("agent_ambient_scale") or 1.0) * float(as_), 4)
        new = {"sun_intensity": sun.get("intensity"), "ambient_scale": fit.get("ambient_scale")}
        return {"old": old, "light": new, "note": f"sun intensity {old['sun_intensity']} -> {new['sun_intensity']}, "
                                                  f"ambient scale {old['ambient_scale']} -> {new['ambient_scale']}"}

    # ── files ────────────────────────────────────────────────────────────────
    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.scene, indent=1))
        return path


def apply_all(scene: dict, actions: list[dict], context=None, run_dir: Optional[Path] = None, who: str = "agent",
              log=print, strict: bool = False) -> tuple[dict, list[dict], list[dict]]:
    """Apply actions to a copy of the scene. Returns (scene, applied records, failures)."""
    ed = Editor(copy.deepcopy(scene), context=context, run_dir=run_dir, log=log)
    failures = []
    for a in actions:
        try:
            ed.apply(a, who=who)
        except EditError as exc:
            failures.append({"input": a, "error": str(exc)})
            log(f"  {a.get('action')}: refused: {exc}")
            if strict:
                raise
    return ed.scene, ed.applied, failures


def load_context(run_dir: Path, config_path: Path):
    """The cached run as a PipelineContext positioned after its last stage (as render-scene does)."""
    import tempfile
    from pipeline.pipeline import PipelineConfiguration
    from pipeline.pipeline_context import PipelineContext
    scratch = Path(tempfile.mkdtemp(prefix="scene-edit-"))
    config = PipelineConfiguration(output=str(scratch), config_path=Path(config_path), log_mode="plain")
    entries = config.stages_yaml
    names = [e["name"] for e in entries]
    enabled = [e["name"] for e in entries if e.get("enabled", True)]
    context = PipelineContext()
    context.load(run_dir, enabled, names)
    for name in enabled:
        context.push_stage(name)
        context.pop_stage()
    context.push_stage(enabled[-1])
    return context


def newest_scene_path(run_dir: Path) -> Path:
    """The scene.json the renderer draws: the newest any stage wrote."""
    cands = sorted(run_dir.glob("*/scene.json"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else run_dir / "Scene Generation" / "scene.json"
