"""Scene Motion: how everything in the built scene moves, written into scene.json for
the client and the offline renderer.

- Water: the flow field over the water sheet (flow_field.py), a river's current or a
  sea's swell toward the shore, as a flow map (R, G: velocity, B: water) plus a
  full-precision copy beside it.
- Boats: underway ones follow the flow line through their position at their speed,
  holding the angle they make with it (so a boat photographed stern-on stays stern-on
  to its course) and the lane; every boat bobs at its size.
- Plants: one sway per kind (stiffness class, bend amplitude, rate), driven by a
  shared wind field so neighbours move together.
- Sky: the cloud layer drifts with the wind aloft.

What moves which way, and how fast, comes from the video's measurements
(motion_measurements, when the motion learning step has run) and the Motion Director's
reading of the photograph, each held to a physical range; the geometry says where
(the river's course, the banks, the boats' lanes). The motion is played by the client
and by render-scene.py --animate; the one thing set at build time is a moored boat's
berth: it lies along its bank, in the water (see _berth).

Reads:  ContextKey.SCENE, metadata_{i}, the water meshes, motion_measurements (optional)
Writes: ContextKey.SCENE (scene.motion, object.motion), water_flow_map (image),
        scene_motion (summary)"""
from __future__ import annotations

import math
import re
from logging import Logger
from typing import Any, Optional

import numpy as np

from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration
from pipeline.object_typing.categories import VEGETATION_CATEGORIES
from pipeline.scene_motion.flow_field import FlowField, build_flow_field

# Ranges every number is held to, whoever proposed it (degrees, Hz).
_STIFFNESS = {
    "conifer":   {"amplitude": (0.5, 4.0), "frequency": (0.15, 0.6), "default": (2.0, 0.3)},
    "broadleaf": {"amplitude": (1.0, 6.0), "frequency": (0.25, 0.9), "default": (3.0, 0.45)},
    "willow":    {"amplitude": (2.0, 8.0), "frequency": (0.2, 0.6), "default": (4.0, 0.35)},
    "palm":      {"amplitude": (2.0, 8.0), "frequency": (0.2, 0.7), "default": (4.0, 0.4)},
    "shrub":     {"amplitude": (2.0, 8.0), "frequency": (0.4, 1.2), "default": (4.0, 0.7)},
    "herb":      {"amplitude": (4.0, 20.0), "frequency": (0.6, 2.0), "default": (9.0, 1.1)},
    "grass":     {"amplitude": (5.0, 25.0), "frequency": (0.8, 2.5), "default": (12.0, 1.4)},
}
_CLOUD_SPEED = {"still": 0.0, "slow": 0.015, "moderate": 0.04, "fast": 0.1}   # degrees of sky per second


def _clamp(v, lo, hi, default):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(v):
        return float(default)
    return float(min(max(v, lo), hi))


def _trs(position, rotation, scale) -> np.ndarray:
    """Unity-style TRS in the server frame (as render-scene.py's transform)."""
    rx, ry, rz = (math.radians(float(rotation[k])) for k in ("x", "y", "z"))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    R = (np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
         @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]]))
    M = np.eye(4)
    M[:3, :3] = R @ np.diag([float(scale[k]) for k in ("x", "y", "z")])
    M[:3, 3] = [float(position[k]) for k in ("x", "y", "z")]
    return M


def kind_key(mesh_key: str) -> str:
    """One kind of plant across its variants and LODs: category_mesh_tree_0_v1 / _card -> category_mesh_tree_0."""
    return re.sub(r"(_v\d+|_card)+$", "", str(mesh_key or ""))


class SceneMotionConfiguration(PipelineStageConfiguration):
    def __init__(self, name: str, device, torch_dtype: Any, log: Logger, keys=None, seed: int = 0,
                 flow_cell_m: float = 1.0, flow_half_extent_m: float = 200.0, river_speed_m_s: float = 0.6,
                 boat_speed_range_m_s: tuple = (1.0, 5.0), path_length_m: float = 800.0, path_step_m: float = 4.0,
                 director_model: str = "anthropic:claude-sonnet-5-5", director_max_usd: float = 0.3,
                 director_no_api: bool = False, director_refresh: bool = False):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.flow_cell_m = float(flow_cell_m)
        self.flow_half_extent_m = float(flow_half_extent_m)
        self.river_speed_m_s = float(river_speed_m_s)
        self.boat_speed_range_m_s = tuple(float(v) for v in boat_speed_range_m_s)
        self.path_length_m = float(path_length_m)
        self.path_step_m = float(path_step_m)
        self.director_model = str(director_model)
        self.director_max_usd = float(director_max_usd)
        self.director_no_api = bool(director_no_api)
        self.director_refresh = bool(director_refresh)


class SceneMotionStage(PipelineStage):
    @classmethod
    def config_class(cls) -> type[SceneMotionConfiguration]:
        return SceneMotionConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: SceneMotionConfiguration = self.config
        scene = context.input_scene(ContextKey.SCENE)
        if scene is None or not scene.objects:
            self.log_info("No scene, skipping")
            return context
        R = np.asarray(scene.extrinsics.rotation, dtype=float).reshape(3, 3)
        cam = np.asarray(scene.extrinsics.translation, dtype=float).ravel()
        fwd = np.array([R[0, 2], R[2, 2]]); fwd /= max(np.linalg.norm(fwd), 1e-9)
        right = np.array([R[0, 0], R[2, 0]]); right /= max(np.linalg.norm(right), 1e-9)

        def world_dir(name: str, at: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
            """A direction in the photograph's terms on the ground plane; toward/away are
            along the line from the camera to `at` when given (a boat off to one side
            moving 'away' moves along its own sight line, not the camera's axis)."""
            axis = fwd
            if at is not None:
                d = np.asarray(at, dtype=float) - cam[[0, 2]]
                if np.linalg.norm(d) > 1e-6:
                    axis = d / np.linalg.norm(d)
            return {"left": -right, "right": right, "away_from_camera": axis, "toward_camera": -axis}.get(name)

        yaw_of = lambda v: float(math.degrees(math.atan2(v[0], v[1])))

        # ── water ───────────────────────────────────────────────────────────
        flow: Optional[FlowField] = None
        verts, faces, off = [], [], 0
        for o in scene.objects:
            if o.name != "water" or not o.mesh:
                continue
            m = context.input_mesh(o.mesh)
            if m is None:
                continue
            T = _trs(o.position, o.rotation, o.scale)
            v = np.asarray(m.mesh.vertices, dtype=float)
            verts.append((np.c_[v, np.ones(len(v))] @ T.T)[:, :3]); faces.append(np.asarray(m.mesh.faces) + off); off += len(v)
        water_y = float(np.median(np.concatenate(verts)[:, 1])) if verts else None
        if verts:
            flow = build_flow_field(np.concatenate(verts), np.concatenate(faces), cfg.flow_half_extent_m, cfg.flow_cell_m,
                                    speed_m_s=cfg.river_speed_m_s)
            self.log_info(f"Water: {flow.kind}, {int(flow.water.sum()) * cfg.flow_cell_m ** 2:.0f} m^2"
                          + (f", ends at {[tuple(round(c) for c in e) for e in flow.ends]}" if flow.ends else ""))
        bank = None
        if flow is not None and flow.water.any():
            from scipy import ndimage
            bank = ndimage.distance_transform_edt(flow.water) * flow.cell_m

        def bank_distance(x: float, z: float) -> float:
            if bank is None:
                return 0.0
            r, c = flow._cell(x, z)
            r = int(np.clip(round(float(r)), 0, bank.shape[0] - 1)); c = int(np.clip(round(float(c)), 0, bank.shape[1] - 1))
            return float(bank[r, c])

        # ── what moves ──────────────────────────────────────────────────────
        boats, plant_kinds = [], {}
        for o in scene.objects:
            idx = o.source_index
            meta = (context.input_object(f"metadata_{idx}") or {}) if idx is not None else {}
            cls = str(meta.get("class") or "")
            key = o.mesh or ""
            if not key:
                continue
            mesh = context.input_mesh(key) if (cls in ("boat", "ship") or key.startswith("category_mesh_boat")) else None
            if mesh is not None:
                ext = np.asarray(mesh.extents, dtype=float) * np.array([abs(float(o.scale[k])) for k in ("x", "y", "z")])
                pos = np.array([float(o.position["x"]), float(o.position["z"])])
                boats.append({"id": len(boats), "obj": o, "name": meta.get("name") or cls or "boat",
                              "length_m": float(max(ext[0], ext[2])), "pos": pos,
                              "photo_box": [float(v) for v in meta["photo_box"]] if meta.get("photo_box") else None,
                              "bank_m": bank_distance(*pos), "on_water": bool(flow is not None and flow.on_water(*pos)[0])})
                continue
            if cls in VEGETATION_CATEGORIES or re.match(r"category_mesh_(tree|flower|plant|bush|shrub|grass)", key):
                k = kind_key(key)
                entry = plant_kinds.setdefault(k, {"id": k, "class": cls or k.split("_")[2], "name": meta.get("name") or cls,
                                                   "heights": [], "count": 0})
                entry["count"] += 1
                if len(entry["heights"]) < 50:
                    m = context.input_mesh(key)
                    if m is not None:
                        entry["heights"].append(float(m.extents[1]) * abs(float(o.scale["y"])))
        for p in plant_kinds.values():
            p["height_m"] = float(np.median(p["heights"])) if p["heights"] else 0.0

        # ── the Motion Director ─────────────────────────────────────────────
        measurements = context.input_object("motion_measurements") or context.object("motion_measurements")
        reading = None
        photo = context.input_image(ContextKey.INPUT)
        if photo is not None and (boats or plant_kinds or flow is not None):
            try:
                from pipeline.scene_motion.motion_director import MotionDirector
                from pipeline.scene_plan.scene_plan import photo_sha256
                director = MotionDirector(cfg.director_model, max_usd_per_run=cfg.director_max_usd, no_api=cfg.director_no_api,
                                          refresh=cfg.director_refresh, log=self.log_info, warn=self.log_warning)
                pil = photo.image.convert("RGB")
                facts = (f"water: {flow.kind if flow else 'none'}"
                         + (f" (a river crossing the scene from {tuple(round(c) for c in flow.ends[0])} to "
                            f"{tuple(round(c) for c in flow.ends[1])} m, camera at the origin looking along +z)" if flow and flow.kind == "river" else ""))
                reading = director.direct(pil, photo_sha256(pil),
                                          [{k: b[k] for k in ("id", "name", "length_m", "photo_box", "bank_m")} for b in boats],
                                          [{k: p[k] for k in ("id", "class", "name", "height_m", "count")} for p in plant_kinds.values()],
                                          facts, measurements)
            except Exception as exc:
                self.log_warning(f"Motion Director unavailable ({type(exc).__name__}: {exc}); motion from the priors")
        reading = reading if isinstance(reading, dict) else {}
        for part in ("wind", "water", "clouds"):
            if not isinstance(reading.get(part), dict):
                reading[part] = {}
        if reading:
            self.log_info(f"Motion Director ({reading.get('model')}, ${reading.get('usd', 0):.3f}): wind toward "
                          f"{reading.get('wind', {}).get('toward')}, water {reading.get('water', {}).get('flow_toward')}, "
                          f"clouds {reading.get('clouds', {}).get('drift_toward')}")

        # ── wind, water, sky ────────────────────────────────────────────────
        wind = reading.get("wind") or {}
        wdir = world_dir(str(wind.get("toward")))
        if wdir is None:
            axis = next((o.sway.get("axisDegrees") for o in scene.objects if o.sway and o.sway.get("axisDegrees") is not None), 0.0)
            wdir = np.array([math.sin(math.radians(axis)), math.cos(math.radians(axis))])
        beaufort = _clamp(wind.get("beaufort"), 0.0, 6.0, 2.0)
        wind_speed = 0.836 * beaufort ** 1.5
        motion: dict = {"wind": {"directionDeg": yaw_of(wdir), "speedMs": round(wind_speed, 2),
                                 "gustiness": _clamp(wind.get("gustiness"), 0.0, 1.0, 0.4)}}

        water = reading.get("water") or {}
        if flow is not None and flow.kind != "none":
            if flow.kind == "river":
                speed = _clamp(water.get("speed_m_s"), 0.1, 2.0, cfg.river_speed_m_s)
                along = np.asarray(flow.ends[1]) - np.asarray(flow.ends[0])
                want = world_dir(str(water.get("flow_toward")))
                sign = -1.0 if (want is not None and float(np.dot(want, along)) < 0) else 1.0
                flow = flow.scaled(speed, sign)
                self.log_info(f"River: {speed:.2f} m/s from {tuple(round(c) for c in flow.ends[0])} to "
                              f"{tuple(round(c) for c in flow.ends[1])}" + ("" if want is not None else " (direction not read; as the geometry found it)"))
            vmax = float(max(np.abs(flow.vx).max(), np.abs(flow.vz).max(), 1e-6))
            rgba = np.zeros(flow.water.shape + (4,), dtype=np.uint8)
            rgba[..., 0] = np.clip(np.rint(127.5 + 127.5 * flow.vx / vmax), 0, 255)
            rgba[..., 1] = np.clip(np.rint(127.5 + 127.5 * flow.vz / vmax), 0, 255)
            rgba[..., 2] = np.where(flow.water, 255, 0); rgba[..., 3] = 255
            from PIL import Image as PILImage
            from util.image_utils import Image
            context.add_image("water_flow_map", Image(PILImage.fromarray(rgba, "RGBA")))
            try:
                flow.save(self.output / "water_flow.npz")
            except Exception as exc:
                self.log_warning(f"water_flow.npz not written ({exc})")
            # The surf's speed toward the shore, measured where the video shows it (Shark
            # Fin's fronts at 3.1 m/s, what shallow water ~1 m deep carries).
            wave_speed = 3.0
            mw = ((measurements or {}).get("regions") or {}).get("water") or {}
            ms = mw.get("speed_m_s") or {}
            if flow.kind == "sea" and float(ms.get("confidence") or 0.0) >= 0.5 and ms.get("median") is not None:
                wave_speed = _clamp(ms["median"], 1.0, 8.0, 3.0)
                self.log_info(f"Surf: fronts run to the shore at {wave_speed:.2f} m/s (measured, n={ms.get('n')})")
            motion["water"] = {"kind": flow.kind, "flowMap": "water_flow_map", "origin": [float(v) for v in flow.origin],
                               "waveSpeedMs": round(wave_speed, 2),
                               "cellM": flow.cell_m, "maxSpeedMs": round(vmax, 3), "speedMs": round(flow.speed_m_s, 3),
                               "swellPeriodS": _clamp(water.get("swell_period_s"), 4.0, 18.0, 10.0) if flow.kind == "sea" else 0.0,
                               "rowsAlongZ": True}

        clouds = reading.get("clouds") or {}
        cdir = world_dir(str(clouds.get("drift_toward"))) if clouds else None
        cdir = wdir if cdir is None else cdir
        motion["sky"] = {"directionDeg": yaw_of(cdir), "angularSpeedDegS": _CLOUD_SPEED.get(str(clouds.get("speed")), 0.015)}

        # ── boats ───────────────────────────────────────────────────────────
        said = {int(b.get("id", -1)): b for b in (reading.get("boats") or []) if isinstance(b, dict)}
        underway = moored = 0
        for b in boats:
            o, L = b["obj"], b["length_m"]
            r = said.get(b["id"], {})
            role = str(r.get("role") or ("moored" if b["bank_m"] < 0.6 * L / 4 + 2.0 else "underway"))
            bob = {"heaveM": round(_clamp(0.02 + 0.6 / max(L, 1.0), 0.02, 0.15, 0.05), 3),
                   "rollDeg": round(_clamp(25.0 / max(L, 1.0), 0.3, 3.0, 1.0), 2),
                   "pitchDeg": round(_clamp(10.0 / max(L, 1.0), 0.1, 1.5, 0.4), 2),
                   "periodS": round(_clamp(0.6 * math.sqrt(max(L, 1.0)), 2.0, 6.0, 3.0), 2),
                   "phase": float(np.random.default_rng((self.seed, b["id"])).uniform(0, 2 * math.pi))}
            entry = {"type": "floating_moored", "bob": bob, "why": str(r.get("why") or "")}
            travel = world_dir(str(r.get("travel_toward")), b["pos"])
            if role == "underway" and flow is not None and b["on_water"] and flow.kind == "river":
                v = flow.sample(*b["pos"])[0]
                if np.linalg.norm(v) < 1e-6:
                    v = np.asarray(flow.ends[1]) - np.asarray(flow.ends[0])
                if travel is None:
                    yaw = math.radians(float(o.rotation["y"]))
                    travel = np.array([math.sin(yaw), math.cos(yaw)])
                sign = 1.0 if float(np.dot(travel, v)) >= 0 else -1.0
                path = flow.streamline(*b["pos"], cfg.path_length_m, cfg.path_step_m, sign)
                if len(path) >= 3:
                    lo, hi = cfg.boat_speed_range_m_s
                    speed = _clamp(r.get("speed_m_s"), lo, hi, _clamp(0.6 * math.sqrt(L), lo, hi, 3.0))
                    t0 = path[1] - path[0]
                    entry = {"type": "floating_underway", "speedMs": round(speed, 2), "bob": bob,
                             "path": [[round(float(x), 2), round(float(z), 2)] for x, z in path],
                             # The angle between the model's +Z and its course, held along the path.
                             "yawOffsetDeg": round(float(o.rotation["y"]) - yaw_of(t0), 2),
                             "withCurrent": bool(sign > 0), "why": str(r.get("why") or "")}
                    underway += 1
                    self.log_info(f"  {o.name} (#{o.source_index}, {L:.0f} m): underway {'with' if sign > 0 else 'against'} the "
                                  f"current at {speed:.1f} m/s, course {len(path) * cfg.path_step_m:.0f} m along the river")
            if entry["type"] == "floating_moored":
                if flow is not None and bank is not None and flow.water.any():
                    berth = self._berth(o, context.input_mesh(o.mesh), flow, bank, water_y)
                    if berth:
                        self.log_info(f"  {o.name} (#{o.source_index}): moored {berth}")
                moored += 1
                self.log_info(f"  {o.name} (#{o.source_index}, {L:.0f} m): {role if role != 'underway' else 'afloat'}, bobbing")
            o.motion = entry

        # ── plants ──────────────────────────────────────────────────────────
        said_p = {str(p.get("id")): p for p in (reading.get("plants") or []) if isinstance(p, dict)}
        veg = {}
        for k, p in plant_kinds.items():
            r = said_p.get(k, {})
            stiff = str(r.get("stiffness") or ("conifer" if "conifer" in str(p["name"]).lower() or "fir" in str(p["name"]).lower()
                                               else "broadleaf" if p["class"] == "tree" else "herb"))
            rng = _STIFFNESS.get(stiff, _STIFFNESS["herb"])
            veg[k] = {"stiffness": stiff,
                      "amplitudeDeg": round(_clamp(r.get("amplitude_deg"), *rng["amplitude"], rng["default"][0]), 2),
                      "frequencyHz": round(_clamp(r.get("frequency_hz"), *rng["frequency"], rng["default"][1]), 3),
                      "heightM": round(p["height_m"], 2)}
        if veg:
            motion["vegetation"] = veg
        scene.motion = motion
        context.add_scene(ContextKey.SCENE, scene)
        summary = {"boats_underway": underway, "boats_moored": moored, "plant_kinds": len(veg),
                   "water": motion.get("water", {}).get("kind", "none"), "director": bool(reading)}
        context.add_object("scene_motion", summary)
        self.log_info(f"Motion: {underway} boat(s) underway, {moored} moored, {len(veg)} kind(s) of plant swaying, "
                      f"wind toward {motion['wind']['directionDeg']:.0f} deg at {wind_speed:.1f} m/s, "
                      f"clouds {motion['sky']['angularSpeedDegS']:.3f} deg/s")
        return context

    @staticmethod
    def _berth(o, mesh, flow: FlowField, bank: np.ndarray, water_y: Optional[float]) -> str:
        """A moored boat alongside its bank, in the water. The photo fit sets a boat
        where its pixels' depth puts it and turns it by its box, which for a hull
        seen end-on or cut by the frame is any angle: Paris r3b's houseboat on the
        left quay stood across the river, its bow on the quay 2 m above the water,
        read as a box on the bank. A moored hull lies along the shore beside it: its
        long axis is turned to the bank's line (the nearer of the two ways round)
        when it is more than 10 deg off, and a hull with more than a fifth of its
        footprint on land is moved to the nearest water a half-beam off the bank
        (at most its own length away) and set on the water. Returns what was done."""
        if mesh is None or abs(float(o.rotation["x"])) > 1e-3 or abs(float(o.rotation["z"])) > 1e-3:
            return ""
        v = np.asarray(mesh.mesh.vertices if hasattr(mesh, "mesh") else mesh.vertices, dtype=float)
        if v.size == 0:
            return ""
        sc = np.array([abs(float(o.scale[k])) for k in ("x", "y", "z")])
        lo, hi = v.min(0) * sc, v.max(0) * sc
        ext = hi - lo
        along_x = ext[0] >= ext[2]
        L, beam = float(max(ext[0], ext[2])), float(min(ext[0], ext[2]))
        if L < 2.0 * beam * 0.75:
            return ""                                     # no long axis to lie along
        from scipy import ndimage
        pos = np.array([float(o.position["x"]), float(o.position["z"])])

        def to_rc(x, z):
            r, c = flow._cell(x, z)
            return (int(np.clip(round(float(r)), 0, bank.shape[0] - 1)), int(np.clip(round(float(c)), 0, bank.shape[1] - 1)))

        def axis(yaw_deg):
            y = math.radians(yaw_deg)
            return np.array([math.cos(y), -math.sin(y)]) if along_x else np.array([math.sin(y), math.cos(y)])

        def on_land(p, yaw_deg):
            a = axis(yaw_deg); n = np.array([-a[1], a[0]])
            pts = [p + a * s * L / 2 + n * t * beam / 2 for s in (-1, -0.5, 0, 0.5, 1) for t in (-1, 0, 1)]
            return float(np.mean([not flow.on_water(*q)[0] for q in pts]))

        # The bank's line: across the gradient of the distance to the shore, at the
        # nearest water a hull's half-beam in.
        want_d = beam / 2.0 + 0.5
        ok = bank >= want_d
        if not ok.any():
            return ""
        r0, c0 = to_rc(*pos)
        dist, (ir, ic) = ndimage.distance_transform_edt(~ok, return_indices=True)
        rr, cc = int(ir[r0, c0]), int(ic[r0, c0])
        gy, gx = np.gradient(ndimage.gaussian_filter(bank.astype(np.float32), 2.0))
        g = np.array([gx[rr, cc], gy[rr, cc]], dtype=float)           # (x, z) per cell
        if np.linalg.norm(g) < 1e-6:
            return ""
        t = np.array([-g[1], g[0]]) / np.linalg.norm(g)
        yaw0 = float(o.rotation["y"])
        base = math.degrees(math.atan2(-t[1], t[0])) if along_x else math.degrees(math.atan2(t[0], t[1]))
        cands = [base, base + 180.0]
        yaw = min(cands, key=lambda c: abs((c - yaw0 + 180.0) % 360.0 - 180.0))
        off = abs((yaw - yaw0 + 180.0) % 360.0 - 180.0)
        done = []
        if off > 10.0:
            o.rotation["y"] = float(yaw % 360.0)
            done.append(f"turned {off:.0f} deg to lie along the bank")
        land = on_land(pos, float(o.rotation["y"]))
        if land > 0.2:
            # The hull's centre at the near water a half-beam off the shore.
            band = ok & (bank <= want_d + 1.5)
            if not band.any():
                band = ok
            d2, (br, bc) = ndimage.distance_transform_edt(~band, return_indices=True)
            nr, nc = int(br[r0, c0]), int(bc[r0, c0])
            new = np.array([flow.origin[0] + (nc + 0.5) * flow.cell_m, flow.origin[1] + (nr + 0.5) * flow.cell_m])
            moved = float(np.linalg.norm(new - pos))
            if moved <= L and on_land(new, float(o.rotation["y"])) < land:
                o.position["x"], o.position["z"] = float(new[0]), float(new[1])
                if water_y is not None:
                    draft = float(np.clip(0.12 * ext[1], 0.2, 1.0))
                    o.position["y"] = float(water_y - draft - lo[1])
                done.append(f"moved {moved:.1f} m off the bank onto the water ({land:.0%} of the hull was on land)")
        return ", ".join(done)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object("scene_motion") is not None
