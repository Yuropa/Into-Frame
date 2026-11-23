"""Build a SceneGraph from a cached run (docs/agentic-scene-graph-plan.md §2).

`export_graph(run_dir)` reads the run the way scripts/replay-stage.py does: the context is
loaded with PipelineContext.load against config.yaml's stage list and, with no stage
pushed, every lookup searches all stages in reverse so the latest writer wins. Only the
values the graph touches are decoded (the context is lazy), nothing here imports torch,
and nothing needs the GPU. Every missing key is a warning in graph.meta, never an error.

Node ids are the ones server/agent/knobs.yaml edits: world, style, camera, sun,
atmosphere, water, ground, treeline, far_field, grass, pop_<class>, surface_<label>,
landmark_<slug>, thing_<i>, asset_<key>.
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml

from pipeline.scene_graph.graph import SceneGraph, Node

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "server" / "config.yaml"

# The plan's evidence words -> the graph's evidence classes (+ confidence).
PLAN_EVIDENCE = {"visible": ("plan-visible", 1.0), "inferred": ("plan-inferred", 1.0),
                 "invented": ("default", 0.3)}

# Classes whose instances are a population (knobs.yaml pop_<class>), plus any class
# with >= POPULATION_MIN instances.
POPULATION_CLASSES = {"flower", "grass_tuft", "tree", "bush", "plant", "rock", "shrub", "reed"}
POPULATION_MIN = 20
VEGETATION_CLASSES = {"flower", "grass_tuft", "tree", "bush", "plant", "shrub", "reed"}   # stand on the meadow
WATER_CLASSES = {"boat", "ship", "buoy", "swan", "duck", "kayak", "canoe", "sailboat", "yacht"}  # float

# Detector classes that name a region of the photograph, not a thing standing in it.
# They are neither populations nor things, however many crops carry them.
REGION_CLASSES = {"sky", "cloud", "clouds", "mountain", "mountains", "hill", "water", "lake", "ocean",
                  "sea", "river", "sand", "beach", "ground", "grass", "road", "trail", "path", "snow",
                  "field", "meadow", "forest", "environment", "indeterminate", "cliff", "wall",
                  "building_facade", "terrain", "glacier", "ice", "fog", "haze", "none", ""}

# RegionType ids (pipeline/panorama_segmentation/panorama_region_result.py), repeated here so
# the exporter does not import the segmentation package.
REGION_TYPES = {0: "sky", 1: "water", 2: "terrain", 3: "ground", 4: "vegetation", 5: "built",
                6: "other", 7: "road", 8: "trail", 9: "forest", 10: "snow"}
SURFACE_SKIP = {"sky", "water", "other"}
NEAR_M = 60.0
MAX_THINGS = 60
MIN_WATER_M2 = 1.0      # fewer water cells than this on the region map is noise, not a body of water

# Landmark structural type by the words in its category / class / name (first match wins).
LINEAR_WORDS = {"trail", "road", "river", "path", "street", "railway", "canal", "quay", "promenade",
                "boulevard", "avenue", "track", "footpath", "shoreline", "coastline", "ridge"}
SURFACE_WORDS = {"snowfield", "glacier", "icefall", "meadow", "snow", "field", "lawn", "beach", "lake",
                 "pond", "plateau", "flat", "bank", "reef", "shelf", "platform"}

# World / sun / ground / backdrop / grass config knobs: (stage, config key) -> (attr, unit).
WORLD_KNOBS = [("Scene Generation", "plan_population_near_m", "population_near_m", "m"),
               ("Scene Generation", "plan_population_density_m2.dense", "population_density_dense_m2", "per m2"),
               ("Scene Generation", "plan_population_density_m2.patchy", "population_density_patchy_m2", "per m2"),
               ("Scene Generation", "plan_size_tolerance", "size_tolerance", ""),
               ("Scene Generation", "species_mesh_near_distance_m", "species_mesh_near_distance_m", "m"),
               ("Scene Generation", "plan_band_rows", "band_rows", ""),
               ("Scene Generation", "overlap_min_separation", "overlap_min_separation", ""),
               ("Species Assets", "variants", "species_variants", ""),
               ("Panorama Asset Generation", "billboard_distance_m", "billboard_distance_m", "m")]
SUN_KNOBS = [("Exposure Fit", "scale_max", "exposure_scale_max", ""),
             ("Exposure Fit", "scale_min", "exposure_scale_min", ""),
             ("Exposure Fit", "near_m", "exposure_near_m", "m"),
             ("Panorama Lighting", "plan_sun_min_elevation_deg", "plan_min_elevation_deg", "deg"),
             ("Panorama Lighting", "plan_sun_max_intensity", "plan_max_intensity", "")]
GROUND_KNOBS = [("Height Map", "near_drop_floor_m", "near_drop_floor_m", "m"),
                ("Height Map", "smooth_sigma", "smooth_sigma", "")]
TREELINE_KNOBS = [("Treeline Backdrop", k, k, "m") for k in
                  ("min_distance_m", "max_distance_m", "min_distance_no_form_m", "min_height_m")]
FAR_FIELD_KNOBS = [("Far Field Backdrop", "far_m", "far_m", "m")]
GRASS_KNOBS = [("Grass Cover", "tufts_per_m2", "tufts_per_m2", "per m2"),
               ("Grass Cover", "max_radius_m", "max_radius_m", "m"),
               ("Grass Cover", "density_floor_fraction", "density_floor_fraction", "")]
# Distribution Synthesis per-class maps -> pop_<class> attrs.
POP_CLASS_KNOBS = [("class_density_scale", "density_scale", ""),
                   ("class_max_paint_radius_m", "max_paint_radius_m", "m"),
                   ("class_falloff_exponent", "falloff_exponent", ""),
                   ("class_min_full_density_radius_m", "min_full_density_radius_m", "m")]
BLOOM_KNOBS = [("bloom_max_boost", "bloom_max_boost", ""), ("bloom_max_density_m2", "bloom_max_density_m2", "per m2")]
FIT_KNOBS = [("enabled", "fit_enabled", ""), ("density_multiplier", "fit_density_multiplier", ""),
             ("cluster_radius_m", "fit_cluster_radius_m", "m"), ("cluster_size", "fit_cluster_size", ""),
             ("max_distance_m", "fit_max_distance_m", "m"), ("seed", "fit_seed", "")]


# ── helpers ───────────────────────────────────────────────────────────────────

def slug(text: str, limit: int = 32) -> str:
    """Lower-case words joined by _, accents stripped, at most `limit` characters."""
    t = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode()
    words = re.findall(r"[a-z0-9]+", t.lower())
    s = "_".join(words) or "unnamed"
    if len(s) > limit:
        s = s[:limit].rstrip("_")
    return s


def _num(v: Any) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _round(v: Any, nd: int = 3) -> Any:
    if isinstance(v, (float, np.floating)):
        return round(float(v), nd)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (list, tuple)):
        return [_round(x, nd) for x in v]
    if isinstance(v, dict):
        return {str(k): _round(x, nd) for k, x in v.items()}
    return v


def _words(*texts: Any) -> set[str]:
    out: set[str] = set()
    for t in texts:
        out.update(re.findall(r"[a-z]+", str(t or "").lower()))
    return out


def _dotted(d: Any, path: str) -> Any:
    for part in path.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


class _Run:
    """The cached run: config, context (lazy), stage order, and a warning sink."""

    def __init__(self, run_dir: Path, config_path: Path):
        self.run_dir = Path(run_dir).resolve()
        self.config_path = Path(config_path)
        self.warnings: list[str] = []
        cfg = yaml.safe_load(self.config_path.read_text()) or {}
        self.stage_entries: list[dict] = [e for e in (cfg.get("stages") or []) if isinstance(e, dict) and e.get("name")]
        self.stage_by_name = {e["name"]: e for e in self.stage_entries}
        self.all_names = [e["name"] for e in self.stage_entries]
        self.enabled_names = [e["name"] for e in self.stage_entries if e.get("enabled", True) is not False]
        order_file = self.run_dir / "_stage_order.json"
        try:
            saved = json.loads(order_file.read_text())
            self.saved_order = list(saved.get("stages") if isinstance(saved, dict) else saved)
        except Exception:
            self.saved_order = sorted(d.name for d in self.run_dir.iterdir() if d.is_dir())
            self.warn("no _stage_order.json; using directory names")
        from pipeline.pipeline_context import PipelineContext  # noqa: WPS433 (server on sys.path)
        self.context = PipelineContext()
        self.context.load(self.run_dir, self.enabled_names, self.all_names)

    def warn(self, text: str) -> None:
        if text not in self.warnings:
            self.warnings.append(text)

    # context values, each failure a warning
    def obj(self, key: str, warn: bool = True) -> Any:
        try:
            v = self.context.object(key)
        except Exception as e:  # a corrupt file is a warning, not a crash
            self.warn(f"{key}: {type(e).__name__}: {e}")
            return None
        if v is None and warn:
            self.warn(f"missing context value {key!r}")
        return v

    def array(self, key: str) -> Optional[np.ndarray]:
        try:
            d = self.context.depth(key)
        except Exception as e:
            self.warn(f"{key}: {type(e).__name__}: {e}")
            return None
        if d is None:
            self.warn(f"missing context array {key!r}")
            return None
        return np.asarray(d.depth)

    def typed(self, getter: str, key: str) -> Any:
        """A typed getter (panorama_regions, object_distribution) with a dict fallback."""
        try:
            v = getattr(self.context, getter)(key)
            if v is not None:
                return v
        except Exception as e:
            self.warn(f"{key}: {type(e).__name__}: {e}")
        for stage in reversed(self.saved_order):
            p = self.run_dir / stage / f"{key}.json"
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except Exception as e:
                    self.warn(f"{p.name}: {e}")
        self.warn(f"missing context value {key!r}")
        return None

    def stage_config(self, stage: str, key: str) -> Any:
        e = self.stage_by_name.get(stage)
        if e is None:
            return None
        return _dotted(e, key)

    def latest_file(self, name: str) -> Optional[Path]:
        """The newest stage directory holding `name` (latest in the saved order)."""
        found = None
        for stage in self.saved_order:
            p = self.run_dir / stage / name
            if p.exists():
                found = p
        return found

    def json_file(self, name: str) -> Any:
        """The newest stage's copy of a JSON file, decoded; None (and a warning) when absent."""
        p = self.latest_file(name)
        if p is None:
            self.warn(f"no {name} in any stage")
            return None
        try:
            return json.loads(p.read_text())
        except Exception as e:
            self.warn(f"{p}: {e}")
            return None

    def scene_json(self) -> dict:
        p = self.latest_file("scene.json")
        if p is None:
            self.warn("no scene.json in any stage")
            return {}
        try:
            return json.loads(p.read_text())
        except Exception as e:
            self.warn(f"{p}: {e}")
            return {}

    def override_files(self) -> list[tuple[str, list]]:
        out = []
        for stage in self.saved_order:
            d = self.run_dir / stage
            if not d.is_dir():
                continue
            for p in sorted(d.glob("plan_overrides_*.json")):
                try:
                    rows = json.loads(p.read_text())
                except Exception as e:
                    self.warn(f"{p.name}: {e}")
                    continue
                if isinstance(rows, list):
                    out.append((stage, rows))
        return out


# ── the exporter ──────────────────────────────────────────────────────────────

class _Exporter:
    def __init__(self, run: _Run):
        self.run = run
        self.g = SceneGraph(run_id=run.run_dir.name)
        self.plan_file = run.obj("scene_plan", warn=False) or {}
        if not self.plan_file:
            run.warn("no Scene Plan; plan-evidence attributes are absent")
        self.plan: dict = self.plan_file.get("plan") or {}
        self.g.photo_sha256 = str(self.plan_file.get("photo_sha256") or "")
        self.patched = self._agent_patched_fields()
        self.scene = run.scene_json()
        self.pop_classes: set[str] = set()
        self.region_map: Optional[np.ndarray] = None
        self.cell_m: Optional[float] = None
        self.far_field_start_m: Optional[float] = None

    # evidence from a plan block
    def plan_ev(self, block: Any, field: str = "") -> tuple[str, float]:
        if field and field in self.patched:
            return "fitted", 1.0
        word = block.get("evidence") if isinstance(block, dict) else None
        return PLAN_EVIDENCE.get(word, ("plan-inferred", 1.0))

    def _agent_patched_fields(self) -> set[str]:
        out: set[str] = set()
        for p in self.plan_file.get("agent_patches") or []:
            if isinstance(p, dict):
                for k in ("field", "path", "attr"):
                    if p.get(k):
                        out.add(str(p[k]))
            elif isinstance(p, str):
                out.add(p)
        return out

    def set_plan(self, node: Node, attr: str, block: dict, key: str, unit: str = "", path: str = "") -> None:
        """Copy block[key] onto the node with the block's plan evidence (skips None)."""
        if not isinstance(block, dict) or block.get(key) is None:
            return
        ev, conf = self.plan_ev(block, path or f"{attr}")
        source = "agent" if ev == "fitted" and (path or attr) in self.patched else "Scene Plan"
        node.set(attr, _round(block[key]), ev, unit, conf, source)

    def set_config(self, node: Node, knobs: list[tuple[str, str, str, str]]) -> None:
        for stage, key, attr, unit in knobs:
            v = self.run.stage_config(stage, key)
            if v is None:
                self.run.warn(f"config {stage}.{key} absent")
                continue
            node.set(attr, _round(v), "default", unit, 1.0, "config")

    def build(self) -> SceneGraph:
        self.world()
        self.style()
        self.camera()
        self.sun()
        self.atmosphere()
        self.ground_and_surfaces()
        self.water()
        self.backdrops()
        self.grass()
        self.populations_and_things()
        self.assets()
        self.landmarks()
        self.overrides()
        self.g.meta.update({"stages": list(self.run.saved_order),
                            "exported": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "config": str(self.run.config_path),
                            "warnings": list(self.run.warnings)})
        return self.g

    # ── world and style ───────────────────────────────────────────────────────
    def world(self) -> None:
        n = self.g.add(Node("world", "World", "terrain", "world"))
        place = self.plan.get("place") or {}
        self.set_plan(n, "place_name", place, "name", path="place.name")
        self.set_plan(n, "place_kind", place, "kind", path="place.kind")
        if place.get("confidence") is not None:
            ev, _ = self.plan_ev(place)
            n.set("place_confidence", _round(place["confidence"]), ev, "", 1.0, "Scene Plan")
        if self.plan_file.get("caption"):
            n.set("caption", self.plan_file["caption"], "measured", "", 1.0, "Captioning")
        grid = self.run.obj("terrain_grid_size_m")
        if _num(grid) is not None:
            n.set("grid_size_m", _round(grid), "measured", "m", 1.0, "Panorama Depth Calibration")
        ffs = self.run.obj("far_field_start_m")
        if _num(ffs) is not None:
            self.far_field_start_m = float(ffs)
            n.set("far_field_start_m", _round(ffs), "measured", "m", 1.0, "Panorama Depth Calibration")
        if self.scene:
            n.set("skybox_rotation_deg", _round(self.scene.get("skyboxRotation", 0.0)), "measured", "deg", 1.0,
                  "Scene Generation")
            n.set("scene_object_count", len(self.scene.get("objects") or []), "measured", "", 1.0, "Scene Generation")
        count = self.run.obj("count")
        if count is not None:
            n.set("object_count", int(count), "measured", "", 1.0, "Scene Generation")
        behind = self.plan.get("behind") or {}
        if behind.get("text"):
            n.set("behind", behind["text"], "plan-inferred", "", 0.5, "Scene Plan")
        hz = self.plan.get("horizon") or {}
        if hz.get("skyline"):
            ev, conf = self.plan_ev(hz)
            n.set("skyline", hz["skyline"], ev, "", conf, "Scene Plan")
        self.set_config(n, WORLD_KNOBS)
        n.geometry = {"region_map": "region_map", "height_map": "height_map", "scene": "scene"}

    def style(self) -> None:
        n = self.g.add(Node("style", "Style", "photograph", "style"))
        n.set("medium", "photograph", "default", "", 1.0, "export")
        n.set("realism", "physical", "default", "", 1.0, "export")

    # ── camera ────────────────────────────────────────────────────────────────
    def camera(self) -> None:
        n = self.g.add(Node("camera", "Camera", "camera", "camera"))
        cam = self.plan.get("camera") or {}
        self.set_plan(n, "height_m", cam, "height_m", "m", "camera.height_m")
        self.set_plan(n, "pitch_deg", cam, "pitch_deg", "deg", "camera.pitch_deg")
        self.set_plan(n, "on_structure", cam, "on_structure", "", "camera.on_structure")
        est = self.run.obj("camera_height_estimate")
        if isinstance(est, dict):
            ev = "measured" if (est.get("evidence") == "measured" or est.get("source") in ("photo", "measured")) else "fitted"
            if _num(est.get("meters")) is not None:
                n.set("height_estimate_m", _round(est["meters"]), ev, "m", 1.0, "Panorama Depth Calibration")
            if est.get("source"):
                n.set("height_source", est["source"], ev, "", 1.0, "Panorama Depth Calibration")
            if _num(est.get("measured")) is not None:
                n.set("height_measured_m", _round(est["measured"]), "measured", "m", 1.0,
                      f"Panorama Depth Calibration ({est.get('measured_source', '?')})")
            for k in ("anchors", "plane_da3", "scale"):
                if _num(est.get(k)) is not None:
                    n.set(f"height_{k}", _round(est[k]), "measured", "", 1.0, "Panorama Depth Calibration")
        intr = self.run.json_file("intrinsics.json")   # stored as a CameraIntrinsics, not a plain object
        if isinstance(intr, dict):
            if _num(intr.get("fov")) is not None:
                n.set("fov_deg", _round(intr["fov"]), "measured", "deg", 1.0, "Depth Generation")
            if intr.get("width") and intr.get("height"):
                n.set("image_size", [int(intr["width"]), int(intr["height"])], "measured", "px", 1.0, "Depth Generation")
            for k in ("fx", "fy"):
                if _num(intr.get(k)) is not None:
                    n.set(k, _round(intr[k]), "measured", "px", 1.0, "Depth Generation")
        if self.scene:
            for key, attr in (("cameraHeight", "scene_camera_height_m"), ("eyeHeightMeters", "eye_height_m")):
                if _num(self.scene.get(key)) is not None:
                    n.set(attr, _round(self.scene[key]), "measured", "m", 1.0, "Scene Generation")

    # ── light and medium ──────────────────────────────────────────────────────
    @staticmethod
    def _az_el(direction: Any) -> tuple[Optional[float], Optional[float]]:
        try:
            x, y, z = (float(v) for v in direction)
        except Exception:
            return None, None
        r = math.sqrt(x * x + y * y + z * z) or 1.0
        return math.degrees(math.atan2(x, z)), math.degrees(math.asin(max(-1.0, min(1.0, y / r))))

    def sun(self) -> None:
        n = self.g.add(Node("sun", "Light", "sun", "sun"))
        sun = self.plan.get("sun") or {}
        for k, unit in (("elevation_deg", "deg"), ("azimuth_deg", "deg"), ("warmth", ""), ("sky", ""), ("time_of_day", "")):
            self.set_plan(n, k, sun, k, unit, f"sun.{k}")
        ms = self.run.latest_file("lighting.lighting/measured_sun.json")
        if ms is None:
            self.run.warn("no measured_sun.json under Panorama Lighting")
        else:
            try:
                m = json.loads(ms.read_text())
                m = m.get("sun", m) if isinstance(m, dict) else {}
            except Exception as e:
                self.run.warn(f"measured_sun.json: {e}")
                m = {}
            az, el = self._az_el(m.get("direction"))
            if _num(m.get("elevation_deg")) is not None:
                el = float(m["elevation_deg"])
            if el is not None:
                n.set("measured_elevation_deg", _round(el), "measured", "deg", 1.0, "Panorama Lighting")
            if az is not None:
                n.set("measured_azimuth_deg", _round(az), "measured", "deg", 1.0, "Panorama Lighting")
            for k in ("intensity", "direction_source", "color", "sun_share", "peak_ratio"):
                if m.get(k) is not None:
                    n.set(f"measured_{k}", _round(m[k]), "measured", "", 1.0, "Panorama Lighting")
        L = (self.scene or {}).get("lighting") or {}
        s = L.get("sun") or {}
        if not s:
            self.run.warn("scene lighting has no sun")
        if _num(s.get("elevation_deg")) is not None:
            n.set("final_elevation_deg", _round(s["elevation_deg"]), "measured", "deg", 1.0, "Panorama Lighting")
        az, el = self._az_el(s.get("direction"))
        if az is not None:
            n.set("final_azimuth_deg", _round(az), "measured", "deg", 1.0, "Panorama Lighting")
            n.set("final_direction", _round(s.get("direction")), "measured", "", 1.0, "Panorama Lighting")
        if s.get("direction_source"):
            n.set("direction_source", s["direction_source"], "measured", "", 1.0, "Panorama Lighting")
        for k in ("color", "intensity", "intensity_base", "solar_scale"):
            if s.get(k) is not None:
                n.set(k, _round(s[k]), "fitted" if k == "intensity" else "measured", "", 1.0, "Panorama Lighting")
        if s.get("fit_sun_scale") is not None:
            n.set("fit_sun_scale", _round(s["fit_sun_scale"]), "fitted", "", 1.0, "Exposure Fit")
        if L.get("ambient_scale") is not None:
            n.set("ambient_scale", _round(L["ambient_scale"]), "fitted", "", 1.0, "Exposure Fit")
        fit = self.run.obj("exposure_fit", warn=False)
        if isinstance(fit, dict) and not fit.get("skipped"):
            for k in ("sun_scale", "ambient_scale", "err_before", "err_after", "n_near", "near_m"):
                if _num(fit.get(k)) is not None:
                    attr = {"sun_scale": "exposure_sun_scale", "ambient_scale": "exposure_ambient_scale",
                            "near_m": "exposure_fit_near_m"}.get(k, k)
                    n.set(attr, _round(fit[k]), "fitted", "m" if k == "near_m" else "", 1.0, "Exposure Fit")
        elif isinstance(fit, dict):
            n.set("exposure_fit_skipped", True, "measured", "", 1.0, "Exposure Fit")
        elif isinstance(L.get("fit"), dict):
            for k in ("err_before", "err_after"):
                if _num(L["fit"].get(k)) is not None:
                    n.set(k, _round(L["fit"][k]), "fitted", "", 1.0, "Exposure Fit")
        else:
            self.run.warn("no exposure_fit record")
        self.set_config(n, SUN_KNOBS)
        self.g.relate("world", "lit_by", "sun", "measured", "export")

    def atmosphere(self) -> None:
        n = self.g.add(Node("atmosphere", "Medium", "air", "atmosphere"))
        atm = self.plan.get("atmosphere") or {}
        self.set_plan(n, "haze_near_m", atm, "haze_near_m", "m", "atmosphere.haze_near_m")
        self.set_plan(n, "haze_far_m", atm, "haze_far_m", "m", "atmosphere.haze_far_m")
        self.set_plan(n, "tint_rgb", atm, "tint_rgb", "", "atmosphere.tint_rgb")
        sa = (self.scene or {}).get("atmosphere") or {}
        if not sa:
            self.run.warn("scene has no atmosphere block")
        ev, conf = PLAN_EVIDENCE.get(sa.get("evidence"), ("fitted", 1.0))
        for k, attr, unit in (("haze_near_m", "scene_haze_near_m", "m"), ("haze_far_m", "scene_haze_far_m", "m"),
                              ("tint_rgb", "scene_tint_rgb", ""), ("haze_color", "haze_color", "")):
            if sa.get(k) is not None:
                n.set(attr, _round(sa[k]), "fitted" if k == "haze_color" else ev, unit, conf, "Scene Generation")
        if self.scene.get("ambientColor") is not None:
            n.set("ambient_color", _round(self.scene["ambientColor"]), "measured", "", 1.0, "Scene Generation")

    # ── ground, surfaces, water ───────────────────────────────────────────────
    def _region_stats(self) -> dict[str, dict]:
        """Per RegionType label: area_m2, mean_distance_m, near_fraction_60m, fraction."""
        rm = self.region_map
        if rm is None or self.cell_m is None:
            return {}
        res_r, res_c = rm.shape[:2]
        ids = np.rint(rm).astype(np.int32)
        axis_z = (np.arange(res_r) + 0.5) * self.cell_m - res_r * self.cell_m / 2.0
        axis_x = (np.arange(res_c) + 0.5) * self.cell_m - res_c * self.cell_m / 2.0
        dist = np.sqrt(axis_z[:, None] ** 2 + axis_x[None, :] ** 2)
        total = float(ids.size)
        out: dict[str, dict] = {}
        for tid, label in REGION_TYPES.items():
            mask = ids == tid
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            d = dist[mask]
            out[label] = {"cells": cnt, "area_m2": cnt * self.cell_m ** 2, "fraction": cnt / total,
                          "mean_distance_m": float(d.mean()), "min_distance_m": float(d.min()),
                          "near_fraction_60m": float((d <= NEAR_M).mean())}
        return out

    def ground_and_surfaces(self) -> None:
        g = self.g.add(Node("ground", "Surface", "ground", "ground"))
        params = self.run.obj("height_map_params")
        if isinstance(params, dict):
            for k, unit in (("grid_size_meters", "m"), ("grid_resolution", ""), ("ground_y_max", "m"),
                            ("camera_height_meters", "m")):
                if _num(params.get(k)) is not None:
                    g.set(k, _round(params[k]), "measured", unit, 1.0, "Height Map")
        terrain = self.plan.get("terrain") or {}
        self.set_plan(g, "terrain_kind", terrain, "kind", "", "terrain.kind")
        self.set_plan(g, "slope_pct", terrain, "slope_pct", "pct", "terrain.slope_pct")
        self.set_plan(g, "ground_description", terrain, "ground", "", "terrain.ground")
        tp = self.run.obj("terrain_plan")
        if isinstance(tp, dict):
            if tp.get("kind"):
                g.set("terrain_plan_kind", tp["kind"], "fitted", "", 1.0, "Region Map")
            if _num(tp.get("hfov_deg")) is not None:
                g.set("hfov_deg", _round(tp["hfov_deg"]), "measured", "deg", 1.0, "Region Map")
            if tp.get("notes"):
                g.set("terrain_plan_notes", list(tp["notes"])[:6], "fitted", "", 1.0, "Region Map")
        self.set_config(g, GROUND_KNOBS)
        g.geometry = {"height_map": "height_map", "region_map": "region_map"}
        g.affordances = ["walkable", "collides"]

        # the region map
        grid_m = _num(self.run.obj("terrain_grid_size_m", warn=False))
        if grid_m is None and isinstance(params, dict):
            grid_m = _num(params.get("grid_size_meters"))
        self.region_map = self.run.array("region_map")
        if self.region_map is not None and grid_m:
            self.cell_m = grid_m / float(self.region_map.shape[0])
            g.set("cell_m", _round(self.cell_m, 4), "measured", "m", 1.0, "Region Map")
        elif self.region_map is not None:
            self.run.warn("region map present but no grid size; surface areas skipped")
        hm = self.run.array("height_map")
        if hm is not None:
            finite = hm[np.isfinite(hm)]
            if finite.size:
                g.set("height_min_m", _round(float(finite.min())), "measured", "m", 1.0, "Terrain Noise Refinement")
                g.set("height_max_m", _round(float(finite.max())), "measured", "m", 1.0, "Terrain Noise Refinement")
                c = hm.shape[0] // 2
                centre = hm[c - 2:c + 3, c - 2:c + 3]
                cf = centre[np.isfinite(centre)]
                if cf.size:
                    g.set("height_at_camera_m", _round(float(cf.mean())), "measured", "m", 1.0, "Terrain Noise Refinement")
        stats = self._region_stats()
        self.region_stats = stats
        pr = self.run.typed("panorama_regions", "panorama_regions")
        fractions = getattr(pr, "type_fractions", None) if pr is not None and not isinstance(pr, dict) else (
            pr.get("type_fractions") if isinstance(pr, dict) else None)
        if isinstance(fractions, dict):
            g.set("panorama_type_fractions", _round(fractions, 4), "measured", "", 1.0, "Panorama Regions")
        for label, st in stats.items():
            if label in SURFACE_SKIP:
                continue
            n = self.g.add(Node(f"surface_{label}", "Surface", label, label))
            n.set("area_m2", _round(st["area_m2"], 1), "measured", "m2", 1.0, "Region Map")
            n.set("fraction", _round(st["fraction"], 4), "measured", "", 1.0, "Region Map")
            n.set("mean_distance_m", _round(st["mean_distance_m"], 1), "measured", "m", 1.0, "Region Map")
            n.set("min_distance_m", _round(st["min_distance_m"], 1), "measured", "m", 1.0, "Region Map")
            n.set("near_fraction_60m", _round(st["near_fraction_60m"], 4), "measured", "", 1.0, "Region Map")
            if isinstance(fractions, dict) and label in fractions:
                n.set("panorama_fraction", _round(fractions[label], 4), "measured", "", 1.0, "Panorama Regions")
            n.geometry = {"region_map": "region_map", "region_type": label}
            n.affordances = ["walkable", "collides"] if label not in ("forest", "built") else ["collides"]
            self.g.relate(n.id, "in", "ground", "measured", "Region Map")
        if not stats:
            self.run.warn("no region-map statistics; surface_* nodes absent")

    def water(self) -> None:
        plan_w = self.plan.get("water") or {}
        kind = plan_w.get("kind")
        st = getattr(self, "region_stats", {}).get("water")
        if st is not None and st["area_m2"] < MIN_WATER_M2:
            if not kind or kind == "none":
                self.run.warn(f"region map has {st['cells']} stray water cells ({st['area_m2']:.2f} m2); no water node")
                st = None
        if (not kind or kind == "none") and not st:
            return
        n = self.g.add(Node("water", "Surface", kind if kind and kind != "none" else "water", "water"))
        for k, unit in (("kind", ""), ("level_below_camera_m", "m"), ("width_m", "m"), ("bank_left", ""),
                        ("bank_right", ""), ("flow_dir_deg", "deg"), ("continues_upstream", ""),
                        ("continues_downstream", "")):
            self.set_plan(n, k, plan_w, k, unit, f"water.{k}")
        if st:
            n.set("area_m2", _round(st["area_m2"], 1), "measured", "m2", 1.0, "Region Map")
            n.set("fraction", _round(st["fraction"], 4), "measured", "", 1.0, "Region Map")
            n.set("mean_distance_m", _round(st["mean_distance_m"], 1), "measured", "m", 1.0, "Region Map")
            n.set("min_distance_m", _round(st["min_distance_m"], 1), "measured", "m", 1.0, "Region Map")
            n.set("near_fraction_60m", _round(st["near_fraction_60m"], 4), "measured", "", 1.0, "Region Map")
        tp = self.run.obj("terrain_plan", warn=False)
        if isinstance(tp, dict):
            if tp.get("kind"):
                n.set("terrain_plan_kind", tp["kind"], "fitted", "", 1.0, "Region Map")
            if _num(tp.get("water_y")) is not None:
                n.set("water_y", _round(tp["water_y"]), "fitted", "m", 1.0, "Region Map")
            if tp.get("width_prior") is not None:
                n.set("width_prior", _round(tp["width_prior"]), "fitted", "m", 1.0, "Region Map")
            if tp.get("channel") is not None:
                n.set("channel", _round(tp["channel"]), "fitted", "", 1.0, "Region Map")
        n.geometry = {"region_map": "region_map", "region_type": "water"}
        n.affordances = ["floats", "animated"]
        self.g.relate("water", "in", "ground", "measured", "Region Map")

    # ── backdrops and grass ───────────────────────────────────────────────────
    def backdrops(self) -> None:
        t = self.g.add(Node("treeline", "Backdrop", "treeline", "treeline"))
        ts = self.run.obj("treeline_backdrop_summary")
        if isinstance(ts, dict):
            for k, v in ts.items():
                if v is None:
                    continue
                unit = "m" if k.endswith("_m") else ""
                t.set(k, _round(v), "measured", unit, 1.0, "Treeline Backdrop")
            t.set("present", bool(ts.get("bins")), "measured", "", 1.0, "Treeline Backdrop")
        self.set_config(t, TREELINE_KNOBS)
        t.geometry = {"mesh": "treeline_backdrop"}
        f = self.g.add(Node("far_field", "Backdrop", "far_field", "far field"))
        fs = self.run.obj("far_field_backdrop_summary")
        if isinstance(fs, dict):
            for k, v in fs.items():
                if v is None:
                    continue
                unit = "m" if (k.endswith("_m") or k.startswith("metric_depth") or k.startswith("radius")) else ""
                f.set(k, _round(v), "measured", unit, 1.0, "Far Field Backdrop")
        self.set_config(f, FAR_FIELD_KNOBS)
        f.geometry = {"mesh": "far_field_backdrop"}
        self.g.relate("treeline", "behind", "ground", "measured", "export")
        self.g.relate("far_field", "behind", "treeline", "measured", "export")

    def grass(self) -> None:
        n = self.g.add(Node("grass", "Population", "grass", "grass"))
        gf = self.run.obj("grass_field", warn=False) or (self.scene or {}).get("grassField")
        if isinstance(gf, dict):
            for k, v in gf.items():
                if isinstance(v, (int, float, bool)):
                    unit = "m" if "Meters" in k else ("per m2" if "SquareMeter" in k else "")
                    n.set(k, _round(v), "measured", unit, 1.0, "Grass Cover")
            n.set("present", True, "measured", "", 1.0, "Grass Cover")
        else:
            self.run.warn("no grass_field (no grass cover in this run)")
            n.set("present", False, "measured", "", 1.0, "Grass Cover")
        gs = self.run.obj("grass_cover_summary", warn=False)
        if isinstance(gs, dict):
            for k, v in gs.items():
                if isinstance(v, (int, float, bool)) and k not in n.attrs:
                    n.set(k, _round(v), "measured", "m" if k.endswith("_m") else "", 1.0, "Grass Cover")
        for sp in self.plan.get("species") or []:
            if sp.get("class") in ("grass", "grass_tuft", "sedge"):
                ev, conf = self.plan_ev(sp)
                n.set("plan_species", sp.get("name"), ev, "", conf, "Scene Plan")
                if sp.get("density"):
                    n.set("plan_density", sp["density"], ev, "", conf, "Scene Plan")
                break
        self.set_config(n, GRASS_KNOBS)
        n.geometry = {"density": "grass_field_density", "colour": "grass_field_colour"}
        n.affordances = ["animated"]
        on = "surface_vegetation" if "surface_vegetation" in self.g.nodes else "ground"
        self.g.relate("grass", "on", on, "measured", "Grass Cover")

    # ── populations, things ───────────────────────────────────────────────────
    def _instances(self) -> list[tuple[int, dict]]:
        count = self.run.obj("count")
        try:
            n = int(count)
        except (TypeError, ValueError):
            self.run.warn("no instance count")
            return []
        out = []
        for i in range(n):
            m = self.run.obj(f"metadata_{i}", warn=False)
            if isinstance(m, dict):
                out.append((i, m))
        return out

    def populations_and_things(self) -> None:
        inst = self._instances()
        by_class: dict[str, list[tuple[int, dict]]] = {}
        for i, m in inst:
            by_class.setdefault(str(m.get("class") or "none").lower(), []).append((i, m))
        region_counts = {c: len(v) for c, v in by_class.items() if c in REGION_CLASSES}
        # A class the pipeline treats as a population stays one however few instances the
        # current draw left (the boat fit on Paris drew 3 and pop_boat vanished from the
        # re-export, so the loop could not annotate its own edit): a fitted model, a
        # Population Fit class or override, or a Distribution Synthesis class override.
        pm_early = self.run.obj("population_models", warn=False)
        fit_early = self.run.stage_by_name.get("Population Fit") or {}
        ds_early = self.run.stage_by_name.get("Distribution Synthesis") or {}
        pipeline_pops = set(pm_early.keys() if isinstance(pm_early, dict) else [])
        pipeline_pops |= set(fit_early.get("classes") or []) | set((fit_early.get("class_overrides") or {}).keys())
        pipeline_pops |= set((ds_early.get("class_overrides") or {}).keys())
        self.pop_classes = {c for c, v in by_class.items()
                            if c not in REGION_CLASSES and (c in POPULATION_CLASSES or c in pipeline_pops
                                                            or len(v) >= POPULATION_MIN)}
        self.g.meta["instance_classes"] = {c: len(v) for c, v in sorted(by_class.items(), key=lambda kv: -len(kv[1]))}
        self.g.meta["region_classes_skipped"] = region_counts

        od = self.run.typed("object_distribution", "object_distribution")
        dists = getattr(od, "distributions", None) if od is not None and not isinstance(od, dict) else (
            od.get("distributions") if isinstance(od, dict) else None) or {}
        pm = self.run.obj("population_models", warn=False)
        if not isinstance(pm, dict):
            pm = {}
        ds = self.run.stage_by_name.get("Distribution Synthesis") or {}
        fit_cfg = self.run.stage_by_name.get("Population Fit")
        species = self.plan.get("species") or []
        objects = self.plan.get("objects") or []

        for cls in sorted(self.pop_classes):
            rows = by_class[cls]
            n = self.g.add(Node(f"pop_{cls}", "Population", cls, cls))
            real = [m for _, m in rows if not m.get("synthetic")]
            n.set("count", len(rows), "measured", "", 1.0, "Scene Generation")
            n.set("count_real", len(real), "measured", "", 1.0, "Object Labeling")
            n.set("count_synthetic", len(rows) - len(real), "generated", "", 1.0, "Distribution Synthesis")
            heights = [h for h in (_num(m.get("world_height")) for _, m in rows) if h is not None]
            if heights:
                n.set("mean_height_m", _round(float(np.mean(heights))), "measured", "m", 1.0, "Scene Generation")
                n.set("max_height_m", _round(float(np.max(heights))), "measured", "m", 1.0, "Scene Generation")
            dists_m = []
            for _, m in rows:
                p = m.get("world_position")
                if isinstance(p, (list, tuple)) and len(p) == 3 and _num(p[0]) is not None and _num(p[2]) is not None:
                    dists_m.append(math.hypot(float(p[0]), float(p[2])))
            if dists_m:
                n.set("median_distance_m", _round(float(np.median(dists_m)), 1), "measured", "m", 1.0, "Scene Generation")
                n.set("max_distance_m", _round(float(np.max(dists_m)), 1), "measured", "m", 1.0, "Scene Generation")
                n.set("count_placed", len(dists_m), "measured", "", 1.0, "Scene Generation")
            srcs: dict[str, int] = {}
            for _, m in rows:
                s = str(m.get("source") or "detector")
                srcs[s] = srcs.get(s, 0) + 1
            n.set("sources", srcs, "measured", "", 1.0, "Scene Generation")
            buckets: dict[str, int] = {}
            for _, m in rows:
                if m.get("bucket") is not None:
                    buckets[str(m["bucket"])] = buckets.get(str(m["bucket"]), 0) + 1
            if buckets:
                n.set("buckets", buckets, "measured", "", 1.0, "Object Category Clustering")
            conf = [c for c in (_num(m.get("confidence")) for m in real) if c is not None]
            if conf:
                n.set("mean_confidence", _round(float(np.mean(conf))), "measured", "", 1.0, "Object Labeling")
            # exemplar counts per region (Object Distribution)
            ex: dict[str, int] = {}
            for region, by_type in (dists or {}).items():
                d = (by_type or {}).get(cls)
                if d is None:
                    continue
                np_ = getattr(d, "n_points", None) if not isinstance(d, dict) else d.get("n_points")
                if np_ is not None:
                    ex[region] = int(np_)
            if ex:
                n.set("exemplars_per_region", ex, "measured", "", 1.0, "Object Distribution")
            # the plan's species and objects of this class
            names, densities, forms, wheres = [], [], [], []
            ev_best: Optional[tuple[str, float]] = None
            for sp in species:
                if str(sp.get("class", "")).lower() == cls:
                    names.append(sp.get("name"))
                    densities.append(sp.get("density"))
                    forms.append(sp.get("form"))
                    wheres.append(sp.get("where"))
                    ev_best = ev_best or self.plan_ev(sp)
            if names:
                ev, c = ev_best or ("plan-inferred", 1.0)
                n.set("plan_species", names, ev, "", c, "Scene Plan")
                n.set("plan_density", densities, ev, "", c, "Scene Plan")
                if any(forms):
                    n.set("plan_form", forms, ev, "", c, "Scene Plan")
                if any(wheres):
                    n.set("plan_where", wheres, ev, "", c, "Scene Plan")
                sizes = [s for s in (_num(sp.get("size_m")) for sp in species if str(sp.get("class", "")).lower() == cls)
                         if s is not None]
                if sizes:
                    n.set("plan_size_m", _round(sizes), ev, "m", c, "Scene Plan")
            pobjs = [o for o in objects if cls in _words(o.get("class"), o.get("category"), o.get("phrase"))]
            if pobjs:
                ev, c = self.plan_ev(pobjs[0])
                n.set("plan_count", sum(int(o.get("count") or 0) for o in pobjs), ev, "", c, "Scene Plan")
                n.set("plan_phrases", [o.get("phrase") for o in pobjs], ev, "", c, "Scene Plan")
            # Population Fit's model summary
            model = pm.get(cls)
            if isinstance(model, dict):
                summary = model.get("summary") if isinstance(model.get("summary"), dict) else model
                for k, v in summary.items():
                    if isinstance(v, (int, float, bool, str, list)):
                        n.set(f"model_{k}", _round(v), "fitted", "m" if k.endswith("_m") else "", 1.0, "Population Fit")
            # config knobs
            for key, attr, unit in POP_CLASS_KNOBS:
                table = ds.get(key)
                if isinstance(table, dict) and cls in table:
                    n.set(attr, _round(table[cls]), "default", unit, 1.0, "config")
            if cls == "flower":
                for key, attr, unit in BLOOM_KNOBS:
                    if ds.get(key) is not None:
                        n.set(attr, _round(ds[key]), "default", unit, 1.0, "config")
            over = (_dotted(fit_cfg, f"class_overrides.{cls}") if fit_cfg else None) or {}
            for key, attr, unit in FIT_KNOBS:
                if key in over:
                    n.set(attr, _round(over[key]), "default", unit, 1.0, "config")
                elif key == "enabled":
                    n.set(attr, False, "default", "", 1.0, "config")
            n.geometry = {"metadata": f"metadata_*.class=={cls}"}
            n.affordances = ["collides"] + (["animated"] if cls in ("flower", "tree", "bush", "plant", "reed", "grass_tuft") else [])
            # relations
            on = "water" if cls in WATER_CLASSES and "water" in self.g.nodes else (
                "surface_vegetation" if "surface_vegetation" in self.g.nodes and cls in VEGETATION_CLASSES else "ground")
            self.g.relate(n.id, "on", on, "measured", "export")
            if cls == "tree" and "treeline" in self.g.nodes:
                self.g.relate(n.id, "bounded_by", "treeline", "measured", "export")
        if fit_cfg is None:
            self.run.warn("config has no 'Population Fit' stage; fit_* knobs default (fit_enabled=false)")
        if not pm:
            self.run.warn("no population_models (Population Fit has not run)")

        # things: real, placed, singular
        things = []
        for i, m in inst:
            cls = str(m.get("class") or "none").lower()
            if cls in self.pop_classes or cls in REGION_CLASSES or m.get("synthetic"):
                continue
            p = m.get("world_position")
            if not (isinstance(p, (list, tuple)) and len(p) == 3 and all(_num(v) is not None for v in p)):
                continue
            things.append((i, m, [float(v) for v in p]))
        self.g.meta["things_total"] = len(things)
        self.g.find("world").set("thing_count", len(things), "measured", "", 1.0, "Scene Generation")
        things.sort(key=lambda t: -(_num(t[1].get("world_height")) or 0.0))
        for i, m, p in things[:MAX_THINGS]:
            cls = str(m.get("class")).lower()
            n = self.g.add(Node(f"thing_{i}", "Thing", cls, str(m.get("name") or cls)))
            n.set("position", _round(p), "measured", "m", 1.0, "Scene Generation")
            n.set("distance_m", _round(math.hypot(p[0], p[2]), 1), "measured", "m", 1.0, "Scene Generation")
            if _num(m.get("world_height")) is not None:
                n.set("height_m", _round(m["world_height"]), "measured", "m", 1.0, "Scene Generation")
            if _num(m.get("world_width")) is not None:
                n.set("width_m", _round(m["world_width"]), "measured", "m", 1.0, "Scene Generation")
            if _num(m.get("confidence")) is not None:
                n.set("confidence", _round(m["confidence"]), "measured", "", 1.0, "Object Labeling")
            if m.get("source"):
                n.set("source", m["source"], "measured", "", 1.0, "Object Labeling")
            if m.get("bucket") is not None:
                n.set("bucket", m["bucket"], "measured", "", 1.0, "Object Category Clustering")
            if m.get("in_captured_fov") is not None:
                n.set("in_captured_fov", bool(m["in_captured_fov"]), "measured", "", 1.0, "Object Labeling")
            n.geometry = {"metadata": f"metadata_{i}", "mesh": m.get("mesh") or f"category_mesh_{cls}_{m.get('bucket', 0)}"}
            n.affordances = ["collides"]
            on = "water" if cls in WATER_CLASSES and "water" in self.g.nodes else "ground"
            self.g.relate(n.id, "on", on, "measured", "export")
        if len(things) > MAX_THINGS:
            self.run.warn(f"{len(things)} things; only the {MAX_THINGS} tallest are nodes")

    # ── assets ────────────────────────────────────────────────────────────────
    def assets(self) -> None:
        sm = self.run.obj("species_meshes", warn=False)
        if not isinstance(sm, dict):
            self.run.warn("no species_meshes (Species Assets has not run)")
            return
        if not sm:
            self.run.warn("species_meshes is empty (no species matched)")
        for key, rec in sm.items():
            if not isinstance(rec, dict):
                continue
            n = self.g.add(Node(f"asset_{key}", "Asset", str(rec.get("form") or "mesh"), str(rec.get("species") or key)))
            for k, unit in (("height_m", "m"), ("faces", ""), ("delta_e", ""), ("variants", ""), ("members", "")):
                if _num(rec.get(k)) is not None:
                    n.set(k, _round(rec[k]), "fitted" if k in ("height_m", "delta_e") else "measured", unit, 1.0, "Species Assets")
            for k in ("phrase", "match", "card"):
                if rec.get(k):
                    n.set(k, rec[k], "fitted", "", 1.0, "Species Assets")
            if isinstance(rec.get("palette"), dict):
                n.set("palette", _round(rec["palette"]), "measured", "rgb", 1.0, "Species Assets")
            n.geometry = {"mesh": key, "files": list(rec.get("files") or [])}
            m = re.match(r"category_mesh_([a-z_]+?)_(\d+)$", key)
            if m:
                cls = m.group(1)
                n.set("bucket", int(m.group(2)), "measured", "", 1.0, "Species Assets")
                pop = f"pop_{cls}"
                if pop in self.g.nodes:
                    self.g.relate(pop, "made_of", n.id, "measured", "Species Assets")
                else:
                    self.run.warn(f"asset {key}: no population node {pop}")

    # ── landmarks ─────────────────────────────────────────────────────────────
    @staticmethod
    def landmark_type(lm: dict) -> str:
        words = _words(lm.get("category"), lm.get("class"), lm.get("name"))
        if words & LINEAR_WORDS:
            return "Linear"
        if words & SURFACE_WORDS:
            return "Surface"
        return "Mass"

    def landmarks(self) -> None:
        placeholders = self.run.obj("structure_placeholders", warn=False)
        placeholders = placeholders if isinstance(placeholders, list) else []
        by_name = {str(p.get("name", "")).lower(): p for p in placeholders if isinstance(p, dict)}
        used: set[str] = set()
        for lm in self.plan.get("landmarks") or []:
            if not isinstance(lm, dict):
                continue
            s = slug(lm.get("name") or lm.get("class") or "landmark")
            nid = f"landmark_{s}"
            k = 2
            while nid in self.g.nodes:
                nid = f"landmark_{s[:29]}_{k}"
                k += 1
            typ = self.landmark_type(lm)
            n = self.g.add(Node(nid, typ, str(lm.get("class") or lm.get("category") or "landmark"), str(lm.get("name") or nid)))
            for key, unit in (("bearing_deg", "deg"), ("distance_m", "m"), ("height_m", "m"), ("category", ""),
                              ("phrase", ""), ("side", "")):
                self.set_plan(n, key, lm, key, unit, f"landmarks.{s}.{key}")
            ph = by_name.get(str(lm.get("name", "")).lower())
            if ph is not None:
                used.add(str(ph.get("name", "")).lower())
                for key, unit in (("status", ""), ("mesh", ""), ("note", ""), ("colour", "rgb"), ("colour_source", "")):
                    if ph.get(key) is not None:
                        n.set(f"placeholder_{key}" if key in ("colour", "colour_source", "note") else key,
                              _round(ph[key]), "measured", unit, 1.0, "Structure Placeholders")
                for key in ("bearing_deg", "distance_m", "height_m"):
                    if _num(ph.get(key)) is not None and ph.get(key) != lm.get(key):
                        n.set(f"placed_{key}", _round(ph[key]), "fitted", "deg" if key.endswith("deg") else "m", 1.0,
                              "Structure Placeholders")
                if ph.get("mesh"):
                    n.geometry = {"mesh": ph["mesh"]}
            n.affordances = ["collides"] if typ == "Mass" else (["walkable"] if typ == "Linear" else [])
            d = _num(lm.get("distance_m"))
            if d is not None and self.far_field_start_m is not None and d > self.far_field_start_m:
                self.g.relate(nid, "behind", "world", "plan-inferred", "export")
            elif typ != "Mass" or d is not None:
                self.g.relate(nid, "on", "ground", "plan-inferred", "export")
        for ph in placeholders:
            if isinstance(ph, dict) and str(ph.get("name", "")).lower() not in used:
                self.run.warn(f"structure placeholder {ph.get('name')!r} matches no plan landmark")

    # ── plan overrides ────────────────────────────────────────────────────────
    def _override_owner(self, field: str) -> str:
        f = field.lower()
        words = _words(f)
        if "sun" in words or "light" in words:
            return "sun"
        if "camera" in words:
            return "camera"
        if words & {"river", "water", "sea", "lake", "shore", "coast"}:
            return "water" if "water" in self.g.nodes else "ground"
        if "far" in words and "field" in words:
            return "far_field"
        if "treeline" in words or "forest" in words and "band" in words:
            tail = f.split(":")[-1].strip()
            return f"pop_{tail}" if f"pop_{tail}" in self.g.nodes else "treeline"
        tail = f.split(":")[-1].strip()
        if ":" in f:
            nid = f"landmark_{slug(tail)}"       # "structure: Pont d'Iéna", "landmark distance (m): Eiffel Tower"
            if nid in self.g.nodes:
                return nid
        for cls in sorted(self.pop_classes, key=len, reverse=True):
            if cls == tail or cls in words:
                return f"pop_{cls}"
        if "haze" in words or "atmosphere" in words:
            return "atmosphere"
        if "terrain" in words or "ground" in words or "slope" in words:
            return "ground"
        return "world"

    def overrides(self) -> None:
        n_rows = 0
        for stage, rows in self.run.override_files():
            for row in rows:
                if not isinstance(row, dict) or not row.get("field"):
                    continue
                n_rows += 1
                owner = self.g.find(self._override_owner(row["field"]))
                if owner is None:
                    owner = self.g.find("world")
                attr = "override_" + slug(row["field"], 48)
                used = row.get("used")
                unit = "m" if re.search(r"\(m\)|_m\b|width|height|depth", row["field"].lower()) else (
                    "deg" if "deg" in row["field"].lower() else "")
                if not isinstance(used, (int, float, list)) or isinstance(used, bool):
                    unit = ""
                owner.set(attr, _round(used), "fitted", unit, 1.0, stage)
                owner.set(attr + "_source", {"source": row.get("source"), "measured": _round(row.get("measured")),
                                             "plan": _round(row.get("plan")), "measured_source": row.get("measured_source"),
                                             "note": row.get("note")}, "fitted", "", 1.0, stage)
        if n_rows == 0:
            self.run.warn("no plan_overrides_*.json in the run")
        self.g.meta["plan_overrides"] = n_rows


def export_graph(run_dir: Path, config_path: Path = DEFAULT_CONFIG) -> SceneGraph:
    """Build the scene graph of a cached run on the CPU. Missing data -> graph.meta['warnings']."""
    server = ROOT / "server"
    if str(server) not in sys.path:
        sys.path.insert(0, str(server))
    run = _Run(Path(run_dir), Path(config_path))
    return _Exporter(run).build()
