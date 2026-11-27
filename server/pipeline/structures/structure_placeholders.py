"""Structure Placeholders: plan-named far structures as geometry.

docs/hybrid-scene-plan.md, "The plan as scene understanding". A bridge or a building
block the plan reads as visible, with a bearing, a distance and a height, usually
reaches Scene Generation as nothing: its detection sits at the depth model's far clamp,
stays in the terrain plate as backdrop, and is absorbed into a neighbouring blob. The
harness got these because its planner wrote them as boxes. This stage does the bounded
version of that: for every visible plan landmark of a placeholder category that Scene
Generation did not place as a mesh near that bearing, it builds simple massing

  bridge    the bridge's own reconstruction when asset generation made one (its
            detection is kept in the far plate, so Scene Generation never places it),
            stretched across the river channel from the terrain plan; without one, a
            deck on piers with parapets. Perpendicular to the channel axis at the plan
            distance either way
  building  a block of the plan height facing the camera

at min(plan distance, far_clip_fraction x far clip), scaled so its angular size is kept,
standing on the reconstructed terrain (water level for piers), coloured from the pixels
of its own detection box in the photograph (median of the box). No default colour: a
structure whose pixels cannot be found is built grey and logged as such.

Input   ContextKey.SCENE, SCENE_PLAN, TERRAIN_MESH, WATER_MESH, HEIGHT_MAP(+PARAMS),
        TERRAIN_PLAN, INPUT, OBJECT_COUNT, metadata_{i}
Output  structure_{slug} (Mesh), objects appended to ContextKey.SCENE,
        ContextKey.STRUCTURE_PLACEHOLDERS (list of records)
"""
from __future__ import annotations

import math
import re
from logging import Logger
from typing import Any, Optional

import numpy as np
import trimesh
from PIL import Image as PILImage

from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration
from pipeline.scene_generation.projection import height_map_y_at, mesh_y_at, terrain_local_xz
from pipeline.scene_plan import ScenePlan
from scene.mesh import Mesh
from scene.object import Object3D, vec3

MARKER = "structure_placeholders_complete"

# Placed meshes of these classes stand for any plan building or bridge at their bearing.
STRUCTURE_CLASSES = {"building", "bridge", "tower", "monument", "landmark", "statue", "skyscraper", "lighthouse",
                     "structure", "church", "cathedral", "castle", "dome", "arch", "house", "palace", "temple"}
# Plan categories that any placed structure satisfies (the plan could not say what kind).
GENERIC_STRUCTURE = {"building", "structure", "landmark", "monument"}
_STOP_WORDS = {"the", "of", "a", "an", "de", "la", "le", "du", "des", "les", "and", "block"}


def _name_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 2 and w not in _STOP_WORDS}


class StructurePlaceholdersConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        categories: list[str] | None = None,
        min_evidence: str = "visible",
        far_clip_fraction: float = 0.95,
        bearing_match_deg: float = 20.0,
        deck_thickness_m: float = 2.5,
        deck_width_m: float = 14.0,
        pier_spacing_m: float = 24.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.categories = [str(c).lower() for c in (categories or ["bridge", "building"])]
        self.min_evidence = str(min_evidence)
        self.far_clip_fraction = float(far_clip_fraction)
        self.bearing_match_deg = float(bearing_match_deg)
        self.deck_thickness_m = float(deck_thickness_m)
        self.deck_width_m = float(deck_width_m)
        self.pier_spacing_m = float(pier_spacing_m)


def _oriented_box(center: np.ndarray, u: np.ndarray, half_u: float, half_y: float, half_w: float) -> tuple[np.ndarray, np.ndarray]:
    """A box in the terrain frame: `u` is a unit XZ direction (long axis), w is the
    horizontal perpendicular, y up. Returns (8 vertices, 12 faces)."""
    w = np.array([-u[2], 0.0, u[0]])
    up = np.array([0.0, 1.0, 0.0])
    verts = []
    for su in (-1, 1):
        for sy in (-1, 1):
            for sw in (-1, 1):
                verts.append(center + su * half_u * u + sy * half_y * up + sw * half_w * w)
    v = np.asarray(verts)
    # index = su*4 + sy*2 + sw with su,sy,sw in {0,1}
    faces = np.array([
        [0, 1, 3], [0, 3, 2],  # -u face
        [4, 6, 7], [4, 7, 5],  # +u face
        [0, 4, 5], [0, 5, 1],  # -y (bottom)
        [2, 3, 7], [2, 7, 6],  # +y (top)
        [0, 2, 6], [0, 6, 4],  # -w
        [1, 5, 7], [1, 7, 3],  # +w
    ])
    return v, faces


def _flat_colour_mesh(parts: list[tuple[np.ndarray, np.ndarray]], rgb: tuple[int, int, int]) -> trimesh.Trimesh:
    verts, faces = [], []
    offset = 0
    for v, f in parts:
        verts.append(v)
        faces.append(f + offset)
        offset += len(v)
    V = np.concatenate(verts, axis=0)
    F = np.concatenate(faces, axis=0)
    tm = trimesh.Trimesh(V, F, process=False)
    tm.fix_normals()
    image = PILImage.new("RGB", (4, 4), tuple(int(c) for c in rgb))
    uv = np.full((len(V), 2), 0.5, dtype=np.float64)
    tm.visual = trimesh.visual.texture.TextureVisuals(uv=uv, image=image)
    return tm


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")[:40] or "structure"


class StructurePlaceholdersStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return StructurePlaceholdersConfiguration

    # ── helpers ─────────────────────────────────────────────────────────────
    @staticmethod
    def _bearing_of_local(x: float, z: float) -> float:
        return float(np.degrees(np.arctan2(x, z)))

    @staticmethod
    def _dbearing(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _detection_box(self, context: PipelineContext, photo: Optional[PILImage.Image], category: str,
                       phrase: str, bearing: float, hfov: float) -> Optional[tuple[int, tuple[float, float, float, float]]]:
        """The detection of this category (or carrying this name) nearest the bearing,
        within 30 degrees: its index and its box in the photograph (x, y, w, h)."""
        if photo is None:
            return None
        count = int(context.input_object(ContextKey.OBJECT_COUNT) or 0)
        W, H = photo.size
        words = {w for w in re.split(r"\W+", phrase.lower()) if len(w) > 2}
        best = None
        for idx in range(count):
            m = context.input_object(f"metadata_{idx}") or {}
            cls = str(m.get("class", "")).lower()
            name = str(m.get("name", "")).lower()
            vetoed = str(m.get("vetoed_class", "")).lower()
            if cls != category and vetoed != category and not (words and words & set(name.split())):
                continue
            pb = m.get("photo_box")
            if not pb:
                continue
            x, y, w, h = (float(v) for v in pb)
            az = ((x + w / 2.0) / W - 0.5) * hfov
            d = self._dbearing(az, bearing)
            if d <= 30.0 and (best is None or d < best[0]):
                best = (d, idx, (x, y, w, h))
        return None if best is None else (best[1], best[2])

    def _detection_colour(self, context: PipelineContext, photo: Optional[PILImage.Image], category: str,
                          phrase: str, bearing: float, hfov: float) -> tuple[Optional[tuple[int, int, int]], str]:
        """Median colour of the photograph inside the best matching detection box."""
        if photo is None:
            return None, "no photograph"
        W, H = photo.size
        best = self._detection_box(context, photo, category, phrase, bearing, hfov)
        if best is None:
            return self._horizon_colour(photo, bearing, hfov)
        idx, (x, y, w, h) = best
        box = (max(0, int(x)), max(0, int(y)), min(W, int(math.ceil(x + w))), min(H, int(math.ceil(y + h))))
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return None, f"detection {idx} box too small"
        arr = np.asarray(photo.convert("RGB").crop(box), dtype=np.float32).reshape(-1, 3)
        # Drop the sky-blue pixels a thin far structure is seen against.
        r, g, b = arr[:, 0], arr[:, 1], arr[:, 2]
        skyish = (b > r + 25) & (b > g + 10)
        keep = arr[~skyish] if (~skyish).sum() >= max(8, 0.2 * len(arr)) else arr
        med = np.median(keep, axis=0)
        return tuple(int(round(float(c))) for c in med), f"median of detection {idx}'s box ({box[2]-box[0]}x{box[3]-box[1]} px)"

    def _reconstructed_bridge(self, context: PipelineContext, photo: Optional[PILImage.Image], phrase: str, bearing: float,
                              hfov: float, centre: np.ndarray, u: np.ndarray, half_span: float, water_y: float,
                              deck_top: float, deck_width: float) -> Optional[tuple[trimesh.Trimesh, str, str]]:
        """The bridge's reconstruction laid across the channel, in terrain-local
        coordinates: its long axis along `u` over the span, its foot just under the
        water, the face it was photographed from towards the camera. Its height is the
        span times the proportions of its detection box in the photograph (Paris's
        Pont d'Iena: 307 x 25 px, 9 m on a 114 m span), no lower than the placeholder
        deck and no taller than the mesh's own proportions. None when asset generation
        made no bridge mesh."""
        source, src = None, None
        for i in range(8):
            cand = context.input_mesh(f"category_mesh_bridge_{i}")
            if cand is not None and cand.mesh is not None and len(cand.mesh.vertices):
                source, src = f"category_mesh_bridge_{i}", cand
                break
        if src is None:
            return None
        tm = src.mesh.copy()
        lo, hi = np.asarray(tm.bounds[0], dtype=float), np.asarray(tm.bounds[1], dtype=float)
        ext = hi - lo
        long_axis = 0 if ext[0] >= ext[2] else 2
        across = 2 - long_axis
        if ext[long_axis] < 1e-6 or ext[1] < 1e-6 or ext[across] < 1e-6:
            return None
        span = 2.0 * half_span
        own_h = span * ext[1] / ext[long_axis]
        target_h, why = deck_top - water_y, "the plan height"
        best = self._detection_box(context, photo, "bridge", phrase, bearing, hfov)
        if best is not None and best[1][2] > 4 and best[1][3] > 2:
            target_h = span * best[1][3] / best[1][2]
            why = f"the proportions of detection {best[0]}'s box ({best[1][2]:.0f}x{best[1][3]:.0f} px)"
        height = float(min(max(target_h, deck_top - water_y), own_h))
        sink = 0.05 * height
        # Local axes of the mesh onto the scene: long axis -> u, up -> up, the third
        # onto the channel axis with the sign that keeps it a rotation (no mirror).
        n = np.array([-u[2], 0.0, u[0]])
        e_long, e_across = (u, n) if long_axis == 0 else (n, u)
        basis = np.zeros((3, 3))
        basis[:, long_axis], basis[:, 1], basis[:, across] = e_long, (0.0, 1.0, 0.0), e_across
        if np.linalg.det(basis) < 0:
            basis[:, across] = -basis[:, across]
        # The reconstruction's +Z is the side it was photographed from; turn it half
        # round when that side would face away from the camera.
        if float(np.dot(basis[:, 2], -centre)) < 0:
            basis[:, 0], basis[:, 2] = -basis[:, 0], -basis[:, 2]
        scale = np.ones(3)
        scale[long_axis], scale[1], scale[across] = span / ext[long_axis], (height + sink) / ext[1], deck_width / ext[across]
        mid = (lo + hi) / 2.0
        m = np.eye(4)
        m[:3, :3] = basis @ np.diag(scale)
        m[:3, 3] = (np.array([centre[0], water_y - sink, centre[2]])
                    - m[:3, :3] @ np.array([mid[0], lo[1], mid[2]]))
        tm.apply_transform(m)
        note = (f"{source} laid {span:.0f} m across the channel, {height:.1f} m above water from {why} "
                f"(its own proportions would stand {own_h:.1f} m), {deck_width:.0f} m deep")
        return tm, source, note

    @staticmethod
    def _horizon_colour(photo: PILImage.Image, bearing: float, hfov: float) -> tuple[Optional[tuple[int, int, int]], str]:
        """No detection to read: the photograph's own pixels at the plan bearing, just
        under the skyline in that column (a far structure stands on the horizon)."""
        W, H = photo.size
        if hfov <= 0 or abs(bearing) > hfov / 2.0 + 2.0:
            return None, "bearing outside the photograph"
        cx = int(round((bearing / hfov + 0.5) * (W - 1)))
        x0, x1 = max(0, cx - 30), min(W, cx + 30)
        arr = np.asarray(photo.convert("RGB"), dtype=np.float32)
        strip = arr[:, x0:x1, :]
        r, g, b = strip[..., 0], strip[..., 1], strip[..., 2]
        # Sky is blue-dominant or a bright, unsaturated cloud.
        skyish = ((b > r + 15) & (b > g + 5)) | ((strip.min(-1) > 185) & (strip.max(-1) - strip.min(-1) < 45))
        ground_rows = np.flatnonzero((~skyish).mean(1) > 0.6)
        if ground_rows.size == 0:
            return None, "no skyline in the column"
        top = int(ground_rows[0])
        band = strip[top:min(H, top + 40)]
        px = band.reshape(-1, 3)
        sk = ((px[:, 2] > px[:, 0] + 15) & (px[:, 2] > px[:, 1] + 5)) | ((px.min(1) > 185) & (px.max(1) - px.min(1) < 45))
        # Foliage is not masonry: a structure on the skyline is read from the pixels
        # that are neither sky nor green.
        green = (px[:, 1] > px[:, 2] + 15) & (px[:, 1] >= px[:, 0] - 10)  # green and yellow-green foliage
        keep = px[~sk & ~green]
        if keep.shape[0] < 24:
            return None, f"only sky and foliage under the skyline at column {cx}"
        med = np.median(keep, axis=0)
        return tuple(int(round(float(c))) for c in med), f"photograph under the skyline at column {cx} (rows {top}-{top + 40}, {keep.shape[0]} px)"

    # ── stage ───────────────────────────────────────────────────────────────
    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: StructurePlaceholdersConfiguration = self.config
        records: list[dict] = []
        scene = context.input_scene(ContextKey.SCENE)
        plan = ScenePlan.from_context(context)
        if scene is None or plan is None:
            self.log_info("No scene or no plan; nothing to place")
            context.add_object(ContextKey.STRUCTURE_PLACEHOLDERS, records)
            context.add_object(MARKER, True)
            return context

        landmarks = [lm for lm in plan.items("landmarks", cfg.min_evidence)
                     if ScenePlan.category_of(lm, "landmarks") in cfg.categories
                     and isinstance(lm.get("bearing_deg"), (int, float))
                     and isinstance(lm.get("distance_m"), (int, float)) and lm.get("distance_m") > 0]
        if not landmarks:
            self.log_info("The plan names no visible structure of a placeholder category")
            context.add_object(ContextKey.STRUCTURE_PLACEHOLDERS, records)
            context.add_object(MARKER, True)
            return context

        terrain_mesh = context.input_mesh(ContextKey.TERRAIN_MESH)
        water_mesh = context.input_mesh(ContextKey.WATER_MESH)
        height_map = context.input_depth(ContextKey.HEIGHT_MAP)
        grid_m = float((context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}).get("grid_size_meters", 0.0))
        terrain_plan = context.input_object(ContextKey.TERRAIN_PLAN) or {}
        channel = terrain_plan.get("channel") if isinstance(terrain_plan, dict) else None
        hfov = float(terrain_plan.get("hfov_deg", 0.0) or 0.0) if isinstance(terrain_plan, dict) else 0.0
        if hfov <= 0.0:
            intr = context.input_object(ContextKey.INTRINSICS)
            hfov = float(getattr(intr, "fov", 60.0) or 60.0)
        photo_img = context.input_image(ContextKey.INPUT)
        photo = photo_img.image if photo_img is not None else None
        yaw = float(scene.skybox_rotation or 0.0)
        cam = scene.extrinsics.translation if scene.extrinsics is not None else (0.0, 0.0, 0.0)
        camera_xz = (float(cam[0]), float(cam[2]))
        far_clip = float(scene.far_clip_plane or 0.0)
        max_d = cfg.far_clip_fraction * far_clip if far_clip > 0 else None
        # The far-field backdrop shell stands at the depth ceiling (Paris: 531 m, band
        # floor 478 m); anything placed beyond it is painted over. A structure stands
        # in front of the shell, at the far-field start at most.
        far_start = float(context.input_object(ContextKey.FAR_FIELD_START_M) or 0.0)
        if far_start > 0.0:
            max_d = min(max_d, far_start) if max_d else far_start

        def ground_y(local_x: float, local_z: float) -> Optional[float]:
            hits = [y for y in (mesh_y_at(local_x, local_z, m) for m in (terrain_mesh, water_mesh) if m is not None)
                    if y is not None]
            if hits:
                return max(hits)
            if height_map is not None and grid_m > 0:
                return height_map_y_at(local_x, local_z, height_map, grid_m)
            return None

        # What Scene Generation already placed as a mesh, by class, bearing and name.
        placed: list[tuple[str, float, float, float, str]] = []
        for obj in scene.objects:
            if obj.source_index is None or not obj.mesh:
                continue
            m = context.input_object(f"metadata_{obj.source_index}") or {}
            lx, lz = terrain_local_xz(obj.position["x"], obj.position["z"], yaw, camera_x=camera_xz[0], camera_z=camera_xz[1])
            placed.append((str(m.get("class", "")).lower(), self._bearing_of_local(lx, lz), float(math.hypot(lx, lz)),
                           float(m.get("world_height") or 0.0), str(m.get("name") or m.get("phrase") or "").lower()))

        task = self.create_progress(len(landmarks), "Structure placeholders…")
        for lm in landmarks:
            category = ScenePlan.category_of(lm, "landmarks")
            name = str(lm.get("name") or category)
            bearing = float(lm["bearing_deg"])
            distance = float(lm["distance_m"])
            height = float(lm.get("height_m") or 0.0)
            if height <= 0.0:
                height = 10.0 if category == "bridge" else 20.0
            tol = 8.0 if category == "tree" else cfg.bearing_match_deg
            # A tree landmark counts as present only when a placed tree of comparable
            # height stands near its bearing and distance (a 2 m sapling at 15 m is not
            # the 9 m lone fir the plan names).
            # A built structure counts as present when a mesh of its category, of any structure
            # class, or carrying its name stands near its bearing: the plan may file the Eiffel
            # Tower under "building" while its detection is a "tower" landmark card, and a 115 m
            # block in front of that card hid the whole skyline (Paris mac1, 2026-11-23).
            name_words = _name_words(name) | _name_words(str(lm.get("phrase") or ""))
            def _same_thing(p) -> bool:
                # The same class, the same name, or -- only for a generic plan category
                # (building, structure, landmark) -- any placed structure. A specific one
                # (bridge, tower) is not stood in for by another kind: the Pont d'Iéna was
                # skipped on 2026-11-23 because the Eiffel card stood 20 deg away.
                if category == "tree":
                    return p[0] == category
                if p[0] == category or bool(name_words & _name_words(p[4])):
                    return True
                return category in GENERIC_STRUCTURE and p[0] in STRUCTURE_CLASSES
            near = [p for p in placed if _same_thing(p) and self._dbearing(p[1], bearing) <= tol
                    and (category != "tree" or (abs(p[2] - distance) <= max(10.0, 0.5 * distance)
                                                and p[3] >= 0.5 * height))]
            if near:
                self.log_info(f"  '{name}' ({category}, bearing {bearing:.0f}): a placed {near[0][0] or category} mesh"
                              + (f" '{near[0][4]}'" if near[0][4] else "") + f" already stands within {tol:.0f} deg"
                              + (f" at {near[0][2]:.0f} m, {near[0][3]:.1f} m tall" if category == "tree" else "")
                              + "; skipped")
                records.append({"name": name, "category": category, "status": "already placed"})
                self.advance_progress(task)
                continue
            d = min(distance, max_d) if max_d else distance
            f = d / distance
            h = height * f
            az = math.radians(bearing)
            lx, lz = d * math.sin(az), d * math.cos(az)
            colour, colour_note = self._detection_colour(context, photo, category, str(lm.get("phrase") or name), bearing, hfov)
            if colour is None:
                colour = (150, 148, 140)
                colour_note = "not measured (" + colour_note + "); neutral grey"
            parts: list[tuple[np.ndarray, np.ndarray]] = []
            if category == "tree":
                # The plan's named tree (Rainier's lone fir at the left edge) as an
                # instance of the species layer's conifer, scaled to the plan height,
                # on the ground at the plan bearing and distance.
                species = context.input_object(ContextKey.SPECIES_MESHES) or {}
                tree_keys = [k for k, v in species.items() if k.startswith("category_mesh_tree_") and "variant_of" not in v]
                if not tree_keys:
                    self.log_info(f"  '{name}' (tree): no species tree mesh to stand in; skipped")
                    records.append({"name": name, "category": category, "status": "no species mesh"})
                    self.advance_progress(task)
                    continue
                gy = ground_y(lx, lz)
                if gy is None:
                    self.log_info(f"  '{name}' (tree): no terrain under bearing {bearing:.0f} at {d:.0f} m; skipped")
                    records.append({"name": name, "category": category, "status": "no ground"})
                    self.advance_progress(task)
                    continue
                tkey = sorted(tree_keys)[0]
                tmesh = context.input_mesh(tkey)
                if tmesh is None:
                    self.advance_progress(task)
                    continue
                mesh_h = float(tmesh.extents[1]) or 1.0
                scale = h / mesh_h
                th = math.radians(yaw)
                wx = camera_xz[0] + lx * math.cos(th) + lz * math.sin(th)
                wz = camera_xz[1] - lx * math.sin(th) + lz * math.cos(th)
                obj = Object3D.mesh(tkey, wx, gy - float(tmesh.mesh.bounds[0][1]) * scale, wz)
                obj.name = name
                obj.scale = vec3(scale, scale, scale)
                obj.rotation = vec3(0.0, float((self.seed * 37 + int(bearing)) % 360), 0.0)
                scene.add_object(obj)
                note = f"{tkey} scaled x{scale:.2f} to {h:.1f} m on ground y {gy:.1f}"
                self.log_info(f"  '{name}' (tree, {lm.get('evidence')}): {note}; bearing {bearing:.0f}, {d:.0f} m")
                records.append({"name": name, "category": category, "status": "placed", "mesh": tkey, "bearing_deg": bearing,
                                "distance_m": d, "height_m": h, "note": note})
                plan.overrides.append({"field": f"structure: {name}", "plan": [bearing, distance, height], "evidence": lm.get("evidence"),
                                       "because": lm.get("because", ""), "measured": None, "measured_source": "none",
                                       "used": [bearing, d, h], "source": "plan", "note": note})
                self.advance_progress(task)
                continue
            if category == "bridge" and isinstance(channel, dict) and channel.get("width"):
                slope = float(channel.get("slope", 0.0))
                offset = float(channel.get("offset", 0.0))
                width = float(channel["width"]) * 1.0
                water_y = float(channel.get("water_y", 0.0))
                bank_y = float(channel.get("bank_y", water_y))
                xc = offset + slope * lz
                if grid_m > 0 and max(abs(xc), abs(lz)) > grid_m / 2.0:
                    # Beyond the terrain there is no river to cross: the bridge stood
                    # over nothing, behind the backdrop that carries its picture, and
                    # showed over the backdrop's canopy as a grey sliver in the sky
                    # (Paris's Pont d'Iena, laid 368 m out past a +-200 m grid).
                    self.log_info(f"  '{name}' (bridge, {lm.get('evidence')}): bearing {bearing:.0f}, {d:.0f} m is beyond the "
                                  f"terrain (+-{grid_m / 2.0:.0f} m), no water under it; left to the backdrop's picture of it")
                    records.append({"name": name, "category": category, "status": "beyond the terrain",
                                    "bearing_deg": bearing, "distance_m": d, "plan_distance_m": distance})
                    self.advance_progress(task)
                    continue
                axis = np.array([slope, 0.0, 1.0]); axis /= np.linalg.norm(axis)
                u = np.array([axis[2], 0.0, -axis[0]])  # across the river
                centre = np.array([xc, 0.0, lz])
                deck_top = max(water_y + h, bank_y + 1.0)
                deck_bot = deck_top - cfg.deck_thickness_m
                half_span = width / 2.0 + 12.0
                built = self._reconstructed_bridge(context, photo, str(lm.get("phrase") or name), bearing, hfov,
                                                   centre, u, half_span, water_y, deck_top, cfg.deck_width_m)
                if built is not None:
                    tm, source, note = built
                    key = f"structure_{_slug(name)}"
                    context.add_mesh(key, Mesh(tm))
                    obj = Object3D.mesh(key, 0.0, 0.0, 0.0)
                    obj.name = name
                    obj.rotation = vec3(0.0, yaw, 0.0)
                    scene.add_object(obj)
                    self.log_info(f"  '{name}' (bridge, {lm.get('evidence')}): {note}; bearing {bearing:.0f}, {d:.0f} m"
                                  + (f" (plan {distance:.0f} m, scaled x{f:.2f})" if f < 0.999 else ""))
                    records.append({"name": name, "category": category, "status": "placed", "mesh": key, "source_mesh": source,
                                    "bearing_deg": bearing, "distance_m": d, "plan_distance_m": distance, "height_m": h,
                                    "note": note})
                    plan.overrides.append({"field": f"structure: {name}", "plan": [bearing, distance, height],
                                           "evidence": lm.get("evidence"), "because": lm.get("because", ""), "measured": None,
                                           "measured_source": "none (absorbed into the backdrop)", "used": [bearing, d, h],
                                           "source": "plan", "note": note})
                    self.advance_progress(task)
                    continue
                parts.append(_oriented_box(centre + np.array([0.0, (deck_top + deck_bot) / 2.0, 0.0]), u, half_span,
                                           cfg.deck_thickness_m / 2.0, cfg.deck_width_m / 2.0))
                for sw in (-1, 1):
                    parts.append(_oriented_box(centre + np.array([0.0, deck_top + 0.6, 0.0]) + sw * (cfg.deck_width_m / 2.0 - 0.4) * np.array([-u[2], 0.0, u[0]]),
                                               u, half_span, 0.6, 0.4))
                n_piers = max(2, int(round(width / cfg.pier_spacing_m)) + 1)
                for k in range(n_piers):
                    t = -width / 2.0 + width * k / (n_piers - 1)
                    pc = centre + t * u
                    parts.append(_oriented_box(np.array([pc[0], (water_y - 2.0 + deck_bot) / 2.0, pc[2]]), u, 1.8,
                                               (deck_bot - water_y + 2.0) / 2.0, cfg.deck_width_m / 2.0 + 0.5))
                base_y = water_y
                note = f"deck {2*half_span:.0f} m across the channel ({width:.0f} m) on {n_piers} piers, top {deck_top - water_y:.1f} m above water"
            else:
                gy = ground_y(lx, lz)
                # The terrain grid may end before the plan distance: walk the ray in
                # until it stands on reconstructed ground, keeping the angular size.
                steps = 0
                while gy is None and d > 40.0 and steps < 24:
                    d *= 0.92
                    steps += 1
                    f = d / distance
                    h = height * f
                    lx, lz = d * math.sin(az), d * math.cos(az)
                    gy = ground_y(lx, lz)
                if gy is None:
                    self.log_info(f"  '{name}' ({category}): no terrain under bearing {bearing:.0f} at {d:.0f} m; skipped")
                    records.append({"name": name, "category": category, "status": "no ground"})
                    self.advance_progress(task)
                    continue
                w = max(h * 1.6, 18.0) * (1.0 if category == "building" else 1.0)
                u = np.array([math.cos(az), 0.0, -math.sin(az)])  # facing the camera
                centre = np.array([lx, gy + h / 2.0, lz])
                parts.append(_oriented_box(centre, u, w / 2.0, h / 2.0, max(8.0, w * 0.4)))
                base_y = gy
                note = f"block {w:.0f} x {h:.0f} m on ground y {gy:.1f}"
            tm = _flat_colour_mesh(parts, colour)
            key = f"structure_{_slug(name)}"
            context.add_mesh(key, Mesh(tm))
            obj = Object3D.mesh(key, 0.0, 0.0, 0.0)
            obj.name = name
            obj.rotation = vec3(0.0, yaw, 0.0)
            scene.add_object(obj)
            self.log_info(f"  '{name}' ({category}, {lm.get('evidence')}): {note}; bearing {bearing:.0f}, "
                          f"{d:.0f} m" + (f" (plan {distance:.0f} m, scaled x{f:.2f})" if f < 0.999 else "")
                          + f"; colour {colour} from {colour_note}")
            records.append({"name": name, "category": category, "status": "placed", "mesh": key, "bearing_deg": bearing,
                            "distance_m": d, "plan_distance_m": distance, "height_m": h, "colour": list(colour),
                            "colour_source": colour_note, "note": note})
            plan.overrides.append({"field": f"structure: {name}", "plan": [bearing, distance, height], "evidence": lm.get("evidence"),
                                   "because": lm.get("because", ""), "measured": None, "measured_source": "none (absorbed into the backdrop)",
                                   "used": [bearing, d, h], "source": "plan", "note": note})
            self.advance_progress(task)
        self.finish_progress(task)
        placed_n = sum(1 for r in records if r["status"] == "placed")
        self.log_info(f"Structure placeholders: {placed_n} built, {len(records) - placed_n} skipped")
        if placed_n:
            context.add_scene(ContextKey.SCENE, scene)
        plan.publish(context, "structures")
        context.add_object(ContextKey.STRUCTURE_PLACEHOLDERS, records)
        context.add_object(MARKER, True)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(MARKER) is True
