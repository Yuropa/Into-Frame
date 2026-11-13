"""Species Assets: procedural 3D plants for the near field, coloured from the photograph.

docs/hybrid-scene-plan.md section 6 (Route A). For every (class, bucket) group of flower
or tree detections that Panorama Asset Generation gave a card, this stage

  1. measures the bucket's palette from its members' masked hires crops,
  2. matches the bucket to one of the Scene Plan's visible species (by the plan's crop
     hint palette when it has one, else its colour words; shape words give the form),
  3. builds a procedural variant of that form in Blender 5.2 headless (CPU, seconds)
     with the BUCKET's measured palette -- never a default colour, never a word --
  4. publishes it as category_mesh_{class}_{bucket}, replacing the reconstruction, and
     records the match in ContextKey.SPECIES_MESHES so Scene Generation keeps the mesh
     within species_mesh_near_distance_m and Category Mesh Rigging gives it sway.

The card stays as the far LOD. Buckets no species matches (bloom palette further than
match_max_delta_e from every candidate) keep their card and are logged.
GLBs are cached across runs by the spec's hash in cache/species/.

Input   ContextKey.SCENE_PLAN, ContextKey.INPUT, ContextKey.OBJECT_COUNT, metadata_{i},
        crop_hires_{i} / crop_{i}, category_mesh_{class}_{bucket}(_card)
Output  category_mesh_{class}_{bucket} (Mesh), ContextKey.SPECIES_MESHES (dict)
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from logging import Logger
from pathlib import Path
from typing import Any, Optional

from PIL import Image as PILImage

from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration
from pipeline.scene_plan import ScenePlan
from pipeline.species.palette import anchors_from_words, measure_palette, palette_distance
from scene.mesh import Mesh

ROOT = Path(__file__).resolve().parents[3]
BUILDER = Path(__file__).resolve().parent / "blender_species.py"
MARKER = "species_assets_complete"

_FORM_WORDS = [
    ("conifer", r"\b(fir|spruce|pine|conifer|spire|hemlock|cedar|larch|evergreen)\b"),
    ("raceme", r"\b(lupin|lupine|raceme|foxglove|delphinium|larkspur|pea.flower)\b"),
    ("seedhead", r"\b(seed.?head|anemone|pasque|fuzzy|dandelion|clock)\b"),
    ("tuft", r"\b(grass|sedge|tussock|rush|carex)\b"),
    ("shrub", r"\b(shrub|bush|heather|willow|huckleberry|sagebrush)\b"),
    ("spike", r"\b(spike|paintbrush|lousewort|bistort|castilleja|pedicularis|orchid|lavender|salvia)\b"),
]


def species_form(species: dict) -> str:
    text = " ".join(str(species.get(k, "")) for k in ("name", "phrase", "form", "class")).lower()
    for form, pattern in _FORM_WORDS:
        if re.search(pattern, text):
            return form
    cls = str(species.get("class", "")).lower()
    return {"tree": "conifer", "grass": "tuft", "shrub": "shrub", "reed": "tuft", "moss": "tuft"}.get(cls, "spike")


class SpeciesAssetsConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        blender_path: str = "~/.local/bin/blender-5.2",
        cache_dir: str | None = None,
        classes: list[str] | None = None,
        max_members: int = 12,
        match_max_delta_e: float = 40.0,
        variants: int = 3,
        tree_variants: int = 2,
        timeout_s: int = 600,
        min_evidence: str = "visible",
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.blender_path = Path(os.path.expanduser(blender_path))
        self.cache_dir = Path(cache_dir) if cache_dir else ROOT / "cache" / "species"
        self.classes = [str(c).lower() for c in (classes or ["flower", "tree"])]
        self.max_members = int(max_members)
        self.match_max_delta_e = float(match_max_delta_e)
        self.variants = max(1, int(variants))
        self.tree_variants = max(1, int(tree_variants))
        self.timeout_s = int(timeout_s)
        self.min_evidence = str(min_evidence)


class SpeciesAssetsStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return SpeciesAssetsConfiguration

    # ── measurement ─────────────────────────────────────────────────────────
    def _bucket_members(self, context: PipelineContext) -> dict[tuple[str, int], list[int]]:
        count = int(context.input_object(ContextKey.OBJECT_COUNT) or 0)
        groups: dict[tuple[str, int], list[int]] = {}
        for idx in range(count):
            m = context.input_object(f"metadata_{idx}") or {}
            cls, bucket = str(m.get("class", "")).lower(), m.get("bucket")
            if cls in self.config.classes and bucket is not None and m.get("in_captured_fov", True):
                groups.setdefault((cls, int(bucket)), []).append(idx)
        return groups

    def _bucket_palette(self, context: PipelineContext, members: list[int], cls: str = "flower") -> dict:
        crops: list[PILImage.Image] = []
        # Photographed detections first: Distribution Synthesis's painted instances
        # carry the class and bucket too, but no crop of their own.
        def photographed(idx: int) -> bool:
            m = context.input_object(f"metadata_{idx}") or {}
            return bool(m.get("photo_box")) or str(m.get("source", "")).startswith(("sam2", "grounding"))
        ordered = [i for i in members if photographed(i)] + [i for i in members if not photographed(i)]
        for idx in ordered[: self.config.max_members]:
            crop = context.input_image(f"crop_hires_{idx}") or context.input_image(f"crop_{idx}")
            if crop is not None:
                crops.append(crop.image)
        # Conifer foliage is dark by nature; the lit-side filter that keeps a bloom
        # from being darkened twice would turn it olive.
        return measure_palette(crops, seed=self.seed, lit_share=0.6 if cls == "flower" else 0.9)

    def _species_palette(self, photo: PILImage.Image, species: dict) -> tuple[Optional[dict], str]:
        hint = species.get("crop_hint")
        if isinstance(hint, list) and len(hint) == 4 and photo is not None:
            x, y, w, h = (float(v) for v in hint)
            W, H = photo.size
            box = (int(x * W), int(y * H), int((x + w) * W), int((y + h) * H))
            if box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
                region = photo.convert("RGBA").crop(box)
                pal = measure_palette([region], seed=self.seed)
                if pal["bloom"] or pal["foliage"]:
                    return pal, "crop_hint"
        anchors = anchors_from_words(species.get("colours") or [])
        if anchors:
            return {"bloom": anchors, "bloom_shares": [1.0 / len(anchors)] * len(anchors), "foliage": [], "stem": []}, "words"
        return None, "none"

    # ── matching ────────────────────────────────────────────────────────────
    def _match(self, cls: str, bucket_pal: dict, candidates: list[tuple[dict, dict, str]]) -> tuple[Optional[dict], Optional[float], str]:
        """(species, dE, how) for the bucket; candidates are (species, palette, source)."""
        best = None
        if cls == "flower":
            if bucket_pal["bloom_fraction"] < 0.02 or not bucket_pal["bloom"]:
                return None, None, "bucket has no bloom pixels"
            for sp, pal, src in candidates:
                if not pal or not pal.get("bloom"):
                    continue
                d = palette_distance(bucket_pal["bloom"], pal["bloom"], bucket_pal.get("bloom_shares"))
                if d is not None and (best is None or d < best[1]):
                    best = (sp, d, src)
            if best is None:
                return None, None, "no candidate species with a bloom palette"
            if best[1] > self.config.match_max_delta_e:
                return None, best[1], f"nearest species '{best[0].get('name')}' is dE {best[1]:.0f} away"
            return best[0], best[1], f"bloom palette dE {best[1]:.0f} ({best[2]})"
        # Trees and shrubs: the class's species, nearest foliage palette when measurable.
        for sp, pal, src in candidates:
            d = palette_distance(bucket_pal["foliage"], (pal or {}).get("foliage") or [], bucket_pal.get("foliage_shares")) if pal else None
            score = d if d is not None else 999.0
            if best is None or score < best[1]:
                best = (sp, score, src)
        if best is None:
            return None, None, "no plan species of this class"
        return best[0], (best[1] if best[1] < 999.0 else None), "class match" + (f", foliage dE {best[1]:.0f}" if best[1] < 999.0 else "")

    # ── building ────────────────────────────────────────────────────────────
    @staticmethod
    def _spec_entry(key: str, species: dict, bucket_pal: dict, seed: int, variants: int) -> dict:
        form = species_form(species)
        size = species.get("size_m")
        height = float(size) if isinstance(size, (int, float)) and size > 0 else (12.0 if form == "conifer" else 0.4)
        width = height * (0.22 if form == "conifer" else 0.6 if form == "shrub" else 0.45)
        foliage = bucket_pal["foliage"] or bucket_pal["stem"] or bucket_pal["bloom"][:1]
        stem = bucket_pal["stem"] or foliage
        # A conifer or a tuft has no bloom: its "bloom" pixels are the snow and sky
        # inside the mask, so the foliage palette stands in for every part.
        bloom = (bucket_pal["bloom"] or foliage) if form in ("spike", "raceme", "seedhead", "shrub") else foliage
        return {
            "id": key, "form": form, "height_m": round(height, 3), "width_m": round(width, 3),
            "bloom_rgb": bloom, "foliage_rgb": foliage, "stem_rgb": stem,
            "density": str(species.get("density") or "patchy"), "variants": int(variants), "seed": int(seed),
        }

    @staticmethod
    def _entry_hash(entry: dict) -> str:
        return hashlib.sha256(json.dumps(entry, sort_keys=True).encode()).hexdigest()[:16]

    def _build(self, entries: list[dict]) -> dict[str, list[Path]]:
        """id -> GLB paths, building what the cache lacks in one Blender run."""
        out: dict[str, list[Path]] = {}
        todo = []
        for e in entries:
            d = self.config.cache_dir / self._entry_hash(e)
            paths = [d / f"{e['id']}_v{k}.glb" for k in range(e["variants"])]
            if all(p.exists() for p in paths):
                out[e["id"]] = paths
            else:
                todo.append((e, d))
        if todo:
            if not self.config.blender_path.exists():
                self.log_warning(f"Blender not found at {self.config.blender_path}; species meshes skipped")
                return out
            if not BUILDER.exists():
                self.log_warning(f"Species builder missing at {BUILDER}; species meshes skipped")
                return out
            work = (self.temp or self.output) / "build"
            work.mkdir(parents=True, exist_ok=True)
            spec = work / "spec.json"
            spec.write_text(json.dumps({"species": [e for e, _ in todo]}, indent=1))
            cmd = [str(self.config.blender_path), "--background", "--factory-startup", "--python", str(BUILDER),
                   "--", str(spec), str(work)]
            self.log_info(f"Blender: building {len(todo)} species mesh(es)")
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.config.timeout_s)
            except subprocess.TimeoutExpired:
                self.log_warning(f"Blender timed out after {self.config.timeout_s}s; species meshes skipped")
                return out
            (work / "blender.log").write_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
            if proc.returncode != 0:
                tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-6:])
                self.log_warning(f"Blender failed ({proc.returncode}): {tail}")
                return out
            for e, d in todo:
                d.mkdir(parents=True, exist_ok=True)
                paths = []
                for k in range(e["variants"]):
                    src = work / f"{e['id']}_v{k}.glb"
                    if src.exists():
                        dst = d / src.name
                        dst.write_bytes(src.read_bytes())
                        paths.append(dst)
                if paths:
                    (d / "spec.json").write_text(json.dumps(e, indent=1))
                    out[e["id"]] = paths
                else:
                    self.log_warning(f"Blender produced no GLB for {e['id']}")
        return out

    # ── stage ───────────────────────────────────────────────────────────────
    def run(self, context: PipelineContext) -> PipelineContext:
        plan = ScenePlan.from_context(context)
        published: dict[str, dict] = {}
        if plan is None:
            self.log_info("No scene plan; species layer skipped")
            context.add_object(ContextKey.SPECIES_MESHES, published)
            context.add_object(MARKER, True)
            return context
        photo_img = context.input_image(ContextKey.INPUT)
        photo = photo_img.image if photo_img is not None else None
        species = [s for s in plan.items("species", self.config.min_evidence)
                   if str(s.get("class", "")).lower() in ("flower", "tree", "shrub")]
        if not species:
            self.log_info("The plan names no visible flower, tree or shrub species; species layer skipped")
            context.add_object(ContextKey.SPECIES_MESHES, published)
            context.add_object(MARKER, True)
            return context
        candidates: dict[str, list[tuple[dict, dict, str]]] = {}
        for sp in species:
            if str(sp.get("class", "")).lower() == "tree" and species_form(sp) != "conifer":
                # No procedural form for a broadleaf tree yet (Paris's plane trees and
                # willows); its buckets keep their cards.
                self.log_info(f"  plan species '{sp.get('name')}': broadleaf tree, no procedural form yet; skipped")
                continue
            pal, src = self._species_palette(photo, sp)
            cls = {"shrub": "bush"}.get(str(sp.get("class")).lower(), str(sp.get("class")).lower())
            candidates.setdefault(cls, []).append((sp, pal, src))
            self.log_info(f"  plan species '{sp.get('name')}' ({species_form(sp)}, {sp.get('size_m')} m): palette from {src}"
                          + (f", bloom {pal['bloom'][:2]}" if pal and pal.get("bloom") else ""))

        groups = self._bucket_members(context)
        if not groups:
            self.log_info("No bucketed flower/tree detections; species layer skipped")
            context.add_object(ContextKey.SPECIES_MESHES, published)
            context.add_object(MARKER, True)
            return context
        task = self.create_progress(len(groups) + 1, "Species assets…")
        entries: list[dict] = []
        matches: dict[str, tuple[dict, dict, str, Optional[float], list[int]]] = {}
        for (cls, bucket), members in sorted(groups.items()):
            key = f"category_mesh_{cls}_{bucket}"
            pal = self._bucket_palette(context, members, cls)
            sp, de, how = self._match(cls, pal, candidates.get(cls, []))
            if sp is None:
                self.log_info(f"  {key} ({len(members)} members): keeps its card -- {how}")
                self.advance_progress(task)
                continue
            self.log_info(f"  {key} ({len(members)} members): '{sp.get('name')}' by {how}; "
                          f"bloom {pal['bloom'][:2]} foliage {pal['foliage'][:1]} ({pal['pixels']} px)")
            entry = self._spec_entry(key, sp, pal, seed=self.seed + bucket * 17 + (0 if cls == "flower" else 101),
                                     variants=self.config.variants if cls == "flower" else self.config.tree_variants)
            entries.append(entry)
            matches[key] = (sp, pal, how, de, members)
            self.advance_progress(task)

        built = self._build(entries)
        for entry in entries:
            key = entry["id"]
            paths = built.get(key)
            if not paths:
                continue
            meshes = []
            for k, path in enumerate(paths):
                try:
                    meshes.append(Mesh.load(path))
                except Exception as exc:
                    self.log_warning(f"  {key}: could not load {path.name}: {exc}")
            if not meshes:
                continue
            mesh = meshes[0]
            ext = mesh.extents
            sp, pal, how, de, members = matches[key]
            context.add_mesh(key, mesh)
            published[key] = {
                "species": sp.get("name"), "phrase": sp.get("phrase"), "form": entry["form"],
                "height_m": entry["height_m"], "match": how, "delta_e": de, "members": len(members),
                "palette": {"bloom": pal["bloom"], "foliage": pal["foliage"], "stem": pal["stem"]},
                "files": [str(p) for p in paths], "faces": mesh.face_count, "variants": len(meshes),
            }
            # Variants k >= 1 under their own keys; Scene Generation spreads a bucket's
            # instances across them (one plant per mesh, so one shape per meadow is
            # what the judge called "uniform sprite-like blobs").
            for k in range(1, len(meshes)):
                vkey = f"{key}_v{k}"
                context.add_mesh(vkey, meshes[k])
                published[vkey] = {"variant_of": key, "variant": k, "faces": meshes[k].face_count}
            self.log_info(f"  {key}: {entry['form']} {ext[0]:.2f}x{ext[1]:.2f}x{ext[2]:.2f} m, {mesh.face_count} faces, "
                          f"{len(meshes)} variant(s)")
        self.finish_progress(task)
        n_base = sum(1 for v in published.values() if "variant_of" not in v)
        self.log_info(f"Species layer: {n_base} bucket mesh(es) replaced ({len(published) - n_base} extra variants), "
                      f"{len(groups) - n_base} keep their cards")
        context.add_object(ContextKey.SPECIES_MESHES, published)
        context.add_object(MARKER, True)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(MARKER) is True
