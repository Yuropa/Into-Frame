import json
import numpy as np
import torch
from logging import Logger
from typing import Any

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.model_generation.model_generation import ModelGenerator, ModelGeneratorType
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import ENVIRONMENT_CATEGORIES as _ENV_CATEGORIES, CategoryFilter
from util.device_utils import DeviceStrategy, preferred_device


class PanoramaAssetGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        billboard_distance_m: float = 10.0,
        generator_type: str = "TRELLIS",
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        score_weight_confidence: float = 0.35,
        score_weight_fill_ratio: float = 0.25,
        score_weight_depth: float = 0.25,
        score_weight_occlusion: float = 0.6,
        occlusion_covered_fraction_threshold: float = 0.35,
        occlusion_disqualify_fraction: float = 0.6,
        occlusion_depth_margin: float = 0.10,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.billboard_distance_m = float(billboard_distance_m)
        self.generator_type = ModelGeneratorType[generator_type.upper()]
        self.category_filter = CategoryFilter(include_categories, exclude_categories)
        self.score_weight_confidence = score_weight_confidence
        self.score_weight_fill_ratio = score_weight_fill_ratio
        self.score_weight_depth = score_weight_depth
        self.score_weight_occlusion = score_weight_occlusion
        self.occlusion_covered_fraction_threshold = occlusion_covered_fraction_threshold
        self.occlusion_disqualify_fraction = occlusion_disqualify_fraction
        self.occlusion_depth_margin = occlusion_depth_margin


class PanoramaAssetGenerationStage(PipelineStage):
    """
    For each object category present in the scene, meshifies one representative
    crop (the closest instance to the camera) and stores it as category_mesh_{class}.
    Categories where every instance is farther than billboard_distance_m are left
    as billboards; SceneGenerationStage will draw from the category's crop pool for
    those.

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i} (with 'class' and 'box'),
            crop_{i}, ContextKey.PANORAMA_OBJECT_DEPTH (depth on the ORIGINAL panorama,
            matching what objects were detected against), ContextKey.PANORAMA
    Writes: category_mesh_{class} for each qualifying category
    Config: billboard_distance_m (default 10.0 m), generator_type (default TRELLIS)
    """

    @classmethod
    def config_class(cls):
        return PanoramaAssetGenerationConfiguration

    def __init__(self, config: PanoramaAssetGenerationConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to process, skipping")
            return context

        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None

        category_best, skipped_debug, billboard_debug, disqualified_debug = self._select_representatives(
            object_count, context.input_object, context.input_image, panorama_depth, pano_w, pano_h,
        )

        self._write_debug(skipped_debug, billboard_debug, category_best, disqualified_debug)

        if not category_best:
            self.log_info("No objects within 3D generation distance")
            return context

        # Second pass: generate one mesh per qualifying category.
        asset_task = self.create_progress(len(category_best), "Generating 3D assets…")
        super().clean_up()
        gen = ModelGenerator(self.preferred_device, type=self.config.generator_type)

        try:
            for obj_class, (idx, depth, score) in category_best.items():
                mesh_key = f"category_mesh_{obj_class}"

                cached = context.mesh(mesh_key)
                if cached is not None:
                    self.log_info(f"  {mesh_key}: cached ({cached.vertex_count}v {cached.face_count}f)")
                    self.advance_progress(asset_task)
                    continue

                self.log_info(f"  {mesh_key}: {depth:.1f} m, score {score:.2f} → 3D mesh (crop_{idx})")
                crop = context.input_image(f"crop_{idx}")
                temp_path = self.temp / mesh_key if self.temp is not None else None
                if temp_path is not None:
                    temp_path.mkdir(parents=True, exist_ok=True)
                super().clean_up()
                mesh = gen.meshify(crop, temp_path, seed=self.seed)
                mesh = mesh.repair()
                mesh.fit_to_box(1.0, 1.0)
                context.add_mesh(mesh_key, mesh)
                self.log_info(f"  {mesh_key}: {mesh.vertex_count}v {mesh.face_count}f")
                self.advance_progress(asset_task)
        finally:
            gen.close()

        self.finish_progress(asset_task)
        return context

    def _select_representatives(
        self, object_count: int, get_metadata, get_image, panorama_depth, pano_w, pano_h,
    ) -> tuple[dict[str, tuple[int, float, float]], list, list, list]:
        """Shared by run() and has_expected_output() (callers pass either the
        input_* accessors to see state as of the previous stage, or the plain
        accessors to see this stage's own already-cached output).

        Returns (category_best, skipped_debug, billboard_debug, disqualified_debug).
        category_best: class -> (idx, depth, score) of the winning representative.
        A candidate whose box is heavily covered by another, nearer instance
        (any class) is disqualified in favor of the next-best-scoring
        candidate in that category; if every candidate for a category is
        disqualified, the least-bad one is kept anyway rather than silently
        dropping the category.
        """
        threshold = self.config.billboard_distance_m
        skipped_debug = []
        billboard_debug = []
        depth_by_idx: dict[int, tuple[list, float]] = {}
        candidates_by_class: dict[str, list[dict]] = {}

        for idx in range(object_count):
            metadata = get_metadata(f"metadata_{idx}")
            if metadata is None:
                continue

            box = metadata.get("box")
            depth = self._sample_object_depth(box, panorama_depth, pano_w, pano_h)
            if box is not None and depth is not None:
                depth_by_idx[idx] = (box, depth)

            obj_class = metadata.get("class")
            if obj_class in _ENV_CATEGORIES or obj_class == "indeterminate":
                skipped_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "reason": "environment" if obj_class in _ENV_CATEGORIES else "indeterminate",
                })
                continue
            if not self.config.category_filter.allows(obj_class or ""):
                skipped_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "reason": "category_filter",
                })
                continue

            if depth is None or depth >= threshold:
                billboard_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "depth_m": round(depth, 2) if depth is not None else None,
                    "threshold_m": threshold,
                })
                continue

            candidates_by_class.setdefault(obj_class, []).append({
                "idx": idx, "box": box, "depth": depth, "metadata": metadata,
            })

        category_best: dict[str, tuple[int, float, float]] = {}
        disqualified_debug = []
        for obj_class, candidates in candidates_by_class.items():
            scored = []
            for candidate in candidates:
                occlusion = self._occlusion_score(candidate["idx"], candidate["box"], candidate["depth"], depth_by_idx)
                crop = get_image(f"crop_{candidate['idx']}")
                score = self._composite_score(candidate["metadata"], crop, candidate["depth"], occlusion, threshold)
                disqualified = occlusion >= self.config.occlusion_disqualify_fraction
                scored.append({**candidate, "occlusion": occlusion, "score": score, "disqualified": disqualified})

            eligible = [s for s in scored if not s["disqualified"]] or scored
            winner = max(eligible, key=lambda s: s["score"])
            category_best[obj_class] = (winner["idx"], winner["depth"], winner["score"])
            for s in scored:
                if s["disqualified"]:
                    disqualified_debug.append({
                        "idx": s["idx"],
                        "class": obj_class,
                        "occlusion": round(s["occlusion"], 3),
                        "chosen_anyway": s["idx"] == winner["idx"],
                    })

        return category_best, skipped_debug, billboard_debug, disqualified_debug

    @staticmethod
    def _mask_fill_ratio(crop) -> float | None:
        """crop_i.image is already cropped tight to its own bbox, so fill
        ratio is just alpha-nonzero-px / (crop.width * crop.height) -- a
        proxy for 'clean single-object segmentation' vs a partial/broken crop."""
        if crop is None or crop.image.mode != "RGBA":
            return None
        alpha = np.asarray(crop.image.getchannel("A"))
        return float((alpha > 0).sum()) / float(alpha.size) if alpha.size else None

    @staticmethod
    def _covered_fraction(box_i: list[float], box_j: list[float]) -> float:
        """Fraction of box_i's area covered by box_j -- containment-style
        overlap, not symmetric IoU, since an occluder can be much larger or
        smaller than the thing it's occluding."""
        ax1, ay1 = box_i[0], box_i[1]
        ax2, ay2 = box_i[0] + box_i[2], box_i[1] + box_i[3]
        bx1, by1 = box_j[0], box_j[1]
        bx2, by2 = box_j[0] + box_j[2], box_j[1] + box_j[3]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        area_i = box_i[2] * box_i[3]
        return inter / area_i if area_i > 0 else 0.0

    def _occlusion_score(self, idx: int, box, depth: float, depth_by_idx: dict[int, tuple[list, float]]) -> float:
        """Max covered_fraction among all OTHER instances (any class, any
        filter status -- an occluder that's itself excluded from asset
        generation, e.g. a bush, still visually cuts off what's behind it)
        whose sampled depth is nearer than this candidate's by more than
        occlusion_depth_margin."""
        if box is None:
            return 0.0
        best = 0.0
        for j, (jbox, jdepth) in depth_by_idx.items():
            if j == idx:
                continue
            if jdepth < depth * (1.0 - self.config.occlusion_depth_margin):
                best = max(best, self._covered_fraction(box, jbox))
        return best

    def _composite_score(self, metadata, crop, depth: float, occlusion: float, threshold: float) -> float:
        cfg = self.config
        confidence = metadata.get("confidence", 0.5)
        fill_ratio = self._mask_fill_ratio(crop)
        fill_ratio = fill_ratio if fill_ratio is not None else 0.5
        depth_score = max(0.0, 1.0 - depth / threshold) if threshold else 0.0
        occlusion_penalty = occlusion if occlusion >= cfg.occlusion_covered_fraction_threshold else 0.0
        return (
            cfg.score_weight_confidence * confidence
            + cfg.score_weight_fill_ratio * fill_ratio
            + cfg.score_weight_depth * depth_score
            - cfg.score_weight_occlusion * occlusion_penalty
        )

    def _write_debug(self, skipped: list, billboards: list, category_best: dict, disqualified: list):
        if self.output is None:
            return
        payload = {
            "billboard_distance_m": self.config.billboard_distance_m,
            "summary": {
                "skipped_env_or_filtered": len(skipped),
                "billboard_only": len(billboards),
                "categories_meshified": len(category_best),
            },
            "skipped": skipped,
            "billboard": billboards,
            "occlusion_disqualified": disqualified,
            "categories": [
                {"class": cls, "representative_idx": idx, "depth_m": round(depth, 2), "score": round(score, 3)}
                for cls, (idx, depth, score) in category_best.items()
            ],
        }
        with open(self.output / "asset_debug.json", "w") as f:
            json.dump(payload, f, indent=2)

    def _sample_object_depth(self, box, panorama_depth, pano_w, pano_h) -> float | None:
        """Sample median depth in a patch around the bbox centre in the panorama depth map."""
        if box is None or panorama_depth is None or pano_w is None or pano_h is None:
            return None

        bx, by, bw, bh = box
        cx = bx + bw / 2.0
        cy = by + bh / 2.0

        sx = panorama_depth.width / pano_w
        sy = panorama_depth.height / pano_h
        dx = int(round(cx * sx))
        dy = int(round(cy * sy))

        r = 5
        x1 = max(0, dx - r)
        x2 = min(panorama_depth.width, dx + r)
        y1 = max(0, dy - r)
        y2 = min(panorama_depth.height, dy + r)

        patch = panorama_depth.depth[y1:y2, x1:x2]
        valid = patch[(patch > 0) & np.isfinite(patch)]
        return float(np.median(valid)) if len(valid) > 0 else None

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            # No OBJECT_COUNT anywhere upstream means Object Segmentation is
            # disabled (permanent, not "pending") -- nothing to generate
            # assets for and never will be. Matches what count == 0 would
            # already return below (the loop is simply skipped); treating
            # None differently forced this stage, and everything after it via
            # the dirty cascade, to rerun on every single invocation.
            return True
        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None

        category_best, _, _, _ = self._select_representatives(
            count, context.object, context.image, panorama_depth, pano_w, pano_h,
        )
        for obj_class in category_best:
            if context.mesh(f"category_mesh_{obj_class}") is None:
                return False
        return True

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names(type=self.config.generator_type)

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        count = context.object(ContextKey.OBJECT_COUNT)
        if count is None or count == 0:
            return None

        seen_classes: set[str] = set()
        images = []
        total_verts = 0
        total_faces = 0

        for i in range(count):
            meta = context.object(f"metadata_{i}") or {}
            obj_class = meta.get("class", f"Object {i + 1}")
            mesh_key = f"category_mesh_{obj_class}"
            mesh = context.mesh(mesh_key)
            crop = context.image(f"crop_{i}")

            if obj_class not in seen_classes:
                seen_classes.add(obj_class)
                if mesh is not None:
                    total_verts += mesh.vertex_count
                    total_faces += mesh.face_count
                if crop is not None and len(images) < 6:
                    label = obj_class
                    if mesh is not None:
                        label += f" ({mesh.vertex_count:,}v)"
                    images.append((crop.image, label))

        reconstructed = len(seen_classes)
        stats = {"Categories reconstructed": str(reconstructed)}
        if reconstructed > 0:
            stats["Total vertices"] = f"{total_verts:,}"
            stats["Total triangles"] = f"{total_faces:,}"
            stats["Generator"] = self.config.generator_type.name
        return ReportSection(
            stage_name=self.name,
            title="3D Object Reconstruction",
            body=(
                "One 3D mesh is generated per object category using the closest "
                f"instance as the representative crop. The {self.config.generator_type.name} "
                "model reconstructs a textured mesh, which is normalised to a 1 m "
                "canonical box. At scene placement each instance randomly uses the "
                "category mesh (with a random rotation) or a billboard drawn from "
                "the category's crop pool."
            ),
            images=images,
            stats=stats,
        )

    def clean_up(self):
        super().clean_up()
