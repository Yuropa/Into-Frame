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
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.billboard_distance_m = float(billboard_distance_m)
        self.generator_type = ModelGeneratorType[generator_type.upper()]
        self.category_filter = CategoryFilter(include_categories, exclude_categories)


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
        threshold = self.config.billboard_distance_m

        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None

        # First pass: for each category, find the closest instance within threshold.
        # That instance's crop becomes the representative mesh for the whole category.
        category_best: dict[str, tuple[int, float]] = {}  # class → (idx, depth)
        skipped_debug = []
        billboard_debug = []

        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}")
            if metadata is None:
                continue
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

            depth = self._sample_object_depth(metadata.get("box"), panorama_depth, pano_w, pano_h)
            if depth is None or depth >= threshold:
                billboard_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "depth_m": round(depth, 2) if depth is not None else None,
                    "threshold_m": threshold,
                })
                continue

            # Keep the closest instance as the category representative.
            if obj_class not in category_best or depth < category_best[obj_class][1]:
                category_best[obj_class] = (idx, depth)

        self._write_debug(skipped_debug, billboard_debug, category_best)

        if not category_best:
            self.log_info("No objects within 3D generation distance")
            return context

        # Second pass: generate one mesh per qualifying category.
        asset_task = self.create_progress(len(category_best), "Generating 3D assets…")
        super().clean_up()
        gen = ModelGenerator(self.preferred_device, type=self.config.generator_type)

        try:
            for obj_class, (idx, depth) in category_best.items():
                mesh_key = f"category_mesh_{obj_class}"

                cached = context.mesh(mesh_key)
                if cached is not None:
                    self.log_info(f"  {mesh_key}: cached ({cached.vertex_count}v {cached.face_count}f)")
                    self.advance_progress(asset_task)
                    continue

                self.log_info(f"  {mesh_key}: {depth:.1f} m → 3D mesh (crop_{idx})")
                crop = context.input_image(f"crop_{idx}")
                temp_path = self.temp / mesh_key if self.temp is not None else None
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

    def _write_debug(self, skipped: list, billboards: list, category_best: dict):
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
            "categories": [
                {"class": cls, "representative_idx": idx, "depth_m": round(depth, 2)}
                for cls, (idx, depth) in category_best.items()
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
            return False
        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None
        threshold = self.config.billboard_distance_m

        seen_classes: set[str] = set()
        for idx in range(count):
            metadata = context.object(f"metadata_{idx}")
            if metadata is None:
                continue
            obj_class = metadata.get("class")
            if obj_class in _ENV_CATEGORIES or obj_class == "indeterminate":
                continue
            if not self.config.category_filter.allows(obj_class or ""):
                continue
            depth = self._sample_object_depth(
                metadata.get("box"), panorama_depth, pano_w, pano_h
            )
            if depth is not None and depth < threshold:
                if obj_class not in seen_classes:
                    seen_classes.add(obj_class)
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
