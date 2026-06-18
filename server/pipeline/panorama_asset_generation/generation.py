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
        lod_max_error_fraction: float = 0.03,
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.billboard_distance_m = float(billboard_distance_m)
        self.generator_type = ModelGeneratorType[generator_type.upper()]
        self.lod_max_error_fraction = float(lod_max_error_fraction)
        self.category_filter = CategoryFilter(include_categories, exclude_categories)


class PanoramaAssetGenerationStage(PipelineStage):
    """
    For each non-environment panorama object (as classified by
    PanoramaObjectClassificationStage), samples the object's depth from
    PANORAMA_DEPTH and decides:
      - depth < billboard_distance_m  → generate a 3D mesh (writes mesh_{i})
      - depth >= billboard_distance_m → leave as billboard (no mesh written)

    SceneGenerationStage already falls back to billboard when mesh_{i} is absent,
    so no changes to that stage are needed for the far-object path.

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i} (with 'class'),
            crop_{i}, ContextKey.PANORAMA_DEPTH, ContextKey.PANORAMA
    Writes: mesh_{i} for objects closer than billboard_distance_m
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

        panorama_depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        threshold = self.config.billboard_distance_m

        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None

        # First pass: decide which objects need 3D generation
        near_indices = []
        skipped_debug = []
        billboard_debug = []
        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}")
            if metadata is None:
                continue
            obj_class = metadata.get("class")
            if obj_class in _ENV_CATEGORIES or obj_class == "indeterminate":
                self.log_info(f"  crop_{idx}: {obj_class} — skipped")
                skipped_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "caption": metadata.get("caption", ""),
                    "confidence": metadata.get("confidence"),
                    "reason": "environment" if obj_class in _ENV_CATEGORIES else "indeterminate",
                })
                continue
            if not self.config.category_filter.allows(obj_class or ""):
                self.log_info(f"  crop_{idx}: '{obj_class}' — excluded by category filter")
                skipped_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "caption": metadata.get("caption", ""),
                    "confidence": metadata.get("confidence"),
                    "reason": "category_filter",
                })
                continue
            if context.input_mesh(f"mesh_{idx}") is not None:
                self.log_info(f"  crop_{idx}: mesh already cached")
                continue

            depth = self._sample_object_depth(metadata.get("box"), panorama_depth, pano_w, pano_h)
            if depth is not None and depth < threshold:
                near_indices.append((idx, depth))
            else:
                label = f"{depth:.1f} m" if depth is not None else "unknown depth"
                self.log_info(f"  crop_{idx}: {label} → billboard")
                billboard_debug.append({
                    "idx": idx,
                    "class": metadata.get("class"),
                    "depth_m": round(depth, 2) if depth is not None else None,
                    "threshold_m": threshold,
                })

        self._write_debug(skipped_debug, billboard_debug, near_indices)

        if not near_indices:
            self.log_info("No objects within 3D generation distance")
            return context

        # Second pass: generate meshes for near objects
        asset_task = self.create_progress(len(near_indices), "Generating 3D assets…")
        super().clean_up()
        gen = ModelGenerator(self.preferred_device, type=self.config.generator_type)

        for idx, depth in near_indices:
            self.log_info(f"  crop_{idx}: {depth:.1f} m → 3D mesh")
            crop = context.input_image(f"crop_{idx}")
            temp_path = self.temp / f"crop_{idx}" if self.temp is not None else None
            super().clean_up()
            mesh = gen.meshify(crop, temp_path, seed=self.seed)
            mesh = mesh.repair()
            context.add_mesh(f"mesh_{idx}", mesh)

            try:
                lod = mesh.simplify(max_error_fraction=self.config.lod_max_error_fraction)
                if crop is not None:
                    lod.apply_crop_texture(crop.rgba())
                context.add_mesh(f"mesh_lod_{idx}", lod)
                self.log_info(f"  crop_{idx}: LOD {mesh.face_count} → {lod.face_count} faces")
            except Exception as e:
                self.log_info(f"  crop_{idx}: LOD generation failed ({e}), skipping")

            self.advance_progress(asset_task)

        gen.close()
        self.finish_progress(asset_task)
        return context

    def _write_debug(self, skipped: list, billboards: list, near: list):
        if self.output is None:
            return
        payload = {
            "billboard_distance_m": self.config.billboard_distance_m,
            "summary": {
                "skipped_env_or_indeterminate": len(skipped),
                "billboard": len(billboards),
                "mesh_3d": len(near),
            },
            "skipped": skipped,
            "billboard": billboards,
            "mesh_3d": [{"idx": idx, "depth_m": round(depth, 2)} for idx, depth in near],
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

        # Scale from panorama pixel space to depth map pixel space
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
        panorama_depth = context.input_depth(ContextKey.PANORAMA_DEPTH)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None
        threshold = self.config.billboard_distance_m

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
                if context.mesh(f"mesh_{idx}") is None:
                    return False
                if context.mesh(f"mesh_lod_{idx}") is None:
                    return False
        return True

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names(type=self.config.generator_type)

    def clean_up(self):
        super().clean_up()
