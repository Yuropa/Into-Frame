"""
Tree Mesh Refinement: volumetric reconstructions for the tree-like category
meshes.

Panorama Asset Generation reconstructs one mesh per (class, visual-variant)
group with SAM3D. For compact, mostly convex things -- a rock, a boat, a bench
-- a single-view reconstructor does well. For a tree it does not: there is no
closed surface to recover, and on every 2026-10-13 run the tree and bush groups
came back as a textured sheet a few centimetres thick, which the scene's shape
gates then refused, so every tree in the scene was a card.

Tree-D Fusion (Lee et al., ECCV 2024) is a reconstructor built for exactly this
input: Magic123's SDS optimisation of a NeRF, with a Zero123 fine-tuned on
trees as the novel-view prior, so the unseen sides are filled with canopy rather
than extruded from the silhouette. It is slow (minutes per tree, not seconds),
so it is not the default generator; it runs here, after Asset Generation, for
the few groups where it matters, and publishes its mesh under the SAME
category_mesh_{class}_{bucket} key. Later stages read the context newest-first,
so Scene Generation and Category Mesh Rigging pick up the volumetric mesh and
SAM3D's sheet stays as the fallback when this stage fails or is disabled.

Its picture of the tree is the best one the pipeline has: the delit, completed
(ObjectCompletionStage) and super-resolved crop Asset Generation itself built
from, in that order of preference.

Reads:  "mesh_sources" (Panorama Asset Generation: which crop each category mesh
        was reconstructed from), the crop images for those indices
Writes: category_mesh_{class}_{bucket} for every refined group,
        "tree_mesh_refinement_summary"
"""
from __future__ import annotations

import math
import tempfile
from logging import Logger
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext
from pipeline.object_typing.categories import CategoryFilter, normalize_category
from pipeline.panorama_asset_generation.generation import MESH_SOURCES_KEY, _delit_key, _SR_KEY, _COMPLETE_KEY
from pipeline.segmentation.hires_crops import prefer_hires
from pipeline.tree_generation.model_generation_treed import ModelGeneratorTreeD
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image

REFINEMENT_SUMMARY_KEY = "tree_mesh_refinement_summary"


class TreeMeshGenerationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        max_meshes: int = 6,
        iterations: int = 2000,
        train_resolution: int = 64,
        mcubes_resolution: int = 128,
        max_mesh_faces: int = 8000,
        repair_meshes: bool = False,
        max_aspect_disagreement: float = 1.6,
        min_class_likeness: float = 0.5,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Which classes to refine. The Zero123 prior was fine-tuned on trees, so by
        # default only the groups with a trunk-and-canopy structure go through it;
        # flowers, plants and everything manufactured keep SAM3D.
        if include_categories is None and exclude_categories is None:
            include_categories = ["tree", "bush"]
        self.category_filter = CategoryFilter(include_categories, exclude_categories)
        # Cap on reconstructions per run, nearest groups first -- each one is a full
        # SDS optimisation (Magic123, ~2000 iterations).
        self.max_meshes = int(max_meshes)
        self.iterations = int(iterations)
        # NeRF training render size and marching-cubes grid, Tree-D Fusion's own recipe.
        self.train_resolution = int(train_resolution)
        self.mcubes_resolution = int(mcubes_resolution)
        # Same post-processing as Panorama Asset Generation applies to its meshes.
        self.max_mesh_faces = int(max_mesh_faces)
        self.repair_meshes = bool(repair_meshes)
        # Tree-D Fusion can collapse a slender conifer into a round lump (Rainier,
        # 2026-10-16: crops 0.3-0.4 wide:tall came back 0.9-1.0). A result whose
        # width:height disagrees with the crop's silhouette by more than this factor
        # is discarded and the Asset Generation mesh kept.
        self.max_aspect_disagreement = float(max_aspect_disagreement)
        # CLIP class-likeness floor for Tree-D's output (scene.mesh_check); 0 disables.
        self.min_class_likeness = float(min_class_likeness)


class TreeMeshGenerationStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return TreeMeshGenerationConfiguration

    def __init__(self, config: TreeMeshGenerationConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)
        self._gen: ModelGeneratorTreeD | None = None

    @staticmethod
    def _source(context: PipelineContext, idx: int):
        """The best picture of crop `idx`: delit (made from the completed or
        super-resolved crop when one exists), else those, else the hi-res cut,
        else the panorama crop."""
        for key in (_delit_key(idx), _SR_KEY.format(idx=idx), _COMPLETE_KEY.format(idx=idx)):
            image = context.input_image(key)
            if image is not None:
                return image, key
        image = prefer_hires(context, idx)
        if image is not None:
            return image, "hires"
        return context.input_image(f"crop_{idx}"), f"crop_{idx}"

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: TreeMeshGenerationConfiguration = self.config
        sources = context.input_object(MESH_SOURCES_KEY) or {}
        jobs = []
        for mesh_key, src in sources.items():
            obj_class = normalize_category(src.get("class"))
            if not obj_class or not cfg.category_filter.allows(obj_class):
                continue
            depth = src.get("depth")
            jobs.append((depth if depth is not None else math.inf, mesh_key, int(src["idx"]), obj_class))
        jobs.sort()
        skipped = max(0, len(jobs) - cfg.max_meshes)
        jobs = jobs[: cfg.max_meshes]
        if not jobs:
            self.log_info("Tree mesh refinement: no tree-like category meshes to refine")
            context.add_object(REFINEMENT_SUMMARY_KEY, {"refined": 0})
            return context
        self.log_info(
            f"Tree mesh refinement: {len(jobs)} group(s) → Tree-D Fusion"
            + (f", {skipped} farther group(s) over max_meshes={cfg.max_meshes} keep their SAM3D mesh" if skipped else "")
        )

        task = self.create_progress(len(jobs), "Refining tree meshes…")
        super().clean_up()
        self._gen = ModelGeneratorTreeD(self.preferred_device)
        refined, failed = [], []
        scratch = None
        try:
            for depth, mesh_key, idx, obj_class in jobs:
                image, source_key = self._source(context, idx)
                if image is None:
                    self.log_info(f"  {mesh_key}: no image for crop_{idx}, keeping the existing mesh")
                    self.advance_progress(task)
                    continue
                pil = image.image.convert("RGBA")
                if self.temp is not None:
                    temp_path = self.temp / mesh_key
                else:
                    scratch = scratch or tempfile.TemporaryDirectory(prefix="treed-")
                    temp_path = Path(scratch.name) / mesh_key
                temp_path.mkdir(parents=True, exist_ok=True)
                self.log_info(
                    f"  {mesh_key}: {depth:.1f} m, {pil.width}x{pil.height} px from {source_key} "
                    f"→ Tree-D Fusion ({cfg.iterations} it, log {temp_path / 'magic123.log'})"
                )
                super().clean_up()
                try:
                    mesh = self._gen.meshify(
                        Image(pil), temp_path, seed=self.seed, species=obj_class,
                        iterations=cfg.iterations, train_resolution=cfg.train_resolution,
                        mcubes_resolution=cfg.mcubes_resolution,
                    )
                    if cfg.repair_meshes or not mesh.has_texture:
                        mesh = mesh.repair()
                    raw_faces = mesh.face_count
                    mesh = mesh.decimate(cfg.max_mesh_faces)
                    if not mesh.has_texture:
                        # The client ignores vertex colours; give it a texture.
                        mesh = mesh.bake_vertex_colours()
                    mesh.fit_to_box(1.0, 1.0)
                    # Tree-D's textures come back dark like SAM3D's (Paris
                    # 2026-10-16: near-black canopies); match the crop it came from.
                    matched = mesh.match_texture_colour(pil)
                    if matched is not None:
                        self.log_info(f"  {mesh_key}: texture mean {np.round(matched[0]).astype(int).tolist()} "
                                      f"-> {np.round(matched[1]).astype(int).tolist()}")
                    crop_aspect = self._silhouette_aspect(pil)
                    ext = mesh.extents
                    mesh_aspect = float(max(ext[0], ext[2]) / max(ext[1], 1e-6))
                    off = max(mesh_aspect, crop_aspect) / max(min(mesh_aspect, crop_aspect), 1e-6)
                    if off > cfg.max_aspect_disagreement:
                        raise ValueError(
                            f"width:height {mesh_aspect:.2f} disagrees with the crop's {crop_aspect:.2f} "
                            f"by {off:.1f}x (over {cfg.max_aspect_disagreement:.1f}x)"
                        )
                    # Rainier 2026-10-17: a flat yellow sheet passed the aspect test.
                    from pipeline.panorama_asset_generation.generation import _check_class_likeness
                    _check_class_likeness(mesh, "tree", cfg.min_class_likeness, self.log_info, mesh_key)
                except Exception as exc:
                    self.log_warning(f"  {mesh_key}: Tree-D Fusion failed ({exc}); keeping the existing mesh")
                    failed.append(mesh_key)
                    self.advance_progress(task)
                    continue
                context.add_mesh(mesh_key, mesh)
                refined.append(mesh_key)
                ext = mesh.extents
                self.log_info(
                    f"  {mesh_key}: {mesh.vertex_count}v {mesh.face_count}f"
                    + (f" (decimated from {raw_faces}f)" if mesh.face_count < raw_faces else "")
                    + (" textured" if mesh.has_texture else " vertex-colour only")
                    + f", extents {ext[0]:.2f} x {ext[1]:.2f} x {ext[2]:.2f}"
                )
                self.advance_progress(task)
        finally:
            self._gen.close()
            self._gen = None
            if scratch is not None:
                scratch.cleanup()
        self.finish_progress(task)
        context.add_object(REFINEMENT_SUMMARY_KEY, {"refined": len(refined), "failed": failed, "keys": refined})
        self.log_info(f"Tree mesh refinement: {len(refined)} mesh(es) replaced, {len(failed)} failed")
        return context

    @staticmethod
    def _silhouette_aspect(pil) -> float:
        """Width:height of the crop's opaque silhouette (the whole image if it has none)."""
        alpha = np.asarray(pil.getchannel("A")) > 127
        if not alpha.any():
            return pil.width / max(pil.height, 1)
        rows, cols = np.where(alpha)
        return (cols.max() - cols.min() + 1) / max(rows.max() - rows.min() + 1, 1)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(REFINEMENT_SUMMARY_KEY) is not None

    def model_names(self) -> list[str]:
        return []

    def clean_up(self):
        if self._gen is not None:
            self._gen.close()
            self._gen = None
        super().clean_up()
