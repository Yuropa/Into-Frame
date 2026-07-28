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
from util.crop_scoring import composite_score, occlusion_score, mask_fill_ratio


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
        min_mesh_area_fraction: float = 0.001,
        generator_type: str = "TRELLIS",
        include_categories: list[str] | None = None,
        exclude_categories: list[str] | None = None,
        billboard_top_k: int = 4,
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
        # A group only earns a bespoke category mesh if its winning representative's
        # detection box covers at least this fraction of the panorama. Meshing is
        # otherwise gated on distance alone (billboard_distance_m), which lets a
        # tiny-but-close subject through -- e.g. foreground alpine-meadow flowers
        # sit 0.6-1 m from the camera yet each occupy only ~0.01-0.07% of the frame
        # (20-80 px), and, split into a bucket per colour, spawn a separate 3D mesh
        # apiece (observed: 7 flower meshes, several from singleton buckets, one
        # from a conf-0.03 "sheep in grass" miscrop). Bespoke meshes are meant for
        # prominent foreground subjects; anything below this stays billboard-only
        # (its pool is still curated, so it isn't dropped -- just not meshified).
        # 0 disables the size gate (distance-only, prior behaviour).
        self.min_mesh_area_fraction = float(min_mesh_area_fraction)
        self.generator_type = ModelGeneratorType[generator_type.upper()]
        self.category_filter = CategoryFilter(include_categories, exclude_categories)
        self.billboard_top_k = billboard_top_k
        self.score_weight_confidence = score_weight_confidence
        self.score_weight_fill_ratio = score_weight_fill_ratio
        self.score_weight_depth = score_weight_depth
        self.score_weight_occlusion = score_weight_occlusion
        self.occlusion_covered_fraction_threshold = occlusion_covered_fraction_threshold
        self.occlusion_disqualify_fraction = occlusion_disqualify_fraction
        self.occlusion_depth_margin = occlusion_depth_margin


class PanoramaAssetGenerationStage(PipelineStage):
    """
    For each (class, bucket) visual-similarity group present in the scene (see
    ObjectCategoryClusteringStage -- bucket sub-divides a class into visually
    distinct variants, e.g. flower colors), curates a top-K billboard crop pool
    (billboard_top_k, ranked by the same composite score regardless of
    distance -- SceneGenerationStage draws from this pool at ANY distance) and,
    if any instance of the group is closer than billboard_distance_m, meshifies
    its best-scoring eligible instance as category_mesh_{class}_{bucket}.
    Groups where every instance is farther than billboard_distance_m stay
    billboard-only.

    metadata_{i}['position_only'] (ObjectCategoryClusteringStage -- a low-confidence
    crop visually corroborated against some class, trusted only for its world
    position) is always excluded here: never a mesh representative, never in a
    billboard pool, regardless of its class/bucket/score. So is
    metadata_{i}['synthetic'] (DistributionSynthesisStage's painted points, which
    run before this stage and are already counted in OBJECT_COUNT) -- a painted
    point has no crop of its own and only ever consumes a pool, never supplies one.

    Reads:  ContextKey.OBJECT_COUNT, metadata_{i} (with 'class', 'bucket', 'box'),
            crop_{i}, ContextKey.PANORAMA_OBJECT_DEPTH (depth on the ORIGINAL panorama,
            matching what objects were detected against), ContextKey.PANORAMA
    Writes: category_mesh_{class}_{bucket} for each qualifying group,
            "billboard_pools" ({"{class}::{bucket}": [idx, ...]}) for every group
    Config: billboard_distance_m (default 10.0 m), billboard_top_k (default 4),
            min_mesh_area_fraction (default 0.001 -- a group whose winning box
            covers less of the panorama than this stays billboard-only),
            generator_type (default TRELLIS)
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

        group_best, billboard_pools, skipped_debug, disqualified_debug, synthetic_skipped = self._curate(
            object_count, context.input_object, context.input_image, panorama_depth, pano_w, pano_h,
        )

        context.add_object("billboard_pools", billboard_pools)
        self._write_debug(skipped_debug, group_best, disqualified_debug, billboard_pools, synthetic_skipped)

        if not group_best:
            self.log_info("No objects within 3D generation distance")
            return context

        # Second pass: generate one mesh per qualifying (class, bucket) group.
        asset_task = self.create_progress(len(group_best), "Generating 3D assets…")
        super().clean_up()
        gen = ModelGenerator(self.preferred_device, type=self.config.generator_type)

        try:
            for (obj_class, bucket), (idx, depth, score) in group_best.items():
                mesh_key = f"category_mesh_{obj_class}_{bucket}"

                cached = context.mesh(mesh_key)
                if cached is not None:
                    self.log_info(f"  {mesh_key}: cached ({cached.vertex_count}v {cached.face_count}f)")
                    self.advance_progress(asset_task)
                    continue

                self.log_info(f"  {mesh_key}: {depth:.1f} m, score {score:.2f} → 3D mesh (crop_{idx})")
                crop = context.input_image(f"crop_{idx}")

                fill_ratio = mask_fill_ratio(crop)
                if fill_ratio is not None and fill_ratio <= 0.0:
                    self.log_info(f"  {mesh_key}: crop_{idx} has an empty mask, skipping mesh (billboard-only)")
                    self.advance_progress(asset_task)
                    continue

                temp_path = self.temp / mesh_key if self.temp is not None else None
                if temp_path is not None:
                    temp_path.mkdir(parents=True, exist_ok=True)
                super().clean_up()
                try:
                    mesh = gen.meshify(crop, temp_path, seed=self.seed)
                    mesh = mesh.repair()
                    mesh.fit_to_box(1.0, 1.0)
                except Exception as e:
                    # A single degenerate crop (e.g. a near-empty mask the
                    # generator's own preprocessing collapses to nothing)
                    # shouldn't take down every other group's mesh --
                    # scene_generation.py already falls back to this group's
                    # billboard pool when category_mesh_{class}_{bucket} is
                    # absent.
                    self.log_info(f"  {mesh_key}: meshify failed ({e}), falling back to billboard-only")
                    self.advance_progress(asset_task)
                    continue
                context.add_mesh(mesh_key, mesh)
                self.log_info(f"  {mesh_key}: {mesh.vertex_count}v {mesh.face_count}f")
                self.advance_progress(asset_task)
        finally:
            gen.close()

        self.finish_progress(asset_task)
        return context

    def _curate(
        self, object_count: int, get_metadata, get_image, panorama_depth, pano_w, pano_h,
    ) -> tuple[dict[tuple[str, int], tuple[int, float, float]], dict[str, list[int]], list, list, int]:
        """Shared by run() and has_expected_output() (callers pass either the
        input_* accessors to see state as of the previous stage, or the plain
        accessors to see this stage's own already-cached output).

        Returns (group_best, billboard_pools, skipped_debug, disqualified_debug,
        synthetic_skipped).
        group_best: (class, bucket) -> (idx, depth, score) of the winning mesh
        representative, only for groups with at least one instance closer than
        billboard_distance_m. billboard_pools: "{class}::{bucket}" -> top-K
        crop indices by score, for EVERY (class, bucket) group regardless of
        distance -- a group's billboard pool must stay usable even when no
        instance qualified for meshing.
        """
        threshold = self.config.billboard_distance_m
        skipped_debug = []
        synthetic_skipped = 0
        depth_by_idx: dict[int, tuple[list, float]] = {}
        candidates_by_group: dict[tuple[str, int], list[dict]] = {}

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
            if metadata.get("position_only"):
                # ObjectCategoryClusteringStage visually corroborated this
                # low-confidence crop enough to trust its position for
                # ObjectDistributionStage, but not enough to anchor a bucket,
                # get meshed, or appear as a billboard from its own crop.
                skipped_debug.append({
                    "idx": idx,
                    "class": obj_class,
                    "reason": "position_only",
                })
                continue
            if metadata.get("synthetic"):
                # DistributionSynthesisStage runs BEFORE this stage and bumps
                # OBJECT_COUNT, so the loop above walks its painted points too --
                # but a painted point has no detection box and no crop_{idx} of
                # its own; it exists to CONSUME a pool, never to populate one.
                # Left in, each one scored on pure defaults (confidence and
                # fill_ratio both fall back to 0.5, depth is unsamplable so it's
                # forced to the far value, giving a flat ~0.30) and so outranked
                # every real detection in a bucket whose instances all sit past
                # billboard_distance_m with a typical sub-0.5 CLIP confidence.
                # Those buckets' billboard_pools then filled with indices that
                # have no image behind them, and SceneGenerationStage rendered
                # crop_{idx} for a crop that was never written -- observed as
                # distant classes (trees) disappearing from the scene while close
                # ones (flowers) survived on their nonzero depth score.
                #
                # Counted rather than listed per-index: a painted population is
                # routinely thousands of points, which would swamp asset_debug.json
                # with entries carrying no information beyond their own count.
                synthetic_skipped += 1
                continue

            bucket = metadata.get("bucket") or 0
            key = (obj_class, int(bucket))
            candidates_by_group.setdefault(key, []).append({
                "idx": idx, "box": box, "depth": depth, "metadata": metadata,
            })

        group_best: dict[tuple[str, int], tuple[int, float, float]] = {}
        billboard_pools: dict[str, list[int]] = {}
        disqualified_debug = []
        for (obj_class, bucket), candidates in candidates_by_group.items():
            scored = []
            for candidate in candidates:
                # A candidate whose depth couldn't be sampled still belongs in
                # the billboard pool (it just can't be scored on depth or
                # compared for occlusion) -- treat it as "far" rather than
                # dropping it, so a bad depth sample doesn't silently shrink
                # the billboard pool.
                depth = candidate["depth"] if candidate["depth"] is not None else threshold * 10.0
                occlusion = occlusion_score(
                    candidate["idx"], candidate["box"], depth, depth_by_idx, self.config.occlusion_depth_margin,
                )
                crop = get_image(f"crop_{candidate['idx']}")
                score = composite_score(
                    candidate["metadata"], crop, depth, occlusion, threshold,
                    self.config.score_weight_confidence, self.config.score_weight_fill_ratio,
                    self.config.score_weight_depth, self.config.score_weight_occlusion,
                    self.config.occlusion_covered_fraction_threshold,
                )
                disqualified = occlusion >= self.config.occlusion_disqualify_fraction
                scored.append({**candidate, "depth": depth, "occlusion": occlusion, "score": score, "disqualified": disqualified})

            scored_by_rank = sorted(scored, key=lambda s: s["score"], reverse=True)
            pool_key = f"{obj_class}::{bucket}"
            billboard_pools[pool_key] = [s["idx"] for s in scored_by_rank[: self.config.billboard_top_k]]

            within_threshold = [s for s in scored if s["depth"] < threshold]
            if not within_threshold:
                continue  # nothing close enough to mesh -- billboard-only group

            eligible = [s for s in within_threshold if not s["disqualified"]] or within_threshold
            winner = max(eligible, key=lambda s: s["score"])

            # Size gate: a bespoke mesh is only worth generating for a prominent
            # subject. A close-but-tiny winner (e.g. a single meadow flower) stays
            # billboard-only -- its pool was already curated above, so it isn't
            # lost, just not meshified. Skipped when panorama dims or the winning
            # box are unavailable (can't measure), preserving distance-only
            # behaviour; disabled entirely at min_mesh_area_fraction == 0.
            wb = winner.get("box")
            if self.config.min_mesh_area_fraction > 0 and wb is not None and pano_w and pano_h:
                win_area_fraction = (wb[2] * wb[3]) / float(pano_w * pano_h)
                if win_area_fraction < self.config.min_mesh_area_fraction:
                    skipped_debug.append({
                        "idx": winner["idx"],
                        "class": obj_class,
                        "reason": "too_small_for_mesh",
                        "bucket": bucket,
                        "area_fraction": round(win_area_fraction, 5),
                    })
                    continue

            group_best[(obj_class, bucket)] = (winner["idx"], winner["depth"], winner["score"])
            for s in within_threshold:
                if s["disqualified"]:
                    disqualified_debug.append({
                        "idx": s["idx"],
                        "class": obj_class,
                        "bucket": bucket,
                        "occlusion": round(s["occlusion"], 3),
                        "chosen_anyway": s["idx"] == winner["idx"],
                    })

        return group_best, billboard_pools, skipped_debug, disqualified_debug, synthetic_skipped

    def _write_debug(
        self, skipped: list, group_best: dict, disqualified: list, billboard_pools: dict,
        synthetic_skipped: int = 0,
    ):
        if self.output is None:
            return
        payload = {
            "billboard_distance_m": self.config.billboard_distance_m,
            "billboard_top_k": self.config.billboard_top_k,
            "summary": {
                "skipped_env_or_filtered": len(skipped),
                "skipped_synthetic": synthetic_skipped,
                "groups_billboard_only": len(billboard_pools) - len(group_best),
                "groups_meshified": len(group_best),
            },
            "skipped": skipped,
            "occlusion_disqualified": disqualified,
            "groups": [
                {"class": cls, "bucket": bucket, "representative_idx": idx, "depth_m": round(depth, 2), "score": round(score, 3)}
                for (cls, bucket), (idx, depth, score) in group_best.items()
            ],
            "billboard_pools": billboard_pools,
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
        if context.object("billboard_pools") is None:
            return False
        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        pano_w = panorama.width if panorama is not None else None
        pano_h = panorama.height if panorama is not None else None

        group_best, _, _, _, _ = self._curate(
            count, context.object, context.image, panorama_depth, pano_w, pano_h,
        )
        for obj_class, bucket in group_best:
            if context.mesh(f"category_mesh_{obj_class}_{bucket}") is None:
                return False
        return True

    def model_names(self) -> list[str]:
        return ModelGenerator.model_names(type=self.config.generator_type)

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        count = context.object(ContextKey.OBJECT_COUNT)
        if count is None or count == 0:
            return None

        seen_groups: set[tuple[str, int]] = set()
        images = []
        total_verts = 0
        total_faces = 0

        for i in range(count):
            meta = context.object(f"metadata_{i}") or {}
            obj_class = meta.get("class", f"Object {i + 1}")
            bucket = int(meta.get("bucket") or 0)
            group = (obj_class, bucket)
            mesh_key = f"category_mesh_{obj_class}_{bucket}"
            mesh = context.mesh(mesh_key)
            crop = context.image(f"crop_{i}")

            if group not in seen_groups:
                seen_groups.add(group)
                if mesh is not None:
                    total_verts += mesh.vertex_count
                    total_faces += mesh.face_count
                if crop is not None and len(images) < 6:
                    label = f"{obj_class} #{bucket}" if bucket else obj_class
                    if mesh is not None:
                        label += f" ({mesh.vertex_count:,}v)"
                    images.append((crop.image, label))

        reconstructed = len(seen_groups)
        stats = {"Categories reconstructed": str(reconstructed)}
        if reconstructed > 0:
            stats["Total vertices"] = f"{total_verts:,}"
            stats["Total triangles"] = f"{total_faces:,}"
            stats["Generator"] = self.config.generator_type.name
        return ReportSection(
            stage_name=self.name,
            title="3D Object Reconstruction",
            body=(
                "One 3D mesh is generated per visual-similarity bucket within each object "
                "category, using the closest instance in that bucket as the representative "
                f"crop. The {self.config.generator_type.name} model reconstructs a textured "
                "mesh, which is normalised to a 1 m canonical box. At scene placement, "
                "instances closer than the mesh distance threshold use the bucket's mesh "
                "(with a random rotation); farther instances use a billboard drawn from the "
                "bucket's curated crop pool."
            ),
            images=images,
            stats=stats,
        )

    def clean_up(self):
        super().clean_up()
