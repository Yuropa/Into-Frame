import colorsys
import json
from logging import Logger
from typing import Any

import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import torch
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist

from pipeline.object_clustering.dinov2_embedder import DinoV2Embedder
from pipeline.object_correlation.object_correlation_result import ObjectCorrelationResult, ObjectGroupStats
from pipeline.object_typing.categories import ENVIRONMENT_CATEGORIES
from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration


class ObjectCategoryClusteringConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        embedding_model_name: str = DinoV2Embedder.MODEL_NAME,
        cluster_distance_threshold: float = 0.3,
        position_only_similarity_threshold: float = 0.75,
        max_buckets_per_class: int = 8,
        min_bucket_size: int = 3,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.embedding_model_name = embedding_model_name
        self.cluster_distance_threshold = cluster_distance_threshold
        self.position_only_similarity_threshold = position_only_similarity_threshold
        # A bucket is meant to be a visual VARIANT of a class (a flower colour, a
        # tree species) -- a handful per class. cluster_distance_threshold alone is
        # an absolute cut on cosine distance with no notion of how many groups that
        # produces, and on a full-panorama detection set it produces one bucket per
        # instance: measured on an alpine-meadow capture, 302 (class, bucket) groups
        # across 77 flower, 65 tree, 57 table and 33 plant buckets. Every one of
        # those asks PanoramaAssetGenerationStage for its own mesh and its own
        # billboard pool, so the pools degenerate to a single crop each (every
        # instance of a "variant" reusing one image) and DistributionSynthesisStage
        # inherits 77 flower variants to sample between.
        #
        # These two bound the outcome instead of the cut: re-cut the same dendrogram
        # at the coarsest level that yields at most max_buckets_per_class, then fold
        # any bucket still under min_bucket_size into its nearest surviving centroid.
        # 0 disables either check.
        self.max_buckets_per_class = max_buckets_per_class
        self.min_bucket_size = min_bucket_size


class ObjectCategoryClusteringStage(PipelineStage):
    """
    Sub-clusters each class's *confident* detections (ObjectTypingStage's
    'low_confidence' flag false -- Grounding DINO independently found that crop's
    proposed label in that crop) by visual similarity into 'bucket' groups -- e.g. splitting
    one undifferentiated "flower" population into distinct color/species
    variants -- so downstream stages (Panorama Asset Generation, Scene
    Generation) can curate and place assets per visual variant instead of
    treating an entire class as one interchangeable pool.

    Runs after Object Correlation, reusing its class -> indices grouping
    directly rather than re-deriving it. Within each class group, confident
    crops are embedded with DINOv2 (a self-supervised visual-similarity model
    -- CLIP's zero-shot *text*-prompted classification can tell "flower" from
    "tree" but isn't suited to telling one flower's embedding from another's)
    and agglomeratively clustered (average-linkage, cosine distance, cut at
    cluster_distance_threshold) -- unlike k-means, this doesn't require
    knowing the number of visual variants ahead of time, which varies per
    scene. Classes with fewer than 2 confident detections trivially get
    bucket 0.

    Low-confidence crops (a label ObjectTypingStage proposed but could not
    corroborate -- see its docstring) never get to
    anchor a bucket or found their own category on that guess alone: instead, each
    one's DINOv2 embedding is compared against every confident bucket's centroid
    embedding, across every class, not just the one it was originally (unreliably)
    guessed as. A strong enough match (position_only_similarity_threshold) reassigns
    it to that class with metadata_{idx}['position_only'] = True -- it's moved into
    that class's ObjectCorrelationResult group (so ObjectDistributionStage counts its
    world position toward that class's spatial pattern) but PanoramaAssetGenerationStage
    and SceneGenerationStage both skip position_only crops outright: no mesh, no
    billboard, no video tracking, nothing rendered from its own uncertain crop --
    only its location is trusted. No match at all demotes it to 'indeterminate',
    dropped everywhere, same as a crop CLIP was never confident about in the first
    place. This is what stops a handful of noise-sized, hallucinated-caption crops
    (a blurry rock crop BLIP captioned as "a person", say) from spawning a whole
    fake category with no real evidence anywhere else in the scene.

    Reads:  ContextKey.OBJECT_CORRELATION (class -> indices), crop_{i}, metadata_{i}
            ('low_confidence', from ObjectTypingStage)
    Writes: metadata_{i} ('bucket': int, scoped within its own class, for confident
            crops; 'class'/'position_only' for reassigned low-confidence crops;
            'class' = 'indeterminate' for rejected ones)
            ContextKey.OBJECT_CORRELATION (re-persisted with low-confidence crops
            moved to their visually-matched class's group, or dropped entirely)
    Debug:  self.output/clustering_debug.json -- per-class cluster counts/sizes,
            plus low-confidence reassignment/rejection counts
            self.output/debug_panorama.png -- panorama with bucket-colored boxes
    Config: embedding_model_name (default facebook/dinov2-base),
            cluster_distance_threshold (default 0.3, cosine distance)
            position_only_similarity_threshold (default 0.75, cosine similarity)
            max_buckets_per_class (default 8) -- re-cut the dendrogram coarser if the
              distance cut produced more buckets than this; 0 disables
            min_bucket_size (default 3) -- fold smaller buckets into the nearest
              surviving one; 0/1 disables
    """

    @classmethod
    def config_class(cls):
        return ObjectCategoryClusteringConfiguration

    def __init__(self, config: ObjectCategoryClusteringConfiguration) -> None:
        super().__init__(config)
        self._embedder = None

    def run(self, context: PipelineContext) -> PipelineContext:
        correlation = context.input_object_correlation(ContextKey.OBJECT_CORRELATION)
        if correlation is None or not correlation.groups:
            self.log_info("No correlated objects to cluster, skipping")
            return context

        if self._embedder is None:
            self._embedder = DinoV2Embedder(self.device, model_name=self.config.embedding_model_name)

        total = sum(len(grp.indices) for grp in correlation.groups.values())
        task = self.create_progress(max(total, 1), "Clustering object categories…")

        debug_by_class: dict[str, dict] = {}
        low_confidence_by_class: dict[str, list[int]] = {}
        centroids_by_class: dict[str, dict[int, np.ndarray]] = {}
        trust_split: dict[str, tuple[int, int]] = {}

        for obj_class, grp in correlation.groups.items():
            # Environment/indeterminate groups (sky, grass, water, ...) are
            # never meshed or billboard-curated downstream (Panorama Asset
            # Generation filters them out the same way) -- clustering them
            # would just burn DINOv2 embedding time on crops nothing reads
            # 'bucket' from.
            if obj_class in ENVIRONMENT_CATEGORIES or obj_class == "indeterminate":
                for _ in grp.indices:
                    self.advance_progress(task)
                continue

            confident_indices, low_confidence_indices = [], []
            for idx in grp.indices:
                metadata = context.input_object(f"metadata_{idx}") or {}
                if metadata.get("low_confidence"):
                    low_confidence_indices.append(idx)
                else:
                    confident_indices.append(idx)

            debug_by_class[obj_class], centroids_by_class[obj_class] = self._cluster_class(
                context, obj_class, confident_indices, task
            )
            if low_confidence_indices:
                low_confidence_by_class[obj_class] = low_confidence_indices
            trust_split[obj_class] = (len(confident_indices), len(low_confidence_indices))

        # The composition of the confident set, per class, before anything is done
        # with it. This is the state that decides the whole outcome: a class with no
        # confident crop has no bucket, so every low-confidence crop it has must
        # match some OTHER class's bucket or be dropped. That is how a capture with
        # 65 typed trees rendered none, and until now the only trace of it was a
        # zero in clustering_debug.json that had to be noticed and interpreted.
        #
        # Wrapped for the same reason SceneGenerationStage's mesh report is: this
        # is reporting, and reporting must not be able to fail the stage.
        try:
            if trust_split:
                self.log_info("  Confident / low-confidence split per class:")
                for cls, (n_conf, n_low) in sorted(trust_split.items(), key=lambda kv: -sum(kv[1])):
                    marker = "  <-- NO ANCHOR" if n_conf == 0 and n_low else ""
                    self.log_info(f"    {cls:<16} confident {n_conf:<4} low-confidence {n_low:<4}{marker}")
                anchorless = [c for c, (n_conf, n_low) in trust_split.items() if n_conf == 0 and n_low]
                if anchorless:
                    self.log_warning(
                        f"  {len(anchorless)} class(es) have low-confidence crops but NO confident "
                        f"crop to anchor them: {', '.join(sorted(anchorless))}. Their crops can only "
                        f"survive by matching another class's bucket."
                    )
        except Exception as e:
            self.log_warning(f"Could not report trust split ({type(e).__name__}: {e})")

        reassigned, rejected = self._reassign_low_confidence(
            context, correlation, low_confidence_by_class, centroids_by_class, task
        )

        context.add_object_correlation(ContextKey.OBJECT_CORRELATION, correlation)
        self.finish_progress(task)
        self._write_debug(context, debug_by_class, reassigned, rejected)
        return context

    def _cluster_class(
        self, context: PipelineContext, obj_class: str, indices: list[int], task
    ) -> tuple[dict, dict[int, np.ndarray]]:
        embeddings = []
        valid_indices = []
        for idx in indices:
            crop = context.input_image(f"crop_{idx}")
            if crop is not None:
                embeddings.append(self._embedder.embed(crop))
                valid_indices.append(idx)
            self.advance_progress(task)

        if len(valid_indices) < 2:
            for idx in valid_indices:
                self._set_bucket(context, idx, 0)
            centroids = {0: embeddings[0]} if embeddings else {}
            return {"count": len(valid_indices), "buckets": 1 if valid_indices else 0}, centroids

        embeddings_arr = np.stack(embeddings)
        labels = self._cluster(embeddings_arr)
        for idx, bucket in zip(valid_indices, labels):
            self._set_bucket(context, idx, int(bucket))

        centroids: dict[int, np.ndarray] = {}
        for b in np.unique(labels):
            centroid = embeddings_arr[labels == b].mean(axis=0)
            centroids[int(b)] = centroid / np.linalg.norm(centroid)

        sizes = {int(b): int((labels == b).sum()) for b in np.unique(labels)}
        self.log_info(f"  {obj_class}: {len(valid_indices)} object(s) -> {len(sizes)} bucket(s) {sizes}")
        return {"count": len(valid_indices), "buckets": len(sizes), "bucket_sizes": sizes}, centroids

    def _reassign_low_confidence(
        self,
        context: PipelineContext,
        correlation: ObjectCorrelationResult,
        low_confidence_by_class: dict[str, list[int]],
        centroids_by_class: dict[str, dict[int, np.ndarray]],
        task,
    ) -> tuple[int, int]:
        # Flatten every confident bucket's centroid into one list to compare each
        # low-confidence crop against, regardless of which class it was originally
        # (unreliably) guessed as -- the match is purely visual, per the user's
        # explicit intent: color/shape similarity, not caption/keyword text.
        all_centroids: list[tuple[str, np.ndarray]] = [
            (cls, centroid)
            for cls, buckets in centroids_by_class.items()
            for centroid in buckets.values()
        ]

        reassigned = 0
        rejected = 0
        threshold = self.config.position_only_similarity_threshold
        # Best similarity reached per originating class, whether or not it cleared
        # the bar. A class whose crops all peak just under threshold is a threshold
        # problem; one that peaks near zero genuinely resembles nothing confident in
        # the scene. The bare reject count cannot tell those apart, and it was the
        # only thing reported when 270 crops were dropped in one line.
        similarity_by_class: dict[str, list[float]] = {}
        for old_class, indices in low_confidence_by_class.items():
            for idx in indices:
                metadata = context.input_object(f"metadata_{idx}") or {}
                crop = context.input_image(f"crop_{idx}")

                best_class, best_sim = None, -1.0
                if crop is not None and all_centroids:
                    embedding = self._embedder.embed(crop)
                    for cls, centroid in all_centroids:
                        sim = float(np.dot(embedding, centroid))
                        if sim > best_sim:
                            best_sim, best_class = sim, cls

                similarity_by_class.setdefault(old_class, []).append(
                    best_sim if best_class is not None else 0.0
                )

                if best_class is not None and best_sim >= threshold:
                    self._move_index(correlation, old_class, best_class, idx)
                    context.add_object(f"metadata_{idx}", {
                        **metadata, "class": best_class, "position_only": True,
                        "visual_match_similarity": round(best_sim, 4),
                    })
                    reassigned += 1
                else:
                    self._move_index(correlation, old_class, None, idx)
                    context.add_object(f"metadata_{idx}", {**metadata, "class": "indeterminate"})
                    rejected += 1
                self.advance_progress(task)

        # Reporting only from here down -- guarded so it cannot fail the stage.
        try:
            self._report_reassignment(
                reassigned, rejected, all_centroids, similarity_by_class, threshold
            )
        except Exception as e:
            self.log_warning(f"Could not report reassignment ({type(e).__name__}: {e})")
        return reassigned, rejected

    def _report_reassignment(
        self,
        reassigned: int,
        rejected: int,
        all_centroids: list,
        similarity_by_class: dict[str, list[float]],
        threshold: float,
    ) -> None:
        if reassigned or rejected:
            self.log_info(
                f"  low-confidence crops: {reassigned} visually corroborated (position-only), "
                f"{rejected} unmatched -> dropped"
            )
            if not all_centroids:
                self.log_warning(
                    "  No confident bucket existed anywhere in the scene, so every "
                    "low-confidence crop was dropped without a comparison. Check "
                    "Object Typing's 'Label verification' lines -- nothing was corroborated."
                )
            else:
                self.log_info(
                    f"  Best visual match per originating class "
                    f"(threshold {threshold:.2f}, max / median):"
                )
                for cls, sims in sorted(similarity_by_class.items(), key=lambda kv: -len(kv[1])):
                    ordered = sorted(sims)
                    median = ordered[len(ordered) // 2]
                    marker = "  <-- all dropped" if max(sims) < threshold else ""
                    self.log_info(
                        f"    {cls:<16} n={len(sims):<4} "
                        f"{max(sims):.3f} / {median:.3f}{marker}"
                    )

    def _move_index(
        self, correlation: ObjectCorrelationResult, old_class: str, new_class: str | None, idx: int,
    ) -> None:
        old_grp = correlation.groups.get(old_class)
        if old_grp is not None and idx in old_grp.indices:
            old_grp.indices.remove(idx)
        if new_class is not None:
            if new_class not in correlation.groups:
                correlation.groups[new_class] = ObjectGroupStats(object_type=new_class)
            correlation.groups[new_class].indices.append(idx)

    def _set_bucket(self, context: PipelineContext, idx: int, bucket: int):
        metadata = context.input_object(f"metadata_{idx}") or {}
        context.add_object(f"metadata_{idx}", {**metadata, "bucket": bucket})

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        distances = pdist(embeddings, metric="cosine")
        z = linkage(distances, method="average")
        labels = fcluster(z, t=self.config.cluster_distance_threshold, criterion="distance")

        # Re-cut the SAME dendrogram at a coarser level rather than raising
        # cluster_distance_threshold: 'maxclust' picks the largest cut that still
        # yields at most max_buckets_per_class, which keeps the threshold meaningful
        # as "how different counts as a different variant" for scenes that genuinely
        # have few variants, while bounding the pathological case. See the config
        # comment for the 302-group measurement that motivates it.
        max_buckets = self.config.max_buckets_per_class
        if max_buckets > 0 and len(set(labels.tolist())) > max_buckets:
            labels = fcluster(z, t=max_buckets, criterion="maxclust")

        labels = self._merge_small_buckets(embeddings, labels)

        # fcluster labels are 1-based and not necessarily contiguous -- remap
        # to a dense 0-based range for a tidy bucket id space.
        remap = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
        return np.array([remap[label] for label in labels])

    def _merge_small_buckets(self, embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fold every bucket smaller than min_bucket_size into the nearest bucket
        that isn't. A singleton bucket is the failure mode this exists for: it wins
        its own mesh and its own billboard pool of exactly one crop, so every
        instance downstream that resolves to it renders the identical image -- worse
        than being grouped with the visually closest real variant.

        Falls back to folding everything into the largest bucket when no bucket
        clears the threshold, which is the right answer for a class with only a
        couple of confident detections: one variant, not several of size one."""
        min_size = self.config.min_bucket_size
        if min_size <= 1:
            return labels

        unique, counts = np.unique(labels, return_counts=True)
        keep = unique[counts >= min_size]
        if len(keep) == len(unique):
            return labels
        if len(keep) == 0:
            # Nothing clears the bar -- collapse to the single largest bucket
            # rather than leaving every instance in a bucket of its own.
            return np.full_like(labels, unique[int(np.argmax(counts))])

        # Cosine similarity against L2-normalised centroids, matching how
        # _cluster_class builds the centroids it hands to low-confidence matching.
        centroids = {}
        for b in keep:
            centroid = embeddings[labels == b].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids[int(b)] = centroid / norm if norm > 0 else centroid
        keep_ids = list(centroids)
        centroid_matrix = np.stack([centroids[b] for b in keep_ids])

        merged = labels.copy()
        for b, count in zip(unique, counts):
            if count >= min_size:
                continue
            members = np.nonzero(labels == b)[0]
            for i in members:
                embedding = embeddings[i]
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                merged[i] = keep_ids[int(np.argmax(centroid_matrix @ embedding))]
        return merged

    def _write_debug(self, context: PipelineContext, debug_by_class: dict[str, dict], reassigned: int, rejected: int):
        if self.output is None:
            return

        with open(self.output / "clustering_debug.json", "w") as f:
            json.dump({
                "cluster_distance_threshold": self.config.cluster_distance_threshold,
                "position_only_similarity_threshold": self.config.position_only_similarity_threshold,
                "max_buckets_per_class": self.config.max_buckets_per_class,
                "min_bucket_size": self.config.min_bucket_size,
                "low_confidence": {"reassigned_position_only": reassigned, "rejected": rejected},
                "classes": debug_by_class,
            }, f, indent=2)

        self._draw_debug_panorama(context)

    def _draw_debug_panorama(self, context: PipelineContext):
        panorama = context.input_panorama(ContextKey.PANORAMA)
        if panorama is None:
            return

        object_count = context.input_object(ContextKey.OBJECT_COUNT) or 0
        keys = []
        for idx in range(object_count):
            metadata = context.object(f"metadata_{idx}")
            if metadata is None or metadata.get("bucket") is None or not metadata.get("box"):
                continue
            keys.append(f"{metadata.get('class')}::{metadata['bucket']}")

        unique_keys = sorted(set(keys))
        colors = {}
        for i, key in enumerate(unique_keys):
            h = i / max(len(unique_keys), 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.80, 0.95)
            colors[key] = (int(r * 255), int(g * 255), int(b * 255))

        base = panorama.rgb().convert("RGBA")
        overlay = PIL.Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = PIL.ImageDraw.Draw(overlay)
        font = PIL.ImageFont.load_default()

        for idx in range(object_count):
            metadata = context.object(f"metadata_{idx}")
            if metadata is None or metadata.get("bucket") is None:
                continue
            box = metadata.get("box")
            if not box:
                continue
            key = f"{metadata.get('class')}::{metadata['bucket']}"
            r, g, b = colors.get(key, (128, 128, 128))
            x, y, w, h = box
            draw.rectangle([x, y, x + w, y + h], outline=(r, g, b, 220), width=2)
            draw.text((x + 4, y + 4), key, fill=(r, g, b, 255), font=font)

        composite = PIL.Image.alpha_composite(base, overlay).convert("RGB")
        composite.save(self.output / "debug_panorama.png")

    def has_expected_output(self, context: PipelineContext) -> bool:
        correlation = context.object_correlation(ContextKey.OBJECT_CORRELATION)
        if correlation is None:
            return True
        for obj_class, grp in correlation.groups.items():
            # Mirror run()'s own skip condition -- environment/indeterminate
            # objects are never bucketed, so metadata_{idx} for them is never
            # written by this stage. Checking has_stage_output for them
            # unconditionally made this stage (and everything downstream, via
            # the dirty cascade) rerun on every single invocation whenever the
            # scene had any such object.
            if obj_class in ENVIRONMENT_CATEGORIES or obj_class == "indeterminate":
                continue
            for idx in grp.indices:
                if not context.has_stage_output(f"metadata_{idx}"):
                    return False
        return True

    def model_names(self) -> list[str]:
        return DinoV2Embedder.model_names()

    def clean_up(self):
        self._embedder = None
        super().clean_up()
