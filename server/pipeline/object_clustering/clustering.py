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
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.embedding_model_name = embedding_model_name
        self.cluster_distance_threshold = cluster_distance_threshold


class ObjectCategoryClusteringStage(PipelineStage):
    """
    Sub-clusters each class's detections by visual similarity into 'bucket'
    groups -- e.g. splitting one undifferentiated "flower" population into
    distinct color/species variants -- so downstream stages (Panorama Asset
    Generation, Scene Generation) can curate and place assets per visual
    variant instead of treating an entire class as one interchangeable pool.

    Runs after Object Correlation, reusing its class -> indices grouping
    directly rather than re-deriving it. Within each class group, crops are
    embedded with DINOv2 (a self-supervised visual-similarity model -- CLIP's
    zero-shot *text*-prompted classification can tell "flower" from "tree" but
    isn't suited to telling one flower's embedding from another's) and
    agglomeratively clustered (average-linkage, cosine distance, cut at
    cluster_distance_threshold) -- unlike k-means, this doesn't require
    knowing the number of visual variants ahead of time, which varies per
    scene. Classes with fewer than 2 detections trivially get bucket 0.

    Reads:  ContextKey.OBJECT_CORRELATION (class -> indices), crop_{i}
    Writes: metadata_{i} ('bucket': int, scoped within its own class)
    Debug:  self.output/clustering_debug.json -- per-class cluster counts/sizes
            self.output/debug_panorama.png -- panorama with bucket-colored boxes
    Config: embedding_model_name (default facebook/dinov2-base),
            cluster_distance_threshold (default 0.3, cosine distance)
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
            debug_by_class[obj_class] = self._cluster_class(context, obj_class, grp.indices, task)

        self.finish_progress(task)
        self._write_debug(context, debug_by_class)
        return context

    def _cluster_class(self, context: PipelineContext, obj_class: str, indices: list[int], task) -> dict:
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
            return {"count": len(valid_indices), "buckets": 1 if valid_indices else 0}

        labels = self._cluster(np.stack(embeddings))
        for idx, bucket in zip(valid_indices, labels):
            self._set_bucket(context, idx, int(bucket))

        sizes = {int(b): int((labels == b).sum()) for b in np.unique(labels)}
        self.log_info(f"  {obj_class}: {len(valid_indices)} object(s) -> {len(sizes)} bucket(s) {sizes}")
        return {"count": len(valid_indices), "buckets": len(sizes), "bucket_sizes": sizes}

    def _set_bucket(self, context: PipelineContext, idx: int, bucket: int):
        metadata = context.input_object(f"metadata_{idx}") or {}
        context.add_object(f"metadata_{idx}", {**metadata, "bucket": bucket})

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        distances = pdist(embeddings, metric="cosine")
        z = linkage(distances, method="average")
        labels = fcluster(z, t=self.config.cluster_distance_threshold, criterion="distance")
        # fcluster labels are 1-based and not necessarily contiguous -- remap
        # to a dense 0-based range for a tidy bucket id space.
        remap = {old: new for new, old in enumerate(sorted(set(labels.tolist())))}
        return np.array([remap[label] for label in labels])

    def _write_debug(self, context: PipelineContext, debug_by_class: dict[str, dict]):
        if self.output is None:
            return

        with open(self.output / "clustering_debug.json", "w") as f:
            json.dump({
                "cluster_distance_threshold": self.config.cluster_distance_threshold,
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
        for grp in correlation.groups.values():
            for idx in grp.indices:
                if not context.has_stage_output(f"metadata_{idx}"):
                    return False
        return True

    def model_names(self) -> list[str]:
        return DinoV2Embedder.model_names()

    def clean_up(self):
        self._embedder = None
        super().clean_up()
