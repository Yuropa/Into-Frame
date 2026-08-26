import colorsys
import json
import re
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
from pipeline.object_typing.categories import (
    ENVIRONMENT_CATEGORIES, BUILT_ENVIRONMENT_CATEGORIES, WATERBORNE_CATEGORIES,
    LEVEL_GROUND_CATEGORIES, DOMESTIC_CAPTION_SUBJECTS, OBJECT_CATEGORIES,
    VEGETATION_CATEGORIES, normalize_category,
)
from pipeline.panorama_segmentation.panorama_region_result import RegionType
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
        min_built_fraction: float = 0.005,
        min_bucket_size: int = 3,
        region_veto_fraction: float = 0.85,
        caption_veto: bool = True,
        outsize_split_ratio: float = 2.5,
        outsize_split_min_height_m: float = 2.0,
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
        # Share of the panorama that must type as RegionType.BUILT before this scene
        # is allowed to contain built-in-place categories at all. Measured off the
        # stored region maps of the five sample captures: Rainier 0.01%, Iceland 0.12%,
        # Shark Fin 1.13%, Irises 1.69%, Paris 7.47% -- 0.5% sits ~4x clear of the
        # nearest on either side. Only ever consulted alongside the scene tags; see
        # _scene_admits_built. 0 disables the veto.
        self.min_built_fraction = float(min_built_fraction)
        # Share of a detection's own box that must land on the offending region
        # before its class is vetoed as physically implausible there -- see
        # _region_veto, WATERBORNE_CATEGORIES and LEVEL_GROUND_CATEGORIES.
        #
        # Deliberately near-total rather than a majority. A box that is 60% water is
        # routinely a real thing at a waterline; a box that is 85%+ water with a
        # picnic table in it is not a picnic table. Measured over the five sample
        # captures, every rejection at 0.85 is junk and Mount Rainier loses only four
        # phantom aircraft it should never have had. 0 disables the veto.
        self.region_veto_fraction = float(region_veto_fraction)
        # Drop a detection whose caption describes something that cannot be in this
        # scene -- see _caption_veto for the four conditions it requires, and
        # DOMESTIC_CAPTION_SUBJECTS for what BLIP writes when it cannot parse a
        # texture. False disables it.
        self.caption_veto = bool(caption_veto)
        # Split the tallest member of a bucket out into a bucket of its own when it is
        # this many times taller IN METRES than the next tallest.
        #
        # A bucket is a promise that one reconstruction, scaled to each member's own
        # detected height, can stand in for all of them. DINOv2 similarity cannot keep
        # that promise on its own: it sees shape and texture, and a landmark against
        # the sky looks like every other tall pale structure against the sky. On the
        # Paris capture the Eiffel Tower (36.9 m) shares tower::2 with five fragments
        # of the cathedral on the far bank (12.6, 7.2, 7.1, 5.4, 4.7 m), and the
        # bucket's representative is chosen by composite_score -- confidence, fill
        # ratio, proximity, occlusion, no size term at all -- which ranks the tower
        # LAST of the six. Every instance in that bucket, the tower included, then
        # renders a cathedral spire stretched to its own height.
        #
        # Metres, not pixels. Angular height in an equirect is a function of distance,
        # so a pixel-height rule splits a near flower from a far one of the same
        # species -- measured on these captures it fires on 14 groups, mostly for
        # exactly that reason. Metric height (angular extent x depth, computed exactly
        # as scene_generation/projection.py computes it) is distance-invariant, which
        # is the property the test needs.
        self.outsize_split_ratio = float(outsize_split_ratio)
        # Floor on the outsized member's own height, so the ratio test only ever runs
        # where being wrong about size is worth a second reconstruction. Without it the
        # rule fires on sub-metre noise -- a 0.3 m "ship" over a 0.1 m one on the
        # Rainier capture, and every small bucket on the Irises painting, whose 10,739
        # detections are all centimetres tall.
        self.outsize_split_min_height_m = float(outsize_split_min_height_m)


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
            min_built_fraction (default 0.005) -- built-region share below which
              BUILT_ENVIRONMENT_CATEGORIES are vetoed for this scene; 0 disables
    """

    @classmethod
    def config_class(cls):
        return ObjectCategoryClusteringConfiguration

    def __init__(self, config: ObjectCategoryClusteringConfiguration) -> None:
        super().__init__(config)
        self._embedder = None

    def _scene_admits_built(self, context: PipelineContext) -> tuple[bool, str]:
        """Whether this scene shows any built environment at all.

        Two independent signals, both already computed upstream, and BOTH must be
        silent before anything is vetoed -- a single one saying "no buildings" is not
        worth overruling a classifier on:

          1. The fraction of the panorama PanoramaRegionStage types RegionType.BUILT.
             Measured off the stored region maps of the five sample captures: Rainier
             0.01%, Iceland 0.12%, Shark Fin 1.13%, Irises 1.69%, Paris 7.47%. The
             0.5% threshold sits ~4x clear of the nearest capture on either side.
          2. RAM++'s scene tags. Rainier's are entirely natural ("mountain range |
             pasture | wildflower | snowy | sky ..."), Paris's are not.

        Returns (admits, why) -- `why` is logged, because "this scene has no built
        environment" is a claim worth being able to check afterwards.
        """
        region_type = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        if region_type is None:
            return True, "no region map — built categories left alone"
        built_fraction = float(
            (np.asarray(region_type.depth) == int(RegionType.BUILT)).mean()
        )
        if built_fraction >= self.config.min_built_fraction:
            return True, f"{built_fraction * 100:.1f}% of the panorama is built"

        tags = context.input_object(ContextKey.RECOGNIZE_TAGS) or ""
        built_tags = sorted({
            tag.strip() for tag in str(tags).split("|")
            if normalize_category(tag.strip()) in BUILT_ENVIRONMENT_CATEGORIES
        })
        if built_tags:
            return True, (
                f"only {built_fraction * 100:.1f}% built region, but the scene tags "
                f"name {', '.join(built_tags)}"
            )
        return False, (
            f"{built_fraction * 100:.1f}% of the panorama is built and no scene tag "
            f"names a structure"
        )

    def _region_veto(self, context: PipelineContext):
        """Build a per-detection test for "can this class be standing here at all".

        _scene_admits_built above asks the same kind of question of the WHOLE scene:
        does this capture contain any built environment? That is the right shape for
        a class that either belongs in the scene or does not, and the wrong shape for
        a capture that genuinely has some of a class and also has a cliff face being
        read as more of it. Shark Fin Cove types 1.13% of its panorama BUILT -- over
        that gate's 0.5% threshold, so its tables and fences survive it -- while
        being a scene with no tables and no fences at all.

        This asks per crop instead, using the box each detection already carries:
        what is actually underneath THIS one. See the block above
        WATERBORNE_CATEGORIES for why the region map is the right witness and why
        both category sets are as narrow as they are.

        Returns a callable (idx, obj_class) -> reason string or None, or None when
        there is no region map to consult (in which case nothing is vetoed, matching
        _scene_admits_built's own behaviour on a missing map).
        """
        if self.config.region_veto_fraction <= 0:
            return None
        region_type = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        if region_type is None:
            return None

        region_map = np.asarray(region_type.depth)
        map_h, map_w = region_map.shape[:2]
        threshold = self.config.region_veto_fraction

        def veto(idx: int, obj_class: str) -> "str | None":
            metadata = context.input_object(f"metadata_{idx}") or {}
            box = metadata.get("box")
            if not box:
                return None
            x, y, w, h = (int(round(float(v))) for v in box[:4])
            # The box is in panorama pixels and the region map is the panorama's own
            # typing, so these are the same space -- but clamp anyway, since a box
            # that runs off the edge would otherwise silently sample a smaller
            # region than the detection actually covers.
            sub = region_map[max(0, y):min(map_h, y + h), max(0, x):min(map_w, x + w)]
            if sub.size == 0:
                return None

            if obj_class not in WATERBORNE_CATEGORIES:
                water = float((sub == int(RegionType.WATER)).mean())
                if water >= threshold:
                    return f"{water * 100:.0f}% of its box is open water"
            if obj_class in LEVEL_GROUND_CATEGORIES:
                # TERRAIN is mountain/cliff/rock here, NOT ordinary ground -- see
                # LEVEL_GROUND_CATEGORIES.
                terrain = float((sub == int(RegionType.TERRAIN)).mean())
                if terrain >= threshold:
                    return f"{terrain * 100:.0f}% of its box is bare rock"
            if obj_class in BUILT_ENVIRONMENT_CATEGORIES:
                # A structure does not grow inside a tree canopy. This is the third
                # face of the same test and it catches a failure the other two miss
                # entirely, because it fires on classes the SCENE genuinely contains.
                #
                # Paris types 38 crops `tower`, and `tower` is in its own RAM++ scene
                # tags, so the caption veto skips them by construction. They are not
                # one over-split Eiffel Tower -- the tower itself is a single clean
                # 126x529 detection. They are ~14 real fragments of the cathedral
                # (correctly sitting in BUILT, 0.48-1.00) plus a run of right-bank
                # TREES typed as towers, sitting in 0.61-1.00 VEGETATION. Only the
                # region map separates those two populations; class, confidence and
                # caption all look alike.
                vegetation = float((sub == int(RegionType.VEGETATION)).mean())
                if vegetation >= threshold:
                    return f"{vegetation * 100:.0f}% of its box is tree canopy"
            return None

        return veto

    def _caption_veto(self, context: PipelineContext):
        """Build a per-detection test for "does this crop's caption describe something
        that could not be in this scene at all".

        The region veto above asks where a crop SITS. This asks what it was SAID to
        be, and catches the junk that sits somewhere plausible -- a rock on a rock
        face typed `person` because BLIP captioned it "a piece of pizza with bacon".

        Four conditions, ALL required. Each one alone is measurably unsafe, and the
        numbers below are the reason each is here rather than a simpler rule:

          1. The class has no support in the RAM++ scene tags. Tags alone are far too
             sparse to veto on -- Rainier's tags name no tree and it has 67 real ones,
             Paris's name no tree and it has 16 -- so this only ever narrows what the
             other conditions may look at.
          2. The caption names something from DOMESTIC_CAPTION_SUBJECTS.
          3. The caption does NOT name the class itself. This is the caption/class
             agreement test used as a PROTECTION rather than as a veto, and the
             direction matters: as a veto its synonym gaps delete real objects ("a
             plane flying through the air" does not contain the word "aircraft"),
             while as a protection a gap merely keeps a junk crop. Measured: without
             it, four real Rainier conifers captioned "a cat sitting on a tree
             branch" are deleted -- real tree, hallucinated cat.
          4. The class is not vegetation. Green, bushy, edible-looking things are
             exactly what a plant IS, so a food caption over vegetation is weak
             evidence: Iceland's moss is captioned "a piece of broccoli on a plate"
             twelve times and `plant` is the right answer every time.

        Replayed over the five captures this removes 16 Shark Fin, 16 Iceland, 9
        Rainier and 2 Paris detections, and every one is junk -- Rainier's nine
        include "a bird flying by a cake" typed aircraft, "a giraffe standing in the
        middle of a pasture" typed animal, and "a man sitting on a couch with a
        remote" typed bench.

        Returns a callable (idx, obj_class) -> reason or None, or None when disabled
        or when there are no scene tags to establish what the capture is of.
        """
        if not self.config.caption_veto:
            return None
        tags = context.input_object(ContextKey.RECOGNIZE_TAGS) or ""
        supported = {
            normalize_category(tag.strip())
            for tag in str(tags).split("|") if tag.strip()
        }
        supported.discard(None)
        if not supported:
            # No tags means no idea what this capture is of, and condition 1 cannot
            # be evaluated. Veto nothing rather than veto blindly.
            return None

        stop = {"a", "an", "the", "of", "photo", "in", "on", "or", "and",
                "with", "to", "at", "is", "there", "close", "up"}

        def words(text: str) -> set:
            return {w.rstrip("s") for w in re.findall(r"[a-z]+", (text or "").lower())}

        def class_vocabulary(obj_class: str) -> set:
            """Every word that would count as the caption naming this class."""
            vocab = {obj_class, obj_class.rstrip("s")} | set(obj_class.split("_"))
            for prompt in OBJECT_CATEGORIES.get(obj_class, []):
                for word in re.findall(r"[a-z]+", prompt.lower()):
                    if word not in stop and len(word) > 2:
                        vocab.add(word)
                        vocab.add(word.rstrip("s"))
            return {w.rstrip("s") for w in vocab}

        domestic = {w.rstrip("s") for w in DOMESTIC_CAPTION_SUBJECTS}

        def veto(idx: int, obj_class: str) -> "str | None":
            if obj_class in supported or obj_class in VEGETATION_CATEGORIES:
                return None
            metadata = context.input_object(f"metadata_{idx}") or {}
            caption_words = words(metadata.get("caption"))
            named = caption_words & domestic
            if not named:
                return None
            if caption_words & class_vocabulary(obj_class):
                return None
            return (
                f"captioned '{', '.join(sorted(named))}' — nothing in this scene's "
                f"tags supports a {obj_class}"
            )

        return veto

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

        admits_built, built_reason = self._scene_admits_built(context)
        if not admits_built:
            self.log_info(f"  Scene has no built environment: {built_reason}")
        vetoed_built: dict[str, int] = {}
        region_veto = self._region_veto(context)
        caption_veto = self._caption_veto(context)
        vetoed_region: dict[str, int] = {}
        vetoed_region_examples: list[str] = []

        for obj_class, grp in correlation.groups.items():
            # A built-in-place category in a scene with no built environment is a
            # misread silhouette, not a structure -- see BUILT_ENVIRONMENT_CATEGORIES.
            # Demoted to indeterminate rather than deleted, which is the same state
            # this stage's own DINOv2 corroboration puts a crop it cannot vouch for:
            # Panorama Asset Generation skips it, Scene Generation skips it, and it
            # keeps its crop so nothing upstream has to be re-run to see it again.
            if not admits_built and obj_class in BUILT_ENVIRONMENT_CATEGORIES:
                for idx in grp.indices:
                    metadata = context.input_object(f"metadata_{idx}") or {}
                    metadata["class"] = "indeterminate"
                    metadata["vetoed_class"] = obj_class
                    metadata["veto_reason"] = "no built environment in this scene"
                    context.add_object(f"metadata_{idx}", metadata)
                    self.advance_progress(task)
                vetoed_built[obj_class] = len(grp.indices)
                continue

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

                # Per-instance region veto. Unlike the built-environment veto above
                # this cannot be decided for the class as a whole -- the same class
                # can be real in one part of a capture and a misread texture in
                # another -- so it is applied here, per crop, and the survivors go on
                # to cluster normally. Demoted to indeterminate exactly as the built
                # veto demotes, so every downstream consumer already handles it.
                reason = region_veto(idx, obj_class) if region_veto is not None else None
                if reason is None and caption_veto is not None:
                    reason = caption_veto(idx, obj_class)
                if reason is not None:
                    metadata["class"] = "indeterminate"
                    metadata["vetoed_class"] = obj_class
                    metadata["veto_reason"] = reason
                    context.add_object(f"metadata_{idx}", metadata)
                    vetoed_region[obj_class] = vetoed_region.get(obj_class, 0) + 1
                    if len(vetoed_region_examples) < 8:
                        caption = str(metadata.get("caption") or "")[:48]
                        vetoed_region_examples.append(
                            f"crop_{idx} ({obj_class}): {reason}"
                            + (f" — captioned '{caption}'" if caption else "")
                        )
                    self.advance_progress(task)
                    continue

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
        if vetoed_built:
            self.log_warning(
                f"  Vetoed {sum(vetoed_built.values())} crop(s) across "
                f"{len(vetoed_built)} built-environment class(es) — "
                + ", ".join(f"{c} x{n}" for c, n in sorted(vetoed_built.items(), key=lambda kv: -kv[1]))
            )

        if vetoed_region:
            self.log_warning(
                f"  Vetoed {sum(vetoed_region.values())} crop(s) standing somewhere "
                f"their class cannot be — "
                + ", ".join(f"{c} x{n}" for c, n in sorted(vetoed_region.items(), key=lambda kv: -kv[1]))
            )
            for example in vetoed_region_examples:
                self.log_info(f"    {example}")

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

        # After reassignment, so a low-confidence crop that just joined a bucket is
        # weighed with it rather than against the bucket it was about to change.
        try:
            splits = self._split_outsized(context)
        except Exception as e:
            # A bucket layout that is merely coarser is a far smaller defect than a
            # stage that fails outright and takes the whole clustering with it.
            self.log_warning(f"Could not split outsized bucket members ({type(e).__name__}: {e})")
            splits = []

        context.add_object_correlation(ContextKey.OBJECT_CORRELATION, correlation)
        self.finish_progress(task)
        self._write_debug(context, debug_by_class, reassigned, rejected, splits)
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

    def _split_outsized(self, context: PipelineContext) -> list[dict]:
        """Give a bucket's outsized member its own bucket. See outsize_split_ratio.

        Runs last, over the buckets as they finally stand (low-confidence crops
        already reassigned), and only moves 'bucket' -- so every downstream consumer
        that reads metadata_{i}['bucket'] picks the split up for free: Panorama Asset
        Generation reconstructs the new bucket from the outsized crop itself and
        curates it its own billboard pool, and Scene Generation resolves its
        instance to that mesh by the same key lookup as any other.

        Returns one debug record per split, for clustering_debug.json.
        """
        ratio = self.config.outsize_split_ratio
        if ratio <= 1.0:
            return []

        panorama = context.input_panorama(ContextKey.PANORAMA)
        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        if panorama is None or panorama_depth is None:
            return []

        from pipeline.scene_generation.projection import unproject_bbox_equirect
        from scene.camera import CameraExtrinsics

        # Identity pose: only the HEIGHT of the returned triple is read, and height is
        # measured in the camera's own frame before the extrinsics transform touches
        # the position. Using the real pose here would change nothing and would make
        # this stage depend on one it currently doesn't.
        extrinsics = CameraExtrinsics.identity()

        # (class, bucket) -> [(height_m, idx), ...], over every detection that still
        # carries a bucket at this point. Read through context.object (not
        # input_object) so this sees the buckets THIS stage just assigned, and the
        # classes _reassign_low_confidence just changed.
        groups: dict[tuple[str, int], list[tuple[float, int]]] = {}
        buckets_in_class: dict[str, set[int]] = {}
        count = context.input_object(ContextKey.OBJECT_COUNT) or 0
        for idx in range(count):
            metadata = context.object(f"metadata_{idx}")
            if not metadata:
                continue
            obj_class, bucket, box = metadata.get("class"), metadata.get("bucket"), metadata.get("box")
            if not obj_class or bucket is None or not box or obj_class == "indeterminate":
                continue
            buckets_in_class.setdefault(obj_class, set()).add(int(bucket))
            if metadata.get("position_only"):
                # Its bucket id is registered above, but it takes no part in the
                # comparison -- neither as the outsized member nor as the runner-up
                # measured against. Panorama Asset Generation declines to mesh a
                # position_only crop or to put it in a billboard pool, so a bucket
                # founded on one would hold no asset at all and its instance would
                # render nothing.
                continue
            unprojected = unproject_bbox_equirect(
                box, panorama.width, panorama.height,
                pano_depth=panorama_depth, extrinsics=extrinsics,
            )
            if unprojected is None:
                continue
            groups.setdefault((obj_class, int(bucket)), []).append((float(unprojected[2]), idx))

        splits: list[dict] = []
        for (obj_class, bucket), members in sorted(groups.items()):
            if len(members) < 2:
                continue
            members.sort(reverse=True)
            (tallest, tallest_idx), (runner_up, _) = members[0], members[1]
            if tallest < self.config.outsize_split_min_height_m:
                continue
            if runner_up <= 0 or tallest < ratio * runner_up:
                continue

            new_bucket = max(buckets_in_class[obj_class]) + 1
            buckets_in_class[obj_class].add(new_bucket)
            # Not _set_bucket: that rebuilds metadata from input_object, i.e. from
            # this crop's state BEFORE the stage ran, which would silently undo
            # _reassign_low_confidence's own writes (class, position_only,
            # visual_match_similarity). Only the bucket changes here.
            metadata = context.object(f"metadata_{tallest_idx}") or {}
            context.add_object(
                f"metadata_{tallest_idx}", {**metadata, "bucket": new_bucket}
            )
            splits.append({
                "class": obj_class,
                "from_bucket": bucket,
                "to_bucket": new_bucket,
                "idx": tallest_idx,
                "height_m": round(tallest, 2),
                "next_height_m": round(runner_up, 2),
                "ratio": round(tallest / runner_up, 2),
            })
            self.log_info(
                f"  {obj_class}::{bucket}: crop_{tallest_idx} is {tallest:.1f} m against "
                f"{runner_up:.1f} m for the next largest ({tallest / runner_up:.1f}x) — "
                f"splitting it into {obj_class}::{new_bucket}"
            )
        return splits

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

    def _write_debug(
        self, context: PipelineContext, debug_by_class: dict[str, dict],
        reassigned: int, rejected: int, splits: list[dict] | None = None,
    ):
        if self.output is None:
            return

        with open(self.output / "clustering_debug.json", "w") as f:
            json.dump({
                "cluster_distance_threshold": self.config.cluster_distance_threshold,
                "position_only_similarity_threshold": self.config.position_only_similarity_threshold,
                "max_buckets_per_class": self.config.max_buckets_per_class,
                "min_bucket_size": self.config.min_bucket_size,
                "low_confidence": {"reassigned_position_only": reassigned, "rejected": rejected},
                "outsize_split_ratio": self.config.outsize_split_ratio,
                "outsize_split_min_height_m": self.config.outsize_split_min_height_m,
                "outsize_splits": splits or [],
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
