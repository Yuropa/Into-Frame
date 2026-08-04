import json
from logging import Logger
from typing import Any
import torch
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.object_typing.image_clip_classifier import ImageClipClassifier
from pipeline.pipeline_context import PipelineContext, ContextKey


class ObjectTypingConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        confidence_threshold: float = 0.9,
        object_margin_threshold: float = 0.9,
        min_confident_area_fraction: float = 0.001,
        object_lead_confidence_threshold: float = 0.5,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.confidence_threshold = confidence_threshold
        self.object_margin_threshold = object_margin_threshold
        self.min_confident_area_fraction = min_confident_area_fraction
        self.object_lead_confidence_threshold = object_lead_confidence_threshold


class ObjectTypingStage(PipelineStage):
    """
    Assigns a fine-grained 'class' to every crop using CLIP zero-shot classification
    (openai/clip-vit-base-patch32), overwriting the caption-based pre-filter value
    with a more reliable label (e.g. 'car', 'tree', 'sky'). Covers both object and
    environment categories so all crops get a meaningful class regardless of what the
    upstream classification stage decided.

    Each crop is classified as a scene-context composite (panorama thumbnail with the
    crop's box highlighted, plus the crop itself — see make_context_composite) rather
    than the bare crop, and gated by both an object-vs-environment confidence check and
    a margin check between the winning and runner-up object labels (see
    ImageClipClassifier._sims_to_result) — the margin check is skipped for crops below
    `min_confident_area_fraction` of the panorama, since a handful of pixels can't earn
    a specific label either way; they just take CLIP's best-effort guess. Ambiguous
    large-crop ties are broken using a scene-level category prior (labels the earlier
    BLIP+keyword pass found anywhere else in the scene).

    Text embeddings for all categories are computed once at load time;
    per-image cost is a single forward pass plus cosine similarity.

    Each run writes typing_debug.json to the stage output directory with per-crop
    CLIP scores, object-margin, and top candidates for every crop.

    Reads:  ContextKey.OBJECT_COUNT, crop_{i}, metadata_{i}, ContextKey.PANORAMA (scene context)
    Writes: metadata_{i} (updates 'class', 'confidence', and 'low_confidence' fields --
            the latter true whenever the real crop pixels alone didn't clear
            confidence_threshold, i.e. the class came from a caption or prior-class
            fallback rather than the image itself; see ObjectCategoryClusteringStage
            for how this gates downstream trust)
    Debug:  typing_debug.json
    Config: confidence_threshold (default 0.9), object_margin_threshold (default 0.9),
            min_confident_area_fraction (default 0.001),
            object_lead_confidence_threshold (default 0.5 — the confidence gate for
            crops where the OBJECT pool leads; see ImageClipClassifier)

            Both gates are stated on ImageClipClassifier._pairwise_certainty's
            scale (a temperature-calibrated softmax probability, rescaled so
            0 = tied and 1 = one-sided), which saturates fast -- hence the high
            defaults. They are NOT comparable to the pre-temperature values
            these replaced; see _pairwise_certainty for why those were
            unreachable.
    """

    @classmethod
    def config_class(cls):
        return ObjectTypingConfiguration

    def __init__(self, config: ObjectTypingConfiguration) -> None:
        super().__init__(config)
        self._classifier = None

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to type, skipping")
            return context

        typing_task = self.create_progress(object_count + 1, "Typing objects…")
        if self._classifier is None:
            self._classifier = ImageClipClassifier(
                self.device,
                confidence_threshold=self.config.confidence_threshold,
                object_margin_threshold=self.config.object_margin_threshold,
                min_confident_area_fraction=self.config.min_confident_area_fraction,
                object_lead_confidence_threshold=self.config.object_lead_confidence_threshold,
            )
        self.advance_progress(typing_task)

        panorama = context.input_panorama(ContextKey.PANORAMA)
        pano_area = float(panorama.width * panorama.height) if panorama is not None else None

        # Scene-level category prior: labels PanoramaObjectClassificationStage's
        # BLIP+keyword pass already found *somewhere* in this scene, before this
        # stage's own CLIP pass overwrites 'class' below. Used only to break
        # large-crop margin-ties (see ImageClipClassifier._sims_to_result) --
        # it's what stops a single oddly-shaped crop from becoming the scene's
        # only "statue" when nothing else in the panorama supports it.
        scene_category_prior = frozenset(
            cls
            for idx in range(object_count)
            if (cls := (context.input_object(f"metadata_{idx}") or {}).get("class")) not in (None, "indeterminate")
        )

        debug_entries = []
        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}

            crop = context.input_image(f"crop_{idx}")
            if crop is None:
                self.advance_progress(typing_task)
                continue

            box = metadata.get("box")
            area_fraction = (box[2] * box[3]) / pano_area if box is not None and pano_area else None

            obj_type, confidence, top, criteria = self._classifier.classify_with_details(
                crop,
                scene_image=panorama,
                box=box,
                area_fraction=area_fraction,
                scene_category_prior=scene_category_prior,
            )
            # The real crop pixels alone didn't clear the confidence bar -- whatever
            # class this ends up as (caption fallback, prior-class fallback) is an
            # unverified guess, not evidence this is actually a distinct object.
            # ObjectCategoryClusteringStage uses this to require independent visual
            # (DINOv2 embedding) corroboration against a confidently-classified crop
            # before trusting it for anything beyond contributing a position sample.
            low_confidence = obj_type == "indeterminate"
            caption = metadata.get("caption", "")
            caption_fallback = False
            if obj_type == "indeterminate" and caption:
                obj_type, confidence, top, criteria = self._classifier.classify_from_caption(
                    caption,
                    area_fraction=area_fraction,
                    scene_category_prior=scene_category_prior,
                )
                caption_fallback = True

            # Don't clobber a valid prior classification with indeterminate — keep
            # the existing class as a fallback so correlation has something to work with.
            prior_class = metadata.get("class", "indeterminate")
            if obj_type == "indeterminate" and prior_class != "indeterminate":
                obj_type = prior_class
                caption_fallback = False

            context.add_object(f"metadata_{idx}", {
                **metadata, "class": obj_type, "confidence": round(confidence, 4),
                "low_confidence": low_confidence and obj_type != "indeterminate",
            })
            suffix = " [caption fallback]" if caption_fallback else ""
            self.log_info(f"  crop_{idx}: '{caption}' → {obj_type} ({confidence:.2f}){suffix}")

            debug_entries.append({
                "idx": idx,
                "caption": caption,
                "class": obj_type,
                "confidence": round(confidence, 4),
                "caption_fallback": caption_fallback,
                **criteria,
                "top_candidates": [[lbl, round(sc, 4)] for lbl, sc in top],
            })

            self.advance_progress(typing_task)

        self.finish_progress(typing_task)
        self._write_debug(debug_entries)
        return context

    def _write_debug(self, entries: list):
        if self.output is None:
            return
        threshold = self._classifier._confidence_threshold if self._classifier else None
        indet = sum(1 for e in entries if e["class"] == "indeterminate")
        caption_fallbacks = sum(1 for e in entries if e.get("caption_fallback"))
        payload = {
            "confidence_threshold": threshold,
            "summary": {"total": len(entries), "indeterminate": indet, "caption_fallbacks": caption_fallbacks},
            "objects": entries,
        }
        with open(self.output / "typing_debug.json", "w") as f:
            json.dump(payload, f, indent=2)

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            # See PanoramaObjectClassificationStage.has_expected_output's own
            # comment -- no OBJECT_COUNT anywhere upstream means Object
            # Segmentation is disabled (a permanent state, not "pending"), so
            # there's nothing to type and never will be. Treating that as a
            # cache miss forces this stage -- and everything after it, via
            # the dirty cascade -- to rerun on every single invocation.
            return True
        return all(context.has_stage_output(f"metadata_{i}") for i in range(count))

    def model_names(self) -> list[str]:
        return ImageClipClassifier.model_names()

    def clean_up(self):
        self._classifier = None
        super().clean_up()
