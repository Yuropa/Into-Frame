import json
from logging import Logger
from typing import Any
import torch
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.object_typing.image_clip_classifier import ImageClipClassifier
from pipeline.object_typing.categories import OBJECT_CATEGORIES
from pipeline.object_detection.grounding_dino import GroundingDino
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.image_utils import flatten_alpha_with_mean_fill


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
        object_lead_confidence_threshold: float = 0.0,
        verify_labels: bool = True,
        verify_threshold: float = 0.25,
        verify_min_box_fraction: float = 0.1,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.confidence_threshold = confidence_threshold
        self.object_margin_threshold = object_margin_threshold
        self.min_confident_area_fraction = min_confident_area_fraction
        self.object_lead_confidence_threshold = object_lead_confidence_threshold
        self.verify_labels = bool(verify_labels)
        self.verify_threshold = float(verify_threshold)
        self.verify_min_box_fraction = float(verify_min_box_fraction)


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

    Every crop that lands on an OBJECT category is then put to an independent model:
    Grounding DINO is asked to find that specific label in that specific crop, and
    'low_confidence' is set from whether it could (see _verify_labels). CLIP proposes
    the label; it does not also get to certify it. That separation is the point --
    CLIP always returns a ranking over its fixed label set, so a capture it has no
    signal on still produces winners, and those winners used to be indistinguishable
    from real ones by any threshold available here.

    Reads:  ContextKey.OBJECT_COUNT, crop_{i}, metadata_{i}, ContextKey.PANORAMA (scene context)
    Writes: metadata_{i} (updates 'class', 'confidence', and 'low_confidence' fields --
            the latter true when the label failed Grounding DINO corroboration, or,
            with verify_labels off, when the crop pixels alone didn't clear
            confidence_threshold; see ObjectCategoryClusteringStage for how this gates
            downstream trust)
    Debug:  typing_debug.json — carries the image pass's own verdict and scores
            ('image_class' / 'image_criteria') alongside whichever pass won, so a crop
            that fell back still records why the image path rejected it.
    Config: confidence_threshold (default 0.9), object_margin_threshold (default 0.9),
            min_confident_area_fraction (default 0.001),
            object_lead_confidence_threshold (default 0.0 — the confidence gate for
            crops where the OBJECT pool leads; see ImageClipClassifier),
            verify_labels (default True), verify_threshold (default 0.25),
            verify_min_box_fraction (default 0.1)

            The first two gates are stated on ImageClipClassifier._pairwise_certainty's
            scale (a temperature-calibrated softmax probability, rescaled so
            0 = tied and 1 = one-sided), which saturates fast -- hence the high
            defaults. They are NOT comparable to the pre-temperature values
            these replaced; see _pairwise_certainty for why those were
            unreachable. verify_threshold is a Grounding DINO box score and shares
            no scale with them.
    """

    @classmethod
    def config_class(cls):
        return ObjectTypingConfiguration

    def __init__(self, config: ObjectTypingConfiguration) -> None:
        super().__init__(config)
        self._classifier = None
        self._verifier = None

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
            # What the crop pixels alone said, before any fallback overwrites it. Kept
            # because the fallback paths below replace `criteria` wholesale with the
            # caption pass's scores, and without this the debug record for a fallen-back
            # crop has no trace of why the image path rejected it -- which is exactly
            # the question asked when a scene comes out missing a whole class.
            image_class = obj_type
            image_criteria = dict(criteria)
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
            prior_fallback = False
            if obj_type == "indeterminate" and prior_class != "indeterminate":
                obj_type = prior_class
                caption_fallback = False
                prior_fallback = True

            context.add_object(f"metadata_{idx}", {
                **metadata, "class": obj_type, "confidence": round(confidence, 4),
                # Provisional. Whether this crop is trusted is settled by the
                # verification pass after the loop, which is the only step here that
                # asks an independent model. Set now so a run with verification
                # disabled still has the old CLIP-only meaning of the flag.
                "low_confidence": image_class == "indeterminate" and obj_type != "indeterminate",
            })
            suffix = (
                " [caption fallback]" if caption_fallback
                else " [prior fallback]" if prior_fallback else ""
            )
            self.log_info(f"  crop_{idx}: '{caption}' → {obj_type} ({confidence:.2f}){suffix}")

            debug_entries.append({
                "idx": idx,
                "caption": caption,
                "class": obj_type,
                "confidence": round(confidence, 4),
                "caption_fallback": caption_fallback,
                "prior_fallback": prior_fallback,
                # The image pass's own verdict and scores, whether or not it won.
                "image_class": image_class,
                "image_criteria": image_criteria,
                **criteria,
                "top_candidates": [[lbl, round(sc, 4)] for lbl, sc in top],
            })

            self.advance_progress(typing_task)

        self.finish_progress(typing_task)
        self._verify_labels(context, debug_entries)
        self._write_debug(debug_entries)
        return context

    def _verify_labels(self, context: PipelineContext, debug_entries: list) -> None:
        """Settle `low_confidence` by asking an independent model to find each label.

        CLIP proposes; this disposes. The reason the proposal cannot also be the
        verdict is that CLIP scores a fixed label set and always returns a ranking,
        so on a capture where it has no signal the ranking is near-uniform noise
        that still has a winner. Measured on the Rainier capture: of 359 crops, the
        20 that survived CLIP's own gates ALL scored inside a 0.019-0.034 top-5
        spread -- the 18 junk `other` crops and the 2 real flowers were
        indistinguishable, so no threshold on that spread could separate them. 18
        junk crops became the scene's only confident category while all 65 trees,
        having failed the same gates, were dropped for want of one to corroborate
        against.

        Grounding DINO is a different question with a different failure mode: it has
        to localize the phrase it is given, and it can return nothing. Asked for
        "lighthouse" on a crop of empty sky it declines; asked for "tree" on a
        conifer it puts a box round the conifer.

        Only object categories are verified. Environment labels (sky, water,
        mountain) are not placed as objects, so their trust flag changes nothing
        downstream, and an open-vocabulary detector is a poor judge of "sky" anyway --
        it is trained to find things, not to describe the backdrop they sit against.
        """
        if not self.config.verify_labels:
            return

        candidates = [
            entry for entry in debug_entries
            if entry["class"] in OBJECT_CATEGORIES and entry["class"] != "indeterminate"
        ]
        if not candidates:
            self.log_info("Label verification: no object-class crops to verify")
            return

        crops = []
        kept: list[dict] = []
        for entry in candidates:
            crop = context.input_image(f"crop_{entry['idx']}")
            if crop is None:
                continue
            crops.append({"image": flatten_alpha_with_mean_fill(crop), "label": entry["class"]})
            kept.append(entry)
        if not crops:
            return

        task = self.create_progress(1, "Verifying labels…")
        if self._verifier is None:
            self._verifier = GroundingDino(self.device)
        verdicts = self._verifier.verify(
            crops,
            self.temp or self.output,
            threshold=self.config.verify_threshold,
            min_box_fraction=self.config.verify_min_box_fraction,
            on_progress=self.make_progress_callback(task),
        )
        self.finish_progress(task)

        verified_by_class: dict[str, list[int]] = {}
        for entry, verdict in zip(kept, verdicts):
            entry["verification"] = {
                "score": round(float(verdict["score"]), 4),
                "box_fraction": round(float(verdict["box_fraction"]), 4),
                "verified": bool(verdict["verified"]),
            }
            # context.object(), NOT context.input_object() -- the latter reads the
            # PREVIOUS stage's value, so merging into it here would write back the
            # pre-typing class and confidence and silently undo this stage's own work
            # for every crop that got verified.
            metadata = context.object(f"metadata_{entry['idx']}") or {}
            context.add_object(f"metadata_{entry['idx']}", {
                **metadata, "low_confidence": not verdict["verified"],
            })
            counts = verified_by_class.setdefault(entry["class"], [0, 0])
            counts[1] += 1
            if verdict["verified"]:
                counts[0] += 1

        total_ok = sum(c[0] for c in verified_by_class.values())
        self.log_info(
            f"Label verification: {total_ok}/{len(kept)} crop(s) corroborated "
            f"(score >= {self.config.verify_threshold}, coverage >= "
            f"{self.config.verify_min_box_fraction})"
        )
        for cls, (ok, total) in sorted(verified_by_class.items(), key=lambda kv: -kv[1][1]):
            self.log_info(f"    {cls}: {ok}/{total}")

        self._log_verification_diagnostics(kept)

    def _log_verification_diagnostics(self, kept: list[dict]) -> None:
        """Everything needed to retune verify_threshold / verify_min_box_fraction.

        Both defaults are estimates -- they were reasoned from Grounding DINO's own
        detection threshold rather than measured, because the capture that motivated
        them could not be re-run where they were written. So the run itself has to
        carry the evidence for moving them, and the useful evidence is not the pass
        rate (which says only that SOME line was drawn) but the score distribution
        either side of it, and which of the two conditions did the rejecting.

        A class that fails wholesale with scores bunched just under the threshold is
        a threshold that is too high. One that fails on coverage with strong scores
        is a mask bounding-box problem, not a labelling one. Those need different
        fixes and are indistinguishable from a pass rate alone.
        """
        def pct(values: list[float], q: float) -> float:
            if not values:
                return 0.0
            ordered = sorted(values)
            return ordered[min(int(q * len(ordered)), len(ordered) - 1)]

        score_t, frac_t = self.config.verify_threshold, self.config.verify_min_box_fraction

        by_class: dict[str, list[dict]] = {}
        for entry in kept:
            by_class.setdefault(entry["class"], []).append(entry)

        self.log_info("  Verification score distribution (p10 / median / p90):")
        for cls, entries in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
            scores = [e["verification"]["score"] for e in entries]
            fracs = [e["verification"]["box_fraction"] for e in entries]
            self.log_info(
                f"    {cls:<14} n={len(entries):<4} "
                f"score {pct(scores, 0.1):.3f} / {pct(scores, 0.5):.3f} / {pct(scores, 0.9):.3f}   "
                f"coverage {pct(fracs, 0.1):.3f} / {pct(fracs, 0.5):.3f} / {pct(fracs, 0.9):.3f}"
            )

        # Which condition rejected each failure. A crop can fail both.
        failed = [e for e in kept if not e["verification"]["verified"]]
        if failed:
            score_only = sum(
                1 for e in failed
                if e["verification"]["score"] < score_t
                and e["verification"]["box_fraction"] >= frac_t
            )
            coverage_only = sum(
                1 for e in failed
                if e["verification"]["score"] >= score_t
                and e["verification"]["box_fraction"] < frac_t
            )
            both = len(failed) - score_only - coverage_only
            self.log_info(
                f"  Rejections by cause: {score_only} score-only, "
                f"{coverage_only} coverage-only, {both} both"
            )

        # Near-misses, both directions. These are the crops whose verdict a small
        # move in either threshold would flip, so they are the ones to look at
        # before moving one -- and the ones to eyeball in the panorama to decide
        # which way is right.
        def near(entry: dict) -> bool:
            v = entry["verification"]
            return (
                (not v["verified"] and v["score"] >= score_t * 0.6
                 and v["box_fraction"] >= frac_t * 0.6)
                or (v["verified"] and (v["score"] < score_t * 1.4
                                       or v["box_fraction"] < frac_t * 1.4))
            )

        near_misses = sorted(
            (e for e in kept if near(e)),
            key=lambda e: -e["verification"]["score"],
        )
        if near_misses:
            self.log_info(
                f"  {len(near_misses)} borderline crop(s) (within ~40% of a threshold) "
                f"— these flip first if either is moved:"
            )
            for entry in near_misses[:20]:
                v = entry["verification"]
                self.log_info(
                    f"    crop_{entry['idx']:<4} {entry['class']:<14} "
                    f"score {v['score']:.3f} coverage {v['box_fraction']:.3f} "
                    f"→ {'PASS' if v['verified'] else 'fail'}   '{entry['caption'][:48]}'"
                )
            if len(near_misses) > 20:
                self.log_info(f"    … {len(near_misses) - 20} more in typing_debug.json")

        # A class that proposed many crops and corroborated none is the shape of the
        # regression this whole pass exists to catch (65 trees typed, 0 trusted, none
        # rendered). Say so loudly rather than leaving it to be inferred from a table.
        for cls, entries in sorted(by_class.items(), key=lambda kv: -len(kv[1])):
            if len(entries) >= 5 and not any(e["verification"]["verified"] for e in entries):
                best = max(e["verification"]["score"] for e in entries)
                self.log_warning(
                    f"  Class '{cls}': {len(entries)} crop(s) proposed, NONE corroborated "
                    f"(best score {best:.3f} vs threshold {score_t}). This class will be "
                    f"dropped entirely by Object Category Clustering."
                )

    def _write_debug(self, entries: list):
        if self.output is None:
            return
        threshold = self._classifier._confidence_threshold if self._classifier else None
        indet = sum(1 for e in entries if e["class"] == "indeterminate")
        caption_fallbacks = sum(1 for e in entries if e.get("caption_fallback"))
        prior_fallbacks = sum(1 for e in entries if e.get("prior_fallback"))
        verified = [e for e in entries if "verification" in e]
        payload = {
            "confidence_threshold": threshold,
            "summary": {
                "total": len(entries),
                "indeterminate": indet,
                "caption_fallbacks": caption_fallbacks,
                "prior_fallbacks": prior_fallbacks,
                # Crops CLIP typed straight from the pixels, before any fallback --
                # the count whose collapse to 20-of-359 hid a whole missing class.
                "image_typed": sum(
                    1 for e in entries if e.get("image_class") not in (None, "indeterminate")
                ),
                "verification_attempted": len(verified),
                "verification_passed": sum(
                    1 for e in verified if e["verification"]["verified"]
                ),
            },
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
        return ImageClipClassifier.model_names() + GroundingDino.model_names()

    def clean_up(self):
        self._classifier = None
        if self._verifier is not None:
            self._verifier.close()
            self._verifier = None
        super().clean_up()
