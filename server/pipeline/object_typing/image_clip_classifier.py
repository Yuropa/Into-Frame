import math

from util.image_utils import Image, make_context_composite, flatten_alpha_with_mean_fill
from pipeline.object_typing.categories import OBJECT_CATEGORIES, ENVIRONMENT_CATEGORIES
from transformers import CLIPProcessor, CLIPModel
import torch

_OBJECT_LABELS = frozenset(OBJECT_CATEGORIES.keys())
_ALL_CATEGORIES = {**OBJECT_CATEGORIES, **ENVIRONMENT_CATEGORIES}


def _pairwise_certainty(winner_score: float, runner_up_score: float, logit_scale: float) -> float:
    """How decisively `winner_score` beats `runner_up_score`, in [0, 1].

    This is the two-way softmax between the pair, evaluated at CLIP's own
    learned `logit_scale`, then rescaled from its natural [0.5, 1] range:

        p_winner = exp(T*a) / (exp(T*a) + exp(T*b))   ->   (p - 0.5) * 2
                 = tanh(T * (a - b) / 2)

    The temperature is what makes this a meaningful measure at all. An earlier
    version compared the two *raw cosine similarities* directly, as
    `(max/(a+b) - 0.5) * 2`. CLIP's image-text cosines all live in a narrow
    band well away from zero (0.55-0.90 across a measured alpine-meadow
    capture), so that ratio is pinned near 0.5 and the rescaled value near 0
    no matter how decisive the win: clearing a 0.1 threshold would have
    required the winner to score 1.22x the runner-up, which never happens.
    Measured on that capture, 2 of 359 crops cleared the two gates below --
    i.e. every object in the scene was flagged low_confidence, every class was
    dropped by ObjectCategoryClusteringStage for want of a confident bucket to
    corroborate against, and the scene came out with 21 objects in it.

    A *difference* of cosines is the quantity that actually carries signal
    (p5 0.002, median 0.032, p95 0.099 for the object-vs-runner-up gap on that
    same capture); exponentiating it at the model's own calibrated temperature
    turns it back into a probability the thresholds can be stated against.
    """
    return math.tanh(logit_scale * (winner_score - runner_up_score) / 2.0)


class ImageClipClassifier:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        device: torch.device,
        confidence_threshold: float = 0.9,
        object_margin_threshold: float = 0.9,
        min_confident_area_fraction: float = 0.001,
        object_lead_confidence_threshold: float = 0.0,
    ):
        self.device = device
        # Both thresholds are stated on _pairwise_certainty's scale: a
        # softmax probability rescaled so 0 = a perfect tie and 1 = one-sided.
        # They sit high (0.9 == a ~0.029 raw-cosine win at ViT-B/32's
        # temperature) because that scale saturates fast by design -- see
        # _pairwise_certainty for the units, and for what the pre-temperature
        # version of these gates did to a real scene.
        # Applied ONLY when the environment pool outscores the object pool -- see
        # _sims_to_result. It answers "is the scenery reading solid enough to commit
        # to", which is a question about scenery.
        self._confidence_threshold = confidence_threshold
        # The same gate for the other direction, when the OBJECT pool leads. Separate,
        # and much lower, because the two directions are not the same question.
        #
        # object-vs-environment is a near-tie for most vegetation by construction: the
        # two pools are scored against prompts that describe the same pixels
        # ("a photo of colorful wildflowers in bloom" against "a photo of a wildflower
        # meadow"; "a photo of a cluster of trees" against "a photo of a grassy lawn or
        # meadow"). On an alpine meadow capture that tie is not ignorance, it is the
        # correct answer to a question with no answer -- and the useful reading is the
        # object one. Gating both directions on one symmetric threshold scored those
        # ties as unknown and discarded them: measured on that capture, 68 of 68 typed
        # trees came back indeterminate from the crop, every one of them then failed
        # ObjectCategoryClusteringStage's corroboration for want of a confident bucket
        # to match against, and the scene rendered zero trees.
        #
        # Back to 0 after a spell at 0.5. The 0.5 was an attempt to stop a Shark Fin
        # Cove capture placing a 38 m "lighthouse" cut from blank sky, whose crop beat
        # the environment pool by 0.0083 raw cosine with every label in the band
        # 0.25-0.29 -- CLIP returning a near-uniform distribution, read as a decision.
        # It did not work, and could not have: measured on the Rainier capture, ALL 20
        # crops that cleared these gates sat inside a 0.019-0.034 top-5 spread, the 18
        # junk ones and the 2 real ones alike. There is no threshold on a near-uniform
        # distribution that keeps the signal and drops the noise, because at that point
        # the two are the same numbers. What 0.5 did accomplish was rejecting the
        # vegetation ties this gate exists to forgive: 65 of 65 trees came back
        # indeterminate and the scene rendered none.
        #
        # The judgement it was reaching for now happens where it can actually be made:
        # ObjectTypingStage puts every proposed object label to Grounding DINO, which
        # has to localize the phrase and can decline. "Is this really a lighthouse" is
        # a question with an answer; "is a 0.008 cosine lead meaningful" is not.
        #
        # Raise toward confidence_threshold to restore the old symmetric strictness.
        self._object_lead_confidence_threshold = object_lead_confidence_threshold
        # Required gap between the winning object label's score and the
        # runner-up object label's score (same rescaled unit as `confidence`
        # below) before a specific object label is trusted.
        # Without this, `confidence` only measures object-vs-environment
        # dominance -- a crop that's a near-tie between e.g. "tree" and
        # "statue" still passed that check confidently, because both clearly
        # beat every environment prompt, even though the *choice* between
        # them was essentially noise. See classify_with_details' `area_fraction`
        # parameter for why this gate is skipped for small crops.
        self._object_margin_threshold = object_margin_threshold
        # Crop bbox area / scene area below which the object-margin gate is
        # skipped -- a handful of pixels can't earn a confident specific label,
        # so small crops get CLIP's best-effort guess instead of being pushed to
        # 'indeterminate' by a bar they could never realistically clear.
        self._min_confident_area_fraction = min_confident_area_fraction
        self.processor = CLIPProcessor.from_pretrained(self.MODEL_NAME)
        self.model = CLIPModel.from_pretrained(self.MODEL_NAME).to(device)
        self.model.eval()
        # CLIP's own learned inverse-temperature (100.0 for ViT-B/32) -- the
        # scale its contrastive objective calibrated cosine gaps against. Read
        # from the checkpoint rather than hard-coded so swapping MODEL_NAME
        # doesn't silently change what the thresholds above mean.
        self._logit_scale = float(self.model.logit_scale.exp().item())

        # Pre-compute text embeddings once — reused for every image
        all_prompts: list[str] = []
        self._category_slices: list[tuple[str, int, int]] = []
        for label, prompts in _ALL_CATEGORIES.items():
            start = len(all_prompts)
            all_prompts.extend(prompts)
            self._category_slices.append((label, start, len(all_prompts)))

        text_inputs = self.processor(text=all_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            text_out = self.model.text_model(**text_inputs)
            text_features = self.model.text_projection(text_out.pooler_output)
            self._text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    @classmethod
    def model_names(cls) -> list[str]:
        return [cls.MODEL_NAME]

    def classify(self, image: Image) -> tuple[str, float]:
        """Returns (label, confidence) — see classify_with_details for the full breakdown."""
        label, confidence, _, _criteria = self.classify_with_details(image)
        return label, confidence

    def classify_with_details(
        self,
        image: Image,
        scene_image: Image | None = None,
        box: list[float] | None = None,
        top_n: int = 5,
        area_fraction: float | None = None,
        scene_category_prior: frozenset[str] | None = None,
    ) -> tuple[str, float, list[tuple[str, float]], dict]:
        """Returns (label, confidence, top_candidates, criteria).

        label          — winning category or 'indeterminate'
        confidence     — see _pairwise_certainty: 0 = perfectly tied between the
                         best object and best environment label, 1 = one-sided
        top_candidates — top_n (label, raw_score) pairs sorted best-first
        criteria       — dict with best_obj_label, best_obj_score, best_env_label,
                         best_env_score for debugging the decision
        scene_image    — full scene for context; if provided, a composite
                         (scene thumbnail + crop) is fed to CLIP instead of the
                         bare crop, giving the model spatial/scene context
        box            — [x, y, w, h] of the crop in scene_image pixel space;
                         drawn as a red highlight on the scene thumbnail
        area_fraction  — this crop's bbox area as a fraction of the scene image's
                         area. Below `min_confident_area_fraction`, the object-margin
                         gate (see _sims_to_result) is skipped: a handful of pixels
                         can't support a confident specific label, so small crops
                         just take CLIP's best-scoring guess instead of being pushed
                         to 'indeterminate' by a margin they could never clear.
        scene_category_prior — object labels with independent corroborating evidence
                         elsewhere in the scene (see ObjectTypingStage). Used only to
                         break large-crop margin-ties: if the runner-up object label is
                         in this set and the winner isn't, the runner-up is preferred.
        """
        if scene_image is not None:
            pil_input = make_context_composite(flatten_alpha_with_mean_fill(image), scene_image.rgb(), box)
        else:
            pil_input = flatten_alpha_with_mean_fill(image)

        inputs = self.processor(images=pil_input, return_tensors="pt").to(self.device)
        with torch.no_grad():
            vision_out = self.model.vision_model(**inputs)
            image_features = self.model.visual_projection(vision_out.pooler_output)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        sims = (image_features @ self._text_features.T).squeeze(0)
        return self._sims_to_result(sims, top_n, area_fraction, scene_category_prior)

    def classify_from_caption(
        self,
        caption: str,
        top_n: int = 5,
        area_fraction: float | None = None,
        scene_category_prior: frozenset[str] | None = None,
    ) -> tuple[str, float, list[tuple[str, float]], dict]:
        """Classify using a text caption instead of an image.

        Encodes the caption with CLIP's text encoder and scores it against the
        same pre-computed category embeddings used for image classification.
        Intended as a fallback when image-based classification is indeterminate.
        """
        text_inputs = self.processor(text=[caption], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_out = self.model.text_model(**text_inputs)
            caption_features = self.model.text_projection(text_out.pooler_output)
            caption_features = caption_features / caption_features.norm(dim=-1, keepdim=True)

        sims = (caption_features @ self._text_features.T).squeeze(0)
        return self._sims_to_result(sims, top_n, area_fraction, scene_category_prior)

    def _sims_to_result(
        self,
        sims: "torch.Tensor",
        top_n: int,
        area_fraction: float | None = None,
        scene_category_prior: "frozenset[str] | None" = None,
    ) -> tuple[str, float, list[tuple[str, float]], dict]:
        best_obj_label, best_obj_score = None, -float("inf")
        second_obj_label, second_obj_score = None, -float("inf")
        best_env_label, best_env_score = None, -float("inf")
        per_label: list[tuple[str, float]] = []
        for label, start, end in self._category_slices:
            score = sims[start:end].max().item()
            per_label.append((label, score))
            if label in _OBJECT_LABELS:
                if score > best_obj_score:
                    second_obj_label, second_obj_score = best_obj_label, best_obj_score
                    best_obj_score, best_obj_label = score, label
                elif score > second_obj_score:
                    second_obj_score, second_obj_label = score, label
            else:
                if score > best_env_score:
                    best_env_score, best_env_label = score, label

        # How decisively the object pool's best beats the environment pool's
        # best (or vice versa) -- "is this a thing, or is it scenery".
        confidence = _pairwise_certainty(
            max(best_obj_score, best_env_score),
            min(best_obj_score, best_env_score),
            self._logit_scale,
        )

        # How one-sided is the winning object label against the best label that
        # ISN'T it -- from either pool. This is what `confidence` alone misses:
        # a crop can be unambiguously "an object, not scenery" while the specific
        # object label chosen is nearly a coin flip.
        #
        # The rival is deliberately not the runner-up OBJECT alone, which is what
        # this used to measure. Restricting it to the object pool means an
        # environment label sitting BETWEEN the winner and the runner-up object is
        # invisible to the gate, so the margin is earned against a label that was
        # never the real competitor. Measured on a Shark Fin Cove capture (a beach,
        # no structures of any kind), for a crop of blank sky:
        #
        #     lighthouse 0.2857   <- winner
        #     cliff      0.2774   <- best environment, IGNORED by the old margin
        #     landmark   0.2515   <- runner-up object, what the margin used
        #
        # margin against `landmark` is 0.94 and clears the 0.9 gate comfortably;
        # against `cliff` it is 0.34 and does not. Both of that scene's two
        # image-typed objects were this exact shape, and both rendered -- one as a
        # 38 m "lighthouse" billboard of empty sky standing in the ocean.
        rival_score = second_obj_score
        rival_label = second_obj_label
        if best_obj_label is not None and best_env_score > rival_score:
            rival_score, rival_label = best_env_score, best_env_label
        object_margin = (
            _pairwise_certainty(best_obj_score, rival_score, self._logit_scale)
            if rival_label is not None else 1.0
        )

        top_candidates = sorted(per_label, key=lambda x: x[1], reverse=True)[:top_n]

        criteria = {
            "best_obj_label": best_obj_label,
            "best_obj_score": round(best_obj_score, 4),
            "second_obj_label": second_obj_label,
            "second_obj_score": round(second_obj_score, 4) if second_obj_label is not None else None,
            # Which label object_margin was actually measured against -- either the
            # runner-up object or, when it outscores that, the best environment label.
            "margin_rival_label": rival_label,
            "margin_rival_score": round(rival_score, 4) if rival_label is not None else None,
            "object_margin": round(object_margin, 4),
            "best_env_label": best_env_label,
            "best_env_score": round(best_env_score, 4),
        }

        # Directional. `confidence` is symmetric -- _pairwise_certainty(max, min) --
        # so a single threshold on it rejects a near-tie whichever pool is ahead. That
        # is right for a crop leaning scenery (an unresolved "is this even a thing"
        # should not become an object) and wrong for one leaning object, where the tie
        # is usually just the two pools describing the same vegetation. See
        # _object_lead_confidence_threshold.
        leads_object = best_obj_score >= best_env_score
        gate = (
            self._object_lead_confidence_threshold if leads_object
            else self._confidence_threshold
        )
        if confidence < gate:
            return "indeterminate", confidence, top_candidates, criteria

        large_crop = area_fraction is None or area_fraction >= self._min_confident_area_fraction
        if (
            large_crop
            and best_obj_score >= best_env_score
            and object_margin < self._object_margin_threshold
        ):
            # Ambiguous winner on a crop large enough that we should be able to
            # do better. If the scene has independent evidence for the runner-up
            # but not the winner, prefer it instead of just giving up -- this is
            # what stops one oddly-shaped crop from becoming the scene's only
            # instance of a category nothing else in the panorama supports.
            if (
                scene_category_prior is not None
                and second_obj_label is not None
                and second_obj_label in scene_category_prior
                and best_obj_label not in scene_category_prior
            ):
                return second_obj_label, confidence, top_candidates, criteria
            return "indeterminate", confidence, top_candidates, criteria

        if best_obj_score >= best_env_score:
            return best_obj_label, confidence, top_candidates, criteria
        return best_env_label, confidence, top_candidates, criteria
