from util.image_utils import Image, make_context_composite, flatten_alpha_with_mean_fill
from pipeline.object_typing.categories import OBJECT_CATEGORIES, ENVIRONMENT_CATEGORIES
from transformers import CLIPProcessor, CLIPModel
import torch

_OBJECT_LABELS = frozenset(OBJECT_CATEGORIES.keys())
_ALL_CATEGORIES = {**OBJECT_CATEGORIES, **ENVIRONMENT_CATEGORIES}


class ImageClipClassifier:
    MODEL_NAME = "openai/clip-vit-base-patch32"

    def __init__(
        self,
        device: torch.device,
        confidence_threshold: float = 0.1,
        object_margin_threshold: float = 0.08,
        min_confident_area_fraction: float = 0.01,
    ):
        self.device = device
        self._confidence_threshold = confidence_threshold
        # Required gap between the winning object label's score and the
        # runner-up object label's score (same [0, 1]-rescaled unit as
        # `confidence` below) before a specific object label is trusted.
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
        confidence     — rescaled to [0, 1]: 0 = perfectly tied, 1 = one-sided
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

        obj_s = max(0.0, best_obj_score)
        env_s = max(0.0, best_env_score)
        total = obj_s + env_s
        # Raw ratio is in [0.5, 1.0]; rescale to [0, 1] so the threshold is meaningful.
        confidence = (max(obj_s, env_s) / total - 0.5) * 2.0 if total > 0 else 0.0

        # Same rescaling as `confidence` above, but within the object pool only:
        # how one-sided is the winning object label against its runner-up. This
        # is what `confidence` alone misses -- a crop can be unambiguously
        # "an object, not scenery" while the specific object label chosen is
        # nearly a coin flip.
        second_s = max(0.0, second_obj_score) if second_obj_label is not None else 0.0
        obj_total = obj_s + second_s
        object_margin = (obj_s / obj_total - 0.5) * 2.0 if obj_total > 0 else 1.0

        top_candidates = sorted(per_label, key=lambda x: x[1], reverse=True)[:top_n]

        criteria = {
            "best_obj_label": best_obj_label,
            "best_obj_score": round(best_obj_score, 4),
            "second_obj_label": second_obj_label,
            "second_obj_score": round(second_obj_score, 4) if second_obj_label is not None else None,
            "object_margin": round(object_margin, 4),
            "best_env_label": best_env_label,
            "best_env_score": round(best_env_score, 4),
        }

        if confidence < self._confidence_threshold:
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
