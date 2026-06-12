from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import OBJECT_CATEGORIES, ENVIRONMENT_CATEGORIES

_ENVIRONMENT_KEYWORDS = frozenset(ENVIRONMENT_CATEGORIES.keys())
_OBJECT_KEYWORDS = frozenset(OBJECT_CATEGORIES.keys())


def _classify_caption(caption: str, confidence_threshold: float = 0.75) -> str:
    """Classify a free-form caption as 'object', 'environment', or 'indeterminate'.

    Confidence = max(obj_score, env_score) / total_keywords. Returns 'indeterminate'
    when no keywords match or the winning side doesn't clear the threshold."""
    words = set(caption.lower().replace(",", " ").replace(".", " ").replace("'", " ").split())
    obj_score = sum(1 for kw in _OBJECT_KEYWORDS if kw in words)
    env_score = sum(1 for kw in _ENVIRONMENT_KEYWORDS if kw in words)
    total = obj_score + env_score
    if total == 0:
        return "indeterminate"
    confidence = max(obj_score, env_score) / total
    if confidence < confidence_threshold:
        return "indeterminate"
    return "object" if obj_score > env_score else "environment"


class PanoramaObjectClassificationConfig(PipelineStageConfiguration):
    def __init__(self, confidence_threshold: float = 0.75, **kwargs):
        super().__init__(**kwargs)
        self.confidence_threshold = confidence_threshold


class PanoramaObjectClassificationStage(PipelineStage):
    """
    Classifies each panorama object crop as 'environment' or 'object' using
    BLIP captioning + keyword matching. Updates metadata_{i} in-place with
    'class' ('object'|'environment') and 'caption' (str) fields.

    Environment-classified crops are preserved in the context but skipped
    by SceneGenerationStage so they don't appear as scene objects.

    Reads:  ContextKey.OBJECT_COUNT, crop_{i}, metadata_{i}
    Writes: metadata_{i} (updated with 'class' and 'caption')
    """

    @classmethod
    def config_class(cls):
        return PanoramaObjectClassificationConfig

    def __init__(self, config: PanoramaObjectClassificationConfig) -> None:
        super().__init__(config)
        self._captioner = None
        self._confidence_threshold = config.confidence_threshold

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to classify, skipping")
            return context

        context.add_object(ContextKey.OBJECT_COUNT, object_count)

        classify_task = self.create_progress(object_count + 1, "Classifying objects…")
        if self._captioner is None:
            self._captioner = ImageCaptioning(self.device)
        self.advance_progress(classify_task)

        counts = {"object": 0, "environment": 0, "indeterminate": 0}
        for idx in range(object_count):
            crop = context.input_image(f"crop_{idx}")
            metadata = context.input_object(f"metadata_{idx}") or {}

            if crop is None:
                context.add_object(f"metadata_{idx}", metadata)
                self.advance_progress(classify_task)
                continue

            context.add_image(f"crop_{idx}", crop)
            caption = self._captioner.caption(crop)
            cls = _classify_caption(caption, self._confidence_threshold)
            counts[cls] = counts.get(cls, 0) + 1

            context.add_object(f"metadata_{idx}", {**metadata, "class": cls, "caption": caption})
            self.log_info(f"  crop_{idx}: '{caption}' → {cls}")
            self.advance_progress(classify_task)

        self.finish_progress(classify_task)
        self.log_info(
            f"Classification complete: {counts['object']} objects, "
            f"{counts['environment']} environment, {counts['indeterminate']} indeterminate"
        )
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            return False
        return all(
            (context.object(f"metadata_{i}") or {}).get("class") is not None
            for i in range(count)
        )

    def model_names(self) -> list[str]:
        return ImageCaptioning.model_names()

    def clean_up(self):
        self._captioner = None
        super().clean_up()
