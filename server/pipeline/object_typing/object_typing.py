from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.object_typing.image_clip_classifier import ImageClipClassifier
from pipeline.pipeline_context import PipelineContext, ContextKey


class ObjectTypingStage(PipelineStage):
    """
    Assigns a fine-grained type label to each crop already classified as 'object'
    using CLIP zero-shot image classification (openai/clip-vit-base-patch32).

    Skips crops whose metadata 'class' is 'environment'. Text embeddings for all
    categories are computed once at load time; per-image cost is a single forward
    pass plus cosine similarity.

    Reads:  ContextKey.OBJECT_COUNT, crop_{i}, metadata_{i} (with 'class' field)
    Writes: metadata_{i} (adds 'type' field, e.g. "car", "building", "person")
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._classifier = None

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to type, skipping")
            return context

        typing_task = self.create_progress(object_count + 1, "Typing objects...")
        if self._classifier is None:
            self._classifier = ImageClipClassifier(self.device)
        self.advance_progress(typing_task)

        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}

            if metadata.get("class") != "object":
                self.advance_progress(typing_task)
                continue

            crop = context.input_image(f"crop_{idx}")
            if crop is None:
                self.advance_progress(typing_task)
                continue

            obj_type, obj_class, confidence = self._classifier.classify(crop)
            context.add_object(f"metadata_{idx}", {**metadata, "type": obj_type, "class": obj_class})
            self.log_info(f"  crop_{idx}: {metadata.get('caption', '')} → {obj_type} ({obj_class}, {confidence:.2f})")
            self.advance_progress(typing_task)

        self.finish_progress(typing_task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            return False
        for i in range(count):
            meta = context.object(f"metadata_{i}") or {}
            if meta.get("class") == "object" and meta.get("type") is None:
                return False
        return True

    def model_names(self) -> list[str]:
        return ImageClipClassifier.model_names()

    def clean_up(self):
        self._classifier = None
        super().clean_up()
