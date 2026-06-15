import json
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.object_typing.image_clip_classifier import ImageClipClassifier
from pipeline.pipeline_context import PipelineContext, ContextKey


class ObjectTypingStage(PipelineStage):
    """
    Assigns a fine-grained 'class' to every crop using CLIP zero-shot classification
    (openai/clip-vit-base-patch32), overwriting the caption-based pre-filter value
    with a more reliable label (e.g. 'car', 'tree', 'sky'). Covers both object and
    environment categories so all crops get a meaningful class regardless of what the
    upstream classification stage decided.

    Text embeddings for all categories are computed once at load time;
    per-image cost is a single forward pass plus cosine similarity.

    Each run writes typing_debug.json to the stage output directory with per-crop
    CLIP scores and top candidates for every indeterminate result.

    Reads:  ContextKey.OBJECT_COUNT, crop_{i}, metadata_{i}, ContextKey.INPUT (scene context)
    Writes: metadata_{i} (updates 'class' and 'confidence' fields)
    Debug:  typing_debug.json
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._classifier = None

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to type, skipping")
            return context

        typing_task = self.create_progress(object_count + 1, "Typing objects…")
        if self._classifier is None:
            self._classifier = ImageClipClassifier(self.device)
        self.advance_progress(typing_task)

        debug_entries = []
        for idx in range(object_count):
            metadata = context.input_object(f"metadata_{idx}") or {}

            crop = context.input_image(f"crop_{idx}")
            if crop is None:
                self.advance_progress(typing_task)
                continue

            obj_type, confidence, top, criteria = self._classifier.classify_with_details(crop)
            context.add_object(f"metadata_{idx}", {**metadata, "class": obj_type, "confidence": round(confidence, 4)})
            self.log_info(f"  crop_{idx}: '{metadata.get('caption', '')}' → {obj_type} ({confidence:.2f})")

            debug_entries.append({
                "idx": idx,
                "caption": metadata.get("caption", ""),
                "class": obj_type,
                "confidence": round(confidence, 4),
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
        payload = {
            "confidence_threshold": threshold,
            "summary": {"total": len(entries), "indeterminate": indet},
            "objects": entries,
        }
        with open(self.output / "typing_debug.json", "w") as f:
            json.dump(payload, f, indent=2)

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            return False
        return all(context.has_stage_output(f"metadata_{i}") for i in range(count))

    def model_names(self) -> list[str]:
        return ImageClipClassifier.model_names()

    def clean_up(self):
        self._classifier = None
        super().clean_up()
