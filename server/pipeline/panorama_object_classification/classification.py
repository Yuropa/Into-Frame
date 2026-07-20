import json
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_typing.categories import OBJECT_CATEGORIES, ENVIRONMENT_CATEGORIES

_ENVIRONMENT_KEYWORDS = frozenset(ENVIRONMENT_CATEGORIES.keys())
_OBJECT_KEYWORDS = frozenset(OBJECT_CATEGORIES.keys())


def _classify_caption(caption: str) -> tuple[str, dict]:
    """Return (label, debug) where label is the best keyword or 'indeterminate'.

    debug contains obj_matches, env_matches, and confidence for JSON output."""
    words = set(caption.lower().replace(",", " ").replace(".", " ").replace("'", " ").split())
    obj_matches = [kw for kw in _OBJECT_KEYWORDS if kw in words]
    env_matches = [kw for kw in _ENVIRONMENT_KEYWORDS if kw in words]
    obj_score = len(obj_matches)
    env_score = len(env_matches)
    total = obj_score + env_score
    debug = {"obj_matches": obj_matches, "env_matches": env_matches, "confidence": 0.0}
    if total == 0:
        return "indeterminate", debug
    confidence = max(obj_score, env_score) / total
    debug["confidence"] = confidence
    return (obj_matches[0] if obj_score >= env_score else env_matches[0]), debug


class PanoramaObjectClassificationConfig(PipelineStageConfiguration):
    pass


class PanoramaObjectClassificationStage(PipelineStage):
    """
    Assigns a coarse class to each panorama crop using BLIP captioning + keyword
    matching against OBJECT_CATEGORIES and ENVIRONMENT_CATEGORIES keys. Writes
    'class' (a fine-grained keyword like 'tree' or 'sky', or 'indeterminate')
    and 'caption' (str) to metadata_{i}.

    BLIP receives a context composite (scene thumbnail with the object highlighted
    on top, crop below) rather than the bare crop, giving it spatial context.

    This is a fast first-pass pre-filter. ObjectTypingStage (CLIP) runs after and
    overwrites 'class' with a more reliable fine-grained result for all crops.

    Reads:  ContextKey.OBJECT_COUNT, crop_{i}, metadata_{i}, ContextKey.INPUT (scene context)
    Writes: metadata_{i} (updated with 'class' and 'caption')
    Debug:  classification_debug.json
    """

    @classmethod
    def config_class(cls):
        return PanoramaObjectClassificationConfig

    def __init__(self, config: PanoramaObjectClassificationConfig) -> None:
        super().__init__(config)
        self._captioner = None

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

        obj_count = env_count = indet_count = 0
        debug_entries = []
        for idx in range(object_count):
            crop = context.input_image(f"crop_{idx}")
            metadata = context.input_object(f"metadata_{idx}") or {}

            if crop is None:
                context.add_object(f"metadata_{idx}", metadata)
                self.advance_progress(classify_task)
                continue

            context.add_image(f"crop_{idx}", crop)
            caption = self._captioner.caption(crop)
            cls, debug = _classify_caption(caption)

            if cls == "indeterminate":
                indet_count += 1
            elif cls in _OBJECT_KEYWORDS:
                obj_count += 1
            else:
                env_count += 1

            self.log_info(f"  crop_{idx}: '{caption}' → {cls}")
            context.add_object(f"metadata_{idx}", {**metadata, "class": cls, "caption": caption})

            debug_entries.append({
                "idx": idx,
                "caption": caption,
                "class": cls,
                "caption_confidence": round(debug["confidence"], 4),
                "obj_matches": debug["obj_matches"],
                "env_matches": debug["env_matches"],
            })

            self.advance_progress(classify_task)

        self.finish_progress(classify_task)
        self.log_info(
            f"Classification complete: {obj_count} objects, "
            f"{env_count} environment, {indet_count} indeterminate"
        )
        self._write_debug(debug_entries, obj_count, env_count, indet_count)
        return context

    def _write_debug(self, entries: list, obj_count: int, env_count: int, indet_count: int):
        if self.output is None:
            return
        payload = {
            "summary": {"objects": obj_count, "environment": env_count, "indeterminate": indet_count},
            "objects": entries,
        }
        with open(self.output / "classification_debug.json", "w") as f:
            json.dump(payload, f, indent=2)

    def has_expected_output(self, context: PipelineContext) -> bool:
        count = context.input_object(ContextKey.OBJECT_COUNT)
        if count is None:
            # No object count anywhere upstream -- either Object Segmentation
            # is disabled (permanently the case, not "not yet run"; see
            # config.yaml's enabled: false) or a genuinely fresh pipeline with
            # nothing cached yet. The latter never actually reaches this
            # return value: _run_pipeline's dirty flag is already forced True
            # from the very first stage on a truly empty cache, which makes
            # this method's result irrelevant (force short-circuits it). So
            # in every case this return value actually matters, there is
            # nothing to classify and nothing ever will be -- treat that as a
            # satisfied, cacheable steady state rather than a permanent cache
            # miss that reruns this stage (and, via the dirty cascade, every
            # stage after it) on every single invocation.
            return True
        return all(
            (context.object(f"metadata_{i}") or {}).get("class") is not None
            for i in range(count)
        )

    def model_names(self) -> list[str]:
        return ImageCaptioning.model_names()

    def clean_up(self):
        if self._captioner is not None:
            self._captioner.close()
            self._captioner = None
        super().clean_up()  # calls torch.cuda.empty_cache() after refs are dropped
