from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext, ContextKey

_ENVIRONMENT_KEYWORDS = frozenset({
    "sky", "cloud", "clouds", "ceiling", "roof", "rooftop",
    "wall", "walls", "floor", "ground", "terrain", "earth",
    "road", "pavement", "sidewalk", "path", "asphalt", "cobblestone", "street",
    "grass", "lawn", "field", "meadow",
    "water", "ocean", "sea", "lake", "river", "pond", "stream", "puddle",
    "mountain", "hill", "cliff",
    "dirt", "sand", "snow", "ice", "mud",
    "building", "buildings", "architecture", "house", "facade", "exterior", "structure",
    "concrete", "landscape", "scenery", "horizon", "background",
    "shadow", "reflection", "surface", "texture", "pattern",
})

_OBJECT_KEYWORDS = frozenset({
    "person", "people", "man", "woman", "child", "boy", "girl", "human", "crowd", "figure",
    "car", "vehicle", "truck", "bus", "bicycle", "motorcycle", "bike", "scooter", "van",
    "boat", "ship", "airplane", "train", "tram",
    "animal", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "deer",
    "chair", "table", "bench", "couch", "sofa", "desk", "furniture", "stool",
    "sign", "signpost", "pole", "lamp", "lamppost", "streetlight", "fire", "hydrant",
    "trash", "bin", "can", "barrel", "box", "container", "dumpster",
    "umbrella", "backpack", "bag", "suitcase", "luggage",
    "statue", "sculpture", "monument",
    "ball", "toy", "cart", "trolley",
    "tree", "trees", "bush", "shrub", "hedge",
    "plant", "plants", "foliage", "vegetation", "leaves", "branch", "branches", "forest", "jungle",
    "rock", "rocks", "boulder", "stone",
})


def _classify_caption(caption: str) -> str:
    """Classify a free-form caption as 'object' or 'environment' via keyword scoring."""
    words = set(caption.lower().replace(",", " ").replace(".", " ").replace("'", " ").split())
    obj_score = sum(1 for kw in _OBJECT_KEYWORDS if kw in words)
    env_score = sum(1 for kw in _ENVIRONMENT_KEYWORDS if kw in words)
    # Require a positive object signal to remove something; unknown/ambiguous → keep
    return "object" if obj_score > env_score else "environment"


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

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._captioner = None

    def run(self, context: PipelineContext) -> PipelineContext:
        object_count = context.input_object(ContextKey.OBJECT_COUNT)
        if not object_count:
            self.log_info("No objects to classify, skipping")
            return context

        classify_task = self.create_progress(object_count + 1, "Classifying objects...")
        if self._captioner is None:
            self._captioner = ImageCaptioning(self.device)
        self.advance_progress(classify_task)

        env_count = 0
        for idx in range(object_count):
            crop = context.input_image(f"crop_{idx}")
            metadata = context.input_object(f"metadata_{idx}") or {}

            if crop is None:
                self.advance_progress(classify_task)
                continue

            caption = self._captioner.caption(crop)
            cls = _classify_caption(caption)
            if cls == "environment":
                env_count += 1

            context.add_object(f"metadata_{idx}", {**metadata, "class": cls, "caption": caption})
            self.log_info(f"  crop_{idx}: '{caption}' → {cls}")
            self.advance_progress(classify_task)

        self.finish_progress(classify_task)
        self.log_info(
            f"Classification complete: {object_count - env_count} objects, {env_count} environment"
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
