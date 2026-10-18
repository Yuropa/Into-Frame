from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.recognize.image_recognize import ImageRecognize, RecognizeCaptioningType


class RecognizeAnythingStage(PipelineStage):
    """
    Runs Recognize Anything Plus (RAM++) on an image to produce a set of
    pipe-separated semantic tags, stored as a string in context.

    Input key  (SemanticKey.INPUT)  → ContextKey.INPUT          (Image, default)
                                       or a Panorama at the same key — e.g.
                                       point it at ContextKey.PANORAMA to tag
                                       the full 360° panorama instead of the
                                       narrow input photo. RAM++'s own square
                                       resize (384×384) squashes the panorama's
                                       ~2:1 aspect ratio, but since this stage
                                       only produces a coarse scene tag list
                                       (not spatial detections), that's an
                                       accepted trade-off rather than something
                                       needing tiling like ObjectDetectionStage.
    Output key (SemanticKey.OUTPUT) → ContextKey.RECOGNIZE_TAGS  (str, pipe-separated)
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._recognize = None

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT,
            SemanticKey.OUTPUT: ContextKey.RECOGNIZE_TAGS,
        })

    def _resolve_source(self, context: PipelineContext, input_key):
        """Image and Panorama both expose .width/.height/.rgb(), so either can be
        the recognition source — try Image (the common case) before Panorama."""
        image = context.input_image(input_key)
        if image is not None:
            return image
        return context.input_panorama(input_key)

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, output_key = self._resolved_keys()

        task = self.create_progress(2, "Recognizing…")

        if self._recognize is None:
            self._recognize = ImageRecognize(self.device)
        self.advance_progress(task)

        input_image = self._resolve_source(context, input_key)
        if input_image is not None:
            result = self._recognize.recognize(
                input_image.rgb(),
                self.temp,
                on_progress=self.make_progress_callback(task),
            )
            tags = [t.strip() for t in str(result.tags).split("|") if t.strip()]
            # The photograph too, when the source is the panorama: a regenerated
            # Paris panorama tagged "boat | harbor | river ..." with no tower or
            # bridge, and the Eiffel Tower dropped out of the vocabulary, the
            # labels and the scene. The photo is what the scene must contain.
            photo = context.input_image(ContextKey.INPUT)
            if photo is not None and input_key != ContextKey.INPUT:
                try:
                    photo_tags = self._recognize.recognize(photo.image.convert("RGB"), self.temp).tags
                    extra = [t.strip() for t in str(photo_tags).split("|")
                             if t.strip() and t.strip() not in tags]
                    if extra:
                        self.log_info(f"Photograph adds tags: {' | '.join(extra)}")
                        tags += extra
                except Exception as exc:
                    self.log_warning(f"Photograph tagging failed ({exc}); panorama tags only")
            joined = " | ".join(tags)
            self.log_info(f"Recognized tags: {joined}")
            context.add_object(output_key, joined)
        else:
            self.log_warning("No input image — skipping recognition")

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return context.object(output_key) is not None

    def model_names(self) -> list[str]:
        return ImageRecognize.model_names()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        _, output_key = self._resolved_keys()
        tags = context.object(output_key)
        if not tags:
            return None
        tag_list = [t.strip() for t in tags.split("|") if t.strip()]
        return ReportSection(
            stage_name=self.name,
            title="Scene Recognition",
            body=(
                "Recognize Anything Plus (RAM++) analysed the input image and extracted "
                "a set of semantic scene tags. These tags describe the objects, materials, "
                "and environmental context present in the photograph and are used by "
                "downstream stages to classify and distribute scene elements."
            ),
            stats={
                "Tag count": str(len(tag_list)),
                "Tags": ", ".join(tag_list),
            },
        )

    def clean_up(self):
        if self._recognize is not None:
            self._recognize.close()
            self._recognize = None
        super().clean_up()
