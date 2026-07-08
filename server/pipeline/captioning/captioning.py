import re
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext, ContextKey

class CaptioningStage(PipelineStage):
    """
    Generates a text caption for the input image using a vision-language model.

    Input key  (SemanticKey.INPUT)  → ContextKey.INPUT          (Image)
    Output key (SemanticKey.OUTPUT) → ContextKey.INPUT_CAPTION   (str, stored as object)
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._caption = None

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT, 
            SemanticKey.OUTPUT: ContextKey.INPUT_CAPTION
        })

    def _clean_caption(self, caption: str) -> str:
        WORDS_TO_REMOVE = {
            "arafed",
            "araffe",
            "arafe",
            "araffed",
        }

        words = caption.split()

        cleaned = []
        for word in words:
            normalized = re.sub(r"[^a-z]", "", word.lower())

            if normalized in WORDS_TO_REMOVE:
                continue

            cleaned.append(word)

        result = " ".join(cleaned)

        result = re.sub(r"\s+", " ", result).strip()

        return result

    def run(self, context: PipelineContext) -> PipelineContext:
        caption_task = self.create_progress(2, "Captioning…")
        if self._caption is None:
            self._caption = ImageCaptioning(self.device)
        self.advance_progress(caption_task)
        
        input_key, output_key = self._resolved_keys()
        
        input_image = context.input_image(input_key)
        if input_image is not None:
            caption = self._caption.caption(input_image)
            caption = self._clean_caption(caption)
            context.add_object(output_key, caption)

        self.advance_progress(caption_task)
        self.finish_progress(caption_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return context.object(output_key) is not None

    def model_names(self) -> list[str]:
        return ImageCaptioning.model_names()

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        input_key, output_key = self._resolved_keys()
        caption = context.object(output_key)
        if caption is None:
            return None
        input_img = context.image(input_key)
        images = []
        if input_img is not None:
            images.append((input_img.image, "Input photograph"))
        return ReportSection(
            stage_name=self.name,
            title="Scene Description",
            body=(
                "The input image was processed by a vision-language model to produce a "
                "natural language description of the scene. This caption is used throughout "
                "the pipeline to guide panorama synthesis and other text-conditioned stages."
            ),
            images=images,
            stats={"Caption": caption},
        )

    def clean_up(self):
        if self._caption is not None:
            self._caption.close()
            self._caption = None
        super().clean_up()  # calls torch.cuda.empty_cache() after refs are dropped