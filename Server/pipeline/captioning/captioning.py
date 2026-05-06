from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext, ContextKey

class CaptioningStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._caption = None

    def run(self, context: PipelineContext) -> PipelineContext:
        caption_task = self.create_progress(2, "Captioning...")
        if self._caption is None:
            self._caption = ImageCaptioning(self.device)
        self.advance_progress(caption_task)

        input_image = context.input_image(ContextKey.INPUT)
        if input_image is not None:
            caption = self._caption.caption(input_image)
            context.add_object(ContextKey.INPUT_CAPTION, caption)

        self.advance_progress(caption_task)
        self.finish_progress(caption_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return (
            context.image(ContextKey.INPUT_CAPTION) is not None
        )

    def model_names(self) -> list[str]:
        return ImageCaptioning.model_names()

    def clean_up(self):
        super().clean_up()
        self._caption = None