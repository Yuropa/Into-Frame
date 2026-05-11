from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.pipeline_context import PipelineContext, ContextKey

class CaptioningStage(PipelineStage):
    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._caption = None

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.INPUT, 
            SemanticKey.OUTPUT: ContextKey.INPUT_CAPTION
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        caption_task = self.create_progress(2, "Captioning...")
        if self._caption is None:
            self._caption = ImageCaptioning(self.device)
        self.advance_progress(caption_task)
        
        input_key, output_key = self._resolved_keys()
        
        input_image = context.input_image(input_key)
        if input_image is not None:
            caption = self._caption.caption(input_image)
            context.add_object(output_key, caption)

        self.advance_progress(caption_task)
        self.finish_progress(caption_task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return (
            context.image(output_key) is not None
        )

    def model_names(self) -> list[str]:
        return ImageCaptioning.model_names()

    def clean_up(self):
        super().clean_up()
        self._caption = None