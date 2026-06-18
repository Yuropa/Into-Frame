from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.supersampling.image_supersampling import ImageSupersampling
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.panorama_utils import Panorama


class SupersamplingStage(PipelineStage):
    """
    Upsamples an image or panorama from the input key and writes the result to
    the output key using a Swin2SR super-resolution model.

    Key mapping (override in config YAML via `keys:`):
      input  (SemanticKey.INPUT)  — context key to read from
      output (SemanticKey.OUTPUT) — context key to write to

    The stage auto-detects whether the value is a Panorama or Image and
    preserves that type on write.
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._samp = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.has_stage_output(self.output_key())

    def run(self, context: PipelineContext) -> PipelineContext:
        task = self.create_progress(2, "Supersampling…")

        if self._samp is None:
            self._samp = ImageSupersampling(self.preferred_device)
        self.advance_progress(task)

        in_key = self.input_key()
        out_key = self.output_key()

        panorama = context.input_panorama(in_key)
        if panorama is not None:
            upsampled = self._samp.supersample(Image(panorama.image), self.temp, on_progress=self.make_progress_callback(task))
            context.add_panorama(out_key, Panorama(upsampled.image))
        else:
            image = context.input_image(in_key)
            upsampled = self._samp.supersample(image, self.temp, on_progress=self.make_progress_callback(task))
            context.add_image(out_key, upsampled)

        if self.temp is not None:
            upsampled.image.save(self.temp / f"{out_key}_supersampled.png")
        if self.output is not None:
            upsampled.image.save(self.output / f"{out_key}.png")

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def model_names(self) -> list[str]:
        return ImageSupersampling.model_names()

    def clean_up(self):
        if self._samp is not None:
            self._samp.close()
            self._samp = None
        super().clean_up()
