from pipeline.panorama_inpainting.panorama_inpainting import PanoramaInpaint
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device


class PanoramaInpaintingStage(PipelineStage):
    """
    Segments and removes foreground objects from the panorama, saving each
    detected object crop for later 3D model generation, then inpaints the
    panorama background to fill the removed regions.

    Input key  (SemanticKey.PANORAMA) → ContextKey.PANORAMA (Panorama)
    Output key (SemanticKey.OUTPUT)   → ContextKey.PANORAMA (Panorama, inpainted)

    Additional context writes:
      pano_object_count — int: number of objects extracted
      pano_object_0 …   — Image: individual object crops
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, self.preferred_format = preferred_device(DeviceStrategy.MEMORY)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.PANORAMA: ContextKey.PANORAMA,
            SemanticKey.OUTPUT: ContextKey.PANORAMA,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        panorama_key, output_key = self._resolved_keys()

        panorama = context.input_panorama(panorama_key)
        if panorama is None:
            self.log_warning("No panorama in context, skipping panorama inpainting")
            return context

        task = self.create_progress(2, "Inpainting Panorama...")

        inpainter = PanoramaInpaint(self.preferred_device, self.preferred_format, seed=self.seed)
        self.advance_progress(task)

        inpainted, crops = inpainter.inpaint(panorama, self.temp)

        context.add_panorama(output_key, inpainted)
        context.add_object("pano_object_count", len(crops))
        for i, crop in enumerate(crops):
            context.add_image(f"pano_object_{i}", crop)
        context.add_object("pano_inpainting_complete", True)

        self.advance_progress(task)
        self.finish_progress(task)

        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object("pano_inpainting_complete") is not None

    def model_names(self) -> list[str]:
        return PanoramaInpaint.model_names()
