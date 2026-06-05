from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.panorama_depth.pano_depth import PanoramaDepth
from util.depth_utils import Depth
from util.device_utils import DeviceStrategy, preferred_device


class PanoramaDepthStage(PipelineStage):
    """
    Runs Depth Any Panorama (DAP) on an equirectangular panorama image and
    stores the resulting equirectangular depth map in context.

    Input key  (SemanticKey.INPUT)  → ContextKey.PANORAMA  (equirectangular Image)
    Output key (SemanticKey.OUTPUT) → ContextKey.PANORAMA_DEPTH (Depth, metres, equirectangular)
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._depth = None
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.PANORAMA,
            SemanticKey.OUTPUT: ContextKey.PANORAMA_DEPTH,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, output_key = self._resolved_keys()

        task = self.create_progress(2, "Panorama Depth...")

        if self._depth is None:
            self._depth = PanoramaDepth(self.preferred_device)
        self.advance_progress(task)

        input_panorama = context.input_panorama(input_key)
        if input_panorama is not None:
            result = self._depth.depth(input_panorama.rgb(), self.temp, on_progress=self.make_progress_callback(task))
            depth = Depth(result.depth)

            if self.temp is not None:
                depth.save_debug_image(self.temp / "pano_depth.png")

            self.log_info(
                f"Panorama depth {depth.min():.1f} → {depth.max():.1f} m "
                f"({depth.width}×{depth.height})"
            )
            context.add_depth(output_key, depth)
            if result.sky_mask is not None:
                context.add_object(ContextKey.PANORAMA_SKY_MASK, result.sky_mask)
                sky_pct = result.sky_mask.mean() * 100
                self.log_info(f"Sky mask: {sky_pct:.1f}% sky pixels")
        else:
            self.log_warning("No panorama found — skipping panorama depth")

        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return context.depth(output_key) is not None

    def model_names(self) -> list[str]:
        return PanoramaDepth.model_names()

    def clean_up(self):
        super().clean_up()
        self._depth = None
