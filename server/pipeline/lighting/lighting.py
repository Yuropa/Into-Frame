from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.lighting.lux_dit import LuxDiT
from scene.lighting import SceneLighting


class PanoramaLightingStage(PipelineStage):
    """
    Runs LuxDiT on the panorama to estimate environment lighting.

    Produces an LDR environment map and a log-domain environment map, stored
    together as a SceneLighting value so downstream stages (e.g. scene
    generation) can embed the lighting data in the scene sent to Unity.

    Input key  (SemanticKey.INPUT)  → ContextKey.PANORAMA  (Panorama)
    Output key (SemanticKey.OUTPUT) → ContextKey.LIGHTING   (SceneLighting)
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._lux_dit = None

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.PANORAMA,
            SemanticKey.OUTPUT: ContextKey.LIGHTING,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, output_key = self._resolved_keys()

        task = self.create_progress(2, "Panorama Lighting...")

        if self._lux_dit is None:
            self._lux_dit = LuxDiT(self.device)
        self.advance_progress(task)

        panorama = context.input_panorama(input_key)
        if panorama is None:
            self.log_warning("No panorama found — skipping lighting estimation")
            self.finish_progress(task)
            return context

        result = self._lux_dit.estimate(
            panorama.rgb(),
            self.temp,
            on_progress=self.make_progress_callback(task),
        )

        lighting = SceneLighting(ldr=result.ldr, log=result.log)

        if self.temp is not None:
            result.ldr.save(self.temp / "lighting_ldr.png")
            result.log.save(self.temp / "lighting_log.png")

        self.log_info(
            f"Lighting estimated: env map {lighting.width}×{lighting.height}"
        )
        context.add_lighting(output_key, lighting)

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        _, output_key = self._resolved_keys()
        return context.lighting(output_key) is not None

    def model_names(self) -> list[str]:
        return LuxDiT.model_names()

    def clean_up(self):
        super().clean_up()
        self._lux_dit = None
