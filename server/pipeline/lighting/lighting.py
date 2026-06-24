from logging import Logger
from typing import Any

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.lighting.lux_dit import LuxDiT
from scene.lighting import SceneLighting


class PanoramaLightingConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        crop_fov_deg: float = 90.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.crop_fov_deg = float(crop_fov_deg)


class PanoramaLightingStage(PipelineStage):
    """
    Runs LuxDiT on the panorama to estimate environment lighting.

    Produces an LDR environment map and a log-domain environment map, stored
    together as a SceneLighting value so downstream stages (e.g. scene
    generation) can embed the lighting data in the scene sent to Unity.

    Input key  (SemanticKey.INPUT)  → ContextKey.PANORAMA  (Panorama)
    Output key (SemanticKey.OUTPUT) → ContextKey.LIGHTING   (SceneLighting)

    Config:
      crop_fov_deg — horizontal FOV of the centre crop fed to LuxDiT (degrees).
                     Set to 0 to use the full panorama. Default: 90.
    """

    def __init__(self, config: PanoramaLightingConfiguration) -> None:
        super().__init__(config)
        self._lux_dit = None

    @classmethod
    def config_class(cls) -> type[PanoramaLightingConfiguration]:
        return PanoramaLightingConfiguration

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.PANORAMA,
            SemanticKey.OUTPUT: ContextKey.LIGHTING,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        input_key, output_key = self._resolved_keys()

        task = self.create_progress(2, "Panorama Lighting…")

        if self._lux_dit is None:
            self._lux_dit = LuxDiT(self.device)
        self.advance_progress(task)

        panorama = context.input_panorama(input_key)
        if panorama is None:
            self.log_warning("No panorama found — skipping lighting estimation")
            self.finish_progress(task)
            return context

        crop_fov = self.config.crop_fov_deg
        if crop_fov > 0:
            img  = panorama.rgb()
            w, h = img.size
            crop_w = max(1, int(w * crop_fov / 360.0))
            x0     = (w - crop_w) // 2
            input_image = img.crop((x0, 0, x0 + crop_w, h))
            self.log_info(f"Lighting: using {crop_fov}° centre crop ({crop_w}×{h}px)")
        else:
            input_image = panorama.rgb()

        result = self._lux_dit.estimate(
            input_image,
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

    def contribute_report(self, context: PipelineContext):
        from pipeline.report.report_section import ReportSection
        _, output_key = self._resolved_keys()
        lighting = context.lighting(output_key)
        if lighting is None:
            return None
        return ReportSection(
            stage_name=self.name,
            title="Environment Lighting Estimation",
            body=(
                "Environment lighting was estimated from the panorama using LuxDiT, a "
                "diffusion-based illumination estimation model. The model produces an LDR "
                "environment map and a log-domain map that together encode the full "
                "spherical lighting of the scene. Both maps are embedded in the scene "
                "description and used by the real-time renderer for physically-based "
                "shading of all reconstructed 3D assets."
            ),
            images=[(lighting.ldr.convert("RGB"), "Estimated LDR environment map")],
            stats={"Environment map resolution": f"{lighting.width} × {lighting.height} px"},
        )

    def clean_up(self):
        if self._lux_dit is not None:
            self._lux_dit.close()
            self._lux_dit = None
        super().clean_up()
