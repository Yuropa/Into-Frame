from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.panorama.panorama_lora import PanoramaLoraType, lora_prompt_prefix
from util.device_utils import DeviceStrategy, preferred_device
from util.panorama_utils import Panorama
import numpy as np
from PIL import Image as PILImage, ImageFilter


class PanoramaLoraCorrectionConfiguration(PipelineStageConfiguration):
    """
    Stage-specific config for PanoramaLoraCorrectionStage.

    Experimental alternative to PanoramaDepthPatchStage for the same underlying
    problem: PanoramaForegroundInpaintingStage's ObjectClear pass has no
    equirectangular awareness at all (no wrap padding, no pole/nadir curvature
    handling — see server/pipeline/inpainting/inpainting_objectclear_imp.py), so
    the content it fabricates inside the removal mask looks like an ordinary flat
    photo pasted into equirect space. The second PanoramaDepthStage pass that
    re-estimates depth on PANORAMA_TERRAIN is a panorama-specialized model, so
    running it over that out-of-distribution content is exactly where the
    resulting geometry tends to go wrong. This stage takes a different tack from
    PanoramaDepthPatchStage: rather than leaving the fabricated pixels alone and
    patching the depth map afterward, it re-touches just those pixels with FLUX.1
    + a LoRA trained on real HDRI/equirectangular panoramas (the same
    PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2 already used by SkyboxInpaintingStage
    for sky regeneration), so the depth model downstream sees content closer to
    its own training distribution.

    strength (float, default 0.35):
        img2img denoise strength (SDEdit-style) passed straight to FluxInpaintPipeline
        — 0 = untouched, 1.0 = full regeneration from noise, same as re-running the
        removal from scratch. ObjectClear's output is usually already a plausible
        photo; this is deliberately a *nudge* toward the LoRA's panoramic-projection
        prior, not a replacement. Raise it if the corrected region still doesn't
        read as equirectangular-consistent; lower it if real detail is being erased.

    lora_scale (float, default 0.85):
        LoRA adapter weight. Matches SkyboxInpaintingStage's Pass 2 value — high
        enough that the panorama LoRA's geometric/structural bias actually wins,
        without fully suppressing FLUX's own texture detail.

    guidance_scale (float, default 4.0):
        Lower than SkyboxInpaintingStage's 5.5 (that pass regenerates the whole
        canvas from a gradient composite; this one only has to nudge an
        already-reasonable image, so a lighter prompt-adherence pull is enough).

    num_inference_steps (int, default 30):
        Diffusion steps for the correction pass.

    max_resolution (int, default 1536):
        FLUX runs at this resolution cap (longest edge), same trick
        SkyboxInpaintingStage uses — running FLUX directly on a multi-thousand-
        pixel panorama produces a visible patch/grid artifact instead of coherent
        detail, and is far slower than necessary for a single masked region.

    mask_extra_dilation_px (int, default 0):
        Additional dilation (in original panorama pixels) applied to
        PANORAMA_FOREGROUND_MASK before correction, beyond what
        PanoramaForegroundInpaintingStage's own mask_dilation_px already gave it.
        0 = reuse the persisted mask exactly as-is.
    """
    def __init__(
        self,
        *args,
        strength: float = 0.35,
        lora_scale: float = 0.85,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 30,
        max_resolution: int = 1536,
        mask_extra_dilation_px: int = 0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.strength = strength
        self.lora_scale = lora_scale
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.max_resolution = max_resolution
        self.mask_extra_dilation_px = mask_extra_dilation_px


class PanoramaLoraCorrectionStage(PipelineStage):
    """
    Re-touches the region PanoramaForegroundInpaintingStage fabricated with a
    partial-strength FLUX.1 + panorama-LoRA img2img pass, so its content is more
    consistent with equirectangular projection before the second PanoramaDepthStage
    pass re-estimates depth over it. See PanoramaLoraCorrectionConfiguration for
    the full rationale and PanoramaDepthPatchStage for the alternative (geometry-
    only) approach this is meant to be compared against.

    Input key  (SemanticKey.PANORAMA) → ContextKey.PANORAMA_TERRAIN
    Output key (SemanticKey.OUTPUT)   → ContextKey.PANORAMA_TERRAIN

    No-ops (passes the input straight through) if PANORAMA_FOREGROUND_MASK isn't
    in context — i.e. PanoramaForegroundInpaintingStage found nothing to remove.
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._flux = None
        self.preferred_device, self.preferred_dtype = preferred_device(DeviceStrategy.MEMORY)

    @classmethod
    def config_class(cls) -> type[PipelineStageConfiguration]:
        return PanoramaLoraCorrectionConfiguration

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.PANORAMA: ContextKey.PANORAMA_TERRAIN,
            SemanticKey.OUTPUT: ContextKey.PANORAMA_TERRAIN,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        panorama_key, output_key = self._resolved_keys()

        panorama = context.input_panorama(panorama_key)
        if panorama is None:
            self.log_warning("No panorama in context, skipping")
            return context

        original_pil = panorama.image.convert("RGB")
        w, h = original_pil.size

        mask_image = context.input_image(ContextKey.PANORAMA_FOREGROUND_MASK)
        if mask_image is None:
            self.log_info("No foreground removal mask in context, nothing to correct")
            context.add_panorama(output_key, Panorama(original_pil))
            return context

        mask_pil = mask_image.image.convert("L")
        if mask_pil.size != (w, h):
            mask_pil = mask_pil.resize((w, h), PILImage.NEAREST)
        mask_arr = np.array(mask_pil) > 127

        if self.config.mask_extra_dilation_px > 0:
            from scipy.ndimage import binary_dilation
            mask_arr = binary_dilation(mask_arr, iterations=self.config.mask_extra_dilation_px)
            mask_pil = PILImage.fromarray((mask_arr * 255).astype(np.uint8), mode="L")

        if not mask_arr.any():
            self.log_info("Removal mask is empty, nothing to correct")
            context.add_panorama(output_key, Panorama(original_pil))
            return context

        self.set_total_tasks(1)
        correct_task = self.create_progress(1, "Correcting inpainted region for equirect projection…")

        if self._flux is None:
            self._flux = InPainting(
                self.preferred_device,
                self.preferred_dtype,
                InPaintingType.FLUX,
                lora_type=PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2,
                lora_scale=self.config.lora_scale,
            )

        # FLUX runs on a downscaled copy — same trick SkyboxInpaintingStage uses
        # (full panorama resolution is several times FLUX's trained resolution and
        # produces a visible patch/grid artifact instead of coherent detail).
        flux_max = self.config.max_resolution
        if w > flux_max or h > flux_max:
            scale = flux_max / max(w, h)
            fw = max(16, (int(w * scale) // 16) * 16)
            fh = max(16, (int(h * scale) // 16) * 16)
            flux_input = original_pil.resize((fw, fh), PILImage.LANCZOS)
            flux_mask = mask_pil.resize((fw, fh), PILImage.NEAREST)
        else:
            flux_input, flux_mask = original_pil, mask_pil
            fw, fh = w, h

        prompt = lora_prompt_prefix(PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2)
        self.log_info(f"FLUX correction: {fw}x{fh}px, strength={self.config.strength}, prompt={prompt!r}")

        flux_pil = self._flux.inpaint(
            flux_input,
            flux_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            strength=self.config.strength,
            seed=self.seed,
        )

        if (fw, fh) != (w, h):
            flux_pil = flux_pil.resize((w, h), PILImage.LANCZOS)

        if self.temp is not None:
            flux_pil.save(self.temp / "panorama_lora_correction_raw.png")

        # Same feather-composite PanoramaForegroundInpaintingStage uses for its own
        # ObjectClear output: FLUX's own pass resamples the whole (downscaled)
        # canvas, so pixels outside the mask drift from the source even though
        # nothing there should change. Restore everything but the mask region.
        feather_radius = max(8, min(w, h) // 100)
        feather_arr = np.array(
            mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        ).astype(np.float32)[..., np.newaxis] / 255.0
        composited = (
            np.array(original_pil) * (1.0 - feather_arr)
            + np.array(flux_pil) * feather_arr
        ).astype(np.uint8)
        result_pil = PILImage.fromarray(composited)

        if self.temp is not None:
            result_pil.save(self.temp / "panorama_lora_correction_composited.png")

        self.finish_progress(correct_task)

        context.add_panorama(output_key, Panorama(result_pil))
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        # context.panorama() would also match ContextKey.PANORAMA_TERRAIN written
        # by the earlier "Panorama Foreground Inpainting" stage (it walks all
        # prior stages), which would make this always look "already done" even
        # before it ever ran, since this stage overwrites the same key in place.
        # has_stage_output() is scoped to this stage's own writes only (see
        # PanoramaDepthCalibrationStage / PanoramaDepthPatchStage, which have the
        # identical problem for the identical reason).
        _, output_key = self._resolved_keys()
        return context.has_stage_output(output_key)

    def model_names(self) -> list[str]:
        return InPainting.model_names(InPaintingType.FLUX, lora_type=PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2)

    def clean_up(self):
        if self._flux is not None:
            self._flux.close()
            self._flux = None
        super().clean_up()
