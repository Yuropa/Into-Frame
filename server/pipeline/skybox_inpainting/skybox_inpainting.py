import numpy as np
from pathlib import Path
from PIL import Image as PILImage, ImageFilter

from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.panorama_utils import Panorama
from scipy.ndimage import binary_dilation


def _sky_prompt(input_caption: str) -> str:
    """Build a Flux prompt for sky in-fill from the scene caption."""
    sky_words = {"sky", "cloud", "clouds", "sun", "moon", "star", "stars",
                 "horizon", "twilight", "dusk", "dawn", "sunset", "sunrise",
                 "overcast", "clear", "atmosphere", "blue"}
    tokens = [t for t in input_caption.lower().split() if t in sky_words]
    sky_desc = " ".join(tokens) if tokens else "clear blue sky"
    return f"photorealistic panoramic sky, {sky_desc}, seamless, high quality"


class SkyboxInpaintingStage(PipelineStage):
    """
    Masks out non-sky regions from the panorama and in-paints them with sky
    content so the skybox contains only sky material.

    Reads:
      ContextKey.PANORAMA                — equirectangular source panorama
      ContextKey.PANORAMA_REGION_TYPE_MAP — per-pixel RegionType indices (float32)

    Writes:
      ContextKey.PANORAMA_SKY_MASK — binary sky mask (Image, L mode, 255=sky)
      ContextKey.PANORAMA_SKY      — sky-complete equirectangular panorama (Panorama)
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)
        self._lama: InPainting | None = None
        self._flux: InPainting | None = None

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.PANORAMA: ContextKey.PANORAMA,
            SemanticKey.OUTPUT: ContextKey.PANORAMA_SKY,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        panorama_key, output_key = self._resolved_keys()

        panorama = context.input_panorama(panorama_key)
        if panorama is None:
            self.log_warning("No panorama in context, skipping")
            return context

        type_map = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP)
        if type_map is None:
            self.log_warning("No region type map in context — run PanoramaRegionStage first")
            return context

        task = self.create_progress(4, "Building sky mask…")

        source_pil = panorama.image.convert("RGB")
        h, w = source_pil.height, source_pil.width
        source_arr = np.array(source_pil)

        # Region type map is stored as float32; cast back to uint8 for comparison.
        type_arr = type_map.depth.astype(np.uint8)
        sky_mask = (type_arr == int(RegionType.SKY))       # True where sky
        fill_mask = ~sky_mask                               # True where we must in-paint

        # Store sky mask for downstream consumers.
        sky_mask_pil = PILImage.fromarray((sky_mask * 255).astype(np.uint8), mode="L")
        context.add_image(ContextKey.PANORAMA_SKY_MASK, Image(sky_mask_pil))

        sky_fraction = sky_mask.mean()
        self.log_info(f"Sky coverage: {sky_fraction * 100:.1f}%  fill: {fill_mask.mean() * 100:.1f}%")

        if self.output is not None:
            sky_mask_pil.save(self.output / "sky_mask.png")

        self.advance_progress(task)

        if fill_mask.sum() == 0:
            self.log_info("Entire panorama is sky, no in-painting needed")
            context.add_panorama(output_key, Panorama(source_pil))
            self.finish_progress(task)
            return context

        # Dilate fill mask so in-painting blends cleanly into the sky boundary.
        dilation_px = max(8, min(w, h) // 100)
        fill_dilated = binary_dilation(fill_mask, iterations=dilation_px).astype(np.float32)
        fill_mask_pil = PILImage.fromarray((fill_dilated * 255).astype(np.uint8), mode="L")

        if self.output is not None:
            fill_mask_pil.save(self.output / "fill_mask.png")

        # Caption from the scene to guide Flux toward appropriate sky content.
        scene_caption = context.input_object(ContextKey.INPUT_CAPTION) or ""
        prompt = _sky_prompt(scene_caption)
        self.log_info(f"Sky prompt: {prompt!r}")

        self.advance_progress(task)

        # Phase 1: LaMa — structural fill of the full panorama.
        self.log_info(f"LaMa: full panorama ({w}×{h}px)")
        lama = InPainting(self.preferred_device, self.torch_dtype, InPaintingType.LAMA)
        lama_pil = lama.inpaint(source_pil, fill_mask_pil, temp_path=self.temp)
        lama.close()
        lama_arr = np.array(lama_pil)

        if self.output is not None:
            lama_pil.save(self.output / "lama.png")

        self.advance_progress(task)

        # Phase 2: Flux — perceptual refinement. Scale down if needed.
        flux_max = 1024
        if w > flux_max or h > flux_max:
            scale = flux_max / max(w, h)
            fw = max(16, (int(w * scale) // 16) * 16)
            fh = max(16, (int(h * scale) // 16) * 16)
            flux_input = lama_pil.resize((fw, fh), PILImage.LANCZOS)
            flux_mask = fill_mask_pil.resize((fw, fh), PILImage.NEAREST)
        else:
            flux_input, flux_mask = lama_pil, fill_mask_pil
            fw, fh = w, h

        self.log_info(f"Flux: {fw}×{fh}px")
        flux = InPainting(self.preferred_device, self.torch_dtype, InPaintingType.FLUX)
        flux_pil = flux.inpaint(
            flux_input,
            flux_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=36,
            guidance_scale=10.0,
            seed=self.seed,
        )
        flux.close()

        if fw != w or fh != h:
            flux_pil = flux_pil.resize((w, h), PILImage.LANCZOS)
        flux_arr = np.array(flux_pil)

        if self.output is not None:
            flux_pil.save(self.output / "flux.png")

        # Composite: keep original sky pixels exactly; blend fill mask boundary.
        feather_radius = max(8, min(w, h) // 100)
        feathered = np.array(
            fill_mask_pil.filter(ImageFilter.GaussianBlur(radius=feather_radius))
        ).astype(np.float32)[..., np.newaxis] / 255.0

        # Blend against the LaMa fill (not the original) so the feather zone doesn't
        # reveal removed content at the seam.
        blended_arr = (lama_arr * (1.0 - feathered) + flux_arr * feathered).astype(np.uint8)

        # Restore original sky pixels unchanged.
        sky_3d = sky_mask[..., np.newaxis]
        result_arr = np.where(sky_3d, source_arr, blended_arr).astype(np.uint8)

        result_pil = PILImage.fromarray(result_arr)
        if self.output is not None:
            result_pil.save(self.output / "skybox.png")

        context.add_panorama(output_key, Panorama(result_pil))

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        return (
            context.image(ContextKey.PANORAMA_SKY_MASK) is not None
            and context.panorama(ContextKey.PANORAMA_SKY) is not None
        )

    def model_names(self) -> list[str]:
        return (
            InPainting.model_names(InPaintingType.LAMA)
            + InPainting.model_names(InPaintingType.FLUX)
        )

    def clean_up(self):
        if self._lama is not None:
            self._lama.close()
            self._lama = None
        if self._flux is not None:
            self._flux.close()
            self._flux = None
        super().clean_up()
