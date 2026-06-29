import numpy as np
from pathlib import Path
from PIL import Image as PILImage

from pipeline.captioning.image_captioning import ImageCaptioning
from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.panorama_utils import Panorama
from scipy.ndimage import binary_dilation, distance_transform_edt


def _crop_sky(source_pil: PILImage.Image, sky_mask: np.ndarray) -> PILImage.Image:
    """Crop the horizontal band containing sky pixels."""
    rows = np.any(sky_mask, axis=1)
    if not rows.any():
        return source_pil
    rmin = int(np.where(rows)[0][0])
    rmax = int(np.where(rows)[0][-1])
    return source_pil.crop((0, rmin, source_pil.width, rmax + 1))


def _sky_prompt(sky_caption: str) -> str:
    return (
        f"photorealistic equirectangular 360-degree sky panorama, {sky_caption}, "
        "seamless sky filling the entire frame in all directions, only sky and clouds, "
        "no ground, no horizon line, no terrain, no landscape, no objects, "
        "consistent natural lighting, ultra detailed, high quality"
    )


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

        type_arr = type_map.depth.astype(np.uint8)
        sky_mask = (type_arr == int(RegionType.SKY))

        # Build a clean rectangular fill mask instead of tracing the mountain
        # silhouette.  The exact sky-shaped mask tells Flux "there is a mountain
        # outline here" and it obliges; a horizontal cutoff gives it a clean sky
        # boundary so it generates pure sky all the way down.
        sky_rows = np.where(np.any(sky_mask, axis=1))[0]
        sky_bottom = int(sky_rows[-1]) + 1 if len(sky_rows) > 0 else h // 2
        fill_mask = np.zeros((h, w), dtype=bool)
        fill_mask[sky_bottom:] = True                  # rectangle below sky
        fill_mask[:sky_bottom] = ~sky_mask[:sky_bottom] # holes inside sky band

        sky_mask_pil = PILImage.fromarray((sky_mask * 255).astype(np.uint8), mode="L")
        context.add_image(ContextKey.PANORAMA_SKY_MASK, Image(sky_mask_pil))

        sky_fraction = sky_mask.mean()
        self.log_info(f"Sky coverage: {sky_fraction * 100:.1f}%  sky_bottom row: {sky_bottom}  fill: {fill_mask.mean() * 100:.1f}%")

        if self.output is not None:
            sky_mask_pil.save(self.output / "sky_mask.png")

        self.advance_progress(task)

        if fill_mask.sum() == 0:
            self.log_info("Entire panorama is sky, no in-painting needed")
            context.add_panorama(output_key, Panorama(source_pil))
            self.finish_progress(task)
            return context

        dilation_px = max(8, min(w, h) // 100)
        fill_dilated = binary_dilation(fill_mask, iterations=dilation_px).astype(np.float32)
        fill_mask_pil = PILImage.fromarray((fill_dilated * 255).astype(np.uint8), mode="L")

        if self.output is not None:
            fill_mask_pil.save(self.output / "fill_mask.png")

        sky_crop = _crop_sky(source_pil, sky_mask)
        captioner = ImageCaptioning(self.preferred_device)
        sky_caption = captioner.caption(Image(sky_crop), prompt="a photo of the sky with")
        del captioner
        prompt = _sky_prompt(sky_caption)
        self.log_info(f"Sky caption: {sky_caption!r}")
        self.log_info(f"Sky prompt: {prompt!r}")

        self.advance_progress(task)

        # Pre-fill non-sky regions by propagating the nearest sky pixel colour outward.
        # This gives FLUX a sky-coloured context everywhere instead of a hard boundary.
        _, nearest_sky_idx = distance_transform_edt(~sky_mask, return_indices=True)
        prefilled_arr = source_arr[nearest_sky_idx[0], nearest_sky_idx[1]]
        prefilled_pil = PILImage.fromarray(prefilled_arr)

        if self.output is not None:
            prefilled_pil.save(self.output / "prefilled.png")

        # Scale to FLUX working resolution
        flux_max = 1536
        if w > flux_max or h > flux_max:
            scale = flux_max / max(w, h)
            fw = max(16, (int(w * scale) // 16) * 16)
            fh = max(16, (int(h * scale) // 16) * 16)
            flux_input = prefilled_pil.resize((fw, fh), PILImage.LANCZOS)
            flux_mask  = fill_mask_pil.resize((fw, fh), PILImage.NEAREST)
        else:
            flux_input, flux_mask = prefilled_pil, fill_mask_pil
            fw, fh = w, h

        self.log_info(f"FLUX pass 1: {fw}×{fh}px")
        flux = InPainting(self.preferred_device, self.torch_dtype, InPaintingType.FLUX)
        flux_pil = flux.inpaint(
            flux_input,
            flux_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=30.0,
            seed=self.seed,
        )

        if self.output is not None:
            flux_pil.save(self.output / "flux_pass1.png")

        self.advance_progress(task)

        # Equirectangular wrap seam repair: if the fill region touches the left or
        # right edge then the inpainted pixels on each side aren't guaranteed to
        # match (FLUX sees a hard cut). Fix with a circular-shift + narrow inpaint.
        flux_mask_arr = np.array(flux_mask)
        seam_needed = (
            fill_dilated[:, :5].any() or fill_dilated[:, -5:].any()
        ) and flux_mask_arr.any()

        if seam_needed:
            self.log_info("FLUX pass 2: wrap seam repair")
            half_fw = fw // 2
            shifted_arr  = np.roll(np.array(flux_pil), half_fw, axis=1)
            shifted_mask = np.roll(flux_mask_arr,       half_fw, axis=1)

            seam_w = max(4, fw // 64)
            seam_mask_arr = np.zeros((fh, fw), dtype=np.uint8)
            seam_mask_arr[:, half_fw - seam_w : half_fw + seam_w] = 255
            # Only touch pixels that were filled, not original sky
            seam_mask_arr = np.minimum(seam_mask_arr, shifted_mask)

            if seam_mask_arr.any():
                seam_pil = flux.inpaint(
                    PILImage.fromarray(shifted_arr),
                    PILImage.fromarray(seam_mask_arr, "L"),
                    temp_path=self.temp,
                    prompt=prompt,
                    num_inference_steps=30,
                    guidance_scale=30.0,
                    seed=self.seed + 1,
                )
                flux_pil = PILImage.fromarray(
                    np.roll(np.array(seam_pil), -half_fw, axis=1)
                )

                if self.output is not None:
                    flux_pil.save(self.output / "flux_pass2.png")

        flux.close()

        if fw != w or fh != h:
            flux_pil = flux_pil.resize((w, h), PILImage.LANCZOS)

        if self.output is not None:
            flux_pil.save(self.output / "skybox.png")

        context.add_panorama(output_key, Panorama(flux_pil))

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
            ImageCaptioning.model_names()
            + InPainting.model_names(InPaintingType.FLUX)
        )

    def clean_up(self):
        if self._flux is not None:
            self._flux.close()
            self._flux = None
        super().clean_up()
