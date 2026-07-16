from typing import Any
from logging import Logger
import torch
import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import (
    gaussian_filter,
    binary_dilation,
    binary_erosion,
    binary_opening,
    binary_fill_holes,
    distance_transform_edt,
    label as ndi_label,
)

from pipeline.inpainting.inpainting import InPainting, InPaintingType
from pipeline.panorama.panorama_lora import PanoramaLoraType, lora_prompt_prefix
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.device_utils import DeviceStrategy, preferred_device
from util.image_utils import Image
from util.panorama_utils import Panorama
from util.seam_repair import heal_wrap_seam


def _clean_sky_mask(sky_mask: np.ndarray, top_rows_frac: float = 0.02) -> np.ndarray:
    """
    Remove segmentation noise from the raw sky mask before it drives the fill
    boundary.  ADE20K segmentation blurs mountain ridgelines and backlit tree
    silhouettes into the sky class near the horizon, letting real landscape
    pixels leak into what should be a clean sky region.

      1. Fill small enclosed non-sky islands within the sky region (isolated
         misclassified pixels inside an otherwise clean sky).
      2. Morphological opening strips thin tendrils — branch- or
         ridgeline-width protrusions where the sky class bleeds down into
         terrain/vegetation at the horizon — while leaving the bulk sky area
         intact.
      3. Discard any surviving component not connected to the top of the
         panorama: real sky is always contiguous with the zenith, so anything
         else is spurious.
    """
    h, w = sky_mask.shape
    if not sky_mask.any():
        return sky_mask

    filled = binary_fill_holes(sky_mask)

    radius = max(2, min(h, w) // 300)
    opened = binary_opening(filled, iterations=radius)

    labeled, n = ndi_label(opened)
    if n == 0:
        return opened

    top_band = max(1, int(h * top_rows_frac))
    top_labels = set(np.unique(labeled[:top_band, :])) - {0}
    if not top_labels:
        return opened
    return np.isin(labeled, list(top_labels))


def _sun_from_lighting(ldr: PILImage.Image, panorama_h: int, panorama_w: int) -> tuple[int, int]:
    """Find sun pixel in panorama coordinates from the LuxDiT LDR environment map."""
    ldr_arr = np.array(ldr.convert("RGB")).astype(np.float32)
    lum = 0.299 * ldr_arr[:, :, 0] + 0.587 * ldr_arr[:, :, 1] + 0.114 * ldr_arr[:, :, 2]
    sigma = max(lum.shape) * 0.025
    smooth = gaussian_filter(lum, sigma=sigma)
    ldr_row, ldr_col = np.unravel_index(smooth.argmax(), smooth.shape)
    # Rescale LDR pixel → panorama pixel
    sun_row = int(ldr_row / ldr_arr.shape[0] * panorama_h)
    sun_col = int(ldr_col / ldr_arr.shape[1] * panorama_w)
    return sun_row, sun_col


def _sun_from_sky_pixels(source_arr: np.ndarray, sky_mask: np.ndarray) -> tuple[int, int]:
    """Estimate sun as the smoothed luminance peak within sky pixels."""
    lum = np.zeros(source_arr.shape[:2], dtype=np.float32)
    lum[sky_mask] = (
        0.299 * source_arr[sky_mask, 0].astype(np.float32)
        + 0.587 * source_arr[sky_mask, 1].astype(np.float32)
        + 0.114 * source_arr[sky_mask, 2].astype(np.float32)
    )
    sigma = max(source_arr.shape[0], source_arr.shape[1]) * 0.025
    smooth = gaussian_filter(lum, sigma=sigma)
    row, col = np.unravel_index(smooth.argmax(), smooth.shape)
    return int(row), int(col)


def _time_of_day_label(sun_el_deg: float) -> str:
    if sun_el_deg > 55:
        return "midday"
    elif sun_el_deg > 30:
        return "afternoon"
    elif sun_el_deg > 10:
        return "golden hour"
    elif sun_el_deg > -3:
        return "sunset"
    else:
        return "dusk"


def _sky_prompt(time_label: str) -> str:
    return (
        f"A photorealistic equirectangular 360-degree view of the upper atmosphere, {time_label}. "
        "Looking straight up into a seamless daytime sky filled with soft, natural clouds "
        "and realistic atmospheric haze. Clean blue sky expanse, cirrus and cumulus cloud textures, "
        "uniform lighting, ultra-detailed, high quality, panoramic backdrop."
    )


def _build_gradient_sky(
    source_arr: np.ndarray,
    sky_mask: np.ndarray,
    sun_row: int,
    sun_col: int,
    n_bins: int = 24,
) -> np.ndarray:
    """
    Full-sphere solar gradient from sky pixel colors.

    Colors are sampled from the panorama's sky region, binned by angular
    distance from the sun, then mapped back over every pixel.  Captures
    solar glow, blue dome, and horizon haze for any time of day.
    """
    h, w = source_arr.shape[:2]

    row_idx = np.arange(h, dtype=np.float32)[:, None]
    col_idx = np.arange(w, dtype=np.float32)[None, :]
    el = (0.5 - row_idx / h) * np.pi
    az = (col_idx / w) * 2.0 * np.pi

    cos_el = np.cos(el)
    px = cos_el * np.cos(az)
    py = cos_el * np.sin(az)
    pz = np.sin(el)

    sun_el = (0.5 - sun_row / h) * np.pi
    sun_az = (sun_col / w) * 2.0 * np.pi
    sx = np.cos(sun_el) * np.cos(sun_az)
    sy = np.cos(sun_el) * np.sin(sun_az)
    sz = np.sin(sun_el)

    dot = np.clip(px * sx + py * sy + pz * sz, -1.0, 1.0)
    angle_from_sun = np.arccos(dot)

    bin_idx = np.clip(
        (angle_from_sun / np.pi * n_bins).astype(np.int32), 0, n_bins - 1
    )

    bin_colors = np.zeros((n_bins, 3), dtype=np.float64)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    for i in range(n_bins):
        in_bin = sky_mask & (bin_idx == i)
        if in_bin.any():
            bin_colors[i] = source_arr[in_bin].astype(np.float64).mean(axis=0)
            bin_counts[i] = in_bin.sum()

    filled = np.where(bin_counts > 0)[0]
    if len(filled) == 0:
        bin_colors[:] = [135, 206, 235]
    else:
        for i in range(n_bins):
            if bin_counts[i] == 0:
                left = filled[filled < i]
                right = filled[filled > i]
                if left.size and right.size:
                    l, r = left[-1], right[0]
                    t = float(i - l) / float(r - l)
                    bin_colors[i] = (1.0 - t) * bin_colors[l] + t * bin_colors[r]
                elif left.size:
                    bin_colors[i] = bin_colors[left[-1]]
                else:
                    bin_colors[i] = bin_colors[right[0]]

    return bin_colors[bin_idx].astype(np.uint8)


class SkyboxInpaintingConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        # FLUX's guidance scale: how strongly the full-canvas sky generation follows
        # the time-of-day sky prompt versus the gradient conditioning image. Higher
        # pushes the generated clouds/atmosphere further from the muted procedural
        # gradient toward the prompt's own texture and contrast.
        guidance_scale: float = 4.0,
        # How much Pass 2 denoises from the gradient composite: 1.0 starts from pure
        # noise (the composite only informs FLUX indirectly), so nothing of the real
        # photo's sky survives; lower values start from a partially-noised version of
        # the composite itself, letting its real sky colour/texture blend through
        # into the result instead of being fully replaced.
        strength: float = 0.85,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.guidance_scale = guidance_scale
        self.strength = strength


class SkyboxInpaintingStage(PipelineStage):
    """
    Replaces the whole panorama with a generated sky, in two passes:

      1. Procedural solar gradient derived from the real sky pixels and sun
         position estimated from the LuxDiT lighting output (or sky luminance
         peak as fallback). Gives the correct time-of-day structure. Used only
         as the conditioning image for pass 2, not part of the final output.

      2. FLUX inpainting (with the FLUX_DEV_PANORAMA_LORA_2 LoRA) over the
         *entire* canvas — not just the non-sky region — to add cloud and
         atmospheric detail, prompted with the time-of-day derived from sun
         elevation. This is the final result — no further blending is applied.

         Regenerating everything (rather than masking to just the non-sky
         region) is deliberate: an earlier version only filled the non-sky
         area, dilating the fill mask into the sky to bury the real horizon's
         silhouette from FLUX. That buried the photo pixels but not the
         *shape* — the mask boundary still traced the mountain contour, and
         FLUX reliably rendered a cloud-top ridge along it, an artifact that
         "followed the seam" between the kept real sky and the generated
         fill. Full-canvas regeneration removes that boundary (and thus the
         artifact) entirely; the gradient composite from pass 1 already
         carries the real sky's colour and sun position across the whole
         frame, so the result stays grounded without it.

         `config.strength` (< 1.0) controls how much of that composite —
         real photographed sky included — survives into the final result
         versus being replaced by FLUX's own generation.

    Reads:
      ContextKey.PANORAMA                — equirectangular source panorama
      ContextKey.PANORAMA_REGION_TYPE_MAP — per-pixel RegionType indices (float32)
      ContextKey.LIGHTING                — SceneLighting from PanoramaLightingStage
                                           (optional; falls back to sky-pixel estimate)

    Writes:
      ContextKey.PANORAMA_SKY_MASK — binary sky mask (Image, L mode, 255=sky)
      ContextKey.PANORAMA_SKY      — sky-complete equirectangular panorama (Panorama)
    """

    @classmethod
    def config_class(cls) -> type[SkyboxInpaintingConfiguration]:
        return SkyboxInpaintingConfiguration

    def __init__(self, config: SkyboxInpaintingConfiguration) -> None:
        super().__init__(config)
        self.preferred_device, _ = preferred_device(DeviceStrategy.MEMORY)

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

        task = self.create_progress(5, "Sky gradient…")

        source_pil = panorama.image.convert("RGB")
        h, w = source_pil.height, source_pil.width
        source_arr = np.array(source_pil)

        type_arr = type_map.depth.astype(np.uint8)
        raw_sky_mask = (type_arr == int(RegionType.SKY))
        sky_mask = _clean_sky_mask(raw_sky_mask)

        if self.temp is not None:
            raw_mask_pil = PILImage.fromarray((raw_sky_mask * 255).astype(np.uint8), mode="L")
            raw_mask_pil.save(self.temp / "panorama_sky_mask_raw.png")

        sky_mask_pil = PILImage.fromarray((sky_mask * 255).astype(np.uint8), mode="L")
        context.add_image(ContextKey.PANORAMA_SKY_MASK, Image(sky_mask_pil))

        if self.output is not None:
            sky_mask_pil.save(self.output / "panorama_sky_mask.png")

        self.log_info(f"Sky coverage: {sky_mask.mean() * 100:.1f}%")
        self.advance_progress(task)

        if sky_mask.all():
            self.log_info("Entire panorama is sky, no fill needed")
            context.add_panorama(output_key, Panorama(source_pil))
            self.finish_progress(task)
            return context

        # --- Sun position ---
        lighting = context.lighting(ContextKey.LIGHTING)
        if lighting is not None:
            sun_row, sun_col = _sun_from_lighting(lighting.ldr, h, w)
            self.log_info("Sun position from LuxDiT lighting")
        else:
            sun_row, sun_col = _sun_from_sky_pixels(source_arr, sky_mask)
            self.log_info("Sun position from sky-pixel luminance (no lighting in context)")

        sun_el_deg = (0.5 - sun_row / h) * 180.0
        sun_az_deg = sun_col / w * 360.0
        time_label = _time_of_day_label(sun_el_deg)
        self.log_info(
            f"Sun: el={sun_el_deg:.1f}°  az={sun_az_deg:.1f}°  → {time_label}"
        )

        # --- Pass 1: gradient sky ---
        gradient = _build_gradient_sky(source_arr, sky_mask, sun_row, sun_col)

        # Gradient composite: real sky over gradient with feathered boundary
        feather_px = max(4, min(w, h) // 80)
        sky_eroded  = binary_erosion(sky_mask,  iterations=feather_px)
        sky_dilated = binary_dilation(sky_mask, iterations=feather_px)
        blend_zone  = sky_dilated & ~sky_eroded

        dist_in  = distance_transform_edt(sky_mask)
        dist_out = distance_transform_edt(~sky_mask)

        if blend_zone.any():
            sky_alpha = np.where(
                sky_eroded,
                1.0,
                np.where(
                    blend_zone,
                    dist_in / (dist_in + dist_out + 1e-9),
                    0.0,
                ),
            ).astype(np.float32)
        else:
            sky_alpha = sky_mask.astype(np.float32)

        sky_alpha3 = sky_alpha[:, :, None]
        gradient_composite = (
            sky_alpha3 * source_arr.astype(np.float32)
            + (1.0 - sky_alpha3) * gradient.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        if self.temp is not None:
            PILImage.fromarray(gradient_composite).save(self.temp / "gradient.png")

        self.advance_progress(task)

        # --- Pass 2: FLUX detail over the whole panorama ---
        # Full-canvas mask — see the class docstring for why this replaced the old
        # non-sky-only fill mask (it was the source of a cloud-ridge artifact tracing
        # the buried mountain silhouette).
        full_mask_arr = np.ones((h, w), dtype=bool)
        fill_mask_pil = PILImage.fromarray((full_mask_arr * 255).astype(np.uint8), mode="L")

        # Lead with the FLUX_DEV_PANORAMA_LORA_2 trigger sequence so the LoRA activates its
        # spherical/HDRI generation mode instead of its base perspective-photo bias.
        prompt = lora_prompt_prefix(PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2) + _sky_prompt(time_label)
        self.log_info(f"FLUX prompt: {prompt!r}")

        flux_max = 1536
        gradient_pil = PILImage.fromarray(gradient_composite)
        if w > flux_max or h > flux_max:
            scale = flux_max / max(w, h)
            fw = max(16, (int(w * scale) // 16) * 16)
            fh = max(16, (int(h * scale) // 16) * 16)
            flux_input = gradient_pil.resize((fw, fh), PILImage.LANCZOS)
            flux_mask  = fill_mask_pil.resize((fw, fh), PILImage.NEAREST)
        else:
            flux_input, flux_mask = gradient_pil, fill_mask_pil
            fw, fh = w, h

        self.log_info(f"FLUX: {fw}×{fh}px")
        flux = InPainting(
            self.preferred_device,
            self.torch_dtype,
            InPaintingType.FLUX,
            lora_type=PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2,
            # Lower than full strength (1.0) so the LoRA's geometric/structural bias wins
            # without stamping out FLUX's own cloud/atmosphere texture detail.
            lora_scale=0.85,
        )
        flux_pil = flux.inpaint(
            flux_input,
            flux_mask,
            temp_path=self.temp,
            prompt=prompt,
            num_inference_steps=50,
            guidance_scale=self.config.guidance_scale,
            strength=self.config.strength,
            seed=self.seed,
        )

        if fw != w or fh != h:
            flux_pil = flux_pil.resize((w, h), PILImage.LANCZOS)

        if self.temp is not None:
            flux_pil.save(self.temp / "flux.png")

        self.advance_progress(task)

        # FLUX inpainting result is the final output directly — no further
        # blending against the gradient (it only muddied the result with the
        # gradient's coarse angular-bin banding without adding anything FLUX
        # hadn't already accounted for). There's also no separate sky/terrain
        # seam to heal any more: Pass 2 regenerated the entire canvas, so
        # there's no kept-vs-generated boundary left over to blend away.
        result_pil = flux_pil

        # --- Pass 3: heal the panorama's own left/right wrap seam ---
        # FLUX generated the cloud/atmosphere fill across the whole canvas in
        # one shot, with no explicit awareness that column 0 and column W-1
        # are the same seam in the final 360° view — so they can disagree.
        # Roll the seam to the center of the frame, inpaint across it there,
        # then roll back. Runs base FLUX (no LoRA) for this touch-up — the
        # panorama LoRA's geometric bias is for the main generation, not a
        # small seam patch.
        flux.set_lora_enabled(False)

        def _sky_seam_inpaint(rolled_img: PILImage.Image, mask_img: PILImage.Image) -> PILImage.Image:
            # Downscale to flux_max like Pass 2 does — FLUX Fill run directly on
            # a full panorama (several times its trained resolution) produces a
            # visible patch/grid artifact instead of coherent detail. heal_wrap_seam
            # resizes the result back up to the caller's resolution on return, so
            # returning it at flux_max here is fine.
            sw, sh = rolled_img.size
            if sw > flux_max or sh > flux_max:
                scale = flux_max / max(sw, sh)
                fw = max(16, (int(sw * scale) // 16) * 16)
                fh = max(16, (int(sh * scale) // 16) * 16)
                seam_input = rolled_img.resize((fw, fh), PILImage.LANCZOS)
                seam_mask = mask_img.resize((fw, fh), PILImage.NEAREST)
            else:
                seam_input, seam_mask = rolled_img, mask_img
            return flux.inpaint(
                seam_input,
                seam_mask,
                temp_path=self.temp,
                prompt=prompt,
                num_inference_steps=30,
                guidance_scale=4.0,
                strength=0.7,
                seed=self.seed + 1,
            )

        result_pil = heal_wrap_seam(
            result_pil,
            inpaint_fn=_sky_seam_inpaint,
            seam_width_px=256,
            feather_px=32,
            # eligible_mask=None: the whole canvas was regenerated in Pass 2, so
            # every row is eligible (no preserved real-photo pixels to protect).
            # Crop to the band instead of handing FLUX the whole panorama —
            # combined with _sky_seam_inpaint's own flux_max downscale, this
            # means only the seam's actual neighbourhood gets downscaled (if
            # at all), instead of the entire multi-thousand-pixel-wide canvas.
            crop_context_px=256,
            debug_dir=self.temp,
            debug_prefix="sky_wrap_seam",
            log_fn=self.log_warning,
        )
        flux.close()

        if self.output is not None:
            result_pil.save(self.output / "panorama_sky.png")

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
        return InPainting.model_names(InPaintingType.FLUX, lora_type=PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2)

    def clean_up(self):
        super().clean_up()
