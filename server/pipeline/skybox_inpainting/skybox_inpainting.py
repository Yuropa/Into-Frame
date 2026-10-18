from typing import Any
from logging import Logger
import math
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
from util.seam_repair import close_wrap_seam, heal_wrap_seam


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


def _sun_from_direction(direction, panorama_h: int, panorama_w: int) -> tuple[int, int]:
    """Panorama pixel for the sun direction the scene is actually lit by.

    Exact inverse of SceneLighting.sun()'s own mapping (see scene/lighting.py): that
    method builds a unit vector [cos(lat)sin(lon), sin(lat), cos(lat)cos(lon)] from
    equirectangular pixel coordinates, and this undoes it. `direction` points FROM the
    scene TOWARD the sun, in the panorama's frame -- the same frame this stage's rows
    and columns are in, so no rotation is involved.

    Preferred over _sun_from_lighting below because the two disagreed, badly. That one
    takes the brightest pixel of the TONEMAPPED LDR environment map after a heavy
    blur, which on the Rainier capture peaked in the horizon haze and returned
    el=-5.6 deg -- "dusk", which then went into the FLUX sky prompt and the gradient
    sky's centre -- while SceneLighting.sun() measured the top-percentile of LINEAR
    radiance and shipped the client a sun at +61.5 deg. The sky the viewer sees and
    the direction their light comes from were ~67 deg apart. Whatever the scene is lit
    by is the one that has to win here.
    """
    x, y, z = (float(direction[0]), float(direction[1]), float(direction[2]))
    latitude = math.asin(max(-1.0, min(1.0, y)))
    longitude = math.atan2(x, z)
    row = (0.5 - latitude / math.pi) * panorama_h - 0.5
    col = (longitude / (2.0 * math.pi) + 0.5) * panorama_w - 0.5
    return (
        int(min(max(round(row), 0), panorama_h - 1)),
        int(min(max(round(col), 0), panorama_w - 1)),
    )


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


def _cloudiness(source_arr: np.ndarray, sky_mask: np.ndarray, cone_cols: "tuple[int, int] | None" = None) -> float:
    """Fraction of the photographed sky that is cloud: bright and unsaturated.
    Measured inside the photographed cone when one is given (the outpainted sky
    is a guess), over every sky pixel otherwise. Rainier's sky: 0.00 in the cone."""
    rgb = source_arr.astype(np.float32) / 255.0
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    cloud = (sat < 0.18) & (mx > 0.65)
    scope = sky_mask.copy()
    if cone_cols is not None:
        c0, c1 = cone_cols
        band = np.zeros_like(scope)
        band[:, max(0, c0):min(scope.shape[1], c1)] = True
        if (scope & band).sum() > 1000:
            scope &= band
    return float(cloud[scope].mean()) if scope.any() else 0.0


def _flatten_horizon(image: PILImage.Image, band_deg: float = 15.0, dark_levels: float = 15.0,
                     foot_deg: float = 1.5, below_deg: float = 0.5, fade_deg: float = 2.5,
                     ) -> "tuple[PILImage.Image, float]":
    """Take invented landscape out of the horizon band of a generated sky.

    FLUX paints land on the skyline however the prompt forbids it (Paris: a
    mountain range and islands, seen through gaps in the far bank). Land is
    darker than the sky above it and stands on the horizon: within `band_deg`
    above the horizon, every connected patch more than `dark_levels` (luma)
    darker than the sky just above the band at its own azimuth AND reaching
    down into the `foot_deg` above the horizon is land, and is filled across
    from the sky either side of it. The reference is per column, not a row
    median, because a sunset row is bright toward the sun and dark away from
    it (Shark Fin). The last `fade_deg` above the horizon fade to the row's
    smoothed haze, which takes the hazy foot of a range the test let through,
    and from `below_deg` under the horizon everything is that haze: that sky
    only shows through gaps in the terrain. Returns (image, fraction of the
    band above the horizon replaced)."""
    from scipy.ndimage import binary_dilation, binary_fill_holes, gaussian_filter, gaussian_filter1d, label
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    h, w = arr.shape[:2]
    horizon = h // 2
    px_per_deg = h / 180.0
    top = max(0, int(horizon - band_deg * px_per_deg))
    foot = max(top, int(horizon - foot_deg * px_per_deg))
    below = min(h, int(horizon + below_deg * px_per_deg))
    fade0 = max(top, int(horizon - fade_deg * px_per_deg))
    band = arr[top:]
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    luma = band @ weights
    # The sky just above the band, smoothed along the row (circular).
    ref_rows = arr[max(0, top - int(2 * px_per_deg)):top + 1] @ weights
    ref = gaussian_filter1d(ref_rows.mean(axis=0), sigma=1.5 * px_per_deg, mode="wrap")
    off = luma < ref[None, :] - dark_levels
    off[below - top:] = False
    # Rows are a circle: label with the seam doubled so a patch across it is one.
    doubled = np.concatenate([off, off], axis=1)
    labels, n = label(doubled, structure=np.ones((3, 3), dtype=int))
    land = np.zeros_like(off)
    if n:
        touching = np.unique(labels[foot - top:below - top])
        touching = touching[touching > 0]
        if touching.size:
            sel = np.isin(labels, touching)
            land = binary_fill_holes(sel[:, :w] | sel[:, w:])
    land[below - top:] = True
    replaced = float(land[:horizon - top].mean())
    # Fill each land span from the sky either side of it along its row, so the
    # glow and gradient of that row carry across; a row with no sky left takes
    # the fill of the row above it.
    grow = binary_dilation(land, iterations=3)
    fill = band.copy()
    cols = np.arange(w)
    for r in range(band.shape[0]):
        sky = ~grow[r]
        if sky.all():
            continue
        if sky.sum() < 8:
            fill[r] = fill[r - 1] if r else np.median(band[r], axis=0)
            continue
        xs = np.flatnonzero(sky)
        for c in range(3):
            fill[r, :, c] = np.interp(cols, xs, band[r, xs, c], period=w)
    # Rows interpolated on their own streak across a wide range (Paris's spanned
    # 77 deg): smooth the fill in both directions before it goes under the mask.
    fill = gaussian_filter(fill, sigma=(1.0 * px_per_deg, 1.0 * px_per_deg, 0), mode=("nearest", "wrap", "nearest"))
    mask = gaussian_filter(grow.astype(np.float32), sigma=2.0)[..., None]
    out_band = band * (1 - mask) + fill * mask
    # The fade to haze: each of the lowest rows toward its own smoothed self.
    haze = gaussian_filter1d(out_band, sigma=3.0 * px_per_deg, axis=1, mode="wrap")
    ramp = np.zeros(band.shape[0], dtype=np.float32)
    if horizon > fade0:
        ramp[fade0 - top:horizon - top] = np.linspace(0.0, 1.0, horizon - fade0, dtype=np.float32)
    ramp[horizon - top:] = 1.0
    out_band = out_band * (1 - ramp[:, None, None]) + haze * ramp[:, None, None]
    out = arr.copy()
    out[top:] = out_band
    return PILImage.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"), replaced


def _sky_prompt(time_label: str, cloudiness: float = 0.0) -> str:
    # The prompt used to ask for cumulus and cirrus whatever the photograph
    # showed, and at strength 0.85 FLUX obliged: a clear saturated-blue Rainier
    # sky came back as a dusk dome over a sea of cloud, whose grey bank stood
    # above the ridge. Describe the sky the photograph actually has.
    if cloudiness < 0.05:
        clouds = "a completely clear, cloudless sky, deep saturated blue at the zenith fading to pale haze at the horizon"
    elif cloudiness < 0.3:
        clouds = "a mostly clear blue sky with a few small, thin, scattered clouds"
    elif cloudiness < 0.7:
        clouds = "a partly cloudy sky with soft natural cumulus clouds"
    else:
        clouds = "an overcast sky, a smooth even layer of grey-white cloud"
    return (
        f"A photorealistic equirectangular 360-degree view of the upper atmosphere, {time_label}. "
        f"Looking straight up into a seamless daytime sky: {clouds}. "
        "Realistic atmospheric haze near the horizon, no landscape, no ground, no sea of clouds, "
        "uniform lighting, ultra-detailed, high quality, panoramic backdrop."
    )


def _sky_structure(source_arr: np.ndarray, sky_mask: np.ndarray, cone_cols: "tuple[int, int] | None" = None) -> float:
    """Fraction of the photographed sky (above the horizon, inside the cone when
    known) more than 25 levels off its row's median colour: clouds of any colour.
    _cloudiness counts bright unsaturated pixels only, and read Shark Fin's sunset
    cirrus -- orange and pink -- as 3% cloud. Measured: Rainier's clear sky 0.024,
    Paris 0.19, Shark Fin 0.20."""
    h, w = source_arr.shape[:2]
    scope = np.ones(w, dtype=bool)
    if cone_cols is not None:
        scope = np.zeros(w, dtype=bool)
        scope[max(cone_cols[0], 0):min(cone_cols[1], w)] = True
    m = sky_mask & scope[None, :] & (np.arange(h)[:, None] < h // 2)
    src = source_arr.astype(np.float32)
    off, total = 0, 0
    for y in np.flatnonzero(m.sum(axis=1) > 50):
        px = src[y][m[y]]
        dev = np.abs(px - np.median(px, axis=0)).max(axis=1)
        off += int((dev > 25).sum())
        total += px.shape[0]
    return off / total if total else 0.0


def _clear_sky(source_arr: np.ndarray, sky_mask: np.ndarray, sun_row: int, sun_col: int,
               cone_cols: "tuple[int, int] | None" = None, n_sun_bins: int = 36) -> np.ndarray:
    """A cloudless sky from the photographed one: colour by elevation, plus sun glow.

    Each row takes the median colour of its sky pixels (smoothed over rows); rows
    below the lowest sky row hold the horizon colour, so wherever the scene's
    geometry fails to cover the old skyline it shows haze rather than whatever a
    generator put behind the mountain (FLUX turned Rainier's buried treeline and
    mountain silhouette into cloud banks, which stood behind the rendered mountain
    as cumulus in a cloudless photograph). The residual against that profile,
    binned by angle from the sun, adds the glow above the horizon.

    The profile comes from the photographed columns (cone_cols) when they hold
    enough sky, and only from blue-dominant pixels: the terrain panorama's sky
    mask also covers snowfields and the haze ObjectClear leaves where trees were,
    which turned the horizon colour beige."""
    from scipy.ndimage import gaussian_filter1d
    h, w = source_arr.shape[:2]
    src = source_arr.astype(np.float64)
    rows = np.full((h, 3), np.nan)
    min_count = max(8, w // 200)
    blue = sky_mask & (src[..., 2] >= src[..., 0])
    scope = np.ones(w, dtype=bool)
    if cone_cols is not None:
        c0, c1 = cone_cols
        cone = np.zeros(w, dtype=bool)
        cone[max(c0, 0):min(c1, w)] = True
        if blue[:, cone].sum() >= 50 * min_count:
            scope = cone
    profile_px = blue & scope[None, :]
    counts = profile_px.sum(axis=1)
    # Above the horizon and a real share of the row: below it the mask's stray
    # "sky" is water reflection and snow, and a row of three pixels is noise.
    need = max(min_count, int(scope.sum()) // 20)
    for r in np.flatnonzero((counts >= need) & (np.arange(h) < h // 2)):
        rows[r] = np.median(src[r][profile_px[r]], axis=0)
    valid = np.flatnonzero(~np.isnan(rows[:, 0]))
    if valid.size == 0:
        return np.broadcast_to(np.array([135, 206, 235], np.uint8), source_arr.shape).copy()
    for c in range(3):
        rows[:, c] = np.interp(np.arange(h), valid, rows[valid, c])
    rows[valid[-1]:] = rows[valid[-1]]  # below the lowest sky row: horizon haze
    rows[: valid[0]] = rows[valid[0]]
    # Smoothed hard enough that where the measured rows stop (the photo's sky meets
    # the mountain) there is no step.
    rows = gaussian_filter1d(rows, sigma=max(4.0, h / 30.0), axis=0, mode="nearest")
    model = np.broadcast_to(rows[:, None, :], (h, w, 3)).copy()

    # Sun glow: what the elevation profile misses, by angle from the sun.
    el = (0.5 - np.arange(h, dtype=np.float64)[:, None] / h) * np.pi
    az = (np.arange(w, dtype=np.float64)[None, :] / w) * 2.0 * np.pi
    sun_el, sun_az = (0.5 - sun_row / h) * np.pi, (sun_col / w) * 2.0 * np.pi
    dot = (np.cos(el) * np.cos(az) * np.cos(sun_el) * np.cos(sun_az)
           + np.cos(el) * np.sin(az) * np.cos(sun_el) * np.sin(sun_az)
           + np.sin(el) * np.sin(sun_el))
    angle = np.arccos(np.clip(dot, -1, 1))
    b = np.clip((angle / np.pi * n_sun_bins).astype(int), 0, n_sun_bins - 1)
    resid = np.zeros((n_sun_bins, 3))
    # Only rows the profile measured: nearer the horizon the mask's "sky" is haze
    # and snow, which came out as blotches of glow along it.
    upper = blue & (np.arange(h)[:, None] <= valid[-1])
    for i in range(n_sun_bins):
        sel = upper & (b == i)
        if sel.sum() >= min_count:
            resid[i] = np.median(src[sel] - model[sel], axis=0)
    resid = gaussian_filter1d(resid, sigma=1.5, axis=0, mode="nearest")
    # Glow brightens; a negative residual is the profile's own spread, not shade.
    # Interpolated between bin centres so it carries no rings.
    centres = (np.arange(n_sun_bins) + 0.5) * np.pi / n_sun_bins
    # Glow is a halo: beyond 45 deg of the sun a positive residual is horizon haze
    # at some other azimuth (Rainier: blotches 140 deg round from the sun).
    resid *= (np.clip(1.0 - centres / np.radians(45.0), 0.0, 1.0) ** 2)[:, None]
    above = np.clip((h // 2 - np.arange(h, dtype=np.float64)) / max(h / 90.0, 1.0), 0.0, 1.0)[:, None]
    for c in range(3):
        model[..., c] += above * np.interp(angle, centres, np.maximum(resid[:, c], 0.0))
    return model.clip(0, 255).astype(np.uint8)


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
        close_sky_wrap_seam: bool = True,
        wrap_seam_band_px: int = 512,
        # Under this fraction of cloud in the photographed sky, the sky is built
        # from the photo's own colours (_clear_sky) instead of generated.
        clear_sky_max_cloud: float = 0.05,
        clear_sky_max_structure: float = 0.08,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.clear_sky_max_cloud = float(clear_sky_max_cloud)
        # ... and under this fraction of it off its row's colour (_sky_structure).
        self.clear_sky_max_structure = float(clear_sky_max_structure)
        self.guidance_scale = guidance_scale
        self.strength = strength
        # Force column 0 and column W-1 to match exactly after the generative seam
        # pass, by spreading their per-row difference back over wrap_seam_band_px
        # columns at each end. See util.seam_repair.close_wrap_seam. false leaves
        # whatever the generative pass produced, which on the stored Paris output
        # is an 11.53-level vertical step in a sky that otherwise varies by 0.145
        # per column.
        self.close_sky_wrap_seam = bool(close_sky_wrap_seam)
        # How far the correction is spread. Bigger is gentler: the steepest
        # gradient it introduces is pi*|delta| / (4*band), so 512 puts Paris's
        # worst row at 0.066 levels per column against the 0.145 the image already
        # varies by. Measured at 256/512/1024 the seam closes to 0.000 at all
        # three and the interior statistics are indistinguishable, so this is
        # chosen for headroom rather than because the outcome is sensitive to it.
        self.wrap_seam_band_px = int(wrap_seam_band_px)


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
      ContextKey.PANORAMA_TERRAIN         — equirectangular source panorama, object-removed +
                                            LoRA-corrected (not the original ContextKey.PANORAMA
                                            -- an un-removed object silhouetted against the sky
                                            would otherwise get excluded from "kept real sky" and
                                            leave a generation artifact right where it stood, and
                                            the sky/terrain split here would disagree with the one
                                            RegionMapStage/HeightMapStage now use for the same
                                            horizon)
      ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN — per-pixel RegionType indices (float32),
                                            classified from that same panorama_terrain
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
            SemanticKey.PANORAMA: ContextKey.PANORAMA_TERRAIN,
            SemanticKey.OUTPUT: ContextKey.PANORAMA_SKY,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        panorama_key, output_key = self._resolved_keys()

        panorama = context.input_panorama(panorama_key)
        if panorama is None:
            self.log_warning("No panorama in context, skipping")
            return context

        type_map = context.input_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_TERRAIN)
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
        measured_sun = lighting.sun() if lighting is not None else None
        if measured_sun is not None and measured_sun.get("direction") is not None:
            sun_row, sun_col = _sun_from_direction(measured_sun["direction"], h, w)
            self.log_info(
                "Sun position from the scene's own key light "
                f"(direction_source: {measured_sun.get('direction_source', 'env map')})"
            )
        elif lighting is not None:
            # sun() returns None only for a degenerate (uniform) environment map, where
            # there is no key light to agree with in the first place.
            sun_row, sun_col = _sun_from_lighting(lighting.ldr, h, w)
            self.log_info("Sun position from LuxDiT lighting (no key light measured)")
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

        cone_cols = None
        try:
            from util.captured_cone import captured_cone_from_context
            cone = captured_cone_from_context(context, w, h)
            if cone is not None:
                u0, u1 = cone.u_range
                cone_cols = (int(u0), int(u1))
        except Exception:
            cone_cols = None
        cloudiness = _cloudiness(source_arr, sky_mask, cone_cols)
        self.log_info(f"Photographed sky: {cloudiness:.0%} cloud")

        structure = _sky_structure(source_arr, sky_mask, cone_cols)
        self.log_info(f"Photographed sky structure: {structure:.1%} of it off its row's colour")
        if cloudiness < self.config.clear_sky_max_cloud and structure < self.config.clear_sky_max_structure:
            # A clear photographed sky: nothing for a generator to add, and what it
            # adds is cloud banks along the buried skyline (see _clear_sky).
            # The model alone, not composited over the terrain panorama's sky: that
            # carries ghosts of the removed trees and the faded mountain.
            result_pil = PILImage.fromarray(_clear_sky(source_arr, sky_mask, sun_row, sun_col, cone_cols))
            self.log_info("Clear sky: elevation profile + sun glow from the photographed sky, no FLUX pass")
            self.advance_progress(task)
        else:
            result_pil = self._flux_sky(context, gradient_composite, source_arr, sky_mask, cloudiness,
                                        time_label, task, h, w)

        # --- Pass 4: close whatever step Pass 3 left, in closed form ---
        # Pass 3 is a generative fill and has no continuity constraint, so it
        # closes the seam only as well as it happens to -- and nothing checked.
        # It did not: this stage's own stored output for the Paris capture has an
        # 11.53-level step at the wrap column against a 0.145 median
        # adjacent-column difference over the whole sky, i.e. the hard vertical
        # line Pass 3 exists to remove, still there and 80x the surrounding
        # variation. Pass 3 is still what invents cloud STRUCTURE across the
        # join; this makes the join itself exact afterwards. See close_wrap_seam.
        if self.config.close_sky_wrap_seam:
            result_pil = close_wrap_seam(
                result_pil,
                band_px=self.config.wrap_seam_band_px,
                log_fn=self.log_info,
                label="sky wrap",
            )

        if self.output is not None:
            result_pil.save(self.output / "panorama_sky.png")

        context.add_panorama(output_key, Panorama(result_pil))

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def _flux_sky(self, context, gradient_composite, source_arr, sky_mask, cloudiness, time_label, task, h, w):
        """Passes 2-3: FLUX detail over the whole panorama, then the wrap seam."""
        # --- Pass 2: FLUX detail over the whole panorama ---
        # Full-canvas mask — see the class docstring for why this replaced the old
        # non-sky-only fill mask (it was the source of a cloud-ridge artifact tracing
        # the buried mountain silhouette).
        full_mask_arr = np.ones((h, w), dtype=bool)
        fill_mask_pil = PILImage.fromarray((full_mask_arr * 255).astype(np.uint8), mode="L")

        # Lead with the FLUX_DEV_PANORAMA_LORA_2 trigger sequence so the LoRA activates its
        # spherical/HDRI generation mode instead of its base perspective-photo bias.
        prompt = lora_prompt_prefix(PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2) + _sky_prompt(time_label, cloudiness)
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

        # FLUX paints land at the horizon however the prompt forbids it (Paris:
        # a mountainous island on the skyline, seen through gaps in the far
        # bank). The band at the horizon is haze in any real sky.
        result_pil, flattened = _flatten_horizon(result_pil)
        if flattened:
            self.log_info(f"Horizon band: replaced {flattened:.1%} of it (land or dark structure) with the row's own haze")
        return result_pil

    def has_expected_output(self, context: PipelineContext) -> bool:
        return (
            context.image(ContextKey.PANORAMA_SKY_MASK) is not None
            and context.panorama(ContextKey.PANORAMA_SKY) is not None
        )

    def model_names(self) -> list[str]:
        return InPainting.model_names(InPaintingType.FLUX, lora_type=PanoramaLoraType.FLUX_DEV_PANORAMA_LORA_2)

    def clean_up(self):
        super().clean_up()
