from typing import Any, Optional
from logging import Logger

import numpy as np
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.panorama.placement import placement_fov_deg
from util.depth_utils import Depth

# {"meters", "source", ...}: the camera height the scene is built at. Read by
# Height Map and Region Map in preference to their own camera_height_meters.
CAMERA_HEIGHT_KEY = "camera_height_estimate"

# Classes whose real height is narrow enough to set the scene's scale, with the
# height:width a plausible single instance has (a "person" box 76x92 is a crowd).
_HEIGHT_ANCHORS = {
    "person": (1.70, 1.6), "car": (1.50, 0.25), "bus": (3.20, 0.25), "motorcycle": (1.10, 0.4),
    "bicycle": (1.10, 0.4), "bench": (0.90, 0.2), "fire_hydrant": (0.75, 0.8), "trash_bin": (1.00, 0.6),
}


class PanoramaDepthCalibrationConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        num_bins: int = 14,
        min_samples: int = 300,
        max_extrapolation_factor: float = 20.0,
        max_depth_m: Optional[float] = None,
        far_fraction: float = 0.7,
        anchor_to_camera_height: bool = True,
        camera_height_meters: float = 1.0,
        anchor_min_elevation_deg: float = 45.0,
        estimate_camera_height: bool = True,
        max_camera_height_m: float = 60.0,
        max_anchor_spread: float = 2.5,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Number of quantile bins used to build the raw-prediction -> metric-depth lookup
        # curve. More bins track a non-linear response curve more closely but need more
        # overlap samples per bin to stay robust to outliers (inpainted foreground objects).
        self.num_bins = num_bins
        # Minimum number of valid (non-sky, finite, positive) overlap samples required to
        # attempt calibration at all. Below this, a scene's hero photo barely overlaps the
        # panorama (extreme FOV, mostly-sky shot) and any fit would be unreliable — skip and
        # leave the panorama depth as DAP produced it.
        self.min_samples = min_samples
        # Caps _apply_panorama_depth_curve's log-space extrapolation for raw predictions
        # beyond the fitted (near-field-only) range at this many times the farthest/nearest
        # fitted median, instead of letting a slope estimated from a handful of tail bins
        # extrapolate unbounded — see that function's docstring for why unbounded blow-up
        # silently discards real distant terrain (mountains) as "out of bounds" once it
        # dominates the metric scale.
        self.max_extrapolation_factor = max_extrapolation_factor
        # Hard ceiling (metres) on the farthest calibrated depth, matching the terrain
        # grid's own footprint (Height Map's grid_size_meters / 2 — anything farther
        # can never land inside that grid anyway). Enforced by compressing only the
        # FAR end of the range (see _compress_far_range), leaving the near field at
        # its calibrated metric scale.
        #
        # This used to be a uniform rescale of the whole map (scale = max_depth_m /
        # farthest non-sky pixel). That preserved relative distances exactly, but the
        # scale is set by the most distant thing in the scene — unbounded for any
        # landscape photo — and it silently broke the pipeline's other metric anchor,
        # camera_height_meters, which HeightMapGenerator relies on to place ground at
        # -camera_height under the camera. Measured on a Mount Rainier capture: a
        # kilometres-distant summit forced a ~6.5x global shrink, ground 1.37 m below
        # the camera came back at 0.21 m of depth, and the reconstructed terrain
        # surface ended up ~0.9 m ABOVE eye level with the camera buried inside it.
        # A flat clamp is still wrong for the opposite reason (it collapses a 120 m
        # ridge and a 900 m peak onto one wall); the compression curve keeps far
        # content ordered and distinguishable without touching the near field.
        # None disables (only the relative max_extrapolation_factor cap above applies).
        self.max_depth_m = max_depth_m
        # Where the far field begins, as a fraction of the deepest non-sky metric
        # depth. Neither depth model measures distance past a few tens of metres: on
        # the Rainier capture DA3 puts the summit at 53 m with a 55 m maximum and the
        # calibrated panorama tops out at 36 m, for a mountain that is kilometres
        # away. The saturated band -- everything at or near the maximum -- is the
        # far field, whatever number it carries, and FarFieldBackdropStage stands it
        # up behind the terrain grid while TerrainReconstructionStage keeps its
        # ridges out of the grid. Written as FAR_FIELD_START_M for both.
        self.far_fraction = float(far_fraction)
        # DA3's metric scale on a single photograph is only approximately metric
        # (the Rainier capture came out ~3x too small: the ground straight below the
        # camera at 0.3 m). The one metric fact the pipeline already commits to is
        # camera_height_meters (Height Map pins the nadir to it), so after the curve
        # fit both maps are rescaled so the ground under the camera sits at that
        # height: the nadir ray at elevation -e below the horizon hits flat ground at
        # h / sin(e), and the median of depth * sin(e) over the rows steeper than
        # anchor_min_elevation_deg is the camera height the depth map implies.
        self.anchor_to_camera_height = bool(anchor_to_camera_height)
        self.camera_height_meters = float(camera_height_meters)
        self.anchor_min_elevation_deg = float(anchor_min_elevation_deg)
        # Measure the camera height from the photograph instead of assuming
        # camera_height_meters, when the photo carries the evidence: the ground
        # plane fitted to DA3's photo depth gives the height in DA3's units, and
        # people / cars of known size give the scale of those units. Paris was shot
        # from a bridge ~10 m up; built at 1 m everything near the camera came out
        # ~10x small and the Seine a pool. Without a usable anchor (Rainier has none)
        # or with anchors that disagree by more than max_anchor_spread, the
        # configured height stands. Published as CAMERA_HEIGHT_KEY for Height Map
        # and Region Map.
        self.estimate_camera_height = bool(estimate_camera_height)
        self.max_camera_height_m = float(max_camera_height_m)
        self.max_anchor_spread = float(max_anchor_spread)


def _hero_photo_rays(h: int, w: int, hfov_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-pixel unit ray directions for the hero photo's pinhole camera, looking down +Z.

    Reproduces perspective_rays() from server/pipeline/panorama/pano_utils.py in plain numpy.
    That module is not imported directly here because its top-level import of
    worldgen.utils.general_utils ties it to the panorama-generation worker environment; this
    stage runs locally alongside HeightMapStage/RegionMapStage and only needs this one
    self-contained formula.
    """
    hfov = np.radians(hfov_deg)
    vfov = 2.0 * np.arctan(np.tan(hfov / 2.0) * h / w)

    u = (np.arange(w, dtype=np.float64) + 0.5) / w
    v = (np.arange(h, dtype=np.float64) + 0.5) / h
    uu, vv = np.meshgrid(u, v)

    phi = (uu - 0.5) * hfov
    theta = (vv - 0.5) * vfov

    x = np.cos(theta) * np.sin(phi)
    y = np.sin(theta)
    z = np.cos(theta) * np.cos(phi)
    norm = np.sqrt(x * x + y * y + z * z)
    return x / norm, y / norm, z / norm


def _equirect_uv_for_rays(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Equirectangular UV in [0, 1) for unit ray directions — matches the
    phi=atan2(x,z) / theta=asin(y) convention in pano_utils.map_image_to_pano."""
    phi = np.arctan2(x, z)
    theta = np.arcsin(np.clip(y, -1.0, 1.0))
    u = (phi / np.pi + 1.0) / 2.0
    v = theta / np.pi + 0.5
    return u, v


def _fit_panorama_depth_curve(
    da3_depth: np.ndarray,
    da3_placement_fov_deg: float,
    dap_pred_raw: np.ndarray,
    sky_mask: Optional[np.ndarray],
    num_bins: int = 14,
    min_samples: int = 300,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Fits a monotonic lookup curve from DAP's raw (0-1) prediction to real metric depth, using
    the region where the hero photo (metric depth `da3_depth`, from Depth Anything 3) was
    placed into the panorama.

    da3_placement_fov_deg must be the horizontal FOV actually used to place the photo into the
    panorama (server/pipeline/panorama/panorama.py sends 2x the camera's own estimated FOV to
    WorldGen) — the correspondence has to match the real placement, not the true camera FOV.

    dap_pred_raw is DAP's native 0-1 output (i.e. panorama depth with the *100 metres
    assumption undone) — same shape as the target panorama.

    Returns (centers_sorted, medians_monotonic) — apply via _apply_panorama_depth_curve — or
    None if too few valid overlap samples survive filtering, signalling the caller should leave
    the panorama depth uncalibrated.
    """
    h, w = da3_depth.shape
    dap_h, dap_w = dap_pred_raw.shape

    x, y, z = _hero_photo_rays(h, w, da3_placement_fov_deg)
    u, v = _equirect_uv_for_rays(x, y, z)
    u_pix = np.clip(np.round(u * (dap_w - 1)).astype(np.int64), 0, dap_w - 1).ravel()
    v_pix = np.clip(np.round(v * (dap_h - 1)).astype(np.int64), 0, dap_h - 1).ravel()

    da3_samples = da3_depth.astype(np.float64).ravel()
    dap_samples = dap_pred_raw[v_pix, u_pix].astype(np.float64)

    valid = np.isfinite(da3_samples) & (da3_samples > 0) & np.isfinite(dap_samples) & (dap_samples > 0)
    if sky_mask is not None and sky_mask.shape == dap_pred_raw.shape:
        valid &= ~sky_mask[v_pix, u_pix]

    if valid.sum() < min_samples:
        return None

    da3_valid = da3_samples[valid]
    dap_valid = dap_samples[valid]

    edges = np.quantile(dap_valid, np.linspace(0.0, 1.0, num_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bin_idx = np.clip(np.digitize(dap_valid, edges) - 1, 0, num_bins - 1)

    centers, medians = [], []
    for b in range(num_bins):
        in_bin = bin_idx == b
        if not np.any(in_bin):
            continue
        centers.append(float(np.median(dap_valid[in_bin])))
        medians.append(float(np.median(da3_valid[in_bin])))

    if len(centers) < 2:
        return None

    order = np.argsort(centers)
    centers_sorted = np.array(centers)[order]
    # Depth should not decrease as DAP's raw prediction increases — enforce that prior so a
    # noisy bin can't punch a dip into the curve.
    medians_monotonic = np.maximum.accumulate(np.array(medians)[order])
    return centers_sorted, medians_monotonic


def _apply_panorama_depth_curve(
    dap_pred_raw: np.ndarray,
    centers_sorted: np.ndarray,
    medians_monotonic: np.ndarray,
    max_extrapolation_factor: float = 20.0,
) -> np.ndarray:
    """Applies a curve fitted by _fit_panorama_depth_curve to a (possibly different) raw
    DAP prediction array — the fitted raw→metric response curve is treated as a property of
    the DAP model, not of the specific image it ran on.

    The hero photo's overlap with the panorama only ever covers its own near-field content
    (a few to a few tens of metres) — it never sees genuinely distant terrain (mountains,
    open horizons). np.interp flat-clamps any raw value outside the fitted range to the
    curve's nearest endpoint, which collapses every one of those far pixels onto a single
    depth (whatever the hero photo's own farthest point happened to be) — mountains a
    kilometre away and a tree just past the fitted range end up at the *same* calibrated
    depth. Extrapolate instead, in log-depth space (DAP's raw output approaches its ceiling
    asymptotically as true depth grows, so metric depth should keep growing past the fitted
    range, not flatten) using the slope from a small tail of the fitted curve.

    DAP's raw output saturates well before true distance does: a mountainside a few hundred
    metres out and hard-clamped sky pixels both read at (or past) the same near-1.0 ceiling,
    right at the edge of a curve fit only against tens of metres of hero-photo overlap. A
    slope estimated from that fit's last few bins, applied over the (potentially large) gap
    out to the ceiling, is extremely sensitive to noise once exponentiated — a small slope
    error compounds into a wildly oversized, effectively fabricated depth (observed: tens of
    kilometres) for every pixel sharing that saturated raw value. That single huge, near-
    uniform outlier then dominates anything downstream that treats panorama depth as a
    trustworthy metric scale (height-map Y-bounds checks, the debug point-cloud export's
    colour normalization), silently discarding real terrain — mountains included — as
    "out of bounds" instead of just "far".

    Cap the extrapolated result at max_extrapolation_factor times the farthest (nearest)
    fitted median instead of letting it grow unbounded. This can't recover the true distance
    for pixels the model genuinely can't resolve — there's no signal left to extrapolate from
    once DAP saturates — but it keeps "far" bounded to a plausible multiple of the fit's own
    scale, monotonically ordered past the fitted range, rather than an arbitrary, ungrounded
    number.
    """
    x = dap_pred_raw.astype(np.float64).ravel()
    corrected = np.interp(x, centers_sorted, medians_monotonic)

    tail = min(3, len(centers_sorted) - 1)
    if tail >= 1:
        log_med = np.log(np.maximum(medians_monotonic, 1e-3))

        far_mask = x > centers_sorted[-1]
        if np.any(far_mask):
            dx = centers_sorted[-1] - centers_sorted[-1 - tail]
            slope = (log_med[-1] - log_med[-1 - tail]) / dx if abs(dx) > 1e-9 else 0.0
            extrapolated = np.exp(log_med[-1] + slope * (x[far_mask] - centers_sorted[-1]))
            far_cap = medians_monotonic[-1] * max_extrapolation_factor
            corrected[far_mask] = np.minimum(extrapolated, far_cap)

        near_mask = x < centers_sorted[0]
        if np.any(near_mask):
            dx = centers_sorted[tail] - centers_sorted[0]
            slope = (log_med[tail] - log_med[0]) / dx if abs(dx) > 1e-9 else 0.0
            extrapolated = np.exp(log_med[0] + slope * (x[near_mask] - centers_sorted[0]))
            near_floor = medians_monotonic[0] / max_extrapolation_factor
            corrected[near_mask] = np.maximum(extrapolated, near_floor)

    corrected = np.clip(corrected, 1e-3, 1e6)
    return corrected.reshape(dap_pred_raw.shape).astype(np.float32)


def _camera_height_scale(
    depth: np.ndarray,
    camera_height_m: float,
    min_elevation_deg: float,
    sky_mask: Optional[np.ndarray],
) -> tuple[float, float]:
    """Scale that puts the ground under the camera at camera_height_m.

    Every pixel in the rows steeper than min_elevation_deg below the horizon is
    assumed to be ground (flat enough there: at 45 deg and below, a 1 m camera
    sees at most 1.4 m away). Each gives a camera-height estimate
    depth * sin(elevation); the median is robust to the odd rock or flower.
    Returns (scale, implied_height), or (1.0, nan) when nothing can be measured.
    """
    h, w = depth.shape
    rows = np.arange(h)
    elevation = (0.5 - (rows + 0.5) / h) * 180.0          # +90 at the top row, -90 at the bottom
    steep = elevation <= -abs(min_elevation_deg)
    if not steep.any():
        return 1.0, float("nan")
    band = depth[steep]
    sin_e = np.sin(np.radians(-elevation[steep]))[:, None]
    valid = np.isfinite(band) & (band > 0)
    if sky_mask is not None and np.asarray(sky_mask).shape == depth.shape:
        valid &= ~np.asarray(sky_mask, dtype=bool)[steep]
    if valid.sum() < 100:
        return 1.0, float("nan")
    implied = float(np.median((band * sin_e)[valid]))
    if not np.isfinite(implied) or implied <= 1e-3:
        return 1.0, float("nan")
    return float(camera_height_m / implied), implied


def _saturation_plateau(depth: np.ndarray, sky_mask: Optional[np.ndarray]) -> float:
    """The depth the model saturates at: the largest non-sky value that survives a
    median filter. The mountain can be a tiny fraction of the non-sky pixels (under
    0.5 % on Rainier, with the meadow filling the lower hemisphere), so no high
    percentile finds it; the filtered maximum does, while a lone hot pixel does not."""
    from scipy.ndimage import median_filter
    h, w = depth.shape
    step = max(1, w // 1024)
    small = depth[::step, ::step].astype(np.float32)
    valid = np.isfinite(small) & (small > 0)
    if sky_mask is not None and np.asarray(sky_mask).shape == depth.shape:
        valid &= ~np.asarray(sky_mask, dtype=bool)[::step, ::step]
    if not valid.any():
        return float(np.nanmax(depth))
    filled = np.where(valid, small, 0.0)
    filtered = median_filter(filled, size=5)
    return float(filtered[valid].max())


def _compress_far_range(
    depth: np.ndarray,
    max_depth_m: float,
    knee_fraction: float = 0.6,
) -> np.ndarray:
    """
    Fit `depth` inside max_depth_m by compressing the FAR end only, leaving the
    near field metrically intact.

    This replaces a uniform global rescale (scale = max_depth_m / farthest
    non-sky pixel, applied to every pixel). That rescale preserved relative
    distances exactly -- a ridge twice as far as another stayed twice as far --
    which sounds like the conservative choice and is the wrong invariant to
    protect. The pipeline has a second, independent, and far more load-bearing
    metric anchor: camera_height_meters. HeightMapGenerator unprojects ground
    as Y = depth * sin(phi) and expects flat ground under the camera to land at
    -camera_height_meters; every panorama UV, grid placement and elevation
    downstream is referred to that same origin. A uniform scale silently
    invalidates it.

    And the scale is set by the single most distant thing in the scene, which
    for any landscape photo is effectively unbounded. Measured on a Mount
    Rainier capture: the mountain is kilometres away, so fitting it into 100 m
    divided the whole map by ~6.5x. Ground 1.37 m below the camera came back as
    0.21 m of depth, putting the terrain surface ~0.9 m ABOVE the eye instead of
    1 m below it -- 97% of the reconstructed grid ended up above camera height,
    the camera was buried inside its own terrain, and every near-field panorama
    UV collapsed into a thin band either side of the horizon (v in [0.29, 0.54]),
    which is what turned the baked ground texture into a radial pinwheel of sky
    and snow. None of that is visible as a depth-map artifact; it only surfaces
    three stages later as geometry and texturing failures.

    So: identity below the knee, then a bounded exponential approach to
    max_depth_m above it.

        d' = d                                                    d <= knee
        d' = knee + (max - knee) * (1 - exp(-(d - knee)/(max - knee)))

    Continuous and C1 at the knee (both sides have unit slope there), strictly
    monotonic everywhere, and asymptotic to max_depth_m so nothing can exceed
    the budget without a hard clamp. Distant content keeps its ordering and
    stays distinguishable -- the real concern behind the original flat-clamp
    objection -- it just stops being proportional, which it never meaningfully
    was once a 15 km mountain had to fit in a 200 m grid.

    No sky_mask parameter: the curve is fixed by (max_depth_m, knee_fraction)
    alone rather than by the data's own maximum, so there is no basis pool to
    contaminate, and sky saturating at the ceiling can no longer drag real
    terrain down with it. That also makes it trivially reproducible across the
    two depth maps that must agree (PANORAMA_DEPTH and PANORAMA_OBJECT_DEPTH):
    applying the same deterministic function to both places the same real-world
    point at the same distance, with no scale factor to thread between them.

    knee_fraction: where identity ends, as a fraction of max_depth_m.
    """
    if max_depth_m is None or max_depth_m <= 0:
        return depth.astype(np.float32)

    knee = float(np.clip(knee_fraction, 0.0, 0.99)) * max_depth_m
    tail = max_depth_m - knee
    if tail <= 0:
        return np.minimum(depth, max_depth_m).astype(np.float32)

    d = depth.astype(np.float64)
    over = d > knee
    out = d.copy()
    out[over] = knee + tail * (1.0 - np.exp(-(d[over] - knee) / tail))
    # Asymptotic, so this only guards float error and non-finite input.
    return np.minimum(out, max_depth_m).astype(np.float32)


def calibrate_panorama_depth(
    da3_depth: np.ndarray,
    da3_placement_fov_deg: float,
    dap_pred_raw: np.ndarray,
    sky_mask: Optional[np.ndarray],
    num_bins: int = 14,
    min_samples: int = 300,
    max_extrapolation_factor: float = 20.0,
    max_depth_m: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Fits a curve (see _fit_panorama_depth_curve) and applies it to dap_pred_raw in one call.
    Returns None if too few valid overlap samples survive filtering."""
    curve = _fit_panorama_depth_curve(
        da3_depth, da3_placement_fov_deg, dap_pred_raw, sky_mask, num_bins, min_samples
    )
    if curve is None:
        return None
    corrected = _apply_panorama_depth_curve(dap_pred_raw, *curve, max_extrapolation_factor=max_extrapolation_factor)
    if max_depth_m is not None:
        corrected = _compress_far_range(corrected, max_depth_m)
    return corrected


class PanoramaDepthCalibrationStage(PipelineStage):
    """
    Recalibrates both panorama depth maps (DAP's equirectangular depth, which converts its
    raw 0-1 output to metres via a flat, unverified *100 assumption and hard-clamps sky to
    1.0 — see server/pipeline/panorama_depth/pano_depth_imp.py) against ContextKey.DEPTH, the
    genuine metric depth Depth Anything 3 already estimated for the original hero photo before
    the panorama existed.

    The hero photo was placed into the panorama by WorldGen using a known ray mapping (see
    server/pipeline/panorama/pano_utils.py); this stage reproduces that mapping to find which
    panorama pixels came from the real photo, fits a monotonic lookup curve from DAP's raw
    prediction to DA3's metric depth over that overlap, and applies it across the whole
    panorama — replacing the fixed *100 guess with a scene-specific, empirically anchored
    scale. Distant terrain (very often beyond DAP's uncalibrated ~100 m ceiling, where it was
    previously indistinguishable from the hard-clamped sky value) is no longer flattened onto
    a uniform-radius arc.

    The curve is fit against ContextKey.PANORAMA_OBJECT_DEPTH (DAP run on the original,
    unmodified panorama) rather than ContextKey.PANORAMA_DEPTH (DAP run on panorama_terrain,
    after foreground inpainting) — the overlap region has to be real, untouched photo content
    to match DA3's genuine hero-photo depth. If part of the hero photo's footprint fell inside
    the foreground-removal mask, PANORAMA_DEPTH there is a hallucinated re-estimate, not the
    same scene DA3 measured. The fitted curve (a property of DAP's raw→metric response, not of
    a specific image) is then applied to both depth maps: PANORAMA_OBJECT_DEPTH (consumed by
    region/object analysis on the original panorama) and PANORAMA_DEPTH (consumed by Height
    Map / terrain reconstruction on the post-inpainting terrain plate).

    If too little of the hero photo's real (non-hallucinated) footprint survives filtering
    (e.g. a very narrow FOV, or a mostly-sky photo), calibration is skipped and both depth
    maps are left as DAP produced them.

    Input keys  (SemanticKey.DEPTH, SemanticKey.INTRINSICS) → ContextKey.DEPTH, ContextKey.INTRINSICS
    Output key  (SemanticKey.OUTPUT)                        → ContextKey.PANORAMA_DEPTH (overwritten in place)
                                                               ContextKey.PANORAMA_OBJECT_DEPTH (overwritten in place, if present)
    """

    @classmethod
    def config_class(cls) -> type[PanoramaDepthCalibrationConfiguration]:
        return PanoramaDepthCalibrationConfiguration

    def __init__(self, config: PanoramaDepthCalibrationConfiguration) -> None:
        super().__init__(config)

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.DEPTH: ContextKey.DEPTH,
            SemanticKey.INTRINSICS: ContextKey.INTRINSICS,
            SemanticKey.OUTPUT: ContextKey.PANORAMA_DEPTH,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: PanoramaDepthCalibrationConfiguration = self.config
        depth_key, intrinsics_key, panorama_depth_key = self._resolved_keys()

        task = self.create_progress(1, "Panorama Depth Calibration…")

        da3_depth = context.input_depth(depth_key)
        intrinsics = context.input_intrinsics(intrinsics_key)
        panorama_depth = context.input_depth(panorama_depth_key)
        panorama_object_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)
        sky_mask = context.input_sky_mask()

        if da3_depth is None or intrinsics is None or panorama_depth is None:
            self.log_warning("Missing metric depth, intrinsics, or panorama depth — skipping calibration")
            self.finish_progress(task)
            return context

        # Fit against the original (pre-foreground-inpainting) panorama depth when available —
        # see class docstring for why. Falls back to PANORAMA_DEPTH itself if it's missing.
        fit_source = panorama_object_depth if panorama_object_depth is not None else panorama_depth
        dap_pred_raw_fit = fit_source.depth.astype(np.float64) / 100.0

        # The photo is placed into the panorama at PLACEMENT_FOV_SCALE x its measured
        # FOV (see pipeline/panorama/placement.py) -- the correspondence here has to
        # use the same width or the overlap compares the wrong pixels.
        curve = _fit_panorama_depth_curve(
            da3_depth.depth,
            placement_fov_deg(intrinsics.fov),
            dap_pred_raw_fit,
            sky_mask,
            num_bins=cfg.num_bins,
            min_samples=cfg.min_samples,
        )

        if curve is None:
            self.log_warning(
                "Not enough hero-photo/panorama overlap to calibrate — leaving panorama depth unchanged"
            )
            self.finish_progress(task)
            return context

        before_min, before_max = panorama_depth.min(), panorama_depth.max()
        dap_pred_raw = panorama_depth.depth.astype(np.float64) / 100.0
        calibrated_arr = _apply_panorama_depth_curve(
            dap_pred_raw, *curve, max_extrapolation_factor=cfg.max_extrapolation_factor
        )
        camera_height = cfg.camera_height_meters
        estimate = {"meters": camera_height, "source": "config"}
        if cfg.estimate_camera_height:
            try:
                measured = self._measure_camera_height(context, da3_depth.depth, intrinsics, panorama_depth.depth.shape)
            except Exception as exc:   # an estimate must never cost the run
                self.log_warning(f"Camera height estimate failed ({exc}); using {camera_height:.2f} m")
                measured = None
            if measured is not None:
                estimate = measured
                camera_height = measured["meters"]
        context.add_object(CAMERA_HEIGHT_KEY, estimate)
        anchor_scale = 1.0
        if estimate.get("source") == "photo":
            # Measured from the photograph: its size anchors give DA3's metric scale
            # directly, and the curve maps the panorama onto DA3's units. The nadir
            # is generated content and disagreed with the photo by 1.55x on Paris
            # (tour boat 66 m away instead of ~40), so it is reported, not used.
            anchor_scale = float(estimate["scale"])
            _, implied = _camera_height_scale(calibrated_arr, camera_height, cfg.anchor_min_elevation_deg, sky_mask)
            calibrated_arr = calibrated_arr * anchor_scale
            self.log_info(
                f"Depth scaled x{anchor_scale:.2f} by the photo's size anchors (camera {camera_height:.2f} m); "
                f"the generated nadir would have implied {implied * anchor_scale:.2f} m"
            )
        elif cfg.anchor_to_camera_height:
            anchor_scale, implied = _camera_height_scale(
                calibrated_arr, camera_height, cfg.anchor_min_elevation_deg, sky_mask,
            )
            if anchor_scale != 1.0:
                calibrated_arr = calibrated_arr * anchor_scale
                self.log_info(
                    f"Depth anchored to the camera height: the nadir implies {implied:.2f} m, "
                    f"camera height is {camera_height:.2f} m ({estimate['source']}) -> both maps scaled x{anchor_scale:.2f}"
                )
            else:
                self.log_info("Depth anchor: nadir height could not be measured, scale unchanged")
        # Far-range compression, not a global scale: the near field has to stay
        # metric because camera_height_meters anchors everything downstream to it.
        # The curve depends only on max_depth_m, so applying the identical
        # function to PANORAMA_OBJECT_DEPTH below keeps both maps agreeing on
        # where any given real-world point sits, with no scale factor to thread
        # between them. See _compress_far_range.
        # The UNCOMPRESSED calibrated depth, kept alongside. PANORAMA_DEPTH below is
        # compressed into max_depth_m because everything that builds the near
        # terrain grid is tuned to that budget; the far field (the mountain that
        # really is kilometres away) is represented by FarFieldBackdropStage from
        # this map instead, at a scale the grid cannot hold. Ridge chains whose
        # metric depth lies beyond the grid are handed to that stage too (see
        # TerrainReconstructionStage.ridge_max_metric_depth_m).
        metric = Depth(calibrated_arr.astype(np.float32))
        context.add_depth(ContextKey.PANORAMA_DEPTH_METRIC, metric)
        non_sky = calibrated_arr
        if sky_mask is not None and np.asarray(sky_mask).shape == calibrated_arr.shape:
            non_sky = calibrated_arr[~np.asarray(sky_mask, dtype=bool)]
        finite = non_sky[np.isfinite(non_sky)]
        if finite.size:
            # The saturation plateau is the top of the distribution, not a high
            # percentile of it: on Rainier the mountain is under 0.5 % of non-sky
            # pixels, so p99.5 (15 m) sat on the meadow and the far field started at
            # 10.8 m. p99.9 stays off single-pixel outliers.
            p50, p90, p995 = np.percentile(finite, [50, 90, 99.5])
            plateau = _saturation_plateau(calibrated_arr, sky_mask)
            far_start = float(cfg.far_fraction * plateau)
            context.add_object(ContextKey.FAR_FIELD_START_M, far_start)
            self.log_info(
                f"Metric panorama depth (uncompressed, non-sky): p50 {p50:.1f} m, p90 {p90:.1f} m, "
                f"p99.5 {p995:.1f} m, saturation plateau {plateau:.0f} m — far field starts at "
                f"{far_start:.1f} m ({cfg.far_fraction:.2f} x the plateau)"
            )
        if cfg.max_depth_m is not None:
            calibrated_arr = _compress_far_range(calibrated_arr, cfg.max_depth_m)
        calibrated = Depth(calibrated_arr)
        if self.temp is not None:
            calibrated.save_debug_image(self.temp / "pano_depth_calibrated.png")

        self.log_info(
            f"Calibrated panorama depth {before_min:.1f} → {before_max:.1f} m (raw) "
            f"into {calibrated.min():.1f} → {calibrated.max():.1f} m (calibrated"
            + (
                f", far range compressed above {0.6 * cfg.max_depth_m:.0f} m "
                f"to fit {cfg.max_depth_m:.0f} m)"
                if cfg.max_depth_m is not None else ")"
            )
        )
        context.add_depth(panorama_depth_key, calibrated)

        if panorama_object_depth is not None:
            obj_before_min, obj_before_max = panorama_object_depth.min(), panorama_object_depth.max()
            calibrated_object_arr = _apply_panorama_depth_curve(
                dap_pred_raw_fit, *curve, max_extrapolation_factor=cfg.max_extrapolation_factor
            )
            calibrated_object_arr = calibrated_object_arr * anchor_scale
            if cfg.max_depth_m is not None:
                # The identical curve PANORAMA_DEPTH got above -- it's a pure
                # function of max_depth_m, so both maps place the same real-world
                # point at the same distance without sharing any fitted state.
                calibrated_object_arr = _compress_far_range(
                    calibrated_object_arr, cfg.max_depth_m
                )
            calibrated_object = Depth(calibrated_object_arr)
            if self.temp is not None:
                calibrated_object.save_debug_image(self.temp / "pano_object_depth_calibrated.png")
            self.log_info(
                f"Calibrated panorama object depth {obj_before_min:.1f} → {obj_before_max:.1f} m (raw) "
                f"into {calibrated_object.min():.1f} → {calibrated_object.max():.1f} m (calibrated"
                + (
                    f", far range compressed to fit {cfg.max_depth_m:.0f} m)"
                    if cfg.max_depth_m is not None else ")"
                )
            )
            context.add_depth(ContextKey.PANORAMA_OBJECT_DEPTH, calibrated_object)

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def _measure_camera_height(self, context: PipelineContext, da3: np.ndarray, intrinsics, pano_shape) -> Optional[dict]:
        """Camera height from the photograph, or None when the evidence is not there.

        1. Ground plane: RANSAC over DA3's photo point cloud below the image middle,
           normal within 20 deg of vertical; the plane's distance from the camera is
           the height in DA3 units.
        2. Scale: every photographed person / car / ... (by its photo box, kept only
           when its height:width is one instance's) measured at DA3's median depth in
           its box; prior height / measured height is the metric scale.
        Needs at least one anchor; several must agree within max_anchor_spread."""
        from util.captured_cone import captured_cone_from_context
        from pipeline.object_typing.categories import normalize_category
        cfg: PanoramaDepthCalibrationConfiguration = self.config
        depth = np.asarray(da3, dtype=np.float64)
        dh, dw = depth.shape[:2]
        # Intrinsics are given at their own resolution; bring them to the depth map's.
        iw = float(getattr(intrinsics, "width", dw) or dw)
        ih = float(getattr(intrinsics, "height", dh) or dh)
        fx = float(intrinsics.fx) * dw / iw
        fy = float(intrinsics.fy) * dh / ih
        px = float(getattr(intrinsics, "px", iw / 2.0)) * dw / iw
        py = float(getattr(intrinsics, "py", ih / 2.0)) * dh / ih

        v, u = np.mgrid[0:dh, 0:dw]
        pts = np.stack([(u + 0.5 - px) / fx * depth, (v + 0.5 - py) / fy * depth, depth], axis=-1)
        sel = (v > dh * 0.55) & np.isfinite(depth) & (depth > 0)
        cloud = pts[sel]
        if len(cloud) < 500:
            return None
        rng = np.random.default_rng(self.config.seed)
        tol = 0.03 * float(np.median(cloud[:, 2]))
        best = None
        for _ in range(400):
            a = cloud[rng.choice(len(cloud), 3, replace=False)]
            n = np.cross(a[1] - a[0], a[2] - a[0])
            norm = np.linalg.norm(n)
            if norm < 1e-12:
                continue
            n = n / norm
            n = n if n[1] > 0 else -n            # image y points down, toward the ground
            if n[1] < np.cos(np.radians(20.0)):
                continue
            inliers = np.abs((cloud - a[0]) @ n) < tol
            if best is None or inliers.sum() > best.sum():
                best = inliers
        if best is None or best.mean() < 0.4:
            self.log_info("Camera height: no clean ground plane in the photo, keeping the configured height")
            return None
        q = cloud[best]
        centre = q.mean(axis=0)
        normal = np.linalg.svd(q - centre)[2][2]
        normal = normal if normal[1] > 0 else -normal
        height_da3 = float(abs(centre @ normal))

        cone = captured_cone_from_context(context, pano_shape[1], pano_shape[0])
        if cone is None:
            return None
        sx, sy = dw / float(cone.photo_w), dh / float(cone.photo_h)
        ratios, weights, used = [], [], []
        # The labeler's photo detections when it published them, else the blobs.
        candidates = context.input_object("photo_detections")
        if candidates is None:
            count = int(context.input_object(ContextKey.OBJECT_COUNT) or 0)
            candidates = [context.input_object(f"metadata_{i}") or {} for i in range(count)]
        for idx, m in enumerate(candidates):
            cls = normalize_category(m.get("vetoed_class") or m.get("class"))
            box = m.get("photo_box")
            if cls not in _HEIGHT_ANCHORS or not box or m.get("synthetic"):
                continue
            prior, min_aspect = _HEIGHT_ANCHORS[cls]
            x, y, w, h = (float(b) for b in box)
            if w <= 0 or h / w < min_aspect or h * sy < 4:
                continue
            patch = depth[int(y * sy):int((y + h) * sy) + 1, int(x * sx):int((x + w) * sx) + 1]
            patch = patch[np.isfinite(patch) & (patch > 0)]
            if patch.size == 0:
                continue
            measured = h * sy * float(np.median(patch)) / fy
            if measured <= 1e-4:
                continue
            ratios.append(prior / measured)
            weights.append(float(m.get("confidence") or m.get("score") or 0.5))
            used.append(f"{cls} {int(w)}x{int(h)} px x{prior / measured:.1f}")
        if not ratios:
            self.log_info(
                f"Camera height: ground plane {height_da3:.2f} (DA3 units) but nothing of known size "
                f"in the photo to scale it, keeping the configured {cfg.camera_height_meters:.2f} m"
            )
            return None
        ratios_arr = np.asarray(ratios)
        spread = float(ratios_arr.max() / ratios_arr.min())
        if spread > cfg.max_anchor_spread:
            self.log_info(
                f"Camera height: size anchors disagree {spread:.1f}x ({', '.join(used)}), "
                f"keeping the configured {cfg.camera_height_meters:.2f} m"
            )
            return None
        scale = float(np.exp(np.average(np.log(ratios_arr), weights=np.asarray(weights))))
        meters = height_da3 * scale
        if not np.isfinite(meters) or meters < 0.5 or meters > cfg.max_camera_height_m:
            self.log_info(f"Camera height: measured {meters:.1f} m is out of range, keeping the configured height")
            return None
        self.log_info(
            f"Camera height measured from the photo: {meters:.2f} m (ground plane {height_da3:.2f} DA3 units "
            f"x scale {scale:.2f} from {len(ratios)} anchor(s): {', '.join(used)})"
        )
        return {"meters": round(meters, 3), "source": "photo", "anchors": len(ratios),
                "plane_da3": round(height_da3, 4), "scale": round(scale, 4)}

    def has_expected_output(self, context: PipelineContext) -> bool:
        # context.depth() would also match ContextKey.PANORAMA_DEPTH written by the earlier
        # "Panorama Depth" stage (it walks all prior stages), which would make this always
        # look "already done" even before it ever ran. has_stage_output() is scoped to this
        # stage's own writes only.
        _, _, panorama_depth_key = self._resolved_keys()
        return context.has_stage_output(panorama_depth_key)

    def model_names(self) -> list[str]:
        return []
