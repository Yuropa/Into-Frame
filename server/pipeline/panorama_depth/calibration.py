from typing import Any, Optional
from logging import Logger

import numpy as np
import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.depth_utils import Depth


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
        max_depth_basis_percentile: float = 95.0,
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
        # can never land inside that grid anyway). Enforced by uniformly *rescaling*
        # the whole depth map down (see _rescale_to_fit), not by clamping outliers to
        # this value: a flat clamp would collapse every distance beyond it (a nearby
        # ridge at 120 m and a hazy peak at 900 m alike) onto one indistinguishable
        # wall at the cutoff. A global scale instead preserves every relative
        # distance/shape — near stays proportionally near, far stays proportionally
        # far — while guaranteeing the farthest point still fits. None disables (no
        # rescale, only the relative max_extrapolation_factor cap above applies).
        self.max_depth_m = max_depth_m
        # Percentile (over non-sky pixels) of calibrated depth used as the rescale
        # basis instead of the literal max — DAP's raw prediction saturates at the
        # same ceiling for both real distant terrain and sky, so the single
        # farthest pixel is usually one member of a wide shared plateau rather
        # than a genuine outlier; excluding sky alone doesn't fix this, since real
        # terrain independently reaches that ceiling too. A lower percentile
        # scales against the bulk of real signal instead of that plateau, at the
        # cost of hard-clipping the (small) remaining tail beyond it. See
        # _rescale_to_fit.
        self.max_depth_basis_percentile = max_depth_basis_percentile


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


def _rescale_to_fit(
    depth: np.ndarray,
    max_depth_m: float,
    sky_mask: Optional[np.ndarray] = None,
    basis_percentile: float = 95.0,
) -> tuple[np.ndarray, float]:
    """
    Uniformly scales `depth` down so its basis_percentile-th percentile (over
    non-sky pixels, when sky_mask is given) is at most max_depth_m, preserving
    every relative distance and shape (a ridge that was twice as far as another
    stays twice as far) instead of flat-clamping every far outlier individually,
    which would collapse distinct far distances onto one indistinguishable wall.

    A high percentile is used instead of the literal max because DAP's raw
    prediction saturates at the same ceiling for both real distant terrain and
    sky (see _apply_panorama_depth_curve's docstring) -- the farthest pixel is
    often just one member of a wide plateau sharing that ceiling, not a genuine
    single outlier. Excluding sky_mask alone doesn't fix this: real terrain
    independently saturates at the identical value (observed on one test
    panorama: 41% of pixels sat at the exact ceiling, most but not all of it
    sky), so a literal non-sky max still lets that plateau dictate an
    unnecessarily aggressive scale. Scaling against a robust percentile of the
    bulk of real signal instead, then hard-clipping the rare remaining tail to
    max_depth_m, keeps that plateau from single-handedly compressing the entire
    near field.

    Only ever shrinks: if the basis value is already within budget (scale would
    be >= 1), the depth map is returned unchanged. Returns (rescaled_depth,
    scale) so callers can apply the same scale factor to a second, related depth
    map (see PanoramaDepthCalibrationStage -- PANORAMA_OBJECT_DEPTH must shrink
    by the same factor as PANORAMA_DEPTH or the two would place the same
    real-world point at two different distances).
    """
    basis_pool = depth
    if sky_mask is not None and sky_mask.shape == depth.shape:
        non_sky = depth[~sky_mask]
        if non_sky.size > 0:
            basis_pool = non_sky
    basis_value = float(np.percentile(basis_pool, basis_percentile))
    if basis_value <= max_depth_m or basis_value <= 0:
        return depth, 1.0
    scale = max_depth_m / basis_value
    rescaled = np.minimum(depth * scale, max_depth_m).astype(np.float32)
    return rescaled, scale


def calibrate_panorama_depth(
    da3_depth: np.ndarray,
    da3_placement_fov_deg: float,
    dap_pred_raw: np.ndarray,
    sky_mask: Optional[np.ndarray],
    num_bins: int = 14,
    min_samples: int = 300,
    max_extrapolation_factor: float = 20.0,
    max_depth_m: Optional[float] = None,
    max_depth_basis_percentile: float = 95.0,
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
        corrected, _ = _rescale_to_fit(corrected, max_depth_m, sky_mask, max_depth_basis_percentile)
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
        sky_mask = context.input_object(ContextKey.PANORAMA_SKY_MASK)
        if isinstance(sky_mask, list):
            sky_mask = np.array(sky_mask, dtype=bool)

        if da3_depth is None or intrinsics is None or panorama_depth is None:
            self.log_warning("Missing metric depth, intrinsics, or panorama depth — skipping calibration")
            self.finish_progress(task)
            return context

        # Fit against the original (pre-foreground-inpainting) panorama depth when available —
        # see class docstring for why. Falls back to PANORAMA_DEPTH itself if it's missing.
        fit_source = panorama_object_depth if panorama_object_depth is not None else panorama_depth
        dap_pred_raw_fit = fit_source.depth.astype(np.float64) / 100.0

        # server/pipeline/panorama/panorama.py sends `intrinsics.fov * 2.0` to WorldGen when
        # placing the hero photo into the panorama — reproduce that exact value here so the
        # correspondence matches the real placement, whatever the doubling's own justification.
        curve = _fit_panorama_depth_curve(
            da3_depth.depth,
            intrinsics.fov * 2.0,
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
        # Single global scale, derived from PANORAMA_DEPTH (the map with the explicit
        # grid-footprint constraint) and reused below for PANORAMA_OBJECT_DEPTH — both
        # must shrink by the same factor or the same real-world point ends up at two
        # different distances in the two maps. See _rescale_to_fit.
        scale = 1.0
        if cfg.max_depth_m is not None:
            calibrated_arr, scale = _rescale_to_fit(
                calibrated_arr, cfg.max_depth_m, sky_mask, cfg.max_depth_basis_percentile
            )
        calibrated = Depth(calibrated_arr)
        if self.temp is not None:
            calibrated.save_debug_image(self.temp / "pano_depth_calibrated.png")

        self.log_info(
            f"Calibrated panorama depth {before_min:.1f} → {before_max:.1f} m (raw) "
            f"into {calibrated.min():.1f} → {calibrated.max():.1f} m (calibrated"
            + (f", scaled ×{scale:.3f} to fit {cfg.max_depth_m:.0f} m)" if scale != 1.0 else ")")
        )
        context.add_depth(panorama_depth_key, calibrated)

        if panorama_object_depth is not None:
            obj_before_min, obj_before_max = panorama_object_depth.min(), panorama_object_depth.max()
            calibrated_object_arr = _apply_panorama_depth_curve(
                dap_pred_raw_fit, *curve, max_extrapolation_factor=cfg.max_extrapolation_factor
            )
            if scale != 1.0:
                # Same global scale as PANORAMA_DEPTH (see comment above) -- but this
                # map's own saturation tail can differ (separate DAP pass), so it still
                # needs its own hard clip rather than assuming the shared scale alone
                # keeps it within budget.
                calibrated_object_arr = np.minimum(calibrated_object_arr * scale, cfg.max_depth_m)
            calibrated_object = Depth(calibrated_object_arr)
            if self.temp is not None:
                calibrated_object.save_debug_image(self.temp / "pano_object_depth_calibrated.png")
            self.log_info(
                f"Calibrated panorama object depth {obj_before_min:.1f} → {obj_before_max:.1f} m (raw) "
                f"into {calibrated_object.min():.1f} → {calibrated_object.max():.1f} m (calibrated"
                + (f", scaled ×{scale:.3f})" if scale != 1.0 else ")")
            )
            context.add_depth(ContextKey.PANORAMA_OBJECT_DEPTH, calibrated_object)

        self.advance_progress(task)
        self.finish_progress(task)
        return context

    def has_expected_output(self, context: PipelineContext) -> bool:
        # context.depth() would also match ContextKey.PANORAMA_DEPTH written by the earlier
        # "Panorama Depth" stage (it walks all prior stages), which would make this always
        # look "already done" even before it ever ran. has_stage_output() is scoped to this
        # stage's own writes only.
        _, _, panorama_depth_key = self._resolved_keys()
        return context.has_stage_output(panorama_depth_key)

    def model_names(self) -> list[str]:
        return []
