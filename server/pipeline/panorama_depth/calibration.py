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


def calibrate_panorama_depth(
    da3_depth: np.ndarray,
    da3_placement_fov_deg: float,
    dap_pred_raw: np.ndarray,
    sky_mask: Optional[np.ndarray],
    num_bins: int = 14,
    min_samples: int = 300,
) -> Optional[np.ndarray]:
    """
    Fits a monotonic lookup curve from DAP's raw (0-1) prediction to real metric depth, using
    the region where the hero photo (metric depth `da3_depth`, from Depth Anything 3) was
    placed into the panorama, then applies that curve to the entire panorama depth map.

    da3_placement_fov_deg must be the horizontal FOV actually used to place the photo into the
    panorama (server/pipeline/panorama/panorama.py sends 2x the camera's own estimated FOV to
    WorldGen) — the correspondence has to match the real placement, not the true camera FOV.

    dap_pred_raw is DAP's native 0-1 output (i.e. panorama depth with the *100 metres
    assumption undone) — same shape as the target panorama.

    Returns None if too few valid overlap samples survive filtering, signalling the caller
    should leave the panorama depth uncalibrated.
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

    corrected = np.interp(dap_pred_raw.astype(np.float64).ravel(), centers_sorted, medians_monotonic)
    return corrected.reshape(dap_pred_raw.shape).astype(np.float32)


class PanoramaDepthCalibrationStage(PipelineStage):
    """
    Recalibrates ContextKey.PANORAMA_DEPTH (DAP's equirectangular depth, which converts its
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

    If too little of the hero photo's real (non-hallucinated) footprint survives filtering
    (e.g. a very narrow FOV, or a mostly-sky photo), calibration is skipped and
    ContextKey.PANORAMA_DEPTH is left as DAP produced it.

    Input keys  (SemanticKey.DEPTH, SemanticKey.INTRINSICS) → ContextKey.DEPTH, ContextKey.INTRINSICS
    Output key  (SemanticKey.OUTPUT)                        → ContextKey.PANORAMA_DEPTH (overwritten in place)
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
        sky_mask = context.input_object(ContextKey.PANORAMA_SKY_MASK)
        if isinstance(sky_mask, list):
            sky_mask = np.array(sky_mask, dtype=bool)

        if da3_depth is None or intrinsics is None or panorama_depth is None:
            self.log_warning("Missing metric depth, intrinsics, or panorama depth — skipping calibration")
            self.finish_progress(task)
            return context

        dap_pred_raw = panorama_depth.depth.astype(np.float64) / 100.0
        # server/pipeline/panorama/panorama.py sends `intrinsics.fov * 2.0` to WorldGen when
        # placing the hero photo into the panorama — reproduce that exact value here so the
        # correspondence matches the real placement, whatever the doubling's own justification.
        corrected = calibrate_panorama_depth(
            da3_depth.depth,
            intrinsics.fov * 2.0,
            dap_pred_raw,
            sky_mask,
            num_bins=cfg.num_bins,
            min_samples=cfg.min_samples,
        )

        if corrected is None:
            self.log_warning(
                "Not enough hero-photo/panorama overlap to calibrate — leaving panorama depth unchanged"
            )
            self.finish_progress(task)
            return context

        before_min, before_max = panorama_depth.min(), panorama_depth.max()
        calibrated = Depth(corrected)
        if self.temp is not None:
            calibrated.save_debug_image(self.temp / "pano_depth_calibrated.png")

        self.log_info(
            f"Calibrated panorama depth {before_min:.1f} → {before_max:.1f} m (raw) "
            f"into {calibrated.min():.1f} → {calibrated.max():.1f} m (calibrated)"
        )
        context.add_depth(panorama_depth_key, calibrated)

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
