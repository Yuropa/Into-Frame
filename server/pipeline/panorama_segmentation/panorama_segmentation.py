import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage
from scipy.ndimage import uniform_filter1d

from remote_connection.remote_client import RemoteClient
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.panorama_segmentation.panorama_region_result import (
    PanoramaRegionResult,
    PanoramaRegion,
    RegionType,
    coarse_type_for_label,
    build_type_idx_map,
    colorize_region_type_map,
)
from util.image_utils import Image

# Minimum fraction of panorama pixels a connected component must cover to be
# reported as an individual region entry.
_MIN_REGION_FRACTION = 0.005

_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-640-640"


class PanoramaSegmentationClient(RemoteClient):
    def __init__(self, device) -> None:
        script_path = Path(__file__).parent / "panorama_segmentation_imp.py"
        super().__init__(device=device, conda_env="frame", script_path=script_path)

    @classmethod
    def model_names(cls) -> list[str]:
        return [_MODEL_ID]

    def segment(self, panorama: PILImage.Image, temp_path: Path) -> dict:
        return self.send(action="segment", input=panorama, temp_path=temp_path)


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy.ndimage import label as ndimage_label
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    labels, n = ndimage_label(mask, structure=structure)
    return labels, n


def _build_result(raw: dict) -> tuple[PanoramaRegionResult, np.ndarray]:
    label_map = np.array(raw["label_map"], dtype=np.int16)
    id2label: dict[int, str] = {int(k): v for k, v in raw["id2label"].items()}
    h, w = label_map.shape
    total_pixels = h * w

    type_idx_map = build_type_idx_map(label_map, id2label)

    result = PanoramaRegionResult()
    type_pixel_counts: dict[RegionType, int] = {rt: 0 for rt in RegionType}

    # Group ADE20K class IDs by coarse region type.
    ids_for_type: dict[RegionType, list[int]] = {rt: [] for rt in RegionType}
    for class_id, label_name in id2label.items():
        region_type = coarse_type_for_label(label_name)
        ids_for_type[region_type].append(class_id)

    for region_type, class_ids in ids_for_type.items():
        if not class_ids:
            continue

        # Build a boolean mask for all pixels of this coarse type.
        type_mask = np.zeros((h, w), dtype=bool)
        for cid in class_ids:
            type_mask |= (label_map == cid)

        type_pixel_counts[region_type] = int(type_mask.sum())
        if type_pixel_counts[region_type] == 0:
            continue

        # Find connected components and emit one PanoramaRegion per component
        # above the minimum area threshold.
        comp_labels, n_comps = _connected_components(type_mask)
        dominant_label_name = _dominant_label_name(
            label_map, type_mask, class_ids, id2label
        )

        for comp_id in range(1, n_comps + 1):
            comp_mask = comp_labels == comp_id
            comp_pixels = int(comp_mask.sum())
            area_fraction = comp_pixels / total_pixels
            if area_fraction < _MIN_REGION_FRACTION:
                continue

            ys, xs = np.where(comp_mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            cx = float(xs.mean())
            cy = float(ys.mean())

            result.regions.append(
                PanoramaRegion(
                    region_type=region_type.label,
                    label_name=dominant_label_name,
                    area_fraction=round(area_fraction, 4),
                    bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                    centroid=(round(cx, 1), round(cy, 1)),
                )
            )

    result.regions.sort(key=lambda r: r.area_fraction, reverse=True)

    result.type_fractions = {
        rt.label: round(count / total_pixels, 4)
        for rt, count in type_pixel_counts.items()
        if count > 0
    }
    result.present_types = [
        rt.label for rt in RegionType if type_pixel_counts.get(rt, 0) > 0
    ]
    return result, type_idx_map


def _clean_nadir_band(
    type_idx_map: np.ndarray,
    nadir_cutoff_deg: float,
    nadir_band_deg: float,
) -> np.ndarray:
    """
    Replace the panorama's near-nadir rows (looking almost straight down at the
    tripod/rig) with an estimate borrowed from the reliable ring just above them,
    rather than trusting SegFormer there directly. That band is heavily warped
    by the equirectangular projection and shows content (the rig, the operator's
    feet, extreme close-range blur) a model trained on rectilinear photos wasn't
    trained on, so its labels there are close to noise (e.g. flagging distorted
    ground as WATER).

    Equirectangular rows map linearly to elevation angle (row 0 = +90°/zenith,
    row h-1 = -90°/nadir), independent of scene geometry, so the cutoff can be
    computed directly from image height. Everything at/below nadir_cutoff_deg is
    replaced by a per-column majority vote taken over the nadir_band_deg of rows
    immediately above the cutoff, then smoothed circularly along the row (it's a
    full 360° ring — the left and right edges are adjacent, not a hard border)
    to remove single-column segmentation noise before extending it straight down
    through the excluded band.
    """
    h, w = type_idx_map.shape
    n_types = len(RegionType)

    # Row -> elevation angle (deg) is linear and independent of world geometry.
    cutoff_row = int(np.clip(np.ceil((90.0 - nadir_cutoff_deg) / 180.0 * h - 0.5), 0, h))
    band_row = int(np.clip(np.floor((90.0 - (nadir_cutoff_deg + nadir_band_deg)) / 180.0 * h - 0.5), 0, cutoff_row))
    if cutoff_row >= h or cutoff_row <= band_row:
        return type_idx_map

    band = type_idx_map[band_row:cutoff_row]
    one_hot = band[:, None, :] == np.arange(n_types, dtype=type_idx_map.dtype)[None, :, None]
    ring = one_hot.sum(axis=0).argmax(axis=0).astype(type_idx_map.dtype)  # (w,) majority vote per column

    kernel_px = max(1, int(round(w * 0.01)))  # ~3.6° of azimuth
    scores = np.stack([
        uniform_filter1d((ring == t).astype(np.float32), size=2 * kernel_px + 1, mode="wrap")
        for t in range(n_types)
    ])
    ring = scores.argmax(axis=0).astype(type_idx_map.dtype)

    cleaned = type_idx_map.copy()
    cleaned[cutoff_row:] = ring[np.newaxis, :]
    return cleaned


def _dominant_label_name(
    label_map: np.ndarray,
    type_mask: np.ndarray,
    class_ids: list[int],
    id2label: dict[int, str],
) -> str:
    """Return the ADE20K label name with the most pixels within type_mask."""
    best_id = class_ids[0]
    best_count = 0
    for cid in class_ids:
        count = int(((label_map == cid) & type_mask).sum())
        if count > best_count:
            best_count = count
            best_id = cid
    return id2label.get(best_id, "unknown")


class PanoramaRegionConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device,
        torch_dtype,
        log,
        keys=None,
        seed: int = 0,
        nadir_cutoff_deg: float = -55.0,
        nadir_band_deg: float = 15.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.nadir_cutoff_deg = nadir_cutoff_deg
        self.nadir_band_deg = nadir_band_deg


class PanoramaRegionStage(PipelineStage):
    """
    Identifies coarse semantic regions in the equirectangular panorama.

    Runs SegFormer-B5 (ADE20K, 150 classes) on the panorama, maps the 150 class
    labels to six coarse types — sky, water, terrain, ground, vegetation, built —
    and finds connected components within each type.  Each component above a
    minimum area threshold is recorded as a PanoramaRegion with its type, bounding
    box, centroid, and area fraction.

    Rows at/below nadir_cutoff_deg elevation (the tripod/rig, heavily warped by
    the equirectangular projection) are excluded from SegFormer's raw output and
    replaced via _clean_nadir_band — see that function's docstring.

    Reads:  ContextKey.PANORAMA
    Writes: ContextKey.PANORAMA_REGIONS (PanoramaRegionResult)
    Debug:  self.output/regions.json      (reflects SegFormer's raw output —
                                           not nadir-cleaned)
            self.output/label_overlay.png (nadir-cleaned)
    """

    @classmethod
    def config_class(cls) -> type[PanoramaRegionConfiguration]:
        return PanoramaRegionConfiguration

    def __init__(self, config: PanoramaRegionConfiguration) -> None:
        super().__init__(config)
        self._client: PanoramaSegmentationClient | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        panorama = context.input_panorama(ContextKey.PANORAMA)
        if panorama is None:
            self.log_info("No panorama in context, skipping")
            return context

        cfg: PanoramaRegionConfiguration = self.config
        task = self.create_progress(3, "Segmenting panorama regions…")

        if self._client is None:
            self._client = PanoramaSegmentationClient(self.device)
        self.advance_progress(task)

        raw = self._client.segment(panorama.rgb(), self.temp)
        self.advance_progress(task)

        result, type_idx_map = _build_result(raw)
        type_idx_map = _clean_nadir_band(type_idx_map, cfg.nadir_cutoff_deg, cfg.nadir_band_deg)
        context.add_panorama_regions(ContextKey.PANORAMA_REGIONS, result)
        context.add_depth(ContextKey.PANORAMA_REGION_TYPE_MAP, type_idx_map.astype(np.float32))

        for region_type in result.present_types:
            frac = result.type_fractions.get(region_type, 0.0)
            self.log_info(f"  {region_type}: {frac * 100:.1f}% of panorama")

        self.advance_progress(task)
        self.finish_progress(task)
        self._write_debug(result, type_idx_map, panorama.rgb())
        return context

    def _write_debug(
        self, result: PanoramaRegionResult, type_idx_map: np.ndarray, panorama_rgb: PILImage.Image, overlay_alpha: float = 0.5
    ):
        if self.output is None:
            return

        with open(self.output / "regions.json", "w") as f:
            json.dump(result.encode(), f, indent=2)

        region_rgb = colorize_region_type_map(type_idx_map)
        PILImage.fromarray(region_rgb).save(self.output / "label_overlay.png")

        # Blend the colorized region map over the source panorama so regions can
        # be checked against the actual photo content, not just the flat map.
        pano_arr = np.array(panorama_rgb.convert("RGB").resize(
            (type_idx_map.shape[1], type_idx_map.shape[0]), PILImage.LANCZOS
        )).astype(np.float32)
        blended = (pano_arr * (1.0 - overlay_alpha) + region_rgb.astype(np.float32) * overlay_alpha).clip(0, 255)
        PILImage.fromarray(blended.astype(np.uint8)).save(self.output / "label_overlay_on_panorama.png")

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.panorama_regions(ContextKey.PANORAMA_REGIONS) is not None

    def model_names(self) -> list[str]:
        return PanoramaSegmentationClient.model_names()

    def clean_up(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        super().clean_up()
