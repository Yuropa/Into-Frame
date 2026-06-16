import json
from pathlib import Path

import numpy as np
from PIL import Image as PILImage

from remote_connection.remote_client import RemoteClient
from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage, SemanticKey
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.panorama_segmentation.panorama_region_result import (
    PanoramaRegionResult,
    PanoramaRegion,
    ALL_REGION_TYPES,
    coarse_type_for_label,
)
from util.image_utils import Image

# Minimum fraction of panorama pixels a connected component must cover to be
# reported as an individual region entry.
_MIN_REGION_FRACTION = 0.005

_MODEL_ID = "nvidia/segformer-b5-finetuned-ade-512-512"


class PanoramaSegmentationClient(RemoteClient):
    def __init__(self, device) -> None:
        script_path = Path(__file__).parent / "panorama_segmentation_imp.py"
        super().__init__(device=device, conda_env="frame", script_path=script_path)

    @classmethod
    def model_names(cls) -> list[str]:
        return [_MODEL_ID]

    def segment(self, panorama: PILImage.Image) -> dict:
        return self.send(action="segment", input=panorama)


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    from scipy.ndimage import label as ndimage_label
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    labels, n = ndimage_label(mask, structure=structure)
    return labels, n


def _build_result(raw: dict) -> PanoramaRegionResult:
    label_map = np.array(raw["label_map"], dtype=np.int16)
    id2label: dict[int, str] = {int(k): v for k, v in raw["id2label"].items()}
    h, w = label_map.shape
    total_pixels = h * w

    result = PanoramaRegionResult()
    type_pixel_counts: dict[str, int] = {t: 0 for t in ALL_REGION_TYPES}

    # Group ADE20K class IDs by coarse region type.
    ids_for_type: dict[str, list[int]] = {t: [] for t in ALL_REGION_TYPES}
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

        for comp_id in range(n_comps):
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
                    region_type=region_type,
                    label_name=dominant_label_name,
                    area_fraction=round(area_fraction, 4),
                    bbox=(x0, y0, x1 - x0 + 1, y1 - y0 + 1),
                    centroid=(round(cx, 1), round(cy, 1)),
                )
            )

    result.regions.sort(key=lambda r: r.area_fraction, reverse=True)

    result.type_fractions = {
        t: round(count / total_pixels, 4)
        for t, count in type_pixel_counts.items()
        if count > 0
    }
    result.present_types = [
        t for t in ALL_REGION_TYPES if type_pixel_counts.get(t, 0) > 0
    ]
    return result


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


class PanoramaRegionStage(PipelineStage):
    """
    Identifies coarse semantic regions in the equirectangular panorama.

    Runs SegFormer-B5 (ADE20K, 150 classes) on the panorama, maps the 150 class
    labels to six coarse types — sky, water, terrain, ground, vegetation, built —
    and finds connected components within each type.  Each component above a
    minimum area threshold is recorded as a PanoramaRegion with its type, bounding
    box, centroid, and area fraction.

    Reads:  ContextKey.PANORAMA
    Writes: ContextKey.PANORAMA_REGIONS (PanoramaRegionResult)
    Debug:  self.output/regions.json
            self.output/label_overlay.png
    """

    def __init__(self, config: PipelineStageConfiguration) -> None:
        super().__init__(config)
        self._client: PanoramaSegmentationClient | None = None

    def run(self, context: PipelineContext) -> PipelineContext:
        panorama = context.input_panorama(ContextKey.PANORAMA)
        if panorama is None:
            self.log_info("No panorama in context, skipping")
            return context

        task = self.create_progress(3, "Segmenting panorama regions…")

        if self._client is None:
            self._client = PanoramaSegmentationClient(self.device)
        self.advance_progress(task)

        raw = self._client.segment(panorama.rgb())
        self.advance_progress(task)

        result = _build_result(raw)
        context.add_panorama_regions(ContextKey.PANORAMA_REGIONS, result)

        for region_type in result.present_types:
            frac = result.type_fractions.get(region_type, 0.0)
            self.log_info(f"  {region_type}: {frac * 100:.1f}% of panorama")

        self.advance_progress(task)
        self.finish_progress(task)
        self._write_debug(result, raw)
        return context

    def _write_debug(self, result: PanoramaRegionResult, raw: dict):
        if self.output is None:
            return

        with open(self.output / "regions.json", "w") as f:
            json.dump(result.encode(), f, indent=2)

        self._write_label_overlay(raw)

    def _write_label_overlay(self, raw: dict):
        label_map = np.array(raw["label_map"], dtype=np.int16)
        h, w = label_map.shape

        _TYPE_COLORS = {
            "sky":         (135, 206, 235),
            "water":       (30,  144, 255),
            "terrain":     (139, 90,  43),
            "ground":      (160, 120, 60),
            "vegetation":  (34,  139, 34),
            "built":       (169, 169, 169),
            "other":       (200, 200, 200),
        }

        id2label = {int(k): v for k, v in raw["id2label"].items()}
        id_to_color = {}
        for class_id, label_name in id2label.items():
            region_type = coarse_type_for_label(label_name)
            id_to_color[class_id] = _TYPE_COLORS.get(region_type, (200, 200, 200))

        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id, color in id_to_color.items():
            mask = label_map == class_id
            overlay[mask] = color

        PILImage.fromarray(overlay).save(self.output / "label_overlay.png")

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.panorama_regions(ContextKey.PANORAMA_REGIONS) is not None

    def model_names(self) -> list[str]:
        return PanoramaSegmentationClient.model_names()

    def clean_up(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        super().clean_up()
