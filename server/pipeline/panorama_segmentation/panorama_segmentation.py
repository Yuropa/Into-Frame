import json
from pathlib import Path
from typing import NamedTuple

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
    is_ambiguous_label,
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


class _SegmentationResult(NamedTuple):
    result: PanoramaRegionResult
    type_idx_map: np.ndarray            # raw per-pixel coarse type, straight from argmax
    resolved_type_idx_map: np.ndarray   # type_idx_map with uncorroborated-ambiguous pixels overwritten by their runner-up type
    confidence_map: np.ndarray          # per-pixel top-1 softmax confidence
    runnerup_type_idx_map: np.ndarray   # per-pixel coarse type of the model's 2nd-best class
    ambiguous_mask: np.ndarray          # bool; True where resolved_type_idx_map differs from type_idx_map


def _build_result(raw: dict) -> _SegmentationResult:
    label_map = np.array(raw["label_map"], dtype=np.int16)
    confidence_map = np.asarray(raw["confidence_map"], dtype=np.float32)
    runnerup_label_map = np.asarray(raw["runnerup_label_map"], dtype=np.int16)
    id2label: dict[int, str] = {int(k): v for k, v in raw["id2label"].items()}
    h, w = label_map.shape
    total_pixels = h * w

    type_idx_map = build_type_idx_map(label_map, id2label)
    runnerup_type_idx_map = build_type_idx_map(runnerup_label_map, id2label)
    resolved_type_idx_map = type_idx_map.copy()
    ambiguous_mask = np.zeros((h, w), dtype=bool)

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

        # Find connected components above the minimum area threshold. First
        # pass gathers each component's own stats -- including its own
        # dominant label (previously computed once from the whole
        # type_mask and shared across every component of this type, which
        # would silently blend a real, unambiguous component's identity
        # into a nearby ambiguous one's; the corroboration check below needs
        # each component's true, individual label to mean anything).
        comp_labels, n_comps = _connected_components(type_mask)
        components: list[dict] = []
        for comp_id in range(1, n_comps + 1):
            comp_mask = comp_labels == comp_id
            comp_pixels = int(comp_mask.sum())
            area_fraction = comp_pixels / total_pixels
            if area_fraction < _MIN_REGION_FRACTION:
                continue

            dominant_label_name = _dominant_label_name(label_map, comp_mask, class_ids, id2label)
            ys, xs = np.where(comp_mask)
            components.append({
                "mask": comp_mask,
                "label_name": dominant_label_name,
                "ambiguous": is_ambiguous_label(dominant_label_name),
                "area_fraction": area_fraction,
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1),
                "centroid": (float(xs.mean()), float(ys.mean())),
                "mean_confidence": float(confidence_map[comp_mask].mean()),
            })

        # Second pass: an ambiguous component is well_supported only if a
        # *different* component of this same coarse type has an unambiguous
        # label -- corroboration has to come from an independent,
        # less-confusable observation. Two separate components that both
        # happen to be the same confusable call (e.g. two "wall"-labeled
        # cliff faces) don't corroborate each other. Every component
        # remaining here already cleared _MIN_REGION_FRACTION, so this needs
        # no extra size check.
        unambiguous_present = any(not c["ambiguous"] for c in components)
        for c in components:
            well_supported = (not c["ambiguous"]) or unambiguous_present
            if c["ambiguous"] and not well_supported:
                ambiguous_mask[c["mask"]] = True
                resolved_type_idx_map[c["mask"]] = runnerup_type_idx_map[c["mask"]]

            result.regions.append(
                PanoramaRegion(
                    region_type=region_type.label,
                    label_name=c["label_name"],
                    area_fraction=round(c["area_fraction"], 4),
                    bbox=c["bbox"],
                    centroid=(round(c["centroid"][0], 1), round(c["centroid"][1], 1)),
                    mean_confidence=round(c["mean_confidence"], 4),
                    ambiguous_label=c["ambiguous"],
                    well_supported=well_supported,
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
    return _SegmentationResult(
        result=result,
        type_idx_map=type_idx_map,
        resolved_type_idx_map=resolved_type_idx_map,
        confidence_map=confidence_map,
        runnerup_type_idx_map=runnerup_type_idx_map,
        ambiguous_mask=ambiguous_mask,
    )


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

    Per-pixel top-1 confidence and the model's runner-up class are also kept
    (see panorama_segmentation_imp.py's _segment). A region whose dominant
    label matches an "ambiguous" _LABEL_RULES entry (a keyword group
    routinely confused with a different plausible coarse type in natural
    scenes -- e.g. ADE20K's "wall" class firing on a sunlit cliff face) is
    only trusted if a *different*, unambiguous region of the same coarse
    type is independently present elsewhere in the panorama (see
    _build_result, PanoramaRegion.well_supported); otherwise every pixel in
    it is resolved to its own runner-up coarse type instead. This corrected
    map -- not the raw argmax output -- is what's written to
    PANORAMA_REGION_TYPE_MAP, so every existing consumer of that key
    (HeightMapGenerator, RegionMapGenerator, skybox inpainting, terrain
    texture generation) benefits automatically without any change on their
    end.

    Reads:  ContextKey.PANORAMA
    Writes: ContextKey.PANORAMA_REGIONS               (PanoramaRegionResult)
            ContextKey.PANORAMA_REGION_TYPE_MAP        (resolved, nadir-cleaned)
            ContextKey.PANORAMA_REGION_TYPE_MAP_RAW    (raw argmax output, not nadir-cleaned)
            ContextKey.PANORAMA_REGION_CONFIDENCE_MAP  (per-pixel top-1 softmax confidence)
            ContextKey.PANORAMA_REGION_RUNNERUP_TYPE_MAP (per-pixel coarse type of the 2nd-best class)
            ContextKey.PANORAMA_REGION_AMBIGUOUS_MASK  (bool; True where the raw and resolved maps differ)
    Debug:  self.output/regions.json               (reflects SegFormer's raw output —
                                                     not nadir-cleaned)
            self.output/label_overlay.png           (raw, not nadir-cleaned or resolved)
            self.output/label_overlay_resolved.png  (nadir-cleaned + ambiguity-resolved -- matches PANORAMA_REGION_TYPE_MAP)
            self.output/ambiguous_mask.png          (pixels resolved from an uncorroborated ambiguous label)
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

        seg = _build_result(raw)
        result = seg.result
        # Nadir cleanup only needs to run on the map that actually becomes the
        # canonical output; the raw map is kept purely for debugging (see
        # PANORAMA_REGION_TYPE_MAP_RAW / regions.json, both intentionally
        # left un-nadir-cleaned, matching this stage's existing convention).
        resolved_type_idx_map = _clean_nadir_band(
            seg.resolved_type_idx_map, cfg.nadir_cutoff_deg, cfg.nadir_band_deg
        )
        context.add_panorama_regions(ContextKey.PANORAMA_REGIONS, result)
        context.add_depth(ContextKey.PANORAMA_REGION_TYPE_MAP, resolved_type_idx_map.astype(np.float32))
        context.add_depth(ContextKey.PANORAMA_REGION_TYPE_MAP_RAW, seg.type_idx_map.astype(np.float32))
        context.add_depth(ContextKey.PANORAMA_REGION_CONFIDENCE_MAP, seg.confidence_map)
        context.add_depth(ContextKey.PANORAMA_REGION_RUNNERUP_TYPE_MAP, seg.runnerup_type_idx_map.astype(np.float32))
        context.add_depth(ContextKey.PANORAMA_REGION_AMBIGUOUS_MASK, seg.ambiguous_mask.astype(np.float32))

        n_ambiguous = int(seg.ambiguous_mask.sum())
        if n_ambiguous > 0:
            frac = n_ambiguous / seg.ambiguous_mask.size
            self.log_info(
                f"  Resolved {frac * 100:.2f}% of panorama from an uncorroborated "
                f"ambiguous label to its runner-up type"
            )

        for region_type in result.present_types:
            frac = result.type_fractions.get(region_type, 0.0)
            self.log_info(f"  {region_type}: {frac * 100:.1f}% of panorama")

        self.advance_progress(task)
        self.finish_progress(task)
        self._write_debug(result, seg.type_idx_map, resolved_type_idx_map, seg.ambiguous_mask, panorama.rgb())
        return context

    def _write_debug(
        self,
        result: PanoramaRegionResult,
        type_idx_map: np.ndarray,
        resolved_type_idx_map: np.ndarray,
        ambiguous_mask: np.ndarray,
        panorama_rgb: PILImage.Image,
        overlay_alpha: float = 0.5,
    ):
        if self.output is None:
            return

        with open(self.output / "regions.json", "w") as f:
            json.dump(result.encode(), f, indent=2)

        def save_overlay(type_map: np.ndarray, name: str) -> None:
            region_rgb = colorize_region_type_map(type_map)
            PILImage.fromarray(region_rgb).save(self.output / f"{name}.png")

            # Blend the colorized region map over the source panorama so regions
            # can be checked against the actual photo content, not just the flat map.
            pano_arr = np.array(panorama_rgb.convert("RGB").resize(
                (type_map.shape[1], type_map.shape[0]), PILImage.LANCZOS
            )).astype(np.float32)
            blended = (pano_arr * (1.0 - overlay_alpha) + region_rgb.astype(np.float32) * overlay_alpha).clip(0, 255)
            PILImage.fromarray(blended.astype(np.uint8)).save(self.output / f"{name}_on_panorama.png")

        save_overlay(type_idx_map, "label_overlay")
        save_overlay(resolved_type_idx_map, "label_overlay_resolved")

        if ambiguous_mask.any():
            PILImage.fromarray((ambiguous_mask * 255).astype(np.uint8), "L").save(
                self.output / "ambiguous_mask.png"
            )

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.panorama_regions(ContextKey.PANORAMA_REGIONS) is not None

    def model_names(self) -> list[str]:
        return PanoramaSegmentationClient.model_names()

    def clean_up(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        super().clean_up()
