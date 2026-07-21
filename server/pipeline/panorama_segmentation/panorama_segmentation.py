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
    ambiguity_strategy_for_label,
    AMBIGUITY_CORROBORATION,
    AMBIGUITY_CONFIDENCE,
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


def _build_result(raw: dict, confidence_margin_threshold: float = 0.2) -> _SegmentationResult:
    label_map = np.array(raw["label_map"], dtype=np.int16)
    confidence_map = np.asarray(raw["confidence_map"], dtype=np.float32)
    runnerup_label_map = np.asarray(raw["runnerup_label_map"], dtype=np.int16)
    runnerup_confidence_map = np.asarray(raw["runnerup_confidence_map"], dtype=np.float32)
    id2label: dict[int, str] = {int(k): v for k, v in raw["id2label"].items()}
    h, w = label_map.shape
    total_pixels = h * w
    n_types = len(RegionType)

    # top1 >= top2 by construction (topk), so this is never negative.
    margin_map = confidence_map - runnerup_confidence_map

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
        # into a nearby ambiguous one's; the checks below need each
        # component's true, individual stats to mean anything).
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
            runnerup_counts = np.bincount(
                runnerup_type_idx_map[comp_mask].astype(np.int64), minlength=n_types
            )
            components.append({
                "mask": comp_mask,
                "label_name": dominant_label_name,
                "strategy": ambiguity_strategy_for_label(dominant_label_name),
                "area_fraction": area_fraction,
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()) - int(xs.min()) + 1, int(ys.max()) - int(ys.min()) + 1),
                "centroid": (float(xs.mean()), float(ys.mean())),
                "mean_confidence": float(confidence_map[comp_mask].mean()),
                "mean_margin": float(margin_map[comp_mask].mean()),
                # Majority vote of this component's own pixels' runner-up
                # type, not just the single dominant pixel's -- a component-
                # level summary for AMBIGUITY_CONFIDENCE's ground-valid check
                # below. Per-pixel resolution below still uses the full
                # per-pixel runnerup_type_idx_map, not this summary.
                "dominant_runnerup_type": RegionType(int(np.argmax(runnerup_counts))),
            })

        # Corroboration support: is a *different* component of this same
        # coarse type present with no ambiguity strategy at all (i.e. an
        # unquestionable label)? Only consulted by AMBIGUITY_CORROBORATION
        # components below -- see that strategy's docstring in
        # panorama_region_result.py. Every component remaining here already
        # cleared _MIN_REGION_FRACTION, so this needs no extra size check.
        unambiguous_present = any(c["strategy"] is None for c in components)

        for c in components:
            strategy = c["strategy"]
            if strategy == AMBIGUITY_CORROBORATION:
                # Two components that are each the same confusable call (e.g.
                # two "wall"-labeled cliff faces) don't corroborate each
                # other -- corroboration has to come from a component with no
                # ambiguity strategy at all.
                well_supported = unambiguous_present
            elif strategy == AMBIGUITY_CONFIDENCE:
                # Only distrust a genuinely close call (low margin) against a
                # plausible ground-valid alternative -- a confidently-called
                # region, or one whose only alternative isn't ground-valid
                # either, is left alone.
                well_supported = (
                    c["mean_margin"] >= confidence_margin_threshold
                    or not c["dominant_runnerup_type"].ground_valid
                )
            else:
                well_supported = True

            if strategy is not None and not well_supported:
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
                    mean_margin=round(c["mean_margin"], 4),
                    ambiguity_strategy=strategy,
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
        confidence_margin_threshold: float = 0.2,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.nadir_cutoff_deg = nadir_cutoff_deg
        self.nadir_band_deg = nadir_band_deg
        # AMBIGUITY_CONFIDENCE threshold (see panorama_region_result.py):
        # below this top1-vs-runner-up softmax margin, a region using that
        # strategy (currently VEGETATION) is considered a genuinely close
        # call rather than a comfortable one. Untuned against real model
        # output as of introduction -- couldn't run the model locally to
        # calibrate (broken scipy in the local dev env); revisit once this
        # has run against real captures.
        self.confidence_margin_threshold = confidence_margin_threshold


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

    Per-pixel top-1/top-2 confidence and the model's runner-up class are also
    kept (see panorama_segmentation_imp.py's _segment). A region whose
    dominant label matches a rule with an ambiguity strategy in
    panorama_region_result.py's _LABEL_RULES (a keyword group routinely
    confused with a different plausible coarse type in natural scenes -- e.g.
    ADE20K's "wall" class firing on a sunlit cliff face, or "tree" firing on
    dark, mottled rock texture) is only trusted if that strategy's own check
    passes: AMBIGUITY_CORROBORATION requires a *different*, unambiguous
    region of the same coarse type elsewhere in the panorama (safe for BUILT,
    which is rare in nature photography, so total absence elsewhere is
    itself strong evidence); AMBIGUITY_CONFIDENCE requires either a
    comfortable top1-vs-runner-up margin or a runner-up that isn't
    ground-valid either (used for VEGETATION instead, since real vegetation
    is common and unremarkable outdoors -- absence elsewhere proves nothing,
    but a genuinely close call against a plausible ground-valid alternative
    still does). See both strategies' docstrings for the full reasoning.
    Regions that fail their check have every pixel resolved to their own
    runner-up coarse type instead (see _build_result, PanoramaRegion.
    well_supported). This corrected map -- not the raw argmax output -- is
    what's written to the type-map output key, so every existing consumer of
    that key (HeightMapGenerator, RegionMapGenerator, skybox inpainting,
    terrain texture generation) benefits automatically without any change on
    their end.

    Input and all five output keys are overridable via config (keys: input,
    regions, type_map, type_map_raw, confidence_map, runnerup_type_map,
    ambiguous_mask), the same pattern PanoramaDepthStage uses for
    PANORAMA_DEPTH vs PANORAMA_OBJECT_DEPTH -- this stage is run twice in
    config.yaml, once (default keys) on ContextKey.PANORAMA for anything that
    needs to know what was really photographed (object detection/
    distribution, skybox inpainting), and once more (keys: input:
    panorama_terrain, ... _terrain output keys) on the object-removed +
    LoRA-corrected panorama for anything that shapes or textures the terrain
    itself, which should never see objects that were deliberately removed.

    Reads:  ContextKey.PANORAMA                       (default; overridable via keys.input)
    Writes: ContextKey.PANORAMA_REGIONS               (PanoramaRegionResult; keys.regions)
            ContextKey.PANORAMA_REGION_TYPE_MAP        (resolved, nadir-cleaned; keys.type_map)
            ContextKey.PANORAMA_REGION_TYPE_MAP_RAW    (raw argmax output, not nadir-cleaned; keys.type_map_raw)
            ContextKey.PANORAMA_REGION_CONFIDENCE_MAP  (per-pixel top-1 softmax confidence; keys.confidence_map)
            ContextKey.PANORAMA_REGION_RUNNERUP_TYPE_MAP (per-pixel coarse type of the 2nd-best class; keys.runnerup_type_map)
            ContextKey.PANORAMA_REGION_AMBIGUOUS_MASK  (bool; True where the raw and resolved maps differ; keys.ambiguous_mask)
    Debug:  self.output/regions.json               (reflects SegFormer's raw output —
                                                     not nadir-cleaned)
            self.output/label_overlay.png           (raw, not nadir-cleaned or resolved)
            self.output/label_overlay_resolved.png  (nadir-cleaned + ambiguity-resolved -- matches the type-map output)
            self.output/ambiguous_mask.png          (pixels resolved from a failed ambiguity check)
    """

    @classmethod
    def config_class(cls) -> type[PanoramaRegionConfiguration]:
        return PanoramaRegionConfiguration

    def __init__(self, config: PanoramaRegionConfiguration) -> None:
        super().__init__(config)
        self._client: PanoramaSegmentationClient | None = None

    def _resolved_keys(self):
        return self.keys({
            SemanticKey.INPUT: ContextKey.PANORAMA,
            SemanticKey.REGIONS: ContextKey.PANORAMA_REGIONS,
            SemanticKey.TYPE_MAP: ContextKey.PANORAMA_REGION_TYPE_MAP,
            SemanticKey.TYPE_MAP_RAW: ContextKey.PANORAMA_REGION_TYPE_MAP_RAW,
            SemanticKey.CONFIDENCE_MAP: ContextKey.PANORAMA_REGION_CONFIDENCE_MAP,
            SemanticKey.RUNNERUP_TYPE_MAP: ContextKey.PANORAMA_REGION_RUNNERUP_TYPE_MAP,
            SemanticKey.AMBIGUOUS_MASK: ContextKey.PANORAMA_REGION_AMBIGUOUS_MASK,
        })

    def run(self, context: PipelineContext) -> PipelineContext:
        (
            input_key, regions_key, type_map_key, type_map_raw_key,
            confidence_key, runnerup_key, ambiguous_key,
        ) = self._resolved_keys()

        panorama = context.input_panorama(input_key)
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

        seg = _build_result(raw, confidence_margin_threshold=cfg.confidence_margin_threshold)
        result = seg.result
        # Nadir cleanup only needs to run on the map that actually becomes the
        # canonical output; the raw map is kept purely for debugging (see
        # the type_map_raw output / regions.json, both intentionally left
        # un-nadir-cleaned, matching this stage's existing convention).
        resolved_type_idx_map = _clean_nadir_band(
            seg.resolved_type_idx_map, cfg.nadir_cutoff_deg, cfg.nadir_band_deg
        )
        context.add_panorama_regions(regions_key, result)
        context.add_depth(type_map_key, resolved_type_idx_map.astype(np.float32))
        context.add_depth(type_map_raw_key, seg.type_idx_map.astype(np.float32))
        context.add_depth(confidence_key, seg.confidence_map)
        context.add_depth(runnerup_key, seg.runnerup_type_idx_map.astype(np.float32))
        context.add_depth(ambiguous_key, seg.ambiguous_mask.astype(np.float32))

        n_ambiguous = int(seg.ambiguous_mask.sum())
        if n_ambiguous > 0:
            frac = n_ambiguous / seg.ambiguous_mask.size
            self.log_info(
                f"  Resolved {frac * 100:.2f}% of panorama from a failed ambiguity "
                f"check (corroboration or confidence) to its runner-up type"
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
        # Resolved, not the hardcoded default -- this stage runs twice (see
        # class docstring), and checking the literal default key here would
        # make the terrain-scoped instance falsely report cached output
        # already present as soon as the first (original-panorama) instance
        # has run, skipping it entirely.
        _, regions_key, *_ = self._resolved_keys()
        return context.panorama_regions(regions_key) is not None

    def model_names(self) -> list[str]:
        return PanoramaSegmentationClient.model_names()

    def clean_up(self):
        if self._client is not None:
            self._client.close()
            self._client = None
        super().clean_up()
