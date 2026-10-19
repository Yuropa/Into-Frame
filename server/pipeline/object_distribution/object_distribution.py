import json
import os
import subprocess
from pathlib import Path
from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from util.json_utils import parse_json_from_stream
from pipeline.object_distribution.object_distribution_result import (
    ObjectDistributionResult,
    TypeDistribution,
)
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.object_typing.categories import DISTRIBUTABLE_CATEGORIES, OUTPAINTED_PLACEABLE_CATEGORIES
from pipeline.scene_generation.projection import unproject_bbox, unproject_bbox_equirect, class_height_range
from pipeline.segmentation.hires_crops import is_photographed, CAPTURED_FOV_RESTRICTED_KEY

_MIN_POINTS = 2
_DEFAULT_BIN_COUNT = 48

_HERE = Path(__file__).resolve().parent


def _find_pcf_cli(configured_path: str | None) -> Path | None:
    if configured_path:
        p = Path(configured_path)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        return None

    candidates = [
        _HERE.parents[2] / "pattern-synthesis" / "build" / "pcf_cli",
        _HERE.parents[2] / "pattern-synthesis" / "build" / "Release" / "pcf_cli",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _region_type_at(x: float, z: float, region_map, grid_size_meters: float) -> str:
    """Look up the top-down REGION_MAP label at world-space (x, z)."""
    h, w = region_map.shape
    half = grid_size_meters / 2.0
    col = int((x + half) / grid_size_meters * w)
    row = int((z + half) / grid_size_meters * h)
    if 0 <= row < h and 0 <= col < w:
        type_idx = int(round(float(region_map[row, col])))
        if 0 <= type_idx < len(RegionType):
            return RegionType(type_idx).label
    return RegionType.OTHER.label


def _run_pcf_cli(
    pcf_cli: Path,
    points: list[tuple[float, float]],
    bin_count: int,
) -> tuple[dict | None, str | None]:
    """Run pcf_cli on one group's points.

    Returns (result, None) on success or (None, reason) on failure, where reason
    carries the exit code and the last line of stderr -- the same contract
    DistributionSynthesisStage's _run_synthesize_cli uses. An earlier version
    returned a bare None on every failure path, so a missing shared library, a
    timeout and a malformed point set all logged the identical "pcf_cli failed"
    line with nothing to distinguish them.
    """
    n = len(points)
    lines = [f"{bin_count}\n{n}"]
    for u, v in points:
        lines.append(f"{u:.17g} {v:.17g}")
    stdin_data = "\n".join(lines) + "\n"

    try:
        result = subprocess.run(
            [str(pcf_cli)],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after 60s with {n} points"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    stderr_lines = (result.stderr or "").strip().splitlines()
    last_stderr = stderr_lines[-1].strip() if stderr_lines else ""
    if result.returncode != 0:
        return None, f"exit {result.returncode}: {last_stderr or 'no stderr'}"
    # Tolerant of anything printed alongside the JSON -- pcf_cli's stdout is clean
    # today (it links neither synthesis_core.cpp nor voronoi-pcf.cpp, both of which
    # print progress), but the identical assumption in DistributionSynthesisStage
    # is what silently discarded every synthesize_cli result. Same class of bug,
    # one line to be immune to it.
    parsed = parse_json_from_stream(result.stdout)
    if parsed is None:
        head = " | ".join(result.stdout.strip().splitlines()[:2])[:200]
        return None, f"no JSON object in stdout: {head or '(empty)'}"
    return parsed, None


class ObjectDistributionStageConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        bin_count: int = _DEFAULT_BIN_COUNT,
        pcf_cli_path: str | None = None,
        outpainted_classes: list[str] | None = None,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.bin_count = bin_count
        self.pcf_cli_path = pcf_cli_path
        # See hires_crops.is_photographed: these classes learn from outpainted
        # exemplars too, since they are placed from them.
        self.outpainted_classes = frozenset(
            outpainted_classes if outpainted_classes is not None else OUTPAINTED_PLACEABLE_CATEGORIES
        )


class ObjectDistributionStage(PipelineStage):
    """
    Computes per-region Voronoi PCF histograms for each distributable object type.

    Unprojects each object's bounding box to a world-space position (same helper
    SceneGenerationStage uses to place real objects), looks up its region type from
    the top-down REGION_MAP at that (x, z), then calls the `pcf_cli` binary from
    pattern-synthesis on each (object type, region type) group. Groups with two or
    more instances get a full PCF histogram; groups with a single instance are
    recorded as singletons (n_points=1, empty hist) so the presence of the object is
    preserved. Each TypeDistribution also carries its raw exemplar points and
    footprint sizes (world-space), which DistributionSynthesisStage uses to paint
    the pattern across the rest of that region type.

    Reads:  ContextKey.OBJECT_CORRELATION, ContextKey.REGION_MAP,
            ContextKey.HEIGHT_MAP_PARAMS, ContextKey.INTRINSICS, ContextKey.EXTRINSICS,
            ContextKey.DEPTH, ContextKey.INPUT, ContextKey.PANORAMA,
            ContextKey.PANORAMA_OBJECT_DEPTH (optional — depth on the ORIGINAL panorama;
            matches what objects were detected against, unlike PANORAMA_DEPTH)
    Writes: ContextKey.OBJECT_DISTRIBUTION (ObjectDistributionResult)
    Config: pcf_cli_path (str, optional) — override binary location
            bin_count    (int, default 48)
    Debug:  self.output/distributions.json
    """

    @classmethod
    def config_class(cls) -> type[ObjectDistributionStageConfiguration]:
        return ObjectDistributionStageConfiguration

    def __init__(self, config: ObjectDistributionStageConfiguration) -> None:
        super().__init__(config)
        self._pcf_cli_path: str | None = config.pcf_cli_path
        self._bin_count: int = config.bin_count

    def run(self, context: PipelineContext) -> PipelineContext:
        correlation = context.input_object_correlation(ContextKey.OBJECT_CORRELATION)
        if correlation is None:
            self.log_info("No correlation data, skipping")
            return context

        pcf_cli = _find_pcf_cli(self._pcf_cli_path)
        if pcf_cli is None:
            self.log_info(
                "pcf_cli binary not found — build pattern-synthesis first "
                "(cmake .. && make pcf_cli). Skipping distribution stage."
            )
            # Still write an (empty) result -- has_expected_output() only checks
            # presence, and leaving OBJECT_DISTRIBUTION unset here makes this stage
            # (and everything downstream, via the dirty cascade) look permanently
            # incomplete and rerun on every single invocation even when the binary
            # genuinely isn't going to appear.
            context.add_object_distribution(ContextKey.OBJECT_DISTRIBUTION, ObjectDistributionResult())
            return context

        region_map_depth = context.input_depth(ContextKey.REGION_MAP)
        if region_map_depth is None:
            self.log_info("No region map, skipping")
            return context
        region_map = region_map_depth.depth
        grid_size_meters = (context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}).get(
            "grid_size_meters", 100.0
        )

        intrinsics = context.input_intrinsics(ContextKey.INTRINSICS)
        extrinsics = context.input_extrinsics(ContextKey.EXTRINSICS)
        depth = context.input_depth(ContextKey.DEPTH)
        input_image = context.input_image(ContextKey.INPUT)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        # See SceneGenerationStage's PANORAMA_OBJECT_DEPTH comment — objects were detected on
        # the original panorama, so unprojection needs depth measured on that same image.
        panorama_depth = context.input_depth(ContextKey.PANORAMA_OBJECT_DEPTH)

        distributable_types = [
            obj_type
            for obj_type, grp in correlation.groups.items()
            if obj_type in DISTRIBUTABLE_CATEGORIES
        ]

        if not distributable_types:
            self.log_info("No distributable object types found")
            result = ObjectDistributionResult()
            context.add_object_distribution(ContextKey.OBJECT_DISTRIBUTION, result)
            return context

        # Collect points + footprint sizes per (obj_type, region_type), in world XZ meters.
        points_by_group: dict[tuple[str, str], list[tuple[float, float]]] = {}
        sizes_by_group: dict[tuple[str, str], list[tuple[float, float]]] = {}
        # Index-aligned with points/sizes -- see TypeDistribution.buckets for why the
        # visual variant travels with each exemplar instead of splitting the groups.
        buckets_by_group: dict[tuple[str, str], list[int]] = {}
        fov_restricted = bool(context.input_object(CAPTURED_FOV_RESTRICTED_KEY) or False)
        unphotographed = 0
        for obj_type in distributable_types:
            grp = correlation.groups[obj_type]
            for idx in grp.indices:
                metadata = context.input_object(f"metadata_{idx}") or {}
                box = metadata.get("box")
                if not box:
                    continue
                # The spatial pattern is learned from what was photographed; an
                # exemplar from the outpainted region is the generator's pattern,
                # not the scene's (and usually a copy of a real one).
                if not is_photographed(metadata, fov_restricted, self.config.outpainted_classes):
                    unphotographed += 1
                    continue

                # Same branch selection SceneGenerationStage uses to place real objects.
                if panorama_depth is not None and panorama is not None:
                    unprojected = unproject_bbox_equirect(
                        box, panorama.width, panorama.height,
                        pano_depth=panorama_depth, extrinsics=extrinsics,
                        height_range_m=class_height_range(obj_type),
                        max_distance_m=0.475 * float(grid_size_meters),
                    )
                elif depth is not None and input_image is not None and intrinsics is not None:
                    unprojected = unproject_bbox(
                        box, input_image.width, input_image.height,
                        depth_map=depth, intrinsics=intrinsics, extrinsics=extrinsics,
                    )
                else:
                    unprojected = None
                if unprojected is None:
                    continue

                position, width, height = unprojected
                x, z = float(position[0]), float(position[2])
                region_type = _region_type_at(x, z, region_map, grid_size_meters)
                key = (obj_type, region_type)
                points_by_group.setdefault(key, []).append((x, z))
                sizes_by_group.setdefault(key, []).append((float(width), float(height)))
                buckets_by_group.setdefault(key, []).append(int(metadata.get("bucket") or 0))

        if unphotographed:
            self.log_info(f"Captured-cone policy: {unphotographed} detection(s) outside the photographed cone not used as exemplars")

        if not points_by_group:
            self.log_info("No objects could be unprojected to world space")
            result = ObjectDistributionResult()
            context.add_object_distribution(ContextKey.OBJECT_DISTRIBUTION, result)
            return context

        task = self.create_progress(len(points_by_group), "Modelling distributions…")
        result = ObjectDistributionResult()

        for (obj_type, region_type), points in points_by_group.items():
            sizes = sizes_by_group[(obj_type, region_type)]
            buckets = buckets_by_group[(obj_type, region_type)]
            if len(points) >= _MIN_POINTS:
                pcf_data, reason = _run_pcf_cli(pcf_cli, points, self._bin_count)
                if pcf_data is None:
                    self.log_warning(f"  {obj_type} [{region_type}]: pcf_cli failed ({reason}), skipping")
                    self.advance_progress(task)
                    continue
                dist = TypeDistribution(
                    object_type=obj_type,
                    region_type=region_type,
                    n_points=pcf_data.get("n_points", len(points)),
                    bin_count=pcf_data.get("bin_count", self._bin_count),
                    hist=pcf_data.get("hist", []),
                    pair_count=pcf_data.get("pair_count", 0),
                    points=points,
                    sizes=sizes,
                    buckets=buckets,
                )
                self.log_info(
                    f"  {obj_type} [{region_type}]: {dist.n_points} points, "
                    f"{dist.pair_count} pairs, "
                    f"peak bin {dist.hist.index(max(dist.hist)) if dist.hist else '?'}"
                )
            else:
                dist = TypeDistribution(
                    object_type=obj_type,
                    region_type=region_type,
                    n_points=len(points),
                    bin_count=self._bin_count,
                    hist=[],
                    pair_count=0,
                    points=points,
                    sizes=sizes,
                    buckets=buckets,
                )
                self.log_info(f"  {obj_type} [{region_type}]: 1 instance (singleton)")

            result.distributions.setdefault(region_type, {})[obj_type] = dist
            self.advance_progress(task)

        context.add_object_distribution(ContextKey.OBJECT_DISTRIBUTION, result)
        self.finish_progress(task)
        self._write_debug(result)
        return context

    def _write_debug(self, result: ObjectDistributionResult):
        if self.output is None:
            return
        import json as _json
        with open(self.output / "distributions.json", "w") as f:
            _json.dump(result.encode(), f, indent=2)

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object_distribution(ContextKey.OBJECT_DISTRIBUTION) is not None
