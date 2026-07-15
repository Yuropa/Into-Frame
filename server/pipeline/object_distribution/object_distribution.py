import json
import os
import subprocess
from pathlib import Path
from typing import Any
from logging import Logger

import torch

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.object_distribution.object_distribution_result import (
    ObjectDistributionResult,
    TypeDistribution,
)
from pipeline.panorama_segmentation.panorama_region_result import RegionType
from pipeline.object_typing.categories import UNIQUE_CATEGORIES
from pipeline.scene_generation.projection import unproject_bbox, unproject_bbox_equirect

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
) -> dict | None:
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
        if result.returncode != 0:
            return None
        return json.loads(result.stdout.strip())
    except Exception:
        return None


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
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.bin_count = bin_count
        self.pcf_cli_path = pcf_cli_path


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
            if obj_type not in UNIQUE_CATEGORIES
        ]

        if not distributable_types:
            self.log_info("No distributable object types found")
            result = ObjectDistributionResult()
            context.add_object_distribution(ContextKey.OBJECT_DISTRIBUTION, result)
            return context

        # Collect points + footprint sizes per (obj_type, region_type), in world XZ meters.
        points_by_group: dict[tuple[str, str], list[tuple[float, float]]] = {}
        sizes_by_group: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for obj_type in distributable_types:
            grp = correlation.groups[obj_type]
            for idx in grp.indices:
                metadata = context.input_object(f"metadata_{idx}") or {}
                box = metadata.get("box")
                if not box:
                    continue

                # Same branch selection SceneGenerationStage uses to place real objects.
                if panorama_depth is not None and panorama is not None:
                    unprojected = unproject_bbox_equirect(
                        box, panorama.width, panorama.height,
                        pano_depth=panorama_depth, extrinsics=extrinsics,
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

        if not points_by_group:
            self.log_info("No objects could be unprojected to world space")
            result = ObjectDistributionResult()
            context.add_object_distribution(ContextKey.OBJECT_DISTRIBUTION, result)
            return context

        task = self.create_progress(len(points_by_group), "Modelling distributions…")
        result = ObjectDistributionResult()

        for (obj_type, region_type), points in points_by_group.items():
            sizes = sizes_by_group[(obj_type, region_type)]
            if len(points) >= _MIN_POINTS:
                pcf_data = _run_pcf_cli(pcf_cli, points, self._bin_count)
                if pcf_data is None:
                    self.log_info(f"  {obj_type} [{region_type}]: pcf_cli failed, skipping")
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
