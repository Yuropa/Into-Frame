import json
import math
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
from pipeline.panorama_segmentation.panorama_region_result import PanoramaRegionResult
from pipeline.object_typing.categories import UNIQUE_CATEGORIES

_MIN_POINTS = 2
_DEFAULT_BIN_COUNT = 48
_GLOBAL_REGION = "global"

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


def _project_center_to_pano(
    box: list[float],
    intrinsics,
    pano_w: int,
    pano_h: int,
) -> tuple[float, float] | None:
    """Project the center of a perspective bounding box onto the equirectangular panorama."""
    x, y, w, h = box
    px = x + w / 2.0
    py = y + h / 2.0

    scale_x = intrinsics.color_width / intrinsics.width if intrinsics.width else 1.0
    scale_y = intrinsics.color_height / intrinsics.height if intrinsics.height else 1.0
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    cx = intrinsics.px * scale_x
    cy = intrinsics.py * scale_y

    x_cam = (px - cx) / fx
    y_cam = -((py - cy) / fy)
    z_cam = 1.0
    n = math.sqrt(x_cam ** 2 + y_cam ** 2 + z_cam ** 2)
    x_cam /= n
    y_cam /= n
    z_cam /= n

    lon = math.atan2(x_cam, z_cam)
    lat = math.atan2(y_cam, math.sqrt(x_cam ** 2 + z_cam ** 2))

    u = lon / (2.0 * math.pi) + 0.5
    v = 0.5 - lat / math.pi
    return u, v


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


def _region_type_for_point(
    u: float,
    v: float,
    pano_w: int,
    pano_h: int,
    regions: PanoramaRegionResult,
) -> str:
    """Return the coarse region type that contains panorama UV point (u, v)."""
    px = u * pano_w
    py = v * pano_h

    # Collect all regions whose bbox contains this point.
    containing = []
    for region in regions.regions:
        rx, ry, rw, rh = region.bbox
        if rx <= px <= rx + rw and ry <= py <= ry + rh:
            containing.append(region)

    if containing:
        # If multiple regions overlap this point, prefer the largest by area.
        return max(containing, key=lambda r: r.area_fraction).region_type

    # Fall back to nearest centroid.
    best = min(
        regions.regions,
        key=lambda r: (r.centroid[0] - px) ** 2 + (r.centroid[1] - py) ** 2,
        default=None,
    )
    return best.region_type if best is not None else _GLOBAL_REGION


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

    Projects each object's bounding-box centre onto the equirectangular panorama,
    assigns it to a coarse region type from PanoramaRegionResult, then calls the
    `pcf_cli` binary from pattern-synthesis on each (object type, region type) group.
    Groups with two or more instances get a full PCF histogram; groups with a single
    instance are recorded as singletons (n_points=1, empty hist) so the presence of
    the object is preserved. Falls back to a synthetic "global" region when
    PANORAMA_REGIONS is not in context.

    Reads:  ContextKey.OBJECT_CORRELATION, ContextKey.INTRINSICS, ContextKey.PANORAMA,
            ContextKey.PANORAMA_REGIONS (optional)
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

        intrinsics = context.input_intrinsics(ContextKey.INTRINSICS)
        panorama = context.input_panorama(ContextKey.PANORAMA)
        if intrinsics is None or panorama is None:
            self.log_info("Intrinsics or panorama missing, skipping")
            return context

        pano_w, pano_h = panorama.size
        regions = context.input_panorama_regions(ContextKey.PANORAMA_REGIONS)
        if regions is None:
            self.log_info("No panorama regions — grouping all objects under 'global'")

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

        # Collect points per (obj_type, region_type).
        points_by_group: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for obj_type in distributable_types:
            grp = correlation.groups[obj_type]
            for idx in grp.indices:
                metadata = context.input_object(f"metadata_{idx}") or {}
                box = metadata.get("box")
                if not box:
                    continue
                projected = _project_center_to_pano(box, intrinsics, pano_w, pano_h)
                if projected is None:
                    continue
                u, v = projected
                region_type = (
                    _region_type_for_point(u, v, pano_w, pano_h, regions)
                    if regions is not None
                    else _GLOBAL_REGION
                )
                points_by_group.setdefault((obj_type, region_type), []).append(projected)

        if not points_by_group:
            self.log_info("No objects could be projected onto the panorama")
            result = ObjectDistributionResult()
            context.add_object_distribution(ContextKey.OBJECT_DISTRIBUTION, result)
            return context

        task = self.create_progress(len(points_by_group), "Modelling distributions…")
        result = ObjectDistributionResult()

        for (obj_type, region_type), points in points_by_group.items():
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
