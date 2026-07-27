import json
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from logging import Logger

import cv2
import numpy as np
import torch
from PIL import Image as PILImage, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull, KDTree

from pipeline.pipeline_stage import PipelineStageConfiguration, PipelineStage
from pipeline.pipeline_context import PipelineContext, ContextKey
from pipeline.panorama_segmentation.panorama_region_result import (
    RegionType,
    colorize_region_type_map,
    paintable_region_types,
)

_HERE = Path(__file__).resolve().parent

# Minimum world-space padding (metres) around a group's exemplar/output boundaries,
# regardless of how tight the points are, so degenerate (near-collinear) point sets
# still produce a usable polygon.
_MIN_PAD_M = 0.5
# Floor on the exemplar candidate grid's point count -- the grid itself is now sized
# to match tile_domain's own density (1/spacing^2, see the loop below), not a fixed
# count; this only guards against a degenerate near-zero-area input boundary.
_MIN_EXEMPLAR_DOMAIN = 16

# OBJECT_COUNT and metadata_{idx} are both inherited from earlier stages, so neither
# can serve as a "did this stage actually run" marker for has_expected_output — an
# ad hoc key it always (and only) writes on a completed run is the only reliable one.
_RAN_MARKER = "distribution_synthesis_complete"


def _find_synthesize_cli(configured_path: str | None) -> Path | None:
    if configured_path:
        p = Path(configured_path)
        if p.is_file() and os.access(p, os.X_OK):
            return p
        return None

    candidates = [
        _HERE.parents[2] / "pattern-synthesis" / "build" / "synthesize_cli",
        _HERE.parents[2] / "pattern-synthesis" / "build" / "Release" / "synthesize_cli",
    ]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _grid_to_world(row: float, col: float, grid_size_meters: float, grid_resolution: int) -> tuple[float, float]:
    half = grid_size_meters / 2.0
    cell_m = grid_size_meters / grid_resolution
    x = (col + 0.5) * cell_m - half
    z = (row + 0.5) * cell_m - half
    return x, z


def _mean_nn_spacing(points: list[tuple[float, float]]) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 1.0
    tree = KDTree(pts)
    dists, _ = tree.query(pts, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(nn.mean()) if len(nn) > 0 else 1.0


def _padded_bbox_polygon(pts: np.ndarray, pad: float) -> np.ndarray:
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def _padded_hull_polygon(points: list[tuple[float, float]], pad: float) -> np.ndarray:
    """Convex hull of `points`, expanded outward by `pad` metres. Falls back to a padded
    bounding box when there are too few points or they're degenerate (collinear)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return _padded_bbox_polygon(pts, pad)
    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except Exception:
        return _padded_bbox_polygon(pts, pad)
    centroid = hull_pts.mean(axis=0)
    directions = hull_pts - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return hull_pts + directions / norms * pad


def _local_grid(bbox_min: np.ndarray, bbox_max: np.ndarray, target_count: int) -> np.ndarray:
    """Regular point grid of ~target_count points spanning [bbox_min, bbox_max]."""
    res = max(2, int(round(math.sqrt(max(target_count, 4)))))
    xs = np.linspace(bbox_min[0], bbox_max[0], res)
    zs = np.linspace(bbox_min[1], bbox_max[1], res)
    X, Z = np.meshgrid(xs, zs)
    return np.stack([X.ravel(), Z.ravel()], axis=1)


def _run_synthesize_cli(
    cli_path: Path,
    domain_points: np.ndarray,
    exemplar_points: np.ndarray,
    input_boundary: np.ndarray,
    output_boundary: np.ndarray,
    n_points: int,
    bin_count: int,
    max_iters: int,
    seed: int,
) -> dict | None:
    lines = [f"{bin_count} {n_points} {max_iters} {seed}"]

    def emit(pts: np.ndarray):
        lines.append(str(len(pts)))
        for x, z in pts:
            lines.append(f"{x:.6f} {z:.6f}")

    emit(domain_points)
    emit(exemplar_points)
    emit(input_boundary)
    emit(output_boundary)
    stdin_data = "\n".join(lines) + "\n"

    try:
        result = subprocess.run(
            [str(cli_path)],
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout.strip())
    except Exception:
        return None


class DistributionSynthesisConfiguration(PipelineStageConfiguration):
    def __init__(
        self,
        name: str,
        device: torch.device,
        torch_dtype: Any,
        log: Logger,
        keys=None,
        seed: int = 0,
        min_blob_area_cells: int = 64,
        max_candidates_per_tile: int = 2000,
        max_iters: int = 400,
        size_jitter: float = 0.15,
        input_boundary_pad_factor: float = 1.5,
        synthesize_cli_path: str | None = None,
        max_workers: int | None = None,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.min_blob_area_cells = min_blob_area_cells
        self.max_candidates_per_tile = max_candidates_per_tile
        self.max_iters = max_iters
        self.size_jitter = size_jitter
        self.input_boundary_pad_factor = input_boundary_pad_factor
        self.synthesize_cli_path = synthesize_cli_path
        # Each tile's synthesize_cli call is an independent, CPU-bound subprocess (no
        # shared state), so tiles run concurrently in a thread pool -- subprocess.run
        # releases the GIL while the child process runs, so this gets real OS-level
        # parallelism across cores despite being threads, not processes. None -> os.cpu_count().
        self.max_workers = max_workers


class DistributionSynthesisStage(PipelineStage):
    """
    Paints each learned (object_type, region_type) distribution across every matching
    patch of the top-down REGION_MAP, filling the whole environment rather than only
    the originally-detected instances.

    "Matching patch" is the region_type's whole paintable group (see
    panorama_region_result.paintable_region_types), not just an exact label match:
    TERRAIN/GROUND/VEGETATION are merged into one ground-like domain, since ADE20K
    segmentation routinely splits one contiguous walkable area into those three
    labels (grass underfoot vs. a distant hillside vs. a tree's own canopy shadow).
    Without this, a distribution learned from exemplars observed on whichever single
    label they happened to land on would only ever paint that same small patch,
    instead of spreading across the much larger surrounding ground-like terrain.

    For each non-singleton TypeDistribution: finds connected components of
    `region_map`'s paintable group for that region_type, splits large components
    into bounded-candidate-count tiles (the underlying synthesize_pattern optimizer
    is O(candidates²), so a single call can't cover a large area), and calls the
    `synthesize_cli` binary from
    pattern-synthesis once per tile — exemplar points/PCF stay fixed per group, only
    the output boundary and RNG seed change per tile. Synthesized points are appended
    as new `metadata_{idx}` entries (`"synthetic": True`, world position + footprint,
    no detection box) and OBJECT_COUNT is bumped to include them, so
    SceneGenerationStage places them through its normal terrain-snap + mesh/billboard
    path.

    Every tile across every group is an independent synthesize_cli subprocess call, so
    they're all built up front (cheap: boundary/contour geometry, no CLI calls) and then
    run concurrently in a thread pool (see max_workers) rather than one at a time — this
    is what previously made the stage take tens of minutes on a large/densely-tiled
    scene despite having many idle cores. Per-tile results are still consumed back in
    the exact same deterministic (group, blob, tile) order the serial version used, so
    the RNG draws for size/jitter — and therefore the output for a given seed — are
    unchanged by parallelizing.

    Reads:  ContextKey.OBJECT_DISTRIBUTION, ContextKey.REGION_MAP,
            ContextKey.HEIGHT_MAP_PARAMS, ContextKey.OBJECT_COUNT
    Writes: metadata_{idx} for each synthesized point, ContextKey.OBJECT_COUNT (bumped)
    Config: synthesize_cli_path (str, optional) — override binary location
            min_blob_area_cells      (int, default 64)
            max_candidates_per_tile  (int, default 2000)
            max_iters                (int, default 400)
            size_jitter              (float, default 0.15) — +/- fraction applied to
                                       sampled footprint sizes
            input_boundary_pad_factor (float, default 1.5) — exemplar hull padding, in
                                       units of mean exemplar nearest-neighbour spacing
            max_workers               (int, optional) — concurrent synthesize_cli
                                       subprocesses; default os.cpu_count()
    Debug:  self.temp/synthesis_{region_type}_{object_type}.png
    """

    @classmethod
    def config_class(cls) -> type[DistributionSynthesisConfiguration]:
        return DistributionSynthesisConfiguration

    def __init__(self, config: DistributionSynthesisConfiguration) -> None:
        super().__init__(config)

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: DistributionSynthesisConfiguration = self.config

        distribution = context.input_object_distribution(ContextKey.OBJECT_DISTRIBUTION)
        if distribution is None:
            self.log_info("No object distribution, skipping")
            # Mark complete even though nothing was painted: has_expected_output()
            # only checks this marker, so returning without it makes this stage --
            # and, via the dirty cascade, Panorama Asset Generation, Scene
            # Generation, Video Generation, Video Object Extraction, Motion
            # Classification, Rigging and Animation -- look permanently incomplete
            # and rerun on every single invocation. Same guard ObjectDistributionStage
            # already applies for its own "binary not found" path.
            context.add_object(_RAN_MARKER, True)
            return context

        region_map_depth = context.input_depth(ContextKey.REGION_MAP)
        if region_map_depth is None:
            self.log_info("No region map, skipping")
            # Mark complete even though nothing was painted: has_expected_output()
            # only checks this marker, so returning without it makes this stage --
            # and, via the dirty cascade, Panorama Asset Generation, Scene
            # Generation, Video Generation, Video Object Extraction, Motion
            # Classification, Rigging and Animation -- look permanently incomplete
            # and rerun on every single invocation. Same guard ObjectDistributionStage
            # already applies for its own "binary not found" path.
            context.add_object(_RAN_MARKER, True)
            return context
        region_map = region_map_depth.depth
        grid_resolution = region_map.shape[0]
        grid_size_meters = (context.input_object(ContextKey.HEIGHT_MAP_PARAMS) or {}).get(
            "grid_size_meters", 100.0
        )

        object_count = context.input_object(ContextKey.OBJECT_COUNT) or 0

        cli_path = _find_synthesize_cli(cfg.synthesize_cli_path)
        if cli_path is None:
            self.log_info(
                "synthesize_cli binary not found — build pattern-synthesis first "
                "(cmake .. && make synthesize_cli). Skipping distribution synthesis."
            )
            # Mark complete even though nothing was painted: has_expected_output()
            # only checks this marker, so returning without it makes this stage --
            # and, via the dirty cascade, Panorama Asset Generation, Scene
            # Generation, Video Generation, Video Object Extraction, Motion
            # Classification, Rigging and Animation -- look permanently incomplete
            # and rerun on every single invocation. Same guard ObjectDistributionStage
            # already applies for its own "binary not found" path.
            context.add_object(_RAN_MARKER, True)
            return context

        groups = [
            (region_type, obj_type, dist)
            for region_type, by_type in distribution.distributions.items()
            for obj_type, dist in by_type.items()
            if dist.n_points >= 2 and len(dist.points) >= 2
        ]
        if not groups:
            self.log_info("No non-singleton distributions to paint")
            # Mark complete even though nothing was painted: has_expected_output()
            # only checks this marker, so returning without it makes this stage --
            # and, via the dirty cascade, Panorama Asset Generation, Scene
            # Generation, Video Generation, Video Object Extraction, Motion
            # Classification, Rigging and Animation -- look permanently incomplete
            # and rerun on every single invocation. Same guard ObjectDistributionStage
            # already applies for its own "binary not found" path.
            context.add_object(_RAN_MARKER, True)
            return context

        rng = np.random.default_rng(self.seed)
        cell_m = grid_size_meters / grid_resolution
        next_idx = object_count
        seed_counter = 0

        # Phase 1: build every tile's synthesize_cli job up front — boundary/contour
        # geometry only, no subprocess calls yet — so all of them (across every group)
        # can run concurrently in phase 2. Jobs are appended in the same deterministic
        # (group, blob, tile) order the old serial loop used, and seeded the same way
        # (self.seed + seed_counter), so results are consumed in that same order below
        # and the RNG draws for size/jitter — hence the final output for a given seed —
        # are unaffected by parallelizing.
        jobs: list[dict] = []
        for group_idx, (region_type, obj_type, dist) in enumerate(groups):
            try:
                region_type_idx = int(RegionType.from_label(region_type))
            except KeyError:
                self.log_info(f"  {obj_type} [{region_type}]: unknown region type, skipping")
                continue

            exemplar_pts = np.asarray(dist.points, dtype=np.float64)
            spacing = max(_mean_nn_spacing(dist.points), 0.1)
            input_boundary = _padded_hull_polygon(
                dist.points, pad=max(spacing * cfg.input_boundary_pad_factor, _MIN_PAD_M)
            )
            # synthesize_pattern infers how many points to paint from the RATIO of
            # candidates that fall inside the input vs. output boundary, as a proxy
            # for the ratio of their real areal densities (see synthesis_core.cpp).
            # That proxy only holds if both candidate grids are built at the same
            # points-per-area rate. tile_domain below is deliberately sized to
            # 1/spacing^2 (matching the exemplar's own nearest-neighbour spacing) --
            # mirror that here instead of a fixed point count, or a small/sparse
            # exemplar cluster (the common case -- most groups have well under a few
            # hundred real instances) makes this grid far denser than the tile grid,
            # which silently divides the inferred output count by that mismatch and
            # paints far sparser than the real exemplar density.
            input_area = float(np.prod(np.maximum(
                input_boundary.max(axis=0) - input_boundary.min(axis=0), 1e-6
            )))
            exemplar_domain_target = max(_MIN_EXEMPLAR_DOMAIN, int(round(input_area / (spacing ** 2))))
            exemplar_domain = _local_grid(
                input_boundary.min(axis=0), input_boundary.max(axis=0), exemplar_domain_target
            )

            # Tile side length in grid cells: pick a tile candidate resolution (points
            # per side) and space those points at the exemplar's own nearest-neighbour
            # spacing, so tiles are fine enough for placement flexibility without
            # blowing the O(candidates²) budget of a single synthesize_pattern call.
            tile_res = max(4, int(round(math.sqrt(cfg.max_candidates_per_tile))))
            tile_side_m = tile_res * spacing
            tile_side_px = max(1, int(round(tile_side_m / cell_m)))

            paintable_idxs = [int(rt) for rt in paintable_region_types(RegionType(region_type_idx))]
            mask = np.isin(region_map, paintable_idxs)
            labels, n_components = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int32))

            for label_id in range(1, n_components + 1):
                blob_mask = labels == label_id
                rows, cols = np.nonzero(blob_mask)
                if len(rows) < cfg.min_blob_area_cells:
                    continue
                r_min, r_max = int(rows.min()), int(rows.max()) + 1
                c_min, c_max = int(cols.min()), int(cols.max()) + 1

                for tr in range(r_min, r_max, tile_side_px):
                    for tc in range(c_min, c_max, tile_side_px):
                        tr_end = min(tr + tile_side_px, r_max)
                        tc_end = min(tc + tile_side_px, c_max)
                        tile_mask = blob_mask[tr:tr_end, tc:tc_end]
                        if int(tile_mask.sum()) < cfg.min_blob_area_cells:
                            continue

                        contours, _ = cv2.findContours(
                            tile_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        if not contours:
                            continue
                        largest = max(contours, key=cv2.contourArea)
                        approx = cv2.approxPolyDP(largest, epsilon=1.0, closed=True)
                        if len(approx) < 3:
                            continue

                        output_boundary = np.array(
                            [_grid_to_world(pt[0][1] + tr, pt[0][0] + tc, grid_size_meters, grid_resolution)
                             for pt in approx],
                            dtype=np.float64,
                        )

                        tile_world_min, tile_world_max = output_boundary.min(axis=0), output_boundary.max(axis=0)
                        pad = max(cell_m, _MIN_PAD_M)
                        tile_domain = _local_grid(
                            tile_world_min - pad, tile_world_max + pad, cfg.max_candidates_per_tile
                        )
                        domain_points = np.concatenate([exemplar_domain, tile_domain], axis=0)

                        seed_counter += 1
                        jobs.append({
                            "group_idx": group_idx,
                            "domain_points": domain_points,
                            "exemplar_points": exemplar_pts,
                            "input_boundary": input_boundary,
                            "output_boundary": output_boundary,
                            "bin_count": dist.bin_count,
                            "seed": self.seed + seed_counter,
                        })

        # (x, z, width, height, bucket) per painted instance.
        group_placed: list[list[tuple[float, float, float, float, int]]] = [[] for _ in groups]

        if jobs:
            max_workers = cfg.max_workers or os.cpu_count() or 1
            self.log_info(
                f"Painting {len(jobs)} tile(s) across {len(groups)} group(s) "
                f"using up to {max_workers} concurrent synthesize_cli worker(s)…"
            )
            task = self.create_progress(len(jobs), "Painting distributions…")

            # Phase 2: run every tile's synthesize_cli call concurrently — each is an
            # independent, CPU-bound subprocess with no shared state. subprocess.run
            # releases the GIL while its child process runs, so threads give real
            # OS-level parallelism across cores here without process-pool overhead or
            # having to pickle the (potentially large) domain-point arrays.
            results: list[dict | None] = [None] * len(jobs)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _run_synthesize_cli,
                        cli_path,
                        domain_points=job["domain_points"],
                        exemplar_points=job["exemplar_points"],
                        input_boundary=job["input_boundary"],
                        output_boundary=job["output_boundary"],
                        n_points=-1,
                        bin_count=job["bin_count"],
                        max_iters=cfg.max_iters,
                        seed=job["seed"],
                    ): i
                    for i, job in enumerate(jobs)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        results[i] = future.result()
                    except Exception as e:
                        self.log_warning(f"synthesize_cli tile {i} failed: {e}")
                    self.advance_progress(task)

            # Phase 3: consume results back in the original deterministic job order
            # (not completion order) so RNG draws for size/jitter stay reproducible.
            for job, synth in zip(jobs, results):
                if not synth or not synth.get("output_points"):
                    continue
                dist = groups[job["group_idx"]][2]
                placed = group_placed[job["group_idx"]]
                for x, z in synth["output_points"]:
                    # Draw ONE exemplar and take both its footprint and its visual
                    # variant, rather than sampling them independently: bucket is an
                    # appearance class and size correlates with it (a bucket of small
                    # buds and one of tall stalks are different plants), so mixing a
                    # tall exemplar's height with a small exemplar's variant produces
                    # instances that match neither. Also inherits the observed mix of
                    # variants for free -- a region that was 80% bucket 0 paints ~80%
                    # bucket 0.
                    e = int(rng.integers(len(dist.sizes)))
                    w, h = dist.sizes[e]
                    bucket = dist.buckets[e] if e < len(dist.buckets) else 0
                    jitter = 1.0 + float(rng.uniform(-cfg.size_jitter, cfg.size_jitter))
                    placed.append((float(x), float(z), w * jitter, h * jitter, int(bucket)))

            self.finish_progress(task)

        for group_idx, (region_type, obj_type, dist) in enumerate(groups):
            placed = group_placed[group_idx]
            for x, z, w, h, bucket in placed:
                context.add_object(f"metadata_{next_idx}", {
                    "class": obj_type,
                    "synthetic": True,
                    "world_position": [x, 0.0, z],
                    "world_width": w,
                    "world_height": h,
                    # Carries ObjectCategoryClusteringStage's visual variant through to
                    # SceneGenerationStage, which resolves it to category_mesh_{cls}_
                    # {bucket} / the "{cls}::{bucket}" billboard pool. Without it that
                    # lookup fell back to `metadata.get("bucket") or 0`, so every
                    # painted instance in the scene rendered as variant 0.
                    "bucket": int(bucket),
                })
                next_idx += 1

            self.log_info(f"  {obj_type} [{region_type}]: painted {len(placed)} instances")
            if self.temp is not None:
                self._write_debug_image(region_map, grid_size_meters, dist.points,
                                         [(p[0], p[1]) for p in placed],
                                         region_type, obj_type)

        context.add_object(ContextKey.OBJECT_COUNT, next_idx)
        context.add_object(_RAN_MARKER, True)
        self.log_info(f"Object count {object_count} -> {next_idx} after painting")
        return context

    def _write_debug_image(
        self,
        region_map: np.ndarray,
        grid_size_meters: float,
        exemplar_points: list[tuple[float, float]],
        synthesized_points: list[tuple[float, float]],
        region_type: str,
        obj_type: str,
    ):
        debug_res = 1024
        rm_small = cv2.resize(
            region_map.astype(np.uint8), (debug_res, debug_res), interpolation=cv2.INTER_NEAREST
        )
        img = PILImage.fromarray(colorize_region_type_map(rm_small)).convert("RGB")
        draw = ImageDraw.Draw(img)
        half = grid_size_meters / 2.0

        def to_px(x: float, z: float) -> tuple[float, float]:
            return (x + half) / grid_size_meters * debug_res, (z + half) / grid_size_meters * debug_res

        for x, z in synthesized_points:
            px, py = to_px(x, z)
            draw.ellipse([px - 2, py - 2, px + 2, py + 2], fill=(255, 255, 0))
        for x, z in exemplar_points:
            px, py = to_px(x, z)
            draw.ellipse([px - 4, py - 4, px + 4, py + 4], fill=(255, 0, 0))

        img.save(self.temp / f"synthesis_{region_type}_{obj_type}.png")

    def has_expected_output(self, context: PipelineContext) -> bool:
        return context.object(_RAN_MARKER) is True
