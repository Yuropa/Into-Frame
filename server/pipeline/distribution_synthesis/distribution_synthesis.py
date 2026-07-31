import json
import math
import os
import subprocess
from collections import Counter
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
from util.json_utils import parse_json_from_stream
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


def _polygon_area(polygon: np.ndarray) -> float:
    """Shoelace area of a simple polygon, in the polygon's own (world, metres) units."""
    x, z = np.asarray(polygon, dtype=np.float64).T
    return float(abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) / 2.0)


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
    timeout_s: float,
) -> tuple[dict | None, str | None]:
    """Run one tile through synthesize_cli. Returns (result, failure_reason) -- the
    reason is None only when the tile actually produced points, so the caller can
    report WHY a group painted nothing instead of silently reporting that it painted
    nothing (which is indistinguishable from "the data said so").

    Crucially that includes a SUCCESSFUL exit that produced no points. synthesize_cli
    returns 0 and prints a well-formed {"output_points":[]} for every one of
    synthesize_pattern's four bail-outs -- too few exemplars inside the input
    boundary, too few candidate positions in the output boundary, exemplar PCF
    failure, initial placement failure -- and names which one on STDERR. Judging the
    call by its exit code alone throws that diagnosis away and makes a total
    synthesis failure look exactly like an empty distribution."""
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
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return None, (
            f"timed out after {timeout_s:.0f}s with "
            f"{len(domain_points):,} candidates x {max_iters} iters"
        )
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"

    diagnostics = (result.stderr or "").strip().splitlines()
    last_stderr = diagnostics[-1].strip() if diagnostics else ""

    if result.returncode != 0:
        return None, f"exit {result.returncode}: {last_stderr or 'no stderr'}"

    parsed = parse_json_from_stream(result.stdout)
    if parsed is None:
        head = " | ".join(result.stdout.strip().splitlines()[:2])[:300]
        return None, f"no JSON object in stdout: {head or '(empty)'}"

    if not parsed.get("output_points"):
        # Clean exit, empty result -- the interesting case. synthesize_pattern already
        # said why on stderr; pass that through verbatim rather than inventing a
        # summary, since the four reasons call for completely different responses
        # (boundary geometry vs. candidate density vs. exemplar quality).
        return None, last_stderr or "exited 0 but produced no points (no stderr)"
    return parsed, None


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
        max_instances_per_group: int = 12000,
        min_exemplar_spacing_m: float = 0.25,
        full_density_radius_m: float = 6.0,
        density_falloff_exponent: float = 2.0,
        max_paint_radius_m: float = 30.0,
        max_exemplar_candidates: int = 4000,
        synthesize_cli_path: str | None = None,
        max_workers: int | None = None,
        cli_timeout_s: float = 120.0,
    ):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        self.min_blob_area_cells = min_blob_area_cells
        self.max_candidates_per_tile = max_candidates_per_tile
        self.max_iters = max_iters
        self.size_jitter = size_jitter
        self.input_boundary_pad_factor = input_boundary_pad_factor
        # Hard ceiling on painted instances per (object_type, region_type) group,
        # and the last thing applied. Each tile is asked for its own share of the
        # learned density (exemplar count / exemplar hull area, times tile area), so
        # the total still scales with painted AREA over an output boundary covering
        # most of the terrain grid. The binding constraint on the far side is the
        # client: SceneObjectManager
        # instantiates one GameObject per placed object with no GPU instancing, so
        # this is a GameObject budget, not a triangle budget.
        self.max_instances_per_group = max_instances_per_group
        # Floor on the exemplar nearest-neighbour spacing that sets painted density.
        # Spacing is measured on world XZ unprojected from panorama bboxes, so a
        # clump of foreground subjects a metre from the camera (meadow flowers being
        # the observed case) measures centimetres apart and asks for ~1/spacing^2
        # instances per square metre. This floors the density the pattern may claim;
        # it does not move the exemplars. Kept low deliberately -- a meadow has to
        # read as continuous ground cover near the viewer, and the radial falloff
        # below, not this, is what keeps the total affordable.
        self.min_exemplar_spacing_m = min_exemplar_spacing_m
        # Radial density falloff about the camera. A uniform cap spends the whole
        # budget spreading thin across the entire grid -- 6000 instances over a
        # 100 m grid is 0.6/m², which reads as scattered dots rather than a field,
        # while the near ground the viewer actually looks at stays bare. Instead,
        # keep the learned density outright inside full_density_radius_m and thin
        # beyond it by (full_density_radius / distance) ** density_falloff_exponent,
        # so the budget concentrates where it resolves. Past max_paint_radius_m
        # nothing is painted at all: at that range the terrain texture and the
        # skybox panorama already carry the look, and individual billboards are
        # sub-pixel. Camera is the grid origin by construction (see _grid_to_world).
        self.full_density_radius_m = full_density_radius_m
        self.density_falloff_exponent = density_falloff_exponent
        self.max_paint_radius_m = max_paint_radius_m
        # Ceiling on the exemplar candidate grid, enforced by RAISING the effective
        # spacing rather than by clamping the count -- synthesize_pattern infers the
        # output count from the input-vs-output candidate density ratio, so the two
        # grids must stay at equal points-per-area (both are 1/spacing^2) or the
        # inference silently skews. Clamping only the exemplar grid would break that.
        #
        # exemplar_domain_target is input_hull_area / spacing^2, which for points
        # roughly filling their own hull is just ~the exemplar count -- harmless.
        # It blows up for CLUMPED groups, where the hull spans the empty gaps
        # between clusters while spacing is measured within them: three patches of
        # 20 flowers 40 m apart gives ~1600 m^2 / 0.25^2 = 25,600 candidates. The
        # optimizer is O(candidates^2) per iteration over max_iters, so that tile
        # hits the 120 s subprocess timeout, _run_synthesize_cli returns None, and
        # the group silently paints nothing -- the exact failure mode that looks
        # like "the distribution just didn't run" rather than like an error.
        self.max_exemplar_candidates = max_exemplar_candidates
        self.synthesize_cli_path = synthesize_cli_path
        # Each tile's synthesize_cli call is an independent, CPU-bound subprocess (no
        # shared state), so tiles run concurrently in a thread pool -- subprocess.run
        # releases the GIL while the child process runs, so this gets real OS-level
        # parallelism across cores despite being threads, not processes. None -> os.cpu_count().
        self.max_workers = max_workers
        # Per-tile wall-clock budget for synthesize_cli. The optimizer is
        # O(candidates^2) per iteration over max_iters, and a tile's candidate count
        # is max_exemplar_candidates + max_candidates_per_tile, so raising either of
        # those without raising this just converts painted instances into timeouts.
        self.cli_timeout_s = float(cli_timeout_s)


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
            max_instances_per_group   (int, default 12000) — ceiling on painted instances
                                       per (object_type, region_type); 0 disables
            min_exemplar_spacing_m    (float, default 0.25) — floor on the exemplar
                                       nearest-neighbour spacing that sets painted density
            full_density_radius_m     (float, default 6.0) — paint at the learned density
                                       out to here, thinning beyond; 0 disables falloff
            density_falloff_exponent  (float, default 2.0) — thinning exponent past that
                                       radius, as (radius / distance) ** exponent
            max_paint_radius_m        (float, default 30.0) — nothing painted past here;
                                       0 disables
            max_exemplar_candidates   (int, default 4000) — ceiling on the exemplar
                                       candidate grid, enforced by raising the effective
                                       spacing so both candidate grids stay at equal
                                       points-per-area; 0 disables
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
            # See min_exemplar_spacing_m -- this floor is what stops a foreground
            # clump's centimetre-scale measured spacing from setting the density
            # for the entire region. It feeds exemplar_domain_target (hence the
            # inferred output count) and the tile size, so raising it thins the
            # painted population directly rather than truncating it after the fact.
            spacing = max(_mean_nn_spacing(dist.points), cfg.min_exemplar_spacing_m, 0.1)
            input_boundary = _padded_hull_polygon(
                dist.points, pad=max(spacing * cfg.input_boundary_pad_factor, _MIN_PAD_M)
            )

            # Raise spacing if this group's hull would otherwise demand more exemplar
            # candidates than the optimizer can chew through inside its timeout -- see
            # max_exemplar_candidates. Recomputed against the hull built above; the
            # boundary padding shifts only marginally with spacing, so it isn't worth
            # iterating to a fixed point.
            if cfg.max_exemplar_candidates > 0:
                hull_area = float(np.prod(np.maximum(
                    input_boundary.max(axis=0) - input_boundary.min(axis=0), 1e-6
                )))
                budget_spacing = math.sqrt(hull_area / cfg.max_exemplar_candidates)
                if budget_spacing > spacing:
                    self.log_info(
                        f"  {obj_type} [{region_type}]: clumped exemplars over a "
                        f"{hull_area:.0f} m² hull would need "
                        f"{int(hull_area / spacing ** 2):,} candidates; raising spacing "
                        f"{spacing:.2f} m -> {budget_spacing:.2f} m to stay inside the "
                        f"optimizer budget"
                    )
                    spacing = budget_spacing
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

            # Instances per square metre, measured directly in world units from this
            # group's own exemplars. Used below to ask synthesize_cli for an explicit
            # per-tile count instead of passing n_points=-1 and letting it infer one
            # from the ratio of candidate positions inside the input vs. output
            # boundary.
            #
            # That proxy is not stable enough to rely on. Both candidate sets come from
            # Delaunay triangle centres of ONE point set (exemplar_domain + this tile's
            # own grid), so how many centres land inside the input boundary depends on
            # where the tile happens to be -- the triangulation spans the gap between
            # the two grids with long thin triangles whose centres fall wherever the
            # geometry puts them. Measured on the Rainier capture, three adjacent tiles
            # of the SAME group reported input_support 349, 4220 and 4220, and asked
            # for 290, 4 and 7 points respectively. The same instability is what made
            # the near-field flower tiles request thousands of points each and blow
            # through the subprocess timeout.
            #
            # Exemplar count over exemplar hull area has none of that dependence: it is
            # a property of the distribution alone, identical for every tile.
            exemplar_density = len(dist.points) / max(_polygon_area(input_boundary), 1e-6)

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

                        # Drop tiles wholly beyond the paint radius before paying for
                        # their synthesize_cli call -- on a 100 m grid these are the
                        # majority of tiles, and every point they'd produce would be
                        # discarded by the falloff below anyway. Nearest corner of the
                        # tile's world-space bbox to the camera at the grid origin.
                        closest = np.minimum(np.maximum(0.0, tile_world_min), tile_world_max)
                        tile_distance = float(np.hypot(*closest))
                        if cfg.max_paint_radius_m > 0 and tile_distance > cfg.max_paint_radius_m:
                            continue

                        pad = max(cell_m, _MIN_PAD_M)
                        tile_domain = _local_grid(
                            tile_world_min - pad, tile_world_max + pad, cfg.max_candidates_per_tile
                        )
                        domain_points = np.concatenate([exemplar_domain, tile_domain], axis=0)

                        # This tile's share of the learned density, in world units, ALREADY
                        # discounted by the radial falloff.
                        #
                        # The falloff used to be applied only as per-point rejection after
                        # synthesis, so a tile 25 m out paid the optimizer's full price to
                        # place points of which ~94% were then thrown away. That wasted
                        # work is most of what pushed tiles past the subprocess timeout.
                        # Discounting the REQUEST costs nothing in fidelity as long as the
                        # rejection below is made relative to the same factor (it is).
                        #
                        # Evaluated at the tile's NEAREST point to the camera, so
                        # tile_keep is the largest falloff value anywhere in the tile and
                        # every per-point residual below stays <= 1 -- no clamping, and the
                        # expected surviving count is identical to the old behaviour.
                        tile_keep = 1.0
                        if cfg.full_density_radius_m > 0 and tile_distance > cfg.full_density_radius_m:
                            tile_keep = (cfg.full_density_radius_m / tile_distance) ** cfg.density_falloff_exponent

                        # Capped at the tile's own candidate count because that is the
                        # most synthesize_pattern can place anyway (it clamps n_points
                        # to the number of output candidates), and because per-iteration
                        # cost scales with it -- an uncapped request is what turned the
                        # dense near-field tiles into 120 s timeouts.
                        tile_points = int(round(
                            exemplar_density * _polygon_area(output_boundary) * tile_keep
                        ))
                        tile_points = max(2, min(tile_points, cfg.max_candidates_per_tile))

                        seed_counter += 1
                        jobs.append({
                            "group_idx": group_idx,
                            "domain_points": domain_points,
                            "exemplar_points": exemplar_pts,
                            "input_boundary": input_boundary,
                            "output_boundary": output_boundary,
                            "bin_count": dist.bin_count,
                            "n_points": tile_points,
                            "tile_keep": tile_keep,
                            "seed": self.seed + seed_counter,
                        })

        # (x, z, width, height, bucket) per painted instance.
        group_placed: list[list[tuple[float, float, float, float, int]]] = [[] for _ in groups]

        # Set when every tile failed -- see the _RAN_MARKER decision at the end of run().
        total_failure = False

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
            failures: list[str] = []
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(
                        _run_synthesize_cli,
                        cli_path,
                        domain_points=job["domain_points"],
                        exemplar_points=job["exemplar_points"],
                        input_boundary=job["input_boundary"],
                        output_boundary=job["output_boundary"],
                        n_points=job["n_points"],
                        bin_count=job["bin_count"],
                        max_iters=cfg.max_iters,
                        seed=job["seed"],
                        timeout_s=cfg.cli_timeout_s,
                    ): i
                    for i, job in enumerate(jobs)
                }
                for future in as_completed(futures):
                    i = futures[future]
                    try:
                        results[i], reason = future.result()
                    except Exception as e:
                        results[i], reason = None, f"{type(e).__name__}: {e}"
                    if reason is not None:
                        failures.append(reason)
                        # Only the first few, verbatim -- a systematic failure (the
                        # usual case: every tile hits the same bail-out on the same
                        # inputs) produces one identical line per tile otherwise.
                        if len(failures) <= 3:
                            self.log_warning(f"  synthesize_cli tile {i} produced nothing: {reason}")
                    self.advance_progress(task)

            if failures:
                # Loud on purpose. A wholly-failed synthesis is invisible otherwise:
                # the stage completes, writes its debug images, and reports "painted
                # 0 instances" for every group -- which reads as "the distribution
                # had nothing to say" rather than "every single call bailed out".
                # That is exactly how the Rainier capture shipped a scene with no
                # painted population at all.
                by_reason = Counter(failures)
                total_failure = len(failures) == len(jobs)
                level = self.log_error if total_failure else self.log_warning
                level(
                    f"{len(failures)}/{len(jobs)} synthesize_cli tile(s) produced no points"
                    + (" — NOTHING was painted" if total_failure else "")
                )
                for reason, count in by_reason.most_common(4):
                    level(f"    {count}x  {reason}")

            # Phase 3: consume results back in the original deterministic job order
            # (not completion order) so RNG draws for size/jitter stay reproducible.
            for job, synth in zip(jobs, results):
                if not synth or not synth.get("output_points"):
                    continue
                dist = groups[job["group_idx"]][2]
                placed = group_placed[job["group_idx"]]
                for x, z in synth["output_points"]:
                    # Radial density falloff about the camera at the grid origin --
                    # see full_density_radius_m. Still rejection-sampled per point so
                    # the surviving set keeps the synthesized pattern's local structure
                    # (a thinned blue-noise field is still blue noise) and the falloff
                    # stays smooth rather than quantised to tile granularity.
                    #
                    # The tile's REQUEST was already discounted by tile_keep (the
                    # falloff at the tile's nearest point), so what's left to apply here
                    # is only the residual within the tile. Dividing by tile_keep is
                    # what keeps the two consistent: the product of the two stages is
                    # exactly the per-point falloff, as before, but the optimizer no
                    # longer places points that were always going to be discarded.
                    if cfg.max_paint_radius_m > 0 or cfg.full_density_radius_m > 0:
                        distance = float(math.hypot(x, z))
                        if cfg.max_paint_radius_m > 0 and distance > cfg.max_paint_radius_m:
                            continue
                        if cfg.full_density_radius_m > 0 and distance > cfg.full_density_radius_m:
                            keep_prob = (cfg.full_density_radius_m / distance) ** cfg.density_falloff_exponent
                            # <= 1 by construction: tile_keep is the falloff at the
                            # closest point of this tile, so no point in it can have a
                            # larger one.
                            if rng.random() >= keep_prob / job["tile_keep"]:
                                continue
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
            # Ceiling on the population, applied after synthesis so the thinned
            # result still spans the whole region (see max_instances_per_group).
            # Uniform random decimation preserves the pattern's shape -- dropping a
            # fixed fraction independently of position leaves the pair-correlation
            # unchanged up to the density scale factor -- whereas keeping the first
            # N would keep whichever tiles happened to be enumerated first and leave
            # the rest of the region empty.
            if cfg.max_instances_per_group > 0 and len(placed) > cfg.max_instances_per_group:
                keep = rng.choice(len(placed), size=cfg.max_instances_per_group, replace=False)
                self.log_info(
                    f"  {obj_type} [{region_type}]: {len(placed)} painted exceeds "
                    f"max_instances_per_group={cfg.max_instances_per_group}, thinning"
                )
                placed = [placed[i] for i in sorted(keep.tolist())]
                group_placed[group_idx] = placed
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
        # Deliberately NOT marked complete when every tile failed. The marker is a
        # cache key -- has_expected_output() returns True on it, so writing it here
        # tells every later run "this stage is done" and skips it. That is right for
        # the no-op paths above (no distribution, no region map, no binary, no
        # non-singleton groups): they are stable properties of the input, rerunning
        # cannot change them, and leaving the marker off would rerun this stage and
        # its whole downstream cascade on every single invocation for nothing.
        #
        # A total tile failure is the opposite: it is an environment fault (on the
        # Rainier capture, every one of 93 synthesize_cli calls failed instantly while
        # the same 93 tiles synthesize 4,844 points fine against a working binary), and
        # it is FIXABLE. Caching it as success is what let that capture ship a scene
        # with no painted population and then keep shipping it, because the rerun that
        # would have picked up a repaired binary skipped the stage instead. Retrying a
        # genuinely broken stage on every run is the cheap failure here; permanently
        # remembering a failure as a success is not.
        if total_failure:
            self.log_error(
                "Not marking Distribution Synthesis complete: every synthesize_cli tile "
                "failed, so this stage will retry on the next run rather than cache the "
                "failure. Check that pattern-synthesis/build/synthesize_cli exists and "
                "runs (scripts/setup.sh builds it; scripts/debug-synthesize-tile.py "
                "--run replays a captured tile through it)."
            )
        else:
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
