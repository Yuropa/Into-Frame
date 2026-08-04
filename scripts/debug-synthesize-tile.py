#!/usr/bin/env python3
"""
Rebuild the exact stdin DistributionSynthesisStage feeds synthesize_cli for one tile
of a captured run, so a silent "painted 0 instances" can be reproduced and diagnosed
in isolation -- without re-running the pipeline.

    python scripts/debug-synthesize-tile.py <debug-context-dir> [--group ground/flower]
    python scripts/debug-synthesize-tile.py <ctx> --run path/to/synthesize_cli

<debug-context-dir> is the per-sample directory inside a .debug bundle, i.e.
    <name>.debug/context/<uuid>/
It must contain 'Object Distribution/distributions.json', a region map, and
'Height Map/height_map_params.json'.

With --run, each tile is executed and BOTH streams are reported. That matters:
synthesize_cli exits 0 and prints {"output_points":[]} for every one of
synthesize_pattern's bail-outs, naming the actual reason only on stderr.
"""

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

from pipeline.panorama_segmentation.panorama_region_result import (  # noqa: E402
    RegionType,
    paintable_region_types,
)

# Deliberately re-implemented rather than imported from distribution_synthesis.py:
# importing that module pulls in torch (via pipeline_stage), so this diagnostic would
# only run inside a full pipeline environment -- exactly the setup you're trying to
# investigate from the outside. These four are small and must stay in step with the
# originals; they are copied verbatim.
_MIN_EXEMPLAR_DOMAIN = 16


def _grid_to_world(row: float, col: float, grid_size_meters: float, grid_resolution: int):
    half = grid_size_meters / 2.0
    cell_m = grid_size_meters / grid_resolution
    return (col + 0.5) * cell_m - half, (row + 0.5) * cell_m - half


def _mean_nn_spacing(points) -> float:
    from scipy.spatial import KDTree
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 1.0
    dists, _ = KDTree(pts).query(pts, k=2)
    nn = dists[:, 1]
    nn = nn[np.isfinite(nn) & (nn > 0)]
    return float(nn.mean()) if len(nn) > 0 else 1.0


def _padded_bbox_polygon(pts: np.ndarray, pad: float) -> np.ndarray:
    lo = pts.min(axis=0) - pad
    hi = pts.max(axis=0) + pad
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])


def _padded_hull_polygon(points, pad: float) -> np.ndarray:
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return _padded_bbox_polygon(pts, pad)
    try:
        hull_pts = pts[ConvexHull(pts).vertices]
    except Exception:
        return _padded_bbox_polygon(pts, pad)
    centroid = hull_pts.mean(axis=0)
    directions = hull_pts - centroid
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return hull_pts + directions / norms * pad


def _polygon_area(polygon: np.ndarray) -> float:
    x, z = np.asarray(polygon, dtype=np.float64).T
    return float(abs(np.dot(x, np.roll(z, -1)) - np.dot(z, np.roll(x, -1))) / 2.0)


def _local_grid(bbox_min: np.ndarray, bbox_max: np.ndarray, target_count: int) -> np.ndarray:
    res = max(2, int(round(math.sqrt(max(target_count, 4)))))
    xs = np.linspace(bbox_min[0], bbox_max[0], res)
    zs = np.linspace(bbox_min[1], bbox_max[1], res)
    X, Z = np.meshgrid(xs, zs)
    return np.stack([X.ravel(), Z.ravel()], axis=1)

# Mirrors server/config.yaml's Distribution Synthesis block. Kept as plain literals
# rather than parsed from the YAML so this script stays runnable against a bundle
# produced by a different config revision than the checkout it's run from.
MIN_BLOB_AREA_CELLS = 64
MAX_CANDIDATES_PER_TILE = 900
MAX_ITERS = 100
INPUT_BOUNDARY_PAD_FACTOR = 1.5
MIN_EXEMPLAR_SPACING_M = 0.25
MAX_PAINT_RADIUS_M = 30.0
MAX_EXEMPLAR_CANDIDATES = 4000
MIN_PAD_M = 0.5
SWEEP_ITERS = (25, 50, 100, 400)
SWEEP_POINT_FRACTIONS = (0.125, 0.25, 0.5, 1.0)
FULL_DENSITY_RADIUS_M = 6.0
DENSITY_FALLOFF_EXPONENT = 2.0
SEED = 0


def load_region_map(ctx: Path) -> np.ndarray:
    # Latest wins, matching how the pipeline context resolves REGION_MAP.
    for stage in ("Region Map Refinement", "Region Map"):
        path = ctx / stage / "region_map.npy"
        if path.exists():
            print(f"region map: {stage}")
            return np.load(path)
    raise SystemExit(f"no region_map.npy under {ctx}")


def build_jobs(ctx: Path, only_group: str | None):
    region_map = load_region_map(ctx)
    grid_resolution = region_map.shape[0]
    grid_size_meters = json.loads(
        (ctx / "Height Map" / "height_map_params.json").read_text()
    ).get("grid_size_meters", 100.0)
    cell_m = grid_size_meters / grid_resolution
    print(f"grid: {grid_resolution}px over {grid_size_meters}m ({cell_m*100:.2f} cm/cell)")

    distributions = json.loads(
        (ctx / "Object Distribution" / "distributions.json").read_text()
    )["distributions"]

    import cv2  # imported late: only this path needs it
    from scipy import ndimage

    jobs = []
    for region_type, by_type in distributions.items():
        for obj_type, dist in by_type.items():
            name = f"{region_type}/{obj_type}"
            if only_group and name != only_group:
                continue
            points = dist["points"]
            if dist["n_points"] < 2 or len(points) < 2:
                print(f"\n[{name}] skipped: singleton distribution")
                continue

            spacing = max(_mean_nn_spacing(points), MIN_EXEMPLAR_SPACING_M, 0.1)
            input_boundary = _padded_hull_polygon(
                points, pad=max(spacing * INPUT_BOUNDARY_PAD_FACTOR, MIN_PAD_M)
            )
            if MAX_EXEMPLAR_CANDIDATES > 0:
                hull_area = float(np.prod(np.maximum(
                    input_boundary.max(axis=0) - input_boundary.min(axis=0), 1e-6)))
                budget_spacing = math.sqrt(hull_area / MAX_EXEMPLAR_CANDIDATES)
                if budget_spacing > spacing:
                    spacing = budget_spacing
                    input_boundary = _padded_hull_polygon(
                        points, pad=max(spacing * INPUT_BOUNDARY_PAD_FACTOR, MIN_PAD_M)
                    )

            input_area = float(np.prod(np.maximum(
                input_boundary.max(axis=0) - input_boundary.min(axis=0), 1e-6)))
            exemplar_domain_target = max(
                _MIN_EXEMPLAR_DOMAIN, int(round(input_area / (spacing ** 2))))
            exemplar_domain = _local_grid(
                input_boundary.min(axis=0), input_boundary.max(axis=0), exemplar_domain_target)

            tile_res = max(4, int(round(math.sqrt(MAX_CANDIDATES_PER_TILE))))
            tile_side_px = max(1, int(round(tile_res * spacing / cell_m)))
            print(f"\n[{name}] exemplars={len(points)} spacing={spacing:.3f}m "
                  f"tile={tile_side_px}px ({tile_side_px*cell_m:.1f}m) "
                  f"exemplar_domain={len(exemplar_domain)}")

            paintable = [int(rt) for rt in paintable_region_types(RegionType.from_label(region_type))]
            mask = np.isin(region_map, paintable)
            labels, n_components = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.int32))

            group_jobs = 0
            for label_id in range(1, n_components + 1):
                blob = labels == label_id
                rows, cols = np.nonzero(blob)
                if len(rows) < MIN_BLOB_AREA_CELLS:
                    continue
                r_min, r_max = int(rows.min()), int(rows.max()) + 1
                c_min, c_max = int(cols.min()), int(cols.max()) + 1
                for tr in range(r_min, r_max, tile_side_px):
                    for tc in range(c_min, c_max, tile_side_px):
                        tile = blob[tr:min(tr + tile_side_px, r_max), tc:min(tc + tile_side_px, c_max)]
                        if int(tile.sum()) < MIN_BLOB_AREA_CELLS:
                            continue
                        contours, _ = cv2.findContours(
                            tile.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if not contours:
                            continue
                        approx = cv2.approxPolyDP(max(contours, key=cv2.contourArea), 1.0, True)
                        if len(approx) < 3:
                            continue
                        output_boundary = np.array(
                            [_grid_to_world(p[0][1] + tr, p[0][0] + tc, grid_size_meters, grid_resolution)
                             for p in approx], dtype=np.float64)
                        lo, hi = output_boundary.min(axis=0), output_boundary.max(axis=0)
                        if MAX_PAINT_RADIUS_M > 0:
                            closest = np.minimum(np.maximum(0.0, lo), hi)
                            if float(np.hypot(*closest)) > MAX_PAINT_RADIUS_M:
                                continue
                        pad = max(cell_m, MIN_PAD_M)
                        tile_domain = _local_grid(lo - pad, hi + pad, MAX_CANDIDATES_PER_TILE)
                        tile_distance = float(np.hypot(*np.minimum(np.maximum(0.0, lo), hi)))
                        tile_keep = 1.0
                        if FULL_DENSITY_RADIUS_M > 0 and tile_distance > FULL_DENSITY_RADIUS_M:
                            tile_keep = (FULL_DENSITY_RADIUS_M / tile_distance) ** DENSITY_FALLOFF_EXPONENT
                        tile_points = int(round(
                            len(points) / max(_polygon_area(input_boundary), 1e-6)
                            * _polygon_area(output_boundary) * tile_keep))
                        jobs.append({
                            "group": name,
                            "domain_points": np.concatenate([exemplar_domain, tile_domain], axis=0),
                            "exemplar_points": np.asarray(points, dtype=np.float64),
                            "input_boundary": input_boundary,
                            "output_boundary": output_boundary,
                            "bin_count": dist["bin_count"],
                            "n_points": max(2, min(tile_points, MAX_CANDIDATES_PER_TILE)),
                            "distance_m": tile_distance,
                        })
                        group_jobs += 1
            print(f"[{name}] {group_jobs} tile(s) inside {MAX_PAINT_RADIUS_M}m")
    return jobs


def to_stdin(job, n_points=None, max_iters=None) -> str:
    n = job["n_points"] if n_points is None else n_points
    iters = MAX_ITERS if max_iters is None else max_iters
    # Folded into the signed 32-bit range exactly as the stage does -- synthesize_cli
    # parses the header with `std::cin >> int`, so a raw 32-bit-unsigned seed used to
    # fail the whole header read.
    lines = [f"{job['bin_count']} {n} {iters} {SEED & 0x7FFFFFFF}"]
    for key in ("domain_points", "exemplar_points", "input_boundary", "output_boundary"):
        pts = job[key]
        lines.append(str(len(pts)))
        lines.extend(f"{x:.6f} {z:.6f}" for x, z in pts)
    return "\n".join(lines) + "\n"


def run_sweep(job, cli: Path, timeout: float):
    """Time one tile across a grid of (n_points, max_iters).

    Separates the two cost terms that get conflated otherwise. Before any iteration
    runs, synthesize_pattern precomputes an O(candidates^2) support table, so every
    cell of a row pays the same fixed floor; the growth ACROSS a row is the iteration
    cost, and the growth DOWN a column is the point count. A row that is flat and
    already over budget means the floor is the problem and only --candidates helps.
    """
    n_full = job["n_points"]
    print(f"    sweep (per-cell timeout {timeout:.0f}s; '>' = hit it)")
    print(f"    {'n_points':>10} | " + " | ".join(f"{it:>5} iters" for it in SWEEP_ITERS))
    for n in [max(2, int(n_full * f)) for f in SWEEP_POINT_FRACTIONS]:
        cells = []
        for iters in SWEEP_ITERS:
            stdin_data = to_stdin(job, n_points=n, max_iters=iters)
            start = time.monotonic()
            try:
                proc = subprocess.run([str(cli)], input=stdin_data, capture_output=True,
                                      text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                cells.append(f">{timeout:.0f}s".rjust(11))
                # Longer iteration budgets on this row can only be slower; skip them.
                cells.extend(["  -".rjust(11)] * (len(SWEEP_ITERS) - len(cells)))
                break
            elapsed = time.monotonic() - start
            placed = ""
            for line in reversed(proc.stdout.strip().splitlines()):
                if line.strip().startswith("{"):
                    try:
                        placed = f"/{json.loads(line).get('n_points')}"
                    except json.JSONDecodeError:
                        pass
                    break
            cells.append(f"{elapsed:6.1f}s{placed}".rjust(11))
        print(f"    {n:>10} | " + " | ".join(cells))


def main():
    global MAX_CANDIDATES_PER_TILE, SEED

    ap = argparse.ArgumentParser()
    ap.add_argument("context_dir", type=Path)
    ap.add_argument("--group", help="only this '<region>/<type>' group, e.g. ground/flower")
    ap.add_argument("--run", type=Path, help="path to synthesize_cli; execute each tile")
    ap.add_argument("--limit", type=int, default=3, help="tiles to run/write (default 3)")
    ap.add_argument("--worst", action="store_true",
                    help="take the tiles with the LARGEST requested point count instead of "
                         "the first ones. Runtime scales with that count, so this is what "
                         "you want when the question is whether tiles fit in the timeout — "
                         "enumeration order has no relationship to cost.")
    ap.add_argument("--out", type=Path, help="directory to write tile stdin files to")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--candidates", type=int,
                    help="override max_candidates_per_tile. The optimizer precomputes an "
                         "O(candidates^2) support table before any iteration runs, so this "
                         "sets a floor on tile cost that no point-count or iteration change "
                         "can get under.")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="tile seed, as the stage would pass it (self.seed + tile index). "
                         "Defaults to 0, which is NOT what a real run uses: the pipeline's "
                         "global seed is random over 0..2**32-1 and is persisted in "
                         "seed.json, so pass the run's own value from there to reproduce a "
                         "seed-dependent failure.")
    ap.add_argument("--sweep", action="store_true",
                    help="run each selected tile across a grid of (n_points, max_iters) and "
                         "report wall time for each, so the cost curve is measured rather "
                         "than guessed. Uses --timeout per cell; keep it short (~30s).")
    args = ap.parse_args()

    SEED = args.seed

    if args.candidates:
        MAX_CANDIDATES_PER_TILE = args.candidates
        print(f"max_candidates_per_tile overridden to {MAX_CANDIDATES_PER_TILE}")

    jobs = build_jobs(args.context_dir, args.group)
    print(f"\n=== {len(jobs)} tile job(s) total, "
          f"{sum(j['n_points'] for j in jobs)} point(s) requested across all of them ===")
    if not jobs:
        print("No tiles built at all — the stage would paint nothing without ever "
              "invoking synthesize_cli. Look at the region map and max_paint_radius_m.")
        return

    selected = list(enumerate(jobs))
    if args.worst:
        selected.sort(key=lambda pair: pair[1]["n_points"], reverse=True)
    selected = selected[:args.limit]

    for i, job in selected:
        stdin_data = to_stdin(job)
        print(f"\n--- tile {i} [{job['group']}] "
              f"domain={len(job['domain_points'])} exemplars={len(job['exemplar_points'])} "
              f"input_boundary={len(job['input_boundary'])} output_boundary={len(job['output_boundary'])} "
              f"dist={job['distance_m']:.1f}m requested_n={job['n_points']}")

        if args.sweep:
            if not args.run:
                raise SystemExit("--sweep needs --run <path to synthesize_cli>")
            run_sweep(job, args.run, args.timeout)
            continue

        if args.out:
            args.out.mkdir(parents=True, exist_ok=True)
            path = args.out / f"tile_{i}.txt"
            path.write_text(stdin_data)
            print(f"    wrote {path}")
        if args.run:
            try:
                proc = subprocess.run([str(args.run)], input=stdin_data, capture_output=True,
                                      text=True, timeout=args.timeout)
            except subprocess.TimeoutExpired:
                print(f"    TIMEOUT after {args.timeout}s")
                continue
            print(f"    exit={proc.returncode}")
            if proc.stderr.strip():
                print(f"    stderr: {proc.stderr.strip()}")
            parsed = None
            for line in reversed(proc.stdout.strip().splitlines()):
                line = line.strip()
                if not line.startswith("{"):
                    print(f"    stdout(noise): {line[:160]}")
                    continue
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                break
            if parsed is None:
                print(f"    NO JSON in stdout: {proc.stdout.strip()[:300]}")
            else:
                print(f"    n_points={parsed.get('n_points')} energy={parsed.get('energy')} "
                      f"iterations={parsed.get('iterations')}")


if __name__ == "__main__":
    main()
