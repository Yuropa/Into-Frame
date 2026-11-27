#!/usr/bin/env python3
"""
The semantic scene graph of a cached run (docs/agentic-scene-graph-plan.md §2).

    python scripts/scene-graph.py export <run-dir|Scene> [--out DIR]   # scene_graph.json + .log.jsonl
    python scripts/scene-graph.py show <run-dir|Scene> [--types T,T]   # graph.summary()
    python scripts/scene-graph.py html <run-dir|Scene> [--out FILE]    # the explorable page (graph, map,
                                                                       # photograph, tables, raw documents)
    python scripts/scene-graph.py validate <run-dir|Scene>

A scene name ("Paris", "Mount Rainier") resolves to the output/<uuid> run whose input.png
matches server/samples/<Scene>.jpg, as scripts/hybrid-run.sh does. Everything runs on the
CPU: CUDA is hidden before any import, nothing here touches a model. `export` writes into
the run directory unless --out says otherwise; `show` and `validate` read the written
scene_graph.json when there is one and build the graph in memory when not; `html` always
builds from the run as it stands (the page the Scene Graph stage writes).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before any import that might look for a GPU

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from pipeline.scene_graph.graph import SceneGraph, EVIDENCE, NODE_TYPES, GRAPH_FILE, LOG_FILE  # noqa: E402
from pipeline.scene_graph.export import export_graph  # noqa: E402


# ── run resolution ─────────────────────────────────────────────────────────────

def resolve_run(scene: str) -> Path:
    p = Path(scene)
    if p.is_dir():
        return p.resolve()
    sample = ROOT / "server" / "samples" / f"{scene}.jpg"
    if not sample.exists():
        raise SystemExit(f"{scene!r} is neither a run directory nor a sample photo under server/samples/")
    import numpy as np
    from PIL import Image

    def thumb(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB").resize((32, 32), Image.BOX), dtype=np.float32)

    want = thumb(sample)
    best = None
    for run in (ROOT / "output").glob("*-*-*-*-*"):
        inp = run / "input.png"
        if inp.is_file():
            err = float(np.abs(thumb(inp) - want).mean())
            if err < 6 and (best is None or err < best[0]):
                best = (err, run)
    if best is None:
        raise SystemExit(f"no cached run under output/ matches {scene!r}")
    return best[1].resolve()


def load_or_build(run: Path, config: Path) -> SceneGraph:
    p = run / GRAPH_FILE
    if p.exists():
        return SceneGraph.load(p)
    print(f"({run.name} has no {GRAPH_FILE}; building the graph in memory)", file=sys.stderr)
    return export_graph(run, config)


# ── commands ──────────────────────────────────────────────────────────────────

def cmd_export(args) -> int:
    run = resolve_run(args.run)
    out = Path(args.out).resolve() if args.out else run
    g = export_graph(run, Path(args.config))
    path = g.save(out)
    # the edit log starts empty but present, so an editor can append to it
    log = path.with_name(LOG_FILE)
    if not log.exists():
        log.write_text("")
    problems = g.validate()
    counts = ", ".join(f"{k} {v}" for k, v in sorted(g.counts().items()))
    print(f"{run.name}: {len(g.nodes)} nodes ({counts}), {len(g.relations)} relations -> {path}")
    for w in g.meta.get("warnings", []):
        print(f"  warning: {w}")
    if problems:
        for pr in problems:
            print(f"  INVALID: {pr}")
        return 1
    return 0


def cmd_show(args) -> int:
    run = resolve_run(args.run)
    g = load_or_build(run, Path(args.config))
    types = [t.strip() for t in args.types.split(",")] if args.types else None
    print(g.summary(max_attrs=args.max_attrs, types=types))
    if args.warnings:
        for w in g.meta.get("warnings", []):
            print(f"warning: {w}")
    return 0


def cmd_validate(args) -> int:
    run = resolve_run(args.run)
    g = load_or_build(run, Path(args.config))
    problems = g.validate()
    # the exporter's own contract on top of the structural checks
    ids = set(g.nodes)
    for required in ("world", "style", "camera", "sun", "atmosphere", "ground", "treeline", "far_field", "grass"):
        if required not in ids:
            problems.append(f"required node {required!r} missing")
    for n in g.nodes.values():
        for k, a in n.attrs.items():
            try:
                json.dumps(a.value)
            except TypeError:
                problems.append(f"{n.id}.{k}: value is not JSON ({type(a.value).__name__})")
    if problems:
        for pr in problems:
            print(f"INVALID: {pr}")
        return 1
    counts = ", ".join(f"{k} {v}" for k, v in sorted(g.counts().items()))
    print(f"{run.name}: valid; {len(g.nodes)} nodes ({counts}), {len(g.relations)} relations, "
          f"{len(g.meta.get('warnings', []))} warnings")
    return 0


# ── html page ─────────────────────────────────────────────────────────────────

def cmd_html(args) -> int:
    """The explorable page the Scene Graph stage writes (pipeline/scene_graph/view.py)."""
    from pipeline.scene_graph import view
    run = resolve_run(args.run)
    doc = view.build_view(run, Path(args.config), title=args.title or (None if Path(args.run).is_dir() else args.run))
    out = Path(args.out) if args.out else run / "Scene Graph" / "scene_graph.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    # --artifact: the page without its document skeleton, as a claude.ai Artifact is published
    out.write_text(view.render_html(doc, standalone=not args.artifact), encoding="utf-8")
    if args.physical:
        Path(args.physical).write_text(json.dumps(view.physical_table(doc), indent=1), encoding="utf-8")
    g = doc["graph"]
    print(f"{out}  ({len(g['nodes'])} nodes, {len(g['relations'])} relations, {len(doc['detections'])} detections, "
          f"{len(doc['models'])} model files, {len(doc['placements']['rows'])} placements, {out.stat().st_size / 1e6:.1f} MB)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "server" / "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("export", help="build scene_graph.json (+ an empty edit log) from a cached run")
    p.add_argument("run"); p.add_argument("--out", default=None, help="directory (default: the run dir)")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("show", help="print graph.summary()")
    p.add_argument("run"); p.add_argument("--types", default=None, help="comma-separated node types")
    p.add_argument("--max-attrs", type=int, default=14); p.add_argument("--warnings", action="store_true")
    p.set_defaults(fn=cmd_show)
    p = sub.add_parser("html", help="write the explorable page of the whole representation")
    p.add_argument("run"); p.add_argument("--out", default=None, help="file (default: <run>/Scene Graph/scene_graph.html)")
    p.add_argument("--title", default=None); p.add_argument("--physical", default=None, help="also write physical_properties.json here")
    p.add_argument("--artifact", action="store_true", help="write the page body only (for publishing as an Artifact)")
    p.set_defaults(fn=cmd_html)
    p = sub.add_parser("validate", help="structural checks plus the exporter's contract")
    p.add_argument("run")
    p.set_defaults(fn=cmd_validate)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
