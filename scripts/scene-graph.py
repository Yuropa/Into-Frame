#!/usr/bin/env python3
"""
The semantic scene graph of a cached run (docs/agentic-scene-graph-plan.md §2).

    python scripts/scene-graph.py export <run-dir|Scene> [--out DIR]   # scene_graph.json + .log.jsonl
    python scripts/scene-graph.py show <run-dir|Scene> [--types T,T]   # graph.summary()
    python scripts/scene-graph.py html <run-dir|Scene> [--out FILE]    # self-contained viewer
    python scripts/scene-graph.py validate <run-dir|Scene>

A scene name ("Paris", "Mount Rainier") resolves to the output/<uuid> run whose input.png
matches server/samples/<Scene>.jpg, as scripts/hybrid-run.sh does. Everything runs on the
CPU: CUDA is hidden before any import, nothing here touches a model. `export` writes into
the run directory unless --out says otherwise; `show`, `html` and `validate` read the
written scene_graph.json when there is one and build the graph in memory when not.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # before any import that might look for a GPU

import argparse
import html
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

from pipeline.scene_graph.graph import SceneGraph, EVIDENCE, NODE_TYPES, GRAPH_FILE, LOG_FILE  # noqa: E402
from pipeline.scene_graph.export import export_graph  # noqa: E402

EVIDENCE_COLOURS = {"measured": "#1b8a4a", "fitted": "#2a6fd6", "plan-visible": "#0f8f8f",
                    "plan-inferred": "#c47f00", "generated": "#8a3fc9", "default": "#7a7a7a"}


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


# ── html viewer ───────────────────────────────────────────────────────────────

def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (list, dict)):
        s = json.dumps(v)
        return s if len(s) <= 160 else s[:157] + "..."
    return str(v)


def render_html(g: SceneGraph, title: str) -> str:
    esc = html.escape
    css_ev = "\n".join(f".ev-{k.replace('-', '_')} {{ color: {c}; }}" for k, c in EVIDENCE_COLOURS.items())
    parts = [f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d1d1f; --muted: #6b6b70; --line: #e3e3e8; --head: #f4f4f7; --code: #f0f0f4; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg: #141417; --fg: #e8e8ec; --muted: #9a9aa2; --line: #2c2c33; --head: #1e1e24; --code: #1a1a20; }} }}
:root[data-theme="dark"] {{ --bg: #141417; --fg: #e8e8ec; --muted: #9a9aa2; --line: #2c2c33; --head: #1e1e24; --code: #1a1a20; }}
body {{ margin: 0; padding: 24px 16px 48px; background: var(--bg); color: var(--fg); font: 14px/1.45 -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
main {{ max-width: 1200px; margin: 0 auto; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }} h2 {{ font-size: 17px; margin: 32px 0 8px; }} h3 {{ font-size: 15px; margin: 20px 0 6px; }}
.meta, .muted {{ color: var(--muted); }}
.legend span {{ display: inline-block; margin: 0 10px 0 0; font-weight: 600; }}
table {{ border-collapse: collapse; width: 100%; margin: 6px 0 14px; font-size: 13px; }}
th, td {{ text-align: left; vertical-align: top; padding: 4px 8px; border-bottom: 1px solid var(--line); }}
th {{ background: var(--head); position: sticky; top: 0; }}
td.v {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; word-break: break-word; }}
td.k {{ white-space: nowrap; }}
.node {{ margin: 0 0 18px; }}
.node .head {{ font-weight: 600; }} .node .head .id {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}
.tag {{ display: inline-block; padding: 0 6px; border: 1px solid var(--line); border-radius: 10px; font-size: 11px; margin-left: 6px; color: var(--muted); }}
.rels {{ font-size: 12px; color: var(--muted); margin: 2px 0 6px; }}
.ev {{ font-weight: 600; white-space: nowrap; }}
pre {{ background: var(--code); padding: 8px; overflow-x: auto; font-size: 12px; }}
details > summary {{ cursor: pointer; }}
nav a {{ margin-right: 12px; }}
{css_ev}
</style></head><body><main>
<h1>{esc(title)}</h1>
<div class="meta">run {esc(g.run_id or '?')} · schema {g.schema_version} · created {esc(g.created)} · exported {esc(str(g.meta.get('exported', '?')))}
 · {len(g.nodes)} nodes, {len(g.relations)} relations</div>
<p class="legend">evidence: """ + " ".join(f'<span class="ev-{k.replace("-", "_")}">{k}</span>' for k in EVIDENCE) + "</p>"]
    present = [t for t in NODE_TYPES if any(n.type == t for n in g.nodes.values())]
    parts.append("<nav>" + " ".join(f'<a href="#type-{t}">{t} ({sum(1 for n in g.nodes.values() if n.type == t)})</a>' for t in present) + "</nav>")
    for t in present:
        parts.append(f'<h2 id="type-{t}">{t}</h2>')
        for n in g.nodes.values():
            if n.type != t:
                continue
            head = f'<span class="id">{esc(n.id)}</span>'
            if n.kind and n.kind != n.id:
                head += f' <span class="muted">/ {esc(n.kind)}</span>'
            if n.name and n.name not in (n.kind, n.id):
                head += f" &nbsp;{esc(n.name)}"
            for a in n.affordances:
                head += f'<span class="tag">{esc(a)}</span>'
            parts.append(f'<div class="node" id="node-{esc(n.id)}"><div class="head">{head}</div>')
            rels = g.related(n.id)
            if rels:
                parts.append('<div class="rels">' + "; ".join(
                    (f'{esc(r.predicate)} <a href="#node-{esc(r.object)}">{esc(r.object)}</a>' if r.subject == n.id
                     else f'<a href="#node-{esc(r.subject)}">{esc(r.subject)}</a> {esc(r.predicate)} this') for r in rels) + "</div>")
            if n.attrs:
                parts.append("<table><thead><tr><th>attribute</th><th>value</th><th>unit</th><th>evidence</th><th>source</th></tr></thead><tbody>")
                for k, a in n.attrs.items():
                    ev = esc(a.evidence)
                    conf = f" {a.confidence:.2f}" if a.confidence != 1.0 else ""
                    parts.append(f'<tr><td class="k">{esc(k)}</td><td class="v">{esc(_fmt(a.value))}</td><td>{esc(a.unit)}</td>'
                                 f'<td class="ev ev-{ev.replace("-", "_")}">{ev}{conf}</td><td class="muted">{esc(a.source)}</td></tr>')
                parts.append("</tbody></table>")
            if n.geometry:
                parts.append(f'<div class="muted">geometry: <code>{esc(_fmt(n.geometry))}</code></div>')
            for note in n.notes:
                parts.append(f'<div class="muted">note: {esc(note)}</div>')
            parts.append("</div>")
    parts.append("<h2>Relations</h2><table><thead><tr><th>subject</th><th>predicate</th><th>object</th><th>evidence</th><th>source</th></tr></thead><tbody>")
    for r in g.relations:
        ev = esc(r.evidence)
        parts.append(f'<tr><td><a href="#node-{esc(r.subject)}">{esc(r.subject)}</a></td><td>{esc(r.predicate)}</td>'
                     f'<td><a href="#node-{esc(r.object)}">{esc(r.object)}</a></td><td class="ev ev-{ev.replace("-", "_")}">{ev}</td><td class="muted">{esc(r.source)}</td></tr>')
    parts.append("</tbody></table>")
    warnings = g.meta.get("warnings", [])
    parts.append(f"<h2>Export notes</h2><details><summary>{len(warnings)} warnings, meta</summary><pre>"
                 + esc(json.dumps({k: v for k, v in g.meta.items() if k != 'stages'}, indent=1)) + "</pre>"
                 + "<div class='muted'>stages: " + esc(", ".join(g.meta.get("stages", []))) + "</div></details>")
    entries = g.log_entries()
    parts.append(f"<h2>Edit log</h2>")
    if not entries:
        parts.append('<p class="muted">no edits yet</p>')
    else:
        parts.append("<table><thead><tr><th>time</th><th>node</th><th>attr</th><th>old</th><th>new</th><th>evidence</th><th>who</th><th>why / note</th></tr></thead><tbody>")
        for e in entries:
            parts.append("<tr>" + "".join(f"<td class='v'>{esc(_fmt(e.get(k, '')))}</td>" for k in
                                          ("time", "node", "attr", "old", "new", "evidence", "who")) +
                         f"<td>{esc(str(e.get('why') or e.get('note') or ''))}</td></tr>")
        parts.append("</tbody></table>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def cmd_html(args) -> int:
    run = resolve_run(args.run)
    g = load_or_build(run, Path(args.config))
    title = f"Scene graph {args.run if not Path(args.run).is_dir() else run.name}"
    out = Path(args.out) if args.out else run / "scene_graph.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(g, title))
    print(out)
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
    p = sub.add_parser("html", help="write a self-contained HTML viewer")
    p.add_argument("run"); p.add_argument("--out", default=None, help="file (default: <run>/scene_graph.html)")
    p.set_defaults(fn=cmd_html)
    p = sub.add_parser("validate", help="structural checks plus the exporter's contract")
    p.add_argument("run")
    p.set_defaults(fn=cmd_validate)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
