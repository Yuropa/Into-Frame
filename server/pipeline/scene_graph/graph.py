"""The semantic scene graph (docs/agentic-scene-graph-plan.md §2, §13).

One JSON file per run, `scene_graph.json`, plus an append-only edit log
`scene_graph.log.jsonl` beside it. Nodes have a closed structural TYPE (what tools are
written against), an open KIND (what the plan or a detector called it), attributes that
each carry their evidence, and relations between nodes. Nothing here touches the GPU or
the pipeline context; `export.py` builds a graph from a cached run, the agent edits it,
and the runner (server/agent/runner.py) turns edits into stage reruns.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1

# Evidence classes, strongest first. `generated` marks what an expansion tool or a video
# clip supplied: a hypothesis that is never graded against the photograph.
EVIDENCE = ("measured", "plan-visible", "fitted", "plan-inferred", "generated", "default")

# Structural types: the closed set tools consume (§13). Kinds are open strings.
NODE_TYPES = ("World", "Style", "Camera", "Light", "Medium", "Surface", "Mass", "Linear",
              "Thing", "Population", "Asset", "Backdrop")

# Affordances: the closed set the client maps to physics and behaviour.
AFFORDANCES = ("walkable", "collides", "floats", "climbable", "boardable", "emits_light",
               "casts_shadow", "animated", "transparent")

RELATIONS = ("on", "in", "beside", "behind", "bounded_by", "lines", "moored_to",
             "excluded_from", "along", "lit_by", "made_of")

GRAPH_FILE = "scene_graph.json"
LOG_FILE = "scene_graph.log.jsonl"


@dataclass
class Attr:
    value: Any
    unit: str = ""
    evidence: str = "default"
    confidence: float = 1.0
    source: str = ""            # stage or agent round that wrote it

    def to_json(self) -> dict:
        d = {"value": self.value, "evidence": self.evidence}
        if self.unit:
            d["unit"] = self.unit
        if self.confidence != 1.0:
            d["confidence"] = round(float(self.confidence), 3)
        if self.source:
            d["source"] = self.source
        return d

    @classmethod
    def from_json(cls, d: Any) -> "Attr":
        if not isinstance(d, dict) or "value" not in d:
            return cls(d)
        return cls(d["value"], d.get("unit", ""), d.get("evidence", "default"),
                   float(d.get("confidence", 1.0)), d.get("source", ""))


@dataclass
class Node:
    id: str
    type: str
    kind: str = ""
    name: str = ""
    attrs: dict[str, Attr] = field(default_factory=dict)
    geometry: dict[str, Any] = field(default_factory=dict)   # context keys, file names, mask refs
    affordances: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)            # critique annotations, one line each

    def get(self, name: str, default: Any = None) -> Any:
        a = self.attrs.get(name)
        return default if a is None else a.value

    def set(self, name: str, value: Any, evidence: str = "default", unit: str = "",
            confidence: float = 1.0, source: str = "") -> Attr:
        if evidence not in EVIDENCE:
            raise ValueError(f"unknown evidence {evidence!r} (one of {EVIDENCE})")
        a = Attr(value, unit, evidence, confidence, source)
        self.attrs[name] = a
        return a

    def to_json(self) -> dict:
        d: dict[str, Any] = {"id": self.id, "type": self.type}
        if self.kind:
            d["kind"] = self.kind
        if self.name:
            d["name"] = self.name
        d["attrs"] = {k: v.to_json() for k, v in self.attrs.items()}
        if self.geometry:
            d["geometry"] = self.geometry
        if self.affordances:
            d["affordances"] = self.affordances
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Node":
        return cls(d["id"], d["type"], d.get("kind", ""), d.get("name", ""),
                   {k: Attr.from_json(v) for k, v in d.get("attrs", {}).items()},
                   dict(d.get("geometry", {})), list(d.get("affordances", [])), list(d.get("notes", [])))


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    evidence: str = "default"
    source: str = ""

    def to_json(self) -> dict:
        d = {"s": self.subject, "p": self.predicate, "o": self.object, "evidence": self.evidence}
        if self.source:
            d["source"] = self.source
        return d

    @classmethod
    def from_json(cls, d: dict) -> "Relation":
        return cls(d["s"], d["p"], d["o"], d.get("evidence", "default"), d.get("source", ""))


class SceneGraph:
    def __init__(self, run_id: str = "", photo_sha256: str = ""):
        self.schema_version = SCHEMA_VERSION
        self.run_id = run_id
        self.photo_sha256 = photo_sha256
        self.created = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.nodes: dict[str, Node] = {}
        self.relations: list[Relation] = []
        self.meta: dict[str, Any] = {}      # exporter notes: stages seen, counts, warnings
        self._log_path: Optional[Path] = None

    # ── nodes and relations ───────────────────────────────────────────────────
    def add(self, node: Node) -> Node:
        if node.type not in NODE_TYPES:
            raise ValueError(f"unknown node type {node.type!r} (one of {NODE_TYPES})")
        if node.id in self.nodes:
            raise ValueError(f"duplicate node id {node.id!r}")
        self.nodes[node.id] = node
        return node

    def node(self, type: str, id: str, kind: str = "", name: str = "", **attrs) -> Node:
        """Create and add a node; keyword attrs become default-evidence attributes."""
        n = Node(id, type, kind, name or kind or id)
        for k, v in attrs.items():
            n.set(k, v)
        return self.add(n)

    def get(self, node_id: str) -> Node:
        return self.nodes[node_id]

    def find(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def by_type(self, *types: str) -> list[Node]:
        return [n for n in self.nodes.values() if n.type in types]

    def relate(self, subject: str, predicate: str, object: str, evidence: str = "default",
               source: str = "") -> Relation:
        if predicate not in RELATIONS:
            raise ValueError(f"unknown relation {predicate!r} (one of {RELATIONS})")
        for n in (subject, object):
            if n not in self.nodes:
                raise KeyError(f"relation {subject} {predicate} {object}: no node {n!r}")
        r = Relation(subject, predicate, object, evidence, source)
        self.relations.append(r)
        return r

    def related(self, node_id: str, predicate: Optional[str] = None) -> list[Relation]:
        return [r for r in self.relations
                if (r.subject == node_id or r.object == node_id) and (predicate is None or r.predicate == predicate)]

    # ── edits (the agent's only action) ───────────────────────────────────────
    def edit(self, node_id: str, attr: str, value: Any, who: str, why: str,
             evidence: str = "fitted", unit: Optional[str] = None, confidence: float = 1.0) -> dict:
        """Set one attribute and append the change to the edit log. Returns the log entry."""
        n = self.get(node_id)
        old = n.attrs.get(attr)
        new = n.set(attr, value, evidence, unit if unit is not None else (old.unit if old else ""),
                    confidence, who)
        entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "node": node_id, "attr": attr,
                 "old": None if old is None else old.value, "new": new.value,
                 "evidence": evidence, "who": who, "why": why}
        self._append_log(entry)
        return entry

    def annotate(self, node_id: str, text: str, who: str) -> None:
        n = self.get(node_id)
        n.notes.append(f"[{who}] {text}")
        self._append_log({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "node": node_id,
                          "note": text, "who": who})

    def _append_log(self, entry: dict) -> None:
        """Append to the log and rewrite the graph file, so an edit is never lost."""
        if self._log_path is not None:
            with self._log_path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            self._log_path.with_name(GRAPH_FILE).write_text(json.dumps(self.to_json(), indent=1))

    def log_entries(self) -> list[dict]:
        if self._log_path is None or not self._log_path.exists():
            return []
        return [json.loads(l) for l in self._log_path.read_text().splitlines() if l.strip()]

    # ── files ────────────────────────────────────────────────────────────────
    def to_json(self) -> dict:
        return {"schema_version": self.schema_version, "run_id": self.run_id,
                "photo_sha256": self.photo_sha256, "created": self.created, "meta": self.meta,
                "nodes": [n.to_json() for n in self.nodes.values()],
                "relations": [r.to_json() for r in self.relations]}

    def save(self, path: Path) -> Path:
        """Write scene_graph.json (path may be the file or its directory)."""
        path = Path(path)
        if path.is_dir() or path.suffix != ".json":
            path.mkdir(parents=True, exist_ok=True)
            path = path / GRAPH_FILE
        path.write_text(json.dumps(self.to_json(), indent=1))
        self._log_path = path.with_name(LOG_FILE)
        return path

    @classmethod
    def load(cls, path: Path) -> "SceneGraph":
        path = Path(path)
        if path.is_dir():
            path = path / GRAPH_FILE
        d = json.loads(path.read_text())
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{path}: schema {d.get('schema_version')}, this code reads {SCHEMA_VERSION}")
        g = cls(d.get("run_id", ""), d.get("photo_sha256", ""))
        g.created = d.get("created", g.created)
        g.meta = dict(d.get("meta", {}))
        for nd in d.get("nodes", []):
            g.nodes[nd["id"]] = Node.from_json(nd)
        g.relations = [Relation.from_json(r) for r in d.get("relations", [])]
        g._log_path = path.with_name(LOG_FILE)
        return g

    @classmethod
    def from_run(cls, run_dir: Path) -> Optional["SceneGraph"]:
        p = Path(run_dir) / GRAPH_FILE
        return cls.load(p) if p.exists() else None

    # ── checks and summaries ─────────────────────────────────────────────────
    def validate(self) -> list[str]:
        problems = []
        for n in self.nodes.values():
            if n.type not in NODE_TYPES:
                problems.append(f"{n.id}: type {n.type!r}")
            for k, a in n.attrs.items():
                if a.evidence not in EVIDENCE:
                    problems.append(f"{n.id}.{k}: evidence {a.evidence!r}")
            for aff in n.affordances:
                if aff not in AFFORDANCES:
                    problems.append(f"{n.id}: affordance {aff!r}")
        for r in self.relations:
            if r.predicate not in RELATIONS:
                problems.append(f"{r.subject} {r.predicate} {r.object}: unknown relation")
            for x in (r.subject, r.object):
                if x not in self.nodes:
                    problems.append(f"{r.subject} {r.predicate} {r.object}: no node {x}")
        return problems

    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for n in self.nodes.values():
            c[n.type] = c.get(n.type, 0) + 1
        return c

    def summary(self, max_attrs: int = 14, types: Optional[Iterable[str]] = None,
                with_notes: bool = True) -> str:
        """Compact text for a model prompt: one block per node, evidence in brackets."""
        lines = [f"scene graph {self.run_id or '?'} (schema {self.schema_version}); nodes: "
                 + ", ".join(f"{k} {v}" for k, v in sorted(self.counts().items()))]
        want = set(types) if types else None
        for n in self.nodes.values():
            if want and n.type not in want:
                continue
            head = f"{n.id} <{n.type}"
            if n.kind and n.kind != n.id:
                head += f"/{n.kind}"
            head += ">"
            if n.name and n.name not in (n.kind, n.id):
                head += f" {n.name!r}"
            lines.append(head)
            for i, (k, a) in enumerate(n.attrs.items()):
                if i >= max_attrs:
                    lines.append(f"    ... {len(n.attrs) - max_attrs} more")
                    break
                v = a.value
                if isinstance(v, float):
                    v = f"{v:.3g}"
                elif isinstance(v, (list, dict)):
                    s = json.dumps(v)
                    v = s if len(s) <= 80 else s[:77] + "..."
                lines.append(f"    {k} = {v}{(' ' + a.unit) if a.unit else ''} [{a.evidence}"
                             + (f" {a.confidence:.2f}" if a.confidence != 1.0 else "") + "]")
            rel = self.related(n.id)
            if rel:
                lines.append("    " + "; ".join(f"{r.predicate} {r.object}" if r.subject == n.id
                                                 else f"<-{r.predicate}- {r.subject}" for r in rel[:8]))
            if with_notes and n.notes:
                for t in n.notes[-3:]:
                    lines.append(f"    note: {t}")
        return "\n".join(lines)

    def node_ids(self, *types: str) -> list[str]:
        return [n.id for n in self.nodes.values() if not types or n.type in types]
