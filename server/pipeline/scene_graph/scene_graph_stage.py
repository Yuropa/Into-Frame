"""Scene Graph: the run's whole representation as one page a person can explore.

Last stage of the first pass. It reads the run the way scripts/scene-graph.py does (the
saved context, never the models), builds the semantic scene graph, adds an Asset node per
placed model file with its measured geometry and its estimated physical properties
(scene_graph/physical.py: class priors, marked `default` until something measures them),
and writes into its own folder:

    scene_graph.html           self-contained page: graph, top-down map, photograph with the
                               detections, tables, and the raw documents (open it in a browser)
    scene_graph.json           the graph the page shows (nodes, attributes with evidence, relations)
    physical_properties.json   per model file: geometry, material, mass, friction, collider

The agent's working graph (<run>/scene_graph.json and its edit log) is not touched: this
stage is a reader. It runs on the CPU in a few seconds and never fails a run: anything
that goes wrong is a warning and the scene passes through.

Input   the saved run (every stage before this one has been written to disk)
Output  files only; the context is unchanged
"""
from __future__ import annotations

import json
import time
from logging import Logger
from pathlib import Path
from typing import Any

from pipeline.pipeline_context import PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration

PAGE_FILE = "scene_graph.html"
GRAPH_FILE = "scene_graph.json"
PHYSICAL_FILE = "physical_properties.json"


class SceneGraphConfiguration(PipelineStageConfiguration):
    def __init__(self, name: str, device, torch_dtype: Any, log: Logger, keys=None, seed: int = 0,
                 sample_points: int = 5000, title: str = ""):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # Points sampled on each model's surface for the page's turntable (0: none).
        self.sample_points = int(sample_points)
        # Page title; empty: the sample photograph's name, else the run id.
        self.title = str(title or "")


class SceneGraphStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return SceneGraphConfiguration

    def run(self, context: PipelineContext) -> PipelineContext:
        cfg: SceneGraphConfiguration = self.config
        task = self.create_progress(1)
        try:
            if self.output is None:
                self.log_info("Scene Graph: files are not saved in this run; nothing written")
                return context
            # The run directory: the one the context was loaded from (also under replay-stage,
            # whose output goes elsewhere), else the parent of this stage's own folder.
            run_dir = Path(getattr(context, "loaded_from", None) or self.output.parent)
            self._write(run_dir, cfg)
        except Exception as e:  # a page for people must not cost the run its scene
            self.log_warning(f"Scene Graph: not written ({type(e).__name__}: {e})")
        finally:
            self.finish_progress(task)
        return context

    def _write(self, run_dir: Path, cfg: SceneGraphConfiguration) -> None:
        from pipeline.scene_graph import view

        t0 = time.time()
        doc = view.build_view(run_dir, title=cfg.title or None, sample_points=cfg.sample_points)
        page = view.render_html(doc)
        (self.output / PAGE_FILE).write_text(page, encoding="utf-8")
        (self.output / GRAPH_FILE).write_text(json.dumps(doc["graph"], indent=1), encoding="utf-8")
        (self.output / PHYSICAL_FILE).write_text(json.dumps(view.physical_table(doc), indent=1), encoding="utf-8")

        graph = doc["graph"]
        meshes = sum(1 for r in doc["placements"]["rows"] if not r[6] & 8)
        self.log_info(
            f"Scene Graph: {len(graph['nodes'])} nodes, {len(graph['relations'])} relations, "
            f"{len(doc['detections'])} detections, {len(doc['models'])} model files, "
            f"{len(doc['placements']['rows'])} placements ({meshes} as meshes); "
            f"{self.output / PAGE_FILE} ({len(page) / 1e6:.1f} MB, {time.time() - t0:.1f}s)")
        for w in doc.get("warnings") or []:
            self.log_info(f"Scene Graph: {w}")
