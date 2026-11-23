"""Scene Edits: replay the agent's accepted scene actions on the pipeline's scene.

The agent edits the scene the renderers draw through agent/editor.py (place, move,
scale, rotate, remove, swap_mesh, set_atmosphere, set_sun) and persists the accepted
actions in <run>/scene_edits.json (scripts/scene-edit.py --accept). This stage applies
that file to ContextKey.SCENE so every later stage (video, animation) and the .frame
archive carry the edits, and a rerun of the first pass from any earlier stage puts
them back without the agent. No file, or an empty one: the scene passes through.

The stage is dirty whenever the edits file changed since it last ran (its marker
records the file's hash), so a `main.py run` after new edits were accepted reruns it
and everything after it; nothing before it is touched.

Input   ContextKey.SCENE, TERRAIN_MESH, WATER_MESH, HEIGHT_MAP(+PARAMS), INTRINSICS, DEPTH,
        every placed mesh key an action names; <run>/scene_edits.json
Output  ContextKey.SCENE (edited), scene_edits_applied (records, failures, hash)
"""
from __future__ import annotations

from logging import Logger
from typing import Any

from pipeline.pipeline_context import ContextKey, PipelineContext
from pipeline.pipeline_stage import PipelineStage, PipelineStageConfiguration
from scene.scene import Scene

MARKER = "scene_edits_applied"


class SceneEditsConfiguration(PipelineStageConfiguration):
    def __init__(self, name: str, device, torch_dtype: Any, log: Logger, keys=None, seed: int = 0,
                 strict: bool = False):
        super().__init__(name, device, torch_dtype, log, keys, seed=seed)
        # true: an action the editor refuses fails the stage; false: it is logged and skipped.
        self.strict = bool(strict)


class SceneEditsStage(PipelineStage):
    @classmethod
    def config_class(cls):
        return SceneEditsConfiguration

    def _edits_path(self, context: PipelineContext):
        from agent.editor import EDITS_FILE
        # The run directory: the one the context was loaded from (also under replay-stage,
        # whose output goes elsewhere), else the parent of this stage's own folder.
        run_dir = getattr(context, "loaded_from", None) or (self.output.parent if self.output is not None else None)
        return (run_dir / EDITS_FILE) if run_dir is not None else None

    def has_expected_output(self, context: PipelineContext) -> bool:
        from agent.editor import edits_sha
        marker = context.object(MARKER)
        if not isinstance(marker, dict):
            return False
        p = self._edits_path(context)
        return marker.get("edits_sha", "") == (edits_sha(p) if p is not None else "")

    def run(self, context: PipelineContext) -> PipelineContext:
        from agent.editor import apply_all, edits_sha, load_edits
        cfg: SceneEditsConfiguration = self.config
        task = self.create_progress(1)
        scene = context.input_scene(ContextKey.SCENE)
        p = self._edits_path(context)
        edits = load_edits(p.parent) if p is not None and p.is_file() else {"edits": []}
        actions = [e.get("input") or e for e in edits.get("edits", []) if isinstance(e, dict)]
        if scene is None or not actions:
            self.log_info("Scene edits: " + ("no scene" if scene is None else "no accepted edits") + "; scene unchanged")
            context.add_object(MARKER, {"edits_sha": edits_sha(p) if p is not None else "", "applied": 0, "failures": []})
            self.finish_progress(task)
            return context
        data = scene.encode()
        new, applied, failures = apply_all(data, actions, context=context, run_dir=p.parent, who="scene_edits",
                                           log=self.log_info, strict=cfg.strict)
        for f in failures:
            self.log_warning(f"Scene edits: {f['input'].get('action')} refused: {f['error']}")
        self.log_info(f"Scene edits: {len(applied)} of {len(actions)} actions applied; {len(new['objects'])} objects")
        context.add_scene(ContextKey.SCENE, Scene.decode(new))
        context.add_object(MARKER, {"edits_sha": edits_sha(p), "applied": len(applied), "failures": failures,
                                    "records": [{k: v for k, v in r.items() if k != "removed"} for r in applied]})
        self.finish_progress(task)
        return context
