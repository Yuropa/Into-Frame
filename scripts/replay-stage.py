#!/usr/bin/env python3
"""
Replay one pipeline stage against a cached run, with the real stage code and the
real config, without re-running anything else.

    python scripts/replay-stage.py <run-dir> "<Stage Name>" [--out DIR] [--config PATH]
                                   [--set key=value ...] [--save] [--list]

<run-dir> is a per-input output directory (output/<uuid>/, or <name>.debug/context/
<uuid>/ from a debug bundle). It is only ever READ: the stage's outputs, debug
images and log go to --out (default: scratch/replay/<Stage Name>/), and only
--save writes the resulting context back into <run-dir>.

Every validation during the 2026-10-13 review was a hand-written stub context;
this is that harness once, done the way the pipeline itself does it: the context
is loaded with PipelineContext.load, every earlier stage is pushed and popped so
the stage's input_* accessors resolve exactly as they would in a run, the stage is
constructed from config.yaml (with --set overrides), reset_stage clears its own
previous output, and run() is called.

Examples
    python scripts/replay-stage.py output/0bc34849-... "Region Map Refinement"
    python scripts/replay-stage.py output/0bc34849-... "Treeline Backdrop" --set min_height_m=2.0
    python scripts/replay-stage.py output/0bc34849-... --list
"""
import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))


def _parse_set(values: list[str]) -> dict:
    import yaml
    out = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("stage", nargs="?", help="stage name as in config.yaml")
    parser.add_argument("--out", type=Path, default=None, help="where the stage writes (default scratch/replay/<stage>)")
    parser.add_argument("--config", type=Path, default=ROOT / "server" / "config.yaml")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a stage config value")
    parser.add_argument("--save", action="store_true", help="write the resulting context back into run_dir")
    parser.add_argument("--list", action="store_true", help="list the stages in config.yaml and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    from pipeline.pipeline import Pipeline, PipelineConfiguration, STAGE_REGISTRY
    from pipeline.pipeline_context import PipelineContext
    from pipeline.pipeline_stage import SemanticKey
    from rich.progress import Progress

    run_dir = args.run_dir.resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"{run_dir} is not a directory")

    out = (args.out or (ROOT / "scratch" / "replay" / (args.stage or "list"))).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # The configuration owns logging and would clear <output>/build on construction,
    # so it points at the replay directory, never at the cached run.
    config = PipelineConfiguration(
        output=str(out), config_path=args.config, log_mode="verbose" if args.verbose else "plain",
    )
    entries = config.stages_yaml
    names = [e["name"] for e in entries]
    if args.list or not args.stage:
        for e in entries:
            print(f"{'' if e.get('enabled', True) else '(disabled) '}{e['name']}  [{e.get('stage')}]")
        return 0
    if args.stage not in names:
        raise SystemExit(f"Unknown stage {args.stage!r}. --list shows the names.")

    entry = dict(entries[names.index(args.stage)])
    stage_class = STAGE_REGISTRY[entry["stage"]]
    raw_keys = entry.get("keys")
    keys = {SemanticKey(k): v for k, v in raw_keys.items()} if raw_keys else None
    kwargs = {k: v for k, v in entry.items() if k not in {"name", "stage", "enabled", "keys"}}
    kwargs.update(_parse_set(args.set))
    stage_config = config.stage_config(args.stage, stage_class.config_class(), keys=keys, **kwargs)
    stage = stage_class(config=stage_config)
    stage.output = out
    stage.temp = out / "build"
    stage.temp.mkdir(parents=True, exist_ok=True)

    enabled = [e["name"] for e in entries if e.get("enabled", True)]
    context = PipelineContext()
    context.load(run_dir, enabled, names)
    if args.stage not in enabled:
        logging.getLogger().warning("%s is disabled in config.yaml; replaying it anyway", args.stage)

    # Walk the earlier stages so the context's "previous stage" is the one just
    # before the target, exactly as in a run.
    for name in enabled:
        if name == args.stage:
            break
        context.push_stage(name)
        context.pop_stage()

    log = config.log
    log.info("Replaying %r against %s -> %s", args.stage, run_dir, out)
    if kwargs:
        log.info("Config: %s", {k: v for k, v in kwargs.items()})

    started = time.time()
    with Progress() as progress:
        main_task = progress.add_task(args.stage, total=1)
        stage._set_progress(progress, main_task)
        stage.total_tasks = 1
        context.push_stage(args.stage)
        context.reset_stage(args.stage)
        try:
            context = stage.run(context)
        finally:
            stage.clean_up()
        context.pop_stage()
    log.info("Done in %.1f s", time.time() - started)

    if args.save:
        context.save(path=run_dir)
        log.info("Context written back to %s", run_dir)
    else:
        log.info("Context NOT saved (pass --save to write it into the run directory)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
