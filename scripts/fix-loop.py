#!/usr/bin/env python3
"""Run the issue-driven fix loop over one built scene (server/agent/fixer.py).

    scripts/fix-loop.py Paris --label fix1 [--rounds 12] [--max-usd 10] [--patience 3]
        [--model anthropic:claude-opus-5-5] [--judge-model anthropic:claude-sonnet-5-5]
        [--kinds missing,extra,scale] [--no-grass] [--width 1200] [--dry-run] [--baseline opus1]

Round 0 renders the scene from the capture camera and the judge lists its issues. Each
round the fixer proposes one to three scene actions for the worst open issue, the loop
applies them to a copy of the scene, renders it (no pipeline rerun) and asks the judge
whether that issue is resolved without regressions; kept actions go to
<run>/scene_edits.json and the Scene Edits stage is replayed into the run. --dry-run
stops after the first proposal. Output: scratch/fix/<label>/ (loop.log, report.md,
round_N/). Refuses to start while a pipeline run, server or pan render holds the GPU.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--label", required=True)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--max-usd", type=float, default=10.0)
    ap.add_argument("--model", default="anthropic:claude-opus-5-5")
    ap.add_argument("--judge-model", default="anthropic:claude-sonnet-5-5")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--kinds", default=None, help="comma list of issue kinds to work on (default: all actionable)")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--attempts", type=int, default=2, help="attempts per issue")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--no-grass", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--baseline", default=None, help="existing scene-check label whose render and verdict stand in for round 0")
    args = ap.parse_args()

    from agent.fixer import FixConfig, FixLoop
    from agent.runner import gpu_busy
    busy = gpu_busy()
    if busy and not (args.dry_run and args.baseline):
        print(f"GPU is held by: {busy}", file=sys.stderr)
        return 3
    loop = FixLoop(FixConfig(scene=args.scene, label=args.label, rounds=args.rounds, max_usd=args.max_usd,
                             model=args.model, judge_model=args.judge_model, base_url=args.base_url,
                             width=args.width, grass=not args.no_grass, patience=args.patience,
                             attempts_per_issue=args.attempts, dry_run=args.dry_run, baseline=args.baseline,
                             only_kinds=set(args.kinds.split(",")) if args.kinds else None))
    rep = loop.run_loop()
    print(rep)
    return 0


if __name__ == "__main__":
    sys.exit(main())
