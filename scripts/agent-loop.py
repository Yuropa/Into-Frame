#!/usr/bin/env python3
"""Run the agent loop over one cached scene (docs/agentic-scene-graph-plan.md §4).

    scripts/agent-loop.py "Mount Rainier" --label agent1 --rounds 3 [--max-usd 10]
        [--model anthropic:claude-opus-5-5] [--judge-model anthropic:claude-haiku-5-5]
        [--baseline band27] [--allow-expensive] [--dry-run] [--no-grass]
        [--base-url http://studio.local:8000/v1]

Round 0 renders and grades the scene as it stands (or takes --baseline, an existing
scratch/check label, to skip that). Each round the orchestrator proposes edits from
server/agent/knobs.yaml, the pipeline reruns from the earliest touched stage, and the
scene is checked again. --dry-run stops after the first proposal (one Opus call, no GPU).
Without --allow-expensive the vocabulary excludes edits that rerun stages before Object
Completion (hours). Output: scratch/agent/<label>/ (loop.log, report.md, round_N/).

Refuses to start while another pipeline run, server or pan render holds the GPU.
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
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-usd", type=float, default=10.0)
    ap.add_argument("--model", default="anthropic:claude-opus-5-5")
    ap.add_argument("--judge-model", default="anthropic:claude-haiku-5-5")
    ap.add_argument("--no-judge", action="store_true", help="metrics only, no critic")
    ap.add_argument("--base-url", default=None, help="OpenAI-compatible server for openai:<model> ids")
    ap.add_argument("--baseline", default=None, help="existing scene-check label to use as round 0")
    ap.add_argument("--allow-expensive", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-grass", action="store_true")
    ap.add_argument("--patience", type=int, default=2)
    args = ap.parse_args()

    from agent.loop import AgentLoop, LoopConfig
    from agent.runner import gpu_busy
    busy = gpu_busy()
    if busy and not args.dry_run:
        print(f"GPU is held by: {busy}", file=sys.stderr)
        return 3
    loop = AgentLoop(LoopConfig(scene=args.scene, label=args.label, rounds=args.rounds, max_usd=args.max_usd,
                                model=args.model, judge_model=None if args.no_judge else args.judge_model,
                                base_url=args.base_url, dry_run=args.dry_run, allow_expensive=args.allow_expensive,
                                baseline=args.baseline, grass=not args.no_grass, patience=args.patience))
    rep = loop.run()
    print(f"best round {rep['best_round']} ({rep['best_label']}): {rep['best_scores']}; spent ${rep['usd']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
