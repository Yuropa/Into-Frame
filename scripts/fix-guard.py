#!/usr/bin/env python3
"""The fix loop's last word: did the scene as a whole get better?

    scripts/fix-guard.py Paris --before agent2 --after agent2-fix [--restore agent2]

The loop keeps an edit on the judge's word about one region, round by round; each step can
read as "improved" while the frame drifts away from the photograph (Paris agent2-fix: the
tour boat scaled up into a marbled blob and four red barges across the river, every round
kept, the independent check 0.28 -> 0.13). This compares the scene-check of the scene before
the loop with the one after it, on measurements the loop did not steer by: the edits stand
unless the judge's whole-frame score fell by more than its call-to-call noise AND the DINO
similarity to the photograph fell too. With --restore the snapshot of that tag is put back
when they do not stand. Exit code 0 = kept, 1 = worse (restored if asked).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JUDGE_NOISE = 0.05   # the same render scores within about this from one judge call to the next
DINO_NOISE = 0.01


def scores(scene: str, label: str) -> dict:
    p = ROOT / "scratch" / "check" / scene.replace(" ", "_") / label / "check.json"
    if not p.is_file():
        sys.exit(f"no scene-check at {p}")
    return json.load(open(p))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene")
    ap.add_argument("--before", required=True, help="scene-check label of the scene before the loop")
    ap.add_argument("--after", required=True, help="scene-check label of the scene with the kept edits")
    ap.add_argument("--restore", default=None, help="snapshot tag to put back when the edits do not stand")
    args = ap.parse_args()
    b, a = scores(args.scene, args.before), scores(args.scene, args.after)
    dj = float(a.get("judge") or 0) - float(b.get("judge") or 0)
    dd = float(a.get("dino_global") or 0) - float(b.get("dino_global") or 0)
    worse = dj < -JUDGE_NOISE and dd < -DINO_NOISE
    print(f"{args.scene}: judge {b.get('judge')} -> {a.get('judge')} ({dj:+.2f}), "
          f"DINO {b.get('dino_global')} -> {a.get('dino_global')} ({dd:+.3f}), "
          f"layout {b.get('dino_layout')} -> {a.get('dino_layout')}, dE {b.get('colour_de')} -> {a.get('colour_de')}: "
          f"{'WORSE, the edits do not stand' if worse else 'the edits stand'}")
    if worse and args.restore:
        rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "scene-snapshot.py"), "restore", args.scene,
                             "--tag", args.restore], cwd=ROOT).returncode
        if rc != 0:
            sys.exit(f"restore of {args.restore} failed ({rc})")
    return 1 if worse else 0


if __name__ == "__main__":
    sys.exit(main())
