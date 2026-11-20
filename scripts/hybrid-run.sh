#!/usr/bin/env bash
# Rebuild cached scenes with the Scene Plan priors and check them (docs/hybrid-scene-plan.md).
#
#   scripts/hybrid-run.sh Paris "Mount Rainier"                 # rerun from Panorama Lighting, check
#   scripts/hybrid-run.sh Paris --from "Region Map" --label h2  # rerun from a later stage
#   scripts/hybrid-run.sh Paris --no-check                      # pipeline only
#
# For each scene: (1) if the cached run has no Scene Plan yet, read the photograph once
# with scripts/replay-stage.py --save (one billed call, then cached in cache/scene_plan/);
# (2) `main.py run --rerun <from>` so every stage from <from> on rebuilds with the plan's
# priors while the panorama and segmentation stay cached; (3) scripts/scene-check.py with
# --label, --grass and --judge. Refuses to start while another pipeline holds the GPU.
# Logs: scratch/hybrid/<label>/<scene>.log, chain log scratch/hybrid/<label>/chain.log.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scenes=(); from="Panorama Lighting"; label="hybrid-$(date +%m%d-%H%M)"; check=1; judge="--judge"
while [ $# -gt 0 ]; do
    case "$1" in
        --from) from="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --no-check) check=0; shift ;;
        --no-judge) judge=""; shift ;;
        -h|--help) sed -n 2,14p "$0"; exit 0 ;;
        *) scenes+=("$1"); shift ;;
    esac
done
[ ${#scenes[@]} -gt 0 ] || { echo "usage: $0 <Scene>... [--from STAGE] [--label L] [--no-check] [--no-judge]" >&2; exit 2; }

if pgrep -f "main\.py (run|server)" >/dev/null; then
    echo "a pipeline run or server is alive (main.py run|server); not starting" >&2; exit 3
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate frame
export HF_HUB_OFFLINE=1
# The check renderer draws up to this many grass tufts (25000 thins the near meadow to hairlines).
export RENDER_GRASS_MAX=60000
logdir="$ROOT/scratch/hybrid/$label"; mkdir -p "$logdir"
chain="$logdir/chain.log"
say() { echo "$(date '+%H:%M:%S') $*" | tee -a "$chain"; }

resolve_run() {  # scene name -> output/<uuid> holding that photograph (by thumbnail), or ""
    python - "$ROOT" "$1" <<'PY'
import sys, numpy as np
from pathlib import Path
from PIL import Image
root, scene = Path(sys.argv[1]), sys.argv[2]
def thumb(p): return np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BOX), dtype=np.float32)
want = thumb(root / "server" / "samples" / f"{scene}.jpg")
best = None
for run in (root / "output").glob("*-*-*-*-*"):
    inp = run / "input.png"
    if inp.is_file():
        err = float(np.abs(thumb(inp) - want).mean())
        if err < 6 and (best is None or err < best[0]):
            best = (err, run)
print(best[1] if best else "")
PY
}

for scene in "${scenes[@]}"; do
    photo="$ROOT/server/samples/$scene.jpg"
    [ -f "$photo" ] || { say "no photo at $photo; skipping"; continue; }
    run="$(resolve_run "$scene")"
    want=$(python -c "import sys; sys.path.insert(0,'$ROOT/server'); from pipeline.scene_plan.schema import SCHEMA_VERSION; print(SCHEMA_VERSION)")
    have=$(python -c "import json,sys; print(json.load(open(sys.argv[1])).get('schema_version'))" "$run/Scene Plan/scene_plan.json" 2>/dev/null || echo none)
    if [ -n "$run" ] && [ "$have" != "$want" ]; then
        say "[$scene] Scene Plan in $run is schema $have, want $want; reading the photograph once"
        python "$ROOT/scripts/replay-stage.py" "$run" "Scene Plan" --save >> "$logdir/$scene.plan.log" 2>&1 \
            || say "[$scene] Scene Plan replay failed (see $scene.plan.log); the run will read it itself"
    fi
    say "[$scene] pipeline from '$from' -> $logdir/$scene.log"
    ( cd "$ROOT/server" && python main.py --log-mode plain run "$photo" --output "$ROOT/output" --rerun "$from" ) \
        > "$logdir/$scene.log" 2>&1
    rc=$?
    say "[$scene] pipeline exit $rc"
    if [ $rc -ne 0 ]; then grep -E "Pipeline error|Error|Traceback" "$logdir/$scene.log" | tail -5 | tee -a "$chain"; continue; fi
    if [ $check -eq 1 ]; then
        say "[$scene] scene-check --label $label"
        python "$ROOT/scripts/scene-check.py" "$scene" --label "$label" --grass $judge > "$logdir/$scene.check.log" 2>&1 \
            && tail -6 "$logdir/$scene.check.log" | tee -a "$chain" \
            || say "[$scene] scene-check failed (see $scene.check.log)"
    fi
done
say "ALL DONE $label"
