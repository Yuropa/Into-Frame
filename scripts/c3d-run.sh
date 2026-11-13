#!/usr/bin/env bash
# Build one of our sample scenes with the 3DCodeVerse harness (LLM-written three.js /
# bpy code, measured + VLM-judged refine loop) so it can be compared with our
# photo-reconstruction pipeline on the same photograph.
#
#   scripts/c3d-run.sh Paris                     # brief = the Captioning stage's caption
#   scripts/c3d-run.sh "Mount Rainier" --language scene_blender --rounds 2
#   scripts/c3d-run.sh Paris --brief "quay on the Seine at noon, ..." --rounds 1
#   scripts/c3d-run.sh Paris --dry-run           # print the command only
#
# The harness lives in external/3dcodeverse (gitignored clone, `conda activate c3d`,
# installed by hand on 2026-11-06: Python 3.13, Node 26, Blender 5.2.3 LTS at
# ~/.local/bin/blender-5.2, puppeteer Chrome). Runs land in
# external/3dcodeverse/harness/runs/<slug>/ (deliverable/, artifacts/renders/rNN/,
# record.json); score them against the photo with scripts/c3d-compare.py.
#
# Backends. The coding agent is the local Claude Code CLI (subscription, no key).
# The planner and the judge are API chat models and need a key, stored once in the
# login keyring (gnome-keyring via the `keyring` CLI in the c3d env):
#   keyring set anthropic 3dcodeverse     -> anthropic:<model> for both (the default here)
#   keyring set gemini 3dcodeverse        -> gemini:gemini-4-argon (the harness's own default)
# ANTHROPIC_API_KEY / GEMINI_API_KEYS in the environment, or the harness's own
# ~/.config/3dcodeverse/config.yaml, work too. Nothing runs without one; the script says so.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS="$ROOT/external/3dcodeverse/harness"

scene=""; brief=""; language="scene_threejs"; rounds=1; minutes=60
generator="claude-code:opus"; planner=""; judge=""; dry=0; extra=()
while [ $# -gt 0 ]; do
    case "$1" in
        --brief) brief="$2"; shift 2 ;;
        --language) language="$2"; shift 2 ;;
        --rounds) rounds="$2"; shift 2 ;;
        --max-minutes) minutes="$2"; shift 2 ;;
        --generator) generator="$2"; shift 2 ;;
        --planner) planner="$2"; shift 2 ;;
        --judge) judge="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --) shift; extra+=("$@"); break ;;
        -*) extra+=("$1"); shift ;;
        *) scene="$1"; shift ;;
    esac
done
[ -n "$scene" ] || { echo "usage: $0 <Scene|photo.jpg> [--brief TEXT] [--language scene_threejs|scene_blender] [--rounds N] [--max-minutes M] [--generator ID] [--planner ID] [--judge ID] [--dry-run] [-- extra 3dcodeverse flags]" >&2; exit 2; }

if [ -f "$scene" ]; then photo="$(realpath "$scene")"; name="$(basename "${scene%.*}")"
else photo="$ROOT/server/samples/$scene.jpg"; name="$scene"; fi
[ -f "$photo" ] || { echo "no photo at $photo" >&2; exit 2; }
src_photo="$photo"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate c3d

# The harness types a reference image by its extension and the API rejects a mismatch;
# several samples are WebP files named .jpg, so hand it a real JPEG (gitignored scratch/).
mkdir -p "$ROOT/scratch/c3d"
photo="$ROOT/scratch/c3d/$name.jpg"
python -c "import sys; from PIL import Image; Image.open(sys.argv[1]).convert('RGB').save(sys.argv[2], 'JPEG', quality=95)" "$src_photo" "$photo"

# The caption our Captioning stage wrote for this photograph, if a run cache has it.
if [ -z "$brief" ]; then
    brief="$(python - "$ROOT" "$src_photo" <<'EOF'
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image
root, photo = Path(sys.argv[1]), Path(sys.argv[2])
def thumb(p):
    return np.asarray(Image.open(p).convert("RGB").resize((32, 32), Image.BOX), dtype=np.float32)
want = thumb(photo)
for run in sorted((root / "output").glob("*-*-*-*-*")):
    cap = run / "Captioning" / "input_caption.json"
    inp = run / "input.png"
    if cap.is_file() and inp.is_file() and np.abs(thumb(inp) - want).mean() < 6:
        print(json.loads(cap.read_text())); break
EOF
)"
    [ -n "$brief" ] || { echo "no cached caption for $name; pass --brief" >&2; exit 2; }
    brief="$brief -- an outdoor place photographed from eye level; match the photograph's time of day, weather, light and vantage point"
fi

# Planner / judge: Anthropic when a key is present, else the harness's Gemini default.
# Keys, in order of preference: the login keyring (encrypted at rest, unlocked with the
# session; `keyring set anthropic 3dcodeverse` once, the key is exported to the harness
# process only), then the environment, then the harness's own ~/.config/3dcodeverse/config.yaml.
cfg="$HOME/.config/3dcodeverse/config.yaml"
has_anthropic=0; has_gemini=0
if [ -z "${ANTHROPIC_API_KEY:-}" ] && command -v keyring >/dev/null; then
    k="$(keyring get anthropic 3dcodeverse 2>/dev/null || true)"
    [ -n "$k" ] && export ANTHROPIC_API_KEY="$k"
fi
if [ -z "${GEMINI_API_KEYS:-}" ] && command -v keyring >/dev/null; then
    k="$(keyring get gemini 3dcodeverse 2>/dev/null || true)"
    [ -n "$k" ] && export GEMINI_API_KEYS="$k"
fi
unset k
[ -n "${ANTHROPIC_API_KEY:-}" ] && has_anthropic=1
[ -n "${GEMINI_API_KEYS:-}${GEMINI_API_KEY:-}" ] && has_gemini=1
[ -f "$cfg" ] && grep -qE '^anthropic_api_key:\s*\S' "$cfg" && has_anthropic=1
[ -f "$cfg" ] && grep -qE '^gemini_api_keys:\s*\S' "$cfg" && has_gemini=1
if [ -z "$planner$judge" ]; then
    if [ "$has_anthropic" = 1 ]; then planner="anthropic:claude-opus-5-5"; judge="anthropic:claude-opus-5-5"
    elif [ "$has_gemini" = 1 ]; then planner="gemini:gemini-4-argon"; judge="gemini:gemini-4-argon"
    elif [ "$dry" = 1 ]; then planner="anthropic:claude-opus-5-5"; judge="$planner"; echo "(dry run: no API key set)" >&2
    else
        echo "The planner and judge are API chat models. Store a key once with: keyring set anthropic 3dcodeverse  (or gemini 3dcodeverse)," >&2
        echo "or export ANTHROPIC_API_KEY / GEMINI_API_KEYS, or put anthropic_api_key: in $cfg (chmod 600)." >&2
        echo "The coding agent ($generator) runs on the local Claude Code subscription and needs no key." >&2
        exit 3
    fi
fi
[ -n "$planner" ] || planner="$judge"; [ -n "$judge" ] || judge="$planner"

slug="$(echo "$name" | tr 'A-Z ' 'a-z_')_${language#scene_}_$(date +%m%d_%H%M)"
cmd=(3dcodeverse make "$brief" --image "$photo" --track scene --language "$language"
     --generator "$generator" --planner "$planner" --judge "$judge"
     --rounds "$rounds" --max-minutes "$minutes" --slug "$slug" --tag into-frame --tag "$name" "${extra[@]}")

echo "brief: $brief"
echo "run:   $HARNESS/runs/$slug"
printf '%q ' "${cmd[@]}"; echo
[ "$dry" = 1 ] && exit 0

cd "$HARNESS"
"${cmd[@]}"
echo "done: $HARNESS/runs/$slug  -> python scripts/c3d-compare.py $slug"
