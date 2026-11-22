#!/bin/bash
# From the Into Frame box: verify the Mac's llama-server speaks what the plan client
# expects (docs/agentic-scene-graph-plan.md §7), using the client's own `_openai` path, and
# print the config lines. Usage:
#
#   scripts/local-models/check-endpoint.sh http://<mac>.local:8000/v1 [photo.jpg]
#
# OPENAI_API_KEY, if the server was started with --api-key, must be in the environment.
# With a photo it runs a real Scene Plan read through PlanModel (no Anthropic call, no
# cache write) and prints the plan's place/camera/sun lines; without one it only lists the
# models and sends a one-line text prompt.
set -euo pipefail
BASE=${1:?base url, e.g. http://studio.local:8000/v1}
PHOTO=${2:-}
BASE=${BASE%/}
cd "$(dirname "$0")/../.."

AUTH=(); [ -n "${OPENAI_API_KEY:-}" ] && AUTH=(-H "Authorization: Bearer $OPENAI_API_KEY")
echo "== models at $BASE"
curl -fsS --max-time 10 "${AUTH[@]}" "$BASE/models" | python3 -c 'import json,sys; print(", ".join(m["id"] for m in json.load(sys.stdin)["data"]))'

if [ -z "$PHOTO" ]; then
    echo "== one text prompt"
    curl -fsS --max-time 120 "${AUTH[@]}" -H 'Content-Type: application/json' \
        -d '{"model":"qwen3-vl","max_tokens":40,"messages":[{"role":"user","content":"Say which model you are, in one line."}]}' \
        "$BASE/chat/completions" | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())'
else
    echo "== a Scene Plan read of $PHOTO through PlanModel (openai provider)"
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate frame 2>/dev/null || true
    export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=""
    OPENAI_BASE_URL="$BASE" python3 - "$PHOTO" <<'PY'
import sys, time, json
sys.path.insert(0, "server")
from PIL import Image
from pipeline.scene_plan.client import PlanModel
from pipeline.scene_plan.schema import PLAN_SCHEMA, SYSTEM_PROMPT, USER_PROMPT
from pipeline.scene_plan.scene_plan import TOOL_NAME
pil = Image.open(sys.argv[1]).convert("RGB")
user = USER_PROMPT.replace("{caption}", "a photograph")   # the stage puts the caption here
m = PlanModel("openai:qwen3-vl", timeout_s=900)
t0 = time.time()
reply = m.read_photo(pil, SYSTEM_PROMPT, user, TOOL_NAME, PLAN_SCHEMA, 8000)
dt = time.time() - t0
d = reply.data
print(f"{reply.input_tokens} in / {reply.output_tokens} out, {dt:.0f} s")
for k in ("place", "camera", "sun", "atmosphere"):
    if k in d: print(f"  {k}: {json.dumps(d[k])[:160]}")
print(f"  species: {[s.get('name') for s in d.get('species', [])][:8]}")
PY
fi

HOSTPORT=${BASE#http://}; HOSTPORT=${HOSTPORT%/v1}
cat <<EOF
== config lines for a role on the Mac (server/config.yaml)
  model: openai:qwen3-vl
  base_url: $BASE
EOF
