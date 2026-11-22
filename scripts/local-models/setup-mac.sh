#!/bin/bash
# Set up a Mac Studio (Apple Silicon, 256 GB) as the vision-language model server for the
# agentic pipeline (docs/agentic-scene-graph-plan.md §7). Run ON THE MAC:
#
#   scripts/local-models/setup-mac.sh [--model 235b|32b] [--port 8000] [--ctx 32768]
#                                     [--api-key KEY] [--no-launchd] [--download-only]
#
# What it does, in order (every step is idempotent, re-run after any failure):
#   1. Homebrew, then `llama.cpp` (llama-server with Metal) and the Hugging Face CLI.
#   2. Downloads the chosen Qwen3-VL GGUF and its vision projector (mmproj) into
#      ~/models/<repo>. 235b = Qwen/Qwen3-VL-235B-A22B-Instruct-GGUF at Q4_K_M (three
#      shards, ~140 GB; mixture, 22 B active, the fast option on 256 GB).
#      32b = unsloth/Qwen3-VL-32B-Instruct-GGUF at Q8_0 (~35 GB; dense, steadier).
#   3. Writes ~/.into-frame/serve-local-model.sh, which starts llama-server on 0.0.0.0 with
#      the model aliased as `qwen3-vl`, and a launchd agent that keeps it running across
#      reboots (unless --no-launchd). Logs: ~/.into-frame/llama-server.log.
#   4. Waits for /v1/models, then times one vision chat completion (a tiny generated PNG)
#      and prints tokens per second, so the number in the design doc is measured, not guessed.
#
# The API key is optional and is passed to llama-server only on its command line; it is
# never written into this repo. If given, the Into Frame box sets OPENAI_API_KEY to match.
# Then on the Into Frame box: scripts/local-models/check-endpoint.sh http://<mac>:8000/v1
set -euo pipefail

MODEL=235b; PORT=8000; CTX=32768; API_KEY=""; LAUNCHD=1; DOWNLOAD_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --model) MODEL=$2; shift 2 ;;
        --port) PORT=$2; shift 2 ;;
        --ctx) CTX=$2; shift 2 ;;
        --api-key) API_KEY=$2; shift 2 ;;
        --no-launchd) LAUNCHD=0; shift ;;
        --download-only) DOWNLOAD_ONLY=1; shift ;;
        -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

say() { printf '\n== %s\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

[ "$(uname -s)" = Darwin ] || die "this script runs on the Mac, not here ($(uname -s))"
[ "$(uname -m)" = arm64 ] || die "Apple Silicon required (uname -m says $(uname -m))"
MEM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
say "macOS $(sw_vers -productVersion), $(sysctl -n machdep.cpu.brand_string), ${MEM_GB} GB unified memory"

case "$MODEL" in
    235b)
        REPO=Qwen/Qwen3-VL-235B-A22B-Instruct-GGUF
        WEIGHTS_GLOB='Qwen3VL-235B-A22B-Instruct-Q4_K_M-split-*.gguf'
        MMPROJ_GLOB='mmproj-Qwen3VL-235B-A22B-Instruct-Q8_0.gguf'
        NEED_GB=150 ;;
    32b)
        REPO=unsloth/Qwen3-VL-32B-Instruct-GGUF
        WEIGHTS_GLOB='Qwen3-VL-32B-Instruct-Q8_0.gguf'
        MMPROJ_GLOB='mmproj-F16.gguf'
        NEED_GB=40 ;;
    *) die "--model must be 235b or 32b" ;;
esac
[ "$MEM_GB" -ge "$NEED_GB" ] || die "$MODEL wants about ${NEED_GB} GB of memory; this machine has ${MEM_GB} GB"

# 1. tooling --------------------------------------------------------------------------
say "tooling"
if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew is not installed. Install it from https://brew.sh (one command there), then re-run."
    exit 1
fi
for pkg in llama.cpp huggingface-cli; do
    if brew list --formula "$pkg" >/dev/null 2>&1; then
        brew upgrade "$pkg" 2>/dev/null || true
    else
        brew install "$pkg"
    fi
done
command -v llama-server >/dev/null || die "llama-server not on PATH after brew install"
HF=$(command -v hf || command -v huggingface-cli) || die "no Hugging Face CLI"
echo "llama-server: $(llama-server --version 2>&1 | head -1)"

# 2. weights ---------------------------------------------------------------------------
MODEL_DIR="$HOME/models/$REPO"
say "weights -> $MODEL_DIR"
mkdir -p "$MODEL_DIR"
# hf download resumes partial files; an interrupted run just continues.
"$HF" download "$REPO" --include "$WEIGHTS_GLOB" "$MMPROJ_GLOB" --local-dir "$MODEL_DIR"
FIRST_SHARD=$(ls "$MODEL_DIR"/$WEIGHTS_GLOB | sort | head -1)
MMPROJ=$(ls "$MODEL_DIR"/$MMPROJ_GLOB | head -1)
[ -f "$FIRST_SHARD" ] && [ -f "$MMPROJ" ] || die "download incomplete (no shard or no mmproj under $MODEL_DIR)"
du -sh "$MODEL_DIR" | sed 's/^/on disk: /'
[ "$DOWNLOAD_ONLY" = 1 ] && { say "download only: done"; exit 0; }

# 3. server script + launchd -----------------------------------------------------------
CFG="$HOME/.into-frame"; mkdir -p "$CFG"
SERVE="$CFG/serve-local-model.sh"
say "server script -> $SERVE"
# Flags: all layers on the GPU, the model's own chat template (--jinja, needed for the
# Qwen3-VL template and for json_schema response_format, which llama-server turns into a
# grammar), two parallel slots so a critique call does not queue behind a plan read.
# -c is the KV budget shared by the slots, so it is doubled to give each slot $CTX.
{
    echo '#!/bin/bash'
    echo "exec llama-server -m '$FIRST_SHARD' --mmproj '$MMPROJ' --alias qwen3-vl \\"
    echo "    --host 0.0.0.0 --port $PORT -c $((CTX * 2)) -ngl 999 -np 2 --jinja --flash-attn on \\"
    [ -n "$API_KEY" ] && echo "    --api-key '$API_KEY' \\"
    echo "    --no-warmup"
} > "$SERVE"
chmod 700 "$SERVE"   # the key, if any, lives in this file only; keep it private

PLIST="$HOME/Library/LaunchAgents/link.jford.into-frame.llama-server.plist"
if [ "$LAUNCHD" = 1 ]; then
    say "launchd agent -> $PLIST"
    mkdir -p "$HOME/Library/LaunchAgents"
    cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>link.jford.into-frame.llama-server</string>
  <key>ProgramArguments</key><array><string>/bin/bash</string><string>$SERVE</string></array>
  <key>EnvironmentVariables</key><dict><key>PATH</key><string>/opt/homebrew/bin:/usr/bin:/bin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$CFG/llama-server.log</string>
  <key>StandardErrorPath</key><string>$CFG/llama-server.log</string>
</dict></plist>
EOF
    launchctl bootout "gui/$(id -u)/link.jford.into-frame.llama-server" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "started under launchd; log: $CFG/llama-server.log"
    echo "stop:  launchctl bootout gui/$(id -u)/link.jford.into-frame.llama-server"
else
    say "starting llama-server in the background (no launchd)"
    nohup bash "$SERVE" >> "$CFG/llama-server.log" 2>&1 &
    echo "pid $!; log: $CFG/llama-server.log"
fi

# 4. wait, then one timed vision call --------------------------------------------------
say "waiting for the server (the first load reads ${NEED_GB} GB from disk; minutes, not seconds)"
AUTH=(); [ -n "$API_KEY" ] && AUTH=(-H "Authorization: Bearer $API_KEY")
for _ in $(seq 1 180); do
    if curl -fsS "${AUTH[@]}" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
    sleep 5
done
curl -fsS "${AUTH[@]}" "http://127.0.0.1:$PORT/v1/models" >/dev/null || die "server did not come up; see $CFG/llama-server.log"

say "one vision call, timed"
# A 64x64 PNG with a red square on green, built with python3 (ships with macOS).
PNG_B64=$(python3 - <<'PY'
import base64, struct, zlib
w = h = 64
rows = []
for y in range(h):
    row = bytearray([0])
    for x in range(w):
        row += bytes((200, 30, 30)) if 16 <= x < 48 and 16 <= y < 48 else bytes((40, 160, 60))
    rows.append(bytes(row))
raw = b"".join(rows)
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
print(base64.b64encode(png).decode())
PY
)
BODY=$(cat <<EOF
{"model":"qwen3-vl","max_tokens":120,"temperature":0.2,
 "messages":[{"role":"user","content":[
   {"type":"image_url","image_url":{"url":"data:image/png;base64,$PNG_B64"}},
   {"type":"text","text":"Describe this image in one sentence: the colours and the shape."}]}]}
EOF
)
T0=$(date +%s.%N)
REPLY=$(curl -fsS "${AUTH[@]}" -H 'Content-Type: application/json' -d "$BODY" "http://127.0.0.1:$PORT/v1/chat/completions")
T1=$(date +%s.%N)
python3 - "$REPLY" "$T0" "$T1" <<'PY'
import json, sys
j = json.loads(sys.argv[1]); dt = float(sys.argv[3]) - float(sys.argv[2])
u = j.get("usage", {}); out = u.get("completion_tokens", 0)
print("reply:", j["choices"][0]["message"]["content"].strip())
print(f"prompt tokens {u.get('prompt_tokens', '?')}, completion tokens {out}, wall {dt:.1f} s, "
      f"{out / dt:.1f} tokens/s end to end (includes image encoding)")
PY

say "done"
HOST=$(scutil --get LocalHostName 2>/dev/null || hostname -s)
cat <<EOF
On the Into Frame box:
  scripts/local-models/check-endpoint.sh http://${HOST}.local:$PORT/v1
and in server/config.yaml, for any role that should use the Mac:
  model: openai:qwen3-vl
  base_url: http://${HOST}.local:$PORT/v1
EOF
[ -n "$API_KEY" ] && echo "  (export OPENAI_API_KEY to the same key before running the pipeline; do not put it in config.yaml)"
exit 0
