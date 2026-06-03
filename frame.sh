#!/bin/bash
# frame.sh — unified CLI for Into Frame (setup, inference, server, remote)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER_DIR="$SCRIPT_DIR/server"
SCRIPTS_DIR="$SCRIPT_DIR/scripts"

# --- Global defaults ---
ENV="frame"
SEED_ARGS=()

# --- Shared defaults ---
PORT="${PORT:-8080}"
ASSET_PORT="${ASSET_PORT:-3000}"
OUTPUT="./output"
DEBUG=""
CONFIG=""

# --- run defaults ---
INPUT="$SCRIPT_DIR/samples/Paris.jpg"

# --- server defaults ---
HOST="localhost"

# --- remote defaults (passed through to scripts/remote-server.sh) ---
REMOTE_USER="${REMOTE_USER:-admin}"
REMOTE_HOST="${REMOTE_HOST:-your-remote-host}"
REMOTE_DIR="${REMOTE_DIR:-~/Research/Into-Frame/server}"
SSH_PORT=""
SSH_KEY=""

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
usage() {
  cat <<EOF
Usage: $(basename "$0") [GLOBAL OPTIONS] [SUBCOMMAND [OPTIONS]]

Subcommands (default: run):
  run       Run the pipeline on an image
  server    Start the local generation server
  download  Download all models for the pipeline
  remote    Start the server on a remote machine via SSH with port forwarding
  setup     Install dependencies and set up the conda environments
  clear     Remove cached output files

Global options:
  --env ENV          Conda environment name  (default: ${ENV})
  --seed VALUE       Random seed (bare integer) or STAGE:VALUE pair; repeatable
  -h, --help         Show this help message

run options:
  INPUT              Input image or directory  (default: ${INPUT})
  -o, --output DIR   Output directory          (default: ${OUTPUT})
  -d, --debug BOOL   Save intermediate files   (default: true)
  --config PATH      Pipeline config YAML      (default: server/config.yaml)

server options:
  --host HOST        Bind host                 (default: ${HOST})
  --port PORT        Server port               (default: ${PORT})
  --asset-port PORT  Asset server port         (default: ${ASSET_PORT})
  -o, --output DIR   Output directory          (default: ${OUTPUT})
  -d, --debug BOOL   Save intermediate files   (default: false)
  --config PATH      Pipeline config YAML      (default: server/config.yaml)

download options:
  --config PATH      Pipeline config YAML      (default: server/config.yaml)

clear options:
  -o, --output DIR   Output directory to clear  (default: server/${OUTPUT})

remote options (SSH + port forwarding, then starts server on the remote):
  SUBCOMMAND         server (default), pull, or clear
  --user USER        Remote SSH username       (default: ${REMOTE_USER})
  --host HOST        Remote hostname or IP     (default: ${REMOTE_HOST})
  --dir DIR          Remote project directory  (default: ${REMOTE_DIR})
  --port PORT        Server port               (default: ${PORT})
  --asset-port PORT  Asset server port         (default: ${ASSET_PORT})
  --ssh-port PORT    SSH port                  (default: 22)
  --key PATH         SSH private key path      (default: none)
  -d, --debug BOOL   Save intermediate files on remote   [server only]
  --config PATH      Remote pipeline config YAML         [server only]
  -o, --output DIR   Output directory to clear           [clear only]

setup options (passed directly to scripts/setup.sh):
  -f                 Force clean reinstall (removes existing envs)
  -s                 Save existing logs instead of wiping them
  -v                 Verbose mode (dump output to terminal)

Environment variables:
  REMOTE_USER, REMOTE_HOST, REMOTE_DIR, ENV, PORT, ASSET_PORT

Examples:
  $(basename "$0") run samples/Iceland.jpg
  $(basename "$0") run samples/Paris.jpg --output ./out --debug true
  $(basename "$0") server --port 9090
  $(basename "$0") download
  $(basename "$0") remote --host 192.168.1.10 --user admin
  $(basename "$0") remote --host 192.168.1.10 --ssh-port 2222 --key ~/.ssh/id_rsa
  $(basename "$0") remote pull --host 192.168.1.10
  $(basename "$0") remote clear --host 192.168.1.10
  $(basename "$0") remote clear --host 192.168.1.10 --output ./my-output
  $(basename "$0") setup
  $(basename "$0") setup -f -v
  $(basename "$0") --env myenv run samples/Paris.jpg --config config_sam3d_test.yaml
  $(basename "$0") clear
  $(basename "$0") clear --output ./my-output
EOF
}

# ---------------------------------------------------------------------------
# Conda activation
# ---------------------------------------------------------------------------
activate_conda() {
  local env="$1"
  local conda_sh=""

  if command -v conda &>/dev/null; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    [[ -f "$base/etc/profile.d/conda.sh" ]] && conda_sh="$base/etc/profile.d/conda.sh"
  fi

  if [[ -z "$conda_sh" ]]; then
    for candidate in \
      "$HOME/miniconda3/etc/profile.d/conda.sh" \
      "$HOME/anaconda3/etc/profile.d/conda.sh" \
      "/opt/miniconda3/etc/profile.d/conda.sh" \
      "/opt/anaconda3/etc/profile.d/conda.sh"; do
      if [[ -f "$candidate" ]]; then
        conda_sh="$candidate"
        break
      fi
    done
  fi

  if [[ -z "$conda_sh" ]]; then
    echo "Error: conda init script not found. Is conda installed?" >&2
    exit 1
  fi

  # shellcheck source=/dev/null
  source "$conda_sh"
  conda activate "$env"
}

# ---------------------------------------------------------------------------
# Build python global-args array
# ---------------------------------------------------------------------------
build_global_args() {
  local args=("--env" "$ENV")
  for s in "${SEED_ARGS[@]+"${SEED_ARGS[@]}"}"; do
    args+=("--seed" "$s")
  done
  echo "${args[@]+"${args[@]}"}"
}

# ---------------------------------------------------------------------------
# Parse global options + detect subcommand
# ---------------------------------------------------------------------------
SUBCOMMAND=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --env)   ENV="$2";          shift 2 ;;
    --seed)  SEED_ARGS+=("$2"); shift 2 ;;
    -h|--help) usage; exit 0 ;;
    run|server|download|remote|setup|clear)
      SUBCOMMAND="$1"; shift; break ;;
    -*) break ;;   # unknown global flag — let subcommand parser handle it
    *)  break ;;   # positional — treat as run INPUT
  esac
done

if [[ -z "$SUBCOMMAND" ]]; then
  usage; exit 0
fi

# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------
if [[ "$SUBCOMMAND" == "run" ]]; then
  if [[ $# -gt 0 && "$1" != -* ]]; then
    INPUT="$1"; shift
  fi

  while [[ $# -gt 0 ]]; do
    case $1 in
      -o|--output) OUTPUT="$2"; shift 2 ;;
      -d|--debug)  DEBUG="$2";  shift 2 ;;
      --config)    CONFIG="$2"; shift 2 ;;
      -h|--help)   usage; exit 0 ;;
      *) echo "Unknown run option: $1" >&2; exit 1 ;;
    esac
  done

  # Resolve relative paths before cd changes the working directory
  [[ "$INPUT" != /* ]] && INPUT="$PWD/$INPUT"
  [[ -n "$OUTPUT" && "$OUTPUT" != /* ]] && OUTPUT="$PWD/$OUTPUT"
  [[ -n "$CONFIG" && "$CONFIG" != /* ]] && CONFIG="$PWD/$CONFIG"

  activate_conda "$ENV"
  cd "$SERVER_DIR"

  ARGS=($(build_global_args) run "$INPUT" --output "$OUTPUT")
  [[ -n "$DEBUG"  ]] && ARGS+=(--debug "$DEBUG")
  [[ -n "$CONFIG" ]] && ARGS+=(--config "$CONFIG")

  exec python3 main.py "${ARGS[@]}"

# ---------------------------------------------------------------------------
# Subcommand: server
# ---------------------------------------------------------------------------
elif [[ "$SUBCOMMAND" == "server" ]]; then
  while [[ $# -gt 0 ]]; do
    case $1 in
      --host)       HOST="$2";       shift 2 ;;
      --port)       PORT="$2";       shift 2 ;;
      --asset-port) ASSET_PORT="$2"; shift 2 ;;
      -o|--output)  OUTPUT="$2";     shift 2 ;;
      -d|--debug)   DEBUG="$2";      shift 2 ;;
      --config)     CONFIG="$2";     shift 2 ;;
      -h|--help)    usage; exit 0 ;;
      *) echo "Unknown server option: $1" >&2; exit 1 ;;
    esac
  done

  [[ -n "$OUTPUT" && "$OUTPUT" != /* ]] && OUTPUT="$PWD/$OUTPUT"
  [[ -n "$CONFIG" && "$CONFIG" != /* ]] && CONFIG="$PWD/$CONFIG"

  activate_conda "$ENV"
  cd "$SERVER_DIR"

  ARGS=($(build_global_args) server --host "$HOST" --port "$PORT" --asset-port "$ASSET_PORT" --output "$OUTPUT")
  [[ -n "$DEBUG"  ]] && ARGS+=(--debug "$DEBUG")
  [[ -n "$CONFIG" ]] && ARGS+=(--config "$CONFIG")

  exec python3 main.py "${ARGS[@]}"

# ---------------------------------------------------------------------------
# Subcommand: download
# ---------------------------------------------------------------------------
elif [[ "$SUBCOMMAND" == "download" ]]; then
  while [[ $# -gt 0 ]]; do
    case $1 in
      --config)  CONFIG="$2"; shift 2 ;;
      -h|--help) usage; exit 0 ;;
      *) echo "Unknown download option: $1" >&2; exit 1 ;;
    esac
  done

  activate_conda "$ENV"
  cd "$SERVER_DIR"

  ARGS=($(build_global_args) download)
  [[ -n "$CONFIG" ]] && ARGS+=(--config "$CONFIG")

  exec python3 main.py "${ARGS[@]}"

# ---------------------------------------------------------------------------
# Subcommand: remote  (delegates to scripts/remote-server.sh)
# ---------------------------------------------------------------------------
elif [[ "$SUBCOMMAND" == "remote" ]]; then
  exec "$SCRIPTS_DIR/remote-server.sh" "$@" --env "$ENV"

# ---------------------------------------------------------------------------
# Subcommand: setup  (delegates to scripts/setup.sh)
# ---------------------------------------------------------------------------
elif [[ "$SUBCOMMAND" == "setup" ]]; then
  exec "$SCRIPTS_DIR/setup.sh" "$@"

# ---------------------------------------------------------------------------
# Subcommand: clear
# ---------------------------------------------------------------------------
elif [[ "$SUBCOMMAND" == "clear" ]]; then
  while [[ $# -gt 0 ]]; do
    case $1 in
      -o|--output) OUTPUT="$2"; shift 2 ;;
      -h|--help)   usage; exit 0 ;;
      *) echo "Unknown clear option: $1" >&2; exit 1 ;;
    esac
  done

  TARGET="$SERVER_DIR/$OUTPUT"
  if [[ ! -d "$TARGET" ]]; then
    echo "Output directory does not exist: $TARGET"
    exit 0
  fi

  echo "Clearing output: $TARGET"
  rm -rf "${TARGET:?}"/*
  echo "Done."
fi
