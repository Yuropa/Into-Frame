#!/bin/bash

# Defaults
REMOTE_USER="${REMOTE_USER:-admin}"
REMOTE_HOST="${REMOTE_HOST:-your-remote-host}"
REMOTE_DIR="${REMOTE_DIR:-~/Research/Into-Frame/server}"
ENV="${ENV:-frame}"
PORT="${PORT:-8080}"
ASSET_PORT="${ASSET_PORT:-3000}"
SSH_PORT=""
SSH_KEY=""
DEBUG=""
CONFIG=""
OUTPUT="output"

usage() {
  cat <<EOF
Usage: $(basename "$0") [ACTION] [OPTIONS]

Actions (default: server):
  server   Start the generation server on the remote machine (with port forwarding)
  pull     Pull the latest git changes on the remote machine
  clear    Clear the remote output directory

Common options:
  --user        Remote SSH username        (default: ${REMOTE_USER})
  --host        Remote hostname or IP      (default: ${REMOTE_HOST})
  --dir         Remote project directory   (default: ${REMOTE_DIR})
  --env         Remote conda environment   (default: ${ENV})
  --ssh-port    SSH port                   (default: 22)
  --key         Path to SSH private key    (default: none)
  -h, --help    Show this help message

server options:
  --port        Server port                (default: ${PORT})
  --asset-port  Asset server port          (default: ${ASSET_PORT})
  -d, --debug   Save intermediate files    (default: none)
  --config      Remote pipeline config     (default: config.yaml)

clear options:
  -o, --output  Output directory to clear  (default: ${OUTPUT})

Environment variables:
  REMOTE_USER, REMOTE_HOST, REMOTE_DIR, ENV, PORT, ASSET_PORT

Examples:
  $(basename "$0")
  $(basename "$0") server --host 192.168.1.10 --user admin
  $(basename "$0") server --port 9090 --asset-port 4000
  $(basename "$0") pull --host 192.168.1.10
  $(basename "$0") clear --host 192.168.1.10
  $(basename "$0") clear --host 192.168.1.10 --output my-output
EOF
}

# Detect optional action positional arg
ACTION="server"
if [[ $# -gt 0 && "$1" != -* ]]; then
  ACTION="$1"; shift
fi

# Parse flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --user)        REMOTE_USER="$2";  shift 2 ;;
    --host)        REMOTE_HOST="$2";  shift 2 ;;
    --dir)         REMOTE_DIR="$2";   shift 2 ;;
    --env)         ENV="$2";          shift 2 ;;
    --port)        PORT="$2";         shift 2 ;;
    --asset-port)  ASSET_PORT="$2";   shift 2 ;;
    --ssh-port)    SSH_PORT="$2";     shift 2 ;;
    --key)         SSH_KEY="$2";      shift 2 ;;
    -d|--debug)    DEBUG="$2";        shift 2 ;;
    --config)      CONFIG="$2";       shift 2 ;;
    -o|--output)   OUTPUT="$2";       shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

# Build SSH options string
SSH_OPTS="-t"
[[ -n "$SSH_PORT" ]] && SSH_OPTS="$SSH_OPTS -p ${SSH_PORT}"
[[ -n "$SSH_KEY"  ]] && SSH_OPTS="$SSH_OPTS -i ${SSH_KEY}"

case "$ACTION" in
  server)
    REMOTE_PY_ARGS="server --port ${PORT} --asset-port ${ASSET_PORT}"
    [[ -n "$DEBUG"  ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --debug $DEBUG"
    [[ -n "$CONFIG" ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --config $CONFIG"

    REMOTE_CMD="source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV} && cd ${REMOTE_DIR} && python3 main.py ${REMOTE_PY_ARGS}"

    echo "Connecting to ${REMOTE_USER}@${REMOTE_HOST} (forwarding :${PORT} and :${ASSET_PORT})..."
    # shellcheck disable=SC2086
    exec ssh $SSH_OPTS \
      -L "${PORT}:localhost:${PORT}" \
      -L "${ASSET_PORT}:localhost:${ASSET_PORT}" \
      "${REMOTE_USER}@${REMOTE_HOST}" \
      "$REMOTE_CMD"
    ;;

  pull)
    REMOTE_REPO="$(dirname "$REMOTE_DIR")"
    REMOTE_CMD="git -C $REMOTE_REPO pull"

    echo "Pulling latest on ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_REPO}..."
    # shellcheck disable=SC2086
    exec ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "$REMOTE_CMD"
    ;;

  clear)
    REMOTE_TARGET="${REMOTE_DIR}/${OUTPUT#./}"
    REMOTE_CMD="if [ -d $REMOTE_TARGET ]; then rm -rf $REMOTE_TARGET/* && echo 'Done.'; else echo \"Output directory does not exist: $REMOTE_TARGET\"; fi"

    echo "Clearing output on ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_TARGET}..."
    # shellcheck disable=SC2086
    exec ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "$REMOTE_CMD"
    ;;

  *)
    echo "Unknown action: $ACTION" >&2
    echo "Valid actions: server, pull, clear" >&2
    exit 1
    ;;
esac
