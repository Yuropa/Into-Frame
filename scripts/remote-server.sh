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

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Runs 'python3 main.py server' on a remote machine via SSH,
forwarding local ports to the remote server.

Options:
  --user        Remote SSH username        (default: ${REMOTE_USER})
  --host        Remote hostname or IP      (default: ${REMOTE_HOST})
  --dir         Remote project directory   (default: ${REMOTE_DIR})
  --env         Remote conda environment   (default: ${ENV})
  --port        Server port                (default: ${PORT})
  --asset-port  Asset server port          (default: ${ASSET_PORT})
  --ssh-port    SSH port                   (default: 22)
  --key         Path to SSH private key    (default: none)
  -d, --debug   Save intermediate files    (default: none)
  --config      Remote pipeline config     (default: config.yaml)
  -h, --help    Show this help message

Environment variables:
  REMOTE_USER, REMOTE_HOST, REMOTE_DIR, ENV, PORT, ASSET_PORT
  can all be set as environment variables to override defaults.

Examples:
  $(basename "$0")
  $(basename "$0") --host 192.168.1.10 --user admin
  $(basename "$0") --port 9090 --asset-port 4000
  $(basename "$0") --host 192.168.1.10 --user admin --dir ~/myapp --ssh-port 2222 --key ~/.ssh/my_key
  REMOTE_HOST=192.168.1.10 $(basename "$0")
EOF
}

# Parse optional flags
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
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

# Build the remote python command
REMOTE_PY_ARGS="server --port ${PORT} --asset-port ${ASSET_PORT}"
[[ -n "$DEBUG"  ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --debug $DEBUG"
[[ -n "$CONFIG" ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --config $CONFIG"

REMOTE_CMD="source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV} && cd ${REMOTE_DIR} && python3 main.py ${REMOTE_PY_ARGS}"

# Build optional SSH args
SSH_OPTS="-t"
[[ -n "$SSH_PORT" ]] && SSH_OPTS="$SSH_OPTS -p ${SSH_PORT}"
[[ -n "$SSH_KEY"  ]] && SSH_OPTS="$SSH_OPTS -i ${SSH_KEY}"

echo "Connecting to ${REMOTE_USER}@${REMOTE_HOST} (forwarding :${PORT} and :${ASSET_PORT})..."
# shellcheck disable=SC2086
exec ssh $SSH_OPTS \
  -L "${PORT}:localhost:${PORT}" \
  -L "${ASSET_PORT}:localhost:${ASSET_PORT}" \
  "${REMOTE_USER}@${REMOTE_HOST}" \
  "$REMOTE_CMD"
