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
SEEDS=()

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
  --seed        Random seed (repeatable)   VALUE or STAGE:VALUE (default: none)

clear options:
  -o, --output  Output directory to clear  (default: ${OUTPUT})

Environment variables:
  REMOTE_USER, REMOTE_HOST, REMOTE_DIR, ENV, PORT, ASSET_PORT

Examples:
  $(basename "$0")
  $(basename "$0") server --host 192.168.1.10 --user admin
  $(basename "$0") server --port 9090 --asset-port 4000
  $(basename "$0") server --seed 12345
  $(basename "$0") server --seed sceneGeneration:1 --seed treeGeneration:2
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
    --seed)        SEEDS+=("$2");     shift 2 ;;
    -o|--output)   OUTPUT="$2";       shift 2 ;;
    -i|--input)    INPUT="$2";        shift 2 ;;
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
    SEED_ARGS=""
    for s in "${SEEDS[@]}"; do
      SEED_ARGS="$SEED_ARGS --seed $s"
    done

    # --output was parsed above but never forwarded, so `remote --output X` silently
    # did nothing and the server fell back to main.py's default of "./output",
    # relative to REMOTE_DIR (the *server* subdir). Whether that resolves to the same
    # cache `frame.sh run` wrote depends entirely on server/output being a symlink to
    # ../output -- which is gitignored and therefore absent on a fresh clone, so
    # main.py just mkdir'd an empty server/output and regenerated the whole pipeline.
    # Resolved against REMOTE_DIR exactly as the clear action below does, so the two
    # can no longer disagree about which directory they mean.
    REMOTE_OUT="${OUTPUT}"
    [[ "$REMOTE_OUT" != /* ]] && REMOTE_OUT="${REMOTE_DIR}/${OUTPUT#./}"

    # Same story as --output: the pipeline cache lives at output/<md5 of the input
    # file's bytes>, so a server started without --input silently serves main.py's
    # default (Mount Rainier.jpg) and lands on a different cache key than whatever
    # `frame.sh run` generated -- which looks exactly like the cache being ignored.
    # Relative paths resolve against REMOTE_DIR, since the remote cd's there first.
    # Default mirrors frame.sh's own INPUT default (samples/Paris.jpg) so that
    # `frame.sh run` and `frame.sh remote` land on the SAME cache key by
    # construction, resolved against the remote repo root rather than a local path.
    REMOTE_IN="${INPUT:-$(dirname "$REMOTE_DIR")/samples/Paris.jpg}"
    [[ -n "$REMOTE_IN" && "$REMOTE_IN" != /* && "$REMOTE_IN" != "~"* ]] \
      && REMOTE_IN="${REMOTE_DIR}/${REMOTE_IN#./}"

    REMOTE_PY_ARGS="server --port ${PORT} --asset-port ${ASSET_PORT} --output ${REMOTE_OUT}"
    [[ -n "$REMOTE_IN" ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --input ${REMOTE_IN}"
    [[ -n "$DEBUG"  ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --debug $DEBUG"
    [[ -n "$CONFIG" ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --config $CONFIG"

    # --seed is a top-level main.py flag, so it must precede the "server" subcommand
    REMOTE_CMD="source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV} && cd ${REMOTE_DIR} && python3 main.py${SEED_ARGS} ${REMOTE_PY_ARGS}"

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
