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
# Forwarded to main.py as a global --log-mode. Every other frame.sh subcommand
# already accepts -v/--plain/--log-mode; `remote` passes its argv straight here,
# so without these cases `frame.sh remote -v` died on "Unknown argument: -v" --
# which is exactly the flag the pipeline's own model-download stall hint tells
# you to re-run with.
LOG_MODE=""

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
  -v, --verbose Verbose remote logging (shows model download progress)
  --plain       Plain remote logging (no panel UI)
  --log-mode    panel | plain | verbose    (default: panel)
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
    -v|--verbose)  LOG_MODE="verbose"; shift ;;
    --plain)       LOG_MODE="plain";   shift ;;
    --log-mode)    LOG_MODE="$2";     shift 2 ;;
    -h|--help)     usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

# Quote a path for the REMOTE shell.
#
# Everything below is assembled into one string that ssh hands to the remote
# shell, so an unquoted path with a space in it gets word-split there: the
# default sample is literally "Mount Rainier.jpg", which arrived as
# `--input .../Mount` plus a stray `Rainier.jpg` argument and made argparse fail
# on a file that exists. The leading `~` has to stay OUTSIDE the quotes or it
# stops being expanded and you get a literal ~ directory instead.
quote_remote_path() {
  local p="$1" rest
  # Escape any embedded single quotes: close, emit an escaped quote, reopen.
  if [[ "$p" == "~/"* ]]; then
    rest="${p#\~/}"
    printf "~/'%s'" "${rest//\'/\'\\\'\'}"
  else
    printf "'%s'" "${p//\'/\'\\\'\'}"
  fi
}

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
    # Default mirrors frame.sh's own INPUT default so that `frame.sh run` and
    # `frame.sh remote` land on the SAME cache key by construction.
    #
    # That default resolves against REMOTE_DIR (the *server* dir), not its parent:
    # the samples live in server/samples/, and there is no samples/ at the repo
    # root, so the previous $(dirname "$REMOTE_DIR")/samples/... pointed at a path
    # that has never existed -- every no-input remote run died on FileNotFoundError
    # before reaching the pipeline. Resolving here the same way an explicit
    # relative --input does also removes the disagreement between the two, which
    # previously sent them to different directories.
    #
    # Mount Rainier matches main.py's own --input default, so a bare `frame.sh
    # remote` and a bare `python3 main.py server` mean the same scene.
    REMOTE_IN="${INPUT:-samples/Mount Rainier.jpg}"
    [[ -n "$REMOTE_IN" && "$REMOTE_IN" != /* && "$REMOTE_IN" != "~"* ]] \
      && REMOTE_IN="${REMOTE_DIR}/${REMOTE_IN#./}"

    REMOTE_PY_ARGS="server --port ${PORT} --asset-port ${ASSET_PORT} --output $(quote_remote_path "$REMOTE_OUT")"
    [[ -n "$REMOTE_IN" ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --input $(quote_remote_path "$REMOTE_IN")"
    [[ -n "$DEBUG"  ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --debug $DEBUG"
    [[ -n "$CONFIG" ]] && REMOTE_PY_ARGS="$REMOTE_PY_ARGS --config $(quote_remote_path "$CONFIG")"

    # --seed and --log-mode are top-level main.py flags, so both must precede the
    # "server" subcommand.
    GLOBAL_ARGS="$SEED_ARGS"
    [[ -n "$LOG_MODE" ]] && GLOBAL_ARGS="$GLOBAL_ARGS --log-mode $LOG_MODE"

    REMOTE_CMD="source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${ENV} && cd $(quote_remote_path "$REMOTE_DIR") && python3 main.py${GLOBAL_ARGS} ${REMOTE_PY_ARGS}"

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
    REMOTE_CMD="git -C $(quote_remote_path "$REMOTE_REPO") pull"

    echo "Pulling latest on ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_REPO}..."
    # shellcheck disable=SC2086
    exec ssh $SSH_OPTS "${REMOTE_USER}@${REMOTE_HOST}" "$REMOTE_CMD"
    ;;

  clear)
    REMOTE_TARGET="${REMOTE_DIR}/${OUTPUT#./}"
    # Quoted for the same reason as the server paths above -- and here it matters
    # more than a failed argparse: an unquoted `rm -rf $REMOTE_TARGET/*` on a path
    # containing a space expands to several targets.
    QUOTED_TARGET="$(quote_remote_path "$REMOTE_TARGET")"
    REMOTE_CMD="if [ -d $QUOTED_TARGET ]; then rm -rf $QUOTED_TARGET/* && echo 'Done.'; else echo \"Output directory does not exist: $REMOTE_TARGET\"; fi"

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
