#!/bin/bash
# Shared Hugging Face download environment helpers.
# Source this from other scripts (e.g. `source "$SCRIPTS_DIR/hf_env.sh"`);
# it isn't meant to be executed directly.

# Speed up Hugging Face model downloads (Xet high-throughput chunking)
export HF_XET_HIGH_PERFORMANCE=1

# Point Hugging Face Hub downloads at a mirror instead of huggingface.co.
# Takes the mirror URL as its only argument (e.g. https://hf-mirror.com).
# Most mirrors don't serve the Xet backend, so Xet is disabled when using one.
# Call this only when a mirror flag is explicitly passed — it should stay
# opt-in, not the default.
configure_hf_mirror() {
    local endpoint="${1:?configure_hf_mirror requires a mirror URL argument}"
    export HF_ENDPOINT="$endpoint"
    export HF_HUB_DISABLE_XET=1
    echo "Using Hugging Face mirror ($endpoint) for model downloads" >&2
}
