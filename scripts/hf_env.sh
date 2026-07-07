#!/bin/bash
# Shared Hugging Face download environment helpers.
# Source this from other scripts (e.g. `source "$SCRIPTS_DIR/hf_env.sh"`);
# it isn't meant to be executed directly.

# Speed up Hugging Face model downloads (Xet high-throughput chunking)
export HF_XET_HIGH_PERFORMANCE=1

# Point Hugging Face Hub downloads at the hf-mirror.com mirror instead of
# huggingface.co. hf-mirror.com doesn't serve the Xet backend, so Xet must
# be disabled when using it. Call this only when a mirror flag is explicitly
# passed — it should stay opt-in, not the default.
configure_hf_mirror() {
    export HF_ENDPOINT="https://hf-mirror.com"
    export HF_HUB_DISABLE_XET=1
    echo "Using Hugging Face mirror (hf-mirror.com) for model downloads" >&2
}
