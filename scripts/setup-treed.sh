#!/bin/bash
set -e

TORCH_URL="https://download.pytorch.org/whl/cu130"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LIB_DIR="$PROJECT_DIR/lib"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
TREED_DIR="$LIB_DIR/TreeDFusion"
BASE_ENV="frame-base-310"

warn()    { printf "\e[33m%s\e[0m\n" "$*"; }
info()    { printf "\e[36m%s\e[0m\n" "$*"; }
success() { printf "\e[32m%s\e[0m\n" "$*"; }

# Torch base (Python 3.10)
if ! conda env list | grep -qE "^${BASE_ENV}[[:space:]]"; then
    info "Creating torch base env (Python 3.10)..."
    conda create -y -q -n "$BASE_ENV" python=3.10 pip setuptools wheel
    conda run --no-capture-output -n "$BASE_ENV" pip install \
        torch torchvision torchaudio --extra-index-url "$TORCH_URL"
fi

# Clone TreeDFusion
if [ ! -d "$TREED_DIR" ]; then
    info "Cloning TreeDFusion..."
    git clone --recursive https://github.com/JaeLee18/TreeDFusion_ECCV24.git "$TREED_DIR"
fi

# Patch C++ standard: CUDA 13 requires C++17, Magic123 hardcodes C++14
sed -i 's/std=c++14/std=c++17/g' "$TREED_DIR/Magic123/gridencoder/backend.py"
sed -i 's/std=c++14/std=c++17/g' "$TREED_DIR/Magic123/raymarching/backend.py"

# PyTorch 2.6 changed torch.load to default weights_only=True, breaking the zero123 checkpoint
sed -i "s/torch.load(ckpt, map_location='cpu')/torch.load(ckpt, map_location='cpu', weights_only=False)/" "$TREED_DIR/Magic123/guidance/zero123_utils.py"

# Symlink tree_lora to the subprocess cwd so diffusers finds it without a network lookup
ln -sfn Magic123/guidance/tree_lora "$TREED_DIR/tree_lora"

# Checkpoint
CKPT_DIR="$CHECKPOINT_DIR/treed"
if [ ! -d "$CKPT_DIR" ] || [ -z "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
    info "Downloading TreeD checkpoint..."
    mkdir -p "$CKPT_DIR"
    conda run --no-capture-output -n frame pip install -q gdown
    conda run --no-capture-output -n frame gdown "1tHWbX-TdkS4JCg6MrUu-BAUDjPDTaMyQ" -O "$CKPT_DIR/"
fi

# Remove and recreate treed env
if conda env list | grep -qE "^treed[[:space:]]"; then
    info "Removing existing treed env..."
    conda env remove -n treed -y
fi
info "Creating treed env..."
conda create -y -q --name treed --clone "$BASE_ENV"

R="conda run --no-capture-output -n treed"

# Core packages — pin transformers<4.45 to avoid torchaudio import via loss_rnnt.py
info "Installing core packages..."
$R pip install \
    numpy scipy scikit-learn matplotlib pandas \
    opencv-python imageio imageio-ffmpeg \
    tqdm rich einops tensorboard tensorboardX \
    huggingface_hub "diffusers>=0.9.0,<0.31" accelerate "transformers<4.45" peft \
    torch-ema xatlas PyMCubes trimesh pymeshlab \
    omegaconf pytorch-lightning kornia \
    "timm==0.6.7" easydict termcolor psutil lpips \
    sentencepiece wandb rembg Pillow

# Git-based deps
info "Installing git-based deps..."
$R pip install "git+https://github.com/openai/CLIP.git"                   || warn "CLIP failed"
$R pip install --no-build-isolation "git+https://github.com/NVlabs/nvdiffrast/" || warn "nvdiffrast failed"
$R pip install taming-transformers-rom1504                                 || warn "taming-transformers failed"
$R pip install carvekit-colab                                              || warn "carvekit-colab failed"

# Re-pin torch/torchaudio — other installs may have upgraded them
info "Re-pinning torch to cu130..."
$R pip install torch torchvision torchaudio --extra-index-url "$TORCH_URL"

# Pre-download SD 1.5 so the cache is complete before Magic123 runs.
info "Pre-downloading stable-diffusion-v1-5..."
$R python - <<'EOF'
from diffusers import StableDiffusionPipeline
import torch
StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
EOF

success "treed env ready."
