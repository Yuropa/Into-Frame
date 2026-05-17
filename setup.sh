#!/bin/bash
FORCE=false

while getopts "f" opt; do
  case $opt in
    f) FORCE=true ;;
    *) echo "Usage: $0 [-f]" >&2; exit 1 ;;
  esac
done

RED='\e[31m'
GREEN='\e[32m'
YELLOW='\e[33m'
BLUE='\e[34m'
CYAN='\e[36m'
BOLD='\e[1m'
RESET='\e[0m'

info()    { printf "${CYAN}%s${RESET}\n" "$*"; }
success() { printf "${GREEN}%s${RESET}\n" "$*"; }
warn()    { printf "${YELLOW}%s${RESET}\n" "$*"; }
error()   { printf "${RED}%s${RESET}\n" "$*"; }
section() {
  local msg="$1"
  info "========================================"
  info "  $msg"
  info "========================================"
}

echo ""

cat << 'EOF'


+-------------------------------------------------------------------------+
|                                                                         |
|                   ___           ___           ___                       |
|       ___        /\__\         /\  \         /\  \                      |
|      /\  \      /::|  |        \:\  \       /::\  \                     |
|      \:\  \    /:|:|  |         \:\  \     /:/\:\  \                    |
|      /::\__\  /:/|:|  |__       /::\  \   /:/  \:\  \                   |
|   __/:/\/__/ /:/ |:| /\__\     /:/\:\__\ /:/__/ \:\__\                  |
|  /\/:/  /    \/__|:|/:/  /    /:/  \/__/ \:\  \ /:/  /                  |
|  \::/__/         |:/:/  /    /:/  /       \:\  /:/  /                   |
|   \:\__\         |::/  /     \/__/         \:\/:/  /                    |
|    \/__/         /:/  /                     \::/  /                     |
|                  \/__/                       \/__/                      |
|       ___           ___           ___           ___           ___       |
|      /\  \         /\  \         /\  \         /\__\         /\  \      |
|     /::\  \       /::\  \       /::\  \       /::|  |       /::\  \     |
|    /:/\:\  \     /:/\:\  \     /:/\:\  \     /:|:|  |      /:/\:\  \    |
|   /::\~\:\  \   /::\~\:\  \   /::\~\:\  \   /:/|:|__|__   /::\~\:\  \   |
|  /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/ |::::\__\ /:/\:\ \:\__\  |
|  \/__\:\ \/__/ \/_|::\/:/  / \/__\:\/:/  / \/__/~~/:/  / \:\~\:\ \/__/  |
|       \:\__\      |:|::/  /       \::/  /        /:/  /   \:\ \:\__\    |
|        \/__/      |:|\/__/        /:/  /        /:/  /     \:\ \/__/    |
|                   |:|  |         /:/  /        /:/  /       \:\__\      |
|                    \|__|         \/__/         \/__/         \/__/      |
|                                                                         |
+-------------------------------------------------------------------------+


EOF

# Make sure conda is installed
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: Conda is not installed or not available in PATH." >&2
    echo "Please install Miniconda or Anaconda first." >&2
    exit 1
fi

info "** Installation can take a while to complete. Please be patient... **"

if sudo -n true 2>/dev/null; then
    # No password prompt
    # Give some time to read the comment
    sleep 5
else
    sudo -v
fi

# Keep sudo alive
while true; do sudo -n true; sleep 60; kill -0 "$$" || exit; done 2>/dev/null &


CONDA_NAME="frame"
BASE_ENV="frame-base"
readonly TORCH_URL="https://download.pytorch.org/whl/cu130"

CONDA_ENVS=("$CONDA_NAME" "$BASE_ENV" "stablepoint" "trellis2" "depthanything" "pano" "cudediff" "dreamcube" "lama")
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LIB_DIR="$SCRIPT_DIR/lib"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
PACKAGES_DIR="$LIB_DIR/packages"
CURRENT_ENV=""

create_base_env() {
    section "Creating base environment"

    conda create -y -q -n "$BASE_ENV" python=3.12 pip setuptools wheel
    conda run -n "$BASE_ENV" pip install torch==2.10.0 torchvision==0.25.0 torchaudio --extra-index-url "$TORCH_URL"
}

create_env() {
    local name="$1"
    local version="${2:-}"

    CURRENT_ENV="$name"
    conda deactivate

    if [[ -n "$version" ]]; then
        conda create -y -q -n "$name" "python=$version" pip setuptools wheel
    else
        conda create -y -q --name "$name" --clone "$BASE_ENV"
    fi
    conda activate "$name" 
}

stop_env() {
    conda deactivate
    conda activate "$CONDA_NAME" 
    CURRENT_ENV=""
}

run_in_env() {
    if [[ -z "${CURRENT_ENV:-}" ]]; then
        error "run_in_env called before create_env"
        return 1
    fi

    conda run -n "$CURRENT_ENV" "$@"
}

source_shell_configs() {
  local found=0
  local file

  for file in \
    ~/.bash_profile \
    ~/.bashrc \
    ~/.zshrc \
    ~/.zprofile
  do
    if [ -f "$file" ]; then
      # shellcheck disable=SC1090
      source "$file"
      found=1
    fi
  done

  if [ "$found" -eq 0 ]; then
    warn "No shell configuration files found" >&2
    return 1
  fi

  return 0
}

clone_if_needed() {
    local repo="$1"
    local dir="$2"
    local extra="${3:-}"

    if [ ! -d "$dir" ]; then
        git clone --recursive $extra "$repo" "$dir"
    fi
}

if [ "$FORCE" = true ]; then
    warn "Removing old Conda environments..."
    if [ -d "$LIB_DIR" ]; then
        rm -rf "$LIB_DIR"
    fi

    conda init
    source_shell_configs

    conda deactivate
    for env in "${CONDA_ENVS[@]}"; do
        conda env remove --name "$env" --yes
    done
fi

conda init
source_shell_configs

# Detect OS and install accordingly
if command -v apt &>/dev/null; then
    sudo apt install -y libwebp-dev
elif command -v dnf &>/dev/null; then
    sudo dnf install -y libwebp-devel
elif command -v pacman &>/dev/null; then
    sudo pacman -S --noconfirm libwebp
elif command -v brew &>/dev/null; then
    brew install webp
else
    echo "WARNING: Could not install libwebp — unsupported package manager"
fi

create_base_env

## ===============
##    Main ENV
## ===============

section "Creating Conda environment '$CONDA_NAME'..."
create_env "$CONDA_NAME"

eval "$(conda shell.bash hook)"
conda activate "$CONDA_NAME"

# Install standard pip packages
section "Installing pip requirements..."
pip install -r "$SCRIPT_DIR/requirements.txt"
pip install --no-build-isolation git+https://github.com/SunzeY/AlphaCLIP.git

mkdir -p "$LIB_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$PACKAGES_DIR"

## =============
##    SAM 2
## =============

section "Installing SAM 2"
clone_if_needed https://github.com/facebookresearch/sam2.git "$LIB_DIR/sam2"
pip install -e "$LIB_DIR/sam2"

## =============
##    TRELLIS
## =============

section "Installing Trellis"
clone_if_needed https://github.com/microsoft/TRELLIS.2.git "$LIB_DIR/TRELLIS.2" -b main

TRELLIS_DIR="$LIB_DIR/TRELLIS.2/"
TRELLIS_SETUP="setup.sh"
chmod +x "$TRELLIS_DIR/$TRELLIS_SETUP"

pushd "$TRELLIS_DIR" > /dev/null || exit 1
create_env "trellis2"
printf "Y\n" | bash "$TRELLIS_SETUP" --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
run_in_env pip install transformers==4.57.6
run_in_env pip install psutil

info "Checking for flash-attn"

VER="2.7.3"
DIR="$HOME/.cache/wheels/flash-attn"
WHEEL="$DIR"/flash_attn-${VER}*.whl

mkdir -p "$DIR"

if ls $WHEEL 1> /dev/null 2>&1; then
    run_in_env pip install $WHEEL
else
    warn "Building flash-attn. This will take a while"
    MAX_JOBS=4 run_in_env pip wheel flash-attn==$VER -w "$DIR" --no-build-isolation
    run_in_env pip install $(ls $WHEEL | head -n 1)
fi

popd > /dev/null || exit 1
ln -sf  "$TRELLIS_DIR/trellis2" "$PACKAGES_DIR/trellis2"
stop_env

## =============
##    SAM 3D
## =============

section "Downloading SAM 3D"

if [ ! -d "$CHECKPOINT_DIR/hf" ]; then
    hf download --repo-type model --local-dir "$CHECKPOINT_DIR/hf-download" --max-workers 1  facebook/sam-3d-objects
    mv  "$CHECKPOINT_DIR/hf-download/checkpoints" "$CHECKPOINT_DIR/hf"
    rm -rf "$CHECKPOINT_DIR/hf-download"
fi

conda deactivate

## ======================
##    Depth Anything
## ======================

section "Installing Depth Anything"
clone_if_needed https://github.com/ByteDance-Seed/depth-anything-3 "$LIB_DIR/depth-anything-3"

warn "Removing xformers"
sed -i '' '/xformers/d' "$LIB_DIR/depth-anything-3/requirements.txt"
sed -i '' '/"xformers"/d' "$LIB_DIR/depth-anything-3/pyproject.toml"

create_env "depthanything" 3.10
pip install -e "$LIB_DIR/depth-anything-3"
stop_env

## ======================
##    Models Download
## ======================

# Hugging Face auth for gated checkpoints
warn ""
warn "⚠️  Model checkpoints require Hugging Face access."
pip install huggingface_hub
python -c "from huggingface_hub import interpreter_login; interpreter_login()"

info "Downloading models..."
python3 "$SCRIPT_DIR/Server/main.py" download

if [ $? -ne 0 ]; then
    error "Error: failed to download models. Access may be required on Hugging Face" >&2
    error "Models will be downloaded later when running pipeline" >&2
fi

conda run -n frame pip install --upgrade --force-reinstall Pillow

## ======================
##    Stable Point 3D
## ======================

section "Installing Stable Point 3D"

create_env "stablepoint" 3.12
run_in_env pip install transformers==4.42.3
clone_if_needed https://github.com/Stability-AI/stable-point-aware-3d "$LIB_DIR/StablePoint"

run_in_env pip install -r "$SCRIPT_DIR/requirements-stable3d.txt"
run_in_env pip install --no-build-isolation git+https://github.com/SunzeY/AlphaCLIP.git
run_in_env pip install --no-build-isolation -e "$LIB_DIR/StablePoint/texture_baker"
run_in_env pip install --no-build-isolation -e "$LIB_DIR/StablePoint/uv_unwrapper"
run_in_env pip install --upgrade transparent-background flet
ln -sf  "$LIB_DIR/StablePoint/spar3d" "$PACKAGES_DIR/spar3d"

stop_env

## ============
##    CubeDiff
## ============

section "Installing CubeDiff"

create_env "cubediff"
clone_if_needed git@github.com:Juan5713/OpenCubeDiff.git "$LIB_DIR/CubeDiff"
run_in_env pip install -r "$SCRIPT_DIR/requirements-cubediff.txt"
ln -sf  "$LIB_DIR/CubeDiff/cubediff" "$PACKAGES_DIR/cubediff"

stop_env

## ============
##    DreamCube
## ============

section "Installing DreamCube"

create_env "dreamcube"
clone_if_needed https://github.com/Yukun-Huang/DreamCube.git "$LIB_DIR/DreamCube"
run_in_env pip install -r "$SCRIPT_DIR/requirements-dreamcube.txt"
run_in_env pip install ninja wheel setuptools
run_in_env pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git"
run_in_env pip install peft
ln -sf  "$LIB_DIR/DreamCube" "$PACKAGES_DIR/dreamcube"

stop_env

## ============
##    Lama
## ============

section "Installing LaMa"

create_env "lama" 3.10
clone_if_needed https://github.com/advimman/lama.git "$LIB_DIR/LaMa"
run_in_env pip install -r "$SCRIPT_DIR/requirements-lama.txt"
run_in_env pip install torchvision
ln -sf  "$LIB_DIR/LaMa" "$PACKAGES_DIR/lama"

LAMA_CHECKPOINT="$CHECKPOINT_DIR/lama"

if [ ! -d "$LAMA_CHECKPOINT" ]; then
    mkdir -p "$LAMA_CHECKPOINT"
    TMP_DIR="$(mktemp -d)"
    ZIP_FILE="$TMP_DIR/big-lama.zip"

    curl -L "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip" -o "$ZIP_FILE"

    unzip "$ZIP_FILE" -d "$LAMA_CHECKPOINT"

    rm -f "$ZIP_FILE"
    rmdir "$TMP_DIR"
fi

stop_env

## ============
## LayerPano3D
## ============

LAYER_PANO_CHECKPOINT="$CHECKPOINT_DIR/layer_pano_3d"
if [ ! -d "$LAYER_PANO_CHECKPOINT" ]; then
    mkdir -p "$LAYER_PANO_CHECKPOINT"
    curl -L "https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA/resolve/main/lora_hubs/pano_lora_720*1440_v1.safetensors?download=true" -o "$LAYER_PANO_CHECKPOINT/pano_lora_720*1440_v1.safetensors"
fi


## ============
##    End
## ============

eval "$(conda shell.bash hook)"
conda activate "$CONDA_NAME"

success ""
success "Setup complete! To start:"
success "  conda activate $CONDA_NAME"
