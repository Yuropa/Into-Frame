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

info "** Installation can take a while to complete. Please be patient... **"

# Give some time to read the comment
sleep 5

CONDA_NAME="frame"
CONDA_ENVS=("$CONDA_NAME" "stablepoint" "trellis2")
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LIB_DIR="$SCRIPT_DIR/lib"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints"
PACKAGES_DIR="$LIB_DIR/packages"

if [ "$FORCE" = true ]; then
    warn "Removing old Conda environments..."
    if [ -d "$LIB_DIR" ]; then
        rm -rf "$LIB_DIR"
    fi

    conda init
    source ~/.bash_profile

    conda deactivate
    for env in "${CONDA_ENVS[@]}"; do
        conda env remove --name "$env" --yes
    done
fi

conda init
source ~/.bash_profile

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

info "Creating Conda environment '$CONDA_NAME'..."
conda create -y -n "$CONDA_NAME" python=3.12 pip setuptools wheel

eval "$(conda shell.bash hook)"
conda activate "$CONDA_NAME"

# Install PyTorch with MPS support (standard pip build includes MPS)
info "Installing PyTorch..."
pip install torch torchvision

# Install standard pip packages
info "Installing pip requirements..."
pip install -r "$SCRIPT_DIR/requirements.txt"
pip install --no-build-isolation git+https://github.com/SunzeY/AlphaCLIP.git

## SAM 3 via Hugging Face transformers (MPS-compatible, no triton dependency)
#info "Installing SAM 3 via transformers..."
#pip install git+https://github.com/huggingface/transformers

# Fix MPS pin_memory bug in transformers SAM3 video processor
#if [[ "$(uname)" == "Darwin" ]]; then
#    warn "Patching MPS compatibility bug..."
#    PROCESSOR_FILE=$(python -c "import transformers.models.sam3_video.processing_sam3_video as m; print(m.__file__)")
#    sed -i '' 's/keep_idx.pin_memory().to(device=out_binary_masks.device/keep_idx.to(device=out_binary_masks.device/' "$PROCESSOR_FILE"
#fi

mkdir -p "$LIB_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$PACKAGES_DIR"

info "Installing SAM 2"
if [ ! -d "$LIB_DIR/sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git "$LIB_DIR/sam2"
fi
pip install -e "$LIB_DIR/sam2"

info "Installing Trellis"
if [ ! -d "$LIB_DIR/TRELLIS.2" ]; then
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive "$LIB_DIR/TRELLIS.2"
fi

TRELLIS_SETUP="$LIB_DIR/TRELLIS.2/setup.sh"
chmod +x "$TRELLIS_SETUP"
bash "$TRELLIS_SETUP" --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

info "Downloading SAM 3D"

if [ ! -d "$CHECKPOINT_DIR/hf" ]; then
    hf download --repo-type model --local-dir "$CHECKPOINT_DIR/hf-download" --max-workers 1  facebook/sam-3d-objects
    mv  "$CHECKPOINT_DIR/hf-download/checkpoints" "$CHECKPOINT_DIR/hf"
    rm -rf "$CHECKPOINT_DIR/hf-download"
fi

info "Installing Depth Anything"
if [ ! -d "$LIB_DIR/depth-anything-3" ]; then
    git clone https://github.com/ByteDance-Seed/depth-anything-3 "$LIB_DIR/depth-anything-3"
fi

if [[ "$(uname)" == "Darwin" ]]; then
    warn "Removing xformers for MPS"
    sed -i '' '/xformers/d' "$LIB_DIR/depth-anything-3/requirements.txt"
    sed -i '' '/"xformers"/d' "$LIB_DIR/depth-anything-3/pyproject.toml"
fi

pip install -e "$LIB_DIR/depth-anything-3"

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

conda deactivate
conda create -n stablepoint python=3.12 -y
conda run -n stablepoint pip install transformers==4.42.3

info "Installing Stable Point 3D"
if [ ! -d "$LIB_DIR/StablePoint" ]; then
    git clone https://github.com/Stability-AI/stable-point-aware-3d --recursive "$LIB_DIR/StablePoint"
fi
conda run -n stablepoint pip install -r "$SCRIPT_DIR/requirements-stable3d.txt"
conda run -n stablepoint pip install --no-build-isolation git+https://github.com/SunzeY/AlphaCLIP.git
conda run -n stablepoint pip install --no-build-isolation -e "$LIB_DIR/StablePoint/texture_baker"
conda run -n stablepoint pip install --no-build-isolation -e "$LIB_DIR/StablePoint/uv_unwrapper"
conda run -n stablepoint pip install --upgrade transparent-background flet
ln -s  "$LIB_DIR/StablePoint/spar3d" "$PACKAGES_DIR/spar3d"

conda deactivate

eval "$(conda shell.bash hook)"
conda activate "$CONDA_NAME"

success ""
success "Setup complete! To start:"
success "  conda activate $CONDA_NAME"
