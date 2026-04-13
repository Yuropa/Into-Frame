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
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
LIB_DIR="$SCRIPT_DIR/lib"

if [ "$FORCE" = true ]; then
    warn "Removing old Conda environment '$CONDA_NAME'..."
    if [ -d "$LIB_DIR" ]; then
        rm -rf "$LIB_DIR"
    fi

    conda init
    source ~/.bash_profile
    conda deactivate

    conda env remove --name "$CONDA_NAME" --yes 
fi

conda init
source ~/.bash_profile

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
bash "$TRELLIS_SETUP" --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

info "Installing Depth Anything"
if [ ! -d "$LIB_DIR/depth-anything-3" ]; then
    git clone https://github.com/ByteDance-Seed/depth-anything-3 "$LIB_DIR/depth-anything-3"
fi

if [[ "$(uname)" == "Darwin" ]]; then
    warn "Removing xformers for MPS"
    sed -i '' '/xformers/d' "$LIB_DIR/depth-anything-3/requirements.txt"
    sed -i '' '/"xformers"/d' "$LIB_DIR/depth-anything-3/pyproject.toml"
    sed -i '' 's/from xformers.ops import SwiGLU/try:\n    from xformers.ops import SwiGLU\nexcept ImportError:\n    SwiGLU = None/' "$LIB_DIR/depth-anything-3/src/depth_anything_3/model/dinov2/layers/swiglu_ffn.py"
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

success ""
success "Setup complete! To start:"
success "  conda activate $CONDA_NAME"