#!/bin/bash
set -e

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

 ** Installation can take a while to complete. Please be patient...


EOF 

SCRIPT_DIR="$(dirname "$(realpath "$0")")"

echo "Creating Conda environment 'frame'..."
conda create -y -n frame python=3.12 pip setuptools wheel

eval "$(conda shell.bash hook)"
conda activate frame

# Install PyTorch with MPS support (standard pip build includes MPS)
echo "Installing PyTorch..."
pip install torch torchvision

# Install standard pip packages
echo "Installing pip requirements..."
pip install -r "$SCRIPT_DIR/requirements.txt"

## SAM 3 via Hugging Face transformers (MPS-compatible, no triton dependency)
#echo "Installing SAM 3 via transformers..."
#pip install git+https://github.com/huggingface/transformers

# Fix MPS pin_memory bug in transformers SAM3 video processor
#if [[ "$(uname)" == "Darwin" ]]; then
#    echo "Patching MPS compatibility bug..."
#    PROCESSOR_FILE=$(python -c "import transformers.models.sam3_video.processing_sam3_video as m; print(m.__file__)")
#    sed -i '' 's/keep_idx.pin_memory().to(device=out_binary_masks.device/keep_idx.to(device=out_binary_masks.device/' "$PROCESSOR_FILE"
#fi

LIB_DIR="$SCRIPT_DIR/lib"
mkdir -p "$LIB_DIR"

echo "Installing SAM 2"
if [ ! -d "$LIB_DIR/sam2" ]; then
    git clone https://github.com/facebookresearch/sam2.git "$LIB_DIR/sam2"
fi
pip install -e "$LIB_DIR/sam2"

echo "Installing Trellis"
if [ ! -d "$LIB_DIR/TRELLIS.2" ]; then
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive "$LIB_DIR/TRELLIS.2"
fi

TRELLIS_SETUP="$LIB_DIR/TRELLIS.2/setup.sh"
chmod +x "$TRELLIS_SETUP"
bash "$TRELLIS_SETUP" --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm

# Hugging Face auth for gated checkpoints
echo ""
echo "⚠️  Model checkpoints require Hugging Face access."
pip install huggingface_hub
python -c "from huggingface_hub import interpreter_login; interpreter_login()"

echo "Downloading models..."
python3 "$SCRIPT_DIR/Server/main.py" download

if [ $? -ne 0 ]; then
    echo "Error: failed to download models. Access may be required on Hugging Face" >&2
    echo "Models will be downloaded later when running pipeline" >&2
fi

echo ""
echo "Setup complete! To start:"
echo "  conda activate frame"