#!/bin/bash
set -e

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

# SAM 3 via Hugging Face transformers (MPS-compatible, no triton dependency)
echo "Installing SAM 3 via transformers..."
pip install git+https://github.com/huggingface/transformers

# Fix MPS pin_memory bug in transformers SAM3 video processor
echo "Patching MPS compatibility bug..."
PROCESSOR_FILE=$(python -c "import transformers.models.sam3_video.processing_sam3_video as m; print(m.__file__)")
sed -i '' 's/keep_idx.pin_memory().to(device=out_binary_masks.device/keep_idx.to(device=out_binary_masks.device/' "$PROCESSOR_FILE"

# Hugging Face auth for gated checkpoints
echo ""
echo "⚠️  SAM 3 checkpoints require Hugging Face access."
echo "   Request access at: https://huggingface.co/facebook/sam3"
pip install huggingface_hub
python -c "from huggingface_hub import interpreter_login; interpreter_login()"

echo ""
echo "Setup complete! To start:"
echo "  conda activate frame"