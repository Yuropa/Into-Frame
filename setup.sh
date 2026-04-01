#!/bin/bash
set -e

# 1. Create a fresh environment with Python
echo "Creating Conda environment 'frame'..."
conda create -y -n frame python=3.10 pip setuptools wheel

# 2. Activate the environment (the hook makes it work inside a script)
eval "$(conda shell.bash hook)"
conda activate frame

# 3. Install the correct PyTorch
echo "Installing PyTorch..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cpu
else
    pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
fi

# 4. Install standard pip packages
echo "Installing pip requirements..."
pip install -r requirements.txt


echo ""
echo "Setup complete! To start:"
echo "  conda activate frame"
