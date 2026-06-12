#!/bin/bash
set -e
FORCE=false
SAVE_LOGS=false
VERBOSE=false

# Current directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"

show_usage() {
    echo -e "Usage: $(basename "$0") [OPTIONS]"
    echo -e ""
    echo -e "Options:"
    echo -e "  -f    Force clean installation (removes existing libraries and conda environments)"
    echo -e "  -s    Save existing installation logs (default behavior wipes the logs directory)"
    echo -e "  -v    Verbose mode (dumps output to terminal instead of a log file)"
    echo -e "  -h    Show this help message and exit"
    exit 0
}

# Parse command line flags
while getopts "fshv" opt; do
  case $opt in
    f) FORCE=true ;;
    s) SAVE_LOGS=true ;;
    v) VERBOSE=true ;;
    h) show_usage ;;
    *) echo "Invalid option. Use -h for help." >&2; exit 1 ;;
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
|                           ___           ___           ___               |
|               ___        /\__\         /\  \         /\  \              |
|              /\  \      /::|  |        \:\  \       /::\  \             |
|              \:\  \    /:|:|  |         \:\  \     /:/\:\  \            |
|              /::\__\  /:/|:|  |__       /::\  \   /:/  \:\  \           |
|           __/:/\/__/ /:/ |:| /\__\     /:/\:\__\ /:/__/ \:\__\          |
|          /\/:/  /    \/__|:|/:/  /    /:/  \/__/ \:\  \ /:/  /          |
|          \::/__/         |:/:/  /    /:/  /       \:\  /:/  /           |
|           \:\__\         |::/  /     \/__/         \:\/:/  /            |
|            \/__/         /:/  /                     \::/  /             |
|                          \/__/                       \/__/              |
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

# Setup logs
if [ "$SAVE_LOGS" = true ]; then
    # Saving prior execution logs
    mkdir -p "$LOG_DIR"
else
    if [ -d "$LOG_DIR" ]; then
        rm -rf "$LOG_DIR"
    fi
    mkdir -p "$LOG_DIR"
fi
LOG_FILE="$LOG_DIR/install-$(date +%Y%m%d-%H%M%S).log"

touch "$LOG_FILE"
echo ""
if [ "$VERBOSE" = true ]; then
    echo "Verbose mode enabled. Outputting directly to terminal (also logging to $LOG_FILE)"
else
    echo "Logging to file: $LOG_FILE"
fi

CONDA_NAME="frame"
BASE_ENV_PREFIX="frame-base"
DEFAULT_PYTHON="3.12"
TORCH_BASE_VERSIONS=("3.12" "3.10")
readonly TORCH_URL="https://download.pytorch.org/whl/cu130"

CONDA_ENVS=("$CONDA_NAME" "stablepoint" "trellis2" "depthanything" "pano" "cubediff" "dreamcube" "lama" "depthpano" "sam3d" "recognize" "lux-dit" "worldgen")
for _v in "${TORCH_BASE_VERSIONS[@]}"; do CONDA_ENVS+=("${BASE_ENV_PREFIX}-${_v//./}"); done
unset _v
LIB_DIR="$PROJECT_DIR/lib"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
PACKAGES_DIR="$LIB_DIR/packages"
REQUIREMENTS_DIR="$PROJECT_DIR/requirements"
CURRENT_ENV=""

FLASH_WHEEL_DIR="$HOME/.cache/wheels/flash-attn"
FLASH_VERSION_SAM3D="2.8.3"
P3D_WHEEL_DIR="$HOME/.cache/wheels/pytorch3d"
P3D_COMMIT="75ebeeaea0908c5527e7b1e305fbc7681382db47"

load_conda() {
    eval "$(conda shell.bash hook)" 2>/dev/null || true
}

_torch_base_name() {
    echo "${BASE_ENV_PREFIX}-${1//./}"
}

_has_torch() {
    conda run --no-capture-output -n "$1" python -c "import torch" 2>/dev/null
}

ensure_torch_base() {
    local version="$1"
    local base
    base="$(_torch_base_name "$version")"

    if ! conda env list | grep -qE "^${base}[[:space:]]"; then
        conda create -y -q -n "$base" "python=$version" pip setuptools wheel
    fi

    if ! _has_torch "$base"; then
        conda run --no-capture-output -n "$base" pip install \
            torch torchvision torchaudio \
            --extra-index-url "$TORCH_URL" \
            || { error "pip install torch failed in '$base'"; exit 1; }
        _has_torch "$base" \
            || { error "torch installed but still not importable in '$base'"; exit 1; }
    fi
}

setup_torch_bases() {
    for version in "${TORCH_BASE_VERSIONS[@]}"; do
        info "Setting up torch base for Python $version..."
        ensure_torch_base "$version"
    done
}

create_env() {
    local name="$1"
    local version="${2:-$DEFAULT_PYTHON}"

    CURRENT_ENV="$name"
    load_conda
    conda deactivate || true

    ensure_torch_base "$version"

    if conda env list | grep -qE "^${name}[[:space:]]"; then
        if ! _has_torch "$name"; then
            conda run --no-capture-output -n "$name" pip install \
                torch torchvision torchaudio \
                --extra-index-url "$TORCH_URL" \
                || { error "pip install torch failed in '$name'"; exit 1; }
        fi
        conda activate "$name"
        return 0
    fi

    conda create -y -q --name "$name" --clone "$(_torch_base_name "$version")"
    conda activate "$name"
}

stop_env() {
    load_conda
    conda deactivate || true
    conda activate "$CONDA_NAME"
    CURRENT_ENV=""
}

run_in_env() {
    if [[ -z "${CURRENT_ENV:-}" ]]; then
        error "run_in_env called before create_env"
        exit 1
    fi

    conda run --no-capture-output -n "$CURRENT_ENV" "$@" \
        || { error "Command failed in '$CURRENT_ENV': $*"; exit 1; }
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
      source "$file" || true
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
    shift 2

    if [ ! -d "$dir" ]; then
        git clone --recursive "$@" "$repo" "$dir"
    fi
}

download_checkpoint() {
    local url="$1"
    local dir="$2"

    local filename
    filename=$(basename "${url%%\?*}")

    local local_checkpoint_dir="$CHECKPOINT_DIR/$dir"
    if [ ! -d "$local_checkpoint_dir" ]; then
        mkdir -p "$local_checkpoint_dir"
        curl -L "$url" -o "$local_checkpoint_dir/$filename"
    fi
}

# Progress
CURRENT_STEP=0
TOTAL_STEPS=$(( $(grep -c "^run_step" "$0") - 1 ))

spinner() {
    local pid=$1
    local msg="$2"
    
    # Define the characters as separate items in a sequence
    local spin=("|" "/" "-" "\\")
    local delay=0.1

    # Hide the cursor so it looks cleaner
    printf "\e[?25l"

    while kill -0 "$pid" 2>/dev/null; do
        # Loop over the indices of the array
        for c in "${spin[@]}"; do
            printf "\r${BLUE}[%s]${RESET} %s\e[K" "$c" "$msg"
            sleep "$delay"
            
            if ! kill -0 "$pid" 2>/dev/null; then break; fi
        done
    done

    # Restore the cursor and clear the line one last time
    printf "\e[?25h\r\e[K"
}

run_step() {
    local desc="$1"
    shift

    CURRENT_STEP=$((CURRENT_STEP + 1))
    local timestamp=$(date +"%H:%M:%S")

    echo ""
    printf "${BLUE}[$CURRENT_STEP/$TOTAL_STEPS]${RESET} ($timestamp) ${BOLD}$desc${RESET}\n"

    if [ "$VERBOSE" = true ]; then
        # Verbose Mode: Run in foreground, show output live, still save to log
        # 'tee -a' duplicates stdout to the log file
        # 2>&1 merges stderr into stdout so you see errors too
        "$@" 2>&1 | tee -a "$LOG_FILE"
        local exit_status=${PIPESTATUS[0]} # Gets exit code of "$@", not tee
        
        if [ $exit_status -eq 0 ]; then
            success "✓ Done"
        else
            local end_timestamp=$(date +"%H:%M:%S")
            error "✗ Failed at $end_timestamp (Exit Code: $exit_status)"
            exit 1
        fi
    else
        # Standard Mode: Original background worker + spinner
        "$@" >>"$LOG_FILE" 2>&1 &
        local pid=$!

        spinner "$pid" "Running..."

        wait "$pid"
        local exit_status=$?

        if [ $exit_status -eq 0 ]; then
            success "✓ Done"
        else
            local end_timestamp=$(date +"%H:%M:%S")
            error "✗ Failed at $end_timestamp (Exit Code: $exit_status)"
            error "See log: $LOG_FILE"
            echo ""
            warn "Last 20 log lines:"
            tail -n 20 "$LOG_FILE"
            exit 1
        fi
    fi
}

cleanup_if_needed() {
    if [ "$FORCE" = true ]; then
        warn "Removing old Conda environments..."
        if [ -d "$LIB_DIR" ]; then
            rm -rf "$LIB_DIR"
        fi

        rm -rf "$FLASH_WHEEL_DIR"

        conda init
        source_shell_configs

        EXISTING_ENVS=$(conda env list --json | grep -o '"/[^" ]*' | xargs -L1 basename 2>/dev/null) || true

        conda deactivate || true
        for env in "${CONDA_ENVS[@]}"; do
            if echo "$EXISTING_ENVS" | grep -qxF "$env"; then
                conda env remove --name "$env" --yes
            fi
        done
        # Remove any torch base envs not in CONDA_ENVS (e.g. from version changes)
        while IFS= read -r env; do
            [[ -n "$env" ]] && conda env remove --name "$env" --yes
        done < <(echo "$EXISTING_ENVS" | grep "^${BASE_ENV_PREFIX}-")
    fi
}

run_step "Cleanup" \
    cleanup_if_needed

setup_shell_env() {
    load_conda
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
}

run_step "Setup Shell Environment" \
    setup_shell_env

run_step "Creating Torch Base Environments" \
    setup_torch_bases

## ===============
##    Main ENV
## ===============

create_main_environment() {
    create_env "$CONDA_NAME"

    load_conda
    conda activate "$CONDA_NAME"

    # Install standard pip packages
    conda run --no-capture-output -n frame pip install -r "$REQUIREMENTS_DIR/requirements.txt" \
        || { error "pip install failed in 'frame'"; exit 1; }
    conda run --no-capture-output -n frame pip install --no-build-isolation git+https://github.com/SunzeY/AlphaCLIP.git \
        || { error "AlphaCLIP install failed in 'frame'"; exit 1; }

    mkdir -p "$LIB_DIR"
    mkdir -p "$CHECKPOINT_DIR"
    mkdir -p "$PACKAGES_DIR"
} 

run_step "Creating Conda Environment '$CONDA_NAME'" \
    create_main_environment

## =============
##    SAM 2
## =============

setup_sam2() {
    clone_if_needed https://github.com/facebookresearch/sam2.git "$LIB_DIR/sam2"
    conda run --no-capture-output -n frame pip install -e "$LIB_DIR/sam2"
}

run_step "Installing SAM 2" \
    setup_sam2

## =============
##    TRELLIS
## =============

TRELLIS_DIR="$LIB_DIR/TRELLIS.2"
FLASH_VERSION="2.7.3"

setup_trellis() {
    local setup_script="setup.sh"

    clone_if_needed https://github.com/microsoft/TRELLIS.2.git "$TRELLIS_DIR" -b main
    chmod +x "$TRELLIS_DIR/$setup_script"

    create_env "trellis2"
    
    # Run the setup script safely using --cwd and explicit environment handling
    conda run --no-capture-output --cwd "$TRELLIS_DIR" -n trellis2 bash "$setup_script" --basic --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm <<< "Y"

    # Explicitly run inside the targeted environment
    conda run --no-capture-output -n trellis2 pip install transformers==4.57.6 psutil
}

install_flash_attn() {
    mkdir -p "$FLASH_WHEEL_DIR"

    local matched_wheels=("$FLASH_WHEEL_DIR"/flash_attn-"$FLASH_VERSION"*.whl)

    # Check if the expansion actually found a real file
    if [ -f "${matched_wheels[0]}" ]; then
        info "Found pre-compiled wheel. Installing..."
        conda run --no-capture-output -n trellis2 pip install "${matched_wheels[0]}"
    else
        warn "Building flash-attn. This will take a while..."
        MAX_JOBS=4 conda run --no-capture-output -n trellis2 pip wheel flash-attn=="$FLASH_VERSION" -w "$FLASH_WHEEL_DIR" -v --no-build-isolation

        # Re-evaluate the expansion array to find the newly built wheel
        matched_wheels=("$FLASH_WHEEL_DIR"/flash_attn-"$FLASH_VERSION"*.whl)
        conda run --no-capture-output -n trellis2 pip install "${matched_wheels[0]}"
    fi

    # Handle your symlink cleanly
    ln -sf "$TRELLIS_DIR" "$PACKAGES_DIR/trellis2"
}

run_step "Installing Trellis" \
    setup_trellis

info "Checking for flash-attn"

if ! ls "$FLASH_WHEEL_DIR"/flash_attn-"$FLASH_VERSION"*.whl 1>/dev/null 2>&1; then
    warn "No pre-compiled wheel found. Building flash-attn from source. This can take over 40 minutes..."
fi

run_step "Building Flash Attention" \
    install_flash_attn

## =============
##    SAM 3D
## =============

download_sam3d() {
    if [ ! -d "$CHECKPOINT_DIR/hf" ]; then
        hf download --repo-type model --local-dir "$CHECKPOINT_DIR/hf-download" --max-workers 1  facebook/sam-3d-objects
        mv  "$CHECKPOINT_DIR/hf-download/checkpoints" "$CHECKPOINT_DIR/hf"
        rm -rf "$CHECKPOINT_DIR/hf-download"
    fi
}

run_step "Downloading SAM 3D" \
    download_sam3d

install_pytorch3d() {
    mkdir -p "$P3D_WHEEL_DIR"

    local matched_wheels=("$P3D_WHEEL_DIR"/pytorch3d-*.whl)

    if [ -f "${matched_wheels[0]}" ]; then
        info "Found pre-compiled pytorch3d wheel. Installing..."
        run_in_env pip install "${matched_wheels[0]}"
    else
        warn "Building pytorch3d from source. This will take a while..."

        # Clone at pinned commit and build as a pure-Python wheel by making
        # setup.py believe CUDA is unavailable. sam3d_objects only uses pytorch3d's
        # pure-Python APIs (look_at_view_transform, Transform3d, quaternion_*)
        # so the _C CUDA extension is not needed.
        local build_dir
        build_dir="$(mktemp -d)"
        git clone https://github.com/facebookresearch/pytorch3d.git "$build_dir"
        git -C "$build_dir" checkout "$P3D_COMMIT"
        python3 -c "
import pathlib
p = pathlib.Path('$build_dir/setup.py')
# Prepend a patch so torch.cuda.is_available() returns False, causing setup.py
# to skip the _C CUDA extension entirely. sam3d_objects only uses pytorch3d's
# pure-Python APIs (look_at_view_transform, Transform3d, quaternion_*) which
# don't need _C. This avoids the Pulsar linker failure on CUDA 13 / sm_89+.
p.write_text(
    'import torch as _t; _t.cuda.is_available = lambda: False\n' + p.read_text()
)
"
        run_in_env pip wheel --no-build-isolation "$build_dir" -w "$P3D_WHEEL_DIR"
        rm -rf "$build_dir"

        matched_wheels=("$P3D_WHEEL_DIR"/pytorch3d-*.whl)
        run_in_env pip install "${matched_wheels[0]}"
    fi
}

setup_sam3d() {
    clone_if_needed https://github.com/facebookresearch/sam-3d-objects.git "$LIB_DIR/SAM3D"

    create_env "sam3d"
    run_in_env pip install -r "$REQUIREMENTS_DIR/requirements-sam3d.txt"

    local sam3d_flash_wheels=("$FLASH_WHEEL_DIR"/flash_attn-"$FLASH_VERSION_SAM3D"*.whl)
    if [ ! -f "${sam3d_flash_wheels[0]}" ]; then
        warn "Building flash-attn $FLASH_VERSION_SAM3D for sam3d. This will take a while..."
        MAX_JOBS=4 run_in_env pip wheel flash-attn=="$FLASH_VERSION_SAM3D" \
            -w "$FLASH_WHEEL_DIR" --no-build-isolation
    fi

    run_in_env pip install -r "$REQUIREMENTS_DIR/requirements-p3d.txt" \
        --find-links "$FLASH_WHEEL_DIR" --no-build-isolation
    install_pytorch3d
    # Install SAM3D's own deps one by one, skipping known-broken or incompatible packages.
    # NOTE: use conda run directly (not run_in_env) so failures are tolerated — run_in_env
    # calls exit 1 on failure, which would terminate the whole script.
    while IFS= read -r pkg; do
        [[ -z "$pkg" || "$pkg" == \#* ]] && continue
        # bpy: not on PyPI
        # *+cu121*: version pinned to CUDA 12.1 (we have cu13 variants, e.g. torchaudio)
        # cuda-python==12.1.0: never published on PyPI
        # nvidia-pyindex: broken setup.py; just adds NVIDIA package index, not a runtime dep
        # Note: spconv-cu121 is intentionally NOT skipped — no cu130 build exists, and
        #       cu121 wheels run fine on CUDA 13 via forward compatibility.
        [[ "$pkg" == bpy* || "$pkg" == *+cu121* || \
           "$pkg" == "cuda-python==12.1.0" || "$pkg" == nvidia-pyindex* ]] && {
            warn "Skipping incompatible dep: $pkg"
            continue
        }
        conda run --no-capture-output -n "$CURRENT_ENV" pip install "$pkg" --quiet \
            || warn "Skipped dep (incompatible): $pkg"
    done < "$LIB_DIR/SAM3D/requirements.txt"
    # Install inference-only deps (gsplat, nvdiffrast, seaborn).
    # Skip kaolin — handled by stub below. Skip gradio — UI dep, not needed for inference.
    # gsplat and nvdiffrast both import torch at build time, so --no-build-isolation is required.
    run_in_env pip install --no-build-isolation \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@2323de5905d5e90e035f792fe65bad0fedd413e7" \
        "git+https://github.com/NVlabs/nvdiffrast.git" \
        seaborn==0.13.2
    # SAM3D's gaussian_render.py uses kernel_size + subpixel_offset, which are Mip-Splatting
    # extensions not in the upstream graphdeco-inria fork. Clone mip-splatting shallowly and
    # install from its diff-gaussian-rasterization submodule.
    # Also patch rasterizer_impl.h: CUDA 13 + C++20 no longer implicitly includes <cstdint>.
    local dgr_build_dir
    dgr_build_dir="$(mktemp -d)"
    git clone --depth 1 --recursive https://github.com/autonomousvision/mip-splatting.git "$dgr_build_dir"
    sed -i '1s/^/#include <cstdint>\n/' "$dgr_build_dir/submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h"
    run_in_env pip install --no-build-isolation "$dgr_build_dir/submodules/diff-gaussian-rasterization"
    rm -rf "$dgr_build_dir"
    run_in_env pip install -e "$LIB_DIR/SAM3D" --no-deps
    # SAM3D's requirements include nvidia-nccl-cu12 which overwrites the cu13 NCCL that
    # PyTorch was compiled against. ncclCommResume was added after 2.21.5 (cu12), so we
    # must force cu13 back after all deps are installed.
    run_in_env pip install --upgrade nvidia-nccl-cu13
    # SAM3D pins opencv-python==4.9.0.80 which is a NumPy 1.x build; upgrade to get a
    # NumPy 2-compatible wheel.
    run_in_env pip install --upgrade "opencv-python>=4.9.0.80"
    # SAM3D pins xformers==0.0.28.post3 (built for torch 2.5.1+cu121). The triton JIT API
    # changed in torch 2.12, breaking xformers' vararg kernel unrolling. Upgrade to 0.0.35
    # which supports torch>=2.10 and ships a py-none (pure-Python) wheel.
    run_in_env pip install xformers==0.0.35
    # kaolin has no wheels for torch 2.x / CUDA 13. The only usage in sam3d_objects is
    # kaolin.utils.testing.check_tensor (a pure-Python shape validator in flexicubes.py).
    # Install a minimal stub instead of the full library.
    local py_ver
    py_ver=$(conda run -n "$CURRENT_ENV" python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    local kaolin_dir="$HOME/miniconda3/envs/$CURRENT_ENV/lib/python${py_ver}/site-packages/kaolin"
    mkdir -p "$kaolin_dir/utils"
    printf '# Minimal kaolin stub — only kaolin.utils.testing.check_tensor is used.\n' > "$kaolin_dir/__init__.py"
    printf '' > "$kaolin_dir/utils/__init__.py"
    cat > "$kaolin_dir/utils/testing.py" << 'PYEOF'
import torch

def check_tensor(obj, shape=None, dtype=None, throw=True):
    if not isinstance(obj, torch.Tensor):
        if throw:
            raise TypeError(f"Expected torch.Tensor, got {type(obj)}")
        return False
    if shape is not None:
        if len(obj.shape) != len(shape):
            if throw:
                raise ValueError(f"Shape rank mismatch: expected {len(shape)}, got {len(obj.shape)}")
            return False
        for i, (actual, expected) in enumerate(zip(obj.shape, shape)):
            if expected is not None and actual != expected:
                if throw:
                    raise ValueError(f"Dim {i}: expected {expected}, got {actual}")
                return False
    if dtype is not None and obj.dtype != dtype:
        if throw:
            raise TypeError(f"dtype mismatch: expected {dtype}, got {obj.dtype}")
        return False
    return True
PYEOF
    stop_env
}

run_step "Setup SAM 3D" \
    setup_sam3d

## ======================
##    Depth Anything
## ======================

setup_depth_anything() {
    clone_if_needed https://github.com/ByteDance-Seed/depth-anything-3 "$LIB_DIR/depth-anything-3"

    warn "Removing xformers"
    perl -pi -e 's/.*xformers.*//g' "$LIB_DIR/depth-anything-3/requirements.txt"
    perl -pi -e 's/.*"xformers".*//g' "$LIB_DIR/depth-anything-3/pyproject.toml"

    create_env "depthanything" 3.10
    run_in_env pip install -e "$LIB_DIR/depth-anything-3"
    stop_env
}

run_step "Installing Depth Anything" \
    setup_depth_anything

## ======================
##        RAM++
## ======================

setup_recognize_anything() {
    clone_if_needed https://github.com/xinyu1205/recognize-anything.git "$LIB_DIR/recognize-anything"
    download_checkpoint "https://huggingface.co/xinyu1205/recognize-anything-plus-model/blob/main/ram_plus_swin_large_14m.pth" "recognize_anything"

    create_env "recognize" 3.10
    run_in_env pip install -e "$LIB_DIR/recognize-anything"
    run_in_env pip install scipy timm fairscale
    run_in_env pip install "transformers<4.45"
    stop_env
}

run_step "Installing Recognize Anything" \
    setup_recognize_anything

## ======================
##    LuxDiT
## ======================

setup_lux_dit() {
    clone_if_needed https://github.com/nv-tlabs/LuxDiT.git "$LIB_DIR/LuxDiT"

    create_env "lux-dit" 3.10
    run_in_env pip install -r "$LIB_DIR/LuxDiT/requirements.txt"
    run_in_env pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git

    if [ ! -d "$CHECKPOINT_DIR/LuxDiT" ]; then
        run_in_env python "$LIB_DIR/LuxDiT/download_weights.py" --repo_id nvidia/LuxDiT --local_dir "$CHECKPOINT_DIR/LuxDiT"
    fi

    stop_env
}

run_step "Installing LuxDiT" \
    setup_lux_dit

## ======================
##    Models Download
## ======================

# Hugging Face auth for gated checkpoints
warn ""
warn "⚠️  Model checkpoints require Hugging Face access."
conda run --no-capture-output -n "$CONDA_NAME" pip install -q huggingface_hub >>"$LOG_FILE" 2>&1
load_conda
conda activate "$CONDA_NAME"
python -c "from huggingface_hub import interpreter_login; interpreter_login()"

# Encapsulating the download wrapper so it interfaces cleanly with your UI spinner
download_pipeline_models() {
    if ! conda run --no-capture-output -n "$CONDA_NAME" python3 "$PROJECT_DIR/server/main.py" download; then
        error "Error: failed to download models. Access may be required on Hugging Face" >&2
        error "Models will be downloaded later when running pipeline" >&2
    fi
}

run_step "Downloading Model Checkpoints" \
    download_pipeline_models

run_step "Installing Updated Pillow" \
    conda run --no-capture-output -n frame pip install --upgrade --force-reinstall Pillow

## ======================
##    Stable Point 3D
## ======================

setup_stable_point() {
    create_env "stablepoint"
    run_in_env pip install transformers==4.42.3
    clone_if_needed https://github.com/Stability-AI/stable-point-aware-3d "$LIB_DIR/StablePoint"

    run_in_env pip install -r "$REQUIREMENTS_DIR/requirements-stable3d.txt"
    run_in_env pip install --no-build-isolation git+https://github.com/SunzeY/AlphaCLIP.git
    run_in_env pip install --no-build-isolation -e "$LIB_DIR/StablePoint/texture_baker"
    run_in_env pip install --no-build-isolation -e "$LIB_DIR/StablePoint/uv_unwrapper"
    run_in_env pip install --upgrade transparent-background flet
    ln -sf  "$LIB_DIR/StablePoint/spar3d" "$PACKAGES_DIR/spar3d"

    stop_env
}

run_step "Installing Stable Point 3D" \
    setup_stable_point

## ============
##    CubeDiff
## ============

setup_cubediff() {
    create_env "cubediff"
    clone_if_needed https://github.com:Juan5713/OpenCubeDiff.git "$LIB_DIR/CubeDiff"
    run_in_env pip install -r "$REQUIREMENTS_DIR/requirements-cubediff.txt"
    ln -sf  "$LIB_DIR/CubeDiff/cubediff" "$PACKAGES_DIR/cubediff"

    stop_env
}

run_step "Installing CubeDiff" \
    setup_cubediff

## ============
##    DreamCube
## ============

setup_dreamcube() {
    create_env "dreamcube"

    clone_if_needed https://github.com/Yukun-Huang/DreamCube.git "$LIB_DIR/DreamCube"
    run_in_env pip install -r "$REQUIREMENTS_DIR/requirements-dreamcube.txt"
    run_in_env pip install ninja wheel setuptools
    install_pytorch3d
    run_in_env pip install peft
    ln -sf  "$LIB_DIR/DreamCube" "$PACKAGES_DIR/dreamcube"

    stop_env
}

run_step "Installing DreamCube" \
    setup_dreamcube

## ============
##    Lama
## ============

setup_lama() {
    create_env "lama" 3.10
    clone_if_needed https://github.com/advimman/lama.git "$LIB_DIR/LaMa"
    run_in_env pip install -r "$REQUIREMENTS_DIR/requirements-lama.txt"
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
}

run_step "Installing LaMa" \
    setup_lama

## ============
## LayerPano3D
## ============

download_layer_pano_3d() {
    download_checkpoint "https://huggingface.co/ysmikey/Layerpano3D-FLUX-Panorama-LoRA/resolve/main/lora_hubs/pano_lora_720*1440_v1.safetensors?download=true" "layer_pano_3d"
}

run_step "Downloading Layer Pano 3D" \
    download_layer_pano_3d

## ======================
## Depth Any Panoramas
## ======================

setup_depth_pano() {
    create_env "depthpano"
    clone_if_needed https://github.com/Insta360-Research-Team/DAP "$LIB_DIR/DAP"
    run_in_env pip install -r "$LIB_DIR/DAP/requirements.txt"
    # DAP's requirements.txt pins torch==2.7.1 without a CUDA suffix, which resolves
    # to the cu126 wheel and overwrites the cu130 base. Force upgrade with the correct index.
    run_in_env pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_URL"

    download_checkpoint "https://huggingface.co/Insta360-Research/DAP-weights/resolve/main/model.pth" "depth_pano"

    ln -sf  "$LIB_DIR/DAP" "$PACKAGES_DIR/dap"
}

run_step "Install Depth Any Panoramas" \
    setup_depth_pano

## ============
##    WorldGen
## ============

setup_worldgen() {
    clone_if_needed https://github.com/ZiYang-xie/WorldGen.git "$LIB_DIR/WorldGen"

    create_env "worldgen" 3.12
    run_in_env pip install torch==2.10.0 torchvision==0.25.0 --extra-index-url "$TORCH_URL"
    run_in_env pip install -r "$REQUIREMENTS_DIR/requirements-worldgen.txt"
    run_in_env pip install git+https://github.com/mit-han-lab/nunchaku.git
    run_in_env pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git

    run_in_env pip uninstall -y xformers
    run_in_env pip install ninja cmake setuptools wheel
    export TORCH_CUDA_ARCH_LIST="12.0"
    run_in_env pip install --no-build-isolation git+https://github.com/facebookresearch/xformers.git
    unset TORCH_CUDA_ARCH_LIST

    run_in_env pip install --no-deps git+https://github.com/EnVision-Research/DA-2.git#subdirectory=src

    ln -sf "$LIB_DIR/WorldGen/src/worldgen" "$PACKAGES_DIR/worldgen"

    stop_env
}

run_step "Installing WorldGen" \
    setup_worldgen

## ============
##    End
## ============

load_conda
conda activate "$CONDA_NAME"

success ""
success "Setup complete! To start:"
success "  conda activate $CONDA_NAME"
