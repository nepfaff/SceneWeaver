#!/bin/bash
# ACDC Tool Dependencies Setup (uv-only, no conda)
# This script sets up the Tabletop-Digital-Cousins (ACDC) tool with all its dependencies.

set -e

# Get script directory and SceneWeaver root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENEWEAVER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() { echo -e "\n${BLUE}======== $1 ========${NC}\n"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error() { echo -e "${RED}✗ $1${NC}"; }
print_info() { echo -e "${BLUE}ℹ $1${NC}"; }

# Load .env if exists
if [ -f "${SCENEWEAVER_DIR}/.env" ]; then
    set -a
    source "${SCENEWEAVER_DIR}/.env"
    set +a
fi

# Configuration
WORKSPACE_DIR="${WORKSPACE_DIR:-${HOME}/workspace}"
ACDC_DIR="${ACDC_DIR:-${WORKSPACE_DIR}/Tabletop-Digital-Cousins}"
ACDC_DEPS_DIR="${ACDC_DIR}/deps"
ACDC_CHECKPOINTS_DIR="${ACDC_DIR}/checkpoints"

print_header "ACDC Tool Dependencies Setup (uv)"
print_info "ACDC directory: ${ACDC_DIR}"
print_info "Dependencies directory: ${ACDC_DEPS_DIR}"
print_info "Checkpoints directory: ${ACDC_CHECKPOINTS_DIR}"

# Check for uv
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_success "uv found"

# 1. Clone or verify ACDC repo exists
print_header "Setting Up ACDC Repository"
mkdir -p "${WORKSPACE_DIR}"

if [ ! -d "${ACDC_DIR}" ]; then
    print_info "Cloning Tabletop-Digital-Cousins..."
    git clone https://github.com/Scene-Weaver/Tabletop-Digital-Cousins.git "${ACDC_DIR}"
    print_success "Cloned Tabletop-Digital-Cousins"
else
    print_success "ACDC directory exists: ${ACDC_DIR}"
fi

# 2. Create uv venv for ACDC
print_header "Creating ACDC Virtual Environment"
cd "${ACDC_DIR}"

if [ -d ".venv" ]; then
    print_warning "ACDC .venv exists"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        uv venv .venv --python 3.10
        print_success "Recreated ACDC venv"
    else
        print_info "Using existing venv"
    fi
else
    uv venv .venv --python 3.10
    print_success "Created ACDC venv"
fi

# Activate venv for subsequent installs
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.1
print_header "Installing PyTorch (CUDA 12.1)"
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
print_success "PyTorch installed"

# Install xformers
print_info "Installing xformers..."
uv pip install xformers --index-url https://download.pytorch.org/whl/cu121 || {
    print_warning "xformers install from cu121 index failed, trying default..."
    uv pip install xformers || print_warning "xformers not installed (optional)"
}

# 4. Install ACDC base requirements
print_header "Installing ACDC Requirements"
if [ -f "requirements.txt" ]; then
    print_info "Installing from requirements.txt..."
    uv pip install -r requirements.txt
    print_success "requirements.txt installed"
fi

if [ -f "pyproject.toml" ] || [ -f "setup.py" ]; then
    print_info "Installing ACDC package..."
    uv pip install -e . || print_warning "ACDC package install failed (may not have setup.py)"
fi

# 5. Clone and install external dependencies
print_header "Installing External Dependencies"
mkdir -p "${ACDC_DEPS_DIR}"
cd "${ACDC_DEPS_DIR}"

# DINOv2 (PYTHONPATH only, no install needed)
if [ ! -d "dinov2" ]; then
    print_info "Cloning DINOv2..."
    git clone https://github.com/facebookresearch/dinov2.git
fi
print_success "DINOv2"

# Segment Anything 2
if [ ! -d "segment-anything-2" ]; then
    print_info "Cloning SAM2..."
    git clone https://github.com/facebookresearch/segment-anything-2.git
    cd segment-anything-2
    uv pip install -e .
    cd ..
    print_success "SAM2 installed"
else
    print_success "SAM2 (already exists)"
fi

# GroundingDINO (needs CUDA_HOME)
if [ ! -d "GroundingDINO" ]; then
    print_info "Cloning GroundingDINO..."
    git clone https://github.com/IDEA-Research/GroundingDINO.git
    cd GroundingDINO

    # Auto-detect CUDA_HOME
    if [ -z "${CUDA_HOME}" ]; then
        for cuda_path in /usr/local/cuda-12.1 /usr/local/cuda-12 /usr/local/cuda; do
            if [ -d "$cuda_path" ]; then
                export CUDA_HOME="$cuda_path"
                break
            fi
        done
    fi

    if [ -z "${CUDA_HOME}" ]; then
        print_warning "CUDA_HOME not found, GroundingDINO may not build correctly"
    else
        print_info "Using CUDA_HOME=${CUDA_HOME}"
    fi

    uv pip install --no-build-isolation -e .
    cd ..
    print_success "GroundingDINO installed"
else
    print_success "GroundingDINO (already exists)"
fi

# PerspectiveFields
if [ ! -d "PerspectiveFields" ]; then
    print_info "Cloning PerspectiveFields..."
    git clone https://github.com/jinlinyi/PerspectiveFields.git
    cd PerspectiveFields
    uv pip install -e .
    cd ..
    print_success "PerspectiveFields installed"
else
    print_success "PerspectiveFields (already exists)"
fi

# Depth-Anything-V2 (PYTHONPATH only, install requirements)
if [ ! -d "Depth-Anything-V2" ]; then
    print_info "Cloning Depth-Anything-V2..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git
    cd Depth-Anything-V2
    if [ -f "requirements.txt" ]; then
        uv pip install -r requirements.txt
    fi
    cd ..
    print_success "Depth-Anything-V2 installed"
else
    print_success "Depth-Anything-V2 (already exists)"
fi

# CLIP from GitHub
print_info "Installing CLIP..."
uv pip install git+https://github.com/openai/CLIP.git
print_success "CLIP installed"

# faiss-gpu (pip version for CUDA 12)
print_info "Installing faiss-gpu..."
if uv pip install faiss-gpu-cu12 2>/dev/null; then
    print_success "faiss-gpu-cu12 installed"
else
    print_warning "faiss-gpu-cu12 failed, trying faiss-cpu..."
    uv pip install faiss-cpu
    print_success "faiss-cpu installed (GPU version not available)"
fi

# objathor for Holodeck data download
print_info "Installing objathor..."
uv pip install objathor
print_success "objathor installed"

# 6. Download Holodeck/Objathor data for asset retrieval
print_header "Downloading Holodeck/Objathor Data"
HOLODECK_DIR="${WORKSPACE_DIR}/Holodeck"
HOLODECK_DATA_DIR="${HOLODECK_DIR}/data"
VERSION="2023_09_23"
mkdir -p "${HOLODECK_DATA_DIR}"

# Download Holodeck base data
if [ ! -d "${HOLODECK_DATA_DIR}/holodeck/${VERSION}" ]; then
    print_info "Downloading Holodeck base data..."
    python -m objathor.dataset.download_holodeck_base_data --version "${VERSION}" --path "${HOLODECK_DATA_DIR}" || print_warning "Holodeck base data download failed"
else
    print_success "Holodeck base data (already exists)"
fi

# Download annotations
if [ ! -f "${HOLODECK_DATA_DIR}/${VERSION}/annotations.json.gz" ]; then
    print_info "Downloading annotations..."
    python -m objathor.dataset.download_annotations --version "${VERSION}" --path "${HOLODECK_DATA_DIR}" || print_warning "Annotations download failed"
else
    print_success "Annotations (already exists)"
fi

# Download features (CLIP embeddings)
if [ ! -d "${HOLODECK_DATA_DIR}/${VERSION}/features" ]; then
    print_info "Downloading features (CLIP embeddings)..."
    python -m objathor.dataset.download_features --version "${VERSION}" --path "${HOLODECK_DATA_DIR}" || print_warning "Features download failed"
else
    print_success "Features (already exists)"
fi

# 7. Download model checkpoints
print_header "Downloading Model Checkpoints"
mkdir -p "${ACDC_CHECKPOINTS_DIR}"
cd "${ACDC_CHECKPOINTS_DIR}"

# GroundingDINO weights
if [ ! -f "groundingdino_swint_ogc.pth" ]; then
    print_info "Downloading GroundingDINO weights (~700MB)..."
    wget -q --show-progress https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
    print_success "GroundingDINO weights downloaded"
else
    print_success "GroundingDINO weights (already exists)"
fi

# SAM2 weights
if [ ! -f "sam2_hiera_large.pt" ]; then
    print_info "Downloading SAM2 weights (~900MB)..."
    wget -q --show-progress https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
    print_success "SAM2 weights downloaded"
else
    print_success "SAM2 weights (already exists)"
fi

# Depth-Anything-V2 weights
if [ ! -f "depth_anything_v2_metric_hypersim_vitl.pth" ]; then
    print_info "Downloading Depth-Anything-V2 weights (~400MB)..."
    wget -q --show-progress https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
    print_success "Depth-Anything-V2 weights downloaded"
else
    print_success "Depth-Anything-V2 weights (already exists)"
fi

# 7. Create activation script with PYTHONPATH
print_header "Creating Activation Script"
cat > "${ACDC_DIR}/activate_acdc.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Activate ACDC environment with proper PYTHONPATH
# Usage: source /path/to/Tabletop-Digital-Cousins/activate_acdc.sh

ACDC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "${ACDC_DIR}/.venv" ]; then
    echo "Error: ACDC venv not found at ${ACDC_DIR}/.venv"
    echo "Run install_acdc_tool_dependencies.sh first"
    return 1
fi

source "${ACDC_DIR}/.venv/bin/activate"

# Add dependencies that need PYTHONPATH
export PYTHONPATH="${ACDC_DIR}/deps/dinov2:${ACDC_DIR}/deps/Depth-Anything-V2:${PYTHONPATH}"

# Set checkpoint paths
export ACDC_CHECKPOINTS="${ACDC_DIR}/checkpoints"

echo "ACDC environment activated"
echo "  ACDC_DIR: ${ACDC_DIR}"
echo "  PYTHONPATH includes: dinov2, Depth-Anything-V2"
echo "  ACDC_CHECKPOINTS: ${ACDC_CHECKPOINTS}"
ACTIVATE_EOF

chmod +x "${ACDC_DIR}/activate_acdc.sh"
print_success "Created activate_acdc.sh"

# 8. Update SceneWeaver .env
print_header "Updating SceneWeaver Configuration"
if [ -f "${SCENEWEAVER_DIR}/.env" ]; then
    # Remove old ACDC entries if they exist
    sed -i '/^ACDC_VENV=/d' "${SCENEWEAVER_DIR}/.env"
    sed -i '/^ACDC_DEPS_DIR=/d' "${SCENEWEAVER_DIR}/.env"
    sed -i '/^ACDC_CHECKPOINTS_DIR=/d' "${SCENEWEAVER_DIR}/.env"
    sed -i '/^SCENEWEAVER_DIR=/d' "${SCENEWEAVER_DIR}/.env"

    # Remove empty lines at end of file
    sed -i -e :a -e '/^\n*$/{$d;N;ba' -e '}' "${SCENEWEAVER_DIR}/.env" 2>/dev/null || true

    # Append new entries
    cat >> "${SCENEWEAVER_DIR}/.env" << ENV_EOF

# ACDC Configuration (generated by install_acdc_tool_dependencies.sh)
SCENEWEAVER_DIR=${SCENEWEAVER_DIR}
ACDC_VENV=${ACDC_DIR}/.venv
ACDC_DEPS_DIR=${ACDC_DEPS_DIR}
ACDC_CHECKPOINTS_DIR=${ACDC_CHECKPOINTS_DIR}
ENV_EOF

    print_success "Updated ${SCENEWEAVER_DIR}/.env"
else
    print_warning ".env file not found at ${SCENEWEAVER_DIR}/.env"
fi

deactivate

# Final summary
print_header "ACDC Setup Complete!"
echo ""
echo "Installation summary:"
echo "  ACDC directory:     ${ACDC_DIR}"
echo "  Virtual environment: ${ACDC_DIR}/.venv"
echo "  Dependencies:        ${ACDC_DEPS_DIR}"
echo "  Checkpoints:         ${ACDC_CHECKPOINTS_DIR}"
echo ""
echo "To use ACDC manually:"
echo "  source ${ACDC_DIR}/activate_acdc.sh"
echo ""
echo "To verify installation:"
echo "  source ${ACDC_DIR}/activate_acdc.sh"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'"
echo ""
print_info "The SceneWeaver Pipeline will automatically use this environment via subprocess."
