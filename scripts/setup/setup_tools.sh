#!/bin/bash
# SceneWeaver Tools Setup Script
# Sets up ACDC and SD 3.5 tools with their separate virtual environments
#
# These tools require separate venvs due to conflicting dependencies:
# - ACDC: requires bpy 3.6.0, specific torch version for Blender integration
# - SD 3.5: requires specific torch/transformers versions for inference

set -e

# Get script directory and SceneWeaver root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENEWEAVER_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TOOLS_DIR="${SCENEWEAVER_DIR}/tools"

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

# Check for uv
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_success "uv found"

# ============================================================================
# ACDC Tool Setup
# ============================================================================
setup_acdc() {
    print_header "Setting Up ACDC Tool"

    ACDC_DIR="${TOOLS_DIR}/acdc"

    if [ ! -d "${ACDC_DIR}" ]; then
        print_error "ACDC directory not found at ${ACDC_DIR}"
        print_info "ACDC should be included in the SceneWeaver repository"
        return 1
    fi

    cd "${ACDC_DIR}"

    # Create venv if it doesn't exist
    if [ ! -d ".venv" ]; then
        print_info "Creating ACDC virtual environment..."
        uv venv .venv --python 3.10
        print_success "Created ACDC venv"
    else
        print_success "ACDC venv already exists"
    fi

    # Activate and install dependencies
    source .venv/bin/activate

    print_info "Installing ACDC dependencies..."

    # PyTorch with CUDA 12.1
    uv pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

    # Blender as Python module
    uv pip install bpy==3.6.0

    # Core dependencies
    uv pip install \
        openai>=1.0.0 \
        numpy \
        scipy \
        Pillow \
        trimesh \
        open3d \
        transformers \
        huggingface_hub \
        clip \
        pyyaml \
        tqdm \
        matplotlib \
        imageio

    # Install ACDC package in development mode
    uv pip install -e .

    deactivate

    print_success "ACDC dependencies installed"

    # Download checkpoints and external dependencies
    print_info "Setting up ACDC checkpoints and external tools..."

    mkdir -p "${ACDC_DIR}/checkpoints"
    mkdir -p "${ACDC_DIR}/deps"

    # Clone GroundingDINO if not present
    if [ ! -d "${ACDC_DIR}/deps/GroundingDINO" ]; then
        print_info "Cloning GroundingDINO..."
        git clone https://github.com/IDEA-Research/GroundingDINO.git "${ACDC_DIR}/deps/GroundingDINO"
    fi

    # Clone Depth-Anything-V2 if not present
    if [ ! -d "${ACDC_DIR}/deps/Depth-Anything-V2" ]; then
        print_info "Cloning Depth-Anything-V2..."
        git clone https://github.com/DepthAnything/Depth-Anything-V2.git "${ACDC_DIR}/deps/Depth-Anything-V2"
    fi

    # Clone segment-anything if not present
    if [ ! -d "${ACDC_DIR}/deps/segment-anything" ]; then
        print_info "Cloning segment-anything..."
        git clone https://github.com/facebookresearch/segment-anything.git "${ACDC_DIR}/deps/segment-anything"
    fi

    # Download checkpoints
    print_info "Downloading model checkpoints (this may take a while)..."

    # GroundingDINO checkpoint
    if [ ! -f "${ACDC_DIR}/checkpoints/groundingdino_swint_ogc.pth" ]; then
        curl -L -o "${ACDC_DIR}/checkpoints/groundingdino_swint_ogc.pth" \
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
    fi

    # Depth-Anything-V2 checkpoint
    if [ ! -f "${ACDC_DIR}/checkpoints/depth_anything_v2_vitl.pth" ]; then
        curl -L -o "${ACDC_DIR}/checkpoints/depth_anything_v2_vitl.pth" \
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
    fi

    # SAM checkpoint
    if [ ! -f "${ACDC_DIR}/checkpoints/sam_vit_h_4b8939.pth" ]; then
        curl -L -o "${ACDC_DIR}/checkpoints/sam_vit_h_4b8939.pth" \
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    fi

    print_success "ACDC setup complete"
}

# ============================================================================
# SD 3.5 Tool Setup
# ============================================================================
setup_sd35() {
    print_header "Setting Up SD 3.5 Tool"

    SD35_DIR="${TOOLS_DIR}/sd3.5"

    if [ ! -d "${SD35_DIR}" ]; then
        print_error "SD 3.5 directory not found at ${SD35_DIR}"
        return 1
    fi

    cd "${SD35_DIR}"

    # Create venv if it doesn't exist
    if [ ! -d ".venv" ]; then
        print_info "Creating SD 3.5 virtual environment..."
        uv venv .venv --python 3.10
        print_success "Created SD 3.5 venv"
    else
        print_success "SD 3.5 venv already exists"
    fi

    # Activate and install dependencies
    source .venv/bin/activate

    print_info "Installing SD 3.5 dependencies..."

    # PyTorch with CUDA 12.1
    uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

    # SD 3.5 requirements
    uv pip install \
        transformers \
        numpy \
        fire \
        pillow \
        einops \
        sentencepiece \
        protobuf \
        webdataset \
        safetensors \
        huggingface_hub

    deactivate

    print_success "SD 3.5 dependencies installed"

    # Download models
    print_header "Downloading SD 3.5 Models"

    print_warning "SD 3.5 models are GATED and require license acceptance."
    print_warning ""
    print_warning "REQUIRED STEPS:"
    print_warning "  1. Visit: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
    print_warning "  2. Click 'Agree and access repository' to accept the license"
    print_warning "  3. Be logged in via: huggingface-cli login"
    print_warning ""

    source .venv/bin/activate

    # Check if HuggingFace CLI is available and logged in
    if python -c "from huggingface_hub import HfApi; api = HfApi(); api.whoami()" 2>/dev/null; then
        print_success "HuggingFace authentication found"

        # Try to download SD 3.5 Medium
        print_info "Attempting to download SD 3.5 Medium model..."
        if python "${SD35_DIR}/download_models.py" --model medium; then
            print_success "SD 3.5 models downloaded successfully"
        else
            print_warning "Model download failed - you likely need to accept the license"
            print_info "After accepting, run: cd ${SD35_DIR} && source .venv/bin/activate && python download_models.py"
        fi
    else
        print_warning "HuggingFace not authenticated."
        print_info "To download models:"
        print_info "  1. huggingface-cli login"
        print_info "  2. Accept license at: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
        print_info "  3. cd ${SD35_DIR} && source .venv/bin/activate && python download_models.py"
    fi

    deactivate

    print_success "SD 3.5 setup complete"
}

# ============================================================================
# Holodeck Data Setup
# ============================================================================
setup_holodeck_data() {
    print_header "Setting Up Holodeck Data"

    HOLODECK_DIR="${SCENEWEAVER_DIR}/data/holodeck"
    mkdir -p "${HOLODECK_DIR}"

    print_info "Holodeck assets provide 3D models for ACDC object retrieval."
    print_info "Downloading Holodeck data (~10GB)..."

    # This would download the Holodeck data
    # For now, just create the directory structure
    mkdir -p "${HOLODECK_DIR}/2023_09_23/assets"

    print_warning "Holodeck data download not implemented in this script."
    print_info "Please download manually from the Holodeck repository."

    print_success "Holodeck data directory created"
}

# ============================================================================
# Main
# ============================================================================
print_header "SceneWeaver Tools Setup"
echo "This script sets up the ACDC and SD 3.5 tools with separate virtual environments."
echo ""
echo "Tools directory: ${TOOLS_DIR}"
echo ""

# Parse arguments
SETUP_ACDC=true
SETUP_SD35=true
SETUP_HOLODECK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --acdc-only)
            SETUP_SD35=false
            shift
            ;;
        --sd35-only)
            SETUP_ACDC=false
            shift
            ;;
        --with-holodeck)
            SETUP_HOLODECK=true
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --acdc-only      Only set up ACDC tool"
            echo "  --sd35-only      Only set up SD 3.5 tool"
            echo "  --with-holodeck  Also download Holodeck data"
            echo "  --help           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run setup
if [ "$SETUP_ACDC" = true ]; then
    setup_acdc
fi

if [ "$SETUP_SD35" = true ]; then
    setup_sd35
fi

if [ "$SETUP_HOLODECK" = true ]; then
    setup_holodeck_data
fi

# Final summary
print_header "Setup Complete!"
echo ""
echo "Virtual environments created:"
if [ "$SETUP_ACDC" = true ]; then
    echo "  - ACDC:  ${TOOLS_DIR}/acdc/.venv"
fi
if [ "$SETUP_SD35" = true ]; then
    echo "  - SD3.5: ${TOOLS_DIR}/sd3.5/.venv"
fi
echo ""
echo "These tools use SEPARATE venvs from SceneWeaver due to conflicting dependencies."
echo ""
echo "To use SceneWeaver with these tools, simply run the pipeline - it will"
echo "automatically invoke the correct Python interpreter for each tool."
