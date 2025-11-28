#!/bin/bash
# SD 3.5 Dependencies Setup (uv-only, no conda)
# This script sets up the Stable Diffusion 3.5 tool with all its dependencies.

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
SD35_DIR="${SD35_DIR:-${WORKSPACE_DIR}/sd3.5}"
SD35_MODELS_DIR="${SD35_DIR}/models"

print_header "SD 3.5 Dependencies Setup (uv)"
print_info "SD 3.5 directory: ${SD35_DIR}"
print_info "Models directory: ${SD35_MODELS_DIR}"

# Check for uv
if ! command -v uv &> /dev/null; then
    print_error "uv is not installed. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_success "uv found"

# 1. Clone or verify SD 3.5 repo exists
print_header "Setting Up SD 3.5 Repository"
mkdir -p "${WORKSPACE_DIR}"

if [ ! -d "${SD35_DIR}" ]; then
    print_info "Cloning Stable Diffusion 3.5..."
    git clone https://github.com/Stability-AI/sd3.5.git "${SD35_DIR}"
    print_success "Cloned SD 3.5"
else
    print_success "SD 3.5 directory exists: ${SD35_DIR}"
fi

# 2. Create uv venv for SD 3.5
print_header "Creating SD 3.5 Virtual Environment"
cd "${SD35_DIR}"

if [ -d ".venv" ]; then
    print_warning "SD 3.5 .venv exists"
    read -p "Remove and recreate? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf .venv
        uv venv .venv --python 3.10
        print_success "Recreated SD 3.5 venv"
    else
        print_info "Using existing venv"
    fi
else
    uv venv .venv --python 3.10
    print_success "Created SD 3.5 venv"
fi

# Activate venv for subsequent installs
source .venv/bin/activate

# 3. Install PyTorch with CUDA 12.1
print_header "Installing PyTorch (CUDA 12.1)"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
print_success "PyTorch installed"

# 4. Install SD 3.5 requirements
print_header "Installing SD 3.5 Requirements"
# Use cu121 instead of cu118 in requirements
uv pip install transformers numpy fire pillow einops sentencepiece protobuf webdataset safetensors huggingface_hub
print_success "SD 3.5 requirements installed"

# 5. Download SD 3.5 models (requires HuggingFace authentication)
print_header "Downloading SD 3.5 Models"
mkdir -p "${SD35_MODELS_DIR}"

print_info "SD 3.5 models are GATED and require license acceptance."
print_warning ""
print_warning "REQUIRED STEPS:"
print_warning "  1. Visit: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
print_warning "  2. Click 'Agree and access repository' to accept the license"
print_warning "  3. Be logged in via: huggingface-cli login"
print_warning ""

# Check if HuggingFace CLI is available and logged in
if python -c "from huggingface_hub import HfApi; api = HfApi(); api.whoami()" 2>/dev/null; then
    print_success "HuggingFace authentication found"

    # Try to download SD 3.5 Medium
    print_info "Attempting to download SD 3.5 Medium model..."
    if python "${SD35_DIR}/download_models.py" --model medium; then
        print_success "SD 3.5 models downloaded successfully"
    else
        print_warning "Model download failed - you likely need to accept the license"
        print_info "After accepting the license, run:"
        print_info "  cd ${SD35_DIR} && source .venv/bin/activate && python download_models.py"
    fi
else
    print_warning "HuggingFace not authenticated."
    print_info "To download models later:"
    print_info "  1. pip install huggingface_hub"
    print_info "  2. huggingface-cli login"
    print_info "  3. Accept license at: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
    print_info "  4. cd ${SD35_DIR} && source .venv/bin/activate && python download_models.py"
fi

# 6. Create run.sh script for ACDC integration
print_header "Creating Run Script"
cat > "${SD35_DIR}/run.sh" << 'RUN_EOF'
#!/bin/bash
# Run SD 3.5 inference from prompt.json
# Used by SceneWeaver ACDC tool

SD35_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SD35_DIR}/.venv/bin/activate"

# Read prompt from prompt.json
if [ -f "${SD35_DIR}/prompt.json" ]; then
    PROMPT=$(python -c "import json; f=open('${SD35_DIR}/prompt.json'); d=json.load(f); print(d.get('prompt', ''))")
    OUTPUT=$(python -c "import json; f=open('${SD35_DIR}/prompt.json'); d=json.load(f); print(d.get('img_savedir', 'output.jpg'))")

    # Get the directory and ensure it exists
    OUTPUT_DIR=$(dirname "${OUTPUT}")
    mkdir -p "${OUTPUT_DIR}"

    # Use a temp directory for SD3.5 output (it creates nested subdirs)
    TEMP_OUT_DIR="${SD35_DIR}/temp_output"
    rm -rf "${TEMP_OUT_DIR}"
    mkdir -p "${TEMP_OUT_DIR}"

    # Run inference
    cd "${SD35_DIR}"
    python sd3_infer.py \
        --prompt "${PROMPT}" \
        --model models/sd3.5_medium.safetensors \
        --out_dir "${TEMP_OUT_DIR}" \
        --steps 28 \
        --cfg 4.5

    # Find the generated image (SD3.5 creates nested subdirs like: <model>/<prompt>/<postfix>/000000.png)
    GENERATED_FILE=$(find "${TEMP_OUT_DIR}" -name "*.png" -type f | head -1)
    if [ -n "${GENERATED_FILE}" ] && [ -f "${GENERATED_FILE}" ]; then
        cp "${GENERATED_FILE}" "${OUTPUT}"
        echo "Image saved to: ${OUTPUT}"
        rm -rf "${TEMP_OUT_DIR}"
    else
        echo "Error: No image generated"
        exit 1
    fi
else
    echo "Error: prompt.json not found"
    exit 1
fi
RUN_EOF

chmod +x "${SD35_DIR}/run.sh"
print_success "Created run.sh"

# 7. Create activation script
print_header "Creating Activation Script"
cat > "${SD35_DIR}/activate_sd35.sh" << 'ACTIVATE_EOF'
#!/bin/bash
# Activate SD 3.5 environment
# Usage: source /path/to/sd3.5/activate_sd35.sh

SD35_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "${SD35_DIR}/.venv" ]; then
    echo "Error: SD 3.5 venv not found at ${SD35_DIR}/.venv"
    echo "Run install_sd35_dependencies.sh first"
    return 1
fi

source "${SD35_DIR}/.venv/bin/activate"

echo "SD 3.5 environment activated"
echo "  SD35_DIR: ${SD35_DIR}"
ACTIVATE_EOF

chmod +x "${SD35_DIR}/activate_sd35.sh"
print_success "Created activate_sd35.sh"

deactivate

# Final summary
print_header "SD 3.5 Setup Complete!"
echo ""
echo "Installation summary:"
echo "  SD 3.5 directory:      ${SD35_DIR}"
echo "  Virtual environment:    ${SD35_DIR}/.venv"
echo "  Models directory:       ${SD35_MODELS_DIR}"
echo ""
echo "To use SD 3.5 manually:"
echo "  source ${SD35_DIR}/activate_sd35.sh"
echo ""
echo "To verify installation:"
echo "  source ${SD35_DIR}/activate_sd35.sh"
echo "  python -c 'import torch; print(f\"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}\")'"
echo ""

if [ ! -f "${SD35_MODELS_DIR}/sd3.5_medium.safetensors" ]; then
    print_warning "Models not downloaded! Please:"
    print_info "1. Accept license at https://huggingface.co/stabilityai/stable-diffusion-3.5-medium"
    print_info "2. Run: huggingface-cli login"
    print_info "3. Re-run this script"
fi
