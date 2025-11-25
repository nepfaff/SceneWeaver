#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
WORKSPACE_DIR="${HOME}/workspace"
INSTALL_MODE="minimal"  # minimal or full

# Optional existing asset paths
export EXISTING_3D_FUTURE=""
export EXISTING_METASCENES=""
export EXISTING_OBJAVERSE=""

# Print functions
print_header() {
    echo -e "\n${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --workspace)
            WORKSPACE_DIR="$2"
            shift 2
            ;;
        --full)
            INSTALL_MODE="full"
            shift
            ;;
        --minimal)
            INSTALL_MODE="minimal"
            shift
            ;;
        --existing-3d-future)
            EXISTING_3D_FUTURE="$2"
            shift 2
            ;;
        --existing-metascenes)
            EXISTING_METASCENES="$2"
            shift 2
            ;;
        --existing-objaverse)
            EXISTING_OBJAVERSE="$2"
            shift 2
            ;;
        --help)
            echo "SceneWeaver Pipeline Setup Script"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --workspace DIR              Set workspace directory (default: ~/workspace)"
            echo "  --minimal                    Install minimal setup (default)"
            echo "  --full                       Install full setup with all tools and datasets"
            echo "  --existing-3d-future PATH    Point to existing 3D FUTURE dataset"
            echo "  --existing-metascenes PATH   Point to existing MetaScenes dataset"
            echo "  --existing-objaverse PATH    Point to existing Objaverse cache"
            echo "  --help                       Show this help message"
            echo ""
            echo "Minimal setup includes:"
            echo "  - SceneWeaver and Infinigen environments"
            echo "  - Infinigen installation with Blender"
            echo "  - LLM-based tools only"
            echo ""
            echo "Full setup additionally includes:"
            echo "  - SD 3.5 for image generation"
            echo "  - Tabletop Digital Cousins for 3D reconstruction"
            echo "  - 3D FUTURE dataset"
            echo "  - IDesign/OpenShape for Objaverse retrieval"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

print_header "SceneWeaver Pipeline Setup"
print_info "Installation mode: ${INSTALL_MODE}"
print_info "Workspace directory: ${WORKSPACE_DIR}"

# Check prerequisites
print_header "Checking Prerequisites"

command -v conda >/dev/null 2>&1 || {
    print_error "conda is not installed. Please install Miniconda or Anaconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
}
print_success "conda found"

command -v git >/dev/null 2>&1 || {
    print_error "git is not installed. Please install git first."
    exit 1
}
print_success "git found"

command -v wget >/dev/null 2>&1 || {
    print_error "wget is not installed. Please install wget first."
    exit 1
}
print_success "wget found"

command -v uv >/dev/null 2>&1 || {
    print_warning "uv is not installed. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    print_success "uv installed"
}
print_success "uv found"

# Create workspace directory
print_header "Setting Up Directories"
mkdir -p "${WORKSPACE_DIR}"
print_success "Workspace directory created: ${WORKSPACE_DIR}"

# Get the SceneWeaver root directory
SCENEWEAVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${SCENEWEAVER_DIR}"
print_success "SceneWeaver directory: ${SCENEWEAVER_DIR}"

# Source the modular setup scripts
source "${SCENEWEAVER_DIR}/scripts/setup/setup_environments.sh"
source "${SCENEWEAVER_DIR}/scripts/setup/setup_config.sh"

if [ "${INSTALL_MODE}" = "full" ]; then
    source "${SCENEWEAVER_DIR}/scripts/setup/download_external_tools.sh"
    source "${SCENEWEAVER_DIR}/scripts/setup/download_datasets.sh"
fi

# Setup environments
setup_environments "${SCENEWEAVER_DIR}"

# Setup configuration
setup_config "${SCENEWEAVER_DIR}" "${WORKSPACE_DIR}"

# Download external tools (full mode only)
if [ "${INSTALL_MODE}" = "full" ]; then
    download_external_tools "${WORKSPACE_DIR}"
    download_datasets "${WORKSPACE_DIR}"
fi

# Print final instructions
print_header "Setup Complete!"

if [ "${INSTALL_MODE}" = "minimal" ]; then
    echo -e "${GREEN}Minimal setup completed successfully!${NC}\n"
    echo "You can now run SceneWeaver with LLM-based tools and Infinigen assets."
    echo ""
    echo "To run the pipeline:"
    echo "  cd Pipeline"
    echo "  conda activate sceneweaver"
    echo "  python main.py --prompt 'Design me a bedroom.' --cnt 1 --basedir ./output/"
    echo ""
    echo "To install additional tools and datasets, run:"
    echo "  bash scripts/setup_pipeline.sh --full"
else
    echo -e "${GREEN}Full setup completed successfully!${NC}\n"
    echo "All tools and datasets have been installed."
    echo ""
    echo "To run the pipeline:"
    echo "  cd Pipeline"
    echo "  conda activate sceneweaver"
    echo "  python main.py --prompt 'Design me a bedroom.' --cnt 1 --basedir ./output/"
fi

echo ""
print_warning "Manual steps required:"
echo "1. Add your Azure OpenAI API key to: Pipeline/key.txt"
echo "2. Update Azure endpoint in: Pipeline/config/config.json"
echo "3. Review and modify tool configuration in: Pipeline/app/agent/scenedesigner.py"
echo ""
echo "See README.md for detailed usage instructions."
