#!/bin/bash
# ACDC Setup Script
# This script sets up the ACDC (Automated Creation of Digital Cousins) pipeline
# Requires: Python 3.10, uv package manager

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==================================================="
echo "ACDC Setup Script"
echo "==================================================="
echo "Project directory: $PROJECT_DIR"

# Check for Python 3.10
if ! command -v python3.10 &> /dev/null; then
    echo "ERROR: Python 3.10 is required but not found"
    echo "Install with: apt install python3.10 python3.10-venv"
    exit 1
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

cd "$PROJECT_DIR"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python 3.10 virtual environment..."
    uv venv --python 3.10 .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies with bpy from Blender's PyPI
echo "Installing ACDC dependencies (including bpy)..."
uv pip install -e . --find-links https://download.blender.org/pypi/bpy/

# Clone external dependencies if not present
DEPS_DIR="$PROJECT_DIR/deps"
mkdir -p "$DEPS_DIR"

# DINOv2
if [ ! -d "$DEPS_DIR/dinov2" ]; then
    echo "Cloning DINOv2..."
    git clone https://github.com/facebookresearch/dinov2.git "$DEPS_DIR/dinov2"
fi

# Depth-Anything-V2
if [ ! -d "$DEPS_DIR/Depth-Anything-V2" ]; then
    echo "Cloning Depth-Anything-V2..."
    git clone https://github.com/DepthAnything/Depth-Anything-V2.git "$DEPS_DIR/Depth-Anything-V2"
fi

# Segment Anything 2
if [ ! -d "$DEPS_DIR/segment-anything-2" ]; then
    echo "Cloning Segment-Anything-2..."
    git clone https://github.com/facebookresearch/segment-anything-2.git "$DEPS_DIR/segment-anything-2"
    cd "$DEPS_DIR/segment-anything-2"
    uv pip install -e .
    cd "$PROJECT_DIR"
fi

# Download checkpoints if not present
CKPT_DIR="$PROJECT_DIR/checkpoints"
mkdir -p "$CKPT_DIR"

# Depth-Anything-V2 checkpoint
if [ ! -f "$CKPT_DIR/depth_anything_v2_vitl.pth" ]; then
    echo "Downloading Depth-Anything-V2 checkpoint..."
    wget -O "$CKPT_DIR/depth_anything_v2_vitl.pth" \
        "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth"
fi

# SAM2 checkpoint
if [ ! -f "$CKPT_DIR/sam2.1_hiera_large.pt" ]; then
    echo "Downloading SAM2 checkpoint..."
    wget -O "$CKPT_DIR/sam2.1_hiera_large.pt" \
        "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
fi

# Create activation script
cat > "$PROJECT_DIR/activate_acdc.sh" << 'EOF'
#!/bin/bash
# ACDC environment activation script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate venv
source "$SCRIPT_DIR/.venv/bin/activate"

# Set environment variables
export ACDC_DIR="$SCRIPT_DIR"
export HOLODECK_DIR="${HOLODECK_DIR:-$HOME/workspace/Holodeck}"
export OBJATHOR_ASSETS_DIR="${OBJATHOR_ASSETS_DIR:-$HOLODECK_DIR/data/2023_09_23/assets}"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/deps/dinov2:$SCRIPT_DIR/deps/Depth-Anything-V2:$PYTHONPATH"
export ACDC_CHECKPOINTS="$SCRIPT_DIR/checkpoints"

echo "ACDC environment activated"
echo "  ACDC_DIR: $ACDC_DIR"
echo "  HOLODECK_DIR: $HOLODECK_DIR"
echo "  PYTHONPATH includes: dinov2, Depth-Anything-V2"
echo "  ACDC_CHECKPOINTS: $ACDC_CHECKPOINTS"
EOF

chmod +x "$PROJECT_DIR/activate_acdc.sh"

echo ""
echo "==================================================="
echo "ACDC Setup Complete!"
echo "==================================================="
echo ""
echo "To activate the environment:"
echo "  source activate_acdc.sh"
echo ""
echo "To run ACDC:"
echo "  python digital_cousins/pipeline/acdc.py --help"
echo ""
