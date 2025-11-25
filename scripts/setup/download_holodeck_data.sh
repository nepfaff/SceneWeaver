#!/bin/bash
# Download Holodeck objaverse data for asset retrieval
# This script downloads pre-computed embeddings and asset data from the objathor package

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENEWEAVER_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Default paths
VERSION="${OBJATHOR_VERSION:-2023_09_23}"
DATA_DIR="${OBJATHOR_DATA_DIR:-$HOME/.objathor-assets}"

echo "Downloading Holodeck/Objathor data..."
echo "Version: $VERSION"
echo "Data directory: $DATA_DIR"

# Activate virtual environment
cd "$SCENEWEAVER_DIR"
source .venv/bin/activate

# Download all required data using objathor
echo ""
echo "Step 1/4: Downloading holodeck base data..."
python -m objathor.dataset.download_holodeck_base_data --version "$VERSION" --path "$DATA_DIR" || {
    echo "Warning: Failed to download holodeck base data, continuing..."
}

echo ""
echo "Step 2/4: Downloading assets..."
python -m objathor.dataset.download_assets --version "$VERSION" --path "$DATA_DIR" || {
    echo "Warning: Failed to download assets, continuing..."
}

echo ""
echo "Step 3/4: Downloading annotations..."
python -m objathor.dataset.download_annotations --version "$VERSION" --path "$DATA_DIR" || {
    echo "Warning: Failed to download annotations, continuing..."
}

echo ""
echo "Step 4/4: Downloading features (embeddings)..."
python -m objathor.dataset.download_features --version "$VERSION" --path "$DATA_DIR" || {
    echo "Warning: Failed to download features, continuing..."
}

echo ""
echo "Download complete!"
echo ""
echo "Data stored in: $DATA_DIR"
echo ""
echo "To use this data, update GPT/constants.py:"
echo "  ABS_PATH_OF_HOLODECK = '$DATA_DIR'"
echo ""
echo "Or set environment variable:"
echo "  export OBJATHOR_ASSETS_BASE_DIR='$DATA_DIR'"
