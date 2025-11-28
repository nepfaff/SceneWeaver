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
# metric_depth is inside Depth-Anything-V2, so we need both:
# - Depth-Anything-V2 for depth_anything_v2 imports
# - Depth-Anything-V2/metric_depth for metric_depth imports
export PYTHONPATH="${ACDC_DIR}/deps/dinov2:${ACDC_DIR}/deps/Depth-Anything-V2:${ACDC_DIR}/deps/Depth-Anything-V2/metric_depth:${PYTHONPATH}"

# Set checkpoint paths
export ACDC_CHECKPOINTS="${ACDC_DIR}/checkpoints"

# Set Holodeck path for asset retrieval
export HOLODECK_DIR="${HOLODECK_DIR:-$(dirname ${ACDC_DIR})/Holodeck}"

echo "ACDC environment activated"
echo "  ACDC_DIR: ${ACDC_DIR}"
echo "  HOLODECK_DIR: ${HOLODECK_DIR}"
echo "  PYTHONPATH includes: dinov2, Depth-Anything-V2"
echo "  ACDC_CHECKPOINTS: ${ACDC_CHECKPOINTS}"
