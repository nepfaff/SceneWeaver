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
