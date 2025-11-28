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

    # Use a temp directory for SD3.5 output
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

    # Find the generated image (it creates subdirs with prompt text)
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
