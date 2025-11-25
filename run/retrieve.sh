#!/bin/bash
# Objaverse asset retrieval script
# Tries Holodeck retrieval first, falls back to OpenShape, then empty fallback

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENEWEAVER_DIR="$(dirname "$SCRIPT_DIR")"

# Run the script from SceneWeaver directory
cd "$SCENEWEAVER_DIR"
source .venv/bin/activate

save_dir=$1
echo "Retrieving assets for: $save_dir"

# Method 1: Try Holodeck/layoutvlm retrieval (simpler, uses pre-downloaded data)
echo "Trying Holodeck retrieval..."
if python infinigen/assets/objaverse_assets/retrieve_holodeck.py "${save_dir}" 2>&1; then
    # Check if we got any results
    if [ -s "${save_dir}/objav_files.json" ] && [ "$(cat "${save_dir}/objav_files.json")" != "{}" ]; then
        echo "Holodeck retrieval completed successfully"
        exit 0
    fi
fi

# Method 2: Try OpenShape retrieval (requires OpenShape/MinkowskiEngine)
echo "Trying OpenShape retrieval..."
if python infinigen/assets/objaverse_assets/retrieve_idesign.py "${save_dir}" 2>&1; then
    echo "OpenShape retrieval completed successfully"
    exit 0
fi

# Fallback: Create empty file
echo "Warning: All retrieval methods failed. Creating empty fallback."
echo "{}" > "${save_dir}/objav_files.json"