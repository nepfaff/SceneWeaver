#!/bin/bash
# ============================================================================
# SceneWeaver Batch Scene Generator
# ============================================================================
# Edit the prompts array below to customize your scenes.
# Each prompt will generate one scene in Pipeline/output/
#
# Usage:
#   cd Pipeline
#   chmod +x run_batch.sh
#   ./run_batch.sh
# ============================================================================

prompts=(
  "A kid's bedroom with a pastel pink twin bed against the back wall, a low white dresser to the right of the bed, and a simple wooden desk to the left of the bed. Toys are scattered on the floor."
  "A contemporary living room with a leather sofa, a small coffee table, and a vintage typewriter."
  "Generate a dining room. The room should have a rectangular, wooden dining table with three chairs surrounding it. There is a utility cart near the table with three plates inside."
  "A living room with a sectional sofa, a wooden coffee table in the middle, and a retro record player in the corner with some records nearby."
  "A dining room with a circular table surrounded by six vintage chairs, and an old wooden ladder against the wall displaying plants and decorative jars."
)

# ============================================================================
# Setup and run
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

source .venv/bin/activate
set -a && source .env && set +a
cd Pipeline

total=${#prompts[@]}
echo "=========================================="
echo "SceneWeaver Batch Generator"
echo "Generating $total scenes..."
echo "=========================================="

for i in "${!prompts[@]}"; do
  scene_num=$((i+1))
  echo ""
  echo "[$scene_num/$total] Starting scene generation..."
  echo "Prompt: ${prompts[$i]:0:60}..."
  echo "------------------------------------------"

  python main.py --prompt "${prompts[$i]}" --cnt 1 --basedir ./output/

  echo "[$scene_num/$total] Complete!"
done

echo ""
echo "=========================================="
echo "Batch complete! $total scenes generated."
echo "Output saved to: Pipeline/output/"
echo "=========================================="
