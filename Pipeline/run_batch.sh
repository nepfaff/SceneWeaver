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
  "Basement transformed into a practice studio contains a full drum set, an electric guitar resting on the wall, and a large speaker mounted on the wall."
  "A Japanese-style living room featuring a coffee table next to the window with two floor cushions placed beside it. A sofa is positioned across from the window, and in front of the sofa is a table with a teapot on top, completing the serene and minimalist setup."
  "A teenager's bedroom features a comfortable twin bed with a backboard in the far corner, with boxes underneath it. At the foot of the bed is a small desk equipped with a monitor, an external keyboard and mouse, and a desk lamp on the right for visibility, accompanied by a rolling chair. Next to the bed, a nightstand with an additional floor lamp nearby provides space for a phone and other valuables. A sizable wooden wardrobe with multiple drawers offers ample storage for clothes, while a coffee table beside it holds books and board games. In the center of the room, a tan-colored rug creates a cozy spot to sit, and the walls are adorned with various posters and pictures."
  "An open-plan dining area featuring a round table surrounded by six mid-century modern chairs. A grand chandelier illuminates the table, while a sideboard with four drawers rests against the wall, topped with three potted plants. Two wall sconces provide additional lighting, flanking a large mirror that hangs between them. A wine cabinet is positioned in the corner, with a serving cart placed nearby for added convenience."
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
