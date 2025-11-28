#!/bin/bash
# Stop on errors
set -e

# Run the script
cd /home/yandan/workspace/digital-cousins

# Source Conda WITHOUT relying on .bashrc
source "/home/yandan/anaconda3/etc/profile.d/conda.sh"

# Deactivate any existing environment
conda deactivate || true

# Activate target environment
conda activate idesign

save_dir=$1
echo $save_dir
python digital_cousins/models/objaverse/idesign_retriever.py ${save_dir}