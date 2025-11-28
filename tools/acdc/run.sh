#!/bin/bash
conda deactivate
cd /home/yandan/workspace/digital-cousins
conda activate acdc


# echo $CONDA_DEFAULT_ENV
# pip list | grep torch

# python --version
# python -c "import digital_cousins"
python digital_cousins/pipeline/acdc_pipeline.py 


# #install
# /home/yandan/software/blender-4.2.0-linux-x64/4.2/python/bin/python3.11 -m pip install kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html
