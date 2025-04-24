# import os
# os.system("bash -i /home/yandan/workspace/infinigen/run_invisible.sh")

import subprocess

cmd = """
source ~/anaconda3/etc/profile.d/conda.sh
conda deactivate
cd /home/yandan/workspace/digital-cousins
conda activate acdc2
python digital_cousins/pipeline/acdc_pipeline.py --gpt_api_key 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 > /home/yandan/workspace/infinigen/Pipeline/run.log 2>&1
"""
subprocess.run(["bash", "-c", cmd])