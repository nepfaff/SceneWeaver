# import os
import sys
sys.path.append("/home/yandan/workspace/digital-cousins/")
from digital_cousins.models.blend.launch_blender import open_blender_for_generation
# import yaml

# from digital_cousins.pipeline.generation import SimulatedSceneGenerator

def render_generation(config_step3_filename):
    render_script = "/home/yandan/workspace/digital-cousins/digital_cousins/pipeline/generation.py"
    open_blender_for_generation(render_script,config_step3_filename)

    return 

# if __name__== "__main__":

#     config_step3_filename = str(sys.argv[-1])

#     with open("/home/yandan/workspace/digital-cousins/digital_cousins/configs/default.yaml","r") as f:
#         config = yaml.load(f, Loader=yaml.FullLoader)


#     step_3 = SimulatedSceneGenerator(
#         verbose=config["verbose"],
#     )

#     del config["verbose"]
#     success, step_3_output_path = step_3(**config)
#     if not success:
#         raise ValueError("Failed ACDC Step 3!")