
from digital_cousins.models.blend.launch_blender import open_blender_for_render
import os

from digital_cousins.models.blend.configs import RENDER_DIR

def render_candidates(candidates,dataset,start_angle,end_angle,cnt):
    os.system(f"rm -r {RENDER_DIR}/*")
    render_script = "digital_cousins/models/blend/blender_render.py"
    
    
    candidates_imgs = []
    for candidate in candidates:
        open_blender_for_render(render_script,candidate,dataset,start_angle,end_angle,cnt,bkg=1)

        # if "/" in candidate and (not candidate.endswith(".blend")):
        #     candidate = candidate.split("/")[-2]

        if candidate.endswith(".blend"):
            save_dir = f"{RENDER_DIR}/infinigen"
        elif dataset=="holodeck":
            if "/" in candidate:
                candidate = candidate.split("/")[-2]
            save_dir = f"{RENDER_DIR}/{candidate}"
        else:
            objname = candidate.split("/")[-1].split(".")[0]
            save_dir = f"{RENDER_DIR}/{objname}"

        if os.path.exists(save_dir):
            files = os.listdir(save_dir)
            files = [f"{save_dir}/{file}" for file in files]
            candidates_imgs += files
        
    return candidates_imgs