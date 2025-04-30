import sys
from typing import Dict

from app.tool.base import BaseTool
import json
import os
import random
import numpy as np
from gpt import GPT4 
from app.utils import extract_json, dict2str, lst2str
import json

from app.tool.update_infinigen import update_infinigen
from app.tool.add_relation import add_relation

from app.prompt.acdc_cand import system_prompt,user_prompt

DESCRIPTION="""
Using image generation and 3D reconstruction to add additional objects into the current scene.

Use Case 1: Add **a group of** small objects on the top of an empty and large furniture, such as a table, cabinet, and desk when there is nothing on its top. 

Do not add objects where there is no available space or there already exists small objects.
You **MUST** not add small objects on the tall furniture, such as wardrob.
Do not add small objects on small supporting surface, such as nightstand.

Strengths: Real. Excellent for adding a group of objects with inter-relations on the top of a large furniture.(e.g., enriching a tabletop), such as adding (laptop,mouse,keyboard) set on the desk and (plate,spoon,food) set on the dining table. Accurate in rotation. 
Weaknesses: Can not add objects on the wall, ground, or ceiling. Can not add objectsinside a container, such as objects in the shelf. Can not add objects when there is already something on the top.

"""


class AddAcdcExecute(BaseTool):
    """A tool for executing Python code with timeout and safety restrictions."""
    
    name: str = "add_acdc"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The ideas to add objects in this step.",
            },
        },
        "required": ["ideas"],
    }


    def execute(self, ideas: str) -> str:

        user_demand = os.getenv("UserDemand")
        iter = int(os.getenv("iter"))
        roomtype = os.getenv("roomtype")
        action = self.name
        try:
            steps = gen_ACDC_cand(user_demand, ideas, roomtype, iter)
            # steps = {'9563833_nightstand': {'prompt for SD': 'An entire 54cm * 57cm * 59cm nightstand with a lamp on it.', 'obj category': 'nightstand', 'obj_id': '9563833_nightstand', 'obj_size': [...]}, '2552790_shelf': {'prompt for SD': 'An entire 74cm * 31cm * 73cm shelf with books on it.', 'obj category': 'shelf', 'obj_id': '2552790_shelf', 'obj_size': [...]}, '1133652_desk': {'prompt for SD': 'An entire 115cm * 57cm * 77cm desk with a plant on it.', 'obj category': 'desk', 'obj_id': '1133652_desk', 'obj_size': [...]}}
            # last_prompt = 'A 120cm * 60cm * 70cm simple desk with a monitor, desk lamp, stack of documents, and a coffee mug.'
            # last_img_filename = "/home/yandan/workspace/infinigen/Pipeline/record/SD_img.jpg"
            # last_json_name = "/home/yandan/workspace/infinigen/Pipeline/record/acdc_output/step_3_output/scene_0/scene_0_info.json"
            last_prompt = None
            last_img_filename = None
            last_json_name = None
            inplace = False
            for obj_id, info in steps.items():
                if last_prompt is None or info["prompt for SD"] != last_prompt:
                    update_infinigen(
                        "export_supporter", iter, json_name="", description=obj_id
                    )
                    cnt = 0
                    while True and cnt < 5:
                        cnt += 1
                        print(info["prompt for SD"])
                        img_filename = gen_img_SD(
                            info["prompt for SD"], obj_id, info["obj_size"]
                        )  # execute until satisfy the requirement
                        # img_filename = '/home/yandan/workspace/infinigen/Pipeline/record/SD_img.jpg'

                        json_name = acdc(img_filename, obj_id, info["obj category"])

                        with open(
                            "/home/yandan/workspace/digital-cousins/args.json", "r"
                        ) as f:
                            j = json.load(f)
                            if j["success"]:
                                break
                    assert j["success"]
                else:
                    img_filename = last_img_filename
                    json_name = last_json_name

                last_prompt = info["prompt for SD"]
                last_img_filename = img_filename
                last_json_name = json_name

                update_infinigen(
                    action, iter, json_name, description=obj_id, inplace=inplace,ideas=ideas
                )
                inplace = True


            return f"Successfully add objects with ACDC."
        except Exception as e:
            return f"Error adding objects with ACDC"

def acdc(img_filename, obj_id, category):
    # objtype = obj_id.split("_")[1:]
    # objtype = "_".join(objtype)
    j = {
        "obj_id": obj_id,
        "objtype": category,
        "img_filename": img_filename,
        "success": False,
        "error": "Unknown",
    }
    with open("/home/yandan/workspace/digital-cousins/args.json", "w") as f:
        json.dump(j, f, indent=4)


    import subprocess

    cmd = """
    source ~/anaconda3/etc/profile.d/conda.sh
    conda deactivate
    cd /home/yandan/workspace/digital-cousins
    conda activate acdc2
    python digital_cousins/pipeline/acdc_pipeline.py --gpt_api_key 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 > /home/yandan/workspace/infinigen/Pipeline/run.log 2>&1
    """
    subprocess.run(["bash", "-c", cmd])
    # os.system("bash -i /home/yandan/workspace/digital-cousins/run.sh")
    save_dir = os.getenv("save_dir")
    json_name = f"{save_dir}/pipeline/acdc_output/step_3_output/scene_0/scene_0_info.json"

    return json_name
 
def gen_img_SD(SD_prompt, obj_id, obj_size):
    # objtype = obj_id.split("_")[1:]
    # objtype = "_".join(objtype)
    # SD_prompt = gen_SD_prompt(prompt,objtype,obj_size)
    save_dir = os.getenv("save_dir")
    img_filename = f"{save_dir}/pipeline/SD_img.jpg"
    j = {"prompt": SD_prompt, "img_savedir": img_filename}
    with open("/home/yandan/workspace/sd3.5/prompt.json", "w") as f:
        json.dump(j, f, indent=4)

    basedir = "/home/yandan/workspace/sd3.5"
    os.system(f"bash {basedir}/run.sh")

    return img_filename

def gen_ACDC_cand(user_demand,ideas,roomtype,iter):
    save_dir = os.getenv("save_dir")
    with open(f"{save_dir}/record_scene/layout_{iter-1}.json", "r") as f:
        layout = json.load(f)
    layout = layout["objects"]

    #convert size
    for key in layout.keys():
        size = layout[key]["size"]
        size_new = [size[1],size[0],size[2]]
        layout[key]["size"] = size_new

    gpt = GPT4(version="4.1")

    user_prompt_1 = user_prompt.format(user_demand=user_demand,
                                       ideas=ideas,
                                       roomtype=roomtype,
                                       scene_layout = layout) 

    prompt_payload = gpt.get_payload(system_prompt, user_prompt_1)

    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    results = extract_json(gpt_text_response)

    with open(f"{save_dir}/pipeline/acdc_candidates_{iter}.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
    