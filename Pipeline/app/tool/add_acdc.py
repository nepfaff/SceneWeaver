import json
import os
import subprocess

from gpt import GPT4

from app.prompt.acdc_cand import system_prompt, user_prompt
from app.tool.base import BaseTool
from app.tool.update_infinigen import update_infinigen
from app.utils import extract_json

# Get SceneWeaver root directory
_SCENEWEAVER_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def get_acdc_dir() -> str:
    """Get ACDC directory - defaults to tools/acdc within SceneWeaver."""
    return os.getenv("ACDC_DIR", os.path.join(_SCENEWEAVER_DIR, "tools", "acdc"))


def get_sd35_dir() -> str:
    """Get SD 3.5 directory - defaults to tools/sd3.5 within SceneWeaver."""
    return os.getenv("SD35_DIR", os.path.join(_SCENEWEAVER_DIR, "tools", "sd3.5"))

DESCRIPTION = """
Using image generation and 3D reconstruction to add additional objects into the current scene.

Use Case 1: Add **a group of** small objects on the top of an empty and large furniture, such as a table, cabinet, and desk when there is nothing on its top. 

You **MUST** not:
1.Do not add objects where there is no available space.
2.Do not add objects where there already exists other small objects.
3.Do not add small objects on any tall furniture, such as wardrob.
4.Do not add small objects on small supporting surface, such as nightstand.
5.Do not add small objects on concave furniture, such as sofa and shelf.

Strengths: Real. Excellent for adding a group of objects with inter-relations on the top of a large furniture.(e.g., enriching a tabletop), such as adding (laptop,mouse,keyboard) set on the desk and (plate,spoon,food) set on the dining table. Accurate in rotation. 
Weaknesses: Very slow. Can not add objects on the wall, ground, or ceiling. Can not add objectsinside a container, such as objects in the shelf. Can not add objects when there is already something on the top.

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
        # 1 generate prompt for sd + 2 use sd to generate image + 3 use acdc to reconstruct 3D scene
        user_demand = os.getenv("UserDemand")
        iter = int(os.getenv("iter"))
        roomtype = os.getenv("roomtype")
        action = self.name
        try:
            # 1 generate prompt for sd
            steps = gen_ACDC_cand(user_demand, ideas, roomtype, iter)

            inplace = False
            acdc_record = dict()
            for obj_id, info in steps.items():
                sd_prompt = info["prompt for SD"]
                if sd_prompt not in acdc_record:
                    update_infinigen(
                        "export_supporter", iter, json_name="", description=obj_id
                    )
                    cnt = 0
                    while True and cnt < 5:
                        cnt += 1
                        print(sd_prompt)
                        # 2 use sd to generate image
                        img_filename = gen_img_SD(
                            sd_prompt, obj_id, info["obj_size"]
                        )  # execute until satisfy the requirement

                        # 3 use acdc to reconstruct 3D scene
                        _ = acdc(img_filename, obj_id, info["obj category"])

                        acdc_dir = get_acdc_dir()
                        with open(os.path.join(acdc_dir, "args.json"), "r") as f:
                            j = json.load(f)
                            if j["success"]:
                                save_dir = os.getenv("save_dir")
                                newid = obj_id.replace(" ", "_")
                                foldername_old = f"{save_dir}/pipeline/acdc_output/step_3_output/scene_0/"
                                foldername_new = f"{save_dir}/pipeline/{newid}"
                                os.system(f"cp -r {foldername_old} {foldername_new}")
                                json_name = f"{foldername_new}/scene_0_info.json"
                                acdc_record[sd_prompt] = json_name
                                break
                    assert j["success"]
                else:
                    json_name = acdc_record[sd_prompt]

                update_infinigen(
                    action,
                    iter,
                    json_name,
                    description=obj_id,
                    inplace=inplace,
                    ideas=ideas,
                )
                inplace = True

            return "Successfully add objects with ACDC."
        except Exception:
            return "Error adding objects with ACDC"


def acdc(img_filename, obj_id, category):
    """Run ACDC pipeline to reconstruct 3D scene from image.

    Uses the ACDC venv at tools/acdc/.venv (separate from SceneWeaver's venv
    due to conflicting dependencies like bpy).
    """
    acdc_dir = get_acdc_dir()

    j = {
        "obj_id": obj_id,
        "objtype": category.lower(),
        "img_filename": img_filename,
        "success": False,
        "error": "Unknown",
    }

    args_path = os.path.join(acdc_dir, "args.json")
    with open(args_path, "w") as f:
        json.dump(j, f, indent=4)

    # Get configuration from environment
    api_key = os.getenv("OPENAI_API_KEY", "")
    save_dir = os.getenv("save_dir", "")
    log_file = os.path.join(_SCENEWEAVER_DIR, "Pipeline", "run.log")

    # ACDC has its own venv due to conflicting dependencies (bpy, specific torch version)
    # Use activate_acdc.sh to set up venv AND proper PYTHONPATH for dependencies
    activate_script = os.path.join(acdc_dir, "activate_acdc.sh")

    # Run ACDC pipeline in its own venv via subprocess
    # Using source activate_acdc.sh sets up venv + PYTHONPATH for deps like metric_depth
    # Source ~/.bashrc to get OPENAI_API_KEY and other env vars
    cmd = f"""
source ~/.bashrc
cd "{acdc_dir}"
source "{activate_script}"
python digital_cousins/pipeline/acdc_pipeline.py --gpt_api_key "$OPENAI_API_KEY" > "{log_file}" 2>&1
"""
    subprocess.run(["bash", "-c", cmd])

    json_name = (
        f"{save_dir}/pipeline/acdc_output/step_3_output/scene_0/scene_0_info.json"
    )

    return json_name


def gen_img_SD(SD_prompt, obj_id, obj_size):
    """Generate image using Stable Diffusion 3.5."""
    sd35_dir = get_sd35_dir()
    save_dir = os.getenv("save_dir")
    save_dir = os.path.abspath(save_dir)  # Convert to absolute path for SD 3.5
    img_filename = f"{save_dir}/pipeline/SD_img.jpg"

    j = {"prompt": SD_prompt, "img_savedir": img_filename}

    prompt_path = os.path.join(sd35_dir, "prompt.json")
    with open(prompt_path, "w") as f:
        json.dump(j, f, indent=4)

    run_script = os.path.join(sd35_dir, "run.sh")
    os.system(f"bash {run_script}")

    return img_filename


def gen_ACDC_cand(user_demand, ideas, roomtype, iter):
    save_dir = os.getenv("save_dir")
    with open(f"{save_dir}/record_scene/layout_{iter-1}.json", "r") as f:
        layout = json.load(f)
    layout = layout["objects"]

    # convert size
    for key in layout.keys():
        size = layout[key]["size"]
        size_new = [size[1], size[0], size[2]]
        layout[key]["size"] = size_new

    gpt = GPT4(version="4.1")

    user_prompt_1 = user_prompt.format(
        user_demand=user_demand, ideas=ideas, roomtype=roomtype, scene_layout=layout
    )

    prompt_payload = gpt.get_payload(system_prompt, user_prompt_1)

    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    results = extract_json(gpt_text_response)

    with open(f"{save_dir}/pipeline/acdc_candidates_{iter}.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
