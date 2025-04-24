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
from app.tool.init_gpt import InitGPTExecute
import app.prompt.gpt.init_gpt as prompts0
import app.prompt.gpt.add_gpt as prompts1
from app.utils import extract_json, dict2str, lst2str

DESCRIPTION="""
Using GPT to add additional objects into the current scene.

Use Case 1: Add large objects in the current scene.
Use Case 2: Add sparse small objects on the top of large furniture. (e.g., add a cup on the table).
Use Case 3: Add small objects inside the large furniture. (e.g., add books in the shelf).

Strengths: The location is accurate. Can add objects inside a container, such as objects in the shelf.
Weaknesses: Can not add objects on the wall or ceiling. The rotation of asset is not accurate. Relation between small objects is not accurate. Can not modify objects in the current scene.  

"""


class AddGPTExecute(InitGPTExecute):
    """A tool for executing Python code with timeout and safety restrictions."""
    
    name: str = "add_gpt"
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
            #find scene
            json_name = self.add_gpt(user_demand, ideas, iter,roomtype)
            
          
            success = update_infinigen(
                action, iter, json_name
            )
            assert success

            return f"Successfully add objects with GPT."
        except Exception as e:
            return f"Error adding objects with GPT"

 
    def add_gpt(self,user_demand, ideas, iter,roomtype):
        json_name = self.generate_scene_iter1_gpt(user_demand, ideas, iter,roomtype)

        return json_name
    

    def generate_scene_iter1_gpt(self,user_demand,ideas,iter,roomtype):

        gpt = GPT4(version="4.1")

        results = dict()
        save_dir = os.getenv("save_dir")
        render_path = f"{save_dir}/record_scene/render_{iter-1}.jpg"
        with open(f"{save_dir}/record_scene/layout_{iter-1}.json", "r") as f:
            layout = json.load(f)

        roomsize = layout["roomsize"]
        
        roomsize_str = f"[{roomsize[0]},{roomsize[1]}]"
        step_1_big_object_prompt_user = prompts1.step_1_big_object_prompt_user.format(demand=user_demand, 
                                                                                    roomtype = roomtype,
                                                                                    ideas = ideas,
                                                                                    roomsize = roomsize_str,
                                                                                    scene_layout=layout["objects"],
                                                                                    structure = layout["structure"])
        
        prompt_payload = gpt.get_payload_scene_image(prompts1.step_1_big_object_prompt_system, 
                                                    step_1_big_object_prompt_user,
                                                    render_path)
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        gpt_dict_response = extract_json(gpt_text_response)
        results = gpt_dict_response

        # #### 2. get object class name in infinigen
        category_list = gpt_dict_response["Number of new furniture"]
        if len(category_list.keys())==0:
            return "Nothing"
        s = lst2str(list(category_list.keys()))
        user_prompt = prompts0.step_3_class_name_prompt_user.format(
            category_list=s, demand=user_demand
        )
        system_prompt = prompts0.step_3_class_name_prompt_system
        prompt_payload = gpt.get_payload(system_prompt, user_prompt)
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        gpt_dict_response = extract_json(
            gpt_text_response.replace("'", '"').replace("None", "null")
        )
        name_mapping = gpt_dict_response["Mapping results"]
        results["name_mapping"] = name_mapping

        save_dir = os.getenv("save_dir")
        json_name = f"{save_dir}/add_gpt_results_{iter}.json"
        with open(json_name, "w") as f:
            json.dump(results, f, indent=4)
        return json_name
