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
from app.prompt.add_relation import system_prompt,user_prompt,example

DESCRIPTION="""
Add explicit relation between objects in the current scene according to the layout. 
Sometimes the relation is encoded in the layout coordinate rather than represented explicitly, making it difficult to manage.
Explicit relations will make the scene more tidy.

Use Case 1: Add explicit relation between objects, according to the layout, to make the scene better-organized. 
Use Case 2: Add new relation between objects, make the scene better-organized. 

Strengths: Can add relation between objects, make the scene tidy and well-organized quickly. 
Weaknesses: Can not fix the layout problem, such as placing the object into the right place accurately. Each object can have only one relation with other object (except for the room).

"""


class AddRelationExecute(BaseTool):
    
    name: str = "add_relation"
    description: str = DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "ideas": {
                "type": "string",
                "description": "(required) The ideas to add relations in this step.",
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
            json_name = add_relation(user_demand, ideas, iter, roomtype)
            success = update_infinigen(
                action, iter, json_name, inplace=False, invisible=True
            )
            assert success
            return f"Successfully add relation between objects."
        except Exception as e:
            return f"Error adding relation between objects"


def add_relation(user_demand, ideas, iter, roomtype):
    save_dir = os.getenv("save_dir")
    if iter==0:
        render_path = f"{save_dir}/record_scene/render_{iter}.jpg"
        with open(
            f"{save_dir}/record_scene/layout_{iter}.json", "r"
        ) as f:
            layout = json.load(f)
    else:
        render_path = f"{save_dir}/record_scene/render_{iter-1}.jpg"
        with open(
            f"{save_dir}/record_scene/layout_{iter-1}.json", "r"
        ) as f:
            layout = json.load(f)
    

    roomsize = layout["roomsize"]
    roomsize = lst2str(roomsize)

    structure = dict2str(layout["structure"])
    layout = dict2str(layout["objects"])

    system_prompt_1 = system_prompt
    user_prompt_1 = (
        user_prompt.format(
            roomtype=roomtype,
            roomsize=roomsize,
            layout=layout,
            structure=structure,
            user_demand=user_demand,
            ideas=ideas,
        )
        + example
    )

    gpt = GPT4(version="4.1")

    prompt_payload = gpt.get_payload_scene_image(
        system_prompt_1, user_prompt_1, render_path=render_path
    )
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)

    json_name = f"{save_dir}/pipeline/add_relation_results_{iter}_response.json"
    with open(json_name, "w") as f:
        json.dump(gpt_text_response, f, indent=4)

    new_layout = extract_json(gpt_text_response)

    json_name = f"{save_dir}/pipeline/add_relation_results_{iter}.json"
    with open(json_name, "w") as f:
        json.dump(new_layout, f, indent=4)

    return json_name