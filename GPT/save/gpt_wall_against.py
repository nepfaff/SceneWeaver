import json
import re
from functools import reduce
import method_4_GPT_iter0_prompt as prompts0
import method_4_GPT_iter1_prompt as prompts1
from gpt import GPT4
from prompt_room import extract_json, dict2str, lst2str
import sys


def generate_scene_iter1(user_demand,iter):

    gpt = GPT4()

    results = dict()
    
    with open(f"/home/yandan/workspace/infinigen/layout{iter-1}.json", "r") as f:
        layout = json.load(f)

    roomsize = layout["roomsize"]
    roomsize_str = f"[{roomsize[0]},{roomsize[1]}]"
    step_1_big_object_prompt_user = prompts1.step_1_big_object_prompt_user.format(demand=user_demand, 
                                                               roomsize = roomsize_str,
                                                               scene_layout=layout["objects"])
    
    prompt_payload = gpt.get_payload(prompts1.step_1_big_object_prompt_system, 
                                                 step_1_big_object_prompt_user,
                                                 render_path)
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)

    gpt_dict_response = extract_json(gpt_text_response)
    results = gpt_dict_response
    


    # #### 2. get object class name in infinigen
    category_list = gpt_dict_response["List of new furniture"]
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

    with open(f"method_4_GPT_iter{iter}_results.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # user_demand = "Add floor lamp next to the armchair and wall art above the sofa."
    user_demand = "Add vase with flowers on the coffee table, magazines, and remote controls on the TV stand."
    generate_scene_iter1(user_demand,iter=4)
