import json
import re
from functools import reduce
import init_gpt_prompt as prompts0
import add_gpt_prompt as prompts1
from gpt import GPT4
from utils import extract_json, lst2str



def generate_scene_iter1(user_demand,ideas,iter):

    gpt = GPT4()

    results = dict()
    render_path = f"/home/yandan/workspace/infinigen/render_{iter-1}.jpg"
    with open(f"/home/yandan/workspace/infinigen/layout_{iter-1}.json", "r") as f:
        layout = json.load(f)

    roomsize = layout["roomsize"]
    roomsize_str = f"[{roomsize[0]},{roomsize[1]}]"
    step_1_big_object_prompt_user = prompts1.step_1_big_object_prompt_user.format(demand=user_demand, 
                                                                                ideas = ideas,
                                                                                roomsize = roomsize_str,
                                                                                scene_layout=layout["objects"])
    
    prompt_payload = gpt.get_payload_scene_image(prompts1.step_1_big_object_prompt_system, 
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

    json_name = f"/home/yandan/workspace/infinigen/Pipeline/record/add_gpt_results_{iter}.json"
    with open(json_name, "w") as f:
        json.dump(results, f, indent=4)
    return json_name

if __name__ == "__main__":
    # user_demand = "Add floor lamp next to the armchair and wall art above the sofa."
    user_demand = "Add vase with flowers on the coffee table, magazines, and remote controls on the TV stand."
    generate_scene_iter1(user_demand,iter=4)
