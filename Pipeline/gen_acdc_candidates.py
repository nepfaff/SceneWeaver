
from gpt import GPT4 
from utils import extract_json, dict2str
import json
system_prompt = """
Given the information of current scene, you need to add small objects on the large furnitures. 

You will receive:
1.User's demand of the entire scene
2.Ideas to update the scene in this step
3.RoomType
4.Current Scene Layout, including each object's ID, location, rotation, and size.

Your response will be sent to generate some images of the partial scene, and then a reconstruction method to reconstruct the image into the partial 3D scene.
You should return the info for updating each large object:
1. prompt for SD: A descriptive prompt suitable for the Stable Diffusion model. 
2. obj category: The category of the large object, ensuring alignment with the SD prompt.
3. obj_id: You need to recognize which large objects should be used to add small objects.
4. obj_size: You also need to return the size of these large objects. 

The prompt for SD should:
1.Clearly specify the furniture type.
2.Include its size in the format 'L*W*H cm'.
3.Incorporate the small objects mentioned in the Ideas. 
4.If multiple large furniture pieces are present, distribute small objects naturally, ensuring slight variations to enhance realism.
5.Ensure natural and concise wording.

Expected Output is in json format:
{
    "2049208_TableFactory":
        {
            "prompt for SD": "An entire 80cm * 60cm * 50cm simple table with tissue and flower on it.",
            "obj category": "table",
            "obj_id": "2049208_TableFactory",
            "obj_size": [0.8,0.6,0.5]
        },
    "2360448_SimpleDeskFactory":
        {
            "prompt for SD": "An entire 120cm * 50cm * 80cm simple desk with a laptop, a keyboard, and a mouse on it.",
            "obj category": "desk",
            "obj_id": "2360448_SimpleDeskFactory",
            "obj_size": [1.2,0.5,0.8]
        }
}

"""

# Example Input:
# User's demand: Add tissue and flower on the table.
# User's demand: a living room
# Ideas: Add tissue and flower on the table
# RoomType: living room
# Current Scene Layout: {scene_layout}

user_prompt = """
Input:
User's demand: {user_demand}
Ideas: {ideas}
RoomType: {roomtype}
Current Scene Layout: {scene_layout}
"""

def gen_ACDC_cand(user_demand,ideas,roomtype,iter):
    
    with open(f"/home/yandan/workspace/infinigen/record_scene/layout_{iter-1}.json", "r") as f:
        layout = json.load(f)
    layout = layout["objects"]

    #convert size
    for key in layout.keys():
        size = layout[key]["size"]
        size_new = [size[1],size[0],size[2]]
        layout[key]["size"] = size_new

    gpt = GPT4()

    user_prompt_1 = user_prompt.format(user_demand=user_demand,
                                       ideas=ideas,
                                       roomtype=roomtype,
                                       scene_layout = layout) 

    prompt_payload = gpt.get_payload(system_prompt, user_prompt_1)

    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    results = extract_json(gpt_text_response)

    with open(f"acdc_candidates_{iter}.json", "w") as f:
        json.dump(results, f, indent=4)

    return results
    