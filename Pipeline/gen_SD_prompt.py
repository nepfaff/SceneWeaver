
from gpt import GPT4 
from utils import extract_json, dict2str
import json

system_prompt = """
"Given a user's demand, a large furniture type, and its size in length, width, and height, generate a descriptive prompt suitable for the Stable Diffusion model.

The output should:
1.Clearly specify the furniture type.
2.Include its size in the format 'L*W*H cm'.
3.Incorporate the small objects mentioned in the user's demand.
4.Ensure natural and concise wording.

Example Input:
User's demand: Add tissue and flower on the table.
Large furniture's type: Table
Large furniture's size: 80*60*50

Expected Output:
A 80cm * 60cm * 50cm table with tissue and flower on it."
"""
user_prompt = """
Input:
User's demand: {user_demand}
Large furniture's type: {objtype}
Large furniture's size: {objsize}
"""

def gen_SD_prompt(prompt,objtype,obj_size):
    
    user_prompt_1 = user_prompt.format(user_demand=prompt,objtype=objtype,objsize=obj_size) 
        
    gpt = GPT4()

    prompt_payload = gpt.get_payload(system_prompt, user_prompt_1)
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    # new_layout = extract_json(gpt_text_response)

    # with open(f"update_gpt_results_{iter}.json", "w") as f:
    #     json.dump(new_layout, f, indent=4)

    return gpt_text_response
    