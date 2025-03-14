import agent_prompt as prompts
from gpt import GPT4 
from prompt_room import extract_json, dict2str
import json


render_path = f"/home/yandan/workspace/infinigen/render_3+.jpg"
with open(f"/home/yandan/workspace/infinigen/layout_deepseek5.json", "r") as f:
    layout = json.load(f)
s = dict2str(layout["objects"])
system_prompt = """
You are an expert in 3D scene evaluation. 
Your task is to : 
1) evaluate the current scene, 
2) tell me what problem it has, 
3) help me solve the problem.

**3D Convention:**
- Right-handed coordinate system.
- The X-Y plane is the floor; the Z axis points up. The origin is at a corner, defining the global frame.
- Original asset (without rotation) faces point along the positive X axis. The Z axis points up. The local origin is centered in X-Y and at the bottom in Z. 
- A 90-degree Z rotation means that the object will face the positive Y axis. The bounding box aligns with the assets local frame.

"""
user_prompt = f"""
This is a classroom. 
The room size is [10,12] in length and width.
This is the scene layout: {s}. 

Please take a moment to relax and carefully look through each object and their relations.
What problem do you think it has? 
Then tell me how to solve these problems.

Fianlly, according to the problem and thoughts, you should modify objects' layout to fix each of the problem.
Keep the objects inside the room.

Before returning the final results, you need to carefully confirm that each issue has been resolved. 
If not, update the layout until each problem is resolved.

Provide me with the new layout in json format.

"""

    
gpt = GPT4()

prompt_payload = gpt.get_payload_scene_image(system_prompt, user_prompt,render_path=render_path)
gpt_text_response = gpt(payload=prompt_payload, verbose=True)
print(gpt_text_response)
