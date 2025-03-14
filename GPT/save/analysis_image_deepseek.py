import agent_prompt as prompts
# from gpt import GPT4 
from deepseek import DeepSeek
from prompt_room import extract_json, dict2str
import json


render_path = f"/home/yandan/workspace/infinigen/render_deepseek6.jpg"
with open(f"/home/yandan/workspace/infinigen/layout_deepseek6.json", "r") as f:
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
- The X-Y plane is the floor; the Z axis points up. The origin is at a corner (the left-top corner of the rendered image), defining the global frame.
- Original asset (without rotation) faces point along the positive X axis. The Z axis points up. The local origin is centered in X-Y and at the bottom in Z. 
- A 90-degree Z rotation means that the object will face the positive Y axis. The bounding box aligns with the assets local frame.
For the image:
- The origin point x,y =[0,0] represents the top-left corner of the image.
- The x-coordinate increases from left to right (positive x is to the right).
- The y-coordinate usually increases from top to bottom (positive y is downward).

"""
user_prompt = f"""
This is a classroom. 
The room size is [10,12] in length and width.
This is the scene layout: {s}. 
This is the image render from the top view: SCENE_IMAGE 

Please take a moment to relax and carefully look through each object and their relations.
What problem do you think it has? 
Then tell me how to solve these problems.

Fianlly, according to the problem and thoughts, you should modify objects' layout to fix each of the problem.
Keep the objects inside the room.

Before returning the final results, you need to carefully confirm that each issue has been resolved. 
If not, update the layout until each problem is resolved.

Provide me with the new layout in json format.


"""

    
ds = DeepSeek()

prompt_payload = ds.get_payload_scene_image(system_prompt, user_prompt,render_path=render_path )
gpt_text_response = ds(payload=prompt_payload, verbose=True)
print(gpt_text_response)
