import json
import os
import subprocess
import time


# Paths
args_path = "args_layoutgpt.json"
roominfo_path = "roominfo_layoutgpt.json"

roomtype="livingroom"
for i in range(9,10):
# for i in [11,13,18]:
    print(f"\n=== Running Task {i} ===")

    # Edit args_idesign.json
    with open(args_path, 'r') as f:
        args_data = json.load(f)
    args_data["json_name"] = f"/mnt/fillipo/yandan/scenesage/LayoutGPT/{roomtype}_scene_graph/layoutgpt_{roomtype}_{i}.json"
    with open(args_path, 'w') as f:
        json.dump(args_data, f, indent=4)

    # Edit roominfo_idesign.json
    with open(roominfo_path, 'r') as f:
        roominfo_data = json.load(f)

   
    with open(args_data["json_name"], 'r') as f:
        j = json.load(f)
        roomsize = j["room_size"]

        
    room_size  = [roomsize["length"],roomsize["width"]]
    room_size = [i+0.28 for i in room_size]
    roominfo_data["roomsize"] = room_size
    roominfo_data["save_dir"] = f"/mnt/fillipo/yandan/scenesage/record_scene/layoutgpt/{roomtype}_{i}"
    with open(roominfo_path, 'w') as f:
        json.dump(roominfo_data, f, indent=4)

    # Run the Blender generation command
    command = [
        "python", 
        # "-m", "infinigen.launch_blender",
        "-m", "infinigen_examples.generate_indoors_idesign",
        # "--", 
        "--seed", "0", "--task", "coarse", "--method", "layoutgpt",
        "--output_folder", "outputs/indoors/coarse_expand_whole_nobedframe",
        "-g", "fast_solve.gin", "overhead.gin", "studio.gin",
        "-p", "compose_indoors.terrain_enabled=False"
    ]
    subprocess.run(command)

    time.sleep(2)  # Optional: wait a bit between tasks
    
print("\n🎯 All tasks finished!")
