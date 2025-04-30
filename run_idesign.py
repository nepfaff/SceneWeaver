import json
import os
import subprocess
import time

# Room sizes for each task (replace with your actual room sizes)
room_sizes = [
#bedroom
[3.0, 4.0, 2.4],
[2.5, 3.0, 2.4],
[3.5, 4.5, 2.4],
[4.0, 5.0, 2.4],
[2.4, 3.5, 2.4],
[3.2, 4.2, 2.4],
[2.8, 3.6, 2.4],
[3.6, 4.8, 2.4],
[4.2, 5.2, 2.4],
[3.0, 3.5, 2.4],
#living room
[4.0, 5.0, 2.8],
[3.5, 4.5, 2.8],
[3.0, 4.0, 2.8],
[4.5, 6.0, 3.0],
[5.0, 7.0, 3.0],
[3.6, 4.8, 2.8],
[4.2, 5.2, 2.8],
[5.5, 6.5, 3.0],
[3.2, 4.2, 2.8],
[6.0, 8.0, 3.0]]

# Paths
args_path = "args_idesign.json"
roominfo_path = "roominfo_idesign.json"

# for i in range(4,5):
for i in [11,13,18]:
    print(f"\n=== Running Task {i} ===")

    # Edit args_idesign.json
    with open(args_path, 'r') as f:
        args_data = json.load(f)
    args_data["json_name"] = f"/mnt/fillipo/yandan/scenesage/idesign/scene_sage/scene_graph/scene_graph_{i}.json"
    with open(args_path, 'w') as f:
        json.dump(args_data, f, indent=4)

    # Edit roominfo_idesign.json
    with open(roominfo_path, 'r') as f:
        roominfo_data = json.load(f)

    room_size  = room_sizes[i][:2]
    room_size = [i+0.28 for i in room_size]
    roominfo_data["roomsize"] = room_size
    roominfo_data["save_dir"] = f"/mnt/fillipo/yandan/scenesage/record_scene/idesign/scene_{i}"
    with open(roominfo_path, 'w') as f:
        json.dump(roominfo_data, f, indent=4)

    # Run the Blender generation command
    command = [
        "python", 
        # "-m", "infinigen.launch_blender",
        "-m", "infinigen_examples.generate_indoors_idesign",
        # "--", 
        "--seed", "0", "--task", "coarse", "--method", "idesign",
        "--output_folder", "outputs/indoors/coarse_expand_whole_nobedframe",
        "-g", "fast_solve.gin", "overhead.gin", "studio.gin",
        "-p", "compose_indoors.terrain_enabled=False"
    ]
    subprocess.run(command)
    # subprocess.run(["bash", "-c", command])

    time.sleep(2)  # Optional: wait a bit between tasks
    
print("\n🎯 All tasks finished!")
