from get_action import get_action0, get_action1
from init_gpt import generate_scene_iter0
from add_gpt import generate_scene_iter1
from update_gpt import update_scene
from gen_SD_prompt import gen_SD_prompt
from get_roomsize import get_roomsize
from gen_acdc_candidates import gen_ACDC_cand
import os
import random
import json
import numpy as np

def get_action(user_demand,iter):
    if iter==0:
        action,ideas,roomtype = get_action0(user_demand,iter)
    else:
        action,ideas,roomtype = get_action1(user_demand,iter)

    return action, ideas, roomtype

def find_physcene(user_demand,ideas,roomtype):
    roomtype = roomtype[:-5]
    basedir = "/home/yandan/workspace/PhyScene/3D_front/generate_filterGPN_clean/"
    
    for filename in random.shuffle(os.listdir(basedir)):
        if filename.endswith(".json") and roomtype in roomtype:
            break
    
    json_name = f"{basedir}/{filename}"

    def calculate_room_size(data):
        min_coords = np.array([float('inf'), float('inf'), float('inf')])
        max_coords = np.array([-float('inf'), -float('inf'), -float('inf')])
        
        for objects in data["ThreedFront"].values():
            for obj in objects:
                position = np.array(obj["position"])  # Object's position
                size = np.array(obj["size"]) / 2  # Half-size for bounding box calculation
                
                # Compute object's bounding box min and max coordinates
                obj_min = position - size
                obj_max = position + size
                
                # Update overall min/max coordinates
                min_coords = np.minimum(min_coords, obj_min)
                max_coords = np.maximum(max_coords, obj_max)
        
        # Calculate room size (max - min)
        room_size = max_coords - min_coords
        return room_size.tolist()[:2]
    
    with open(json_name,"r") as f:
        data = json.load(f)
        room_size = calculate_room_size(data)

    return json_name,room_size

def find_metascene(user_demand,ideas,roomtype):
    if roomtype.endswith("room"):
        roomtype = roomtype[:-4].strip()
    basedir = "/mnt/fillipo/yandan/metascene/export_stage2_sm"
    
    with open(f"{basedir}/statistic.json","r") as f:
        j = json.load(f)
    
    scenes = j["scenes"]
    scene_ids = list(scenes.keys())
    random.shuffle(scene_ids)
    HasFind = False
    for scene_id in scene_ids:
        scene_type = scenes[scene_id]["roomtype"]
        for info in scene_type:
            if roomtype in info["predicted"] and info["confidence"]>0.8:
                HasFind = True
                break
        if HasFind:
            break

    json_name = scene_id

    with open("/mnt/fillipo/yandan/metascene/export_stage2_sm/roomsize.json","r") as f:
        data = json.load(f)
        room_size = data[scene_id]
        room_size = [round(room_size["size_x"],1),round(room_size["size_y"],1)]

    return json_name,room_size

def gen_gpt_scene(user_demand,ideas,roomtype):
    json_name = generate_scene_iter0(user_demand,ideas,roomtype)
    with open(json_name,"r") as f:
        j = json.load(f)
    roomsize = j["roomsize"]
    return json_name,roomsize

def add_gpt(user_demand,ideas,iter):
    json_name = generate_scene_iter1(user_demand,ideas,iter)
    return json_name

def prepare_acdc(user_demand,ideas,roomtype,iter):
    
    result = gen_ACDC_cand(user_demand,ideas,roomtype,iter)
    
    return result

def gen_img_SD(SD_prompt,obj_id,obj_size):
    # objtype = obj_id.split("_")[1:]
    # objtype = "_".join(objtype)
    # SD_prompt = gen_SD_prompt(prompt,objtype,obj_size)
    img_filename = "/home/yandan/workspace/infinigen/Pipeline/record/SD_img.jpg"
    j = {"prompt":SD_prompt,
         "img_savedir": img_filename}
    with open("/home/yandan/workspace/sd3.5/prompt.json","w") as f:
        json.dump(j,f,indent=4)
    
    basedir = "/home/yandan/workspace/sd3.5"
    os.system(f"bash {basedir}/run.sh")
    
    return img_filename

def update_infinigen(action,iter,json_name,description=None):
    j = {"iter":iter,
         "action":action,
         "json_name":json_name,
        #  "roomsize": roomsize,
         "description":description}
    
    with open(f"/home/yandan/workspace/infinigen/args.json","w") as f:
        json.dump(j,f,indent=4)

    os.system("bash -i /home/yandan/workspace/infinigen/run.sh")

    return

def acdc(img_filename,obj_id,category):
    # objtype = obj_id.split("_")[1:]
    # objtype = "_".join(objtype)
    j = {"obj_id":obj_id,
         "objtype": category,
         "img_filename": img_filename}
    with open("/home/yandan/workspace/digital-cousins/args.json","w") as f:
        json.dump(j,f,indent=4)

    # os.system("conda activate infinigen_python")
    # os.system("bash -i /home/yandan/workspace/digital-cousins/run.sh")
    json_name = "/home/yandan/workspace/infinigen/Pipeline/record/acdc_output/step_3_output/scene_0/scene_0_info.json"
    return json_name

def update_gpt(user_demand,ideas,iter,roomtype):
    update_scene(user_demand,ideas,iter,roomtype)
    json_name = f"update_gpt_results_{iter}.json"
    return json_name

# def choose_solve():
#     return solve_action


iter = 0
user_demand = "An office room for 8 people."

while(iter<10):
    if iter == 0:
        # action, ideas, roomtype = get_action(user_demand,iter)
        action = "init_metascene"
        ideas = "Create a foundational layout for an office room designed to accommodate 8 people."

        # action='init_gpt'
        # ideas='Create a foundational layout for an office room designed for 8 people, including desks, chairs, and basic office equipment.'
        roomtype = 'office'

        if action == "init_physcene":
            json_name,roomsize = find_physcene(user_demand,ideas)
            roomsize = get_roomsize(user_demand,ideas,roomsize,roomtype)
        elif action == "init_metascene":
            json_name,roomsize = find_metascene(user_demand,ideas,roomtype)
            roomsize = get_roomsize(user_demand,ideas,roomsize,roomtype)
        elif action == "init_gpt":
            json_name,roomsize = gen_gpt_scene(user_demand,ideas,roomtype)
            # json_name='/home/yandan/workspace/infinigen/Pipeline/record/init_gpt_results.json'
            roomsize=[5,7]
        else:
            raise ValueError(f"Action is wrong: {action}") 
        
        with open("/home/yandan/workspace/infinigen/roominfo.json","w") as f :
            info = {"action": action,
                    "ideas": ideas,
                    "roomtype": roomtype,
                    "roomsize": roomsize
                    }
            json.dump(info,f,indent=4)
        update_infinigen(action,iter,json_name)
    else:
        # action, ideas, roomtype  = get_action(user_demand,iter)
        action = "add_acdc"
        ideas = "Add computers on each desk, small plants, and personal items like notepads and pens.",
        roomtype="office"
        if action == "add_gpt":
            json_name = add_gpt(user_demand,ideas,iter)
            update_infinigen(action,iter,json_name)
                
        elif action == "add_acdc":
            # steps = prepare_acdc(user_demand,ideas,roomtype,iter)
            steps = {'5778780_SimpleDeskFactory': {'prompt for SD': 'A 60cm * 100cm * 100cm desk with a computer, a small plant, a notepad, and a pen on it.', 'obj category': 'desk', 'obj_id': '5778780_SimpleDeskFactory', 'obj_size': [...]}, '9647033_SimpleDeskFactory': {'prompt for SD': 'A 60cm * 100cm * 100cm desk with a computer, a small plant, and personal items like a notepad and pens.', 'obj category': 'desk', 'obj_id': '9647033_SimpleDeskFactory', 'obj_size': [...]}, '9726544_SimpleDeskFactory': {'prompt for SD': 'A 60cm * 100cm * 100cm desk equipped with a computer, a small plant, and various personal items.', 'obj category': 'desk', 'obj_id': '9726544_SimpleDeskFactory', 'obj_size': [...]}, '4158242_SimpleDeskFactory': {'prompt for SD': 'A 60cm * 100cm * 100cm desk featuring a computer, a small plant, and personal items such as notepads and pens.', 'obj category': 'desk', 'obj_id': '4158242_SimpleDeskFactory', 'obj_size': [...]}, '7874330_SimpleDeskFactory': {'prompt for SD': 'A 60cm * 100cm * 100cm desk with a computer, a small plant, and personal items like notepads and pens.', 'obj category': 'desk', 'obj_id': '7874330_SimpleDeskFactory', 'obj_size': [...]}, '3459997_SimpleDeskFactory': {'prompt for SD': 'A 60cm * 100cm * 100cm desk with a computer, a small plant, and personal items like notepads and pens.', 'obj category': 'desk', 'obj_id': '3459997_SimpleDeskFactory', 'obj_size': [...]}, '7222356_SimpleDeskFactory': {'prompt for SD': 'A 60cm * 100cm * 100cm desk with a computer, a small plant, and personal items like notepads and pens.', 'obj category': 'desk', 'obj_id': '7222356_SimpleDeskFactory', 'obj_size': [...]}, '2360448_SimpleDeskFactory': {'prompt for SD': 'A 60cm * 100cm * 100cm desk with a computer, a small plant, and personal items like notepads and pens.', 'obj category': 'desk', 'obj_id': '2360448_SimpleDeskFactory', 'obj_size': [...]}}
            for obj_id, info in steps.items():
                # update_infinigen("export_supporter",iter,json_name="",description=obj_id)
                # img_filename = gen_img_SD(info["prompt for SD"],obj_id,info["obj_size"])
                img_filename = "/home/yandan/workspace/infinigen/Pipeline/record/SD_img.jpg"
                json_name = acdc(img_filename,obj_id,info["obj category"])
                update_infinigen(action,iter,json_name,description=obj_id)
            
        elif action == "update":
            json_name = update_gpt(user_demand,ideas,iter,roomtype)
            update_infinigen(action,iter,json_name)
            
        elif action == "finish":
            update_infinigen(action,iter,json_name)
            break

        else:
            raise ValueError(f"Action is wrong: {action}") 

    # update_infinigen(action,iter,json_name)
    # solve_action = choose_solve()
    # update_infinigen("solve_large",iter)
    # update_infinigen("solve_large_and_medium",iter)
    # update_infinigen("solve_small",iter)


action = "finalize_scene"
update_infinigen(action,iter)