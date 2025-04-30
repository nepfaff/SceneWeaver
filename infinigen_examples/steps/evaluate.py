import json

import bpy
import trimesh

from infinigen.core.constraints.constraint_language import util as iu

# from infinigen.core import tags as t
from infinigen.core.constraints.evaluator.node_impl.trimesh_geometry import any_touching
import os
from infinigen.core.constraints.constraint_language.util import delete_obj

def eval_metric(state, iter,remove_bad=False):
    results,map_names = eval_physics_score(state,remove_bad=remove_bad)
    save_dir = os.getenv("save_dir")
    with open(f"{save_dir}/record_files/metric_{iter}.json", "w") as file:
        json.dump(results, file, indent=4)

    with open(f"{save_dir}/record_files/name_map_{iter}.json", "w") as file:
        json.dump(map_names, file, indent=4)
    return results


import bpy
import bmesh


def calc_intersection_volume(obj_a, obj_b):
    # 检查两个物体是否存在
    if obj_a is None or obj_b is None:
        print("Objects not found.")
        return 0.0

    # 复制 A 的数据
    temp_mesh = obj_a.data.copy()
    temp_obj = bpy.data.objects.new(name="TempIntersection", object_data=temp_mesh)
    bpy.context.collection.objects.link(temp_obj)
    
    # 选择 temp_obj
    bpy.context.view_layer.objects.active = temp_obj
    temp_obj.select_set(True)
    
    # 加布尔修改器
    boolean_mod = temp_obj.modifiers.new(name="Intersect", type='BOOLEAN')
    boolean_mod.operation = 'INTERSECT'
    boolean_mod.object = obj_b
    boolean_mod.solver = 'EXACT'  # 🔥 使用更稳定的 EXACT solver！！

    # 应用 Modifier
    bpy.ops.object.modifier_apply(modifier=boolean_mod.name)

    # 检查交集是否有面
    if len(temp_obj.data.polygons) == 0:
        print("No intersection: resulting mesh is empty.")
        bpy.data.objects.remove(temp_obj, do_unlink=True)
        return 0.0

    # 计算体积
    bm = bmesh.new()
    bm.from_mesh(temp_obj.data)
    volume = bm.calc_volume(signed=False)
    bm.free()

    # 删除临时物体
    bpy.data.objects.remove(temp_obj, do_unlink=True)

    return volume


def eval_physics_score(state,remove_bad=False):
    scene = state.trimesh_scene
    collision_objs = []
    map_names = dict()
    for name, info in state.objs.items():
        if name.startswith("window") or name == "newroom_0-0" or name == "entrance" or name.endswith("RugFactory"):
            continue
        else:
            name_obj = state.objs[name].populate_obj
            map_names[name_obj] = name
            collision_objs.append(name_obj)  # mesh
    
    Nobj = len(collision_objs)
    print("Nobj: ", Nobj)

    OOB_objs = []
    room_obj = state.objs["newroom_0-0"].obj
    normal_b = [0, 0, 1]
    origin_b = [0, 0, 0]
    b_trimesh = iu.meshes_from_names(scene, room_obj.name)[0]
    projected_b = trimesh.path.polygons.projected(b_trimesh, normal_b, origin_b)
    for name in collision_objs:
        # target_obj = bpy.data.objects.get(name)
        a_trimesh = iu.meshes_from_names(scene, name)[0]
        # try:
        #     projected_a = trimesh.path.polygons.projected(a_trimesh, normal_b, origin_b)
        # except:
        #     projected_a = trimesh.path.polygons.projected(a_trimesh.convex_hull, normal_b, origin_b)
        projected_a = trimesh.path.polygons.projected(
            a_trimesh.convex_hull, normal_b, origin_b
        )
        res = projected_a.within(projected_b.buffer(1e-2))
        if not res:
            OOB_objs.append(map_names[name])
    # collision_objs=["MetaCategoryFactory(2675461).spawn_asset(3827780)","MetaCategoryFactory(160109).spawn_asset(3161408)"]
    OOB = len(OOB_objs)
    print("OOB: ", OOB, OOB_objs)
    # state.trimesh_scene.show()
    
   

    
    collide_pairs = []
    # for name1 in collision_objs:
    #     for name2 in collision_objs: 
    #         scene = state.trimesh_scene
    #         mesh1 = scene.geometry[name1+"_mesh"]
    #         mesh2 = scene.geometry[name2+"_mesh"]
    #         intersection = mesh1.intersection(mesh2)
    #         # Check if result is valid and compute volume
    #         if intersection.is_volume:
    #             volume = intersection.volume
    #             a = 1
    collision_objs_norug = [i for i in collision_objs if "rug" not in map_names[i]]
    collide_volume = []
    for name in collision_objs_norug:
        touch = any_touching(
            scene, name, collision_objs_norug, bvh_cache=state.bvh_cache
        )
        if name =='ObjaverseCategoryFactory(1083614).spawn_asset(8785162)':
            a = 1
        if isinstance(touch.names[0], str):
            touch_names = [touch.names[0]]
        elif len(touch.names[0])==len(collision_objs_norug)-1:
            continue
        else:
            touch_names = touch.names[0]
        threshold = 0.001
        for contact in touch.contacts:
            if contact.depth > threshold:
                # import pdb
                # pdb.set_trace()
                name_col = list(contact.names)
                name_col.remove("__external")
                name_col = name_col[0]
                if name_col != name:
                    name1 = map_names[name_col]
                    name2 = map_names[name]
                    collide_pair = [max(name1,name2),min(name1,name2)] 
                    if collide_pair not in collide_pairs:
                        #check again
                        # Example usage:
                        obj1 = state.trimesh_scene.geometry[state.objs[collide_pair[0]].obj.name+"_mesh"]
                        obj2 = state.trimesh_scene.geometry[state.objs[collide_pair[1]].obj.name+"_mesh"]
                        vol = trimesh.boolean.boolean_manifold([obj1,obj2],"intersection").volume
                        # print(f"Intersection volume: {vol:.6f}")
                        if vol > 0.00001:
                            print(collide_pair,vol)
                            collide_volume.append(vol)
                            collide_pairs.append(collide_pair)
                        # scene = state.trimesh_scene
                        # # print(scene.geometry.keys())  # prints the names like ['Cube', 'Plane', 'Mesh_01', ...]

                        # # Pick two meshes by name
                        # mesh1 = scene.geometry[name_col+"_mesh"]
                        # mesh2 = scene.geometry[name+"_mesh"]

                        # # Combine them into a new scene
                        # combined_scene = trimesh.Scene()
                        # combined_scene.add_geometry(mesh1)
                        # combined_scene.add_geometry(mesh2)
                        # # Show the combined scene
                        # combined_scene.show()
        # for name_col in touch_names :
        #     if name_col != name:
        #         name1 = map_names[name_col]
        #         name2 = map_names[name]
        #         collide_pair = [max(name1,name2),min(name1,name2)] 
        #         if collide_pair not in collide_pairs:
        #             collide_pairs.append(collide_pair)
        #             scene = state.trimesh_scene
        #             # print(scene.geometry.keys())  # prints the names like ['Cube', 'Plane', 'Mesh_01', ...]

        #             # Pick two meshes by name
        #             mesh1 = scene.geometry[name_col+"_mesh"]
        #             mesh2 = scene.geometry[name+"_mesh"]

        #             # Combine them into a new scene
        #             combined_scene = trimesh.Scene()
        #             combined_scene.add_geometry(mesh1)
        #             combined_scene.add_geometry(mesh2)
        #             # Show the combined scene
        #             combined_scene.show()

    collide_names = collide_pairs

        # collide_pairs = [[map_names[name1], map_names[name2]] for name1, name2 in touch.names if name1 != name2]
    # collide_names = list(set([map_names[name1] for name1, name2 in touch.names if name1 != name2]))
    # collide_pairs = [[max(name1,name2),min(name1,name2)] for name1,name2 in touch.names if name1!=name2]
    # collide_pairs = set(collide_pairs)
    BBL = len(collide_names)
    print("BBL: ", BBL, collide_names)
    # import pdb
    # pdb.set_trace()

    results = {
        "Nobj":Nobj,
        "OOB":OOB,
        "OOB Objects": OOB_objs,
        "BBL":BBL,
        "BBL objects": collide_names,
        "collide volume":collide_volume
    }

    return results,map_names


def eval_general_score(image_path_1, layout, image_path_2=None):
    # real = 0
    # func = 0
    # complet = 0

    # return real, func, complet

    import argparse
    import base64
    import json
    import re

    import numpy as np
    import requests

    # TODO : OpenAI API Key
    api_key = "YOUR_API_KEY"

    # TODO : Path to your image
    image_path_1 = "FIRST_IMAGE_PATH.png"
    image_path_2 = "SECOND_IMAGE_PATH.png"

    # TODO : User preference Text
    user_preference = "USER_PREFERNCE_TEXT"

    # Function to encode the image
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    example_json = """
    {
    "realism_and_3d_geometric_consistency": {
        "grade": 8,
        "comment": "The renders appear to have appropriate 3D geometry and lighting that is fairly consistent with real-world expectations. The proportions and perspective look realistic."
    },
    "functionality_and_activity_based_alignment": {
        "grade": 7,
        "comment": "The room includes a workspace, sleeping area, and living area as per the user preference. The L-shaped couch facing the bed partially meets the requirement for watching TV comfortably. However, there does not appear to be a TV depicted in the render, so it's not entirely clear if the functionality for TV watching is fully supported."
    },
    "layout_and_furniture": {
        "grade": 7,
        "comment": "The room has a bed that’s not centered and with space at the foot, and a large desk with a chair. However, it's unclear if the height of the bed meets the user's preference, and the layout does not clearly show the full-length mirror in relation to the wardrobe, so its placement in accordance to user preferences is uncertain."
    },
    "completion_and_richness_of_detail": {
        "grade": 9,
        "comment": "The render includes detailed elements such as books on the desk, a rug under the coffee table, and small decorative items on the shelves. These touches add a sense of realism and completeness to the room, making it feel lived-in and thoughtfully designed."
    }
    """

    # Getting the base64 string
    base64_image_1 = encode_image(image_path_1)
    # base64_image_2 = encode_image(image_path_2)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""
            Give a grade from 1 to 10 or unknown to the following room renders and layout based on how well they correspond together to the user preference (in triple backquotes) in the following aspects: 
            - Realism and 3D Geometric Consistency
            - Functionality and Activity-based Alignment
            - Layout and furniture     
            - Completion and richness of detail  
            User Preference:
            ```{user_preference}```
            Room layout:
            ```{layout}```
            Return the results in the following JSON format:
            ```json
            {example_json}
            ```
            """,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image_1}"
                        },
                    },
                    # {
                    # "type": "image_url",
                    # "image_url": {
                    #     "url" : f"data:image/jpeg;base64,{base64_image_2}"
                    # }
                    # }
                ],
            }
        ],
        "max_tokens": 1024,
    }
    grades = {
        "realism_and_3d_geometric_consistency": [],
        "functionality_and_activity_based_alignment": [],
        "layout_and_furniture": [],
        # "color_scheme_and_material_choices": [],
        "completion_and_richness_of_detail": [],
    }
    for _ in range(3):
        response = requests.post(
            "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
        )
        grading_str = response.json()["choices"][0]["message"]["content"]
        print(grading_str)
        print("-" * 50)
        pattern = r"```json(.*?)```"
        matches = re.findall(pattern, grading_str, re.DOTALL)
        json_content = matches[0].strip() if matches else None
        if json_content is None:
            grading = json.loads(grading_str)
        else:
            grading = json.loads(json_content)
        for key in grades:
            grades[key].append(grading[key]["grade"])
    # Save the mean and std of the grades
    for key in grades:
        grades[key] = {
            "mean": round(sum(grades[key]) / len(grades[key]), 2),
            "std": round(np.std(grades[key]), 2),
        }
    # Save the grades
    with open(f"{'_'.join(image_path_1.split('_')[:-1])}_grades.json", "w") as f:
        json.dump(grades, f)

    return grades

def del_top_collide_obj(state,iter):
    results = eval_metric(state,iter)
    collide_pairs = results["BBL objects"]
    if len(collide_pairs)==0:
        return None
    vol = results["collide volume"]
    record = dict()
    for pair,v in zip(collide_pairs,vol):
        for objname in pair:
            if objname not in record:
                record[objname] = 0
            record[objname] += v
    # max_key = max(record, key=record.get)
    max_value = max(record.values())
    max_keys = [k for k, v in record.items() if v == max_value]
    obj_volumes = [state.trimesh_scene.geometry[state.objs[max_key].obj.name+"_mesh"].volume for max_key in max_keys]
    index = obj_volumes.index(min(obj_volumes))
    max_key = max_keys[index]
    if "nightstand" in max_key:
        AssertionError

    print(f"### Object {max_key} has biggest collision volume: {record[max_key]}, remove it !")

    objname = state.objs[max_key].obj.name
    delete_obj(state.trimesh_scene,objname,delete_blender=True, delete_asset=True)
    state.objs.pop(max_key)

    return max_key