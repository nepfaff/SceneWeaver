# import torch as th
import bpy
import numpy as np
from pathlib import Path
from PIL import Image
from copy import deepcopy
import os
import json
import imageio

import mathutils
import yaml
import sys
# Dynamically get the project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)
from digital_cousins.models.blend.blender_render import add_light
from digital_cousins.utils.processing_utils import NumpyTorchEncoder, unprocess_depth_linear, compute_point_cloud_from_depth, \
    get_reproject_offset, resize_image
from digital_cousins.utils.scene_utils import compute_relative_cam_pose_from, align_model_pose, compute_object_z_offset, \
    compute_obj_bbox_info, align_obj_with_wall, get_vis_cam_trajectory, set_bbox_center_position_orientation, set_position_orientation, \
    get_position_orientation, get_aabb_center, get_aabb
# from digital_cousins.utils.scene_utils import find_large_name
import digital_cousins.utils.transform_utils as T

from digital_cousins.models.blend.load_objaverse import load_pickled_3d_asset, load_openshape
from digital_cousins.models.blend.blender_render import load_obj_from_blend,merge_obj_from_blend
from digital_cousins.models.blend.blender_util import are_bbox_colliding, get_unbiased_verts_faces, are_mesh_colliding

# Set of non-collidable categories
NON_COLLIDABLE_CATEGORIES = {
    "towel",
    "rug",
    "mirror",
    "picture",
    "painting",
    "window",
    "art",
}

CATEGORIES_MUST_ON_FLOOR = {
    "rug",
    "carpet"
}

class SimulatedSceneGenerator:
    """
    3rd Step in ACDC pipeline. This takes in the output from Step 2 (Digital Cousin Matching) and generates
    fully populated digital cousin scenes

    Foundation models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - CLIP (https://github.com/openai/CLIP)
        - DINOv2 (https://github.com/facebookresearch/dinov2)

    Inputs:
        - Output from Step 2, which includes the following:
            - Per-object (category,, model, pose) digital cousin information

    Outputs:
        - Ordered digital cousin (category, model, pose) information per detected object from Step 1
    """
    SAMPLING_METHODS = {
        "random",
        "ordered",
    }

    def __init__(
            self,
            verbose=False,
    ):
        """
        Args:
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        self.verbose = verbose

    def __call__(
            self,
            step_1_output_path,
            step_2_output_path,
            n_scenes=1,
            dataset="holodeck",
            sampling_method="random",
            resolve_collision=False,
            discard_objs=None,
            save_dir=None,
            visualize_scene=False,
            visualize_scene_tilt_angle=0,
            visualize_scene_radius=5,
            save_visualization=True,
            tabletype="",
    ):
        """
        Runs the simulated scene generator. This does the following steps for all detected objects from Step and all
        matched cousin assets from Step 2:

        1. Compute camera pose and world origin point from step 1 output.
        2. Separately set each object in correct position and orientation w.r.t. the viewer camera,
           and save the relative transformation between the object and the camera.
        3. Put all objects in a single scene.
        4. Infer objects OnTop relationship. We currently only support OnTop cross-object relationship, so there might
            be artifacts if an object is 'In' another object, like books in a bookshelf.
        5. Process collisions and put objects onto the floor or objects beneath to generate a physically plausible scene.
        6. (Optionally) visualize the reconstructed scene.

        Args:
            step_1_output_path (str): Absolute path to the output file generated from Step 1 (RealWorldExtractor)
            step_2_output_path (str): Absolute path to the output file generated from Step 2 (DigitalCousinMatcher)
            n_scenes (int): Number of scenes to generate. This number cannot be greater than the number of cousins
                generated from Step 2 if @sampling_method="ordered" or greater than the product of all possible cousin
                combinations if @sampling_method="random"
            sampling_method (str): Sampling method to use when generating scenes. "random" will randomly select a cousin
                for each detected object in Step 1 (total combinations: N_cousins ^ N_objects). "ordered" will
                sequentially iterate over each detected object and generate scenes with corresponding ordered cousins,
                i.e.: a scene with all 1st cousins, a scene with all 2nd cousins, etc. (total combinations: N_cousins).
                Note that in both cases, the first scene generated will always be composed of all the closest (first)
                cousins. Default is "random"
            resolve_collision (bool): Whether to depenetrate collisions. When the point cloud is not denoised properly,
                or the mounting type is wrong, the object can be unreasonably large. Or when two objects in the input image
                intersect with each other, we may move an object by a non-trivial distance to depenetrate collision, so
                objects on top may fall down to the floor, and other objects may also need to be moved to avoid collision
                with this object. Under both cases, we recommend setting @resolve_collision to False to visualize the
                raw output.
            discard_objs (str): Names of objects to discard during reconstruction, seperated by comma, i.e., obj_1,obj_2,obj_3.
                Do not add space between object names.
            save_dir (None or str): If specified, the absolute path to the directory to save all generated outputs. If
                not specified, will generate a new directory in the same directory as @step_2_output_path
            visualize_scene (bool): Whether to visualize the scene after reconstruction. If True, the viewer camera will
                rotate around the scene's center point with a @visualize_scene_tilt_angle tilt cangle, and a 
                @visualize_scene_radius radius.
            visualize_scene_tilt_angle (float): The camera tilt angle in degree when visualizing the reconstructed scene. 
                This parameter is only used when @visualize_scene is set to True
            visualize_scene_radius (float): The camera rotating raiud in meters when visualizing the reconstructed scene.
                This parameter is only used when @visualize_scene is set to True
            save_visualization (bool): Whether to save the visualization results. This parameter is only used when 
                @visualize_scene is set to True

        Returns:
            2-tuple:
                bool: True if the process completed successfully. If successful, this will write all relevant outputs to
                    the directory specified in the second output
                None or str: If successful, this will be the absolute path to the main output file. Otherwise, None
        """
        # Load step 2 info
        print(step_2_output_path)
        with open(step_2_output_path, "r") as f:
            step_2_output_info = json.load(f)

        # Load relevant information from prior steps
        n_cousins = step_2_output_info["metadata"]["n_cousins"]
        n_objects = step_2_output_info["metadata"]["n_objects"]
        cousins = step_2_output_info["objects"]
        # Sanity check number of scenes to generate
        assert sampling_method in self.SAMPLING_METHODS, \
            f"Got invalid sampling_method! Valid methods: {self.SAMPLING_METHODS}, got: {sampling_method}"\

        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(step_2_output_path)
        save_dir = os.path.join(save_dir, "step_3_output")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if discard_objs:
            discard_objs = set(discard_objs.split(","))

        if self.verbose:
            print(f"Generating simulated scenes given output {step_2_output_path}...")

        if self.verbose:
            print("""

####################################################
####  Generating simulated scenes in OmniGibson ####
####################################################

            """)

        # Load relevant input information
        with open(step_1_output_path, "r") as f:
            step_1_output_info = json.load(f)

        with open(step_1_output_info["detected_categories"], "r") as f:
            detected_categories = json.load(f)

        seg_dir = detected_categories["segmentation_dir"]
        K = np.array(step_1_output_info["K"])
        rgb = np.array(Image.open(step_1_output_info["input_rgb"]))
        raw_depth = np.array(Image.open(step_1_output_info["input_depth"]))
        depth_limits = np.array(step_1_output_info["depth_limits"])
        depth = unprocess_depth_linear(depth=raw_depth, out_limits=depth_limits)
        pc = compute_point_cloud_from_depth(depth=depth, K=K)

        z_dir = np.array(step_1_output_info["z_direction"])
        # wall_mask_planes = step_1_output_info["wall_mask_planes"]
        origin_pos = np.array(step_1_output_info["origin_pos"])
        cam_pos, cam_quat = compute_relative_cam_pose_from(z_dir=z_dir, origin_pos=origin_pos)

        # Launch omnigibson
        # og.launch()

        # Loop over all sample indices to generate individual scenes
        for scene_count in range(n_scenes):

            print("#" * 30)
            print(f"[Scene {scene_count + 1} / {n_scenes}]")

            # Make dir for saving this scene
            scene_save_dir = f"{save_dir}/scene_{scene_count}"
            Path(scene_save_dir).mkdir(parents=True, exist_ok=True)

            # Parse the index to know what configuration of cousins to use
            if sampling_method == "random":
                # Ordering is inferred iteratively
                cousin_idxs = dict()
                for i, obj_name in enumerate(cousins.keys()):
                    cousin_idxs[obj_name] = np.random.randint(0, n_cousins)
            elif sampling_method == "ordered":
                # Cousin selection is simply the current scene idx
                cousin_idxs = {obj_name: scene_count for obj_name in cousins.keys()}
            else:
                raise ValueError(f"sampling_method {sampling_method} not supported!")

            # render settings
            
            h, w, _ = rgb.shape
            scene = bpy.context.scene
            for obj in scene.objects:
                if "Point" in obj.name or "Area" in obj.name  or "Camera" in obj.name:
                    continue
                bpy.data.objects.remove(obj) 

            scene.render.resolution_x = w  # Width
            scene.render.resolution_y = h  # Height
            scene.render.resolution_percentage = 100  # Use full resolution
            if not bpy.context.scene.camera:
                bpy.ops.object.camera_add()
                camera = bpy.context.object
                bpy.context.scene.camera = camera
            else:
                camera = bpy.context.scene.camera
            
            camera.location = cam_pos
            camera.rotation_mode = 'QUATERNION'
            blender_quat = [cam_quat[3], cam_quat[0], cam_quat[1], cam_quat[2]]
            camera.rotation_quaternion = mathutils.Quaternion(blender_quat)
            camera.data.lens = 17
            

            # Loop over all cousins and load them
            #加载资产，计算并对齐姿态，与点云对齐，保存相关的场景信息和可视化。
            for obj_idx, (obj_name, obj_cousin_idx) in enumerate(cousin_idxs.items()):
                if self.verbose:
                    print("-----------------")
                    print(f"[Scene {scene_count + 1} / {n_scenes}] [Object {obj_idx + 1} / {n_objects}] generating...")
                
                #如果某些对象被列入 discard_objs 排除列表，则跳过这些对象。
                if discard_objs and obj_name in discard_objs:
                    continue
                # Load and prune object mask
                obj_info = step_2_output_info["objects"][obj_name]
                is_articulated = obj_info["articulated"]
                obj_mask = np.array(Image.open(f"{seg_dir}/{obj_name}_nonprojected_mask_pruned.png"))
                #使用掩码提取对象的点云数据。
                pc_obj = pc.reshape(-1, 3)[np.array(obj_mask).flatten().nonzero()[0]]

                # Infer cousin category and model
                # Assumes path is XXX/.../<CATEGORY>/model/<MODEL>/<MODEL>_<ANGLE>
                print(obj_name,obj_cousin_idx)
                cousin_info = obj_info["cousins"][obj_cousin_idx]

                # if False:
                if cousins[obj_name]["mount"]["floor"]:
                    candidate = os.path.join(_PROJECT_ROOT, "tests", "obj.blend")
                    if not os.path.exists(candidate):
                        candidate = "tests/obj.blend"
                    obj = merge_obj_from_blend(candidate)
                    obj.name = obj_name
                elif dataset=="holodeck":
                    from digital_cousins.models.objaverse.constants import OBJATHOR_ASSETS_DIR
                    basedir = OBJATHOR_ASSETS_DIR
                    candidate = cousins[obj_name]['cousins'][obj_cousin_idx]['model']
                    filename = f'{basedir}/{candidate}/{candidate}.pkl.gz'
                    obj = load_pickled_3d_asset(filename)
                else:
                    step_2_dir = save_dir.replace("step_3_output","step_2_output")
                    with open(f"{step_2_dir}/objav_files.json", "r") as f:
                        candidates_objav = json.load(f)
                    modelname = cousins[obj_name]['cousins'][obj_cousin_idx]['model']
                    candidates = candidates_objav["_".join(obj_name.split("_")[:-1])]
                    filename = [i for i in candidates if modelname in i]
                    obj = load_openshape(filename[0])
                
                # og.sim.step()

                # Determine the reprojection offset based on the object's pose
                ## 根据对象姿态计算重投影偏移
                bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                # import pdb
                # pdb.set_trace()
                pan_angle_offset, _ = get_reproject_offset(
                    pc_obj=deepcopy(pc_obj),
                    z_dir=z_dir,
                    xy_dist=2.30,   # from OG dataset generation process
                    z_dist=0.65    # from OG dataset generation process
                )
               
             
                # Align the object model to its corresponding point cloud
                ## 将对象模型与点云对齐,返回对象的缩放比例（obj_scale）、包围盒大小（obj_bbox_extent）和相机到对象的变换矩阵（tf_from_cam）。
                obj_scale, obj_bbox_extent, tf_from_cam = align_model_pose(
                    obj=obj,
                    pc_obj=pc_obj,
                    obj_z_angle=float(cousin_info["z_angle"]) + pan_angle_offset,
                    obj_ori_offset=cousin_info["ori_offset"],
                    z_dir=deepcopy(z_dir),
                    cam_pos=cam_pos,
                    cam_quat=cam_quat,
                    is_articulated=is_articulated,
                    verbose=self.verbose,
                )
                
                obj_pos, obj_quat = get_position_orientation(obj)
                

                # Save information and current visualization
                ## 保存场景信息
                obj_save_dir = f"{scene_save_dir}/{obj_name}"
                Path(obj_save_dir).mkdir(parents=True, exist_ok=True)
                obj_info = cousins[obj_name]['cousins'][obj_cousin_idx]
                obj_scene_info = {
                    "category": obj_info['category'],
                    "model": obj_info['model'],
                    "scale": obj_scale,
                    "bbox_extent": obj_bbox_extent,
                    "tf_from_cam": tf_from_cam,
                    "mount": detected_categories["mount"][obj_idx],
                }
                with open(f"{obj_save_dir}/{obj_name}_scene_info.json", "w+") as f:
                    json.dump(obj_scene_info, f, indent=4, cls=NumpyTorchEncoder)

                # Take photo # 拍摄照片
                obj_scene_rgb = SimulatedSceneGenerator.take_photo()
                H, W, _ = obj_scene_rgb.shape

                # Append to non-projcted image, then save
                # 加载非投影的图像并拼接保存
                nonprojected_rgb = resize_image(np.array(Image.open(f"{seg_dir}/{obj_name}_nonprojected.png")), height=H)
                # import pdb
                # pdb.set_trace()
                if nonprojected_rgb.shape[-1]==4:
                    nonprojected_rgb = nonprojected_rgb[:,:,:-1]
                obj_rgb = np.concatenate([nonprojected_rgb, obj_scene_rgb], axis=1)
                Image.fromarray(obj_rgb).save(f"{obj_save_dir}/{obj_name}_scene_visualization.png")

                # Remove the object from the scene
                bpy.data.objects.remove(obj) 
                

            # Store final scene information
            scene_graph_info_path = f"{scene_save_dir}/scene_{scene_count}_graph.json"
            scene_info = dict()
            scene_info["resolution"] = [H, W]
            scene_info["scene_graph"] = scene_graph_info_path
            scene_info["cam_pose"] = [cam_pos, cam_quat]
            scene_info["objects"] = dict()
            
            for obj_name in cousin_idxs.keys():
                if discard_objs and obj_name in discard_objs:
                    continue
                with open(f"{scene_save_dir}/{obj_name}/{obj_name}_scene_info.json", "r") as f:
                    scene_obj_info = json.load(f)
                scene_info["objects"][obj_name] = scene_obj_info

            # Load the entire scene
            scene = SimulatedSceneGenerator.load_cousin_scene(scene_info=scene_info, dataset=dataset,save_dir=save_dir,visual_only=True)
            
            if self.verbose:
                print(f"[Scene {scene_count + 1} / {n_scenes}] refining scene graph...")

            # Infer scene graph based on relative object poses
            all_obj_bbox_info = dict()
            for obj_name, obj_info in scene_info["objects"].items():
                if discard_objs and obj_name in discard_objs:
                    continue
                # Grab object and relevant info
                # obj = scene.object_registry("name", obj_name)
                obj = bpy.data.objects.get(obj_name)
                obj_bbox_info = compute_obj_bbox_info(obj=obj)
                obj_bbox_info["articulated"] = step_2_output_info["objects"][obj_name]["articulated"]
                obj_bbox_info["mount"] = obj_info["mount"]
                all_obj_bbox_info[obj_name] = obj_bbox_info
            sorted_z_obj_bbox_info = dict(sorted(all_obj_bbox_info.items(), key=lambda x: x[1]['lower'][2]))  # sort by lower corner's height (z)

            scene_graph_info = {
                "floor": {
                    "objOnTop": [],
                    "objBeneath": None,  # This must be empty, i.e., no obj is beneath floor
                    "mount": {
                        "floor": True,
                        "wall": False,
                    },
                },
            }
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            # import pdb
            # pdb.set_trace()
            # 存储到最终场景信息中。
            # large_name = find_large_name(step_2_output_info)
            # assert large_name!=None
            final_scene_info = deepcopy(scene_info)

            index = detected_categories["phrases"].index(tabletype)
            table_objname = detected_categories["names"][index]
            final_scene_info["supporter"] = table_objname

            
            # Find the maximum Z
            obj_supporter = bpy.data.objects.get(table_objname)
            bbox_global = [obj_supporter.matrix_world @ mathutils.Vector(corner) for corner in obj_supporter.bound_box]
            supporter_top_z = max(corner.z for corner in bbox_global)
            # supporter_bottom_z = min(corner.z for corner in bbox_global)
            # supporter_top_z = obj_supporter.location[2]+obj_supporter.dimensions[2]/2
            objs_under = []
            for name in sorted_z_obj_bbox_info:
                if name == table_objname:
                    continue
                # import pdb
                # pdb.set_trace()
                obj = bpy.data.objects.get(name)

                #remove object below table
                bbox_global = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
                top_z = max(corner.z for corner in bbox_global)
                bottom_z = min(corner.z for corner in bbox_global)
                # top_z = obj.location[2]+obj.dimensions[2]/2
                # size_z = obj.dimensions[2]

                if top_z-supporter_top_z<-0.05 or (bottom_z - supporter_top_z)<-0.2:  #objet is not on top
                    objs_under.append(name)

            for name in objs_under:
                sorted_z_obj_bbox_info.pop(name)
                obj = bpy.data.objects.get(name)
                bpy.data.objects.remove(obj)
                final_scene_info["objects"].pop(name)


            for name in sorted_z_obj_bbox_info:
                obj_name_beneath, z_offset = compute_object_z_offset(
                    target_obj_name=name,
                    sorted_obj_bbox_info=sorted_z_obj_bbox_info,
                    verbose=self.verbose,
                )
                
                if name!="floor" and name!=table_objname:
                    obj_name_beneath = table_objname

                obj = bpy.data.objects.get(name)

                if scene_info["objects"][name]["category"] in CATEGORIES_MUST_ON_FLOOR:
                    obj_name_beneath = "floor"
                    z_offset = -sorted_z_obj_bbox_info[name]["lower"][-1]

                # Add information to scene graph info
                if name not in scene_graph_info.keys():
                    scene_graph_info[name] = {
                        "objOnTop": [],
                        "objBeneath": obj_name_beneath,
                        "mount": None,
                    }
                else:
                    scene_graph_info[name]["objBeneath"] = obj_name_beneath

                if obj_name_beneath not in scene_graph_info.keys():
                    scene_graph_info[obj_name_beneath] = {
                        "objOnTop": [name],
                        "objBeneath": None,
                        "mount": None,
                    }
                else:
                    scene_graph_info[obj_name_beneath]["objOnTop"].append(name)

                mount_type = scene_info["objects"][name]["mount"]  # a list
                scene_graph_info[name]["mount"] = mount_type
                # obj.keep_still()
                # Modify object pose if z_offset is not 0
                if z_offset != 0:
                    # if (not mount_type["floor"]) and z_offset <= 0:
                    #     # If the object in mounted on the wall, and we want to lower it, omit that
                    #     continue
                    new_center = sorted_z_obj_bbox_info[name]["center"] + np.array([0.0, 0.0, z_offset])
                    set_bbox_center_position_orientation(obj,new_center, quat=None)
                    # og.sim.step_physics()

                    # Grab updated obj bbox info
                    obj_bbox_info = compute_obj_bbox_info(obj=obj)
                    sorted_z_obj_bbox_info[name].update(obj_bbox_info)

                # Update scene_info
                obj_pos, obj_quat = get_position_orientation(obj)
                rel_tf = T.relative_pose_transform(np.array(obj_pos), np.array(obj_quat), cam_pos, cam_quat)
                final_scene_info["objects"][name]["tf_from_cam"] = T.pose2mat(rel_tf)
                obj.rotation_mode = 'XYZ'
                final_scene_info["objects"][name]["location"] = np.array(obj_pos)
                final_scene_info["objects"][name]["rotation"] = np.array(obj.rotation_euler)
                final_scene_info["objects"][name]["size"] = np.array(obj.dimensions)

            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            # import pdb
            # pdb.set_trace()

            with open(scene_graph_info_path, "w+") as f:
                json.dump(scene_graph_info, f, indent=4, cls=NumpyTorchEncoder)

            # for _ in range(3):
            #     og.sim.render()
            # Put objects down
            if self.verbose:
                print(f"[Scene {scene_count + 1} / {n_scenes}] placing all objects down...")

            bpy.context.view_layer.update() 

            # Process collisions
            sorted_x_obj_bbox_info = dict(sorted(sorted_z_obj_bbox_info.items(), key=lambda x: x[1]['lower'][0], reverse=True))  # sort by lower corner's x
            obj_names = list(sorted_x_obj_bbox_info.keys())
            
            final_scene_info = self.move_ontop(obj_names,scene_graph_info, cam_pos, cam_quat,final_scene_info)

            if resolve_collision:
                if self.verbose:
                    print(f"[Scene {scene_count + 1} / {n_scenes}] depenetrating collisions...")
                self.resolve_collision(obj_names,scene_graph_info, cam_pos, cam_quat,final_scene_info)    
            else:
                if self.verbose:
                    print(f"[Scene {scene_count + 1} / {n_scenes}] skip depenetrating collisions.")
               
            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

            # Take final physics step, then save visualization + info
            bpy.context.view_layer.update() 
            scene_rgb = SimulatedSceneGenerator.take_photo()
            H, W, _ = scene_rgb.shape
            resized_rgb = resize_image(rgb, height=H)
            if resized_rgb.shape[-1]==4:
                resized_rgb = resized_rgb[:,:,:-1]
            concat_scene_rgb = np.concatenate([resized_rgb, scene_rgb], axis=1)
            Image.fromarray(concat_scene_rgb).save(f"{scene_save_dir}/scene_{scene_count}_visualization.png")

            # Save final info
            with open(f"{scene_save_dir}/scene_{scene_count}_info.json", "w+") as f:
                json.dump(final_scene_info, f, indent=4, cls=NumpyTorchEncoder)

            # if visualize_scene:
            #     SimulatedSceneGenerator.render_video(scene_save_dir,visualize_scene_tilt_angle,visualize_scene_radius,save_visualization)
            
            # Save the current scene
            blender_save_path = f"{scene_save_dir}/scene_{scene_count}.blend"
            bpy.ops.wm.save_as_mainfile(filepath=blender_save_path)
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
        # import pdb
        # pdb.set_trace()
        # # Compile final results across all scenes
        step_3_output_info = dict()
        for scene_count in range(n_scenes):
            scene_name = f"scene_{scene_count}"
            final_scene_info_path = f"{save_dir}/{scene_name}/{scene_name}_info.json"
            with open(final_scene_info_path, "r") as f:
                final_scene_info = json.load(f)
            step_3_output_info[scene_name] = final_scene_info

        step_3_output_path = f"{save_dir}/step_3_output_info.json"
        with open(step_3_output_path, "w+") as f:
            json.dump(step_3_output_info, f, indent=4, cls=NumpyTorchEncoder)


        with open("args.json","r") as f:
            j = json.load(f)
            obj_id = j["obj_id"]

        step_3_output_path_save = "/".join(save_dir.split("/")[:-2])+"/step_3_output_"+obj_id
        os.system(f"cp -r {save_dir} {step_3_output_path_save}")

        print("""

#############################################
### Completed Simulated Scene Generation! ###
#############################################

        """)

        return True, step_3_output_path
    
    def move_ontop(self,obj_names,scene_graph_info, cam_pos, cam_quat,final_scene_info):
        #遍历场景中的所有物体
        # import pdb
        # pdb.set_trace()
        #这段代码的目标是将一个物体 obj1 放置在支撑物 obj_beneath 上方，直到检测到两者碰撞为止，
        # 然后记录其最终位姿信息，特别是相对于相机的位姿变换。
        for obj1_idx, obj1_name in enumerate(obj_names):
            print(obj1_name)
            # import pdb
            # pdb.set_trace()

            # Skip any non-collidable categories 跳过不可碰撞的物体类别：
            if any(cat in obj1_name for cat in NON_COLLIDABLE_CATEGORIES):
                continue

            # If object is on floor, or mounted on a wall, don't move 检查物体的位置和安装状态：
            # 如果物体位于地板上或者安装在墙上，或者没有任何安装信息，则跳过这些物体。
            if scene_graph_info[obj1_name]['objBeneath'] == "floor": # or all_obj_bbox_info[obj1_name]["mount"]==None or not all_obj_bbox_info[obj1_name]["mount"]["floor"]:
                continue
            # if obj1_name=="desk":
            #     continue

            # Infer object that is beneath obj1
            # 推断物体下方的支撑物：
            obj_beneath_name = scene_graph_info[obj1_name]["objBeneath"]
            obj_beneath = bpy.data.objects.get(obj_beneath_name)

            # Skip any non-collidable categories, and objects without top
            #跳过没有顶部或不可碰撞的支撑物： 如果支撑物类别中包含 no_top 或支撑物是不可碰撞的类别，则跳过。
            if any(cat in obj_beneath_name for cat in NON_COLLIDABLE_CATEGORIES):
                continue

            # 准备物体状态：
            # 将当前物体 obj1 和支撑物体 obj_beneath 设置为不可移动（keep_still）且可碰撞（visual_only = False）。
            obj1 = bpy.data.objects.get(obj1_name)

            obj1_lower_corner, _ = get_aabb(obj1)
            obj1_low_z = obj1_lower_corner[-1]
            obj_beneath_lower_corner, obj_beneath_higher_corner = get_aabb(obj_beneath)
            obj_beneath_low_z = obj_beneath_lower_corner[-1]
            obj_beneath_high_z = obj_beneath_higher_corner[-1]
            center_step_size = 0.005
            bpy.context.view_layer.update() 
            
            # 计算碰撞：
            # import pdb
            # pdb.set_trace()
            if obj1_low_z <= obj_beneath_high_z - center_step_size:
            # if are_bbox_colliding(obj1,obj_beneath) or are_mesh_colliding(collision_info,obj1,obj_beneath):
                while obj1_low_z <= obj_beneath_high_z - center_step_size: # or are_bbox_colliding(obj1,obj_beneath) or are_mesh_colliding(collision_info,obj1,obj_beneath):
                    new_center = get_position_orientation(obj1)[0] + mathutils.Vector([0, 0, 1.0])  * center_step_size
                    obj1_low_z += center_step_size
                    set_position_orientation(obj1,new_center)
                    bpy.context.view_layer.update() 
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

                # 如果检测到碰撞,更新物体的最终位置并计算相对于相机的位姿变换。
                # final_position = get_position_orientation(obj1)[0] - mathutils.Vector([0, 0, 1.0]) * center_step_size
                # set_position_orientation(obj1,final_position)
                obj_pos, obj_quat = get_position_orientation(obj1)
                rel_tf = T.relative_pose_transform(np.array(obj_pos), np.array(obj_quat), cam_pos, cam_quat)
                #记录物体的最终信息：
                final_scene_info["objects"][obj1_name]["tf_from_cam"] = T.pose2mat(rel_tf)

            # 计算碰撞：
            # import pdb
            # pdb.set_trace()
            if obj1_low_z >= obj_beneath_high_z + center_step_size:
            # if are_bbox_colliding(obj1,obj_beneath) or are_mesh_colliding(collision_info,obj1,obj_beneath):
                while obj1_low_z >= obj_beneath_high_z + center_step_size: # or are_bbox_colliding(obj1,obj_beneath) or are_mesh_colliding(collision_info,obj1,obj_beneath):
                    new_center = get_position_orientation(obj1)[0] - mathutils.Vector([0, 0, 1.0])  * center_step_size
                    obj1_low_z -= center_step_size
                    set_position_orientation(obj1,new_center)
                    bpy.context.view_layer.update() 
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)

                # 如果检测到碰撞,更新物体的最终位置并计算相对于相机的位姿变换。
                # final_position = get_position_orientation(obj1)[0] - mathutils.Vector([0, 0, 1.0]) * center_step_size
                # set_position_orientation(obj1,final_position)
                obj_pos, obj_quat = get_position_orientation(obj1)
                rel_tf = T.relative_pose_transform(np.array(obj_pos), np.array(obj_quat), cam_pos, cam_quat)
                #记录物体的最终信息：
                final_scene_info["objects"][obj1_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                final_scene_info["objects"][obj1_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                final_scene_info["objects"][obj1_name]["location"] = np.array(obj_pos)
                obj1.rotation_mode = 'XYZ'
                final_scene_info["objects"][obj1_name]["rotation"] = np.array(obj1.rotation_euler)
            # else:
                # og.sim.load_state(old_state)
        return final_scene_info

    def resolve_collision_complex(self,obj_names,scene_graph_info, cam_pos, cam_quat,final_scene_info):
        

        # record collision mesh
        collision_info = dict()
        for obj1_idx, obj1_name in enumerate(obj_names):
            # Skip any non-collidable categories
            if any(cat in obj1_name for cat in NON_COLLIDABLE_CATEGORIES):
                continue
            points,faces = get_unbiased_verts_faces(obj1_name)
            collision_info[obj1_name] = {"faces":faces,"points":points}

            
        # Iterate over all objects; check for collision
        for obj2_name in obj_names:
            # Skip any non-collidable categories
            if any(cat in obj2_name for cat in NON_COLLIDABLE_CATEGORIES):
                continue
            # Grab the object, make it collidable
            obj2 = bpy.data.objects.get(obj2_name)

            detectCollision = True
            while(detectCollision):
                detectCollision = False

                for obj1_idx, obj1_name in enumerate(obj_names):
                    # Skip any non-collidable categories
                    if any(cat in obj1_name for cat in NON_COLLIDABLE_CATEGORIES):
                        continue

                    if obj1_name==obj2_name:
                        continue
                
                    # If the objects are related by a vertical relationship, continue -- collision is expected
                    if (obj2_name in scene_graph_info[obj1_name]['objOnTop']) or (
                            scene_graph_info[obj1_name]["objBeneath"] == obj2_name):
                        continue
                    # Grab the object, make it collidable
                    obj1 = bpy.data.objects.get(obj1_name)


                    # If we're in contact, move the object with smaller x value
                    if are_bbox_colliding(obj1,obj2) and are_mesh_colliding(collision_info,obj1,obj2):
                        detectCollision = True
                        # Adjust the object with smaller x
                        if self.verbose:
                            print(f"Detected collision between {obj1_name} and {obj2_name}")
                        # Get obj 2's x and y axes
                        obj2_ori_mat = T.quat2mat(np.array(get_position_orientation(obj2)[1]))
                        obj2_x_dir = obj2_ori_mat[:, 0]
                        obj2_y_dir = obj2_ori_mat[:, 1]

                        center_step_size = 0.001  # 1cm
                        obj2_to_obj1 = np.array(get_position_orientation(obj1)[0] - get_position_orientation(obj2)[0])

                        # chosen_axis = obj2_x_dir if abs(np.dot(obj2_x_dir, obj2_to_obj1)) > abs(np.dot(obj2_y_dir, obj2_to_obj1)) else obj2_y_dir
                        # center_step_dir = -chosen_axis if np.dot(chosen_axis, obj2_to_obj1) > 0 else chosen_axis
                        import math
                        magnitude = math.sqrt(obj2_to_obj1[0]**2 + obj2_to_obj1[1]**2)
                        center_step_dir = [-obj2_to_obj1[0] / magnitude, -obj2_to_obj1[1] / magnitude, 0]

                        while are_bbox_colliding(obj1,obj2) and are_mesh_colliding(collision_info,obj1,obj2):
                            new_center = get_position_orientation(obj2)[0] +  mathutils.Vector(center_step_dir) * center_step_size
                            set_position_orientation(obj2,new_center)
                            print(obj1_name,obj2_name,new_center)
                            bpy.context.view_layer.update() 
                            bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                            
                        # Finally, load the collision-free state, update relative transformation
                        set_position_orientation(obj2,new_center)
                        obj_pos, obj_quat = get_position_orientation(obj2)
                        rel_tf = T.relative_pose_transform(np.array(obj_pos), np.array(obj_quat), cam_pos, cam_quat)
                        final_scene_info["objects"][obj2_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                        bpy.context.view_layer.update() 
        return final_scene_info

    def resolve_collision(self,obj_names,scene_graph_info, cam_pos, cam_quat,final_scene_info):
        

        # record collision mesh
        collision_info = dict()
        for obj1_idx, obj1_name in enumerate(obj_names):
            # Skip any non-collidable categories
            if any(cat in obj1_name for cat in NON_COLLIDABLE_CATEGORIES):
                continue
            points,faces = get_unbiased_verts_faces(obj1_name)
            collision_info[obj1_name] = {"faces":faces,"points":points}

            
        # Iterate over all objects; check for collision
        for obj1_idx, obj1_name in enumerate(obj_names):

            # Skip any non-collidable categories
            if any(cat in obj1_name for cat in NON_COLLIDABLE_CATEGORIES):
                continue

            # Grab the object, make it collidable
            obj1 = bpy.data.objects.get(obj1_name)

            # Check all subsequent downstream objects for collision
            for obj2_name in obj_names[obj1_idx + 1:]:

                # Skip any non-collidable categories
                if any(cat in obj2_name for cat in NON_COLLIDABLE_CATEGORIES):
                    continue

                # Sanity check to make sure the two objects aren't the same
                assert obj1_name != obj2_name

                # If the objects are related by a vertical relationship, continue -- collision is expected
                if (obj2_name in scene_graph_info[obj1_name]['objOnTop']) or (
                        scene_graph_info[obj1_name]["objBeneath"] == obj2_name):
                    continue

                # Grab the object, make it collidable
                obj2 = bpy.data.objects.get(obj2_name)

                # If we're in contact, move the object with smaller x value
                if are_bbox_colliding(obj1,obj2) and are_mesh_colliding(collision_info,obj1,obj2):
                    # Adjust the object with smaller x
                    if self.verbose:
                        print(f"Detected collision between {obj1_name} and {obj2_name}")
                    # Get obj 2's x and y axes
                    obj2_ori_mat = T.quat2mat(np.array(get_position_orientation(obj2)[1]))
                    obj2_x_dir = obj2_ori_mat[:, 0]
                    obj2_y_dir = obj2_ori_mat[:, 1]

                    center_step_size = 0.001  # 1cm
                    obj2_to_obj1 = np.array(get_position_orientation(obj1)[0] - get_position_orientation(obj2)[0])

                    # chosen_axis = obj2_x_dir if abs(np.dot(obj2_x_dir, obj2_to_obj1)) > abs(np.dot(obj2_y_dir, obj2_to_obj1)) else obj2_y_dir
                    # center_step_dir = -chosen_axis if np.dot(chosen_axis, obj2_to_obj1) > 0 else chosen_axis
                    import math
                    magnitude = math.sqrt(obj2_to_obj1[0]**2 + obj2_to_obj1[1]**2)
                    center_step_dir = [-obj2_to_obj1[0] / magnitude, -obj2_to_obj1[1] / magnitude, 0]

                    while are_bbox_colliding(obj1,obj2) and are_mesh_colliding(collision_info,obj1,obj2):
                        new_center = get_position_orientation(obj2)[0] +  mathutils.Vector(center_step_dir) * center_step_size
                        set_position_orientation(obj2,new_center)
                        print(obj1_name,obj2_name,new_center)
                        bpy.context.view_layer.update() 
                        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
                        
                    # Finally, load the collision-free state, update relative transformation
                    set_position_orientation(obj2,new_center)
                    obj_pos, obj_quat = get_position_orientation(obj2)
                    rel_tf = T.relative_pose_transform(np.array(obj_pos), np.array(obj_quat), cam_pos, cam_quat)
                    final_scene_info["objects"][obj2_name]["tf_from_cam"] = T.pose2mat(rel_tf)
                    final_scene_info["objects"][obj2_name]["location"] = np.array(obj_pos)
                    obj2.rotation_mode = 'XYZ'
                    final_scene_info["objects"][obj2_name]["rotation"] = np.array(obj2.rotation_euler)
    
                    bpy.context.view_layer.update() 
        return final_scene_info

    @staticmethod
    def load_cousin_scene(scene_info, dataset="openshape",save_dir="",visual_only=False):
        """
        Loads the cousin scene specified by info at @scene_info_fpath

        Args:
            scene_info (dict or str): If dict, scene information to load. Otherwise, should be absolute path to the
                scene info that should be loaded
            visual_only (bool): Whether to load all objects as visual only or not

        Returns:
            Scene: loaded OmniGibson scene
        """
        # Stop sim, clear it, then load empty scene

        # Load scene information if it's a path
        if isinstance(scene_info, str):
            with open(scene_info, "r") as f:
                scene_info = json.load(scene_info)

        scene = bpy.context.scene
        for obj in scene.objects:
            bpy.data.objects.remove(obj) 

        #camera # Set viewer camera to proper pose
        if not bpy.context.scene.camera:
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            bpy.context.scene.camera = camera
        else:
            camera = bpy.context.scene.camera
        # camera.data.lens = 17
        cam_pose = scene_info["cam_pose"]
        set_position_orientation(camera,cam_pose[0],cam_pose[1])
            
        # import pdb
        # pdb.set_trace()
        for obj_name, obj_info in scene_info["objects"].items():
            # if False : 
            if obj_info['mount']['floor']:
                candidate = os.path.join(_PROJECT_ROOT, "tests", "obj.blend")
                if not os.path.exists(candidate):
                    candidate = "tests/obj.blend"
                obj = merge_obj_from_blend(candidate)
            elif dataset=="holodeck":
                from digital_cousins.models.objaverse.constants import OBJATHOR_ASSETS_DIR
                basedir = OBJATHOR_ASSETS_DIR
                candidate = obj_info['model']
                filename = f'{basedir}/{candidate}/{candidate}.pkl.gz'
                obj = load_pickled_3d_asset(filename)
            else:
                step_2_dir = save_dir.replace("step_3_output","step_2_output")
                with open(f"{step_2_dir}/objav_files.json", "r") as f:
                    candidates_objav = json.load(f)
                modelname = obj_info['model']
                candidates = candidates_objav["_".join(obj_name.split("_")[:-1])]
                filename = [i for i in candidates if modelname in i]
                obj = load_openshape(filename[0])

            obj_pos, obj_quat = T.mat2pose(T.pose_in_A_to_pose_in_B(
                pose_A=np.array(obj_info["tf_from_cam"]),
                pose_A_in_B=T.pose2mat(cam_pose),
            ))
            set_position_orientation(obj,obj_pos,obj_quat)
            
            obj.scale = obj_info['scale']
            obj.name = obj_name
            bpy.context.view_layer.update() 
        
        # Initialize all objects by taking one step
        bpy.context.view_layer.update() 
        return scene
    
    @staticmethod
    def take_photo():
        """
        Takes photo with current scene configuration with current camera

        Args:
            n_render_steps (int): Number of rendering steps to take before taking the photo

        Returns:
            np.ndarray: (H,W,3) RGB frame from viewer camera perspective
        """
        # Render a bit,

        
        add_light()
        # Render the image
        # Set the render settings
        image_path = "debug.png" 
        bpy.context.scene.render.filepath = image_path# Change the filepath as needed
        bpy.context.scene.render.image_settings.file_format = 'PNG'  # Set the desired image format
        # Render the image
        bpy.ops.render.render(write_still=True)
    
        image = Image.open(image_path)
        # Convert the image to a NumPy array (RGBA format by default)
        image_np = np.array(image)
        # If the image is RGBA and you want to discard the alpha channel, use:
        rgb = image_np[:, :, :3]
    

        return rgb

    @staticmethod
    def render_video(scene_save_dir,visualize_scene_tilt_angle,visualize_scene_radius,save_visualization):
        #calculate scene size 
        aabb_points = []
        scene = bpy.context.scene
        # Set pass indices for objects (manually set based on your scene)
        for obj in scene.objects:
            if "Point" in obj.name or "Camera" in obj.name or "Area" in obj.name:
                continue
            p1, p2 = get_aabb(obj)
            aabb_points.append(p1)
            aabb_points.append(p2)
            # if ToggledOn in obj.states:
            #     obj.states[ToggledOn].link.visible = False

        min_x = min([p[0] for p in aabb_points])
        min_y = min([p[1] for p in aabb_points])
        max_x = max([p[0] for p in aabb_points])
        max_y = max([p[1] for p in aabb_points])
    
        # Get camera trajectory
        camera = bpy.context.scene.camera
        vis_cam_pos, vis_cam_ori = get_position_orientation(camera)
        vis_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0, vis_cam_pos[-1])
        cam_commands = get_vis_cam_trajectory(center_pos=vis_center, cam_pos=vis_cam_pos, cam_quat=vis_cam_ori, \
                                            d_tilt=visualize_scene_tilt_angle, radius=visualize_scene_radius, n_steps=100)
        
        video_path = f"{scene_save_dir}/visualization_video.mp4"
        img_dir = f"{scene_save_dir}/scene_visualization"
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        video_writer = imageio.get_writer(video_path, fps=20)
        filter_names = {"floors", "background"}
        view_layer = bpy.context.scene.view_layers["ViewLayer"]
        view_layer.use_pass_object_index = True

        
        object_ids = {obj.name: i for i, obj in enumerate(bpy.data.objects, start=1) if "Point" in obj.name or "Camera" in obj.name}
        for obj_name, idx in object_ids.items():
                obj = bpy.data.objects.get(obj_name)
                if obj:
                    obj.pass_index = idx

        # iterate camera trajectory
        for i, (pos, quat) in enumerate(cam_commands):
            set_position_orientation(camera,pos, quat)
                                
            bpy.context.scene.frame_set(i)  # Set frame in Blender's timeline
            SimulatedSceneGenerator.capture_and_process_frame(i,img_dir,object_ids,filter_names,video_writer)
        
        # Close video writer
        video_writer.close()

        return
    
    @staticmethod
    def capture_and_process_frame(frame_number,output_image_dir,object_ids,filter_names,video_writer):
        # Render the RGB image
        bpy.context.scene.render.filepath = f"{output_image_dir}/frame_{frame_number}.png"
        bpy.ops.render.render(write_still=True)
        
        # Load the rendered image as RGB
        rgb_image = bpy.data.images.load(f"{output_image_dir}/frame_{frame_number}.png")
        rgb_pixels = np.array(rgb_image.pixels[:]).reshape(rgb_image.size[1], rgb_image.size[0], 4) * 255
        rgb_pixels = rgb_pixels.astype(np.uint8)  # Convert to uint8 for processing
        # import pdb
        # pdb.set_trace()
        seg_mask = rgb_pixels[:,:,-1].reshape(rgb_image.size[1], rgb_image.size[0]) 

        rgb_image.user_clear()
        bpy.data.images.remove(rgb_image)
        
        
        # Apply mask to RGB image
        masked_rgb = rgb_pixels.copy()
        masked_rgb[seg_mask == 0, :3] = [0, 0, 0]  # Set RGB to black for filtered areas
        masked_rgb[:, :, 3] = seg_mask  # Set alpha channel to mask
        masked_rgb = masked_rgb[::-1,:,:]

        # Save masked image
        Image.fromarray(masked_rgb).save(f"{output_image_dir}/vis_frame_{frame_number}.png")
        
        # Append masked image to video writer
        video_writer.append_data(masked_rgb)

        return
    

if __name__== "__main__":

    import random
    random.seed(42)
    import numpy as np
    np.random.seed(42)
    
    config_step3_filename = str(sys.argv[-1])
   
    print(config_step3_filename)
    with open(config_step3_filename,"r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    step_3 = SimulatedSceneGenerator(
        verbose=config["verbose"],
    )

    del config["verbose"]
    success, step_3_output_path = step_3(**config)
    if not success:
        raise ValueError("Failed ACDC Step 3!")