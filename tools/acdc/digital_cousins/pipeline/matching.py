import torch
from torchvision.ops.boxes import _box_xyxy_to_cxcywh
from groundingdino.util.inference import load_image
import numpy as np
from pathlib import Path
from PIL import Image
import os
import json
import cv2
import faiss
import re
import warnings
import math
import digital_cousins
import digital_cousins.utils.transform_utils as T
from digital_cousins.models.clip import CLIPEncoder
from digital_cousins.models.gpt import GPT
from digital_cousins.utils.processing_utils import NumpyTorchEncoder, compute_bbox_from_mask

from digital_cousins.models.blend.render_candidates import render_candidates
from digital_cousins.models.objaverse import get_candidates,get_candidates_all

DO_NOT_MATCH_CATEGORIES = {"walls", "floors", "ceilings"}
IMG_SHAPE_OG = (720, 1280)

class DigitalCousinMatcher:
    """
    2nd Step in ACDC pipeline. This takes in the output from Step 1 (Real World Extraction) and generates
    ordered digital cousin candidates from a given dataset (default is Behavior-1K dataset)

    Foundation models used:
        - GPT-4O (https://openai.com/index/hello-gpt-4o/)
        - CLIP (https://github.com/openai/CLIP)
        - DINOv2 (https://github.com/facebookresearch/dinov2)

    Inputs:
        - Output from Step 1, which includes the following:
            - Camera Intrinsics Matrix
            - Detected Categories information
            - Floor segmentation mask
            - Wall(s) segmentation mask(s)
            - Estimated z-direction in the camera frame
            - Selected origin position in the camera frame
            - Input RGB image
            - Input (linear) Depth image (potentially synthetically generated)
            - Depth limits (min, max)
            - Mount type

    Outputs:
        - Ordered digital cousin (category, model, pose) information per detected object from Step 1
    """

    def __init__(
            self,
            feature_matcher,
            verbose=False,
    ):
        """
        Args:
            feature_matcher (FeatureMatcher): Feature matcher class instance to use for segmenting objects
                and matching features
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        self.fm = feature_matcher
        self.fm.eval()
        self.verbose = verbose
        self.device = self.fm.device

    def __call__(
            self,
            step_1_output_path,
            dataset,
            gpt_api_key,
            gpt_version="4o",
            top_k_categories=3,
            top_k_models=8,
            top_k_poses=3,
            n_digital_cousins=3,
            n_cousins_reselect_cand=3,
            remove_background=False,
            gpt_select_cousins=False,
            n_cousins_link_count_threshold=3,
            save_dir=None,
            start_at_name=None,
    ):
        
        """
        Runs the digital cousin matcher. This does the following steps for each detected object from Step 1:

        1. Use CLIP embeddings to find the top-K nearest OmniGibson dataset categories for a given box + mask
        2. Select digital cousins using encoder features + GPT

        Args:
            step_1_output_path (str): Absolute path to the output file generated from Step 1 (RealWorldExtractor)
            gpt_api_key (str): Valid GPT-4O compatible API key
            gpt_version (str): GPT version to use. Valid options are {"4o", "4v"}.
                Default is "4o", which we've found to work empirically better than 4V
            top_k_categories (int): Number of closest categories from the OmniGibson dataset from which digital
                cousin candidates will be selected
            top_k_models (int): Number of closest asset digital cousin models from the OmniGibson dataset to select
            top_k_poses (int): Number of closest asset digital cousin model poses to select
            n_digital_cousins (int): Number of digital cousins to select. This number cannot be greater than
                @top_k_models
            n_cousins_reselect_cand (int): The frequency of reselecting digital cousin candidates.
                If set to 1, reselect candidates for each digital cousin.
            remove_background (bool): Whether to remove background before computing visual encoder features when
                computing digital cousin candidates
            gpt_select_cousins (bool): Whether to prompt GPT to select nearest asset as a digital cousin.
                If False, the nearest digital cousin will be the one with the least DINOv2 embedding distance.
            start_at_name (None or str): If specified, the name of the object to start at. This is useful in case
                the pipeline crashes midway, and can resume progress without starting from the beginning
            n_cousins_link_count_threshold (int): The number of digital cousins to apply door/drawer count threshold
                during selection. When selecting digital cousin candidates for articulated objects, setting this as a
                positive integer will leverage the GPT-driven door / drawer annotations from Step 1 to further constrain
                the potential candidates during visual encoder selection.
                If set to 0, this threshold will not be used.
                If set larger than n_digital_cousins, this threshold will always be used.
            save_dir (None or str): If specified, the absolute path to the directory to save all generated outputs. If
                not specified, will generate a new directory in the same directory as @step_1_output_path

        Returns:
            2-tuple:
                bool: True if the process completed successfully. If successful, this will write all relevant outputs to
                    the directory specified in the second output
                None or str: If successful, this will be the absolute path to the main output file. Otherwise, None
        """
        # Sanity check values
        assert n_digital_cousins <= top_k_models, \
            f"n_digital_cousins ({n_digital_cousins}) cannot be greater than top_k_models ({top_k_models})!"

        # Parse save_dir, and create the directory if it doesn't exist
        if save_dir is None:
            save_dir = os.path.dirname(step_1_output_path)
        save_dir = os.path.join(save_dir, "step_2_output")
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Computing digital cousins given output {step_1_output_path}...")

        if self.verbose:
            print("""

##################################################################
### 1. Use CLIP embeddings to find top-K categories per object ###
##################################################################

            """)

        # Load meta info
        with open(step_1_output_path, "r") as f:
            step_1_output_info = json.load(f) # 读取步骤1的输出信息

        with open(step_1_output_info["detected_categories"], "r") as f:
            detected_categories_info = json.load(f) # 读取检测到的类别信息

        # Split into non-/articulated groups
        names = detected_categories_info["names"]
        phrases_recaptioned = detected_categories_info["phrases_recaptioned"]  # 获取重新描述的短语
        # 获取关节计数
        # articulation_counts = {int(k): v for k, v in detected_categories_info["articulation_counts"].items()}
        # # 获取所有类别，并排除特定类别
        # all_categories = list(get_all_dataset_categories(do_not_include_categories=DO_NOT_MATCH_CATEGORIES, replace_underscores=True))
        # # 获取所有关节类别
        # all_articulated_categories = list(get_all_articulated_categories(do_not_include_categories=DO_NOT_MATCH_CATEGORIES, replace_underscores=True))
        # # 获取关节类别的索
        # articulation_indexes = list(articulation_counts.keys())
        # # 获取非关节类别的索引
        # non_articulation_indexes = [idx for idx in range(len(phrases_recaptioned)) if idx not in articulation_counts]

        # # Run CLIP to determine top-K category matches
        # clip = CLIPEncoder(backbone_name="ViT-B/32", device=self.device) # 初始化CLIP编码器
        # # 创建FAISS GPU资源
        # res = faiss.StandardGpuResources()
        # # 创建L2距离的扁平索引
        # index_flat = faiss.IndexFlatL2(clip.embedding_dim)
        # # 将索引移动到GPU
        # gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)
        # # 用于存储选择的类别
        # selected_categories = dict()
        

        # # 遍历非关节和关节对象的索引及其对应的类别
        # for obj_indexes, categories in zip(
        #         (non_articulation_indexes, articulation_indexes),
        #         (all_categories, all_articulated_categories),
        # ):
        #     # import pdb
        #     # pdb.set_trace()
        #     obj_phrases = [phrases_recaptioned[idx] for idx in obj_indexes]  # 获取对象的短语
        #     if len(obj_phrases) > 0: # 确保短语列表不为空 

        #         if self.verbose:
        #             print(f"Computing top-{top_k_categories} for phrases: {obj_phrases}")

        #         text_features = clip.get_text_features(text=categories) # 获取类别的文本特征 (1800, 512)
        #         cand_text_features = clip.get_text_features(text=obj_phrases) # 获取对象短语的文本特征 (7, 512)
        #         gpu_index_flat.reset()
        #         gpu_index_flat.add(text_features) # 将类别特征添加到索引中
        #         dists, idxs = gpu_index_flat.search(cand_text_features, top_k_categories) # 搜索前K个匹配 (7,top_k_categories)
        #         for obj_idx, topk_idxs in zip(obj_indexes, idxs):   # 遍历每个对象和其前K个索引
        #             # 保存选择的类别
        #             selected_categories[names[obj_idx]] = [categories[topk_idx] for topk_idx in topk_idxs]

        # # Store these results
        # topk_categories_info = {
        #     "topk_categories": selected_categories,
        # }
        # topk_categories_path = f"{save_dir}/topk_categories.json"
        # with open(topk_categories_path, "w+") as f:
        #     json.dump(topk_categories_info, f, indent=4)

        # # Clean up resources to avoid OOM error
        # del res
        # del clip

        if self.verbose:
            print("""

        

##############################################################
### 2. Select digital cousins using encoder features + GPT ###
##############################################################

            """)

        
        input_rgb_path = step_1_output_info["input_rgb"]
        boxes = torch.tensor(detected_categories_info["boxes"])
        logits = torch.tensor(detected_categories_info["logits"])
        phrases = detected_categories_info["phrases"]
        mounts = detected_categories_info["mount"]
        seg_dir = detected_categories_info["segmentation_dir"]

        # Create GPT instance
        assert gpt_api_key is not None, "gpt_api_key must be specified in order to use GPT model!"
        # gpt = GPT(api_key=gpt_api_key, version=gpt_version)

        should_start = start_at_name is None
        n_instances = len(names)
        if dataset=="openshape":
            candidates_all = get_candidates_all(names,dataset,save_dir)
        
        for instance_idx, (box, logit, phrase, name, mount) in enumerate(
                zip(boxes, logits, phrases, names, mounts)
        ):  
            # if "monitor" not in name:
            #     continue
            # Skip if starting at name has not been reached yet
            if not should_start:
                if start_at_name == name:
                    should_start = True
                else:
                    # Immediately keep looping
                    continue

            og_categories = name
            obj_save_dir = f"{save_dir}/{name}"
            topk_model_candidates_dir = f"{obj_save_dir}/top_k_model_candidates"
            topk_pose_candidates_dir = f"{obj_save_dir}/top_k_pose_candidates"
            cousin_visualization_dir = f"{obj_save_dir}/cousin_visualization"
            obj_mask_fpath = f"{seg_dir}/{name}_nonprojected_mask.png"
            mask = np.array(Image.open(obj_mask_fpath))
            is_articulated = False
            Path(cousin_visualization_dir).mkdir(parents=True, exist_ok=True)

            # Need to replace the category with underscores to preserve the original naming scheme
            category = phrase.replace(" ", "_")


            if self.verbose:
                print("-----------------")
                print(f"[Object {instance_idx + 1} / {n_instances}] Finding digital cousins for object {name}, category: {category}...")

            # Load the unpruned mask and bboxes to use for digital cousin selection
            obj_masks = mask.reshape(1, 1, *mask.shape)
            bboxes = _box_xyxy_to_cxcywh(torch.tensor(compute_bbox_from_mask(obj_mask_fpath))).unsqueeze(dim=0)

            cousin_results = {
                "articulated": False,
                "mount":mount,
                "cousins": [],
            }
            selected_models = set()

            # Select digital cousins iteratively

            for i in range(n_digital_cousins):
                # Reselect candidates
                if i % n_cousins_reselect_cand == 0 or i >= n_cousins_link_count_threshold:
                    if self.verbose:
                        print(f"Reselecting candidates using {self.fm.encoder_name}...")

                    # Find Top-K candidates
                    # candidate_imgs_fdirs = [f"{digital_cousins.ASSET_DIR}/objects/{og_category.replace(' ', '_')}/snapshot" for og_category in og_categories]

                    # # Possibly filter based on articulated models
                    
                    # candidate_imgs = list(sorted(f"{candidate_imgs_fdir}/{model}"
                    #                             for candidate_imgs_fdir in candidate_imgs_fdirs
                    #                             for model in os.listdir(candidate_imgs_fdir)
                    #                             if model not in selected_models))
                   
                    if dataset=="openshape":
                        candidates = candidates_all["_".join(name.split("_")[:-1])]
                    else: #holodeck
                        candidates = get_candidates(name,phrase,dataset)
                    candidate_imgs = render_candidates(candidates,dataset,start_angle=0,end_angle=315,cnt=7)
                
                    # Run feature-matching!
                    if self.verbose:
                        print(f"Selecting Top-{top_k_models} nearest models using {self.fm.encoder_name}...")
                    
                    model_results = self.fm.find_nearest_neighbor_candidates(
                        input_category=category,
                        input_img_fpath=input_rgb_path,
                        candidate_imgs_fdirs=None,
                        candidate_imgs=candidate_imgs,
                        candidate_filter=None,
                        n_candidates=top_k_models,
                        save_dir=topk_model_candidates_dir,
                        visualize_resolution=(640, 480),
                        boxes=bboxes,
                        logits=logit.unsqueeze(dim=0),
                        phrases=[phrase],
                        obj_masks=obj_masks,
                        save_prefix=f"{name}_iter{i}",
                        remove_background=remove_background,
                    )

                    

                    # Rename bbox and mask images
                    os.rename(f"{topk_model_candidates_dir}/{name}_iter{i}_annotated_bboxes.png", f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png")
                    os.rename(f"{topk_model_candidates_dir}/{name}_iter{i}_mask.png", f"{topk_model_candidates_dir}/{name}_mask.png")

                    current_candidates = model_results["candidates"]

                gpt_select_cousins=False
                if gpt_select_cousins:
                    # Select the nearest model via GPT
                    if self.verbose:
                        print(f"Selecting cousin #{i} final model using GPT...")

                    if is_articulated:
                        # Use prompt specifically for articulation
                        nn_selection_payload = gpt.payload_articulated_nearest_neighbor(
                            caption=phrases_recaptioned[instance_idx],
                            img_path=input_rgb_path,
                            bbox_img_path=f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png",
                            candidates_fpaths=current_candidates
                        )
                    else:
                        nn_selection_payload = gpt.payload_nearest_neighbor(
                            caption=phrases_recaptioned[instance_idx],
                            img_path=input_rgb_path,
                            bbox_img_path=f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png",
                            candidates_fpaths=current_candidates,
                            nonproject_obj_img_path=f"{seg_dir}/{name}_nonprojected.png",
                        )
                    # Query GPT
                    attempt = 0
                    while True:
                        attempt += 1
                        gpt_text_response = gpt(nn_selection_payload)
                        if gpt_text_response is not None:
                            break
                            # Failed, terminate early
                        elif attempt>3: 
                            return False, None
                        else:
                            import time
                            time.sleep(8)

                    # Extract the first non-negative integer from the response
                    match = re.search(r'\b\d+\b', gpt_text_response)
                    if match:
                        nn_model_index = int(match.group()) - 1
                    else:
                        # No valid integer found, handle this case
                        return False, None

                    # Fallback to 0 if invalid value
                    if nn_model_index >= len(current_candidates):
                        nn_model_index = 0
                else:
                    # Select the nearest model via DINOv2 embedding distance
                    if self.verbose:
                        print(f"Selecting cousin #{i} final model using DINOv2...")
                    nn_model_index = 0

                if len(current_candidates) == 0:
                    raise ValueError(f"Not enough candidates to choose digital cousins for {name}!")

                candidate_model = current_candidates[nn_model_index]
                candidate_name = candidate_model.split('/')[-2]
                selected_models.add(candidate_name)

                # Given the selected model, generate pose candidates using our visual encoder
                if self.verbose:
                    print(f"Selecting Top-{top_k_poses} nearest poses using {self.fm.encoder_name}...")

                # og_category = candidate_model.split("/objects/")[-1].split("/snapshot/")[0]
                # og_model = candidate_model.split(".")[0].split(f"{og_category}_")[-1]
                cousin_topk_pose_candidates_dir = f"{topk_pose_candidates_dir}/cousin{i}"

                # # Articulated objects have link count > 0, which indicate that the frontal face can be seen,
                # # so we only search best pose among orientations where the frontal face can be seen
                # start_idx, end_idx = ARTICULATION_VALID_ANGLES.get(og_category, {}).get(og_model, [0, 99])
                # candidate_imgs = [f"{digital_cousins.ASSET_DIR}/objects/{og_category}/model/{og_model}/{og_model}_{rot_idx}.png" for rot_idx in range(start_idx, end_idx + 1)]

                if mount["floor"]:
                # if False:
                    # candidates = ["/home/yandan/Desktop/desk.blend"] 
                    candidates = ["/home/yandan/workspace/infinigen/record_files/obj.blend"] 
                    if not os.path.exists(candidates[0]):
                        candidates = ["tests/obj.blend"]
                else:
                    
                    if dataset=="openshape":
                        candidates = [i for i in candidates if candidate_name in i]
                    else:
                        candidates = [candidate_name]
                    #choose 90 degree pose
                    candidate_imgs = render_candidates(candidates,dataset,start_angle=0,end_angle=0,cnt=0)

                    pose_results = self.fm.find_nearest_neighbor_candidates(
                        input_category=category,
                        input_img_fpath=input_rgb_path,
                        candidate_imgs_fdirs=None,
                        candidate_imgs=candidate_imgs,
                        n_candidates=len(candidate_imgs),
                        save_dir=cousin_topk_pose_candidates_dir,
                        visualize_resolution=(640, 480),
                        boxes=bboxes,
                        logits=logit.unsqueeze(dim=0),
                        phrases=[phrase],
                        obj_masks=obj_masks,
                        save_prefix=name+"_90degree",
                        remove_background=remove_background,
                    )

                    candidates_new = dict()
                    for c in pose_results["candidates"]:
                        name_c = c.split("/")[-2]
                        if name_c not in candidates_new:
                            candidates_new[name_c] = c

                    candidates = candidates_new.values()

                    ##choose z accurate pose
                    # candidates = ['/home/yandan/workspace/digital-cousins/output/render//84dc5cb99c5f428e838a5ce399262dcf/0_0_90_0.png']
                    if dataset=="openshape":
                        candidate_name = list(candidates)[0].split("/")[-2]
                        candidates = [i for i in candidates_all["_".join(name.split("_")[:-1])] if candidate_name in i]
                    
                candidate_imgs = render_candidates(candidates,dataset=dataset,start_angle=0,end_angle=340,cnt=20)
                pose_results = self.fm.find_nearest_neighbor_candidates(
                    input_category=category,
                    input_img_fpath=input_rgb_path,
                    candidate_imgs_fdirs=None,
                    candidate_imgs=candidate_imgs,
                    n_candidates=top_k_poses,
                    save_dir=cousin_topk_pose_candidates_dir,
                    visualize_resolution=(640, 480),
                    boxes=bboxes,
                    logits=logit.unsqueeze(dim=0),
                    phrases=[phrase],
                    obj_masks=obj_masks,
                    save_prefix=name,
                    remove_background=remove_background,
                )


                # Use GPT to select final pose
                if self.verbose:
                    print(f"Selecting cousin #{i} final pose using GPT...")

                # nn_selection_payload = gpt.payload_nearest_neighbor_pose(
                #     caption=phrases_recaptioned[instance_idx],
                #     img_path=input_rgb_path,
                #     bbox_img_path=f"{topk_model_candidates_dir}/{name}_annotated_bboxes.png",
                #     nonproject_obj_img_path=f"{seg_dir}/{name}_nonprojected.png",
                #     candidates_fpaths=pose_results["candidates"]
                # )

                # # Query GPT
                # attempt = 0
                # while True:
                #     attempt += 1
                #     gpt_text_response = gpt(nn_selection_payload)
                #     if gpt_text_response is not None:
                #         break
                #         # Failed, terminate early
                #     elif attempt>3: 
                #         return False, None
                #     else:
                #         import time
                #         time.sleep(8)
                
                gpt_text_response = "None"

                # Extract the first non-negative integer from the response
                match = re.search(r'\b\d+\b', gpt_text_response)
                if match:
                    nn_pose_index = int(match.group())
                else:
                    # No valid integer found
                    warnings.warn(f"Got invalid response! Valid options are pose indices, got: '{gpt_text_response}'")
                    nn_pose_index = 0

                if nn_pose_index >= len(pose_results["candidates"]):
                    nn_pose_index = 0

                # Add results to final cousins
                snapshot_path = pose_results["candidates"][nn_pose_index]
                # _, _, ori_offset, z_angle = extract_info_from_model_snapshot(snapshot_path)
                angles = snapshot_path.split("/")[-1].split(".")[0]
                rot_z,rot_axis,angle_axis,angle = angles.split("_")
                z_angle = float(snapshot_path.split("/")[-1].split(".")[0].split("_")[-1])/180*math.pi

                cousin_info = {
                    "category": name,
                    "model": candidate_name,
                    "ori_offset": None,
                    # "rotation":{
                    #     "rot_z": rot_z,
                    #     "rot_axis": rot_axis,
                    #     "angle_axis": angle_axis,
                    #     "angle": angle,
                    # },
                    "z_angle": z_angle,
                    "snapshot": snapshot_path,
                }
                cousin_results["cousins"].append(cousin_info)

                # Generate visualization
                image_source, _image = load_image(input_rgb_path)
                ref_img_vis = cv2.resize(image_source, (640, 480))
                H_ref, W_ref, _ = image_source.shape

                imgs = [ref_img_vis]
                nn_img = np.array(Image.open(snapshot_path).convert("RGB"))
                imgs.append(cv2.resize(nn_img, (640, 480)))
                concat_img = np.concatenate(imgs, axis=1)
                Image.fromarray(concat_img).save(
                    f"{cousin_visualization_dir}/cousin{i}_visualization.png")

                # Prune selected cousin for next iteration
                current_candidates_new = [i for i in current_candidates if candidate_name not in i]
                if len(current_candidates_new)>0:
                    current_candidates = current_candidates_new

            # Finally save cousin results
            with open(f"{obj_save_dir}/cousin_results.json", "w+") as f:
                json.dump(cousin_results, f, indent=4, cls=NumpyTorchEncoder)

            if self.verbose:
                print("-----------------\n")

        # Compile final results
        step_2_output_info = dict()
        step_2_output_info["metadata"] = {
            "n_cousins": n_digital_cousins,
            "n_objects": n_instances,
        }
        obj_info = dict()
        for name in names:
            obj_cousin_results_path = f"{save_dir}/{name}/cousin_results.json"
            with open(obj_cousin_results_path, "r") as f:
                obj_cousin_results = json.load(f)
            obj_info[name] = obj_cousin_results
        step_2_output_info["objects"] = obj_info

        step_2_output_path = f"{save_dir}/step_2_output_info.json"
        with open(step_2_output_path, "w+") as f:
            json.dump(step_2_output_info, f, indent=4, cls=NumpyTorchEncoder)

        if self.verbose:
            print(f"Saved Step 2 Output information to {step_2_output_path}")

        print("""

##########################################
### Completed Digital Cousin Matching! ###
##########################################

        """)

        return True, step_2_output_path



