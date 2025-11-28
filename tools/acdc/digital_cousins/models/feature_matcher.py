import numpy as np
import torch
from PIL import Image
import cv2
from pathlib import Path
import json
import faiss
import os
from groundingdino.util.inference import load_image, annotate
from groundingdino.datasets.transforms import RandomResize
from torchvision.ops import box_convert
from copy import deepcopy

from digital_cousins.models.clip import CLIPEncoder
from digital_cousins.models.dino_v2 import DinoV2Encoder
# from digital_cousins.models.dift import DIFTEncoder
from digital_cousins.models.grounded_sam_v2 import GroundedSAMv2


class FeatureMatcher(torch.nn.Module):
    ENCODERS = {
        "DinoV2Encoder": DinoV2Encoder,
        "CLIPEncoder": CLIPEncoder,
        # "DIFTEncoder": DIFTEncoder,
    }
    def __init__(
            self,

            # Encoder kwargs
            encoder="DinoV2Encoder",
            encoder_kwargs=None,

            # Grounded SAM v2 kwargs
            gsam_box_threshold=0.25,  # Grounded SAM边界框的阈值
            gsam_text_threshold=0.25, # Grounded SAM文本注释的阈值

            # General kwargs
            device="cuda",
            verbose=False, # 是否打印详细信息
    ):
        """
        Args:
            encoder (str): Type of visual encoder used to generate visual embeddings.
                Valid options are {"DinoV2Encoder", "CLIPEncoder"}
            encoder_kwargs (None or dict): If specified, encoder-specific kwargs to pass to the encoder constructor
            gsam_box_threshold (float): Confidence threshold for generating GroundedSAM bounding box. 
                If there are undetected objects in the input scene, consider decrease this value.
            gsam_text_threshold (float): Confidence threshold for generating GroundedSAM text annotation.
                If there are undetected objects in the input scene, consider decrease this value.
            device (str): Device to use for storing tensors
            verbose (bool): Whether to display verbose print outs during execution or not
        """
        # Call super first
        super().__init__()

        # Initialize internal variables
        self.device = device
        self.verbose = verbose
        self.encoder_name = encoder

        # Create encoder # 创建编码器实例
        assert encoder in self.ENCODERS, \
            f"Invalid encoder specified! Valid options are: {self.ENCODERS.keys()}, got: {encoder}"
        encoder_kwargs = dict() if encoder_kwargs is None else encoder_kwargs
        encoder_kwargs["device"] = device
        self.encoder = self.ENCODERS[self.encoder_name](**encoder_kwargs)

        # Create SAM model
        self.gsam = GroundedSAMv2(
            box_threshold=gsam_box_threshold,
            text_threshold=gsam_text_threshold,
            device=self.device,
        )
        self.eval()

    def compute_segmentation_mask( # 计算分割掩膜
            self,
            input_category, # 要分割的对象类别
            input_img_fpath, 
            save_dir=None,
            save_prefix=None, 
            multi_results=False, # 是否计算多个结果,是否计算多个边界框
    ):
        """
        Args:
            input_category (str): Name of the desired object category to segment from
                @input_img_fpath. It is this category that is assumed will be attempted
                to be segmented from the image located at @input_img_fpath
            input_img_fpath (str): Absolute filepath to the input object image
            save_dir (None or str): If specified, the absolute path to the directory where the results should be saved.
                If None, will default to the same directory of @input_img_fpath. Note that in either case the
                file saved is named f"{input_category}_mask.png"
            save_prefix (None or str): If specified, the prefix string name for saved outputs.
                If None, saved outputs will be prepended with @input_category instead
            multi_results (bool): Whether to compute multiple bounding boxes or only the highest probability one

        Returns:
            2-tuple:
                - list of np.ndarray: (H, W) segmented mask(s) (>1 if multi_results=True)
                - list of str: List of absolute paths to the segmented mask(s) (>1 if multi_results=True)
        """
        # Standardize save dir and make sure it exists
        save_dir = str(Path(input_img_fpath).parent) if save_dir is None else save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_prefix = input_category if save_prefix is None else save_prefix

        # Load the input image 加载输入图像
        image_source, image = load_image(input_img_fpath)

        # Predict the bounding boxes  # 预测边界框
        boxes, logits, phrases = self.gsam.predict_boxes(img=image, caption=f"{input_category}.")

        # Save this image 保存注释图像
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        cv2.imwrite(f"{save_dir}/{save_prefix}_annotated_bboxes.png", annotated_frame)

        # Only keep pixels within the segmented category box
        # Sort them based on filtering mechanism 只保留分割类别框内的像素
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Calculate the number of actually category-level bboxes, make sure we have at least 1
        # 计算有效类别边界框的数量，确保至少有一个
        category_boxes = [(box, logit) for box, logit in zip(boxes_xyxy, logits)]
        n_obj_bboxes = len(category_boxes)
        assert n_obj_bboxes > 0, "Did not find any valid category-level obj bboxes!"

        if n_obj_bboxes > 1:
            if not multi_results: # 如果有多个边界框且不需要多个结果
                # Grab highest probability one
                obj_bboxes = sorted(category_boxes, key=lambda x: x[1])
                obj_bbox = [obj_bboxes[-1][0]] # 选择概率最高的边界框
        else:
            obj_bbox = [category_boxes[0][0]]

        masks, file_dirs = [], []
        if n_obj_bboxes > 1 and multi_results: # 如果有多个边界框且需要多个结果
            for i, (box, logit) in enumerate(category_boxes):
                # Get segmentation mask for the object category
                obj_masks = self.gsam.predict_segmentation(
                    img_source=image_source,
                    boxes=np.array([box]),
                    cxcywh=False,
                )
                obj_mask = obj_masks[0].squeeze(axis=0)
                save_fpath = f"{save_dir}/{save_prefix}_{i}_mask.png"
                Image.fromarray(obj_mask).save(save_fpath)  # 保存掩膜
                masks.append(obj_mask) # 添加掩膜到列表
                file_dirs.append(save_fpath) # 添加文件路径到列表
        else:
            # Get segmentation mask for the object category
            obj_masks = self.gsam.predict_segmentation(
                img_source=image_source,
                boxes=np.array(obj_bbox),
                cxcywh=False,
            )
            obj_mask = obj_masks[0].squeeze(axis=0)
            save_fpath = f"{save_dir}/{save_prefix}_mask.png"
            Image.fromarray(obj_mask).save(save_fpath)
            masks.append(obj_mask)
            file_dirs.append(save_fpath)

        return masks, file_dirs # 返回掩膜和文件路径

    def find_nearest_neighbor_candidates(
            self,
            input_category, #输入图像中要分割的目标物体类别。
            input_img_fpath,# 输入图像的绝对路径。
            candidate_imgs_fdirs=None, #候选图像目录路径。
            candidate_imgs=None, # 候选图像的路径，如果提供了，直接使用这些图像。
            candidate_filter=None,#用于筛选候选图像的过滤器。
            n_candidates=4, #返回最相似的候选图像的数量。
            save_dir=None, #保存结果的目录路径。如果为 `None`，则使用输入图像所在的目录。
            visualize_resolution=(640, 480),
            boxes=None,#如果已计算，传入物体的边界框。
            logits=None,#如果已计算，传入分类的概率分布。
            phrases=None,#物体的描述词。
            obj_masks=None,#如果已计算，传入物体的分割掩码。
            save_prefix=None,#结果保存时的前缀。如果为 `None`，则使用 `input_category`。
            remove_background=True,#是否在计算 DINO 特征前移除背景。
            use_input_img_without_bbox=False #是否直接使用输入图像计算特征，不使用边界框。
    ):
        """
        Args:
            input_category (str): Name of the desired object category to segment from
                @input_img_fpath. It is this category that is assumed will be attempted
                to be matched to nearest neighbor candidate(s)
            input_img_fpath (str): Absolute filepath to the input object image
            candidate_imgs_fdirs (None or str or list of str): Absolute filepath(s) to the candidate images directory(s)
            candidate_imgs (None or list of str): Absolute filepath(s) to the candidate images. If this is not None, directly use this. Otherwise, use candidate_imgs_fdirs
            candidate_filter (None or TextFilter): If specified, TextFilter for pruning all possible
                candidates from @candidate_imgs_fdir
            n_candidates (int): The number of nearest neighbor candidates to return.
            save_dir (None or str): If specified, the absolute path to the directory where the results should be saved.
                If None, will default to the same directory of @input_img_fpath.
            visualize_resolution (2-tuple): (H, W) when visualizing candidate results
            boxes (None or tensor): If specified, pre-computed SAM boxes to use
            logits (None or tensor): If specified, pre-computed SAM logits to use
            phrases (None or list): If specified, pre-computed SAM phrases to use
            obj_masks (None or np.array): If specified, pre-computed SAM segmentation mask to use
            save_prefix (None or str): If specified, the prefix string name for saved outputs.
                If None, saved outputs will be prepended with @input_category instead
            remove_background (bool): Whether to remove background before computing DINO features
            use_input_img_without_bbox (bool): Whether to directly use the input image to compute dino score,
                or with a bounding box of the target object

        Returns:
            dict: Dictionary of outputs. Note that this will also be saved to f"{save_prefix}_feature_matcher_results.json"
                in @save_dir
        """
        # 检查输入参数是否有效
        assert " " not in input_category# 类别名称不能包含空格
        assert (candidate_imgs_fdirs is not None) or (candidate_imgs is not None)

        # Standardize save dir and make sure it exists
        save_dir = str(Path(input_img_fpath).parent) if save_dir is None else save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_prefix = input_category if save_prefix is None else save_prefix

        # Standardize other inputs
        if candidate_imgs is None:
            candidate_imgs_fdirs = [candidate_imgs_fdirs] if isinstance(candidate_imgs_fdirs, str) else candidate_imgs_fdirs

        # Load the input image
        image_source, image = load_image(input_img_fpath)
        ref_img_vis = cv2.resize(image_source, visualize_resolution)
        H_ref, W_ref, _ = image_source.shape

        # Predict the bounding boxes
        if self.verbose:
            print(f"{self.__class__.__name__}: Computing GroundedSAMv2 obj boxes...")

        # First find category-level boxes
        # 首先，找到与类别相关的边界框, 获取物体的分割掩码
        if not use_input_img_without_bbox:
            if boxes is None or logits is None or phrases is None:
                # Either all or none of must be None
                # 如果没有提供边界框或分类信息，则需要通过模型进行预测
                assert boxes is None and logits is None and phrases is None,\
                    "All of boxes, logits, and phrases must be None if at least one of them is None!"
                boxes, logits, phrases = self.gsam.predict_boxes(img=image, caption=f"{input_category.replace('_', ' ')}.")

            # Save this image
            # 将预测的边界框标注在图像上并保存
            annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
            cv2.imwrite(f"{save_dir}/{save_prefix}_annotated_bboxes.png", annotated_frame)

            # Only keep pixels within the segmented category box
            # Sort them based on filtering mechanism
            # 只保留目标类别的物体边界框
            boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            category_boxes = []

            # 如果边界框中有多个物体，选择属于目标类别的边界框
            # Infer which of the bboxes belong to the object itself if there's more than 1
            if len(logits) > 1:
                for box, logit, phrase in zip(boxes_xyxy, logits, phrases):
                    if phrase.replace("_", " ") == input_category.replace("_", " "):
                        category_boxes.append((box, logit))
            else:
                category_boxes = [(boxes_xyxy[0], logits[0])]

            # 确保找到至少一个有效的目标类别边界框
            # Calculate the number of actually category-level bboxes, make sure we have at least 1
            n_obj_bboxes = len(category_boxes)
            assert n_obj_bboxes > 0, "Did not find any valid category-level obj bboxes!"
            if n_obj_bboxes > 1:
                # Grab highest probability one
                obj_bboxes = sorted(category_boxes, key=lambda x: x[1])
                obj_bbox = obj_bboxes[-1][0]
            else:
                obj_bbox = category_boxes[0][0]

            # 计算物体的像素区域
            # Calculate the obj pixels based on its bbox
            box_pixels = (obj_bbox * np.array([W_ref, H_ref, W_ref, H_ref])).astype(int)

            # 获取物体的分割掩码
            # Get segmentation mask for the object itself
            if obj_masks is None:
                obj_masks = self.gsam.predict_segmentation(
                    img_source=image_source,
                    boxes=np.array([obj_bbox]),
                    cxcywh=False,
                )
            obj_mask = obj_masks[0].squeeze(axis=0)
            Image.fromarray(obj_mask).save(f"{save_dir}/{save_prefix}_mask.png")
        else:
            # 如果使用不带边界框的输入图像
            height, width, _ = image_source.shape
            obj_mask = np.ones((height, width))

        # Mask the original image source and image and then infer the corresponding part-level segmentations
        # 使用分割掩码处理原始图像并进行特征提取
        # TODO: Black out all background pixels from ref_img_cropped using segmentation mask
        preprocess = RandomResize([800], max_size=1333)
        obj_mask_resized = np.expand_dims(np.array(preprocess(Image.fromarray(obj_mask))[0]), axis=0)
        image_source_masked = image_source * np.expand_dims(np.where(obj_mask > 0.5, 1.0, 0.0), axis=-1).astype(np.uint8) if remove_background else image_source
        image_masked = image * torch.tensor(obj_mask_resized, requires_grad=False) if remove_background else image
        if use_input_img_without_bbox:
            ref_img_cropped = image_source_masked
            obj_mask_cropped = obj_mask
        else:
            ref_img_cropped = image_source_masked[box_pixels[1]:box_pixels[3], box_pixels[0]:box_pixels[2]]
            obj_mask_cropped = obj_mask[box_pixels[1]:box_pixels[3], box_pixels[0]:box_pixels[2]]
        ref_img_masked_vis = cv2.resize(image_source_masked, visualize_resolution)
        Image.fromarray(ref_img_masked_vis).save(f"{save_dir}/{save_prefix}_mask_img.png")
        
        # Get all valid candidates and load them # 加载所有候选图像
        models = list(sorted(f"{candidate_imgs_fdir}/{model}"
                             for candidate_imgs_fdir in candidate_imgs_fdirs for model in os.listdir(candidate_imgs_fdir)
                             if (candidate_filter is None or candidate_filter.process(model)))) \
            if candidate_imgs is None else sorted(candidate_imgs)
        model_imgs = np.array([np.array(Image.open(model).convert("RGB")) for model in models])

        # 计算输入图像和候选图像的DINO特征，并将它们重塑为(N, D)的数组
        # Compute DINO features and reshape them to be (N, D) arrays
        ref_img_feats = self.encoder.get_features(ref_img_cropped,input_category).squeeze(axis=0)  # (84, 112, 384)  #(37, 49, 1280)
        model_imgs_feats = self.encoder.get_features(model_imgs,input_category)    # (64, 84, 112, 384)  #(70, 37, 49, 1280)
        ref_feat_vecs = ref_img_feats.reshape(-1, self.encoder.embedding_dim)   # (9408, 384)

        # TODO: Remove background pixels from candidate feat vecs, via better dataset parsing (use alpha channel = 0.0)
        model_feat_vecs = model_imgs_feats.reshape(-1, self.encoder.embedding_dim)  # (602112, 384)

        
        # Reshape cropped image to be the same shape as the feature size
        if self.encoder_name == "DinoV2Encoder":
            H, W, C = ref_img_feats.shape
            obj_mask_cropped_resized = cv2.resize(obj_mask_cropped.astype(np.uint8), (W, H))
        elif self.encoder_name == "CLIPEncoder":
            obj_mask_cropped_resized = obj_mask_cropped
        elif self.encoder_name == "DIFTEncoder":
            H, W, C = ref_img_feats.shape
            obj_mask_cropped_resized = cv2.resize(obj_mask_cropped.astype(np.uint8), (W, H))
        else:
            raise ValueError(f"Got invalid encoder_name! Valid options: {self.ENCODERS.keys()}, got: {self.encoder_name}")
        
        # 根据前景像素进行特征匹配
        # Get set of idxs corresponding to foreground
        foreground_idxs = set(obj_mask_cropped_resized.flatten().nonzero()[0])
        
        
        # Match features; compute top-K likely models
        top_k_models = []
        models_copy = deepcopy(models)
        model_imgs_copy = np.array(model_imgs)
        feat_vecs = np.array(model_feat_vecs)   # (602112, 384)
        imgs = [ref_img_vis, ref_img_masked_vis]

        if self.verbose:
            print(f"{self.__class__.__name__}: Computing top-{n_candidates} candidates using encoder {self.encoder_name}...")

        n_candidates = min(n_candidates, len(models))
        
        # 使用FAISS进行高效的特征匹配
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatL2(self.encoder.embedding_dim)
        gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

        # 计算最接近的候选图像
        if self.encoder_name in ["DinoV2Encoder","DIFTEncoder"]:
            for i in range(n_candidates):
                if self.verbose:
                    print(f"{self.__class__.__name__}: Computing DINO candidate {i+1}...")
                gpu_index_flat.reset()
                gpu_index_flat.add(feat_vecs)
                dists, idxs = gpu_index_flat.search(ref_feat_vecs, 1)   # (9408, 1)
                idxs = idxs // (H * W)
                freqs = dict()
                for j, (idx, dist) in enumerate(zip(idxs.reshape(-1), dists.reshape(-1))):
                    # If j is not part of foreground, skip
                    if j not in foreground_idxs:
                        continue

                    if idx not in freqs:
                        freqs[idx] = 0
                    # Add weighting
                    freqs[idx] += 1

                freqs = {k: v for k, v in sorted(freqs.items(), key=lambda item: item[1])}
                top_1_idx = list(freqs.keys())[-1]

                top_k_models.append(models_copy[top_1_idx])
                imgs.append(cv2.resize(model_imgs_copy[top_1_idx], visualize_resolution))
                # Prune the selected one
                del models_copy[top_1_idx]
                model_imgs_copy = np.delete(model_imgs_copy, top_1_idx, axis=0)
                feat_vecs = np.delete(feat_vecs, np.arange(H * W * top_1_idx, H * W * (top_1_idx + 1)), axis=0)

        elif self.encoder_name == "CLIPEncoder":
            gpu_index_flat.reset()
            gpu_index_flat.add(feat_vecs)
            dists, idxs = gpu_index_flat.search(ref_feat_vecs, n_candidates)

            # Store top-k models and distances
            for idx in idxs[0]:
                top_k_models.append(models_copy[idx])
                imgs.append(cv2.resize(model_imgs_copy[idx], visualize_resolution))

        # Record results # 记录结果并保存
        if self.verbose:
            print(f"Top-{n_candidates} models: {[model.split('/')[-1] for model in top_k_models]}")
        concat_img = np.concatenate(imgs, axis=1)
        Image.fromarray(concat_img).save(f"{save_dir}/{save_prefix}_feature_matcher_results_visualization.png")
        results = {
            "k": n_candidates,
            "candidates": top_k_models,
        }
        with open(f"{save_dir}/{save_prefix}_feature_matcher_results.json", "w+") as f:
            json.dump(results, f)

        return results
