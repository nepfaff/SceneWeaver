<div align="center">
<img src="docs/images/sceneweaver.png" width="300"></img>
</div>

<h2 align="center">
  <b>SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent</b>
</h2>
 <div align="center" margin-bottom="6em">
  <a target="_blank" href="https://yandanyang.github.io/">Yandan Yang</a><sup>✶</sup>,
  <a target="_blank" href="https://buzz-beater.github.io/">Baoxiong Jia</a><sup>✶</sup>,
  <a target="_blank" href="https://github.com/hiShujie">Shujie Zhang</a>,
  <a target="_blank" href="https://siyuanhuang.com/">Siyuan Huang</a>

</div>
<br>
<div align="center">
    <!-- <a href="https://cvpr.thecvf.com/virtual/2023/poster/22552" target="_blank"> -->
    <a href="https://arxiv.org/abs/2404.09465" target="_blank"> 
      <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://sceneweaver.github.io" target="_blank">
      <img src="https://img.shields.io/badge/Page-SceneWeaver-blue" alt="Project Page"/></a>
</div>
<br>
<div style="text-align: center">
<img src="docs/images/teaser.jpg"  />
</div>


<!-- This is the official repository of [**PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI**](https://arxiv.org/abs/2211.05272). -->


For more information, please visit our [**project page**](https://sceneweaver.github.io).

## Available Tools:

Here we adapt the following tools to our framework. You could choose what you want from the following tools or expand the framework to other tools (such as architecture, Text-2-3D)  you need.
Initializer:
LLM: GPT
Dataset: MetaScenes
Model: PhyScene/DiffuScene/ATISS

Implementer:
Visual: SD + Digital Cousin
LLM: GPT
Rule

Modifier:
Update Layout/Rotation/Size
Add Relation
Remove Objects

## ⚙️ Installation & Dependencies
Infinigen
install Infinigen as a Blender Python script:
```
git clone https://github.com/princeton-vl/infinigen.git
cd infinigen
conda create --name infinigen python=3.11
conda activate infinigen 
```

Then, install using one of the options below:
```

# Minimal installation (recommended setting for use in the Blender UI)
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh

# Normal install
bash scripts/install/interactive_blender.sh

# Enable OpenGL GT
INFINIGEN_INSTALL_CUSTOMGT=True bash scripts/install/interactive_blender.sh
```
More details can refer to [official repo of Infinigen](https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md#installing-infinigen-as-a-blender-python-script).


ACDC
sd3.5
IDesign
                                                                                                          
## 🛒 Prepare Data                                                 
You can refer to [DATA.md](DATA.md) to download the original datasets and preprocess the data.
metascene
physcene room
3D future
holodeck-objaverse

## 🚀 Diffusion Model for Scene Synthesis
<!-- ####  Training 
```
sh run/train_livingroom.sh exp_dir
# for example:
# sh run/train_livingroom.sh outputs/livingroom
``` -->
#### Pretrained Weight
The pretrained weight of PhyScene can be downloaded [here](https://huggingface.co/datasets/yangyandan/PhyScene/blob/main/weights/livingroom/model_134000).
#### Evaluation
1) Generate scenes (save as json file) and test CKL & physical metrics.
```
sh run/test_livingroom.sh exp_dir
```

2) Load json file to generate images
```
sh run/test_livingroom_gen_image.sh exp_dir
```
3) Test SCA, KID, and FID
```
# SCA
python synthetic_vs_real_classifier.py --path_to_real_renderings data/preprocessed_data/LivingRoom/ --path_to_synthesized_renderings your/generated/image/folder
# KID and FID
python compute_fid_scores.py --path_to_real_renderings data/preprocessed_data/LivingRoom/ --path_to_synthesized_renderings your/generated/image/folder
```

## 🏡 Test Procthor Floor Plan
We also provide scripts for generating scenes from an unseen floor plan, such as room in ProTHOR.
This script generate scene layout without reliance on any dataset, which provides a lite solution for user to apply on their own furniture dataset.
See [Procthor.md](Procthor.md) for more details.

## ⏱️ Modules 
- [x] Base Model 
- [x] Preprossed datasets
- [x] Pretrained models
- [ ] Training scripts
- [ ] Tutorial.ipynb 

## 🪧 Citation
If you find our work useful in your research, please consider citing:

```
@inproceedings{yang2024physcene,
          title={PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI},
          author={Yang, Yandan and Jia, Baoxiong and Zhi, Peiyuan and Huang, Siyuan},
          booktitle={Proceedings of Conference on Computer Vision and Pattern Recognition (CVPR)},
          year={2024}
        }
```

![gif](https://github.com/YandanYang/PhyScene/blob/main/demo/gif.gif)


## 👋🏻 Acknowledgements
The code of this project is adapted from [ATISS](https://github.com/nv-tlabs/ATISS) and [DiffuScene](https://github.com/tangjiapeng/DiffuScene), we sincerely thank the authors for open-sourcing their awesome projects. We also thank Ms. Zhen Chen from BIGAI for refining the figures, and all colleagues from the BIGAI TongVerse project for fruitful discussions and help on simulation developments.
