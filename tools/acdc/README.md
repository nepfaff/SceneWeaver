[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)

# TableTop Digital Cousins

This repository is forked from [**ACDC**](https://github.com/cremebrule/digital-cousins), which is designed to generate fully interactive scenes from a single RGB image in a completely automated fashion.

Different from the original codebase, we 
- Remove the need of OmniGibson and robomimic. 
- Apply Blender for visualization.
- Modify the code to focus on generating a tabletop layout given a single RGB and a supporter (table, desk, cabinet, etc.).
- Replace the asset dataset from categorized BEHAVIOR dataset to open-vocabulary Objaverse dataset.

<div align="center">
<img src="examples/images/tabletop.png"></img>
</div>

## Requirements
- Linux machine
- Conda
- NVIDIA RTX-enabled GPU (recommended 24+ GB VRAM) + CUDA (12.1+)


## Getting Started


### Download

Clone this repo:

```bash
git clone git@github.com:YandanYang/Tabletop-Digital-Cousins.git
cd Tabletop-Digital-Cousins
```


### Installation

#### Step-by-Step
1. Create a new conda environment to be used for this repo and activate the repo:
    ```bash
    conda create -y -n acdc python=3.10
    conda activate acdc
    ```

2. Install ACDC
    ```bash
    conda install conda-build
    pip install torch torchvision xformers --index-url https://download.pytorch.org/whl/cu121
    pip install -r requirements.txt
    pip install -e .
    ```

3. Install the following key dependencies used in our pipeline. **NOTE**: Make sure to install in the exact following order:

    - Make sure we're in dependencies directory
   
        ```bash
        mkdir -p deps && cd deps
        ```

    - [dinov2](https://github.com/facebookresearch/dinov2)
   
        ```bash
        git clone https://github.com/facebookresearch/dinov2.git && cd dinov2
        conda-develop . && cd ..      # Note: Do NOT run 'pip install -r requirements.txt'!!
        ```
    
    - [segment-anything-2](https://github.com/facebookresearch/segment-anything-2)
   
        ```bash
        git clone https://github.com/facebookresearch/segment-anything-2.git && cd segment-anything-2
        pip install -e . && cd ..
        ```
    
    - [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

        Switch CUDA Version to cuda-12.3
        ```
        #Option 1: Update Symlink​
        sudo ln -s /PATH/TO/cuda-12.3 /PATH/TO/cuda

        #Option 2: Use update-alternatives (Debian/Ubuntu)​​
        sudo update-alternatives --config cuda  
        ```

        ```bash
        export CUDA_HOME=/PATH/TO/cuda-12.3   # Make sure to set this!
        git clone https://github.com/IDEA-Research/GroundingDINO.git && cd GroundingDINO
        pip install --no-build-isolation -e . && cd ..
        ```

    - [PerspectiveFields](https://github.com/jinlinyi/PerspectiveFields)
   
        ```bash
        git clone https://github.com/jinlinyi/PerspectiveFields.git && cd PerspectiveFields
        pip install -e . && cd ..
        ```

    - [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)
   
        ```bash
        git clone https://github.com/DepthAnything/Depth-Anything-V2.git && cd Depth-Anything-V2
        pip install -r requirements.txt
        conda-develop . && cd ..
        ```
      
    - [CLIP](https://github.com/openai/CLIP)
   
        ```bash
        pip install git+https://github.com/openai/CLIP.git
        ```
      
    - [faiss-gpu](https://github.com/facebookresearch/faiss/tree/main)
   
        ```bash
        conda install -c pytorch -c nvidia faiss-gpu=1.8.0
        ```

    - [Blender](https://www.blender.org/)
    
        Download blender. Here we use blender-4.2.0-linux-x64. Then modify the `PATH_TO_BLENDER` in `digital_cousins/models/blend/launch_blender.py` to your blender path.
#### Download checkpoint
```
mkdir -p checkpoints
cd checkpoints
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Large/resolve/main/depth_anything_v2_metric_hypersim_vitl.pth
cd ..
```

### Assets
Here we use open-vocabulary Objaverse as the retrieve dataset.

We provide two resource & retrieve pipeline for Objaverse (OpenShape & Holodeck), you can following one/both of the two pipelines to retrieve assets. If you need to convert the generated tabletop scene into [SceneWeaver](https://github.com/Scene-Weaver/SceneWeaver), we recommend you to choose Holodeck retrieving pipeline.


1. OpenShape

    Refer to [IDesign official repo](https://github.com/atcelen/IDesign/tree/main) and build the `idesign` conda env.

    Run the [inference code](https://github.com/atcelen/IDesign/tree/main?tab=readme-ov-file#inference) to download and build the openshape repo.
    
    Then run `bash  retrieve.sh debug/`. If success, you will get a new file named`debug/objav_files.json`. 

2. Holodeck
    
    Refer to [Holodeck official repo](https://github.com/allenai/Holodeck?tab=readme-ov-file#data), build the conda env and then download the data.
    Then modify the `ABS_PATH_OF_HOLODECK` in `digital_cousins/models/objaverse/constants.py` to your downloaded directory.

#### api-key for LLM model
Save your `api-key` of GPT in `key.txt`. We use AzureOpenAI here, you can modify this module to fit your own LLM api.


### Testing

To validate that the entire installation process completed successfully, please run our set of unit tests:

```bash
cd Tabletop-Digital-Cousins
python tests/test_models.py 
```

## Usage

### ACDC Pipeline
Usage is straightforward, simply run our ACDC pipeline on any input image you'd like via our entrypoint:
```sh
#Generate table top with a single RGB image and a given supporter mesh save in obj.blend.
python digital_cousins/pipeline/acdc.py --input_path <INPUT_IMG_PATH> --tabletype <Supporter Type> [--dataset <"holodeck" or "openshape">] [--config <CONFIG>]

```
- `--input_path` specifies the path to the input RGB image ot use
- `--tabletype` The category of the supporter in the image, such as countertop, desk, and cabinet.
- `--dataset` Method to retrieve objaverse assets. openshape or holodeck.
- `--config` (optional) specifies the path to the config to use. If not set, will use the default config at [`acdc/configs/default.yaml`](https://github.com/cremebrule/acdc/blob/main/acdc/configs/default.yaml)

By default, this will generate all outputs to a directory named `acdc_outputs` in the same directory as `<INPUT_IMG_PATH>`.

We include complex input images published in our work under `examples/images`.

<!-- To load the result in an user-interactable way, simply run:
```sh
python digital_cousins/scripts/load_scene.py --scene_info_path <SCENE_OUTPUT_JSON_FILE>
```
The user can use keyboard and mouse commands to interact with the scene.
 -->
