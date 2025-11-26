<div align="center">
<img src="docs/images/sceneweaver.png" width="300"></img>
</div>

<h2 align="center">
  <b>SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent</b>
</h2>
 <div align="center" margin-bottom="6em">
  <a target="_blank" href="https://yandanyang.github.io/">Yandan Yang</a><sup>✶</sup>,
  <a target="_blank" href="https://buzz-beater.github.io/">Baoxiong Jia</a><sup>✶</sup>,
  <a target="_blank" href="https://hishujie.github.io/">Shujie Zhang</a>,
  <a target="_blank" href="https://siyuanhuang.com/">Siyuan Huang</a>

</div>
<br>
<div align="center">
    <!-- <a href="https://cvpr.thecvf.com/virtual/2023/poster/22552" target="_blank"> -->
    <a href="https://arxiv.org/abs/2509.20414" target="_blank"> 
      <img src="https://img.shields.io/badge/Paper-arXiv-green" alt="Paper arXiv"></a>
    <a href="https://scene-weaver.github.io" target="_blank">
      <img src="https://img.shields.io/badge/Page-SceneWeaver-blue" alt="Project Page"/></a>
</div>
<br>
<div style="text-align: center">
<img src="docs/images/teaser.jpg"  />
</div>


<!-- This is the official repository of [**PhyScene: Physically Interactable 3D Scene Synthesis for Embodied AI**](https://arxiv.org/abs/2211.05272). -->


For more information, please visit our [**project page**](https://scene-weaver.github.io).

## Requirements
- Linux machine
- Python 3.10
- [uv](https://docs.astral.sh/uv/)

## ⚙️ Installation & Dependencies

### Quick Start: Automated Setup (Recommended)

We provide an automated setup script that handles all installation steps:

> **Note:** The setup includes a modification to `scripts/install/interactive_blender.sh` to add `--find-links` for the Blender repository. This is necessary for installing bpy into Blender's Python. See `scripts/patches/README.md` for details.

```bash
# Minimal setup (LLM-based tools + Infinigen only)
bash scripts/setup_pipeline.sh --minimal

# Full setup (all tools and datasets)
bash scripts/setup_pipeline.sh --full

# Custom workspace directory
bash scripts/setup_pipeline.sh --workspace /path/to/workspace

# Use existing datasets (optional)
bash scripts/setup_pipeline.sh --full \
  --existing-3d-future /path/to/3D-FUTURE \
  --existing-metascenes /path/to/metascenes \
  --existing-objaverse /path/to/objaverse
```

The script will:
- ✅ Create the Python environment with uv
- ✅ Install all Python dependencies
- ✅ Download and install Blender 3.6
- ✅ Clone external tools (SD 3.5, Tabletop Digital Cousins)
- ✅ Download datasets (3D FUTURE, etc.)
- ✅ Set up configuration templates

### Post-Setup Configuration

After running the setup script, complete these manual steps:

#### 1. Configure Azure OpenAI API

Edit `Pipeline/key.txt` and add your API key:
```
your-actual-azure-openai-api-key-here
```

Edit `Pipeline/config/config.json`:
```json
{
    "llm_name": "gpt-4.1-2025-04-14",
    "llm_config": {
        "base_url": "https://YOUR_ENDPOINT.openai.azure.com/openai/deployments/YOUR_DEPLOYMENT_ID",
        "api_version": "2025-03-01-preview"
    }
}
```

#### 2. Configure Available Tools

Edit `Pipeline/app/agent/scenedesigner.py` (lines 66-79) to enable/disable tools based on what you installed:

**Minimal Setup (default):**
```python
available_tools0 = [InitGPTExecute()]  # LLM-based initialization only
available_tools1 = [                   # Basic modifiers + LLM-based addition
    AddGPTExecute(),
    UpdateLayoutExecute(),
    UpdateRotationExecute(),
    UpdateSizeExecute(),
    RemoveExecute(),
    Terminate()
]
```

**Full Setup (if you installed SD 3.5 + ACDC):**
```python
available_tools0 = [
    InitGPTExecute(),
    InitMetaSceneExecute(),  # If you have MetaScenes dataset
    InitPhySceneExecute()     # Uses sample data in data/physcene/
]
available_tools1 = [
    AddAcdcExecute(),         # SD 3.5 + Tabletop Digital Cousins
    AddGPTExecute(),
    AddCrowdExecute(),
    AddRelationExecute(),
    UpdateLayoutExecute(),
    UpdateRotationExecute(),
    UpdateSizeExecute(),
    RemoveExecute(),
    Terminate()
]
```

#### 3. Verify Environment Paths

Check the auto-generated `.env` file to ensure paths are correct:
```bash
cat .env
```

If you need to adjust paths, edit `.env` and update:
- `WORKSPACE_DIR` - Where external tools and datasets are stored
- `SD35_DIR` - Path to SD 3.5
- `ACDC_DIR` - Path to Tabletop Digital Cousins
- `FUTURE_3D_DIR` - Path to 3D FUTURE dataset
- `METASCENES_DIR` - Path to MetaScenes dataset (if available)

#### 4. Test Your Setup

Run a minimal test:
```bash
cd Pipeline
source ../.venv/bin/activate
python main.py --prompt "Design me a simple bedroom with a bed and nightstand." --cnt 1 --basedir ./test_output/
```

The output will be saved in `./test_output/` with:
- `scene_*.blend` - Blender scene files
- `layout_*.json` - Object layouts
- `render_*.jpg` - Top-down renders
- `pipeline/` - Agent logs and tool results

### Running the Pipeline

#### Mode 1: Background (Headless)
```bash
cd Pipeline
source ../.venv/bin/activate
python main.py --prompt "Design me a bedroom." --cnt 1 --basedir ./output/
```

**Arguments:**
- `--prompt`: Natural language scene description
- `--cnt`: Number of scenes to generate (default: 1)
- `--basedir`: Output directory path

#### Mode 2: Foreground (Interactive with Blender UI)

**Terminal 1** - Start Blender with socket:
```bash
cd SceneWeaver
source .venv/bin/activate  # Use the infinigen environment
python -m infinigen.launch_blender -m infinigen_examples.generate_indoors_vis \
  --save_dir debug/ -- --seed 0 --task coarse --output_folder debug/ \
  -g fast_solve.gin overhead.gin studio.gin \
  -p compose_indoors.terrain_enabled=False
```

**Terminal 2** - Run the agent:
```bash
cd Pipeline
source ../.venv/bin/activate
python main.py --prompt "Design me a bedroom." --cnt 1 --basedir ./output/ --socket True
```

You'll see the scene being generated in real-time in the Blender window.

### Troubleshooting

**Issue: "Could not find a version that satisfies the requirement bpy==3.6.0"**
- Solution: The setup script automatically configures the Blender repository. If you see this, run:
  ```bash
  uv sync
  ```

**Issue: "API key not found"**
- Solution: Make sure you added your Azure OpenAI key to `Pipeline/key.txt`

**Issue: "ModuleNotFoundError: No module named 'infinigen'"**
- Solution: Make sure you activated the environment:
  ```bash
  source .venv/bin/activate  # From the SceneWeaver root directory
  ```

**Issue: "git submodule error"**
- Solution: Re-initialize submodules:
  ```bash
  rm -rf infinigen/datagen/customgt/dependencies/glm
  git submodule update --init --recursive
  ```

**Issue: External tools not found**
- Solution: Check `.env` file and verify paths are correct. External tools should be in:
  - `~/workspace/sd3.5/`
  - `~/workspace/Tabletop-Digital-Cousins/`
  - Or the custom workspace directory you specified

**Issue: Out of memory**
- Solution: Reduce scene complexity or use fewer objects. LLM-based tools (minimal setup) use less memory than SD+ACDC.

---

### Manual Installation

If you prefer to install manually or need more control:

#### Option 1: Using uv (Recommended)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

#### Download this repo in your workspace
```bash
cd ~/workspace
git clone https://github.com/Scene-Weaver/SceneWeaver.git
cd SceneWeaver
```

#### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Set LLM api
Save your api-key of GPT in `Pipeline/key.txt`. We use AzureOpenAI here. You can modify this module to fit your own LLM api.

#### Set up the executor environment (infinigen)
```bash
# Initialize git submodules (required for building)
git submodule update --init --recursive

# Install all dependencies (uv automatically uses Python 3.10 from .python-version)
# This installs bpy from the Blender repository and all other dependencies
uv sync
```

Then, install Infinigen using one of the options below:
```bash
# Minimal installation (recommended setting for use in the Blender UI)
INFINIGEN_MINIMAL_INSTALL=True bash scripts/install/interactive_blender.sh

# Normal install
bash scripts/install/interactive_blender.sh

# Enable OpenGL GT
INFINIGEN_INSTALL_CUSTOMGT=True bash scripts/install/interactive_blender.sh
```

More details can refer to [official repo of Infinigen](https://github.com/princeton-vl/infinigen/blob/main/docs/Installation.md#installing-infinigen-as-a-blender-python-script).


## Available Tools

### Tool Status Summary

| Tool | Status | Notes |
|------|--------|-------|
| `init_gpt` | ✅ Works | Initialize scene with GPT layout generation |
| `init_metascene` | ❌ Requires external data | Needs MetaScene dataset (not included) |
| `init_physcene` | ✅ Works | Uses sample data in `data/physcene/` |
| `add_gpt` | ✅ Works | Add objects using GPT |
| `add_acdc` | ❌ Requires external systems | Needs SD 3.5 + ACDC (not included) |
| `add_crowd` | ✅ Works | Add crowded placement using GPT |
| `add_relation` | ✅ Works | Add explicit relations between objects |
| `update_layout` | ✅ Works | Update object positions |
| `update_rotation` | ✅ Works | Update object rotations |
| `update_size` | ✅ Works | Update object sizes |
| `remove_obj` | ✅ Works | Remove objects from scene |
| `terminate` | ✅ Works | End pipeline |

You can expand the framework to other tools (such as architecture, Text-2-3D) as needed.
Modify `available_tools0` and `available_tools1` in [Pipeline/app/agent/scenedesigner.py](Pipeline/app/agent/scenedesigner.py#L67) to configure which tools are available.

### Unavailable Tools - External Dependencies

These tools require external datasets/systems from the original authors that are not included in this repository:

- **`init_metascene`**: Requires MetaScene dataset at `/mnt/fillipo/yandan/metascene/`
- **`add_acdc`**: Requires [SD 3.5](https://github.com/Scene-Weaver/sd3.5) + [Tabletop Digital Cousins](https://github.com/Scene-Weaver/Tabletop-Digital-Cousins) + conda environment

### Working Tools by Category

**Initializers:**
- [x] LLM: GPT (`init_gpt`)
- [ ] Dataset: MetaScenes (requires external data)
- [x] Model: PhyScene/DiffuScene/ATISS (sample data in `data/physcene/`)

**Implementers:**
- [ ] Visual: SD + Tabletop Digital Cousin (requires external setup)
- [x] LLM: GPT (both sparse & crowded)
- [x] Rule-based relations

**Modifiers:**
- [x] Update Layout/Rotation/Size
- [x] Add Relation
- [x] Remove Objects


## 🛒 Assets      
We here support different source of assets. **You can choose any of them to fit your own requirements**. But in this project, we choose different asset according to the usage of tool.


#### MetaScenes
For tool using Dataset such as MetaScenes, we employ its assets directly, since each scene contains several assets with delicated mesh and layout information.

#### 3D FUTURE
For tool using Model such as PhyScene/DiffuScene/ATISS, we employ 3D FUTURE, since the model is trained on this dataset.
You can download 3D FUTURE in [huggingface](https://huggingface.co/datasets/yangyandan/PhyScene/blob/main/dataset/3D-FUTURE-model.zip).


#### Infinigen 
For other tools, we use [Infinigen's asset generation code](infinigen/assets/objects) to generate standard assets in common categories, such as bed, sofa, and plate. The asset will be generated in a delicated rule procedure in the scene generation process. 

#### Objaverse     
For those catrgories that are not supported by Infinigen, such as clock, laptop, and washing machine, we employ open-vocabulary Objaverse dataset.

We provide two resource & retrieve pipeline for Objaverse (OpenShape & Holodeck), you can following one/both of the two pipelines to retrieve assets. Note if you use **Tabletop Digital Cousin** tool, we recommend you to use Holodeck pipeline.

1. OpenShape
    Refer to [IDesign official repo](https://github.com/atcelen/IDesign/tree/main) and build the `idesign` conda env.
    Run the [inference code](https://github.com/atcelen/IDesign/tree/main?tab=readme-ov-file#inference) to download and build the openshape repo.
    Then run `bash  SceneWeaver/run/retrieve.sh debug/`. If success, you will get a new file named`debug/objav_files.json`.

2. Holodeck
    Refer to [Holodeck official repo](https://github.com/allenai/Holodeck?tab=readme-ov-file#data), build the conda env and then download the data.
    Then modify the `ABS_PATH_OF_HOLODECK` in `digital_cousins/models/objaverse/constants.py` to your downloaded directory.
                                          

## Usage

#### Mode 1: Run with Blender in the background
```bash
cd Pipeline
source ../.venv/bin/activate
python main.py --prompt "Design me a bedroom." --cnt 1 --basedir PATH/TO/SAVE
```
Then you can check the scene in `PATH/TO/SAVE`. The intermediate scene in each step is saved in `record_files`. You can open relative `.blend` file in blender to check the result of each step.

#### Mode 2: Run with Blender in the foreground
Interactable & convenient to check generating process.

You need to open **two** terminal.

**Terminal 1**: Run infinigen with socket to connect with blender
```bash
cd SceneWeaver
source .venv/bin/activate
python -m infinigen.launch_blender -m infinigen_examples.generate_indoors_vis --save_dir debug/ -- --seed 0 --task coarse  --output_folder debug/ -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False
```
**Terminal 2**: Run SceneWeaver to launch the agent
```bash
cd SceneWeaver/Pipeline
source ../.venv/bin/activate
python main.py --prompt "Design me a bedroom." --cnt 1 --basedir PATH/TO/SAVE --socket
```
Then you can check the scene in the `Blender` window and `PATH/TO/SAVE`

#### Generated Folder Structure
We record the intermediate info of each step of the agent and the generated scene. 
The folder structure is as follows:
```
PATH/TO/SAVE/
  Scene_Name/                         # folder name for this scene
    |-- args                          # saved args info for each iter 
      |-- args_{iter}.json
    |-- pipeline                      # saved info for agent
      |-- acdc_output                 # save folder of table top scene
      |-- {tool}_results_{iter}.json  # tool result
      |-- eval_iter_{iter}.json       # eval result
      |-- grade_iter_{iter}.json      # evaluated result (GPT score)
      |-- memory_{iter}.json          # agent memory record
      |-- metric_{iter}.json          # evaluated result (physics & GPT score)
      |-- roomtype.txt                # roomtype
      |-- trajs_{iter}.json           # overall record of previous steps

    |-- record_files                  # record files of intermediate scene
      |-- metric_{iter}.json          # evaluated result (physics)
      |-- name_map_{iter}.json        # name map between object id and blender name
      |-- scene_{iter}.blend          # saved intermediate scene
      |-- obj.blend (optional)        # save supporter for acdc
      |-- env_{iter}.pkl              # record file of infinigen
      |-- house_bbox_{iter}.pkl 
      |-- MaskTag.json
      |-- p_{iter}.pkl
      |-- solved_bbox_{iter}.pkl
      |-- solver_{iter}.pkl
      |-- state_{iter}.pkl
      |-- terrain_{iter}.pkl

    |-- record_scene
      |-- layout_{iter}.json           # object layout & room size
      |-- render_{iter}.jpg            # top-down rendered scene
      |-- render_{iter}_bbox.png       # top-down rendered 3D Mark (bbox, axies, direction, semantic label)
      |-- render_{iter}_marked.jpg     # top-down rendered scene & 3D Mark

    |-- args.json                      # args info for running infinigen 
    |-- objav_cnts.json                # objects to retrieve from objaverse
    |-- objav_files.json               # retrieved results
    |-- roominfo.json                  # room info to start building a new scene
```



## Evaluate 

```
python evaluation_ours.py
```


## Export to USD for Isaac Sim 

```
python -m infinigen.tools.export --input_folder BLENDER_FILE_FOLDER --output_folder USD_SAVE_FOLDER -f usdc -r 1024 --omniverse
```

## 🪧 Citation
If you find our work useful in your research, please consider citing:

```
@inproceedings{yang2025sceneweaver,
          title={SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent},
          author={Yang, Yandan and Jia, Baoxiong and Zhang, Shujie and Huang, Siyuan},
          booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
          year={2025}
        }
```

## 👋🏻 Acknowledgements
The code of this project is adapted from [Infinigen](https://github.com/princeton-vl/infinigen/tree/main). We sincerely thank the authors for open-sourcing their awesome projects. 
