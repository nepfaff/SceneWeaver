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
- ✅ Download datasets (3D FUTURE, etc.)
- ✅ Set up configuration templates

### Setting Up ACDC Tool (Optional)

The ACDC tool (SD 3.5 + Tabletop Digital Cousins) is now **integrated into SceneWeaver** in the `tools/` directory. It requires a **separate virtual environment** due to conflicting dependencies (bpy 3.6.0, specific torch version).

```bash
# Set up ACDC and SD 3.5 tools (creates separate venvs in tools/)
bash scripts/setup/setup_tools.sh

# Or set up only one tool
bash scripts/setup/setup_tools.sh --acdc-only
bash scripts/setup/setup_tools.sh --sd35-only
```

**Important:** The tools use separate venvs:
- `tools/acdc/.venv` - ACDC dependencies (bpy, open3d, etc.)
- `tools/sd3.5/.venv` - SD 3.5 dependencies (torch, transformers, etc.)

This is because ACDC requires `bpy==3.6.0` which conflicts with SceneWeaver's main environment.

#### SD 3.5 Model Download (Gated)

SD 3.5 models require accepting a license on HuggingFace:

1. Visit: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
2. Click "Agree and access repository"
3. Login to HuggingFace CLI: `huggingface-cli login`
4. Run the setup script again or manually download:
   ```bash
   cd tools/sd3.5
   source .venv/bin/activate
   python download_models.py --model medium
   ```

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

Edit `Pipeline/app/agent/scenedesigner.py` (lines 69-79) to enable/disable tools based on what you installed.


**Minimal Setup (recommended for quick testing):**
```python
available_tools0 = ToolCollection(
    InitGPTExecute()  # LLM-based initialization only
)
available_tools1 = ToolCollection(
    AddGPTExecute(),
    UpdateLayoutExecute(),
    UpdateRotationExecute(),
    UpdateSizeExecute(),
    RemoveExecute(),
    Terminate()
)
```

**Full Setup (with SD 3.5 + ACDC and all datasets):**
```python
available_tools0 = ToolCollection(
    InitGPTExecute(),
    InitMetaSceneExecute(),  # If you have MetaScenes dataset
    InitPhySceneExecute()     # Uses sample data in data/physcene/
)
available_tools1 = ToolCollection(
    AddAcdcExecute(),         # SD 3.5 + Tabletop Digital Cousins
    AddGPTExecute(),
    AddCrowdExecute(),
    AddRelationExecute(),
    UpdateLayoutExecute(),
    UpdateRotationExecute(),
    UpdateSizeExecute(),
    RemoveExecute(),
    Terminate()
)
```

#### 3. Verify Environment Paths

Check the auto-generated `.env` file to ensure paths are correct:
```bash
cat .env
```

If you need to adjust paths, edit `.env` and update:
- `WORKSPACE_DIR` - Where datasets are stored
- `FUTURE_3D_DIR` - Path to 3D FUTURE dataset
- `METASCENES_DIR` - Path to MetaScenes dataset (if available)
- `HOLODECK_DIR` - Path to Holodeck data (for ACDC object retrieval)

**Note:** ACDC and SD 3.5 are now integrated in `tools/` and don't require separate path configuration.

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

#### Batch Processing (Multiple Scenes)

To generate multiple scenes at once, use the batch script:

```bash
cd Pipeline
chmod +x run_batch.sh
./run_batch.sh
```

The script includes 5 example prompts. To customize:

1. Open `Pipeline/run_batch.sh` in a text editor
2. Edit the `prompts` array at the top of the file
3. Add, remove, or modify prompts as needed
4. Run the script

Example custom prompts:
```bash
prompts=(
  "A cozy home office with a large desk and ergonomic chair."
  "A modern kitchen with an island and bar stools."
  "A minimalist bathroom with a freestanding tub."
)
```

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

**Issue: ACDC tool not working**
- Solution: ACDC and SD 3.5 are now integrated in `tools/`. Run the setup script:
  ```bash
  bash scripts/setup/setup_tools.sh
  ```
  This creates separate venvs in `tools/acdc/.venv` and `tools/sd3.5/.venv`.

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
| `add_acdc` | ✅ Works (with setup) | Run `bash scripts/setup/setup_tools.sh` |
| `add_crowd` | ✅ Works | Add crowded placement using GPT |
| `add_relation` | ✅ Works | Add explicit relations between objects |
| `update_layout` | ✅ Works | Update object positions |
| `update_rotation` | ✅ Works | Update object rotations |
| `update_size` | ✅ Works | Update object sizes |
| `remove_obj` | ✅ Works | Remove objects from scene |
| `terminate` | ✅ Works | End pipeline |

You can expand the framework to other tools (such as architecture, Text-2-3D) as needed.
Modify `available_tools0` and `available_tools1` in [Pipeline/app/agent/scenedesigner.py](Pipeline/app/agent/scenedesigner.py#L67) to configure which tools are available.

### Tools Requiring Additional Setup

Some tools require additional setup beyond the base installation:

- **`init_metascene`**: Requires MetaScene dataset (not included in this repository)
- **`add_acdc`**: **Now integrated!** Run `bash scripts/setup/setup_tools.sh` to set up SD 3.5 and ACDC with their separate virtual environments

### Working Tools by Category

**Initializers:**
- [x] LLM: GPT (`init_gpt`)
- [ ] Dataset: MetaScenes (requires external data)
- [x] Model: PhyScene/DiffuScene/ATISS (sample data in `data/physcene/`)

**Implementers:**
- [x] Visual: SD 3.5 + ACDC (integrated in `tools/`, run setup script)
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

SceneWeaver includes evaluation tools that measure both physics metrics (collisions, out-of-bounds objects) and GPT-based quality scores (realism, functionality, layout, completion).

### Automatic Metrics Generation

Physics metrics are automatically computed during scene generation. When you run the pipeline, `metric_*.json` files are saved in `record_files/` for each iteration. This uses the original evaluation code in `infinigen_examples/steps/evaluate.py` for exact reproducibility.

### Computing Metrics for Existing Scenes

If you need to compute metrics for scenes that don't have them (e.g., older scenes), you can use the standalone script:

```bash
# Compute metrics for a single scene
./blender/blender --background PATH/TO/scene_14.blend \
  --python infinigen_examples/compute_metrics.py \
  -- --output PATH/TO/record_files/metric_14.json

# Example:
./blender/blender --background \
  "Pipeline/output/Design_me_a_bedroom_0/record_files/scene_5.blend" \
  --python infinigen_examples/compute_metrics.py \
  -- --output "Pipeline/output/Design_me_a_bedroom_0/record_files/metric_5.json"
```

The script outputs:
- `Nobj`: Total number of objects
- `OOB`: Objects outside room bounds
- `BBL`: Number of collision pairs (Bounding Box overlap/collision)

### Step 2: Run Evaluation Script

After computing physics metrics, you can run the evaluation script:

```bash
cd Pipeline
source ../.venv/bin/activate

# Evaluate a single scene (displays physics metrics)
python evaluation_ours.py --scene_path PATH/TO/SCENE_FOLDER

# Evaluate with GPT scoring (requires OpenAI API key)
python evaluation_ours.py --scene_path PATH/TO/SCENE_FOLDER --gpt_eval

# Example:
python evaluation_ours.py --scene_path ./output/Design_me_a_bedroom_0

# Batch evaluation (original mode - evaluates multiple scenes)
python evaluation_ours.py
```

### Evaluation Metrics

| Category | Metric | Description |
|----------|--------|-------------|
| Physics | `Nobj` | Total object count |
| Physics | `OOB` | Objects outside room bounds (lower is better) |
| Physics | `BBL` | Collision pairs (lower is better) |
| GPT | `realism` | How realistic the scene appears (0-10) |
| GPT | `functionality` | How well it supports intended activities (0-10) |
| GPT | `layout` | Logical furniture arrangement (0-10) |
| GPT | `completion` | How complete/finished the room feels (0-10) |


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
