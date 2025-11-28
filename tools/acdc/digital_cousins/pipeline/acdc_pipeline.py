"""
Top level entry point for ACDC
"""
import digital_cousins
# If you store the offline dataset elsewhere, please uncomment the following line and put the directory here
# digital_cousins.ASSET_DIR = "~/assets"

import yaml
import argparse
import os
from copy import deepcopy
from digital_cousins.models.feature_matcher import FeatureMatcher
from digital_cousins.pipeline.extraction import RealWorldExtractor
from digital_cousins.pipeline.matching import DigitalCousinMatcher
# from digital_cousins.pipeline.generation import SimulatedSceneGenerator
from digital_cousins.models.blend.render_generation import render_generation
# import omnigibson as og

class ACDC:
    """
    End-to-end pipeline for running ACDC
    """
    def __init__(self, config=None):
        """
        Args:
            config (None or str): Configuration to use when running ACDC. If None, will use default
                located at <PATH_TO_ACDC>/configs/default.yaml
        """
        # Load config if not specified
        config = f"{digital_cousins.__path__[0]}/configs/pipeline.yaml" if config is None else config
        with open(config, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.config = config

    def run(
            self,
            input_path,
            tabletype="",
            save_dir=None,
            run_step_1=True,
            run_step_2=True,
            run_step_3=True,
            step_1_output_path=None,
            step_2_output_path=None,
            gpt_api_key=None,
            gpt_version=None,
            dataset=None
    ):
        """
        Executes ACDC, running the following steps:
        1. Real World Extraction
        2. Digital Cousin Matching (per-object)
        3. Simulated Scene Generation

        Optionally skips some steps in case this run crashes mid-execution.

        Args:
            input_path (str): Absolute path to the input RGB image to use for ACDC
            save_dir (None or str): If specified save directory to use for ACDC. Otherwise, will create a directory
                called "acdc_output" in the same directory as @input_path. Note: save_dir should NOT be specified
                in the loaded config!
            run_step_1 (bool): Whether to run Step 1 or not
            run_step_2 (bool): Whether to run Step 2 or not
            run_step_3 (bool): Whether to run Step 3 or not
            step_1_output_path (None or str): If specified, the output path from Step 1 to use. This is only
                necessary if @run_step_1 is False and @run_step_2 is True
            step_2_output_path (None or str): If specified, the output path from Step 2 to use. This is only
                necessary if @run_step_2 is False and @run_step_3 is True
            gpt_api_key (None or str): If specified, the GPT API key to use (will override any value found in the
                loaded config)
            gpt_version (None or str): If specified, the GPT version to use (will override any value found in the
                loaded config)
        """
        # Copy config, and potentially overwrite GPT API key
        config = deepcopy(self.config)
        save_dir = f"{os.path.dirname(input_path)}/acdc_output"
        
        for step in ["RealWorldExtractor", "DigitalCousinMatcher", "SimulatedSceneGenerator"]:
            cur_save_dir = config["pipeline"][step]["call"].get("save_dir", None)
            assert cur_save_dir is None, f"save_dir should not be specified in {step} config! Got: {cur_save_dir}"
            config["pipeline"][step]["call"]["save_dir"] = save_dir
        if gpt_api_key is not None:
            config["pipeline"]["RealWorldExtractor"]["call"]["gpt_api_key"] = gpt_api_key
            config["pipeline"]["DigitalCousinMatcher"]["call"]["gpt_api_key"] = gpt_api_key
        if gpt_version is not None:
            config["pipeline"]["RealWorldExtractor"]["call"]["gpt_version"] = gpt_version
            config["pipeline"]["DigitalCousinMatcher"]["call"]["gpt_version"] = gpt_version

        print(f"""

{"#" * 50}
{"#" * 50}
# Starting ACDC!
{"#" * 50}
{"#" * 50}

        """)
        if run_step_1 or run_step_2:
            # We are running at least step 1 or step 2, so create FeatureMatcher
            fm = FeatureMatcher(**config["models"]["FeatureMatcher"])

            # Create RealWorldExtractor and run
            if run_step_1:
                if os.path.exists(save_dir):
                    os.system(f"rm -r {save_dir}")

                print(f"""

{"#" * 50}
{"#" * 50}
# Running ACDC: Step 1 -- Real World Extraction
{"#" * 50}
{"#" * 50}

                        """)
                step_1 = RealWorldExtractor(
                    feature_matcher=fm,
                    verbose=config["pipeline"]["verbose"],
                )
                success, step_1_output_path = step_1(
                    input_path=input_path,
                    tabletype = tabletype,
                    **config["pipeline"]["RealWorldExtractor"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 1!")

            if run_step_2:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running ACDC: Step 2 -- Digital Cousin Matching
{"#" * 50}
{"#" * 50}

                        """)
                
                step_2 = DigitalCousinMatcher(
                    feature_matcher=fm,
                    verbose=config["pipeline"]["verbose"],
                )
                success, step_2_output_path = step_2(
                    step_1_output_path=step_1_output_path,
                    dataset=dataset,
                    **config["pipeline"]["DigitalCousinMatcher"]["call"],
                )
                if not success:
                    raise ValueError("Failed ACDC Step 2!")

        if run_step_3:

                print(f"""

{"#" * 50}
{"#" * 50}
# Running ACDC: Step 3 -- Simulated Scene Generation
{"#" * 50}
{"#" * 50}

                        """)

                config_step3 = {
                    "verbose":config["pipeline"]["verbose"],
                    "step_1_output_path":step_1_output_path,
                    "step_2_output_path":step_2_output_path,
                    "tabletype" : tabletype,
                    "dataset":dataset,
                    **config["pipeline"]["SimulatedSceneGenerator"]["call"]
                    }
                config_step3_filename = "digital_cousins/configs/config_step3.yaml"
                with open(config_step3_filename, 'w') as file:
                    yaml.dump(config_step3, file, default_flow_style=False, sort_keys=False)

                render_generation(config_step3_filename)

                


def main(args):
    # Create ACDC and run
    pipeline = ACDC(config=args.config)
    pipeline.run(
        input_path=args.input_path,
        tabletype=args.tabletype,
        run_step_1=not args.skip_step_1,
        run_step_2=not args.skip_step_2,
        run_step_3=not args.skip_step_3,
        step_1_output_path=args.step_1_output_path,
        step_2_output_path=args.step_2_output_path,
        gpt_api_key=args.gpt_api_key,
        dataset=args.dataset
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default=None,
                        help="Absolute path to input RGB file to use")
    parser.add_argument("--config", type=str, default=None,
                        help="Absolute path to config file to use. If not specified, will use default.")
    parser.add_argument("--tabletype", type=str, default="tv stand",
                        help="The category of the supporter in the image, such as countertop, desk, and cabinet.")
    parser.add_argument("--gpt_api_key", type=str, default=None,
                        help="GPT API key to use. If not specified, will use value found from config file.")
    parser.add_argument("--skip_step_1", action="store_true",
                        help="If set, will skip ACDC Step 1 (Real World Extraction)")
    parser.add_argument("--skip_step_2", action="store_true",
                        help="If set, will skip ACDC Step 2 (Digital Cousin Matching)")
    parser.add_argument("--skip_step_3", action="store_true",
                        help="If set, will skip ACDC Step 3 (Simulated Scene Generation)")
    parser.add_argument("--step_1_output_path", type=str, default=None,
                        help="output path from Step 1 to use. Only necessary if --skip_step_1 is set and --skip_step_2 is not set.")
    parser.add_argument("--step_2_output_path", type=str, default=None,
                        help="output path from Step 2 to use. Only necessary if --skip_step_2 is set and --skip_step_3 is not set.")
    parser.add_argument("--dataset", type=str, default="holodeck",
                        help="Method to retrieve objaverse assets. openshape or holodeck.")

    args = parser.parse_args()

    import json
    with open("args.json","r") as f:
        j = json.load(f)
        args.input_path = j["img_filename"]
        args.tabletype = j["objtype"]

    import random
    random.seed(42)
    import numpy as np
    np.random.seed(42)

    main(args)
    
    with open("args.json","w") as f:
        
        j["success"] =  True
        j["error"] =  None
        json.dump(j,f,indent=4)

