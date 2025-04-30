

from app.agent.scenedesigner import SceneDesigner
from app.logger import logger
import os

def main():
    agent = SceneDesigner()
    try:
        prompt = "Design me a bedroom."
        # prompt = "Design me a bedroom."
        save_dir = "/mnt/fillipo/yandan/scenesage/record_scene/manus/"+prompt.replace(" ","_").replace(".","").replace(",","_").replace("[","").replace("]","")
        if not os.path.exists(save_dir):
            os.system(f"mkdir {save_dir}")
            os.system(f"mkdir {save_dir}/pipeline")
        os.environ["save_dir"] = save_dir
        os.environ["UserDemand"] = prompt
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.warning("Processing your request...")
        agent.run(prompt)
        logger.info("Request processing completed.")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted.")


if __name__ == "__main__":
    main()
