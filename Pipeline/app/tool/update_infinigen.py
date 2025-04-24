import json
import os
import subprocess


def update_infinigen(
    action, iter, json_name, description=None, inplace=False, invisible=False
):
    j = {
        "iter": iter,
        "action": action,
        "json_name": json_name,
        #  "roomsize": roomsize,
        "description": description,
        "inplace": inplace,
        "success":False
    }

    argsfile = f"/home/yandan/workspace/infinigen/args.json"
    with open(argsfile, "w") as f:
        json.dump(j, f, indent=4)

    # if invisible:
    if True:
        
        cmd = """
        source ~/anaconda3/etc/profile.d/conda.sh
        conda activate infinigen_python
        cd /home/yandan/workspace/infinigen
        python -m infinigen_examples.generate_indoors --seed 0 --task coarse --output_folder outputs/indoors/coarse_expand_whole_nobedframe -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False compose_indoors.invisible_room_ceilings_enabled=True > /home/yandan/workspace/infinigen/Pipeline/run.log 2>&1
        """
        subprocess.run(["bash", "-c", cmd])
    else:
        os.system("bash -i /home/yandan/workspace/infinigen/run.sh > run.log 2>&1")

    with open(argsfile, "r") as f:
        j = json.load(f)

    assert j["success"]
    print("infinigen success")
    return j["success"]
 