from digital_cousins.models.objaverse.retrieve import ObjectRetriever
from digital_cousins.models.objaverse.constants import OBJATHOR_ASSETS_DIR
import json
import os

def get_candidates(category,phrase="",dataset="openshape",threshold = 0.3,maxcnt = 10):
    if dataset=="holodeck":
        basedir = OBJATHOR_ASSETS_DIR
        retriever = ObjectRetriever()
        cat_lst = ["a " + category]
        for cat in cat_lst:
            objects = retriever.retrieve_object_by_cat(cat)
            candidates = [obj[0] for obj in objects if obj[1]>threshold]
            candidates_objav = []
            for candidate in candidates:
                filename = f'{basedir}/{candidate}/{candidate}.pkl.gz'
                if os.path.exists(filename):
                    candidates_objav.append(candidate)
            candidates_objav = candidates_objav[:maxcnt]
            assert len(candidates_objav)>0

        # with open("category_lst.json","w") as f:
        #     json.dump(out,f,indent=4)
    else:
        pass
    return candidates_objav


def get_candidates_all(categories,dataset="openshape",save_dir="",phrase="",threshold = 0.3,maxcnt = 10):
    if dataset=="openshape":
        from collections import Counter
        categories = ["_".join(x.split("_")[:-1]) for x in categories]
        LoadObjavCnts = dict(Counter(categories))
        with open(f"{save_dir}/objav_cnts.json", "w") as f:
            json.dump(LoadObjavCnts, f, indent=4)
        os.system(
            f'env -i bash --norc --noprofile -c "/home/yandan/workspace/digital-cousins/retrieve.sh {save_dir}" > run.log 2>&1'
        )
        with open(f"{save_dir}/objav_files.json", "r") as f:
            candidates_objav = json.load(f)
    else:
        AssertionError()

    return candidates_objav

def retrieve_objav_assets(category_cnt, name_mapping=None):
    def get_case_insensitive(dictionary, key):
        return next(
            (v for k, v in dictionary.items() if k.lower() == key.lower()), None
        )

    # retrieve objaverse
    LoadObjavCnts = dict()
    for name in category_cnt.keys():
        if name_mapping is not None and name not in name_mapping:
            name = name.lower()
        if name_mapping is None or name_mapping[name] is None:
            LoadObjavCnts[name] = get_case_insensitive(category_cnt, name)

    with open(f"{save_dir}/objav_cnts.json", "w") as f:
        json.dump(LoadObjavCnts, f, indent=4)

    # cmd = """
    # source ~/anaconda3/etc/profile.d/conda.sh
    # conda activate idesign
    # python /home/yandan/workspace/infinigen/infinigen/assets/objaverse_assets/retrieve_idesign.py > run.log 2>&1
    # """
    # subprocess.run(["bash", "-c", cmd])
    os.system(
        f'env -i bash --norc --noprofile -c "./retrieve.sh {save_dir}" > run.log 2>&1'
    )
    with open(f"{save_dir}/objav_files.json", "r") as f:
        self.LoadObjavFiles = json.load(f)
    return
