import re
import json
from typing import Dict

import random
import math
from typing import Dict, Any

def extract_json(input_string):
    # Using regex to identify the JSON structure in the string
    json_match = re.search(r'{.*}', input_string, re.DOTALL)
    if json_match:
        extracted_json = json_match.group(0)
        try:
            # Convert the extracted JSON string into a Python dictionary
            json_dict = json.loads(extracted_json)
            json_dict = check_dict(json_dict)
            return json_dict
        except json.JSONDecodeError:
            print(input_string)
            print("Error while decoding the JSON.")
            return None
    else:
        print("No valid JSON found.")
        return None
    
def check_dict(dict):
    valid = True
    attributes = ["index", "category", "size", "objects_on_top", "objects_inside"]
    for key, value in dict.items():
        if not isinstance(key, str): valid = False; break

        if not isinstance(value, Dict): valid = False; break

        for attribute in attributes:
            if attribute not in value: valid = False; break
        
        if not isinstance(value["index"], int): valid = False; break

        if not isinstance(value["category"], str): valid = False; break

        if not isinstance(value["size"], list) or len(value["size"]) != 3 or not all(isinstance(i, int) for i in value["size"]): dict[key]["size"] = None

        if not isinstance(value["objects_on_top"], list): dict[key]["objects_on_top"] = []

        if not isinstance(value["objects_inside"], list): dict[key]["objects_inside"] = []

        for name in ["objects_on_top", "objects_inside"]:
            for i, child in enumerate(value[name]):
                if not isinstance(child, Dict): valid = False; break
                
                for attribute in ["object_name", "quantity", "variance_type"]:
                    if attribute not in child: valid = False; break
                
                if not isinstance(child["object_name"], str): valid = False; break

                if not isinstance(child["quantity"], int): dict[key][name][i]["quantity"] = 1

                if not isinstance(child["variance_type"], str) or child["variance_type"] not in ["same", "varied"]: dict[key][name][i]["variance_type"] = "same"

    if not valid: return None
    else: return dict


def custom_distribution():
    while True:
        sample = random.uniform(0,1)
        probability = math.exp(-((sample-0.5)**2)/0.02)
        if random.uniform(0,1)<probability:
            return sample


def get_asset_metadata(obj_data: Dict[str, Any]):
    if "assetMetadata" in obj_data:
        return obj_data["assetMetadata"]
    elif "thor_metadata" in obj_data:
        return obj_data["thor_metadata"]["assetMetadata"]
    else:
        raise ValueError("Can not find assetMetadata in obj_data")


def get_bbox_dims(obj_data: Dict[str, Any]):
    am = get_asset_metadata(obj_data)

    bbox_info = am["boundingBox"]

    if "x" in bbox_info:
        return bbox_info

    if "size" in bbox_info:
        return bbox_info["size"]

    mins = bbox_info["min"]
    maxs = bbox_info["max"]

    return {k: maxs[k] - mins[k] for k in ["x", "y", "z"]}

if __name__ == "__main__":
    s = 'for the living room design, my strategy is to create a cozy and inviting space by combining functional and decorative elements. for the multi-seat sofa, i will add various throw pillows, blankets, and a small side table on top. inside the sofa, i will place some magazines and a small tray for snacks. the lounge chair will have a side table with a decorative plant on top and a cozy throw inside. the corner side tables will be adorned with a table lamp, picture frames, and a small vase of flowers. the cabinet will showcase some decorative items and storage boxes inside. the console table will have a decorative bowl with keys on top and some books and a decorative basket inside. lastly, the coffee table will feature a decorative tray with coasters and a stack of magazines on top, while inside, there will be a decorative box, coasters, and a plant.\n\n{\n    "4":{\n        "index": 4,\n        "category": "multi_seat_sofa",\n        "size": [233, 99, 107],\n        "objects_on_top": [\n            {"object_name": "throw pillow", "quantity": 4, "variance_type": "varied"},\n            {"object_name": "blanket", "quantity": 1, "variance_type": "same"},\n            {"object_name": "side table", "quantity": 1, "variance_type": "same"}\n        ],\n        "objects_inside": [\n            {"object_name": "magazine", "quantity": 3, "variance_type": "varied"},\n            {"object_name": "snack tray", "quantity": 1, "variance_type": "same"}\n        ]\n    },\n    "5":{\n        "index": 5,\n        "category": "lounge_chair",\n        "size": [63, 91, 88],\n        "objects_on_top": [\n            {"object_name": "side table", "quantity": 1, "variance_type": "same"},\n            {"object_name": "decorative plant", "quantity": 1, "variance_type": "same"}\n        ],\n        "objects_inside": [\n            {"object_name": "throw", "quantity": 1, "variance_type": "same"}\n        ]\n    },\n    "6":{\n        "index": 6,\n        "category": "corner_side_table",\n        "size": [54, 53, 47],\n        "objects_on_top": [\n            {"object_name": "table lamp", "quantity": 1, "variance_type": "same"},\n            {"object_name": "picture frame", "quantity": 2, "variance_type": "varied"},\n            {"object_name": "vase of flowers", "quantity": 1, "variance_type": "same"}\n        ],\n        "objects_inside": []\n    },\n    "8":{\n        "index": 8,\n        "category": "cabinet",\n        "size": [80, 49, 121],\n        "objects_on_top": [\n            {"object_name": "decorative items", "quantity": 3, "variance_type": "varied"}\n        ],\n        "objects_inside": [\n            {"object_name": "storage box", "quantity": 2, "variance_type": "varied"}\n        ]\n    },\n    "9":{\n        "index": 9,\n        "category": "console_table",\n        "size": [159, 44, 81],\n        "objects_on_top": [\n            {"object_name": "decorative bowl with keys", "quantity": 1, "variance_type": "same"}\n        ],\n        "objects_inside": [\n            {"object_name": "book", "quantity": 3, "variance_type": "varied"},\n            {"object_name": "decorative basket", "quantity": 1, "variance_type": "same"}\n        ]\n    },\n    "10":{\n        "index": 10,\n        "category": "coffee_table",\n        "size": [155, 52, 49],\n        "objects_on_top": [\n            {"object_name": "decorative tray with coasters", "quantity": 1, "variance_type": "same"},\n            {"object_name": "stack of magazines", "quantity": 1, "variance_type": "same"}\n        ],\n        "objects_inside": [\n            {"object_name": "decorative box", "quantity": 1, "variance_type": "same"},\n            {"object_name": "coasters", "quantity": 4, "variance_type": "varied"},\n            {"object_name": "plant", "quantity": 1, "variance_type": "same"}\n        ]\n    }\n}'
    extract_json(s)
