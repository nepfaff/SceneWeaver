import json
import re
from functools import reduce

import init_gpt_prompt as prompts
from gpt import GPT4
from utils import extract_json, dict2str, lst2str
import sys


def generate_scene_iter0(user_demand,ideas,roomtype):

    gpt = GPT4()

    results = dict()

    ### 1. get big object, count, and relation
    user_prompt = prompts.step_1_big_object_prompt_user.format(demand=user_demand,ideas=ideas,roomtype=roomtype)
    prompt_payload = gpt.get_payload(prompts.step_1_big_object_prompt_system, user_prompt)
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)

    # gpt_text_response ='{\n    "Roomtype": "Living Room",\n    "Category list of big object": {\n        "sofa": 2,\n        "armchair": 2,\n        "coffee table": 1,\n        "TV stand": 1,\n        "large shelf": 1,\n        "side table": 2,\n        "floor lamp": 2\n    },\n    "Object against the wall": ["TV stand", "large shelf"],\n    "Relation between big objects": [\n        ["armchair", "coffee table", "front_against"],\n        ["sofa", "coffee table", "front_against"],\n        ["side table", "sofa", "side_by_side"],\n        ["floor lamp", "armchair", "side_by_side"]\n    ]\n}'

    # response = [i for i in gpt_text_response.split("\n") if len(i)>0]
    gpt_dict_response = extract_json(gpt_text_response)
    roomsize = gpt_dict_response["Room size"]
    big_category_dict = gpt_dict_response["Category list of big object"]
    big_category_list = list(big_category_dict.keys())
    category_against_wall = gpt_dict_response["Object against the wall"]
    relation_big_object = gpt_dict_response["Relation between big objects"]


    # # Category list of big objects: [1 checkout counter, 5 bookshelves, 2 reading tables, 8 chairs]
    # # Object against the wall: [bookshelves]
    # # Relation between big objects: [chair, reading table, front_against]


    ##### 5  generate position big
    big_category_dict_str = dict2str(big_category_dict)
    category_against_wall_str = lst2str(category_against_wall)
    relation_big_object_str = lst2str(relation_big_object)
    roomsize_str = lst2str(roomsize)

    user_prompt = prompts.step_5_position_prompt_user.format(
        big_category_dict=big_category_dict_str,
        category_against_wall=category_against_wall_str,
        relation_big_object=relation_big_object_str,
        demand=user_demand,
        roomsize=roomsize_str,
    )
    prompt_payload = gpt.get_payload(prompts.step_5_position_prompt_system, user_prompt)
    success = False
    iter = 0
    while not success and iter < 5:
        iter += 1
        gpt_text_response = gpt(payload=prompt_payload, verbose=True)
        print(gpt_text_response)

        # gpt_text_response = '{\n    "Roomtype": "Bookstore",\n    "list of given category names": ["sofa", "armchair", "coffee table", "TV stand", "large shelf", "side table", "floor lamp", "remote control", "book", "magazine", "decorative bowl", "photo frame", "vase", "candle", "coaster", "plant"],\n    "Mapping results": {\n        "sofa": "seating.SofaFactory",\n        "armchair": "seating.ArmChairFactory",\n        "coffee table": "tables.CoffeeTableFactory",\n        "TV stand": "shelves.TVStandFactory",\n        "large shelf": "shelves.LargeShelfFactory",\n        "side table": "tables.SideTableFactory",\n        "floor lamp": "lamp.FloorLampFactory",\n        "remote control": null,\n        "book": "table_decorations.BookStackFactory",\n        "magazine": null,\n        "decorative bowl": "tableware.BowlFactory",\n        "photo frame": null,\n        "vase": "table_decorations.VaseFactory",\n        "candle": null,\n        "coaster": null,\n        "plant": "tableware.PlantContainerFactory"\n    }\n}'
        try:
            gpt_dict_response = extract_json(
                gpt_text_response.replace("'", '"').replace("None", "null")
            )
            success = True
        except:
            success = False
    Placement_big = gpt_dict_response["Placement"]

    small_category_list = []
    relation_small_object = []
    Placement_small = []
    
    # #### 2. get small object and relation
    # s = lst2str(big_category_list)
    # # s = "[bookshelves, reading tables, chairs, checkout counter]"
    # user_prompt = prompts.step_2_small_object_prompt_user.format(
    #     big_category_list=s, demand=user_demand
    # )
    # prompt_payload = gpt.get_payload(prompts.step_2_small_object_prompt_system, user_prompt)
    # gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    # print(gpt_text_response)
    # # gpt_text_response = '{\n    "Roomtype": "Living Room",\n    "List of big furniture": ["sofa", "armchair", "coffee table", "TV stand", "large shelf", "side table", "floor lamp"],\n    "List of small furniture": ["remote control", "book", "magazine", "decorative bowl", "photo frame", "vase", "candle", "coaster", "plant"],\n    "Relation": [\n        ["remote control", "coffee table", "ontop", 2],\n        ["book", "large shelf", "on", 5],\n        ["magazine", "coffee table", "ontop", 3],\n        ["decorative bowl", "coffee table", "ontop", 1],\n        ["photo frame", "side table", "ontop", 2],\n        ["vase", "side table", "ontop", 1],\n        ["candle", "large shelf", "on", 3],\n        ["coaster", "coffee table", "ontop", 4],\n        ["plant", "side table", "ontop", 1]\n    ]\n}'

    # gpt_dict_response = extract_json(gpt_text_response)
    # small_category_list = gpt_dict_response["List of small furniture"]
    # relation_small_object = gpt_dict_response["Relation"]

    # # List of small furniture: ["books", "lamps", "magazines", "decorative items", "cash register"]
    # # Relation: [
    # #     ["books", "bookshelves", "on", 50],
    # #     ["lamps", "reading tables", "ontop", 4],
    # #     ["magazines", "reading tables", "ontop", 6],
    # #     ["decorative items", "checkout counter", "ontop", 3],
    # #     ["cash register", "checkout counter", "ontop", 1]
    # # ]


    ################## load results
    # with open("results.json", "r") as f:
    #     results = json.load(f)

    # roomtype = results["roomtype"]
    # roomsize = results["roomsize"]
    # big_category_dict = results["big_category_dict"]
    # category_against_wall = results["category_against_wall"]
    # relation_big_object = results["relation_big_object"]
    # small_category_list = results["small_category_list"]
    # relation_small_object = results["relation_small_object"]
    # name_mapping = results["name_mapping"]
    # gpt_text_response = results["gpt_text_response"]
    # Placement_big = results["Placement"]
    # # Placement_small = results["Placement_small"]

    # big_category_dict_str = dict2str(big_category_dict)
    # category_against_wall_str = lst2str(category_against_wall)
    # relation_big_object_str = lst2str(relation_big_object)
    # roomsize_str = lst2str(roomsize)
    # big_category_list = list(big_category_dict.keys())



    # ##### 6  generate position small
    # small_category_list_str = lst2str(small_category_list)
    # relation_small_object_str = lst2str(relation_small_object)
    # placement_big = dict2str(Placement_big)
    # user_prompt = prompts.step_6_small_position_prompt_user.format(
    #     big_category_dict=big_category_dict_str,
    #     demand=user_demand,
    #     roomsize=roomsize_str,
    #     small_category_lst=small_category_list_str,
    #     relation_small_big=relation_small_object_str,
    #     placement_big=placement_big
    # )
    # prompt_payload = gpt.get_payload(prompts.step_6_small_position_prompt_system, user_prompt)
    # success = False
    # iter = 0
    # while not success and iter < 5:
    #     iter += 1
    #     gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    #     print(gpt_text_response)

    #     # gpt_text_response = '{\n    "Roomtype": "Bookstore",\n    "list of given category names": ["sofa", "armchair", "coffee table", "TV stand", "large shelf", "side table", "floor lamp", "remote control", "book", "magazine", "decorative bowl", "photo frame", "vase", "candle", "coaster", "plant"],\n    "Mapping results": {\n        "sofa": "seating.SofaFactory",\n        "armchair": "seating.ArmChairFactory",\n        "coffee table": "tables.CoffeeTableFactory",\n        "TV stand": "shelves.TVStandFactory",\n        "large shelf": "shelves.LargeShelfFactory",\n        "side table": "tables.SideTableFactory",\n        "floor lamp": "lamp.FloorLampFactory",\n        "remote control": null,\n        "book": "table_decorations.BookStackFactory",\n        "magazine": null,\n        "decorative bowl": "tableware.BowlFactory",\n        "photo frame": null,\n        "vase": "table_decorations.VaseFactory",\n        "candle": null,\n        "coaster": null,\n        "plant": "tableware.PlantContainerFactory"\n    }\n}'
    #     try:
    #         gpt_dict_response = extract_json(
    #             gpt_text_response.replace("'", '"').replace("None", "null")
    #         )
    #         success = True
    #     except:
    #         success = False
    # Placement_small = gpt_dict_response["Placement of small furniture"]


    # #### 3. get object class name in infinigen
    category_list = big_category_list + small_category_list
    s = lst2str(category_list)

    user_prompt = prompts.step_3_class_name_prompt_user.format(
        category_list=s, demand=user_demand
    )
    system_prompt = prompts.step_3_class_name_prompt_system

    prompt_payload = gpt.get_payload(system_prompt, user_prompt)
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)

    # gpt_text_response = '{\n    "Roomtype": "Bookstore",\n    "list of given category names": ["sofa", "armchair", "coffee table", "TV stand", "large shelf", "side table", "floor lamp", "remote control", "book", "magazine", "decorative bowl", "photo frame", "vase", "candle", "coaster", "plant"],\n    "Mapping results": {\n        "sofa": "seating.SofaFactory",\n        "armchair": "seating.ArmChairFactory",\n        "coffee table": "tables.CoffeeTableFactory",\n        "TV stand": "shelves.TVStandFactory",\n        "large shelf": "shelves.LargeShelfFactory",\n        "side table": "tables.SideTableFactory",\n        "floor lamp": "lamp.FloorLampFactory",\n        "remote control": null,\n        "book": "table_decorations.BookStackFactory",\n        "magazine": null,\n        "decorative bowl": "tableware.BowlFactory",\n        "photo frame": null,\n        "vase": "table_decorations.VaseFactory",\n        "candle": null,\n        "coaster": null,\n        "plant": "tableware.PlantContainerFactory"\n    }\n}'

    gpt_dict_response = extract_json(
        gpt_text_response.replace("'", '"').replace("None", "null")
    )
    name_mapping = gpt_dict_response["Mapping results"]
    # Mapping results: {
    #     "books": 'table_decorations.BookFactory',
    #     "bookmarks": None,
    #     "lamps": 'lamp.DeskLampFactory',
    #     "reading glasses": None,
    #     "cash register": None,
    #     "decorative items": None
    # }

    # #### 4. generate rule code


    # def get_rule_prompt(
    #     big_category_list,
    #     small_category_list,
    #     relation_big_object,
    #     relation_small_object,
    #     big_category_dict,
    #     user_demand,
    #     name_mapping,
    # ):
    #     var_dict = dict()
    #     for name in big_category_list:
    #         if name_mapping[name] is None:
    #             big_category_dict.pop(name)
    #             continue
    #         var_name = name.replace(" ", "_") + "_obj"
    #         info = var_name + " = "
    #         if name in category_against_wall:
    #             info += "wallfurn"
    #         else:
    #             info += "furniture"
    #         info += "[" + name_mapping[name] + "]"
    #         var_dict[name] = {"var_name": var_name, "info": info}

    #     for name in small_category_list:
    #         if name_mapping[name] is None:
    #             continue
    #         var_name = name.replace(" ", "_") + "_obj"
    #         info = var_name + " = "
    #         info += "obj"
    #         info += "[" + name_mapping[name] + "]"
    #         var_dict[name] = {"var_name": var_name, "info": info}

    #     rel_small_big_object_name = set()
    #     for rel in relation_small_object.copy():
    #         obj1name, obj2name, relation, cnt = rel
    #         if obj1name not in var_dict or obj2name not in var_dict:
    #             relation_small_object.remove(rel)
    #             continue
    #         else:
    #             rel_small_big_object_name.add(obj2name)

    #     for rel in relation_big_object.copy():
    #         obj1name, obj2name, relation = rel
    #         if obj1name not in var_dict or obj2name not in var_dict:
    #             relation_big_object.remove(rel)
    #             continue
    #         relation = "cu." + relation
    #         if "related" in var_dict[obj1name]["info"]:
    #             import pdb

    #             pdb.set_trace()
    #         if obj1name in rel_small_big_object_name:
    #             continue
    #         var_name = var_dict[obj2name]["var_name"]
    #         var_dict[obj1name]["info"] += f".related_to({var_name},{relation})"

    #     vars_definition_1 = [
    #         var["info"] for var in var_dict.values() if "related" not in var["info"]
    #     ]
    #     vars_definition_2 = [
    #         var["info"] for var in var_dict.values() if "related" in var["info"]
    #     ]
    #     vars_definition = "\n".join(vars_definition_1 + vars_definition_2)

    #     big_category_cnt_str = json.dumps(big_category_dict)
    #     relation_big_object_str = lst2str(relation_big_object)
    #     relation_small_object_str = lst2str(relation_small_object)

    #     user_prompt = prompts.step_4_rule_prompt_user.format(
    #         big_category_cnt=big_category_cnt_str,
    #         relation_big_object=relation_big_object_str,
    #         relation_small_object=relation_small_object_str,
    #         vars_definition=vars_definition,
    #         demand=user_demand,
    #     )
    #     print(user_prompt)
    #     return user_prompt


    # user_prompt = get_rule_prompt(
    #     big_category_list,
    #     small_category_list,
    #     relation_big_object,
    #     relation_small_object,
    #     big_category_dict,
    #     user_demand,
    #     name_mapping,
    # )
    # prompt_payload = gpt.get_payload(prompts.step_4_rule_prompt_system, user_prompt)
    # gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    # gpt_text_response = gpt_text_response.replace("{{", "{").replace("}}", "}")
    # print(gpt_text_response)


    results["user_demand"] = user_demand
    results["roomsize"] = roomsize
    results["big_category_dict"] = big_category_dict
    results["category_against_wall"] = category_against_wall
    results["relation_big_object"] = relation_big_object
    results["small_category_list"] = small_category_list
    results["relation_small_object"] = relation_small_object
    results["name_mapping"] = name_mapping
    results["gpt_text_response"] = gpt_text_response
    results["Placement_big"] = Placement_big
    results["Placement_small"] = Placement_small

    json_name = "/home/yandan/workspace/infinigen/Pipeline/record/init_gpt_results.json"
    with open(json_name, "w") as f:
        json.dump(results, f, indent=4)

    return json_name

if __name__ == "__main__":
    user_demand = "Bedroom"
