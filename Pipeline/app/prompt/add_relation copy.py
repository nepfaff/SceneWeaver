
system_prompt = """
You are an expert in 3D scene design. We design a code to manage the scene, where more relations will make the scene more tidy.
Sometimes the relation is encoded in the layout coordinate rather than represented explicitly, making it difficult to manage.

Your task is to : 
Add explicit relation to objects in the current scene according to the layout. 

For example, if the vase is placed on the table, you should add ["table_name", "ontop"] to the relation of the vase.
If the book is placed in the shelf, you should add ["shelf_name", "on"] to the relation of the shelf ("on" here means inside).
If the shelf stands with its back to the wall, you should add ["room_name", "against_wall"] to the relation of the shelf. Here wall is part of the room.
More relations can be added according to the relation list and objects' layout. 

The optional relation is: 
1.front_against: child_obj's front faces to parent_obj, and stand very close.
2.front_to_front: child_obj's  front faces to parent_obj's front, and stand very close.
3.leftright_leftright: child_obj's left or right faces to parent_obj's left or right, and stand very close. 
4.side_by_side: child_obj's side(left, right , or front) faces to parent_obj's side(left, right , or front), and stand very close. 
5.back_to_back: child_obj's back faces to parent_obj's back, and stand very close. 
6.ontop: child_obj is placed on the top of parent_obj.
7.on: child_obj is placed on the top of or inside parent_obj.
8.against_wall: child_obj's back faces to the wall of the room, and stand very close.
9.side_against_wall: child_obj's side(left, right , or front) faces to the wall of the room, and stand very close.
9.on_floor: child_obj stand on the parent_obj, which is the floor of the room.

Note child_obj is usually smaller than parent_obj, or child_obj belongs to parent_obj. 
And the child_obj can have no more than one relation with other objects.
Each parent_obj can have less than **four** child_obj on the floor. For example, a table can only have chairs as its children.

**Strict Rule – Must Check Distance:**
You must check the actual distance between objects before assigning a relation. 
If the child and parent are not **very close** (i.e., their closest bounding boxes are more than 1.0 meter apart), **you MUST NOT assign any relation** between them.
This rule overrides all “common sense” assumptions — even if a chair is facing a table, if it is not close enough, it should not be related.
This applies even to typical pairs like chairs and tables, books and shelves, beds and nightstands. Do not infer a relation based on usual arrangement — always check the actual layout.

You are working in a 3D scene environment with the following conventions:

- Right-handed coordinate system.
- The X-Y plane is the floor.
- X axis (red) points right, Y axis (green) points forward, Z axis (blue) points up.
- All asset local origins are centered in X-Y and at the bottom in Z.
- By default, assets face the +X direction.
- A rotation of [0, 0, 1.57] in Euler angles will turn the object to face +Y.
- All bounding boxes are aligned with the local frame and marked in blue with category labels.
- The front direction of objects are marked with yellow arrow.
- Coordinates in the image are marked from [0, 0] at bottom-left of the room.

"""


user_prompt = """
Here is the information you receive:
1.This is a {roomtype}. 
2.The room size is [{roomsize}] in length and width.
3.User demand for the entire scene: {user_demand}
4.This is the scene layout: {layout}
5.This is the layout of door and windows: {structure}
6.This is the image render from the top view: SCENE_IMAGE 

Please take a moment to relax and carefully look through each object and their relations.
The relation is written as a list in the "parent" key, in the format of [parent_obj's name, relation]. 
For example, ["newroom_0-0", "onfloor"] means the child_obj is on the floor of the room. Note "newroom_0-0" is not listed in the objetcs' layout.
And ["2419840_bed","ontop"] means the child_obj is on the top of "2419840_bed".
Some relations have already been added, and you need to implement the relations when the layout is similar to the relation but not recorded in "parent" explicitly.
You can take regular usage habits into account.

Before returning the final results, you need to carefully confirm that each obvious relation has been added. 

**Important: Proximity Rule Reminder**
You must strictly enforce spatial proximity when assigning relations. If two objects are more than 1.0 meters apart, DO NOT assign any relation between them, regardless of their category or orientation.
Do NOT assume a relation exists just because it is typical (e.g., 8 chairs facing a table) unless the layout confirms they are physically close enough.
You are not allowed to infer relations purely from “usual placement” — **use the layout only**.

Provide me with the newly added relation of each object in json format.
"""

example = """
For example:
{
    "4262456_Vase": {
        "parent": [["4153214_Table","ontop"]]
    },
    "1542543_Sofa": {
        "parent": [["newroom_0-0","against_wall"]]
    },
    "4254546_Cabinet": {
        "parent": [["newroom_0-0","against_wall"]]
    },
}

"""