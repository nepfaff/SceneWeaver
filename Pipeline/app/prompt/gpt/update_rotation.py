
system_prompt = """
You are given a list of objects in a 3D room scene, including their positions, rotations, size, and types (e.g., bed, desk, chair). The scenen is rendered from the top-down view.
Your task is to identify incorrect or suboptimal rotations and provide corrected rotations that improve the room’s usability and organization.

For each object, check for rotation issues such as:
1. Facing away from functional targets. e.g., a chair not facing a desk, a TV facing a wall.
2. Backwards placement. e.g., bed headboard against the open space instead of the wall.
3. Not aligned with room geometry. e.g., objects diagonally rotated in a square room with no justification.
4. Obstructed orientation. e.g., a chair facing a wall or window with no clearance. Nothing should face the wall. 
5. Inconsistent alignment with other furniture. e.g., dining chairs misaligned around a table.

For each object with a rotation issue:
Briefly describe the problem.
Suggest a corrected rotation (as angle or quaternion).
Justify the correction based on spatial logic or common use cases.

You **MUST** modify objects' rotation to **fix the problem as much as possible**.
You can change the rotation of the objects. If necessary, you can also change their location together.
For objects that remain unchanged, you must keep their original layout in the response rather than omit it. 
**You can not add any new object.**

You are working in a 3D scene environment with the following conventions:

- Right-handed coordinate system.
- The X-Y plane is the floor.
- X axis (red) points right, Y axis (green) points top, Z axis (blue) points up.
- For the location [x,y,z], x,y means the location of object's center in x- and y-axis, z means the location of the object's bottom in z-axis.
- All asset local origins are centered in X-Y and at the bottom in Z.
- All bounding boxes are aligned with the local frame and marked in blue with category labels.
- Coordinates in the image are marked from [0, 0] at bottom-left of the room.

- By default, assets face the +X direction.Y.
- A rotation of [0, 0, 0.0] in Euler angles will turn the object to face +X.
- A rotation of [0, 0, 1.57] in Euler angles will turn the object to face +Y.
- A rotation of [0, 0, -1.57] in Euler angles will turn the object to face -Y.
- A rotation of [0, 0, 3.14] in Euler angles will turn the object to face -X.
- To modify the rotation, every 1.57 euler-angle increase in the z-axis represents a 90 degree counter clockwise rotation.
- The **front direction** of objects in current scene are marked with **yellow arrow**.

"""

user_prompt = """
Here is the information you receive:
1.This is a {roomtype}. 
2.The room size is [{roomsize}] in length and width.
3.User demand for the entire scene: {user_demand}
4.Ideas for this step (only for reference): {ideas}
5.This is the scene layout: {layout}
6.This is the layout of door and windows: {structure}
7.This is the image render from the top view: SCENE_IMAGE 

Please take a moment to relax and carefully look through each object's rotation.
The euler rotation angle is recorded in the scene layout. And each object's front direction is marked in yellow arrow in the image.
What rotation problem do you think it has? 
Then tell me how to solve these problems.

Before returning the final results, you need to carefully confirm that each issue has been resolved. 
If not, update the layout (rotation and location) until each problem is resolved.

Provide me with some explaination and the new layout of each object in json format.
Do not add any comment in the json. For example:
False:
"rotation": [0.0, 0.0, 3.14],  // Adjusted to avoid overlap
True:
"rotation": [0.0, 0.0, 3.14],

"""