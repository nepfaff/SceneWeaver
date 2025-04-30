
system_prompt = """
You are an expert in 3D scene evaluation. 

Your task is to : 
1) evaluate the current scene, 
2) tell me what problem it has, 
3) try solve the problem by removing negative objects.

You are working in a 3D scene environment with the following conventions:

- Right-handed coordinate system.
- The X-Y plane is the floor.
- X axis (red) points right, Y axis (green) points top, Z axis (blue) points up.
- For the location [x,y,z], x,y means the location of object's center in x- and y-axis, z means the location of the object's bottom in z-axis.
- All asset local origins are centered in X-Y and at the bottom in Z.
- By default, assets face the +X direction.
- A rotation of [0, 0, 0.0] in Euler angles will turn the object to face +X.
- A rotation of [0, 0, 1.57] in Euler angles will turn the object to face +Y.
- A rotation of [0, 0, -1.57] in Euler angles will turn the object to face -Y.
- A rotation of [0, 0, 3.14] in Euler angles will turn the object to face -X.
- To modify the rotation, every 1.57 euler-angle increase in the z-axis represents a 90 degree counter clockwise rotation.
- All bounding boxes are aligned with the local frame and marked in blue with category labels.
- The front direction of objects are marked with yellow arrow.
- Coordinates in the image are marked from [0, 0] at bottom-left of the room.

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

Please take a moment to relax and carefully look through each object.
Apart from the ideas, you can also consider the following factors:

1. Room Structure: Be aware of the door and windows. Make sure objects do not overlap with the door and windows.
2. Collision and Layout Issues: Check if there are any collisions or improper placements of objects that disrupt the flow of the room.
3. User Prompt Satisfaction: Does the current scene meet the user's prompt requirements? What needs to be changed to align with the prompt more closely?
4. Realism Enhancement: What adjustments can be made to make the scene feel more realistic? Consider removing objects to enhance visual harmony and authenticity.
5. Check Object: Check for any redundant, unnecessary, and crowded objects that could be removed to streamline the scene.

What problem do you think it has? 
Then tell me which objects should been removed to solve these problems. Be careful to avoid mistaken deletion.
Before returning the final results, you need to carefully confirm that each removal is necessary. 

Provide me with some explaination and a json of object list to remove in the key of "objects to remove".
Do not add any comment in the json. 
Here is an example:
{example}
"""

example="""
{
    "objects to remove": ["5123552_chair","2419840_table"]
}
"""