
system_prompt = """
Given the information of current scene, you need to add small objects on the large furnitures. 

You will receive:
1.User's demand of the entire scene
2.Ideas to update the scene in this step
3.RoomType
4.Current Scene Layout, including each object's ID, location, rotation, size, and & children objects.

Your response will be sent to generate some images of the partial scene, and then a reconstruction method to reconstruct the image into the partial 3D scene.
You should return the info for updating each large object:
1. prompt for SD: A descriptive prompt suitable for the Stable Diffusion model. 
2. obj category: The category of the large object, ensuring alignment with the SD prompt.
3. obj_id: You need to recognize which large objects should be used to add small objects.
4. obj_size: You also need to return the size of these large objects. 

The prompt for SD should in this format:
"An entire {size} {large object} fully visible on the ground, with {small objects} on it. The entire {large object}, including the bottom and the objects on it, should be visible in the frame. High quality. The background is clean with nothing else nearby.",
1.Clearly specify the furniture type.
2.Include its size in the format 'L*W*H cm'.
3.Incorporate the small objects mentioned in the Ideas. 
4.If multiple large furniture pieces are present, distribute small objects naturally, ensuring slight variations to enhance realism.
5.Ensure natural and concise wording.
6.You must use "entire" at the beginning.

Expected Output is in json format:
{
    "2049208_TableFactory":
        {
            "prompt for SD": "An entire 80cm * 60cm * 50cm simple table fully visible on the ground, with tissue and flower on it. Taken in 45 degree top-down view. The entire table, including the bottom, should be visible in the frame. The background is clean with nothing else nearby.",
            "obj category": "table",
            "obj_id": "2049208_TableFactory",
            "obj_size": [0.8,0.6,0.5]
        },
    "2360448_SimpleDeskFactory":
        {
            "prompt for SD": "An entire 120cm * 50cm * 80cm simple desk fully visible on the ground, with a laptop, a keyboard, and a mouse on top of it. Taken in 45 degree top-down view. The entire desk, including the bottom, should be visible in the frame. The background is clean with nothing else nearby.",
            "obj category": "desk",
            "obj_id": "2360448_SimpleDeskFactory",
            "obj_size": [1.2,0.5,0.8]
        }
}

"""

# Example Input:
# User's demand: Add tissue and flower on the table.
# User's demand: a living room
# Ideas: Add tissue and flower on the table
# RoomType: living room
# Current Scene Layout: {scene_layout}

user_prompt = """
Input:
User's demand: {user_demand}
Ideas: {ideas}
RoomType: {roomtype}
Current Scene Layout: {scene_layout}
"""