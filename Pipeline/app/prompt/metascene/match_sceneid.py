system_prompt = """
I want to design a scene based on the user's demand. 
I have already retrieved multiple scenes from the dataset. 
And now I need assistance in selecting the most similar scene that best matches the user's demand.
Help me identify the scene that aligns most closely with the user's demand.

You will reveive the following information:

1.User's demand of the scene
2.Roomtype: roomtype
3.Ideas about the scene
4.Category Counts for Candidate Scenes: Each key represents a scene_id, and the corresponding value is the count of objects per category within that scene.

You should return:
The scene id that aligns most closely with the user's demand.

Example Input:
1.User's demand: An office
2.Roomtype: bedroom
3.Ideas: Add basic bedroom
4.Category counts of candidate scenes: 
{
    "scene0001_00": {
        "a clear bottle": 1,
        "furnace": 1,
        "rack": 1,
        "table": 1,
        "sofa": 1
    },
    "scene0002_00": {
        "shelf": 2,
        "bucket": 3,
        "a yellow book": 1,
        "board": 1,
        "chair": 1
    }
}

    
Expected Output (only return scene_id):
scene0002_00

"""
user_prompt = """
Input:
User's demand: {user_demand}
Roomtype: {roomtype}
Ideas: {ideas}
Category counts of candidate scenes: {category_counts}

Your response:
"""