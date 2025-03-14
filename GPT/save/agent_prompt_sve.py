# # <system prompt>
# system_prompt = """
# You are an expert in 3D scene generation and optimization. 
# You need to design the scene in several iterations to make it real, accurate, and controllable.
# Given the user's prompt, you need to analyze the scene and selecting the best method out of multiple options to improve it. 
# Your goal is to choose the method that offers the most significant improvement based on the current scene data, resources, and constraints, according to the prompt requirments. 
# Provide a brief explanation for your choice.

# Here are some aspect you may pay attention to improve the scene:
# 1. If the scene is empty, how to build a basic scene to fit the user's prompt?
# 2. If the scene is sparse, do we need to add more furniture on the ground or in the corner?
# 3. Do we need to add some small objects on the top or inside some furnitures, such as shelf and desk?
# 4. Does the objects have collisions or inproper layout? Do we need to modify the position of some objects?
# 5. Does the scene satisfy the user's prompt? What need to be changed?
# 6. How to make the room more real? Change(add/delete/move) any object?
# You can add other aspects to remind yourself if you want.

# """

system_prompt = """
You are an expert in 3D scene generation and optimization. Your task is to iteratively design a scene to make it as realistic, accurate, and controllable as possible.

Given the user's prompt, carefully analyze the scene and choose the most effective method from a set of available options to improve it. Your goal is to select the method that will provide the most significant enhancement based on the current scene’s data, available resources, and constraints.

Aspects to consider for improving the scene:

Initial Scene Construction: If the scene is empty or lacks a foundational structure, what is the best method to generate a basic layout that fits the user's prompt (e.g., living room, office)?
Adding Furniture and Objects: If the scene feels sparse, assess whether more furniture needs to be added to the ground or corners, or if additional small objects should be placed on shelves, desks, or tables.
Detailing: Evaluate if the scene could benefit from small decorative items or functional objects placed inside or on top of furniture (e.g., books, lamps, plants).
Collision and Layout Issues: Check if there are any collisions or improper placements of objects that disrupt the flow of the room. Do any objects need repositioning or resizing for better usability or aesthetic?
User Prompt Satisfaction: Does the current scene meet the user's prompt requirements? What needs to be changed or added to align with the prompt more closely?
Realism Enhancement: What adjustments can be made to make the scene feel more realistic? Consider adding, removing, or repositioning objects to enhance visual harmony and authenticity.
Other considerations you may find helpful:

Evaluate the lighting and shadows to ensure they complement the scene and enhance realism.
Assess the textures and materials of objects and surfaces to make sure they fit the intended atmosphere (e.g., natural wood textures for a cozy living room).
Check for any redundant or unnecessary objects that could be removed to streamline the scene.
Make your choice based on which method will achieve the highest quality and efficiency in improving the scene while keeping the user's needs and constraints in mind.

"""


methods_prompt = """
We have 5 different resources/methods to design/modify the scene:

(1) Load real2sim indoor scene data. 
It has data prior.
The roomtype includes living room, dining room, bedroom, bathroom, office, and classroom.
It provides an easy-to-implement foundational layout for common room types.

(2) Scene synthesis by neural network. 
It has data prior.
The model is trained with 3D Front indoor dataset. 
The roomtype includes living room, bedroom and dining room.
It provides an easy-to-implement foundational layout for common room types.

(3) Image generation + 3D reconstruction. 
It has 2D image prior.
We use stable diffusion to generate image and use reconstruction method to convert image to 3D scene. 
It is recommended when we need to generate a more detailed partial scene, like complete a corner of room or enrich the tabletop.
-> code : given prompt, gen layout

(4) Generated Scene Using GPT. 
It has LLM prior.
The roomtype is not limited to specific types. 
It is recommended if the roomtype can not been provided by method 1 and 2.

(5) Modify by rule. 
It has no prior of scene. Only recommend for detail modify on specific object.
It can 1) add/move/delete object to change object existence and position 
and 2) add/remove/modify the relation constraints between objects to improve layout. 

Attribute comparison (higher is better):
controllability: 4>3>5>2>1
real: 3=1=4=2>5
accurate: 5>2>1>3=4
layout diversity: 4>3>2>1>5
concise: 5>2>3>5>1
detailed: 3>4>1>2>5
field of view: 1=4>2>3>5
roomtype diversity: 3=4=5>1>2
cost (lower is better): 3>4>1>5>2
speed: 2>1>5>4>3
"""


previous_guide = """
In iterations xxx, you 
    thoughts:
    idea: 
    resource:
    action:
    result/feedback/evaluation:
"""

# sceneinfo_prompt = """
# The current layout of the scene is {scene_layout}.
# The image rendered from the top view is {scene_image}.
# The evaluated score is {eval_score}.
# "None" means the scene is empty.
# """

sceneinfo_prompt = """
The current layout of the scene is {scene_layout}.
The image rendered from the top view is SCENE_IMAGE.
"None" means the scene is empty.
"""

feedback_reflections_prompt = """
You are an experienced layout designer to design a 3D scene. 
Your goal is to match the given open-vocabulary category name with the standard category name.


You will receive:
1. The roomtype you need to design.
2. A list of given open-vocabulary category names.

You need to return a dict including:
1. The mapping of given category name with the most similar standard category name. 

*** Important ***
The standard category list: ['appliances.BeverageFridgeFactory', 'appliances.DishwasherFactory', 'appliances.MicrowaveFactory', 'appliances.OvenFactory', 'appliances.MonitorFactory', 'appliances.TVFactory', 'bathroom.BathroomSinkFactory', 'bathroom.StandingSinkFactory', 'bathroom.BathtubFactory', 'bathroom.HardwareFactory', 'bathroom.ToiletFactory', 'decor.AquariumTankFactory', 'elements.DoorCasingFactory', 'elements.GlassPanelDoorFactory', 'elements.LiteDoorFactory', 'elements.LouverDoorFactory', 'elements.PanelDoorFactory', 'elements.NatureShelfTrinketsFactory', 'elements.PillarFactory', 'elements.RugFactory', 'elements.CantileverStaircaseFactory', 'elements.CurvedStaircaseFactory', 'elements.LShapedStaircaseFactory', 'elements.SpiralStaircaseFactory', 'elements.StraightStaircaseFactory', 'elements.UShapedStaircaseFactory', 'elements.PalletFactory', 'elements.RackFactory', 'lamp.CeilingClassicLampFactory', 'lamp.CeilingLightFactory', 'lamp.DeskLampFactory', 'lamp.FloorLampFactory', 'lamp.LampFactory', 'seating.BedFactory', 'seating.BedFrameFactory', 'seating.BarChairFactory', 'seating.ChairFactory', 'seating.OfficeChairFactory', 'seating.MattressFactory', 'seating.PillowFactory', 'seating.ArmChairFactory', 'seating.SofaFactory', 'shelves.CellShelfFactory', 'shelves.TVStandFactory', 'shelves.CountertopFactory', 'shelves.KitchenCabinetFactory', 'shelves.KitchenIslandFactory', 'shelves.KitchenSpaceFactory', 'shelves.LargeShelfFactory', 'shelves.SimpleBookcaseFactory', 'shelves.SidetableDeskFactory', 'shelves.SimpleDeskFactory', 'shelves.SingleCabinetFactory', 'shelves.TriangleShelfFactory', 'table_decorations.BookColumnFactory', 'table_decorations.BookStackFactory', 'table_decorations.SinkFactory', 'table_decorations.TapFactory', 'table_decorations.VaseFactory', 'tables.TableCocktailFactory', 'tables.CoffeeTableFactory', 'tables.SideTableFactory', 'tables.TableDiningFactory', 'tables.TableTopFactory', 'tableware.BottleFactory', 'tableware.BowlFactory', 'tableware.CanFactory', 'tableware.ChopsticksFactory', 'tableware.CupFactory', 'tableware.FoodBagFactory', 'tableware.FoodBoxFactory', 'tableware.ForkFactory', 'tableware.SpatulaFactory', 'tableware.FruitContainerFactory', 'tableware.JarFactory', 'tableware.KnifeFactory', 'tableware.LidFactory', 'tableware.PanFactory', 'tableware.LargePlantContainerFactory', 'tableware.PlantContainerFactory', 'tableware.PlateFactory', 'tableware.PotFactory', 'tableware.SpoonFactory', 'tableware.WineglassFactory', 'wall_decorations.BalloonFactory', 'wall_decorations.RangeHoodFactory', 'wall_decorations.MirrorFactory', 'wall_decorations.WallArtFactory', 'wall_decorations.WallShelfFactory']
You can only use category name from the standard list. If no standard category is matched, return null.

Here is the example: 
{
    "Roomtype": "{roomtype}"
    "list of given category names": ["bed", "nightstand", "lamp", "wardrobe"]
    "Mapping results": {"bed": "seating.BedFactory","nightstand": "shelves.SingleCabinetFactory","lamp": "lamp.DeskLampFactory", "wardrobe": null}
}
"""

feedback_reflections_prompt_system = """
{system_prompt}
{methods_prompt}
"""

feedback_reflections_prompt_user = """

The user's prompt is: {user_prompt}.

Now we are in the {iter} iteration.

Previously you generated the following guide/action to improve the scene.
{previous_guide}

{sceneinfo_prompt}
Describe the scene from the given image.

Below is some feedback on how the scene you generated performed when we tested it.
Your goal is to focus on improving the generated scene based on the feedback provided.
{feedback_examples}

Based on the feedback given and the scene you generated previously, what are some conclusions you can draw from the feedback? 
Make sure to cite the specific examples in the feedback to justify your analysis.

And what is the most important improvement that you can make to the scene layout in the next iteration that you think will have the most impact? 
Note you can update the scene in multiple iterations. 
You can randomly choose a method if more than one method is suitable for this iteration. 
A recommend choice is to take advantages of more methods in different iterations.
For this iteration, you may find the most convenient method to solve the main problem. And the rest problem can be put into next iterations.

Here is an example of methods order in 4 iterations, which is flexible: 
iter 0: You can first use method 1 or 2 to load a basic layout of given scene type if they can provide, 
iter 1: and then complete the scene using image/LLM prior with method 3 or 4. 
iter 2: Then using method 3 or 4 to add more details in the corner or on the table.
iter 3: Fiinally using method 5 to modify details for specific objects.
Try not to use the same method and action in different iterations.


Be as specific and concrete as possible, and write them out in a json format.
For example: 
{idea_example}

Your response:
"""
idea_example="""

{
"Conclusions from feedback": "Since there is no feedback provided, I will make an assessment based on the scene's prompt (bedroom) and the available resources and methods. At this stage, we're still in the first iteration, and the main goal is to create a foundational layout for the bedroom.",
"Thoughts": "A solid and realistic starting point for the bedroom layout is crucial, so I should select a method that provides the most accurate and real representation of a bedroom. For this, Method 1 (real2sim indoor scene data) seems like the best option. This method can offer an accurate, data-driven layout of a typical bedroom, which will give us a solid base to work with.The other methods, such as Method 2 (scene synthesis by neural network), are also viable for providing a foundational layout, but Method 1 is a stronger choice due to its direct real-world data and higher realness.",
"Recommendation": "For this iteration, I recommend using Method 1 (real2sim indoor scene data) to generate the initial bedroom layout. This will provide a reliable, realistic foundation for the scene, and in subsequent iterations, we can add more specific details or improve the layout using other methods. For example, we can add decorations or adjust the positioning of objects in the room in later iterations using Method 5 (modify by rule) or Method 3 (image generation + 3D reconstruction)."This approach ensures the scene starts with a solid and accurate layout, leaving room for refinements in future iterations.",
"Method number": 1,
"Goal for this iteration": "Create a realistic, foundational layout of a bedroom."
}

{
  "Conclusions from feedback": "Since there is no feedback provided, I will assess the scene based on the prompt (living room) and the available resources and methods. In the first iteration, the goal is to create a foundational layout of the living room.",
  "Thoughts": "While Method 1 (real2sim indoor scene data) is a good choice, Method 2 (scene synthesis by neural network) can also provide a solid foundational layout. Although Method 1 is based on real-world data, Method 2 is quicker and may offer more flexibility with room layouts, especially if we want to explore different configurations for the living room. The model trained on the 3D Front indoor dataset provides reliable results for common room types, and this could allow for faster prototyping and adjustments.",
  "Recommendation": "For this iteration, I recommend using Method 2 (scene synthesis by neural network) to generate the initial living room layout. This method will allow us to quickly generate a realistic starting point for the room, providing a diverse and flexible foundation. In subsequent iterations, we can enhance the layout with additional details or refine object placements using other methods like Method 3 (image generation + 3D reconstruction) or Method 5 (modify by rule). This approach offers a balance of speed and flexibility while ensuring an accurate foundational layout.",
  "Method number": 2,
  "Goal for this iteration": "Create a flexible and diverse foundational layout of the living room."
}
"""

# Do we need to improve the scene in another iteration or stop at this stage?
improvement_idea_prompt = """
{system_prompt}

{methods_prompt}

Now we are in the {iter} iteration.
The user's prompt is: {user_prompt}.

Previously you generated the following guide/action to improve the scene.
{previous_guide}

{sceneinfo_prompt}

{feedback_reflection}

Based on the function, feedback, and conclusions you drew, what is the most important improvement that
you can make to the scene layout that you think will have the most impact? Choose a method out of the 5 methods and describe the goal. 
Be as specific and concrete as possible, and write them out in the following format:
Thoughts: <your thoughts here>
Idea : <your idea here> 
"""
# Here’s an example of what this might look like for the improvement idea:
# • Thoughts: I should consider the number of cards left in the deck when evaluating
# the value of a state.
# • Idea : I should add a term to the value function that penalizes states where there
# are fewer cards left in the deck.

# If need another iteration, show me your:
#     thoughts:
#     idea 1: 
#     idea 2: 
#     ...
#     resource: 1 out of 5
#     action prompt:

# -> code: save improvement ideas, do action, evaluate. Go to next iteration


