

system_prompt = """
You are an expert in 3D scene generation and optimization. Your task is to iteratively design a scene to make it as realistic, accurate, and controllable as possible.

Given the user's prompt, carefully analyze the scene and choose the most effective method from a set of available options to improve it. Your goal is to select the method that will provide the most significant enhancement based on the current scene’s data, available resources, and constraints.
"""


methods_prompt = """
Available Methods for 3D Scene Design/Modification:
We have five methods to design or modify a 3D scene, each with different strengths and weaknesses. The goal is to select the most effective method for the current iteration based on the user's prompt and the existing scene data.

Method 1: Load real2sim indoor scene data
  Data Prior: Yes
  Supported Room Types: Living room, dining room, bedroom, bathroom, office, and classroom.
  Use Case: Ideal for generating foundational layouts for common room types.
  Strengths: Provides a ready-made, realistic layout based on real-world data. Easy to implement.
  Weaknesses: Limited room type diversity compared to other methods.

Method 2: Scene synthesis by neural network
  Data Prior: Yes
  Model: Trained on the 3D Front indoor dataset.
  Supported Room Types: Living room, bedroom, and dining room.
  Use Case: Creates a foundational layout for common room types.
  Strengths: Fast and flexible, making it ideal for rapid scene prototyping. Offers diverse configurations for living room, bedroom, and dining room.
  Weaknesses: Less accurate compared to method 1 (real2sim) and other more data-driven methods.

Method 3: Image generation + 3D reconstruction
  Data Prior: 2D image prior
  Use Case: Suitable for generating detailed partial scenes or completing specific room sections.
  How It Works: We use Stable Diffusion to generate images, then apply 3D reconstruction techniques to convert the image into a full 3D scene.
  Strengths: Excellent for adding detail or creating partial elements of a scene (e.g., completing a room corner or enriching a tabletop).
  Weaknesses: More time-consuming and complex than other methods. Not as effective for generating the entire scene or layout.

Method 4: Generated Scene Using GPT
  Data Prior: LLM prior
  Room Type Diversity: Not limited to any specific room types.
  Use Case: Recommended for room types that are not supported by methods 1 or 2 (e.g., more niche or custom room types).
  Strengths: Highly versatile and capable of generating scenes for any room type. Flexible with respect to room design and customization.
  Weaknesses: May not be as real or accurate as data-driven methods.

Method 5: Modify by rule
  Data Prior: None
  Use Case: Suitable for fine-tuning or making detailed modifications to specific objects within an existing scene.
  Capabilities: Can add, remove, or move objects. Allows for modifying object relationships (positioning, constraints, etc.).
  Strengths: Highly controllable for adjusting the scene layout and objects. Useful for precise, targeted edits.
  Weaknesses: Not suitable for creating the scene from scratch or generating a foundational layout.

Decision-Making Guide:
Based on the scene data, available methods, and specific requirements of the user’s prompt, choose the best method for the current iteration. The ideal method should balance speed, realism, controllability, and cost, depending on the specific problem you are aiming to solve.

If you need a foundational layout quickly: Methods 1 or 2 are good starting points, with Method 1 being more accurate and method 2 offering more flexibility and speed.
If you need detailed additions or partial scenes: Method 3 (image generation + 3D reconstruction) is ideal for enhancing specific areas of the scene.
For unique or complex room types: Method 4 (Generated Scene Using GPT) is best for generating a custom room when other methods fall short.
For precise object placement and layout modification: Method 5 (Modify by rule) is the most controllable and flexible, though it’s not ideal for generating scenes from scratch.
Use multiple methods over several iterations to improve the scene incrementally, starting with foundational layouts and progressing to finer details.

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
Layout: {scene_layout}.
Rendered Image from the top view: SCENE_IMAGE.
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
User’s Prompt: {user_prompt}
Current Iteration: {iter}
Previous Guide/Action:
{previous_guide}

Current Scene:
{sceneinfo_prompt}
Describe the scene based on the provided image and the current data.

Feedback from Testing:
Below is the feedback based on how the previously generated scene performed during testing. 

{feedback_examples}

Based on the feedback provided, analyze the issues or improvements identified. You can also consider the following factors:

Initial Scene Construction: If the scene is empty or lacks a foundational structure, what is the best method to generate a basic layout that fits the user's prompt (e.g., living room, office)?
Adding Furniture and Objects: If the scene feels sparse, assess whether more furniture needs to be added to the ground or corners, or if additional small objects should be placed on shelves, desks, or tables.
Detailing: Evaluate if the scene could benefit from small decorative items or functional objects placed inside or on top of furniture (e.g., books, lamps, plants).
Collision and Layout Issues: Check if there are any collisions or improper placements of objects that disrupt the flow of the room. Do any objects need repositioning or resizing for better usability or aesthetic?
User Prompt Satisfaction: Does the current scene meet the user's prompt requirements? What needs to be changed or added to align with the prompt more closely?
Realism Enhancement: What adjustments can be made to make the scene feel more realistic? Consider adding, removing, or repositioning objects to enhance visual harmony and authenticity.
Check Object: Check for any redundant or unnecessary objects that could be removed to streamline the scene.

Make your choice based on which method will achieve the highest quality and efficiency in improving the scene while keeping the user's needs and constraints in mind.

Main Goal for the Next Iteration:
Identify the most crucial improvement that should be made in this iteration based on the feedback. This should address the most pressing issue and make the biggest impact on the overall scene.

You may update the scene over multiple iterations, and you can select the most appropriate method(s) for each iteration. It’s often useful to leverage different methods in various iterations to make incremental improvements.

Recommended Approach & Method(s) for 4 Iterations:
You can consider using a sequence of methods that build on each other in multiple iterations. A sample approach could be:

Iteration 0: Use Method 1 or 2 to create a basic layout for the given scene type (if supported).
Iteration 1: Enhance the scene with Method 3 or 4, adding more details or making improvements based on the initial layout.
Iteration 2: Add more refinement and smaller details in specific areas (e.g., corners, objects on tables) using Method 3 or 4.
Iteration 3: Make fine-tuned modifications to specific objects using Method 5 (Modify by rule), focusing on precise adjustments.
Try to avoid repeating the same method or actions in multiple iterations.


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


