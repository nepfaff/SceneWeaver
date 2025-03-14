import agent_prompt as prompts
from gpt import GPT4 
from prompt_room import extract_json, dict2str
import json

iter = 1
user_prompt = "Classroom"

while(iter<=5):

    ### 1. get big object, count, and relation
    system_prompt = prompts.system_prompt.format(iter=iter,user_prompt=user_prompt)
    methods_prompt = prompts.methods_prompt
    previous_guide = None
    render_path = f"/home/yandan/workspace/infinigen/render{iter-1}.jpg"
    with open(f"/home/yandan/workspace/infinigen/layout{iter-1}.json", "r") as f:
        layout = json.load(f)
    
    sceneinfo_prompt =  prompts.sceneinfo_prompt.format(scene_layout=dict2str(layout["objects"]))
    feedback_examples = "None"

    # responses = [{
    #     "iter": 0,
    #     "Conclusions from feedback": "Since no feedback has been provided yet, I will assess the scene based on the prompt (living room) and available methods. In this initial iteration, the goal is to create a foundational layout of the living room that offers flexibility and realism.",
    #     "Thoughts": "Method 2 (scene synthesis by neural network) is an ideal choice for generating an initial layout. This method uses a model trained on the 3D Front indoor dataset, which provides reliable results for common room types, including living rooms. It allows for quick generation of the foundational layout with flexibility in terms of room configuration. This will give us a starting point that can easily be modified in later iterations, and it is more efficient compared to using real-world data in Method 1.",
    #     "Recommendation": "For this iteration, I recommend using Method 2 (scene synthesis by neural network) to generate the initial living room layout. This method is fast, flexible, and will create a diverse, realistic foundation for the living room. In subsequent iterations, we can add more details or refine specific areas using other methods like Method 3 (image generation + 3D reconstruction) or Method 5 (modify by rule). This approach ensures we have a strong, adaptable base layout while saving time for future refinements.",
    #     "Method number": 2,
    #     "Goal for this iteration": "Generate a flexible and realistic foundational layout for the living room."
    # }   ,
    # {
    #     "iter": 1,
    #     "Conclusions from feedback": "The layout appears to have a basic arrangement of furniture typical for a living room, including seating, storage, and a TV stand. However, the scene could benefit from additional details and refinement to enhance realism and functionality.",
    #     "Thoughts": "The current layout provides a good starting point but lacks detailed elements that could make the living room feel more lived-in and realistic. Method 3 (image generation + 3D reconstruction) is suitable for adding detailed elements to specific areas of the room, such as enhancing the tabletop or adding decorative items. This method allows for the generation of detailed 2D images that can be converted into 3D models, filling in the gaps in the current scene with high-quality, realistic details.",
    #     "Recommendation": "For this iteration, I recommend using Method 3 (image generation + 3D reconstruction) to add detailed elements to the living room, focusing on areas like the coffee table and the spaces around the seating area. This will help in creating a more inviting and realistic environment. Subsequent iterations can focus on refining object placements and relationships using Method 5 (modify by rule) to fine-tune the layout and ensure optimal spacing and functionality.",
    #     "Method number": 3,
    #     "Goal for this iteration": "Enhance the living room scene by adding detailed and realistic elements to specific areas, such as the coffee table and surrounding spaces."
    # },
    # {
    #     "iter": 2,
    #     "Conclusions from feedback": "Since no specific feedback was provided, the assessment of the scene must be based on the visual inspection of the image and the previous iterations' goals. The current scene shows a basic but coherent layout of a living room with essential furniture like a sofa, a coffee table, and a bookshelf. The scene has been enhanced with detailed elements such as decorations on the coffee table, which was the focus of the last iteration. However, the scene could still benefit from further refinement in terms of object relationships and spatial arrangement to improve functionality and aesthetic appeal.",
    #     "Thoughts": "The scene currently lacks a bit in terms of the realistic placement and interaction of objects. For instance, the sofa and coffee table could be better aligned to enhance the flow and usability of the space. Method 5 (modify by rule) is ideal for this purpose as it allows for precise adjustments to object positions and relationships without the need for generating new elements. This method will enable us to fine-tune the existing layout, adjust spacing, and perhaps add or remove objects to better meet typical living room functionality.",
    #     "Recommendation": "For this iteration, I recommend using Method 5 (modify by rule) to refine the placement and relationships of objects within the living room. This will include adjusting the distances between furniture like the sofa and the coffee table, possibly repositioning the bookshelf for better accessibility, and ensuring that the space feels balanced and functional. This method will help in achieving a more polished and practical layout, enhancing both the aesthetic and practical aspects of the living room.",
    #     "Method number": 5,
    #     "Goal for this iteration": "Refine object placements and relationships to enhance functionality and aesthetic appeal of the living room."
    # }]
    responses = [{
            "iter": 0,
            # "Conclusions from feedback": "Since there is no feedback provided and the scene is currently empty, the primary focus is to establish a foundational layout for the living room according to the user's prompt.",
            # "Thoughts": "Given that the scene is empty, the first step should be to create a basic yet comprehensive layout that aligns with the typical elements of a living room. Method 1 (real2sim indoor scene data) offers a data-driven approach that can provide a realistic and detailed layout based on real-world living room configurations. This method ensures that the layout is not only accurate but also practical, incorporating common living room elements such as sofas, coffee tables, and entertainment units.",
            # "Recommendation": "For this iteration, I recommend using Method 1 (real2sim indoor scene data) to generate the initial layout of the living room. This method will establish a solid and realistic foundation for the scene. In subsequent iterations, we can focus on adding more personalized details, adjusting object placements, and enhancing the scene's realism using methods like Method 3 (image generation + 3D reconstruction) or Method 5 (modify by rule). Starting with a strong, data-driven layout will provide a good basis for these enhancements.",
            "Method number": 4,
            # "Goal for this iteration": "Establish a realistic and practical foundational layout of the living room.",
            "Action": "Generate new room.",
            "Feedback": "Finish."
        },
        ]
       
    previous_guide = [dict2str(r) for r in responses]
    previous_guide = "\n".join(previous_guide)

    idea_example = prompts.idea_example
    feedback_reflections_system_payload = prompts.feedback_reflections_prompt_system.format(system_prompt=system_prompt,
                                                                                                    methods_prompt=methods_prompt)
    feedback_reflections_user_payload = prompts.feedback_reflections_prompt_user.format(iter=iter,
                                                                                    user_prompt=user_prompt,
                                                                                    previous_guide=previous_guide,
                                                                                    sceneinfo_prompt=sceneinfo_prompt,
                                                                                    feedback_examples=feedback_examples,
                                                                                    idea_example=idea_example
                                                                                    )
    gpt = GPT4()
    
    prompt_payload = gpt.get_payload_scene_image(feedback_reflections_system_payload, feedback_reflections_user_payload,render_path=render_path )
    gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    print(gpt_text_response)
    gpt_dict_response = extract_json(gpt_text_response)
    with open(f"record/iter_{iter}.json","w") as f:
        gpt_dict_response["user_prompt"] = user_prompt
        gpt_dict_response["iter"] = iter
        gpt_dict_response["system_prompt"] = system_prompt
        gpt_dict_response["previous_guide"] = previous_guide
        gpt_dict_response["sceneinfo_prompt"] = sceneinfo_prompt
        gpt_dict_response["feedback_examples"] = feedback_examples
        gpt_dict_response["idea_example"] = idea_example
        gpt_dict_response["methods_prompt"] = methods_prompt
        json.dump(gpt_dict_response,f,indent=4)

    method_this_iter = gpt_dict_response["Method number"]
    goal_this_iter = gpt_dict_response["Goal for this iteration"]

    if method_this_iter==0:
        if iter == 0:
            from method_4_GPT_iter0 import generate_scene_iter0
            results = generate_scene_iter0(user_prompt,goal_this_iter)




    feedback_reflection = "None"
    # Do we need to improve the scene in another iteration or stop at this stage?
    improvement_idea = prompts.improvement_idea_prompt.format(system_prompt=system_prompt,
                                                                methods_prompt=methods_prompt,
                                                                iter=iter,
                                                                user_prompt=user_prompt,
                                                                previous_guide=previous_guide,
                                                                sceneinfo_prompt=sceneinfo_prompt,
                                                                feedback_reflection=feedback_reflection)


    print(improvement_idea)
    # prompt_payload = gpt.get_payload(prompts.step_1_big_object_prompt_system, user_prompt)
    # gpt_text_response = gpt(payload=prompt_payload, verbose=True)
    # print(gpt_text_response)

    iter += 1