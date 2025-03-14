import pickle
from infinigen.core.util import blender as butil
from infinigen.core.constraints.example_solver.room import decorate as room_dec
from infinigen.assets.materials import invisible_to_camera
import bpy
from infinigen_examples.util.generate_indoors_util import (
    apply_greedy_restriction,
    create_outdoor_backdrop,
    hide_other_rooms,
    place_cam_overhead,
    restrict_solving,
)
from infinigen_examples.util.visible import invisible_others, visible_others
from infinigen.core import execute_tasks, init, placement, surface, tagging
import os
import types
import importlib
import dill
import trimesh
import trimesh.parent

def export_layout(state,solver,save_dir):
    import json
    results = dict()
    results["objects"] = dict()
    results["roomsize"] = [solver.dimensions[0],solver.dimensions[1]]
    for objkey,objinfo in state.objs.items():
        if objkey.startswith("window") or objkey.startswith("entrance") or objkey.startswith("newroom_0-0"):
            continue
        results["objects"][objkey] = dict()
        results["objects"][objkey]["location"] = [round(a, 2) for a in list(objinfo.obj.location)]
        results["objects"][objkey]["rotation"] = [round(a, 2) for a in list(objinfo.obj.rotation_euler)]
        results["objects"][objkey]["size"] = [round(a, 2) for a in list(objinfo.obj.dimensions)]

    with open(save_dir,"w") as f:
        json.dump(results,f,indent=4)
        
def render_scene(p,solved_bbox,camera_rigs,state,filename="debug.jpg"):
    
    rooms_meshed = butil.get_collection("placeholders:room_meshes")
    rooms_split = room_dec.split_rooms(list(rooms_meshed.objects))

    def invisible_room_ceilings():
        rooms_split["exterior"].hide_viewport = True
        rooms_split["exterior"].hide_render = True
        rooms_split["ceiling"].hide_render = True
        invisible_to_camera.apply(list(rooms_split["ceiling"].objects))
        invisible_to_camera.apply(
            [o for o in bpy.data.objects if "CeilingLight" in o.name]
        )

    p.run_stage("invisible_room_ceilings", invisible_room_ceilings, use_chance=False)

    
    p.run_stage(
        "overhead_cam",
        place_cam_overhead,
        cam=camera_rigs[0],
        bbox=solved_bbox,
        use_chance=False,
    )

    # camera_rigs[0].rotation_euler = [0,0,1.57]
    bpy.context.scene.camera = camera_rigs[0]

    invisible_others(hide_placeholder=True)
    bpy.context.scene.render.resolution_x = 1920
    bpy.context.scene.render.resolution_y = 1080
    bpy.context.scene.render.filepath = os.path.join(filename)
    bpy.context.scene.render.image_settings.file_format='JPEG'
    bpy.ops.render.render(write_still=True)
    visible_others()

    modified_output_path = bpy.path.abspath("render_8_coord.jpg")
    world_to_image(filename, modified_output_path)

    bpy.context.scene.camera = None
    return

def world_to_image(image_path, output_path):
    
    import bpy_extras
    from PIL import Image, ImageDraw, ImageFont
    import os
    from mathutils import Vector

    def calc_point(x,y,z=0):
        world_coords = Vector([x,y,z])
        # Convert world coordinates to camera view space (normalized)
        co_2d = bpy_extras.object_utils.world_to_camera_view(scene, cam, world_coords)
        
        # Convert normalized coordinates to image pixel coordinates
        pixel_x = int(co_2d.x * res_x)
        pixel_y = int((1 - co_2d.y) * res_y)  # Flip Y-axis (Blender's origin is bottom-left)
        print(f"3D World Coords (0,0,0) : {world_coords}")
        print(f"Projected 2D Image Coords: ({pixel_x}, {pixel_y})")

        draw.ellipse(
            [(pixel_x - dot_size, pixel_y - dot_size), (pixel_x + dot_size, pixel_y + dot_size)],
            fill="red", outline="red"
        )

        # Draw the text label next to the point
        draw.text((pixel_x + 10, pixel_y - 10), f"({x}, {y})", fill="red", font=font)
        
        return

    # Get the scene and camera
    scene = bpy.context.scene
    cam = scene.camera.children[1]

    # Get render resolution and aspect ratio
    render = scene.render
    res_x = render.resolution_x * render.pixel_aspect_x
    res_y = render.resolution_y * render.pixel_aspect_y

    # Load the rendered image using PIL
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Try to load a font, otherwise use default
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default(30)

    # Draw a red dot at the calculated 2D coordinate
    dot_size = 5

    # for point in [(0,0),(10,12)]:
    #     calc_point(point[0],point[1])
    for x in range(0,11,2):
        for y in range(0,13,2):
            calc_point(x,y)
   
    # Save the modified image
    image.save(output_path)
    print(f"Image with marked point saved at {output_path}")



# def save_record(state,solver,stages,consgraph,iter=0):
def save_record(state,solver,terrain,house_bbox,solved_bbox,iter):
    # state.trimesh_scene = None
    for name in state.trimesh_scene.geometry.keys():
        state.trimesh_scene.geometry[name].fcl_obj = None
        state.trimesh_scene.geometry[name].col_obj = None

    state.bvh_cache = None

    for obj_name in state.objs.keys():
        #blender obj
        state.objs[obj_name].obj = state.objs[obj_name].obj.name
        #material
        try:
            material = generator.material_params
            params = generator.params
            for mat in material.keys():
                material[mat] = material[mat].name
                params[mat] = material[mat]
        except:
            pass

        #generator
        generator = state.objs[obj_name].generator
        if generator is not None:
            for attr in dir(generator):
                m = getattr(generator,attr)
                if isinstance(m, types.ModuleType):
                    setattr(generator, attr, m.__name__)
    
    with open(f"record_files/state_{iter}.pkl", "wb") as file:
        dill.dump(state, file)

    tagging.tag_system.save_tag()

    with open(f"record_files/solver_{iter}.pkl", "wb") as file:
        dill.dump(solver, file)

    # with open(f"record_files/stages_{iter}.pkl", "wb") as file:
    #     pickle.dump(stages, file)

    # with open(f"record_files/consgraph_{iter}.pkl", "wb") as file:
    #     pickle.dump(consgraph, file)

    # with open(f"record_files/limits_{iter}.pkl", "wb") as file:
    #     pickle.dump(limits, file)

    with open(f"record_files/terrain_{iter}.pkl", "wb") as file:
        pickle.dump(terrain, file)

    # with open(f"record_files/solved_rooms_{iter}.pkl", "wb") as file:
    #     pickle.dump(solved_rooms, file)

    with open(f"record_files/house_bbox_{iter}.pkl", "wb") as file:
        pickle.dump(house_bbox, file)

    with open(f"record_files/solved_bbox_{iter}.pkl", "wb") as file:
        pickle.dump(solved_bbox, file)
    
    # with open(f"record_files/camera_rigs_{iter}.pkl", "wb") as file:
    #     pickle.dump(camera_rigs, file)

    save_path = f"record_files/scene_{iter}.blend"
    bpy.ops.wm.save_as_mainfile(filepath=save_path)  

    env_file = f"record_files/env_{iter}.pkl"
    with open(env_file, "wb") as f:
        pickle.dump(dict(os.environ), f)

    for obj_name in state.objs.keys():
        state.objs[obj_name].obj = bpy.data.objects.get(state.objs[obj_name].obj)

    return 

def load_record(iter):
    with open(f"record_files/solver_{iter}.pkl", "rb") as file:
        solver = dill.load(file)

    # with open(f"record_files/stages_{iter}.pkl", "wb") as file:
    #     stages = pickle.load(file)

    # with open(f"record_files/consgraph_{iter}.pkl", "wb") as file:
    #     consgraph = pickle.load(file)

    # with open(f"record_files/limits_{iter}.pkl", "wb") as file:
    #     limits = pickle.load(file)

    # with open(f"record_files/p_{iter}.pkl", "wb") as file:
    #     p = pickle.load(file)
    
    with open(f"record_files/terrain_{iter}.pkl", "rb") as file:
        terrain = pickle.load(file)


    # with open(f"record_files/solved_rooms_{iter}.pkl", "wb") as file:
    #     solved_rooms = pickle.load(file)

    with open(f"record_files/house_bbox_{iter}.pkl", "rb") as file:
        house_bbox = pickle.load(file)

    with open(f"record_files/solved_bbox_{iter}.pkl", "rb") as file:
        solved_bbox = pickle.load(file)

    # with open(f"record_files/camera_rigs_{iter}.pkl", "wb") as file:
    #     camera_rigs = pickle.load(file)

    tagging.tag_system.load_tag()

    save_path = f"record_files/scene_{iter}.blend"
    bpy.ops.wm.open_mainfile(filepath=save_path)

    with open(f"record_files/state_{iter}.pkl", "rb") as file:
        state = dill.load(file)
    for obj_name in state.objs.keys():
        #blender obj
        state.objs[obj_name].obj = bpy.data.objects.get(state.objs[obj_name].obj)

        #generator
        generator = state.objs[obj_name].generator
        if generator is not None:
            for attr in dir(generator):
                if attr=="__module__":
                    continue
                module_name = getattr(generator,attr)
                try: 
                    m = importlib.import_module(module_name)
                    setattr(generator, attr, m)
                except:
                    pass
                
        #material
        try:
            material = generator.material_params
            params = generator.params
            for mat in material.keys():
                m = bpy.data.materials.get(material[mat])
                material[mat] = m
                params[mat] = m
        except:
            pass

    # state.__post_init__()


    solver.state = state

    with open(f"record_files/env_{iter}.pkl", "rb") as f:
        env_vars = pickle.load(f)
    json_name = os.getenv("JSON_RESULTS")
    os.environ.update(env_vars)
    os.environ["JSON_RESULTS"] = json_name

    return state,solver,terrain,house_bbox,solved_bbox