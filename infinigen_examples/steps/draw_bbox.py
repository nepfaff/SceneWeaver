import bpy
from mathutils import Matrix, Vector
import mathutils


def add_rotated_bbox_wireframe(obj, cat_name):
    if obj.type != "MESH":
        return

    # Create a cube with unit size at origin
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, 0))
    bbox_cube = bpy.context.active_object
    bbox_cube.name = f"RotatedBBOX_{obj.name}"
    print(bbox_cube.name)

    # Scale it to match the object's local bounding box
    local_bbox = [Vector(corner) for corner in obj.bound_box]
    min_corner = Vector((min(c[i] for c in local_bbox) for i in range(3)))
    max_corner = Vector((max(c[i] for c in local_bbox) for i in range(3)))
    size = max_corner - min_corner
    center = min_corner + size / 2

    bbox_cube.scale = size
    scale_matrix = Matrix.Diagonal(size).to_4x4()
    bbox_cube.matrix_world = (
        obj.matrix_world @ Matrix.Translation(center) @ scale_matrix
    )

    # Add wireframe modifier
    mod = bbox_cube.modifiers.new("Wireframe", type="WIREFRAME")
    mod.thickness = 0.03

    mat = bpy.data.materials.new(name="WireframeMaterial")
    mat.use_nodes = True
    # Access the Principled BSDF node
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.0, 0.3, 1.0, 1.0)  # RGBA: blue
        bsdf.inputs["Specular"].default_value = 0  # Optional: make it less shiny
    bbox_cube.data.materials.append(mat)
    bbox_cube.display_type = "WIRE"
    bbox_cube.hide_render = False

    add_text_on_bbox_surface(bbox_cube, text_content=cat_name)
    add_front_arrow_to_bbox(bbox_cube)
    return

def create_arrow(start=(0, 0, 0), direction=(0, 0, 1), shaft_length=2, shaft_radius=0.02, head_length=0.3, head_radius=0.1,
        color=(1, 0, 0, 1)):
    # Normalize the direction vector
    dir_vector = mathutils.Vector(direction).normalized()

    # Calculate the end point of the shaft
    shaft_end = mathutils.Vector(start) + dir_vector * shaft_length

    # Create the shaft (cylinder)
    bpy.ops.mesh.primitive_cylinder_add(
        radius=shaft_radius,
        depth=shaft_length,
        location=(mathutils.Vector(start) + shaft_end) / 2
    )
    shaft = bpy.context.object

    # Align the shaft to the direction vector
    shaft_direction = dir_vector
    shaft.rotation_mode = 'QUATERNION'
    shaft.rotation_quaternion = shaft_direction.to_track_quat('Z', 'Y')

    # Create the head (cone)
    bpy.ops.mesh.primitive_cone_add(
        radius1=head_radius,
        depth=head_length,
        location=shaft_end + dir_vector * (head_length / 2)
    )
    head = bpy.context.object

    # Align the head to the direction vector
    head.rotation_mode = 'QUATERNION'
    head.rotation_quaternion = shaft_direction.to_track_quat('Z', 'Y')

    # Join shaft and head into a single object
    bpy.ops.object.select_all(action='DESELECT')
    shaft.select_set(True)
    head.select_set(True)
    bpy.context.view_layer.objects.active = shaft
    bpy.ops.object.join()
    arrow = bpy.context.object

    # Create a red material
    mat = bpy.data.materials.new(name="RedMaterial")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color  # Red (RGBA)

    # Assign the material to the arrow
    if arrow.data.materials:
        arrow.data.materials[0] = mat
    else:
        arrow.data.materials.append(mat)


def add_front_arrow_to_bbox(bbox_obj, length=0.5, color=(1, 0.5, 0, 1)):
    
    # Get object's center in world coordinates
    start = list(bbox_obj.matrix_world.translation)
    start[2] += bbox_obj.dimensions[2]/2

    # Get local X axis (front) in world coordinates
    local_x = mathutils.Vector((1, 0, 0))  # Y+ is typically "front"
    direction = bbox_obj.matrix_world.to_3x3() @ local_x
    direction.normalize()

    shaft_length = bbox_obj.dimensions[0]*0.75
    shaft_radius = bbox_obj.dimensions[0]*0.03
    head_length = bbox_obj.dimensions[0]*0.2
    head_radius = bbox_obj.dimensions[0]*0.1

    # Call arrow creation utility (assumes get_arrow or create_arrow is defined)
    create_arrow(start=start, 
                 direction=direction, 
                 shaft_length=shaft_length, 
                 shaft_radius = shaft_radius,
                 head_length=head_length, 
                 head_radius=head_radius,
                 color=color)


def get_coord(solver):
   
    
    def add_coordinate(bbox_obj, x,y,z):
        text_content = f"({x},{y})"
        # Create text object
        bpy.ops.object.text_add(location=(0, 0, 0))
        text_obj = bpy.context.active_object
        text_obj.name = text_content
        text_obj.data.body = text_content

        # Rotate and scale
        text_obj.rotation_euler = (0, 0, 0)
        scale = 0.2
        text_obj.scale = (scale, scale, scale)

        # Create material for white text
        text_mat = bpy.data.materials.new(name="TextWhite")
        text_mat.use_nodes = True
        bsdf = text_mat.node_tree.nodes.get("Principled BSDF")
        if bsdf:
            bsdf.inputs["Base Color"].default_value = (1, 0, 0, 1)  # red
            bsdf.inputs["Emission"].default_value = (1, 0, 0, 1)  # Glow a bit
            bsdf.inputs["Emission Strength"].default_value = 2.0
            bsdf.inputs["Roughness"].default_value = 0.5
        text_obj.data.materials.append(text_mat)

        # Position text above bbox
        bbox_center = bbox_obj.matrix_world.translation
        bbox_size = bbox_obj.dimensions
        text_offset = Vector(
            (-text_obj.dimensions[0] * text_obj.scale[0] / 2, 0.1, bbox_size.z / 2 + 0.02)
        )
        text_obj.location = bbox_center + text_offset


    def add_circle(x,y,z=0):
        # Add a filled circle (Ngon)
        bpy.ops.mesh.primitive_circle_add(
            vertices=64,
            radius=0.05,
            fill_type='NGON',
            location=(x, y, z)
        )
        circle = bpy.context.active_object
        circle.name = "RedCircle"

        # Create a red material
        mat = bpy.data.materials.new(name="RedMaterial")
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (1.0, 0.0, 0.0, 1.0)  # RGBA: Red
        bsdf.inputs["Emission"].default_value = (1, 0, 0, 1)  # Glow a bit
        bsdf.inputs["Emission Strength"].default_value = 2.0
        bsdf.inputs["Roughness"].default_value = 0.5

        # Assign the material to the circle
        if circle.data.materials:
            circle.data.materials[0] = mat
        else:
            circle.data.materials.append(mat)
        return circle


    z = 0
    roomsize = [solver.dimensions[0], solver.dimensions[1]]
    for x in range(round(roomsize[0])+1):
        for y in range(round(roomsize[1])+1):
            circle = add_circle(x,y,z=z)
            add_coordinate(circle,x,y,z)
    return 


def get_bbox(state):
    for name in state.objs:
        if name.startswith("window") or name == "newroom_0-0" or name == "entrance":
            continue
        obj = state.objs[name].obj
        cat_name = "_".join(name.split("_")[1:])
        if cat_name.endswith("Factory"):
            cat_name = cat_name[:-7]
        add_rotated_bbox_wireframe(obj, cat_name)
    save_path = "debug.blend"
    bpy.ops.wm.save_as_mainfile(filepath=save_path)
    return

def get_arrow(state):

    # Example usage:
    create_arrow(start=(0, 0, 0), direction=(1, 0, 0), shaft_length=1,color=(1, 0, 0, 1))
    create_arrow(start=(0, 0, 0), direction=(0, 1, 0), shaft_length=1,color=(0, 1, 0, 1))
    create_arrow(start=(0, 0, 0), direction=(0, 0, 1), shaft_length=1,color=(0, 0, 1, 1))  


def add_text_on_bbox_surface(bbox_obj, text_content="ObjectName"):
    # Create text object
    bpy.ops.object.text_add(location=(0, 0, 0))
    text_obj = bpy.context.active_object
    text_obj.name = f"Label_{bbox_obj.name}"
    text_obj.data.body = text_content

    # Rotate and scale
    text_obj.rotation_euler = (0, 0, 0)
    scale = 0.15
    text_obj.scale = (scale, scale, scale)

    # Create material for white text
    text_mat = bpy.data.materials.new(name="TextWhite")
    text_mat.use_nodes = True
    bsdf = text_mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)  # White
        bsdf.inputs["Emission"].default_value = (1, 1, 1, 1)  # Glow a bit
        bsdf.inputs["Emission Strength"].default_value = 2.0
    text_obj.data.materials.append(text_mat)

    # Position text above bbox
    bbox_center = bbox_obj.matrix_world.translation
    bbox_size = bbox_obj.dimensions
    text_offset = Vector(
        (-text_obj.dimensions[0] * text_obj.scale[0] / 2, 0.1, bbox_size.z / 2 + 0.02)
    )
    text_obj.location = bbox_center + text_offset

    # Create background plane
    #    bpy.ops.mesh.primitive_plane_add(size=1, location=text_obj.location - Vector((0, 0, 0.005)))
    bpy.ops.mesh.primitive_plane_add(size=1)
    bg_plane = bpy.context.active_object
    bg_plane.name = f"TextBackground_{bbox_obj.name}"
    #    bg_plane.scale = (2+len(text_content)*0.34, 0.8, 1)
    #    bg_plane.location = [0.8+0.17*len(text_content),0.3,-0.01]

    padding = 0.05
    bpy.context.view_layer.update()
    text_size = text_obj.dimensions.xy
    bg_plane.scale.x = (text_size.x + padding) / text_obj.scale[0]
    bg_plane.scale.y = (text_size.y + padding) / text_obj.scale[1]
    bg_plane.location.x = (text_size.x / 2) / text_obj.scale[0]
    bg_plane.location.y = (text_size.y / 2) / text_obj.scale[1]
    bg_plane.location.z = -0.01
    # Match rotation if needed
    #    bg_plane.rotation_euler = text_obj.rotation_euler

    # Blue material for background
    bg_mat = bpy.data.materials.new(name="BG_Blue")
    bg_mat.use_nodes = True
    bg_bsdf = bg_mat.node_tree.nodes.get("Principled BSDF")
    if bg_bsdf:
        bg_bsdf.inputs["Base Color"].default_value = (0.0, 0.3, 1.0, 1.0)  # Blue
        bg_bsdf.inputs["Roughness"].default_value = 1.0

    bg_plane.data.materials.append(bg_mat)

    # Optional: parent text and background to bbox
    #    text_obj.parent = bbox_obj
    bg_plane.parent = text_obj

    return text_obj, bg_plane


# #
# #bbox_obj = bpy.data.objects.get("RotatedBBOX_MetaCategoryFactory(3629900).spawn_asset(9447620)")
# #add_text_on_bbox_surface(bbox_obj, text_content="ObjectName")
