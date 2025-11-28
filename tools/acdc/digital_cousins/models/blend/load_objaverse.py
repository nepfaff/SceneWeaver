import bpy
import numpy as np
import gzip
import pickle
from blender_util import SelectObjects
from mathutils import  Vector
import mathutils

import math

def load_pickled_3d_asset(file_path,idx=0):
    
    # Open the compressed pickled file
    with gzip.open(file_path, 'rb') as f:
        # Load the pickled object
        loaded_object_data = pickle.load(f)

    # Create a new mesh object in Blender
    mesh = bpy.data.meshes.new(name='LoadedMesh')
    obj = bpy.data.objects.new('LoadedObject', mesh)

    # Link the object to the scene
    bpy.context.scene.collection.objects.link(obj)

    # Set the mesh data for the object
    obj.data = mesh

    # Update the mesh with the loaded data
    # print(loaded_object_data.keys())
    # print(loaded_object_data['triangles'])
    # triangles = [vertex_index for face in loaded_object_data['triangles'] for vertex_index in face]
    triangles = np.array(loaded_object_data['triangles']).reshape(-1,3)
    vertices = []

    for v in loaded_object_data['vertices']:
        vertices.append([v['x'],v['z'],v['y']])

    mesh.from_pydata(vertices, [], triangles)

    uvs = []
    for uv in loaded_object_data['uvs']:
        uvs.append([uv['x'],uv['y']]) 

    mesh.update()

    # Ensure UV coordinates are stored
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")

    uv_layer = mesh.uv_layers["UVMap"]
    for poly in mesh.polygons:
        for loop_index in poly.loop_indices:
            vertex_index = mesh.loops[loop_index].vertex_index
            uv_layer.data[loop_index].uv = uvs[vertex_index]
    

    material = bpy.data.materials.new(name="AlbedoMaterial")
    obj.data.materials.append(material)

    # Assign albedo color to the material
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled_bsdf = nodes.get("Principled BSDF")

    texture_node = nodes.new(type='ShaderNodeTexImage')

    image_path = f"{'/'.join(file_path.split('/')[:-1])}/albedo.jpg"  # Replace with your image file path

    image = bpy.data.images.load(image_path)

    # Assign the image to the texture node
    texture_node.image = image

    # Connect the texture node to the albedo color
    material.node_tree.links.new(
        texture_node.outputs["Color"],
        principled_bsdf.inputs["Base Color"]
    )

    # normal
    image_path = f"{'/'.join(file_path.split('/')[:-1])}/normal.jpg"
    img_normal = bpy.data.images.load(image_path)
    image_texture_node_normal = material.node_tree.nodes.new(type='ShaderNodeTexImage')
    image_texture_node_normal.image = img_normal    
    image_texture_node_normal.image.colorspace_settings.name = 'Non-Color'

    normal_map_node = material.node_tree.nodes.new(type='ShaderNodeNormalMap')

    material.node_tree.links.new(normal_map_node.outputs["Normal"], principled_bsdf.inputs["Normal"])
    material.node_tree.links.new(image_texture_node_normal.outputs["Color"], normal_map_node.inputs["Color"])

    # Assign the material to the object
    obj.data.materials[0] = material    
    
    bbox = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    
    xx = [v[0] for v in bbox]
    yy = [v[1] for v in bbox]
    zz = [v[2] for v in bbox]
    
    length = max(xx) - min(xx)
    width = max(yy) - min(yy)
    height = max(zz) - min(zz)
    
    
    obj.location = [-(max(xx) + min(xx))/2,
                    -(max(xx) + min(xx))/2,
                    -min(zz)]
    # obj.location = [0,0,-height/2]
    with SelectObjects(obj):
        bpy.ops.object.transform_apply(location=True,rotation=False,scale=False)
#    i = idx//10
#    j = idx%10
#    obj.location =    [0.2*i ,0.5*j, 0  ]

    # Update mesh to apply UV changes
    mesh.update()


    return obj

    

def load_openshape(candidate):
    try:
        # Delete the cube
        cube = bpy.data.objects["Cube"]
        bpy.context.view_layer.objects.active = cube  # Set the cube as active
        bpy.ops.object.delete()
    except:
        pass

    bpy.ops.import_scene.gltf(filepath=candidate)

    # preprocess directary
    parent_obj = bpy.context.selected_objects[0]
    # parents = get_highest_parent_objects()

    bpy.ops.object.select_all(action="DESELECT")
    select_meshes_under_empty(parent_obj.name)

    bpy.ops.object.join()
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    joined_object = bpy.context.view_layer.objects.active
    if joined_object is not None:
        joined_object.name = parent_obj.name + "-joined"
        joined_object.location = (0, 0, 0)
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
        bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
        joined_object.location = (0, 0, 0)
        bpy.context.view_layer.objects.active = joined_object
        bpy.ops.object.select_all(action="DESELECT")
        joined_object.select_set(True)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        joined_object.rotation_mode = "XYZ"
        radians = math.radians(0 + 90)
        joined_object.rotation_euler[2] = (
            radians  # Rotate around Z-a to face front
        )
        bpy.ops.object.transform_apply(
            location=False, rotation=True, scale=False
        )
        bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    bpy.ops.object.select_all(action="DESELECT")
    delete_object_with_children(parent_obj)

    imported_obj = joined_object

    imported_obj.location = [0, 0, 0]
    # imported_obj.rotation_euler = [0,0,0]
    bpy.context.view_layer.objects.active = imported_obj
    bpy.ops.object.select_all(action="DESELECT")
    imported_obj.select_set(True)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)


    # update scale
    if max(imported_obj.dimensions)>1:
        scale = 1.0/max(imported_obj.dimensions)
        scale = (scale, scale, scale)
    imported_obj.scale = scale
    bpy.context.view_layer.objects.active = imported_obj  # Set as active object
    bpy.ops.object.select_all(action="DESELECT")
    imported_obj.select_set(True)  # Select the object
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)


    set_origin(imported_obj)

    return imported_obj

    
def select_meshes_under_empty(empty_object_name):
    # Get the empty object
    empty_object = bpy.data.objects.get(empty_object_name)
    print(empty_object is not None)
    if empty_object is not None and empty_object.type == 'EMPTY':
        # Iterate through the children of the empty object
        for child in empty_object.children:
            # Check if the child is a mesh
            if child.type == 'MESH':
                # Select the mesh
                child.select_set(True)
                bpy.context.view_layer.objects.active = child
                
            else:
                select_meshes_under_empty(child.name)
    return 

def delete_object_with_children(obj):
    if not obj:
        print(f"Object '{obj.name}' not found.")
        return

    # 收集所有子对象（递归）
    def get_all_children(o):
        children = []
        for child in o.children:
            if child.type!="EMPTY":
                a = 1
            children.append(child)
            children.extend(get_all_children(child))
        return children

    all_objs = get_all_children(obj)
    all_objs.append(obj)  # 最后也删除自己

    # 取消选择，避免干扰
    bpy.ops.object.select_all(action='DESELECT')

    # 选择并删除
    for o in all_objs:
        o.select_set(True)

    bpy.ops.object.delete()

def set_origin(imported_obj):
    imported_obj.location = [0, 0, 0]
    bbox_corners = [mathutils.Vector(corner) for corner in imported_obj.bound_box]

    min_z = min(corner.z for corner in bbox_corners)
    imported_obj.location.z -= min_z

    mean_x = np.mean([corner.x for corner in bbox_corners])
    imported_obj.location.x -= mean_x
    mean_y = np.mean([corner.y for corner in bbox_corners])
    imported_obj.location.y -= mean_y
    # if self.category == "bookshelf":
    #     import pdb
    #     pdb.set_trace()
    pos_bias = [mean_x, mean_y, min_z]
    bpy.context.scene.cursor.location = [0,0,0]

    bpy.ops.object.origin_set(type="ORIGIN_CURSOR", center="BOUNDS")
    return imported_obj, pos_bias

