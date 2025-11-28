import bpy
import numpy as np
import os
import shutil
import OpenEXR
import Imath
import cv2
import time

import matplotlib.pyplot as plt
from PIL import Image
import argparse
glcam_in_cvcam = np.array([[1, 0, 0, 0],
                           [0, -1, 0, 0],
                           [0, 0, -1, 0],
                           [0, 0, 0, 1]])

def export_camera_info(idx):

    # Select the camera object
    camera = bpy.data.objects[f'Camera{idx}']  # Replace 'Camera' with your camera's name

    # Get the camera world matrix and invert it for world-to-camera transformation
    camera_extrinsics = np.array(camera.matrix_world) @ glcam_in_cvcam
    camera_extrinsics = np.linalg.inv(camera_extrinsics)
#    camera_extrinsics = np.array(camera.matrix_world.inverted())
    print("Camera Extrinsics:\n", camera_extrinsics)
    # Save as .npy
    np.save(f"/home/yandan/workspace/d3fields/data/{folder_name}/camera{idx}/camera_extrinsics.npy", camera_extrinsics)

    # Camera data
    cam_data = camera.data
    resolution_x = bpy.context.scene.render.resolution_x
    resolution_y = bpy.context.scene.render.resolution_y
    scale = bpy.context.scene.render.resolution_percentage / 100.0

    # Focal lengths in pixels
    fx = cam_data.lens * (resolution_x * scale) / cam_data.sensor_width
    fy = cam_data.lens * (resolution_y * scale) / cam_data.sensor_height

    # Principal points (center of the image)
    cx = resolution_x * scale / 2.0
    cy = resolution_y * scale / 2.0

# 
#    camd = camera.data
#    scene = bpy.context.scene
#    f_in_mm = camd.lens
#    scale = scene.render.resolution_percentage / 100
#    resolution_x_in_px = scale * scene.render.resolution_x
#    resolution_y_in_px = scale * scene.render.resolution_y
#    sensor_size_in_mm = get_sensor_size(camd.sensor_fit, camd.sensor_width, camd.sensor_height)
#    sensor_fit = get_sensor_fit(
#        camd.sensor_fit,
#        scene.render.pixel_aspect_x * resolution_x_in_px,
#        scene.render.pixel_aspect_y * resolution_y_in_px
#    )
#    pixel_aspect_ratio = scene.render.pixel_aspect_y / scene.render.pixel_aspect_x
#    if sensor_fit == 'HORIZONTAL':
#        view_fac_in_px = resolution_x_in_px
#    else:
#        view_fac_in_px = pixel_aspect_ratio * resolution_y_in_px
#    pixel_size_mm_per_px = sensor_size_in_mm / f_in_mm / view_fac_in_px
#    fx = 1 / pixel_size_mm_per_px
#    fy = 1 / pixel_size_mm_per_px / pixel_aspect_ratio

#    # Parameters of intrinsic calibration matrix K
#    cx = resolution_x_in_px / 2 - camd.shift_x * view_fac_in_px
#    cy = resolution_y_in_px / 2 + camd.shift_y * view_fac_in_px / pixel_aspect_ratio

    # Camera parameters
    camera_params = np.array([fx, fy, cx, cy])
    print("Camera Parameters:\n", camera_params)

    # Save as .npy
    np.save(f"/home/yandan/workspace/d3fields/data/{folder_name}/camera{idx}/camera_params.npy", camera_params)
    
    return 


def get_sensor_size(sensor_fit, sensor_x, sensor_y):
    if sensor_fit == 'VERTICAL':
        return sensor_y
    return sensor_x

def get_sensor_fit(sensor_fit, size_x, size_y):
    if sensor_fit == 'AUTO':
        if size_x >= size_y:
            return 'HORIZONTAL'
        else:
            return 'VERTICAL'
    return sensor_fit


def render_depth(idx):
    #render depth

    '''Set the render setting for the camera and the scene'''
    # bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'                   # NOTE: set color render to TEXTURE

    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    bpy.context.scene.render.film_transparent = True 

    basedir = f"/home/yandan/workspace/d3fields/data/{folder_name}/camera{idx}"
    output_path = f"{basedir}/depth/"

    view_layer = bpy.context.scene.view_layers[0]
    view_layer.use_pass_z = True

    scene = bpy.context.scene
    if not scene.node_tree:
        scene.use_nodes = True
        
    node_tree = scene.node_tree

    for node in node_tree.nodes:
        node_tree.nodes.remove(node)

    render_layers_node = node_tree.nodes.new(type='CompositorNodeRLayers')

    # rgb output
    file_output_rgb_node = node_tree.nodes.new(type='CompositorNodeOutputFile')
    file_output_rgb_node.base_path = os.path.join(output_path, 'rgb')
    file_output_rgb_node.file_slots.new('Image')
    file_output_rgb_node.format.file_format = 'PNG'
    file_output_rgb_node.label = "RGB Output"

    # depth output
    file_output_depth_node = node_tree.nodes.new(type='CompositorNodeOutputFile')
    file_output_depth_node.base_path = os.path.join(output_path, 'depth')
    file_output_depth_node.file_slots.new('Depth')
    file_output_depth_node.format.file_format = 'OPEN_EXR'
    file_output_depth_node.label = "Depth Output"

    node_tree.links.new(render_layers_node.outputs['Image'], file_output_rgb_node.inputs['Image'])
    node_tree.links.new(render_layers_node.outputs['Depth'], file_output_depth_node.inputs['Depth'])

    bpy.ops.render.render(write_still=True)
    
    if not os.path.exists(f"{basedir}/color"):
        os.mkdir(f"{basedir}/color")
    os.system(f"mv {output_path}/rgb/Image0001.png {basedir}/color/0.png")
    
    return 

def load_depth_from_exr(idx):
    file_path = f"/home/yandan/workspace/d3fields/data/{folder_name}/camera{idx}/depth/depth/Depth0001.exr"
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    channels = header['channels']
    print("Available channels:", channels)  # Verify the channels
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    print(f"Image size: {width} x {height}")
    depth_channel = exr_file.channel('V')  # Depth data is stored in the 'V' channel
    depth_image = np.frombuffer(depth_channel, dtype=np.float32)
    if depth_image.size != width * height:
        raise ValueError(f"Array size {depth_image.size} does not match expected size {width * height}")
    depth_image = depth_image.reshape((height, width))
    return depth_image

def convert_depth(idx):
    #convert depth
    depth_map = load_depth_from_exr(idx)
    depth_map_new = depth_map.copy()
    import pdb
    pdb.set_trace()
    depth_map_new = np.nan_to_num(depth_map_new, nan=0.0, posinf=0.0, neginf=0.0)

    depth_map_new[depth_map_new > 1000] = 0.0
    
    depth_map_new[depth_map_new <0] = 0.0
    import pdb
    
    pdb.set_trace()
    depth_map_16bit = np.uint16(depth_map_new *1000)  # Convert to millimeters (for example)

    # Save the depth map using cv2
    cv2.imwrite( f"/home/yandan/workspace/d3fields/data/{folder_name}/camera{idx}/depth/0.png", depth_map_16bit)

#def render_color(idx=0):
#    #render color image
#    # Enable transparency
#    bpy.context.scene.render.film_transparent = True

#    # Set output file format to PNG with RGBA
#    bpy.context.scene.render.image_settings.file_format = 'PNG'
#    bpy.context.scene.render.image_settings.color_mode = 'RGBA'

#    # Set output path
#    output_path = f"/home/yandan/workspace/d3fields/data/new/camera{idx}/color/0.png"  # Replace with your desired path
#    bpy.context.scene.render.filepath = output_path

#    # Render the image
#    bpy.ops.render.render(write_still=True)
#    time.sleep(0.5)
    

# 获取场景依赖图
depsgraph = bpy.context.evaluated_depsgraph_get()
global folder_name
folder_name = "monitor1"

for idx in range(4):
#if True:
#    idx = 2
    # 获取目标摄像机对象
    target_camera = bpy.data.objects[f"Camera{idx}"]  # 替换 "CameraName" 为目标摄像机的名称
    # 将目标摄像机设置为活动摄像机
    bpy.context.scene.camera = target_camera
    depsgraph.update()  # 强制更新依赖图
    
    render_depth(idx)
    export_camera_info(idx)
#    render_color(idx)