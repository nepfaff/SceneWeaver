# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory of this source tree.

import copy
import functools
import logging
import math
import random

# Authors: Karhan Kayan
from typing import Union

import bpy
import fcl
import gin
import numpy as np
import trimesh
from mathutils import Matrix, Vector
from shapely import LineString, MultiPolygon, Point, Polygon
from sklearn.decomposition import PCA
from trimesh import Scene

from infinigen.core import tagging
from infinigen.core.constraints.expand import expand_mesh
from infinigen.core.util import blender as butil

logger = logging.getLogger(__name__)


@gin.configurable
def bvh_caching_config(enabled=True):
    return enabled


@functools.cache
def group(scene, x):
    if isinstance(x, (list, set)):
        x = tuple(x)
    return subset(scene, x)


def meshes_from_names(scene, names):
    if isinstance(names, str):
        names = [names]
    return [scene.geometry[g] for _, g in (scene.graph[n] for n in names)]


def blender_objs_from_names(names):
    if isinstance(names, str):
        names = [names]
    return [bpy.data.objects[n] for n in names]


def name_from_mesh(scene, mesh):
    mesh_name = None
    for name, mesh in scene.geometry.items():
        if mesh == mesh:
            mesh_name = name
            break
    return mesh_name


def project_to_xy_path2d(mesh: trimesh.Trimesh) -> trimesh.path.Path2D:
    poly = trimesh.path.polygons.projected(mesh, (0, 0, 1), (0, 0, 0))
    d = trimesh.path.exchange.misc.polygon_to_path(poly)
    return trimesh.path.Path2D(entities=d["entities"], vertices=d["vertices"])


def project_to_xy_poly(mesh: trimesh.Trimesh):
    poly = trimesh.path.polygons.projected(mesh, (0, 0, 1), (0, 0, 0))
    return poly


def closest_edge_to_point_poly(polygon, point):
    closest_distance = float("inf")
    closest_edge = None

    for i, coord in enumerate(polygon.exterior.coords[:-1]):
        start, end = coord, polygon.exterior.coords[i + 1]
        line = LineString([start, end])
        distance = line.distance(point)

        if distance < closest_distance:
            closest_distance = distance
            closest_edge = line

    return closest_edge


def closest_edge_to_point_edge_list(edge_list: list[LineString], point):
    closest_distance = float("inf")
    closest_edge = None

    for line in edge_list:
        distance = line.distance(point)

        if distance < closest_distance:
            closest_distance = distance
            closest_edge = line

    return closest_edge


def compute_outward_normal(line, polygon):
    dx = line.xy[0][1] - line.xy[0][0]  # x1 - x0
    dy = line.xy[1][1] - line.xy[1][0]  # y1 - y0

    # Candidate normal vectors (perpendicular to edge)
    normal_vector_1 = np.array([dy, -dx])
    normal_vector_2 = -normal_vector_1

    # Normalize the vectors (optional but recommended for consistency)
    normal_vector_1 = normal_vector_1 / np.linalg.norm(normal_vector_1)
    normal_vector_2 = normal_vector_2 / np.linalg.norm(normal_vector_2)

    # Midpoint of the line segment
    mid_point = line.interpolate(0.5, normalized=True)

    # Move a tiny bit in the direction of the normals to check which points outside
    test_point_1 = mid_point.coords[0] + 0.01 * normal_vector_1
    mid_point.coords[0] + 0.01 * normal_vector_2

    # Return the normal for which the test point lies outside the polygon
    if polygon.contains(Point(test_point_1)):
        return normal_vector_2
    else:
        return normal_vector_1


def get_transformed_axis(scene, obj_name):
    obj = bpy.data.objects[obj_name]
    trimesh_mesh = meshes_from_names(scene, obj_name)[0]
    axis = trimesh_mesh.axis
    rot_mat = np.array(obj.matrix_world.to_3x3())
    return rot_mat @ np.array(axis)


def set_axis(scene, objs: Union[str, list[str]], canonical_axis):
    if isinstance(objs, str):
        objs = [objs]
    obj_meshes = meshes_from_names(scene, objs)
    for obj_name, obj in zip(objs, obj_meshes):
        obj.axis = canonical_axis
        obj.axis = get_transformed_axis(scene, obj_name)


def get_plane_from_3dmatrix(matrix):
    """Extract the plane_normal and plane_origin from a transformation matrix."""
    # The normal of the plane can be extracted from the 3x3 rotation part of the matrix
    plane_normal = matrix[:3, 2]
    plane_origin = matrix[:3, 3]
    return plane_normal, plane_origin


def project_points_onto_plane(points, plane_origin, plane_normal):
    """Project 3D points onto a plane."""
    d = np.dot(points - plane_origin, plane_normal)[:, None]
    return points - d * plane_normal


def to_2d_coordinates(points, plane_normal):
    """Convert 3D points to 2D using the plane defined by its normal."""
    # Compute two perpendicular vectors on the plane
    u = np.cross(plane_normal, [1, 0, 0])
    if np.linalg.norm(u) < 1e-10:
        u = np.cross(plane_normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(plane_normal, u)
    v /= np.linalg.norm(v)

    # Convert 3D points to 2D using dot products
    return np.column_stack([points.dot(u), points.dot(v)])


def ensure_correct_order(points):
    """
    Ensures the points are in counter-clockwise order.
    If not, it reverses them.
    """
    # Calculate signed area
    n = len(points)
    area = (
        sum(
            (points[i][0] * points[(i + 1) % n][1])
            - (points[(i + 1) % n][0] * points[i][1])
            for i in range(n)
        )
        / 2.0
    )
    # Return the points in reverse order if area is negative
    return points[::-1] if area < 0 else points


def sample_random_point(polygon):
    """
    Sample a random point from inside the given Shapely polygon.
    """
    minx, miny, maxx, maxy = polygon.bounds
    while True:
        p = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(p):
            return p



def delete_obj(scene, a, delete_blender=True, delete_asset=False):
    if isinstance(a, str):
        a = [a]
    if delete_blender:
        obj_list = [bpy.data.objects[obj_name] for obj_name in a if obj_name in bpy.data.objects]
        butil.delete(obj_list)
    for obj_name in a:
        # bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
        if scene:
            scene.graph.transforms.remove_node(obj_name)
            scene.delete_geometry(obj_name + "_mesh")

    if delete_asset:
        asset_names = [name.replace(
                "bbox_placeholder", "spawn_asset"
            ).replace("spawn_placeholder", "spawn_asset") for name in a]
        asset_names = [name for name in asset_names if name in bpy.data.objects]
        
        if delete_blender:
            obj_list = [bpy.data.objects[obj_name] for obj_name in asset_names]
            butil.delete(obj_list)
        for obj_name in asset_names:
            # bpy.data.objects.remove(bpy.data.objects[obj_name], do_unlink=True)
            if scene:
                scene.graph.transforms.remove_node(obj_name)
                scene.delete_geometry(obj_name + "_mesh")


def global_vertex_coordinates(obj, local_vertex) -> Vector:
    return obj.matrix_world @ local_vertex.co


def global_polygon_normal(obj, polygon):
    loc, rot, scale = obj.matrix_world.decompose()
    rot = rot.to_matrix()
    normal = rot @ polygon.normal
    return normal / np.linalg.norm(normal)


def is_planar(obj, tolerance=1e-6):
    if len(obj.data.polygons) != 1:
        return False

    polygon = obj.data.polygons[0]
    global_normal = global_polygon_normal(obj, polygon)

    # Take the first vertex as a reference point on the plane
    ref_vertex = global_vertex_coordinates(obj, obj.data.vertices[polygon.vertices[0]])

    # Check if all vertices lie on the plane defined by the reference vertex and the global normal
    for vertex in obj.data.vertices:
        distance = (global_vertex_coordinates(obj, vertex) - ref_vertex).dot(
            global_normal
        )
        if not math.isclose(distance, 0, abs_tol=tolerance):
            return False

    return True


def planes_parallel(plane_obj_a, plane_obj_b, tolerance=1e-6):
    if plane_obj_a.type != "MESH" or plane_obj_b.type != "MESH":
        raise ValueError("Both objects should be of type 'MESH'")

    # # Check if the objects are planar
    # if not is_planar(plane_obj_a) or not is_planar(plane_obj_b):
    #     raise ValueError("One or both objects are not planar")

    global_normal_a = global_polygon_normal(plane_obj_a, plane_obj_a.data.polygons[0])
    global_normal_b = global_polygon_normal(plane_obj_b, plane_obj_b.data.polygons[0])

    dot_product = global_normal_a.dot(global_normal_b)

    return math.isclose(dot_product, 1, abs_tol=tolerance) or math.isclose(
        dot_product, -1, abs_tol=tolerance
    )


def distance_to_plane(point, plane_point, plane_normal):
    """Compute the distance from a point to a plane defined by a point and a normal."""
    return abs((point - plane_point).dot(plane_normal))


def subset(scene: Scene, incl):
    if isinstance(incl, str):
        incl = [incl]

    objs = []
    for n in scene.graph.nodes:
        T, g = scene.graph[n]
        if g is None:
            continue
        otags = scene.geometry[g].metadata["tags"]
        if any(t in incl for t in otags):
            objs.append(n)

    # assert len(objs) > 0, incl

    return objs


def add_object_cached(col, name, col_obj, fcl_obj):
    geom = fcl_obj  # 将 fcl_obj 赋值给 geom
    o = col_obj  # 将 col_obj 赋值给 o
    # 添加碰撞对象到集合
    if name in col._objs:  # 如果名称已经在对象集合中
        col._manager.unregisterObject(col._objs[name])  # 从管理器中注销之前的对象
    col._objs[name] = {"obj": o, "geom": geom}  # 将新的对象和几何体存储到集合中
    # 存储几何体的名称
    col._names[id(geom)] = name  # 用几何体的 ID 作为键，存储名称

    col._manager.registerObject(o)  # 将新对象注册到管理器中
    col._manager.update()  # 更新管理器状态
    return o  # 返回新注册的对象


def col_from_subset(scene, names, tags=None, bvh_cache=None, expand=False,export=False,return_geom=False):
    if isinstance(names, str):
        names = [names]

    if (not expand) and (not return_geom) and bvh_cache is not None and bvh_caching_config():
        tag_key = frozenset(tags) if tags is not None else None
        key = (frozenset(names), tag_key)
        res = bvh_cache.get(key)
        if res is not None:
            return res

    col = trimesh.collision.CollisionManager()
    geoms = []
    T = trimesh.transformations.identity_matrix()  # 设置变换矩阵为单位矩阵
    t = fcl.Transform(T[:3, :3], T[:3, 3])  # 创建 FCL 变换对象
    
    for name in names:
        
        _, g = scene.graph[name]  # 从场景图中获取变换矩阵和几何体索引
        geom = scene.geometry[g]  # 获取几何体
        if "SingleCabinetFactory" in name and not geom.is_watertight:
            a = 1
        # if geom.is_watertight is False: # fix mesh bug in objaverse
        #     print(name)
        #     print(geom.is_watertight)  # Should be False, but check if there are issues in edge connectivity
        #     success = geom.fill_holes()
        #     import pdb
        #     pdb.set_trace()
        #     print(geom.is_manifold) 
        #     geom_new = copy.deepcopy(geom)
        #     geom_new = geom_new.repair()

        #     # broken = trimesh.repair.broken_faces(geom_new, color=[255,0,0,255])    
        #     # success = geom_new.fill_holes() 
        #     # if success is False:
        #     #     import pdb
        #     #     pdb.set_trace()
        #     # Step 2: Remove unreferenced vertices (this helps clean the mesh)
        #     geom_new.remove_unreferenced_vertices()

        #     # Step 3: Remove non-manifold edges (to fix problematic edges)
        #     # repaired_mesh = geom_new.repair.fill_non_manifold()
           

        #     geom_new.current_transform = trimesh.transformations.identity_matrix()
        #     geom_new.fcl_obj = col._get_fcl_obj(geom_new)
           
        #     geom_new.col_obj = fcl.CollisionObject(geom_new.fcl_obj, t)
        #     geom = geom_new
        


        if expand:
            # print(name)
            mesh_expand = expand_mesh(geom, name)
            geom = copy.deepcopy(mesh_expand)
            
            
            geom.fcl_obj = col._get_fcl_obj(geom)  # 获取 FCL 对象
            geom.col_obj = fcl.CollisionObject(geom.fcl_obj, t)  # 创建碰撞对象
           
            if export:
                geom.export(name+".obj")
              

        if tags is not None and len(tags) > 0:  # 如果存在标签并且标签数量大于零
            obj = blender_objs_from_names(name)[0]  # 从名称中获取Blender对象
            mask = tagging.tagged_face_mask(obj, tags)  # 获取面标记掩码
            if not mask.any():  # 如果没有面被标记
                logger.warning(f"{name=} had {mask.sum()=} for {tags=}")
                continue
            geom = geom.submesh(np.where(mask), append=True)  # 获取被标记的子网格
            
            geom.fcl_obj = col._get_fcl_obj(geom)  # 获取 FCL 对象
            geom.col_obj = fcl.CollisionObject(geom.fcl_obj, t)  # 创建碰撞对象
            assert len(geom.faces) == mask.sum()  # 确保面数匹配
        if geom.col_obj is None:
            geom.fcl_obj = col._get_fcl_obj(geom)  # 获取 FCL 对象
            geom.col_obj = fcl.CollisionObject(geom.fcl_obj, t)  # 创建碰撞对象
        # col.add_object(name, geom, T)
        add_object_cached(col, name, geom.col_obj, geom.fcl_obj)  # 使用缓存添加对象

        # if geom.is_watertight is False: # fix mesh bug in objaverse
        #     print(name)
        #     geom_new = copy.deepcopy(geom)
        #     success = geom_new.fill_holes()
        #     geom_new.current_transform = trimesh.transformations.identity_matrix()
        #     geom_new.fcl_obj = col._get_fcl_obj(geom_new)
           
        #     geom_new.col_obj = fcl.CollisionObject(geom_new.fcl_obj, t)
        #     geom = geom_new

        geoms.append(geom)
        

    if len(col._objs) == 0:  # 如果没有对象被添加
        logger.debug(f"{names=} got no objs, returning None")
        col = None

    if (not expand) and (not return_geom) and bvh_cache is not None and bvh_caching_config():  # 如果存在缓存并且缓存配置有效
        bvh_cache[key] = col  # 将结果存入缓存
    if return_geom:
        return col, geoms
    else:
        return col  # 返回碰撞体集合



def plot_geometry(ax, geom, color="blue"):
    if isinstance(geom, Polygon):
        x, y = geom.exterior.xy
        ax.fill(x, y, alpha=0.5, fc=color, ec="black")
    elif isinstance(geom, MultiPolygon):
        for sub_geom in geom:
            x, y = sub_geom.exterior.xy
            ax.fill(x, y, alpha=0.5, fc=color, ec="black")
    elif isinstance(geom, LineString):
        x, y = geom.xy
        ax.plot(x, y, color=color)
    elif isinstance(geom, Point):
        ax.plot(geom.x, geom.y, "o", color=color)


def sync_trimesh(scene: trimesh.Scene, obj_name: str):  # MARK trimesh
    bpy.context.view_layer.update()  # 更新Blender的视图层，以反映当前场景的变化
    blender_obj = bpy.data.objects[obj_name]  # 获取指定名称的Blender对象
    mesh = meshes_from_names(scene, obj_name)[0]  # 从场景中根据对象名称获取网格数据
    T_old = mesh.current_transform  # 保存网格的当前变换
    T = np.array(blender_obj.matrix_world)  # 获取Blender对象的世界变换矩阵
    mesh.apply_transform(T @ np.linalg.inv(T_old))  # 应用变换，将新变换应用到网格
    mesh.current_transform = np.array(
        blender_obj.matrix_world
    )  # 更新网格的当前变换为Blender对象的世界变换
    t = fcl.Transform(T[:3, :3], T[:3, 3])  
    if mesh.col_obj is not None:# 创建一个Transform对象，包含旋转和位移
        mesh.col_obj.setTransform(t)  # 将变换应用到网格的碰撞对象上
    else:
        col = trimesh.collision.CollisionManager()
        mesh.fcl_obj = col._get_fcl_obj(mesh)  # 获取 FCL 对象
        mesh.col_obj = fcl.CollisionObject(mesh.fcl_obj, t) 

def translate(scene: trimesh.Scene, a: str, translation):
    blender_obj = bpy.data.objects[a]
    blender_obj.location += Vector(translation)
    
    if scene:
        sync_trimesh(scene, a)

    asset_name = a.replace(
            "bbox_placeholder", "spawn_asset"
        ).replace("spawn_placeholder", "spawn_asset")

    if asset_name in bpy.data.objects:
        blender_asset_obj = bpy.data.objects.get(asset_name)
        blender_asset_obj.location = blender_obj.location
        blender_asset_obj.scale = blender_obj.scale
        if scene:
            sync_trimesh(scene, asset_name)
    


def rotate(scene: trimesh.Scene, a: str, axis, angle):  # MARK rotation
    blender_obj = bpy.data.objects[a]
    rotation_matrix = trimesh.transformations.rotation_matrix(angle, axis)
    transform_matrix = Matrix(rotation_matrix).to_4x4()
    loc, rot, scale = blender_obj.matrix_world.decompose()
    rot = rot.to_matrix().to_4x4()
    rot = transform_matrix @ rot
    rot = rot.to_quaternion()
    blender_obj.matrix_world = Matrix.LocRotScale(loc, rot, scale)

    if scene:
        sync_trimesh(scene, a)

    asset_name = a.replace(
            "bbox_placeholder", "spawn_asset"
        ).replace("spawn_placeholder", "spawn_asset")

    if asset_name in bpy.data.objects:
        blender_asset_obj = bpy.data.objects.get(asset_name)
        blender_asset_obj.matrix_world = blender_obj.matrix_world 
        if scene:
            sync_trimesh(scene, asset_name)

def set_location(scene: trimesh.Scene, obj_name: str, location):
    blender_mesh = bpy.data.objects[obj_name]
    blender_mesh.location = location
    sync_trimesh(scene, obj_name)

    asset_name = obj_name.replace(
            "bbox_placeholder", "spawn_asset"
        ).replace("spawn_placeholder", "spawn_asset")

    if asset_name in bpy.data.objects:
        blender_asset_obj = bpy.data.objects.get(asset_name)
        blender_asset_obj.location = blender_mesh.location
        blender_asset_obj.scale = blender_mesh.scale
        if scene:
            sync_trimesh(scene, asset_name)


def set_rotation(scene: trimesh.Scene, obj_name: str, rotation):
    blender_mesh = blender_objs_from_names(obj_name)[0]
    blender_mesh.rotation_euler = rotation
    sync_trimesh(scene, obj_name)

    asset_name = obj_name.replace(
            "bbox_placeholder", "spawn_asset"
        ).replace("spawn_placeholder", "spawn_asset")

    if asset_name in bpy.data.objects:
        blender_asset_obj = bpy.data.objects.get(asset_name)
        blender_asset_obj.rotation_mode = 'XYZ'
        blender_asset_obj.rotation_euler = blender_mesh.rotation_euler
        blender_asset_obj.scale = blender_mesh.scale
        if scene:
            sync_trimesh(scene, asset_name)


# for debugging. does not actually find centroid
def blender_centroid(a):
    return np.mean([a.matrix_world @ v.co for v in a.data.vertices], axis=0)


def order_objects_by_principal_axis(objects: list[bpy.types.Object]):
    locations = [obj.location for obj in objects]
    location_matrix = np.array(locations)
    pca = PCA(n_components=1)
    pca.fit(location_matrix)
    locations_projected = pca.transform(location_matrix)
    sorted_indices = np.argsort(locations_projected.ravel())
    return [objects[i] for i in sorted_indices]
