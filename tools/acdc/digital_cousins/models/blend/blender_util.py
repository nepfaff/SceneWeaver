
import logging
# from contextlib import nullcontext
# from itertools import chain
# from math import prod
# from pathlib import Path

# import bmesh
import bpy
import trimesh

# import mathutils
# import numpy as np
# import trimesh
# from tqdm import tqdm
import numpy as np
from mathutils import Vector



logger = logging.getLogger(__name__)



class SelectObjects:
    def __init__(self, objects, active=0):
        self.objects = list(objects) if hasattr(objects, "__iter__") else [objects]
        self.active = active

        self.saved_objs = None
        self.saved_active = None

    def _check_selectable(self):
        unlinked = [o for o in self.objects if o.name not in bpy.context.scene.objects]
        if len(unlinked) > 0:
            raise ValueError(
                f"{SelectObjects.__name__} had objects {unlinked=} which are not in bpy.context.scene.objects and cannot be selected"
            )

        hidden = [o for o in self.objects if o.hide_viewport]
        if len(hidden) > 0:
            raise ValueError(
                f"{SelectObjects.__name__} had objects {hidden=} which are hidden and cannot be selected"
            )

    def _get_intended_active(self):
        if isinstance(self.active, int):
            if self.active >= len(self.objects):
                return None
            else:
                return self.objects[self.active]
        else:
            return self.active

    def _validate(self, error=False):
        if error:

            def msg(str):
                raise ValueError(str)
        else:
            msg = logger.warning

        difference = set(self.objects) - set(bpy.context.selected_objects)
        if len(difference):
            msg(
                f"{SelectObjects.__name__} failed to select {self.objects=}, result was {bpy.context.selected_objects=}. "
                "The most common cause is that the objects are in a collection with col.hide_viewport=True"
            )

        intended = self._get_intended_active()
        if intended is not None and bpy.context.active_object != intended:
            msg(
                f"{SelectObjects.__name__} failed to set active object to {intended=}, result was {bpy.context.active_object=}"
            )

    def __enter__(self):
        self.saved_objects = list(bpy.context.selected_objects)
        self.saved_active = bpy.context.active_object

        select_none()
        select(self.objects)

        intended = self._get_intended_active()
        if intended is not None:
            bpy.context.view_layer.objects.active = intended

        self._validate()

    def __exit__(self, *_):
        # our saved selection / active objects may have been deleted, update them to only include valid ones
        def enforce_not_deleted(o):
            try:
                return o if o.name in bpy.data.objects else None
            except ReferenceError:
                return None

        self.saved_objects = [enforce_not_deleted(o) for o in self.saved_objects]
        self.saved_objects = [o for o in self.saved_objects if o is not None]

        select_none()
        select(self.saved_objects)
        if self.saved_active is not None:
            bpy.context.view_layer.objects.active = enforce_not_deleted(
                self.saved_active
            )

def select_none():
    if hasattr(bpy.context, "active_object") and bpy.context.active_object is not None:
        bpy.context.active_object.select_set(False)
    if hasattr(bpy.context, "selected_objects"):
        for obj in bpy.context.selected_objects:
            obj.select_set(False)


def select(objs: bpy.types.Object | list[bpy.types.Object]):
    select_none()
    if not isinstance(objs, list):
        objs = [objs]
    for o in objs:
        if o.name not in bpy.context.scene.objects:
            raise ValueError(f"Object {o.name=} not in scene and cant be selected")
        o.select_set(True)



def are_bbox_colliding(obj1, obj2):
    # Get world space bounding boxes

    bbox1 = [obj1.matrix_world @ Vector(corner) for corner in obj1.bound_box]
    bbox2 = [obj2.matrix_world @ Vector(corner) for corner in obj2.bound_box]
    
    # Get min/max coordinates for each bounding box
    min1, max1 = Vector(map(min, zip(*bbox1))), Vector(map(max, zip(*bbox1)))
    min2, max2 = Vector(map(min, zip(*bbox2))), Vector(map(max, zip(*bbox2)))
    
    # Check for overlap
    return all(min1[i] <= max2[i] and max1[i] >= min2[i] for i in range(3))



def get_unbiased_verts_faces(obj1_name):
    # Grab the object, make it collidable
    obj1 = bpy.data.objects.get(obj1_name)
    mesh = obj1.data

    # Extract vertices as a list of (x, y, z) coordinates
    vertices = [vertex.co for vertex in mesh.vertices]
    # Extract faces as a list of tuples of vertex indices
    faces = [tuple(face.vertices) for face in mesh.polygons]
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    _ = trimesh.interfaces.vhacd.convex_decomposition(mesh,convexhullDownsampling=1,maxNumVerticesPerCH=1024)
    mesh =  trimesh.load("decomp.obj")
    if isinstance(mesh, trimesh.Scene):
        # Merge all geometries in the scene into a single mesh
        mesh = trimesh.util.concatenate(
            [geometry for geometry in mesh.geometry.values()]
        )
   
    points = mesh.vertices
    faces = mesh.faces
    return points,faces

def are_mesh_colliding(collision_info, obj1, obj2):
    import torch
    from kaolin.ops.mesh import check_sign
    verts = collision_info[obj1.name]["points"]
    faces = collision_info[obj1.name]["faces"]
    
    verts = [obj1.matrix_world @ Vector(vert) for vert in verts]
    verts = np.array([vert.to_tuple() for vert in verts])
    verts = torch.tensor(verts,device = "cuda").unsqueeze(0)
    faces = torch.tensor(faces,device = "cuda").long()

    points = collision_info[obj2.name]["points"]
    points = [obj2.matrix_world @ Vector(point) for point in points]
    points = np.array([point.to_tuple() for point in points])
    pointscuda = torch.tensor(points,device = "cuda").unsqueeze(0)

    occupancy = check_sign(verts,faces,pointscuda)
    print(obj1.name,obj2.name,occupancy.max()>0)
    # import pdb
    # pdb.set_trace()
    mesh = trimesh.Trimesh(vertices=verts[0].cpu(),faces=faces.cpu())  
    mesh.export('output.obj')
    if occupancy.max()>0:
        pcdinside = pointscuda[0][occupancy.cpu().numpy()[0]==1]
        return True #, pcdinside
    
    return False #, None