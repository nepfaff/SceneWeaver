#!/usr/bin/env python
"""
Compute collision metrics for an existing scene using Blender BVH collision detection.
This produces metrics in the same format as infinigen_examples/steps/evaluate.py.

Must be run with Blender:
    ./blender/blender --background scene.blend --python infinigen_examples/compute_metrics.py -- --output metrics.json

Example:
    ./blender/blender --background \
        Pipeline/output/Design_me_a_bedroom_0/record_files/scene_14.blend \
        --python infinigen_examples/compute_metrics.py -- \
        --output Pipeline/output/Design_me_a_bedroom_0/record_files/metric_14.json

Note: For exact reproducibility with the original evaluation, enable eval_metric() during
scene generation by uncommenting line 370 in infinigen_examples/generate_indoors.py.
This standalone script uses Blender's BVH collision detection which may produce
slightly different results than the original trimesh-based evaluation.
"""

import argparse
import json
import os
import re
import sys

import bpy
import bmesh
from mathutils.bvhtree import BVHTree


def parse_object_name(name):
    """
    Parse object name to extract factory type, instance IDs, mesh type, and suffix.

    Examples:
        'BedFactory(3164690).spawn_asset(7191960)' -> ('BedFactory', '3164690', '7191960', 'spawn_asset', '')
        'PillowFactory(3164690).spawn_asset(7191960).001' -> ('PillowFactory', '3164690', '7191960', 'spawn_asset', '.001')
        'OfficeChairFactory(2033975).bbox_placeholder(9406388)' -> ('OfficeChairFactory', '2033975', '9406388', 'bbox_placeholder', '')

    Returns: (factory_type, factory_id, instance_id, mesh_type, suffix) or (name, None, None, None, '') if not parseable
    """
    # Pattern: FactoryName(factory_id).method(instance_id) possibly with .001 suffix
    match = re.match(r'(\w+)\((\d+)\)\.(\w+)\((\d+)\)(\.?\d*)', name)
    if match:
        return match.group(1), match.group(2), match.group(4), match.group(3), match.group(5)
    return name, None, None, None, ''


def get_logical_object_id(name):
    """
    Get a unique ID for the logical object (including suffix for multi-instance objects).

    Returns: 'FactoryType_factoryId_instanceId_suffix' or original name if not parseable
    """
    factory_type, factory_id, instance_id, mesh_type, suffix = parse_object_name(name)
    if factory_id and instance_id:
        return f"{factory_type}_{factory_id}_{instance_id}{suffix}"
    return name


def categorize_collision_pairs(collision_pairs):
    """
    Categorize collision pairs into:
    - internal_mesh: same object, different mesh type (spawn_asset vs placeholder) - FILTER OUT
    - component: different instances of same parent (.001 vs .002) - COUNT AS REAL
    - real: different logical objects - COUNT AS REAL

    Returns dict with categorized pairs and counts.
    """
    internal_mesh_pairs = []  # Same object, different mesh representation - filter out
    component_pairs = []      # Different instances from same parent (pillow.001 vs pillow.002)
    real_pairs = []           # Different logical objects
    real_unique_set = set()

    for pair in collision_pairs:
        name1, name2 = pair
        factory1, fid1, iid1, mesh1, suffix1 = parse_object_name(name1)
        factory2, fid2, iid2, mesh2, suffix2 = parse_object_name(name2)

        # Same factory, instance, AND suffix but different mesh type = truly internal
        # e.g., spawn_asset vs spawn_placeholder for the same object
        if (fid1 and fid2 and fid1 == fid2 and iid1 == iid2 and suffix1 == suffix2):
            internal_mesh_pairs.append(pair)
        # Same factory and instance but different suffix = component collision (pillow.001 vs pillow.002)
        elif (fid1 and fid2 and fid1 == fid2 and iid1 == iid2 and suffix1 != suffix2):
            component_pairs.append(pair)
            # Track as real collision
            logical_id1 = get_logical_object_id(name1)
            logical_id2 = get_logical_object_id(name2)
            unique_pair = tuple(sorted([logical_id1, logical_id2]))
            real_unique_set.add(unique_pair)
        else:
            real_pairs.append(pair)
            # Track unique logical object pairs
            logical_id1 = get_logical_object_id(name1)
            logical_id2 = get_logical_object_id(name2)
            unique_pair = tuple(sorted([logical_id1, logical_id2]))
            real_unique_set.add(unique_pair)

    return {
        'internal_mesh': internal_mesh_pairs,  # Filter these out (same object, diff mesh)
        'component': component_pairs,          # Count these (different instances like pillow.001 vs .002)
        'real': real_pairs,                    # Count these (different objects)
        'real_unique': [list(p) for p in real_unique_set]  # Unique logical object pairs
    }


def get_mesh_objects():
    """Get all mesh objects in the scene, excluding room and windows."""
    mesh_objs = []
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue
        name = obj.name
        # Skip room, windows, entrance, and rugs (same exclusions as original)
        if (name.startswith("window") or
            name == "newroom_0-0" or
            name.startswith("newroom_") or
            name == "entrance" or
            name.endswith("RugFactory") or
            "rug" in name.lower()):
            continue
        mesh_objs.append(obj)
    return mesh_objs


def get_bvh_tree(obj):
    """Create a BVH tree from a mesh object in world space."""
    bm = bmesh.new()

    # Get the evaluated mesh with modifiers applied
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh()

    bm.from_mesh(mesh)
    bm.transform(obj.matrix_world)

    bvh = BVHTree.FromBMesh(bm)

    bm.free()
    obj_eval.to_mesh_clear()

    return bvh


def get_room_bounds():
    """Get the room boundary polygon for OOB checking."""
    room_obj = None
    for obj in bpy.data.objects:
        if obj.name == "newroom_0-0" or obj.name.startswith("newroom_"):
            room_obj = obj
            break

    if room_obj is None:
        return None

    # Get room bounds in world space
    bbox = [room_obj.matrix_world @ v.co for v in room_obj.data.vertices]
    min_x = min(v.x for v in bbox)
    max_x = max(v.x for v in bbox)
    min_y = min(v.y for v in bbox)
    max_y = max(v.y for v in bbox)

    return (min_x, max_x, min_y, max_y)


def is_out_of_bounds(obj, room_bounds, buffer=0.01):
    """Check if object center is outside room bounds."""
    if room_bounds is None:
        return False

    min_x, max_x, min_y, max_y = room_bounds

    # Get object center in world space
    center = obj.matrix_world.translation

    # Check if center is outside bounds (with small buffer)
    if (center.x < min_x - buffer or center.x > max_x + buffer or
        center.y < min_y - buffer or center.y > max_y + buffer):
        return True

    return False


def count_unique_objects(mesh_objs):
    """Count unique logical objects (ignoring mesh type variants like spawn_asset vs placeholder)."""
    unique_ids = set()
    for obj in mesh_objs:
        logical_id = get_logical_object_id(obj.name)
        unique_ids.add(logical_id)
    return len(unique_ids), list(unique_ids)


def count_factory_objects(mesh_objs):
    """
    Count unique factory instances (entire factory as single object).
    E.g., bed + mattress + pillows with same factory_id = 1 object.
    """
    factory_ids = set()
    for obj in mesh_objs:
        factory_type, factory_id, instance_id, mesh_type, suffix = parse_object_name(obj.name)
        if factory_id:
            # Group by factory_id only (all components of a bed = 1 object)
            factory_ids.add(factory_id)
        else:
            # Non-factory objects count individually
            factory_ids.add(obj.name)
    return len(factory_ids)


def compute_metrics():
    """Compute physics metrics for the current scene."""
    print("Computing scene metrics...")

    # Get all relevant mesh objects
    mesh_objs = get_mesh_objects()
    print(f"Found {len(mesh_objs)} mesh objects")

    # Count unique logical objects (without placeholder duplicates)
    num_unique, unique_ids = count_unique_objects(mesh_objs)
    print(f"Found {num_unique} unique logical objects")

    # Count factory objects (entire factory as single object)
    num_factory = count_factory_objects(mesh_objs)
    print(f"Found {num_factory} factory objects (bed+components=1)")

    # Build BVH trees for all objects
    bvh_cache = {}
    for obj in mesh_objs:
        try:
            bvh_cache[obj.name] = get_bvh_tree(obj)
        except Exception as e:
            print(f"Warning: Could not build BVH for {obj.name}: {e}")

    print(f"Converted {len(bvh_cache)} meshes")

    # Get room bounds
    room_bounds = get_room_bounds()
    if room_bounds:
        print("Found room bounds")

    # Check for out of bounds objects
    oob_objects = []
    for obj in mesh_objs:
        if is_out_of_bounds(obj, room_bounds):
            oob_objects.append(obj.name)

    print(f"OOB: {len(oob_objects)}", oob_objects if oob_objects else "")

    # Check for collisions
    print("Checking collisions...")
    collision_pairs = []
    checked = set()

    for i, obj1 in enumerate(mesh_objs):
        if obj1.name not in bvh_cache:
            continue
        bvh1 = bvh_cache[obj1.name]

        for obj2 in mesh_objs[i+1:]:
            if obj2.name not in bvh_cache:
                continue

            # Skip if already checked
            pair_key = tuple(sorted([obj1.name, obj2.name]))
            if pair_key in checked:
                continue
            checked.add(pair_key)

            bvh2 = bvh_cache[obj2.name]

            # Check for overlap
            overlap = bvh1.overlap(bvh2)
            if len(overlap) > 0:
                # Sort names for consistent output
                name1, name2 = sorted([obj1.name, obj2.name], reverse=True)
                collision_pairs.append([name1, name2])
                print(f"  Collision: {obj1.name} <-> {obj2.name}")

    print("Checking bounds...")

    # Categorize collision pairs
    categorized = categorize_collision_pairs(collision_pairs)

    print(f"\nCollision breakdown:")
    print(f"  Internal mesh (filtered out): {len(categorized['internal_mesh'])}")
    print(f"  Component collisions (.001 vs .002): {len(categorized['component'])}")
    print(f"  Inter-object collisions: {len(categorized['real'])}")
    print(f"  Total real unique pairs: {len(categorized['real_unique'])}")

    results = {
        "Nobj": len(mesh_objs),
        "Nobj_unique": num_unique,  # Unique logical objects (without placeholder duplicates)
        "Nobj_factory": num_factory,  # Factory objects (bed+mattress+pillows=1)
        "OOB": len(oob_objects),
        "OOB Objects": oob_objects,
        # Raw BBL (all mesh-level collisions) - for backwards compatibility
        "BBL": len(collision_pairs),
        "BBL objects": collision_pairs,
        # Categorized BBL metrics
        "BBL_internal_mesh": len(categorized['internal_mesh']),
        "BBL_internal_mesh_objects": categorized['internal_mesh'],
        "BBL_component": len(categorized['component']),
        "BBL_component_objects": categorized['component'],
        "BBL_inter_object": len(categorized['real']),
        "BBL_inter_object_objects": categorized['real'],
        "BBL_real_unique": len(categorized['real_unique']),
        "BBL_real_unique_objects": categorized['real_unique'],
    }

    return results


def main():
    # Parse arguments after '--'
    argv = sys.argv
    if '--' in argv:
        argv = argv[argv.index('--') + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description='Compute collision metrics for a scene')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output JSON file path')
    args = parser.parse_args(argv)

    # Try to infer output path from blend file if not provided
    blend_path = bpy.data.filepath
    if args.output is None and blend_path:
        base_dir = os.path.dirname(blend_path)
        blend_name = os.path.basename(blend_path)
        if blend_name.startswith('scene_') and blend_name.endswith('.blend'):
            iter_num = blend_name[6:-6]
            args.output = os.path.join(base_dir, f'metric_{iter_num}.json')

    results = compute_metrics()

    print(f"\nResults:")
    print(f"  Total mesh objects: {results['Nobj']}")
    print(f"  Unique logical objects: {results['Nobj_unique']}")
    print(f"  Factory objects (bed+parts=1): {results['Nobj_factory']}")
    print(f"  Out of bounds: {results['OOB']}")
    print(f"  Collisions (raw): {results['BBL']}")
    print(f"  Collisions (internal mesh, filtered): {results['BBL_internal_mesh']}")
    print(f"  Collisions (component, e.g. pillow vs pillow): {results['BBL_component']}")
    print(f"  Collisions (inter-object): {results['BBL_inter_object']}")
    print(f"  Collisions (real unique pairs): {results['BBL_real_unique']}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to: {args.output}")

    return results


if __name__ == '__main__':
    main()
