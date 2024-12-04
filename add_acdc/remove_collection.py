import bpy

# Specify the name of the object you want to keep
object_to_keep = "SimpleDeskFactory(8569017).spawn_asset(5056988)"  # Replace with the name of your object


def keep_object_from_collections(collection, target_name):
    # 递归检查子集合
    for child_collection in collection.children:
        keep_object_from_collections(child_collection, target_name)
        bpy.data.collections.remove(child_collection)

    # 检查当前集合中的对象
    for obj in collection.objects:
        if obj.name != target_name:
            bpy.data.objects.remove(obj, do_unlink=True)
        else:
            obj.location = [0, 0, 0]
            obj.rotation_euler = [0, 0, 0]
            c.objects.link(obj)
            collection.objects.unlink(obj)

    return


global c
c = bpy.context.scene.collection
keep_object_from_collections(c, object_to_keep)

save_path = "debug.blend"
bpy.ops.wm.save_as_mainfile(filepath=save_path)
