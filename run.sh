#!/bin/bash
conda deactivate
cd /home/yandan/workspace/infinigen
conda activate infinigen_python
# python -m infinigen.launch_blender -m infinigen_examples.generate_indoors -- --seed 0 --task coarse --output_folder outputs/indoors/debug -g debug.gin overhead.gin singleroom.gin -p compose_indoors.terrain_enabled=False compose_indoors.overhead_cam_enabled=True restrict_solving.solve_max_rooms=1 compose_indoors.invisible_room_ceilings_enabled=True compose_indoors.restrict_single_supported_roomtype=True 
# python -m infinigen_examples.generate_indoors -- --seed 0 --task coarse --output_folder outputs/indoors/debug -g debug.gin overhead.gin singleroom.gin -p compose_indoors.terrain_enabled=False compose_indoors.overhead_cam_enabled=True restrict_solving.solve_max_rooms=1 compose_indoors.invisible_room_ceilings_enabled=True compose_indoors.restrict_single_supported_roomtype=True 

#python -m infinigen.launch_blender -m infinigen_examples.generate_indoors -- --seed 0 --task coarse --output_folder outputs/indoors/coarse_expand_whole_nobedframe -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False

# python -m infinigen.tools.export --input_folder outputs/indoors/coarse_p --output_folder outputs/my_export_coarse_p -f usdc -r 1024 --omniverse

# python examples/isaac_sim.py --scene-path /home/yandan/workspace/infinigen/outputs/my_export_coarse_p/export_scene.blend/export_scene.usdc --json-path /home/yandan/workspace/infinigen/outputs/my_export_coarse_p/export_scene.blend/solve_state_acdc.json 
# python examples/isaac_sim.py --scene-path /home/yandan/workspace/infinigen/outputs/debug2/export_scene.blend/export_scene.usdc --json-path /home/yandan/workspace/infinigen/outputs/my_export_debug3/solve_state.json 

# /home/yandan/workspace/infinigen/blender/3.6/python/bin/python3.10 -m pip install dill

# blender viewport
python -m infinigen.launch_blender -m infinigen_examples.generate_indoors -- --seed 0 --task coarse --output_folder outputs/indoors/coarse_expand_whole_nobedframe -g fast_solve.gin overhead.gin studio.gin -p compose_indoors.terrain_enabled=False


# ffmpeg -framerate 2 -i render_%d.jpg -vf "scale=1280:-1:flags=lanczos" -loop 0 output.gif
# from infinigen_examples.util.visible import invisible_others, visible_others

# invisible_others(hide_placeholder=True)
# bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
# visible_others()