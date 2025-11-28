from pathlib import Path
import subprocess
import os
import shutil
import sys

# Check if bpy is available as a Python package
try:
    import bpy
    BPY_AVAILABLE = True
except ImportError:
    BPY_AVAILABLE = False

# Use environment variable or find blender in system PATH
PATH_TO_BLENDER = os.environ.get(
    "BLENDER_PATH",
    shutil.which("blender") or "/snap/bin/blender"
)

# Prefer bpy package over subprocess by default
USE_BPY_PACKAGE = os.environ.get("ACDC_USE_BPY", "1") == "1" and BPY_AVAILABLE


def _run_render_script_direct(render_script, bkg, candidate, dataset, start_angle, end_angle, cnt):
    """Run the render script directly using bpy package."""
    # Set up sys.argv as the script expects
    original_argv = sys.argv.copy()
    sys.argv = [str(render_script), str(bkg), candidate, dataset, str(start_angle), str(end_angle), str(cnt)]

    try:
        # Add the blend directory to path for imports
        script_dir = os.path.dirname(os.path.abspath(render_script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        # Execute the render script
        with open(render_script) as f:
            exec(compile(f.read(), render_script, 'exec'), {'__name__': '__main__'})
    finally:
        sys.argv = original_argv


def _run_generation_script_direct(render_script, config_step3_filename):
    """Run the generation script directly using bpy package."""
    original_argv = sys.argv.copy()
    sys.argv = [str(render_script), config_step3_filename]

    try:
        script_dir = os.path.dirname(os.path.abspath(render_script))
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        with open(render_script) as f:
            exec(compile(f.read(), render_script, 'exec'), {'__name__': '__main__'})
    finally:
        sys.argv = original_argv


def open_blender_for_render(render_script, candidate, dataset, start_angle, end_angle, cnt, bkg=1):
    """Open blender for rendering. Uses bpy package if available, otherwise subprocess."""
    if USE_BPY_PACKAGE:
        print(f"Using bpy package for rendering (bpy version: {bpy.app.version_string})")
        _run_render_script_direct(render_script, bkg, candidate, dataset, start_angle, end_angle, cnt)
    else:
        # Fall back to subprocess
        if bkg == 0:
            process = subprocess.Popen([
                str(PATH_TO_BLENDER), '--python', str(render_script),
                str(bkg), candidate, dataset, str(start_angle), str(end_angle), str(cnt)
            ])
        else:
            process = subprocess.Popen([
                str(PATH_TO_BLENDER), '--background', '--python', str(render_script),
                str(bkg), candidate, dataset, str(start_angle), str(end_angle), str(cnt)
            ])
        process.wait()


def open_blender_for_generation(render_script, config_step3_filename):
    """Open blender for scene generation. Uses bpy package if available, otherwise subprocess."""
    if USE_BPY_PACKAGE:
        print(f"Using bpy package for generation (bpy version: {bpy.app.version_string})")
        _run_generation_script_direct(render_script, config_step3_filename)
    else:
        process = subprocess.Popen([
            str(PATH_TO_BLENDER), '--python', str(render_script), config_step3_filename
        ])
        process.wait()