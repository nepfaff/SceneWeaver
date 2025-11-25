# SceneWeaver Patches

This directory contains patches for modifications made to the codebase.

## blender-install-find-links.patch

**File Modified:** `scripts/install/interactive_blender.sh`

**Purpose:** Adds `--find-links https://download.blender.org/pypi/bpy/` to the pip install command when installing the package into Blender's bundled Python environment.

**Why this is needed:**
- When installing infinigen into Blender's Python, it tries to install all dependencies from `pyproject.toml`
- One of the dependencies is `bpy==3.6.0`
- Blender's pip doesn't automatically know about the Blender package repository
- Without the `--find-links` flag, pip can't find bpy and the installation fails

**Is infinigen external?**
No, the `infinigen` directory is part of the SceneWeaver repository (it's a fork/adaptation of the original Infinigen). Only some subdirectories are git submodules (like `infinigen/OcMesher`, `infinigen/infinigen_gpl`, etc.). Therefore, this modification is part of SceneWeaver's customizations.

**Applying the patch:**
```bash
cd /path/to/SceneWeaver
git apply scripts/patches/blender-install-find-links.patch
```

**Reverting the patch:**
```bash
cd /path/to/SceneWeaver
git apply -R scripts/patches/blender-install-find-links.patch
```

**Alternative approaches:**

If you want to avoid modifying the install script, you can:

1. **Remove bpy from pyproject.toml** (since Blender already includes bpy):
   - Remove line 25 in `pyproject.toml`
   - But this will cause issues when installing in other Python environments

2. **Set environment variable before running install script:**
   ```bash
   export PIP_FIND_LINKS=https://download.blender.org/pypi/bpy/
   bash scripts/install/interactive_blender.sh
   ```

3. **Skip bpy installation** by modifying the pip install command to use `--no-deps` and manually install other dependencies

**Recommendation:** Keep the modification as-is. It's a minimal change that ensures the installation works correctly for all users.
