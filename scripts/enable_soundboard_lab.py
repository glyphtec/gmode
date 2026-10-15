"""One-time startup script: re-register and enable the Soundboard Lab
extension after its dev junction was recreated. Not a pytest test.

Run with: blender --python scripts\\enable_soundboard_lab.py
(leaves Blender open afterwards; this only runs once at startup)
"""
import bpy

REPO_ID = "vscode_development"
PKG_ID = "soundboard_lab"
MODULE_NAME = f"bl_ext.{REPO_ID}.{PKG_ID}"

# Rescan local repositories so Blender notices the recreated junction.
try:
    bpy.ops.preferences.extension_repo_refresh_all()
    print("Repo refresh: OK")
except Exception as exc:
    print("Repo refresh failed (continuing):", exc)

try:
    bpy.ops.preferences.addon_enable(module=MODULE_NAME)
    print("Enabled:", MODULE_NAME)
except Exception as exc:
    print("Enable failed:", exc)

bpy.ops.wm.save_userpref()
print("Preferences saved.")
