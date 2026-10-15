"""Standalone smoke test for Soundboard Lab's Blender extension hooks.

Run with: blender --background --python scripts\addon_smoke_test.py
Not a pytest test (requires bpy / Blender runtime) - temporary dev script.
"""
import sys

sys.path.insert(0, r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\addon")

import bpy
import soundboard_lab as addon

print("=== Soundboard Lab smoke test: register ===")
addon.register()

assert hasattr(bpy.types.Object, "soundboard_lab"), "Object.soundboard_lab missing"
assert "set_role_lining" in dir(bpy.ops.soundboard), "operator not registered"

bpy.ops.mesh.primitive_cube_add()
obj = bpy.context.active_object
obj.soundboard_lab.role = 'SOUNDBOARD'
assert obj.soundboard_lab.role == 'SOUNDBOARD'

bpy.ops.object.select_all(action='SELECT')
result = bpy.ops.soundboard.set_role_lining()
assert result == {'FINISHED'}, f"operator returned {result}"
assert obj.soundboard_lab.role == 'LINING', "set_role_lining did not set role"

# Terminal condition defaults to Fixed; Spring exposes a stiffness field.
assert obj.soundboard_lab.terminal_condition == 'FIXED'
obj.soundboard_lab.terminal_condition = 'SPRING'
obj.soundboard_lab.terminal_stiffness = 12.5
assert obj.soundboard_lab.terminal_stiffness == 12.5

# N-panel ("Soundboard" tab, VIEW_3D) panels.
n_panel_categories = {
    cls.__name__: getattr(cls, "bl_category", None)
    for cls in bpy.types.Panel.__subclasses__()
    if cls.__name__.startswith("SOUNDBOARD_PT_") and cls.bl_space_type == 'VIEW_3D'
}
print("Registered Soundboard N-panel panels:", n_panel_categories)
assert all(cat == "Soundboard" for cat in n_panel_categories.values())
assert len(n_panel_categories) == 6, f"expected 6 N-panel panels, got {len(n_panel_categories)}"

# Properties-editor (Object tab) mini-panel.
assert addon.SOUNDBOARD_PT_object_role.bl_space_type == 'PROPERTIES'
assert addon.SOUNDBOARD_PT_object_role.bl_context == 'object'

print("=== Soundboard Lab smoke test: unregister ===")
addon.unregister()
assert not hasattr(bpy.types.Object, "soundboard_lab"), "Object.soundboard_lab not cleaned up"

print("SMOKE TEST PASSED")
