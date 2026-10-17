"""Standalone smoke test for Soundboard Lab's Blender extension hooks.

Run with: blender --background --python scripts\addon_smoke_test.py
Not a pytest test (requires bpy / Blender runtime) - temporary dev script.
"""
import sys

sys.path.insert(0, r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\addon")

import bmesh
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
assert len(n_panel_categories) == 7, f"expected 7 N-panel panels, got {len(n_panel_categories)}"

# Properties-editor (Object tab) mini-panel.
assert addon.SOUNDBOARD_PT_object_role.bl_space_type == 'PROPERTIES'
assert addon.SOUNDBOARD_PT_object_role.bl_context == 'object'

# Properties-editor (Material tab) mini-panel.
assert addon.SOUNDBOARD_PT_material.bl_space_type == 'PROPERTIES'
assert addon.SOUNDBOARD_PT_material.bl_context == 'material'

# --- Grain orientation: property, material-kind gating, align operators ---

assert "grain_rotation" in obj.soundboard_lab.bl_rna.properties
assert tuple(obj.soundboard_lab.grain_rotation) == (0.0, 0.0, 0.0)

mat = bpy.data.materials.new("Spruce Proxy")
assert hasattr(mat, "soundboard_lab"), "Material.soundboard_lab missing"
assert mat.soundboard_lab.kind == 'ISOTROPIC'  # default
obj.data.materials.append(mat)
obj.soundboard_lab.role = 'LINING'

# Lining is not orthotropic-capable even with an orthotropic material.
mat.soundboard_lab.kind = 'ORTHOTROPIC'
assert not addon.is_grain_orientable(obj), "LINING role must not be grain-orientable"

# Soundboard + isotropic material: still not grain-orientable.
obj.soundboard_lab.role = 'SOUNDBOARD'
mat.soundboard_lab.kind = 'ISOTROPIC'
assert not addon.is_grain_orientable(obj), "isotropic material must not be grain-orientable"

# Soundboard + orthotropic material: grain-orientable, gizmo group polls True.
mat.soundboard_lab.kind = 'ORTHOTROPIC'
assert addon.is_grain_orientable(obj)
# poll() is plain Python and testable headlessly; full registration into
# bpy.types needs a window manager, unavailable under --background (see
# the bpy.app.background guard in register()/unregister()) and is instead
# verified manually in the GUI.
assert addon.SOUNDBOARD_GGT_grain_rosette.poll(bpy.context), "rosette gizmo should poll True"

obj.soundboard_lab.grain_rotation = (0.3, 0.4, 0.5)
result = bpy.ops.soundboard.align_grain_world()
assert result == {'FINISHED'}, f"align_grain_world returned {result}"
assert tuple(obj.soundboard_lab.grain_rotation) == (0.0, 0.0, 0.0), "grain not reset to world axes"

bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='DESELECT')
bm = bmesh.from_edit_mesh(obj.data)
bm.edges.ensure_lookup_table()
bm.edges[0].select = True
bm.select_history.add(bm.edges[0])
bpy.context.view_layer.update()
assert addon.SOUNDBOARD_OT_align_grain_to_edge.poll(bpy.context), "align-to-edge should poll True"
result = bpy.ops.soundboard.align_grain_to_edge()
assert result == {'FINISHED'}, f"align_grain_to_edge returned {result}"
assert tuple(obj.soundboard_lab.grain_rotation) != (0.0, 0.0, 0.0), "grain not aligned to edge"
bpy.ops.object.mode_set(mode='OBJECT')

print("=== Soundboard Lab smoke test: unregister ===")
addon.unregister()
assert not hasattr(bpy.types.Object, "soundboard_lab"), "Object.soundboard_lab not cleaned up"
assert not hasattr(bpy.types.Material, "soundboard_lab"), "Material.soundboard_lab not cleaned up"

print("SMOKE TEST PASSED")
