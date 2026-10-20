"""Standalone smoke test for Soundboard Lab's Blender extension hooks.

Run with: blender --background --python scripts\addon_smoke_test.py
Not a pytest test (requires bpy / Blender runtime) - temporary dev script.
"""
import sys

sys.path.insert(0, r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\addon")

import bmesh
import bpy
import soundboard_lab as addon
from pathlib import Path

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

# --- export_triangulated_stl: must bake world transform, not local ---
# Real bug, caught via a real round-trip test with the user: a rotated
# +translated object's exported geometry came out wrong (local-space
# coordinates, e.g. its thin axis in the wrong direction) because
# `mesh.vertices[i].co` is local/object-space and was never multiplied
# by `matrix_world`. A non-identity transform (rotated, moved off
# origin) is completely normal for a real modeled object, not an edge
# case -- this must be verified, not just assumed fixed. Non-uniform
# scale on each axis (not a symmetric cube) matters here: a symmetric
# cube's bounding box is unchanged by a 90-degree rotation, so it
# wouldn't actually catch a missing/wrong rotation, only a missing
# translation.
import math

bpy.ops.mesh.primitive_cube_add(size=2.0)  # local bounds: [-1, 1]^3
transformed_obj = bpy.context.active_object
transformed_obj.scale = (1.0, 2.0, 4.0)  # local bounds become X:[-1,1] Y:[-2,2] Z:[-4,4]
transformed_obj.location = (10.0, 20.0, 30.0)
transformed_obj.rotation_euler = (math.radians(90), 0.0, 0.0)  # Rx(90): (x,y,z) -> (x,-z,y)
bpy.context.view_layer.update()

stl_path = Path(bpy.app.tempdir) / "transform_check.stl"
tri_count = addon.analysis.export_triangulated_stl(bpy.context, transformed_obj, stl_path)
assert tri_count == 12, f"expected 12 triangles for a cube, got {tri_count}"

stl_text = stl_path.read_text()
vertex_lines = [line for line in stl_text.splitlines() if line.strip().startswith("vertex")]
coords = [[float(n) for n in line.split()[1:4]] for line in vertex_lines]
xs, ys, zs = zip(*coords)
world_bounds_min = (min(xs), min(ys), min(zs))
world_bounds_max = (max(xs), max(ys), max(zs))
# Scale (1,2,4) then Rx(90) [(x,y,z)->(x,-4z,2y)] then translate (10,20,30):
# world X = x+10 in [9,11]; world Y = -4z+20 in [16,24]; world Z = 2y+30 in [28,32].
expected_min = (9.0, 16.0, 28.0)
expected_max = (11.0, 24.0, 32.0)
assert all(abs(a - b) < 1e-4 for a, b in zip(world_bounds_min, expected_min)), (
    f"exported STL is not in world space: got min {world_bounds_min}, expected {expected_min}"
)
assert all(abs(a - b) < 1e-4 for a, b in zip(world_bounds_max, expected_max)), (
    f"exported STL is not in world space: got max {world_bounds_max}, expected {expected_max}"
)
print("export_triangulated_stl world-transform check passed")

# --- Analysis & Solve: real end-to-end subprocess pipeline ---
# Unlike the gizmo, this needs no window manager -- it's just
# subprocess launching -- so it can be fully exercised headlessly,
# including actually running gmsh + ccx for real. Uses the real
# neck-block-notch fixture (not the synthetic cube) so the shell/
# extrusion path actually gets exercised on representative geometry
# (soundhole, sharp notch corners, the known reversed-winding-
# duplicate defect), not a trivial thick cube.
import time

obj.soundboard_lab.role = 'NONE'  # free up the sole-Soundboard slot temporarily
bpy.ops.wm.stl_import(
    filepath=r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\tests\data\real_soundboard_with_neck_block_notch.stl"
)
fixture_obj = bpy.context.selected_objects[0]
fixture_obj.soundboard_lab.role = 'SOUNDBOARD'
fixture_obj.soundboard_lab.mesh_element_size = 15.0

CCX_EXE = r"C:\Program Files\FreeCAD 1.1\bin\ccx.exe"
ENGINE_PYTHON = r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\.venv\Scripts\python.exe"

assert addon.analysis.find_soundboard_object(bpy.context).name == fixture_obj.name

handle = addon.analysis.start_solve(bpy.context, fixture_obj, CCX_EXE, ENGINE_PYTHON)
result = None
for _ in range(600):  # up to ~5 minutes -- real shell-mesh solves can be slow
    status = addon.analysis.read_solve_status(handle)
    result = addon.analysis.poll_solve(handle)
    if result is not None:
        break
    time.sleep(0.5)
assert result is not None, "engine subprocess did not finish in time"
assert result["status"] == "ok", result
assert result["triangle_count"] > 1
assert len(result["modes"]) == 10
assert "mesh_shape" in result and "mode_shapes" in result
assert len(result["mesh_shape"]["xy"]) == len(result["mode_shapes"]["1"])
print("Shell-path analysis result: status=ok, triangle_count=", result["triangle_count"])

# read_solve_status must have surfaced a real step checklist at some
# point while the subprocess was running (not just at the very end).
assert status is not None, "expected at least one non-None step-status read while the solve was in progress"
assert {s["id"] for s in status["steps"]} == {"repair", "mesh", "write_input", "solve", "parse_results"}

# _apply_result is what the real timer callback uses to populate scene
# properties (poll_solve itself just returns the raw dict) -- verify
# that wiring directly since the pipeline test above bypassed it.
addon.analysis._apply_result(bpy.context.scene, result)
assert bpy.context.scene.soundboard_lab_status == 'DONE'
assert len(bpy.context.scene.soundboard_lab_results) == 10
assert bpy.context.scene.soundboard_lab_results[0].mode_number == 1
# approx, not == : Blender's FloatProperty is 32-bit internally, so an
# exact round-trip against the original 64-bit JSON value isn't
# guaranteed once a real (not suspiciously-round) frequency is stored.
stored_freq = bpy.context.scene.soundboard_lab_results[9].frequency_hz
expected_freq = result["modes"][9]["frequency_hz"]
assert abs(stored_freq - expected_freq) < 1e-3 * abs(expected_freq), (stored_freq, expected_freq)

# Selecting a mode (as the UIList binding does via its update callback)
# must build/replace the real "Soundboard Mode Shape Preview" object.
bpy.context.scene.soundboard_lab_selected_mode_index = 0
preview_obj = bpy.data.objects.get("Soundboard Mode Shape Preview")
assert preview_obj is not None, "selecting a mode should build the mode-shape preview object"
assert len(preview_obj.data.vertices) == len(result["mesh_shape"]["xy"])
assert "ModeShapeZ" in preview_obj.data.color_attributes
print("Mode selection -> preview rebuild check passed")

error_result = {"status": "error", "message": "synthetic test error"}
addon.analysis._apply_result(bpy.context.scene, error_result)
assert bpy.context.scene.soundboard_lab_status == 'ERROR'
assert bpy.context.scene.soundboard_lab_message == "synthetic test error"
assert len(bpy.context.scene.soundboard_lab_results) == 0

bpy.data.objects.remove(fixture_obj, do_unlink=True)
obj.soundboard_lab.role = 'SOUNDBOARD'  # restore state the rest of this script expects

# --- Mode-shape heat-map preview: static Color Attribute on a real object ---
# Not an animated/exaggerated deformation (explicitly not wanted, per the
# user) -- a real 3D mesh object with a Z-displacement heat map, viewable
# from any angle. Synthetic 4-vertex case with a known gradient: exact
# min/mid/max colors are checked, not just "didn't crash".
preview_xy = [[0, 0], [1, 0], [1, 1], [0, 1]]
preview_top_z = [0.0, 0.0, 0.0, 0.0]
preview_triangles = [[0, 1, 2], [0, 2, 3]]
preview_dz = [-5.0, 0.0, 5.0, -2.5]  # vertex 0 = min (blue), vertex 2 = max (red), vertex 1 = mid (green)

preview_obj = addon.analysis.build_mode_shape_preview(
    bpy.context, "Smoke Test Mode Preview", preview_xy, preview_top_z, preview_triangles, preview_dz
)
assert len(preview_obj.data.vertices) == 4
assert len(preview_obj.data.polygons) == 2
color_attr = preview_obj.data.color_attributes["ModeShapeZ"]
assert tuple(color_attr.data[0].color) == (0.0, 0.0, 1.0, 1.0), "min displacement should be pure blue"
assert tuple(color_attr.data[1].color) == (0.0, 1.0, 0.0, 1.0), "mid displacement should be pure green"
assert tuple(color_attr.data[2].color) == (1.0, 0.0, 0.0, 1.0), "max displacement should be pure red"

# Re-calling with the same name must replace, not duplicate, the object.
addon.analysis.build_mode_shape_preview(
    bpy.context, "Smoke Test Mode Preview", preview_xy, preview_top_z, preview_triangles, preview_dz
)
assert len([o for o in bpy.data.objects if o.name.startswith("Smoke Test Mode Preview")]) == 1

# Asymmetric (one-signed) mode case -- a real, confirmed scenario, not
# hypothetical: a real solve's mode 1 ranged -244.6 to +0.18, almost
# entirely negative. Plain min/max normalization would put the
# clamped boundary's zero displacement near the RED (max) end instead
# of the visual middle, making a zero-displacement edge look "hot".
# Symmetric-around-zero normalization must keep zero at green
# regardless of which way a mode happens to be signed.
asymmetric_dz = [-10.0, 0.0, 1.0, -5.0]  # bound = max(|-10|, |1|) = 10
addon.analysis.build_mode_shape_preview(
    bpy.context, "Smoke Test Asymmetric Mode", preview_xy, preview_top_z, preview_triangles, asymmetric_dz
)
asym_attr = bpy.data.objects["Smoke Test Asymmetric Mode"].data.color_attributes["ModeShapeZ"]
assert tuple(asym_attr.data[1].color) == (0.0, 1.0, 0.0, 1.0), "zero displacement must stay green even when a mode is one-signed"
assert tuple(asym_attr.data[0].color) == (0.0, 0.0, 1.0, 1.0), "the most-negative vertex should still be pure blue"

print("build_mode_shape_preview check passed")


# Scene-collection errors: zero candidates, then multiple candidates.
obj.soundboard_lab.role = 'RIB'
try:
    addon.analysis.find_soundboard_object(bpy.context)
    raise AssertionError("expected SoundboardLabSceneError for zero candidates")
except addon.analysis.SoundboardLabSceneError:
    pass
obj.soundboard_lab.role = 'SOUNDBOARD'

bpy.ops.mesh.primitive_cube_add()
obj2 = bpy.context.active_object
obj2.soundboard_lab.role = 'SOUNDBOARD'
try:
    addon.analysis.find_soundboard_object(bpy.context)
    raise AssertionError("expected SoundboardLabSceneError for multiple candidates")
except addon.analysis.SoundboardLabSceneError:
    pass
bpy.data.objects.remove(obj2, do_unlink=True)

print("=== Soundboard Lab smoke test: unregister ===")
addon.unregister()
assert not hasattr(bpy.types.Object, "soundboard_lab"), "Object.soundboard_lab not cleaned up"
assert not hasattr(bpy.types.Material, "soundboard_lab"), "Material.soundboard_lab not cleaned up"

print("SMOKE TEST PASSED")
