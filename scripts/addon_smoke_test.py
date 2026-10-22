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
assert len(n_panel_categories) == 6, f"expected 6 N-panel panels, got {len(n_panel_categories)}"

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

# --- Material physical properties: fields, defaults, extraction ---
def approx(a, b, tol=1e-6):
    # Blender's FloatProperty is 32-bit internally -- a round-tripped
    # value isn't guaranteed to exactly `==` the original 64-bit
    # Python/JSON literal for a non-round decimal (confirmed earlier
    # this session: exact comparison is fragile for real frequency
    # values too, same underlying reason).
    return abs(a - b) < tol


ortho_mat = bpy.data.materials.new("Test Orthotropic")
ortho_mat.soundboard_lab.kind = 'ORTHOTROPIC'
# Defaults should be SPRUCE_PLACEHOLDER's own constants, an editable
# starting point (not meaningless zeros) -- see docs/CLAUDE.md.
assert ortho_mat.soundboard_lab.density == 460.0  # exact integer, representable in float32
assert ortho_mat.soundboard_lab.e_longitudinal == 13800.0
ortho_dict = addon.extract_material(ortho_mat)
assert ortho_dict["kind"] == "ORTHOTROPIC"
assert ortho_dict["name"] == "Test Orthotropic"
# kg/m^3 -> t/mm^3 conversion (factor 1e-12), matching SPRUCE_PLACEHOLDER's
# own documented 460 kg/m^3 == 4.6e-10 t/mm^3 exactly.
assert approx(ortho_dict["density"], 4.6e-10, tol=1e-13), ortho_dict["density"]
assert ortho_dict["e_longitudinal"] == 13800.0  # MPa, no conversion
assert approx(ortho_dict["nu_lr"], 0.38)

iso_mat = bpy.data.materials.new("Test Isotropic")
assert iso_mat.soundboard_lab.kind == 'ISOTROPIC'  # default
assert iso_mat.soundboard_lab.iso_e == 10000.0
iso_dict = addon.extract_material(iso_mat)
assert iso_dict["kind"] == "ISOTROPIC"
assert iso_dict["name"] == "Test Isotropic"
# density is a single field shared by both kinds (same physical meaning
# regardless of kind) -- its default (460.0 kg/m^3) matches the
# orthotropic/spruce placeholder, not a separate isotropic-specific one.
assert approx(iso_dict["density"], 4.6e-10, tol=1e-13)
assert iso_dict["e"] == 10000.0
assert approx(iso_dict["nu"], 0.3)
print("extract_material (both kinds) check passed")

# --- Grain orientation -> *ORIENTATION vector resolution ---
bpy.ops.mesh.primitive_cube_add()
grain_obj = bpy.context.active_object
grain_obj.soundboard_lab.grain_rotation = (0.0, 0.0, 0.0)
identity_orientation = addon.analysis._compute_grain_orientation(grain_obj)
assert all(abs(a - b) < 1e-9 for a, b in zip(identity_orientation, (1.0, 0.0, 0.0, 0.0, 1.0, 0.0))), (
    identity_orientation
)

# A 90-degree Z rotation on the OBJECT ITSELF (not grain_rotation) must
# carry through to the resolved world-space vectors -- world X -> world Y,
# world Y -> world -X (right-hand rotation about Z).
grain_obj.rotation_euler = (0.0, 0.0, math.radians(90))
bpy.context.view_layer.update()
rotated_orientation = addon.analysis._compute_grain_orientation(grain_obj)
expected = (0.0, 1.0, 0.0, -1.0, 0.0, 0.0)
assert all(abs(a - b) < 1e-6 for a, b in zip(rotated_orientation, expected)), (rotated_orientation, expected)
print("_compute_grain_orientation check passed")

# --- Global solution parameters: Scene-level, project-persisted ---
# num_layers is deliberately NOT here any more -- moved to per-object
# (mesh_num_layers, alongside mesh_element_size/mesh_thickness_hint)
# since each independently-meshed body may need its own layer count
# for its own thickness, once multi-body meshing is in play.
assert bpy.context.scene.soundboard_lab_n_frequencies == 10
assert bpy.context.scene.soundboard_lab_boundary_condition == 'FIXED_EDGE'
bpy.context.scene.soundboard_lab_n_frequencies = 6
assert bpy.context.scene.soundboard_lab_n_frequencies == 6
assert "mesh_num_layers" in obj.soundboard_lab.bl_rna.properties
assert obj.soundboard_lab.mesh_num_layers == 3  # default
print("Global solution parameters (Scene-level) check passed")

# --- 'Body/Lining' boundary condition: validated, not yet implemented ---
# check_body_lining_boundary_condition is a no-op unless BODY_LINING is
# actually selected -- must not raise for the default/other choices.
addon.analysis.check_body_lining_boundary_condition(bpy.context)

# Deterministic regardless of this machine's own default startup scene
# (which may carry leftover role assignments from past interactive
# testing) -- explicitly clear any stray Lining-role objects first.
for stray in bpy.context.scene.objects:
    if stray.soundboard_lab.role == 'LINING':
        stray.soundboard_lab.role = 'NONE'

bpy.context.scene.soundboard_lab_boundary_condition = 'BODY_LINING'
try:
    addon.analysis.check_body_lining_boundary_condition(bpy.context)
    assert False, "expected a SoundboardLabSceneError (no Body/Lining-role object exists yet)"
except addon.analysis.SoundboardLabSceneError as exc:
    assert "no body/lining" in str(exc).lower(), str(exc)

# Add a real Body/Lining-role object: the error must now be about it not
# being implemented yet, not about a missing object.
lining_obj = bpy.data.objects.new("Lining Test Object", bpy.data.meshes.new("Lining Test Mesh"))
bpy.context.collection.objects.link(lining_obj)
lining_obj.soundboard_lab.role = 'LINING'
try:
    addon.analysis.check_body_lining_boundary_condition(bpy.context)
    assert False, "expected a SoundboardLabSceneError (not yet implemented)"
except addon.analysis.SoundboardLabSceneError as exc:
    assert "not yet implemented" in str(exc).lower(), str(exc)

bpy.data.objects.remove(lining_obj, do_unlink=True)
bpy.context.scene.soundboard_lab_boundary_condition = 'FIXED_EDGE'
print("Body/Lining boundary condition validation check passed")

# --- Add-on prefs fallback cache: survives Blender's own prefs being lost ---
# Real bug, observed live twice in one session: the ccx/engine-python
# paths in AddonPreferences disappeared across routine dev-reload
# cycles, even with "Save Preferences" already clicked and Auto-Save
# Preferences already on -- not user error. This cache (a plain file
# outside anything Blender's extension-install bookkeeping manages) is
# the mitigation. Monkeypatch the cache PATH (not the save/load logic)
# to a tmp file so this test can't clobber the real user's actual
# cached prefs on this machine.
_real_prefs_cache_path = addon.analysis._prefs_cache_path
_test_cache_path = Path(bpy.app.tempdir) / "soundboard_lab_prefs_cache_TEST.json"
addon.analysis._prefs_cache_path = lambda: _test_cache_path
try:
    addon.analysis._save_prefs_cache("C:\\fake\\ccx.exe", "C:\\fake\\python.exe")
    loaded = addon.analysis._load_prefs_cache()
    assert loaded == {"ccx_path": "C:\\fake\\ccx.exe", "engine_python_path": "C:\\fake\\python.exe"}, loaded
    print("Prefs fallback cache save/load check passed")
finally:
    addon.analysis._prefs_cache_path = _real_prefs_cache_path
    if _test_cache_path.is_file():
        _test_cache_path.unlink()

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

# Real GUI bug, confirmed live: a fixed-size surface_offset alone isn't
# enough -- depth-buffer precision degrades with camera distance, so
# the same offset that looks fine up close Z-fights again once zoomed
# out. The real fix is to hide the original soundboard object while a
# preview is showing (no two coincident solid surfaces on screen at
# once), and restore it as soon as a new solve starts.
assert fixture_obj.hide_get(), "soundboard object should be hidden once its mode-shape preview is showing"
addon.analysis._set_soundboard_visibility(fixture_obj, visible=True)
assert not fixture_obj.hide_get(), "start_solve restoring visibility should actually unhide the object"
print("Soundboard visibility hide/restore check passed")

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

# Real GUI bug, confirmed: the preview mesh sitting at the exact same
# world-space Z as the original soundboard's top surface Z-fights with
# it in the viewport. Must sit strictly above the real top_z, not
# coincide with it.
for v, expected_top_z in zip(preview_obj.data.vertices, preview_top_z):
    assert v.co.z > expected_top_z, "preview mesh must be offset above the real top surface, not coincide with it"

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
