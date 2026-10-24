"""Standalone smoke test for Soundboard Lab's Blender extension hooks.

Run with: blender --background --python scripts\addon_smoke_test.py
Not a pytest test (requires bpy / Blender runtime) - temporary dev script.
"""
import sys

sys.path.insert(0, r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\addon")

import bmesh
import bpy
from mathutils import Matrix
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

# --- 'Body/Lining' boundary condition: scene validation ---
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

# Add a real Body/Lining-role object: the check now passes; a second
# lining is ambiguous and must be rejected.
lining_obj = bpy.data.objects.new("Lining Test Object", bpy.data.meshes.new("Lining Test Mesh"))
bpy.context.collection.objects.link(lining_obj)
lining_obj.soundboard_lab.role = 'LINING'
addon.analysis.check_body_lining_boundary_condition(bpy.context)
lining_obj2 = bpy.data.objects.new("Lining Test Object 2", bpy.data.meshes.new("Lining Test Mesh 2"))
bpy.context.collection.objects.link(lining_obj2)
lining_obj2.soundboard_lab.role = 'LINING'
try:
    addon.analysis.check_body_lining_boundary_condition(bpy.context)
    assert False, "expected a SoundboardLabSceneError (two Body/Lining objects)"
except addon.analysis.SoundboardLabSceneError as exc:
    assert "multiple" in str(exc).lower(), str(exc)
bpy.data.objects.remove(lining_obj2, do_unlink=True)

# Spring terminal without a stiffness is rejected before any solve.
lining_obj.soundboard_lab.terminal_condition = 'SPRING'
try:
    addon.bonds.terminal_spec(lining_obj)
    assert False, "expected a SoundboardLabSceneError (zero spring stiffness)"
except addon.analysis.SoundboardLabSceneError as exc:
    assert "stiffness" in str(exc).lower(), str(exc)
lining_obj.soundboard_lab.terminal_stiffness = 5000.0
assert addon.bonds.terminal_spec(lining_obj) == {"condition": "spring", "stiffness": 5000.0}

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
preview_objs = [o for o in bpy.data.objects if o.soundboard_lab.generated_kind == 'MODE_SHAPE']
assert len(preview_objs) == 1, "selecting a mode should build exactly one tagged mode-shape preview object"
preview_obj = preview_objs[0]
assert len(preview_obj.data.vertices) == len(result["mesh_shape"]["xy"])
assert "ModeShapeZ" in preview_obj.data.color_attributes
_root, _bonding, results_coll = addon.generated.ensure_generated_collections(bpy.context.scene)
assert list(preview_obj.users_collection) == [results_coll], "preview must live only in Analysis Results"
assert preview_obj.soundboard_lab.source_object == fixture_obj
assert results_coll.get(addon.generated.COLLECTION_TAG) == "results"
assert _root.name in bpy.context.scene.collection.children
# Generated objects are never assembly inputs, even if a role is set.
preview_obj.soundboard_lab.role = 'SOUNDBOARD'
assert addon.analysis.find_soundboard_object(bpy.context) == fixture_obj
preview_obj.soundboard_lab.role = 'NONE'
print("Mode selection -> preview rebuild check passed")

# Real GUI bug, confirmed live: a fixed-size display offset isn't
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

# No display offset: generated objects sit on true geometry (Z-fighting
# with the authored soundboard is handled by hiding it, checked above).
for v, expected_top_z in zip(preview_obj.data.vertices, preview_top_z):
    assert v.co.z == expected_top_z, "preview mesh must sit exactly on the solved top surface (no display offset)"

# Re-calling must replace, not duplicate, the tagged object.
addon.analysis.build_mode_shape_preview(
    bpy.context, "Smoke Test Mode Preview", preview_xy, preview_top_z, preview_triangles, preview_dz
)
assert len([o for o in bpy.data.objects if o.soundboard_lab.generated_kind == 'MODE_SHAPE']) == 1

# Asymmetric (one-signed) mode case -- a real, confirmed scenario, not
# hypothetical: a real solve's mode 1 ranged -244.6 to +0.18, almost
# entirely negative. Plain min/max normalization would put the
# clamped boundary's zero displacement near the RED (max) end instead
# of the visual middle, making a zero-displacement edge look "hot".
# Symmetric-around-zero normalization must keep zero at green
# regardless of which way a mode happens to be signed.
asymmetric_dz = [-10.0, 0.0, 1.0, -5.0]  # bound = max(|-10|, |1|) = 10
asym_obj = addon.analysis.build_mode_shape_preview(
    bpy.context, "Smoke Test Asymmetric Mode", preview_xy, preview_top_z, preview_triangles, asymmetric_dz
)
asym_attr = asym_obj.data.color_attributes["ModeShapeZ"]
assert tuple(asym_attr.data[1].color) == (0.0, 1.0, 0.0, 1.0), "zero displacement must stay green even when a mode is one-signed"
assert tuple(asym_attr.data[0].color) == (0.0, 0.0, 1.0, 1.0), "the most-negative vertex should still be pure blue"

print("build_mode_shape_preview check passed")

# --- Bond detection: generated collections, proxies, results, staleness ---
# Real subprocess run on the bonding_exercise fixtures (ribs/bridge plate
# sit on the soundboard within the 0.1 tolerance).
BOND_DATA = Path(r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\tests\data\bonding_exercise")
obj.soundboard_lab.role = 'NONE'
bond_objs = {}
for stem, role, element_size in (
    ("soundboard", 'SOUNDBOARD', 10.0),
    ("rib1", 'RIB', 2.0),
    ("bridge_plate2", 'BRIDGE_PLATE', 2.0),
):
    bpy.ops.wm.stl_import(filepath=str(BOND_DATA / f"{stem}.stl"))
    imported = bpy.context.selected_objects[0]
    imported.soundboard_lab.role = role
    imported.soundboard_lab.mesh_element_size = element_size
    bond_objs[stem] = imported

assert addon.bonds.bond_color(0.0, 0.1)[:3] == (0.1, 0.9, 0.2), "exact contact should be green"
assert addon.bonds.bond_color(0.5, 0.1)[0] == 1.0 and addon.bonds.bond_color(0.5, 0.1)[1] < 0.2, "penetration is red"
assert addon.bonds.bond_color(-0.3, 0.1) == (1.0, 0.5, 0.0, 0.6), "near gap is orange"
assert addon.bonds.bond_color(-5.0, 0.1)[3] < 0.2, "far surface is faint"

bond_bodies = addon.bonds.find_bond_bodies(bpy.context)
assert {b.name for b in bond_bodies} == {bond_objs["rib1"].name, bond_objs["bridge_plate2"].name}
bond_handle = addon.bonds.start_bond_detection(
    bpy.context, bond_objs["soundboard"], bond_bodies, ENGINE_PYTHON, 0.1
)
bond_result = None
for _ in range(600):
    bond_result = addon.analysis.poll_solve(bond_handle)
    if bond_result is not None:
        break
    time.sleep(0.5)
assert bond_result is not None, "bond detection subprocess did not finish in time"
assert bond_result["status"] == "ok", bond_result
addon.bonds.apply_bond_result(bpy.context, bond_result, bond_handle)

scene = bpy.context.scene
assert scene.soundboard_lab_bond_status == 'DONE', scene.soundboard_lab_bond_message
bond_rows = {item.name: item for item in scene.soundboard_lab_bond_results}
assert set(bond_rows) == {bond_objs["rib1"].name, bond_objs["bridge_plate2"].name}
for name, item in bond_rows.items():
    assert item.status == 'CONTACT', (name, item.status, item.message)
    assert item.contact_area > 0
    assert item.proxy_object is not None
    assert item.source_object == bpy.data.objects[name]
print("Bond rows:", {n: (i.status, round(i.contact_area, 1)) for n, i in bond_rows.items()})

_root, bonding_coll, results_coll = addon.generated.ensure_generated_collections(scene)
assert bonding_coll.get(addon.generated.COLLECTION_TAG) == "bonding"
assert len([c for c in scene.collection.children if c.get(addon.generated.COLLECTION_TAG) == "root"]) == 1, \
    "collections must be found by tag, not duplicated"
proxies = [o for o in scene.objects if o.soundboard_lab.generated_kind in addon.generated.BOND_PROXY_KINDS]
assert len(proxies) == 3, [o.name for o in proxies]
for proxy in proxies:
    assert list(proxy.users_collection) == [bonding_coll]
    assert proxy.soundboard_lab.role == 'NONE'
    assert proxy.hide_render
    assert proxy.matrix_world == Matrix.Identity(4), "proxies sit at identity with world-space vertices"
    assert proxy.soundboard_lab.source_fingerprint
    assert not proxy.soundboard_lab.bond_stale
rib_proxy = bond_rows[bond_objs["rib1"].name].proxy_object
assert addon.bonds.BOND_DISTANCE_ATTR in rib_proxy.data.attributes
assert addon.bonds.BOND_STATUS_ATTR in rib_proxy.data.color_attributes
sb_proxy = next(o for o in proxies if o.soundboard_lab.generated_kind == 'SOUNDBOARD_PROXY')
assert sb_proxy.soundboard_lab.source_object == bond_objs["soundboard"]
assert addon.analysis.find_soundboard_object(bpy.context) == bond_objs["soundboard"], \
    "proxies must never count as assembly bodies"
assert len(addon.bonds.find_bond_bodies(bpy.context)) == 2
assert addon.bonds.check_bond_staleness(scene, bpy.context.evaluated_depsgraph_get()) == 0, \
    "fresh proxies must not read as stale"

# Refresh replaces only tagged proxies; user objects placed in the
# generated collection survive.
user_mesh = bpy.data.meshes.new("User Note Mesh")
user_obj = bpy.data.objects.new("User Note", user_mesh)
bonding_coll.objects.link(user_obj)
addon.bonds.apply_bond_result(bpy.context, bond_result, bond_handle)
assert bpy.data.objects.get("User Note") is not None, "refresh must not delete user objects"
assert user_obj.name in bonding_coll.objects
proxies = [o for o in scene.objects if o.soundboard_lab.generated_kind in addon.generated.BOND_PROXY_KINDS]
assert len(proxies) == 3, "refresh must replace, not accumulate, proxies"
assert not any(m.name.startswith("Bond Proxy") and m.users == 0 for m in bpy.data.meshes), "orphaned proxy meshes"
print("Bond proxies / collections / refresh check passed")

# Staleness: moving a rib flags its proxy (and its result row) out of date.
rib_obj = bond_objs["rib1"]
rib_row = scene.soundboard_lab_bond_results[rib_obj.name]
plate_row = scene.soundboard_lab_bond_results[bond_objs["bridge_plate2"].name]
rib_obj.location.z += 1.0
bpy.context.view_layer.update()
print("depsgraph handler flagged stale during view_layer.update():", rib_row.proxy_object.soundboard_lab.bond_stale)
assert addon.bonds.check_bond_staleness(scene, bpy.context.evaluated_depsgraph_get()) == 1
assert rib_row.proxy_object.soundboard_lab.bond_stale
assert addon.bonds.result_is_stale(scene, rib_row)
assert not addon.bonds.result_is_stale(scene, plate_row)
rib_obj.location.z -= 1.0
bpy.context.view_layer.update()
assert addon.bonds.check_bond_staleness(scene, bpy.context.evaluated_depsgraph_get()) == 0, \
    "moving back to the detected position should clear the stale flag"
# Moving the soundboard makes every result stale.
bond_objs["soundboard"].location.x += 2.0
bpy.context.view_layer.update()
addon.bonds.check_bond_staleness(scene, bpy.context.evaluated_depsgraph_get())
assert addon.bonds.result_is_stale(scene, plate_row)
print("Bond staleness check passed")

# --- Assembly solve: GO! gating + bonded bodies tied into the solve ---
try:
    addon.bonds.gather_bonded_bodies(bpy.context)
    raise AssertionError("stale bond results must block the solve")
except addon.analysis.SoundboardLabSceneError as exc:
    assert "out of date" in str(exc), exc
bond_objs["soundboard"].location.x -= 2.0
bpy.context.view_layer.update()
assert addon.bonds.check_bond_staleness(scene, bpy.context.evaluated_depsgraph_get()) == 0
bonded, excluded = addon.bonds.gather_bonded_bodies(bpy.context)
assert {b.name for b in bonded} == {bond_objs["rib1"].name, bond_objs["bridge_plate2"].name}, bonded
assert excluded == []


def _world_points(obj):
    return [obj.matrix_world @ v.co for v in obj.data.vertices]


rib_pts = _world_points(bond_objs["rib1"])
rib_top_z = max(p.z for p in rib_pts)
rib_top = [p for p in rib_pts if p.z > rib_top_z - 1e-3]
board_pts = _world_points(bond_objs["soundboard"])
board_bottom_z = min(p.z for p in board_pts)
compensator = bpy.data.objects.new("Compensator", None)
compensator.location = (
    sum(p.x for p in rib_top) / len(rib_top), sum(p.y for p in rib_top) / len(rib_top), rib_top_z
)
bridge_mass = bpy.data.objects.new("BridgeMass", None)
bridge_mass.location = (compensator.location.x, compensator.location.y, board_bottom_z)
for empty in (compensator, bridge_mass):
    scene.collection.objects.link(empty)
compensator.soundboard_lab.role = 'POINT_ATTACHMENT'
compensator.soundboard_lab.attach_stiffness = 2.72
compensator.soundboard_lab.attach_mass_g = 3.0
bridge_mass.soundboard_lab.role = 'MASS_LOAD'
bridge_mass.soundboard_lab.attach_stiffness = 5.0  # ignored: a Mass Load has no spring
bridge_mass.soundboard_lab.attach_radius = 10.0
bpy.context.view_layer.update()
assert {o.name for o in addon.analysis.find_attachment_objects(bpy.context)} == {"Compensator", "BridgeMass"}
try:
    addon.analysis.gather_point_attachments(bpy.context)
    raise AssertionError("a Mass Load without mass must be rejected")
except addon.analysis.SoundboardLabSceneError as exc:
    assert "BridgeMass" in str(exc) and "Mass" in str(exc), exc
bridge_mass.soundboard_lab.attach_mass_g = 30.0
attachments = {a["name"]: a for a in addon.analysis.gather_point_attachments(bpy.context)}
assert abs(attachments["Compensator"]["stiffness"] - 2.72 * 9.80665) < 1e-4, attachments  # float32 property
assert attachments["BridgeMass"]["stiffness"] == 0.0, attachments
assert addon.bonds.gather_bonded_bodies(bpy.context) == (bonded, excluded), "Empties are never bodies"
assembly_handle = addon.analysis.start_solve(
    bpy.context,
    bond_objs["soundboard"],
    CCX_EXE,
    ENGINE_PYTHON,
    n_frequencies=6,
    boundary_condition="fixed_edge",
    bonded_bodies=[
        {"object": b, "material": None, "orientation": addon.analysis._compute_grain_orientation(b)}
        for b in bonded
    ],
    tie_tolerance=scene.soundboard_lab_bond_tolerance,
    attachments=list(attachments.values()),
)
assembly_result = None
for _ in range(1200):  # up to ~10 minutes
    assembly_result = addon.analysis.poll_solve(assembly_handle)
    if assembly_result is not None:
        break
    time.sleep(0.5)
assert assembly_result is not None, "assembly solve did not finish in time"
assert assembly_result["status"] == "ok", assembly_result
tied = {b["name"]: b["tie_nodes"] for b in assembly_result["bonded_bodies"]}
assert set(tied) == {b.name for b in bonded} and all(n > 0 for n in tied.values()), tied
assert assembly_result["modes"][0]["frequency_hz"] > 1.0, "fixed edge must have no rigid modes"
addon.analysis._apply_result(scene, assembly_result)
assert "2 bonded bodies tied" in scene.soundboard_lab_message, scene.soundboard_lab_message
placed = {p["name"]: p for p in assembly_result["point_attachments"]}
assert placed["Compensator"]["body"] == bond_objs["rib1"].name, placed
assert placed["BridgeMass"]["body"] == "Soundboard" and placed["BridgeMass"]["nodes"] >= 1, placed
assert "Compensator on " in scene.soundboard_lab_message, scene.soundboard_lab_message
for empty in (compensator, bridge_mass):
    bpy.data.objects.remove(empty, do_unlink=True)
print("Point attachments passed:", placed)
print("Assembly solve passed: ties", tied, "f1 =", assembly_result["modes"][0]["frequency_hz"])
scene.soundboard_lab_results.clear()

# --- Body/Lining: lining is detected like any body, then holds the rim ---
bpy.ops.wm.stl_import(filepath=str(BOND_DATA / "body_lining.stl"))
lining = bpy.context.selected_objects[0]
lining.soundboard_lab.role = 'LINING'
lining.soundboard_lab.mesh_element_size = 4.0
bond_objs["body_lining"] = lining
assert addon.bonds.gather_bonded_bodies(bpy.context)[0], "lining is ignored under Fixed Edge"
scene.soundboard_lab_boundary_condition = 'BODY_LINING'
addon.analysis.check_body_lining_boundary_condition(bpy.context)
try:
    addon.bonds.gather_bonded_bodies(bpy.context)
    raise AssertionError("an undetected lining must block a Body/Lining solve")
except addon.analysis.SoundboardLabSceneError as exc:
    assert "Detect Bonds" in str(exc), exc
lining_bodies = addon.bonds.find_bond_bodies(bpy.context)
assert lining in lining_bodies
lining_handle = addon.bonds.start_bond_detection(bpy.context, bond_objs["soundboard"], lining_bodies, ENGINE_PYTHON, 0.1)
lining_bond = None
for _ in range(600):
    lining_bond = addon.analysis.poll_solve(lining_handle)
    if lining_bond is not None:
        break
    time.sleep(0.5)
assert lining_bond is not None and lining_bond["status"] == "ok", lining_bond
addon.bonds.apply_bond_result(bpy.context, lining_bond, lining_handle)
assert scene.soundboard_lab_bond_results[lining.name].status == 'CONTACT', scene.soundboard_lab_bond_results[lining.name].message
bonded, excluded = addon.bonds.gather_bonded_bodies(bpy.context)
assert lining in bonded and excluded == [], (bonded, excluded)
lining_solve = addon.analysis.start_solve(
    bpy.context,
    bond_objs["soundboard"],
    CCX_EXE,
    ENGINE_PYTHON,
    n_frequencies=4,
    boundary_condition="body_lining",
    bonded_bodies=[
        {"object": b, "material": None, "orientation": addon.analysis._compute_grain_orientation(b),
         "terminal": addon.bonds.terminal_spec(b)}
        for b in bonded
    ],
    tie_tolerance=scene.soundboard_lab_bond_tolerance,
)
lining_result = None
for _ in range(1200):
    lining_result = addon.analysis.poll_solve(lining_solve)
    if lining_result is not None:
        break
    time.sleep(0.5)
assert lining_result is not None and lining_result["status"] == "ok", lining_result
lining_row = next(b for b in lining_result["bonded_bodies"] if b["name"] == lining.name)
assert lining_row["tie_nodes"] > 0 and lining_row["terminal_nodes"] > 0, lining_row
assert lining_result["modes"][0]["frequency_hz"] > 1.0, "a Fixed lining terminal must leave no rigid modes"
print("Body/Lining solve passed:", lining_row, "f1 =", lining_result["modes"][0]["frequency_hz"])
scene.soundboard_lab_boundary_condition = 'FIXED_EDGE'

for o in [*bond_objs.values(), user_obj]:
    bpy.data.objects.remove(o, do_unlink=True)
generated_before_cleanup = addon.generated.remove_generated_objects(scene, addon.generated.BOND_PROXY_KINDS)
assert generated_before_cleanup == 4
scene.soundboard_lab_bond_results.clear()
obj.soundboard_lab.role = 'SOUNDBOARD'

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
assert addon.bonds._on_depsgraph_update not in bpy.app.handlers.depsgraph_update_post, "bond handler not removed"
assert not hasattr(bpy.types.Scene, "soundboard_lab_bond_results"), "bond results not cleaned up"

print("SMOKE TEST PASSED")
