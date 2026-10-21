"""Analysis & Solve: scene-collection, STL export, and the subprocess
launch/poll cycle for running the engine's CLI (see src/soundboard_lab/
cli.py) out-of-process.

The engine/UI boundary rule (docs/CLAUDE.md) means this addon never
imports the engine's gmsh/meshio-dependent code directly -- it launches
the engine venv's own Python running `python -m soundboard_lab.cli` as
a subprocess and polls for completion.

`start_solve`/`poll_solve` are deliberately plain functions (no bpy.app
.timers involved) so the launch/poll cycle itself is testable headlessly
(a --background smoke test can call poll_solve in a blocking loop);
only the thin "don't block Blender's UI thread" wrapper around them
needs bpy.app.timers, and that wrapper is intentionally minimal.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import bpy
from mathutils import Vector


# ---------------------------------------------------------------------------
# Add-on preferences: ccx and engine-venv-python paths (never hardcoded,
# never PATH-dependent -- see docs/CLAUDE.md Stack decision).
# ---------------------------------------------------------------------------


def _prefs_cache_path() -> Path:
    """Location for the fallback preferences cache -- deliberately
    OUTSIDE anything tied to this extension's own install/registration
    state (`bpy.utils.user_resource('CONFIG')` is a stable per-user
    Blender config directory, independent of which extension repo or
    install this add-on currently lives under).

    Exists because Blender's own `AddonPreferences` storage has been
    observed, live, to lose its values across what should be routine
    dev-reload cycles (disable/enable, or a dev-tooling reinstall) --
    twice in one session, with "Save Preferences" already clicked and
    Auto-Save Preferences already on, so this isn't simple user error.
    The leading theory (not fully confirmed): a dev-reload that
    uninstalls/reinstalls the extension package can reset Blender's own
    per-addon preferences entry even though the registered `bl_idname`
    string itself stays the same. A plain file next to nothing Blender
    manages sidesteps that entirely.
    """
    return Path(bpy.utils.user_resource('CONFIG')) / "soundboard_lab_prefs_cache.json"


def _save_prefs_cache(ccx_path: str, engine_python_path: str) -> None:
    """Best-effort, non-fatal -- a failed cache write must never break
    the actual preferences field the user just edited."""
    try:
        _prefs_cache_path().write_text(json.dumps({"ccx_path": ccx_path, "engine_python_path": engine_python_path}))
    except Exception as exc:
        print(f"Soundboard Lab: failed to write prefs cache (non-fatal): {exc!r}")


def _load_prefs_cache() -> dict | None:
    try:
        cache_path = _prefs_cache_path()
        if not cache_path.is_file():
            return None
        return json.loads(cache_path.read_text())
    except Exception as exc:
        print(f"Soundboard Lab: failed to read prefs cache (non-fatal): {exc!r}")
        return None


def _on_prefs_path_update(self, context):
    _save_prefs_cache(self.ccx_path, self.engine_python_path)


def _restore_prefs_from_cache_if_empty():
    """Deferred (via bpy.app.timers, NOT called directly from
    register()): `get_preferences()` needs `context.preferences.
    addons[__package__]`, which Blender populates as part of the
    formal addon-enable sequence -- confirmed not yet present at the
    moment register() itself runs (see get_preferences's own
    docstring). A short-delay one-shot timer runs after that sequence
    completes, which is the earliest point this can safely run.
    """
    try:
        prefs = get_preferences(bpy.context)
    except Exception as exc:
        print(f"Soundboard Lab: couldn't restore prefs from cache (non-fatal): {exc!r}")
        return None  # don't reschedule; nothing more we can do this session
    if prefs.ccx_path and prefs.engine_python_path:
        return None  # already populated -- nothing to restore
    cached = _load_prefs_cache()
    if cached is None:
        return None
    if not prefs.ccx_path:
        prefs.ccx_path = cached.get("ccx_path", "")
    if not prefs.engine_python_path:
        prefs.engine_python_path = cached.get("engine_python_path", "")
    print("Soundboard Lab: restored ccx/engine Python paths from the fallback prefs cache")
    return None


class SoundboardLabPreferences(bpy.types.AddonPreferences):
    # Must match the installed package's import name; extensions load
    # under a `bl_ext.<repo>.<pkg_id>` namespace that varies by install
    # location, so this is resolved dynamically rather than hardcoded.
    bl_idname = __package__

    ccx_path: bpy.props.StringProperty(
        name="CalculiX (ccx) Executable",
        description="Full path to ccx.exe (never resolved via PATH)",
        subtype='FILE_PATH',
        update=_on_prefs_path_update,
    )
    engine_python_path: bpy.props.StringProperty(
        name="Engine Python (venv)",
        description=(
            "Full path to the engine venv's python.exe (has gmsh/meshio "
            "installed; Blender's own bundled Python deliberately does not)"
        ),
        subtype='FILE_PATH',
        update=_on_prefs_path_update,
    )

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "ccx_path")
        layout.prop(self, "engine_python_path")
        for label, path in (("ccx", self.ccx_path), ("Engine Python", self.engine_python_path)):
            if path and not Path(path).is_file():
                layout.label(text=f"{label} path does not exist", icon='ERROR')


def get_preferences(context) -> SoundboardLabPreferences:
    return context.preferences.addons[__package__].preferences


# ---------------------------------------------------------------------------
# Scene collection: find the one Soundboard-role object
# ---------------------------------------------------------------------------


class SoundboardLabSceneError(RuntimeError):
    """Raised for scene-collection problems (missing/ambiguous bodies)
    that must be reported to the user, not silently worked around.
    """


def find_soundboard_object(context) -> bpy.types.Object:
    candidates = [
        obj for obj in context.scene.objects
        if obj.type == 'MESH' and obj.soundboard_lab.role == 'SOUNDBOARD'
    ]
    if not candidates:
        raise SoundboardLabSceneError("No object with role 'Soundboard' found in the scene")
    if len(candidates) > 1:
        names = ", ".join(o.name for o in candidates)
        raise SoundboardLabSceneError(
            f"Multiple Soundboard-role objects found ({names}) -- "
            "free-free analysis currently expects exactly one"
        )
    return candidates[0]


def export_triangulated_stl(context, obj: bpy.types.Object, stl_path: Path) -> int:
    """Write `obj`'s evaluated (modifiers applied), triangulated mesh
    as an ASCII STL, in WORLD space. Returns the triangle count.

    Uses the depsgraph-evaluated mesh (not `obj.data` directly) so
    modifiers on the actual modeled object are respected, not just on
    a bare base mesh -- matters for testing a real modeled object
    rather than a trivial primitive.

    Applies `matrix_world` to every vertex: `mesh.vertices[i].co` is
    LOCAL (object-space), not world-space. Without this, any object
    with a non-identity transform (rotated, moved off the origin, or
    unapplied scale -- all completely normal in a real modeled scene,
    not an edge case) would silently export geometrically wrong
    geometry with no error at all. Caught empirically: a real-world
    test object exported this way, before the fix, came out with a
    ~490-unit Z span instead of its true ~2mm thickness, because its
    object transform rotates its local "thin" axis away from world Z.
    """
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        matrix_world = eval_obj.matrix_world
        triangles = [
            [tuple(matrix_world @ mesh.vertices[i].co) for i in tri.vertices] for tri in mesh.loop_triangles
        ]
    finally:
        eval_obj.to_mesh_clear()

    with open(stl_path, "w") as f:
        f.write(f"solid {obj.name}\n")
        for a, b, c in triangles:
            ax, ay, az = a
            bx, by, bz = b
            cx, cy, cz = c
            ux, uy, uz = bx - ax, by - ay, bz - az
            vx, vy, vz = cx - ax, cy - ay, cz - az
            nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
            f.write(f"  facet normal {nx:.6e} {ny:.6e} {nz:.6e}\n")
            f.write("    outer loop\n")
            for x, y, z in (a, b, c):
                f.write(f"      vertex {x:.6e} {y:.6e} {z:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")
        f.write(f"endsolid {obj.name}\n")

    return len(triangles)


# ---------------------------------------------------------------------------
# Subprocess launch / poll (plain functions, testable without bpy.app.timers)
# ---------------------------------------------------------------------------


def start_solve(
    context,
    obj: bpy.types.Object,
    ccx_path: str,
    engine_python_path: str,
    n_frequencies: int = 10,
    boundary_condition: str = "fixed_edge",
    num_layers: int = 3,
    material: dict | None = None,
    orientation: tuple[float, float, float, float, float, float] | None = None,
) -> dict:
    """Export `obj` and launch the engine CLI subprocess. Returns a
    handle dict for `poll_solve`. Raises `FileNotFoundError` for
    problems that must stop before ever spawning a process.

    Takes explicit paths rather than resolving add-on preferences
    itself -- preferences only exist via Blender's formal addon-enable
    mechanism (confirmed: merely calling register_class does not
    populate `context.preferences.addons`), which is a bpy/UI-only
    concept this otherwise-plain, independently-testable function
    shouldn't be coupled to. The operator resolves preferences and
    passes the paths in.

    `n_frequencies`/`boundary_condition`/`num_layers` default to the
    same values that used to be hardcoded here, so every existing
    direct caller (tests, this function's own prior behavior) is
    unaffected; the operator now reads the real values from the
    Scene's global solution-parameter properties instead of relying
    on these defaults.

    `material`, if given, is a plain dict (see
    `soundboard_lab.materials.material_to_dict`/`_from_dict` on the
    engine side) written to a temp JSON file and passed via
    `--material-json`; `None` leaves the engine's own
    `SPRUCE_PLACEHOLDER` default in place (e.g. no Material assigned
    to `obj` yet -- graceful degradation, not a hard error, since a
    missing material shouldn't block a first solve attempt).
    `orientation` is the six raw `*ORIENTATION` numbers (see
    `_compute_grain_orientation`); `None` leaves the engine's identity
    default in place.
    """
    if not ccx_path or not Path(ccx_path).is_file():
        raise FileNotFoundError("ccx executable path is not configured or does not exist (add-on preferences)")
    if not engine_python_path or not Path(engine_python_path).is_file():
        raise FileNotFoundError(
            "Engine Python (venv) path is not configured or does not exist (add-on preferences)"
        )

    # Restore the real object's visibility before launching: a previous
    # solve's mode-shape preview may have hidden it (see
    # `_set_soundboard_visibility`), and it shouldn't stay hidden while
    # a brand new solve is running (or if it then errors out).
    _set_soundboard_visibility(obj, visible=True)

    settings = obj.soundboard_lab
    work_dir = Path(tempfile.mkdtemp(prefix="soundboard_lab_"))
    stl_path = work_dir / f"{obj.name}.stl"
    output_json = work_dir / "result.json"
    status_json = work_dir / "status.json"
    export_triangulated_stl(context, obj, stl_path)

    cmd = [
        engine_python_path,
        "-m", "soundboard_lab.cli",
        "--stl", str(stl_path),
        "--ccx-exe", ccx_path,
        "--output-json", str(output_json),
        "--element-size", str(settings.mesh_element_size),
        "--n-frequencies", str(n_frequencies),
        "--mesh-type", "shell",
        "--boundary-condition", boundary_condition,
        "--num-layers", str(num_layers),
        "--status-json", str(status_json),
    ]
    if settings.mesh_thickness_hint > 0:
        cmd += ["--thickness-hint", str(settings.mesh_thickness_hint)]
    if material is not None:
        material_json_path = work_dir / "material.json"
        material_json_path.write_text(json.dumps(material))
        cmd += ["--material-json", str(material_json_path)]
    if orientation is not None:
        cmd += ["--orientation", ",".join(str(v) for v in orientation)]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {
        "process": process,
        "output_json": output_json,
        "status_json": status_json,
        "work_dir": work_dir,
        "object_name": obj.name,
    }


def read_solve_status(handle: dict) -> dict | None:
    """Return the current step checklist (see cli.py's `StepTracker`)
    if the subprocess has written one yet, else `None`. Safe to call
    on every poll tick regardless of whether the subprocess has
    finished -- independent of `poll_solve`'s own completion check, so
    the UI can show live step-by-step status while still waiting for
    the final result. Not a numeric progress bar (ccx itself is an
    opaque subprocess call with no fine-grained progress available) --
    a named-step checklist with a clear per-step message instead.
    """
    status_json: Path = handle["status_json"]
    if not status_json.is_file():
        return None
    try:
        return json.loads(status_json.read_text())
    except (json.JSONDecodeError, OSError):
        # The subprocess may be mid-write when we poll; a torn read is
        # expected and transient, not a real error -- just means "no
        # status to show yet", same as the file not existing at all.
        return None


def poll_solve(handle: dict) -> dict | None:
    """Return the parsed result dict once the subprocess has finished,
    or `None` if it's still running. Does not raise on a failed solve
    (status=='error' is a normal, expected result, not a Python
    exception) -- only genuinely unexpected conditions (process exited
    but wrote no output at all) raise.
    """
    process: subprocess.Popen = handle["process"]
    if process.poll() is None:
        return None

    output_json: Path = handle["output_json"]
    if not output_json.is_file():
        stdout, stderr = process.communicate()
        raise SoundboardLabSceneError(
            f"Engine subprocess exited (code {process.returncode}) without writing a result: "
            f"stdout={stdout!r} stderr={stderr!r}"
        )
    return json.loads(output_json.read_text())


# ---------------------------------------------------------------------------
# Mode-shape visualization: a static Z-displacement heat map on a real
# 3D mesh object (not an animated/exaggerated deformation -- explicitly
# not wanted; a plain color-coded heat map on the real geometry is the
# goal, viewable from any angle since it's a real object, not a flat
# projection image).
# ---------------------------------------------------------------------------

# 5-point blue -> cyan -> green -> yellow -> red ramp (the same
# rainbow/jet-style convention the FreeCAD FEM reference images use for
# their own "Displacement Z" heat maps -- not an exact color match
# requirement, just the same familiar low-to-high convention).
_JET_CONTROL_POINTS = [
    (0.00, (0.0, 0.0, 1.0)),
    (0.25, (0.0, 1.0, 1.0)),
    (0.50, (0.0, 1.0, 0.0)),
    (0.75, (1.0, 1.0, 0.0)),
    (1.00, (1.0, 0.0, 0.0)),
]


def _jet_colormap(t: float) -> tuple[float, float, float]:
    """Piecewise-linear interpolation through `_JET_CONTROL_POINTS`.
    `t` is clamped to [0, 1] first (defensive: a displacement array's
    own min/max define the normalization range by construction, so `t`
    should already be in range, but a degenerate all-zero array could
    produce a NaN from 0/0 upstream -- clamping here keeps a garbage
    input from raising rather than just drawing a slightly wrong color).
    """
    t = max(0.0, min(1.0, t))
    for (t0, c0), (t1, c1) in zip(_JET_CONTROL_POINTS, _JET_CONTROL_POINTS[1:]):
        if t0 <= t <= t1:
            local_t = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return tuple(a + (b - a) * local_t for a, b in zip(c0, c1))
    return _JET_CONTROL_POINTS[-1][1]  # unreachable given the clamp above; defensive fallback


def _set_viewport_vertex_color_shading():
    """Best-effort: switch any visible VIEW_3D area to Solid shading
    with vertex-color display, so the mode-shape heat map is visible
    with zero material setup (same rationale as docs/CLAUDE.md's
    "Mesh preview & validation" decision for the mesh-quality heat
    map). Non-fatal, like `_tag_ui_redraw` -- a window-less context
    (e.g. this function called from a background script) must not
    raise just because there's no viewport to update.
    """
    try:
        window = bpy.context.window
        if window is None:
            return
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = 'SOLID'
                    space.shading.color_type = 'VERTEX'
    except Exception as exc:
        print(f"Soundboard Lab: viewport shading update failed (non-fatal): {exc!r}")


def _set_soundboard_visibility(obj: bpy.types.Object, visible: bool) -> None:
    """Show/hide the real Soundboard-role object in the viewport.

    `surface_offset` alone (see `build_mode_shape_preview`) only pushes
    the Z-fighting threshold out to a larger camera distance -- it
    can't eliminate it, since depth-buffer precision degrades with
    distance from the camera. Confirmed visually in a real GUI session:
    clean up close, the same clashing/striping reappears once zoomed
    out further. The robust fix is to not have two coincident solid
    surfaces on screen at the same time at all: hide the real object
    while its mode-shape preview is showing, and restore it as soon as
    a new solve is kicked off (whether or not that solve succeeds --
    a stale preview from a *previous* solve shouldn't linger hiding the
    real object if the new one errors out).

    Non-fatal, like `_set_viewport_vertex_color_shading` -- a missing
    view-layer entry (e.g. called from a background script context)
    must not raise and abort the preview/solve flow over this.
    """
    try:
        obj.hide_set(not visible)
    except Exception as exc:
        print(f"Soundboard Lab: soundboard visibility update failed (non-fatal): {exc!r}")


def build_mode_shape_preview(
    context,
    name: str,
    xy: list,
    top_z: list,
    triangles: list,
    z_displacement: list,
    surface_offset: float = 0.5,
) -> bpy.types.Object:
    """Build (or replace, if an object of this `name` already exists)
    a real 3D mesh object -- vertices at
    `(xy[i][0], xy[i][1], top_z[i] + surface_offset)` in WORLD space
    (matching `export_triangulated_stl`'s world-space convention; this
    object's own transform is left at identity so the mesh data's own
    coordinates are used directly, not double-transformed) -- with a
    per-vertex Color Attribute encoding `z_displacement` via a rainbow
    colormap, normalized to that mode's own min/max. A real object
    (not a flat image) can be freely orbited/viewed from any angle in
    the 3D viewport, which is the actual goal here (an animated/
    exaggerated deformed shape was explicitly NOT wanted -- a static
    heat map on the real geometry is "table stakes" and sufficient).

    `surface_offset` nudges the preview mesh slightly above the
    original soundboard object's own top surface (both occupy the
    EXACT same world-space Z otherwise, since `top_z` comes directly
    from the solved shell's real top surface): without it, the two
    coplanar surfaces Z-fight in the viewport (confirmed in a real
    GUI session, not hypothetical -- the original object's solid-fill
    shading and the preview's vertex-color shading flicker against
    each other). 0.5 is small enough to look "on the surface" at any
    reasonable zoom for a soundboard-scale object (~2mm thick), while
    clearly separating the two surfaces in depth.
    """
    mesh = bpy.data.meshes.get(name)
    if mesh is not None:
        mesh.clear_geometry()
    else:
        mesh = bpy.data.meshes.new(name)

    vertices = [(x, y, z + surface_offset) for (x, y), z in zip(xy, top_z)]
    mesh.from_pydata(vertices, [], [tuple(tri) for tri in triangles])
    mesh.update()

    color_attr = mesh.color_attributes.new(name="ModeShapeZ", type='FLOAT_COLOR', domain='POINT')
    # Normalize SYMMETRICALLY around zero, not plain (dz-min)/(max-min):
    # an eigenmode's absolute sign/scale is mathematically arbitrary, so
    # a mode can easily come back almost entirely one-signed (confirmed
    # on a real solve: one mode's values ranged -244.6 to +0.18, nearly
    # all negative). Min/max normalization would then put a CLAMPED
    # BOUNDARY'S zero displacement near one extreme of the color ramp
    # instead of the visual middle, making a zero-displacement edge
    # look like a "hot" region rather than neutral. Symmetric
    # normalization keeps zero at the same green midpoint regardless of
    # which way a given mode happens to be signed.
    bound = max(abs(min(z_displacement)), abs(max(z_displacement)))
    for i, dz in enumerate(z_displacement):
        t = (dz + bound) / (2 * bound) if bound > 1e-12 else 0.5
        r, g, b = _jet_colormap(t)
        color_attr.data[i].color = (r, g, b, 1.0)

    obj = bpy.data.objects.get(name)
    if obj is None:
        obj = bpy.data.objects.new(name, mesh)
        context.collection.objects.link(obj)
    else:
        obj.data = mesh

    _set_viewport_vertex_color_shading()
    return obj


# ---------------------------------------------------------------------------
# Results storage (Scene-level: one active free-free analysis at a time)
# ---------------------------------------------------------------------------


class SoundboardLabModeResult(bpy.types.PropertyGroup):
    mode_number: bpy.props.IntProperty(name="Mode")
    frequency_hz: bpy.props.FloatProperty(name="Frequency (Hz)")


class SoundboardLabStepStatus(bpy.types.PropertyGroup):
    """One row of the process-step checklist (see cli.py's
    `StepTracker`) -- not a numeric progress bar, since ccx itself is
    an opaque subprocess call with no fine-grained progress available;
    a named-step checklist with per-step status and a clear message if
    something goes sideways is both honest and actually useful.
    """

    step_id: bpy.props.StringProperty(name="Step ID")
    label: bpy.props.StringProperty(name="Step")
    status: bpy.props.StringProperty(name="Status")  # 'pending' | 'running' | 'done' | 'error'
    message: bpy.props.StringProperty(name="Message")


def _tag_ui_redraw():
    """Same rationale as the grain-rosette gizmo's redraw helper: force
    the N-panel to refresh after background state changes outside the
    normal operator-redraw path. Must never raise (see that helper's
    docstring for why).
    """
    try:
        window = bpy.context.window
        if window is None:
            return
        for area in window.screen.areas:
            for region in area.regions:
                region.tag_redraw()
    except Exception as exc:
        print(f"Soundboard Lab: results redraw tag failed (non-fatal): {exc!r}")


# ---------------------------------------------------------------------------
# Operator + timer-based poll (the only genuinely bpy.app.timers-coupled
# part; start_solve/poll_solve above are plain and separately testable)
# ---------------------------------------------------------------------------

_active_handle = None

# Cached per-solve data the mode-shape preview needs (solved mesh
# footprint + every mode's per-node Z-displacement) -- not stored as
# bpy Scene properties, since those don't hold nested Python
# lists/arrays efficiently; a plain module-level cache is simplest and
# is read only by the same session's mode-selection handler below.
_last_mesh_shape = None
_last_mode_shapes = None


def _update_step_status(scene, status: dict) -> None:
    """Mirror cli.py's step-checklist JSON into
    `scene.soundboard_lab_steps` for the Analysis panel to display
    live, called on every poll tick regardless of completion.
    """
    scene.soundboard_lab_steps.clear()
    for step in status.get("steps", []):
        item = scene.soundboard_lab_steps.add()
        item.step_id = step.get("id", "")
        item.label = step.get("label", "")
        item.status = step.get("status", "pending")
        item.message = step.get("message", "")


def _apply_result(scene, result: dict) -> None:
    global _last_mesh_shape, _last_mode_shapes
    scene.soundboard_lab_results.clear()
    if result["status"] == "ok":
        scene.soundboard_lab_status = 'DONE'
        message = f"{result['triangle_count']} triangles, {result['material_name']}"
        repair = result.get("mesh_repair")
        if repair is not None:
            if repair["was_already_valid"]:
                message += " | mesh was already valid"
            else:
                problems = "; ".join(repair["problems_found"])
                message += f" | mesh repaired ({problems})"
                if repair["components_removed"]:
                    message += f", removed {repair['components_removed']} disconnected piece(s)"
        scene.soundboard_lab_message = message
        for mode in result["modes"]:
            item = scene.soundboard_lab_results.add()
            item.mode_number = mode["mode_number"]
            item.frequency_hz = mode["frequency_hz"]
        _last_mesh_shape = result.get("mesh_shape")
        _last_mode_shapes = result.get("mode_shapes")
        scene.soundboard_lab_selected_mode_index = 0
    else:
        scene.soundboard_lab_status = 'ERROR'
        scene.soundboard_lab_message = result.get("message", "Unknown error")
        _last_mesh_shape = None
        _last_mode_shapes = None


def _poll_timer():
    global _active_handle
    if _active_handle is None:
        return None  # stop; nothing active (shouldn't normally happen)

    status = read_solve_status(_active_handle)
    if status is not None:
        _update_step_status(bpy.context.scene, status)
        _tag_ui_redraw()

    try:
        result = poll_solve(_active_handle)
    except SoundboardLabSceneError as exc:
        result = {"status": "error", "message": str(exc)}
    if result is None:
        return 0.5  # still running; check again shortly

    _apply_result(bpy.context.scene, result)
    _active_handle = None
    _tag_ui_redraw()
    return None  # stop the timer


def _update_mode_shape_preview(scene) -> None:
    """Rebuild the mode-shape preview object for whichever row is
    currently selected in the Results UIList. Called from the
    selected-index `update` callback, so it fires whenever the user
    clicks a different mode in the list.
    """
    if _last_mesh_shape is None or _last_mode_shapes is None:
        return
    index = scene.soundboard_lab_selected_mode_index
    if index < 0 or index >= len(scene.soundboard_lab_results):
        return
    item = scene.soundboard_lab_results[index]
    z_displacement = _last_mode_shapes.get(str(item.mode_number))
    if z_displacement is None:
        return
    build_mode_shape_preview(
        bpy.context,
        "Soundboard Mode Shape Preview",
        _last_mesh_shape["xy"],
        _last_mesh_shape["top_z"],
        _last_mesh_shape["triangles"],
        z_displacement,
    )
    try:
        soundboard_obj = find_soundboard_object(bpy.context)
    except SoundboardLabSceneError as exc:
        print(f"Soundboard Lab: couldn't update soundboard visibility (non-fatal): {exc!r}")
        return
    _set_soundboard_visibility(soundboard_obj, visible=False)


def _on_selected_mode_index_update(self, context):
    _update_mode_shape_preview(self)


def _compute_grain_orientation(obj: bpy.types.Object) -> tuple[float, float, float, float, float, float]:
    """Resolve `obj.soundboard_lab.grain_rotation` (a per-object Euler,
    relative to the object's own LOCAL axes -- see docs/CLAUDE.md
    "Grain orientation") into the six raw `*ORIENTATION` numbers the
    engine's CLI `--orientation` flag expects: a world-space vector
    along the local 1-axis (radial), then a world-space vector in the
    local 1-2 plane (longitudinal) -- see `write_shell_frequency_inp`'s
    own docstring for the CalculiX card format.

    Grain orientation is deliberately independent of the Material
    ID-block (the same material gets reused across boards with
    different grain directions), so this is resolved from the OBJECT,
    not from `extract_material`.

    World space, not object-local: the engine's own mesh coordinates
    are already world-space (`export_triangulated_stl` bakes
    `matrix_world` into the exported STL), so the orientation vectors
    must be transformed the same way. Only the rotation component of
    `matrix_world` is used (via `to_quaternion()`), not scale/shear --
    an axis direction has no meaningful "scale".
    """
    grain_rotation_matrix = obj.soundboard_lab.grain_rotation.to_matrix()
    world_rotation_matrix = obj.matrix_world.to_quaternion().to_matrix()
    a = world_rotation_matrix @ (grain_rotation_matrix @ Vector((1.0, 0.0, 0.0)))
    b = world_rotation_matrix @ (grain_rotation_matrix @ Vector((0.0, 1.0, 0.0)))
    return (a.x, a.y, a.z, b.x, b.y, b.z)


class SOUNDBOARD_OT_analyze(bpy.types.Operator):
    bl_idname = "soundboard.analyze"
    bl_label = "Analyze"
    bl_description = (
        "Mesh (shell/extrusion) and solve the Soundboard-role object via the engine subprocess, "
        "using the project's global solution parameters and the object's assigned material/grain orientation"
    )
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        global _active_handle
        return _active_handle is None

    def execute(self, context):
        global _active_handle
        from . import extract_material  # lazy: avoids a circular import with __init__.py at module load time

        try:
            obj = find_soundboard_object(context)
            prefs = get_preferences(context)
            material = extract_material(obj.active_material) if obj.active_material is not None else None
            orientation = _compute_grain_orientation(obj)
            scene = context.scene
            _active_handle = start_solve(
                context,
                obj,
                prefs.ccx_path,
                prefs.engine_python_path,
                n_frequencies=scene.soundboard_lab_n_frequencies,
                boundary_condition=scene.soundboard_lab_boundary_condition.lower(),
                num_layers=scene.soundboard_lab_num_layers,
                material=material,
                orientation=orientation,
            )
        except (SoundboardLabSceneError, FileNotFoundError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        context.scene.soundboard_lab_status = 'RUNNING'
        context.scene.soundboard_lab_message = ""
        context.scene.soundboard_lab_results.clear()
        context.scene.soundboard_lab_steps.clear()
        bpy.app.timers.register(_poll_timer, first_interval=0.5)
        self.report({'INFO'}, f"Soundboard Lab: solving '{obj.name}'...")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    SoundboardLabPreferences,
    SoundboardLabModeResult,
    SoundboardLabStepStatus,
    SOUNDBOARD_OT_analyze,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    if not bpy.app.background:
        # Deferred: context.preferences.addons[__package__] isn't
        # populated yet at this point in the formal addon-enable
        # sequence (see _restore_prefs_from_cache_if_empty's own
        # docstring) -- a short one-shot timer runs just after it is.
        # Skipped under --background: timers generally don't fire
        # there anyway (no running event loop), and smoke tests pass
        # explicit paths directly, never relying on this mechanism.
        bpy.app.timers.register(_restore_prefs_from_cache_if_empty, first_interval=0.2)
    bpy.types.Scene.soundboard_lab_results = bpy.props.CollectionProperty(type=SoundboardLabModeResult)
    bpy.types.Scene.soundboard_lab_steps = bpy.props.CollectionProperty(type=SoundboardLabStepStatus)
    bpy.types.Scene.soundboard_lab_selected_mode_index = bpy.props.IntProperty(
        name="Selected Mode", default=0, update=_on_selected_mode_index_update
    )
    bpy.types.Scene.soundboard_lab_status = bpy.props.EnumProperty(
        items=[
            ('IDLE', "Idle", "No analysis run yet"),
            ('RUNNING', "Running", "Engine subprocess in progress"),
            ('DONE', "Done", "Solve completed successfully"),
            ('ERROR', "Error", "Solve failed"),
        ],
        default='IDLE',
    )
    bpy.types.Scene.soundboard_lab_message = bpy.props.StringProperty()
    # Global solution parameters: project-level (Scene), not per-body,
    # since they describe the analysis itself rather than any one
    # object -- saved with the .blend file automatically like any
    # other Scene property (no extra persistence code needed within a
    # project; a reusable cross-project default would be a separate,
    # bigger piece of work -- not built here).
    bpy.types.Scene.soundboard_lab_n_frequencies = bpy.props.IntProperty(
        name="Number of Modes", description="How many eigenfrequencies/mode shapes to extract", default=10, min=1
    )
    bpy.types.Scene.soundboard_lab_boundary_condition = bpy.props.EnumProperty(
        name="Boundary Condition",
        items=[
            ('FIXED_EDGE', "Fixed Edge", "Every perimeter node clamped (instrument-mounted analogy)"),
            ('FREE_FREE', "Free-Free", "No supports at all (bare-plate validation rig analogy)"),
        ],
        default='FIXED_EDGE',
    )
    bpy.types.Scene.soundboard_lab_num_layers = bpy.props.IntProperty(
        name="Through-Thickness Layers", description="Element layers through the shell's thickness", default=3, min=2
    )


def unregister():
    global _active_handle, _last_mesh_shape, _last_mode_shapes
    _active_handle = None
    _last_mesh_shape = None
    _last_mode_shapes = None
    del bpy.types.Scene.soundboard_lab_num_layers
    del bpy.types.Scene.soundboard_lab_boundary_condition
    del bpy.types.Scene.soundboard_lab_n_frequencies
    del bpy.types.Scene.soundboard_lab_message
    del bpy.types.Scene.soundboard_lab_status
    del bpy.types.Scene.soundboard_lab_selected_mode_index
    del bpy.types.Scene.soundboard_lab_steps
    del bpy.types.Scene.soundboard_lab_results
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
