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


# ---------------------------------------------------------------------------
# Add-on preferences: ccx and engine-venv-python paths (never hardcoded,
# never PATH-dependent -- see docs/CLAUDE.md Stack decision).
# ---------------------------------------------------------------------------


class SoundboardLabPreferences(bpy.types.AddonPreferences):
    # Must match the installed package's import name; extensions load
    # under a `bl_ext.<repo>.<pkg_id>` namespace that varies by install
    # location, so this is resolved dynamically rather than hardcoded.
    bl_idname = __package__

    ccx_path: bpy.props.StringProperty(
        name="CalculiX (ccx) Executable",
        description="Full path to ccx.exe (never resolved via PATH)",
        subtype='FILE_PATH',
    )
    engine_python_path: bpy.props.StringProperty(
        name="Engine Python (venv)",
        description=(
            "Full path to the engine venv's python.exe (has gmsh/meshio "
            "installed; Blender's own bundled Python deliberately does not)"
        ),
        subtype='FILE_PATH',
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
    as an ASCII STL. Returns the triangle count.

    Uses the depsgraph-evaluated mesh (not `obj.data` directly) so
    modifiers on the actual modeled object are respected, not just on
    a bare base mesh -- matters for testing a real modeled object
    rather than a trivial primitive.
    """
    depsgraph = context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        mesh.calc_loop_triangles()
        triangles = [[tuple(mesh.vertices[i].co) for i in tri.vertices] for tri in mesh.loop_triangles]
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


def start_solve(context, obj: bpy.types.Object, ccx_path: str, engine_python_path: str) -> dict:
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
    """
    if not ccx_path or not Path(ccx_path).is_file():
        raise FileNotFoundError("ccx executable path is not configured or does not exist (add-on preferences)")
    if not engine_python_path or not Path(engine_python_path).is_file():
        raise FileNotFoundError(
            "Engine Python (venv) path is not configured or does not exist (add-on preferences)"
        )

    settings = obj.soundboard_lab
    work_dir = Path(tempfile.mkdtemp(prefix="soundboard_lab_"))
    stl_path = work_dir / f"{obj.name}.stl"
    output_json = work_dir / "result.json"
    export_triangulated_stl(context, obj, stl_path)

    cmd = [
        engine_python_path,
        "-m", "soundboard_lab.cli",
        "--stl", str(stl_path),
        "--ccx-exe", ccx_path,
        "--output-json", str(output_json),
        "--element-size", str(settings.mesh_element_size),
        "--n-frequencies", "10",
    ]
    if settings.mesh_thickness_hint > 0:
        cmd += ["--thickness-hint", str(settings.mesh_thickness_hint)]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"process": process, "output_json": output_json, "work_dir": work_dir, "object_name": obj.name}


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
# Results storage (Scene-level: one active free-free analysis at a time)
# ---------------------------------------------------------------------------


class SoundboardLabModeResult(bpy.types.PropertyGroup):
    mode_number: bpy.props.IntProperty(name="Mode")
    frequency_hz: bpy.props.FloatProperty(name="Frequency (Hz)")


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


def _apply_result(scene, result: dict) -> None:
    scene.soundboard_lab_results.clear()
    if result["status"] == "ok":
        scene.soundboard_lab_status = 'DONE'
        scene.soundboard_lab_message = f"{result['tetra_count']} elements, {result['material_name']}"
        for mode in result["modes"]:
            item = scene.soundboard_lab_results.add()
            item.mode_number = mode["mode_number"]
            item.frequency_hz = mode["frequency_hz"]
    else:
        scene.soundboard_lab_status = 'ERROR'
        scene.soundboard_lab_message = result.get("message", "Unknown error")


def _poll_timer():
    global _active_handle
    if _active_handle is None:
        return None  # stop; nothing active (shouldn't normally happen)
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


class SOUNDBOARD_OT_analyze_free_free(bpy.types.Operator):
    bl_idname = "soundboard.analyze_free_free"
    bl_label = "Analyze (Free-Free)"
    bl_description = "Mesh and solve the Soundboard-role object, unconstrained, via the engine subprocess"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        global _active_handle
        return _active_handle is None

    def execute(self, context):
        global _active_handle
        try:
            obj = find_soundboard_object(context)
            prefs = get_preferences(context)
            _active_handle = start_solve(context, obj, prefs.ccx_path, prefs.engine_python_path)
        except (SoundboardLabSceneError, FileNotFoundError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        context.scene.soundboard_lab_status = 'RUNNING'
        context.scene.soundboard_lab_message = ""
        context.scene.soundboard_lab_results.clear()
        bpy.app.timers.register(_poll_timer, first_interval=0.5)
        self.report({'INFO'}, f"Soundboard Lab: solving '{obj.name}'...")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    SoundboardLabPreferences,
    SoundboardLabModeResult,
    SOUNDBOARD_OT_analyze_free_free,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.soundboard_lab_results = bpy.props.CollectionProperty(type=SoundboardLabModeResult)
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


def unregister():
    global _active_handle
    _active_handle = None
    del bpy.types.Scene.soundboard_lab_message
    del bpy.types.Scene.soundboard_lab_status
    del bpy.types.Scene.soundboard_lab_results
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
