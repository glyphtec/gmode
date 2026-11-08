"""Export Results: the Results & Verification dialog that dumps the last
solve (images, mode frequencies, mobility, inputs) into an experiment
folder under the preferences' Export Folder, for browsing outside
Blender (see src/soundboard_lab/experiment.py for the layout).

Like the solve, the work runs in the engine venv as a subprocess
(`soundboard_lab.export_cli`) -- the add-on never imports the engine --
so `start_export`/`poll_export` are plain, headlessly testable
functions and only the thin timer wrapper needs `bpy.app.timers`.
"""

import json
import subprocess
from pathlib import Path

import bpy

from . import analysis


def start_export(
    work_dir: Path,
    out_root: Path,
    engine_python_path: str,
    name: str,
    notes: str = "",
    image_width: int = 1200,
    line_width_mm: float = 0.6,
    displacement_images: bool = True,
    energy_images: bool = True,
    overlays: bool = True,
    fields: bool = True,
    solver_files: bool = True,
) -> dict:
    """Launch the export subprocess for the solve in `work_dir`; returns
    a handle for `poll_export`. Raises `FileNotFoundError` for problems
    that must stop before spawning anything."""
    if not engine_python_path or not Path(engine_python_path).is_file():
        raise FileNotFoundError(
            "Engine Python (venv) path is not configured or does not exist (add-on preferences)"
        )
    work_dir = Path(work_dir)
    if not (work_dir / "result.json").is_file():
        raise FileNotFoundError(f"no solve result to export in {work_dir}")

    output_json = work_dir / "export_result.json"
    output_json.unlink(missing_ok=True)
    cmd = [
        engine_python_path, "-m", "soundboard_lab.export_cli",
        "--work-dir", str(work_dir),
        "--out-root", str(out_root),
        "--name", name,
        "--notes", notes,
        "--output-json", str(output_json),
        "--image-width", str(image_width),
        "--line-width-mm", str(line_width_mm),
    ]
    for enabled, flag in (
        (displacement_images, "--no-displacement-images"),
        (energy_images, "--no-energy-images"),
        (overlays, "--no-overlays"),
        (fields, "--no-fields"),
        (solver_files, "--no-solver-files"),
    ):
        if not enabled:
            cmd.append(flag)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {"process": process, "output_json": output_json}


def poll_export(handle: dict) -> dict | None:
    """The parsed `{"status": "ok", "folder": ...}` (or error) result
    once the subprocess has finished, `None` while it is still running."""
    process: subprocess.Popen = handle["process"]
    if process.poll() is None:
        return None
    output_json: Path = handle["output_json"]
    if not output_json.is_file():
        _stdout, stderr = process.communicate()
        return {"status": "error", "message": f"export exited (code {process.returncode}) without a result: {stderr!r}"}
    return json.loads(output_json.read_text())


_active_export = None


def _default_experiment_name(context) -> str:
    if bpy.data.filepath:
        return Path(bpy.data.filepath).stem
    try:
        return analysis.find_soundboard_object(context).name
    except analysis.SoundboardLabSceneError:
        return "experiment"


def _resolved_export_path(context) -> Path | None:
    configured = analysis.get_preferences(context).export_path
    return Path(bpy.path.abspath(configured)) if configured else None


class SOUNDBOARD_OT_export_results(bpy.types.Operator):
    bl_idname = "soundboard.export_results"
    bl_label = "Export Results"
    bl_description = (
        "Review and export the last solve -- displacement and strain-energy images, mode frequencies, mobility "
        "and the solve inputs -- to a new experiment folder under the Export Folder in the add-on preferences"
    )
    bl_options = {'REGISTER'}

    experiment_name: bpy.props.StringProperty(name="Experiment", description="Name of this experiment")
    notes: bpy.props.StringProperty(name="Notes", description="Free-text notes stored with the experiment")
    displacement_images: bpy.props.BoolProperty(name="Displacement images", default=True)
    energy_images: bpy.props.BoolProperty(name="Strain energy images", default=True)
    overlays: bpy.props.BoolProperty(
        name="Contours and rib directions",
        description="Draw the iso-energy contours and bending ticks on the strain energy images",
        default=True,
    )
    image_width: bpy.props.IntProperty(name="Image width (px)", default=1200, min=400, max=3000)
    include_fields: bpy.props.BoolProperty(
        name="Mode fields (fields.npz)",
        description="The raw per-vertex displacement/energy/curvature fields, for comparing modes across experiments",
        default=True,
    )
    include_solver_files: bpy.props.BoolProperty(
        name="Solver files (STL, .inp, .dat)",
        description="Geometry and solver input/output, so the experiment can be reproduced",
        default=True,
    )

    @classmethod
    def poll(cls, context):
        return analysis.last_solve_dir() is not None and _active_export is None

    def invoke(self, context, event):
        self.experiment_name = _default_experiment_name(context)
        return context.window_manager.invoke_props_dialog(self, width=460)

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        destination = _resolved_export_path(context)

        box = layout.box()
        n_modes = len(scene.soundboard_lab_results)
        summary = f"{n_modes} modes"
        if scene.soundboard_lab_mobility.available:
            summary += ", bridge mobility"
        box.label(text=summary, icon='CHECKMARK')
        if scene.soundboard_lab_message:
            box.label(text=scene.soundboard_lab_message)
        if destination is None:
            box.label(text="Set the Export Folder in the add-on preferences first", icon='ERROR')
        else:
            box.label(text=str(destination), icon='FILE_FOLDER')

        layout.prop(self, "experiment_name")
        layout.prop(self, "notes")
        column = layout.column(heading="Include", align=True)
        column.prop(self, "displacement_images")
        column.prop(self, "energy_images")
        sub = column.row()
        sub.enabled = self.energy_images
        sub.prop(self, "overlays")
        column.prop(self, "include_fields")
        column.prop(self, "include_solver_files")
        if self.displacement_images or self.energy_images:
            layout.prop(self, "image_width")
            n_images = n_modes * (int(self.displacement_images) + int(self.energy_images))
            layout.label(text=f"{n_images} images", icon='IMAGE_DATA')

    def execute(self, context):
        global _active_export
        scene = context.scene
        work_dir = analysis.last_solve_dir()
        destination = _resolved_export_path(context)
        if work_dir is None:
            self.report({'ERROR'}, "No solve results to export -- run Analysis & Solve first")
            return {'CANCELLED'}
        if destination is None:
            self.report({'ERROR'}, "Set the Export Folder in the Soundboard Lab add-on preferences")
            return {'CANCELLED'}
        try:
            _active_export = start_export(
                work_dir,
                destination,
                analysis.get_preferences(context).engine_python_path,
                self.experiment_name.strip() or _default_experiment_name(context),
                self.notes,
                image_width=self.image_width,
                line_width_mm=scene.soundboard_lab_line_width,
                displacement_images=self.displacement_images,
                energy_images=self.energy_images,
                overlays=self.overlays,
                fields=self.include_fields,
                solver_files=self.include_solver_files,
            )
        except FileNotFoundError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}

        _active_export["scene"] = scene.name
        scene.soundboard_lab_export_message = "Exporting..."
        bpy.app.timers.register(_poll_timer, first_interval=0.5)
        self.report({'INFO'}, f"Soundboard Lab: exporting to {destination}")
        return {'FINISHED'}


def _poll_timer():
    global _active_export
    if _active_export is None:
        return None
    result = poll_export(_active_export)
    if result is None:
        return 0.5
    scene = bpy.data.scenes.get(_active_export["scene"])
    _active_export = None
    if scene is not None:
        if result.get("status") == "ok":
            scene.soundboard_lab_export_folder = result["folder"]
            scene.soundboard_lab_export_message = f"Exported: {Path(result['folder']).name}"
        else:
            scene.soundboard_lab_export_message = f"Export failed: {result.get('message', 'unknown error')}"
    analysis._tag_ui_redraw()
    return None


class SOUNDBOARD_OT_open_export_folder(bpy.types.Operator):
    bl_idname = "soundboard.open_export_folder"
    bl_label = "Open Export Folder"
    bl_description = "Open the most recent experiment folder in the file browser"

    @classmethod
    def poll(cls, context):
        folder = context.scene.soundboard_lab_export_folder
        return bool(folder) and Path(folder).is_dir()

    def execute(self, context):
        bpy.ops.wm.path_open(filepath=context.scene.soundboard_lab_export_folder)
        return {'FINISHED'}


classes = (SOUNDBOARD_OT_export_results, SOUNDBOARD_OT_open_export_folder)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.soundboard_lab_export_message = bpy.props.StringProperty()
    bpy.types.Scene.soundboard_lab_export_folder = bpy.props.StringProperty(subtype='DIR_PATH')


def unregister():
    global _active_export
    _active_export = None
    del bpy.types.Scene.soundboard_lab_export_folder
    del bpy.types.Scene.soundboard_lab_export_message
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
