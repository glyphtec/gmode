"""Bond detection: launch the engine's `bond_cli` subprocess, turn its
result into bonding proxies (in the generated "Bonding Proxies"
collection), draw the translucent contact overlay, and flag proxies
whose source changed afterwards.

Same engine/UI boundary as analysis.py: no gmsh/trimesh in Blender's
Python. `start_bond_detection` / `apply_bond_result` /
`check_bond_staleness` are plain functions so a --background smoke
test can drive the whole cycle without timers, handlers or a viewport.

Design (docs/CLAUDE.md "Generated collections & bonding proxies"):
- Proxies are ordinary mesh objects, tagged add-on-owned, at identity
  transform with world-space vertices -- never displaced, never solve
  inputs. A refresh replaces only tagged proxies.
- Each body proxy carries per-vertex "BondDistance" (signed distance to
  the solver-meshed Soundboard; positive = inside it) and "BondStatus"
  (RGBA overlay color) attributes, so the overlay survives a reload.
- The overlay is a gpu draw handler, not a material: it shows in Solid
  mode without touching user materials or global shading, is drawn
  without depth testing (the glue interface is sandwiched between the
  bodies), and draws only proxies whose collection is visible -- the
  user switches views with Blender's own collection toggles.
"""

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import bpy
import numpy as np

from . import analysis, generated

BODY_ROLES = ('RIB', 'BRIDGE_PLATE', 'LINING')

BOND_DISTANCE_ATTR = "BondDistance"
BOND_STATUS_ATTR = "BondStatus"

NEAR_GAP_FACTOR = 5.0

_STATUS_ITEMS = [
    ('CONTACT', "Contact", "Body touches the Soundboard within tolerance"),
    ('NOT_TOUCHING', "Not Touching", "No contact within tolerance"),
    ('INTERPENETRATING', "Interpenetrating", "Body and Soundboard overlap by more than tolerance"),
    ('GEOMETRY_ERROR', "Geometry Error", "Body could not be repaired or meshed"),
]


# ---------------------------------------------------------------------------
# Result storage
# ---------------------------------------------------------------------------


class SoundboardLabBondResult(bpy.types.PropertyGroup):
    source_object: bpy.props.PointerProperty(type=bpy.types.Object)
    proxy_object: bpy.props.PointerProperty(type=bpy.types.Object)
    status: bpy.props.EnumProperty(items=_STATUS_ITEMS, default='NOT_TOUCHING')
    message: bpy.props.StringProperty()
    contact_area: bpy.props.FloatProperty()
    nearest_distance: bpy.props.FloatProperty()
    max_penetration: bpy.props.FloatProperty()


# ---------------------------------------------------------------------------
# Fingerprints (staleness)
# ---------------------------------------------------------------------------


def source_fingerprint(obj: bpy.types.Object, depsgraph) -> str:
    """Hash of the object's world transform plus evaluated vertex
    positions -- what the exported STL was built from. Rounded so float
    noise from re-evaluation doesn't read as a change."""
    eval_obj = obj.evaluated_get(depsgraph)
    mesh = eval_obj.to_mesh()
    try:
        coords = np.empty(len(mesh.vertices) * 3, dtype=np.float64)
        mesh.vertices.foreach_get("co", coords)
    finally:
        eval_obj.to_mesh_clear()
    matrix = np.array([list(row) for row in eval_obj.matrix_world], dtype=np.float64)
    h = hashlib.sha1()
    h.update(np.round(matrix, 6).tobytes())
    h.update(np.round(coords, 5).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Launch / apply (plain functions)
# ---------------------------------------------------------------------------


def find_bond_bodies(context) -> list[bpy.types.Object]:
    bodies = []
    for role in BODY_ROLES:
        bodies.extend(analysis.find_role_objects(context, role))
    return bodies


def start_bond_detection(
    context,
    soundboard: bpy.types.Object,
    bodies: list[bpy.types.Object],
    engine_python_path: str,
    tolerance: float,
) -> dict:
    """Export the Soundboard and `bodies` (world space), write the
    manifest and launch `soundboard_lab.bond_cli`. Returns a handle for
    `analysis.poll_solve` / `apply_bond_result`."""
    if not engine_python_path or not Path(engine_python_path).is_file():
        raise FileNotFoundError(
            "Engine Python (venv) path is not configured or does not exist (add-on preferences)"
        )
    if tolerance <= 0:
        raise analysis.SoundboardLabSceneError("Bond tolerance must be positive")
    if not bodies:
        raise analysis.SoundboardLabSceneError(
            "No Rib / Bridge Plate objects to check -- assign roles first"
        )

    work_dir = Path(tempfile.mkdtemp(prefix="soundboard_lab_bonds_"))
    depsgraph = context.evaluated_depsgraph_get()

    soundboard_stl = work_dir / "soundboard.stl"
    analysis.export_triangulated_stl(context, soundboard, soundboard_stl)
    manifest_bodies = []
    for index, body in enumerate(bodies):
        stl = work_dir / f"body_{index}.stl"
        analysis.export_triangulated_stl(context, body, stl)
        manifest_bodies.append(
            {"name": body.name, "stl": str(stl), "element_size": body.soundboard_lab.mesh_element_size}
        )
    manifest = {
        "tolerance": tolerance,
        "soundboard": {"stl": str(soundboard_stl), "element_size": soundboard.soundboard_lab.mesh_element_size},
        "bodies": manifest_bodies,
    }
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    output_json = work_dir / "bond_result.json"

    cmd = [
        engine_python_path,
        "-m", "soundboard_lab.bond_cli",
        "--manifest", str(manifest_path),
        "--output-json", str(output_json),
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return {
        "process": process,
        "output_json": output_json,
        "work_dir": work_dir,
        "soundboard_name": soundboard.name,
        "body_names": [b.name for b in bodies],
        "fingerprints": {o.name: source_fingerprint(o, depsgraph) for o in [soundboard, *bodies]},
    }


def bond_color(distance: float, tolerance: float) -> tuple[float, float, float, float]:
    """Overlay color for one proxy vertex. `distance` > 0 is inside the
    Soundboard (penetration)."""
    if distance > tolerance:
        return (1.0, 0.1, 0.1, 0.85)
    gap = abs(distance)
    if gap <= tolerance:
        t = gap / tolerance
        return (0.1 + 0.9 * t, 0.9, 0.2 - 0.1 * t, 0.85)
    if gap <= NEAR_GAP_FACTOR * tolerance:
        return (1.0, 0.5, 0.0, 0.6)
    return (0.6, 0.6, 0.6, 0.15)


def _build_body_proxy(name, collection, source, points, triangles, distances, tolerance, fingerprint):
    obj = generated.new_generated_mesh_object(name, collection, 'BOND_PROXY', source, points, triangles)
    mesh = obj.data
    dist_attr = mesh.attributes.new(BOND_DISTANCE_ATTR, 'FLOAT', 'POINT')
    dist_attr.data.foreach_set("value", [float(d) for d in distances])
    color_attr = mesh.color_attributes.new(BOND_STATUS_ATTR, 'FLOAT_COLOR', 'POINT')
    flat = [c for d in distances for c in bond_color(float(d), tolerance)]
    color_attr.data.foreach_set("color", flat)
    obj.display_type = 'WIRE'
    obj.soundboard_lab.source_fingerprint = fingerprint
    return obj


def apply_bond_result(context, result: dict, handle: dict) -> None:
    """Replace the tagged bonding proxies and the scene's bond results
    with `result` (bond_cli's JSON)."""
    scene = context.scene
    scene.soundboard_lab_bond_results.clear()
    generated.remove_generated_objects(scene, generated.BOND_PROXY_KINDS)
    _invalidate_overlay_cache()

    if result.get("status") != "ok":
        scene.soundboard_lab_bond_status = 'ERROR'
        scene.soundboard_lab_bond_message = result.get("message", "Unknown error")
        return

    _root, bonding_collection, _results = generated.ensure_generated_collections(scene)
    fingerprints = handle["fingerprints"]
    tolerance = float(result["tolerance"])

    soundboard = bpy.data.objects.get(handle["soundboard_name"])
    surface = result["soundboard_surface"]
    sb_proxy = generated.new_generated_mesh_object(
        f"Bond Reference: {handle['soundboard_name']}",
        bonding_collection,
        'SOUNDBOARD_PROXY',
        soundboard,
        surface["points"],
        surface["triangles"],
    )
    sb_proxy.display_type = 'WIRE'
    sb_proxy.soundboard_lab.source_fingerprint = fingerprints.get(handle["soundboard_name"], "")

    counts = {}
    for body in result["bodies"]:
        source = bpy.data.objects.get(body["name"])
        item = scene.soundboard_lab_bond_results.add()
        item.name = body["name"]
        item.source_object = source
        item.status = body["status"].upper()
        item.message = body.get("message", "")
        counts[item.status] = counts.get(item.status, 0) + 1
        if "points" not in body:
            continue
        item.contact_area = body["contact_area"]
        item.nearest_distance = body["nearest_distance"]
        item.max_penetration = body["max_penetration"]
        item.proxy_object = _build_body_proxy(
            f"Bond Proxy: {body['name']}",
            bonding_collection,
            source,
            body["points"],
            body["triangles"],
            body["signed_distances"],
            tolerance,
            fingerprints.get(body["name"], ""),
        )

    scene.soundboard_lab_bond_status = 'DONE'
    labels = {key: name for key, name, _desc in _STATUS_ITEMS}
    scene.soundboard_lab_bond_message = ", ".join(
        f"{count} {labels[status].lower()}" for status, count in counts.items()
    )


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------


def _bond_proxies(scene):
    return [o for o in scene.objects if o.soundboard_lab.generated_kind in generated.BOND_PROXY_KINDS]


def check_bond_staleness(scene, depsgraph, changed: set | None = None) -> int:
    """Recompute source fingerprints (only for `changed` source objects,
    if given) and set `bond_stale` on proxies whose source no longer
    matches. Writes only on an actual change, so calling this from a
    depsgraph handler can't loop. Returns the number of stale proxies."""
    fingerprints = {}
    stale_count = 0
    for proxy in _bond_proxies(scene):
        settings = proxy.soundboard_lab
        source = settings.source_object
        if source is not None and (changed is None or source.name in changed):
            if source.name not in fingerprints:
                fingerprints[source.name] = source_fingerprint(source, depsgraph)
            stale = fingerprints[source.name] != settings.source_fingerprint
            if stale != settings.bond_stale:
                settings.bond_stale = stale
                _invalidate_overlay_cache()
        elif source is None and not settings.bond_stale:
            settings.bond_stale = True  # source deleted
        stale_count += settings.bond_stale
    return stale_count


def result_is_stale(scene, item) -> bool:
    """A body's result is out of date if its own proxy or the Soundboard
    reference is."""
    if item.proxy_object is not None and item.proxy_object.soundboard_lab.bond_stale:
        return True
    if item.source_object is None:
        return True
    return any(
        o.soundboard_lab.bond_stale for o in _bond_proxies(scene)
        if o.soundboard_lab.generated_kind == 'SOUNDBOARD_PROXY'
    )


def gather_bonded_bodies(context) -> tuple[list[bpy.types.Object], list[str]]:
    """Bodies the solve ties to the Soundboard: those with a current
    CONTACT result. Returns `(bonded, excluded_names)`. Raises
    `SoundboardLabSceneError` if bodies exist but the bond results are
    missing or out of date -- solving with stale bonds would silently
    tie the wrong geometry.

    The Body/Lining body only takes part when the Boundary Condition is
    Body/Lining (it *is* the rim support then, so it must be bonded);
    under Fixed Edge / Free the board edge rule replaces it."""
    scene = context.scene
    use_lining = scene.soundboard_lab_boundary_condition == 'BODY_LINING'
    bodies = [
        b for b in find_bond_bodies(context)
        if use_lining or b.soundboard_lab.role != 'LINING'
    ]
    if not bodies:
        return [], []
    by_source = {
        item.source_object.name: item
        for item in scene.soundboard_lab_bond_results
        if item.source_object is not None
    }
    if scene.soundboard_lab_bond_status != 'DONE' or any(b.name not in by_source for b in bodies):
        raise analysis.SoundboardLabSceneError(
            "Bonded bodies have no bond results -- run Detect Bonds first"
        )
    if any(result_is_stale(scene, by_source[b.name]) for b in bodies):
        raise analysis.SoundboardLabSceneError("Bond results are out of date -- run Detect Bonds again")
    for b in bodies:
        if b.soundboard_lab.role == 'LINING' and by_source[b.name].status != 'CONTACT':
            raise analysis.SoundboardLabSceneError(
                f"Body/Lining '{b.name}' is not bonded to the Soundboard "
                f"({by_source[b.name].status.replace('_', ' ').lower()}) -- it must be for the Body/Lining boundary condition"
            )
    bonded = [b for b in bodies if by_source[b.name].status == 'CONTACT']
    excluded = [b.name for b in bodies if by_source[b.name].status != 'CONTACT']
    return bonded, excluded


def terminal_spec(obj: bpy.types.Object) -> dict | None:
    """Engine `terminal` entry for a Body/Lining object, else None."""
    settings = obj.soundboard_lab
    if settings.role != 'LINING':
        return None
    if settings.terminal_condition == 'SPRING' and settings.terminal_stiffness <= 0:
        raise analysis.SoundboardLabSceneError(
            f"Body/Lining '{obj.name}': Spring terminal needs a Terminal Stiffness above 0"
        )
    return {"condition": settings.terminal_condition.lower(), "stiffness": settings.terminal_stiffness}


@bpy.app.handlers.persistent
def _on_depsgraph_update(scene, depsgraph):
    sources = {
        o.soundboard_lab.source_object.name
        for o in _bond_proxies(scene)
        if o.soundboard_lab.source_object is not None
    }
    if not sources:
        return
    changed = set()
    for update in depsgraph.updates:
        id_ = update.id
        if isinstance(id_, bpy.types.Object) and (update.is_updated_geometry or update.is_updated_transform):
            name = id_.original.name
            if name in sources:
                changed.add(name)
    if changed:
        try:
            check_bond_staleness(scene, depsgraph, changed)
        except Exception as exc:
            print(f"Soundboard Lab: bond staleness check failed (non-fatal): {exc!r}")


# ---------------------------------------------------------------------------
# Viewport overlay (gpu draw handler; never registered in --background)
# ---------------------------------------------------------------------------

_draw_handle = None
_overlay_cache = {}
STALE_ALPHA_FACTOR = 0.3


def _invalidate_overlay_cache():
    _overlay_cache.clear()


def _proxy_batch(obj, shader):
    from gpu_extras.batch import batch_for_shader

    mesh = obj.data
    stale = obj.soundboard_lab.bond_stale
    key = (mesh.as_pointer(), stale)
    batch = _overlay_cache.get(key)
    if batch is not None:
        return batch
    color_attr = mesh.color_attributes.get(BOND_STATUS_ATTR)
    if color_attr is None:
        return None
    n = len(mesh.vertices)
    positions = np.empty(n * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", positions)
    colors = np.empty(n * 4, dtype=np.float32)
    color_attr.data.foreach_get("color", colors)
    colors = colors.reshape(n, 4)
    if stale:
        colors[:, 3] *= STALE_ALPHA_FACTOR
    mesh.calc_loop_triangles()
    indices = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", indices)
    batch = batch_for_shader(
        shader,
        'TRIS',
        {"pos": positions.reshape(n, 3), "color": colors},
        indices=indices.reshape(-1, 3),
    )
    _overlay_cache[key] = batch
    return batch


def _draw_overlay():
    import gpu

    context = bpy.context
    scene = context.scene
    if scene is None:
        return
    proxies = [
        o for o in scene.objects
        if o.soundboard_lab.generated_kind == 'BOND_PROXY' and o.type == 'MESH' and o.visible_get()
    ]
    if not proxies:
        return
    shader = gpu.shader.from_builtin('SMOOTH_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('NONE')
    gpu.state.depth_mask_set(False)
    try:
        for obj in proxies:
            batch = _proxy_batch(obj, shader)
            if batch is None:
                continue
            with gpu.matrix.push_pop():
                gpu.matrix.multiply_matrix(obj.matrix_world)
                batch.draw(shader)
    finally:
        gpu.state.blend_set('NONE')
        gpu.state.depth_test_set('LESS_EQUAL')
        gpu.state.depth_mask_set(True)


@bpy.app.handlers.persistent
def _on_load_post(*_args):
    _invalidate_overlay_cache()


# ---------------------------------------------------------------------------
# Operator + poll timer
# ---------------------------------------------------------------------------

_active_bond_handle = None


def _bond_poll_timer():
    global _active_bond_handle
    if _active_bond_handle is None:
        return None
    try:
        result = analysis.poll_solve(_active_bond_handle)
    except analysis.SoundboardLabSceneError as exc:
        result = {"status": "error", "message": str(exc)}
    if result is None:
        return 0.5
    handle, _active_bond_handle = _active_bond_handle, None
    apply_bond_result(bpy.context, result, handle)
    analysis._tag_ui_redraw()
    return None


class SOUNDBOARD_OT_detect_bonds(bpy.types.Operator):
    """Mesh the Soundboard and every Rib / Bridge Plate as the solver
    would, and classify each body's contact with the Soundboard"""

    bl_idname = "soundboard.detect_bonds"
    bl_label = "Detect Bonds"
    bl_options = {'REGISTER'}

    @classmethod
    def poll(cls, context):
        return context.scene.soundboard_lab_bond_status != 'RUNNING'

    def execute(self, context):
        global _active_bond_handle
        scene = context.scene
        try:
            soundboard = analysis.find_soundboard_object(context)
            prefs = analysis.get_preferences(context)
            _active_bond_handle = start_bond_detection(
                context,
                soundboard,
                find_bond_bodies(context),
                prefs.engine_python_path,
                scene.soundboard_lab_bond_tolerance,
            )
        except (analysis.SoundboardLabSceneError, FileNotFoundError) as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        scene.soundboard_lab_bond_status = 'RUNNING'
        scene.soundboard_lab_bond_message = "Meshing and classifying..."
        bpy.app.timers.register(_bond_poll_timer, first_interval=0.5)
        return {'FINISHED'}


_BOND_STATUS_ICONS = {
    'CONTACT': 'CHECKMARK',
    'NOT_TOUCHING': 'UNLINKED',
    'INTERPENETRATING': 'ERROR',
    'GEOMETRY_ERROR': 'CANCEL',
}


def draw_bond_section(layout, context) -> None:
    scene = context.scene
    box = layout.box()
    box.label(text="Bond Detection", icon='LINKED')
    box.prop(scene, "soundboard_lab_bond_tolerance")
    box.operator(SOUNDBOARD_OT_detect_bonds.bl_idname, icon='VIEWZOOM')

    status = scene.soundboard_lab_bond_status
    if status == 'RUNNING':
        box.label(text=scene.soundboard_lab_bond_message, icon='TIME')
        return
    if status == 'ERROR':
        box.label(text=scene.soundboard_lab_bond_message, icon='ERROR')
        return
    if status != 'DONE':
        return
    box.label(text=scene.soundboard_lab_bond_message)
    col = box.column(align=True)
    for item in scene.soundboard_lab_bond_results:
        row = col.row()
        row.label(text=item.name, icon=_BOND_STATUS_ICONS[item.status])
        if item.status == 'CONTACT':
            row.label(text=f"{item.contact_area:.0f} mm²")
        elif item.status == 'NOT_TOUCHING':
            row.label(text=f"gap {item.nearest_distance:.2f}")
        elif item.status == 'INTERPENETRATING':
            row.label(text=f"overlap {item.max_penetration:.2f}")
        else:
            row.label(text="geometry error")
        if item.status == 'GEOMETRY_ERROR' and item.message:
            col.label(text=item.message)
        if result_is_stale(scene, item):
            col.label(text="Out of date -- run Detect Bonds again", icon='ERROR')


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    SoundboardLabBondResult,
    SOUNDBOARD_OT_detect_bonds,
)


def register():
    global _draw_handle
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Scene.soundboard_lab_bond_tolerance = bpy.props.FloatProperty(
        name="Bond Tolerance",
        description="Contact gap/overlap tolerance (mesh units, e.g. mm)",
        default=0.1,
        min=1e-6,
        precision=3,
    )
    bpy.types.Scene.soundboard_lab_bond_status = bpy.props.EnumProperty(
        items=[
            ('IDLE', "Idle", "Bond detection not run yet"),
            ('RUNNING', "Running", "Bond detection subprocess in progress"),
            ('DONE', "Done", "Bond detection completed"),
            ('ERROR', "Error", "Bond detection failed"),
        ],
        default='IDLE',
    )
    bpy.types.Scene.soundboard_lab_bond_message = bpy.props.StringProperty()
    bpy.types.Scene.soundboard_lab_bond_results = bpy.props.CollectionProperty(type=SoundboardLabBondResult)
    bpy.app.handlers.depsgraph_update_post.append(_on_depsgraph_update)
    bpy.app.handlers.load_post.append(_on_load_post)
    if not bpy.app.background:
        _draw_handle = bpy.types.SpaceView3D.draw_handler_add(_draw_overlay, (), 'WINDOW', 'POST_VIEW')


def unregister():
    global _draw_handle, _active_bond_handle
    _active_bond_handle = None
    if _draw_handle is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_draw_handle, 'WINDOW')
        _draw_handle = None
    _invalidate_overlay_cache()
    for handlers, fn in (
        (bpy.app.handlers.depsgraph_update_post, _on_depsgraph_update),
        (bpy.app.handlers.load_post, _on_load_post),
    ):
        if fn in handlers:
            handlers.remove(fn)
    del bpy.types.Scene.soundboard_lab_bond_results
    del bpy.types.Scene.soundboard_lab_bond_message
    del bpy.types.Scene.soundboard_lab_bond_status
    del bpy.types.Scene.soundboard_lab_bond_tolerance
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
