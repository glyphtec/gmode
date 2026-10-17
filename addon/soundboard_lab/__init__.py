"""Soundboard Lab: Blender add-on entry point.

This module is currently a smoke test of the extension hooks the add-on
will use: N-panel tab + sub-panels, Object property storage, a
right-click context menu entry and a keymap shortcut. No solver logic
lives here (see docs/CLAUDE.md for the engine/UI boundary rule).
"""

import math

import bmesh
import bpy
from mathutils import Matrix, Vector

# Roles whose material may be orthotropic and therefore carries a grain axis
# (see docs/CLAUDE.md, "Material library"). Bridge plate / mass load / lining
# are isotropic by default and get no grain-orientation UI.
ORTHOTROPIC_CAPABLE_ROLES = {'SOUNDBOARD', 'RIB'}

# ---------------------------------------------------------------------------
# Property storage (backs the panels below)
# ---------------------------------------------------------------------------


class SoundboardLabObjectSettings(bpy.types.PropertyGroup):
    role: bpy.props.EnumProperty(
        name="Role",
        description="Soundboard Lab role assigned to this object",
        items=[
            ('NONE', "None", "Not part of the assembly"),
            ('SOUNDBOARD', "Soundboard", "Soundboard solid body"),
            ('RIB', "Rib / Stiffener", "Bracing or rib member"),
            ('BRIDGE_PLATE', "Bridge Plate", "Bridge plate body"),
            ('LINING', "Body / Lining", "Rim lining body, bonded to the soundboard at the glue strip"),
            ('POINT_ATTACHMENT', "Point Attachment", "Bridge feet / compensator point"),
            ('MASS_LOAD', "Mass Load", "Added mass"),
        ],
        default='NONE',
    )
    terminal_condition: bpy.props.EnumProperty(
        name="Terminal Condition",
        description=(
            "Condition applied to this Body/Lining object's surface nodes that are "
            "not part of a declared bond to another body"
        ),
        items=[
            ('FIXED', "Fixed", "Rigid bench/vice anchor"),
            ('FREE', "Free", "Unconstrained; floating-mass / string-suspended analogy"),
            ('SPRING', "Spring", "Grounded springs at the unbonded nodes"),
        ],
        default='FIXED',
    )
    terminal_stiffness: bpy.props.FloatProperty(
        name="Terminal Stiffness",
        description="Grounded spring stiffness used when Terminal Condition is Spring (placeholder)",
        default=0.0,
        min=0.0,
    )
    grain_rotation: bpy.props.FloatVectorProperty(
        name="Grain Orientation",
        description=(
            "Orthotropic material axis orientation relative to this object's local "
            "axes (X = radial, Y = longitudinal/grain, Z = tangential)"
        ),
        subtype='EULER',
        size=3,
        default=(0.0, 0.0, 0.0),
    )


class SoundboardLabMaterialSettings(bpy.types.PropertyGroup):
    """Physical-material kind, attached to bpy.types.Material.

    This is a stub: it currently carries only enough (`kind`) to gate
    role-driven UI such as the grain-orientation rosette gizmo below.
    Isotropic/orthotropic are distinct kinds (not one type with a switch,
    see docs/CLAUDE.md); the full per-kind constant sets are a follow-up.
    """

    kind: bpy.props.EnumProperty(
        name="Material Kind",
        description="Physical material kind assigned to this Material data-block",
        items=[
            ('ISOTROPIC', "Isotropic", "Single modulus / Poisson ratio (bridge plate, mass loads)"),
            ('ORTHOTROPIC', "Orthotropic", "Directional stiffness with a grain axis (spruce, rib composites)"),
        ],
        default='ISOTROPIC',
    )


def is_grain_orientable(obj):
    """True if obj's role + assigned material warrant a grain-axis UI."""
    if obj is None or obj.type != 'MESH':
        return False
    if obj.soundboard_lab.role not in ORTHOTROPIC_CAPABLE_ROLES:
        return False
    mat = obj.active_material
    return mat is not None and mat.soundboard_lab.kind == 'ORTHOTROPIC'


# ---------------------------------------------------------------------------
# Operator (stub, used by both the context menu and the keymap below)
# ---------------------------------------------------------------------------


class SOUNDBOARD_OT_set_role_lining(bpy.types.Operator):
    bl_idname = "soundboard.set_role_lining"
    bl_label = "Set Role: Body / Lining"
    bl_description = "Smoke-test stub: assign the Body / Lining role to selected objects"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for obj in context.selected_objects:
            obj.soundboard_lab.role = 'LINING'
        self.report({'INFO'}, "Soundboard Lab: Set Role: Body / Lining (stub)")
        return {'FINISHED'}


class SOUNDBOARD_OT_align_grain_world(bpy.types.Operator):
    bl_idname = "soundboard.align_grain_world"
    bl_label = "Align Grain to World Axes"
    bl_description = (
        "Reset the grain orientation to this object's local axes "
        "(X=radial, Y=longitudinal/grain, Z=tangential)"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return is_grain_orientable(context.object)

    def execute(self, context):
        context.object.soundboard_lab.grain_rotation = (0.0, 0.0, 0.0)
        self.report({'INFO'}, "Soundboard Lab: grain orientation reset to local axes")
        return {'FINISHED'}


class SOUNDBOARD_OT_align_grain_to_edge(bpy.types.Operator):
    bl_idname = "soundboard.align_grain_to_edge"
    bl_label = "Align Grain to Active Edge"
    bl_description = (
        "Set the grain (Y) axis to match the active mesh edge direction "
        "(edit mode, one edge selected) — snaps to a visible grain line"
    )
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.object
        if context.mode != 'EDIT_MESH' or not is_grain_orientable(obj):
            return False
        bm = bmesh.from_edit_mesh(obj.data)
        return any(e.select for e in bm.edges)

    def execute(self, context):
        obj = context.object
        bm = bmesh.from_edit_mesh(obj.data)
        active = bm.select_history.active
        edge = active if isinstance(active, bmesh.types.BMEdge) else next(
            (e for e in bm.edges if e.select), None
        )
        if edge is None:
            self.report({'WARNING'}, "Soundboard Lab: select an edge to align the grain to")
            return {'CANCELLED'}
        local_dir = (edge.verts[1].co - edge.verts[0].co).normalized()
        rotation = Vector((0.0, 1.0, 0.0)).rotation_difference(local_dir)
        obj.soundboard_lab.grain_rotation = rotation.to_euler()
        self.report({'INFO'}, "Soundboard Lab: grain aligned to active edge")
        return {'FINISHED'}


# ---------------------------------------------------------------------------
# Grain orientation rosette (interactive viewport gizmo)
# ---------------------------------------------------------------------------

# Dial colors match the axis convention: X=radial, Y=longitudinal/grain,
# Z=tangential (docs/CLAUDE.md).
GRAIN_AXIS_DIAL_COLORS = (
    (0.9, 0.15, 0.15),
    (0.15, 0.75, 0.2),
    (0.2, 0.35, 0.95),
)

# Extra rotation applied to the object's matrix so each dial's own Z axis
# (its rotation axis) lines up with the object's local X / Y / Z axis in
# turn. Verified numerically: Y+90 maps local Z -> local X; X-90 maps
# local Z -> local Y; identity leaves local Z -> local Z.
GRAIN_AXIS_DIAL_OFFSETS = (
    Matrix.Rotation(math.radians(90.0), 4, 'Y'),
    Matrix.Rotation(math.radians(-90.0), 4, 'X'),
    Matrix.Identity(4),
)

# Nested (not coincident) ring radii, like a gimbal/armillary sphere, so the
# three rings can't visually merge or hide inside a thin flat soundboard --
# the Z/tangential ring's plane is otherwise coincident with the board face.
GRAIN_AXIS_DIAL_SCALE = (1.0, 1.25, 1.5)


def _grain_rotation_get(index):
    """Build a dial get-callback bound to one grain_rotation component.

    GIZMO_GT_dial_3d's "offset" target is a plain scalar (confirmed via a
    RuntimeError: "index 1 must be below 1" -- Blender validates `index`
    against the *gizmo type's* declared target length, not our 3-component
    source property). target_set_prop(..., index=i) therefore cannot bind
    one ring per Euler component; target_set_handler with explicit
    get/set callbacks (the pattern Blender's own templates use for this
    exact case) is required instead.
    """

    def get_cb():
        obj = bpy.context.object
        return obj.soundboard_lab.grain_rotation[index] if obj else 0.0

    return get_cb


def _tag_ui_redraw():
    """Force the N-panel (and any other open editor) to refresh.

    target_set_handler's custom Python callback bypasses the automatic
    UI-redraw tagging that Blender's native target_set_prop binding gets
    for free (it goes through the normal RNA property-set path); without
    this, the N-panel's grain_rotation fields only catch up on an
    unrelated redraw (e.g. reselecting the object).

    This runs inside a gizmo modal callback, invoked on every mouse-move
    during a drag -- a context that can be more restricted than a normal
    operator's. If anything here raised, Blender's gizmo modal dispatch
    appears to treat the property-set call as failed and reverts the
    value to what it was at drag-start, so this must never propagate.
    """
    try:
        window = bpy.context.window
        if window is None:
            return
        for area in window.screen.areas:
            for region in area.regions:
                region.tag_redraw()
    except Exception as exc:
        print(f"Soundboard Lab: grain rosette redraw tag failed (non-fatal): {exc!r}")


def _grain_rotation_set(index):
    def set_cb(value):
        obj = bpy.context.object
        if obj is not None:
            obj.soundboard_lab.grain_rotation[index] = value
            _tag_ui_redraw()

    return set_cb


class SOUNDBOARD_GGT_grain_rosette(bpy.types.GizmoGroup):
    bl_idname = "SOUNDBOARD_GGT_grain_rosette"
    bl_label = "Grain Orientation Rosette"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'PERSISTENT'}

    @classmethod
    def poll(cls, context):
        return is_grain_orientable(context.object)

    def setup(self, context):
        self.dials = []
        axes = zip(GRAIN_AXIS_DIAL_COLORS, GRAIN_AXIS_DIAL_OFFSETS, GRAIN_AXIS_DIAL_SCALE)
        for i, (color, offset, scale) in enumerate(axes):
            try:
                dial = self.gizmos.new("GIZMO_GT_dial_3d")
                dial.target_set_handler(
                    "offset", get=_grain_rotation_get(i), set=_grain_rotation_set(i)
                )
                dial.color = color
                dial.color_highlight = tuple(min(c + 0.25, 1.0) for c in color)
                dial.alpha = 0.9
                dial.alpha_highlight = 1.0
                dial.line_width = 3
                # scale_basis is a constant-screen-size multiplier (like the
                # built-in Move/Rotate gizmo, which stays the same apparent
                # size regardless of zoom or object scale) -- NOT world
                # units. Per-axis scale nests the three rings (see above).
                dial.scale_basis = 1.2 * scale
                dial.use_draw_value = True
                # Without this, GIZMO_GT_dial_3d draws only a bare ring and
                # never the radial line marking the current angle value.
                dial.draw_options = {'ANGLE_VALUE'}
                self.dials.append((dial, offset))
            except Exception as exc:
                # A failure on one axis must not silently drop the rest.
                print(f"Soundboard Lab: grain rosette axis {i} failed to set up: {exc!r}")
        print(f"Soundboard Lab: grain rosette set up {len(self.dials)}/3 dials")
        self.refresh(context)

    def refresh(self, context):
        obj = context.object
        if obj is None or not self.dials:
            return
        base = obj.matrix_world.normalized()
        for dial, offset in self.dials:
            dial.matrix_basis = base @ offset


# ---------------------------------------------------------------------------
# N-panel tab ("Soundboard") and sub-panels
# ---------------------------------------------------------------------------


class SOUNDBOARD_PT_main(bpy.types.Panel):
    bl_label = "Soundboard Lab"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Soundboard"

    def draw(self, context):
        layout = self.layout
        layout.label(text="Soundboard Lab (smoke test)", icon='MOD_SOFT')


class SOUNDBOARD_PT_bodies(bpy.types.Panel):
    bl_label = "Bodies & Roles"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Soundboard"
    bl_parent_id = "SOUNDBOARD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if obj is not None:
            layout.prop(obj.soundboard_lab, "role")
            layout.operator(SOUNDBOARD_OT_set_role_lining.bl_idname, icon='MOD_SOFT')
        else:
            layout.label(text="No active object")


class SOUNDBOARD_PT_grain(bpy.types.Panel):
    bl_label = "Grain Orientation"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Soundboard"
    bl_parent_id = "SOUNDBOARD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if obj is None or obj.soundboard_lab.role not in ORTHOTROPIC_CAPABLE_ROLES:
            layout.label(text="Select a Soundboard / Rib object")
            return
        if not is_grain_orientable(obj):
            layout.label(text="Assign an Orthotropic material to edit grain orientation")
            return
        layout.prop(obj.soundboard_lab, "grain_rotation", text="")
        layout.label(text="Drag the rosette in the viewport, or type values above")
        layout.separator()
        layout.operator(SOUNDBOARD_OT_align_grain_world.bl_idname, icon='EMPTY_AXIS')
        layout.operator(SOUNDBOARD_OT_align_grain_to_edge.bl_idname, icon='EDGESEL')


class SOUNDBOARD_PT_object_role(bpy.types.Panel):
    """Quick-glance role readout in the Properties editor's Object tab.

    This mirrors (a subset of) the N-panel's Bodies & Roles sub-panel; the
    N-panel tab remains the primary editing surface.
    """

    bl_label = "Soundboard Lab"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'object'

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if obj is not None:
            layout.prop(obj.soundboard_lab, "role")
        else:
            layout.label(text="No active object")


class SOUNDBOARD_PT_material(bpy.types.Panel):
    """Physical material kind, in the Properties editor's Material tab.

    Full isotropic/orthotropic constant fields are a follow-up task (see
    docs/CLAUDE.md); this currently exposes just the kind selector that
    other UI (e.g. the grain-orientation rosette) gates on.
    """

    bl_label = "Soundboard Lab"
    bl_space_type = 'PROPERTIES'
    bl_region_type = 'WINDOW'
    bl_context = 'material'

    def draw(self, context):
        layout = self.layout
        mat = context.material
        if mat is not None:
            layout.prop(mat.soundboard_lab, "kind")
            layout.label(text="Physical constants: placeholder (follow-up task)")
        else:
            layout.label(text="No active material")


class SOUNDBOARD_PT_bonds(bpy.types.Panel):
    bl_label = "Bonds"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Soundboard"
    bl_parent_id = "SOUNDBOARD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        self.layout.label(text="Bond / tie-completeness status (placeholder)")


class SOUNDBOARD_PT_boundary(bpy.types.Panel):
    bl_label = "Boundary & Attachments"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Soundboard"
    bl_parent_id = "SOUNDBOARD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        obj = context.object
        if obj is not None and obj.soundboard_lab.role == 'LINING':
            layout.prop(obj.soundboard_lab, "terminal_condition")
            if obj.soundboard_lab.terminal_condition == 'SPRING':
                layout.prop(obj.soundboard_lab, "terminal_stiffness")
        else:
            layout.label(text="Select a Body / Lining object to configure its terminal condition")
        layout.separator()
        layout.label(text="Point attachments (placeholder)")


class SOUNDBOARD_PT_analysis(bpy.types.Panel):
    bl_label = "Analysis & Solve"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Soundboard"
    bl_parent_id = "SOUNDBOARD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        self.layout.label(text="Analysis sequence / solve launch (placeholder)")


class SOUNDBOARD_PT_results(bpy.types.Panel):
    bl_label = "Results & Verification"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Soundboard"
    bl_parent_id = "SOUNDBOARD_PT_main"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        self.layout.label(text="Mode viewer / MAC comparison (placeholder)")


# ---------------------------------------------------------------------------
# Right-click (Object) context menu entry
# ---------------------------------------------------------------------------


def draw_object_context_menu(self, context):
    self.layout.separator()
    self.layout.operator(SOUNDBOARD_OT_set_role_lining.bl_idname, icon='MOD_SOFT')


# ---------------------------------------------------------------------------
# Keymap
# ---------------------------------------------------------------------------

addon_keymaps = []


def register_keymaps():
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon
    if kc:
        km = kc.keymaps.new(name='Object Mode', space_type='EMPTY')
        kmi = km.keymap_items.new(
            SOUNDBOARD_OT_set_role_lining.bl_idname, 'V', 'PRESS', ctrl=True, shift=True
        )
        addon_keymaps.append((km, kmi))


def unregister_keymaps():
    for km, kmi in addon_keymaps:
        km.keymap_items.remove(kmi)
    addon_keymaps.clear()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

classes = (
    SoundboardLabObjectSettings,
    SoundboardLabMaterialSettings,
    SOUNDBOARD_OT_set_role_lining,
    SOUNDBOARD_OT_align_grain_world,
    SOUNDBOARD_OT_align_grain_to_edge,
    SOUNDBOARD_PT_main,
    SOUNDBOARD_PT_bodies,
    SOUNDBOARD_PT_grain,
    SOUNDBOARD_PT_object_role,
    SOUNDBOARD_PT_material,
    SOUNDBOARD_PT_bonds,
    SOUNDBOARD_PT_boundary,
    SOUNDBOARD_PT_analysis,
    SOUNDBOARD_PT_results,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    if not bpy.app.background:
        # GizmoGroups need a window manager; `--background` runs (smoke
        # tests, CI) have none and registration never fully completes,
        # so skip it there rather than hit "missing bl_rna" on unregister.
        bpy.utils.register_class(SOUNDBOARD_GGT_grain_rosette)
    bpy.types.Object.soundboard_lab = bpy.props.PointerProperty(type=SoundboardLabObjectSettings)
    bpy.types.Material.soundboard_lab = bpy.props.PointerProperty(type=SoundboardLabMaterialSettings)
    bpy.types.VIEW3D_MT_object_context_menu.append(draw_object_context_menu)
    register_keymaps()
    print("Soundboard Lab: registered")


def unregister():
    unregister_keymaps()
    bpy.types.VIEW3D_MT_object_context_menu.remove(draw_object_context_menu)
    del bpy.types.Material.soundboard_lab
    del bpy.types.Object.soundboard_lab
    if not bpy.app.background:
        bpy.utils.unregister_class(SOUNDBOARD_GGT_grain_rosette)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Soundboard Lab: unregistered")
