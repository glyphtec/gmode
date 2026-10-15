"""Soundboard Lab: Blender add-on entry point.

This module is currently a smoke test of the extension hooks the add-on
will use: N-panel tab + sub-panels, Object property storage, a
right-click context menu entry and a keymap shortcut. No solver logic
lives here (see docs/CLAUDE.md for the engine/UI boundary rule).
"""

import bpy

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
    SOUNDBOARD_OT_set_role_lining,
    SOUNDBOARD_PT_main,
    SOUNDBOARD_PT_bodies,
    SOUNDBOARD_PT_object_role,
    SOUNDBOARD_PT_bonds,
    SOUNDBOARD_PT_boundary,
    SOUNDBOARD_PT_analysis,
    SOUNDBOARD_PT_results,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)
    bpy.types.Object.soundboard_lab = bpy.props.PointerProperty(type=SoundboardLabObjectSettings)
    bpy.types.VIEW3D_MT_object_context_menu.append(draw_object_context_menu)
    register_keymaps()
    print("Soundboard Lab: registered")


def unregister():
    unregister_keymaps()
    bpy.types.VIEW3D_MT_object_context_menu.remove(draw_object_context_menu)
    del bpy.types.Object.soundboard_lab
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
    print("Soundboard Lab: unregistered")
