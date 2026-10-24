"""Add-on-owned generated collections and objects.

Layout (see docs/CLAUDE.md "Generated collections & bonding proxies"):

    Soundboard Lab          (root)
    +-- Bonding Proxies     (bond-detection proxies)
    +-- Analysis Results    (mode-shape display)

Switching between authored objects, bonding inspection and results is
done with Blender's own collection visibility toggles -- this module
only creates/finds the collections and keeps generated objects in them.

Collections are found by a custom-property tag, not by name, so a user
renaming them in the Outliner doesn't make the add-on create
duplicates. Generated objects are tagged via
`obj.soundboard_lab.generated_kind`; cleanup only ever touches objects
carrying that tag, never authored content.
"""

import bpy

COLLECTION_TAG = "soundboard_lab_generated"

ROOT_NAME = "Soundboard Lab"
BONDING_NAME = "Bonding Proxies"
RESULTS_NAME = "Analysis Results"

BOND_PROXY_KINDS = {'BOND_PROXY', 'SOUNDBOARD_PROXY'}


def _find_tagged_child(parent: bpy.types.Collection, tag: str):
    for child in parent.children:
        if child.get(COLLECTION_TAG) == tag:
            return child
    return None


def _ensure_tagged_child(parent: bpy.types.Collection, tag: str, name: str, color_tag: str | None = None):
    collection = _find_tagged_child(parent, tag)
    if collection is None:
        collection = bpy.data.collections.new(name)
        collection[COLLECTION_TAG] = tag
        if color_tag is not None:
            collection.color_tag = color_tag
        parent.children.link(collection)
    return collection


def ensure_generated_collections(scene: bpy.types.Scene):
    """Return `(root, bonding, results)`, creating whichever are missing."""
    root = _ensure_tagged_child(scene.collection, "root", ROOT_NAME, color_tag='COLOR_05')
    bonding = _ensure_tagged_child(root, "bonding", BONDING_NAME)
    results = _ensure_tagged_child(root, "results", RESULTS_NAME)
    return root, bonding, results


def is_generated(obj: bpy.types.Object) -> bool:
    return obj.soundboard_lab.generated_kind != 'NONE'


def link_exclusively(obj: bpy.types.Object, collection: bpy.types.Collection) -> None:
    """Make `collection` the object's only collection. Only ever called
    on add-on-owned objects."""
    for other in list(obj.users_collection):
        if other != collection:
            other.objects.unlink(obj)
    if collection not in obj.users_collection:
        collection.objects.link(obj)


def remove_generated_objects(scene: bpy.types.Scene, kinds: set[str]) -> int:
    """Delete this scene's generated objects of the given kinds (and
    their meshes, once unused). Returns how many were removed."""
    doomed = [obj for obj in scene.objects if obj.soundboard_lab.generated_kind in kinds]
    for obj in doomed:
        mesh = obj.data if obj.type == 'MESH' else None
        bpy.data.objects.remove(obj, do_unlink=True)
        if mesh is not None and mesh.users == 0:
            bpy.data.meshes.remove(mesh)
    return len(doomed)


def new_generated_mesh_object(
    name: str,
    collection: bpy.types.Collection,
    kind: str,
    source: bpy.types.Object | None,
    vertices,
    triangles,
) -> bpy.types.Object:
    """Create a tagged, world-space (identity transform) mesh object in
    `collection`. Never placed in an assembly role, never rendered."""
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in vertices], [], [tuple(t) for t in triangles])
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    collection.objects.link(obj)
    settings = obj.soundboard_lab
    settings.generated_kind = kind
    settings.role = 'NONE'
    settings.source_object = source
    obj.hide_render = True
    return obj
