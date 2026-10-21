# Materials Library

`materials_library.blend` is a starter personal materials library for Soundboard Lab, per `docs/CLAUDE.md`'s "Material library" decision: physical material records live on native `bpy.types.Material` ID-blocks, so a `materials_library.blend` full of named Materials with their Soundboard Lab physical-property fields set is a real, reusable library across any project -- no custom add-on-side catalog needed.

It currently contains two starter materials (both placeholders -- see each material's own fields, and `src/soundboard_lab/materials.py`'s `SPRUCE_PLACEHOLDER`/`ISOTROPIC_PLACEHOLDER` docstrings, for the "replace with real lab data" caveat):

- **Spruce (Soundboard Lab Placeholder)** -- Orthotropic.
- **Generic Isotropic (Bridge Plate Placeholder)** -- Isotropic.

## Using it in a project

1. In your soundboard `.blend` file: **File > Append** (or **Link**, if you want live updates from the library file itself), browse into `materials_library.blend > Material`, and pick the material you want.
2. Assign it to an object via the Soundboard Lab N-panel's Bodies & Roles sub-panel (material picker) or the Properties editor's Material tab.
3. Edit the physical-property fields freely per-project -- Append copies the data-block into your file, it does not stay tied to the library (use Link instead if you want shared/live-updating materials across many projects).

You can also add this folder as an **Asset Library** (Preferences > File Paths > Asset Libraries) and mark materials as Assets (right-click a material in the Outliner/Shader Editor > Mark as Asset) for drag-and-drop reuse via the Asset Browser, instead of File > Append.

## Regenerating

`build_materials_library.py` is the generator script (not part of the installed extension) -- run it with:

```
blender --background --python library\build_materials_library.py
```

It overwrites `materials_library.blend`. Re-run it if you add new starter materials or the add-on's own placeholder defaults change.
