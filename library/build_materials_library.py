"""One-off generator for library/materials_library.blend -- run with:
    blender --background --python library\\build_materials_library.py

Builds a small starter materials library: named Material ID-blocks with
real Soundboard Lab physical-property fields set, saved as their own
.blend file. This is the actual mechanism docs/CLAUDE.md's "Material
library" decision describes for reuse across projects -- a user's own
materials_library.blend, pulled into any project via Blender's native
File > Append (or Link), optionally cataloged in the Asset Browser.

Not part of the installed extension itself; a standalone convenience
asset. Re-run this script (it overwrites the file) if the starter
materials' field set or defaults change.
"""
import sys

sys.path.insert(0, r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\addon")

import bpy
import soundboard_lab as addon

addon.register()

# Clear the default scene's starter objects/materials -- this file
# exists purely to hold Material ID-blocks for Append/Link, not a
# scene to open and work in directly.
for obj in list(bpy.data.objects):
    bpy.data.objects.remove(obj, do_unlink=True)

spruce = bpy.data.materials.new("Spruce (Soundboard Lab Placeholder)")
spruce.soundboard_lab.kind = 'ORTHOTROPIC'
# Already SPRUCE_PLACEHOLDER's own defaults (see SoundboardLabMaterialSettings) --
# set explicitly anyway so this file stays correct even if the add-on's
# own defaults are ever changed later.
spruce.soundboard_lab.density = 460.0
spruce.soundboard_lab.e_longitudinal = 13800.0
spruce.soundboard_lab.e_radial = 811.76
spruce.soundboard_lab.e_tangential = 593.4
spruce.soundboard_lab.g_lr = 828.0
spruce.soundboard_lab.g_lt = 621.0
spruce.soundboard_lab.g_rt = 41.4
spruce.soundboard_lab.nu_lr = 0.38
spruce.soundboard_lab.nu_lt = 0.40
spruce.soundboard_lab.nu_rt = 0.45
spruce.use_fake_user = True  # survives with 0 users (nothing in this file references it)

bridge_plate = bpy.data.materials.new("Generic Isotropic (Bridge Plate Placeholder)")
bridge_plate.soundboard_lab.kind = 'ISOTROPIC'
bridge_plate.soundboard_lab.density = 1600.0
bridge_plate.soundboard_lab.iso_e = 10000.0
bridge_plate.soundboard_lab.iso_nu = 0.3
bridge_plate.use_fake_user = True

assert len(bpy.data.objects) == 0
assert {"Spruce (Soundboard Lab Placeholder)", "Generic Isotropic (Bridge Plate Placeholder)"}.issubset(
    {m.name for m in bpy.data.materials}
)

out_path = r"C:\GlyphstoneCloud\CloudStation\LocalDev\gmode\library\materials_library.blend"
bpy.ops.wm.save_as_mainfile(filepath=out_path)
print(f"Wrote {out_path}")
