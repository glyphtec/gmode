# Modeling tips: building parts with Boolean cuts

Soundboard Lab needs every body to be a clean, closed solid. These habits keep
Boolean-built parts (waffle stiffeners, cut-out linings, pickup cavities) meshable.
They come from real failures; see `docs/CLAUDE.md`, "Mesh repair traps".

## 1. Make cutters overshoot, never coincide

A cutter whose face lies exactly on a face of the part leaves a zero-thickness
film or sliver behind. This is the most common cause of a part that "looks fine"
but won't mesh. Extend every cutter past **both** faces it passes through (a
millimetre or two above the top and below the bottom), even if that looks wasteful.

If you duplicate the soundboard, thicken it, and raise it onto the board, build the
cutters from a copy of the part and extrude them well below its underside, not from
the underside plane itself.

## 2. Keep cutter loops apart

- Don't let two cutter loops touch, overlap, or run almost parallel a hair apart.
  Near-touching edges produce sliver triangles and stray fragments in the result.
- If loops must overlap, join them into one cutter first so the result is a single
  clean outline.
- After "Simplify" on curves, run Merge by Distance (about 0.01 mm) so nearly
  duplicate points don't survive.

## 3. Trim edges with an overshooting cutter, then hunt for leftovers

When trimming the outer edge (for example to make room for a lining), make the
trimming body larger than the cut in every direction. Afterwards check for
bits you missed outside the final perimeter:

- Edit Mode, Select > Select All by Trait > Non Manifold: should select nothing.
- Mesh > Separate > By Loose Parts, then delete anything that isn't the part.

## 4. Check before you solve

- Apply scale and rotation (Ctrl+A) before any Boolean.
- Use the Exact (or Manifold, if your Blender offers it) Boolean solver.
- Run Mesh > Clean Up > Delete Loose, and Merge by Distance.
- Be careful with the 3D Print Toolbox's Make Manifold. It is best-effort and fills
  open edges with flat faces. If a Boolean left open edges at the bottom of a cut
  (a cutter that sat exactly on a face), Make Manifold can cap them with a
  zero-thickness sheet, turning a through-hole into an invalid film. Check
  Select > Select All by Trait > Non Manifold right after the Boolean, before
  cleaning up: open edges there mean the cutter needs to overshoot, not that the
  mesh needs patching.
- Then run Detect Bonds. A "geometry error" there points at the body, and the message
  now says what is wrong and where.

## What Soundboard Lab repairs for you, and what it won't

It automatically removes stray disconnected fragments, rebuilds flat caps whose
triangles overlap or are inverted, removes zero-thickness films (they become
through-holes), collapses tiny self-crossings where a closed outline overshoots its
start, and fills small holes. It refuses to continue, with a message, if a "repair"
would delete real geometry, so you never get an answer for a shape you didn't build.
The repair report in the result says what was changed.
