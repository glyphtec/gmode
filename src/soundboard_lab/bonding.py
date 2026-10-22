"""Multi-body bonding: contact-footprint detection between
independently-meshed bodies (see docs/CLAUDE.md "Boundary & Bonds UI").

Pure Python, no `bpy` imports (see docs/CLAUDE.md, "Engine/UI boundary").

Scope (decided with the user): role-scoped to the Soundboard only --
every non-Soundboard body (Rib, Bridge Plate, eventually Lining) is
checked against the Soundboard specifically, not against every other
body (no general N:M any-pair contact mapping in V1). This module
only answers "does this one body touch the soundboard, and where" --
it says nothing about bonding two non-Soundboard bodies to each other.

This is a detection step only: it finds candidate contact footprints
for a human to review (and, later, visualize as a color map) -- it
does not itself decide a body IS bonded, and it does not generate any
CalculiX `*TIE`/`*SURFACE` cards. Matches the project's established
"explicit, reviewable, not silently baked into solve" culture (see
the step-checklist / mesh-quality-preview precedents).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh


@dataclass(frozen=True)
class ContactFootprint:
    """Result of checking one non-Soundboard body's mesh against the
    Soundboard's mesh for contact, within `tolerance`.

    `body_contact_vertex_mask`/`body_vertex_distances` are indexed by
    the BODY mesh's own vertex order -- directly usable both as a
    candidate CalculiX `*TIE` slave node set (nodes where the mask is
    True) and as a per-vertex Color Attribute for visualization (the
    distances themselves, not just the boolean mask, so "how close"
    is visible even for non-contact vertices -- important for making
    incomplete/jagged bonding at edges visually obvious, per the
    user's own stated visualization goal).

    `soundboard_contact_face_indices` are indices into the
    SOUNDBOARD mesh's own face array -- the candidate CalculiX `*TIE`
    master face set (master must be face-based, see the real
    CalculiX 2.22 manual's *TIE/*SURFACE semantics, verified before
    the `*TIE` mechanics proof this session, not assumed).
    """

    body_contact_vertex_mask: np.ndarray
    body_vertex_distances: np.ndarray
    soundboard_contact_face_indices: np.ndarray

    @property
    def contact_fraction(self) -> float:
        """Fraction of the body's own vertices classified as in
        contact -- a quick, cheap completeness signal (e.g. "almost
        none of this rib actually touches the board" is an obvious
        red flag), not a substitute for the real visualization.
        """
        if len(self.body_contact_vertex_mask) == 0:
            return 0.0
        return float(self.body_contact_vertex_mask.sum()) / len(self.body_contact_vertex_mask)


def find_contact_footprint(
    body_points: np.ndarray,
    soundboard_mesh: trimesh.Trimesh,
    tolerance: float,
) -> ContactFootprint:
    """For every point in `body_points`, find the nearest point on
    `soundboard_mesh`'s surface (`trimesh.proximity.closest_point`,
    an exact nearest-point-on-triangle query, not a coarse bounding-
    volume approximation) and classify it as "in contact" if that
    distance is within `tolerance`.

    Takes a plain point array for the body side, not a full
    `trimesh.Trimesh` -- confirmed empirically (not assumed) that this
    matters: a body's own raw DESIGN-time STL (e.g. a simple box-
    topology rib with only 8 corner vertices) is far too sparse to
    detect contact along most of a real touching footprint -- it can
    only ever "see" the handful of corner vertices that happen to
    exist, nothing along the middle of a long touching edge/face.
    The real, accurate check needs the MESHED (post-gmsh) node
    positions, which are dense across the whole surface -- see
    `find_contact_footprint_from_meshed_body`. This function itself
    doesn't care where the points came from, by design, so it works
    correctly either way once given the right points.

    `tolerance` must be positive and is the user-controllable epsilon
    (default 0.1mm, per the user's own call -- "easy to see when
    modeling") -- deliberately a parameter here, never hardcoded.

    Both the body points and the soundboard mesh are expected in the
    same (world) coordinate space -- this function does no alignment
    of its own; getting the two bodies' geometry into a shared frame
    is the caller's job (for the addon, this is already true:
    `export_triangulated_stl` bakes `matrix_world` into every
    exported body).
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    _closest_points, distances, triangle_ids = trimesh.proximity.closest_point(soundboard_mesh, body_points)
    contact_mask = distances <= tolerance
    contacted_faces = np.unique(triangle_ids[contact_mask]) if contact_mask.any() else np.array([], dtype=int)

    return ContactFootprint(
        body_contact_vertex_mask=contact_mask,
        body_vertex_distances=distances,
        soundboard_contact_face_indices=contacted_faces,
    )


def find_contact_footprint_from_files(body_stl_path: Path, soundboard_stl_path: Path, tolerance: float) -> ContactFootprint:
    """Coarse/design-time convenience wrapper: uses the body's own raw
    (un-meshed) STL vertices. Fast and useful for early "does this
    even roughly line up" modeling feedback, but NOT what should back
    any real bond decision or `*TIE` card -- see this module's and
    `find_contact_footprint`'s own docstrings for why (a simple
    box-topology body has only its 8 corner vertices to sample from,
    regardless of how long the real touching footprint actually is).
    """
    body_mesh = trimesh.load(str(body_stl_path), force="mesh")
    soundboard_mesh = trimesh.load(str(soundboard_stl_path), force="mesh")
    return find_contact_footprint(body_mesh.vertices, soundboard_mesh, tolerance)


def find_contact_footprint_from_meshed_body(
    body_mesh_points: np.ndarray, soundboard_stl_path: Path, tolerance: float
) -> ContactFootprint:
    """The real, accurate check: `body_mesh_points` should be the
    MESHED (post-gmsh) node positions of the non-Soundboard body
    (e.g. `mesh_surface(...).points`), not its raw design-time STL
    vertices -- dense across the whole surface, so contact is
    correctly detected along the full touching footprint, not just at
    a handful of corner points.
    """
    soundboard_mesh = trimesh.load(str(soundboard_stl_path), force="mesh")
    return find_contact_footprint(body_mesh_points, soundboard_mesh, tolerance)
