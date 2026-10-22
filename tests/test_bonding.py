"""Contact-footprint detection (docs/CLAUDE.md "Boundary & Bonds UI" --
bonding design). Synthetic torture tests first (precise, known-correct
geometry), then real-geometry verification on the actual bonding-
exercise fixtures (soundboard + tapered ribs + bridge plate).
"""

from pathlib import Path

import numpy as np
import pytest
import trimesh

from soundboard_lab.bonding import (
    find_contact_footprint,
    find_contact_footprint_from_files,
    find_contact_footprint_from_meshed_body,
)
from soundboard_lab.mesh import mesh_surface

DATA_DIR = Path(__file__).parent / "data" / "bonding_exercise"


def _box(extents, translation=(0.0, 0.0, 0.0)):
    b = trimesh.creation.box(extents=extents)
    b.apply_translation(translation)
    return b


def test_exactly_coincident_faces_are_fully_in_contact():
    """Two boxes stacked with EXACTLY coincident touching faces (no
    gap at all) -- every vertex on the touching face must register as
    in contact, at a tight tolerance.
    """
    bottom = _box((10, 10, 10), translation=(0, 0, 5))  # z in [0, 10]
    top = _box((10, 10, 10), translation=(0, 0, 15))  # z in [10, 20]

    footprint = find_contact_footprint(top.vertices, bottom, tolerance=0.01)

    # top's own bottom-face vertices (z == 10) must all be in contact.
    bottom_face_vertex_mask = np.isclose(top.vertices[:, 2], 10.0)
    assert bottom_face_vertex_mask.sum() == 4, "sanity check on the test box itself"
    assert all(footprint.body_contact_vertex_mask[bottom_face_vertex_mask])

    # top's own TOP-face vertices (z == 20, far from bottom) must NOT
    # be in contact.
    top_face_vertex_mask = np.isclose(top.vertices[:, 2], 20.0)
    assert not any(footprint.body_contact_vertex_mask[top_face_vertex_mask])

    assert len(footprint.soundboard_contact_face_indices) > 0


def test_gap_larger_than_tolerance_is_not_in_contact():
    """A real, deliberate gap bigger than the tolerance must register
    as NO contact at all -- not a false positive.
    """
    bottom = _box((10, 10, 10), translation=(0, 0, 5))  # z in [0, 10]
    top = _box((10, 10, 10), translation=(0, 0, 15.5))  # z in [10.5, 20.5] -- 0.5mm gap

    footprint = find_contact_footprint(top.vertices, bottom, tolerance=0.1)

    assert not footprint.body_contact_vertex_mask.any(), "0.5mm gap must not register as contact at 0.1mm tolerance"
    assert footprint.contact_fraction == 0.0
    assert len(footprint.soundboard_contact_face_indices) == 0


def test_gap_smaller_than_tolerance_is_in_contact():
    """The inverse of the above: a gap SMALLER than the tolerance must
    register as contact -- the tolerance is doing real work, not just
    "exact coincidence only".
    """
    bottom = _box((10, 10, 10), translation=(0, 0, 5))  # z in [0, 10]
    top = _box((10, 10, 10), translation=(0, 0, 15.05))  # z in [10.05, 20.05] -- 0.05mm gap

    footprint = find_contact_footprint(top.vertices, bottom, tolerance=0.1)

    bottom_face_vertex_mask = np.isclose(top.vertices[:, 2], 10.05)
    assert all(footprint.body_contact_vertex_mask[bottom_face_vertex_mask]), (
        "0.05mm gap must register as contact at 0.1mm tolerance"
    )


def test_rejects_nonpositive_tolerance():
    bottom = _box((10, 10, 10))
    top = _box((10, 10, 10), translation=(0, 0, 10))
    with pytest.raises(ValueError):
        find_contact_footprint(top.vertices, bottom, tolerance=0.0)
    with pytest.raises(ValueError):
        find_contact_footprint(top.vertices, bottom, tolerance=-0.1)


def test_real_rib_touches_real_soundboard_partially():
    """Real geometry, not synthetic: a real tapered rib against the
    real soundboard it was modeled flush against. Must find SOME
    contact (they really do touch -- confirmed via earlier live
    testing in Blender) but NOT the whole rib (only the bottom face
    touches; the top, sides, and tapered tips mostly don't).
    """
    footprint = find_contact_footprint_from_files(
        DATA_DIR / "rib1.stl", DATA_DIR / "soundboard.stl", tolerance=0.1
    )
    assert 0.0 < footprint.contact_fraction < 0.9, footprint.contact_fraction
    assert len(footprint.soundboard_contact_face_indices) > 0
    assert footprint.body_contact_vertex_mask.any()


def test_real_bridge_plate_touches_real_soundboard_partially():
    footprint = find_contact_footprint_from_files(
        DATA_DIR / "bridge_plate2.stl", DATA_DIR / "soundboard.stl", tolerance=0.1
    )
    assert 0.0 < footprint.contact_fraction < 0.9, footprint.contact_fraction
    assert len(footprint.soundboard_contact_face_indices) > 0


def test_raw_design_stl_undercounts_contact_vs_real_meshed_body():
    """Real bug, found empirically while building this detection code,
    not hypothetical: rib1.stl's raw design-time STL has only 8 total
    vertices (plain box-corner topology) -- far too sparse to detect
    contact along most of a 300mm touching edge, since it can only
    ever "see" the 4 bottom corners. The MESHED (post-gmsh) node
    positions are dense across the whole surface and must detect a
    much richer, more complete footprint on the exact same real
    geometry -- proof `find_contact_footprint_from_meshed_body` is
    doing meaningfully better than the coarse convenience wrapper,
    not just accepting a different-shaped input for its own sake.
    """
    coarse = find_contact_footprint_from_files(DATA_DIR / "rib1.stl", DATA_DIR / "soundboard.stl", tolerance=0.1)
    assert len(coarse.soundboard_contact_face_indices) <= 4, "sanity check: the coarse case really is this limited"

    meshed = mesh_surface(DATA_DIR / "rib1.stl", element_size=2.0)
    fine = find_contact_footprint_from_meshed_body(meshed.points, DATA_DIR / "soundboard.stl", tolerance=0.1)

    assert len(fine.soundboard_contact_face_indices) > len(coarse.soundboard_contact_face_indices) * 5, (
        len(fine.soundboard_contact_face_indices), len(coarse.soundboard_contact_face_indices)
    )
    # Still a genuine partial footprint (only the bottom touches), not
    # the whole meshed rib -- the fix made detection richer, not
    # indiscriminate.
    assert 0.0 < fine.contact_fraction < 0.5, fine.contact_fraction
