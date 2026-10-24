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
    BodyInput,
    BondStatus,
    classify_bond,
    detect_bonds,
    find_contact_footprint,
    find_contact_footprint_from_files,
    find_contact_footprint_from_meshed_body,
    shell_surface,
    tet_boundary_skin,
)
from soundboard_lab.mesh import mesh_box, mesh_surface
from soundboard_lab.shell_mesh import ShellMesh

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

# ---------------------------------------------------------------------------
# Solver-mesh-based detection (detect_bonds and its building blocks)
# ---------------------------------------------------------------------------

PLATE = _box((100, 100, 2), translation=(0, 0, 1))  # z in [0, 2]


def _block_skin(origin, size=10.0, element_size=2.5):
    return tet_boundary_skin(mesh_box(size, size, size, element_size=element_size, origin=origin))


def test_tet_boundary_skin_is_closed_and_outward():
    points, triangles = _block_skin((0, 0, 0))
    skin = trimesh.Trimesh(points, triangles, process=False)
    assert skin.is_watertight
    assert skin.volume == pytest.approx(1000.0, rel=1e-6), "positive volume = outward winding"


def test_shell_surface_is_closed_solid_of_the_footprint():
    shell = ShellMesh(
        xy=np.array([[0, 0], [10, 0], [10, 20], [0, 20]], dtype=float),
        triangles=np.array([[0, 1, 2], [0, 2, 3]]),
        bottom_z=np.zeros(4),
        top_z=np.full(4, 2.0),
    )
    surface = shell_surface(shell)
    assert surface.is_watertight
    assert surface.volume == pytest.approx(400.0)


def test_block_resting_on_plate_is_contact_with_its_footprint_area():
    points, triangles = _block_skin((-5, -5, 2))
    detection = classify_bond(points, triangles, PLATE, tolerance=0.1)
    assert detection.status is BondStatus.CONTACT
    assert detection.contact_area == pytest.approx(100.0, rel=1e-6)
    assert detection.max_penetration <= 0.1


def test_block_above_gap_is_not_touching_and_reports_the_gap():
    points, triangles = _block_skin((-5, -5, 2.5))
    detection = classify_bond(points, triangles, PLATE, tolerance=0.1)
    assert detection.status is BondStatus.NOT_TOUCHING
    assert detection.contact_area == 0.0
    assert detection.nearest_distance == pytest.approx(0.5, abs=1e-6)


def test_block_sunk_into_plate_is_interpenetrating():
    points, triangles = _block_skin((-5, -5, 1.0))
    detection = classify_bond(points, triangles, PLATE, tolerance=0.1)
    assert detection.status is BondStatus.INTERPENETRATING
    assert detection.max_penetration == pytest.approx(1.0, abs=1e-6)


def test_soundboard_poking_into_body_between_its_nodes_is_interpenetrating():
    """A thin Soundboard edge entering a coarse body that has no skin
    node inside the Soundboard -- the body-side check alone sees only
    clearance, so the reverse (Soundboard-inside-body) check must catch it.
    """
    strip = _box((100, 6, 2), translation=(0, 0, 1))  # x in [-50, 50], y in [-3, 3], z in [0, 2]
    block = _box((10, 10, 10), translation=(50, 0, 1))  # straddles the strip's x=50 end
    detection = classify_bond(np.asarray(block.vertices), np.asarray(block.faces), strip, tolerance=0.1)
    assert not detection.penetrating_vertex_mask.any(), "sanity: no body vertex is inside the strip"
    assert detection.status is BondStatus.INTERPENETRATING
    assert detection.soundboard_penetration == pytest.approx(2.0, abs=1e-6)


def test_rejects_nonpositive_tolerance_for_classify_bond():
    points, triangles = _block_skin((-5, -5, 2))
    with pytest.raises(ValueError):
        classify_bond(points, triangles, PLATE, tolerance=0.0)


def test_detect_bonds_on_real_fixtures(tmp_path):
    """Real geometry against the solver-meshed Soundboard: every rib and
    the bridge plate are candidate bonds; rib1's contact area matches
    its real 300 x 6 mm glue footprint; an unreadable body is reported
    per body instead of aborting the others.
    """
    bodies = {name: BodyInput(DATA_DIR / f"{name}.stl", element_size=2.0)
              for name in ("rib1", "rib2", "rib3", "rib4", "bridge_plate2")}
    bodies["missing"] = BodyInput(tmp_path / "does_not_exist.stl", element_size=2.0)

    report = detect_bonds(DATA_DIR / "soundboard.stl", bodies, soundboard_element_size=10.0, tolerance=0.1)

    assert report.soundboard_surface.is_watertight
    for name in ("rib1", "rib2", "rib3", "rib4", "bridge_plate2"):
        assert report.bodies[name].status is BondStatus.CONTACT, (name, report.bodies[name])
    assert report.bodies["rib1"].detection.contact_area == pytest.approx(1800.0, rel=1e-3)

    missing = report.bodies["missing"]
    assert missing.status is BondStatus.GEOMETRY_ERROR
    assert missing.detection is None
    assert missing.message
