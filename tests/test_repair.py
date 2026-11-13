"""Tests for mesh repair (repair.py), upstream of gmsh meshing.

Covers realistic Blender-authored-geometry defect patterns (holes,
non-manifold duplicate faces, disconnected debris, degenerate faces),
not just the trivial single-hole-in-a-box case used during initial
exploration, plus a genuinely unrepairable case and an end-to-end
integration proving a repaired mesh actually feeds gmsh successfully.
"""

import numpy as np
import pytest
import trimesh
from pathlib import Path

from soundboard_lab.mesh import mesh_surface
from soundboard_lab.repair import (
    MeshRepairError,
    _boundary_loops,
    _repair_topology,
    diagnose,
    find_zero_thickness_film,
    repair_mesh,
    repair_stl,
)


def _box():
    return trimesh.creation.box(extents=(10, 10, 10))


def test_clean_mesh_reports_already_valid():
    box = _box()
    vertices, faces, report = repair_mesh(box.vertices, box.faces)
    assert report.was_already_valid
    assert report.problems_found == []
    assert len(faces) == len(box.faces)
    # Unmodified: should return the same geometry, not silently alter
    # an already-correct mesh.
    assert np.allclose(sorted(vertices.flatten()), sorted(box.vertices.flatten()))


def test_hole_in_box_is_repaired():
    box = _box()
    keep = ~np.all(np.isclose(box.face_normals, [0, 0, 1]), axis=1)
    broken = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[keep], process=False)
    assert not broken.is_watertight

    vertices, faces, report = repair_mesh(broken.vertices, broken.faces)
    assert not report.was_already_valid
    assert "not watertight" in report.problems_found[0]

    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert repaired.is_watertight
    assert np.isclose(repaired.volume, 1000.0, rtol=1e-3)


def test_duplicate_face_nonmanifold_is_repaired():
    box = _box()
    dup_faces = np.vstack([box.faces, box.faces[0:1]])
    broken = trimesh.Trimesh(vertices=box.vertices, faces=dup_faces, process=False)
    assert not broken.is_watertight

    vertices, faces, report = repair_mesh(broken.vertices, broken.faces)
    assert not report.was_already_valid
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert repaired.is_watertight
    assert np.isclose(repaired.volume, 1000.0, rtol=1e-3)


def test_disconnected_debris_is_removed_and_reported():
    box = _box()
    extra_verts = np.array([[100.0, 100.0, 100.0], [101.0, 100.0, 100.0], [100.0, 101.0, 100.0]])
    debris_verts = np.vstack([box.vertices, extra_verts])
    n = len(box.vertices)
    debris_faces = np.vstack([box.faces, [[n, n + 1, n + 2]]])
    broken = trimesh.Trimesh(vertices=debris_verts, faces=debris_faces, process=False)
    assert len(broken.split(only_watertight=False)) == 2

    vertices, faces, report = repair_mesh(broken.vertices, broken.faces)
    assert not report.was_already_valid
    assert report.components_removed >= 1
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert repaired.is_watertight
    assert np.isclose(repaired.volume, 1000.0, rtol=1e-3)


def test_degenerate_face_is_repaired():
    box = _box()
    degen_faces = np.vstack([box.faces, [[0, 0, 1]]])
    broken = trimesh.Trimesh(vertices=box.vertices, faces=degen_faces, process=False)
    assert int((~broken.nondegenerate_faces()).sum()) == 1

    vertices, faces, report = repair_mesh(broken.vertices, broken.faces)
    assert not report.was_already_valid
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert repaired.is_watertight
    assert int((~repaired.nondegenerate_faces()).sum()) == 0


def _box_with_capped_top():
    """10 mm box whose flat top is triangulated with a cap: Z sits 1 um
    off the X-Y segment, as Blender's n-gon triangulation leaves them."""
    box = _box()
    keep = ~np.all(np.isclose(box.face_normals, [0, 0, 1]), axis=1)
    A, B, C, D = (-5, -5), (5, -5), (5, 5), (-5, 5)
    X, Y, Z = (-2, 0), (2, 0), (0, 0.001)
    top = [(A, B, Y), (A, Y, X), (A, X, D), (B, C, Y), (X, Y, Z), (Y, C, Z), (Z, C, D), (X, Z, D)]
    top_tris = np.array([[(x, y, 5.0) for x, y in tri] for tri in top])
    tris = np.vstack([box.vertices[box.faces[keep]], top_tris])
    return trimesh.Trimesh(**trimesh.triangles.to_kwargs(tris), process=True)


def test_coplanar_cap_triangles_are_flipped():
    capped = _box_with_capped_top()
    assert capped.is_watertight
    assert np.degrees(capped.face_angles).max() > 179

    vertices, faces, report = repair_mesh(capped.vertices, capped.faces)
    assert report.was_already_valid
    assert report.cap_triangles_flipped >= 1
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert repaired.is_watertight and repaired.is_winding_consistent
    assert np.isclose(repaired.volume, 1000.0)
    assert np.array_equal(vertices, capped.vertices)
    assert np.degrees(repaired.face_angles).max() < 170


def test_unrepairable_mesh_raises_clearly():
    """A single flat triangle can never become a watertight solid --
    must fail loudly, not hand ambiguous geometry to gmsh.
    """
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    with pytest.raises(MeshRepairError):
        repair_mesh(vertices, faces)


def test_repair_stl_then_gmsh_meshes_successfully(tmp_path):
    """End-to-end: a mesh with a hole that would otherwise confuse
    gmsh's classifySurfaces/addVolume (open surface loop) is repaired
    first, then actually meshes cleanly -- not just "trimesh says
    watertight", the full downstream step too.
    """
    box = _box()
    keep = ~np.all(np.isclose(box.face_normals, [0, 0, 1]), axis=1)
    broken = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[keep], process=False)
    broken_stl = tmp_path / "broken.stl"
    broken.export(broken_stl)

    repaired_stl = tmp_path / "repaired.stl"
    report = repair_stl(broken_stl, repaired_stl)
    assert not report.was_already_valid

    mesh = mesh_surface(repaired_stl, element_size=5.0)
    tetra_count = next(c.data.shape[0] for c in mesh.cells if c.type == "tetra10")
    assert tetra_count > 1


DATA_DIR = Path(__file__).parent / "data"


def _l_prism_with_bad_top():
    """L-shaped 10 mm-thick prism whose top cap is a naive fan from the
    vertex at (10, 0): that vertex can't see the whole L, so one fan
    triangle is inverted and overlaps its neighbours -- watertight and
    consistently wound, yet gmsh-hostile (what Blender's n-gon fill makes)."""
    outline = [(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)]
    n = len(outline)
    vertices = np.array([(x, y, 0.0) for x, y in outline] + [(x, y, 10.0) for x, y in outline])
    faces = []
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]  # outward side walls (outline is CCW)
    faces += [(0, 2, 1), (0, 3, 2), (0, 4, 3), (0, 5, 4)]  # bottom, fan from vertex 0, faces down
    faces += [(n + 1, n + 2, n + 3), (n + 1, n + 3, n + 4), (n + 1, n + 4, n + 5), (n + 1, n + 5, n + 0)]  # bad top
    return trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)


def test_inverted_cap_triangles_are_retriangulated():
    broken = _l_prism_with_bad_top()
    assert broken.is_watertight and broken.is_winding_consistent
    assert diagnose(broken) == []

    vertices, faces, report = repair_mesh(broken.vertices, broken.faces)
    assert report.caps_retriangulated == 1
    assert not report.was_already_valid
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert repaired.is_watertight and repaired.is_winding_consistent
    assert np.isclose(repaired.volume, 750.0)
    top = repaired.triangles[:, :, 2].min(axis=1) > 9.99
    assert np.all(repaired.face_normals[top][:, 2] > 0.99), "every top triangle must face up"
    assert np.isclose(repaired.area_faces[top].sum(), 75.0)


def test_retriangulated_cap_feeds_gmsh(tmp_path):
    broken = _l_prism_with_bad_top()
    stl = tmp_path / "bad_top.stl"
    broken.export(stl)
    repair_stl(stl, tmp_path / "fixed.stl")
    mesh = mesh_surface(tmp_path / "fixed.stl", element_size=3.0)
    assert next(c.data.shape[0] for c in mesh.cells if c.type == "tetra10") > 1


def test_consistent_caps_are_left_alone():
    box = _box()
    vertices, faces, report = repair_mesh(box.vertices, box.faces)
    assert report.caps_retriangulated == 0


def test_boundary_loops_split_at_a_pinch_vertex():
    """Two squares meeting at one corner: the boundary walk must give two
    simple loops, not one figure-eight."""
    xy = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 1], [2, 2], [1, 2]], dtype=float)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [2, 4], [4, 5], [5, 6], [6, 2]])
    # Vertex 2 is the shared corner: lower square 0-1-2-3, upper square 2-4-5-6.
    loops = _boundary_loops(edges, xy, region_on_left=True)
    assert loops is not None and sorted(len(loop) for loop in loops) == [4, 4]
    assert all(2 in loop for loop in loops)


def _ring_with_pocket_film():
    """20 mm square block, 10 mm thick, with a 10 mm square pocket cut down
    from the top exactly to the bottom face: the pocket floor lies on the
    flat underside, a zero-thickness film. Valid topology, not a valid solid."""
    outer = [(-10, -10), (10, -10), (10, 10), (-10, 10)]
    inner = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    vertices = np.array(
        [(x, y, 0.0) for x, y in outer] + [(x, y, 10.0) for x, y in outer]
        + [(x, y, 10.0) for x, y in inner] + [(x, y, 0.0) for x, y in inner]
    )
    o0, o1, i1, i0 = 0, 4, 8, 12  # first index of outer-bottom, outer-top, inner-top, inner-bottom
    faces = [(o0, o0 + 2, o0 + 1), (o0, o0 + 3, o0 + 2)]  # underside
    faces += [(i0, i0 + 1, i0 + 2), (i0, i0 + 2, i0 + 3)]  # pocket floor
    for k in range(4):
        m = (k + 1) % 4
        faces += [(o0 + k, o0 + m, o1 + m), (o0 + k, o1 + m, o1 + k)]  # outer walls
        faces += [(i1 + k, i1 + m, i0 + m), (i1 + k, i0 + m, i0 + k)]  # pocket walls
        faces += [(o1 + k, o1 + m, i1 + m), (o1 + k, i1 + m, i1 + k)]  # top rim
    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=False)
    mesh.fix_normals()
    return mesh


def test_zero_thickness_film_is_removed_as_a_through_hole(tmp_path):
    film = _ring_with_pocket_film()
    assert film.is_watertight and film.is_winding_consistent
    assert diagnose(film) == []
    assert find_zero_thickness_film(film.vertices, film.faces) is not None

    vertices, faces, report = repair_mesh(film.vertices, film.faces)
    assert np.isclose(report.film_area_removed, 100.0)
    repaired = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    assert repaired.is_watertight and repaired.is_winding_consistent
    assert find_zero_thickness_film(vertices, faces) is None
    assert np.isclose(repaired.volume, 20 * 20 * 10 - 10 * 10 * 10)  # the film had no volume

    stl = tmp_path / "pocket.stl"
    repaired.export(stl)
    mesh = mesh_surface(stl, element_size=3.0)
    assert next(c.data.shape[0] for c in mesh.cells if c.type == "tetra10") > 1


def test_ordinary_box_has_no_film():
    box = _box()
    assert find_zero_thickness_film(box.vertices, box.faces) is None


def test_repair_that_discards_geometry_is_refused(monkeypatch):
    """MeshFix returning a valid but much smaller solid means it threw
    real geometry away; that must stop the run, not feed gmsh wrong shape."""
    box = _box()
    keep = ~np.all(np.isclose(box.face_normals, [0, 0, 1]), axis=1)
    broken = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[keep], process=False)
    small = trimesh.creation.box(extents=(5, 5, 5))
    monkeypatch.setattr(
        "soundboard_lab.repair.pymeshfix.clean_from_arrays",
        lambda v, f: (np.asarray(small.vertices), np.asarray(small.faces)),
    )
    with pytest.raises(MeshRepairError, match="discarded"):
        repair_mesh(broken.vertices, broken.faces)


def test_debris_removal_does_not_run_meshfix(monkeypatch):
    """Disconnected debris is dropped directly; if what remains is a valid
    solid, MeshFix (which once deleted most of a real model) isn't invoked."""
    box = _box()
    n = len(box.vertices)
    vertices = np.vstack([box.vertices, [[100, 100, 100], [101, 100, 100], [100, 101, 100]]])
    faces = np.vstack([box.faces, [[n, n + 1, n + 2]]])

    def boom(v, f):
        raise AssertionError("pymeshfix should not run")

    monkeypatch.setattr("soundboard_lab.repair.pymeshfix.clean_from_arrays", boom)
    out_v, out_f, report = repair_mesh(vertices, faces)
    assert report.components_removed == 1 and not report.was_already_valid
    assert np.isclose(trimesh.Trimesh(out_v, out_f, process=False).volume, 1000.0)


def test_real_waffle_stiffener_with_films_and_debris():
    """luna_waffle_stiffener1.stl: six zero-volume debris slivers that once
    made MeshFix delete 58% of the stiffener, plus pocket floors lying on
    the underside (zero-thickness film) from the lining-trim step."""
    mesh = trimesh.load(DATA_DIR / "luna_waffle_stiffener1.stl", process=True)
    vertices, faces, report = _repair_topology(mesh.vertices, mesh.faces)
    assert report.components_removed == 6
    assert np.isclose(trimesh.Trimesh(vertices, faces, process=False).volume, 137661.8, rtol=1e-4)

    vertices, faces, report = repair_mesh(mesh.vertices, mesh.faces)
    assert np.isclose(report.film_area_removed, 16202, rtol=1e-3)
    repaired = trimesh.Trimesh(vertices, faces, process=False)
    assert repaired.is_watertight and repaired.is_winding_consistent
    assert find_zero_thickness_film(vertices, faces) is None
    assert np.isclose(repaired.volume, 137661.8, rtol=1e-3)


def test_untrimmed_waffle_stiffener_repairs_and_meshes(tmp_path):
    """luna_waffle_stiffener_untrimmed.stl: debris, a tiny film, inverted cap
    triangles and a 0.23 mm self-crossing where the outline's closing seam
    overshoots. Once repaired it must mesh, with the volume untouched."""
    report = repair_stl(DATA_DIR / "luna_waffle_stiffener_untrimmed.stl", tmp_path / "fixed.stl")
    assert report.components_removed == 4
    assert report.seam_crossings_collapsed == 2  # the same twist at the top and the bottom rim
    assert report.film_area_removed > 0 and report.caps_retriangulated == 2
    fixed = trimesh.load(tmp_path / "fixed.stl", process=True)
    assert fixed.is_watertight and fixed.is_winding_consistent
    assert np.isclose(fixed.volume, 184038.9, rtol=1e-4)
    mesh = mesh_surface(tmp_path / "fixed.stl", element_size=4.0)
    assert next(c.data.shape[0] for c in mesh.cells if c.type == "tetra10") > 1000
    # The add-on's default per-body element size (10 mm) failed here when the
    # rebuilt caps were flipped after rebuilding.
    mesh_surface(tmp_path / "fixed.stl", element_size=10.0)

