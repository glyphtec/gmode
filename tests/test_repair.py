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

from soundboard_lab.mesh import mesh_surface
from soundboard_lab.repair import MeshRepairError, diagnose, repair_mesh, repair_stl


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
