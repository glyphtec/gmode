"""Mesh repair for Blender-authored surface geometry, upstream of gmsh.

Pure Python, no `bpy` imports (see docs/CLAUDE.md, "Engine/UI boundary").

gmsh's STL-remeshing workflow (mesh.py's `mesh_surface`) assumes a
closed, consistent, manifold input surface. Invalid topology (holes,
non-manifold edges, degenerate/duplicate faces, disconnected debris)
produces confusing, hard-to-diagnose "gmsh artifacts" that are
actually symptoms of unrepaired upstream geometry, not a gmsh bug --
this is why mesh repair is its own pipeline stage, named ahead of
meshing in docs/CLAUDE.md's Stack decision, not something gmsh itself
is expected to compensate for.

Uses `pymeshfix` (a Python wrapper around Marco Attene's MeshFix
algorithm, "A lightweight approach to repairing digitized polygon
meshes", The Visual Computer 2010) for the actual repair, and
`trimesh` for diagnosis both before and after -- `pymeshfix` can print
a "MeshFix could not fix everything" warning even on a fully
successful repair (observed empirically, not gated by its own
`verbose` flag), so success is verified independently via `trimesh`
rather than trusted from pymeshfix's own output.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymeshfix
import trimesh


class MeshRepairError(RuntimeError):
    """Raised when a mesh still isn't a valid watertight solid after
    repair -- must stop the pipeline here, before ever reaching gmsh,
    rather than handing ambiguous geometry downstream.
    """


@dataclass(frozen=True)
class RepairReport:
    was_already_valid: bool
    problems_found: list[str]
    original_face_count: int
    repaired_face_count: int
    components_removed: int


def diagnose(mesh: trimesh.Trimesh) -> list[str]:
    """Human-readable list of problems found; empty if the mesh is
    already a valid, repair-free input for gmsh.
    """
    problems = []
    if not mesh.is_watertight:
        problems.append("not watertight (open holes and/or non-manifold edges)")
    if not mesh.is_winding_consistent:
        problems.append("inconsistent face winding/normals")
    bad_face_count = int((~mesh.nondegenerate_faces()).sum())
    if bad_face_count:
        problems.append(f"{bad_face_count} degenerate (zero-area) face(s)")
    n_components = len(mesh.split(only_watertight=False))
    if n_components > 1:
        problems.append(f"{n_components} disconnected components (expected 1)")
    return problems


def repair_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, RepairReport]:
    """Diagnose and, if needed, repair a triangulated surface into a
    single watertight solid. Raises `MeshRepairError` if the result
    still isn't valid afterward.
    """
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    problems = diagnose(mesh)
    original_face_count = len(mesh.faces)

    if not problems:
        return (
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
            RepairReport(
                was_already_valid=True,
                problems_found=[],
                original_face_count=original_face_count,
                repaired_face_count=original_face_count,
                components_removed=0,
            ),
        )

    original_components = len(mesh.split(only_watertight=False))
    v_clean, f_clean = pymeshfix.clean_from_arrays(
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int32),
    )
    repaired = trimesh.Trimesh(vertices=v_clean, faces=f_clean, process=False)
    remaining_problems = diagnose(repaired)
    if remaining_problems:
        raise MeshRepairError(
            f"Mesh could not be repaired to a valid watertight solid: "
            f"{'; '.join(remaining_problems)} (found before repair: {'; '.join(problems)})"
        )

    repaired_components = len(repaired.split(only_watertight=False))
    report = RepairReport(
        was_already_valid=False,
        problems_found=problems,
        original_face_count=original_face_count,
        repaired_face_count=len(repaired.faces),
        components_removed=max(0, original_components - repaired_components),
    )
    return np.asarray(repaired.vertices), np.asarray(repaired.faces), report


def repair_stl(stl_path: Path, repaired_stl_path: Path) -> RepairReport:
    """Load an STL, repair if needed, and always write the (possibly
    unchanged) result to `repaired_stl_path` -- callers get one
    consistent path to feed to gmsh regardless of whether repair
    actually happened.

    Loads with default processing (`process=True`, not `False`): STL
    is an unindexed format -- every triangle gets its own 3 vertex
    coordinates, with no shared-vertex topology at all in the file
    itself. Without merging coincident vertices on load, an otherwise
    perfectly fine mesh looks like one disconnected single-triangle
    "component" per face (confirmed: triggered a false "10
    disconnected components" diagnosis on a 10-triangle test box).
    `repair_mesh`'s own `process=False` is correct by contrast, since
    it's handed already-indexed in-memory arrays that must not be
    silently altered before diagnosis runs.
    """
    mesh = trimesh.load(stl_path, process=True)
    vertices, faces, report = repair_mesh(mesh.vertices, mesh.faces)
    out_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    out_mesh.export(repaired_stl_path)
    return report
