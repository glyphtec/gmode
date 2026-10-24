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
    cap_triangles_flipped: int = 0


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


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _triangle_cosines(v, tri) -> list[float]:
    """Cosine of the interior angle at each of `tri`'s three corners."""
    cosines = []
    for k in range(3):
        a = v[tri[k]]
        u, w = _sub(v[tri[(k + 1) % 3]], a), _sub(v[tri[(k + 2) % 3]], a)
        denom = (_dot(u, u) * _dot(w, w)) ** 0.5
        cosines.append(_dot(u, w) / denom if denom > 0 else 1.0)
    return cosines


def flip_cap_triangles(
    vertices: np.ndarray, faces: np.ndarray, max_angle_deg: float = 170.0, max_passes: int = 50
) -> tuple[np.ndarray, int]:
    """Remove "cap" triangles (one interior angle near 180 deg) by
    flipping their long edge with a coplanar neighbour.

    Blender triangulates a large flat n-gon (e.g. a rim lining's top
    face) into many caps whose apex sits microns off the opposite
    edge. They're valid topology, but gmsh's boundary-recovery then
    decides "segment and facet intersect" on floating-point roundoff
    alone -- a 1e-6 coordinate change from re-exporting flipped one
    lining between meshing and failing. Flipping the diagonal of a
    planar quad moves no vertex and changes no geometry, only the
    triangulation, so it's safe to always apply. Caps whose long edge
    is a crease (non-coplanar neighbour) are left alone.
    """
    v_arr = np.asarray(vertices, dtype=np.float64)
    v = [tuple(p) for p in v_arr.tolist()]
    f = [tuple(t) for t in np.asarray(faces, dtype=np.int64).tolist()]
    plane_tol = 1e-6 * max(float(np.ptp(v_arr, axis=0).max()), 1.0)
    cos_limit = float(np.cos(np.radians(max_angle_deg)))
    flips = 0
    edge_faces: dict[tuple[int, int], list[int]] = {}
    for i, tri in enumerate(f):
        for k in range(3):
            a, b = tri[k], tri[(k + 1) % 3]
            edge_faces.setdefault((min(a, b), max(a, b)), []).append(i)

    def _reassign(a: int, b: int, old: int, new: int) -> None:
        owners = edge_faces[(min(a, b), max(a, b))]
        owners[owners.index(old)] = new

    candidates = [i for i, tri in enumerate(f) if min(_triangle_cosines(v, tri)) <= cos_limit]
    for _ in range(max_passes):
        locked: set[int] = set()
        next_candidates: set[int] = set()
        for i in candidates:
            if i in locked:
                continue
            cosines = _triangle_cosines(v, f[i])
            apex_k = min(range(3), key=cosines.__getitem__)
            if cosines[apex_k] > cos_limit:
                continue
            p, q, r = (f[i][(apex_k + k) % 3] for k in range(3))
            owners = edge_faces.get((min(q, r), max(q, r)), [])
            if len(owners) != 2:
                continue
            j = owners[0] if owners[1] == i else owners[1]
            if j in locked:
                next_candidates.add(i)
                continue
            s = next(x for x in f[j] if x != q and x != r)
            fj = f[j]
            n_j = _cross(_sub(v[fj[1]], v[fj[0]]), _sub(v[fj[2]], v[fj[0]]))
            n_len = _dot(n_j, n_j) ** 0.5
            if n_len == 0 or abs(_dot(_sub(v[p], v[q]), n_j)) / n_len > plane_tol:
                continue
            ps = (min(p, s), max(p, s))
            if ps in edge_faces:
                continue
            tri_a, tri_b = (p, q, s), (p, s, r)
            # Both new triangles must face the same way as the quad, i.e.
            # the quad is convex and the new diagonal lies inside it.
            if any(_dot(_cross(_sub(v[t[1]], v[t[0]]), _sub(v[t[2]], v[t[0]])), n_j) <= 0 for t in (tri_a, tri_b)):
                continue
            ca, cb = _triangle_cosines(v, tri_a), _triangle_cosines(v, tri_b)
            if min(min(ca), min(cb)) <= cosines[apex_k]:
                continue
            f[i], f[j] = tri_a, tri_b
            del edge_faces[(min(q, r), max(q, r))]
            edge_faces[ps] = [i, j]
            _reassign(q, s, j, i)
            _reassign(r, p, i, j)
            locked.update((i, j))
            flips += 1
            # The new triangles, and caps across the quad's outer edges
            # (their neighbour just changed), may now flip.
            next_candidates.update((i, j))
            for a, b in ((p, q), (q, s), (s, r), (r, p)):
                next_candidates.update(edge_faces[(min(a, b), max(a, b))])
        if not next_candidates:
            break
        candidates = sorted(next_candidates)
    return np.asarray(f, dtype=np.int64).reshape(-1, 3), flips


def repair_mesh(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, RepairReport]:
    """Diagnose and, if needed, repair a triangulated surface into a
    single watertight solid, then flip coplanar cap triangles (see
    `flip_cap_triangles`). Raises `MeshRepairError` if the result
    still isn't valid afterward.
    """
    v_out, f_out, report = _repair_topology(vertices, faces)
    f_out, flips = flip_cap_triangles(v_out, f_out)
    if flips:
        report = RepairReport(**{**report.__dict__, "cap_triangles_flipped": flips})
    return v_out, f_out, report


def _repair_topology(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray, RepairReport]:
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
