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
    caps_retriangulated: int = 0
    film_area_removed: float = 0.0
    seam_crossings_collapsed: int = 0


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
    single watertight solid, then re-triangulate any flat cap whose
    triangulation overlaps itself (`retriangulate_flat_caps`) and flip
    coplanar cap triangles (`flip_cap_triangles`). Raises
    `MeshRepairError` if the result still isn't valid afterward.
    """
    v_out, f_out, report = _repair_topology(vertices, faces)
    v_out, f_out, n_seams = collapse_rim_crossings(v_out, f_out)
    if n_seams:
        report = RepairReport(
            **{
                **report.__dict__,
                "was_already_valid": False,
                "problems_found": [
                    *report.problems_found,
                    f"{n_seams} tiny self-crossing(s) in an outline collapsed (a closed curve overshooting its start)",
                ],
                "seam_crossings_collapsed": n_seams,
            }
        )
    film = find_zero_thickness_film(v_out, f_out)
    v_out, f_out, n_caps = retriangulate_flat_caps(v_out, f_out, film[0] if film is not None else None)
    if film is not None:
        remaining = find_zero_thickness_film(v_out, f_out)
        if remaining is not None:
            pairs, area = remaining
            centres = v_out[f_out[pairs[:, 0]]].mean(axis=1)
            (x0, y0, z0), (x1, y1, _) = centres.min(axis=0), centres.max(axis=0)
            raise MeshRepairError(
                f"The model has a zero-thickness film that could not be removed automatically: {area:.0f} mm^2 "
                f"of upward-facing surface lies exactly on top of downward-facing surface near Z = {z0:.3f} "
                f"(X {x0:.0f}..{x1:.0f}, Y {y0:.0f}..{y1:.0f}). It usually comes from a pocket or cut made "
                "exactly flush with a face (or capped over by Make Manifold): make the cutter go slightly "
                "past that face, then re-run."
            )
        report = RepairReport(
            **{
                **report.__dict__,
                "problems_found": [
                    *report.problems_found,
                    f"zero-thickness film removed ({film[1]:.0f} mm^2 -- now a through-hole)",
                ],
                "film_area_removed": film[1],
            }
        )
    if n_caps:
        report = RepairReport(
            **{
                **report.__dict__,
                "was_already_valid": False,
                "problems_found": [
                    *report.problems_found,
                    f"{n_caps} flat cap(s) rebuilt (overlapping, inverted or film-covered)",
                ],
                "caps_retriangulated": n_caps,
            }
        )
    f_out, flips = flip_cap_triangles(v_out, f_out)
    if flips:
        report = RepairReport(**{**report.__dict__, "cap_triangles_flipped": flips})
    return v_out, f_out, report


# MeshFix may only add surface (filling holes) or drop slivers; losing more
# than this fraction of the surface area means it threw real geometry away
# (seen: 58% of a waffle stiffener's volume deleted, with no error).
MAX_REPAIR_AREA_LOSS = 0.05


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

    components = mesh.split(only_watertight=False)
    original_components = len(components)
    if original_components > 1:
        # Drop disconnected debris (e.g. zero-thickness slivers left by a
        # botched boolean) ourselves, first: if what remains is a valid
        # solid, MeshFix never has to touch it.
        mesh = max(components, key=lambda c: float(c.area))
        if not diagnose(mesh):
            return (
                np.asarray(mesh.vertices),
                np.asarray(mesh.faces),
                RepairReport(
                    was_already_valid=False,
                    problems_found=problems,
                    original_face_count=original_face_count,
                    repaired_face_count=len(mesh.faces),
                    components_removed=original_components - 1,
                ),
            )

    area_before = float(mesh.area)
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
    if float(repaired.area) < (1.0 - MAX_REPAIR_AREA_LOSS) * area_before:
        raise MeshRepairError(
            f"Automatic repair would have discarded {100 * (1 - repaired.area / area_before):.0f}% of the "
            f"model's surface ({original_face_count} -> {len(repaired.faces)} faces), which means it "
            "deleted real geometry rather than fixing defects. Fix the model instead -- e.g. remove "
            "self-intersections / overlapping or doubled faces, and make the solid a single closed "
            f"manifold shape (found before repair: {'; '.join(problems)})"
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


def _boundary_loops(directed_edges: np.ndarray, xy: np.ndarray, region_on_left: bool) -> list[list[int]] | None:
    """Closed vertex loops from a region's directed boundary edges (each
    edge directed as the region's own winding has it, so the region lies
    on one fixed side). Where two loops touch at a pinch vertex, the
    walk takes the tightest turn toward the region, which splits them
    into separate simple loops rather than one tangled one. None if the
    edges don't close up."""
    out: dict[int, list[int]] = {}
    remaining = set()
    for a, b in directed_edges.tolist():
        out.setdefault(a, []).append(b)
        remaining.add((a, b))

    def turn(a: int, b: int, c: int) -> float:
        back = np.arctan2(xy[a, 1] - xy[b, 1], xy[a, 0] - xy[b, 0])
        onward = np.arctan2(xy[c, 1] - xy[b, 1], xy[c, 0] - xy[b, 0])
        rotation = (back - onward) if region_on_left else (onward - back)
        return float(rotation % (2 * np.pi)) or 2 * np.pi

    loops = []
    while remaining:
        start = min(remaining)
        remaining.discard(start)
        loop, edge = [start[0]], start
        while True:
            a, b = edge
            candidates = [(b, c) for c in out.get(b, ()) if (b, c) in remaining or (b, c) == start]
            if not candidates:
                return None
            nxt = min(candidates, key=lambda e: turn(a, b, e[1]))
            if nxt == start:
                break
            loop.append(b)
            remaining.discard(nxt)
            edge = nxt
        if len(loop) < 3:
            return None
        loops.append(loop)
    return loops


def _signed_area_xy(xy: np.ndarray) -> float:
    return 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    x, y = point
    x1, y1 = polygon[:, 0], polygon[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    straddles = (y1 > y) != (y2 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
    return bool(np.count_nonzero(straddles & (x < x_cross)) % 2)


def _triangulate_region(
    v: np.ndarray, loops: list[list[int]], depths: list[int], parents: list[int | None]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Fresh 2D triangulation of the planar region bounded by `loops`
    (even depth = outer boundary or island, odd = hole), reusing the
    loops' own vertices so the cap still meets the side walls exactly.
    Returns (new_points, triangles) with triangle indices into
    `[v..., new_points...]`, or None if gmsh can't do it."""
    import gmsh

    big = 10.0 * float(np.ptp(v, axis=0).max())
    z_mid = float(np.median(v[np.concatenate(loops), 2]))
    point_tag: dict[int, int] = {}
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("flat_cap")

        def loop_tag(loop: list[int]) -> int:
            tags = []
            for vi in loop:
                if vi not in point_tag:  # loops that touch at a pinch vertex share its point
                    point_tag[vi] = gmsh.model.geo.addPoint(v[vi, 0], v[vi, 1], 0.0, big)
                tags.append(point_tag[vi])
            lines = [gmsh.model.geo.addLine(tags[i], tags[(i + 1) % len(tags)]) for i in range(len(tags))]
            return gmsh.model.geo.addCurveLoop(lines)

        tags = [loop_tag(loop) for loop in loops]
        for i, depth in enumerate(depths):
            if depth % 2 == 0:
                holes = [tags[j] for j in range(len(loops)) if parents[j] == i]
                gmsh.model.geo.addPlaneSurface([tags[i], *holes])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.asarray(node_coords, dtype=np.float64).reshape(-1, 3)
        node_to_row = {int(t): i for i, t in enumerate(node_tags)}
        original_row = {}
        for vi, ptag in point_tag.items():
            tags_at_point, _, _ = gmsh.model.mesh.getNodes(0, ptag)
            original_row[node_to_row[int(tags_at_point[0])]] = vi
        elem_types, _, elem_nodes = gmsh.model.mesh.getElements(dim=2)
        tris = [np.asarray(n, dtype=np.int64).reshape(-1, 3) for t, n in zip(elem_types, elem_nodes) if t == 2]
    except Exception:
        return None
    finally:
        gmsh.finalize()
    if not tris:
        return None
    tris = np.vstack(tris)
    tri_rows = np.vectorize(node_to_row.get)(tris)

    new_rows = [r for r in range(len(coords)) if r not in original_row]
    new_index = {r: len(v) + i for i, r in enumerate(new_rows)}
    mapping = {**original_row, **new_index}
    new_points = np.column_stack([coords[new_rows, :2], np.full(len(new_rows), z_mid)]) if new_rows else np.zeros((0, 3))
    return new_points, np.vectorize(mapping.get)(tri_rows)


def collapse_rim_crossings(
    vertices: np.ndarray, faces: np.ndarray, plane_tol: float = 1e-4, max_connector: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, int]:
    """Remove tiny bow-ties from flat caps' outlines: two rim edges that
    cross each other where their far ends are joined by a very short edge.
    This is what a closed curve leaves when its start and end overshoot
    each other at the seam (seen: a 0.23 mm twist at the bottom of a waffle
    stiffener's outline). The walls overlap there, and gmsh rejects the
    result as "overlapping facets".

    The two joined end vertices are merged at the crossing point -- the
    shortest possible change, well under `max_connector` x model size -- and
    the faces that collapse to nothing are dropped. Crossings with no short
    connecting edge are left alone: they need a human.

    Returns (vertices, faces, number of crossings collapsed).
    """
    from rtree import index

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    size = max(float(np.ptp(v, axis=0).max()), 1.0)
    flat = np.nonzero(np.ptp(v[f][:, :, 2], axis=1) <= plane_tol * size)[0]
    if len(flat) == 0:
        return v, f, 0
    edges = np.sort(f[flat][:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    rim = unique_edges[counts == 1]
    if len(rim) < 4:
        return v, f, 0
    rim_set = {(int(a), int(b)) for a, b in rim}
    start, end = v[rim[:, 0], :2], v[rim[:, 1], :2]
    tree = index.Index(properties=index.Property(dimension=2))
    for i in range(len(rim)):
        tree.insert(i, (*np.minimum(start[i], end[i]), *np.maximum(start[i], end[i])))

    merge: dict[int, int] = {}
    target: dict[int, np.ndarray] = {}
    for i in range(len(rim)):
        for j in tree.intersection((*np.minimum(start[i], end[i]), *np.maximum(start[i], end[i]))):
            if j <= i or set(rim[i]) & set(rim[j]):
                continue
            p, d1 = start[i], end[i] - start[i]
            r, d2 = start[j], end[j] - start[j]
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-14:
                continue
            t = ((r - p)[0] * d2[1] - (r - p)[1] * d2[0]) / den
            u = ((r - p)[0] * d1[1] - (r - p)[1] * d1[0]) / den
            if not (1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9):
                continue
            crossing = p + t * d1
            for a in rim[i]:
                for b in rim[j]:
                    key = (int(min(a, b)), int(max(a, b)))
                    if key in rim_set and np.linalg.norm(v[a, :2] - v[b, :2]) <= max_connector * size:
                        merge[int(b)] = int(a)
                        target[int(a)] = crossing
    if not merge:
        return v, f, 0

    root = np.arange(len(v))
    for b, a in merge.items():
        root[b] = a
    v_out = v.copy()
    for a, crossing in target.items():
        group = [a] + [b for b, r in merge.items() if r == a]
        v_out[a, :2] = crossing
        v_out[a, 2] = v[group, 2].mean()
    f_out = root[f]
    f_out = f_out[(f_out[:, 0] != f_out[:, 1]) & (f_out[:, 1] != f_out[:, 2]) & (f_out[:, 2] != f_out[:, 0])]
    used = np.unique(f_out)
    remap = np.full(len(v_out), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v_out[used], remap[f_out], len(target)


# A cap triangle facing against the cap's outward direction (an inverted
# fold) bigger than this fraction of the cap's area gets the cap rebuilt;
# smaller is float noise on a sliver.
INVERTED_AREA_TOL = 1e-5


def find_zero_thickness_film(
    vertices: np.ndarray, faces: np.ndarray, plane_tol: float = 1e-4
) -> tuple[np.ndarray, float] | None:
    """Horizontal sheets with no thickness: an up-facing flat triangle
    lying (within `plane_tol` x model size) on top of a down-facing one.
    A solid can't have its inside on both sides of a plane, so these are
    zero-thickness films, typically a pocket cut exactly flush with the
    underside by a boolean (or capped over by Make Manifold). They are valid
    topology (trimesh and MeshFix accept them) but gmsh rejects them as
    "overlapping facets".

    Returns (pairs, area): `pairs` is a (k, 2) array of (up-facing triangle,
    the down-facing triangle it lies on) indices, `area` their total
    up-facing area in mm^2. None if there is no film.
    """
    from rtree import index

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    tol = plane_tol * max(float(np.ptp(v, axis=0).max()), 1.0)
    tri = v[f]
    flat = np.ptp(tri[:, :, 2], axis=1) <= tol
    cross_z = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2]
    area = 0.5 * np.abs(cross_z)
    up = np.nonzero(flat & (cross_z > 0))[0]
    down = np.nonzero(flat & (cross_z < 0))[0]
    if len(up) == 0 or len(down) == 0:
        return None

    xy = tri[:, :, :2]
    z = tri[:, :, 2].mean(axis=1)
    tree = index.Index(properties=index.Property(dimension=2))
    for i in down:
        tree.insert(int(i), (*xy[i].min(axis=0), *xy[i].max(axis=0)))

    def inside(p: np.ndarray, t: np.ndarray) -> bool:
        a, b, c = t
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if d == 0:
            return False
        l1 = ((b[1] - c[1]) * (p[0] - c[0]) + (c[0] - b[0]) * (p[1] - c[1])) / d
        l2 = ((c[1] - a[1]) * (p[0] - c[0]) + (a[0] - c[0]) * (p[1] - c[1])) / d
        return l1 > 0 and l2 > 0 and l1 + l2 < 1

    pairs = []
    for i in up:
        centre = xy[i].mean(axis=0)
        for j in tree.intersection((*centre, *centre)):
            if abs(z[i] - z[j]) <= tol and inside(centre, xy[j]):
                pairs.append((int(i), int(j)))
                break
    if not pairs:
        return None
    pairs_array = np.array(pairs)
    film_area = float(area[pairs_array[:, 0]].sum())
    return pairs_array, film_area


def retriangulate_flat_caps(
    vertices: np.ndarray,
    faces: np.ndarray,
    film_pairs: np.ndarray | None = None,
    plane_tol: float = 1e-4,
    area_tol: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Re-triangulate horizontal (constant-Z) flat regions whose own
    triangulation is self-inconsistent, e.g. a top or bottom cap where
    Blender's n-gon fill left inverted, overlapping triangles. Such a
    mesh is still topologically watertight (and trimesh calls it valid),
    but two boundary facets overlap, which gmsh rejects with "Invalid
    boundary mesh (overlapping facets)" when it builds the volume.

    A region is "inconsistent" when its triangles' total area differs
    from the area of the polygon its boundary loops enclose (outer minus
    holes): in a clean cap they are equal. The boundary loops (taken
    where the region meets the side walls) don't depend on the cap's own
    triangulation, so the cap is replaced by a fresh 2D triangulation of
    them; the side walls and the watertightness are untouched. Consistent
    caps, regions with non-manifold or pinched boundaries, and non-flat
    surfaces are left alone. Horizontal only, matching the world-Z thin
    axis the rest of the pipeline assumes.

    `film_pairs` (from `find_zero_thickness_film`) merges each zero-thickness
    film with the underside it lies on into one region, always rebuilt. The
    film's own boundary loops become holes in that new surface: the film
    has no volume, so dropping it just turns its outline into the
    through-hole it was meant to be.

    Returns (vertices, faces, number of caps re-triangulated).
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return v, f, 0
    tol = plane_tol * max(float(np.ptp(v, axis=0).max()), 1.0)
    face_z = v[f][:, :, 2]
    flat = np.nonzero(np.ptp(face_z, axis=1) <= tol)[0]
    if len(flat) < 2:
        return v, f, 0

    edges = np.sort(f[:, [[0, 1], [1, 2], [2, 0]]], axis=2).reshape(-1, 2)
    unique_edges, edge_id = np.unique(edges, axis=0, return_inverse=True)
    edge_id = edge_id.reshape(-1, 3)
    edge_total = np.bincount(edge_id.ravel(), minlength=len(unique_edges))

    # Flat faces sharing an edge and the same Z level form one region.
    z_mean = face_z.mean(axis=1)
    local = {int(face): i for i, face in enumerate(flat)}
    rows, cols = [], []
    owners: dict[int, int] = {}
    for face in flat:
        for e in edge_id[face]:
            other = owners.setdefault(int(e), int(face))
            if other != face and abs(z_mean[other] - z_mean[face]) <= tol:
                rows.append(local[other])
                cols.append(local[int(face)])
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(flat), len(flat)))
    n_regions, label = connected_components(graph, directed=False)

    film_regions: set[int] = set()
    if film_pairs is not None and len(film_pairs):
        linked = [(label[local[u]], label[local[d]]) for u, d in film_pairs.tolist() if u in local and d in local]
        if linked:
            a, b = np.array(linked).T
            merge = coo_matrix((np.ones(len(a)), (a, b)), shape=(n_regions, n_regions))
            n_regions, merged = connected_components(merge, directed=False)
            label = merged[label]
            film_regions = {int(label[local[u]]) for u, _ in film_pairs.tolist() if u in local}

    drop = np.zeros(len(f), dtype=bool)
    new_faces: list[np.ndarray] = []
    new_points: list[np.ndarray] = []
    next_vertex = len(v)
    count = 0
    for region in range(n_regions):
        members = flat[label == region]
        if len(members) < 2:
            continue
        in_region = np.zeros(len(f), dtype=bool)
        in_region[members] = True
        region_count = np.bincount(edge_id[in_region].ravel(), minlength=len(unique_edges))
        touched = region_count > 0
        if np.any(edge_total[touched] != 2):
            continue
        boundary_mask = region_count[edge_id[members]] == 1
        directed = f[members][:, [[0, 1], [1, 2], [2, 0]]][boundary_mask]
        if len(directed) == 0:
            continue
        net_boundary_area = 0.5 * float(
            np.sum(v[directed[:, 0], 0] * v[directed[:, 1], 1] - v[directed[:, 1], 0] * v[directed[:, 0], 1])
        )
        loops = _boundary_loops(directed, v[:, :2], region_on_left=net_boundary_area > 0)
        if not loops:
            continue

        xy_loops = [v[loop, :2] for loop in loops]
        areas = [abs(_signed_area_xy(xy)) for xy in xy_loops]
        depths, parents = [], []
        for i, loop in enumerate(loops):
            others = set().union(*(set(other) for j, other in enumerate(loops) if j != i))
            probe = next((vi for vi in loop if vi not in others), loop[0])
            containing = [j for j in range(len(loops)) if j != i and _point_in_polygon(v[probe, :2], xy_loops[j])]
            depths.append(len(containing))
            parents.append(min(containing, key=lambda j: areas[j]) if containing else None)
        polygon_area = sum(a if d % 2 == 0 else -a for a, d in zip(areas, depths))

        tri_xy = v[f[members]][:, :, :2]
        signed = 0.5 * (
            (tri_xy[:, 1, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 2, 1] - tri_xy[:, 0, 1])
            - (tri_xy[:, 2, 0] - tri_xy[:, 0, 0]) * (tri_xy[:, 1, 1] - tri_xy[:, 0, 1])
        )
        neg_area = float(np.abs(signed[(signed < 0) == (net_boundary_area > 0)]).sum())
        mismatch = abs(float(np.abs(signed).sum()) - polygon_area) > area_tol * polygon_area
        inconsistent = mismatch or neg_area > INVERTED_AREA_TOL * polygon_area
        if polygon_area <= 0 or not (inconsistent or region in film_regions):
            continue

        result = _triangulate_region(v, loops, depths, parents)
        if result is None:
            continue
        points, triangles = result
        points = np.asarray(points)
        all_xy = np.vstack([v[:, :2], points[:, :2]]) if len(points) else v[:, :2]
        a, b, c = all_xy[triangles[:, 0]], all_xy[triangles[:, 1]], all_xy[triangles[:, 2]]
        tri_signed = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
        # Outward direction comes from the boundary edges' own direction (set
        # by the intact side walls): a bottom cap faces -Z, so its triangles
        # wind clockwise seen from above.
        want_up = net_boundary_area > 0
        flip = (tri_signed < 0) == want_up
        triangles[flip] = triangles[flip][:, [0, 2, 1]]

        # New triangle indices >= len(v) refer to this region's new points.
        triangles = np.where(triangles >= len(v), triangles - len(v) + next_vertex, triangles)
        next_vertex += len(points)
        new_points.append(points)
        new_faces.append(triangles)
        drop[members] = True
        count += 1

    if not count:
        return v, f, 0

    v_out = np.vstack([v, *new_points]) if new_points else v
    f_out = np.vstack([f[~drop], *new_faces])
    used = np.unique(f_out)
    remap = np.full(len(v_out), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return v_out[used], remap[f_out], count


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
