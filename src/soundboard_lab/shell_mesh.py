"""Multi-layer prism (C3D15) extrusion meshing, for objects we know in
advance are thin/flat (role=Soundboard) -- stacks N layers between a
bottom and top reference surface sharing the same in-plane footprint,
instead of asking generic 3D Delaunay tetrahedralization to resolve a
thin gap (see docs/CLAUDE.md "known traps": that approach forces a
uniform element size small enough for the thin dimension everywhere,
blowing up element count across the whole in-plane area -- measured
empirically at 1.36M elements for a simple 300x400x2mm plate at the
minimum viable 2-layer resolution).

Pure Python, no `bpy` imports (see docs/CLAUDE.md, "Engine/UI boundary").
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soundboard_lab.materials import OrthotropicWoodMaterial, to_calculix_engineering_constants, validate_positive_definite
from soundboard_lab.mesh import _frequency_step_cards


@dataclass(frozen=True)
class ShellMesh:
    """A 2D in-plane triangulated footprint, shared by a bottom and
    top reference surface (as per-vertex Z offsets, allowing a
    non-uniform/CNC-contoured thickness later -- see docs/CLAUDE.md
    Scope, "curved orthotropic material orientation" sibling idea).
    """

    xy: np.ndarray  # (V, 2) in-plane vertex positions
    triangles: np.ndarray  # (T, 3) vertex index triples
    bottom_z: np.ndarray  # (V,) per-vertex bottom-surface Z
    top_z: np.ndarray  # (V,) per-vertex top-surface Z


def _reject_wrong_level_faces(
    mesh, top_faces: np.ndarray, bottom_faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Drop faces that passed the normal-direction test but sit at the
    WRONG surface's Z level -- a real, confirmed defect: a botched
    boolean soundhole cut can leave a reversed-winding duplicate of a
    top-surface sliver triangle (same (x, y, z) positions, opposite
    face winding), which normal-direction classification alone can't
    tell apart from a genuine bottom face, since its normal correctly
    points "down" even though it's physically sitting at the top
    surface's Z level. Left unfiltered, this contaminates the
    bottom/top vertex clouds used for bottom_z/top_z lookup, producing
    a handful of zero-thickness vertices.

    Assumes each surface sits at a roughly CONSTANT Z (true for a flat
    plate, both of this pipeline's real test files) -- a face is kept
    only if its own vertices' Z values are closer to its own
    candidate group's median Z than to the other group's median Z.
    This is a global, not local, consistency check, so it won't
    generalize to a genuinely curved/CNC-contoured thickness (a
    documented limitation also noted on `shell_from_closed_mesh`'s
    bottom_z/top_z lookup).
    """
    top_z_est = float(np.median(mesh.vertices[np.unique(mesh.faces[top_faces]), 2]))
    bottom_z_est = float(np.median(mesh.vertices[np.unique(mesh.faces[bottom_faces]), 2]))

    def keep(face_indices: np.ndarray, own_z: float, other_z: float) -> np.ndarray:
        face_z = mesh.vertices[mesh.faces[face_indices], 2]  # (n, 3)
        closer_to_own = np.abs(face_z - own_z) < np.abs(face_z - other_z)
        return face_indices[np.all(closer_to_own, axis=1)]

    kept_top = keep(top_faces, top_z_est, bottom_z_est)
    kept_bottom = keep(bottom_faces, bottom_z_est, top_z_est)
    n_rejected = (len(top_faces) - len(kept_top)) + (len(bottom_faces) - len(kept_bottom))
    if n_rejected:
        print(
            f"Soundboard Lab: rejected {n_rejected} face(s) whose normal direction matched "
            "top/bottom but whose Z position didn't -- likely a reversed-winding duplicate "
            "defect (see docs/CLAUDE.md soundhole/fan-triangulation trap)"
        )
    return kept_top, kept_bottom


def shell_from_closed_mesh(mesh_path: Path, element_size: float, normal_threshold: float = 0.9) -> ShellMesh:
    """Convert a closed/watertight mesh -- as exported from the typical
    Blender "draw the outline, extrude up by thickness" soundboard
    workflow -- into a `ShellMesh`, the input `extrude_shell_to_prisms`/
    `write_shell_frequency_inp` expect.

    Assumes the object's thin/thickness axis is world Z. This matches
    the Z=tangential material-axis convention already adopted
    elsewhere (docs/CLAUDE.md "Meshing and numerics"): tangential
    growth-ring direction is typically through-thickness for a
    quartersawn soundboard, so "thin axis is Z" isn't a new
    assumption, it's the same one made explicit here.

    Faces are classified by their normal's Z component: "top"
    (normal_z > normal_threshold) and "bottom" (normal_z <
    -normal_threshold) are the near-horizontal reference surfaces;
    everything else is a side wall.

    The outer-edge and soundhole curves are recovered from the SIDE
    WALL structure (the edges between its "upper" and "lower" vertices
    at each quad/strip), NOT by looking for boundary edges in the top
    surface's own interior triangulation. This matters a lot in
    practice: verified empirically (see docs/CLAUDE.md) that a
    realistic Blender workflow -- draw the outline, boolean-cut a
    soundhole, let Blender's n-gon fill triangulate the resulting cap
    faces -- can produce a degenerate single-vertex "fan" triangulation
    for the top/bottom caps, where one fan ray happens to touch the
    true outer boundary and creates a false 4-way junction there. That
    corrupts any boundary-edge-counting approach applied to the cap
    faces directly (confirmed: no amount of clever loop-walking could
    reliably recover the two true loops from that corrupted topology --
    the junction genuinely conflates them). The side walls don't share
    this problem: they trace the original extrusion curve directly
    (confirmed empirically: side-wall rim vertices all have exactly 2
    boundary neighbors, a clean closed set of simple loops, even on the
    same defective file).

    The recovered curves are then meshed FRESH with gmsh at
    `element_size` -- replacing the cap's own interior triangulation
    entirely, so its quality (good or bad) never matters. Each new
    vertex's top_z/bottom_z is looked up by nearest (x, y) neighbor
    against the original top/bottom surfaces: exact for a flat plate
    or straight vertical extrusion (both of this pipeline's real test
    files), an approximation for a curved/CNC-contoured thickness.
    """
    import gmsh
    import trimesh
    from scipy.spatial import cKDTree

    mesh = trimesh.load(str(mesh_path), process=True)
    if not mesh.is_watertight:
        raise ValueError(f"{mesh_path}: mesh is not watertight -- repair it first (see repair.py)")

    normal_z = mesh.face_normals[:, 2]
    top_faces = np.nonzero(normal_z > normal_threshold)[0]
    bottom_faces = np.nonzero(normal_z < -normal_threshold)[0]
    side_faces = np.nonzero(np.abs(normal_z) <= normal_threshold)[0]
    if len(top_faces) == 0 or len(bottom_faces) == 0 or len(side_faces) == 0:
        raise ValueError(
            f"{mesh_path}: couldn't find clear top/bottom/side faces at "
            f"normal_threshold={normal_threshold} -- is the object's thin axis actually world Z?"
        )
    top_faces, bottom_faces = _reject_wrong_level_faces(mesh, top_faces, bottom_faces)

    top = mesh.submesh([top_faces], append=True)
    bottom = mesh.submesh([bottom_faces], append=True)
    side = mesh.submesh([side_faces], append=True)

    mid_z = (side.vertices[:, 2].max() + side.vertices[:, 2].min()) / 2
    top_rim_edges: list[tuple[int, int]] = []
    for face in side.faces:
        upper = [v for v in face if side.vertices[v, 2] > mid_z]
        if len(upper) == 2:
            a, b = upper
            top_rim_edges.append((a, b) if a < b else (b, a))
    if not top_rim_edges:
        raise ValueError(f"{mesh_path}: side walls produced no top-rim edges -- unexpected mesh structure")

    rim_xy = side.vertices[:, :2]
    outer_rim, hole_rims = _classify_loops(rim_xy, _loops_from_edges(top_rim_edges))
    # Resample the rim down toward element_size: the ORIGINAL curve's
    # point density (inherited from however finely Blender sampled the
    # bezier curve) has nothing to do with the requested mesh
    # resolution -- confirmed empirically: without this, gmsh just
    # reuses every original point as a mesh node directly (a straight
    # Line between 2 points already closer together than element_size
    # is never coarsened, only ever subdivided further), producing
    # ~5x more triangles than element_size alone would suggest.
    min_spacing = 0.5 * element_size
    outer_rim = _simplify_rim(rim_xy, outer_rim, min_spacing)
    hole_rims = [_simplify_rim(rim_xy, hole, min_spacing) for hole in hole_rims]

    gmsh.initialize()
    try:
        gmsh.model.add("soundboard_lab_shell_footprint")

        def add_curve_loop(rim: list[int]) -> int:
            point_tags = [gmsh.model.geo.addPoint(rim_xy[v, 0], rim_xy[v, 1], 0.0, element_size) for v in rim]
            n = len(point_tags)
            line_tags = [gmsh.model.geo.addLine(point_tags[i], point_tags[(i + 1) % n]) for i in range(n)]
            return gmsh.model.geo.addCurveLoop(line_tags)

        outer_loop_tag = add_curve_loop(outer_rim)
        hole_loop_tags = [add_curve_loop(hole) for hole in hole_rims]
        gmsh.model.geo.addPlaneSurface([outer_loop_tag, *hole_loop_tags])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(node_coords, dtype=np.float64).reshape(-1, 3)
        tag_to_index = {tag: i for i, tag in enumerate(node_tags)}

        elem_types, _elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
        triangles = None
        for elem_type, node_tags_for_type in zip(elem_types, elem_node_tags):
            if elem_type == 2:  # gmsh element type 2 = 3-node triangle
                tri_tags = np.array(node_tags_for_type, dtype=np.int64).reshape(-1, 3)
                triangles = np.vectorize(tag_to_index.get)(tri_tags)
        if triangles is None:
            raise RuntimeError(f"{mesh_path}: gmsh produced no 2D triangles for the recovered footprint")
    finally:
        gmsh.finalize()

    xy_new = coords[:, :2]

    # extrude_shell_to_prisms needs consistent CCW winding (viewed from
    # +Z) to produce positive-Jacobian C3D15 elements -- confirmed
    # empirically: this gmsh curve-loop construction consistently
    # produces CW triangles instead (all 6101 of them, on the real
    # test geometry, not a mix -- ccx failed on effectively every
    # extruded element with "nonpositive jacobian", not just one).
    signed_area_x2 = (
        (xy_new[triangles[:, 1], 0] - xy_new[triangles[:, 0], 0])
        * (xy_new[triangles[:, 2], 1] - xy_new[triangles[:, 0], 1])
        - (xy_new[triangles[:, 2], 0] - xy_new[triangles[:, 0], 0])
        * (xy_new[triangles[:, 1], 1] - xy_new[triangles[:, 0], 1])
    )
    clockwise = signed_area_x2 < 0
    triangles[clockwise] = triangles[clockwise][:, [0, 2, 1]]  # swap 2 verts to flip winding

    top_tree = cKDTree(top.vertices[:, :2])
    _, nearest_top_idx = top_tree.query(xy_new)
    bottom_tree = cKDTree(bottom.vertices[:, :2])
    _, nearest_bottom_idx = bottom_tree.query(xy_new)

    return ShellMesh(
        xy=xy_new,
        triangles=triangles,
        bottom_z=bottom.vertices[nearest_bottom_idx, 2].copy(),
        top_z=top.vertices[nearest_top_idx, 2].copy(),
    )



def _loops_from_edges(edges: list[tuple[int, int]]) -> list[list[int]]:
    """Walk an undirected edge list into simple closed loops.

    Assumes a clean 2-manifold-with-boundary graph (every vertex has
    exactly 2 incident edges) -- true for a triangulation's own
    boundary edges (edges used by exactly one triangle) in a
    well-formed mesh, and true by construction for the side-wall "rim"
    edges `shell_from_closed_mesh` builds (see its docstring for why
    that's the more robust source for real, possibly messy, meshes).
    Prints a non-fatal warning (not a fix -- see docs/CLAUDE.md
    soundhole/fan-triangulation trap for why a junction can't be
    reliably untangled by clever walking alone) if a vertex has more
    or fewer than 2 incident edges, since the resulting loops may then
    be wrong rather than just absent.
    """
    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

    bad_degree = {v: nbrs for v, nbrs in adjacency.items() if len(nbrs) != 2}
    if bad_degree:
        print(
            f"Soundboard Lab: {len(bad_degree)} vertex/vertices have != 2 boundary-edge "
            "neighbors (not a clean set of simple loops) -- results below may be unreliable; "
            "see docs/CLAUDE.md soundhole/fan-triangulation trap"
        )

    loops: list[list[int]] = []
    visited: set[int] = set()
    for start in adjacency:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        prev, current = None, start
        while True:
            neighbors = [n for n in adjacency[current] if n != prev]
            next_vertex = next((n for n in neighbors if n not in visited), None)
            if next_vertex is None:
                break  # closed the loop back to `start`
            loop.append(next_vertex)
            visited.add(next_vertex)
            prev, current = current, next_vertex
        loops.append(loop)
    return loops


def find_boundary_loops(triangles: np.ndarray) -> list[list[int]]:
    """Find every boundary loop in a 2D triangulation -- the outer
    perimeter AND any interior holes (e.g. a soundhole) are both
    topologically "boundary loops" by this test; distinguishing which
    is which is `classify_boundary_loops`'s job, not this function's.

    A boundary edge is one used by exactly one triangle (an interior
    edge is always shared by exactly two) -- deterministic, no
    heuristics, for a clean triangulation. For a messy/defective
    real-world triangulation, prefer sourcing loops from
    `shell_from_closed_mesh`'s side-wall rim extraction instead (see
    its docstring) -- this function trusts the triangulation's own
    (possibly wrong) topology as-is.
    """
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            edge_count[key] += 1
    boundary_edges = [e for e, count in edge_count.items() if count == 1]
    return _loops_from_edges(boundary_edges)


def _polygon_area(xy: np.ndarray, loop: list[int]) -> float:
    """Shoelace formula; works for any simple polygon, convex or not."""
    pts = xy[loop]
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0)


def _classify_loops(xy: np.ndarray, loops: list[list[int]]) -> tuple[list[int], list[list[int]]]:
    """Split already-found loops into (outer perimeter, [hole loops]).

    The outer loop is the one enclosing the largest area -- a sound
    (not heuristic) rule: any interior hole's loop necessarily
    encloses less area than the boundary that contains it, since the
    outer loop's enclosed region is a strict superset (the hole's
    footprint plus the material around it).
    """
    if not loops:
        raise ValueError("no boundary loops found -- is this triangulation closed with no perimeter at all?")
    areas = [_polygon_area(xy, loop) for loop in loops]
    outer_index = int(np.argmax(areas))
    outer = loops[outer_index]
    holes = [loop for i, loop in enumerate(loops) if i != outer_index]
    return outer, holes


def classify_boundary_loops(xy: np.ndarray, triangles: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """`find_boundary_loops` + `_classify_loops` for a given triangulation."""
    return _classify_loops(xy, find_boundary_loops(triangles))


def _simplify_rim(xy: np.ndarray, rim: list[int], min_spacing: float) -> list[int]:
    """Drop points that are closer to their predecessor than
    `min_spacing`. Two jobs, one mechanism:

    1. Near-duplicate removal: a real curve can be locally
       over-sampled right where it's sharpest (confirmed: a real
       soundboard outline had 2 consecutive tail-tip points 0.23mm
       apart against a ~4mm typical spacing), and feeding that
       straight into gmsh as 2 separate points/lines can produce a
       razor-thin, technically self-intersecting kink that gmsh's
       curve loop meshing then rejects outright.
    2. Resampling toward the target element size: the original rim's
       point density is inherited from however finely the source curve
       happened to be sampled (e.g. in Blender), which has nothing to
       do with the requested FEA mesh resolution -- confirmed
       empirically: a Line between 2 points already closer together
       than `element_size` is never coarsened by gmsh, only ever
       subdivided further, so without this step the boundary alone can
       force ~5x more triangles than `element_size` would suggest.

    Called with `min_spacing = 0.5 * element_size` for (2); doesn't
    lose real shape detail at any reasonable target element size,
    since gmsh will subdivide the remaining longer segments down to
    `element_size` anyway where finer resolution is actually needed.
    """
    simplified = [rim[0]]
    for v in rim[1:]:
        if np.linalg.norm(xy[v] - xy[simplified[-1]]) >= min_spacing:
            simplified.append(v)
    if np.linalg.norm(xy[simplified[-1]] - xy[simplified[0]]) < min_spacing and len(simplified) > 3:
        simplified.pop()  # closing point also collapses into the start
    return simplified



@dataclass(frozen=True)
class ExtrudedPrismMesh:
    """Result of `extrude_shell_to_prisms`."""

    nodes: np.ndarray  # (n, 3) 3D coordinates
    elements: np.ndarray  # (num_layers * len(triangles), 15) 1-indexed node IDs
    corner_node_ids: np.ndarray  # (num_layers + 1, V) node ID at each (level, in-plane vertex)


def extrude_shell_to_prisms(shell: ShellMesh, num_layers: int) -> ExtrudedPrismMesh:
    """Stack `num_layers` C3D15 (15-node quadratic wedge) elements
    between `shell`'s bottom and top surfaces.

    `elements` is a (num_layers * len(triangles), 15) array of
    1-indexed node IDs in CalculiX's own C3D15 order (verified against
    calculix/examples/elements/C3D15.inp, not assumed): 1-3 bottom
    corners, 4-6 top corners (directly above 1-3), 7-9 bottom
    mid-edges (1-2, 2-3, 3-1), 10-12 top mid-edges (4-5, 5-6, 6-4),
    13-15 vertical mid-edges (1-4, 2-5, 3-6).

    `corner_node_ids` lets callers (e.g. boundary-condition writers)
    map an in-plane vertex index at a given extrusion level back to
    its global node ID -- needed to fix an outer-edge node across
    every layer, not just one face.
    """
    V = len(shell.xy)
    N = num_layers
    node_list: list[tuple[float, float, float]] = []

    def add_node(x: float, y: float, z: float) -> int:
        node_list.append((x, y, z))
        return len(node_list)  # 1-indexed CalculiX node ID

    # Corner nodes at each of the N+1 levels, shared across adjacent
    # layers (level k's "top" is level k+1's "bottom" -- same nodes).
    corner_id = np.zeros((N + 1, V), dtype=np.int64)
    for level in range(N + 1):
        t = level / N
        for v in range(V):
            x, y = shell.xy[v]
            z = shell.bottom_z[v] + t * (shell.top_z[v] - shell.bottom_z[v])
            corner_id[level, v] = add_node(x, y, z)

    # In-plane mid-edge nodes at each level, deyed by the *global* edge
    # (sorted vertex pair) so triangles sharing an edge share the node
    # -- not per-triangle-local, which would duplicate and disconnect
    # the mesh at shared edges.
    inplane_mid: dict[tuple[int, tuple[int, int]], int] = {}

    def inplane_mid_node(level: int, va: int, vb: int) -> int:
        key = (level, (va, vb) if va < vb else (vb, va))
        if key not in inplane_mid:
            a_id, b_id = corner_id[level, va], corner_id[level, vb]
            ax, ay, az = node_list[a_id - 1]
            bx, by, bz = node_list[b_id - 1]
            inplane_mid[key] = add_node((ax + bx) / 2, (ay + by) / 2, (az + bz) / 2)
        return inplane_mid[key]

    # Vertical mid-edge nodes: unique per (layer, in-plane vertex) --
    # shared across triangles that meet at that vertex within the same
    # layer, but not shared between different layers.
    vertical_mid: dict[tuple[int, int], int] = {}

    def vertical_mid_node(layer: int, v: int) -> int:
        key = (layer, v)
        if key not in vertical_mid:
            bottom_id, top_id = corner_id[layer, v], corner_id[layer + 1, v]
            bx, by, bz = node_list[bottom_id - 1]
            tx, ty, tz = node_list[top_id - 1]
            vertical_mid[key] = add_node((bx + tx) / 2, (by + ty) / 2, (bz + tz) / 2)
        return vertical_mid[key]

    elements = []
    for layer in range(N):
        for tri in shell.triangles:
            v1, v2, v3 = tri
            elements.append(
                [
                    corner_id[layer, v1], corner_id[layer, v2], corner_id[layer, v3],
                    corner_id[layer + 1, v1], corner_id[layer + 1, v2], corner_id[layer + 1, v3],
                    inplane_mid_node(layer, v1, v2),
                    inplane_mid_node(layer, v2, v3),
                    inplane_mid_node(layer, v3, v1),
                    inplane_mid_node(layer + 1, v1, v2),
                    inplane_mid_node(layer + 1, v2, v3),
                    inplane_mid_node(layer + 1, v3, v1),
                    vertical_mid_node(layer, v1),
                    vertical_mid_node(layer, v2),
                    vertical_mid_node(layer, v3),
                ]
            )

    return ExtrudedPrismMesh(
        nodes=np.array(node_list), elements=np.array(elements, dtype=np.int64), corner_node_ids=corner_id
    )


def write_shell_frequency_inp(
    shell: ShellMesh,
    material: OrthotropicWoodMaterial,
    num_layers: int,
    inp_path: Path,
    n_frequencies: int = 10,
    fixed_edge: bool = False,
) -> int:
    """Write a `*FREQUENCY` CalculiX `.inp` from a multi-layer C3D15
    extrusion of `shell`. Returns the element count.

    `fixed_edge=False` (default) is the free-free case: no supports at
    all, used for the mesh/material validation ladder so far.

    `fixed_edge=True` clamps every node (all translational DOFs 1,2,3
    -- solid elements have no rotational DOFs, so there's nothing else
    to fix) on the shell's OUTER perimeter, across every extrusion
    layer, not just one face. The outer perimeter is found via
    `classify_boundary_loops`, which distinguishes it from any
    interior hole (e.g. a soundhole) by enclosed area -- the hole's
    loop is always smaller since the outer loop's region is a strict
    superset. Interior-hole nodes are deliberately left unconstrained.

    Orientation is currently identity (same known gap as
    write_frequency_inp_orthotropic in mesh.py -- grain rotation isn't
    wired through yet, tracked separately as a fast-follow).
    """
    validate_positive_definite(material)
    prisms = extrude_shell_to_prisms(shell, num_layers)
    nodes, elements, corner_node_ids = prisms.nodes, prisms.elements, prisms.corner_node_ids
    e1, e2, e3, nu12, nu13, nu23, g12, g13, g23 = to_calculix_engineering_constants(material)

    lines = [
        "** Soundboard Lab -- multi-layer C3D15 prism extrusion of a",
        "** thin/flat shell (bottom + top reference surfaces).",
        "*NODE, NSET=Nall",
    ]
    for i, (x, y, z) in enumerate(nodes, start=1):
        lines.append(f"{i}, {x:.10g}, {y:.10g}, {z:.10g}")

    lines.append("*ELEMENT, TYPE=C3D15, ELSET=Evolumes")
    for i, conn in enumerate(elements, start=1):
        node_ids = ", ".join(str(n) for n in conn)
        lines.append(f"{i}, {node_ids}")

    lines += [
        "*ORIENTATION, NAME=GrainAxes",
        "1., 0., 0., 0., 1., 0.",
        f"*MATERIAL, NAME={material.name}",
        "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
        f"{e1:.10g}, {e2:.10g}, {e3:.10g}, {nu12:.10g}, {nu13:.10g}, {nu23:.10g}, {g12:.10g}, {g13:.10g},",
        f"{g23:.10g}",
        "*DENSITY",
        f"{material.density:.10g}",
        f"*SOLID SECTION, ELSET=Evolumes, MATERIAL={material.name}, ORIENTATION=GrainAxes",
    ]

    if fixed_edge:
        outer_loop, _hole_loops = classify_boundary_loops(shell.xy, shell.triangles)
        fixed_node_ids = sorted({int(corner_node_ids[level, v]) for level in range(num_layers + 1) for v in outer_loop})
        lines.append("*NSET, NSET=Nfixed")
        for i in range(0, len(fixed_node_ids), 10):
            lines.append(", ".join(str(n) for n in fixed_node_ids[i : i + 10]))
        lines.append("*BOUNDARY")
        lines.append("Nfixed, 1, 3")

    lines += _frequency_step_cards(n_frequencies)
    Path(inp_path).write_text("\n".join(lines) + "\n")
    return len(elements)

