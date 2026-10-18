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


def find_boundary_loops(triangles: np.ndarray) -> list[list[int]]:
    """Find every boundary loop in a 2D triangulation -- the outer
    perimeter AND any interior holes (e.g. a soundhole) are both
    topologically "boundary loops" by this test; distinguishing which
    is which is `classify_boundary_loops`'s job, not this function's.

    A boundary edge is one used by exactly one triangle (an interior
    edge is always shared by exactly two) -- deterministic, no
    heuristics. Loops are found by walking the boundary-edge
    adjacency graph; assumes a clean 2-manifold-with-boundary
    triangulation (each boundary vertex touches exactly 2 boundary
    edges), true for any triangulation gmsh or a simple grid produces.
    """
    edge_count: dict[tuple[int, int], int] = defaultdict(int)
    for tri in triangles:
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            key = (a, b) if a < b else (b, a)
            edge_count[key] += 1
    boundary_edges = [e for e, count in edge_count.items() if count == 1]

    adjacency: dict[int, list[int]] = defaultdict(list)
    for a, b in boundary_edges:
        adjacency[a].append(b)
        adjacency[b].append(a)

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


def _polygon_area(xy: np.ndarray, loop: list[int]) -> float:
    """Shoelace formula; works for any simple polygon, convex or not."""
    pts = xy[loop]
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) / 2.0)


def classify_boundary_loops(xy: np.ndarray, triangles: np.ndarray) -> tuple[list[int], list[list[int]]]:
    """Split boundary loops into (outer perimeter, [hole loops]).

    The outer loop is the one enclosing the largest area -- a sound
    (not heuristic) rule: any interior hole's loop necessarily
    encloses less area than the boundary that contains it, since the
    outer loop's enclosed region is a strict superset (the hole's
    footprint plus the material around it).
    """
    loops = find_boundary_loops(triangles)
    if not loops:
        raise ValueError("no boundary loops found -- is this triangulation closed with no perimeter at all?")
    areas = [_polygon_area(xy, loop) for loop in loops]
    outer_index = int(np.argmax(areas))
    outer = loops[outer_index]
    holes = [loop for i, loop in enumerate(loops) if i != outer_index]
    return outer, holes


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

