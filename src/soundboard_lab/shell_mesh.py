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


def extrude_shell_to_prisms(shell: ShellMesh, num_layers: int) -> tuple[np.ndarray, np.ndarray]:
    """Stack `num_layers` C3D15 (15-node quadratic wedge) elements
    between `shell`'s bottom and top surfaces.

    Returns (nodes, elements): `nodes` is an (n, 3) array of 3D
    coordinates; `elements` is an (num_layers * len(triangles), 15)
    array of 1-indexed node IDs in CalculiX's own C3D15 order
    (verified against calculix/examples/elements/C3D15.inp, not
    assumed): 1-3 bottom corners, 4-6 top corners (directly above
    1-3), 7-9 bottom mid-edges (1-2, 2-3, 3-1), 10-12 top mid-edges
    (4-5, 5-6, 6-4), 13-15 vertical mid-edges (1-4, 2-5, 3-6).
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

    return np.array(node_list), np.array(elements, dtype=np.int64)


def write_shell_frequency_inp(
    shell: ShellMesh,
    material: OrthotropicWoodMaterial,
    num_layers: int,
    inp_path: Path,
    n_frequencies: int = 10,
) -> int:
    """Write a free-free `*FREQUENCY` CalculiX `.inp` from a
    multi-layer C3D15 extrusion of `shell`. Returns the element count.

    Orientation is currently identity (same known gap as
    write_frequency_inp_orthotropic in mesh.py -- grain rotation isn't
    wired through yet, tracked separately as a fast-follow).
    """
    validate_positive_definite(material)
    nodes, elements = extrude_shell_to_prisms(shell, num_layers)
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
        *_frequency_step_cards(n_frequencies),
    ]
    Path(inp_path).write_text("\n".join(lines) + "\n")
    return len(elements)

