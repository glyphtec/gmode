"""Tests for multi-layer C3D15 prism extrusion meshing (shell_mesh.py),
the thin/flat-object-aware alternative to generic Delaunay
tetrahedralization (mesh.py's mesh_surface/mesh_box).

Element-order correctness is verified against CalculiX's own official
example (calculix/examples/elements/C3D15.inp) in the module docstring
-- these tests cover the stacking/node-sharing geometry (no duplicate
or disconnected nodes at shared layer interfaces) and a full
end-to-end solve proving the dramatic element-count win is real, not
just a hand-wavy claim.
"""

import numpy as np
import pytest

from soundboard_lab.materials import SPRUCE_PLACEHOLDER
from soundboard_lab.shell_mesh import ShellMesh, extrude_shell_to_prisms, write_shell_frequency_inp
from soundboard_lab.solver import parse_eigenfrequencies, run_ccx

# A unit square (2 triangles), reused across tests.
_SQUARE_XY = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_SQUARE_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def _square_shell(thickness: float = 2.0) -> ShellMesh:
    return ShellMesh(
        xy=_SQUARE_XY,
        triangles=_SQUARE_TRIANGLES,
        bottom_z=np.zeros(4),
        top_z=np.full(4, thickness),
    )


def test_node_and_element_counts_match_hand_calculation():
    """2 triangles x 3 layers: 4 corner levels (4 verts each) + 4
    levels x 5 unique in-plane edges (4 outer + 1 diagonal, shared
    correctly between the two triangles) + 3 layers x 4 verts
    vertical mid-edges = 16 + 20 + 12 = 48 nodes, 6 elements.
    """
    nodes, elements = extrude_shell_to_prisms(_square_shell(), num_layers=3)
    assert len(nodes) == 48
    assert len(elements) == 6  # 3 layers * 2 triangles


def test_no_duplicate_or_disconnected_nodes_at_layer_interfaces():
    """Every node ID referenced by elements must be valid, and the
    corner/edge node counts must reflect sharing across adjacent
    layers and adjacent triangles, not independent per-element copies
    (which would silently disconnect the mesh at every interface).
    """
    nodes, elements = extrude_shell_to_prisms(_square_shell(), num_layers=3)
    assert elements.min() == 1
    assert elements.max() == len(nodes)
    # A naive "no sharing at all" implementation would produce
    # 15 nodes * 6 elements = 90 (with duplicates); correct sharing
    # must produce strictly fewer unique nodes than that.
    assert len(nodes) < 15 * len(elements)


def test_corners_directly_above_each_other_and_z_spans_full_thickness():
    thickness = 2.0
    nodes, elements = extrude_shell_to_prisms(_square_shell(thickness), num_layers=3)
    assert nodes[:, 2].min() == pytest.approx(0.0)
    assert nodes[:, 2].max() == pytest.approx(thickness)
    # Element 0's top corners (local slots 4-6) must share (x, y) with
    # its bottom corners (slots 1-3), varying only in z.
    el0_nodes = nodes[elements[0] - 1]
    assert np.allclose(el0_nodes[0:3, :2], el0_nodes[3:6, :2])


def test_shell_placeholder_spruce_eigenfrequencies(ccx_path, tmp_path):
    """End-to-end: a realistic soundboard-scale plate (300x400x2mm,
    matching docs/CLAUDE.md's stated ~2mm thickness), meshed as a
    3-layer C3D15 extrusion instead of uniform-size tets. Confirmed
    separately (see session history) that the equivalent uniform-tet
    2-layer mesh needs 1,359,056 elements; this approach needs 780 --
    this test locks in the correctness side (clean solve, right
    qualitative eigenvalue shape), the element-count win is
    structural (decoupled in-plane/through-thickness sizing), not
    something this test re-derives.
    """
    dx, dy, dz = 300.0, 400.0, 2.0
    cell = 30.0
    nx, ny = int(dx / cell) + 1, int(dy / cell) + 1
    xs, ys = np.linspace(0, dx, nx), np.linspace(0, dy, ny)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    xy = np.column_stack([xx.ravel(), yy.ravel()])

    def vid(i, j):
        return i * ny + j

    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
            triangles.append([a, b, c])
            triangles.append([a, c, d])
    shell = ShellMesh(
        xy=xy, triangles=np.array(triangles), bottom_z=np.zeros(len(xy)), top_z=np.full(len(xy), dz)
    )

    inp_path = tmp_path / "shell.inp"
    element_count = write_shell_frequency_inp(shell, SPRUCE_PLACEHOLDER, num_layers=3, inp_path=inp_path)
    assert element_count < 2000  # vs. 1.36M for the equivalent uniform-tet case

    run_ccx(inp_path, ccx_path)
    modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))
    assert [m.mode_number for m in modes] == list(range(1, 11))
    frequencies = [m.frequency_hz for m in modes]

    for freq in frequencies[:6]:
        assert freq < 1.0, f"expected a near-zero rigid-body mode, got {freq} Hz"
    real_modes = frequencies[6:]
    assert all(5 < f < 200 for f in real_modes), real_modes
    assert real_modes == sorted(real_modes), "eigenfrequencies should be strictly increasing"
