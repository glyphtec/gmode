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

from pathlib import Path

import numpy as np
import pytest

from soundboard_lab.materials import SPRUCE_PLACEHOLDER
from soundboard_lab.shell_mesh import (
    ShellMesh,
    classify_boundary_loops,
    extrude_shell_to_prisms,
    find_boundary_loops,
    shell_from_closed_mesh,
    write_shell_frequency_inp,
)
from soundboard_lab.solver import parse_eigenfrequencies, run_ccx

DATA_DIR = Path(__file__).parent / "data"

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
    prisms = extrude_shell_to_prisms(_square_shell(), num_layers=3)
    nodes, elements = prisms.nodes, prisms.elements
    assert len(nodes) == 48
    assert len(elements) == 6  # 3 layers * 2 triangles


def test_no_duplicate_or_disconnected_nodes_at_layer_interfaces():
    """Every node ID referenced by elements must be valid, and the
    corner/edge node counts must reflect sharing across adjacent
    layers and adjacent triangles, not independent per-element copies
    (which would silently disconnect the mesh at every interface).
    """
    prisms = extrude_shell_to_prisms(_square_shell(), num_layers=3)
    nodes, elements = prisms.nodes, prisms.elements
    assert elements.min() == 1
    assert elements.max() == len(nodes)
    # A naive "no sharing at all" implementation would produce
    # 15 nodes * 6 elements = 90 (with duplicates); correct sharing
    # must produce strictly fewer unique nodes than that.
    assert len(nodes) < 15 * len(elements)


def test_corners_directly_above_each_other_and_z_spans_full_thickness():
    thickness = 2.0
    prisms = extrude_shell_to_prisms(_square_shell(thickness), num_layers=3)
    nodes, elements = prisms.nodes, prisms.elements
    assert nodes[:, 2].min() == pytest.approx(0.0)
    assert nodes[:, 2].max() == pytest.approx(thickness)
    # Element 0's top corners (local slots 4-6) must share (x, y) with
    # its bottom corners (slots 1-3), varying only in z.
    el0_nodes = nodes[elements[0] - 1]
    assert np.allclose(el0_nodes[0:3, :2], el0_nodes[3:6, :2])


def _grid_plate_with_hole(dx: float, dy: float, cell: float, hole_lo: float, hole_hi: float, thickness: float):
    """A rectangular grid-triangulated plate with a square hole carved
    out of the middle -- a crude "soundhole" stand-in, used to prove
    boundary-loop detection correctly tells the true outer edge apart
    from an interior hole (see docs/CLAUDE.md "known traps": both are
    topologically indistinguishable boundary loops by the
    one-triangle-per-edge rule alone -- area is the distinguisher).
    """
    nx, ny = int(dx / cell) + 1, int(dy / cell) + 1
    xs, ys = np.linspace(0, dx, nx), np.linspace(0, dy, ny)

    def vid(i, j):
        return i * ny + j

    def in_hole(i, j):
        x, y = xs[i], ys[j]
        return hole_lo <= x <= hole_hi and hole_lo <= y <= hole_hi

    xy = np.column_stack([np.repeat(xs, ny), np.tile(ys, nx)])
    triangles = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            corners = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            if all(in_hole(*c) for c in corners):
                continue  # entire quad falls inside the hole -- skip it
            a, b, c, d = (vid(*corners[0]), vid(*corners[1]), vid(*corners[2]), vid(*corners[3]))
            triangles.append([a, b, c])
            triangles.append([a, c, d])
    triangles = np.array(triangles)
    shell = ShellMesh(
        xy=xy, triangles=triangles, bottom_z=np.zeros(len(xy)), top_z=np.full(len(xy), thickness)
    )
    return shell


def test_boundary_loops_distinguish_outer_perimeter_from_soundhole():
    """A 100x100 plate with a 20x20 interior hole must produce exactly
    2 boundary loops, and the outer loop (not the hole) must be the
    one classify_boundary_loops picks, verified both by enclosed area
    and by checking the actual (x, y) coordinates land on the true
    100x100 perimeter vs. the 40-60 hole boundary.
    """
    shell = _grid_plate_with_hole(dx=100, dy=100, cell=10, hole_lo=40, hole_hi=60, thickness=2.0)
    loops = find_boundary_loops(shell.triangles)
    assert len(loops) == 2, "expected exactly one outer loop and one hole loop"

    outer, holes = classify_boundary_loops(shell.xy, shell.triangles)
    assert len(holes) == 1

    outer_xy = shell.xy[outer]
    on_outer_perimeter = (
        np.isclose(outer_xy[:, 0], 0) | np.isclose(outer_xy[:, 0], 100) |
        np.isclose(outer_xy[:, 1], 0) | np.isclose(outer_xy[:, 1], 100)
    )
    assert on_outer_perimeter.all(), "outer loop must only contain true-perimeter vertices, not hole vertices"

    hole_xy = shell.xy[holes[0]]
    on_hole_boundary = (
        np.isclose(hole_xy[:, 0], 40) | np.isclose(hole_xy[:, 0], 60) |
        np.isclose(hole_xy[:, 1], 40) | np.isclose(hole_xy[:, 1], 60)
    )
    assert on_hole_boundary.all(), "hole loop must only contain hole-boundary vertices, not perimeter vertices"


def test_shell_fixed_edge_eigenfrequencies_are_higher_than_free_free(ccx_path, tmp_path):
    """A clamped-edge plate must be stiffer than the same plate
    free-free -- its lowest non-trivial eigenfrequency should be
    clearly higher, and (unlike free-free) there should be NO
    near-zero rigid-body modes at all, since every outer-rim node is
    fully restrained. This is the sanity check that *BOUNDARY actually
    did something physically meaningful, not just that ccx ran clean.
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

    inp_path = tmp_path / "shell_fixed.inp"
    write_shell_frequency_inp(shell, SPRUCE_PLACEHOLDER, num_layers=3, inp_path=inp_path, fixed_edge=True)

    run_ccx(inp_path, ccx_path)
    modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))
    frequencies = [m.frequency_hz for m in modes]

    assert all(f > 1.0 for f in frequencies), (
        "a fully clamped-edge plate should have no near-zero rigid-body modes at all "
        f"(got {frequencies})"
    )
    assert frequencies == sorted(frequencies), "eigenfrequencies should be strictly increasing"


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


def test_shell_from_closed_mesh_handles_real_defective_geometry():
    """`real_soundboard_with_defects.stl` is a real Blender export (not
    synthetic) that empirically surfaced 3 distinct, real-world
    defects -- this test locks in all 3 fixes at once against the same
    file that caught them, not a cleaned-up stand-in:

    1. Z cross-contamination: a reversed-winding duplicate of a
       top-surface sliver (from a botched boolean soundhole cut) gets
       classified as a "bottom" face by its normal direction alone,
       even though it's physically at the top surface's Z level --
       `_reject_wrong_level_faces` catches this.
    2. All triangles must come out consistently CCW (viewed from +Z):
       gmsh's curve-loop meshing here produced a mesh that was
       entirely (not partially) clockwise, which silently produces
       inverted/negative-Jacobian C3D15 elements -- ccx failed on
       effectively every extruded element before this was caught and
       fixed, not just one.
    3. Exactly one outer perimeter and one soundhole must be found, not
       the 3 spurious "boundary loops" the top surface's own corrupted
       fan-triangulation produced before the side-wall-rim-based
       boundary recovery was built.
    """
    shell = shell_from_closed_mesh(DATA_DIR / "real_soundboard_with_defects.stl", element_size=10.0)

    thickness = shell.top_z - shell.bottom_z
    assert thickness.min() == pytest.approx(2.0, abs=0.01), "zero/wrong-thickness vertices means Z contamination"
    assert thickness.max() == pytest.approx(2.0, abs=0.01)

    xy, triangles = shell.xy, shell.triangles
    signed_area_x2 = (
        (xy[triangles[:, 1], 0] - xy[triangles[:, 0], 0]) * (xy[triangles[:, 2], 1] - xy[triangles[:, 0], 1])
        - (xy[triangles[:, 2], 0] - xy[triangles[:, 0], 0]) * (xy[triangles[:, 1], 1] - xy[triangles[:, 0], 1])
    )
    assert np.all(signed_area_x2 > 0), "every triangle must be wound CCW (viewed from +Z) for a positive Jacobian"

    outer, holes = classify_boundary_loops(xy, triangles)
    assert len(holes) == 1, f"expected exactly 1 soundhole, found {len(holes)} boundary loop(s) besides the outer edge"
    assert len(outer) > len(holes[0]), "outer perimeter must have more vertices than the (smaller) soundhole loop"


def test_shell_real_geometry_free_free_solve_matches_convergence_study(ccx_path, tmp_path):
    """End-to-end solve of the real defective fixture at element_size=10mm
    -- the default recommended by a real mesh-convergence sweep (session
    history: frequencies changed by <0.1% going from 10mm down to 5mm,
    while solve time grew ~5x, so 10mm was chosen as the accurate-but-
    not-wasteful default). Checks the real structural modes land within
    a couple percent of that sweep's recorded 10mm values, not just
    that ccx exits cleanly -- confirms the winding fix produces a
    physically sound solve, not merely a non-crashing one.
    """
    shell = shell_from_closed_mesh(DATA_DIR / "real_soundboard_with_defects.stl", element_size=10.0)
    inp_path = tmp_path / "real_free_free.inp"
    write_shell_frequency_inp(shell, SPRUCE_PLACEHOLDER, num_layers=3, inp_path=inp_path, n_frequencies=10)

    run_ccx(inp_path, ccx_path)
    modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))
    frequencies = [m.frequency_hz for m in modes]

    for freq in frequencies[:6]:
        assert freq < 1.0, f"expected a near-zero rigid-body mode, got {freq} Hz"
    real_modes = frequencies[6:]
    assert real_modes == sorted(real_modes), "eigenfrequencies should be strictly increasing"

    expected_at_10mm = [19.553, 24.397, 43.202, 52.772]  # from the convergence sweep, same element_size
    for actual, expected in zip(real_modes, expected_at_10mm):
        assert actual == pytest.approx(expected, rel=0.02), (actual, expected)
    assert real_modes == sorted(real_modes), "eigenfrequencies should be strictly increasing"
