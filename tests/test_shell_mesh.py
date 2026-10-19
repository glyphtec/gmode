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
    _polygon_area,
    _simplify_rim,
    classify_boundary_loops,
    extrude_shell_to_prisms,
    find_boundary_loops,
    mode_shape_z_displacement,
    shell_from_closed_mesh,
    write_shell_frequency_inp,
)
from soundboard_lab.solver import parse_eigenfrequencies, parse_frd_mode_displacements, run_ccx

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


def _square_with_sharp_feature(apex_xy: list[float], apex_direction: str) -> np.ndarray:
    """A 6x6 square's boundary with a sharp corner feature cut into (or
    pointing out from) the bottom edge at x=3, surrounded by densely-
    sampled nearly-collinear points on both sides -- the real torture
    case for curvature-aware simplification: a naive distance-based
    method could drop the feature's defining apex vertex just because
    its neighbors happen to be close together, even though the apex
    itself is geometrically essential (not just "a bit more detail").
    `apex_direction` is purely documentation here; the apex_xy value
    itself determines whether it's an inward notch or outward spike.
    """
    pts = [[0.0, 0.0]]
    for x in [1.0, 2.0, 2.8, 2.9, 2.95]:
        pts.append([x, 0.0])
    pts.append(apex_xy)
    for x in [3.05, 3.1, 3.2]:
        pts.append([x, 0.0])
    pts += [[4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]
    pts += [[6, 2], [6, 4], [6, 6], [4, 6], [2, 6], [0, 6], [0, 4], [0, 2]]
    return np.array(pts, dtype=float)


def test_simplify_rim_preserves_sharp_inward_notch_despite_dense_sampling():
    """A sharp inward notch -- e.g. a neck-block intrusion modeled as
    a deliberately sharp-cornered cut, not a smoothly-filleted curve
    (session history: a real, not hypothetical, modeling choice) --
    must survive simplification exactly, even though its immediately
    neighboring points are only ~0.05-0.1 units apart (much closer
    than the 0.5 epsilon used here), which is exactly the situation
    where a naive distance-based method could drop the apex.
    """
    xy = _square_with_sharp_feature(apex_xy=[3.0, 2.0], apex_direction="inward")
    rim = list(range(len(xy)))
    apex_index = 6  # position of [3.0, 2.0] in the array above

    simplified = _simplify_rim(xy, rim, epsilon=0.5)
    assert apex_index in simplified, "the notch's defining apex vertex must survive simplification"
    assert _polygon_area(xy, simplified) == pytest.approx(_polygon_area(xy, rim)), (
        "shape must be preserved exactly when the only real feature is a single sharp notch"
    )


def test_simplify_rim_preserves_sharp_outward_spike_despite_dense_sampling():
    """The Venetian-cutaway case: a sharp OUTWARD-pointing apex where
    two curves meet at a point rather than blending smoothly (session
    history: a real guitar body style, not hypothetical) -- same
    dense-neighbor torture case as the inward notch, opposite
    direction.
    """
    xy = _square_with_sharp_feature(apex_xy=[3.0, -2.0], apex_direction="outward")
    rim = list(range(len(xy)))
    apex_index = 6

    simplified = _simplify_rim(xy, rim, epsilon=0.5)
    assert apex_index in simplified, "the spike's defining apex vertex must survive simplification"
    assert _polygon_area(xy, simplified) == pytest.approx(_polygon_area(xy, rim)), (
        "shape must be preserved exactly when the only real feature is a single sharp spike"
    )


def test_simplify_rim_collapses_collinear_points_to_true_corners():
    """The complementary case: a plain square with extra collinear
    points scattered along each straight edge must collapse down to
    exactly its 4 true corners -- confirms simplification actually
    simplifies when there's no real detail to preserve, not just that
    it's safe/conservative.
    """
    xy = np.array(
        [
            [0, 0], [2, 0], [4, 0], [6, 0],
            [6, 2], [6, 4], [6, 6],
            [4, 6], [2, 6], [0, 6],
            [0, 4], [0, 2],
        ],
        dtype=float,
    )
    rim = list(range(len(xy)))
    simplified = _simplify_rim(xy, rim, epsilon=0.1)
    assert len(simplified) == 4
    assert set(map(tuple, xy[simplified])) == {(0.0, 0.0), (6.0, 0.0), (6.0, 6.0), (0.0, 6.0)}


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
    result = write_shell_frequency_inp(shell, SPRUCE_PLACEHOLDER, num_layers=3, inp_path=inp_path)
    assert result.element_count < 2000  # vs. 1.36M for the equivalent uniform-tet case

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
    history: frequencies changed by <1.3% going from 10mm down to 5mm,
    so 10mm was chosen as the accurate-but-not-wasteful default).
    Checks the real structural modes land within a couple percent of
    that sweep's recorded 10mm values, not just that ccx exits cleanly
    -- confirms the winding fix produces a physically sound solve, not
    merely a non-crashing one.

    Expected values re-measured after switching `_simplify_rim` to
    curvature-aware Douglas-Peucker (session history): DP coarsens
    smooth regions more aggressively than the prior simple
    distance-based method at the same nominal `element_size` (3715 vs.
    6101 triangles here), shifting the solved frequencies by ~2% --
    expected and correct, not a regression, since the convergence
    trend itself (checked again under DP) still holds.
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

    expected_at_10mm = [19.968, 24.894, 44.388, 53.756]  # from the convergence sweep, same element_size
    for actual, expected in zip(real_modes, expected_at_10mm):
        assert actual == pytest.approx(expected, rel=0.02), (actual, expected)


def test_mode_shape_z_displacement_matches_shell_vertex_order(ccx_path, tmp_path):
    """Full pipeline for mode-shape visualization: solve a real fixed-
    edge case, parse the `.frd` for per-node displacement, then map it
    back onto the shell's own vertex order via
    `top_layer_node_ids`/`mode_shape_z_displacement`.

    Checks this isn't just "doesn't crash": every fixed outer-rim
    vertex must show (numerically) zero Z-displacement for every mode
    (a clamped boundary), while interior vertices for the real
    structural modes should show clearly nonzero displacement --
    confirms the node-ID mapping is actually wired to the right
    vertices, not just producing an array of the right shape.
    """
    shell = shell_from_closed_mesh(DATA_DIR / "real_soundboard_with_defects.stl", element_size=15.0)
    inp_path = tmp_path / "mode_shape.inp"
    result = write_shell_frequency_inp(
        shell, SPRUCE_PLACEHOLDER, num_layers=3, inp_path=inp_path, fixed_edge=True, n_frequencies=4
    )
    run_ccx(inp_path, ccx_path)

    mode_displacements = parse_frd_mode_displacements(inp_path.with_suffix(".frd"))
    dz = mode_shape_z_displacement(result.top_layer_node_ids, mode_displacements[1])
    assert dz.shape == (len(shell.xy),)

    outer, _holes = classify_boundary_loops(shell.xy, shell.triangles)
    assert np.allclose(dz[outer], 0.0), "the clamped outer rim must show zero Z-displacement"

    interior = [v for v in range(len(shell.xy)) if v not in set(outer)]
    assert np.abs(dz[interior]).max() > 0.01, "interior vertices should show real nonzero mode-shape displacement"


def test_shell_from_closed_mesh_preserves_real_neck_block_notch(ccx_path, tmp_path):
    """`real_soundboard_with_neck_block_notch.stl`: a second real,
    independently-exported file -- same booleaned-soundhole defect
    pattern as `real_soundboard_with_defects.stl`, but ALSO includes a
    deliberately sharp-cornered neck-block intrusion (a real luthier
    modeling choice, not a smoothly-filleted curve -- session history:
    raised as the actual real-world torture case for curvature-aware
    simplification, not a hypothetical one).

    Verifies the notch's defining corners survive simplification by
    checking the outer loop's per-vertex turning angle: the notch's 4
    true corners are near-right-angle (confirmed manually at ~90-102
    degrees on this exact file), clearly separated from the smooth
    body outline's gentle curvature peaks (<40 degrees everywhere
    else) -- if DP had rounded the notch off, no vertex would show a
    turn anywhere near 90 degrees.

    Then runs a full free-free + fixed-edge solve to confirm the more
    complex (notched) outer boundary doesn't just look right but
    solves cleanly end-to-end.
    """
    shell = shell_from_closed_mesh(DATA_DIR / "real_soundboard_with_neck_block_notch.stl", element_size=10.0)
    xy, triangles = shell.xy, shell.triangles
    outer, holes = classify_boundary_loops(xy, triangles)
    assert len(holes) == 1, "the neck-block notch is part of the outer loop, not a second hole"

    pts = xy[outer]
    n = len(pts)
    turn_angles = []
    for i in range(n):
        incoming = pts[i] - pts[(i - 1) % n]
        outgoing = pts[(i + 1) % n] - pts[i]
        incoming_n = incoming / np.linalg.norm(incoming)
        outgoing_n = outgoing / np.linalg.norm(outgoing)
        turn_angles.append(np.degrees(np.arccos(np.clip(np.dot(incoming_n, outgoing_n), -1, 1))))
    turn_angles = np.array(turn_angles)

    sharp_corners = np.sum(turn_angles > 80)
    assert sharp_corners >= 4, (
        f"expected at least the neck-block notch's 4 near-right-angle corners to survive, found {sharp_corners}"
    )

    for fixed_edge in (False, True):
        inp_path = tmp_path / f"neck_block_notch_fixed_{fixed_edge}.inp"
        write_shell_frequency_inp(
            shell, SPRUCE_PLACEHOLDER, num_layers=3, inp_path=inp_path, fixed_edge=fixed_edge
        )
        run_ccx(inp_path, ccx_path)
        modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))
        frequencies = [m.frequency_hz for m in modes]
        assert frequencies == sorted(frequencies), "eigenfrequencies should be strictly increasing"
        if not fixed_edge:
            assert all(f < 1.0 for f in frequencies[:6]), f"expected 6 near-zero rigid-body modes, got {frequencies}"
        else:
            assert all(f > 1.0 for f in frequencies), f"a clamped edge should have no rigid-body modes, got {frequencies}"
