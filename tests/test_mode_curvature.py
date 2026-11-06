"""Tests for the mode-shape curvature field (mode_curvature.py): exact
checks of the Hessian recovery, principal-direction and plate-stiffness
math on synthetic inputs, then real fixed-edge ccx solves checking the
whole chain against ccx's own strain energy -- in the default grain
orientation and rotated 90 degrees (which exercises the rotated
orthotropic plate stiffness, not just the identity case).
"""

import math
from pathlib import Path

import numpy as np
import pytest

from soundboard_lab.materials import SPRUCE_PLACEHOLDER, IsotropicMaterial
from soundboard_lab.mode_curvature import (
    build_mode_curvature,
    default_tick_spacing,
    make_tick_grid,
    plate_bending_matrix,
    principal_bending,
    sample_direction_ticks,
    triangle_hessians,
    vertex_hessians,
)
from soundboard_lab.mode_energy import build_mode_energy
from soundboard_lab.shell_mesh import shell_from_closed_mesh, write_shell_frequency_inp
from soundboard_lab.solver import (
    parse_eigenfrequencies,
    parse_frd_mode_displacements,
    parse_frd_mode_energy_density,
    run_ccx,
)

DATA_DIR = Path(__file__).parent / "data"
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def _quadratic(x, y):
    return 3.0 * x**2 - 2.0 * x * y + 0.5 * y**2


def _grid_mesh(n=6, size=60.0):
    """n x n square cells split into triangles, on [0, size]^2."""
    xs = np.linspace(0.0, size, n + 1)
    xy = np.array([[x, y] for y in xs for x in xs])
    triangles = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            b, c, d = a + 1, a + n + 1, a + n + 2
            triangles += [[a, b, d], [a, d, c]]
    return xy, np.array(triangles)


def _quadratic_field_on(xy, triangles, func):
    w_corner = func(xy[:, 0], xy[:, 1])
    mids = 0.5 * np.stack(
        [xy[triangles[:, 0]] + xy[triangles[:, 1]], xy[triangles[:, 1]] + xy[triangles[:, 2]],
         xy[triangles[:, 2]] + xy[triangles[:, 0]]],
        axis=1,
    )
    return w_corner, func(mids[:, :, 0], mids[:, :, 1])


def test_triangle_hessian_is_exact_for_a_quadratic_field_on_an_irregular_triangle():
    xy = np.array([[1.0, 2.0], [7.0, 3.0], [2.5, 9.0]])
    triangles = np.array([[0, 1, 2]])
    w_corner, w_mid = _quadratic_field_on(xy, triangles, _quadratic)
    h = triangle_hessians(xy, triangles, w_corner, w_mid)
    assert h[0] == pytest.approx([6.0, 1.0, -2.0])


def test_vertex_hessians_of_a_quadratic_field_are_uniform_across_the_mesh():
    xy, triangles = _grid_mesh()
    w_corner, w_mid = _quadratic_field_on(xy, triangles, _quadratic)
    vertex_h = vertex_hessians(xy, triangles, triangle_hessians(xy, triangles, w_corner, w_mid))
    assert np.allclose(vertex_h, [6.0, 1.0, -2.0])


def test_principal_bending_picks_the_larger_magnitude_axis_and_its_angle():
    k1, k2, angle = principal_bending(np.array([[2.0, -5.0, 0.0]]))
    assert (k1[0], k2[0]) == pytest.approx((-5.0, 2.0))
    assert angle[0] == pytest.approx(math.pi / 2)

    # The same field rotated by 30 degrees: w = -5 * (x sin a ... ) style rotation of H.
    a = math.radians(30.0)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    h = rot @ np.diag([-5.0, 2.0]) @ rot.T
    k1, k2, angle = principal_bending(np.array([[h[0, 0], h[1, 1], h[0, 1]]]))
    assert (k1[0], k2[0]) == pytest.approx((-5.0, 2.0))
    assert angle[0] == pytest.approx(a)


def test_isotropic_plate_stiffness_matches_the_textbook_formulas():
    e, nu, t = 10000.0, 0.3, 2.0
    d = plate_bending_matrix(IsotropicMaterial(name="iso", density=1e-9, e=e, nu=nu), IDENTITY, thickness=t)
    d11 = e * t**3 / (12.0 * (1.0 - nu**2))
    assert d[0, 0] == pytest.approx(d11) and d[1, 1] == pytest.approx(d11)
    assert d[0, 1] == pytest.approx(nu * d11)
    assert d[2, 2] == pytest.approx(e * t**3 / (24.0 * (1.0 + nu)))
    assert d[0, 2] == pytest.approx(0.0) and d[1, 2] == pytest.approx(0.0)


def test_orthotropic_plate_stiffness_swaps_axes_when_the_grain_is_rotated_ninety_degrees():
    d0 = plate_bending_matrix(SPRUCE_PLACEHOLDER, IDENTITY, thickness=2.0)
    d90 = plate_bending_matrix(SPRUCE_PLACEHOLDER, (0.0, 1.0, 0.0, -1.0, 0.0, 0.0), thickness=2.0)
    assert d90[0, 0] == pytest.approx(d0[1, 1]) and d90[1, 1] == pytest.approx(d0[0, 0])
    assert d90[0, 1] == pytest.approx(d0[0, 1]) and d90[2, 2] == pytest.approx(d0[2, 2])
    assert d0[1, 1] > 10.0 * d0[0, 0], "the grain (Y) axis must be far stiffer than cross-grain"
    # A 45-degree grain couples bending and twisting.
    d45 = plate_bending_matrix(SPRUCE_PLACEHOLDER, (1.0, 1.0, 0.0, -1.0, 1.0, 0.0), thickness=2.0)
    assert abs(d45[0, 2]) > 0.0


def test_ticks_follow_the_bending_axis_and_skip_flat_regions():
    xy, triangles = _grid_mesh(n=8, size=80.0)
    top_z = np.zeros(len(xy))
    grid = make_tick_grid(xy, top_z, triangles, spacing=10.0)

    # w = x^2 bends along x only: every tick must lie along x (no dy).
    w_corner, w_mid = _quadratic_field_on(xy, triangles, lambda x, y: x**2)
    vertex_h = vertex_hessians(xy, triangles, triangle_hessians(xy, triangles, w_corner, w_mid))
    ticks = sample_direction_ticks(grid, triangles, vertex_h)
    assert len(ticks) == len(grid.points)
    assert np.allclose(ticks[:, 4] - ticks[:, 1], 0.0) and np.all(ticks[:, 3] > ticks[:, 0])

    # A flat field has no bending to show.
    assert len(sample_direction_ticks(grid, triangles, np.zeros_like(vertex_h))) == 0

    # Only the strongly curved half gets ticks when curvature varies.
    varying = vertex_h.copy()
    varying[xy[:, 0] < 40.0] = 0.0
    kept = sample_direction_ticks(grid, triangles, varying)
    assert 0 < len(kept) < len(grid.points)
    assert kept[:, [0, 3]].min() > 30.0


@pytest.mark.parametrize(
    "orientation", [IDENTITY, (0.0, 1.0, 0.0, -1.0, 0.0, 0.0)], ids=["grain_along_y", "grain_rotated_90"]
)
def test_curvature_field_reproduces_ccx_strain_energy_on_a_real_solve(ccx_path, tmp_path, orientation):
    """The thin-plate energy from the recovered curvature field and the
    orthotropic plate stiffness must land close to (just under, since
    plate theory leaves out transverse shear) the strain energy ccx
    itself reports -- which can only hold if the Hessian recovery, the
    node mapping, and the (rotated) stiffness are all right."""
    shell = shell_from_closed_mesh(DATA_DIR / "real_soundboard_with_defects.stl", element_size=15.0)
    inp_path = tmp_path / "curvature.inp"
    result = write_shell_frequency_inp(
        shell, SPRUCE_PLACEHOLDER, num_layers=3, inp_path=inp_path, fixed_edge=True, n_frequencies=3,
        orientation=orientation,
    )
    run_ccx(inp_path, ccx_path)
    modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))
    displacements = parse_frd_mode_displacements(inp_path.with_suffix(".frd"))
    energy_density = parse_frd_mode_energy_density(inp_path.with_suffix(".frd"))
    thickness = shell.top_z - shell.bottom_z
    grid = make_tick_grid(shell.xy, shell.top_z, shell.triangles, default_tick_spacing(shell.xy, shell.triangles))

    for mode in modes:
        energy = build_mode_energy(
            shell.xy, shell.top_z, shell.triangles, result.corner_node_ids, thickness, energy_density[mode.mode_number]
        )
        payload = build_mode_curvature(
            shell.xy, shell.triangles, thickness, result.top_layer_node_ids, result.top_mid_node_ids,
            displacements[mode.mode_number], SPRUCE_PLACEHOLDER, orientation, grid,
            reference_energy=energy["total_energy"],
        )
        assert 0.90 < payload["plate_energy_ratio"] < 1.02, payload["plate_energy_ratio"]
        assert len(payload["hessian"]) == len(shell.xy)
        assert len(payload["ticks"]) > 0 and all(len(t) == 6 for t in payload["ticks"])
