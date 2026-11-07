"""Tests for the modal strain-energy map (mode_energy.py): the
through-thickness integration, normalisation and iso-line math on
synthetic inputs, then a real fixed-edge ccx solve that checks the
map against the one quantity it must reproduce -- the mode's total
strain energy, omega^2 / 2 for ccx's mass-normalised eigenvectors.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from soundboard_lab.materials import SPRUCE_PLACEHOLDER
from soundboard_lab.mode_energy import (
    build_mode_energy,
    contour_segments,
    display_scale,
    energy_per_area,
    integrate_over_area,
    normalize_to_unit_energy,
)
from soundboard_lab.shell_mesh import classify_boundary_loops, shell_from_closed_mesh, write_shell_frequency_inp
from soundboard_lab.solver import parse_eigenfrequencies, parse_frd_mode_energy_density, run_ccx

DATA_DIR = Path(__file__).parent / "data"

_SQUARE_XY = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
_SQUARE_TRIANGLES = np.array([[0, 1, 2], [0, 2, 3]])


def test_energy_per_area_integrates_density_through_thickness_by_trapezoid():
    # 3 levels over a 4 mm plate (2 mm apart); density 0, 1, 4 at one vertex.
    corner_ids = np.array([[1], [2], [3]])
    density = {1: 0.0, 2: 1.0, 3: 4.0}
    result = energy_per_area(corner_ids, np.array([4.0]), density)
    assert result == pytest.approx([2.0 * (0.5 * 0.0 + 1.0 + 0.5 * 4.0)])


def test_energy_per_area_clamps_negative_extrapolation_noise():
    corner_ids = np.array([[1], [2]])
    result = energy_per_area(corner_ids, np.array([1.0]), {1: -0.5, 2: 1.0})
    assert result[0] == pytest.approx(0.5 * 1.0)


def test_normalized_map_integrates_to_one_and_reports_total():
    energy = np.array([1.0, 2.0, 3.0, 4.0])
    normalized, total = normalize_to_unit_energy(_SQUARE_XY, _SQUARE_TRIANGLES, energy)
    assert total == pytest.approx(integrate_over_area(_SQUARE_XY, _SQUARE_TRIANGLES, energy))
    assert integrate_over_area(_SQUARE_XY, _SQUARE_TRIANGLES, normalized) == pytest.approx(1.0)


def test_normalize_rejects_a_mode_with_no_energy():
    with pytest.raises(ValueError):
        normalize_to_unit_energy(_SQUARE_XY, _SQUARE_TRIANGLES, np.zeros(4))


def test_contour_of_a_linear_field_is_a_straight_line_at_the_right_place():
    values = _SQUARE_XY[:, 0].copy()  # f = x
    (segments,) = contour_segments(_SQUARE_XY, _SQUARE_TRIANGLES, values, [0.25])
    assert len(segments) > 0
    assert np.allclose(segments[:, [0, 2]], 0.25)
    # Together the segments span the full square height (y from 0 to 1).
    ys = np.concatenate([segments[:, 1], segments[:, 3]])
    assert ys.min() == pytest.approx(0.0) and ys.max() == pytest.approx(1.0)
    assert sum(abs(s[3] - s[1]) for s in segments) == pytest.approx(1.0)


def test_contour_interpolates_a_third_coordinate_along_the_edge():
    points = np.column_stack([_SQUARE_XY, _SQUARE_XY[:, 0] * 10.0])  # z = 10 x
    (segments,) = contour_segments(points, _SQUARE_TRIANGLES, _SQUARE_XY[:, 0].copy(), [0.5])
    assert segments.shape[1] == 6
    assert np.allclose(segments[:, 2], 5.0) and np.allclose(segments[:, 5], 5.0)


def test_contour_outside_the_value_range_is_empty():
    (segments,) = contour_segments(_SQUARE_XY, _SQUARE_TRIANGLES, _SQUARE_XY[:, 0].copy(), [2.0])
    assert len(segments) == 0


def test_display_scale_ignores_a_few_extreme_peaks():
    values = np.concatenate([np.ones(99), [1000.0]])
    assert display_scale(values) == pytest.approx(1.0)


def test_mode_energy_matches_modal_strain_energy_on_a_real_fixed_edge_solve(ccx_path, tmp_path):
    """Per-mode strain energy integrated from the map must equal
    omega^2 / 2 (ccx mass-normalises its eigenvectors, so the kinetic
    and strain energy of a mode are both omega^2 / 2) to within the
    nodal-extrapolation error -- not just "is non-negative". The
    clamped rim must also hold the peak density, as clamped-edge plate
    theory predicts (w = w' = 0 there, w'' maximal).
    """
    shell = shell_from_closed_mesh(DATA_DIR / "real_soundboard_with_defects.stl", element_size=15.0)
    inp_path = tmp_path / "energy.inp"
    result = write_shell_frequency_inp(
        shell, SPRUCE_PLACEHOLDER, num_layers=3, inp_path=inp_path, fixed_edge=True, n_frequencies=3
    )
    run_ccx(inp_path, ccx_path)
    modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))
    energy_by_mode = parse_frd_mode_energy_density(inp_path.with_suffix(".frd"))
    outer = set(classify_boundary_loops(shell.xy, shell.triangles)[0])
    thickness = shell.top_z - shell.bottom_z

    for mode in modes:
        payload = build_mode_energy(
            shell.xy, shell.top_z, shell.triangles, result.corner_node_ids, thickness, energy_by_mode[mode.mode_number]
        )
        omega = 2.0 * math.pi * mode.frequency_hz
        assert payload["total_energy"] == pytest.approx(0.5 * omega**2, rel=0.10)

        density = np.array(payload["energy_per_area"])
        assert integrate_over_area(shell.xy, shell.triangles, density) == pytest.approx(1.0)
        assert int(np.argmax(density)) in outer
        assert payload["display_scale"] > 0.0
        assert [c["fraction"] for c in payload["contours"]] == [0.4, 0.7, 1.0]
        assert all(len(c["segments"]) > 0 for c in payload["contours"][:3])
