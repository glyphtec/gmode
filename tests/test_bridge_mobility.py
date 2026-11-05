"""Static bridge monopole mobility (mobility.py): the static solve, its
cross-check against the eigenmodes, and the MM = 2 pi f / k formula."""

import json
import math
from pathlib import Path

import pytest

from soundboard_lab import cli

PLATE_STL = Path(__file__).parent / "data" / "soundboard_plate.stl"  # 150 x 200 x 1.5 mm, centred on the origin
PLATE_TOP_Z = 0.75
DRIVES = [
    {"name": "bass", "position": [-20.0, 0.0, PLATE_TOP_Z], "radius": 4.0},
    {"name": "treble", "position": [20.0, 0.0, PLATE_TOP_Z], "radius": 4.0},
]


def _run(tmp_path, ccx_path, boundary="fixed_edge", n_frequencies=30, mobility=None, extra=()):
    tmp_path.mkdir(parents=True, exist_ok=True)
    spec = tmp_path / "mobility.json"
    spec.write_text(json.dumps(mobility if mobility is not None else {"drive_points": DRIVES}))
    output = tmp_path / "result.json"
    code = cli.main(
        [
            "--stl", str(PLATE_STL), "--ccx-exe", ccx_path, "--output-json", str(output),
            "--mesh-type", "shell", "--element-size", "8", "--num-layers", "2",
            "--boundary-condition", boundary, "--n-frequencies", str(n_frequencies),
            "--mobility-json", str(spec), *extra,
        ]
    )
    return code, json.loads(output.read_text())


_CACHE: dict = {}


@pytest.fixture
def plate_result(tmp_path_factory, ccx_path):
    if "mm" not in _CACHE:
        code, result = _run(tmp_path_factory.mktemp("mobility"), ccx_path)
        assert code == 0, result
        _CACHE["mm"] = result["bridge_mobility"]
    return _CACHE["mm"]


def test_mobility_follows_the_static_formula(plate_result):
    mm = plate_result
    k_n_per_m = mm["stiffness_kgf_per_mm"] * 9.80665 * 1000.0
    omega = 2.0 * math.pi * mm["m1_frequency_hz"]
    assert mm["compliance_mm_per_kgf"] > 0
    assert mm["mobility_m_per_s_per_n"] == pytest.approx(omega / k_n_per_m, rel=1e-9)
    assert mm["effective_mass_g"] == pytest.approx(k_n_per_m / omega**2 * 1000.0, rel=1e-9)
    assert mm["mobility_m_per_s_per_n"] == pytest.approx(1.0 / math.sqrt(k_n_per_m * mm["effective_mass_g"] / 1000.0), rel=1e-9)


def test_modes_reproduce_the_static_compliance(plate_result):
    """Mass-normalised eigenvectors: the modal sum of phi_drive * phi_read / omega^2
    approaches the static solve from below as modes are added."""
    mm = plate_result
    assert 0.85 < mm["modal_sum_fraction"] < 1.02
    first = min(mm["modes"], key=lambda m: m["frequency_hz"])
    assert mm["m1_mode"] == first["mode_number"], "the lowest mode dominates a centred bridge load on a clamped plate"
    assert first["fraction"] > 0.5


def test_symmetric_drive_gives_equal_saddle_deflections(plate_result):
    bass, treble = plate_result["drive_deflection_mm_per_kgf"]
    assert bass == pytest.approx(treble, rel=0.02)
    assert plate_result["read_position"] == pytest.approx([0.0, 0.0, PLATE_TOP_Z], abs=1e-6)


def test_mode_override_changes_only_the_frequency(tmp_path, ccx_path, plate_result):
    code, result = _run(tmp_path, ccx_path, mobility={"drive_points": DRIVES, "mode": 3})
    assert code == 0, result
    mm = result["bridge_mobility"]
    assert mm["m1_mode"] == 3 and mm["m1_chosen_by"] == "requested"
    assert mm["compliance_mm_per_kgf"] == pytest.approx(plate_result["compliance_mm_per_kgf"], rel=1e-9)


def test_free_rim_is_rejected_before_solving(tmp_path, ccx_path):
    code, result = _run(tmp_path, ccx_path, boundary="free_free")
    assert code == 1
    assert "held rim" in result["message"]
