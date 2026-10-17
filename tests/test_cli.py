"""Tests for the engine CLI entry point -- this is what the Blender
addon's Analysis & Solve operator will invoke as a subprocess (see
cli.py's module docstring for why: the engine has zero bpy imports,
and gmsh/meshio can't go into Blender's bundled Python).
"""

import json
from pathlib import Path

from soundboard_lab.cli import main

DATA_DIR = Path(__file__).parent / "data"


def test_cli_success(ccx_path, tmp_path):
    output_json = tmp_path / "result.json"
    exit_code = main(
        [
            "--stl", str(DATA_DIR / "soundboard_plate.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
            "--element-size", "15.0",
            "--n-frequencies", "10",
        ]
    )
    assert exit_code == 0

    result = json.loads(output_json.read_text())
    assert result["status"] == "ok"
    assert result["tetra_count"] > 1
    modes = result["modes"]
    assert [m["mode_number"] for m in modes] == list(range(1, 11))
    assert all(m["frequency_hz"] < 1.0 for m in modes[:6])
    real_freqs = [m["frequency_hz"] for m in modes[6:]]
    assert all(10 < f < 1000 for f in real_freqs)
    assert real_freqs == sorted(real_freqs)


def test_cli_rejects_insufficient_through_thickness_layers(ccx_path, tmp_path):
    """element_size=15mm against a 3mm-thick plate gives far fewer
    than the required layers -- this must fail loudly, before ever
    meshing or solving, not silently produce an over-stiff result.
    """
    output_json = tmp_path / "result.json"
    exit_code = main(
        [
            "--stl", str(DATA_DIR / "soundboard_plate.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
            "--element-size", "15.0",
            "--thickness-hint", "3.0",
        ]
    )
    assert exit_code == 1

    result = json.loads(output_json.read_text())
    assert result["status"] == "error"
    assert "layers" in result["message"]


def test_cli_reports_missing_stl_as_error_json(ccx_path, tmp_path):
    output_json = tmp_path / "result.json"
    exit_code = main(
        [
            "--stl", str(tmp_path / "does_not_exist.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
        ]
    )
    assert exit_code == 1
    result = json.loads(output_json.read_text())
    assert result["status"] == "error"
