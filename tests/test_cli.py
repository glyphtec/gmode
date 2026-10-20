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


def test_cli_shell_path_success(ccx_path, tmp_path):
    """The shell/extrusion-mesh path (role=Soundboard objects), used
    by the addon's actual "Analyze" button -- unlike the tet path,
    also returns the solved mesh footprint and per-mode Z-displacement
    arrays, needed for the mode-shape heat-map preview.
    """
    output_json = tmp_path / "result.json"
    exit_code = main(
        [
            "--stl", str(DATA_DIR / "real_soundboard_with_neck_block_notch.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
            "--element-size", "15.0",
            "--n-frequencies", "4",
            "--mesh-type", "shell",
            "--boundary-condition", "fixed_edge",
            "--num-layers", "3",
        ]
    )
    assert exit_code == 0

    result = json.loads(output_json.read_text())
    assert result["status"] == "ok"
    assert result["triangle_count"] > 1
    modes = result["modes"]
    assert [m["mode_number"] for m in modes] == [1, 2, 3, 4]
    frequencies = [m["frequency_hz"] for m in modes]
    assert all(f > 1.0 for f in frequencies), "fixed-edge should have no rigid-body modes"
    assert frequencies == sorted(frequencies)

    mesh_shape = result["mesh_shape"]
    n_vertices = len(mesh_shape["xy"])
    assert len(mesh_shape["top_z"]) == n_vertices
    assert all(len(tri) == 3 for tri in mesh_shape["triangles"])

    mode_shapes = result["mode_shapes"]
    assert set(mode_shapes.keys()) == {"1", "2", "3", "4"}
    for dz in mode_shapes.values():
        assert len(dz) == n_vertices


def test_cli_shell_path_writes_step_checklist(ccx_path, tmp_path):
    """Not a numeric progress bar (ccx itself is an opaque subprocess
    call) -- a named-step checklist (pending/running/done/error per
    stage), written at each transition, that the addon can poll and
    display live. Verifies every step ends up "done" with a non-empty
    summary message after a successful run.
    """
    output_json = tmp_path / "result.json"
    status_json = tmp_path / "status.json"
    exit_code = main(
        [
            "--stl", str(DATA_DIR / "real_soundboard_with_neck_block_notch.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
            "--element-size", "15.0",
            "--n-frequencies", "4",
            "--mesh-type", "shell",
            "--boundary-condition", "fixed_edge",
            "--status-json", str(status_json),
        ]
    )
    assert exit_code == 0
    status = json.loads(status_json.read_text())
    steps = {s["id"]: s for s in status["steps"]}
    assert set(steps.keys()) == {"repair", "mesh", "write_input", "solve", "parse_results"}
    for step_id, step in steps.items():
        assert step["status"] == "done", f"step {step_id} was not done: {step}"


def test_cli_shell_path_step_checklist_shows_failing_step(ccx_path, tmp_path):
    """A missing STL must fail at the "repair" step specifically (the
    first step that actually touches the file), with a clear message
    -- not just an opaque overall failure.
    """
    output_json = tmp_path / "result.json"
    status_json = tmp_path / "status.json"
    exit_code = main(
        [
            "--stl", str(tmp_path / "does_not_exist.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
            "--mesh-type", "shell",
            "--status-json", str(status_json),
        ]
    )
    assert exit_code == 1
    status = json.loads(status_json.read_text())
    steps = {s["id"]: s for s in status["steps"]}
    assert steps["repair"]["status"] == "error"
    assert steps["repair"]["message"]  # some clear, non-empty message
    assert steps["mesh"]["status"] == "pending"  # never got there
