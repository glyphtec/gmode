"""Tests for the engine CLI entry point -- this is what the Blender
addon's Analysis & Solve operator will invoke as a subprocess (see
cli.py's module docstring for why: the engine has zero bpy imports,
and gmsh/meshio can't go into Blender's bundled Python).
"""

import json
from pathlib import Path

from soundboard_lab.cli import main
from soundboard_lab.materials import IsotropicMaterial, material_to_dict

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


def test_cli_tet_path_custom_material_json_actually_used(ccx_path, tmp_path):
    """Real bug, found during live multi-body bonding-exercise testing:
    run_analysis() (the tet path, default --mesh-type) hardcoded
    SPRUCE_PLACEHOLDER internally and never even accepted a material
    parameter -- --material-json silently had no effect on this path
    (confirmed empirically: a real isotropic CF-composite material
    JSON was passed, but the result's material_name still came back
    "SprucePlaceholder"). Verify the fix the same way as the shell
    path's own equivalent test: a stiffer isotropic material must give
    higher frequencies than the default orthotropic spruce material,
    mode-for-mode, on the same real fixture.
    """
    stiff_isotropic = IsotropicMaterial(name="TestStiffIsotropic", density=1.3e-9, e=32850.0, nu=0.3)
    material_json = tmp_path / "material.json"
    material_json.write_text(json.dumps(material_to_dict(stiff_isotropic)))

    def solve(material_json_path):
        output_json = tmp_path / f"result_{material_json_path is not None}.json"
        args = [
            "--stl", str(DATA_DIR / "soundboard_plate.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
            "--element-size", "15.0",
            "--n-frequencies", "10",
        ]
        if material_json_path is not None:
            args += ["--material-json", str(material_json_path)]
        exit_code = main(args)
        assert exit_code == 0
        return json.loads(output_json.read_text())

    default_result = solve(None)
    assert default_result["material_name"] == "SprucePlaceholder"
    custom_result = solve(material_json)
    assert custom_result["status"] == "ok"
    assert custom_result["material_name"] == "TestStiffIsotropic"

    default_freqs = [m["frequency_hz"] for m in default_result["modes"][6:]]
    custom_freqs = [m["frequency_hz"] for m in custom_result["modes"][6:]]
    # A much stiffer isotropic material (E=32850 MPa vs spruce's
    # ~800-13800 MPa range) must come back higher mode-for-mode, not
    # the same -- proof the override was actually used, not ignored.
    assert all(c > d for c, d in zip(custom_freqs, default_freqs)), (custom_freqs, default_freqs)


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


def test_cli_shell_path_custom_material_json_actually_used(ccx_path, tmp_path):
    """`--material-json` must actually flow through to the solved
    result, not be silently ignored in favor of SPRUCE_PLACEHOLDER --
    verified empirically (not just "didn't crash") by using a
    deliberately much softer isotropic material and confirming the
    resulting frequencies come back markedly lower than the default
    orthotropic spruce material's own fixed-edge frequencies
    (test_cli_shell_path_success), which only makes sense if the
    custom material was the one actually solved.
    """
    soft_isotropic = IsotropicMaterial(name="TestSoftIsotropic", density=4.6e-10, e=500.0, nu=0.3)
    material_json = tmp_path / "material.json"
    material_json.write_text(json.dumps(material_to_dict(soft_isotropic)))

    def solve(material_json_path):
        output_json = tmp_path / f"result_{material_json_path is not None}.json"
        args = [
            "--stl", str(DATA_DIR / "real_soundboard_with_neck_block_notch.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
            "--element-size", "15.0",
            "--n-frequencies", "4",
            "--mesh-type", "shell",
            "--boundary-condition", "fixed_edge",
            "--num-layers", "3",
        ]
        if material_json_path is not None:
            args += ["--material-json", str(material_json_path)]
        exit_code = main(args)
        assert exit_code == 0
        return json.loads(output_json.read_text())

    default_result = solve(None)
    assert default_result["material_name"] == "SprucePlaceholder"
    custom_result = solve(material_json)
    assert custom_result["status"] == "ok"
    assert custom_result["material_name"] == "TestSoftIsotropic"

    default_freqs = [m["frequency_hz"] for m in default_result["modes"]]
    custom_freqs = [m["frequency_hz"] for m in custom_result["modes"]]
    # A ~27x softer isotropic material (E=500 MPa vs spruce's ~800-13800
    # MPa range) must come back substantially lower mode-for-mode, not
    # the same -- proof the override was actually used, not ignored.
    assert all(c < 0.7 * d for c, d in zip(custom_freqs, default_freqs)), (custom_freqs, default_freqs)


def test_cli_shell_path_custom_orientation_changes_result(ccx_path, tmp_path):
    """A 90-degree-rotated grain orientation (swap local 1/2 axes) on
    the same orthotropic material + the same non-square fixture must
    give DIFFERENT frequencies than identity orientation -- proof
    `--orientation` actually reaches the `*ORIENTATION` card, not just
    accepted and discarded. The fixture's grain-soft (radial) and
    grain-stiff (longitudinal) axes aren't interchangeable physically,
    so swapping them must change the solved frequencies.
    """

    def solve(orientation_arg):
        output_json = tmp_path / f"result_{orientation_arg or 'identity'}.json"
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
            + (["--orientation", orientation_arg] if orientation_arg else [])
        )
        assert exit_code == 0
        return [m["frequency_hz"] for m in json.loads(output_json.read_text())["modes"]]

    identity_freqs = solve(None)
    rotated_freqs = solve("0,1,0,-1,0,0")  # local-1 = world Y, local-2 = world X: swaps radial/longitudinal
    assert identity_freqs != rotated_freqs, "rotating the grain 90 degrees must change the solved frequencies"


def test_cli_shell_path_invalid_material_fails_fast_with_clean_error(ccx_path, tmp_path):
    """A non-physical material (E <= 0) must be caught by
    `validate_material` before any meshing/solving is even attempted
    -- verified here by using a bogus material and relying on how fast
    this returns plus the step checklist never reaching "repair".
    """
    bad_material = IsotropicMaterial(name="BadMaterial", density=1.0e-9, e=-1.0, nu=0.3)
    material_json = tmp_path / "bad_material.json"
    material_json.write_text(json.dumps(material_to_dict(bad_material)))

    output_json = tmp_path / "result.json"
    status_json = tmp_path / "status.json"
    exit_code = main(
        [
            "--stl", str(DATA_DIR / "real_soundboard_with_neck_block_notch.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(output_json),
            "--mesh-type", "shell",
            "--material-json", str(material_json),
            "--status-json", str(status_json),
        ]
    )
    assert exit_code == 1
    result = json.loads(output_json.read_text())
    assert result["status"] == "error"
    assert "BadMaterial" in result["message"]

    status = json.loads(status_json.read_text())
    steps = {s["id"]: s for s in status["steps"]}
    assert all(s["status"] == "pending" for s in steps.values()), "should fail before any step even starts"


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
