"""bond_cli: the subprocess entry point the add-on's Detect Bonds
operator launches. Real fixtures through `main()`, checking the JSON
contract the add-on reads, plus the always-write-a-result error path.
"""

import json
from pathlib import Path

from soundboard_lab.bond_cli import main

DATA_DIR = Path(__file__).parent / "data" / "bonding_exercise"


def _write_manifest(path: Path, bodies: list[dict]) -> Path:
    path.write_text(json.dumps({
        "tolerance": 0.1,
        "soundboard": {"stl": str(DATA_DIR / "soundboard.stl"), "element_size": 10.0},
        "bodies": bodies,
    }))
    return path


def test_bond_cli_writes_per_body_results_and_surfaces(tmp_path):
    manifest = _write_manifest(tmp_path / "manifest.json", [
        {"name": "Rib 1", "stl": str(DATA_DIR / "rib1.stl"), "element_size": 2.0},
        {"name": "Ghost", "stl": str(tmp_path / "missing.stl"), "element_size": 2.0},
    ])
    output = tmp_path / "result.json"

    assert main(["--manifest", str(manifest), "--output-json", str(output)]) == 0
    result = json.loads(output.read_text())

    assert result["status"] == "ok"
    assert result["tolerance"] == 0.1
    assert len(result["soundboard_surface"]["points"]) > 0
    assert len(result["soundboard_surface"]["triangles"]) > 0

    rib, ghost = result["bodies"]
    assert rib["name"] == "Rib 1"
    assert rib["status"] == "contact"
    assert abs(rib["contact_area"] - 1800.0) < 2.0
    assert len(rib["signed_distances"]) == len(rib["points"])
    assert max(max(t) for t in rib["triangles"]) < len(rib["points"])

    assert ghost["status"] == "geometry_error"
    assert ghost["message"]
    assert "points" not in ghost


def test_bond_cli_reports_soundboard_failure_as_error_result(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "tolerance": 0.1,
        "soundboard": {"stl": str(tmp_path / "missing.stl"), "element_size": 10.0},
        "bodies": [],
    }))
    output = tmp_path / "result.json"

    assert main(["--manifest", str(manifest), "--output-json", str(output)]) == 1
    result = json.loads(output.read_text())
    assert result["status"] == "error"
    assert result["message"]
