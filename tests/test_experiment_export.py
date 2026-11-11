"""Tests for the experiment export (experiment.py / export_cli.py): the
folder layout and manifest on a synthetic result (fast), the
sign-normalisation and failure-cleanup contracts, and one real
end-to-end solve -> export.
"""

import csv
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from soundboard_lab.cli import main as solve_main
from soundboard_lab.export_cli import main as export_main
from soundboard_lab.experiment import SCHEMA, SCHEMA_VERSION, ExportOptions, slugify, write_experiment

DATA_DIR = Path(__file__).parent / "data"


def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def _grid(n=8, width=200.0, height=150.0):
    xs, ys = np.linspace(0.0, width, n + 1), np.linspace(0.0, height, n + 1)
    xy = np.array([[x, y] for y in ys for x in xs])
    triangles = []
    for j in range(n):
        for i in range(n):
            a = j * (n + 1) + i
            triangles += [[a, a + 1, a + n + 2], [a, a + n + 2, a + n + 1]]
    return xy, np.array(triangles)


def _synthetic_work_dir(tmp_path: Path, with_mobility=True) -> Path:
    work = tmp_path / "work"
    work.mkdir()
    xy, triangles = _grid()
    v = len(xy)
    x, y = xy[:, 0] / 200.0, xy[:, 1] / 150.0
    shapes = {
        "1": (np.sin(np.pi * x) * np.sin(np.pi * y)),  # peak positive already
        "2": -(np.sin(np.pi * x) * np.sin(2 * np.pi * y)) * 3.0,  # peak negative -> must be flipped
        "3": np.sin(2 * np.pi * x) * np.sin(np.pi * y),
    }
    energy = {}
    curvature = {}
    for key in ("1", "2"):  # mode 3 deliberately has no energy/curvature
        density = 1e-5 * (1.0 + np.cos(2 * np.pi * x) ** 2 + float(key) * y)
        energy[key] = {
            "energy_per_area": density.tolist(),
            "display_scale": float(np.percentile(density, 98)),
            "total_energy": 1234.5 * float(key),
            "contours": [
                {"fraction": 0.5, "segments": [[20.0, 20.0, 0.0, 180.0, 20.0, 0.0], [20.0, 20.0, 0.0, 20.0, 130.0, 0.0]]},
                {"fraction": 1.0, "segments": []},
            ],
        }
        curvature[key] = {
            "hessian": np.full((v, 3), 2.0).tolist(),
            "ticks": [[100.0, 75.0, 0.0, 110.0, 80.0, 0.0]],
            "plate_energy_ratio": 0.955,
        }
    result = {
        "status": "ok",
        "material_name": "TestWood",
        "triangle_count": len(triangles),
        "modes": [{"mode_number": 1, "frequency_hz": 101.25}, {"mode_number": 2, "frequency_hz": 182.5},
                  {"mode_number": 3, "frequency_hz": 240.0}],
        "mesh_repair": {"was_already_valid": True, "problems_found": [], "components_removed": 0},
        "mesh_shape": {
            "xy": xy.tolist(), "top_z": np.full(v, 2.0).tolist(), "thickness": np.full(v, 2.0).tolist(),
            "triangles": triangles.tolist(),
        },
        "mode_shapes": {k: s.tolist() for k, s in shapes.items()},
        "mode_energy": energy,
        "mode_curvature": curvature,
        "bonded_bodies": [{"name": "rib1", "tie_nodes": 120}],
        "point_attachments": [],
    }
    if with_mobility:
        result["bridge_mobility"] = {
            "m1_mode": 1, "m1_frequency_hz": 101.25, "stiffness_kgf_per_mm": 3.3, "effective_mass_g": 66.0,
            "mobility_m_per_s_per_n": 0.021,
            "modes": [
                {"mode_number": 1, "frequency_hz": 101.25, "phi_drive": 0.1, "phi_read": 0.1,
                 "compliance_mm_per_kgf": 0.25, "fraction": 0.8},
                {"mode_number": 2, "frequency_hz": 182.5, "phi_drive": -0.05, "phi_read": -0.05,
                 "compliance_mm_per_kgf": 0.05, "fraction": 0.15},
            ],
        }
    (work / "result.json").write_text(json.dumps(result))
    (work / "inputs.json").write_text(json.dumps({"blender_version": "5.2.2", "parameters": {"element_size": 15.0}}))
    (work / "analysis.inp").write_text("*NODE\n")
    (work / "analysis.dat").write_text("eigen\n")
    (work / "board.stl").write_text("solid x\nendsolid x\n")
    (work / "analysis.frd").write_text("huge, must not be copied")
    (work / "status.json").write_text("{}")
    return work


def test_slugify_makes_safe_folder_names():
    assert slugify("My Board #3: v2 / test") == "My-Board-3-v2-test"
    assert slugify("???") == "experiment"
    assert len(slugify("x" * 200)) == 60


def test_synthetic_export_writes_the_documented_layout(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    folder = write_experiment(work, tmp_path / "exports", "Braceless v1", notes="first try", options=ExportOptions(image_width=300))

    assert folder.parent == tmp_path / "exports" and folder.name.endswith("_Braceless-v1")
    assert not list((tmp_path / "exports").glob("*.partial")), "staging folder must be renamed away"

    manifest = json.loads((folder / "experiment.json").read_text())
    assert (manifest["schema"], manifest["schema_version"]) == (SCHEMA, SCHEMA_VERSION)
    assert manifest["name"] == "Braceless v1" and manifest["notes"] == "first try"
    assert manifest["folder"] == folder.name and len(manifest["id"]) == 32
    assert manifest["summary"]["frequencies_hz"] == [101.25, 182.5, 240.0]
    assert manifest["summary"]["m1_mode"] == 1 and manifest["summary"]["stiffness_kgf_per_mm"] == 3.3
    assert manifest["inputs"]["parameters"]["element_size"] == 15.0
    assert manifest["software"]["blender"] == "5.2.2"
    assert manifest["mesh"]["vertices"] == 81 and manifest["mesh"]["thickness_mm"] == {"min": 2.0, "max": 2.0}
    assert manifest["mesh"]["bonded_bodies"] == [{"name": "rib1", "tie_nodes": 120}]
    assert manifest["mobility"]["m1_mode"] == 1

    spec = manifest["image_spec"]
    assert spec["width_px"] == 300 and len(spec["colormap"]) == 5
    for entry in manifest["modes"]:
        n = entry["mode_number"]
        image = folder / entry["displacement"]["image"]
        assert image.name == f"mode_{n:02d}_displacement.png"
        assert _png_size(image) == (spec["width_px"], spec["height_px"])

    m1, m2, m3 = manifest["modes"]
    assert m1["is_m1"] and not m2["is_m1"]
    assert m1["mobility"]["fraction"] == 0.8 and m3["mobility"] is None
    # Mode 1 alone at the bridge: k = 1 / 0.25 mm/kgf, m = k / omega^2.
    assert m1["mobility"]["modal_stiffness_kgf_per_mm"] == pytest.approx(4.0)
    assert m1["mobility"]["modal_mass_g"] == pytest.approx(4.0 * 9.80665 * 1000.0 / (2 * np.pi * 101.25) ** 2 * 1000.0, rel=1e-6)
    assert m2["mobility"]["modal_stiffness_kgf_per_mm"] == pytest.approx(20.0)
    assert m1["strain_energy"]["plate_energy_ratio"] == 0.955
    assert (folder / m1["strain_energy"]["image"]).is_file() and (folder / m2["strain_energy"]["image"]).is_file()
    assert m3["strain_energy"] is None, "a mode without an energy map is recorded, not faked"
    assert not (folder / "images" / "mode_03_energy.png").exists()

    rows = list(csv.DictReader((folder / "modes.csv").open()))
    assert [r["mode_number"] for r in rows] == ["1", "2", "3"]
    assert rows[0]["is_m1"] == "True" and float(rows[1]["frequency_hz"]) == 182.5
    assert rows[2]["total_strain_energy"] == "" and rows[2]["mobility_fraction"] == ""

    solver = sorted(p.name for p in (folder / "solver").iterdir())
    assert solver == ["analysis.dat", "analysis.inp", "board.stl", "inputs.json"]
    assert manifest["files"]["solver"] == solver


def test_modes_are_sign_normalised_in_fields_images_and_hessian(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    folder = write_experiment(work, tmp_path / "out", "s", options=ExportOptions(image_width=200))
    fields = np.load(folder / "fields.npz")
    manifest = json.loads((folder / "experiment.json").read_text())

    assert fields["displacement"].shape == (3, 81) and fields["hessian"].shape == (3, 81, 3)
    for row in fields["displacement"]:
        assert row[np.argmax(np.abs(row))] > 0, "the largest-magnitude displacement is positive"
    assert [e["displacement"]["sign_flipped"] for e in manifest["modes"]] == [False, True, False]
    # Mode 2 was flipped, so its Hessian was too; mode 1 was not.
    assert np.allclose(fields["hessian"][0], 2.0) and np.allclose(fields["hessian"][1], -2.0)
    assert np.isnan(fields["energy_per_area"][2]).all() and np.isnan(fields["hessian"][2]).all()
    assert np.array_equal(fields["mode_numbers"], [1, 2, 3])
    assert fields["thickness"].shape == (81,) and fields["triangles"].shape[1] == 3


def test_net_volume_displacement_is_exported_per_mode(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    folder = write_experiment(work, tmp_path / "out", "vol", options=ExportOptions(displacement_images=False,
                              energy_images=False, fields=False, solver_files=False))
    manifest = json.loads((folder / "experiment.json").read_text())
    m1, m2, m3 = (m["displacement"] for m in manifest["modes"])
    # Mode 1 is sin(pi x/200) sin(pi y/150): integral = 200 * 150 * (2/pi)^2 (the 8x8 mesh underestimates it slightly).
    assert m1["net_volume"] == pytest.approx(200 * 150 * (2 / np.pi) ** 2, rel=0.05)
    assert m1["net_volume_ratio"] == pytest.approx(1.0)
    # Modes 2 and 3 have an odd lobe pair: the net volume cancels while the gross does not.
    assert abs(m2["net_volume"]) < 1e-6 * 3 * 200 * 150 and abs(m2["net_volume_ratio"]) < 1e-6
    assert abs(m3["net_volume_ratio"]) < 1e-6
    rows = list(csv.DictReader((folder / "modes.csv").open()))
    assert float(rows[0]["net_volume"]) == m1["net_volume"] and float(rows[0]["net_volume_ratio"]) == 1.0
    assert "net_volume" in manifest["conventions"]


def test_net_volume_follows_the_sign_normalisation(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    result = json.loads((work / "result.json").read_text())
    result["mode_shapes"]["1"] = (-np.asarray(result["mode_shapes"]["1"])).tolist()  # solver sign flipped
    (work / "result.json").write_text(json.dumps(result))
    folder = write_experiment(work, tmp_path / "out", "vol", options=ExportOptions(displacement_images=False,
                              energy_images=False, fields=False, solver_files=False))
    m1 = json.loads((folder / "experiment.json").read_text())["modes"][0]["displacement"]
    assert m1["sign_flipped"] is True and m1["net_volume"] > 0, "volume is reported in the exported sign convention"


def test_mobility_phi_follows_the_same_sign_convention_as_the_images(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    folder = write_experiment(work, tmp_path / "out", "phi", options=ExportOptions(displacement_images=False,
                              energy_images=False, fields=False, solver_files=False))
    manifest = json.loads((folder / "experiment.json").read_text())
    m1, m2, _m3 = manifest["modes"]
    # Mode 2 was flipped on export; its raw phi were (-0.05, -0.05). Both flip together, so the product is unchanged.
    assert m2["displacement"]["sign_flipped"] is True
    assert (m2["mobility"]["phi_drive"], m2["mobility"]["phi_read"]) == (0.05, 0.05)
    assert (m1["mobility"]["phi_drive"], m1["mobility"]["phi_read"]) == (0.1, 0.1)
    row = list(csv.DictReader((folder / "modes.csv").open()))[1]
    assert (row["phi_drive"], row["phi_read"]) == ("0.05", "0.05")


def test_a_mode_that_opposes_the_static_deflection_has_no_modal_stiffness_or_mass(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    result = json.loads((work / "result.json").read_text())
    result["bridge_mobility"]["modes"][1]["compliance_mm_per_kgf"] = -0.02
    result["bridge_mobility"]["modes"][1]["fraction"] = -0.05
    (work / "result.json").write_text(json.dumps(result))
    folder = write_experiment(work, tmp_path / "out", "neg", options=ExportOptions(displacement_images=False,
                              energy_images=False, fields=False, solver_files=False))
    manifest = json.loads((folder / "experiment.json").read_text())
    assert manifest["modes"][1]["mobility"]["modal_stiffness_kgf_per_mm"] is None
    assert manifest["modes"][1]["mobility"]["modal_mass_g"] is None
    assert manifest["modes"][1]["mobility"]["fraction"] == -0.05, "the share itself is kept"
    row = list(csv.DictReader((folder / "modes.csv").open()))[1]
    assert row["modal_stiffness_kgf_per_mm"] == "" and row["modal_mass_g"] == "" and row["mobility_fraction"] == "-0.05"


def test_options_switch_off_images_fields_and_solver_files(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    folder = write_experiment(
        work, tmp_path / "out", "lean",
        options=ExportOptions(displacement_images=False, energy_images=False, fields=False, solver_files=False),
    )
    manifest = json.loads((folder / "experiment.json").read_text())
    assert not (folder / "images").exists() and not (folder / "fields.npz").exists()
    assert not (folder / "solver").exists()
    assert manifest["image_spec"] is None and manifest["files"]["fields"] is None
    assert all(m["displacement"]["image"] is None for m in manifest["modes"])
    assert (folder / "modes.csv").is_file()


def test_overlays_change_the_energy_image_but_not_the_displacement_image(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    with_overlays = write_experiment(work, tmp_path / "a", "o", options=ExportOptions(image_width=300, overlays=True))
    without = write_experiment(work, tmp_path / "b", "o", options=ExportOptions(image_width=300, overlays=False))
    assert (with_overlays / "images/mode_01_energy.png").read_bytes() != (without / "images/mode_01_energy.png").read_bytes()
    assert (with_overlays / "images/mode_01_displacement.png").read_bytes() == (
        without / "images/mode_01_displacement.png"
    ).read_bytes()


def test_same_name_twice_gets_distinct_folders(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    options = ExportOptions(displacement_images=False, energy_images=False, fields=False, solver_files=False)
    first = write_experiment(work, tmp_path / "out", "dup", options=options)
    second = write_experiment(work, tmp_path / "out", "dup", options=options)
    assert first != second and first.is_dir() and second.is_dir()


def test_a_failed_solve_is_refused_and_leaves_nothing_behind(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "result.json").write_text(json.dumps({"status": "error", "message": "ccx exploded"}))
    with pytest.raises(ValueError, match="ccx exploded"):
        write_experiment(work, tmp_path / "out", "bad")
    assert not (tmp_path / "out").exists(), "a refused export must not even create the output folder"

    with pytest.raises(FileNotFoundError):
        write_experiment(tmp_path / "missing", tmp_path / "out", "bad")
    assert not (tmp_path / "out").exists()


def test_a_crash_mid_export_cleans_up_its_staging_folder(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    result = json.loads((work / "result.json").read_text())
    del result["mode_shapes"]["2"]
    (work / "result.json").write_text(json.dumps(result))
    with pytest.raises(KeyError):
        write_experiment(work, tmp_path / "out", "crash")
    assert list((tmp_path / "out").iterdir()) == [], "no .partial or half-written folder may remain"


def test_export_cli_reports_the_folder_or_a_readable_error(tmp_path):
    work = _synthetic_work_dir(tmp_path)
    ok_json = tmp_path / "ok.json"
    code = export_main(["--work-dir", str(work), "--out-root", str(tmp_path / "out"), "--name", "cli test",
                        "--output-json", str(ok_json), "--image-width", "200", "--no-solver-files"])
    assert code == 0
    reported = json.loads(ok_json.read_text())
    assert reported["status"] == "ok" and (Path(reported["folder"]) / "experiment.json").is_file()
    assert not (Path(reported["folder"]) / "solver").exists()

    bad_json = tmp_path / "bad.json"
    code = export_main(["--work-dir", str(tmp_path / "nope"), "--out-root", str(tmp_path / "out"), "--name", "x",
                        "--output-json", str(bad_json)])
    assert code == 1
    failure = json.loads(bad_json.read_text())
    assert failure["status"] == "error" and "result.json" in failure["message"]


def test_real_solve_exports_images_and_metrics_for_every_mode(ccx_path, tmp_path):
    work = tmp_path / "solve"
    work.mkdir()
    exit_code = solve_main(
        [
            "--stl", str(DATA_DIR / "real_soundboard_with_neck_block_notch.stl"),
            "--ccx-exe", str(ccx_path),
            "--output-json", str(work / "result.json"),
            "--element-size", "15.0", "--n-frequencies", "3", "--mesh-type", "shell",
            "--boundary-condition", "fixed_edge", "--num-layers", "3",
        ]
    )
    assert exit_code == 0
    solved = json.loads((work / "result.json").read_text())

    folder = write_experiment(work, tmp_path / "exports", "real", options=ExportOptions(image_width=500))
    manifest = json.loads((folder / "experiment.json").read_text())
    assert [m["frequency_hz"] for m in manifest["modes"]] == pytest.approx(
        [m["frequency_hz"] for m in solved["modes"]], rel=1e-7
    )
    spec = manifest["image_spec"]
    for entry in manifest["modes"]:
        for view, key in (("displacement", "displacement"), ("energy", "strain_energy")):
            assert _png_size(folder / entry[key]["image"]) == (spec["width_px"], spec["height_px"]), view
        assert 0.90 < entry["strain_energy"]["plate_energy_ratio"] < 1.02
    assert manifest["mesh"]["thickness_mm"]["min"] > 0
    assert {p.name for p in (folder / "solver").iterdir()} >= {"analysis.inp", "analysis.dat"}
    assert not any(p.suffix == ".frd" for p in (folder / "solver").iterdir())
