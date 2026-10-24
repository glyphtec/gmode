"""Point attachments (assembly.PointAttachment): grounded Z springs and
lumped masses on the nearest body surface, checked against closed-form
rigid-body physics and against where they land in a bonded assembly."""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import trimesh

from soundboard_lab import cli
from soundboard_lab.assembly import AssemblyBody, PointAttachment, write_assembly_frequency_inp
from soundboard_lab.materials import ISOTROPIC_PLACEHOLDER, SPRUCE_PLACEHOLDER
from soundboard_lab.mesh import mesh_surface
from soundboard_lab.repair import repair_stl
from soundboard_lab.shell_mesh import extrude_shell_to_prisms, shell_from_closed_mesh

DATA_DIR = Path(__file__).parent / "data" / "bonding_exercise"
PLATE_STL = Path(__file__).parent / "data" / "soundboard_plate.stl"  # 150 x 200 x 1.5 mm, centred


def _run_plate(tmp_path, ccx_path, attachments):
    tmp_path.mkdir(parents=True, exist_ok=True)
    attachments_json = tmp_path / "attachments.json"
    attachments_json.write_text(json.dumps({"attachments": attachments}))
    output = tmp_path / "result.json"
    code = cli.main([
        "--stl", str(PLATE_STL), "--ccx-exe", ccx_path, "--output-json", str(output),
        "--element-size", "15", "--mesh-type", "shell", "--n-frequencies", "8",
        "--attachments-json", str(attachments_json),
    ])
    result = json.loads(output.read_text())
    assert code == 0, result
    return result


def test_spring_and_mass_match_rigid_body_physics(ccx_path, tmp_path):
    """A free plate on one soft vertical spring under its centre bounces
    at sqrt(k/M)/2pi; adding a point mass m gives sqrt(k/(M+m))/2pi. The
    spring is soft enough (~3.5 Hz) to stay clear of plate bending
    (~70 Hz), so the closed form holds to well under 1%."""
    k = 0.01  # N/mm
    plate_mass = trimesh.load(PLATE_STL).volume * SPRUCE_PLACEHOLDER.density  # t
    spring = {"name": "spring", "position": [0, 0, 0.75], "radius": 20, "stiffness": k}
    mass = {"name": "mass", "position": [0, 0, -0.75], "radius": 0, "mass_g": 20.0}

    sprung = _run_plate(tmp_path / "spring", ccx_path, [spring])
    loaded = _run_plate(tmp_path / "loaded", ccx_path, [spring, mass])

    bounce = [m["frequency_hz"] for m in sprung["modes"]][5]
    bounce_loaded = [m["frequency_hz"] for m in loaded["modes"]][5]
    assert bounce == pytest.approx(math.sqrt(k / plate_mass) / (2 * math.pi), rel=0.01)
    assert bounce_loaded == pytest.approx(math.sqrt(k / (plate_mass + 20e-6)) / (2 * math.pi), rel=0.01)
    assert sprung["point_attachments"][0]["body"] == "Soundboard"
    assert sprung["point_attachments"][0]["nodes"] > 1
    assert loaded["point_attachments"][1]["nodes"] == 1


@pytest.fixture(scope="module")
def board_and_rib(tmp_path_factory):
    work = tmp_path_factory.mktemp("attach")
    repair_stl(DATA_DIR / "soundboard.stl", work / "board.stl")
    repair_stl(DATA_DIR / "rib1.stl", work / "rib.stl")
    shell = shell_from_closed_mesh(work / "board.stl", element_size=10.0)
    rib = mesh_surface(work / "rib.stl", element_size=2.0)
    return shell, rib


def _attachment_node_coords(inp: Path, points: np.ndarray, elset: str) -> np.ndarray:
    lines = inp.read_text().splitlines()
    start = lines.index(next(line for line in lines if line.endswith(f"ELSET={elset}")))
    ids = []
    for line in lines[start + 1 :]:
        if line.startswith("*"):
            break
        ids.append(int(line.split(",")[1]))
    return points[np.array(ids) - 1]


def test_attachment_lands_on_the_nearest_body(board_and_rib, tmp_path):
    """A spring on top of the rib acts on the rib; one on the board's top
    face acts on the board's top face only -- never leaking through to
    its bottom face, which is within the patch radius."""
    shell, rib = board_and_rib
    lo, hi = rib.points.min(axis=0), rib.points.max(axis=0)
    rib_top = ((lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, hi[2])
    board_top_z = float(shell.top_z.max())
    board_point = (0.0, float(shell.xy[:, 1].min()) + 60.0, board_top_z)
    inp = tmp_path / "a.inp"
    result = write_assembly_frequency_inp(
        shell, SPRUCE_PLACEHOLDER, 3, [AssemblyBody("rib1", rib, ISOTROPIC_PLACEHOLDER)], 0.1, inp,
        attachments=[
            PointAttachment("on_rib", rib_top, radius=3.0, stiffness=10.0),
            PointAttachment("on_board", board_point, radius=8.0, mass_g=3.0),
        ],
    )
    on_rib, on_board = result.attachments
    assert on_rib.body == "rib1" and on_rib.node_count > 1
    assert on_board.body == "Soundboard" and on_board.node_count > 1

    points = np.vstack([extrude_shell_to_prisms(shell, 3).nodes, rib.points])
    rib_nodes = _attachment_node_coords(inp, points, "Eattk1")
    board_nodes = _attachment_node_coords(inp, points, "Eattm2")
    assert np.allclose(rib_nodes[:, 2], hi[2], atol=1e-6)
    assert np.allclose(board_nodes[:, 2], board_top_z, atol=0.05)
    assert re.search(r"\*SPRING, ELSET=Eattk1\n3\n[0-9.]+E[+-]\d+", inp.read_text())


def test_attachment_rejects_bad_specs(board_and_rib, tmp_path):
    shell, _ = board_and_rib
    top = (0.0, float(shell.xy[:, 1].min()) + 60.0, float(shell.top_z.max()))
    cases = [
        (PointAttachment("far", (top[0], top[1], top[2] + 20.0), stiffness=1.0), "from the nearest body surface"),
        (PointAttachment("empty", top), "neither a spring stiffness nor a mass"),
        (PointAttachment("negative", top, mass_g=-1.0), "must not be negative"),
    ]
    for attachment, message in cases:
        with pytest.raises(ValueError, match=message):
            write_assembly_frequency_inp(shell, SPRUCE_PLACEHOLDER, 3, [], 0.1, tmp_path / "x.inp", attachments=[attachment])
