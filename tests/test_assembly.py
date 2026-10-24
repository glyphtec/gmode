"""Multi-body assembly solve: Soundboard shell extrusion + tet bodies
joined by *TIE (see soundboard_lab/assembly.py)."""

import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from soundboard_lab import cli
from soundboard_lab.assembly import AssemblyBody, outer_wall_nodes, tie_slave_nodes, write_assembly_frequency_inp
from soundboard_lab.bonding import shell_surface
from soundboard_lab.materials import ISOTROPIC_PLACEHOLDER, SPRUCE_PLACEHOLDER
from soundboard_lab.mesh import mesh_surface
from soundboard_lab.repair import repair_stl
from soundboard_lab.shell_mesh import shell_from_closed_mesh

DATA_DIR = Path(__file__).parent / "data" / "bonding_exercise"
PLATE_STL = Path(__file__).parent / "data" / "soundboard_plate.stl"


@pytest.fixture(scope="module")
def board_and_rib(tmp_path_factory):
    work = tmp_path_factory.mktemp("assembly")
    repair_stl(DATA_DIR / "soundboard.stl", work / "board.stl")
    repair_stl(DATA_DIR / "rib1.stl", work / "rib.stl")
    shell = shell_from_closed_mesh(work / "board.stl", element_size=10.0)
    rib = mesh_surface(work / "rib.stl", element_size=2.0)
    return shell, rib


def test_tie_slave_nodes_are_the_rib_glue_face(board_and_rib):
    shell, rib = board_and_rib
    slave = tie_slave_nodes(rib, shell_surface(shell), tolerance=0.1)
    assert len(slave) > 0
    # Only the rib's underside (z ~ 2.0, the board's top face) -- never
    # its top (z ~ 10) or side walls.
    assert np.all(np.abs(rib.points[slave, 2] - 2.0) < 0.1)
    # Corner AND mid-side nodes of the glue faces are tied.
    tets = next(c.data for c in rib.cells if c.type == "tetra10")
    assert np.isin(slave, tets[:, 4:]).any()


def test_assembly_inp_ties_each_body_and_rejects_floating_ones(board_and_rib, tmp_path):
    shell, rib = board_and_rib
    inp = tmp_path / "assembly.inp"
    result = write_assembly_frequency_inp(
        shell, SPRUCE_PLACEHOLDER, 3, [AssemblyBody("rib1", rib, ISOTROPIC_PLACEHOLDER)], 0.1, inp, fixed_edge=True
    )
    text = inp.read_text()
    assert "*ELEMENT, TYPE=C3D15, ELSET=Eboard" in text
    assert "*ELEMENT, TYPE=C3D10, ELSET=Ebody1" in text
    assert "*TIE, NAME=Tie1, POSITION TOLERANCE=0.1, ADJUST=NO" in text
    assert "Stie1, Sboard" in text
    assert "MATERIAL=Mbody1" in text and "MATERIAL=Mboard" in text
    assert result.tie_node_counts["rib1"] > 0
    assert len(result.top_layer_node_ids) == len(shell.xy)

    lifted = rib.copy()
    lifted.points = rib.points + np.array([0.0, 0.0, 1.0])
    with pytest.raises(ValueError, match="no glue face"):
        write_assembly_frequency_inp(
            shell, SPRUCE_PLACEHOLDER, 3, [AssemblyBody("floating", lifted, SPRUCE_PLACEHOLDER)], 0.1, tmp_path / "x.inp"
        )


def _frame_lining_stl(path: Path, half_x=75.0, half_y=100.0, width=8.0, z0=0.75, height=8.0) -> Path:
    """Watertight rectangular frame sitting flush on soundboard_plate.stl's
    top face -- a clean stand-in lining (outer wall = the plate edge)."""
    outer = [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]
    inner = [(x - np.sign(x) * width, y - np.sign(y) * width) for x, y in outer]
    z1 = z0 + height
    verts = [(*p, z) for z in (z0, z1) for p in outer + inner]  # 0-3 outer@z0, 4-7 inner@z0, 8-15 @z1
    faces = []
    for i in range(4):
        j = (i + 1) % 4
        o0, o1, i0, i1 = i, j, 4 + i, 4 + j
        faces += [(o0, i1, o1), (o0, i0, i1)]  # bottom (normal -z)
        faces += [(o0 + 8, o1 + 8, i1 + 8), (o0 + 8, i1 + 8, i0 + 8)]  # top (+z)
        faces += [(o0, o1, o1 + 8), (o0, o1 + 8, o0 + 8)]  # outer wall
        faces += [(i0, i1 + 8, i1), (i0, i0 + 8, i1 + 8)]  # inner wall
    mesh = trimesh.Trimesh(verts, faces, process=True)
    mesh.fix_normals()
    assert mesh.is_watertight and mesh.volume > 0
    mesh.export(path)
    return path


def test_outer_wall_nodes_are_the_frame_outside(tmp_path):
    shell = shell_from_closed_mesh(PLATE_STL, element_size=15.0)
    frame = mesh_surface(_frame_lining_stl(tmp_path / "frame.stl"), element_size=5.0)
    outer = outer_wall_nodes(frame, shell)
    p = frame.points[outer]
    on_outside = np.isclose(np.abs(p[:, 0]), 75.0, atol=1e-6) | np.isclose(np.abs(p[:, 1]), 100.0, atol=1e-6)
    assert on_outside.all(), "inner wall / top / bottom nodes picked up"
    for side in (p[:, 0] > 74.9, p[:, 0] < -74.9, p[:, 1] > 99.9, p[:, 1] < -99.9):
        assert side.sum() > 10, "every side of the frame has outer-wall nodes"
    all_outside = np.isclose(np.abs(frame.points[:, 0]), 75.0) | np.isclose(np.abs(frame.points[:, 1]), 100.0)
    assert len(outer) == all_outside.sum(), "the whole outer wall is selected"


def _run_body_lining(tmp_path, ccx_path, terminal):
    tmp_path.mkdir(parents=True, exist_ok=True)
    bodies_json = tmp_path / "bodies.json"
    bodies_json.write_text(json.dumps({
        "tolerance": 0.1,
        "bodies": [{"name": "lining", "stl": str(_frame_lining_stl(tmp_path / "frame.stl")), "element_size": 5.0,
                    "material": None, "orientation": None, "terminal": terminal}],
    }))
    output = tmp_path / "result.json"
    code = cli.main([
        "--stl", str(PLATE_STL), "--ccx-exe", ccx_path, "--output-json", str(output),
        "--element-size", "15", "--mesh-type", "shell", "--n-frequencies", "8",
        "--boundary-condition", "body_lining", "--bodies-json", str(bodies_json),
    ])
    result = json.loads(output.read_text())
    assert code == 0, result
    return result


def test_cli_body_lining_terminals(ccx_path, tmp_path):
    """Fixed outer wall: no rigid modes. Free: the lining just rides
    along, so the assembly floats -- exactly 6 rigid modes. A stiff
    spring converges to Fixed; a soft one is lower."""
    fixed = _run_body_lining(tmp_path / "fixed", ccx_path, {"condition": "fixed"})
    free = _run_body_lining(tmp_path / "free", ccx_path, {"condition": "free"})
    stiff = _run_body_lining(tmp_path / "stiff", ccx_path, {"condition": "spring", "stiffness": 1e8})
    soft = _run_body_lining(tmp_path / "soft", ccx_path, {"condition": "spring", "stiffness": 1e3})
    f = {name: [m["frequency_hz"] for m in r["modes"]] for name, r in
         (("fixed", fixed), ("free", free), ("stiff", stiff), ("soft", soft))}

    assert fixed["bonded_bodies"][0]["terminal_nodes"] > 0
    assert fixed["bonded_bodies"][0]["tie_nodes"] > 0
    assert f["fixed"][0] > 10.0, f
    assert sum(1 for x in f["free"] if x < 5.0) == 6, f
    assert f["free"][6] > 50.0, f
    assert f["stiff"][0] == pytest.approx(f["fixed"][0], rel=0.02), f
    assert f["soft"][0] < f["stiff"][0], f


def test_cli_body_lining_requires_a_terminal_body(tmp_path):
    output = tmp_path / "result.json"
    code = cli.main([
        "--stl", str(PLATE_STL), "--ccx-exe", "unused", "--output-json", str(output),
        "--mesh-type", "shell", "--boundary-condition", "body_lining",
    ])
    assert code == 1
    assert "terminal" in json.loads(output.read_text())["message"]


def test_cli_bonded_free_free_solve_is_one_connected_body(ccx_path, tmp_path):
    """The real proof, same idea as test_tie_mechanics: a free-free
    Soundboard + rib has exactly 6 rigid-body modes when tied (12 if
    the rib were floating free)."""
    bodies_json = tmp_path / "bodies.json"
    bodies_json.write_text(json.dumps({
        "tolerance": 0.1,
        "bodies": [{"name": "rib1", "stl": str(DATA_DIR / "rib1.stl"), "element_size": 2.0,
                    "material": None, "orientation": None}],
    }))
    output = tmp_path / "result.json"
    code = cli.main([
        "--stl", str(DATA_DIR / "soundboard.stl"),
        "--ccx-exe", ccx_path,
        "--output-json", str(output),
        "--element-size", "10",
        "--mesh-type", "shell",
        "--boundary-condition", "free_free",
        "--bodies-json", str(bodies_json),
    ])
    result = json.loads(output.read_text())
    assert code == 0, result
    assert result["bonded_bodies"][0]["name"] == "rib1"
    assert result["bonded_bodies"][0]["tie_nodes"] > 0
    freqs = [m["frequency_hz"] for m in result["modes"]]
    assert sum(1 for f in freqs if f < 1.0) == 6, freqs
    assert freqs[6] > 5.0, freqs
    assert len(result["mode_shapes"]["7"]) == len(result["mesh_shape"]["xy"])
