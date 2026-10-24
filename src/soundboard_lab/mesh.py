"""gmsh-based mesh generation and CalculiX `.inp` card writing.

Pure Python, no `bpy` imports (see docs/CLAUDE.md, "Engine/UI boundary").
Stage 2 (box meshed programmatically via gmsh's own geometry kernel)
and Stage 3 (an arbitrary, already-triangulated surface -- i.e. the
Blender-authored-geometry path) both live here, plus orthotropic
material card writing (Stage 4).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import gmsh
import meshio

from soundboard_lab.materials import (
    IsotropicMaterial as GenericIsotropicMaterial,
    Material,
    OrthotropicWoodMaterial,
    to_calculix_engineering_constants,
    validate_material,
)


@dataclass(frozen=True)
class IsotropicMaterial:
    """Isotropic material constants, in whatever unit system the mesh
    geometry uses (e.g. mm/MPa/t per docs/CLAUDE.md's unit convention).
    """

    name: str
    youngs_modulus: float
    poissons_ratio: float
    density: float


def mesh_box(
    dx: float, dy: float, dz: float, element_size: float, origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> meshio.Mesh:
    """Mesh a solid box with gmsh into quadratic (10-node) tets.

    Returns a `meshio.Mesh` containing only the volume (`tetra10`)
    cells. gmsh's own lower-dimensional point/edge/surface cells (used
    internally for geometric tagging) are discarded -- they have no
    CalculiX equivalent, and meshio's Abaqus writer can't translate
    them (confirmed empirically: a bare "vertex" cell block raises
    `KeyError` there).
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("box")
        ox, oy, oz = origin
        gmsh.model.occ.addBox(ox, oy, oz, dx, dy, dz)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        gmsh.model.mesh.generate(3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            msh_path = Path(tmp_dir) / "mesh.msh"
            gmsh.write(str(msh_path))
            full_mesh = meshio.read(msh_path)
    finally:
        gmsh.finalize()

    volume_cells = [c for c in full_mesh.cells if c.type == "tetra10"]
    if not volume_cells:
        raise RuntimeError("gmsh produced no tetra10 (quadratic tet) volume elements")
    return meshio.Mesh(points=full_mesh.points, cells=volume_cells)


def mesh_surface(stl_path: Path, element_size: float, angle_deg: float = 40.0) -> meshio.Mesh:
    """Remesh an arbitrary, already-triangulated watertight surface
    (e.g. exported from a Blender-authored object) into quadratic
    (10-node) volume tets.

    Unlike `mesh_box` (which defines geometry directly via gmsh's own
    OCC kernel), this takes a raw STL -- gmsh has no B-rep/CAD
    knowledge of it yet, just triangles. The workflow (merge -> detect
    sharp edges -> reparametrize into a real geometry -> build a
    volume from the resulting surface loop -> mesh) follows gmsh's
    own official tutorial for exactly this case (`t13.py`,
    "Remeshing an STL file without an underlying CAD model"), not
    improvised: https://gitlab.onelab.info/gmsh/gmsh/-/blob/master/tutorials/python/t13.py

    `angle_deg` is the dihedral-angle threshold above which an edge
    between two triangles is treated as a sharp geometric feature
    (gmsh's own tutorial default is 40); a flat-faced plate's actual
    edges are far sharper than that; a smoothly curved soundboard
    surface would need no hard edges detected at all mid-surface.
    """
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.merge(str(stl_path))

        angle_rad = angle_deg * 3.141592653589793 / 180.0
        # Split surfaces into patches gmsh can always parametrize: without
        # it a long thin ring (the rim lining's top/bottom faces) can make
        # the 2D mesher spin forever -- a hang, not a catchable error.
        force_parametrizable = True
        include_boundary = True
        curve_angle_rad = 180.0 * 3.141592653589793 / 180.0
        gmsh.model.mesh.classifySurfaces(
            angle_rad, include_boundary, force_parametrizable, curve_angle_rad
        )
        gmsh.model.mesh.createGeometry()

        surfaces = gmsh.model.getEntities(2)
        surface_loop = gmsh.model.geo.addSurfaceLoop([s[1] for s in surfaces])
        gmsh.model.geo.addVolume([surface_loop])
        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("Mesh.MeshSizeMax", element_size)
        gmsh.option.setNumber("Mesh.ElementOrder", 2)
        # Straight mid-side nodes: the STL is faceted, so curving them
        # onto gmsh's reparametrized surface adds no accuracy and can
        # invert thin tets (seen on the rim lining: nonpositive jacobian).
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
        gmsh.model.mesh.generate(3)
        with tempfile.TemporaryDirectory() as tmp_dir:
            msh_path = Path(tmp_dir) / "mesh.msh"
            gmsh.write(str(msh_path))
            full_mesh = meshio.read(msh_path)
    finally:
        gmsh.finalize()

    volume_cells = [c for c in full_mesh.cells if c.type == "tetra10"]
    if not volume_cells:
        raise RuntimeError("gmsh produced no tetra10 (quadratic tet) volume elements")
    return meshio.Mesh(points=full_mesh.points, cells=volume_cells)


def _node_element_cards(mesh: meshio.Mesh) -> list[str]:
    """Shared `*NODE`/`*ELEMENT` card lines for both material paths."""
    tetra_cells = next(c.data for c in mesh.cells if c.type == "tetra10")
    lines = ["*NODE, NSET=Nall"]
    for i, (x, y, z) in enumerate(mesh.points, start=1):
        lines.append(f"{i}, {x:.10g}, {y:.10g}, {z:.10g}")
    lines.append("*ELEMENT, TYPE=C3D10, ELSET=Evolumes")
    for i, conn in enumerate(tetra_cells, start=1):
        # meshio connectivity is 0-indexed; CalculiX node IDs are 1-indexed.
        node_ids = ", ".join(str(n + 1) for n in conn)
        lines.append(f"{i}, {node_ids}")
    return lines


def _frequency_step_cards(n_frequencies: int) -> list[str]:
    """Shared `*STEP`/`*FREQUENCY` output-request cards."""
    return [
        "*STEP",
        "*FREQUENCY",
        str(n_frequencies),
        "*NODE FILE",
        "U",
        "*EL FILE",
        "S, E",
        "*OUTPUT, FREQUENCY=1",
        "*END STEP",
    ]


def write_frequency_inp(
    mesh: meshio.Mesh,
    material: IsotropicMaterial,
    inp_path: Path,
    n_frequencies: int = 10,
) -> None:
    """Write a free-free `*FREQUENCY` CalculiX `.inp` from a quadratic-
    tet mesh. Card style matches tests/data/stage1_single_hex.inp (same
    keyword set) so results are directly comparable.

    `TYPE=C3D10` is written explicitly. meshio's own Abaqus writer
    defaults to the equivalent-but-differently-named "C3D10MH" variant
    (confirmed: its `abaqus_to_meshio_type` table maps C3D10, C3D10H,
    C3D10I, C3D10M and C3D10MH all to the same meshio cell type
    "tetra10" -- they share identical node connectivity/order and
    differ only in solver formulation, so the variant meshio happens
    to pick when reversing that lookup is incidental, not a
    requirement). Writing our own cards keeps the element type
    traceable to the project's stated C3D10 convention (docs/CLAUDE.md,
    "known traps") rather than a library default.

    Node order: meshio's gmsh reader applies a documented permutation
    (`[0,1,2,3,4,5,6,7,9,8]`) converting gmsh's native tet10 node order
    into meshio's own canonical order, and meshio's Abaqus writer
    applies no further permutation when writing -- i.e. meshio's
    canonical order is designed to already match Abaqus/CalculiX's
    C3D10 convention, so `mesh.cells` connectivity is used as-is here.
    This is corroborated, not just assumed: CalculiX itself flags a
    genuinely wrong node order immediately as a negative-Jacobian
    error, and the Stage 2 regression test confirms a clean solve.
    """
    lines = [
        "** Soundboard Lab -- isotropic quadratic-tet mesh, gmsh-generated.",
        *_node_element_cards(mesh),
        f"*MATERIAL, NAME={material.name}",
        "*ELASTIC",
        f"{material.youngs_modulus:.10g}, {material.poissons_ratio:.10g}",
        "*DENSITY",
        f"{material.density:.10g}",
        f"*SOLID SECTION, ELSET=Evolumes, MATERIAL={material.name}",
        *_frequency_step_cards(n_frequencies),
    ]
    Path(inp_path).write_text("\n".join(lines) + "\n")


def write_frequency_inp_orthotropic(
    mesh: meshio.Mesh,
    material: Material,
    inp_path: Path,
    n_frequencies: int = 10,
    orientation: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
) -> None:
    """Write a free-free `*FREQUENCY` CalculiX `.inp` from a quadratic-
    tet mesh. Despite the name (kept for backward compatibility with
    existing callers/tests), `material` may be either an
    `OrthotropicWoodMaterial` (writes `*ELASTIC, TYPE=ENGINEERING
    CONSTANTS` plus an `*ORIENTATION` card) or a `materials.
    IsotropicMaterial` (writes `*ELASTIC, TYPE=ISO`; no orientation --
    stiffness is directionally invariant for an isotropic material) --
    same kind-dispatch pattern as `shell_mesh.write_shell_frequency_inp`.

    `validate_material` is called unconditionally before any card is
    written -- for an orthotropic material, a non-positive-definite
    stiffness tensor is known (docs/CLAUDE.md "known traps") to
    produce chaotic, increasingly-wrong high-mode eigenfrequencies
    rather than a clean solver error, so this must fail loudly here,
    before ever reaching ccx.

    `orientation` is the raw six `*ORIENTATION` numbers (a vector along
    the local 1-axis, then a vector in the local 1-2 plane); identity
    by default (mesh authored/oriented with its grain running along
    global Y). Ignored for an isotropic material, same as the shell
    path. The addon resolves a per-object grain rotation into this
    pair of vectors before calling in (`analysis._compute_grain_
    orientation`), since grain orientation is an object-level property,
    independent of the Material ID-block.
    """
    validate_material(material)

    if isinstance(material, OrthotropicWoodMaterial):
        e1, e2, e3, nu12, nu13, nu23, g12, g13, g23 = to_calculix_engineering_constants(material)
        a1, a2, a3, b1, b2, b3 = orientation
        lines = [
            "** Soundboard Lab -- orthotropic quadratic-tet mesh, gmsh-generated.",
            *_node_element_cards(mesh),
            "*ORIENTATION, NAME=GrainAxes",
            f"{a1:.10g}, {a2:.10g}, {a3:.10g}, {b1:.10g}, {b2:.10g}, {b3:.10g}",
            f"*MATERIAL, NAME={material.name}",
            "*ELASTIC, TYPE=ENGINEERING CONSTANTS",
            f"{e1:.10g}, {e2:.10g}, {e3:.10g}, {nu12:.10g}, {nu13:.10g}, {nu23:.10g}, {g12:.10g}, {g13:.10g},",
            f"{g23:.10g}",
            "*DENSITY",
            f"{material.density:.10g}",
            f"*SOLID SECTION, ELSET=Evolumes, MATERIAL={material.name}, ORIENTATION=GrainAxes",
        ]
    elif isinstance(material, GenericIsotropicMaterial):
        lines = [
            "** Soundboard Lab -- isotropic quadratic-tet mesh, gmsh-generated.",
            *_node_element_cards(mesh),
            f"*MATERIAL, NAME={material.name}",
            "*ELASTIC, TYPE=ISO",
            f"{material.e:.10g}, {material.nu:.10g}",
            "*DENSITY",
            f"{material.density:.10g}",
            f"*SOLID SECTION, ELSET=Evolumes, MATERIAL={material.name}",
        ]
    else:
        raise TypeError(f"Unknown material type: {type(material)!r}")

    lines += _frequency_step_cards(n_frequencies)
    Path(inp_path).write_text("\n".join(lines) + "\n")

