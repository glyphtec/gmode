"""Stage 2 of the validation ladder (docs/CLAUDE.md): the same simple
box as Stage 1, now meshed programmatically via gmsh (quadratic C3D10
tets) and converted through meshio, instead of hand-typed.

Unlike Stage 1's single hand-specified element, gmsh's triangulation
is a heuristic algorithm -- exact node/element counts aren't pinned
here as a hard regression (they could shift slightly across gmsh
versions). What's checked is the thing Stage 2 actually exists to
prove: the gmsh -> meshio -> `.inp` -> ccx -> parse pipeline runs
clean (no negative-Jacobian / malformed-element errors) and produces
the expected qualitative eigenvalue shape -- 6 near-zero rigid-body
modes followed by real, physically-sensible structural modes.

Captured on one real run (gmsh 4.15.2, ccx 2.22, element_size=5.0,
1108 tetra10 elements): modes 7-10 were 142095/142109/191243/191252
Hz -- lower than Stage 1's crude single-linear-hex result
(172038/172038/243291/243299 Hz), consistent with docs/CLAUDE.md's
"known traps" note that linear elements run artificially stiff.
"""

from soundboard_lab.mesh import IsotropicMaterial, mesh_box, write_frequency_inp
from soundboard_lab.solver import parse_eigenfrequencies, run_ccx

GENERIC_STEEL = IsotropicMaterial(
    name="GenericSteel", youngs_modulus=200000, poissons_ratio=0.3, density=7.9e-09
)


def test_stage2_gmsh_box_eigenfrequencies(ccx_path, tmp_path):
    mesh = mesh_box(dx=10, dy=10, dz=10, element_size=5.0)

    tetra_count = next(c.data.shape[0] for c in mesh.cells if c.type == "tetra10")
    assert tetra_count > 1, "expected gmsh to actually subdivide the box, not return one element"

    inp_path = tmp_path / "stage2.inp"
    write_frequency_inp(mesh, GENERIC_STEEL, inp_path, n_frequencies=10)

    run_ccx(inp_path, ccx_path)
    modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))

    assert [m.mode_number for m in modes] == list(range(1, 11))
    frequencies = [m.frequency_hz for m in modes]

    # Modes 1-6: rigid-body modes of a free-free box -- near zero.
    for freq in frequencies[:6]:
        assert freq < 1.0, f"expected a near-zero rigid-body mode, got {freq} Hz"

    # Modes 7-10: real structural modes -- strictly positive, clearly
    # separated from the rigid-body modes, and in a physically
    # sensible range for a 10mm steel cube (same order of magnitude as
    # Stage 1's cruder single-element result).
    real_modes = frequencies[6:]
    assert all(50_000 < f < 400_000 for f in real_modes), real_modes
    assert real_modes == sorted(real_modes), "eigenfrequencies should be non-decreasing"
