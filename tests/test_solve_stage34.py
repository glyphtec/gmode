"""Stage 3 + 4 of the validation ladder (docs/CLAUDE.md): a Blender-
authored surface mesh (not gmsh's own geometry primitives), remeshed
into quadratic C3D10 tets, with an orthotropic spruce material
instead of isotropic -- the combination the project actually exists
for (a soundboard, not a generic steel test block).

tests/data/soundboard_plate.stl is a simple flat rectangular plate
(300x400x3mm, 12 triangles), generated once via a Blender background
script and committed as a static fixture -- this test does not invoke
Blender itself (the engine is developed and tested with pytest
outside Blender, per docs/CLAUDE.md's engine/UI boundary rule).

The spruce constants below are an explicitly-labeled PLACEHOLDER, not
the project owner's lab-measured values (see docs/CLAUDE.md: "Material
constants come from her lab-measured library, never hardcoded
defaults"). They combine two sources: density/E_L/E_R/G_LR/nu_LR from
real measurements on actual spruce guitar soundboards (Giro et al.,
"Predicting the Effects of Bracing Pattern Modifications on Acoustic
Guitar Soundboards", Forum Acusticum 2023) -- a 2D thin-plate model
that doesn't cover through-thickness (tangential) properties -- with
E_T/G_LT/G_RT/nu_LT/nu_RT estimated from general wood-science
literature ratios (Wood Handbook-style Sitka spruce E_T/E_L, and
typical low-G_RT/moderate-nu_RT wood behavior) to fill the gap. This
is for plumbing validation only.
"""

from pathlib import Path

from soundboard_lab.materials import OrthotropicWoodMaterial, validate_positive_definite
from soundboard_lab.mesh import mesh_surface, write_frequency_inp_orthotropic
from soundboard_lab.solver import parse_eigenfrequencies, run_ccx

DATA_DIR = Path(__file__).parent / "data"

SPRUCE_PLACEHOLDER = OrthotropicWoodMaterial(
    name="SprucePlaceholder",
    density=4.6e-10,  # 460 kg/m^3 in t/mm^3
    e_longitudinal=13800,  # MPa
    e_radial=811.76,
    e_tangential=593.4,
    g_lr=828.0,
    g_lt=621.0,
    g_rt=41.4,
    nu_lr=0.38,
    nu_lt=0.40,
    nu_rt=0.45,
)


def test_spruce_placeholder_is_positive_definite():
    """The placeholder constants themselves must pass the stability
    check -- if this ever fails, the regression test below would be
    exercising an already-known-broken material, which defeats the
    point.
    """
    validate_positive_definite(SPRUCE_PLACEHOLDER)


def test_stage34_blender_plate_orthotropic_spruce(ccx_path, tmp_path):
    mesh = mesh_surface(DATA_DIR / "soundboard_plate.stl", element_size=15.0)

    tetra_count = next(c.data.shape[0] for c in mesh.cells if c.type == "tetra10")
    assert tetra_count > 1, "expected gmsh to actually subdivide the plate, not return one element"

    inp_path = tmp_path / "stage34.inp"
    write_frequency_inp_orthotropic(mesh, SPRUCE_PLACEHOLDER, inp_path, n_frequencies=10)

    run_ccx(inp_path, ccx_path)
    modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))

    assert [m.mode_number for m in modes] == list(range(1, 11))
    frequencies = [m.frequency_hz for m in modes]

    # Modes 1-6: rigid-body modes of a free-free plate -- near zero.
    for freq in frequencies[:6]:
        assert freq < 1.0, f"expected a near-zero rigid-body mode, got {freq} Hz"

    # Modes 7-10: real structural modes -- strictly positive,
    # increasing (no chaotic/out-of-order spikes -- the failure
    # pattern this whole pipeline is explicitly guarding against), and
    # in a physically sensible range for a soundboard-sized spruce
    # plate (same order of magnitude as her real measured targets,
    # ~100-200 Hz, per docs/CLAUDE.md's Evergreen section).
    real_modes = frequencies[6:]
    assert all(10 < f < 1000 for f in real_modes), real_modes
    assert real_modes == sorted(real_modes), "eigenfrequencies should be strictly increasing"
    # Unlike the symmetric test cube in Stage 2, a 300x400mm rectangle
    # has no geometric symmetry, so its low modes should NOT come in
    # near-degenerate pairs.
    gaps = [b - a for a, b in zip(real_modes, real_modes[1:])]
    assert all(gap > 1.0 for gap in gaps), (
        f"expected clearly separated modes for an asymmetric plate, got gaps {gaps}"
    )
