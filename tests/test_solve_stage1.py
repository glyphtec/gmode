"""Stage 1 of the validation ladder (docs/CLAUDE.md): a hand-built,
already-meshed `.inp` -- no gmsh, no Blender -- run through the real
ccx binary, parsing eigenfrequencies out of its `.dat` output.

Expected frequencies below were captured by actually running
tests/data/stage1_single_hex.inp through ccx 2.22 and reading its
output, not invented -- see session history. Modes 1-3 are rigid-body
modes with negative (numerically noisy) eigenvalues, for which ccx
reports an exact 0.0 Hz real frequency; modes 4-6 are rigid-body modes
with tiny positive eigenvalues (near-zero, not exactly zero -- pure
numerical noise, so only bounded, not pinned exactly); modes 7-10 are
the real structural modes of this single steel hex element.
"""

import shutil
from pathlib import Path

import pytest

from soundboard_lab.solver import CcxError, parse_eigenfrequencies, run_ccx

DATA_DIR = Path(__file__).parent / "data"

# (mode_number, expected_frequency_hz) for the real structural modes.
# Pairs 7/8 and 9/10 are near-degenerate, as expected for a symmetric cube.
EXPECTED_REAL_MODES = {
    7: 172037.6,
    8: 172038.5,
    9: 243291.4,
    10: 243299.1,
}


def test_stage1_single_hex_eigenfrequencies(ccx_path, tmp_path):
    job_path = tmp_path / "stage1.inp"
    shutil.copy(DATA_DIR / "stage1_single_hex.inp", job_path)

    run_ccx(job_path, ccx_path)
    modes = parse_eigenfrequencies(job_path.with_suffix(".dat"))

    assert [m.mode_number for m in modes] == list(range(1, 11))

    by_mode = {m.mode_number: m.frequency_hz for m in modes}

    # Modes 1-3: negative/noisy eigenvalues -> ccx reports exactly 0.0 Hz.
    for mode_number in (1, 2, 3):
        assert by_mode[mode_number] == 0.0

    # Modes 4-6: tiny positive (numerically noisy) rigid-body frequencies.
    for mode_number in (4, 5, 6):
        assert 0.0 <= by_mode[mode_number] < 0.01

    # Modes 7-10: real structural modes, matched within 1% of the
    # captured reference values.
    for mode_number, expected_hz in EXPECTED_REAL_MODES.items():
        assert by_mode[mode_number] == pytest.approx(expected_hz, rel=0.01)


def test_run_ccx_raises_on_malformed_input(ccx_path, tmp_path):
    """A malformed .inp (dangling node references) crashes ccx outright
    -- no graceful "*ERROR" text, just a nonzero native exit code (seen
    empirically as -1073740940 on Windows). run_ccx() must surface this
    as a clear exception regardless, never assume a tidy failure mode.
    """
    job_path = tmp_path / "bad.inp"
    shutil.copy(DATA_DIR / "stage1_malformed_dangling_nodes.inp", job_path)

    with pytest.raises(CcxError) as exc_info:
        run_ccx(job_path, ccx_path)
    assert exc_info.value.returncode != 0
