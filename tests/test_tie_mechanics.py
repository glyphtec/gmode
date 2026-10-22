"""*TIE mechanics proof (bonding system design, see docs/CLAUDE.md
"Boundary & Bonds UI"): before tackling real multi-body geometry, prove
CalculiX's `*TIE` card actually reproduces a truly continuous
structure's behavior when run through this project's own ccx
invocation -- not assumed from the manual alone (verified against the
real CalculiX 2.22 manual's *TIE/*SURFACE semantics before writing
these fixtures, see tie_proof_tied.inp's own comments), empirically
confirmed by running all three real cases and comparing frequencies.

Expected frequencies below were captured by actually running these
three .inp files through ccx 2.22 and reading the output, not invented.
"""

import shutil
from pathlib import Path

import pytest

from soundboard_lab.solver import parse_eigenfrequencies, run_ccx

DATA_DIR = Path(__file__).parent / "data"


def _run_and_parse(name: str, ccx_path, tmp_path):
    job_path = tmp_path / f"{name}.inp"
    shutil.copy(DATA_DIR / f"{name}.inp", job_path)
    run_ccx(job_path, ccx_path)
    return parse_eigenfrequencies(job_path.with_suffix(".dat"))


def test_tie_reproduces_continuous_structure(ccx_path, tmp_path):
    """The core proof: two independently-numbered hexes joined by
    *TIE must behave almost identically to the same geometry meshed
    with genuinely shared nodes at the interface -- both in rigid-body
    mode COUNT (6, not 12 -- proving the tie makes them act as one
    connected body) and in the real elastic-mode frequencies (matched
    to within 0.01%, not just "roughly the same ballpark").
    """
    continuous = _run_and_parse("tie_proof_continuous", ccx_path, tmp_path)
    tied = _run_and_parse("tie_proof_tied", ccx_path, tmp_path)

    assert [m.mode_number for m in continuous] == list(range(1, 17))
    assert [m.mode_number for m in tied] == list(range(1, 17))

    # Modes 1-6: rigid-body modes, near-zero in both cases -- exactly
    # 6, not some other count, for a single connected 10x10x20 block.
    assert sum(1 for m in continuous if m.frequency_hz < 1.0) == 6
    assert sum(1 for m in tied if m.frequency_hz < 1.0) == 6

    # Modes 7-16: real elastic modes. The tied (independent-node) case
    # must closely reproduce the continuous (shared-node) case's own
    # frequencies -- proof the *TIE card's MPCs are doing the same
    # mechanical job as genuinely shared nodes, not just "something".
    for i in range(6, 16):
        f_continuous = continuous[i].frequency_hz
        f_tied = tied[i].frequency_hz
        assert f_tied == pytest.approx(f_continuous, rel=1e-4), (
            f"mode {i + 1}: continuous={f_continuous} Hz, tied={f_tied} Hz"
        )


def test_no_tie_leaves_bodies_mechanically_disconnected(ccx_path, tmp_path):
    """Negative control: the SAME two independently-numbered hexes,
    with no *TIE at all, must show 12 near-zero rigid-body modes (6
    per independent body), not 6 -- confirming the tie in the other
    test is doing real mechanical work, not a no-op that would have
    passed either way.
    """
    disconnected = _run_and_parse("tie_proof_disconnected", ccx_path, tmp_path)
    assert [m.mode_number for m in disconnected] == list(range(1, 17))
    assert sum(1 for m in disconnected if m.frequency_hz < 1.0) == 12
