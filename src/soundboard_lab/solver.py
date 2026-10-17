"""CalculiX (ccx) subprocess invocation and `.dat` result parsing.

Pure Python, no `bpy` imports (see docs/CLAUDE.md, "Engine/UI boundary").
This is the Stage 1 validation-ladder slice: invoke ccx on an already
meshed `.inp` and parse its eigenfrequency table. No mesh generation
lives here.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class CcxError(RuntimeError):
    """Raised when ccx fails to run or exits with a nonzero return code.

    ccx's failure modes are not uniformly graceful -- a malformed `.inp`
    can produce a clean "*ERROR" message and a nonzero exit, or a hard
    native crash with no diagnostic text at all (observed: a dangling
    element-to-node reference crashed the solver outright). Treat any
    nonzero return code as failure and always attach whatever stdout/
    stderr was captured, since that may be the only clue available.
    """

    def __init__(self, message: str, returncode: int, stdout: str, stderr: str):
        super().__init__(message)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class EigenMode:
    """One row of ccx's `.dat` "E I G E N V A L U E   O U T P U T" table."""

    mode_number: int
    frequency_hz: float


def run_ccx(job_path: Path, ccx_exe: Path, timeout: float | None = None) -> None:
    """Invoke ccx on `job_path.inp`, writing results alongside it.

    `job_path` is the `.inp` file *without* its extension's worth of
    meaning stripped away by ccx's own `-i jobname` convention: ccx is
    invoked as `ccx -i jobname` (no `.inp` suffix) and looks for
    `jobname.inp` in its working directory, writing `jobname.dat` /
    `jobname.frd` / etc. next to it. `job_path` here is that `.inp`
    file's full path; its parent directory becomes the subprocess cwd
    and its stem becomes the `-i` argument, so results land next to it.

    Always uses the explicit `ccx_exe` path, never PATH lookup (Windows
    hazard noted in docs/CLAUDE.md).
    """
    job_path = Path(job_path)
    ccx_exe = Path(ccx_exe)
    if not job_path.is_file():
        raise FileNotFoundError(f"No such .inp file: {job_path}")
    if not ccx_exe.is_file():
        raise FileNotFoundError(f"No such ccx executable: {ccx_exe}")

    result = subprocess.run(
        [str(ccx_exe), "-i", job_path.stem],
        cwd=job_path.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise CcxError(
            f"ccx exited with code {result.returncode} for {job_path.name}",
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


_EIGENVALUE_HEADER = re.compile(r"E\s+I\s+G\s+E\s+N\s+V\s+A\s+L\s+U\s+E\s+O\s+U\s+T\s+P\s+U\s+T")
_MODE_ROW = re.compile(
    r"^\s*(\d+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s+([-+0-9.EeDd]+)\s*$"
)


def parse_eigenfrequencies(dat_path: Path) -> list[EigenMode]:
    """Extract (mode number, frequency in Hz) from a ccx `.dat` file.

    ccx's eigenvalue table has 4 numeric columns after the mode number:
    eigenvalue (rad/time)^2, angular frequency omega (rad/time),
    frequency in Hz (cycles/time), and an imaginary part (rad/time,
    nonzero only for negative/unstable eigenvalues). Verified
    numerically against real ccx output (not the header text, whose
    column alignment is misleading): the 3rd numeric column equals
    sqrt(eigenvalue) to 7 significant figures (omega), and the 4th
    equals omega / (2*pi) (Hz) -- confirmed on both a rigid-body mode
    (omega ~ 1.7e-3) and a real structural mode (omega ~ 9e5). The
    4th column, not the 3rd, is what we want. The table ends at the
    next blank line, before "PARTICIPATION FACTORS".
    """
    dat_path = Path(dat_path)
    if not dat_path.is_file():
        raise FileNotFoundError(
            f"No .dat output at {dat_path} -- ccx may have run but produced no results"
        )
    text = dat_path.read_text()

    header_match = _EIGENVALUE_HEADER.search(text)
    if header_match is None:
        raise CcxError(
            f"No eigenvalue output table found in {dat_path.name}",
            returncode=0,
            stdout=text,
            stderr="",
        )

    modes: list[EigenMode] = []
    lines = text[header_match.end():].splitlines()
    in_table = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if in_table:
                break  # blank line after at least one row ends the table
            continue
        row_match = _MODE_ROW.match(line)
        if row_match is None:
            continue  # column-header lines before the first data row
        in_table = True
        mode_number = int(row_match.group(1))
        frequency_hz = float(row_match.group(4))
        modes.append(EigenMode(mode_number=mode_number, frequency_hz=frequency_hz))

    if not modes:
        raise CcxError(
            f"Eigenvalue table in {dat_path.name} had no parseable rows",
            returncode=0,
            stdout=text,
            stderr="",
        )
    return modes
