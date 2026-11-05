"""Bridge monopole mobility (MM), the project's measure of record.

The guitar measurement is static: load both saddles together, read the
bridge-centre displacement (compliance, mm/kgf), then

    k = 1 / compliance          static stiffness at the bridge
    m = k / (2 pi f)^2          effective mass, f = the M1 frequency
    MM = 1 / sqrt(k m) = 2 pi f / k

so the model does the same: a `*STATIC` solve for the compliance
(`assembly.write_assembly_frequency_inp` with a `BridgeProbe`) and the
M1 frequency from the eigen solve. M1 is taken as the mode adding the
most to that static compliance -- the same bias toward M1 the static
method has -- unless the caller names a mode.

Units follow the engine's mm / tonne / N / s set, so mass-normalised
eigenvector products divided by omega^2 come out directly in mm/N.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from soundboard_lab.solver import CcxError, EigenMode

KGF_TO_N = 9.80665

_DISP_HEADER = re.compile(r"displacements \(vx,vy,vz\) for set (\S+) and time", re.IGNORECASE)
MIN_ELASTIC_HZ = 1.0


def parse_node_print_displacements(dat_path: Path) -> dict[str, dict[int, tuple[float, float, float]]]:
    """`*NODE PRINT, ... U` blocks of a ccx `.dat` as `{SETNAME: {node: (ux, uy, uz)}}`
    (set names are upper-case, as ccx writes them)."""
    text = Path(dat_path).read_text()
    blocks: dict[str, dict[int, tuple[float, float, float]]] = {}
    current: dict[int, tuple[float, float, float]] | None = None
    for line in text.splitlines():
        header = _DISP_HEADER.search(line)
        if header is not None:
            current = blocks.setdefault(header.group(1).upper(), {})
            continue
        if current is None:
            continue
        parts = line.split()
        if not parts:
            if current:
                current = None
            continue
        if len(parts) == 4:
            try:
                current[int(parts[0])] = (float(parts[1]), float(parts[2]), float(parts[3]))
            except ValueError:
                current = None
    if not blocks:
        raise CcxError(
            f"No node-print displacements found in {Path(dat_path).name}",
            returncode=0,
            stdout=text,
            stderr="",
        )
    return blocks


@dataclass(frozen=True)
class ModalContribution:
    mode_number: int
    frequency_hz: float
    phi_drive: float
    phi_read: float
    compliance_mm_per_n: float  # this mode's share of the static compliance
    fraction: float  # of the static compliance


def _patch_mean_z(node_ids, displacements: dict[int, tuple[float, float, float]]) -> float:
    return float(np.mean([displacements[int(n)][2] for n in node_ids]))


def modal_contributions(
    modes: list[EigenMode],
    mode_displacements: dict[int, dict[int, tuple[float, float, float]]],
    drive_node_ids: list[np.ndarray],
    read_node_ids: np.ndarray,
    static_compliance_mm_per_n: float,
) -> list[ModalContribution]:
    """Each elastic mode's share of the static compliance,
    phi_drive * phi_read / omega^2 (needs ccx's mass-normalised
    eigenvectors; the sum against the static solve is the check).
    `phi_drive` is the equal-weight mean over the drive patches, the
    same sharing as the static load."""
    contributions = []
    for mode in modes:
        if mode.frequency_hz < MIN_ELASTIC_HZ:
            continue
        shape = mode_displacements[mode.mode_number]
        phi_drive = float(np.mean([_patch_mean_z(nodes, shape) for nodes in drive_node_ids]))
        phi_read = _patch_mean_z(read_node_ids, shape)
        omega = 2.0 * math.pi * mode.frequency_hz
        compliance = phi_drive * phi_read / omega**2
        contributions.append(
            ModalContribution(
                mode.mode_number,
                mode.frequency_hz,
                phi_drive,
                phi_read,
                compliance,
                compliance / static_compliance_mm_per_n,
            )
        )
    return contributions


def static_compliance_mm_per_n(
    static_displacements: dict[str, dict[int, tuple[float, float, float]]], read_node_ids: np.ndarray, force_n: float
) -> float:
    """Downward bridge-centre deflection per newton (the load is -Z)."""
    if force_n <= 0:
        raise ValueError("force_n must be positive")
    read = static_displacements.get("NREAD")
    if not read:
        raise CcxError("Static solve printed no displacements for the bridge read patch", 0, "", "")
    deflection_mm = -_patch_mean_z(read_node_ids, read)
    if deflection_mm <= 0:
        raise ValueError(f"static deflection at the bridge is not downward ({-deflection_mm:g} mm)")
    return deflection_mm / force_n


def evaluate_bridge_mobility(
    static_dat_path: Path,
    placement,
    probe,
    force_n: float,
    modes: list[EigenMode],
    mode_displacements: dict[int, dict[int, tuple[float, float, float]]],
    mm_mode: int | None = None,
) -> dict:
    """Parse the static solve and combine it with the eigen solve into
    the `bridge_mobility` result block (`placement` is the writer's
    `ProbePlacement`, `probe` the `BridgeProbe` that was requested)."""
    static_displacements = parse_node_print_displacements(static_dat_path)
    compliance = static_compliance_mm_per_n(static_displacements, placement.read_node_ids, force_n)
    contributions = modal_contributions(
        modes, mode_displacements, placement.drive_node_ids, placement.read_node_ids, compliance
    )
    result = bridge_mobility(
        static_displacements, placement.drive_node_ids, placement.read_node_ids, force_n, contributions, mm_mode
    )
    result["drive_points"] = [
        {"name": d.name, "nodes": p.node_count, "snap_distance": p.snap_distance}
        for d, p in zip(probe.drive_points, placement.drive_placements)
    ]
    result["read_position"] = list(placement.read_position)
    result["read_nodes"] = placement.read_placement.node_count
    return result


def bridge_mobility(
    static_displacements: dict[str, dict[int, tuple[float, float, float]]],
    drive_node_ids: list[np.ndarray],
    read_node_ids: np.ndarray,
    force_n: float,
    contributions: list[ModalContribution],
    mm_mode: int | None = None,
) -> dict:
    """The result block for the JSON: static compliance and stiffness at
    the bridge centre, the M1 mode and frequency used, effective mass,
    MM, plus the per-mode breakdown for cross-checking."""
    compliance_mm_per_n = static_compliance_mm_per_n(static_displacements, read_node_ids, force_n)
    deflection_mm = compliance_mm_per_n * force_n
    stiffness_n_per_mm = 1.0 / compliance_mm_per_n

    drive_deflection_mm = [
        -_patch_mean_z(nodes, static_displacements[f"NDRIVE{d}"]) for d, nodes in enumerate(drive_node_ids, start=1)
    ]

    if mm_mode is not None:
        chosen = next((c for c in contributions if c.mode_number == mm_mode), None)
        if chosen is None:
            raise ValueError(f"mode {mm_mode} is not among the solved elastic modes")
    else:
        if not contributions:
            raise ValueError("no elastic modes to take the M1 frequency from")
        chosen = max(contributions, key=lambda c: c.compliance_mm_per_n)

    omega = 2.0 * math.pi * chosen.frequency_hz
    stiffness_n_per_m = stiffness_n_per_mm * 1000.0
    effective_mass_kg = stiffness_n_per_m / omega**2
    mobility_m_per_s_per_n = 1.0 / math.sqrt(stiffness_n_per_m * effective_mass_kg)

    return {
        "force_n": force_n,
        "bridge_deflection_mm": deflection_mm,
        "compliance_mm_per_kgf": compliance_mm_per_n * KGF_TO_N,
        "stiffness_kgf_per_mm": stiffness_n_per_mm / KGF_TO_N,
        "drive_deflection_mm_per_kgf": [d / force_n * KGF_TO_N for d in drive_deflection_mm],
        "m1_mode": chosen.mode_number,
        "m1_frequency_hz": chosen.frequency_hz,
        "m1_chosen_by": "requested" if mm_mode is not None else "largest share of static compliance",
        "effective_mass_g": effective_mass_kg * 1000.0,
        "mobility_m_per_s_per_n": mobility_m_per_s_per_n,
        "modal_sum_fraction": sum(c.fraction for c in contributions),
        "modes": [
            {
                "mode_number": c.mode_number,
                "frequency_hz": c.frequency_hz,
                "phi_drive": c.phi_drive,
                "phi_read": c.phi_read,
                "compliance_mm_per_kgf": c.compliance_mm_per_n * KGF_TO_N,
                "fraction": c.fraction,
            }
            for c in contributions
        ],
    }
