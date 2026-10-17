"""Orthotropic wood material constants and CalculiX card conversion.

Pure Python, no `bpy` imports (see docs/CLAUDE.md, "Engine/UI boundary").

Material constants are entered in the wood-science L/R/T convention
(as published literature reports them), then converted to CalculiX's
1/2/3 index convention and validated for physical stability before any
card gets written -- see docs/CLAUDE.md "known traps": a
non-positive-definite orthotropic stiffness tensor (e.g. from small or
inconsistent Poisson ratios) has caused chaotic, increasingly-wrong
high-mode eigenfrequencies in prior work with other FEA tools.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrthotropicWoodMaterial:
    """Orthotropic material in wood-science L/R/T convention.

    Poisson ratios follow the standard wood-science convention: `nu_lr`
    means "stress applied along L (longitudinal), strain response
    measured along R (radial)" -- the convention published literature
    (e.g. Wood Handbook, guitar-soundboard acoustics papers) reports
    them in. This is deliberately NOT CalculiX's 1/2/3 index
    convention; see `to_calculix_engineering_constants()` for why a
    naive component-for-component copy would be wrong.
    """

    name: str
    density: float  # consistent mesh units, e.g. t/mm^3
    e_longitudinal: float  # E_L, along the grain
    e_radial: float  # E_R
    e_tangential: float  # E_T
    g_lr: float
    g_lt: float
    g_rt: float
    nu_lr: float  # stress in L, response in R
    nu_lt: float  # stress in L, response in T
    nu_rt: float  # stress in R, response in T


def to_calculix_engineering_constants(
    m: OrthotropicWoodMaterial,
) -> tuple[float, float, float, float, float, float, float, float, float]:
    """Convert to CalculiX's `*ELASTIC, TYPE=ENGINEERING CONSTANTS` order:
    `(E1, E2, E3, nu12, nu13, nu23, G12, G13, G23)`.

    CalculiX's local directions 1, 2, 3 are, per docs/CLAUDE.md's
    established convention, X=radial, Y=longitudinal, Z=tangential.
    CalculiX defines `nu_ij` as the strain response in direction j to a
    unit stress in direction i -- the loaded direction is always
    listed FIRST. Wood-science literature conventionally publishes
    nu_LR, nu_LT, nu_RT with the loaded direction first too, but that
    direction order doesn't always match CalculiX's 1->2->3 order once
    mapped onto radial/longitudinal/tangential:

    - nu_12 (CalculiX: load radial[1], response longitudinal[2]) is
      the REVERSE of published nu_LR (load longitudinal, response
      radial). Converting requires the standard orthotropic
      reciprocity relation nu_ij / E_i == nu_ji / E_j:
      nu_12 = nu_LR * E_radial / E_longitudinal.
    - nu_13 (load radial[1], response tangential[3]) already matches
      published nu_RT's direction directly: nu_13 = nu_RT.
    - nu_23 (load longitudinal[2], response tangential[3]) already
      matches published nu_LT directly: nu_23 = nu_LT.

    Shear moduli are plane-based, not direction-ordered (G_XY == G_YX),
    so they map directly by plane: G12 is the radial-longitudinal
    plane (G_LR), G13 is the radial-tangential plane (G_RT), G23 is
    the longitudinal-tangential plane (G_LT).
    """
    e1, e2, e3 = m.e_radial, m.e_longitudinal, m.e_tangential
    nu12 = m.nu_lr * (m.e_radial / m.e_longitudinal)
    nu13 = m.nu_rt
    nu23 = m.nu_lt
    g12, g13, g23 = m.g_lr, m.g_rt, m.g_lt
    return e1, e2, e3, nu12, nu13, nu23, g12, g13, g23


class NonPhysicalMaterialError(ValueError):
    """Raised when orthotropic constants would produce a
    non-positive-definite stiffness tensor -- not solvable, and known
    to produce chaotic/garbage high-mode eigenfrequencies rather than
    a clean failure if run anyway (see module docstring).
    """


def validate_positive_definite(m: OrthotropicWoodMaterial) -> None:
    """Raise `NonPhysicalMaterialError` unless the 3x3 normal-stress
    compliance sub-block is positive definite (all eigenvalues > 0).

    The full 6x6 orthotropic compliance matrix decouples into this 3x3
    normal block plus three independent, diagonal shear terms
    (1/G12, 1/G13, 1/G23); the shear terms can only fail positive-
    definiteness if a modulus is non-positive, checked separately. The
    normal block is built directly from the 6 constants CalculiX
    actually stores (E1, E2, E3, nu12, nu13, nu23) using the standard
    symmetric compliance form (off-diagonal terms -nu_ij/E_i are
    already symmetric by the reciprocity relation, so no separate
    nu21/nu31/nu32 terms are needed).
    """
    e1, e2, e3, nu12, nu13, nu23, g12, g13, g23 = to_calculix_engineering_constants(m)

    for name, value in [("E1", e1), ("E2", e2), ("E3", e3), ("G12", g12), ("G13", g13), ("G23", g23)]:
        if value <= 0:
            raise NonPhysicalMaterialError(f"{m.name}: {name} = {value} must be positive")

    compliance = np.array(
        [
            [1.0 / e1, -nu12 / e1, -nu13 / e1],
            [-nu12 / e1, 1.0 / e2, -nu23 / e2],
            [-nu13 / e1, -nu23 / e2, 1.0 / e3],
        ]
    )
    eigenvalues = np.linalg.eigvalsh(compliance)
    if np.any(eigenvalues <= 0):
        raise NonPhysicalMaterialError(
            f"{m.name}: orthotropic stiffness is not positive definite "
            f"(compliance matrix eigenvalues {eigenvalues.tolist()}) -- "
            "these constants are physically invalid and would produce "
            "unstable/garbage eigenfrequencies rather than a clean solver error"
        )
