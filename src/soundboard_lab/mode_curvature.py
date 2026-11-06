"""Curvature (Hessian) field of a mode shape and the bending direction
it implies -- the "which way would a rib have to run" half of the
stiffener-placement picture that the scalar strain-energy map
(`mode_energy.py`) cannot give.

A rib along direction t resists curvature along its own length: its
first-order effect on a mode is proportional to `EI * (t' H t)^2`, with
H the Hessian of the out-of-plane displacement w. So at any point the
best rib direction is the principal direction of H with the largest
|eigenvalue|, independent of the plate's own material. (It is NOT the
gradient of the energy map, which vanishes at the very hot spots.)

H comes from the solver's own displacement field, not from differencing
noisy vertex values: the top surface of a C3D15 prism mesh is made of
6-node quadratic triangles (corners plus mid-edge nodes), so w is an
exact quadratic inside each one and its Hessian is exactly constant per
triangle. Triangle values are averaged (area-weighted) to the vertices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from soundboard_lab.materials import IsotropicMaterial, Material, OrthotropicWoodMaterial
from soundboard_lab.mode_energy import triangle_areas


def triangle_hessians(
    xy: np.ndarray, triangles: np.ndarray, w_corner: np.ndarray, w_mid: np.ndarray
) -> np.ndarray:
    """(T, 3) `[w_xx, w_yy, w_xy]`, constant per triangle, of the
    quadratic interpolant through `w_corner` ((V,), at `xy`) and
    `w_mid` ((T, 3), at the mid-points of edges v1-v2, v2-v3, v3-v1).

    With barycentric coordinates z_i (gradients b_i), the 6-node shape
    functions are z_c (2 z_c - 1) at the corners and 4 z_i z_j at the
    mid-edges, whose Hessians are 4 b_c b_c' and 4 (b_i b_j' + b_j b_i').
    """
    p = xy[triangles]  # (T, 3, 2)
    x, y = p[:, :, 0], p[:, :, 1]
    two_area = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    b = np.empty((len(triangles), 3, 2))
    for i, (j, k) in enumerate(((1, 2), (2, 0), (0, 1))):
        b[:, i, 0] = (y[:, j] - y[:, k]) / two_area
        b[:, i, 1] = (x[:, k] - x[:, j]) / two_area

    wc = w_corner[triangles]  # (T, 3)
    hxx = np.zeros(len(triangles))
    hyy = np.zeros(len(triangles))
    hxy = np.zeros(len(triangles))
    for c in range(3):
        hxx += 4 * wc[:, c] * b[:, c, 0] ** 2
        hyy += 4 * wc[:, c] * b[:, c, 1] ** 2
        hxy += 4 * wc[:, c] * b[:, c, 0] * b[:, c, 1]
    for e, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        hxx += 8 * w_mid[:, e] * b[:, i, 0] * b[:, j, 0]
        hyy += 8 * w_mid[:, e] * b[:, i, 1] * b[:, j, 1]
        hxy += 4 * w_mid[:, e] * (b[:, i, 0] * b[:, j, 1] + b[:, i, 1] * b[:, j, 0])
    return np.column_stack([hxx, hyy, hxy])


def vertex_hessians(xy: np.ndarray, triangles: np.ndarray, tri_hessians: np.ndarray) -> np.ndarray:
    """(V, 3) area-weighted average of the Hessians of the triangles
    around each vertex -- a continuous field from the element-wise
    constant one."""
    areas = triangle_areas(xy, triangles)
    total = np.zeros((len(xy), 3))
    weight = np.zeros(len(xy))
    for corner in range(3):
        np.add.at(total, triangles[:, corner], tri_hessians * areas[:, None])
        np.add.at(weight, triangles[:, corner], areas)
    return total / weight[:, None]


def principal_bending(hessian: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(k1, k2, angle)` for (N, 3) `[w_xx, w_yy, w_xy]` rows: `k1` is
    the principal curvature of larger magnitude, `k2` the other, and
    `angle` (radians in [0, pi)) the direction of k1's axis from +x --
    the direction a rib resists the most curvature along."""
    mean = 0.5 * (hessian[:, 0] + hessian[:, 1])
    radius = np.hypot(0.5 * (hessian[:, 0] - hessian[:, 1]), hessian[:, 2])
    plus, minus = mean + radius, mean - radius
    angle_plus = 0.5 * np.arctan2(2 * hessian[:, 2], hessian[:, 0] - hessian[:, 1])
    plus_wins = np.abs(plus) >= np.abs(minus)
    k1 = np.where(plus_wins, plus, minus)
    k2 = np.where(plus_wins, minus, plus)
    angle = np.where(plus_wins, angle_plus, angle_plus + 0.5 * math.pi) % math.pi
    return k1, k2, angle


def plate_bending_matrix(
    material: Material, orientation: tuple[float, float, float, float, float, float], thickness: float = 1.0
) -> np.ndarray:
    """3x3 plate bending stiffness D (Kirchhoff, plane stress) in the
    global x/y frame, for curvature `(w_xx, w_yy, 2 w_xy)` -- so the
    bending energy density is `0.5 * kappa' D kappa`. The material's
    in-plane axes are CalculiX's 1 (radial) and 2 (longitudinal); the
    1-axis lies along the in-plane part of `orientation[:3]`.
    """
    if isinstance(material, IsotropicMaterial):
        e1 = e2 = material.e
        nu12 = material.nu
        g12 = material.e / (2.0 * (1.0 + material.nu))
        theta = 0.0
    elif isinstance(material, OrthotropicWoodMaterial):
        from soundboard_lab.materials import to_calculix_engineering_constants

        e1, e2, _e3, nu12, _nu13, _nu23, g12, _g13, _g23 = to_calculix_engineering_constants(material)
        theta = math.atan2(orientation[1], orientation[0])
    else:
        raise TypeError(f"Unknown material type: {type(material)!r}")

    nu21 = nu12 * e2 / e1
    denom = 1.0 - nu12 * nu21
    q11, q22, q12, q66 = e1 / denom, e2 / denom, nu12 * e2 / denom, g12
    m, n = math.cos(theta), math.sin(theta)
    m2, n2, mn = m * m, n * n, m * n
    shear_term = q12 + 2.0 * q66
    qbar = np.array(
        [
            [
                q11 * m2 * m2 + 2.0 * shear_term * m2 * n2 + q22 * n2 * n2,
                (q11 + q22 - 4.0 * q66) * m2 * n2 + q12 * (m2 * m2 + n2 * n2),
                (q11 - q12 - 2.0 * q66) * m2 * mn - (q22 - q12 - 2.0 * q66) * mn * n2,
            ],
            [
                0.0,
                q11 * n2 * n2 + 2.0 * shear_term * m2 * n2 + q22 * m2 * m2,
                (q11 - q12 - 2.0 * q66) * mn * n2 - (q22 - q12 - 2.0 * q66) * m2 * mn,
            ],
            [0.0, 0.0, (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m2 * n2 + q66 * (m2 * m2 + n2 * n2)],
        ]
    )
    qbar = np.triu(qbar) + np.triu(qbar, 1).T
    return thickness**3 / 12.0 * qbar


def plate_bending_energy(
    xy: np.ndarray,
    triangles: np.ndarray,
    thickness: np.ndarray,
    tri_hessians: np.ndarray,
    material: Material,
    orientation: tuple[float, float, float, float, float, float],
) -> float:
    """Kirchhoff plate strain energy `0.5 * integral kappa' D kappa dA`
    of a mode from its per-triangle Hessians -- the thin-plate
    prediction of the same quantity ccx reports as `ENER` (see
    `mode_energy.py`); the two agree when the plate really is thin."""
    unit_d = plate_bending_matrix(material, orientation, thickness=1.0)  # already includes the 1/12
    kappa = np.column_stack([tri_hessians[:, 0], tri_hessians[:, 1], 2.0 * tri_hessians[:, 2]])
    density_per_t3 = 0.5 * np.einsum("ti,ij,tj->t", kappa, unit_d, kappa)
    t_tri = thickness[triangles].mean(axis=1)
    return float(np.sum(triangle_areas(xy, triangles) * density_per_t3 * t_tri**3))


@dataclass(frozen=True)
class TickGrid:
    """Where on the footprint direction ticks are sampled: the regular
    grid points inside the mesh, each with its containing triangle and
    barycentric weights. Depends only on the geometry, so it is built
    once per solve and reused for every mode."""

    points: np.ndarray  # (S, 2)
    triangle: np.ndarray  # (S,) index of the containing triangle
    weights: np.ndarray  # (S, 3) barycentric weights in that triangle
    z: np.ndarray  # (S,) interpolated top-surface Z
    spacing: float


def make_tick_grid(xy: np.ndarray, top_z: np.ndarray, triangles: np.ndarray, spacing: float) -> TickGrid:
    origin = xy.min(axis=0) + 0.5 * spacing
    n = np.floor((xy.max(axis=0) - origin) / spacing).astype(int) + 1
    owner = np.full((n[1], n[0]), -1, dtype=np.int64)
    weights = np.zeros((n[1], n[0], 3))
    p = xy[triangles]
    lo_idx = np.ceil((p.min(axis=1) - origin) / spacing).astype(int)
    hi_idx = np.floor((p.max(axis=1) - origin) / spacing).astype(int)
    for t in range(len(triangles)):
        i0, j0 = np.maximum(lo_idx[t], 0)
        i1, j1 = np.minimum(hi_idx[t], n - 1)
        if i0 > i1 or j0 > j1:
            continue
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
        px = origin[0] + ii * spacing
        py = origin[1] + jj * spacing
        a, b, c = p[t]
        det = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        u = ((px - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (py - a[1])) / det
        v = ((b[0] - a[0]) * (py - a[1]) - (px - a[0]) * (b[1] - a[1])) / det
        inside = (u >= 0) & (v >= 0) & (u + v <= 1)
        owner[jj[inside], ii[inside]] = t
        weights[jj[inside], ii[inside]] = np.column_stack([1.0 - u[inside] - v[inside], u[inside], v[inside]])

    jj, ii = np.nonzero(owner >= 0)
    triangle = owner[jj, ii]
    w = weights[jj, ii]
    z = np.einsum("sk,sk->s", w, top_z[triangles[triangle]])
    points = np.column_stack([origin[0] + ii * spacing, origin[1] + jj * spacing])
    return TickGrid(points=points, triangle=triangle, weights=w, z=z, spacing=spacing)


def sample_direction_ticks(
    grid: TickGrid,
    triangles: np.ndarray,
    vertex_hessian: np.ndarray,
    min_fraction: float = 0.1,
    percentile: float = 98.0,
) -> np.ndarray:
    """(S, 6) short line segments `(x0, y0, z0, x1, y1, z1)` at the
    grid points, each along the local principal bending axis (the best
    rib direction) and as long as the local |k1| relative to its
    `percentile` over the grid. Points whose |k1| is below
    `min_fraction` of that scale are dropped, so flat regions stay
    clear. H is interpolated linearly inside each triangle."""
    if len(grid.points) == 0:
        return np.empty((0, 6))
    hess = np.einsum("sk,skc->sc", grid.weights, vertex_hessian[triangles[grid.triangle]])
    k1, _k2, angle = principal_bending(hess)
    scale = float(np.percentile(np.abs(k1), percentile))
    if scale <= 0.0:
        return np.empty((0, 6))
    strength = np.minimum(np.abs(k1) / scale, 1.0)
    keep = strength >= min_fraction
    centre = grid.points[keep]
    half = (0.45 * grid.spacing * strength[keep])[:, None] * np.column_stack(
        [np.cos(angle[keep]), np.sin(angle[keep])]
    )
    zc = grid.z[keep]
    return np.column_stack([centre - half, zc, centre + half, zc])


def default_tick_spacing(xy: np.ndarray, triangles: np.ndarray) -> float:
    """1.5 mean element edge lengths: dense enough to read the field,
    sparse enough that ticks don't overlap."""
    return 1.5 * float(np.sqrt(triangle_areas(xy, triangles).mean() * 2.0))


def build_mode_curvature(
    xy: np.ndarray,
    triangles: np.ndarray,
    thickness: np.ndarray,
    top_corner_node_ids: np.ndarray,
    top_mid_node_ids: np.ndarray,
    displacement_by_node: dict[int, tuple[float, float, float]],
    material: Material,
    orientation: tuple[float, float, float, float, float, float],
    tick_grid: TickGrid,
    reference_energy: float | None = None,
) -> dict:
    """JSON-ready curvature description of one mode: the vertex
    `hessian` (`[w_xx, w_yy, w_xy]` rows, in the mode's own arbitrary
    scale), principal-axis `ticks` (see `sample_direction_ticks`) and,
    given the mode's ccx strain energy as `reference_energy`, the
    `plate_energy_ratio` of the thin-plate bending energy to it -- a
    consistency check on the whole field (near 1 when plate theory
    holds)."""
    w_corner = np.array([displacement_by_node[int(n)][2] for n in top_corner_node_ids])
    w_mid = np.array([[displacement_by_node[int(n)][2] for n in row] for row in top_mid_node_ids])
    tri_h = triangle_hessians(xy, triangles, w_corner, w_mid)
    vertex_h = vertex_hessians(xy, triangles, tri_h)
    ticks = sample_direction_ticks(tick_grid, triangles, vertex_h)
    payload = {
        "hessian": [[float(f"{v:.6g}") for v in row] for row in vertex_h],
        "ticks": np.round(ticks, 3).tolist(),
    }
    if reference_energy:
        payload["plate_energy_ratio"] = (
            plate_bending_energy(xy, triangles, thickness, tri_h, material, orientation) / reference_energy
        )
    return payload
