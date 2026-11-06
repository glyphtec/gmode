"""Modal strain-energy maps: where each eigenmode stores its bending
energy, the places a stiffener (or a mass) can actually change that mode.

The Rayleigh quotient of a mode is `omega^2 = (phi' K phi) / (phi' M phi)`,
so a rib's first-order effect on it is its own stiffness contribution
(the local strain energy it adds) minus its added mass term. Both only
matter where the mode has energy density. ccx's `ENER` output (energy
per volume, extrapolated to the nodes) is integrated here through the
plate thickness into energy per plan area, then normalised so each
mode's map integrates to 1 -- an eigenmode's absolute scale is
arbitrary, but this "fraction of the mode's energy per mm^2" is
physically grounded and comparable across modes.
"""

from __future__ import annotations

import numpy as np


def energy_per_area(
    corner_node_ids: np.ndarray, thickness: np.ndarray, energy_density_by_node: dict[int, float]
) -> np.ndarray:
    """(V,) energy per unit plan area at every shell vertex: the
    per-node `ENER` of one mode, trapezoid-integrated through the
    thickness over the extrusion levels of `corner_node_ids`
    ((num_layers + 1, V), evenly spaced in Z). Nodal extrapolation can
    leave tiny negative values where the true density is ~0; those are
    clamped to 0."""
    n_levels = corner_node_ids.shape[0]
    density = np.array(
        [[energy_density_by_node[int(node)] for node in row] for row in corner_node_ids], dtype=np.float64
    )
    density = np.maximum(density, 0.0)
    weights = np.ones(n_levels)
    weights[[0, -1]] = 0.5
    return np.asarray(thickness, dtype=np.float64) / (n_levels - 1) * (weights[:, None] * density).sum(axis=0)


def triangle_areas(xy: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    a, b, c = (xy[triangles[:, i]] for i in range(3))
    ab, ac = b - a, c - a
    return 0.5 * np.abs(ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def integrate_over_area(xy: np.ndarray, triangles: np.ndarray, values: np.ndarray) -> float:
    """Integral of a per-vertex field over the footprint (linear
    interpolation inside each triangle)."""
    return float(np.sum(triangle_areas(xy, triangles) * values[triangles].mean(axis=1)))


def normalize_to_unit_energy(
    xy: np.ndarray, triangles: np.ndarray, energy_area: np.ndarray
) -> tuple[np.ndarray, float]:
    """`(map, total)`: `map` is `energy_area / total`, so it integrates
    to 1 over the footprint; `total` is the mode's strain energy in the
    solved plate (the Soundboard body only), returned so callers can
    check it against `omega^2 / 2` for a mass-normalised mode."""
    total = integrate_over_area(xy, triangles, energy_area)
    if total <= 0.0:
        raise ValueError("mode has no strain energy on the footprint")
    return energy_area / total, total


def display_scale(values: np.ndarray, percentile: float = 98.0) -> float:
    """Value that maps to the top of the colour ramp (and the last
    contour). The clamped rim concentrates energy (its peak is 20-50x
    the median in a fixed-edge solve), so scaling to the max would
    flatten the whole interior, where ribs actually go; a high
    percentile keeps the interior readable and lets the few rim
    peaks clip."""
    return float(np.percentile(values, percentile))


CONTOUR_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def build_mode_energy(
    xy: np.ndarray,
    top_z: np.ndarray,
    triangles: np.ndarray,
    corner_node_ids: np.ndarray,
    thickness: np.ndarray,
    energy_density_by_node: dict[int, float],
) -> dict | None:
    """JSON-ready energy map of one mode, or None if it has no strain
    energy: `energy_per_area` (unit-integral, per shell vertex),
    `total_energy`, `display_scale` and `contours` -- one
    `{"fraction", "segments"}` per `CONTOUR_FRACTIONS` entry (fractions
    of `display_scale`), with segments as `[x0, y0, z0, x1, y1, z1]`
    on the top surface."""
    density = energy_per_area(corner_node_ids, thickness, energy_density_by_node)
    try:
        normalized, total = normalize_to_unit_energy(xy, triangles, density)
    except ValueError:
        return None
    scale = display_scale(normalized)
    points = np.column_stack([xy, top_z])
    segments = contour_segments(points, triangles, normalized, [f * scale for f in CONTOUR_FRACTIONS])
    return {
        "energy_per_area": normalized.tolist(),
        "total_energy": total,
        "display_scale": scale,
        "contours": [
            {"fraction": f, "segments": np.round(s, 3).tolist()} for f, s in zip(CONTOUR_FRACTIONS, segments)
        ],
    }


def contour_segments(
    points: np.ndarray, triangles: np.ndarray, values: np.ndarray, levels: list[float]
) -> list[np.ndarray]:
    """Iso-lines of a per-vertex field by marching triangles: for each
    level an (S, 2d) array of segments `(p0..., p1...)` (`points` is
    (V, d), 2-D or 3-D), one per triangle the level crosses (a
    triangle is crossed by 0 or 2 edges).
    """
    tri_points = points[triangles]  # (T, 3, d)
    tri_v = values[triangles]  # (T, 3)
    edges = ((0, 1), (1, 2), (2, 0))
    result = []
    for level in levels:
        above = tri_v >= level
        crossing = []
        crossing_points = []
        for i, j in edges:
            crossing.append(above[:, i] != above[:, j])
            vi, vj = tri_v[:, i], tri_v[:, j]
            denom = np.where(vj != vi, vj - vi, 1.0)
            t = np.clip((level - vi) / denom, 0.0, 1.0)[:, None]
            crossing_points.append(tri_points[:, i] + t * (tri_points[:, j] - tri_points[:, i]))
        segments = []
        for e0, e1 in ((0, 1), (0, 2), (1, 2)):
            both = crossing[e0] & crossing[e1]
            segments.append(np.hstack([crossing_points[e0][both], crossing_points[e1][both]]))
        result.append(np.vstack(segments))
    return result
