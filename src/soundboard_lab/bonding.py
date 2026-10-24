"""Multi-body bonding: contact-footprint detection between
independently-meshed bodies (see docs/CLAUDE.md "Boundary & Bonds UI").

Pure Python, no `bpy` imports (see docs/CLAUDE.md, "Engine/UI boundary").

Scope (decided with the user): role-scoped to the Soundboard only --
every non-Soundboard body (Rib, Bridge Plate, eventually Lining) is
checked against the Soundboard specifically, not against every other
body (no general N:M any-pair contact mapping in V1). This module
only answers "does this one body touch the soundboard, and where" --
it says nothing about bonding two non-Soundboard bodies to each other.

This is a detection step only: it finds candidate contact footprints
for a human to review (and, later, visualize as a color map) -- it
does not itself decide a body IS bonded, and it does not generate any
CalculiX `*TIE`/`*SURFACE` cards. Matches the project's established
"explicit, reviewable, not silently baked into solve" culture (see
the step-checklist / mesh-quality-preview precedents).

`detect_bonds` is the real pipeline: it meshes the Soundboard exactly
as the solver does (shell/extrusion, including its perimeter
simplification) and each other body with the tet mesher, then
classifies every body against the MESHED Soundboard surface -- never
against the authored geometry or the mode-shape display object. The
per-body skin it returns is the same surface the add-on's bonding
proxy objects display, so what is shown is exactly what was checked.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import meshio
import numpy as np
import trimesh

from soundboard_lab.mesh import mesh_surface
from soundboard_lab.repair import repair_stl
from soundboard_lab.shell_mesh import ShellMesh, find_boundary_loops, shell_from_closed_mesh


@dataclass(frozen=True)
class ContactFootprint:
    """Result of checking one non-Soundboard body's mesh against the
    Soundboard's mesh for contact, within `tolerance`.

    `body_contact_vertex_mask`/`body_vertex_distances` are indexed by
    the BODY mesh's own vertex order -- directly usable both as a
    candidate CalculiX `*TIE` slave node set (nodes where the mask is
    True) and as a per-vertex Color Attribute for visualization (the
    distances themselves, not just the boolean mask, so "how close"
    is visible even for non-contact vertices -- important for making
    incomplete/jagged bonding at edges visually obvious, per the
    user's own stated visualization goal).

    `soundboard_contact_face_indices` are indices into the
    SOUNDBOARD mesh's own face array -- the candidate CalculiX `*TIE`
    master face set (master must be face-based, see the real
    CalculiX 2.22 manual's *TIE/*SURFACE semantics, verified before
    the `*TIE` mechanics proof this session, not assumed).
    """

    body_contact_vertex_mask: np.ndarray
    body_vertex_distances: np.ndarray
    soundboard_contact_face_indices: np.ndarray

    @property
    def contact_fraction(self) -> float:
        """Fraction of the body's own vertices classified as in
        contact -- a quick, cheap completeness signal (e.g. "almost
        none of this rib actually touches the board" is an obvious
        red flag), not a substitute for the real visualization.
        """
        if len(self.body_contact_vertex_mask) == 0:
            return 0.0
        return float(self.body_contact_vertex_mask.sum()) / len(self.body_contact_vertex_mask)


def find_contact_footprint(
    body_points: np.ndarray,
    soundboard_mesh: trimesh.Trimesh,
    tolerance: float,
) -> ContactFootprint:
    """For every point in `body_points`, find the nearest point on
    `soundboard_mesh`'s surface (`trimesh.proximity.closest_point`,
    an exact nearest-point-on-triangle query, not a coarse bounding-
    volume approximation) and classify it as "in contact" if that
    distance is within `tolerance`.

    Takes a plain point array for the body side, not a full
    `trimesh.Trimesh` -- confirmed empirically (not assumed) that this
    matters: a body's own raw DESIGN-time STL (e.g. a simple box-
    topology rib with only 8 corner vertices) is far too sparse to
    detect contact along most of a real touching footprint -- it can
    only ever "see" the handful of corner vertices that happen to
    exist, nothing along the middle of a long touching edge/face.
    The real, accurate check needs the MESHED (post-gmsh) node
    positions, which are dense across the whole surface -- see
    `find_contact_footprint_from_meshed_body`. This function itself
    doesn't care where the points came from, by design, so it works
    correctly either way once given the right points.

    `tolerance` must be positive and is the user-controllable epsilon
    (default 0.1mm, per the user's own call -- "easy to see when
    modeling") -- deliberately a parameter here, never hardcoded.

    Both the body points and the soundboard mesh are expected in the
    same (world) coordinate space -- this function does no alignment
    of its own; getting the two bodies' geometry into a shared frame
    is the caller's job (for the addon, this is already true:
    `export_triangulated_stl` bakes `matrix_world` into every
    exported body).
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    _closest_points, distances, triangle_ids = trimesh.proximity.closest_point(soundboard_mesh, body_points)
    contact_mask = distances <= tolerance
    contacted_faces = np.unique(triangle_ids[contact_mask]) if contact_mask.any() else np.array([], dtype=int)

    return ContactFootprint(
        body_contact_vertex_mask=contact_mask,
        body_vertex_distances=distances,
        soundboard_contact_face_indices=contacted_faces,
    )


def find_contact_footprint_from_files(body_stl_path: Path, soundboard_stl_path: Path, tolerance: float) -> ContactFootprint:
    """Coarse/design-time convenience wrapper: uses the body's own raw
    (un-meshed) STL vertices. Fast and useful for early "does this
    even roughly line up" modeling feedback, but NOT what should back
    any real bond decision or `*TIE` card -- see this module's and
    `find_contact_footprint`'s own docstrings for why (a simple
    box-topology body has only its 8 corner vertices to sample from,
    regardless of how long the real touching footprint actually is).
    """
    body_mesh = trimesh.load(str(body_stl_path), force="mesh")
    soundboard_mesh = trimesh.load(str(soundboard_stl_path), force="mesh")
    return find_contact_footprint(body_mesh.vertices, soundboard_mesh, tolerance)


def find_contact_footprint_from_meshed_body(
    body_mesh_points: np.ndarray, soundboard_stl_path: Path, tolerance: float
) -> ContactFootprint:
    """The real, accurate check: `body_mesh_points` should be the
    MESHED (post-gmsh) node positions of the non-Soundboard body
    (e.g. `mesh_surface(...).points`), not its raw design-time STL
    vertices -- dense across the whole surface, so contact is
    correctly detected along the full touching footprint, not just at
    a handful of corner points.
    """
    soundboard_mesh = trimesh.load(str(soundboard_stl_path), force="mesh")
    return find_contact_footprint(body_mesh_points, soundboard_mesh, tolerance)


# ---------------------------------------------------------------------------
# Solver-mesh-based detection (the real pipeline)
# ---------------------------------------------------------------------------


class BondStatus(str, Enum):
    """Per-body classification against the meshed Soundboard.

    CONTACT is a *candidate* bond for review, not an accepted one.
    INTERPENETRATING and GEOMETRY_ERROR are modeling problems the user
    must fix before any tie can be trusted.
    """

    CONTACT = "contact"
    NOT_TOUCHING = "not_touching"
    INTERPENETRATING = "interpenetrating"
    GEOMETRY_ERROR = "geometry_error"


def shell_surface(shell: ShellMesh) -> trimesh.Trimesh:
    """Closed, outward-wound surface of the Soundboard exactly as the
    shell/extrusion solver meshes it: the gmsh footprint triangulation
    at `bottom_z` and `top_z`, joined by side walls along every
    boundary loop (outer perimeter and soundhole).

    This is the bonding reference surface, deliberately NOT the
    authored STL: the solver's perimeter is simplified (see
    `shell_mesh._simplify_rim`), and a bond decided against geometry
    the solver never sees could silently disagree with the real tie.
    """
    n = len(shell.xy)
    vertices = np.vstack([
        np.column_stack([shell.xy, shell.bottom_z]),
        np.column_stack([shell.xy, shell.top_z]),
    ])
    faces = [shell.triangles[:, [0, 2, 1]], shell.triangles + n]
    for loop in find_boundary_loops(shell.triangles):
        a = np.asarray(loop)
        b = np.roll(a, -1)
        faces.append(np.column_stack([a, b, b + n]))
        faces.append(np.column_stack([a, b + n, a + n]))
    surface = trimesh.Trimesh(vertices=vertices, faces=np.vstack(faces), process=True)
    trimesh.repair.fix_normals(surface)
    if not surface.is_watertight or surface.volume <= 0:
        raise ValueError("meshed Soundboard surface is not a closed, outward-facing solid")
    return surface


_TET_FACES = ((0, 2, 1, 3), (0, 1, 3, 2), (1, 2, 3, 0), (0, 3, 2, 1))  # (face corners..., opposite corner)


def tet_boundary_skin(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Outer triangular skin of a `tetra10` volume mesh, as compact
    `(points, triangles)` with outward winding -- corner nodes only
    (mid-side nodes lie on the same flat faces, so they add nothing
    to a contact or display surface).

    A face is on the skin when exactly one tet uses it. Winding is
    oriented away from each tet's opposite corner, so it is correct
    regardless of gmsh's own element orientation.
    """
    tets = next(c.data for c in mesh.cells if c.type == "tetra10")[:, :4]
    candidates = []
    opposite = []
    for i, j, k, o in _TET_FACES:
        candidates.append(tets[:, [i, j, k]])
        opposite.append(tets[:, o])
    candidates = np.vstack(candidates)
    opposite = np.concatenate(opposite)

    _, inverse, counts = np.unique(np.sort(candidates, axis=1), axis=0, return_inverse=True, return_counts=True)
    on_skin = counts[inverse.ravel()] == 1
    faces = candidates[on_skin]
    opposite = opposite[on_skin]

    p = mesh.points
    normals = np.cross(p[faces[:, 1]] - p[faces[:, 0]], p[faces[:, 2]] - p[faces[:, 0]])
    inward = np.einsum("ij,ij->i", normals, p[opposite] - p[faces[:, 0]]) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]

    used = np.unique(faces)
    remap = np.full(len(p), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return p[used].copy(), remap[faces]


@dataclass(frozen=True)
class BondDetection:
    """One body's skin classified against the meshed Soundboard.

    `points`/`triangles` are the body's meshed skin (what its bonding
    proxy displays). `signed_distances` is per skin vertex, positive
    INSIDE the Soundboard (trimesh's convention): contact is
    `|d| <= tolerance`, penetration is `d > tolerance`.
    `soundboard_penetration` is the deepest Soundboard surface vertex
    inside this body -- catches a thin Soundboard edge poking into a
    body between the body's own skin nodes, which the per-vertex body
    check alone can miss.
    """

    status: BondStatus
    tolerance: float
    points: np.ndarray
    triangles: np.ndarray
    signed_distances: np.ndarray
    soundboard_penetration: float

    @property
    def contact_vertex_mask(self) -> np.ndarray:
        return np.abs(self.signed_distances) <= self.tolerance

    @property
    def penetrating_vertex_mask(self) -> np.ndarray:
        return self.signed_distances > self.tolerance

    @property
    def contact_area(self) -> float:
        """Area of skin triangles with all three corners in contact --
        an area, not a vertex fraction, so it's comparable against the
        body's intended glue footprint and independent of mesh density.
        """
        in_contact = self.contact_vertex_mask[self.triangles].all(axis=1)
        if not in_contact.any():
            return 0.0
        return float(trimesh.triangles.area(self.points[self.triangles[in_contact]]).sum())

    @property
    def nearest_distance(self) -> float:
        """Smallest unsigned distance from the skin to the Soundboard --
        tells a NOT_TOUCHING body how far off it is."""
        return float(np.abs(self.signed_distances).min())

    @property
    def max_penetration(self) -> float:
        return max(float(self.signed_distances.max()), self.soundboard_penetration, 0.0)


def classify_bond(
    points: np.ndarray, triangles: np.ndarray, soundboard_surface: trimesh.Trimesh, tolerance: float
) -> BondDetection:
    """Classify one body's closed skin against the Soundboard surface.

    Both inputs must share the same (world) frame and be closed solids.
    Interpenetration beyond `tolerance` wins over contact: overlapping
    volumes are a modeling error, and any footprint found there is
    not a trustworthy glue face.
    """
    if tolerance <= 0:
        raise ValueError(f"tolerance must be positive, got {tolerance}")

    signed = trimesh.proximity.signed_distance(soundboard_surface, points)
    body = trimesh.Trimesh(vertices=points, faces=triangles, process=False)
    board_inside_body = trimesh.proximity.signed_distance(body, soundboard_surface.vertices)
    soundboard_penetration = max(float(board_inside_body.max()), 0.0)

    if signed.max() > tolerance or soundboard_penetration > tolerance:
        status = BondStatus.INTERPENETRATING
    elif (np.abs(signed) <= tolerance).any():
        status = BondStatus.CONTACT
    else:
        status = BondStatus.NOT_TOUCHING

    return BondDetection(
        status=status,
        tolerance=tolerance,
        points=points,
        triangles=triangles,
        signed_distances=signed,
        soundboard_penetration=soundboard_penetration,
    )


@dataclass(frozen=True)
class BodyInput:
    """A non-Soundboard body to check: its world-space STL and its own
    tet element size (per-object, matching the add-on's per-object
    `mesh_element_size`)."""

    stl_path: Path
    element_size: float


@dataclass(frozen=True)
class BodyBondResult:
    """`detection` is None exactly when `status` is GEOMETRY_ERROR;
    `message` then says why (repair or meshing failure)."""

    name: str
    status: BondStatus
    detection: BondDetection | None
    message: str = ""


@dataclass(frozen=True)
class BondDetectionReport:
    soundboard_surface: trimesh.Trimesh
    bodies: dict[str, BodyBondResult]


def detect_bonds(
    soundboard_stl: Path,
    bodies: Mapping[str, BodyInput],
    soundboard_element_size: float,
    tolerance: float,
    work_dir: Path | None = None,
) -> BondDetectionReport:
    """Mesh the Soundboard (shell/extrusion, as solved) and every body
    (tet), then classify each body against the meshed Soundboard.

    Every STL goes through `repair_stl` first, same as the solve
    pipeline. A body that fails repair or meshing is reported as
    GEOMETRY_ERROR with the reason rather than aborting the other
    bodies' results; a Soundboard failure does raise, since nothing
    can be checked without it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(work_dir) if work_dir is not None else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)

        repaired_board = work / "soundboard_repaired.stl"
        repair_stl(Path(soundboard_stl), repaired_board)
        board = shell_surface(shell_from_closed_mesh(repaired_board, element_size=soundboard_element_size))

        results: dict[str, BodyBondResult] = {}
        for index, (name, body) in enumerate(bodies.items()):
            try:
                repaired = work / f"body_{index}_repaired.stl"
                repair_stl(Path(body.stl_path), repaired)
                points, triangles = tet_boundary_skin(mesh_surface(repaired, element_size=body.element_size))
            except Exception as exc:
                results[name] = BodyBondResult(name, BondStatus.GEOMETRY_ERROR, None, f"{type(exc).__name__}: {exc}")
                continue
            detection = classify_bond(points, triangles, board, tolerance)
            results[name] = BodyBondResult(name, detection.status, detection)

    return BondDetectionReport(soundboard_surface=board, bodies=results)
