"""Multi-body assembly solve: the Soundboard (multi-layer C3D15 shell
extrusion, exactly as the single-body solve meshes it) plus every
bonded body (C3D10 tets), joined by CalculiX `*TIE` cards.

Each body's tie is built from the same classification detection shows
the user (see bonding.py): a body skin triangle is a glue face when
all three corners are within `tolerance` of the meshed Soundboard
surface -- the faces counted in the reported contact area. Every node
on those faces (corners and mid-side nodes) becomes the tie's slave
set; the master is the Soundboard's top and bottom element faces, so
bodies glued to either side are covered (V1: Soundboard is always
master -- docs/CLAUDE.md "Boundary & Bonds UI").

Pure Python, no `bpy` imports (see docs/CLAUDE.md, "Engine/UI boundary").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import meshio
import numpy as np
import trimesh

from soundboard_lab.bonding import _TET_FACES, shell_surface
from soundboard_lab.materials import Material, validate_material
from soundboard_lab.mesh import _frequency_step_cards
from soundboard_lab.shell_mesh import (
    ShellMesh,
    classify_boundary_loops,
    extrude_shell_to_prisms,
    material_section_cards,
)

IDENTITY_ORIENTATION = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

# meshio/CalculiX tetra10 mid-side node index for each corner pair.
_TET10_MIDSIDE = {
    frozenset((0, 1)): 4,
    frozenset((1, 2)): 5,
    frozenset((0, 2)): 6,
    frozenset((0, 3)): 7,
    frozenset((1, 3)): 8,
    frozenset((2, 3)): 9,
}

# CalculiX C3D15 face labels: S1 = bottom triangle (nodes 1-2-3),
# S2 = top triangle (nodes 4-5-6).
_WEDGE_BOTTOM_FACE = "S1"
_WEDGE_TOP_FACE = "S2"


@dataclass(frozen=True)
class Terminal:
    """Rim termination for a Body/Lining body (docs/CLAUDE.md "Rim
    lining"), applied to the nodes of its outer wall -- where the sides
    would be glued -- minus any glue-face nodes. `fixed` clamps them,
    `free` adds nothing, `spring` grounds each node in x/y/z with
    `stiffness` (N/mm, the total per direction) shared equally."""

    condition: str  # "fixed" | "free" | "spring"
    stiffness: float = 0.0


TERMINAL_CONDITIONS = ("fixed", "free", "spring")

GRAMS_TO_TONNES = 1e-6


@dataclass(frozen=True)
class PointAttachment:
    """A grounded vertical spring and/or a lumped mass at one location
    (docs/CLAUDE.md "Point attachments": compensator feet, bridge mass,
    test masses). It acts on the nearest body surface -- the Soundboard
    or any bonded body -- spread equally over that surface's nodes
    within `radius` of the nearest surface point (`radius` 0 = the
    single nearest node). The spring's far end is ground, so no
    body-to-body attachment graph is involved."""

    name: str
    position: tuple[float, float, float]
    radius: float = 5.0
    stiffness: float = 0.0  # N/mm, Z only, total over the patch
    mass_g: float = 0.0  # grams, total over the patch
    max_snap_distance: float = 5.0  # mm from `position` to the nearest surface


@dataclass(frozen=True)
class AttachmentPlacement:
    name: str
    body: str  # "Soundboard" or the bonded body's name
    node_count: int
    snap_distance: float  # mm from the attachment's position to the surface


@dataclass(frozen=True)
class AssemblyBody:
    name: str
    mesh: meshio.Mesh  # tetra10, world space
    material: Material
    orientation: tuple[float, float, float, float, float, float] = IDENTITY_ORIENTATION
    terminal: Terminal | None = None


@dataclass(frozen=True)
class AssemblyInpWriteResult:
    element_count: int
    top_layer_node_ids: np.ndarray  # same meaning as ShellInpWriteResult's
    tie_node_counts: dict[str, int]  # slave nodes per body
    terminal_node_counts: dict[str, int] = field(default_factory=dict)  # bodies with a terminal only
    attachments: list[AttachmentPlacement] = field(default_factory=list)


def _skin_faces(mesh: meshio.Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Skin faces of a tetra10 mesh as `(faces, opposite)`: `faces` is
    (n, 6) 0-based node ids (3 corners then 3 mid-sides), `opposite` the
    owning tet's fourth corner (the outward normal points away from it)."""
    tets = next(c.data for c in mesh.cells if c.type == "tetra10")
    faces_local = []
    opposite_local = []
    for i, j, k, o in _TET_FACES:
        mids = [_TET10_MIDSIDE[frozenset((a, b))] for a, b in ((i, j), (j, k), (k, i))]
        faces_local.append(tets[:, [i, j, k] + mids])
        opposite_local.append(tets[:, o])
    faces = np.vstack(faces_local)
    opposite = np.concatenate(opposite_local)
    _, inverse, counts = np.unique(np.sort(faces[:, :3], axis=1), axis=0, return_inverse=True, return_counts=True)
    on_skin = counts[inverse.ravel()] == 1
    return faces[on_skin], opposite[on_skin]


def tie_slave_nodes(mesh: meshio.Mesh, board_surface: trimesh.Trimesh, tolerance: float) -> np.ndarray:
    """0-based indices into `mesh.points` of every node on the body's
    glue faces: skin faces whose three corners lie within `tolerance`
    of `board_surface` (the same rule as `BondDetection.contact_area`)."""
    skin, _ = _skin_faces(mesh)
    corner_ids = np.unique(skin[:, :3])
    distance = np.full(len(mesh.points), np.inf)
    distance[corner_ids] = np.abs(trimesh.proximity.signed_distance(board_surface, mesh.points[corner_ids]))
    glue = skin[(distance[skin[:, :3]] <= tolerance).all(axis=1)]
    return np.unique(glue)


def outer_wall_nodes(mesh: meshio.Mesh, shell: ShellMesh, max_normal_z: float = 0.5, min_alignment: float = 0.5) -> np.ndarray:
    """0-based node ids on a lining's outer wall: skin faces that are
    near-vertical (|n_z| < `max_normal_z`) and face outward -- their
    horizontal normal agrees (dot > `min_alignment`) with the outward
    normal of the nearest segment of the Soundboard's outer perimeter.
    Using the nearest perimeter segment, rather than distance to it,
    keeps this right through the waist and for linings set in from the
    board edge; inner walls and kerfed slopes face the other way."""
    skin, opposite = _skin_faces(mesh)
    p = mesh.points
    a, b, c = p[skin[:, 0]], p[skin[:, 1]], p[skin[:, 2]]
    normal = np.cross(b - a, c - a)
    centroid = (a + b + c) / 3.0
    flip = np.einsum("ij,ij->i", normal, p[opposite] - centroid) > 0
    normal[flip] *= -1
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    candidate = np.abs(normal[:, 2]) < max_normal_z
    if not candidate.any():
        return np.empty(0, dtype=int)

    outer_loop, _holes = classify_boundary_loops(shell.xy, shell.triangles)
    loop = shell.xy[outer_loop]
    seg_a, seg_b = loop, np.roll(loop, -1, axis=0)
    edge = seg_b - seg_a
    signed_area = 0.5 * np.sum(seg_a[:, 0] * seg_b[:, 1] - seg_b[:, 0] * seg_a[:, 1])
    outward = np.stack([edge[:, 1], -edge[:, 0]], axis=1) * (1.0 if signed_area > 0 else -1.0)
    outward /= np.linalg.norm(outward, axis=1, keepdims=True)

    q = centroid[candidate, :2]
    rel = q[:, None, :] - seg_a[None, :, :]
    t = np.clip(np.einsum("fsk,sk->fs", rel, edge) / np.einsum("sk,sk->s", edge, edge), 0.0, 1.0)
    closest = seg_a[None] + t[..., None] * edge[None]
    nearest = np.argmin(np.sum((q[:, None, :] - closest) ** 2, axis=2), axis=1)

    n_xy = normal[candidate, :2]
    n_xy = n_xy / np.linalg.norm(n_xy, axis=1, keepdims=True)
    facing_out = np.einsum("fk,fk->f", n_xy, outward[nearest]) > min_alignment
    return np.unique(skin[candidate][facing_out])


def _id_lines(ids, per_line: int = 10) -> list[str]:
    ids = [str(int(i)) for i in ids]
    return [", ".join(ids[i : i + per_line]) for i in range(0, len(ids), per_line)]


def _real(value: float) -> str:
    """Always-real number text: ccx reads an integer-looking `*SPRING` /
    `*MASS` data line such as `1` as "card without data"."""
    return f"{value:.10E}"


@dataclass(frozen=True)
class _AttachSurface:
    label: str
    faces: np.ndarray  # (k, 6) global 1-based node ids, corners first
    normals: np.ndarray  # (k, 3) unit outward normals


def _oriented_normals(points: np.ndarray, faces: np.ndarray, outward_hint: np.ndarray) -> np.ndarray:
    """Unit normals of `faces` (corner ids index `points`), flipped to
    agree with `outward_hint` (per-face vectors)."""
    a, b, c = points[faces[:, 0]], points[faces[:, 1]], points[faces[:, 2]]
    normal = np.cross(b - a, c - a)
    normal[np.einsum("ij,ij->i", normal, outward_hint) < 0] *= -1
    return normal / np.linalg.norm(normal, axis=1, keepdims=True)


def _board_attach_surfaces(prisms, num_layers: int, n_tris: int) -> list[_AttachSurface]:
    """The Soundboard's top and bottom faces (C3D15 corner + mid-edge
    node columns), as two candidate surfaces."""
    points = np.vstack([np.zeros(3), prisms.nodes])  # 1-based lookup
    top = prisms.elements[(num_layers - 1) * n_tris :][:, [3, 4, 5, 9, 10, 11]]
    bottom = prisms.elements[:n_tris][:, [0, 1, 2, 6, 7, 8]]
    up = np.tile([0.0, 0.0, 1.0], (n_tris, 1))
    return [
        _AttachSurface("Soundboard", top, _oriented_normals(points, top, up)),
        _AttachSurface("Soundboard", bottom, _oriented_normals(points, bottom, -up)),
    ]


def _body_attach_surface(body: AssemblyBody, node_offset: int) -> _AttachSurface:
    skin, opposite = _skin_faces(body.mesh)
    p = body.mesh.points
    centroid = p[skin[:, :3]].mean(axis=1)
    normals = _oriented_normals(p, skin, centroid - p[opposite])
    return _AttachSurface(body.name, skin + node_offset + 1, normals)


def attachment_node_ids(
    attachment: PointAttachment, surfaces: list[_AttachSurface], points: np.ndarray
) -> tuple[np.ndarray, AttachmentPlacement]:
    """Global node ids the attachment acts on, and where it landed.
    `points` is 1-based (row 0 unused). The patch is the nodes of the
    nearest surface's faces that face the same way as the contact face
    (so a patch on the board's top never leaks onto its bottom) within
    `radius` of the contact point; empty or radius 0 -> nearest node."""
    if attachment.radius < 0:
        raise ValueError(f"attachment {attachment.name!r}: radius must not be negative")
    if attachment.stiffness < 0 or attachment.mass_g < 0:
        raise ValueError(f"attachment {attachment.name!r}: stiffness and mass must not be negative")
    if attachment.stiffness == 0 and attachment.mass_g == 0:
        raise ValueError(f"attachment {attachment.name!r} has neither a spring stiffness nor a mass")
    position = np.asarray(attachment.position, dtype=float)
    best = None
    for surface in surfaces:
        tri_points = points[surface.faces[:, :3]]
        tm = trimesh.Trimesh(vertices=tri_points.reshape(-1, 3), faces=np.arange(len(tri_points) * 3).reshape(-1, 3), process=False)
        closest, distance, tri_id = trimesh.proximity.closest_point(tm, position[None])
        if best is None or distance[0] < best[1]:
            best = (surface, float(distance[0]), closest[0], int(tri_id[0]))
    surface, snap, contact, tri_id = best
    if snap > attachment.max_snap_distance:
        raise ValueError(
            f"attachment {attachment.name!r} is {snap:.2f} mm from the nearest body surface "
            f"(limit {attachment.max_snap_distance:g} mm) -- place it on the surface it acts on"
        )
    facing = surface.faces[surface.normals @ surface.normals[tri_id] > 0.5]
    candidates = np.unique(facing)
    distance = np.linalg.norm(points[candidates] - contact, axis=1)
    nodes = candidates[distance <= attachment.radius] if attachment.radius > 0 else candidates[:0]
    if len(nodes) == 0:
        nodes = candidates[[int(np.argmin(distance))]]
    return nodes, AttachmentPlacement(attachment.name, surface.label, int(len(nodes)), snap)


def write_assembly_frequency_inp(
    shell: ShellMesh,
    soundboard_material: Material,
    num_layers: int,
    bodies: list[AssemblyBody],
    tie_tolerance: float,
    inp_path: Path,
    n_frequencies: int = 10,
    fixed_edge: bool = False,
    soundboard_orientation: tuple[float, float, float, float, float, float] = IDENTITY_ORIENTATION,
    attachments: list[PointAttachment] | None = None,
) -> AssemblyInpWriteResult:
    """Write a `*FREQUENCY` input for the Soundboard plus `bodies`, each
    tied to the Soundboard. Raises `ValueError` if a body has no glue
    face within `tie_tolerance` -- every body passed in is meant to be
    bonded, so an untied one would silently float free in the solve.

    `fixed_edge` clamps the Soundboard's outer perimeter (as in
    `write_shell_frequency_inp`); bodies get no supports of their own.
    `attachments` add grounded Z springs (SPRING1) and lumped masses
    (MASS elements) on the nearest surface (see `PointAttachment`).
    """
    if tie_tolerance <= 0:
        raise ValueError(f"tie_tolerance must be positive, got {tie_tolerance}")
    validate_material(soundboard_material)
    for body in bodies:
        validate_material(body.material)

    prisms = extrude_shell_to_prisms(shell, num_layers)
    board_surface = shell_surface(shell)
    n_tris = len(shell.triangles)

    lines = [
        "** Soundboard Lab -- multi-body assembly: C3D15 Soundboard shell",
        "** extrusion plus C3D10 bodies, joined by *TIE (Soundboard = master).",
        "*NODE, NSET=Nall",
    ]
    for i, (x, y, z) in enumerate(prisms.nodes, start=1):
        lines.append(f"{i}, {x:.10g}, {y:.10g}, {z:.10g}")
    lines.append("*ELEMENT, TYPE=C3D15, ELSET=Eboard")
    for i, conn in enumerate(prisms.elements, start=1):
        lines.append(f"{i}, {', '.join(str(n) for n in conn)}")

    node_offset = len(prisms.nodes)
    element_offset = len(prisms.elements)
    body_sections = []
    tie_sets = []
    tie_counts: dict[str, int] = {}
    terminal_sets = []
    attach_surfaces = _board_attach_surfaces(prisms, num_layers, n_tris)
    all_points = [np.zeros((1, 3)), prisms.nodes]
    for b, body in enumerate(bodies, start=1):
        slave = tie_slave_nodes(body.mesh, board_surface, tie_tolerance)
        if len(slave) == 0:
            raise ValueError(f"body {body.name!r} has no glue face within {tie_tolerance} of the Soundboard")
        tie_counts[body.name] = int(len(slave))

        lines.append(f"** Body {b}: {body.name}")
        lines.append("*NODE, NSET=Nall")
        for i, (x, y, z) in enumerate(body.mesh.points, start=node_offset + 1):
            lines.append(f"{i}, {x:.10g}, {y:.10g}, {z:.10g}")
        tets = next(c.data for c in body.mesh.cells if c.type == "tetra10")
        lines.append(f"*ELEMENT, TYPE=C3D10, ELSET=Ebody{b}")
        for i, conn in enumerate(tets, start=element_offset + 1):
            lines.append(f"{i}, {', '.join(str(n + node_offset + 1) for n in conn)}")

        body_sections += material_section_cards(
            body.material, body.orientation, elset=f"Ebody{b}", material_name=f"Mbody{b}", orientation_name=f"Obody{b}"
        )
        tie_sets.append((b, slave + node_offset + 1))
        if body.terminal is not None:
            terminal_sets.append((b, body, _terminal_node_ids(body, shell, slave) + node_offset + 1))
        attach_surfaces.append(_body_attach_surface(body, node_offset))
        all_points.append(body.mesh.points)
        node_offset += len(body.mesh.points)
        element_offset += len(tets)

    lines += material_section_cards(
        soundboard_material, soundboard_orientation, elset="Eboard", material_name="Mboard", orientation_name="Oboard"
    )
    lines += body_sections

    if tie_sets:
        lines.append("*SURFACE, NAME=Sboard, TYPE=ELEMENT")
        top_layer_first = (num_layers - 1) * n_tris + 1
        for t in range(n_tris):
            lines.append(f"{top_layer_first + t}, {_WEDGE_TOP_FACE}")
            lines.append(f"{t + 1}, {_WEDGE_BOTTOM_FACE}")
        for b, slave_ids in tie_sets:
            lines.append(f"*NSET, NSET=Ntie{b}")
            lines += _id_lines(slave_ids)
            lines.append(f"*SURFACE, NAME=Stie{b}, TYPE=NODE")
            lines.append(f"Ntie{b}")
            lines.append(f"*TIE, NAME=Tie{b}, POSITION TOLERANCE={tie_tolerance:.10g}, ADJUST=NO")
            lines.append(f"Stie{b}, Sboard")

    terminal_counts: dict[str, int] = {}
    for b, body, node_ids in terminal_sets:
        terminal_counts[body.name] = int(len(node_ids))
        if body.terminal.condition == "free":
            continue
        lines.append(f"*NSET, NSET=Nterm{b}")
        lines += _id_lines(node_ids)
        if body.terminal.condition == "fixed":
            lines.append("*BOUNDARY")
            lines.append(f"Nterm{b}, 1, 3")
            continue
        per_node = body.terminal.stiffness / len(node_ids)
        for dof in (1, 2, 3):
            lines.append(f"*ELEMENT, TYPE=SPRING1, ELSET=Espring{b}d{dof}")
            for n in node_ids:
                element_offset += 1
                lines.append(f"{element_offset}, {int(n)}")
            lines.append(f"*SPRING, ELSET=Espring{b}d{dof}")
            lines.append(str(dof))
            lines.append(_real(per_node))

    placements: list[AttachmentPlacement] = []
    points = np.vstack(all_points)
    for a, attachment in enumerate(attachments or [], start=1):
        node_ids, placement = attachment_node_ids(attachment, attach_surfaces, points)
        placements.append(placement)
        lines.append(f"** Point attachment {a}: {attachment.name} on {placement.body}, {len(node_ids)} nodes")
        if attachment.stiffness > 0:
            lines.append(f"*ELEMENT, TYPE=SPRING1, ELSET=Eattk{a}")
            for n in node_ids:
                element_offset += 1
                lines.append(f"{element_offset}, {int(n)}")
            lines.append(f"*SPRING, ELSET=Eattk{a}")
            lines.append("3")
            lines.append(_real(attachment.stiffness / len(node_ids)))
        if attachment.mass_g > 0:
            lines.append(f"*ELEMENT, TYPE=MASS, ELSET=Eattm{a}")
            for n in node_ids:
                element_offset += 1
                lines.append(f"{element_offset}, {int(n)}")
            lines.append(f"*MASS, ELSET=Eattm{a}")
            lines.append(_real(attachment.mass_g * GRAMS_TO_TONNES / len(node_ids)))

    if fixed_edge:
        outer_loop, _holes = classify_boundary_loops(shell.xy, shell.triangles)
        fixed = sorted({int(prisms.corner_node_ids[level, v]) for level in range(num_layers + 1) for v in outer_loop})
        lines.append("*NSET, NSET=Nfixed")
        lines += _id_lines(fixed)
        lines.append("*BOUNDARY")
        lines.append("Nfixed, 1, 3")

    lines += _frequency_step_cards(n_frequencies)
    Path(inp_path).write_text("\n".join(lines) + "\n")
    return AssemblyInpWriteResult(
        element_count=element_offset,
        top_layer_node_ids=prisms.corner_node_ids[num_layers, :].copy(),
        tie_node_counts=tie_counts,
        terminal_node_counts=terminal_counts,
        attachments=placements,
    )


def _terminal_node_ids(body: AssemblyBody, shell: ShellMesh, glue_nodes: np.ndarray) -> np.ndarray:
    """Outer-wall nodes for `body.terminal`, minus glue-face nodes (those
    already follow the Soundboard through the tie; constraining them too
    would over-constrain the tie)."""
    terminal = body.terminal
    if terminal.condition not in TERMINAL_CONDITIONS:
        raise ValueError(f"body {body.name!r}: unknown terminal condition {terminal.condition!r}")
    if terminal.condition == "spring" and terminal.stiffness <= 0:
        raise ValueError(f"body {body.name!r}: spring terminal needs a positive stiffness")
    nodes = np.setdiff1d(outer_wall_nodes(body.mesh, shell), glue_nodes)
    if terminal.condition != "free" and len(nodes) == 0:
        raise ValueError(f"body {body.name!r}: no outer wall found for its terminal condition")
    return nodes
