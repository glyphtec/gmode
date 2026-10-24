"""CLI entry point for running an analysis from outside the engine's
own Python process -- this is what the Blender addon invokes as a
subprocess (per docs/CLAUDE.md's engine/UI boundary rule: the engine
has zero `bpy` imports, and gmsh/meshio can't be installed into
Blender's bundled Python, so the addon launches this venv's own
Python running this module rather than importing the engine in-process).

Usage:
    python -m soundboard_lab.cli --stl PATH --ccx-exe PATH --output-json PATH
        [--element-size FLOAT] [--n-frequencies INT] [--thickness-hint FLOAT]
        [--mesh-type tet|shell] [--boundary-condition free_free|fixed_edge|body_lining]
        [--num-layers INT] [--status-json PATH] [--bodies-json PATH] [--attachments-json PATH]

Writes a JSON result (or error) to --output-json; the addon polls for
that file's existence rather than parsing subprocess stdout, so the
interface between the two stays simple and stable.

`--status-json`, if given, is updated at each pipeline-stage
transition with a step checklist (pending/running/done/error per
named step, not a numeric percentage -- ccx itself is an opaque
subprocess call with no fine-grained progress available, so a real
progress bar would just be fake; a step checklist with a clear
per-step error message is both honest and actually useful).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from soundboard_lab.assembly import (
    IDENTITY_ORIENTATION,
    AssemblyBody,
    PointAttachment,
    Terminal,
    write_assembly_frequency_inp,
)
from soundboard_lab.materials import (
    SPRUCE_PLACEHOLDER,
    Material,
    NonPhysicalMaterialError,
    material_from_dict,
    validate_material,
)
from soundboard_lab.mesh import mesh_surface, write_frequency_inp_orthotropic
from soundboard_lab.repair import MeshRepairError, repair_stl
from soundboard_lab.shell_mesh import mode_shape_z_displacement, shell_from_closed_mesh, write_shell_frequency_inp
from soundboard_lab.solver import CcxError, parse_eigenfrequencies, parse_frd_mode_displacements, run_ccx

MIN_THROUGH_THICKNESS_LAYERS = 2

SHELL_PIPELINE_STEPS = [
    ("repair", "Repair mesh"),
    ("mesh", "Build shell mesh"),
    ("write_input", "Write solver input"),
    ("solve", "Solve (CalculiX)"),
    ("parse_results", "Parse results"),
]


class StepTracker:
    """Writes a process-step checklist to `status_path` as the
    pipeline progresses. Not a numeric progress bar -- see module
    docstring for why. Each step transitions pending -> running ->
    done (with an optional one-line summary) or error (with the
    actual failure message), written after every transition so the
    addon's poll loop always sees the latest state, including which
    specific step failed and why if something goes sideways.

    Writing status must never itself crash the real pipeline --
    wrapped in a bare except for that reason, matching the same
    "non-fatal diagnostic helper" discipline used elsewhere (e.g. the
    addon's `_tag_ui_redraw`).
    """

    def __init__(self, status_path: Path | None, steps: list[tuple[str, str]]):
        self.status_path = status_path
        self.order = [step_id for step_id, _ in steps]
        self.steps = {
            step_id: {"id": step_id, "label": label, "status": "pending", "message": ""} for step_id, label in steps
        }
        self._write()

    def _write(self) -> None:
        if self.status_path is None:
            return
        try:
            payload = {"steps": [self.steps[step_id] for step_id in self.order]}
            self.status_path.write_text(json.dumps(payload))
        except Exception:
            pass

    def start(self, step_id: str) -> None:
        self.steps[step_id]["status"] = "running"
        self._write()

    def done(self, step_id: str, message: str = "") -> None:
        self.steps[step_id]["status"] = "done"
        self.steps[step_id]["message"] = message
        self._write()

    def fail(self, step_id: str, message: str) -> None:
        self.steps[step_id]["status"] = "error"
        self.steps[step_id]["message"] = message
        self._write()



def check_through_thickness_layers(element_size: float, thickness_hint: float) -> None:
    """Reactive sanity check for the "known trap" (docs/CLAUDE.md:
    enforce >=2-3 element layers through thickness, not one global
    size) -- this does not yet implement a proper spatially-varying
    gmsh size field (deferred, bigger piece of work); it just refuses
    to proceed if the chosen uniform element_size obviously can't give
    enough layers across a stated thickness, rather than silently
    producing an over-stiff mesh. Checked before meshing (cheap, using
    just the two input numbers) so a doomed run fails fast.
    """
    estimated_layers = thickness_hint / element_size
    if estimated_layers < MIN_THROUGH_THICKNESS_LAYERS:
        raise ValueError(
            f"element_size={element_size:.4g} gives only ~{estimated_layers:.1f} "
            f"layers through the stated thickness_hint={thickness_hint:.4g} "
            f"(need >= {MIN_THROUGH_THICKNESS_LAYERS}): linear-locking-prone, "
            "systematically too-stiff bending frequencies -- reduce element_size "
            "or raise thickness_hint if it was wrong"
        )


def run_analysis(
    stl_path: Path,
    ccx_exe: Path,
    output_json: Path,
    element_size: float,
    n_frequencies: int,
    thickness_hint: float | None,
    material: Material = SPRUCE_PLACEHOLDER,
    orientation: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
) -> None:
    validate_material(material)

    if thickness_hint is not None:
        check_through_thickness_layers(element_size, thickness_hint)

    work_dir = output_json.parent
    repaired_stl = work_dir / "repaired.stl"
    repair_report = repair_stl(stl_path, repaired_stl)

    mesh = mesh_surface(repaired_stl, element_size=element_size)
    tetra_count = next(c.data.shape[0] for c in mesh.cells if c.type == "tetra10")

    inp_path = work_dir / "analysis.inp"
    write_frequency_inp_orthotropic(mesh, material, inp_path, n_frequencies=n_frequencies, orientation=orientation)

    run_ccx(inp_path, ccx_exe)
    modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))

    result = {
        "status": "ok",
        "material_name": material.name,
        "tetra_count": tetra_count,
        "modes": [{"mode_number": m.mode_number, "frequency_hz": m.frequency_hz} for m in modes],
        "mesh_repair": {
            "was_already_valid": repair_report.was_already_valid,
            "problems_found": repair_report.problems_found,
            "components_removed": repair_report.components_removed,
        },
    }
    output_json.write_text(json.dumps(result, indent=2))


def run_shell_analysis(
    stl_path: Path,
    ccx_exe: Path,
    output_json: Path,
    element_size: float,
    n_frequencies: int,
    num_layers: int,
    fixed_edge: bool,
    status_json: Path | None = None,
    material: Material = SPRUCE_PLACEHOLDER,
    orientation: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    bonded_bodies: dict | None = None,
    attachments: list[PointAttachment] | None = None,
) -> None:
    """Shell/extrusion-mesh analysis path (`shell_mesh.py`) -- the
    thin/flat-aware alternative to `run_analysis`'s generic tet path,
    for role=Soundboard objects.

    `bonded_bodies`, if given, is `{"tolerance": float, "bodies": [{"name",
    "stl", "element_size", "material" (dict or null), "orientation"
    (6 floats or null), "terminal" (optional {"condition":
    fixed|free|spring, "stiffness"})}]}`: each body is repaired, tet-meshed
    and tied to the Soundboard (`assembly.write_assembly_frequency_inp`);
    a body with a terminal is the rim lining (`assembly.Terminal`). None
    or an empty list is the plain single-body solve.

    `attachments` (`assembly.PointAttachment`: grounded Z springs and
    lumped masses) also route through the assembly writer; the result
    lists where each landed under `point_attachments`.

    `material`/`orientation` default to `SPRUCE_PLACEHOLDER`/identity
    (preserving every existing caller/test that doesn't pass them),
    but the addon now resolves the object's actual assigned Material
    ID-block + per-object grain rotation and passes them in via
    `main()`'s `--material-json`/`--orientation` flags -- no longer
    hardcoded end-to-end.

    Unlike `run_analysis`, also writes the solved shell's own footprint
    (`xy`/`triangles`) and every mode's per-node Z-displacement into
    the result JSON -- needed for the addon's mode-shape heat-map
    preview, which `run_analysis`'s tet path has no equivalent of.
    """
    tracker = StepTracker(status_json, SHELL_PIPELINE_STEPS)
    validate_material(material)
    work_dir = output_json.parent

    try:
        tracker.start("repair")
        repaired_stl = work_dir / "repaired.stl"
        repair_report = repair_stl(stl_path, repaired_stl)
        summary = "already valid" if repair_report.was_already_valid else "; ".join(repair_report.problems_found)
        tracker.done("repair", summary)
    except Exception as exc:
        tracker.fail("repair", str(exc))
        raise

    try:
        tracker.start("mesh")
        shell = shell_from_closed_mesh(repaired_stl, element_size=element_size)
        assembly_bodies = _mesh_bonded_bodies(bonded_bodies, work_dir)
        summary = f"{len(shell.triangles)} triangles"
        if assembly_bodies:
            summary += f" + {len(assembly_bodies)} bonded bodies"
        tracker.done("mesh", summary)
    except Exception as exc:
        tracker.fail("mesh", str(exc))
        raise

    try:
        tracker.start("write_input")
        inp_path = work_dir / "analysis.inp"
        if assembly_bodies or attachments:
            write_result = write_assembly_frequency_inp(
                shell,
                material,
                num_layers=num_layers,
                bodies=assembly_bodies,
                tie_tolerance=float((bonded_bodies or {}).get("tolerance", 0.1)),
                inp_path=inp_path,
                n_frequencies=n_frequencies,
                fixed_edge=fixed_edge,
                soundboard_orientation=orientation,
                attachments=attachments,
            )
        else:
            write_result = write_shell_frequency_inp(
                shell,
                material,
                num_layers=num_layers,
                inp_path=inp_path,
                n_frequencies=n_frequencies,
                fixed_edge=fixed_edge,
                orientation=orientation,
            )
        message = f"{write_result.element_count} elements"
        tie_counts = getattr(write_result, "tie_node_counts", {})
        if tie_counts:
            message += f", {len(tie_counts)} ties"
        if attachments:
            message += f", {len(attachments)} point attachments"
        tracker.done("write_input", message)
    except Exception as exc:
        tracker.fail("write_input", str(exc))
        raise

    try:
        tracker.start("solve")
        run_ccx(inp_path, ccx_exe)
        tracker.done("solve")
    except Exception as exc:
        tracker.fail("solve", str(exc))
        raise

    try:
        tracker.start("parse_results")
        modes = parse_eigenfrequencies(inp_path.with_suffix(".dat"))
        mode_displacements = parse_frd_mode_displacements(inp_path.with_suffix(".frd"))
        mode_shapes = {
            str(m.mode_number): mode_shape_z_displacement(
                write_result.top_layer_node_ids, mode_displacements[m.mode_number]
            ).tolist()
            for m in modes
        }
        tracker.done("parse_results")
    except Exception as exc:
        tracker.fail("parse_results", str(exc))
        raise

    result = {
        "status": "ok",
        "material_name": material.name,
        "triangle_count": len(shell.triangles),
        "modes": [{"mode_number": m.mode_number, "frequency_hz": m.frequency_hz} for m in modes],
        "mesh_repair": {
            "was_already_valid": repair_report.was_already_valid,
            "problems_found": repair_report.problems_found,
            "components_removed": repair_report.components_removed,
        },
        "mesh_shape": {
            "xy": shell.xy.tolist(),
            "top_z": shell.top_z.tolist(),
            "triangles": shell.triangles.tolist(),
        },
        "mode_shapes": mode_shapes,
        "bonded_bodies": [
            {
                "name": name,
                "tie_nodes": count,
                **(
                    {"terminal_nodes": write_result.terminal_node_counts[name]}
                    if name in getattr(write_result, "terminal_node_counts", {})
                    else {}
                ),
            }
            for name, count in getattr(write_result, "tie_node_counts", {}).items()
        ],
        "point_attachments": [
            {"name": p.name, "body": p.body, "nodes": p.node_count, "snap_distance": p.snap_distance}
            for p in getattr(write_result, "attachments", [])
        ],
    }
    output_json.write_text(json.dumps(result))


def _mesh_bonded_bodies(bonded_bodies: dict | None, work_dir: Path) -> list[AssemblyBody]:
    if not bonded_bodies or not bonded_bodies.get("bodies"):
        return []
    meshed = []
    for index, body in enumerate(bonded_bodies["bodies"]):
        name = body["name"]
        try:
            repaired = work_dir / f"body_{index}_repaired.stl"
            repair_stl(Path(body["stl"]), repaired)
            mesh = mesh_surface(repaired, element_size=float(body["element_size"]))
        except Exception as exc:
            hint = ""
            if "overlapping facets" in str(exc):
                hint = " -- its surface intersects itself (e.g. an unmerged mirror seam; try Merge by Distance)"
            raise ValueError(f"bonded body {name!r}: {type(exc).__name__}: {exc}{hint}") from exc
        body_material = material_from_dict(body["material"]) if body.get("material") else SPRUCE_PLACEHOLDER
        body_orientation = tuple(body["orientation"]) if body.get("orientation") else IDENTITY_ORIENTATION
        terminal = None
        if body.get("terminal"):
            terminal = Terminal(str(body["terminal"]["condition"]), float(body["terminal"].get("stiffness", 0.0)))
        meshed.append(AssemblyBody(name, mesh, body_material, body_orientation, terminal))
    return meshed


def load_attachments(spec: dict) -> list[PointAttachment]:
    """`{"attachments": [{name, position: [x,y,z] mm, radius mm,
    stiffness N/mm (Z, total), mass_g}]}` -> `PointAttachment`s."""
    attachments = []
    for item in spec.get("attachments", []):
        position = tuple(float(v) for v in item["position"])
        if len(position) != 3:
            raise ValueError(f"attachment {item.get('name')!r}: position needs 3 coordinates")
        attachments.append(
            PointAttachment(
                name=str(item["name"]),
                position=position,
                radius=float(item.get("radius", 5.0)),
                stiffness=float(item.get("stiffness", 0.0)),
                mass_g=float(item.get("mass_g", 0.0)),
            )
        )
    return attachments


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", required=True, type=Path)
    parser.add_argument("--ccx-exe", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--element-size", type=float, default=15.0)
    parser.add_argument("--n-frequencies", type=int, default=10)
    parser.add_argument("--thickness-hint", type=float, default=None)
    parser.add_argument("--mesh-type", choices=["tet", "shell"], default="tet")
    parser.add_argument(
        "--boundary-condition",
        choices=["free_free", "fixed_edge", "body_lining"],
        default="free_free",
        help="body_lining: board edge left free; the rim is held by the lining body's own terminal",
    )
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--status-json", type=Path, default=None)
    parser.add_argument(
        "--bodies-json",
        type=Path,
        default=None,
        help=(
            "Shell path only: JSON {tolerance, bodies:[{name, stl, element_size, material, orientation}]} "
            "of bodies to tie to the Soundboard"
        ),
    )
    parser.add_argument(
        "--attachments-json",
        type=Path,
        default=None,
        help=(
            "Shell path only: JSON {attachments:[{name, position:[x,y,z], radius, stiffness (N/mm, Z), mass_g}]} "
            "of grounded springs / point masses on the nearest body surface"
        ),
    )
    parser.add_argument(
        "--material-json",
        type=Path,
        default=None,
        help="Path to a JSON material descriptor (see materials.material_to_dict/_from_dict); defaults to SPRUCE_PLACEHOLDER",
    )
    parser.add_argument(
        "--orientation",
        type=str,
        default=None,
        help=(
            "Six comma-separated floats 'a1,a2,a3,b1,b2,b3' for the *ORIENTATION card "
            "(local-1 axis vector, then a vector in the local 1-2 plane); defaults to identity"
        ),
    )
    args = parser.parse_args(argv)

    try:
        material = SPRUCE_PLACEHOLDER
        if args.material_json is not None:
            material = material_from_dict(json.loads(args.material_json.read_text()))
        orientation = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
        if args.orientation is not None:
            parts = [float(x) for x in args.orientation.split(",")]
            if len(parts) != 6:
                raise ValueError(f"--orientation must be 6 comma-separated floats, got {args.orientation!r}")
            orientation = tuple(parts)
        bonded_bodies = None
        if args.bodies_json is not None:
            if args.mesh_type != "shell":
                raise ValueError("--bodies-json requires --mesh-type shell")
            bonded_bodies = json.loads(args.bodies_json.read_text())
        attachments = None
        if args.attachments_json is not None:
            if args.mesh_type != "shell":
                raise ValueError("--attachments-json requires --mesh-type shell")
            attachments = load_attachments(json.loads(args.attachments_json.read_text()))
        if args.boundary_condition == "body_lining":
            bodies = (bonded_bodies or {}).get("bodies", [])
            if args.mesh_type != "shell" or not any(b.get("terminal") for b in bodies):
                raise ValueError(
                    "--boundary-condition body_lining needs --mesh-type shell and a --bodies-json body with a terminal"
                )

        if args.mesh_type == "shell":
            run_shell_analysis(
                stl_path=args.stl,
                ccx_exe=args.ccx_exe,
                output_json=args.output_json,
                element_size=args.element_size,
                n_frequencies=args.n_frequencies,
                num_layers=args.num_layers,
                fixed_edge=(args.boundary_condition == "fixed_edge"),
                status_json=args.status_json,
                material=material,
                orientation=orientation,
                bonded_bodies=bonded_bodies,
                attachments=attachments,
            )
        else:
            run_analysis(
                stl_path=args.stl,
                ccx_exe=args.ccx_exe,
                output_json=args.output_json,
                element_size=args.element_size,
                n_frequencies=args.n_frequencies,
                thickness_hint=args.thickness_hint,
                material=material,
                orientation=orientation,
            )
        return 0
    except (NonPhysicalMaterialError, CcxError, MeshRepairError, ValueError, FileNotFoundError) as exc:
        # Known, expected failure modes -- write a clean error result
        # rather than letting the addon see only a nonzero exit code.
        args.output_json.write_text(
            json.dumps({"status": "error", "error_type": type(exc).__name__, "message": str(exc)}, indent=2)
        )
        return 1
    except Exception as exc:  # noqa: BLE001 -- genuinely unexpected; still must not vanish silently
        args.output_json.write_text(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
                indent=2,
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
