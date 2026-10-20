"""CLI entry point for running an analysis from outside the engine's
own Python process -- this is what the Blender addon invokes as a
subprocess (per docs/CLAUDE.md's engine/UI boundary rule: the engine
has zero `bpy` imports, and gmsh/meshio can't be installed into
Blender's bundled Python, so the addon launches this venv's own
Python running this module rather than importing the engine in-process).

Usage:
    python -m soundboard_lab.cli --stl PATH --ccx-exe PATH --output-json PATH
        [--element-size FLOAT] [--n-frequencies INT] [--thickness-hint FLOAT]
        [--mesh-type tet|shell] [--boundary-condition free_free|fixed_edge]
        [--num-layers INT] [--status-json PATH]

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

from soundboard_lab.materials import SPRUCE_PLACEHOLDER, NonPhysicalMaterialError, validate_positive_definite
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
) -> None:
    material = SPRUCE_PLACEHOLDER
    validate_positive_definite(material)

    if thickness_hint is not None:
        check_through_thickness_layers(element_size, thickness_hint)

    work_dir = output_json.parent
    repaired_stl = work_dir / "repaired.stl"
    repair_report = repair_stl(stl_path, repaired_stl)

    mesh = mesh_surface(repaired_stl, element_size=element_size)
    tetra_count = next(c.data.shape[0] for c in mesh.cells if c.type == "tetra10")

    inp_path = work_dir / "analysis.inp"
    write_frequency_inp_orthotropic(mesh, material, inp_path, n_frequencies=n_frequencies)

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
) -> None:
    """Shell/extrusion-mesh analysis path (`shell_mesh.py`) -- the
    thin/flat-aware alternative to `run_analysis`'s generic tet path,
    for role=Soundboard objects. Material and grain orientation are
    still hardcoded to `SPRUCE_PLACEHOLDER`/identity (same known gap
    as the tet path; not addressed here -- tracked separately).

    Unlike `run_analysis`, also writes the solved shell's own footprint
    (`xy`/`triangles`) and every mode's per-node Z-displacement into
    the result JSON -- needed for the addon's mode-shape heat-map
    preview, which `run_analysis`'s tet path has no equivalent of.
    """
    tracker = StepTracker(status_json, SHELL_PIPELINE_STEPS)
    material = SPRUCE_PLACEHOLDER
    validate_positive_definite(material)
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
        tracker.done("mesh", f"{len(shell.triangles)} triangles")
    except Exception as exc:
        tracker.fail("mesh", str(exc))
        raise

    try:
        tracker.start("write_input")
        inp_path = work_dir / "analysis.inp"
        write_result = write_shell_frequency_inp(
            shell, material, num_layers=num_layers, inp_path=inp_path, n_frequencies=n_frequencies, fixed_edge=fixed_edge
        )
        tracker.done("write_input", f"{write_result.element_count} elements")
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
    }
    output_json.write_text(json.dumps(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", required=True, type=Path)
    parser.add_argument("--ccx-exe", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--element-size", type=float, default=15.0)
    parser.add_argument("--n-frequencies", type=int, default=10)
    parser.add_argument("--thickness-hint", type=float, default=None)
    parser.add_argument("--mesh-type", choices=["tet", "shell"], default="tet")
    parser.add_argument("--boundary-condition", choices=["free_free", "fixed_edge"], default="free_free")
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--status-json", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
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
            )
        else:
            run_analysis(
                stl_path=args.stl,
                ccx_exe=args.ccx_exe,
                output_json=args.output_json,
                element_size=args.element_size,
                n_frequencies=args.n_frequencies,
                thickness_hint=args.thickness_hint,
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
