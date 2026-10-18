"""CLI entry point for running an analysis from outside the engine's
own Python process -- this is what the Blender addon invokes as a
subprocess (per docs/CLAUDE.md's engine/UI boundary rule: the engine
has zero `bpy` imports, and gmsh/meshio can't be installed into
Blender's bundled Python, so the addon launches this venv's own
Python running this module rather than importing the engine in-process).

Usage:
    python -m soundboard_lab.cli --stl PATH --ccx-exe PATH --output-json PATH
        [--element-size FLOAT] [--n-frequencies INT] [--thickness-hint FLOAT]

Writes a JSON result (or error) to --output-json; the addon polls for
that file's existence rather than parsing subprocess stdout, so the
interface between the two stays simple and stable.
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
from soundboard_lab.solver import CcxError, parse_eigenfrequencies, run_ccx

MIN_THROUGH_THICKNESS_LAYERS = 2


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stl", required=True, type=Path)
    parser.add_argument("--ccx-exe", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--element-size", type=float, default=15.0)
    parser.add_argument("--n-frequencies", type=int, default=10)
    parser.add_argument("--thickness-hint", type=float, default=None)
    args = parser.parse_args(argv)

    try:
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
