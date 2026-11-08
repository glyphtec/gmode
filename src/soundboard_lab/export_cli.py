"""CLI: export a finished solve (its work directory) into an experiment
folder -- see `experiment.py` for the layout. Run by the add-on's
"Export Results" dialog in the engine venv, same as the solve:

    python -m soundboard_lab.export_cli --work-dir DIR --out-root DIR --name NAME --output-json PATH

Writes `{"status": "ok", "folder": ...}` (or `{"status": "error",
"message": ...}`) to `--output-json`, which the add-on polls.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

from soundboard_lab.experiment import ExportOptions, write_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--notes", default="")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--image-width", type=int, default=ExportOptions.image_width)
    parser.add_argument("--line-width-mm", type=float, default=ExportOptions.line_width_mm)
    parser.add_argument("--no-displacement-images", action="store_true")
    parser.add_argument("--no-energy-images", action="store_true")
    parser.add_argument("--no-overlays", action="store_true")
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--no-solver-files", action="store_true")
    args = parser.parse_args(argv)

    try:
        options = ExportOptions(
            image_width=args.image_width,
            displacement_images=not args.no_displacement_images,
            energy_images=not args.no_energy_images,
            overlays=not args.no_overlays,
            line_width_mm=args.line_width_mm,
            fields=not args.no_fields,
            solver_files=not args.no_solver_files,
        )
        folder = write_experiment(args.work_dir, args.out_root, args.name, args.notes, options)
        args.output_json.write_text(json.dumps({"status": "ok", "folder": str(folder)}, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 -- the add-on must always get a readable result
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
    raise SystemExit(main())
