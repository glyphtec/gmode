"""CLI entry point for bond detection, launched by the Blender add-on as
a subprocess (same engine/UI boundary as `cli.py`: the add-on never
imports gmsh/trimesh into Blender's bundled Python).

Usage:
    python -m soundboard_lab.bond_cli --manifest PATH --output-json PATH

The manifest is JSON:
    {
      "tolerance": 0.1,
      "soundboard": {"stl": "...", "element_size": 10.0},
      "bodies": [{"name": "Rib 1", "stl": "...", "element_size": 2.0}, ...]
    }

The result JSON carries, per body, its classification plus the meshed
skin and per-vertex signed distances the add-on's bonding proxies
display, and the meshed Soundboard surface itself. Like `cli.py`, any
failure is written as `{"status": "error", ...}` rather than left as a
bare nonzero exit code.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from soundboard_lab.bonding import BodyInput, detect_bonds


def run_bond_detection(manifest: dict, output_json: Path) -> None:
    soundboard = manifest["soundboard"]
    bodies = {
        body["name"]: BodyInput(stl_path=Path(body["stl"]), element_size=float(body["element_size"]))
        for body in manifest["bodies"]
    }
    report = detect_bonds(
        Path(soundboard["stl"]),
        bodies,
        soundboard_element_size=float(soundboard["element_size"]),
        tolerance=float(manifest["tolerance"]),
        work_dir=output_json.parent / "bond_work",
    )

    body_results = []
    for name, body in report.bodies.items():
        entry = {"name": name, "status": body.status.value, "message": body.message}
        detection = body.detection
        if detection is not None:
            entry.update(
                contact_area=detection.contact_area,
                nearest_distance=detection.nearest_distance,
                max_penetration=detection.max_penetration,
                points=detection.points.tolist(),
                triangles=detection.triangles.tolist(),
                signed_distances=detection.signed_distances.tolist(),
            )
        body_results.append(entry)

    result = {
        "status": "ok",
        "tolerance": float(manifest["tolerance"]),
        "soundboard_surface": {
            "points": report.soundboard_surface.vertices.tolist(),
            "triangles": report.soundboard_surface.faces.tolist(),
        },
        "bodies": body_results,
    }
    output_json.write_text(json.dumps(result))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        run_bond_detection(json.loads(args.manifest.read_text()), args.output_json)
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
    sys.exit(main())
