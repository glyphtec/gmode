"""Experiment export: dump one solve into a self-contained folder an
external gallery/browser app can index without Blender or a re-solve.

    <out_root>/<YYYYMMDD-HHMMSS>_<name>/
        experiment.json   manifest (schema below): identity, inputs, mesh,
                          per-mode frequencies/metrics/image paths, mobility
        modes.csv         one row per mode, the headline numbers
        images/           mode_NN_displacement.png, mode_NN_energy.png
        fields.npz        the raw per-vertex fields (re-render, resample,
                          or compare modes across experiments later)
        solver/           STLs, .inp/.dat and the input JSONs (reproducible)

Written to `<folder>.partial` and renamed once complete, so a scanner
never sees a half-written experiment.

Conventions a consumer needs:

* Mode shapes are mass-normalised (ccx) and **sign-normalised on
  export**: the largest-magnitude displacement is positive, so the same
  mode in two experiments has the same colors (an eigenvector's sign is
  otherwise arbitrary). `fields.npz` and the displacement images use it;
  `sign_flipped` says whether it differed from the solver's.
* Displacement images are colored symmetrically about zero over
  `[-max_abs, +max_abs]` (zero is green); strain-energy images over
  `[0, display_scale]` (clipped above) in energy per plan area
  normalised so each mode integrates to 1 over the footprint.
* All lengths are mm, in the world frame the soundboard was exported in.
* `schema_version` bumps on any breaking change to this layout.
"""

from __future__ import annotations

import csv
import json
import platform
import re
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from soundboard_lab.mode_energy import triangle_areas
from soundboard_lab.mode_images import (
    JET_CONTROL_POINTS,
    OVERLAY_ORANGE,
    build_raster,
    make_frame,
    render_image,
    write_png,
)

SCHEMA = "soundboard_lab.experiment"
SCHEMA_VERSION = 1
KGF_TO_N = 9.80665


@dataclass(frozen=True)
class ExportOptions:
    image_width: int = 1200
    displacement_images: bool = True
    energy_images: bool = True
    overlays: bool = True  # contours + bending-direction ticks on the energy images
    line_width_mm: float = 0.6
    fields: bool = True
    solver_files: bool = True
    supersample: int = 2


def slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-.")[:60]
    return slug or "experiment"


def _engine_version() -> str | None:
    from importlib import metadata

    for dist in ("soundboard_lab", "soundboard-lab"):
        try:
            return metadata.version(dist)
        except metadata.PackageNotFoundError:
            continue
    return None


def sign_normalize(displacement: np.ndarray) -> float:
    """+1 or -1: the factor that makes the largest-magnitude entry positive."""
    return 1.0 if displacement[int(np.argmax(np.abs(displacement)))] >= 0.0 else -1.0


def vertex_areas(xy: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """(V,) a third of the area of every triangle around each vertex."""
    areas = np.zeros(len(xy))
    third = triangle_areas(xy, triangles) / 3.0
    for corner in range(3):
        np.add.at(areas, triangles[:, corner], third)
    return areas


def _segments(energy: dict | None, curvature: dict | None) -> np.ndarray:
    parts = []
    if energy is not None:
        parts += [np.asarray(c["segments"], dtype=float).reshape(-1, 6) for c in energy["contours"]]
    if curvature is not None:
        parts.append(np.asarray(curvature["ticks"], dtype=float).reshape(-1, 6))
    return np.vstack(parts) if parts else np.empty((0, 6))


def _round(value) -> float | None:
    return None if value is None else float(f"{float(value):.8g}")


def _unique_folder(out_root: Path, stem: str) -> Path:
    folder = out_root / stem
    index = 2
    while folder.exists() or folder.with_name(folder.name + ".partial").exists():
        folder = out_root / f"{stem}-{index}"
        index += 1
    return folder


def write_experiment(
    work_dir: Path,
    out_root: Path,
    name: str,
    notes: str = "",
    options: ExportOptions = ExportOptions(),
) -> Path:
    """Export the solve whose files are in `work_dir` (`result.json`,
    and ideally the `inputs.json` the add-on writes beside it) into a
    new experiment folder under `out_root`; returns its path."""
    work_dir, out_root = Path(work_dir), Path(out_root)
    result_path = work_dir / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"no result.json in {work_dir}")
    result = json.loads(result_path.read_text())
    if result.get("status") != "ok":
        raise ValueError(f"cannot export a failed solve: {result.get('message', 'no message')}")
    inputs_path = work_dir / "inputs.json"
    inputs = json.loads(inputs_path.read_text()) if inputs_path.is_file() else {}

    out_root.mkdir(parents=True, exist_ok=True)
    created = datetime.now().astimezone()
    final = _unique_folder(out_root, f"{created.strftime('%Y%m%d-%H%M%S')}_{slugify(name)}")
    staging = final.with_name(final.name + ".partial")
    staging.mkdir()
    try:
        _write_into(staging, final.name, work_dir, result, inputs, name, notes, options, created)
        staging.rename(final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _write_into(
    folder: Path,
    folder_name: str,
    work_dir: Path,
    result: dict,
    inputs: dict,
    name: str,
    notes: str,
    options: ExportOptions,
    created: datetime,
) -> None:
    shape = result["mesh_shape"]
    xy = np.asarray(shape["xy"], dtype=float)
    top_z = np.asarray(shape["top_z"], dtype=float)
    triangles = np.asarray(shape["triangles"], dtype=np.int64)
    thickness = np.asarray(shape["thickness"], dtype=float) if shape.get("thickness") else None
    areas = vertex_areas(xy, triangles)

    mobility = result.get("bridge_mobility")
    mobility_by_mode = {m["mode_number"]: m for m in (mobility or {}).get("modes", [])}
    m1_mode = (mobility or {}).get("m1_mode")

    want_images = options.displacement_images or options.energy_images
    raster = None
    frame = None
    if want_images:
        (folder / "images").mkdir()
        frame = make_frame(xy, options.image_width)
        raster = build_raster(frame, xy, triangles, options.supersample)

    mode_numbers, frequencies = [], []
    displacement_rows, energy_rows, hessian_rows = [], [], []
    mode_entries = []
    for mode in result["modes"]:
        number = int(mode["mode_number"])
        key = str(number)
        displacement = np.asarray(result["mode_shapes"][key], dtype=float)
        sign = sign_normalize(displacement)
        displacement = displacement * sign
        energy = (result.get("mode_energy") or {}).get(key)
        curvature = (result.get("mode_curvature") or {}).get(key)

        peak = int(np.argmax(np.abs(displacement)))
        bound = float(np.abs(displacement).max())
        entry = {
            "mode_number": number,
            "frequency_hz": _round(mode["frequency_hz"]),
            "is_m1": number == m1_mode,
            "displacement": {
                "image": None,
                "color_range": [-bound, bound],
                "max_abs": _round(bound),
                "peak_xy_mm": [_round(xy[peak, 0]), _round(xy[peak, 1])],
                "sign_flipped": sign < 0,
            },
            "strain_energy": None,
            "mobility": None,
        }
        if options.displacement_images:
            relative = f"images/mode_{number:02d}_displacement.png"
            write_png(folder / relative, render_image(raster, displacement, (-bound, bound)))
            entry["displacement"]["image"] = relative

        energy_values = None
        if energy is not None:
            energy_values = np.asarray(energy["energy_per_area"], dtype=float)
            weight = areas * energy_values
            centroid = (weight[:, None] * xy).sum(axis=0) / weight.sum()
            entry["strain_energy"] = {
                "image": None,
                "color_range": [0.0, _round(energy["display_scale"])],
                "total_energy": _round(energy["total_energy"]),
                "plate_energy_ratio": _round((curvature or {}).get("plate_energy_ratio")),
                "centroid_xy_mm": [_round(centroid[0]), _round(centroid[1])],
                "overlays": bool(options.overlays),
            }
            if options.energy_images:
                relative = f"images/mode_{number:02d}_energy.png"
                segments = _segments(energy, curvature) if options.overlays else None
                write_png(
                    folder / relative,
                    render_image(
                        raster,
                        energy_values,
                        (0.0, float(energy["display_scale"])),
                        segments,
                        options.line_width_mm,
                    ),
                )
                entry["strain_energy"]["image"] = relative

        if number in mobility_by_mode:
            share = mobility_by_mode[number]
            compliance = float(share["compliance_mm_per_kgf"])
            modal_stiffness = modal_mass = None
            if compliance > 0.0:  # a mode that opposes the static deflection has no meaningful stiffness/mass
                modal_stiffness = 1.0 / compliance  # kgf/mm, this mode acting alone at the bridge
                omega = 2.0 * np.pi * float(mode["frequency_hz"])
                modal_mass = modal_stiffness * KGF_TO_N * 1000.0 / omega**2 * 1000.0  # g
            entry["mobility"] = {
                "fraction": _round(share["fraction"]),
                "compliance_mm_per_kgf": _round(compliance),
                "modal_stiffness_kgf_per_mm": _round(modal_stiffness),
                "modal_mass_g": _round(modal_mass),
                # Same sign convention as the images and fields (both flip together; their product does not).
                "phi_drive": _round(share["phi_drive"] * sign),
                "phi_read": _round(share["phi_read"] * sign),
            }
        mode_entries.append(entry)

        mode_numbers.append(number)
        frequencies.append(float(mode["frequency_hz"]))
        displacement_rows.append(displacement)
        energy_rows.append(energy_values if energy_values is not None else np.full(len(xy), np.nan))
        hessian = np.asarray(curvature["hessian"], dtype=float) * sign if curvature else None
        hessian_rows.append(hessian if hessian is not None else np.full((len(xy), 3), np.nan))

    files = {"manifest": "experiment.json", "modes_csv": "modes.csv", "fields": None, "solver": []}
    if options.fields:
        arrays = {
            "xy": xy,
            "top_z": top_z,
            "triangles": triangles,
            "mode_numbers": np.asarray(mode_numbers),
            "frequencies_hz": np.asarray(frequencies),
            "displacement": np.asarray(displacement_rows),
            "energy_per_area": np.asarray(energy_rows),
            "hessian": np.asarray(hessian_rows),
        }
        if thickness is not None:
            arrays["thickness"] = thickness
        np.savez_compressed(folder / "fields.npz", **arrays)
        files["fields"] = "fields.npz"

    if options.solver_files:
        files["solver"] = _copy_solver_files(work_dir, folder / "solver")

    _write_modes_csv(folder / "modes.csv", mode_entries)

    lo, hi = xy.min(axis=0), xy.max(axis=0)
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "id": uuid.uuid4().hex,
        "folder": folder_name,
        "name": name,
        "notes": notes,
        "created": created.isoformat(timespec="seconds"),
        "software": {
            "engine": _engine_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "blender": inputs.get("blender_version"),
            "addon": inputs.get("addon_version"),
        },
        "summary": {
            "n_modes": len(mode_entries),
            "frequencies_hz": [e["frequency_hz"] for e in mode_entries],
            "m1_mode": m1_mode,
            "m1_frequency_hz": _round((mobility or {}).get("m1_frequency_hz")),
            "stiffness_kgf_per_mm": _round((mobility or {}).get("stiffness_kgf_per_mm")),
            "effective_mass_g": _round((mobility or {}).get("effective_mass_g")),
            "mobility_m_per_s_per_n": _round((mobility or {}).get("mobility_m_per_s_per_n")),
        },
        "inputs": inputs,
        "mesh": {
            "vertices": len(xy),
            "triangles": len(triangles),
            "bounds_mm": [_round(lo[0]), _round(lo[1]), _round(hi[0]), _round(hi[1])],
            "thickness_mm": (
                {"min": _round(thickness.min()), "max": _round(thickness.max())} if thickness is not None else None
            ),
            "material_name": result.get("material_name"),
            "repair": result.get("mesh_repair"),
            "bonded_bodies": result.get("bonded_bodies") or [],
            "point_attachments": result.get("point_attachments") or [],
        },
        "modes": mode_entries,
        "mobility": mobility,
        "image_spec": (
            {
                "width_px": frame.width,
                "height_px": frame.height,
                "bounds_mm": [_round(frame.xmin), _round(frame.ymin), _round(frame.xmax), _round(frame.ymax)],
                "px_per_mm": _round(frame.px_per_mm),
                "supersample": options.supersample,
                "orientation": "+x right, +y up (viewed from above); transparent outside the board",
                "colormap": [[t, list(rgb)] for t, rgb in JET_CONTROL_POINTS],
                "overlay_color": list(OVERLAY_ORANGE),
                "line_width_mm": options.line_width_mm,
            }
            if want_images
            else None
        ),
        "conventions": {
            "sign_normalized": "the largest-magnitude displacement of each mode is positive",
            "displacement_units": "mass-normalised eigenvector (arbitrary scale)",
            "energy_per_area": "per-mode strain energy per plan area, normalised to integrate to 1 over the footprint",
            "mobility_modes": (
                "per mode at the bridge: phi_drive/phi_read are the mass-normalised mode's Z motion (1/sqrt(tonne)) "
                "averaged over the drive patches / read patch, in the same sign convention as the images; "
                "compliance = phi_drive*phi_read/omega^2 is the mode's share of the static compliance (fraction); "
                "modal_stiffness = 1/compliance and modal_mass = stiffness/omega^2 are the mode acting alone at the bridge"
            ),
            "lengths": "mm",
        },
        "files": files,
    }
    (folder / "experiment.json").write_text(json.dumps(manifest, indent=2))


_SOLVER_PATTERNS = ("*.stl", "analysis*.inp", "analysis*.dat", "material.json", "bodies.json",
                    "attachments.json", "mobility.json", "inputs.json")


def _copy_solver_files(work_dir: Path, destination: Path) -> list[str]:
    destination.mkdir()
    copied = []
    for pattern in _SOLVER_PATTERNS:
        for source in sorted(work_dir.glob(pattern)):
            if source.name not in copied:
                shutil.copy2(source, destination / source.name)
                copied.append(source.name)
    return sorted(copied)


def _write_modes_csv(path: Path, entries: list[dict]) -> None:
    columns = [
        "mode_number", "frequency_hz", "is_m1", "peak_x_mm", "peak_y_mm", "sign_flipped",
        "energy_centroid_x_mm", "energy_centroid_y_mm", "total_strain_energy", "plate_energy_ratio",
        "mobility_fraction", "compliance_mm_per_kgf", "modal_stiffness_kgf_per_mm", "modal_mass_g",
        "phi_drive", "phi_read",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for e in entries:
            energy = e["strain_energy"] or {}
            share = e["mobility"] or {}
            centroid = energy.get("centroid_xy_mm") or ["", ""]
            writer.writerow(
                [
                    e["mode_number"], e["frequency_hz"], e["is_m1"],
                    e["displacement"]["peak_xy_mm"][0], e["displacement"]["peak_xy_mm"][1],
                    e["displacement"]["sign_flipped"],
                    centroid[0], centroid[1], energy.get("total_energy", ""), energy.get("plate_energy_ratio", ""),
                    share.get("fraction", ""), share.get("compliance_mm_per_kgf", ""),
                    "" if share.get("modal_stiffness_kgf_per_mm") is None else share["modal_stiffness_kgf_per_mm"],
                    "" if share.get("modal_mass_g") is None else share["modal_mass_g"],
                    share.get("phi_drive", ""), share.get("phi_read", ""),
                ]
            )
