# Soundboard Lab experiment format (schema_version 1)

The on-disk contract between the Soundboard Lab engine (writer) and any consumer such as the planned gallery app (reader). A consumer needs only this document, `json` and `numpy`: no Blender, no engine import, no solver.

The writer is `src/soundboard_lab/experiment.py` (pytest: `tests/test_experiment_export.py`). If this document and the code disagree, the code wins; fix the document.

## Rules for readers

1. Identify an experiment by a folder containing `experiment.json` whose `schema` is `"soundboard_lab.experiment"`.
2. **Ignore unknown keys.** New fields are added without bumping the version. Only a breaking change bumps `schema_version`; refuse a major version you do not know and say so, rather than guessing.
3. **Any value may be `null`/absent when it does not apply** (for example no mobility run, no energy map for a mode, images switched off). Never assume a key is non-null without checking; the Notes column below says where `null` occurs.
4. **Skip folders ending `.partial`.** The writer builds `<name>.partial` and renames it when complete, so a finished folder is never half-written.
5. **Exports are immutable.** Never write into an experiment folder. Tags, favourites, notes and imported reference spectra belong in a sidecar elsewhere, keyed by the manifest `id`.
6. Identity: `id` (32-hex uuid) is unique per export; `folder` is the folder name. Copying a folder keeps the `id`, so key sidecars on `id` and expect duplicates.

## Folder layout

```
<YYYYMMDD-HHMMSS>_<name-slug>/        a numeric suffix (-2, -3) is added on name clashes
    experiment.json                   manifest (below)
    modes.csv                         one row per mode, the headline numbers
    images/                           mode_NN_displacement.png, mode_NN_energy.png (NN = zero-padded mode number)
    fields.npz                        raw per-vertex fields (optional)
    solver/                           STLs, ccx .inp/.dat, input JSONs (optional); reproducibility only
```

`images/`, `fields.npz` and `solver/` are each optional (export toggles). `manifest.files` and `manifest.image_spec` say what is present; `image_spec` is `null` when there are no images.

## Units and conventions

| Quantity | Convention |
|---|---|
| Lengths | mm, in the world frame the soundboard was exported in (+X right, +Y up when viewed from above; Z is out of plane) |
| Frequency | Hz |
| Mode shapes | CalculiX mass-normalised eigenvectors (units 1/sqrt(tonne), arbitrary overall scale), **sign-normalised on export**: the largest-magnitude displacement of each mode is positive, so the same mode looks the same across experiments. `displacement.sign_flipped` records whether the solver's sign differed. |
| Everything signed per mode | `phi_drive`, `phi_read`, `net_volume`, `fields.npz` `displacement` and `hessian` all use the *exported* (normalised) sign. Their products with each other (e.g. `phi_drive * phi_read`) are sign-independent. |
| Stiffness / compliance | `kgf/mm` and `mm/kgf` (1 kgf = 9.80665 N) |
| Effective / modal mass | grams |
| MM (monopole mobility) | m/s/N |
| Strain energy per area | per-mode, normalised so each mode integrates to 1 over the plan footprint |
| Material (in `inputs`) | density t/mm^3, moduli MPa |

## `experiment.json`

Top level:

| Key | Type | Notes |
|---|---|---|
| `schema` | string | always `"soundboard_lab.experiment"` |
| `schema_version` | int | `1` |
| `id` | string | 32-hex uuid |
| `folder` | string | folder name |
| `name`, `notes` | string | user-supplied at export |
| `created` | string | ISO 8601 with offset, seconds precision |
| `software` | object | `engine`, `python`, `numpy`, `blender`, `addon` (version strings; any may be `null`) |
| `summary` | object | headline numbers for grids; below |
| `inputs` | object | what defined the solve; below. May be `{}` for exports made without an add-on `inputs.json` |
| `mesh` | object | solved mesh facts; below |
| `modes` | array | one entry per solved mode, ascending mode number; below |
| `mobility` | object or `null` | raw bridge-mobility result; below |
| `image_spec` | object or `null` | how to interpret the PNGs; below |
| `conventions` | object | human-readable restatement of the conventions above; do not parse it |
| `files` | object | `manifest`, `modes_csv`, `fields` (`"fields.npz"` or `null`), `solver` (sorted list of file names) |

### `summary`

`n_modes`; `frequencies_hz` (list); `m1_mode`, `m1_frequency_hz`, `stiffness_kgf_per_mm`, `effective_mass_g`, `mobility_m_per_s_per_n` (all `null` when no mobility run was done).

"M1" is the elastic mode with the largest share of the static bridge compliance (or the mode the user named; see `mobility.m1_chosen_by`). It is not necessarily mode 1.

### `modes[]`

```jsonc
{
  "mode_number": 1,                 // CalculiX elastic mode number (1-based)
  "frequency_hz": 178.4,
  "is_m1": true,
  "displacement": {
    "image": "images/mode_01_displacement.png",   // null if images off
    "color_range": [-0.12, 0.12],   // symmetric about zero; zero is green in the colormap
    "max_abs": 0.12,
    "net_volume": 1.6,              // integral of the (sign-normalised) Z displacement over the footprint; mm^2/sqrt(tonne)
    "net_volume_ratio": 0.62,       // net / integral(|w|), in [-1, 1]; ~1 breathing mode, ~0 lobes cancel; null if the field is all zero
    "peak_xy_mm": [123.4, 56.7],    // where |w| is largest
    "sign_flipped": false
  },
  "strain_energy": {                // null for a mode with no energy map
    "image": "images/mode_01_energy.png",          // null if energy images off
    "color_range": [0.0, 3.2e-4],   // clipped above; plan-area-normalised units
    "total_energy": 1.7e4,          // strain energy in the solved plate (Soundboard body only)
    "plate_energy_ratio": 0.96,     // thin-plate bending energy from the curvature field / ccx strain energy; ~1 when plate theory holds (a consistency check, not a mode property); may be null
    "centroid_xy_mm": [101.0, 80.2],// energy-weighted centroid
    "overlays": true                // iso-contours and bending-direction ticks drawn on the image
  },
  "mobility": {                     // null if no mobility run or this mode is not in it
    "fraction": 0.81,               // this mode's share of the static bridge compliance (may be negative)
    "compliance_mm_per_kgf": 0.25,
    "modal_stiffness_kgf_per_mm": 4.0,   // null when compliance <= 0 (mode opposes the static deflection)
    "modal_mass_g": 66.0,                // null likewise
    "phi_drive": 0.1,               // mode Z motion at the drive patches (mean), 1/sqrt(tonne)
    "phi_read": 0.1                 // ... at the read patch
  }
}
```

`compliance_mm_per_kgf = phi_drive * phi_read / omega^2` (converted to mm/kgf), `modal_stiffness = 1/compliance`, `modal_mass = modal_stiffness / omega^2`, with `omega = 2*pi*frequency_hz`.

### `mobility` (raw `bridge_mobility` result)

```jsonc
{
  "force_n": 9.80665,                       // total static load, shared equally between drive points
  "bridge_deflection_mm": 0.31,
  "compliance_mm_per_kgf": 0.31,            // static solve, read patch
  "stiffness_kgf_per_mm": 3.2,
  "drive_deflection_mm_per_kgf": [0.30, 0.32],
  "m1_mode": 1, "m1_frequency_hz": 178.4, "m1_chosen_by": "largest share of static compliance",
  "effective_mass_g": 70.1,                 // k / omega_M1^2
  "mobility_m_per_s_per_n": 0.021,          // MM = 2 pi f / k, strongly M1-biased by design
  "modal_sum_fraction": 1.0,                // sum of mode fractions; < 1 means too few modes solved to cover the static compliance
  "modes": [ { "mode_number", "frequency_hz", "phi_drive", "phi_read", "compliance_mm_per_kgf", "fraction" } ],
                                            // NOTE: here phi_* are the solver's raw sign; use modes[].mobility for the normalised sign
  "drive_points": [ { "name", "nodes", "snap_distance" } ],
  "read_position": [x, y, z],
  "read_nodes": 12
}
```

Prefer `modes[].mobility` (sign-normalised) over `mobility.modes[]` for anything that is plotted per mode.

### `inputs`

Written by the add-on at solve time, so it describes the solve even if the scene has changed since. Shape (additive; treat every key as optional):

```jsonc
{
  "solved_at": "2026-11-10T16:20:00-08:00",
  "blend_file": "C:/.../board.blend",
  "blender_version": "5.2.2", "addon_version": "0.1.0",
  "soundboard": {
    "object": "Soundboard",
    "stl_sha256": "<hex>",                  // identical hash = identical geometry: use to group A/B runs
    "element_size": 5.0, "thickness_hint": 0.0, "num_layers": 3,
    "material": { "kind": "ORTHOTROPIC", "name": "...", "density": 4.0e-10, "e_longitudinal": 13800, "e_radial": 0, "e_tangential": 0,
                  "g_lr": 0, "g_lt": 0, "g_rt": 0, "nu_lr": 0, "nu_lt": 0, "nu_rt": 0 },
                                            // or { "kind": "ISOTROPIC", "name", "density", "e", "nu" }; may be null (engine default)
    "orientation": [a1, a2, a3, b1, b2, b3] // CalculiX *ORIENTATION: local-1 axis, then a vector in the local 1-2 plane; null = identity
  },
  "parameters": { "n_frequencies": 12, "boundary_condition": "fixed_edge", "tie_tolerance": 0.5 },
                                            // boundary_condition: "free_free" | "fixed_edge" | "body_lining"
  "bodies": [ { "name", "role", "stl_sha256", "element_size", "material", "orientation",
                "terminal": { "condition": "fixed|free|spring", "stiffness": 0.0 } | null } ],
                                            // role: RIB | BRIDGE_PLATE | LINING | ...
  "attachments": [ { "name", "position": [x,y,z], "radius", "stiffness" /* N/mm */, "mass_g" } ],
  "mobility": { "drive_points": [ { "name", "position": [x,y,z], "radius" } ] } | null
}
```

For grouping and "what differs between these runs" views, compare: `soundboard.stl_sha256`, the set of `(bodies[].name, stl_sha256)`, `soundboard.material` / `orientation`, `parameters`, `attachments`.

### `mesh`

`vertices`, `triangles` (counts); `bounds_mm` = `[xmin, ymin, xmax, ymax]`; `thickness_mm` = `{min, max}` or `null`; `material_name`; `repair` (STL repair report); `bonded_bodies` = `[{name, tie_nodes, ...}]`; `point_attachments`.

Two experiments have **the same soundboard mesh** (so vertex-wise comparison, MAC and difference maps are valid) when `fields.npz` `xy` and `triangles` are equal (compare `vertices` and `bounds_mm` first as a cheap filter).

### `image_spec`

```jsonc
{
  "width_px": 1200, "height_px": 640,
  "bounds_mm": [xmin, ymin, xmax, ymax],   // world rectangle the image covers: the board bounds plus a 3% margin; px_per_mm = width_px / (xmax - xmin)
  "px_per_mm": 3.1, "supersample": 2,
  "orientation": "+x right, +y up (viewed from above); transparent outside the board",
  "colormap": [[0.0, [r,g,b]], ...],      // jet control points, t in [0,1], rgb 0-255
  "overlay_color": [r,g,b], "line_width_mm": 0.6
}
```

Images carry no text and no colorbar. Caption them from the manifest: map a pixel's colour back to a value with `color_range` and `colormap`. All images of one experiment share the same size and bounds, so tiles from different experiments of the same board line up. Mode images across *different* boards share neither bounds nor scale; use `bounds_mm` to place them.

## `modes.csv`

UTF-8, header row, one row per mode. Blank cell = not applicable. Columns:

`mode_number, frequency_hz, is_m1, peak_x_mm, peak_y_mm, sign_flipped, net_volume, net_volume_ratio, energy_centroid_x_mm, energy_centroid_y_mm, total_strain_energy, plate_energy_ratio, mobility_fraction, compliance_mm_per_kgf, modal_stiffness_kgf_per_mm, modal_mass_g, phi_drive, phi_read`

Booleans are `True`/`False`. The CSV is a convenience for spreadsheets; read the manifest in code.

## `fields.npz`

`numpy.load("fields.npz")`; V = vertices of the solved soundboard footprint mesh, T = triangles, M = modes (same order as `experiment.json` `modes`).

| Array | Shape | Notes |
|---|---|---|
| `xy` | (V, 2) | mm |
| `top_z` | (V,) | top-surface height, mm |
| `triangles` | (T, 3) int64 | indices into `xy` |
| `thickness` | (V,) | mm; only present when known |
| `mode_numbers` | (M,) | |
| `frequencies_hz` | (M,) | |
| `displacement` | (M, V) | per-vertex Z motion of each mode, sign-normalised, mass-normalised scale |
| `energy_per_area` | (M, V) | NaN row for a mode with no energy map |
| `hessian` | (M, V, 3) | `[w_xx, w_yy, w_xy]` curvature of the (sign-normalised) mode; NaN row if absent |

Every mode is stored at every vertex, so a drive/read point can be moved anywhere on the board after the fact without re-solving.

Useful derived quantities (all in numpy from these arrays):

- Footprint integral of a vertex field `f`: `sum(tri_area * f[triangles].mean(axis=1))`, `tri_area = 0.5*|cross(b-a, c-a)|`. `net_volume` in the manifest is exactly this applied to `displacement`.
- MAC between two modes on the same mesh: `(a.b)^2 / ((a.a)(b.b))` on `displacement` rows (sign does not matter).
- `phi` at an arbitrary point `p`: interpolate `displacement[m]` barycentrically in the triangle containing `p`.

## Estimated response curve (what the stored numbers support)

Driving-point mobility at the bridge from the modal sum, with a damping assumption `zeta` (e.g. 0.02):

```
H(w) = sum_n  phi_drive_n * phi_read_n / (w_n^2 - w^2 + 2j*zeta*w_n*w)  +  residual
```

`residual` is the static compliance not carried by the solved modes: `(1 - mobility.modal_sum_fraction) * static_compliance` (convert units: `phi*phi/omega^2` is mm/N; compliance mm/kgf divided by 9.80665). At `w -> 0` the sum plus residual equals the solved static bridge compliance.

Sound-radiating (net-volume) view: replace `phi_read_n` by `net_volume_n` and multiply by `j*w`:

```
Y_V(w) = sum_n  j*w * phi_drive_n * net_volume_n / (w_n^2 - w^2 + 2j*zeta*w_n*w)
```

`net_volume` and `phi_drive` share the exported sign convention, so the product is meaningful. A mode with `net_volume_ratio` near 0 radiates poorly (monopole) however large its mobility. This is a rough, in-vacuo, back-less, string-less estimate: label it as modelled. Coupling an air cavity / Helmholtz resonance is gallery-side and not stored in the export.

## Versioning and compatibility

- Additive changes (new keys, new CSV columns at the end, new arrays) keep `schema_version`. `net_volume` and `net_volume_ratio` were added this way after the first release: **an older export lacks them**, so treat them as optional and show "n/a" or fall back (compute the integral from `fields.npz` if present).
- Breaking changes (rename, unit change, layout change) bump `schema_version`; the writer's module docstring and this file change together.

## Seeding a separate gallery project

Intended shape, so a new repo can start from this document plus [gallery_features.md](./gallery_features.md):

- **Separate repository; do not import `soundboard_lab`.** The only coupling is this format. This keeps the engine and the add-on (GPL) separate from the gallery so the gallery's licence can be decided independently (get legal review before any release; not advice). Re-implement the small amount of maths above from the formulas here instead of copying engine code.
- **Read folders directly, no database.** A small Python server scans a root folder for `experiment.json`, serves a JSON API and the images, and hosts the built front end. Python because mode matching (MAC) and the response estimator want numpy. Front end: React + Vite + TypeScript.
- **Sidecar** (tags, favourites, notes, imported reference spectra) in a folder next to the exports, keyed by experiment `id`.
- **Test fixtures:** commit two or three small exports (`--no-solver-files`, narrow `image_width`) covering: a full export with mobility; a mode without an energy map; an export with no mobility; an older export without `net_volume` (delete the keys by hand). A synthetic generator exists in the engine tests (`_synthetic_work_dir` in `tests/test_experiment_export.py`) that produces a small export without Blender or ccx; copy that approach.
- **Start with Phase 1 (browse)** from `gallery_features.md`, which needs only `experiment.json` and the PNGs. `fields.npz` is needed from Phase 2 (compare, difference maps) and Phase 3 (movable drive point).
