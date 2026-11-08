# Soundboard Lab Gallery — feature list

Status: **planned, nothing built yet.** The data side exists: Results & Verification > Export Results writes self-describing experiment folders (layout and conventions: `src/soundboard_lab/experiment.py`, summary in `docs/CLAUDE.md` "Experiment export"). The gallery is a separate app that reads those folders; it never needs Blender.

## Architecture (agreed direction)

- **Read the export folder directly — no database.** A small Python server (engine venv, e.g. `python -m soundboard_lab.gallery --root <Export Folder>`) scans for `experiment.json`, serves a JSON API, the images and the built front end. Python because cross-experiment mode matching and the response estimator want numpy.
- **Front end: React + Vite + TypeScript** (Svelte is the alternative if React is not a hard preference).
- **Exports stay immutable.** Anything the user adds (tags, favourites, notes, imported reference spectra) goes in a sidecar next to the exports, never into `experiment.json`.
- **Schema tolerant.** Read `schema_version`, ignore unknown keys, so old exports stay browsable.

## Phase 1 — browse

- Experiment grid: name, date, M1 frequency, K, effective mass, MM, thumbnail; sort and filter on any of these.
- Experiment detail: mode table (frequency, bridge-mobility share, modal stiffness/mass), displacement and strain-energy images per mode, inputs (material, grain, boundary condition, bodies and roles, versions), notes.
- Group runs by geometry using the STL hashes in `inputs.json` (same soundboard, one body differing = a natural A/B pair).

## Phase 2 — compare

- **Frequency ladder:** one strip per experiment on a shared frequency axis.
- **Side-by-side mode compare** with synced zoom, across experiments. Matching tiers: by mode number, by nearest frequency, by shape similarity (MAC) when the soundboard meshes match (same `xy`).
- **Difference maps** (strain energy, displacement) when two experiments share a soundboard mesh.
- Tags, favourites and notes (sidecar).

## Phase 3 — estimated response curve

A modal-sum estimate from numbers already in the export; a rough estimate by design, labelled as such.

- **Bridge driving-point mobility:** `H(f) = sum(phi_d * phi_r / (w_i^2 - w^2 + 2j*zeta_i*w_i*w)) + residual`, from each mode's frequency and `phi_drive`/`phi_read`; the residual is the static compliance not carried by the solved modes (`1 - modal_sum_fraction`), so the low-frequency limit is the solved static bridge compliance.
- **Range:** 0–1 kHz is what matters; the usual 12 solved modes already reach about 1 kHz.
- **Sound view** (for like-for-like comparison with recorded spectra): weight each mode by its net volume displacement `integral(phi dA)` (needs a small export addition, below) and optionally couple an air layer: cavity volume (soundboard area x average depth), net hole area (need not be circular), and a rigid-box Helmholtz frequency as the main control (avoids guessing the end correction). Gives the A0 peak, the dip above it and the in-situ shift of M1. Air on/off toggle; the curve is labelled "coupled" so in-vacuo table frequencies are not confused with it. Back plate omitted, and said so on the chart.
- **Sliders (defaults to be fitted on the owner's real-guitar model):** damping (about 2% for the low modes, optionally rising with frequency), drive/read point (click anywhere on the board — every mode is stored at every vertex), residual on/off, frequency range and log/dB scale, view (bridge mobility / sound), air layer on/off with depth, hole area and Helmholtz frequency.
- **Honesty banner:** frequencies and relative strengths are modelled; absolute levels depend on assumed damping; no strings, no back, rim assumed held by the lining.
- **Overlay several experiments** on one axis (e.g. ring-only vs waist gusset).

## Reference spectra (imported measurements)

- Import CSV/TSV (delimiter, header and decimal comma auto-detected; typical REW / ARTA / Audacity exports). The owner's Sola files (`tests/data/guitar_spectrum/`) are tab-separated `Frequency (Hz)  Level (dB)` at about 2.93 Hz spacing.
- Metadata on import: instrument, strung/unstrung, weighting (none/A/C), excitation (tap/sweep), microphone distance. Weighted and unweighted traces are never compared silently.
- Stored under a `references/` folder with a sidecar, outside the experiments.
- Display: auto-normalise to a chosen peak or manual offset, fractional-octave smoothing, model evaluated on the reference's frequency grid, several overlays at once.
- **Peak labelling:** pick measured peaks, pair them with modelled modes and show the frequency deltas. A consistent offset points at material/boundary bias; a single mismatch points at one mode. This is also how damping and the air layer get calibrated on a real instrument.
- Caveats: measured spectra include radiation and room effects (compare peak positions, levels only loosely); strung data includes string loading the model omits (unstrung is the cleaner comparison).

## Data prerequisites (export changes needed first)

- Per-mode net volume displacement `integral(phi dA)` in the manifest (for the sound view).
- Optional small thumbnails per image if the number of experiments grows large.
- The drive/read positions are already in `inputs.json` (`mobility`) and per-vertex mode fields are in `fields.npz`, so a movable drive point needs nothing more.

## Later / ideas

- Re-render images from `fields.npz` with different colour ranges or captions.
- Tap-tone sonification from the modal sum.
- A rib-impact gauge view (planned separately, built on the curvature field).

## Open question — distribution and licensing (undecided)

Owner's note (2026-11-08): the gallery looks like a good candidate for a **paid-only** feature. This differs from the earlier repository decision that Soundboard Lab itself stays a fully free GPL Blender add-on, monetised through consulting and data packs, with no locked or paid code paths. Because the gallery is a separate app that reads a documented on-disk format, the add-on can stay free either way. Still to decide: whether the gallery is free or paid, and — if it reuses engine code (GPL) — what that means for how it can be distributed (not legal advice; get it checked before any release).
