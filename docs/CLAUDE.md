# Soundboard Lab: a Blender add-on for luthier soundboard experiments

Working title. Read this first, every session. It frames *what* we're building and the decisions already made, so you don't re-litigate them.

## Purpose

A **domain-specific** tool that lets luthiers run rapid, precise experiments on soundboard acoustic treatments (bracing, contouring, mass loading) as eigenmode analyses, from inside Blender. It is **not** a general-purpose FEA add-on. A small, fixed vocabulary of guitar concepts is what makes the tool both easy to use and verifiable.

Primary user: the author (experimental luthier, "Evergreen" stress-neutralized soundboard system, orthotropic spruce, floating bridge, force compensators). Secondary: the wider luthier community, as an open-source gift. Design for the second without compromising the first.

The author has a full experimental validation stack (tap spectra, Chladni, static-deflection material characterization, laser speckle interferometry in progress). **Correctness is judged against her measurements, not published tables and not by you.** Don't explain acoustics measurement or material characterization to her; implement.

## Stack (decided)

- Blender **5.2 LTS**, extensions system (`blender_manifest.toml`), Windows 11, VS Code.
- **CalculiX 2.22**: `C:\Program Files\FreeCAD 1.1\bin\ccx.exe` (standalone-runnable; no `-v` flag, run bare to see the banner). Path lives in add-on preferences, never hardcoded, never PATH-dependent.
- **gmsh + meshio + mesh-repair lib** in a normal system **venv**, called by subprocess with explicit full paths. Never import them into Blender's bundled Python (3.13 in 5.2 LTS).
- Verify CalculiX card syntax against the 2.22 manual before generating; do not write cards from memory.

## Architecture rules

1. **Engine/UI boundary.** The engine (assembly model, mesh repair, gmsh meshing, `.inp` generation, solve, result parsing, verifiers) is a pure Python package with **zero `bpy` imports**, developed and tested with pytest outside Blender. The Blender layer is thin: collect scene → write files → launch engine subprocess → poll via `bpy.app.timers` → import results. Reversing this later is expensive. Guard it.
2. **Intermediate assembly model.** Blender scene → solver-agnostic assembly description → `.inp`. Never generate cards straight from Blender objects. The description holds:
   - **Bodies**: mesh ref, role, material (isotropic, or orthotropic + axis orientation). The rim **lining** is a body like any other (role `Body / Lining`), not a face selection — see "Rim lining" below.
   - **Bonds**: body pairs + interface type. Where adhesive quality must itself be an experimental variable (soundboard-to-lining, and any other glued joint), model the glue as a thin interposed body (its own mesh, thickness, and adhesive material: aliphatic resin, hide glue, etc.), `*TIE`'d rigidly to both adherents — the layer's own flexibility is the bond-stiffness knob, not an ad hoc parameter on the tie itself.
   - **Rim lining**: a Body-role object bonded to the soundboard at the real glue-strip footprint (~6-11 mm). Its own material/geometry governs how much it flexes under load. Whichever of its surface nodes are **not** part of a declared bond get an automatic **terminal condition** (object-level property, no face/node picking): `Fixed` (rigid bench/vice anchor, default), `Free` (unconstrained, floating-mass/string-suspended analogy — produces rigid-body modes like the free-free bare-plate stage; filter the same way), or `Spring` (grounded springs at the unbonded nodes with a stiffness value, approximating partial support). This is forward-compatible with V2+ back/side/air-cavity bodies: once something else is bonded to the lining's outer face, those nodes simply stop being "unbonded" and the terminal condition no longer applies there — no rework needed.
   - **Point attachments**: named locations (bridge feet / compensator points) + grounded spring K and/or applied force
   - **Analysis sequence**: optional static preload step, then eigenmode extraction. Build it expecting more than one analysis type later.
3. **Typed roles are first-class**: soundboard, rib/stiffener, bridge plate, body/lining, force/spring point, mass load. Roles drive defaults, verifiers and operator names. Name operators after guitar concepts ("Place Bridge Feet", "Apply String Preload"), not generic FEA steps. Role assignment (including the lining's terminal condition) must be checkbox/dropdown-easy, no manual face/node picking.
4. **Soundboard is a full 3D solid**, not a shell with thickness. Tapered, CNC-contoured and rib-less relief designs must work without re-architecture. Ribs, bridge plate, lining etc. are separate bodies by default.
5. **Boundary conditions are per-analysis configuration, never a pipeline default.** Free-free is only correct for bare-plate validation runs matching the free-suspended tap rig. The instrument model terminates at the rim lining body (see above), whose terminal condition and own material/geometry together determine how much energy the rim leaks — not a single scalar stiffness.

## Meshing and numerics (known traps)

- Use **quadratic tets (C3D10)**. Linear tets lock in thin bending and give eigenfrequencies that are systematically too high.
- Enforce **>= 2-3 element layers through thickness** (soundboard is ~2 mm). Use a thickness-aware gmsh size field, not one global size. Tapered thickness is its own explicit pass/fail test.
- Orthotropic stiffness must be positive definite; validate the constants before solving (small minor Poisson ratios have caused negative eigenfrequencies in her earlier work). Axis convention: **X = radial, Y = longitudinal/grain, Z = tangential**. Material constants come from her lab-measured library, never hardcoded defaults.
- Point-attachment locations must survive remeshing as identifiable nodes. Register them in gmsh as embedded/constrained points and carry node IDs into the spring/force cards. For Evergreen, these coincide with the Bridge Plate's boss extensions (registered CF-composite features, not arbitrary flat-plate locations) -- an easier case to keep robust through remeshing than an unmarked point.

## Bonding: must be bulletproof (the reason this project exists)

SimScale and FreeCAD FEM both failed her on multi-material bonding via **silent** tie failure. Design the failure out:

1. **Pre-mesh proximity/contact detection** with an explicit, user-visible tolerance, classifying each body pair as bonded / not touching / interpenetrating. Role-aware: a rib bond is a thin strip along the glue face, a bridge-plate bond is a flat area, a lining bond is the rim glue strip (~6-11 mm wide); flag geometry that doesn't match the role's expected contact.
2. **Tie-completeness check** after meshing: every declared bond must produce non-empty paired surfaces on both sides. Fail loudly before calling `ccx`.
3. **Post-solve sanity checks**: with the rim lining's terminal condition applied there should be no unexpected rigid-body modes (unless the terminal condition is deliberately `Free`); a near-zero mode concentrated in one body flags a probable tie failure.
4. Bond only via **facet/surface-based `*TIE`** (no node-matched bonding). Master/slave chosen by a stated rule (coarser/stiffer as master), not by accident.
5. **Visual verification in the viewport**: role colors per body; a "Verify Assembly" operator drawing green/yellow/red bond regions via a `gpu` draw handler (non-destructive); click-to-inspect showing gap, bodies and tolerance in the sidebar. Eigenmodes imported as shape keys so a floating part is visible by eye.

## Scope

**In (v1):** eigenmode analysis of soundboard + bridge plate + rib array; material library (orthotropic spruce proxy, isotropic bridge plate, rib composites) applied per body; rim lining body with its own material/geometry and a per-object terminal condition (fixed / free / spring); two point springs (compensators) at the bridge plate; optional static string-force preload run A/B against no-preload; mass loading; MAC comparison against measured mode shapes; mode-shape viewer.

**Evergreen-specific:** two compensator springs (~2.72 kgf/mm each, ~3 g), bolted to the bridge plate; plate stiffness is largely swamped by them, so they must be in the model. Goal is a **calibrated baseline model of the current instrument** that reproduces measured modal frequencies (M1/T1 about 178-182 Hz, A0 about 98-102 Hz), then use it to test bracing candidates toward M1 about 190-200 Hz.

**Out for now (deferred, keep the analysis-sequence model extensible):** mobility / driven damped response, full-box coupling (back, sides, air cavity), other solvers. The lining's terminal-condition rule (applies only to currently-unbonded nodes) is designed so adding these later doesn't require reworking the rim model.

## Evergreen force path (patent-derived, see `docs/1431-02 - Drawings_r1.pdf`)

The soundboard is in a genuine **no-static-stress regime** by mechanical design, not by simplifying assumption:

- **Longitudinal string tension (~160 lbf x2)** is carried entirely by an internal tension member (a stainless-steel cable) running the full body length under the soundboard, anchored at a rigid Load Bearing Structure at the neck block on one end and at the Force Compensation Structure (anchored to the back) on the other. It never loads the soundboard.
- **The tailpiece is suspended above the soundboard** by string tension and the tension-member cable, not resting on it. The cable doubles around a low-friction bearing (a pulley) at the tail block, converting the string pull into an upward force on the Force Compensation Structure's springs (parabolic leaf springs, CF composite).
- **The bridge is the only point of contact** between this entire tensioning system and the soundboard. It floats (not glued), held down by a net residual transverse force (~36 lbf) that the compensator springs balance against string tension.
- **Round 1 consequence**: skip the `NLGEOM` static-preload -> frequency chain entirely for the Evergreen baseline case; a plain linear modal analysis is physically correct here, not an approximation. Reserve the `NLGEOM` preload chain for the deliberately-detuned "advanced case" (compensator K / preset height tuned so the full-tension equilibrium pose is +1 to 1.5 mm convex instead of neutral).
- **Bridge-to-soundboard interface**: model as bonded (not true floating contact). ~36 lbf of continuous clamping force is far larger than any expected dynamic separation force, so no lift-off/rocking in the small-amplitude eigenmode regime.
- **Point Attachment locations**: the two Bridge Plate boss extensions (CF composite) directly opposite the compensator feet. Bridge feet register against these bosses, not the spruce itself -- load bears on the boss. Apply both compensator mass (~3 g) and spring K there.
- **Maximum dynamic string forcing: +/- 1 lbf** (a deliberately generous outside estimate for aggressive strumming). Tiny relative to the ~36 lbf static clamping preload -- confirms no bridge-separation risk even dynamically, and is the relevant magnitude once driven/forced-response analysis (deferred, see Scope) is built.

## Validation ladder (each stage gates the next)

1. Hand-built isotropic flat-plate `.inp` (ccx invocation + output parsing)
2. Same plate generated programmatically by the engine
3. Blender-authored geometry through repair/remesh, including a **tapered-thickness** variant
4. Orthotropic material on the same geometry (card generation)
5. Real instrument geometry vs. her measured modes via MAC

Don't build stage N+1 on an unvalidated stage N. Keep each reference `.inp` and its expected results as regression tests.

## Working agreements

- Windows hazards: PATH is unreliable for subprocesses launched from Blender; symlinking the dev dir needs Developer Mode; Defender may slow solver/temp-file churn.
- Prefer small, testable increments with failing-loudly checks over clever abstractions. Surface assumptions (tolerances, master/slave choice, mesh sizes) as visible parameters.
- When something is uncertain (a CalculiX card, a gmsh option), say so and check the manual rather than guessing.
