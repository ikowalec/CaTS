# Zeolite COM Co-Adsorbate Workflow

This workflow targets in-cage adsorption screening for zeolites using a user-provided cage center point (`cage_com`) instead of ring indices.

## Suggested Adjustments (Implemented)

1. COM-first API:
`cage_com=[x, y, z]` is now the primary input, so you do not need ring atom indices.

2. Backward-compatible fallback:
`ring_indices` is still accepted as optional fallback to compute COM when needed.

3. Site separation control:
`min_site_separation` can reject multi-adsorbate placements where selected framework sites are too close.

4. In-cage validation after rough relaxation:
`enforce_inside_target_radius` plus `inside_radius_tolerance` filters out structures that drift outside the inferred cage boundary during constrained MACE.

5. Initial overlap resolution:
`resolve_initial_overlaps` plus `overlap_resolution_max_steps` pushes adsorbates inward and separates clashes before MACE starts.

6. Robust constrained MACE retries:
If `FixInternals` fails to converge, the workflow retries with softer constraints (`internals_only`, then `framework_only`) and logs the used mode.

## Entry Script

Use:

- `utilities/similarity/zeolite_com_workflow.py`

## Core API

Implementation:

- `similarity/workflows/zeolite_ring_adsorbate.py`

Recommended entry point:

```python
run_zeolite_com_similarity_workflow(
    zeolite_atoms,
    cage_com=[x, y, z],
    adsorbates=[...],
    ...
)
```

Compatible entry point (supports both `cage_com` and fallback `ring_indices`):

```python
run_zeolite_ring_similarity_workflow(
    zeolite_atoms,
    cage_com=[x, y, z],  # preferred
    ring_indices=None,   # optional fallback only
    adsorbates=[...],
    ...
)
```

## Workflow Stages

1. Select framework candidate sites (`site_symbols`) within `target_radius` of `cage_com`.
2. Infer cage boundary radius from nearest framework atoms around `cage_com` (default behavior).
3. Place adsorbates from each site in the direction of COM by `insertion_distance`.
4. Shift adsorbate groups inward to respect the inferred cage boundary (`cage_padding`).
5. Resolve initial overlaps:
enforce minimum framework-adsorbate distance and optional adsorbate-adsorbate separation before pre-screening.
6. Remove near-duplicates via REMatch similarity pre-screening.
7. Run constrained MACE relaxation:
framework atoms fixed, adsorbate internals constrained (`FixInternals`) with automatic fallback retries.
8. Keep structures that are physically plausible and (optionally) still inside cage radius.
9. Run final MACE relaxation:
framework fixed, adsorbates fully relaxed.

## Key Controls

- `target_radius`: max distance from COM for candidate site selection only.
- `max_site_candidates`: cap number of nearest candidate sites from the pool already filtered by `target_radius`.
- `use_framework_cage_boundary`: use nearest framework atoms to determine cage boundary size (default: `True`).
- `cage_boundary_n_nearest`: number of nearest framework atoms used to infer cage boundary radius.
- `cage_boundary_margin`: subtractive safety margin from inferred cage boundary radius.
- `relax_cage_boundary_if_empty`: if inferred boundary yields zero candidates, retry once with a relaxed boundary.
- `relaxed_cage_boundary_radius`: relaxed boundary radius for fallback retry (`None` means use `target_radius`).
- `max_structures`: cap total generated structures.
- `min_site_separation`: minimum distance between selected framework sites for multi-adsorbate placements.
- `min_initial_adsorbate_framework_distance`: initial clash filter before relaxation.
- `min_initial_adsorbate_adsorbate_distance`: optional minimum separation between different adsorbates during initial placement and pre-relax overlap resolution.
- `enforce_nonbonded_coadsorbates`: require different adsorbates to remain non-bonded to each other.
- `nonbonded_coadsorbate_scale`: non-bond criterion scale for inter-adsorbate atom pairs, using `distance/(r_cov_i+r_cov_j)`.
- `resolve_initial_overlaps`: enable iterative geometric pushing/separation before MACE.
- `overlap_resolution_max_steps`: max iterations for overlap resolution.
- `placement_progress_interval`: print placement-stage progress every N processed candidates.
- `overlap_detailed_logging`: print per-step overlap-resolution updates during placement stage.
- `overlap_progress_interval`: per-step logging interval when `overlap_detailed_logging=True`.
- `min_physical_adsorbate_framework_distance`: post-relax framework-adsorbate physical filter.
- `enforce_inside_target_radius`: drop structures outside cage radius after constrained stage.
- `relax_inside_radius_filter_if_empty`: if no structure passes both `physical_ok` and in-radius filter, fallback to `physical_ok` only.
- `inside_radius_tolerance`: allowed overshoot beyond inferred cage boundary radius in radius validation.
- `constrain_adsorbate_angles`: include bond-angle `FixInternals` constraints. Default is now `False` for stability.
- `adsorbate_bond_constraint_mode`: choose bond constraints for internals: `nearest` (recommended) or `all`.
- `crude_mace_maxstep`: maximum per-step geometry displacement for constrained MACE stage.
- `final_mace_maxstep`: maximum per-step geometry displacement for final MACE stage.
- `fixinternals_epsilon`: optional ASE `FixInternals` solver tolerance (`None` uses ASE default).
- `fixinternals_maxiter`: optional ASE `FixInternals` max internal iterations (`None` uses ASE default).
- `fixinternals_relaxed_retry`: if `True`, run an additional internals retry with relaxed solver settings before falling back to framework-only.
- `fixinternals_relaxed_epsilon`: relaxed retry solver tolerance (default `1e-4`).
- `fixinternals_relaxed_maxiter`: relaxed retry solver max iterations (default `800`).
- `fixinternals_relaxed_step_scale`: scale applied to constrained `maxstep` for relaxed retry attempt (default `0.5`).

## New Additions (Fully Documented)

### 1) Initial overlap handling

- `resolve_initial_overlaps` (default: `True`)
  - Runs iterative geometric cleanup before similarity screening and again before each MACE stage.
  - Shifts adsorbates toward cage interior, pushes them away from framework clashes, and optionally separates adsorbate-adsorbate overlaps.
- `overlap_resolution_max_steps` (default: `24`)
  - Maximum number of overlap-resolution iterations.
  - Increase for crowded pores or larger adsorbates.
- `min_initial_adsorbate_adsorbate_distance` (default: `0.8`, or `None` to disable)
  - Minimum allowed closest-contact distance between different adsorbates during initial placement and pre-MACE cleanup.
- `enforce_nonbonded_coadsorbates` (default: `True`)
  - Ensures atoms from different adsorbates are not within bond-like distances.
- `nonbonded_coadsorbate_scale` (default: `1.10`)
  - Enforces `distance/(r_cov_i+r_cov_j) >= scale` for all inter-adsorbate atom pairs.
  - Increase this value for stricter separation.

### 1b) Decoupled site radius vs cage boundary

- Site selection is controlled by `target_radius`.
- Cage-boundary geometry control (placement/clamping/in-cage checks) is controlled by inferred framework radius when `use_framework_cage_boundary=True`.
- The inferred boundary uses `cage_boundary_n_nearest` nearest framework atoms to `cage_com`, then applies `cage_boundary_margin`.
- If this inferred boundary is too tight and produces zero candidates, optional fallback retry can use `relaxed_cage_boundary_radius` (or `target_radius` when `None`).

### 2) Robust constrained-MACE fallback

For constrained stage (`constrain_adsorbate_internals=True`), optimization now automatically retries if a constraint setup fails (including `FixInternals.adjust_positions did not converge`):

If `constrain_adsorbate_angles=True`:

1. `internals+angles`
2. `internals+angles_relaxed_solver`
3. `internals_only` (drop angle constraints)
4. `internals_relaxed_solver` (bond-only internals with relaxed `FixInternals` solver settings)
5. `framework_only` (drop all adsorbate internal constraints)

If `constrain_adsorbate_angles=False`:

1. `internals`
2. `internals_relaxed_solver`
3. `framework_only`

This prevents one unstable structure from aborting the whole workflow.

Default behavior now favors stability:

- `constrain_adsorbate_angles=False`
- `adsorbate_bond_constraint_mode="nearest"`

This constrains only nearest bonded distances and avoids angle constraints unless explicitly requested.

For difficult systems where you still want angle constraints, increase `FixInternals` robustness first before removing angles:

- set `fixinternals_maxiter` (for example `300-800`)
- set `fixinternals_epsilon` (for example `1e-5` to `1e-4`)
- keep `fixinternals_relaxed_retry=True`

### 3) Angle-constraint stabilization

- Near-linear angle constraints are skipped when building `FixInternals` (angles outside 10-170 degrees).
- This reduces singular/ill-conditioned internal-coordinate updates for complex adsorbates.

### 4) Runtime progress feedback (placement -> pre-MACE)

When `verbose=True`, the workflow now reports:

- Placement-stage start summary (`site_combinations`, `adsorbate_orders`, estimated candidate count).
- Periodic placement heartbeat with counters:
  - `processed`
  - `kept`
  - `rejected_site_sep`
  - `rejected_overlap`
  - `rejected_distances`
- Placement completion summary.
- Explicit message before similarity pre-screen begins.

Optional detailed overlap-step logging is enabled with:

- `overlap_detailed_logging=True`
- `overlap_progress_interval=<N>`

### 5) Constrained-filter fallback to avoid hard stop

If constrained relaxation returns no structures passing both:

- `physical_ok=True`
- `inside_target_radius=True`

and `relax_inside_radius_filter_if_empty=True`, the workflow automatically falls back to:

- `physical_ok=True` (radius filter relaxed for this transition)

This avoids aborting due only to temporary radius drift after constrained stage.

### 6) Small-step MACE stabilization

To reduce “molecule exploding” behavior during optimization, the workflow now uses small BFGS `maxstep` values by default:

- `crude_mace_maxstep=0.01`
- `final_mace_maxstep=0.01`

If needed, tighten further (for example `0.01-0.02`). These are passed directly to `BFGS.maxstep`.

## Outputs

The workflow writes:

- Candidate DB.
- Similarity-prefilter DB.
- Constrained-relaxed DB.
- Final-relaxed DB.
- Similarity/constrained/final CSV reports.
- Summary TXT.
- Optional MACE trajectories/logs directory.

Each DB row includes placement metadata (sites, COM, placement labels, distances) and relaxation metadata:

- `energy_eV`, `max_ads_force_eV_A`, `converged`
- `physical_ok`, `inside_target_radius`
- constrained filter diagnostics in summary:
  - total count
  - `physical_ok` count
  - `inside_target_radius` count
  - `physical_and_inside` count
  - filter mode used (`physical_and_inside`, `physical_only`, `fallback_physical_only`)
- `relax_status` (`internals+angles`, `fallback:internals_only`, `fallback:framework_only`, or `failed`)
- `optimisation_error` (exception text when an attempt fails)
- pre-relax overlap diagnostics (`pre_relax_*` fields)

New/extended metadata fields written by this update:

- Candidate metadata:
  - `target_radius` (site selection radius)
  - `cage_boundary_radius`
  - `initial_min_adsorbate_adsorbate_distance`
  - `initial_min_inter_adsorbate_bond_ratio`
  - `initial_overlap_resolved`
  - `initial_overlap_resolution_steps`
  - `initial_nonbonded_coadsorbates_ok`
  - `enforce_nonbonded_coadsorbates`
  - `nonbonded_coadsorbate_scale`
- Constrained/final relaxation metadata:
  - `relax_status`
  - `used_adsorbate_internals`
  - `used_adsorbate_angles`
  - `used_fixinternals_epsilon`
  - `used_fixinternals_maxiter`
  - `used_optimizer_maxstep`
  - `optimisation_error`
  - `pre_relax_overlap_resolved`
  - `pre_relax_min_framework_adsorbate_distance`
  - `pre_relax_min_adsorbate_adsorbate_distance`
  - `pre_relax_overlap_resolution_steps`
  - `pre_relax_min_inter_adsorbate_bond_ratio`

These fields are included in output DB rows and in constrained/final CSV reports.

## Run

```bash
python utilities/similarity/zeolite_com_workflow.py
```
