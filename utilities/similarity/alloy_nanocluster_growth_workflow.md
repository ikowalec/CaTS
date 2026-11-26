# Alloy Nanocluster Growth Workflow

This extends the single-metal nanocluster growth workflow to alloys.

Shared multi-GPU, restart, and debug behavior is documented in
[`mace_parallel_workflows.md`](mace_parallel_workflows.md).

At each generation it:

1. Takes the kept parent structures from the previous generation.
2. Adds one atom from `--add-symbols` to each parent.
3. Removes near-duplicate structures with REMatch.
4. Relaxes candidates with MACE.
5. Removes near-duplicates again.
6. Ranks by bulk mixing energy when staged bulk references are available,
   otherwise by atomisation energy, never raw total energy across compositions.
7. Keeps the lowest-energy structures per composition.
8. Optionally creates three rattled copies of each kept parent, reoptimizes them,
   and uses the lowest-energy result when it beats the original parent.

Atomization energy is normalized per atom for ranking:

```text
E_atomization = sum(n_i * E_gas_atom_i) - E_cluster
score = E_atomization / N_atoms
```

`E_gas_atom_i` is calculated once from single-atom MACE energies for every
species in the seed and `--add-symbols`.

## Quick Start

Use the project virtual environment:

```powershell
.venv\Scripts\python.exe utilities\similarity\alloy_nanocluster_growth_workflow.py seed.xyz --add-symbols Au Cu --n-additions 8 --keep-per-composition 10 --mace-model-path similarity\run\mace-omat-0-medium.model
```

If no seed structure is provided, the workflow starts separate runs from one
atom of each `--add-symbols` species:

```powershell
.venv\Scripts\python.exe utilities\similarity\alloy_nanocluster_growth_workflow.py --add-symbols Au Cu --n-additions 8 --keep-per-composition 10 --mace-model-path similarity\run\mace-omat-0-medium.model
```

On Linux/macOS:

```bash
.venv/bin/python utilities/similarity/alloy_nanocluster_growth_workflow.py seed.xyz --add-symbols Au Cu --n-additions 8 --keep-per-composition 10 --mace-model-path similarity/run/mace-omat-0-medium.model
```

## Staged Beam Search

Staged mode explores every composition until `--staging-start-atoms`, then
periodically restarts from the globally best terminal structures. Each retained
seed gets its own concentration-limited branch. Branches are combined and
deduplicated at the next dynamically calculated checkpoint.

```powershell
.venv\Scripts\python.exe utilities\similarity\alloy_nanocluster_growth_workflow.py `
  --add-symbols Au Cu `
  --max-atoms 60 `
  --staging-start-atoms 30 `
  --concentration-symbol Au `
  --concentration-span 0.05 `
  --restart-beam-width 3 `
  --mace-model-path similarity\run\mace-omat-0-medium.model `
  --mace-device cuda `
  --output-root scratch\aucu_staged_beam_search
```

For a 30-atom `Au10Cu20` restart seed, a +/-5 percentage-point window requires
six additions, so its first staged checkpoint is 36 atoms. Integer atom counts
are rounded to the nearest representable concentration and the requested and
effective bounds are recorded in `campaign.json`. Use
`--strict-concentration-span` to exclude the nearest grid point when it falls
just outside the requested range.

`--restart-beam-width 3` retains up to three globally ranked, structurally
distinct terminal structures. Distinct compositions are preferred before a
second structure from the same composition is selected. Pure-element candidates
are excluded by default; use `--staging-allow-pure` to include them.

New staged checkpoint and parent selections target atomisation energy per atom
by default. Use `--target-mixing-energy` to instead select the lowest
bulk-referenced mixing energy per atom when optimized elemental bulk references
are available. `selection_criterion.json`, `ranked.csv`, and
`checkpoint_*/restart_ranking.csv` record the criterion and score. Published
checkpoints are reused without retroactive reselection. If bulk references are
unavailable or the system contains support atoms, the workflow prints a clear
`STRUCTURE SELECTION CRITERION: atomization_energy_per_atom_eV` message and
records that fallback.

The directly runnable seedless example is
`utilities/similarity/alloy_staged_beam_search_example.py`.

## Main Options

- `seed_structure`
  - Starting cluster readable by ASE.
  - Optional. If omitted, generation 1 reuses each unique `--add-symbols`
    gas-atom reference in one output tree; those atoms become generation-2 parents.
- `--add-symbols Au Cu`
  - Species allowed to be added at every step.
- `--n-additions 8`
  - Number of growth generations.
- `--max-atoms 60`
  - Final alloy atom count for staged mode. If omitted, staged mode derives it
    from `--n-additions`.
- `--staging-start-atoms 30`
  - Enables staged mode after unrestricted growth reaches this atom count.
- `--concentration-symbol Au`
  - Element whose concentration defines each staged branch window.
- `--concentration-span 0.05`
  - Absolute concentration half-width; `0.05` means +/-5 percentage points.
- `--restart-beam-width 3`
  - Number of globally best checkpoint structures retained as independent seeds.
- `--target-mixing-energy`
  - Opts in to bulk-referenced mixing energy as the growth and staging target.
    Atomisation energy per atom is the default target.
- `--keep-per-composition 10`
  - Number of relaxed structures retained for each composition bucket.
- `--min-per-composition 1`
  - Prints a warning if a bucket has fewer retained structures.
- `--max-structures-per-parent-symbol`
  - Caps candidate generation per parent per added species.
- `--similarity-threshold 0.9999`
  - One-alloy-atom baseline for the primary REMatch duplicate threshold.
- `--secondary-similarity-threshold`
  - Optional one-alloy-atom baseline for the post-relaxation REMatch threshold.
  - Each threshold tightens automatically with alloy size as
    `1 - (1 - baseline) / alloy_atom_count`; support atoms are excluded.
- `--similarity-batch-size 100`
  - REMatch batch size for both pruning stages. Use `0` to disable batching.
- `--no-mace-enable-cueq`
  - Disables MACE cuEquivariance acceleration, for example when it is incompatible
    with the MACE model or installation on the target HPC.
- `--rattle-parent`
  - Optional. Creates and reoptimizes three rattled copies of each selected parent.
- `--parent-rattle-stdev 0.3`
  - Standard deviation of the ASE displacement in angstrom.
- `--parent-rattle-fmax 0.01`
  - Force convergence threshold in eV/angstrom for rattled-parent reoptimization.
  - Normal candidate relaxation uses `--mace-fmax 0.05` by default.

## Outputs

The default output folder is `scratch/alloy_nanocluster_growth`.

Each generation writes:

- `candidates.db`
- `pre_relax.db`
- `relaxed.db`
- `final_unique.db`
- `primary_similarity.csv`
- `secondary_similarity.csv`
- `relaxation.csv`
- `ranked.csv`
- `selected.xyz`

With `--rattle-parent`, every non-final generation also writes
`selected_pre_rattle.xyz` and a `parent_rattle/` folder containing
`rattled_frames.db`, `reoptimized.db`, `reoptimization.csv`, `summary.csv`, and
`selected_parents.db`.

Full MACE optimisation trajectories are not saved. `relaxation.csv` and
`parent_rattle/reoptimization.csv` record endpoint energy, force, convergence,
`nsteps`, `force_evaluations`, and `elapsed_seconds`. Set
`MACE_WORKFLOW_DEBUG=1` for one coordinator log line per completed optimisation.

The top-level folder also writes:

- `alloy_growth_summary.csv`
- `stable_alloy_summary.csv`
- `selected_alloy_growth.db`
- `selected_alloy_growth.traj`
- `stable_alloy_structures.db`
- `stable_alloy_structures.traj`
- `atomization_energy_phase_scatter.png`
- `atomization_energy_phase_lines.png`
- `atomization_energy_phase_lines.html`
- `mixing_energy_phase_lines.png`

Staged campaigns additionally write `campaign_config.json`, `campaign.json`,
`current_restart_seeds.db`, checkpoint folders, independent stage/branch folders,
and combined stable structure DB/trajectory/CSV files. They also create:

- `explored_compositions.csv` and `explored_compositions.png`, combining every
  composition present in every stage's `ranked.csv` file.
- `stable_coordination_summary.csv`, with total metal-metal, heterometal, and
  directional mean coordination numbers for every most-stable structure.
- `stable_atomization_vs_cross_coordination.png` and the corresponding
  `stable_atomization_vs_<M1/M2>_coordination.png` plots, retaining the
  atomisation-energy views.
- `stable_bulk_mixing_energy_vs_cross_coordination.png`, plotting
  bulk-referenced mixing energy per atom against mean M1-M2 coordination.
- `stable_bulk_mixing_energy_vs_<M1>_coordination.png` and
  `stable_bulk_mixing_energy_vs_<M2>_coordination.png`, resolving the mean
  coordination number of each metal separately.
- `bulk_mixing_energy_phase_diagram.png`, plotting all bulk-referenced mixing
  energies against composition, with alloy size shown by colour.
- `bulk_mixing_energy_concentration_minima.png`, retaining only the lowest
  mixing energy at each visited concentration and labelling points with their
  alloy atom counts. The selected rows are retained in the matching CSV.

Both the atomisation and bulk-mixing coordination plots use alloy atom count as
the colour dimension. The same coordination CSV and both plot families are also
written into every `prestage_*` and `stage_*` folder, using only the most stable
structures retained within that stage.

Before producing these plots, staged mode builds each element's stable reference
crystal with ASE `bulk()`, verifies that the selected MACE model provides stress,
and relaxes the atoms and unit cell with ASE `ExpCellFilter` and BFGS. The final
structures, energies, settings, and convergence information are retained under
`bulk_references/`; optimisation trajectories are not written. Use
`--bulk-fmax` and `--bulk-max-steps` to control these two small optimisations.
If stress is not implemented, `bulk_reference_status.json` records that outcome
and bulk-referenced plots are skipped.

The plotted energy is

```text
E_mix,bulk = E_cluster - n_M1 * E_bulk(M1)/N_M1 - n_M2 * E_bulk(M2)/N_M2
```

and is divided by the alloy atom count. Lower values are more stable. This
quantity is only generated for unsupported alloy clusters; elemental bulk
references alone cannot consistently remove a support's energy.

The coordination cutoff is 1.35 times the sum of the atoms' covalent radii.
Input branch folders are never merged in place. Restarting a completed staged
campaign with the same command reuses its stored single-atom references and
combined stable outputs, skips MACE growth, and regenerates these analysis files
from the retained data.

`stable_alloy_summary.csv` has one row per most stable size/composition bucket.
For example, filter `n_atoms == 20` to inspect the most stable
structure for each 20-atom alloy composition. `stable_structure_id` matches the
order in `stable_alloy_structures.db` and `stable_alloy_structures.traj`.
It records `atomization_energy_eV` and `atomization_energy_per_atom_eV` from
gas-atom references. It also records `mixing_energy_eV` and
`mixing_energy_per_atom_eV` when same-size pure-metal references exist.

`atomization_energy_phase_scatter.png` plots atomization energy per atom against the
last `--add-symbols` species concentration. Colour shows mixing energy per atom
when it is available for every point, otherwise cluster size.
`atomization_energy_phase_lines.png` connects the most stable compositions with the
same atom count. `atomization_energy_phase_lines.html` is the same view as an
interactive Plotly plot when Plotly is installed.

`mixing_energy_phase_lines.png` uses the most stable pure-metal
nanoparticles discovered during growth as the endpoints for each cluster size:

```text
E_mix,NP = E_alloy - sum(n_i * E_pure(i, N) / N)
```

For example, the reference for Au5Cu5 is `5 * E(Au10) / 10 + 5 * E(Cu10) / 10`.
Sizes without a discovered pure nanoparticle for every alloy element are omitted.

## Notes

- This is still a greedy growth workflow. It keeps a frontier of stable structures per composition, not every possible alloy ordering.
- Atomization reference energies are single-atom MACE energies calculated at startup.
- Matching relaxation rows are reused on restart; checkpoint databases are
  atomically published by rank 0.
- If co-adsorbate placement fails for a cluster, the workflow falls back to simple radial atom addition.

## Self Check

This does not run MACE. It checks atomization ranking and mixing-energy helpers:

```powershell
.venv\Scripts\python.exe utilities\similarity\alloy_nanocluster_growth_workflow.py --self-check
```
