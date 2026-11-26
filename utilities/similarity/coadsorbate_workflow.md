# Co-Adsorbate Similarity Workflow

This stage-2 workflow does five things in sequence:

1. Generate co-adsorption candidates.
2. Estimate a primary REMatch threshold and pre-filter near-duplicates.
3. Relax candidates with MACE while fixing all atoms from the original input structure.
4. Run a secondary REMatch sweep on relaxed minima.
5. Prune very similar/identical minima and write a final DB for DFT.

Shared multi-GPU, restart, and debug behavior is documented in
[`mace_parallel_workflows.md`](mace_parallel_workflows.md).

## Lean Entry Script

Use:

- `utilities/similarity/coadsorbate_workflow.py`

The script is intentionally short and only keeps editable parameters.

## Core Function

The implementation is in:

- `similarity/workflows/coadsorbate_similarity.py`

Main API:

```python
run_coadsorbate_similarity_workflow(
    input_atoms,
    adsorbates=[...],
    candidates_db_path="scratch/coadsorption_candidates.db",
    pre_relax_db_path="scratch/coadsorption_prefilter.db",
    relaxed_db_path="scratch/coadsorption_relaxed.db",
    final_db_path="scratch/coadsorption_final_for_dft.db",
    single_a_final_db_path="scratch/coadsorption_single_a_final_for_dft.db",
    single_b_final_db_path="scratch/coadsorption_single_b_final_for_dft.db",
    site_probe_report_csv_path="scratch/coadsorption_site_probe_similarity.csv",
    candidate_report_csv_path="scratch/coadsorption_candidate_similarity.csv",
    relaxed_minimum_report_csv_path="scratch/coadsorption_relaxed_minimum_similarity.csv",
    relaxation_report_csv_path="scratch/coadsorption_relaxation_report.csv",
    summary_csv_path="scratch/coadsorption_workflow_summary.csv",
    summary_txt_path="scratch/coadsorption_workflow_summary.txt",
    candidate_reference_row_id=1,
    relaxed_minimum_reference_row_id=1,
    site_probe_symbol="H",
    site_probe_similarity_cutoff=0.99,
    single_mode_duplicate_cutoff=0.99,
    candidate_duplicate_cutoff=None,
    candidate_suggested_cutoff_scale=1.0,
    relaxed_minimum_duplicate_cutoff=None,
    relaxed_minimum_suggested_cutoff_scale=1.0,
    mace_model_path="mace-omat-0-medium.model",
    generation_kwargs={...},
    similarity_kwargs={...},
    show_final=False,
    verbose=True,
)
```

## Inputs

- `input_atoms`: ASE `Atoms` structure (slab/support or cluster-only).
- `adsorbates`: list of adsorbates accepted by the co-adsorbate generator (`str` and/or ASE `Atoms`).

## Outputs

- Candidates DB: all generated structures.
- Pre-relax DB: structures retained after primary de-duplication.
- Relaxed DB: MACE-relaxed structures with support fixed and adsorbates relaxed.
- Final DB: de-duplicated relaxed minima ready for DFT.
- Single-A and single-B final DBs: the already relaxed, validity-filtered, and de-duplicated support+adsorbate complexes ready for DFT.
- Candidate and relaxed-minimum similarity CSV reports.
- Site-probe CSV report with original site indices, keep/prune decisions, closest retained sites, and similarities.
- Relaxation CSV report with energy, maximum force, convergence, `nsteps`,
  `force_evaluations`, and `elapsed_seconds`.
- Workflow summary in CSV and TXT (including relaxed-minimum pruning mappings).

## Processing order

1. Detect geometrically plausible adsorption sites near the requested origin.
2. Place one temporary probe atom, reject bond-invalid placements, and remove similarity-equivalent sites.
3. Generate and relax each adsorbate independently at the retained sites.
4. Reject broken, detached, or (by default) unconverged single modes; sort the survivors by energy, de-duplicate them, and save A/support and B/support DFT databases.
5. Combine the retained single modes and de-duplicate the unrelaxed combinations.
6. Relax the combinations, apply validity filters, sort by energy, and de-duplicate the relaxed minima.
7. Restore the input constraints and write the final DFT database.

Cutoff names follow one rule: `*_duplicate_cutoff` means structures with similarity greater than or equal to the value are duplicates. Lower values prune more. `*_suggested_cutoff_scale` is used only when the corresponding fixed cutoff is `None`.

If an A+B pair clashes at coincident or near-coincident sites, pairing moves A and B by equal increments along outward, diverging lateral directions from the full-system center of mass. It stops at the first geometry where the adsorbates are no longer mutually bonded and the full bonding screen passes. Recovered candidates are marked `sequential_pairing_with_lateral_spacing` with `a_lateral_shift_A` and `b_lateral_shift_A` in database metadata.

Candidate metadata also stores the A/B atom counts. Post-relaxation integrity and support-binding checks use those boundaries: internal bonds are validated independently for A and B, and each adsorbate must contact the support. New bonds between A and B do not by themselves count as internal adsorbate damage.

## Parameter migration

| Previous name | Current name |
|---|---|
| `reference_row_id` | `candidate_reference_row_id` |
| `secondary_reference_row_id` | `relaxed_minimum_reference_row_id` |
| `similarity_threshold` | `candidate_duplicate_cutoff` |
| `primary_similarity_relaxation=x` | `candidate_suggested_cutoff_scale=1+x` |
| `secondary_similarity_threshold` | `relaxed_minimum_duplicate_cutoff` |
| `secondary_similarity_relaxation=x` | `relaxed_minimum_suggested_cutoff_scale=1+x` |
| `single_preoptimisation_similarity_*` | `site_probe_*` and `single_mode_duplicate_cutoff` |
| `primary_report_csv_path` | `candidate_report_csv_path` |
| `secondary_report_csv_path` | `relaxed_minimum_report_csv_path` |

## Pairing With TDDFT AL

Use:

- `utilities/similarity/batch_tddft_from_similarity.py`

This example reads `coadsorption_final_for_dft.db` and runs one
single-structure TDDFT active-learning optimisation per DB row in separate
working directories (`tddft_batch/row_<id>/`), making it straightforward
to chain this workflow into DFT/TDDFT analysis. The batch summary also compares
initial MACE energies against final DFT energies using relative-energy columns:
`rel_mace_energy_eV`, `rel_dft_energy_eV`, `rel_energy_shift_eV`.

Explicit example:

```python
import os
import sys

sys.path.append(os.getcwd())

from utilities.similarity.batch_tddft_from_similarity import (
    run_batch_tddft_from_similarity_db,
)

summary_rows = run_batch_tddft_from_similarity_db(
    similarity_db_path="scratch/coadsorption_final_for_dft.db",
    batch_root="tddft_batch",  # kept outside scratch for easier rsync exclusions
    row_ids=None,          # e.g. [1, 2, 5] for a subset
    max_structures=10,     # None -> run all rows
    skip_completed=True,
    workflow_kwargs={
        "n_iter": 1,
        "bond_scan": None,
        "al_steps_per_iter": 100,
        "use_tddft_optimisation": True,
        "xc_functional": "pbe0",   # or "cam-b3lyp"
        "solvent": "Water",        # or e.g. "Methanol", "CPCM(Acetonitrile)", or None
        "use_scan_geometry": True,
        "optimize_to_convergence": True,
        "max_al_steps_per_iter": None,  # None/<=0 disables total per-iteration cap
        "fmax": 0.05,
        "initial_fmax": 0.01,
        "run_initial_dft_if_missing": True,
        "ensure_initial_dft_converged": True,
    },
    summary_csv_path="tddft_batch/batch_summary.csv",
)
```

## Notes

- The workflow deletes existing output files at the selected output paths before writing new results.
- Set `verbose=True` for stage and 10%-milestone progress. Set
  `MACE_WORKFLOW_DEBUG=1` for one line per completed optimisation with steps,
  force evaluations, and elapsed time.
- Relaxed structures are validity-filtered, sorted by increasing MACE energy, then greedily de-duplicated. This preserves the lowest-energy representative of each similarity group.
- `require_converged=True` excludes unconverged single modes and combined candidates. Single modes must also remain intact and bound.
- A fixed `*_duplicate_cutoff` overrides automatic selection. Otherwise the suggested cutoff is multiplied by the corresponding `*_suggested_cutoff_scale` and capped at `0.999999`.
- Checkpoints include the input geometry and relaxation settings; stale rows are recalculated rather than silently reused.
- By default the workflow points to `similarity/run/mace-omat-0-medium.model`.
- `slab_shave_distance` in `similarity_kwargs` is optional (`None` disables shaving).
- `z_cutoff` in `generation_kwargs` is optional (`None` disables explicit cutoff; leave `z_buffer=None` to fully disable z filtering).
- Set `origin_atom_index` (0-based) and `site_cutoff` (Angstrom) together in `generation_kwargs` to generate candidates only at sites near one input atom.
- With two sequentially paired adsorbates, `site_probe_symbol` is placed once at each detected site. Bond-invalid placements are removed first, then sites with probe similarity at or above `site_probe_similarity_cutoff` are treated as duplicates.
- Inspect `site_probe_report_csv_path` when tuning the probe cutoff; `max_similarity_to_kept` records the similarity that caused each rejected site to be pruned.
- Set `keep_bound_only=True` to discard relaxed structures whose adsorbate components have no covalent-radius contact with the support.
- `skip_abnormal_bond_screening` in `generation_kwargs` is optional (`True` keeps only the 0.5 Angstrom quick clash check and skips radii-based abnormal-bond screening).
- Shared similarity keywords across suggestion and filtering are:
  `support_species`, `slab_shave_distance`, `kernel_gamma`, `kernel_alpha`, `kernel_threshold`, `exclude_self`.

## Default MACE Machine

Default target for co-adsorbate MACE screening:

- SSH: `ssh <username>@<hpc-host>`
- Scheduler: SLURM (`main` partition)
- Constraint: this machine should be used for MACE-only jobs.

Use `mace_device="cuda"` in `utilities/similarity/coadsorbate_workflow.py` when running via GPU allocation.

Suggested submission script:

```bash
#!/bin/bash
#SBATCH -J testgpu
#SBATCH -p main
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive

# Load the Python venv
. /srv/research/mace_venv/bin/activate

# Run the job
input="input.py"
time python -u "$input" > "log.${SLURM_JOB_ID}"
```

You can also use the checked-in template:

- `utilities/similarity/sub_mace_local_main.slurm`

Submission example:

```bash
ssh <username>@<hpc-host>
cd /path/to/repo
sbatch utilities/similarity/sub_mace_local_main.slurm
```
