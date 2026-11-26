# Submission Webpage Guide

This guide documents every option on `/jobs/submit/`, what it does, and which workflow modes use it.

## Workflow Modes

Reaction Hub submit supports these run modes:

- `Single optimisation`
- `TDDFT bond-length scan workflow`
- `Charge-transfer spectrum (Multiwfn)`
- `Active-learning NEB / TDDFT MLNEB`
- `Co-adsorbate placement workflow` (screening)

High-level behavior:

- Single optimisation:
  - runs one optimisation (`n_iter=1`)
  - disables TDDFT optimisation path
  - can continue in the same job with a final TDDFT spectrum, isolated-cluster
    difference spectrum, and Multiwfn charge-transfer spectrum
  - enabling charge transfer automatically enables the final ORCA TDDFT calculation;
    fragment indices are entered as ASE/Python 0-based indices
- TDDFT bond scan:
  - runs iterative scan logic with bond constraints and TDDFT IROOT tracking
- Charge-transfer spectrum:
  - starts from a completed existing ORCA workflow run
  - the workflow-page `Request CT Spectrum` button submits immediately with the
    source run's workflow, HPC, and TDDFT settings
  - infers the metal nanocluster as fragment 1 (`NC`) and each disconnected,
    covalently bonded non-metal component as its own labelled adsorbate fragment
    (for example `CO` and `O`, rather than one formula-only `CO2` fragment)
  - converts ASE/Python 0-based indices to Multiwfn 1-based fragments, runs one
    multi-fragment IFCT calculation, and retains replot-ready numerical data
  - produces one combined directional plot (`NC→CO`, `CO→NC`, `NC→O`, etc.)
    plus a separate bidirectional plot for every fragment pair; all CT plots use
    the retained final-TDDFT wavelength range and show resolved excitation lines
  - tags the follow-up run as `charge_transfer`
- Co-adsorbate placement:
  - runs the MACE screening pipeline
  - uses reference cluster + adsorbate inputs
  - ignores most TDDFT/single-only optimisation controls

## Core Selection

These are always shown at the top of the form.

| Field | Meaning | Used By |
| --- | --- | --- |
| `Run Name` | Unique run label; letters/numbers/`._-` only. | All modes |
| `Run Mode` | Selects workflow behavior and visible controls. | All modes |
| `Workflow` | Python workflow definition to execute. | All modes |
| `HPC System` | Target Slurm cluster + SSH credentials. | All modes |
| `Settings Profile` | DFT/TDDFT chemistry defaults (functional/basis/solvent). | All modes |
| `Reference Type` | Mark run as reusable reference (`adsorbate` or `nanocluster`). | Single + TDDFT |
| `Reference Name` | Name for created/reused reference entity. | Single + TDDFT |

Notes:

- Workflow choices are filtered by mode.
- Co-adsorbate mode always forces reference intent to `none`.

## Geometry Section

### Geometry Source

| Option | Meaning | Used By |
| --- | --- | --- |
| `Upload structure file` | Upload local `.xyz/.traj/.db/.cif/.pdb` or VASP `POSCAR`/`CONTCAR` filenames. | Single + TDDFT + Co-adsorbate |
| `ASE molecule builder` | Build from ASE molecule name/formula. | Single + TDDFT + Co-adsorbate |
| `PubChem lookup` | Fetch structure by CID or query. | Single + TDDFT + Co-adsorbate |
| `Use structure from existing workflow run` | Uses an artifact selected from a prior run. | Single + TDDFT + Co-adsorbate |

Fields in geometry card:

| Field | Meaning | Used By |
| --- | --- | --- |
| `Structure File` | Uploaded geometry file. | Upload source |
| `Trajectory Frame` | Frame index for `.traj/.db` (`-1` = last). | Upload source |
| `Submit all trajectory frames` | Submit one job per frame (`run_name_0`, `run_name_1`, ...). | Upload/Existing run with `.traj/.db` |
| `ASE Molecule Formula/Name` | ASE molecule identifier. | ASE source |
| `PubChem CID` | Numeric PubChem compound ID. | PubChem source |
| `PubChem Query` | Text query when CID is not provided. | PubChem source |
| Existing-run hidden IDs | Internally selected source run + artifact IDs. | Existing-run source |

Important rules:

- Batch frame submission is only valid for `.traj`/`.db`.
- Batch submission cannot be combined with reference tagging.
- In co-adsorbate mode, geometry source defines the base structure when provided.
- In co-adsorbate mode, a reference cluster can be used as fallback base structure when geometry source inputs are left empty.

## Workflow Controls

### Common Controls

| Field | Meaning | Used By |
| --- | --- | --- |
| `fmax` | Force convergence threshold (eV/A). | Single + TDDFT |
| `Active-learning Optimiser` | Local optimiser (`BFGS`/`FIRE2`). | Single + TDDFT |
| `Spin Multiplicity` | Optional ORCA spin multiplicity override. | Single + TDDFT |

### TDDFT Controls

| Field | Meaning | Used By |
| --- | --- | --- |
| `N Iter` | Number of workflow scan iterations. | TDDFT |
| `Use TDDFT Optimisation` | If enabled, optimisation stages use TDDFT engrad path. | TDDFT |
| `Target Wavelength (nm)` | Target wavelength for IROOT scoring. | TDDFT |
| `TDDFT Scan NRoots` | `NRoots` used in TDDFT scan sweeps before IROOT selection. | TDDFT |

TDDFT defaults:

- `Target Wavelength (nm)` default: `520.0`
- `TDDFT Scan NRoots` default: `50`

### Charge-Transfer Controls

Open a completed workflow and click `Request CT Spectrum`. The source run must remain available on the selected HPC system and contain either `<basename>.molden.input`, or `<basename>.gbw` plus `<basename>.out`.

| Field | Meaning |
| --- | --- |
| `Nanoparticle Fragment (0-based)` | Inclusive ranges or comma-separated ASE indices, for example `0-62`. |
| `Adsorbate Fragment (0-based)` | ASE indices for the adsorbate, for example `63,64`. |
| `CT Spectrum NRoots` | State limit used for plotting; default `400`. |
| `ORCA Basename` | Source ORCA basename; default `orca`. |

All Slurm fields remain editable. One CT request is still one serial analysis; this page does not implicitly pack unrelated requests into a task farm.

### Single-Optimisation Behavior

Single mode ignores TDDFT scan controls and enforces:

- `n_iter=1`
- `use_tddft_optimisation=False`

Single mode can still apply optional optimisation constraints from submit:

- `Fixed Symbols` (FixAtoms by symbol)
- optional constrained bond via `Bond Index i/j`
- optional `Target Distance (A)` + `Fix` before applying bond constraint

### Co-adsorbate Workflow Controls

Co-adsorbate mode ignores TDDFT/single optimisation controls (`fmax`, active-learning optimiser, spin multiplicity, scan controls) and enforces:

- `n_iter=1`
- `use_tddft_optimisation=False`

The screening-specific controls are in the co-adsorbate panel below.

## Co-Adsorbate Screening Panel

### Required Inputs

| Field | Meaning |
| --- | --- |
| `Reference Cluster` | Optional fallback base support structure when geometry-source inputs are empty. |
| Adsorbate source(s) | At least one must be provided across reference adsorbates / ASE list / PubChem list / uploaded files. |

### Adsorbate Sources

| Field | Meaning |
| --- | --- |
| `Reference Adsorbates` | Existing reference adsorbates matching selected settings profile. |
| `ASE Molecules` | Molecule names/formulas, one per line or comma-separated. |
| `PubChem CIDs` | Numeric CIDs, one per line or comma-separated. |
| `PubChem Queries` | Query strings, one per line or comma-separated. |
| `Adsorbate Structure Files` | Upload one or more `.xyz/.traj/.db/.cif/.pdb` files or VASP `POSCAR`/`CONTCAR` filenames. |

### MACE/Generation/Similarity Controls

| Field | Meaning | Default |
| --- | --- | --- |
| `MACE Model Path` | Path on HPC to MACE model file. | HPC-specific prefill when available |
| `MACE Device` | `cuda` or `cpu`. | `cuda` |
| `MACE fmax` | Relaxation force threshold. | `0.05` |
| `MACE Max Steps` | Max optimisation steps. | `200` |
| `Generation Site Symbols` | Symbols treated as candidate adsorption sites (auto-populated from selected structure, editable). | all detected symbols |
| `Generation Passive Symbols` | Fixed/passive support symbols in generation (auto-populated from selected structure, editable). | all detected symbols |
| `Generation Height (A)` | Initial adsorbate height. | `2.0` |
| `Generation Slab z Cutoff` | Optional absolute z-coordinate cutoff (A) for generated adsorbates (periodic surfaces only). Candidate adsorbate atoms must satisfy `z >= cutoff`. | empty |
| `Similarity Support Species` | Species included in similarity kernels (auto-populated from selected structure, editable). | all detected symbols |
| `Similarity Slab Shave Distance` | Optional depth (A) below the highest support-species z used to shave support atoms before similarity kernels (periodic surfaces only). | empty |
| `Site Probe Element` | Temporary atom used to compare candidate adsorption-site environments. | `H` |
| `Site Probe Similarity Cutoff` | Similar probe sites at or above this cutoff are duplicates. Lower values prune more. | `0.99` |
| `Single-Mode Duplicate Cutoff` | Duplicate cutoff after independent single-adsorbate relaxation. | `0.99` |
| `Combined-Candidate Duplicate Cutoff` | Optional fixed cutoff before combined relaxation; blank uses automatic selection. | empty |
| `Relaxed-Minimum Duplicate Cutoff` | Optional fixed cutoff after combined relaxation; blank uses automatic selection. | empty |

The generated `coadsorption_site_probe_similarity.csv` artifact records every bond-valid probe site, its closest retained site, and `max_similarity_to_kept`. Use it to tune the site-probe cutoff when the retained-site count is too aggressive.

The workflow also publishes `coadsorption_single_a_final_for_dft.db` and `coadsorption_single_b_final_for_dft.db`. These reuse the single-mode MACE relaxations after validity and similarity filtering; no duplicate optimisation is performed.

Validation highlights:

- Reference adsorbates must match the selected settings profile.
- PubChem CID values must be positive integers.

## Optimisation Constraints

| Field | Meaning | Rules/Default |
| --- | --- | --- |
| `Bond Index i`, `Bond Index j` | 0-based atom indices for constrained bond. | Single: optional but both required together. TDDFT: required |
| `Fixed Symbols` | Symbols constrained during optimisation. | Single: optional (blank default). TDDFT: used during scan |
| `Target Distance (A)` | Optional absolute target bond length before applying bond constraint. | Single: optional (requires bond indices). TDDFT: optional |
| `Fix` | ASE `set_distance(..., fix=...)` parameter when setting target distance. | Default `0.5`; must be `0.0` to `1.0` |
| `Increment (A)` | Bond distance change per iteration if no absolute target distance is set. | TDDFT only; required there |
| `Distance Change Fraction` | Fraction of remaining distance applied per step when target distance is set. | TDDFT only; default `1.0` |
| `Enable TS Smoothing` | Enables post-TS smoothing optimisation pass. | TDDFT only; default off |
| `Refinement Points / Side` | Extra constrained points around TS. | TDDFT only; default `0` |

## Slurm Resources

| Field | Meaning | Used By |
| --- | --- | --- |
| `Slurm Preset` | Preset bundle for resource values. | All modes |
| `Nodes` | Slurm nodes requested. | All modes |
| `Tasks Per Node` | Slurm tasks per node. | Single + TDDFT |
| `CPUs Per Task` | Slurm CPUs per task. | Single + TDDFT |
| `Walltime` | Job wall clock limit. | All modes |
| `Account` | Slurm account/project code. | All modes |
| `Partition` | Slurm partition/queue. | All modes |
| `QoS` | Slurm quality-of-service. | All modes |

If `Account`, `Partition`, or `QoS` is left blank, that directive is omitted from the generated Slurm script.

Mode-aware submit-page behavior:

- TDDFT mode + custom preset:
  - auto-defaults `qos`/`walltime` to TDDFT defaults (`long` / `96:00:00`) if currently blank/standard.
- Co-adsorbate mode + custom preset:
  - nudges to GPU-friendly values (`partition=gpu`, `qos=gpu-shd`, `walltime=12:00:00`, `nodes=1`, `tasks_per_node=1`, `cpus_per_task=1`) when fields are blank or still generic defaults.

Current presets include:

- `TRH408 MACE GPU (home run)`
- `Falcon` presets (where configured)

Co-adsorbate submission backend specifics:

- Single-node execution requests one GPU and uses the backend-specific launch.
- On `TRH408`, generated scripts use `#SBATCH --gres=gpu:1`, include `--cpus-per-task`, set `--exclusive`, and run Python directly (no `srun` wrapper).
- On `TRH408`, submissions run in-place under the run directory (`$HOME`-based root by default), with no scratch stage-in/stage-out copy cycle.

See the shared
[parallel MACE workflow guide](../../utilities/similarity/mace_parallel_workflows.md)
for calculator reuse, MPI logging, checkpoint/restart behavior, and debug metrics.

## HPC Paths and Environment

| Field | Meaning | Used By |
| --- | --- | --- |
| `Remote Runs Root` | Persistent run root on HPC. | All modes |
| `Scratch Base` | Scratch base for temporary run execution. | All modes |
| `Venv Activate Path` | Remote shell `source` path for Python env. | All modes |
| `Module Loads` | One module per line/comma. | All modes |
| `PYTHONPATH Entries` | Extra `PYTHONPATH` entries. | All modes |

Falcon NWChem notes:

- Build NWChem with `reaction_hub/scripts/compile_nwchem_7.2.2_falcon.sh`.
- NWChem single-optimisation submissions auto-use Intel/MPI modules on Falcon and ASE launches `/shared/home2/app_shared/PROJECT/nwchem-7.2.2/bin/nwchem` with `mpirun`.
- Override the executable with `NWCHEM_EXECUTABLE` if the binary is installed elsewhere.
- Falcon submissions request the selected allocation with `#SBATCH --ntasks` and `#SBATCH --tasks-per-node`; NWChem uses that rank count unless `NWCHEM_MPI_RANKS` is set.
- Falcon submissions default to `#SBATCH --mem=100G`. NWChem memory is derived from that per-node request divided by launched NWChem ranks, so the 192-rank default uses `533 mb`.
- Submitted NWChem workflows default `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and `MKL_DYNAMIC=FALSE` to avoid MKL oversubscription.

## Quick Mode Recipes

### Single optimisation (recommended minimal)

- Set mode: `Single optimisation`
- Provide geometry
- Set `fmax`
- Choose Slurm preset or custom resources
- Submit

### TDDFT bond scan

- Set mode: `TDDFT bond-length scan workflow`
- Set `bond_i`, `bond_j`, `increment` (and optional `target_distance`)
- Set `N Iter`, `Target Wavelength (nm)`, `TDDFT Scan NRoots`
- Optional: `Enable TS Smoothing` + refinement points
- Optional restart extension: set `Restart Extension Steps` and (optionally) `Restart Extension Increment (A)`
  - This appends extra scan points from the last completed geometry on resubmission/restart.
  - If extension increment is blank, the base scan `increment` is reused.
  - `Restart Extension Increment (A)` is only valid when `Restart Extension Steps` is greater than 0.
- Use TDDFT-oriented Slurm settings (`long`, longer walltime) unless intentionally overridden

### Co-adsorbate screening

- Set mode: `Co-adsorbate placement workflow`
- Choose `Reference Cluster`
- Provide one or more adsorbate sources
- Set MACE model/device and generation/similarity symbol lists as needed
- Use GPU-oriented Slurm settings (custom values or cluster-specific preset if your deployment defines one)

## Troubleshooting Common Submit Errors

| Error Pattern | Cause | Fix |
| --- | --- | --- |
| `Bond indices are required` | TDDFT mode missing `bond_i/bond_j`. | Set both indices. |
| `Bond indices must refer to two different atoms` | Same atom used twice. | Pick different indices. |
| `Increment cannot be zero when target distance is not set` | TDDFT step size invalid. | Set non-zero increment or provide target distance. |
| `Co-adsorbate submission requires at least one adsorbate` | No adsorbate source provided. | Provide any adsorbate source input. |
| `Selected reference adsorbates do not match chosen settings profile` | Settings profile mismatch. | Re-select reference adsorbates or settings profile. |
| `Run directory already exists` | `Run Name` already used in target runs root. | Change `Run Name` or archive/remove old run directory. |
