# CaTS - Catalysis and Theoretical Simulations
Main author: Igor Kowalec

Other contributors before first release: Thomas Hill, Oliver Morris, Yunyu Zhang.

Computational photo- and thermal catalysis workflows and analysis tools, developed through research within CPLAS.
Focus is on streamlined processes and agentic capabilities.

Agentic tools in use: Codex and Gemini.

## Repository Structure

- `cats/` — Core atomistic simulation toolkit: ORCA, NWChem, Multiwfn, VASP, POV-Ray rendering, spectrum/structure visualization, and active-learning / TDDFT / NEB / dimer workflows.
- `reaction_hub/` — Django + Celery web control plane for Slurm job submission, scheduler sync, artifact ingestion, and campaign/results management.
- `similarity/` — REMatch similarity metrics, MACE batch optimization, and cluster/adsorbate structure-matching workflows.
- `utilities/` — Standalone workflow runners, database conversion/pruning scripts, platform setup helpers (such as offline GPU venv builders), and workflow tests.
- `gui_tools/` — Desktop GUI utilities (MACE trajectory screener).
- `.agents/` — Agent orchestrator schemas, task templates, and local agent skills.

## Version control scope

Track reusable libraries, workflows, documented examples, tests and documentation.
Keep calculation data, model weights, generated figures, manuscript outputs and
one-off campaign scripts local and ignored. Use `.data/`, `scratch/` or `adhoc/`
for new research-specific files; do not ignore entire workflow source directories.
Small scientific fixtures required by tests or documented examples can be tracked
with explicit `.gitignore` exceptions. Removing a file from tracking preserves its
local copy and does not remove earlier versions from Git history.

## Similarity Tools

- REMatch threshold selection guide: `similarity/rematch/threshold_selection.md`
- Co-adsorbate generation, MACE relaxation, and de-duplication workflow: `utilities/similarity/coadsorbate_workflow.md`
- Alloy nanocluster growth with MACE and mixing-energy ranking: `utilities/similarity/alloy_nanocluster_growth_workflow.md`
- Batch TDDFT active-learning example from co-adsorbate DB: `utilities/similarity/batch_tddft_from_similarity.py`

### Batch MACE optimisation

Optimise every structure in any ASE-readable file and write only endpoint
structures plus a CSV summary:

```powershell
.venv\Scripts\python.exe -m similarity.run.mace_batch_optimisation structures.extxyz --model medium --device cuda
```

The default outputs are `structures_mace_optimized.db` and
`structures_mace_summary.csv`. Trajectories are disabled unless
`--trajectory-dir trajectories` is supplied. Run with `--help` for calculator
and optimiser options. For a multi-node Slurm launch, use the existing
`similarity.workflows.mace_mpi_entry` launcher described in
`utilities/similarity/mace_parallel_workflows.md`.

## MACE Dataset Utilities

Use the project virtual environment when running these scripts:

```powershell
.venv\Scripts\python.exe cats\working_directory\evaluated_traj_to_xyz.py reaction_hub\.remote_cache -o mace_training.xyz --audit mace_training_audit.csv --sample-interval 10
```

`evaluated_traj_to_xyz.py` recursively finds `evaluated.traj` files, optionally only under folders whose names start with `--prefix`, then writes about one image per `--sample-interval` optimisation steps from each eligible metal trajectory, always including endpoints. Use `--species Au Cu` to keep only frames whose symbols are all in that list; this excludes reference gas adsorbates unless their symbols are also selected. It keeps Cu/Au ground-state ORCA runs matching the Reaction Hub settings profile `cam-b3lyp-lanl2dz` (or CAM-B3LYP/`lanl2dz_mixed` summary settings), plus tagged reference adsorbate gas runs. Gas references contribute only their final frame, and duplicate references are reduced to the lowest-energy final frame when an energy is available. The script skips NEB/barrier/constrained runs and full TDDFT optimisations, rejects ASE frames carrying constraints, and writes an audit CSV with every include/skip reason.

```powershell
.venv\Scripts\python.exe cats\working_directory\split_xyz_dataset.py training.xyz -o mace_split --seed 42
```

`split_xyz_dataset.py` writes `train.xyz`, `validation.xyz`, and `test.xyz` using an 80/10/10 split. Before shuffling the remaining structures, it reserves the lowest-energy structure for every composition in the training set. If the number of compositions exceeds the nominal training size, the training set grows to contain them all.

### MACE Trajectory Screener GUI

For curating combined `.xyz` trajectories with endpoint preservation, high-force barrier retention ($F_{\max} \ge 5.0\text{ eV/\AA}$), and pair distance RMSD similarity filtering:

- **Desktop Executable**: Launch `gui_tools/MACE_Trajectory_Screener.exe` (or `gui_tools/MACETrajectoryScreener.exe`).
- **Python GUI**: Run `.venv\Scripts\python.exe gui_tools\mace_trajectory_screener_gui.py`.
- **Protocol Documentation**: See [docs/mace_trajectory_screening_protocol.md](docs/mace_trajectory_screening_protocol.md) for full algorithmic details and benchmarks.

### Notes in Progress

- Add new notes here as they are collected.
- `/srv/research` holds shared resources.
- In `/srv/research`, `mace_venv` is used by the submission script.
- Example for running MACE: `/srv/research/examples/mace_mp`.
- Copy the example into your home directory before running it.
- Submit the job with `sbatch` using the provided submission script.
- `/srv/research/mace_models` contains downloaded ML models.
- To use a model, set the corresponding model name and update the model path in `input.py`.
- Ignore the `snap` folder; it is a software manager directory and mainly relevant when using the desktop PC directly.
- If a job is briefly held with `InvalidAccount` and then continues, this is usually a transient Slurm account validation delay - the job will start normally.

### FAQ

- Warnings in output: a wall of deprecation warnings is usually nothing to worry about, and results are not affected.
