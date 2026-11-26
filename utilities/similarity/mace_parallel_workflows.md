# Parallel MACE Workflows

This guide covers:

- cluster growth: `nanocluster_growth_workflow.py`;
- alloy and staged alloy growth: `alloy_nanocluster_growth_workflow.py`;
- co-adsorbate placement: `coadsorbate_similarity.py`.

## Execution model

The workflows detect visible CUDA/ROCm devices at startup. A single-process run
creates one cached MACE calculator per visible device and schedules independent
optimisations concurrently. An MPI run uses one Slurm rank per GPU; rank 0
coordinates tasks and is the only rank that publishes checkpoints.

Calculators are created lazily and reused for the whole process. Consequently,
eight calculator banners are neither expected nor printed: worker output is
suppressed to keep logs ordered, and only `min(number of ready tasks, GPU count)`
GPUs can be active. For example, binary single-atom references provide only two
tasks, while a batch of at least eight relaxations can use all eight GPUs.

The same MPI ranks also process independent similarity batches and
post-relaxation integrity checks on their allocated CPUs. CPU-only tasks do not
create a MACE calculator, and rank 0 restores deterministic input ordering before
combining their results. Without a multi-rank `mpi4py` launch these stages run
serially.

## Progress and evaluation diagnostics

Normal logs are coordinator-only and report aggregate progress near 10%
milestones. Enable one line per completed optimisation with:

```bash
export MACE_WORKFLOW_DEBUG=1
```

Example:

```text
[MACE full relaxation] Completed 5/6 (83%); row=5, steps=24, evaluations=25, elapsed=18.7s
```

The debug line is emitted when rank 0 receives the result, before checkpoint
publication. Checkpoint DB metadata and relaxation CSV files also record:

- `nsteps`: exact ASE optimiser steps;
- `force_evaluations`: `nsteps + 1`, including the initial force evaluation;
- `elapsed_seconds`: wall time for that optimisation.

## Checkpoints and restarts

- Only rank 0 reads or writes workflow outputs.
- ASE databases use `serial=True`; ASE structure I/O uses `parallel=False`.
- SQLite databases are assembled in memory and atomically published through a
  `.db.tmp` file, so Lustre never hosts a live SQLite transaction or journal.
- Relaxation snapshots are published near 10% milestones and at stage completion.
- Restart hashes include the input geometry and relaxation settings. Matching
  rows are reused; changed inputs are recalculated.
- An interrupted run can lose only results completed since the latest snapshot.

Rerun the same command to restart. Do not remove valid checkpoint DBs. A stale
`.db.tmp` is incomplete and is ignored; the last complete `.db` remains valid.

## Saved geometry

BFGS does not save full optimisation trajectories. Endpoint structures and
metrics are stored in checkpoint DBs and CSV reports. Some workflows still
write `.xyz` or `.traj` files containing selected endpoint ensembles; these are
not step-by-step optimiser histories.

## Quick troubleshooting

| Symptom | Meaning or fix |
| --- | --- |
| Eight Python processes are visible | Expected: one MPI rank per GPU, one logical workflow. |
| `ModuleNotFoundError: mpi4py` | Recreate the venv with `--system-site-packages` from `cray-python`. |
| `GTL library is not linked` | Run `unset MPICH_GPU_SUPPORT_ENABLED` before `srun`. |
| Only a few GPUs are busy | The current stage has fewer ready tasks than GPUs. |
| No worker calculator banners | Expected; worker output is captured and calculators are lazy. |
| PyTorch kernel-cache warning | Use the writable per-rank cache in `gpu-bind.sh`. |
| `selected.xyz` is absent during relaxation | It is published only after every task in that stage completes. |
| Restart DB is read-only | Restore owner write permission on the DB, sidecars, and containing directory. |
