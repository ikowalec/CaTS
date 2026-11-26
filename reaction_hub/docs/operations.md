# Reaction Hub Operations Guide

## Prerequisites

- Python 3.12+
- Redis server for queued analysis and automatic background processing
- Access to target Slurm clusters (if scheduler sync is enabled)

## Conceptual Usage Model

1. Your workflow project submits jobs to HPC and executes them in a run directory.
2. Reaction Hub stores metadata for the workflow run, job record, scheduler status events, and output artifacts.
3. `sync_hpc_jobs` polls the scheduler and updates `JobRecord.status`.
4. `ingest_workflow_outputs` scans completed run directories and upserts artifacts/properties.
5. Repeated polling and ingestion are safe and idempotent.

Remote deletion detection:

- During `sync_hpc_jobs`, Reaction Hub also checks whether each run root directory
  (`WorkflowRun.working_directory`, typically the `/work/.../reaction_hub_runs/<run_name>` path)
  still exists on the HPC filesystem.
- If that run root directory is missing for a failed or cancelled job, the job is transitioned to `deleted`.
- Completed jobs are not marked `deleted` when their remote run directory is missing, so old completed run folders can be moved or backed up without hiding them in the app.
- `deleted` is treated as a terminal job state and indicates the remote run root was removed.
- In Django admin (`Jobs -> Job records`), `deleted` rows are shown as `Marked For Deletion`,
  and can be removed in bulk with the action `Delete selected jobs marked for deletion`.

`Scheduler status events` are an audit trail of job status changes (for example:
`initiated -> pending -> running -> completed`).

Reaction Hub is the control plane and index. It tracks and ingests, and can submit
Slurm runs from the web UI (`/jobs/submit/`) when configured.

## Scratch and `workflow-root` Semantics

- Execution happens in scratch space on the HPC cluster.
- Output is then copied back to a persistent run directory.
- `--workflow-root` and `REACTION_HUB_INGEST_ROOT` should point to the persistent tree, not the temporary scratch path.
- cplas discovery relies on marker placement (`TDDFT_active_learning_shared/workflow_root.txt`) and run layout. Marker content can still be a scratch path after copy-back, and ingestion still works.

Example:

- Scratch execution path: `/scratch/<project>/$USER/$SLURM_JOB_ID`
- Persistent ingest root: `/shared/home/$USER/cplas_runs`
- Ingest command: `python manage.py ingest_workflow_outputs --workflow-root /shared/home/$USER/cplas_runs`

## Installation

Use [Getting started](getting_started.md) for installation into the repository-root
venv, database initialisation and brief software/path-linking tips. Environment
settings are documented by category in [Environment configuration](environment.md)
without listing configuration values.

## Runtime Components

- Django app server
- Celery worker
- Celery beat scheduler
- Optional periodic/cron invocation of management commands

Default scheduler cadence (Celery beat):

- `sync_hpc_jobs`: every 600 seconds
- `ingest_workflow_outputs`: every 600 seconds
- Minimum enforced interval for both is 60 seconds, even if lower values are provided in env.
- Configure with `REACTION_HUB_SYNC_HPC_JOBS_INTERVAL_SECONDS` and `REACTION_HUB_INGEST_INTERVAL_SECONDS`.

Remote `orca.out` handling:

- By default, Reaction Hub does **not** read/download raw remote `orca.out` content during remote ingestion.
- The artifact row is still created with remote location metadata (`remote_path`) and marked `remote_only`.
- This keeps large ORCA logs retained on HPC while preserving where they are located.
- Optional override (not recommended for large files): set `REACTION_HUB_INGEST_REMOTE_ORCA_READ=True`.

Remote trajectory caching (`evaluated.traj`):

- During remote ingestion, Reaction Hub looks for `*/active_learning/evaluated.traj` under the run root.
- These trajectory files are downloaded to a local cache so they are viewable in UI and ASE GUI.
- Artifact metadata stores both:
  - `remote_path` (original HPC path)
  - `local_cache_path` (local mirrored file path)
- Configure cache root with `REACTION_HUB_REMOTE_CACHE_ROOT` (default: `reaction_hub/.remote_cache`).

## Local Development Startup

Follow the [Redis and background processing guide](getting_started.md#4-start-redis-and-background-processing)
for Windows with WSL or fully Linux installations. The user starts and manages
the worker and beat scheduler. Reuse existing processes; new jobs do not require
a worker restart. Analysis tasks use the submissions queue, while periodic
polling and ingestion use the normal Celery queue.

## Web Submit Flow (No Manual File Copy)

Reaction Hub now supports direct Slurm submission from the web UI.

For a complete field-by-field reference of every submit option (including
mode-specific behavior, defaults, and validation), see
`docs/submission_guide.md`.

1. Open `http://127.0.0.1:8000/jobs/submit/`.
2. Choose geometry source:
   - upload geometry file (`.xyz`, `.traj`, `.db`, `.cif`, `.pdb`)
   - ASE molecule builder (type formula/name, for example `CO`, `H2O`)
   - PubChem lookup (CID or query text)
3. Choose workflow, settings profile, and HPC system.
4. Optional: in `Reference Type`, mark the run as:
   - `Reference adsorbate calculation`, or
   - `Reference nanocluster calculation`.
   If selected, provide `Reference Name` (or leave blank to reuse `Run Name`).
5. In `Slurm Resources`, choose a preset from the dropdown (or `Custom`) and review populated fields (`account`, `partition`, `qos`, walltime, nodes).
6. Click `Submit to HPC`.
7. Backend creates run files remotely, executes `sbatch`, and creates `WorkflowRun` + `JobRecord`.
8. Existing `sync_hpc_jobs` + ingestion flow recovers status and results automatically.

What is auto-generated remotely in run directory:

- `orca.xyz` (from selected geometry source: upload, ASE molecule, or PubChem)
- `run_workflow_submit.py` (workflow wrapper)
- `submit_workflow.sh` (submission script)

Resource consistency note:

- Generated `submit_workflow.sh` exports `WORKFLOW_NODES`, `WORKFLOW_TASKS_PER_NODE`, and `WORKFLOW_CPUS_PER_TASK`.
- `run_workflow_submit.py` derives workflow ORCA resources from those exact values:
  - `nodes = WORKFLOW_NODES`
  - `cpus_per_node = WORKFLOW_TASKS_PER_NODE * WORKFLOW_CPUS_PER_TASK`

Notes for non-file geometry sources:

- ASE molecule source uses `ase.build.molecule(...)`.
- PubChem source uses `ase.data.pubchem.pubchem_atoms_search(...)`.
- PubChem lookup requires network access from the host running Django/worker.

Requirements for this flow:

- Selected workflow must have a valid `code_path` (Python module path).
- Selected HPC system must be active Slurm with working SSH credentials.
- Settings profile must use supported functional/basis for workflow submit path.
- Remote run directory must not already exist (submit fails fast if it does).
- Automatic recovery requires scheduler sync and ingestion to run periodically (Celery beat or cron).

Reference tagging behavior:

- If `Reference Type` is selected, Reaction Hub creates or reuses the selected reference entity and links it on the run:
  - `Reference adsorbate calculation` -> `WorkflowRun.reference_adsorbate`
  - `Reference nanocluster calculation` -> `WorkflowRun.reference_nanocluster`
- Reference entities are keyed by `Reference Name`.
- Settings profile consistency is enforced when reusing existing references.

Run-mode aware controls in submit UI:

- `Single optimisation` mode:
  - Runs one optimisation only (`n_iter=1`, TDDFT optimisation disabled).
  - Uses user-provided `fmax` convergence threshold from submit form.
  - TDDFT bond-scan fields are hidden.
- `TDDFT bond-length scan workflow` mode:
  - Shows bond scan controls (`bond_i`, `bond_j`, `increment`, `fix`, `fixed_symbols`, optional `target_distance`).
  - Supports restart-only path extension via `restart_extend_steps` and optional `restart_extend_increment_A`.
  - Uses user-provided `fmax` convergence threshold from submit form.
  - Auto-applies TDDFT-oriented scheduler defaults:
    - `qos` from `REACTION_HUB_SUBMIT_TDDFT_DEFAULT_QOS` (default `long`)
    - `walltime` from `REACTION_HUB_SUBMIT_TDDFT_DEFAULT_WALLTIME` (default `96:00:00`)

Active-learning optimiser failsafe (`NaN` / trust-distance instability / force spikes):

- The TDDFT/DFT active-learning step now prioritizes reaching convergence over preserving a single uninterrupted ML restart stream.
- The submit UI exposes an active-learning optimiser choice (`BFGS` default, optional `FIRE2`).
- If LocalAL becomes unstable (for example messages like `Outside of the trust distance`, `nan` coordinates, or `True fmax` force spikes above `10 eV/A`), the workflow automatically:
  1. Restores the last finite atom geometry snapshot.
  2. Archives volatile LocalAL restart files in the active-learning state directory:
     - `predicted.traj`
     - `predicted_evaluated.traj`
     - `converged.traj`
     - `initial_struc.traj`
     - `ml_summary.txt`
     - `ml_time.txt`
  3. Archives the current AL training trajectory (`evaluated.traj`) and rebuilds a clean training seed for recovery.
  4. Retries LocalAL with tighter controls:
     - reduced trust distance (`dtrust *= 0.5`, floor `0.25`)
     - reduced local optimiser step size (`maxstep *= 0.5`, floor `5e-4`)
  5. Applies recovery reseed guards to avoid restarting from obviously diverged geometries:
     - selects the last viable frame from the archived `evaluated.traj` (cached force `<= 1.0 eV/A`) when available
     - if re-evaluation of that geometry exceeds the guard, keeps the cached viable frame in the rebuilt training set
- Up to 3 recovery attempts are made per AL chunk.
- If instability still persists after retries, the run raises a clear runtime error rather than silently continuing with corrupted state.

Operational implications:

- This is intentionally more conservative and may increase walltime, but is more robust for difficult geometries.
- Recovery archives are timestamped (for example `evaluated.traj.failed.<UTC timestamp>`) so previous state can still be inspected.

HPC-specific Slurm presets:

- Presets are selected from a dropdown in `Slurm Resources`.
- Preset mappings are keyed by HPC system name and can be extended in
  `REACTION_HUB_SUBMIT_SLURM_PRESETS` (or overridden via `REACTION_HUB_SUBMIT_SLURM_PRESETS_JSON`).

Structure inspection tools in submit UI:

- `Preview in Browser`: parses the selected geometry source and shows
  - atom index table (`index`, `symbol`, `x`, `y`, `z`),
  - element counts,
  - XYZ preview text.
- `Open in ASE GUI`: launches ASE GUI on the machine running Django using
  `REACTION_HUB_ASE_GUI_COMMAND` (default `python -m ase gui`).
  - You can include `{file}` in the command; if omitted, Reaction Hub appends the temporary XYZ file path.
  - This requires local GUI/display support where the server process runs.

Structure inspection from workflow page:

- On `/workflows/<run_id>/`, structure/trajectory artifacts include an `Open in ASE GUI` action.
- This opens the local artifact path on the machine running Django.
- Remote-only artifacts (retained on HPC) are intentionally blocked from ASE GUI launch.

Frame batch submission:

- When uploading a `.traj` or `.db` file, enable `Submit all trajectory frames`.
- Reaction Hub submits one run per frame using numbered names:
  - `run_name_0`, `run_name_1`, `run_name_2`, ...
- Each frame inherits the same workflow/settings/HPC/Slurm inputs from the single form.
- Batch mode is disabled for files other than `.traj` and `.db`.
- Reference tagging is intentionally single-run only (batch + reference is blocked to avoid ambiguous links).
- Stored structure metadata now includes `structure_sha256` (hash of symbols + positions),
  which can be used to group DFT and TDDFT runs that start from the same geometry.

Workflow plot viewing:

- Key workflow plots are now shown directly on:
  - `/workflows/<run_id>/`
  - `/results/energy/<result_id>/`
- Supported key plots:
  - `stacked_uvvis.png`
  - `stacked_ct.png`
  - `energy_vs_reaction_coordinate.png`
- Any linked image artifact can be opened at:
  - `/results/artifact/<artifact_id>/content/`

Easy database extraction:

- Dashboard CSV export:
  - `/results/export/dashboard.csv`
- REST API (JSON):
  - `/api/runs/`
  - `/api/jobs/`
  - `/api/artifacts/`
  - `/api/properties/`
  - `/api/adsorption-energies/`

## Add a New HPC System (Slurm) Step-by-Step

`HPCSystem` is a Django model (`jobs.models.HPCSystem`) that stores the connection
details for a scheduler host.

Current behavior:

- You add `HPCSystem` through Django admin.
- `sync_hpc_jobs` uses `HPCSystem` records to SSH and call Slurm commands.
- There is no `/api/hpcsystems/` route registered yet, so use admin or Django shell.

### `HPCSystem` Field Reference (Admin Form)

| Field | Required | Example | Notes |
| --- | --- | --- | --- |
| `name` | Yes | `falcon` | Human-readable unique key in Reaction Hub. |
| `scheduler_type` | Yes | `slurm` | Only Slurm is supported currently. |
| `host` | Yes | `hpc-login.example.org` | Scheduler login host Reaction Hub can SSH to. |
| `port` | Yes | `22` | SSH port. |
| `username` | Yes | `abc123` | HPC account username used for polling. |
| `ssh_key_path` | Optional | `/home/app/.ssh/hpc_ed25519` | Path on the machine where `sync_hpc_jobs` runs. Leave blank to use OpenSSH default identities/agent. |
| `is_active` | Yes | checked | Must be enabled or the system is ignored. |

### First connection and first calculation

The [getting-started guide](getting_started.md#5-link-scientific-software-and-one-hpc-system)
covers local versus remote dependencies, existing non-interactive SSH access,
system registration and the first web submission. Users are assumed to know how
to install/link software and select valid paths for their institution.

Keep batch-mode key/agent authentication enabled. Use existing `HPCSystem`
records and approved submission/diagnostic operations; do not disable batch mode,
create manual scheduler scripts or manually duplicate submitted job records.
Queued analysis belongs on the existing Celery submission route. Authentication
failures require restoring the established key/agent access, not bypassing it.

The web submission pipeline generates staging and restart files. Persistent run
roots retain outputs; scratch is temporary. Preserve the generated run layout and
lineage so ingestion can associate results with the recorded workflow. Use the
[submission reference](submission_guide.md) for detailed resource and mode fields.

## Management Command Runbook

- Manual ingest scan:

```bash
python manage.py ingest_workflow_outputs --workflow-root /work/PROJECT/GROUP/$USER/cplas_runs
```

- Manual scheduler sync:

```bash
python manage.py sync_hpc_jobs
```

- Re-seed defaults if needed:

```bash
python manage.py backfill_references
```

## Ingestion Setup

- Set `REACTION_HUB_INGEST_ROOT` in `.env` to your persistent workflow tree root.
- Current adapter targets cplas-style TDDFT run directories.
- Ingestion is idempotent and safe to run repeatedly.

## TDDFT Database Mapping (Detailed)

This section documents exactly what Reaction Hub stores for cplas TDDFT active-learning
runs discovered by `ingest_workflow_outputs`.

All models listed here also include automatic timestamps from `TimeStampedModel`:
`created_at` and `updated_at`.

### 1) Run Discovery (What Counts as a TDDFT Run)

A run directory is discovered when either condition is true:

- `TDDFT_active_learning_shared/workflow_root.txt` exists under that run directory.
- A `TDDFT/` directory exists and either `DFT/` or `TDDFT_active_learning_shared/` exists beside it.

Discovered runs are assigned workflow name `cplas_tddft_active_learning`.

### 2) Workflow Status Type (How Run Type Is Inferred)

Run type is inferred from files in this priority order:

1. If any `error.log` exists anywhere under the run directory:
   - `WorkflowRun.status = failed`
2. Else if any `iteration_complete.txt` exists:
   - `WorkflowRun.status = completed`
3. Else if any `orca.out` exists:
   - `WorkflowRun.status = running`
4. Else:
   - `WorkflowRun.status = initiated`

Important: this is file-presence based. File contents are not used for status typing.

### 3) `WorkflowDefinition` and `WorkflowRun` Records

On ingest, Reaction Hub creates or updates:

#### `WorkflowDefinition`

- `name`: `cplas_tddft_active_learning`
- `code_path`: `cats/working_directory/tddft_active_learning_workflow.py` (on first creation)
- `description`: fixed TDDFT adapter description (on first creation)

#### `WorkflowRun`

Lookup key:

- Deterministic `run_uid` from run directory absolute path (`uuid5`), so repeated ingest
  maps to the same run record.

Values written:

| Field | Stored Value |
| --- | --- |
| `workflow` | `WorkflowDefinition(name="cplas_tddft_active_learning")` |
| `run_name` | Name of run directory (last path component) |
| `status` | Inferred status (`initiated/running/completed/failed`) |
| `working_directory` | Absolute run directory path |
| `settings_profile` | `default_pbe0_water` profile (created if missing) |
| `metadata` | `{"ingested_from": "cplas_adapter", "path": "<run_dir>"}` |
| `started_at` | Set when status is not `initiated` and value is empty |
| `completed_at` | Set only when status is `completed` and value is empty |

Notes for users:

- `reference_adsorbate` and `reference_nanocluster` are not populated by TDDFT ingestion.
- Existing `settings_profile` and `metadata` are overwritten by ingest to the default values above.
- `failed` runs do not automatically set `completed_at` in current implementation.

### 4) Artifacts Stored Per Run (`results.Artifact`)

Artifacts are upserted by unique key `(workflow_run, path)`.

Candidate artifacts collected:

| Kind | File Pattern | Label Stored |
| --- | --- | --- |
| `database` | `TDDFT_active_learning_shared/scan_endpoints.db` | `scan_endpoints_db` |
| `csv` | `TDDFT_active_learning_shared/iroot_scan_endpoints.csv` | `iroot_scan_endpoints` |
| `csv` | `TDDFT_active_learning_shared/iroot_geometry_log.csv` | `iroot_geometry_log` |
| `log` | `**/orca.out` | `orca_out:<parent_dir_name>` |
| `structure` | `**/*.xyz` | `structure:<filename>` |
| `trajectory` | `**/*.traj` | `trajectory:<filename>` |

For each stored artifact:

| Field | Stored Value |
| --- | --- |
| `workflow_run` | Current run |
| `path` | Path relative to run root (or absolute if not under root) |
| `kind` | Mapped from candidate kind |
| `label` | As listed above |
| `metadata` | `{"source": "cplas_adapter"}` |

Not auto-populated by ingest:

- `job`
- `checksum`

### 5) Computed Properties Stored (`results.ComputedProperty`)

For each discovered `orca.out`, ingestion attempts to parse:

- Regex: `FINAL SINGLE POINT ENERGY <float>`

If parsing succeeds, property is upserted:

| Field | Stored Value |
| --- | --- |
| `workflow_run` | Current run |
| `name` | `final_single_point_energy` |
| `unit` | `Eh` |
| `value_float` | Parsed final single-point energy |
| `source_artifact` | The `Artifact` row for that `orca.out` |
| `metadata` | `{"parsed_from": "<artifact_path>"}` |

Important behavior:

- Property uniqueness is `(workflow_run, name, unit)`.
- This means there is one `final_single_point_energy` per run.
- If multiple `orca.out` files exist, later upserts overwrite earlier value/source.

### 6) What Changes by Run Type

The table below shows what differs by inferred run type:

| Inferred Type | Trigger Files | `WorkflowRun.status` | `started_at` | `completed_at` | Artifacts | `final_single_point_energy` |
| --- | --- | --- | --- | --- | --- | --- |
| `initiated` | none of `error.log`, `iteration_complete.txt`, `orca.out` found | `initiated` | not set by ingest | not set by ingest | stored if matching files exist | usually absent |
| `running` | at least one `orca.out`, no `error.log`, no `iteration_complete.txt` | `running` | set if empty | not set by ingest | stored | stored if regex found |
| `completed` | at least one `iteration_complete.txt`, no `error.log` | `completed` | set if empty | set if empty | stored | stored if regex found |
| `failed` | at least one `error.log` (highest priority) | `failed` | set if empty | not set by ingest | stored | stored if regex found |

### 7) Job Polling Records Related to TDDFT Runs

`sync_hpc_jobs` updates existing `jobs.JobRecord` rows linked to TDDFT runs:

- It only polls jobs where status is not terminal (`completed`, `failed`, `cancelled`).
- Scheduler states are normalized to app states (`pending/running/completed/failed/cancelled/unknown`).
- On status change, it stores one `JobStatusEvent` with:
  - `previous_status`
  - `status`
  - `message` (`Scheduler update: <state>`)
  - `payload` (snapshot fields like `job_id`, `state`, `runtime`, `name`, `workdir`, optional `exit_code`)

Plain-language meaning:

- Each scheduler status event is one timestamped transition record in the job history.
- You can read these as "what changed, when it changed, and what the scheduler reported".

Automatic side effect:

- If a job transitions to `completed`, ingestion is triggered for that run directory.

Current limitation:

- Polling does not currently copy snapshot fields into `JobRecord.remote_workdir`,
  `JobRecord.scheduler_payload`, `JobRecord.exit_code`, `stdout_path`, or `stderr_path`.
  Those remain user-populated unless written elsewhere.

### 8) Data Not Currently Captured from TDDFT Files

For clarity, these are not currently parsed into dedicated DB fields by ingestion:

- `workflow_root.txt` content value
- `iteration_complete.txt` content (for example `mode=tddft`)
- IROOT/scan metrics from CSV into `ComputedProperty` rows
- Excited-state transition tables (wavelength/intensity/CT)
- Auto-creation of `AdsorptionEnergyResult` from run outputs

## Non-Slurm HPC Note

Current scheduler sync implementation uses Slurm commands (`squeue`, `sacct`) via SSH. To support another scheduler:

1. Add a new scheduler choice in `jobs/models.py`.
2. Add a scheduler client in `ingest/hpc.py`.
3. Route by scheduler type in `ingest/services.py::sync_hpc_jobs`.
4. Add/adjust state normalization logic in `ingest/services.py::_normalize_scheduler_state`.

## Troubleshooting

- Symptom: No runs ingested.
  - Check `REACTION_HUB_INGEST_ROOT`.
  - Confirm expected marker files exist in run directories.
  - Confirm you are ingesting from the persistent `/work` tree, not scratch.

- Symptom: Scheduler sync updates nothing.
  - Verify `HPCSystem.is_active=True`.
  - Check SSH connectivity and credentials.
  - Confirm non-terminal job records exist.

- Symptom: Structure viewer not rendered.
  - Verify at least one `structure` artifact points to existing `xyz/cif/pdb`.
  - Install `py3Dmol`.

- Symptom: Website `Render SVG/GIF` fails with `libcairo`/`cairo-2.dll` errors, or only SVG is generated.
  - Root cause is usually missing Cairo runtime in the active Python environment.
  - On Windows, install GTK runtime:
    - Repo: `https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer`
    - Direct installer: `https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases/download/2021-01-30/gtk2-runtime-2.24.33-2021-01-30-ts-win64.exe`
  - Then run:
    - `powershell -NoProfile -ExecutionPolicy Bypass -File "<repo>\\reaction_hub\\scripts\\configure_venv_gtk.ps1" -VenvPath "<repo>\\.venv" -GtkRoot "C:\\Program Files\\GTK2-Runtime Win64"`
  - Open a new shell and reactivate `.venv`.
  - Verify:
    - `python -c "import ctypes; ctypes.CDLL('libcairo-2.dll'); print('cairo-ok')"`
    - `xyzrender --help`

- Symptom: Celery tasks do not run.
  - Confirm worker and beat processes are both running.
  - Verify broker URL and backend settings.

- Symptom: Admin page is unstyled (plain text, missing CSS).
  - Install dependencies including `whitenoise` from `requirements.txt`.
  - Run `python manage.py collectstatic` (required when `DEBUG=False`).
  - Confirm `/static/admin/css/base.css` returns `200`.
  - Hard refresh browser cache (`Ctrl+F5` in Edge).

## Testing

```bash
cd reaction_hub
source .venv/bin/activate
pytest
```

## Backup and Recovery

- For SQLite, back up `db.sqlite3` and media/artifact roots.
- For Postgres, use regular DB dumps and tested restore procedures.
- Keep `.env` secrets out of version control.
