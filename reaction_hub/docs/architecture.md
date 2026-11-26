# Reaction Hub Architecture

## Design Goals

- Make workflow execution traceable from scheduler submission to final computed quantities.
- Enforce consistent references and settings for adsorption-energy reporting.
- Support idempotent ingestion so repeated scans do not duplicate records.
- Keep data ingestion read-only relative to external workflow directories.

## Execution vs. Ingestion Paths

- Workflows typically execute in temporary HPC scratch directories.
- Outputs are copied back to persistent storage after completion.
- Reaction Hub should ingest from the persistent tree via `REACTION_HUB_INGEST_ROOT` or `--workflow-root`.
- cplas run discovery depends on marker location and run structure; marker file content can still reference scratch paths.

## App Responsibilities

- `core`: shared base model (`TimeStampedModel`).
- `references`: canonical models for adsorbates, nanoclusters, settings profiles.
- `workflows`: workflow definitions and per-run metadata/state.
- `jobs`: HPC systems, job records, and job status events.
- `results`: artifacts, computed properties, adsorption-energy records.
- `dashboard`: summary/trend pages and operational views.
- `ingest`: adapters, parsers, scheduler polling, and ingestion services.

## Data Model Overview

- `CalculationSettingsProfile`
- `ReferenceAdsorbate`
- `ReferenceNanocluster`
- `WorkflowDefinition`
- `WorkflowRun`
- `HPCSystem`
- `JobRecord`
- `JobStatusEvent`
- `Artifact`
- `ComputedProperty`
- `AdsorptionEnergyResult`

## Key Relationships

- `WorkflowRun.workflow -> WorkflowDefinition`.
- `WorkflowRun.settings_profile -> CalculationSettingsProfile`.
- `WorkflowRun.reference_adsorbate -> ReferenceAdsorbate`.
- `WorkflowRun.reference_nanocluster -> ReferenceNanocluster`.
- `JobRecord.workflow_run -> WorkflowRun`.
- `JobRecord.hpc_system -> HPCSystem`.
- `JobStatusEvent.job -> JobRecord`.
- `Artifact.workflow_run -> WorkflowRun`.
- `Artifact.job -> JobRecord` (optional).
- `ComputedProperty.workflow_run -> WorkflowRun`.
- `ComputedProperty.source_artifact -> Artifact` (optional).
- `AdsorptionEnergyResult.workflow_run -> WorkflowRun` (one-to-one).

## Consistency Guarantees

`AdsorptionEnergyResult.clean()` enforces:

- `AdsorptionEnergyResult.settings_profile == WorkflowRun.settings_profile`.
- If run-level adsorbate is set, adsorption result adsorbate must match.
- If run-level nanocluster is set, adsorption result nanocluster must match.

This keeps adsorption-energy values attributable to approved and consistent settings.

## Lifecycle: File Ingestion

1. `ingest_workflow_outputs` discovers run directories with read-only adapters.
2. For each run directory, `ingest_single_run` resolves workflow identity and status.
3. Artifacts are upserted by `(workflow_run, path)` unique constraint.
4. Derived properties (for example final ORCA single-point energy) are upserted.
5. Re-running ingestion is safe and does not duplicate records.

## Lifecycle: Scheduler Polling

1. `sync_hpc_jobs` selects active non-terminal jobs.
2. Slurm snapshots are fetched via `SlurmSSHClient`.
3. Scheduler states are normalized to app states.
4. `JobRecord.transition_to` records status events and timestamps.
5. Completed jobs trigger ingestion of their workflow run directory.

## Dashboard Data Flow

- Dashboard queries `JobRecord` for aggregate counts by status.
- Trend chart groups records by day over recent history.
- Recent completions and failures are simple filtered querysets.

## API Surface

Routers are configured in `reaction_hub/reaction_hub/urls.py`:

- `/api/workflows/`
- `/api/runs/`
- `/api/jobs/`
- `/api/artifacts/`
- `/api/properties/`
- `/api/adsorption-energies/`

See `reaction_hub/docs/api.md` for examples.

## Security and Trust Boundaries

- No write operations are performed in external workflow directories.
- SSH polling should be restricted to controlled host/user/key tuples.
- Admin writes are limited to the Reaction Hub database.
- Artifact paths are treated as file references and resolved from run working directories.
