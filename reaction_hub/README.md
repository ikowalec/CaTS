# Reaction Hub

Django dashboard and database for computational chemistry workflow tracking.

## What This Project Covers

- Track workflow definitions, runs, HPC jobs, and status transitions.
- Keep reference adsorbates, nanoclusters, and settings profiles in one place.
- Link output artifacts and computed properties to each workflow run.
- Ingest existing run outputs from sibling repositories in a read-only way.
- Provide a dashboard for initiated/running/completed/failed jobs.
- Expose REST API endpoints for jobs, runs, artifacts, and results.
- Render structure artifacts (`xyz`, `cif`, `pdb`) with a notebook-style embedded viewer.

## Documentation Map

- Architecture and data model: `reaction_hub/docs/architecture.md`
- Conceptual usage, web submit, HPC onboarding, TDDFT DB mapping, and runbook: `reaction_hub/docs/operations.md`
- API reference and examples: `reaction_hub/docs/api.md`
- Slurm submission template example: `reaction_hub/docs/examples/slurm_submission_template.sh.tmpl`

## Project Layout

- Django project package: `reaction_hub/reaction_hub/`
- Apps:
  - `core/`
  - `references/`
  - `workflows/`
  - `jobs/`
  - `results/`
  - `dashboard/`
  - `ingest/`
- Templates: `reaction_hub/templates/`
- Tests: `reaction_hub/tests/`
- Dependencies: `reaction_hub/requirements.txt`
- Env template: `reaction_hub/.env.example`

## Quick Start

Follow [Getting started](docs/getting_started.md) for repository-root venv
installation, database setup, Windows WSL/Linux Redis installation, user-managed
Celery startup and the first HPC calculation.

Users are assumed to be capable of installing/linking required scientific
software and configuring local/remote paths. The guide provides brief linking
tips; dashboard dependencies do not install cluster software or model weights.

- [Categorised environment reference](docs/environment.md): variable names,
  types and effects, without configuration values.
- [Windows SVG/GIF rendering utility](../utilities/platforms/windows_rendering/README.md):
  native Cairo/GTK linking and the tracked activation helper.
- `requirements.txt`: direct dependencies reconciled with the inspected local venv.
- `requirements-postgres.txt`: optional PostgreSQL driver.

## Main Commands

- Merge parsed legacy ORCA catalogs:

```bash
python manage.py import_legacy_catalog --catalog-root /path/to/parsed/catalogs --dry-run
python manage.py import_legacy_catalog --catalog-root /path/to/parsed/catalogs
```

The import is idempotent and creates jobless workflow runs. Numerical TDDFT,
UV/Vis, and charge-transfer data is stored in the versioned
`spectral_dataset` computed property so plots can be recreated without access
to the original files. Exact ORCA inputs/settings, final atomic charges, scan
coordinates/forces, and a checksummed companion-file manifest are also
retained. Referenced artifacts are stored relative to each calculation so
their identity survives filesystem moves; missing files do not stop the
import. New catalogs also carry content-derived per-calculation fingerprints,
so reparsing an unchanged tree after an HPC path-prefix move does not create
new runs.

Rerunning `python -m cats.orca.legacy_catalog` reuses unchanged schema-compatible
catalog entries by default. The parser hashes the complete calculation file
manifest and reparses only calculations whose inputs or companion files have
changed. Invoke `python -m cats.orca.legacy_catalog ... --no-reuse` to force a
clean reparse.

If the source tree has moved since parsing, point the import at its current
root. For a parallel catalog tree, the catalog-relative subdirectory is
automatically appended:

```bash
python manage.py import_legacy_catalog \
  --catalog-root /path/to/parsed/catalogs \
  --source-root /new/hpc/prefix/or/archived/source/root
```

Rerun the same command with a new `--source-root` after a later move to update
run working directories without changing run or artifact identities.

- Backfill the same replot-ready spectral representation for existing modern
  runs from retained transition properties and accessible spectrum CSVs:

```bash
python manage.py backfill_spectral_data --dry-run
python manage.py backfill_spectral_data
```

- Ingest outputs from files:

```bash
python manage.py ingest_workflow_outputs --workflow-root /work/path/to/workflow/root
```

Use the persistent copy of runs, not temporary scratch directories.

- Poll scheduler state and update jobs:

```bash
python manage.py sync_hpc_jobs
```

Force re-ingestion for completed runs if a previous sync/SSH session dropped mid-ingest:

```bash
python manage.py sync_hpc_jobs --force-reingestion
```

Retry ingestion for failed runs (off by default):

```bash
python manage.py sync_hpc_jobs --retry-failed-ingestion
```

Enable verbose sync diagnostics:

```bash
python manage.py sync_hpc_jobs --debug
```

- Seed baseline references:

```bash
python manage.py backfill_references
```

- Collect static files (required for admin styling when `DEBUG=False`):

```bash
python manage.py collectstatic
```

- Run tests:

```bash
pytest
```

- Web submit page:

`http://127.0.0.1:8000/jobs/submit/`

## Environment Variables

See the [categorised environment reference](docs/environment.md). It covers
web/database configuration, Redis/Celery, queue controls, ingestion, SSH, remote
submission, named HPC profiles, models and local rendering. Configuration values
belong in your local `.env`; template values remain in `.env.example`.

## Notes

- Ingestion adapters are read-only wrappers around outputs in sibling trees like `cats/` and `similarity/`.
- Scheduler sync currently targets Slurm over SSH via the system OpenSSH client (`ssh`).
- The dashboard reads live database state; no static snapshotting is used.
- Submit-form startup ensures canonical `TRH408` and `FALCON` Slurm systems are present in the HPC dropdown.
