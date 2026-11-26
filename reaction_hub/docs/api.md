# Reaction Hub API Reference

Base URL examples assume local development server:

- `http://127.0.0.1:8000/api/`

## Endpoints

- `GET/POST /api/workflows/`
- `GET/POST /api/runs/`
- `GET/POST /api/jobs/`
- `GET/POST /api/artifacts/`
- `GET/POST /api/properties/`
- `GET/POST /api/adsorption-energies/`

`HPCSystem` is not currently exposed as a routed API endpoint. Create and edit
HPC systems through Django admin (`/admin/jobs/hpcsystem/`) or Django shell.

Each endpoint supports detail operations by appending `/<id>/`.

## Common Query Features

- Filtering by configured model fields.
- Text search via `search=` on selected fields.
- Sorting via `ordering=`.
- Pagination via `page=`.

## Example: List Completed Jobs

```bash
curl "http://127.0.0.1:8000/api/jobs/?status=completed&ordering=-completed_at"
```

## Example: Search Runs by Name

```bash
curl "http://127.0.0.1:8000/api/runs/?search=row_0001"
```

## Example: Artifacts for a Run

```bash
curl "http://127.0.0.1:8000/api/artifacts/?workflow_run=1&ordering=kind"
```

## Example: Create a Job Record

```bash
curl -X POST "http://127.0.0.1:8000/api/jobs/" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_run": 1,
    "hpc_system": 1,
    "scheduler_job_id": "123456",
    "job_name": "tddft_iter_1",
    "status": "initiated"
  }'
```

## Example: Create Adsorption Energy Result

```bash
curl -X POST "http://127.0.0.1:8000/api/adsorption-energies/" \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_run": 1,
    "adsorbate": 1,
    "nanocluster": 1,
    "settings_profile": 1,
    "total_energy_eV": "-100.000000",
    "reference_energy_eV": "-98.500000",
    "adsorption_energy_eV": "-1.500000"
  }'
```

## API Auth

By default this scaffold uses Django defaults and no token auth configuration.

For multi-user deployments, add a production auth layer, for example:

- Session auth behind SSO/reverse proxy.
- Token/JWT auth for service clients.
