# Reaction Hub Docs Index

- [Getting started](getting_started.md): first installation, WSL/Linux Redis,
  local startup, software linking and first HPC calculation.
- [Environment configuration](environment.md): categorised variable reference
  without configuration values.
- [Windows rendering setup](../../utilities/platforms/windows_rendering/README.md):
  native Cairo/GTK linking helper.

- `architecture.md`: app boundaries, data model, ingestion/scheduler lifecycle.
- `operations.md`: conceptual usage model, web submit flow, HPC scratch/persistent root flow, HPC onboarding, TDDFT database field mapping, runbook, troubleshooting.
- `submission_guide.md`: comprehensive field-by-field reference for `/jobs/submit/` (all modes, defaults, and validation rules).
- `api.md`: endpoint list, query patterns, and request examples.
- `energy_terms.md`: definitions and formulas for `Eads`, `Ed`, and `Eint` shown in results.
- `../protocols/CO_OXIDATION_WORKFLOW.md`: shared CO oxidation workflow order, evidence gates, and source reports.
- `examples/slurm_submission_template.sh.tmpl`: starter Slurm submission template with module loading and scratch stage-in/stage-out.
- [Au CO oxidation subgroup discovery](au_subgroup_discovery.md) — dataset preparation, provenance, descriptors and search.
