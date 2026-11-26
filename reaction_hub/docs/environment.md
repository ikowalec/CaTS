# Environment configuration reference

Reaction Hub loads `reaction_hub/.env` from its project directory using
`django-environ`. Existing process environment variables take precedence over
that file. Keep real configuration local. This reference lists names, expected
types and effects only: no configured values, example assignments or defaults.
Use `.env.example` for the editable template and
`reaction_hub/reaction_hub/settings.py` for implementation defaults and limits.

Users are assumed to be able to link required software and choose suitable
paths. A path must exist on the host that uses it: local cache/render paths are
local; workflow environments, models, scratch and run roots are remote. A Windows
drive path cannot identify a Linux cluster directory. See
[getting started](getting_started.md) for brief software and Redis setup tips.

Not every variable needs changing. Review web/database settings for local use,
broker settings before queued work, and SSH/submission settings before HPC use.
The feature labels below describe when configuration matters, not whether Django
requires an explicit assignment. New calculations do not require worker restarts.
When intentionally changing process configuration, it must be loaded into the
affected user-managed process before it can take effect.

## Web application and database

| Variable | Type | Purpose and when to review |
| --- | --- | --- |
| `DEBUG` | Boolean | Development error reporting and static-file behaviour; review before deployment. |
| `DJANGO_SECRET_KEY` | Secret text | Django signing key; create and retain a private installation-specific key. |
| `ALLOWED_HOSTS` | CSV | Hostnames accepted by Django; match how the dashboard is accessed. |
| `DATABASE_URL` | Database URL | Database backend and location; use a filesystem location visible to the app or a reachable database service. PostgreSQL needs the optional driver requirements. |
| `DEFAULT_FROM_EMAIL` | Text | Sender identity for Django email. This does not configure an email transport. |

## Redis, Celery and periodic processing

| Variable | Type | Purpose and when to review |
| --- | --- | --- |
| `CELERY_BROKER_URL` | Broker URL | Redis connection used for Celery messages and the existing single-flight task lock. Review whenever Redis runs on a different host or service endpoint. |
| `CELERY_RESULT_BACKEND` | Backend identifier or URL | Where task results are retained; distinct from the broker and calculation artifacts. |
| `REACTION_HUB_SYNC_HPC_JOBS_INTERVAL_SECONDS` | Seconds | Beat interval for scheduler polling; settings enforce a minimum interval. |
| `REACTION_HUB_INGEST_INTERVAL_SECONDS` | Seconds | Beat interval for output ingestion; settings enforce a minimum interval. |
| `REACTION_HUB_ENABLE_TASKFARM_BACKFILL_BEAT` | Boolean | Enables periodic taskfarm backfill publication; enable only for a deliberately configured campaign. |
| `REACTION_HUB_ENABLE_GPU_DISPATCH_BEAT` | Boolean | Enables periodic dispatch of queued GPU workflows. |

The broker must be a reachable Redis service for the existing locking code;
installing the Python Redis client does not install the server. Worker queue
subscriptions and registered task names are operational checks, not `.env`
variables. Beat publishes periodic work; a worker executes it.

## Queue capacity and dispatch controls

| Variable | Type | Purpose and when to review |
| --- | --- | --- |
| `REACTION_HUB_SHORT_QUEUE_MAX_JOBS` | Integer | Short-queue capacity; settings enforce bounded limits. |
| `REACTION_HUB_SERIAL_QUEUE_MAX_JOBS` | Integer | Serial-queue capacity; settings enforce bounded limits. |
| `REACTION_HUB_GPU_QUEUE_MAX_JOBS` | Integer | GPU queue capacity; settings enforce bounded limits. |
| `REACTION_HUB_TASKFARM_QUEUE_MAX_JOBS` | Integer | Taskfarm capacity; settings enforce bounded limits. |
| `REACTION_HUB_SHORT_QUEUE_POLL_SECONDS` | Seconds | Retry/poll interval for the short-queue capacity check. |
| `REACTION_HUB_CT_SERIAL_WALLTIME` | Scheduler duration | Walltime used for serial charge-transfer work. |
| `REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS` | Boolean | Pause control for TRH GPU submissions; review if GPU work does not dispatch. |

Capacity settings are application-side controls. They do not create scheduler
allocations or override institution limits.

## Ingestion, cache and completion checks

| Variable | Type | Purpose and when to review |
| --- | --- | --- |
| `REACTION_HUB_INGEST_ROOT` | Local path | Persistent tree visible to the ingestion process; do not point it at inaccessible remote scratch. |
| `REACTION_HUB_REMOTE_CACHE_ROOT` | Local path | Root used to retain ingested remote artifacts locally. |
| `REACTION_HUB_INGEST_REMOTE_ORCA_READ` | Boolean | Allows remote ORCA output content to be read/downloaded rather than retaining only remote-location metadata. Consider output size. |
| `REACTION_HUB_INGEST_DOWNLOAD_CUBES` | Boolean | Controls downloading cube data during ingestion. Consider cache space and transfer size. |
| `REACTION_HUB_SYNC_STRICT_COMPLETED_REQUIRES_PROPERTIES` | Boolean | Treats scheduler-completed work without ingested computed properties as failed when enabled. |

## SSH and reachability

| Variable | Type | Purpose and when to review |
| --- | --- | --- |
| `REACTION_HUB_SYNC_PREFLIGHT_REACHABILITY` | Boolean | Enables the TCP reachability check before scheduler SSH operations. |
| `REACTION_HUB_SYNC_PREFLIGHT_TIMEOUT_SECONDS` | Seconds | Timeout for the preflight TCP check. |
| `REACTION_HUB_SSH_BATCH_MODE` | Boolean | Enforces non-interactive key/agent authentication; keep enabled under project policy. |
| `REACTION_HUB_SSH_FORCE_TTY` | Boolean | Controls forced TTY allocation; this is not permission to use interactive authentication. |
| `REACTION_HUB_SSH_TIMEOUT_SECONDS` | Seconds | SSH connection/command timeout used by the supported operations. |
| `REACTION_HUB_SSH_CLEAN_BASH` | Boolean | Runs remote commands through a clean Bash invocation without startup-file side effects. Ensure required modules and paths are configured explicitly. |
| `REACTION_HUB_SSH_BASH_PATH` | Executable path/name | Bash executable used by the clean-shell option. |

Actual connection host, port, username and optional identity-file path are
recorded in `HPCSystem`. Identity paths refer to files on the worker host; do not
copy keys into this repository or include key contents in configuration.
Use existing non-interactive access. Failed authentication is not a reason to
disable batch mode or bypass the Celery submission route.

## Generic remote submission defaults

| Variable | Type | Purpose and when to review |
| --- | --- | --- |
| `REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT` | Remote path | Persistent root for generated calculation run directories. |
| `REACTION_HUB_SUBMIT_SCRATCH_BASE` | Remote path | Scratch base for execution when the selected backend uses staging. |
| `REACTION_HUB_SUBMIT_DEFAULT_MODULES` | CSV | Site software modules offered as submission defaults. |
| `REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH` | Remote path | Activation script for the remote calculation environment. |
| `REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH` | CSV of remote paths | Repository and dependency import roots on compute hosts. |
| `REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT` | Text | Scheduler project/account prefill; must match a valid allocation. |
| `REACTION_HUB_SUBMIT_DEFAULT_PARTITION` | Text | Scheduler partition prefill. |
| `REACTION_HUB_SUBMIT_DEFAULT_QOS` | Text | Scheduler quality-of-service prefill. |
| `REACTION_HUB_SUBMIT_DEFAULT_WALLTIME` | Scheduler duration | Generic requested runtime. |
| `REACTION_HUB_SUBMIT_SLURM_PRESETS_JSON` | JSON object | Resource-preset mapping keyed by HPC system name; includes preset identifiers, labels and resource fields. Consult the settings implementation for schema. |
| `REACTION_HUB_SUBMIT_TDDFT_DEFAULT_QOS` | Text | Quality-of-service prefill for TDDFT scan mode. |
| `REACTION_HUB_SUBMIT_TDDFT_DEFAULT_WALLTIME` | Scheduler duration | Requested-runtime prefill for TDDFT scan mode. |
| `REACTION_HUB_SUBMIT_AL_STEPS_PER_ITER` | Integer | Active-learning step allowance per iteration. |
| `REACTION_HUB_SUBMIT_MAX_AL_STEPS_PER_ITER` | Optional integer | Upper limit on active-learning iteration steps; blank/unlimited handling is implemented in the settings parser. |

These variables prefill or configure existing submission paths; they do not
install software or grant scheduler access. Review the actual submission form
because workflow and system-specific settings can override generic defaults.

## TRH408 system prefill

These names are tied to the existing TRH408 profile, not a generic new-cluster
namespace. The submit form ensures that named system exists and fills missing
connection details; an existing populated record is not simply replaced by
changing the environment. Review the database record as well.

| Variable | Type | Purpose |
| --- | --- | --- |
| `REACTION_HUB_SUBMIT_TRH408_HOST` | Hostname | Canonical system's SSH host prefill. |
| `REACTION_HUB_SUBMIT_TRH408_PORT` | Integer | SSH port prefill. |
| `REACTION_HUB_SUBMIT_TRH408_USERNAME` | Text | SSH account prefill. |
| `REACTION_HUB_SUBMIT_TRH408_VENV_ACTIVATE_PATH` | Remote path | Calculation environment activation script. |
| `REACTION_HUB_SUBMIT_TRH408_PYTHONPATH` | CSV of remote paths | Calculation import roots. |
| `REACTION_HUB_SUBMIT_TRH408_MODULES` | CSV | Calculation software modules. |
| `REACTION_HUB_SUBMIT_TRH408_REMOTE_RUNS_ROOT` | Remote path | Persistent run root for this system. |
| `REACTION_HUB_SUBMIT_TRH408_SCRATCH_BASE` | Remote path | Scratch configuration; check the system's in-place execution behaviour. |

## Falcon system prefill

| Variable | Type | Purpose |
| --- | --- | --- |
| `REACTION_HUB_SUBMIT_FALCON_HOST` | Hostname | Canonical Falcon system's SSH host prefill. |
| `REACTION_HUB_SUBMIT_FALCON_PORT` | Integer | SSH port prefill. |
| `REACTION_HUB_SUBMIT_FALCON_USERNAME` | Text | SSH account prefill. |
| `REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH` | Remote path | Calculation environment activation script. |
| `REACTION_HUB_SUBMIT_FALCON_PYTHONPATH` | CSV of remote paths | Repository and calculation-dependency import roots. |
| `REACTION_HUB_SUBMIT_FALCON_MODULES` | CSV | Software modules loaded for the calculation. |
| `REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT` | Remote path | Persistent run root. |
| `REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE` | Remote path | Scratch execution base. |
| `REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT` | Text | Falcon scheduler-account prefill. |
| `REACTION_HUB_SUBMIT_FALCON_MEMORY` | Scheduler memory request | Memory passed to Falcon submission scripts. |

## Model selection

| Variable | Type | Purpose and when to review |
| --- | --- | --- |
| `REACTION_HUB_COADSORBATE_MACE_MODEL_PATH` | Remote path | Generic model-file choice for co-adsorbate MACE workflows. |
| `REACTION_HUB_COADSORBATE_MACE_MODEL_PATH_BY_HPC_JSON` | JSON object | Per-system model-file mapping; keys are normalised system names, and values identify files on those systems. |

The parsed per-system model mapping is an internal Django setting. Set the JSON
environment variable rather than assuming the internal setting name is loaded
from `.env`.

## Local viewing and rendering

| Variable | Type | Purpose and when to review |
| --- | --- | --- |
| `REACTION_HUB_ASE_GUI_COMMAND` | Command template | ASE GUI launcher on the host running Django, with an optional file placeholder; requires a local display. |
| `REACTION_HUB_XYZRENDER_COMMAND` | Command template | Override for local SVG/GIF rendering; leave detection to the existing helper unless a specific executable is needed. |
| `REACTION_HUB_XYZRENDER_GIF_ROT` | Rotation specification | Rotation axis/plane passed to the renderer. |
| `REACTION_HUB_XYZRENDER_ROT_FRAMES` | Integer | Number of frames requested for rotation rendering. |
| `REACTION_HUB_XYZRENDER_GIF_SIZE` | Canvas size | Overall GIF canvas specification. |
| `REACTION_HUB_XYZRENDER_GIF_WIDTH` | Integer | Optional canvas-width override. |
| `REACTION_HUB_XYZRENDER_GIF_HEIGHT` | Integer | Optional canvas-height override. |
| `REACTION_HUB_XYZRENDER_FIXED_SCALE_ANCHOR_SYMBOL` | Chemical symbol | Temporary anchor used by the rendering helper to stabilise scale. |

Windows native runtime linking uses process-level `GTK_RUNTIME_ROOT`,
`CAIROCFFI_DLL_DIRECTORIES` and `PATH`. The
[Windows rendering utility](../../utilities/platforms/windows_rendering/README.md)
sets these during environment activation. They are not Django settings and need
to be visible before the process loads Cairo. POV-Ray report figures use their
own established renderer rather than the SVG/GIF viewer path.

## Backend executable environment

| Variable | Type | Purpose |
| --- | --- | --- |
| `NWCHEM_EXECUTABLE` | Executable path/name | Backend command override read by the NWChem calculation code. Set on the host executing that calculation. |
| `NWCHEM_MPI_RANKS` | Integer | NWChem rank override read by calculation/submission code; coordinate it with allocated resources. |

ORCA, Multiwfn, POV-Ray and model-specific configuration also have existing
backend/workflow mechanisms. A local `.env` is not a blanket export of arbitrary
variables into every remote job; use the supported backend and submission fields.

## Names which are not currently loaded from `.env`

The following are read as Django settings or used as internal job state, but
have no corresponding environment loader in the inspected settings module.
Adding them to `.env` alone does not configure them. Do not confuse a settings
fallback or generated remote environment variable with a supported local `.env`
option.

| Name | Role |
| --- | --- |
| `REACTION_HUB_ACTIVE_RESERVATIONS` | Internal reservation configuration. |
| `REACTION_HUB_FALCON_DEV_QUEUE_MAX_JOBS` | Django-setting fallback for development queue capacity. |
| `REACTION_HUB_FALCON_HTC_QUEUE_MAX_JOBS` | Django-setting fallback for Falcon throughput capacity. |
| `REACTION_HUB_LONG_QUEUE_MAX_JOBS` | Django-setting fallback for long queue capacity. |
| `REACTION_HUB_STANDARD_QUEUE_MAX_JOBS` | Django-setting fallback for standard queue capacity. |
| `REACTION_HUB_NEB_MACE_MODEL_PATH` | Django-setting prefill for NEB model selection. |
| `REACTION_HUB_SUBMIT_DEFAULT_RESERVATION` | Django-setting prefill for a reservation. |
| `REACTION_HUB_SUBMIT_MAX_AL_STEPS_PER_ITER_SINGLE_OPTIMISATION` | Django-setting fallback for single-optimisation step limits. |
| `REACTION_HUB_SUBMIT_TRH408_DEFAULT_ACCOUNT` | Django-setting fallback for TRH408 account. |
| `REACTION_HUB_SUBMIT_TRH408_DEFAULT_PARTITION` | Django-setting fallback for TRH408 partition. |
| `REACTION_HUB_SUBMIT_TRH408_DEFAULT_QOS` | Django-setting fallback for TRH408 quality of service. |
| `REACTION_HUB_SUBMIT_TRH408_MACE_MODEL_PATH` | Django-setting fallback for the TRH408 MACE model. |
| `REACTION_HUB_SUBMIT_SLURM_PRESETS` | Parsed internal preset mapping; configure its JSON environment input instead. |
| `REACTION_HUB_COADSORBATE_MACE_MODEL_PATH_BY_HPC` | Parsed internal model mapping; configure its JSON environment input instead. |

Generated job variables controlling staging, restart lineage and rsync are
implementation details of the submission pipeline and are not local onboarding
settings. Preserve their generation through the existing submission routes.
