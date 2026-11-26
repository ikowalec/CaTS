# Getting started

This guide covers a local development dashboard followed by the first configured
HPC calculation. Use the repository-root `.venv` throughout. Python 3.12 is the
version used by the inspected development environment.

Users are assumed to be able to install and link scientific software, configure
their institution's access, and select valid executable and filesystem paths.
Reaction Hub does not install ORCA, NWChem, Multiwfn, POV-Ray, model weights or
remote Python environments. The tips below identify which machine needs each
dependency; they do not replace the relevant software or cluster documentation.

## 1. Install local Python dependencies

Start in the repository root with an existing `.venv`. If creating a new one,
use your chosen Python 3.12 installation's virtual-environment facility first.
Do not create a second environment inside `reaction_hub`.

Windows Command Prompt:

```bat
.venv\Scripts\python.exe -m pip install -r reaction_hub\requirements.txt
copy reaction_hub\.env.example reaction_hub\.env
```

Linux Bash:

```bash
.venv/bin/python -m pip install -r reaction_hub/requirements.txt
cp reaction_hub/.env.example reaction_hub/.env
```

Do not overwrite an existing local `.env`. Requirements pin direct installed
dependencies to the inspected venv, rather than freezing every unrelated package
in it. The browser viewer dependency was retained from the previous requirements
but was absent from that venv. PostgreSQL users also install
`reaction_hub/requirements-postgres.txt`; SQLite needs no PostgreSQL driver.
Calculation dependencies on HPC must be installed separately for the selected
workflow.

The inspected venv is not a clean dependency lock: its installed matscipy declares
a NumPy constraint incompatible with its installed NumPy. Matscipy is not part of
the dashboard requirements. Review that conflict before reusing the same
environment for workflows that need it; these documentation changes do not alter
the existing scientific environment. Run the chosen venv's package consistency
check after installation, then Django's system check below. A fresh installation
of this requirements list has not yet been validated.

## 2. Configure only the features you need

Edit `reaction_hub/.env` locally. Review Django security, allowed hosts and database
selection first. For background work, review broker and result-backend settings.
The [environment reference](environment.md) categorises every setting loaded by
the project settings module and marks feature-specific configuration.
It deliberately contains no configuration values; use `.env.example` and the
settings implementation to inspect template values and defaults.

Keep private configuration out of version control. Cluster-specific template
entries are placeholders, not working cluster registrations. Review them before
opening the submission form, which ensures the named cluster records exist.
Keep batch-mode key authentication enabled. Host, username and optional key-file
location belong in the existing `HPCSystem` record and established local SSH setup.

## 3. Initialise the local database

Windows, from the repository root:

```bat
.venv\Scripts\python.exe reaction_hub\manage.py check
.venv\Scripts\python.exe reaction_hub\manage.py migrate
.venv\Scripts\python.exe reaction_hub\manage.py createsuperuser
.venv\Scripts\python.exe reaction_hub\manage.py backfill_references
.venv\Scripts\python.exe reaction_hub\manage.py runserver
```

Linux, from the repository root:

```bash
.venv/bin/python reaction_hub/manage.py check
.venv/bin/python reaction_hub/manage.py migrate
.venv/bin/python reaction_hub/manage.py createsuperuser
.venv/bin/python reaction_hub/manage.py backfill_references
.venv/bin/python reaction_hub/manage.py runserver
```

Open the address printed by `runserver`; the dashboard is at the site root and
administration is at `/admin/`. A new database initially has no calculated runs.
Backfilled reference entries are metadata, not completed reference calculations.
This local development server is not a production deployment recipe.

## 4. Start Redis and background processing

Dashboard browsing does not require a worker. Queued analysis requires Redis
and a user-started Celery worker. Automatic polling and ingestion also require
one Celery beat scheduler. Do not start duplicate workers or beat instances if
your installation already has them running.

### Windows with WSL

In an administrator Windows terminal, install WSL with Ubuntu if not already
available, then complete the distribution's normal first-run setup. Restart
Windows if requested.

```bat
wsl --install -d Ubuntu
wsl --list --verbose
```

Follow [Microsoft's WSL installation guide](https://learn.microsoft.com/en-us/windows/wsl/install)
if installation or virtualisation prerequisites need attention.
In the Ubuntu WSL terminal:

```bash
sudo apt-get update
sudo apt-get install redis-server
sudo service redis-server start
redis-cli ping
```

The ping should return `PONG`. On a systemd-enabled distribution, Redis can also
be managed through its systemd service. Redis must be running again after WSL
is shut down. See [Redis's Windows/WSL instructions](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/install-redis-on-windows/).

Keep Django and Celery in the Windows venv when following this Windows setup;
Redis alone runs in WSL. The Python `redis` package is a client and does not
install the Redis server. Check that the configured broker can be reached from
Windows, not just from inside WSL. From the repository root:

```bat
.venv\Scripts\python.exe -c "import os, sys; sys.path.insert(0, 'reaction_hub'); os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'reaction_hub.settings'); import django; django.setup(); from django.conf import settings; import redis; print(redis.Redis.from_url(settings.CELERY_BROKER_URL).ping())"
```

If that fails while the WSL ping works, check WSL localhost forwarding and local
firewall policy. Keep Redis restricted to the local machine rather than exposing
its service to solve a connectivity problem.

In separate Windows Command Prompt terminals, each starting at the repository root:

```bat
cd reaction_hub
..\.venv\Scripts\celery.exe -A reaction_hub worker -l info --pool=solo -Q celery,submissions
```

```bat
cd reaction_hub
..\.venv\Scripts\celery.exe -A reaction_hub beat -l info
```

The solo pool is the existing Windows development setup. See Celery's
[platform support statement](https://docs.celeryq.dev/en/stable/faq.html#does-celery-support-windows)
when choosing a supported deployment platform.

### Fully Linux installation

For Ubuntu/Debian, the same package-installation and Redis ping commands above
apply in the Linux terminal. For other distributions, or Redis's own package
repository, follow the [official Linux installation guide](https://redis.io/docs/latest/operate/oss_and_stack/install/archive/install-redis/install-redis-on-linux/).

Run Django, Redis and Celery on the Linux host with its Linux venv. In separate
terminals, each starting at the repository root:

```bash
cd reaction_hub
../.venv/bin/celery -A reaction_hub worker -l info -Q celery,submissions
```

```bash
cd reaction_hub
../.venv/bin/celery -A reaction_hub beat -l info
```

Check worker startup output for both configured queues and the registered
`jobs.tasks.submit_analysis_request_task`. Celery publication, Slurm submission,
calculation completion and result ingestion are distinct stages.

## 5. Link scientific software and one HPC system

| Component | Where it is needed | Brief setup tip |
| --- | --- | --- |
| Repository Python modules | Local host and selected remote workflow environment | Keep the repository root importable alongside `reaction_hub`; inspect the remote Python-path entries and environment activation path. |
| ORCA or NWChem | Compute environment used by the calculation | Select the matching backend and load the site's modules or configure the executable path. Check MPI compatibility against the cluster's software instructions. |
| Workflow Python dependencies | Remote workflow environment | Install the selected workflow's dependencies, such as CatLearn or MACE, there; installing dashboard requirements locally does not provision HPC. |
| Multiwfn | Host executing the requested population/density analysis | Use the existing executable configuration and ensure the worker's remote environment can resolve it. |
| MACE weights | Selected compute host | Model paths must refer to files readable there; local Windows paths cannot identify remote Linux model files. |
| POV-Ray | Host executing POV-Ray rendering | Make its executable discoverable using the existing renderer configuration. Manuscript/report structure figures use the established POV-Ray workflow. |
| Cairo/GTK | Windows host running local SVG/GIF rendering | Follow the [Windows rendering utility](../../utilities/platforms/windows_rendering/README.md). Python packages do not supply the native runtime. |
| OpenSSH | Local submission/worker host | Reuse non-interactive keys/agent access and existing `HPCSystem` fields. Do not disable batch mode to make a task prompt for authentication. |

Sign in at `/admin/`, open **Jobs → Hpc systems**, and configure the target system.
The [operations guide](operations.md#hpcsystem-field-reference-admin-form) explains
its fields. Review persistent run root, optional scratch root, remote environment,
modules, account and scheduler resources for that system. Local ingestion paths
are paths visible to the local process; inaccessible remote trees need the
existing remote ingestion/cache mechanisms.

Use `/jobs/submit/` for the first small calculation: choose a geometry, backend,
settings profile, configured system and suitable resources. The
[submission guide](submission_guide.md) explains the fields. Review charge and
spin explicitly. Follow the resulting run and job pages through scheduler state
and ingested outputs; an accepted request alone is not evidence of completion.
Use the existing Celery analysis operations for subsequent queued analysis.
Do not create manual scheduler scripts or duplicate database records to bypass
submission or diagnose a queued request.

## 6. First-use troubleshooting

| Symptom | First check |
| --- | --- |
| Django cannot import `xyzrender` | Install requirements into the interpreter actually launching Django. |
| Repository modules cannot be imported | Make the repository root importable in the local or remote environment reporting the error. |
| Redis connection refused | Check server ping in WSL/Linux and broker reachability from the worker host. |
| Queued analysis stays queued | Check the existing worker's registered tasks and queue subscriptions; new jobs do not require a worker restart. |
| Job status does not refresh automatically | Check that one beat scheduler and a worker consuming the polling queue are running. |
| SSH authentication failure | Restore the existing key/agent access for the user running the worker; keep batch mode enabled. |
| Rendering reports a missing Cairo library | Check native runtime linking and activate the patched Windows environment before launching the process. |
| Metadata exists but results do not | Distinguish queued work, scheduler execution and ingestion; inspect the recorded job using approved diagnostics. |
