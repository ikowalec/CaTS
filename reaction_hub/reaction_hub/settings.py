from __future__ import annotations

import os
import json
from pathlib import Path
from urllib.parse import urlparse

from kombu import Queue


BASE_DIR = Path(__file__).resolve().parent.parent


def _str_to_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _split_csv(value: str | None, *, default: list[str]) -> list[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        return int(text)
    except ValueError:
        return default


def _optional_positive_int_env(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    if text.lower() in {"none", "null"}:
        return None
    try:
        parsed = int(text)
    except ValueError:
        return default
    if parsed <= 0:
        return None
    return int(parsed)


def _json_env(name: str, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    text = raw.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default



def _database_config_from_url(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme in {"sqlite", "sqlite3"}:
        db_path = parsed.path or ""
        if db_path.startswith("/"):
            db_path = db_path[1:]
        resolved = BASE_DIR / db_path if db_path else BASE_DIR / "db.sqlite3"
        return {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": str(resolved),
        }
    if parsed.scheme in {"postgres", "postgresql"}:
        return {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": parsed.path.lstrip("/"),
            "USER": parsed.username or "",
            "PASSWORD": parsed.password or "",
            "HOST": parsed.hostname or "",
            "PORT": str(parsed.port or "5432"),
        }
    raise ValueError(f"Unsupported DATABASE_URL scheme: {parsed.scheme}")


try:
    import environ  # type: ignore

    env = environ.Env(DEBUG=(bool, True))
    environ.Env.read_env(BASE_DIR / ".env")
    SECRET_KEY = env("DJANGO_SECRET_KEY", default="dev-only-change-me")
    DEBUG = env("DEBUG", default=True)
    ALLOWED_HOSTS = _split_csv(env("ALLOWED_HOSTS", default="127.0.0.1,localhost,testserver"), default=["127.0.0.1", "localhost", "testserver"])
    DATABASES = {
        "default": env.db(
            "DATABASE_URL",
            default=f"sqlite:///{BASE_DIR / 'db.sqlite3'}",
        )
    }
    CELERY_BROKER_URL = env("CELERY_BROKER_URL", default="redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = env("CELERY_RESULT_BACKEND", default="django-db")
except ImportError:
    SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-only-change-me")
    DEBUG = _str_to_bool(os.environ.get("DEBUG"), default=True)
    ALLOWED_HOSTS = _split_csv(
        os.environ.get("ALLOWED_HOSTS"),
        default=["127.0.0.1", "localhost", "testserver"],
    )
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        DATABASES = {"default": _database_config_from_url(database_url)}
    else:
        DATABASES = {
            "default": {
                "ENGINE": "django.db.backends.sqlite3",
                "NAME": BASE_DIR / "db.sqlite3",
            }
        }
    CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "django-db")

if DATABASES.get("default", {}).get("ENGINE") == "django.db.backends.sqlite3":
    opts = DATABASES["default"].setdefault("OPTIONS", {})
    opts["timeout"] = 60
    opts["init_command"] = "PRAGMA journal_mode=WAL; PRAGMA busy_timeout=60000; PRAGMA synchronous=NORMAL;"




if DEBUG and "testserver" not in ALLOWED_HOSTS:
    ALLOWED_HOSTS.append("testserver")


INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django_celery_results",
    "django_celery_beat",
    "rest_framework",
    "django_filters",
    "django_tables2",
    "core",
    "references",
    "workflows",
    "jobs",
    "results",
    "dashboard",
    "ingest",
    "xyzrender",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "reaction_hub.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "reaction_hub.wsgi.application"
ASGI_APPLICATION = "reaction_hub.asgi.application"

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

_staticfiles_backend = (
    "whitenoise.storage.CompressedStaticFilesStorage"
    if DEBUG
    else "whitenoise.storage.CompressedManifestStaticFilesStorage"
)
STORAGES = {
    "default": {
        "BACKEND": "django.core.files.storage.FileSystemStorage",
    },
    "staticfiles": {
        "BACKEND": _staticfiles_backend,
    },
}
WHITENOISE_MANIFEST_STRICT = False

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

REST_FRAMEWORK = {
    "DEFAULT_FILTER_BACKENDS": [
        "django_filters.rest_framework.DjangoFilterBackend",
        "rest_framework.filters.SearchFilter",
        "rest_framework.filters.OrderingFilter",
    ],
    "DEFAULT_PAGINATION_CLASS": "rest_framework.pagination.PageNumberPagination",
    "PAGE_SIZE": 25,
}

CELERY_TASK_TRACK_STARTED = True
CELERY_TIMEZONE = TIME_ZONE
CELERY_WORKER_CONCURRENCY = 1
CELERY_WORKER_PREFETCH_MULTIPLIER = 1
CELERY_TASK_QUEUES = (Queue("celery"), Queue("submissions"))
CELERY_TASK_ROUTES = {
    "jobs.tasks.submit_analysis_request_task": {"queue": "submissions"},
    "jobs.tasks.submit_mulliken_recovery_task": {"queue": "submissions"},
    "jobs.tasks.submit_full_ct_workup_task": {"queue": "submissions"},
    "jobs.tasks.restart_cancelled_job_task": {"queue": "submissions"},
    "jobs.tasks.resubmit_failed_job_task": {"queue": "submissions"},
    "jobs.tasks.restore_ct_serial_job_task": {"queue": "submissions"},
    "jobs.tasks.submit_queued_gpu_workflow_task": {"queue": "submissions"},
}
REACTION_HUB_CT_SERIAL_WALLTIME = os.environ.get(
    "REACTION_HUB_CT_SERIAL_WALLTIME", "24:00:00"
)
REACTION_HUB_SHORT_QUEUE_MAX_JOBS = min(
    4, max(1, int(os.environ.get("REACTION_HUB_SHORT_QUEUE_MAX_JOBS", "4")))
)
REACTION_HUB_SERIAL_QUEUE_MAX_JOBS = min(
    8, max(1, int(os.environ.get("REACTION_HUB_SERIAL_QUEUE_MAX_JOBS", "8")))
)
REACTION_HUB_GPU_QUEUE_MAX_JOBS = min(
    2, max(1, int(os.environ.get("REACTION_HUB_GPU_QUEUE_MAX_JOBS", "1")))
)
REACTION_HUB_TASKFARM_QUEUE_MAX_JOBS = min(
    128, max(1, int(os.environ.get("REACTION_HUB_TASKFARM_QUEUE_MAX_JOBS", "128")))
)
REACTION_HUB_SHORT_QUEUE_POLL_SECONDS = max(
    1.0, _float_env("REACTION_HUB_SHORT_QUEUE_POLL_SECONDS", 30.0)
)
REACTION_HUB_MIN_BEAT_INTERVAL_SECONDS = 60.0
REACTION_HUB_SYNC_HPC_JOBS_INTERVAL_SECONDS = max(
    REACTION_HUB_MIN_BEAT_INTERVAL_SECONDS,
    _float_env("REACTION_HUB_SYNC_HPC_JOBS_INTERVAL_SECONDS", 600.0),
)
REACTION_HUB_INGEST_INTERVAL_SECONDS = max(
    REACTION_HUB_MIN_BEAT_INTERVAL_SECONDS,
    _float_env("REACTION_HUB_INGEST_INTERVAL_SECONDS", 600.0),
)
CELERY_BEAT_SCHEDULE = {
    "sync-hpc-jobs-every-10-min": {
        "task": "jobs.tasks.sync_hpc_jobs_task",
        "schedule": REACTION_HUB_SYNC_HPC_JOBS_INTERVAL_SECONDS,
    },
    "ingest-workflow-outputs-every-10-min": {
        "task": "ingest.tasks.ingest_workflow_outputs_task",
        "schedule": REACTION_HUB_INGEST_INTERVAL_SECONDS,
    },
}
if _str_to_bool(os.environ.get("REACTION_HUB_ENABLE_TASKFARM_BACKFILL_BEAT"), default=False):
    CELERY_BEAT_SCHEDULE["taskfarm-backfill-every-minute"] = {
        "task": "jobs.tasks.submit_analysis_request_task",
        "schedule": REACTION_HUB_MIN_BEAT_INTERVAL_SECONDS,
        "args": ("taskfarm_backfill", {
            "target_jobs": 128,
            "max_new_submissions": 128,
        }),
        "options": {"queue": "submissions"},
    }
if _str_to_bool(os.environ.get("REACTION_HUB_ENABLE_GPU_DISPATCH_BEAT"), default=False):
    CELERY_BEAT_SCHEDULE["dispatch-next-trh-gpu-every-minute"] = {
        "task": "jobs.tasks.dispatch_next_queued_gpu_workflow_task",
        "schedule": REACTION_HUB_MIN_BEAT_INTERVAL_SECONDS,
    }

REACTION_HUB_INGEST_ROOT = os.environ.get(
    "REACTION_HUB_INGEST_ROOT",
    str((BASE_DIR.parent / "utilities" / "similarity").resolve()),
)
REACTION_HUB_REMOTE_CACHE_ROOT = os.environ.get(
    "REACTION_HUB_REMOTE_CACHE_ROOT",
    str((BASE_DIR / ".remote_cache").resolve()),
)
REACTION_HUB_INGEST_REMOTE_ORCA_READ = _str_to_bool(
    os.environ.get("REACTION_HUB_INGEST_REMOTE_ORCA_READ"),
    default=False,
)
REACTION_HUB_INGEST_DOWNLOAD_CUBES = _str_to_bool(
    os.environ.get("REACTION_HUB_INGEST_DOWNLOAD_CUBES"),
    default=False,
)
REACTION_HUB_SYNC_STRICT_COMPLETED_REQUIRES_PROPERTIES = _str_to_bool(
    os.environ.get("REACTION_HUB_SYNC_STRICT_COMPLETED_REQUIRES_PROPERTIES"),
    default=False,
)
REACTION_HUB_SYNC_PREFLIGHT_REACHABILITY = _str_to_bool(
    os.environ.get("REACTION_HUB_SYNC_PREFLIGHT_REACHABILITY"),
    default=True,
)
REACTION_HUB_SYNC_PREFLIGHT_TIMEOUT_SECONDS = max(
    0.1,
    _float_env("REACTION_HUB_SYNC_PREFLIGHT_TIMEOUT_SECONDS", 1.5),
)
REACTION_HUB_SSH_FORCE_TTY = _str_to_bool(
    os.environ.get("REACTION_HUB_SSH_FORCE_TTY"),
    default=False,
)
REACTION_HUB_SSH_BATCH_MODE = _str_to_bool(
    os.environ.get("REACTION_HUB_SSH_BATCH_MODE"),
    default=True,
)
REACTION_HUB_SSH_CLEAN_BASH = _str_to_bool(
    os.environ.get("REACTION_HUB_SSH_CLEAN_BASH"),
    default=False,
)
REACTION_HUB_SSH_TIMEOUT_SECONDS = max(
    5,
    _int_env("REACTION_HUB_SSH_TIMEOUT_SECONDS", 60),
)
REACTION_HUB_SSH_BASH_PATH = (os.environ.get("REACTION_HUB_SSH_BASH_PATH", "bash") or "bash").strip() or "bash"

DEFAULT_FROM_EMAIL = os.environ.get("DEFAULT_FROM_EMAIL", "noreply@reaction-hub.local")

REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT = os.environ.get(
    "REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT",
    "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
)
REACTION_HUB_SUBMIT_SCRATCH_BASE = os.environ.get(
    "REACTION_HUB_SUBMIT_SCRATCH_BASE",
    "/scratch/PROJECT/GROUP",
)
REACTION_HUB_SUBMIT_DEFAULT_MODULES = _split_csv(
    os.environ.get("REACTION_HUB_SUBMIT_DEFAULT_MODULES"),
    default=["epcc-setup-env", "cray-python", "orca/6.0.0"],
)
REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH = os.environ.get(
    "REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH",
    "",
)
REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH = _split_csv(
    os.environ.get("REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH"),
    default=[],
)
REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT = os.environ.get(
    "REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT",
    "SLURM_ACCOUNT",
)
REACTION_HUB_SUBMIT_DEFAULT_PARTITION = os.environ.get(
    "REACTION_HUB_SUBMIT_DEFAULT_PARTITION",
    "standard",
)
REACTION_HUB_SUBMIT_DEFAULT_QOS = os.environ.get(
    "REACTION_HUB_SUBMIT_DEFAULT_QOS",
    "standard",
)
REACTION_HUB_SUBMIT_DEFAULT_WALLTIME = os.environ.get(
    "REACTION_HUB_SUBMIT_DEFAULT_WALLTIME",
    "24:00:00",
)
REACTION_HUB_SUBMIT_TDDFT_DEFAULT_QOS = os.environ.get(
    "REACTION_HUB_SUBMIT_TDDFT_DEFAULT_QOS",
    "long",
)
REACTION_HUB_SUBMIT_TDDFT_DEFAULT_WALLTIME = os.environ.get(
    "REACTION_HUB_SUBMIT_TDDFT_DEFAULT_WALLTIME",
    "96:00:00",
)
REACTION_HUB_SUBMIT_AL_STEPS_PER_ITER = max(
    1,
    _int_env("REACTION_HUB_SUBMIT_AL_STEPS_PER_ITER", 200),
)
REACTION_HUB_SUBMIT_MAX_AL_STEPS_PER_ITER = _optional_positive_int_env(
    "REACTION_HUB_SUBMIT_MAX_AL_STEPS_PER_ITER",
    default=None,
)
REACTION_HUB_ACTIVE_RESERVATIONS: list[dict] = []
REACTION_HUB_ASE_GUI_COMMAND = os.environ.get(
    "REACTION_HUB_ASE_GUI_COMMAND",
    "python -m ase gui",
)
REACTION_HUB_XYZRENDER_COMMAND = os.environ.get(
    "REACTION_HUB_XYZRENDER_COMMAND",
    "",
)
REACTION_HUB_XYZRENDER_GIF_ROT = os.environ.get(
    "REACTION_HUB_XYZRENDER_GIF_ROT",
    "xy",
)
REACTION_HUB_XYZRENDER_ROT_FRAMES = max(
    1,
    _int_env("REACTION_HUB_XYZRENDER_ROT_FRAMES", 200),
)
REACTION_HUB_XYZRENDER_GIF_SIZE = os.environ.get(
    "REACTION_HUB_XYZRENDER_GIF_SIZE",
    "900x900",
)
REACTION_HUB_XYZRENDER_GIF_WIDTH = _int_env("REACTION_HUB_XYZRENDER_GIF_WIDTH", 0)
REACTION_HUB_XYZRENDER_GIF_HEIGHT = _int_env("REACTION_HUB_XYZRENDER_GIF_HEIGHT", 0)
REACTION_HUB_XYZRENDER_FIXED_SCALE_ANCHOR_SYMBOL = os.environ.get(
    "REACTION_HUB_XYZRENDER_FIXED_SCALE_ANCHOR_SYMBOL",
    "H",
)

REACTION_HUB_SUBMIT_TRH408_HOST = os.environ.get(
    "REACTION_HUB_SUBMIT_TRH408_HOST",
    "trh408",
).strip()
REACTION_HUB_SUBMIT_TRH408_PORT = max(
    1,
    _int_env("REACTION_HUB_SUBMIT_TRH408_PORT", 22),
)
REACTION_HUB_SUBMIT_TRH408_USERNAME = os.environ.get(
    "REACTION_HUB_SUBMIT_TRH408_USERNAME",
    "",
).strip()
REACTION_HUB_SUBMIT_TRH408_VENV_ACTIVATE_PATH = os.environ.get(
    "REACTION_HUB_SUBMIT_TRH408_VENV_ACTIVATE_PATH",
    "/srv/research/mace_venv/bin/activate",
)
REACTION_HUB_SUBMIT_TRH408_PYTHONPATH = _split_csv(
    os.environ.get("REACTION_HUB_SUBMIT_TRH408_PYTHONPATH"),
    default=["/srv/research/github/cplas"],
)
REACTION_HUB_SUBMIT_TRH408_MODULES = _split_csv(
    os.environ.get("REACTION_HUB_SUBMIT_TRH408_MODULES"),
    default=[],
)
REACTION_HUB_SUBMIT_TRH408_REMOTE_RUNS_ROOT = os.environ.get(
    "REACTION_HUB_SUBMIT_TRH408_REMOTE_RUNS_ROOT",
    "/home/{username}/reaction_hub_runs",
)
REACTION_HUB_SUBMIT_TRH408_SCRATCH_BASE = os.environ.get(
    "REACTION_HUB_SUBMIT_TRH408_SCRATCH_BASE",
    "",
)
REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS = _bool_env(
    "REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS",
    True,
)

REACTION_HUB_SUBMIT_FALCON_HOST = os.environ.get(
    "REACTION_HUB_SUBMIT_FALCON_HOST",
    "falcon",
)
REACTION_HUB_SUBMIT_FALCON_PORT = max(
    1,
    _int_env("REACTION_HUB_SUBMIT_FALCON_PORT", 22),
)
REACTION_HUB_SUBMIT_FALCON_USERNAME = os.environ.get(
    "REACTION_HUB_SUBMIT_FALCON_USERNAME",
    "",
).strip()
REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH = os.environ.get(
    "REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH",
    "/scratch/PROJECT/shared/venv/mace_venv/bin/activate",
)
REACTION_HUB_SUBMIT_FALCON_PYTHONPATH = _split_csv(
    os.environ.get("REACTION_HUB_SUBMIT_FALCON_PYTHONPATH"),
    default=[
        "/scratch/PROJECT/shared/software/github/cplas",
        "/scratch/PROJECT/shared/software/github/carmm",
        "/scratch/PROJECT/shared/software/github/CatLearn",
    ],
)
REACTION_HUB_SUBMIT_FALCON_MODULES = _split_csv(
    os.environ.get("REACTION_HUB_SUBMIT_FALCON_MODULES"),
    default=[
        "Python/3.11.5-GCCcore-13.2.0",
        "ORCA",
    ],
)
REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT = os.environ.get(
    "REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT",
    "/shared/home1/$USER/reaction_hub_runs",
)
REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE = os.environ.get(
    "REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE",
    "/scratch/PROJECT/$USER/",
)
REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT = os.environ.get(
    "REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT",
    "FALCON_SLURM_ACCOUNT",
)
REACTION_HUB_SUBMIT_FALCON_MEMORY = os.environ.get(
    "REACTION_HUB_SUBMIT_FALCON_MEMORY",
    "720G",
)
REACTION_HUB_COADSORBATE_MACE_MODEL_PATH = str(
    os.environ.get("REACTION_HUB_COADSORBATE_MACE_MODEL_PATH", "")
).strip()
_coadsorbate_model_path_by_hpc_default = {
    "trh408": "/srv/research/mace_models/mace-omat-0-medium.model",
}
_coadsorbate_model_path_by_hpc_raw = _json_env(
    "REACTION_HUB_COADSORBATE_MACE_MODEL_PATH_BY_HPC_JSON",
    default=_coadsorbate_model_path_by_hpc_default,
)
if isinstance(_coadsorbate_model_path_by_hpc_raw, dict):
    REACTION_HUB_COADSORBATE_MACE_MODEL_PATH_BY_HPC = {
        str(key).strip().lower(): str(value).strip()
        for key, value in _coadsorbate_model_path_by_hpc_raw.items()
        if str(key).strip() and str(value).strip()
    }
else:
    REACTION_HUB_COADSORBATE_MACE_MODEL_PATH_BY_HPC = dict(_coadsorbate_model_path_by_hpc_default)

REACTION_HUB_SUBMIT_SLURM_PRESETS = _json_env(
    "REACTION_HUB_SUBMIT_SLURM_PRESETS_JSON",
    default={
        "__default__": {
            "default": "custom",
            "presets": [
                {"id": "custom", "label": "Custom (manual)", "values": {}},
            ],
        },
        "trh408": {
            "default": "trh408_mace_gpu",
            "presets": [
                {"id": "custom", "label": "Custom (manual)", "values": {}},
                {
                    "id": "trh408_mace_gpu",
                    "label": "TRH408 MACE GPU (home run)",
                    "values": {
                        "nodes": 1,
                        "tasks_per_node": 1,
                        "cpus_per_task": 8,
                        "walltime": "24:00:00",
                        "account": "",
                        "partition": "",
                        "qos": "",
                    },
                },
            ],
        },
        "falcon": {
            "default": "falcon_dev_1h",
            "presets": [
                {"id": "custom", "label": "Custom (manual)", "values": {}},
                {
                    "id": "falcon_dev_1h",
                    "label": "FALCON Dev/Test (1h, dev)",
                    "values": {
                        "nodes": 1,
                        "tasks_per_node": 192,
                        "cpus_per_task": 1,
                        "walltime": "01:00:00",
                        "account": REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT,
                        "partition": "dev",
                        "qos": "",
                    },
                },
                {
                    "id": "falcon_standard_24h",
                    "label": "FALCON Standard (24h, compute)",
                    "values": {
                        "nodes": 1,
                        "tasks_per_node": 192,
                        "cpus_per_task": 1,
                        "walltime": "24:00:00",
                        "account": REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT,
                        "partition": "compute",
                        "qos": "",
                    },
                },
                {
                    "id": "falcon_standard_72h",
                    "label": "FALCON Standard (72h, compute)",
                    "values": {
                        "nodes": 1,
                        "tasks_per_node": 192,
                        "cpus_per_task": 1,
                        "walltime": "72:00:00",
                        "account": REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT,
                        "partition": "compute",
                        "qos": "",
                    },
                },
            ],
        },
    },
)
