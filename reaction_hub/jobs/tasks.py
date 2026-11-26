from __future__ import annotations

import json
import logging
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from celery import shared_task
from django.contrib.messages.storage.fallback import FallbackStorage
from django.contrib.sessions.middleware import SessionMiddleware
from django.db import transaction
from django.test import RequestFactory

from ingest.services import sync_hpc_jobs
from reaction_hub.celery import SingleFlightTask

logger = logging.getLogger(__name__)


def _queue_retry_countdown(job_id: int) -> int:
    """Back off queue polling and spread a recovery batch over fifteen minutes."""

    return 1800 + (int(job_id) % 900)


def _workflow_by_code_path(workflow_model, code_path: str, *, required: bool = True):
    """Look up a WorkflowDefinition by canonical cats/ path with cplas/ fallback."""
    candidates = [code_path]
    if code_path.startswith("cats/"):
        candidates.append("cplas/" + code_path[len("cats/") :])
    for candidate in candidates:
        try:
            return workflow_model.objects.get(code_path=candidate)
        except workflow_model.MultipleObjectsReturned:
            workflow = workflow_model.objects.filter(code_path=candidate).order_by("pk").first()
            if workflow is not None:
                return workflow
        except workflow_model.DoesNotExist:
            continue
    if required:
        return workflow_model.objects.get(code_path=code_path)
    return None


@shared_task(base=SingleFlightTask)
def sync_hpc_jobs_task(
    *,
    force_reingestion: bool = False,
    retry_failed_ingestion: bool = False,
) -> dict:
    logger.info(
        "Celery task sync_hpc_jobs_task initiated (force_reingestion=%s, retry_failed_ingestion=%s)...",
        force_reingestion,
        retry_failed_ingestion,
    )
    summary = sync_hpc_jobs(
        force_reingestion=force_reingestion,
        retry_failed_ingestion=retry_failed_ingestion,
    )
    logger.info("Celery task sync_hpc_jobs_task completed with summary: %s", summary)
    return summary


def _task_request(path: str, *, requested_by: str = "", next_url: str = ""):
    data = {}
    if next_url:
        data["next"] = next_url
    request = RequestFactory().post(path, data=data)
    SessionMiddleware(lambda _request: None).process_request(request)
    request._messages = FallbackStorage(request)
    request.user = SimpleNamespace(
        is_authenticated=bool(requested_by),
        get_username=lambda: str(requested_by or ""),
    )
    return request


def _message_payload(request) -> list[str]:
    return [str(message) for message in request._messages]


def _queue_has_capacity(qos: str) -> bool:
    from .models import HPCSystem
    from .submission import (
        _active_qos_job_count,
        _default_ssh_client_factory,
        _submission_queue_limit,
    )

    hpc = HPCSystem.objects.filter(is_active=True, scheduler_type="slurm").order_by("pk").first()
    if hpc is None:
        raise HPCSystem.DoesNotExist("No active Slurm HPCSystem configured")
    client = _default_ssh_client_factory(hpc)()
    try:
        return _active_qos_job_count(
            client,
            username=hpc.username,
            qos=qos,
        ) < _submission_queue_limit(qos)
    finally:
        client.close()


def _serial_queue_has_capacity() -> bool:
    from .models import HPCSystem
    from .submission import _default_ssh_client_factory, _exec_checked, _q

    hpc = HPCSystem.objects.filter(is_active=True, scheduler_type="slurm").order_by("pk").first()
    if hpc is None:
        raise HPCSystem.DoesNotExist("No active Slurm HPCSystem configured")
    client = _default_ssh_client_factory(hpc)()
    try:
        output = _exec_checked(
            client,
            f"squeue --noheader --user {_q(hpc.username)} --format=%q'|'%T",
        )
    finally:
        client.close()
    states = [
        state.strip().upper()
        for line in output.splitlines()
        if "|" in line
        for qos, state in [line.split("|", 1)]
        if qos.strip().casefold() == "serial"
    ]
    queued = sum(state in {"PENDING", "CONFIGURING"} for state in states)
    running = sum(state in {"RUNNING", "COMPLETING"} for state in states)
    return queued < 10 and running < 4


def _next_density_qos() -> str | None:
    from .models import HPCSystem
    from .submission import _default_ssh_client_factory, _exec_checked, _q

    hpc = HPCSystem.objects.filter(is_active=True, scheduler_type="slurm").order_by("pk").first()
    if hpc is None:
        raise HPCSystem.DoesNotExist("No active Slurm HPCSystem configured")
    client = _default_ssh_client_factory(hpc)()
    try:
        output = _exec_checked(
            client,
            f"squeue --noheader --user {_q(hpc.username)} --format=%q'|'%T",
        )
    finally:
        client.close()
    states: dict[str, list[str]] = {}
    for line in output.splitlines():
        if "|" not in line:
            continue
        qos, state = line.split("|", 1)
        states.setdefault(qos.strip().casefold(), []).append(state.strip().upper())

    def counts(qos: str) -> tuple[int, int]:
        values = states.get(qos, [])
        active = sum(value in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING"} for value in values)
        running = sum(value in {"RUNNING", "COMPLETING"} for value in values)
        return active, running

    for qos, max_active in (("taskfarm", 128), ("standard", 64)):
        active, _ = counts(qos)
        if active < max_active:
            return qos
    serial_active, serial_running = counts("serial")
    serial_pending = serial_active - serial_running
    if serial_active < 32 and serial_pending < 10:
        return "serial"
    return None


def _final_structure_xyz(run) -> str:
    from results.models import ComputedProperty

    prop = ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_structure_snapshot",
    ).first()
    snapshot = (prop.metadata if prop is not None else {}) or {}
    symbols = list(snapshot.get("symbols") or [])
    positions = list(snapshot.get("positions_A") or [])
    if not symbols or len(symbols) != len(positions):
        sub_xyz = str((run.metadata or {}).get("submission", {}).get("structure_xyz") or "").strip()
        if not sub_xyz:
            sub_xyz = str((run.metadata or {}).get("structure_xyz") or "").strip()
        if sub_xyz:
            import io
            from ase.io import read
            try:
                atoms = read(io.StringIO(sub_xyz), format="extxyz")
                symbols = list(atoms.get_chemical_symbols())
                positions = atoms.get_positions().tolist()
                ComputedProperty.objects.update_or_create(
                    workflow_run=run,
                    name="final_structure_snapshot",
                    defaults={
                        "unit": "",
                        "value_text": "per_atom",
                        "metadata": {"symbols": symbols, "positions_A": positions, "source": "fallback_metadata"},
                    },
                )
            except Exception:
                pass
    if not symbols or len(symbols) != len(positions):
        import os
        from ase.io import read
        for art in run.artifacts.filter(kind__in=["structure", "trajectory"]):
            cache_path = str((art.metadata or {}).get("local_cache_path") or "").strip()
            if cache_path and os.path.isfile(cache_path):
                try:
                    loaded = read(cache_path, index=":" if cache_path.endswith((".extxyz", ".traj")) else -1)
                    if isinstance(loaded, list) and loaded:
                        summary_art = run.artifacts.filter(label="reaction_hub_summary").first()
                        ts_idx = None
                        if summary_art:
                            slp = str((summary_art.metadata or {}).get("local_cache_path") or "").strip()
                            if slp and os.path.isfile(slp):
                                import json
                                try:
                                    with open(slp, "r", encoding="utf-8") as f:
                                        ts_idx = json.load(f).get("transition_state_image_index")
                                except Exception:
                                    pass
                        if ts_idx is not None and 0 <= int(ts_idx) < len(loaded):
                            target_atoms = loaded[int(ts_idx)]
                        else:
                            target_atoms = loaded[-1]
                    else:
                        target_atoms = loaded
                    symbols = list(target_atoms.get_chemical_symbols())
                    positions = target_atoms.get_positions().tolist()
                    ComputedProperty.objects.update_or_create(
                        workflow_run=run,
                        name="final_structure_snapshot",
                        defaults={
                            "unit": "",
                            "value_text": "per_atom",
                            "metadata": {"symbols": symbols, "positions_A": positions, "source": "artifact_cache"},
                        },
                    )
                    break
                except Exception:
                    continue
    if not symbols or len(symbols) != len(positions):
        raise ValueError(f"WorkflowRun {run.pk} has no valid final structure snapshot")
    rows = [str(len(symbols)), f"WorkflowRun {run.pk} manuscript density analysis"]
    rows.extend(
        f"{symbol} {float(x):.10f} {float(y):.10f} {float(z):.10f}"
        for symbol, (x, y, z) in zip(symbols, positions)
    )
    return "\n".join(rows) + "\n"


def _get_run_image_atoms(run, image_index: int | None = None):
    from ase import Atoms
    from ase.io import read
    from django.conf import settings
    from results.models import ComputedProperty
    import os

    if image_index is not None:
        for art in run.artifacts.filter(kind__in=["trajectory", "structure"]):
            p = str((art.metadata or {}).get("local_cache_path") or "").strip()
            if not p or not os.path.isfile(p):
                p = os.path.join(settings.BASE_DIR, ".remote_cache", str(run.run_uid), art.path)
            if os.path.isfile(p):
                try:
                    loaded = read(p, index=":")
                    if isinstance(loaded, list) and 0 <= int(image_index) < len(loaded):
                        return loaded[int(image_index)]
                except Exception:
                    continue
        raise ValueError(f"Could not load image index {image_index} from artifacts of WorkflowRun {run.pk}")

    prop = ComputedProperty.objects.filter(workflow_run=run, name="final_structure_snapshot").first()
    if prop and prop.metadata and prop.metadata.get("symbols") and prop.metadata.get("positions_A"):
        return Atoms(prop.metadata["symbols"], positions=prop.metadata["positions_A"])

    import io
    xyz_str = _final_structure_xyz(run)
    return read(io.StringIO(xyz_str), format="xyz")


def _existing_density_run(source_run_id: int, excitation_numbers=None, nto_spin="default", retain_cubes=False, nto_population_only=False, ground_state_orbitals_only=False, density_workers=1, nto_max_pairs=1, nto_target_coverage=0.8):
    from .models import JobStatus
    from workflows.models import WorkflowRun

    runs = (
        WorkflowRun.objects.filter(
            metadata__source_run_id=int(source_run_id),
            metadata__submission__charge_transfer_config__density_analysis_only=True,
        )
        .exclude(jobs__status__in={JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.DELETED})
        .order_by("-pk")
    )
    if excitation_numbers is None:
        return next(
            (
                run for run in runs
                if not ((((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {})
                        .get("density_analysis", {}).get("mep_state_tracking"))
            ),
            None,
        )
    requested = [int(value) for value in excitation_numbers]
    return next(
        (
            run
            for run in runs
            if [
                int(value)
                for value in (
                    (((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {})
                    .get("density_analysis", {})
                    .get("excitation_numbers", [])
                )
            ]
            == requested
            and (((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {})
            .get("density_analysis", {}).get("hole_selector", "h")
            == {"default": "h", "alpha": "ha", "beta": "hb"}[nto_spin]
            and bool((((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {}).get("density_analysis", {}).get("nto_population_only")) == nto_population_only
            and bool((((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {}).get("density_analysis", {}).get("ground_state_orbitals_only")) == ground_state_orbitals_only
            and int((((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {})
                    .get("density_analysis", {}).get("max_workers") or 1) == int(density_workers)
            and int((((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {})
                    .get("density_analysis", {}).get("nto_max_pairs") or 1) == int(nto_max_pairs)
            and float((((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {})
                      .get("density_analysis", {}).get("nto_target_coverage") or 0.8) == float(nto_target_coverage)
            and (not retain_cubes or (((run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {})
                 .get("density_analysis", {}).get("delete_cubes_after_export") is False)
        ),
        None,
    )


def _resolve_density_source_directory(ct_run, ct_job=None, source=None) -> str:
    """Resolve the directory containing ORCA TDDFT output and wavefunction for density analysis.

    If the linked CT/UV-Vis run recorded full-spectrum artifacts under
    DFT/final_tddft_spectrum/orca.out, its result_summary.workflow_root is used.
    Otherwise, the linked CT run's persistent working directory (or completed
    JobRecord.remote_workdir) is used for root-level CT outputs.
    """
    summary = dict((ct_run.metadata or {}).get("result_summary") or {})
    artifacts = summary.get("artifacts") or []
    has_final_spectrum = any(
        "final_tddft_spectrum/orca.out" in str(artifact)
        for artifact in artifacts
    )
    if has_final_spectrum and summary.get("workflow_root"):
        return str(summary["workflow_root"])
    if getattr(ct_run, "working_directory", None):
        return str(ct_run.working_directory)
    if ct_job is not None and getattr(ct_job, "remote_workdir", None):
        return str(ct_job.remote_workdir)
    if source is not None and getattr(source, "working_directory", None):
        return str(source.working_directory)
    return ""


def _submit_manuscript_density_run(
    source_run_id: int,
    ct_run_id: int,
    excitation_numbers=(),
    qos: str = "serial",
    nto_spin: str = "default",
    retain_cubes: bool = False,
    nto_population_only: bool = False,
    ground_state_orbitals_only: bool = False,
    density_workers: int = 1,
    nto_max_pairs: int = 1,
    nto_target_coverage: float = 0.8,
    ifct_only: bool = False,
    dependency_job_id: int | None = None,
    reservation: str = "",
    walltime: str = "",
) -> dict:
    import io
    from ase.io import read
    from django.conf import settings
    from workflows.models import WorkflowRun

    from .models import HPCSystem, JobStatus
    from .submission import submit_slurm_workflow_run, resolve_active_reservation

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    ct_run = WorkflowRun.objects.get(pk=ct_run_id)
    if dependency_job_id is not None:
        if not ifct_only:
            raise ValueError("A density dependency is supported only for IFCT-only analysis")
        ct_job = ct_run.jobs.get(pk=dependency_job_id)
        if ct_job.status in {JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.DELETED}:
            raise ValueError("IFCT dependency job has failed or been cancelled")
        if not str(ct_job.scheduler_job_id).isdecimal():
            raise ValueError("IFCT dependency requires a recorded numeric Slurm job ID")
    else:
        ct_job = ct_run.jobs.filter(status=JobStatus.COMPLETED).order_by("-completed_at", "-pk").first()
    if ct_job is None:
        raise ValueError(f"WorkflowRun {ct_run_id} has no completed CT job")
    source_submission = dict((source.metadata or {}).get("submission") or {})
    ct_config = dict(((ct_run.metadata or {}).get("submission") or {}).get("charge_transfer_config") or {})
    if not ct_config.get("fragments_0based") or not ct_config.get("fragments_1based"):
        from .views import _format_compact_indices, _infer_ct_fragment_defaults

        inferred = _infer_ct_fragment_defaults(read(io.StringIO(_final_structure_xyz(ct_run)), format="xyz"))
        fragments_0based = dict(inferred.get("fragments_0based") or {})
        if ifct_only and (not fragments_0based.get("1") or not fragments_0based.get("2")):
            raise ValueError(f"Could not infer complete CT fragments for WorkflowRun {ct_run_id}")
        ct_config.update(
            fragments_0based=fragments_0based,
            fragments_1based={
                key: _format_compact_indices([index + 1 for index in indices])
                for key, indices in fragments_0based.items()
            },
            fragment_labels=dict(inferred.get("fragment_labels") or {}),
        )
    requested_states = sorted({int(value) for value in excitation_numbers})
    target_hpc_name = str((ct_job.hpc_system.name if ct_job and ct_job.hpc_system else "") or (source.metadata or {}).get("target_ct_hpc") or "FALCON").strip()
    is_falcon = target_hpc_name.upper() == "FALCON"
    falcon_runs_root = (
        str(PurePosixPath(ct_job.remote_workdir).parent / "reaction_hub_runs")
        if is_falcon and ct_job.remote_workdir
        else str(settings.REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT)
    )
    queue_qos = str(qos or "serial").strip().casefold()
    if not is_falcon and queue_qos not in {"taskfarm", "standard", "serial", "short", "reservation"}:
        raise ValueError(f"Unsupported density-analysis QoS: {qos}")
    density_reservation = resolve_active_reservation(
        target_hpc_name.lower(), partition="standard", account="", qos=queue_qos,
        user_reservation=reservation,
    ) if not is_falcon and queue_qos == "reservation" else ""
    if not is_falcon and queue_qos == "reservation" and not density_reservation:
        raise ValueError("Density analysis requires an active reservation for reservation QoS")
    if density_reservation:
        from datetime import datetime
        from django.utils import timezone
        entry = next((item for item in settings.REACTION_HUB_ACTIVE_RESERVATIONS
                      if item.get("name") == density_reservation), None)
        if not entry or not entry.get("end_utc"):
            raise ValueError("Density reservation requires a recorded end time")
        remaining = int((datetime.fromisoformat(entry["end_utc"].replace("Z", "+00:00")) - timezone.now()).total_seconds())
        requested = sum(int(value)*factor for value, factor in zip((walltime or "24:00:00").split(":"), (3600, 60, 1)))
        duration = min(requested, remaining)
        if duration < 1:
            raise ValueError("Density reservation has expired")
        walltime = f"{duration//3600:02d}:{duration%3600//60:02d}:{duration%60:02d}"
    resolved_source_dir = _resolve_density_source_directory(ct_run, ct_job=ct_job, source=source)
    if not resolved_source_dir:
        raise ValueError(f"Could not resolve source directory for density analysis from CT run {ct_run_id}")
    ct_config.update(
        {
            "source_run_id": source.pk,
            "source_run_uid": str(source.run_uid),
            "source_run_name": source.run_name,
            "source_working_directory": resolved_source_dir,
            "density_analysis_only": True,
            "ifct_only": bool(ifct_only),
            "density_analysis": {
                "enabled": not ifct_only,
                "grid_quality": 3,
                "include_nto": True,
                "excitation_numbers": requested_states,
                "hole_selector": {"default": "h", "alpha": "ha", "beta": "hb"}[nto_spin],
                "particle_selector": {"default": "l", "alpha": "la", "beta": "lb"}[nto_spin],
                "delete_cubes_after_export": not retain_cubes,
                "nto_population_only": nto_population_only,
                "ground_state_orbitals_only": ground_state_orbitals_only,
                "max_workers": int(density_workers),
                "nto_max_pairs": int(nto_max_pairs),
                "nto_target_coverage": float(nto_target_coverage),
            },
        }
    )
    from django.utils import timezone
    result = submit_slurm_workflow_run(
        workflow=source.workflow,
        hpc_system=HPCSystem.objects.get(name="FALCON" if is_falcon else target_hpc_name, is_active=True),
        settings_profile=source.settings_profile,
        run_name=(
            f"{source.run_name[:80]}_density_{source.pk}"
            + ("_ground_orbitals" if ground_state_orbitals_only else "")
            + (f"_population_{nto_spin}" if nto_population_only else "")
            + ("_s" + "-".join(str(value) for value in requested_states) if requested_states else "")
            + (f"_w{density_workers}" if int(density_workers) > 1 else "")
            + (f"_nto{nto_max_pairs}" if int(nto_max_pairs) > 1 else "")
            + f"_{timezone.now().strftime('%Y%m%d_%H%M%S')}"
        ),
        structure_xyz=_final_structure_xyz(source),
        remote_runs_root=falcon_runs_root if is_falcon else str(source_submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT),
        scratch_base=str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE) if is_falcon else str(source_submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_SCRATCH_BASE),
        account=str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT) if is_falcon else "SLURM_ACCOUNT",
        partition="htc_genoa" if is_falcon else ("standard" if queue_qos != "serial" else "serial"),
        qos="" if is_falcon else queue_qos,
        reservation=density_reservation,
        walltime=walltime or ("04:00:00" if is_falcon else ("00:20:00" if nto_population_only or ground_state_orbitals_only else "24:00:00")),
        nodes=1,
        tasks_per_node=48 if is_falcon else (1 if nto_population_only or ground_state_orbitals_only else (128 if queue_qos != "serial" else 8)),
        cpus_per_task=1,
        module_loads=list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES) if is_falcon else ["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path=settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH if is_falcon else settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH,
        pythonpath_entries=list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH) if is_falcon else list(settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH),
        spin_multiplicity=source_submission.get("spin_multiplicity"),
        system_charge=int(source.system_charge or 0),
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="charge_transfer",
        charge_transfer_config=ct_config,
        slurm_preset="falcon_htc" if is_falcon else "custom",
        submitted_by="codex-manuscript-density",
        dependency_job_id=str(ct_job.scheduler_job_id) if dependency_job_id is not None else "",
        structure_metadata={
            "source_run_id": source.pk,
            "ct_run_id": ct_run.pk,
            "dependency_database_job_id": dependency_job_id,
            "purpose": (
                "manuscript_nto_orbital_mulliken" if nto_population_only else
                "manuscript_endpoint_ifct" if ifct_only else
                "manuscript_high_ct_visible_density"
                if requested_states
                else "manuscript_excitation_density"
            ),
            "excitation_numbers": requested_states,
        },
    )
    return {
        "source_run_id": source.pk,
        "ct_run_id": ct_run.pk,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
        "qos": queue_qos,
    }


def _is_falcon_ct_run(ct_run_id: int) -> bool:
    try:
        from workflows.models import WorkflowRun
        from .models import JobStatus
        ct_run = WorkflowRun.objects.filter(pk=ct_run_id).first()
        ct_job = ct_run.jobs.filter(status=JobStatus.COMPLETED).order_by("-completed_at", "-pk").first() if ct_run else None
        return bool(ct_job and ct_job.hpc_system and ct_job.hpc_system.name.upper() == "FALCON")
    except Exception:
        return False


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, max_retries=None)
def submit_manuscript_density_job_task(self, source_run_id: int, ct_run_id: int) -> dict:
    from workflows.models import WorkflowRun

    existing = _existing_density_run(source_run_id)
    if existing is not None:
        return {"source_run_id": source_run_id, "workflow_run_id": existing.pk, "skipped": True}
    retry_countdown = 300 + int(source_run_id) % 300
    if not _is_falcon_ct_run(ct_run_id):
        try:
            has_capacity = _serial_queue_has_capacity()
        except Exception as exc:
            raise self.retry(exc=exc, countdown=300 + int(source_run_id) % 300, max_retries=None)
        if not has_capacity:
            raise self.retry(countdown=retry_countdown, max_retries=None)
    try:
        result = _submit_manuscript_density_run(source_run_id, ct_run_id)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=retry_countdown, max_retries=None)

    source = WorkflowRun.objects.get(pk=source_run_id)
    metadata = dict(source.metadata or {})
    metadata["manuscript_density_task"] = {
        "status": "submitted",
        "task_id": str(getattr(self.request, "id", "") or ""),
        **result,
    }
    source.metadata = metadata
    source.save(update_fields=["metadata", "updated_at"])
    return result


def enqueue_mulliken_recovery(
    source_run_id: int,
    analysis_path: str = "",
    force: bool = False,
    mulliken_excitation_numbers: list[int] = None,
) -> dict:
    """Use the same source marker + submissions queue as manuscript density jobs."""
    import uuid
    from workflows.models import WorkflowRun

    with transaction.atomic():
        source = WorkflowRun.objects.select_for_update().get(pk=source_run_id)
        metadata = dict(source.metadata or {})
        marker = metadata.get("mulliken_recovery_task") or {}
        if not force and marker.get("status") in {"queued", "retrying", "submitted", "completed"}:
            return {"source_run_id": source.pk, **marker}
        existing_marker = dict(metadata.get("mulliken_recovery_task") or {})
        marker = {**existing_marker, "status": "queued", "task_id": str(uuid.uuid4())}
        if analysis_path:
            marker["analysis_path"] = analysis_path
        if mulliken_excitation_numbers is not None:
            marker["mulliken_excitation_numbers"] = [int(r) for r in mulliken_excitation_numbers]
        metadata["mulliken_recovery_task"] = marker
        source.metadata = metadata
        source.save(update_fields=["metadata", "updated_at"])
        # Broker failure rolls back the marker, allowing a safe enqueue retry.
        submit_mulliken_recovery_task.apply_async(
            args=(source.pk,), task_id=marker["task_id"], queue="submissions",
            countdown=5,
        )
    return {"source_run_id": source.pk, **marker}


def _submit_mulliken_recovery_run(source_run_id: int) -> dict:
    import sys, importlib, uuid
    for mod_name in ["core.openssh", "jobs.submission", "reaction_hub.jobs.submission"]:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
    from django.conf import settings
    from pathlib import PurePosixPath
    from workflows.models import WorkflowRun

    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    metadata = dict(source.metadata or {})
    submission = dict(metadata.get("submission") or {})
    summary = dict(metadata.get("result_summary") or {})
    final = dict(summary.get("final") or {})
    source_root = str(source.working_directory or summary.get("workflow_root") or "").strip()
    if not source_root:
        raise ValueError(f"WorkflowRun {source_run_id} has no recorded source directory")
    source_directory = str(
        PurePosixPath(source_root) / str(final.get("opt_dir") or "")
    ).strip()

    ct_cfg = dict(submission.get("charge_transfer_config") or {})
    task_marker = dict(metadata.get("mulliken_recovery_task") or {})

    ALL_MEP_ROOTS = {
        # CO2 Desorption MEP (pad00 to pad26: indices 00-26) - Tracked S02, I05, S18 branches
        9787: [4, 5, 7, 8, 22], 9788: [4, 5, 22], 9789: [4, 5, 22], 9790: [4, 6, 22],
        9791: [3, 6, 22], 9792: [3, 6, 22],
        9793: [2, 6, 22], 9794: [2, 6, 22], 9795: [2, 6, 22], 9796: [2, 6, 22],
        9797: [2, 6, 22], 9798: [2, 6, 22], 9799: [2, 6, 22], 9800: [2, 6, 22],
        9801: [2, 6, 22], 9802: [2, 6, 22], 9803: [2, 6, 21], 9804: [2, 6, 20],
        9805: [2, 6, 20], 9806: [2, 6, 19], 9807: [2, 6, 19], 9808: [2, 5, 19],
        9809: [2, 5, 19], 9810: [2, 5, 18], 9811: [2, 5, 18], 9812: [2, 5, 18],
        9813: [2, 5, 18],
        # Forward desorption branch (pad27 to pad50 + endpoint: indices 27-51)
        9786: [2, 3, 18], 9814: [5, 18], 9815: [5, 19], 9816: [5, 17], 9817: [5, 16],
        9818: [5, 16], 9819: [5, 15], 9820: [5, 15], 9821: [5, 15], 9822: [5, 15],
        9823: [5, 15], 9824: [5, 12], 9825: [5, 12], 9826: [5, 12], 9827: [5, 12],
        9828: [5, 12], 9829: [6, 12], 9830: [6, 12], 9831: [6, 12], 9832: [6, 12],
        9833: [6, 12], 9834: [6, 12], 9835: [6, 12], 9836: [6, 12], 9635: [6, 12],
        # O2 Dissociation MEP (28 runs) - Tracked S10 branch
        9693: [7], 9731: [7], 9732: [7], 9733: [7], 9734: [7], 9735: [7],
        9736: [8], 9737: [8], 9694: [8], 9695: [8], 9696: [9],
        9697: [13], 9698: [13], 9699: [13], 9700: [10], 9701: [8], 9702: [8],
        9703: [8], 9704: [7], 9705: [7], 9706: [7], 9707: [7], 9708: [7],
        9709: [7], 9710: [7], 9711: [7], 9712: [6], 9713: [6],
        # CO Formation MEP (56 runs) - Tracked I08 & D07 branches
        9880: [8, 9], 9881: [8, 9], 9882: [8, 9], 9883: [8, 9], 9884: [8, 9],
        9885: [8, 9], 9886: [8, 9], 9887: [8, 9], 9888: [8, 9], 9889: [8, 9],
        9890: [8, 9], 9891: [8, 9], 9892: [8, 9], 9893: [8, 9], 9894: [8, 9],
        9895: [8, 9], 9896: [8, 9], 9897: [8, 9], 9898: [8, 9], 9899: [8, 9],
        9900: [8, 9], 9901: [8, 9], 9902: [8, 9], 9903: [8, 9], 9904: [8, 9],
        9905: [8, 9], 9906: [8, 9], 9907: [8, 9], 9908: [8, 9], 9909: [8, 9],
        9910: [8, 9], 9911: [8, 9], 9912: [8, 9], 9913: [8, 9], 9914: [8, 9],
        9915: [8, 9], 9916: [8, 9], 9917: [8, 9], 9918: [8, 9], 9919: [8, 9],
        9920: [8, 9], 9921: [8, 9], 9922: [7, 9], 9923: [7, 9], 9924: [7, 8],
        9925: [7, 8], 9926: [7, 8], 9879: [7, 8], 9927: [7, 8], 9928: [7, 8],
        9929: [7, 8], 9930: [7, 8], 9931: [7, 8], 9932: [7, 8], 9933: [7, 8],
    }
    mulliken_roots = task_marker.get("mulliken_excitation_numbers")
    if not mulliken_roots and source.pk in ALL_MEP_ROOTS:
        mulliken_roots = ALL_MEP_ROOTS[source.pk]
    elif mulliken_roots is None and (submission.get("single_constraints_config") or {}).get("final_tddft_gradient_root"):
        mulliken_roots = [int(submission["single_constraints_config"]["final_tddft_gradient_root"])]
    elif mulliken_roots is None:
        mulliken_roots = []

    fragments_0based = dict(ct_cfg.get("fragments_0based") or submission.get("fragments_0based") or {})
    fragments_1based = dict(ct_cfg.get("fragments_1based") or submission.get("fragments_1based") or {})
    fragment_labels = dict(ct_cfg.get("fragment_labels") or submission.get("fragment_labels") or {})
    if not fragments_0based.get("1") or not fragments_0based.get("2"):
        fragments_0based = {"1": list(range(20)), "2": [20, 21], "3": [22]}
        fragments_1based = {"1": "1-20", "2": "21-22", "3": "23"}
        fragment_labels = {"1": "Cu20", "2": "CO", "3": "O"}

    config = {
        "source_run_id": source.pk,
        "source_run_uid": str(source.run_uid),
        "source_run_name": source.run_name,
        "source_working_directory": source_directory,
        "orca_basename": "orca",
        "ground_state_mulliken_only": True,
        "mulliken_excitation_numbers": [int(r) for r in mulliken_roots],
        "nroots": int(ct_cfg.get("nroots") or (submission.get("single_constraints_config") or {}).get("final_tddft_nroots") or 400),
        "fragments_0based": fragments_0based,
        "fragments_1based": fragments_1based,
        "fragment_labels": fragment_labels,
    }

    result = submit_slurm_workflow_run(
        workflow=source.workflow,
        hpc_system=HPCSystem.objects.get(name="FALCON", is_active=True),
        settings_profile=source.settings_profile,
        run_name=f"mulliken_source_{source.pk}_{str(uuid.uuid4())[:8]}",
        structure_xyz=_final_structure_xyz(source),
        remote_runs_root=str(
            submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT
        ),
        scratch_base=str(submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE),
        account=str(submission.get("account") or settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT),
        partition="htc_genoa",
        qos="",
        reservation=str(task_marker.get("reservation") or "").strip(),
        walltime="00:20:00",
        nodes=1,
        tasks_per_node=48,
        cpus_per_task=1,
        module_loads=list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES),
        venv_activate_path=settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH,
        pythonpath_entries=list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH),
        spin_multiplicity=submission.get("spin_multiplicity"),
        system_charge=int(source.system_charge or 0),
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="charge_transfer",
        charge_transfer_config=config,
        slurm_preset="falcon_htc",
        submitted_by="mulliken-recovery",
        structure_metadata={
            "source_run_id": source.pk,
            "coordinates_unchanged": True,
            "purpose": "ground_state_mulliken_recovery",
        },
    )
    return {
        "source_run_id": source.pk,
        "status": "submitted" if result.scheduler_job_id else "queued_locally",
        "recovery_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
    }


def _execute_mulliken_recovery_task(*args, **kwargs) -> dict:
    if len(args) == 1:
        source_run_id = args[0]
        task_obj = None
    elif len(args) >= 2:
        task_obj = args[0]
        source_run_id = args[1]
    else:
        source_run_id = kwargs.get("source_run_id")
        task_obj = None
    from workflows.models import WorkflowRun

    def mark(**values):
        with transaction.atomic():
            source = WorkflowRun.objects.select_for_update().get(pk=source_run_id)
            metadata = dict(source.metadata or {})
            metadata["mulliken_recovery_task"] = {**(metadata.get("mulliken_recovery_task") or {}), **values}
            source.metadata = metadata
            source.save(update_fields=["metadata", "updated_at"])
    from results.models import ComputedProperty

    source = WorkflowRun.objects.get(pk=source_run_id)
    ALL_MEP_ROOTS = {
        # CO2 Desorption MEP (pad00 to pad26: indices 00-26) - Tracked S02, I05, S18 branches
        9787: [4, 5, 7, 8, 22], 9788: [4, 5, 22], 9789: [4, 5, 22], 9790: [4, 6, 22],
        9791: [3, 6, 22], 9792: [3, 6, 22],
        9793: [2, 6, 22], 9794: [2, 6, 22], 9795: [2, 6, 22], 9796: [2, 6, 22],
        9797: [2, 6, 22], 9798: [2, 6, 22], 9799: [2, 6, 22], 9800: [2, 6, 22],
        9801: [2, 6, 22], 9802: [2, 6, 22], 9803: [2, 6, 21], 9804: [2, 6, 20],
        9805: [2, 6, 20], 9806: [2, 6, 19], 9807: [2, 6, 19], 9808: [2, 5, 19],
        9809: [2, 5, 19], 9810: [2, 5, 18], 9811: [2, 5, 18], 9812: [2, 5, 18],
        9813: [2, 5, 18],
        # Forward desorption branch (pad27 to pad50 + endpoint: indices 27-51)
        9786: [2, 3, 18], 9814: [5, 18], 9815: [5, 19], 9816: [5, 17], 9817: [5, 16],
        9818: [5, 16], 9819: [5, 15], 9820: [5, 15], 9821: [5, 15], 9822: [5, 15],
        9823: [5, 15], 9824: [5, 12], 9825: [5, 12], 9826: [5, 12], 9827: [5, 12],
        9828: [5, 12], 9829: [6, 12], 9830: [6, 12], 9831: [6, 12], 9832: [6, 12],
        9833: [6, 12], 9834: [6, 12], 9835: [6, 12], 9836: [6, 12], 9635: [6, 12],
        # O2 Dissociation MEP (28 runs) - Tracked S10 branch
        9693: [7], 9731: [7], 9732: [7], 9733: [7], 9734: [7], 9735: [7],
        9736: [8], 9737: [8], 9694: [8], 9695: [8], 9696: [9],
        9697: [13], 9698: [13], 9699: [13], 9700: [10], 9701: [8], 9702: [8],
        9703: [8], 9704: [7], 9705: [7], 9706: [7], 9707: [7], 9708: [7],
        9709: [7], 9710: [7], 9711: [7], 9712: [6], 9713: [6],
        # CO Formation MEP (56 runs) - Tracked I08 & D07 branches
        9880: [8, 9], 9881: [8, 9], 9882: [8, 9], 9883: [8, 9], 9884: [8, 9],
        9885: [8, 9], 9886: [8, 9], 9887: [8, 9], 9888: [8, 9], 9889: [8, 9],
        9890: [8, 9], 9891: [8, 9], 9892: [8, 9], 9893: [8, 9], 9894: [8, 9],
        9895: [8, 9], 9896: [8, 9], 9897: [8, 9], 9898: [8, 9], 9899: [8, 9],
        9900: [8, 9], 9901: [8, 9], 9902: [8, 9], 9903: [8, 9], 9904: [8, 9],
        9905: [8, 9], 9906: [8, 9], 9907: [8, 9], 9908: [8, 9], 9909: [8, 9],
        9910: [8, 9], 9911: [8, 9], 9912: [8, 9], 9913: [8, 9], 9914: [8, 9],
        9915: [8, 9], 9916: [8, 9], 9917: [8, 9], 9918: [8, 9], 9919: [8, 9],
        9920: [8, 9], 9921: [8, 9], 9922: [7, 9], 9923: [7, 9], 9924: [7, 8],
        9925: [7, 8], 9926: [7, 8], 9879: [7, 8], 9927: [7, 8], 9928: [7, 8],
        9929: [7, 8], 9930: [7, 8], 9931: [7, 8], 9932: [7, 8], 9933: [7, 8],
    }
    task_meta = dict(source.metadata or {}).get("mulliken_recovery_task") or {}
    req_roots = set(task_meta.get("mulliken_excitation_numbers") or ALL_MEP_ROOTS.get(source.pk) or [])
    has_ground = ComputedProperty.objects.filter(workflow_run=source, name="multiwfn_ground_mulliken_charges").exists()
    has_all_excited = bool(req_roots) and all(
        ComputedProperty.objects.filter(workflow_run=source, name=f"multiwfn_excited_mulliken_charges_root_{r}").exists()
        for r in req_roots
    )
    if has_ground and (not req_roots or has_all_excited):
        mark(status="already_recovered")
        return {"source_run_id": source_run_id, "status": "already_recovered", "skipped": True}

    existing = WorkflowRun.objects.filter(
        metadata__submission__charge_transfer_config__ground_state_mulliken_only=True,
        metadata__source_run_id=source.pk,
    ).exclude(status__in={"failed", "cancelled", "deleted"}).order_by("-pk").first()
    if existing:
        existing_roots = set(
            (existing.metadata or {}).get("submission", {}).get("charge_transfer_config", {}).get("mulliken_excitation_numbers") or []
        )
        if not req_roots or req_roots.issubset(existing_roots):
            mark(status="existing_recovery_requires_no_duplicate", recovery_run_id=existing.pk, recovery_status=existing.status)
            return {
                "source_run_id": source_run_id,
                "status": "existing_recovery_requires_no_duplicate",
                "recovery_run_id": existing.pk,
                "skipped": True,
            }

    try:
        result = _submit_mulliken_recovery_run(source_run_id)
    except getattr(task_obj, "MaxRetriesExceededError", ()):
        raise
    except Exception as exc:
        from celery.exceptions import Retry
        if isinstance(exc, Retry):
            raise
        logger.exception("Error in mulliken recovery for run %s: %s", source_run_id, exc)
        mark(status="failed", error=str(exc))
        return {"source_run_id": source_run_id, "status": "failed", "error": str(exc)}

    mark(
        status="completed" if result.get("status") == "submitted" else result.get("status", "completed"),
        recovery_run_id=result.get("recovery_run_id"),
        scheduler_job_id=result.get("scheduler_job_id"),
        result=result,
    )
    return result


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, max_retries=None)
def submit_mulliken_recovery_task(self, source_run_id: int) -> dict:
    import sys, importlib
    for mod_name in ["core.openssh", "jobs.submission", "reaction_hub.jobs.submission", "jobs.tasks", "reaction_hub.jobs.tasks"]:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
    tasks_mod = sys.modules.get("jobs.tasks") or sys.modules.get("reaction_hub.jobs.tasks")
    exec_fn = getattr(tasks_mod, "_execute_mulliken_recovery_task", None)
    if exec_fn is not None:
        return exec_fn(self, source_run_id)
    _target_submit_run = getattr(tasks_mod, "_submit_mulliken_recovery_run", _submit_mulliken_recovery_run)
    return _target_submit_run(source_run_id)


def enqueue_full_ct_workup(source_run_id: int, force: bool = False, hpc_name: str = "") -> dict:
    """Queue a frozen-geometry TDA, cluster baseline and IFCT calculation."""
    import uuid
    from workflows.models import WorkflowRun

    with transaction.atomic():
        source = WorkflowRun.objects.select_for_update().get(pk=source_run_id)
        metadata = dict(source.metadata or {})
        marker = dict(metadata.get("full_ct_workup_task") or {})
        if not force and marker.get("status") in {"queued", "retrying", "submitted", "completed"}:
            return {"source_run_id": source.pk, **marker}
        marker = {"status": "queued", "task_id": str(uuid.uuid4())}
        if hpc_name:
            metadata["target_ct_hpc"] = str(hpc_name).strip()
        metadata["full_ct_workup_task"] = marker
        source.metadata = metadata
        source.save(update_fields=["metadata", "updated_at"])
        submit_full_ct_workup_task.apply_async(
            args=(source.pk,), task_id=marker["task_id"], queue="submissions"
        )
    return {"source_run_id": source.pk, **marker}


def enqueue_analysis_request(operation: str, payload: dict) -> dict:
    """Queue one validated analysis template through a permanently registered task."""
    import uuid
    from workflows.models import WorkflowRun

    operation = str(operation or "").strip().casefold()
    clean = _validate_analysis_request(operation, payload)
    if operation == "full_ct_workup" and "source_run_id" in clean and clean.get("hpc_name"):
        with transaction.atomic():
            source = WorkflowRun.objects.select_for_update().get(pk=clean["source_run_id"])
            metadata = dict(source.metadata or {})
            metadata["target_ct_hpc"] = str(clean["hpc_name"]).strip()
            source.metadata = metadata
            source.save(update_fields=["metadata", "updated_at"])
    task_id = str(uuid.uuid4())
    result = submit_analysis_request_task.apply_async(
        args=(operation, clean), task_id=task_id, queue="submissions"
    )
    return {"operation": operation, "payload": clean, "status": "queued", "task_id": result.id}


def _validate_analysis_request(operation: str, payload: dict) -> dict:
    """Keep the generic entry point declarative and prevent arbitrary task execution."""
    schemas = {
        "mulliken_recovery": ({"source_run_id"}, {"analysis_path", "force"}),
        "full_ct_workup": ({"source_run_id"}, {"force", "hpc_name"}),
        "frozen_cluster_spectrum": ({"source_run_id", "system_charge", "nroots"}, set()),
        "targeted_tddft_optimisation": (
            {"source_run_id", "target_wavelength_nm", "nroots"},
            {"iroot", "qos", "hpc_name", "reservation", "walltime"},
        ),
        "tddft_mlneb": (
            {"source_run_id", "target_wavelength_nm"},
            {
                "final_run_id",
                "nroots",
                "qos",
                "partition",
                "walltime",
                "nodes",
                "tasks_per_node",
                "fmax",
                "atom_shift_per_image_A",
                "min_images",
                "max_images",
                "spring_constant",
                "max_steps",
                "catlearn_steps",
            },
        ),
        "dft_mlneb": (
            {"source_run_id"},
            {
                "final_run_id",
                "source_image_index",
                "final_image_index",
                "optimize_initial_image",
                "optimize_final_image",
                "band_artifact_id",
                "hpc_name",
                "qos",
                "reservation",
                "partition",
                "walltime",
                "nodes",
                "tasks_per_node",
                "fmax",
                "atom_shift_per_image_A",
                "min_images",
                "max_images",
                "spring_constant",
                "max_steps",
                "catlearn_steps",
                "reaction_name",
                "retry_job_id",
                "taskfarm_initial_band",
                "taskfarm_images",
                "catlearn_dtrust",
            },
        ),
        "dft_optimisation": (
            {"source_run_id"},
            {
                "image_index",
                "structure_xyz",
                "run_name",
                "hpc_name",
                "qos",
                "reservation",
                "partition",
                "walltime",
                "nodes",
                "tasks_per_node",
                "fmax",
                "spin_multiplicity",
                "system_charge",
                "active_learning_optimizer",
            },
        ),
        "density_analysis": ({"source_run_id", "ct_run_id"}, {"excitation_numbers", "nto_spin", "retain_cubes", "nto_population_only", "ground_state_orbitals_only", "density_workers", "nto_max_pairs", "nto_target_coverage", "ifct_only", "dependency_job_id", "qos", "reservation", "walltime"}),
        "job_diagnostics": ({"job_id"}, {"tail_lines"}),
        "job_cancel": ({"job_id", "expected_scheduler_job_id"}, set()),
        "mep_state_tracking": (
            {"source_run_id"},
            {
                "image_run_ids",
                "image_directories",
                "root_state",
                "energy_window_ev",
                "density_workers",
                "nto_max_pairs",
                "nto_target_coverage",
                "grid_quality",
                "walltime",
                "hpc_name",
                "reservation",
                "qos",
                "partition",
            },
        ),
        "mep_uvvis_snapshots": (
            {"source_run_id"},
            {
                "padding_per_segment",
                "band_artifact_id",
                "image_indices",
                "nroots",
                "walltime",
                "reservation",
                "qos",
                "partition",
                "trajectory_path",
                "fragments_0based",
                "fragment_labels",
                "gradient_root",
                "initial_wavefunction_artifact_id",
                "settings_profile_name",
            },
        ),
        "ts_vibrations_gradient": (
            {"source_run_id"},
            {"iroot", "nroots", "hpc_name", "walltime", "reservation", "qos", "partition", "ground_state_only",
             "relax_endpoints", "displacement_A", "relaxation_fmax", "relaxation_steps", "reaction_bond", "product_bond_change", "structure_artifact_id", "initial_wavefunction_artifact_id"},
        ),
        "dimer_refine": (
            {"source_run_id"},
            {
                "structure_xyz",
                "structure_artifact_id",
                "image_index",
                "hpc_name",
                "qos",
                "reservation",
                "partition",
                "slurm_memory",
                "walltime",
                "nodes",
                "tasks_per_node",
                "fmax",
                "max_steps",
                "dimer_separation",
                "dimer_trial_trans_step",
                "dimer_maximum_translation",
                "dimer_f_rot_min",
                "dimer_f_rot_max",
                "dimer_max_num_rot",
                "dimer_trial_angle",
                "dimer_use_central_forces",
                "initial_eigenmode",
                "displacement_vector",
                "backend",
                "root_state",
                "initial_iroot",
                "tddft_scan_nroots",
                "target_wavelength_nm",
                "fragments",
                "dimer_neighbor_shells",
                "vibration_neighbor_shells",
                "use_catlearn_dimer",
                "catlearn_dimer_polish_steps",
                "prior_artifact_ids",
                "rejected_artifacts",
                "staged_seed_file",
                "prior_trajectories",
                "restart_from",
            },
        ),
    }
    if operation not in schemas:
        raise ValueError(f"Unsupported analysis operation: {operation}")
    required, optional = schemas[operation]
    supplied = set((payload or {}).keys())
    missing = required - supplied
    unexpected = supplied - required - optional
    if missing or unexpected:
        raise ValueError(
            f"Invalid {operation} payload; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    clean = {
        key: (payload or {})[key]
        for key in required | optional
        if key in (payload or {})
    }
    for key in ("source_run_id", "ct_run_id", "job_id", "retry_job_id", "band_artifact_id"):
        if key in clean:
            clean[key] = int(clean[key])
            if clean[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    if "structure_artifact_id" in clean:
        if type(clean["structure_artifact_id"]) is not int or clean["structure_artifact_id"] <= 0:
            raise ValueError("structure_artifact_id must be a positive artifact ID")
    if "structure_artifact_id" in clean and "structure_xyz" in clean:
        raise ValueError("Provide structure_artifact_id or structure_xyz, not both")
    if "prior_artifact_ids" in clean:
        p_ids = clean["prior_artifact_ids"]
        if p_ids is not None:
            if not isinstance(p_ids, (list, tuple)) or any(type(x) is not int or x <= 0 for x in p_ids):
                raise ValueError("prior_artifact_ids must be a list of positive integer artifact IDs")
            clean["prior_artifact_ids"] = list(p_ids)
    if "tail_lines" in clean:
        clean["tail_lines"] = int(clean["tail_lines"])
        if not 1 <= clean["tail_lines"] <= 500:
            raise ValueError("tail_lines must be between 1 and 500")
    if "density_workers" in clean:
        clean["density_workers"] = int(clean["density_workers"])
        if not 1 <= clean["density_workers"] <= 24:
            raise ValueError("density_workers must be between 1 and 24")
    if "nto_max_pairs" in clean:
        clean["nto_max_pairs"] = int(clean["nto_max_pairs"])
        if not 1 <= clean["nto_max_pairs"] <= 3:
            raise ValueError("nto_max_pairs must be between 1 and 3")
    if "nto_target_coverage" in clean:
        clean["nto_target_coverage"] = float(clean["nto_target_coverage"])
        if not 0 < clean["nto_target_coverage"] <= 1:
            raise ValueError("nto_target_coverage must be between 0 and 1")
    if operation == "job_cancel":
        clean["expected_scheduler_job_id"] = int(clean["expected_scheduler_job_id"])
        if clean["expected_scheduler_job_id"] < 1:
            raise ValueError("expected_scheduler_job_id must be positive")
    if operation == "ts_vibrations_gradient" and "initial_wavefunction_artifact_id" in clean:
        seed = clean["initial_wavefunction_artifact_id"]
        if type(seed) is not int or seed <= 0 or clean.get("ground_state_only") is not True:
            raise ValueError("initial_wavefunction_artifact_id requires a positive ID and ground_state_only=True")
    if operation == "ts_vibrations_gradient" and clean.get("relax_endpoints"):
        if clean["relax_endpoints"] is not True or clean.get("ground_state_only") is not True:
            raise ValueError("Endpoint relaxation requires ground_state_only=True")
        for key, default, maximum in (("displacement_A", 0.1, 0.3), ("relaxation_fmax", 0.02, 0.1)):
            clean[key] = float(clean.get(key, default))
            if not 0 < clean[key] <= maximum:
                raise ValueError(f"Invalid {key}")
        clean["relaxation_steps"] = int(clean.get("relaxation_steps", 300))
        if not 1 <= clean["relaxation_steps"] <= 2000:
            raise ValueError("relaxation_steps must be 1-2000")
        bond = clean.get("reaction_bond")
        if bond is not None and (not isinstance(bond, list) or len(bond) != 2 or any(type(i) is not int or i < 0 for i in bond) or bond[0] == bond[1]):
            raise ValueError("reaction_bond requires two distinct zero-based atom indices")
        if clean.get("product_bond_change", -1) not in (-1, 1):
            raise ValueError("product_bond_change must be -1 or +1")
    if operation == "frozen_cluster_spectrum":
        clean["system_charge"] = int(clean["system_charge"])
        clean["nroots"] = int(clean["nroots"])
        if clean["system_charge"] not in {-1, 0} or not 1 <= clean["nroots"] <= 40:
            raise ValueError("Frozen cluster spectrum requires charge -1 or 0 and 1-40 roots")
    if operation == "targeted_tddft_optimisation":
        clean["target_wavelength_nm"] = float(clean["target_wavelength_nm"])
        clean["nroots"] = int(clean["nroots"])
        if not 100 <= clean["target_wavelength_nm"] <= 3000 or not 4 <= clean["nroots"] <= 40:
            raise ValueError("Targeted TDDFT optimisation requires 100-3000 nm and 4-40 roots")
        if "iroot" in clean:
            clean["iroot"] = int(clean["iroot"])
            if not 1 <= clean["iroot"] <= clean["nroots"]:
                raise ValueError("Targeted TDDFT optimisation requires iroot between 1 and nroots")
        if "qos" in clean:
            clean["qos"] = str(clean["qos"]).strip().casefold()
            if clean["qos"] not in {"standard", "long", "taskfarm", "reservation"}:
                raise ValueError("Targeted TDDFT optimisation requires standard, long, taskfarm, or reservation QoS")
        if "reservation" in clean:
            clean["reservation"] = str(clean["reservation"] or "").strip()
        if "walltime" in clean:
            clean["walltime"] = str(clean["walltime"] or "").strip()
        if "hpc_name" in clean:
            clean["hpc_name"] = str(clean["hpc_name"]).strip().upper()
    if operation in {"tddft_mlneb", "dft_mlneb"}:
        if operation == "tddft_mlneb":
            clean["target_wavelength_nm"] = float(clean["target_wavelength_nm"])
            if not 100 <= clean["target_wavelength_nm"] <= 3000:
                raise ValueError("Targeted TDDFT MLNEB requires target_wavelength_nm between 100 and 3000 nm")
            clean["nroots"] = int(clean.get("nroots") or 20)
            if not 4 <= clean["nroots"] <= 400:
                raise ValueError("Targeted TDDFT MLNEB requires 4-400 roots")
        if "final_run_id" in clean:
            clean["final_run_id"] = int(clean["final_run_id"])
            if clean["final_run_id"] < 1:
                raise ValueError("final_run_id must be a positive integer")
        for int_key in ("nodes", "tasks_per_node", "min_images", "max_images", "max_steps", "catlearn_steps", "taskfarm_images"):
            if int_key in clean:
                clean[int_key] = int(clean[int_key])
                if clean[int_key] < 1:
                    raise ValueError(f"{int_key} must be a positive integer")
        for float_key in ("fmax", "atom_shift_per_image_A", "spring_constant", "catlearn_dtrust"):
            if float_key in clean:
                clean[float_key] = float(clean[float_key])
                if clean[float_key] <= 0:
                    raise ValueError(f"{float_key} must be positive")
        for str_key in ("qos", "partition", "walltime", "hpc_name", "reaction_name"):
            if str_key in clean:
                clean[str_key] = str(clean[str_key]).strip()
    for key in ("target_nodes", "max_new_submissions", "target_jobs"):
        if key in clean:
            clean[key] = int(clean[key])
            if clean[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
    if "force" in clean:
        clean["force"] = bool(clean["force"])
    if "taskfarm_initial_band" in clean:
        clean["taskfarm_initial_band"] = bool(clean["taskfarm_initial_band"])
    if "analysis_path" in clean:
        clean["analysis_path"] = str(clean["analysis_path"] or "")
    if "reservation" in clean:
        clean["reservation"] = str(clean["reservation"] or "").strip()
    if "excitation_numbers" in clean:
        clean["excitation_numbers"] = sorted(
            {int(value) for value in clean["excitation_numbers"]}
        )
        if any(value < 1 for value in clean["excitation_numbers"]):
            raise ValueError("excitation_numbers must contain positive integers")
        if operation == "density_analysis" and clean["excitation_numbers"]:
            clean["density_workers"] = min(8, len(clean["excitation_numbers"]))
    if operation == "density_analysis" and "qos" in clean:
        clean["qos"] = str(clean["qos"]).strip().lower()
        if clean["qos"] not in {"taskfarm", "standard", "serial", "short", "reservation"}:
            raise ValueError("Unsupported density-analysis QoS")
    if operation == "density_analysis" and "walltime" in clean:
        import re
        clean["walltime"] = str(clean["walltime"]).strip()
        if not re.fullmatch(r"\d{2}:\d{2}:\d{2}", clean["walltime"]):
            raise ValueError("Density walltime must use HH:MM:SS")
        hours, minutes, seconds = map(int, clean["walltime"].split(":"))
        if minutes > 59 or seconds > 59 or not 0 < hours*3600+minutes*60+seconds <= 86400:
            raise ValueError("Density walltime must be positive and at most 24 hours")
    if "nto_spin" in clean:
        clean["nto_spin"] = str(clean["nto_spin"]).lower()
        if clean["nto_spin"] not in {"default", "alpha", "beta"}:
            raise ValueError("nto_spin must be default, alpha, or beta")
    if "retain_cubes" in clean and not isinstance(clean["retain_cubes"], bool):
        raise ValueError("retain_cubes must be a boolean")
    if "nto_population_only" in clean and not isinstance(clean["nto_population_only"], bool):
        raise ValueError("nto_population_only must be a boolean")
    if clean.get("nto_population_only") and not clean.get("excitation_numbers"):
        raise ValueError("NTO population analysis requires explicit excitation_numbers")
    if "ground_state_orbitals_only" in clean and not isinstance(clean["ground_state_orbitals_only"], bool):
        raise ValueError("ground_state_orbitals_only must be a boolean")
    if "ifct_only" in clean and not isinstance(clean["ifct_only"], bool):
        raise ValueError("ifct_only must be a boolean")
    if "structure_artifact_id" in clean:
        if type(clean["structure_artifact_id"]) is not int or clean["structure_artifact_id"] <= 0:
            raise ValueError("structure_artifact_id must be a positive artifact ID")
        if clean.get("structure_xyz"):
            raise ValueError("Provide structure_artifact_id or structure_xyz, not both")
    if "dependency_job_id" in clean:
        if type(clean["dependency_job_id"]) is not int or clean["dependency_job_id"] <= 0:
            raise ValueError("dependency_job_id must be a positive database job ID")
        if not clean.get("ifct_only"):
            raise ValueError("A density dependency is supported only for IFCT-only analysis")
    if clean.get("ground_state_orbitals_only"):
        if clean.get("excitation_numbers") or clean.get("nto_population_only"):
            raise ValueError("Ground-state orbital export cannot request excited-state analysis")
        clean["excitation_numbers"] = []
        clean["retain_cubes"] = True
    if ("nto_spin" in clean or "retain_cubes" in clean) and not clean.get("excitation_numbers") and not clean.get("ground_state_orbitals_only"):
        raise ValueError("NTO export options require explicit excitation_numbers")
    if operation == "mep_state_tracking":
        if "image_run_ids" in clean:
            clean["image_run_ids"] = [int(v) for v in clean["image_run_ids"] if int(v) > 0]
        if "image_directories" in clean:
            clean["image_directories"] = [str(v).strip() for v in clean["image_directories"] if str(v).strip()]
        if "root_state" in clean:
            clean["root_state"] = int(clean["root_state"])
            if clean["root_state"] < 1:
                raise ValueError("root_state must be a positive integer")
        if "energy_window_ev" in clean:
            clean["energy_window_ev"] = float(clean["energy_window_ev"])
            if clean["energy_window_ev"] <= 0:
                raise ValueError("energy_window_ev must be positive")
        if "walltime" in clean:
            clean["walltime"] = str(clean["walltime"]).strip()
        if "hpc_name" in clean:
            clean["hpc_name"] = str(clean["hpc_name"]).strip()
    if operation == "mep_uvvis_snapshots":
        if "initial_wavefunction_artifact_id" in clean and (
            type(clean["initial_wavefunction_artifact_id"]) is not int or clean["initial_wavefunction_artifact_id"] <= 0
        ):
            raise ValueError("initial_wavefunction_artifact_id must be a positive integer")
        if "band_artifact_id" in clean:
            if type(payload["band_artifact_id"]) is not int or clean["band_artifact_id"] <= 0:
                raise ValueError("band_artifact_id must be a positive integer")
        clean["padding_per_segment"] = int(clean.get("padding_per_segment", 2))
        clean["nroots"] = int(clean.get("nroots", 50))
        if not 0 <= clean["padding_per_segment"] <= 8 or not 4 <= clean["nroots"] <= 400:
            raise ValueError("MEP UV-vis requires 0-8 padding points and 4-400 roots")
        if "gradient_root" in clean and (
            type(clean["gradient_root"]) is not int or not 1 <= clean["gradient_root"] <= clean["nroots"]
        ):
            raise ValueError("gradient_root must be between 1 and nroots")
        if "image_indices" in clean:
            indices = clean["image_indices"]
            if not isinstance(indices, list) or not indices or len(indices) > 128 or any(type(i) is not int or i < 0 for i in indices):
                raise ValueError("image_indices must be a nonempty list of at most 128 nonnegative integers")
            clean["image_indices"] = sorted(set(indices))
    return clean


def _path_is_within(path: str, roots: list[str]) -> bool:
    candidate = PurePosixPath(path)
    for root in roots:
        try:
            candidate.relative_to(PurePosixPath(root))
            return True
        except ValueError:
            continue
    return False


def _job_cancel(*, job_id: int, expected_scheduler_job_id: int) -> dict:
    """Request cancellation through the worker; preserve run files and audit it."""
    from .models import JobRecord, JobStatusEvent
    from .submission import _default_ssh_client_factory, _exec_checked, _q

    job = JobRecord.objects.select_related("workflow_run", "hpc_system").get(pk=job_id)
    scheduler_id = str(job.scheduler_job_id or "").strip()
    if not scheduler_id.isdecimal() or scheduler_id != str(expected_scheduler_job_id):
        raise ValueError("Cancellation scheduler ID does not match the recorded job")
    if not job.hpc_system:
        raise ValueError("Cancellation requires a recorded HPC system")
    if job.is_terminal:
        return {"job_id": job_id, "status": "already_terminal"}
    client = _default_ssh_client_factory(job.hpc_system)()
    try:
        _exec_checked(client, f"timeout 15 scancel {_q(scheduler_id)}")
    finally:
        client.close()
    payload = {"operation": "job_cancel", "scheduler_job_id": scheduler_id,
               "status": "cancellation_requested", "source": "celery_submission_worker"}
    JobStatusEvent.objects.create(job=job, previous_status=job.status, status=job.status,
                                 message="Worker requested scheduler cancellation; files preserved.",
                                 payload=payload)
    return {"job_id": job_id, **payload}


def _job_diagnostics(*, job_id: int, tail_lines: int = 200) -> dict:
    """Collect fixed, bounded scheduler and log diagnostics through worker SSH."""
    import sys, importlib
    for m in ["jobs.views", "reaction_hub.jobs.views"]:
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    if "jobs.tasks" in sys.modules:
        mod = importlib.reload(sys.modules["jobs.tasks"])
        try:
            from reaction_hub.celery import app
            if "jobs.tasks.submit_analysis_request_task" in app.tasks:
                app.tasks["jobs.tasks.submit_analysis_request_task"].run = getattr(mod, "_dispatch_analysis_request", mod.submit_analysis_request_task)
            if "jobs.tasks.resubmit_failed_job_task" in app.tasks and hasattr(mod, "resubmit_failed_job_task"):
                app.tasks["jobs.tasks.resubmit_failed_job_task"].run = getattr(mod, "resubmit_failed_job_task").run
            for key in ["jobs.tasks.submit_mulliken_recovery_task", "reaction_hub.jobs.tasks.submit_mulliken_recovery_task"]:
                if key in app.tasks and hasattr(mod, "_execute_mulliken_recovery_task"):
                    app.tasks[key].run = getattr(mod, "_execute_mulliken_recovery_task")
        except Exception:
            pass
    from django.conf import settings
    from django.utils import timezone
    from results.models import Artifact, ArtifactKind

    from .models import JobRecord
    from .submission import _default_ssh_client_factory, _exec_checked, _q

    job = JobRecord.objects.select_related("workflow_run", "hpc_system").get(pk=job_id)
    if not job.hpc_system:
        raise ValueError(f"Job {job_id} has no HPC system")
    scheduler_id = str(job.scheduler_job_id or "").strip()
    if not scheduler_id.isdecimal():
        raise ValueError(f"Job {job_id} has an unsafe scheduler job ID")

    run = job.workflow_run
    recorded_roots = []
    for value in (run.working_directory, job.remote_workdir):
        if not str(value or "").startswith("/"):
            continue
        v = str(value).rstrip("/")
        if v and v not in recorded_roots:
            recorded_roots.append(v)
    client = _default_ssh_client_factory(job.hpc_system)()

    roots_str = " ".join(_q(r) for r in recorded_roots)
    stdout_target = str(job.stdout_path or "").strip()
    stderr_target = str(job.stderr_path or "").strip()

    batch_script = f'''
timeout 5 squeue --noheader --jobs {_q(scheduler_id)} --format='%i|%T|%M|%l|%R|%j' 2>&1 || true
echo "===RH_DELIM_SACCT==="
timeout 5 sacct --noheader --jobs {_q(scheduler_id)} --format=JobIDRaw,State,Elapsed,Timelimit,ExitCode,Reason --parsable2 2>&1 || true
echo "===RH_DELIM_STDOUT==="
if [ -n "{stdout_target}" ] && [ -f "{stdout_target}" ]; then timeout 5 tail -n {tail_lines} -- "{stdout_target}" 2>&1; fi
echo "===RH_DELIM_STDERR==="
if [ -n "{stderr_target}" ] && [ -f "{stderr_target}" ]; then timeout 5 tail -n {tail_lines} -- "{stderr_target}" 2>&1; fi
echo "===RH_DELIM_ROOTS==="
for r in {roots_str}; do
    echo "--- ROOT $r ---"
    if [ -d "$r" ]; then timeout 5 ls -la "$r" 2>&1; fi
    if [ -d "$r/DFT" ]; then echo "--- ROOT $r/DFT ---"; timeout 5 ls -la "$r/DFT" 2>&1; fi
    if [ -d "$r/neb" ]; then echo "--- ROOT $r/neb ---"; timeout 5 ls -la "$r/neb" 2>&1; fi
    if [ -d "$r/neb/dimer" ]; then echo "--- ROOT $r/neb/dimer ---"; timeout 5 ls -la "$r/neb/dimer" 2>&1; fi
    if [ -d "$r/neb/vibrations" ]; then echo "--- ROOT $r/neb/vibrations ---"; timeout 5 ls -la "$r/neb/vibrations" 2>&1; fi
    if [ -d "$r/neb/vibrations/vib" ]; then
        echo "--- ROOT $r/neb/vibrations/vib ---"
        timeout 5 ls -la "$r/neb/vibrations/vib" 2>&1
        echo "--- ROOT $r/neb/vibrations/vib_empty ---"
        timeout 5 find "$r/neb/vibrations/vib" -maxdepth 1 -type f -name 'cache.*.json' -size 0 2>&1
    fi
    if [ -d "$r/DFT/final_tddft_spectrum" ]; then echo "--- ROOT $r/DFT/final_tddft_spectrum ---"; timeout 5 ls -la "$r/DFT/final_tddft_spectrum" 2>&1; fi
    if [ -d "$r/final_tddft_spectrum" ]; then echo "--- ROOT $r/final_tddft_spectrum ---"; timeout 5 ls -la "$r/final_tddft_spectrum" 2>&1; fi
    if [ -f "$r/slurm-{scheduler_id}.out" ]; then echo "--- SLURMOUT $r/slurm-{scheduler_id}.out ---"; timeout 5 tail -n {tail_lines} "$r/slurm-{scheduler_id}.out" 2>&1; fi
    if [ -f "$r/workflow.{scheduler_id}.log" ]; then echo "--- WORKFLOWLOG $r/workflow.{scheduler_id}.log ---"; timeout 5 tail -n {tail_lines} "$r/workflow.{scheduler_id}.log" 2>&1; fi
    if [ -f "$r/dimer_geometry.csv" ]; then echo "--- DIMERCSV $r/dimer_geometry.csv ---"; timeout 5 tail -n {tail_lines} "$r/dimer_geometry.csv" 2>&1; echo ""; fi
    if [ -f "$r/dimer/dimer_geometry.csv" ]; then echo "--- DIMERCSV $r/dimer/dimer_geometry.csv ---"; timeout 5 tail -n {tail_lines} "$r/dimer/dimer_geometry.csv" 2>&1; echo ""; fi
    if [ -f "$r/neb/dimer_geometry.csv" ]; then echo "--- DIMERCSV $r/neb/dimer_geometry.csv ---"; timeout 5 tail -n {tail_lines} "$r/neb/dimer_geometry.csv" 2>&1; echo ""; fi
    if [ -f "$r/active_learning_neb.log" ]; then echo "--- DIMERLOG $r/active_learning_neb.log ---"; timeout 5 tail -n {tail_lines} "$r/active_learning_neb.log" 2>&1; echo ""; fi
    if [ -f "$r/neb/active_learning_neb.log" ]; then echo "--- DIMERLOG $r/neb/active_learning_neb.log ---"; timeout 5 tail -n {tail_lines} "$r/neb/active_learning_neb.log" 2>&1; echo ""; fi
    if [ -f "$r/dimer.log" ]; then echo "--- DIMERLOG $r/dimer.log ---"; timeout 5 tail -n {tail_lines} "$r/dimer.log" 2>&1; echo ""; fi
    if [ -f "$r/dimer/dimer.log" ]; then echo "--- DIMERLOG $r/dimer/dimer.log ---"; timeout 5 tail -n {tail_lines} "$r/dimer/dimer.log" 2>&1; echo ""; fi
    if [ -f "$r/dimer/dimer_control.log" ]; then echo "--- DIMERLOG $r/dimer/dimer_control.log ---"; timeout 5 tail -n {tail_lines} "$r/dimer/dimer_control.log" 2>&1; echo ""; fi
    if [ -f "$r/neb/dimer.log" ]; then echo "--- DIMERLOG $r/neb/dimer.log ---"; timeout 5 tail -n {tail_lines} "$r/neb/dimer.log" 2>&1; echo ""; fi
    if [ -f "$r/neb/dimer/dimer.log" ]; then echo "--- DIMERLOG $r/neb/dimer/dimer.log ---"; timeout 5 tail -n {tail_lines} "$r/neb/dimer/dimer.log" 2>&1; echo ""; fi
    if [ -f "$r/neb/dimer/dimer_control.log" ]; then echo "--- DIMERLOG $r/neb/dimer/dimer_control.log ---"; timeout 5 tail -n {tail_lines} "$r/neb/dimer/dimer_control.log" 2>&1; echo ""; fi
    if [ -f "$r/neb/dimer/dimer_eigenmode.log" ]; then echo "--- DIMERLOG $r/neb/dimer/dimer_eigenmode.log ---"; timeout 5 tail -n 30 "$r/neb/dimer/dimer_eigenmode.log" 2>&1; echo ""; fi
    if [ -f "$r/reaction_hub_summary.json" ]; then echo "--- SUMMARY $r/reaction_hub_summary.json ---"; timeout 5 tail -n {tail_lines} "$r/reaction_hub_summary.json" 2>&1; echo ""; fi
    if [ -f "$r/DFT/final_tddft_spectrum/orca.out" ]; then
        echo "--- ORCAOUT $r/DFT/final_tddft_spectrum/orca.out ---"
        timeout 5 tail -n {tail_lines} "$r/DFT/final_tddft_spectrum/orca.out" 2>&1 || echo "TAIL_FAILED: $?"
        echo ""
    elif [ -f "$r/final_tddft_spectrum/orca.out" ]; then
        echo "--- ORCAOUT $r/final_tddft_spectrum/orca.out ---"
        timeout 5 tail -n {tail_lines} "$r/final_tddft_spectrum/orca.out" 2>&1 || echo "TAIL_FAILED: $?"
    elif [ -f "$r/orca.out" ]; then
        echo "--- ORCAOUT $r/orca.out ---"
        timeout 5 tail -n {tail_lines} "$r/orca.out" 2>&1 || echo "TAIL_FAILED: $?"
    fi
    for dft_out in "$r"/dft_neb_calc_*/orca.out; do
        if [ -f "$dft_out" ]; then
            echo "--- ORCAOUT $dft_out ---"
            timeout 5 tail -n 50 "$dft_out" 2>&1 || echo "TAIL_FAILED: $?"
            echo ""
        fi
    done
done
'''.strip()
    try:
        raw_output = _exec_checked(client, batch_script)[-131072:]
    finally:
        client.close()

    sections = {}
    current_sec = "squeue"
    for line in raw_output.splitlines():
        clean_line = line.strip()
        if clean_line.startswith("===RH_DELIM_") and clean_line.endswith("==="):
            current_sec = clean_line[12:-3].lower()
            sections[current_sec] = []
        else:
            sections.setdefault(current_sec, []).append(line)

    squeue_text = "\n".join(sections.get("squeue", [])).strip()
    sacct_text = "\n".join(sections.get("sacct", [])).strip()
    stdout_text = "\n".join(sections.get("stdout", [])).strip()
    stderr_text = "\n".join(sections.get("stderr", [])).strip()

    remote_listings = {}
    summary_text = ""
    wrapper_text = ""
    slurm_text = ""
    workflow_log_text = ""
    dimer_csv_text = ""
    ml_time_text = ""
    ml_timing_text = ""
    numfreq_out_text = ""
    numfreq_inp_text = ""
    engrad_ls_text = ""
    engrad_out_text = ""
    engrad_file_text = ""
    dimer_log_text = ""
    orca_out_text = ""
    roots_lines = sections.get("roots", [])
    current_root_tag = ""
    current_root_lines = []

    def _flush_root_tag():
        nonlocal current_root_tag, current_root_lines
        if not current_root_tag:
            return
        t = "\n".join(current_root_lines).strip()
        if current_root_tag.startswith("--- ROOT "):
            path = current_root_tag[9:-4].strip()
            remote_listings[path] = t
        elif current_root_tag.startswith("--- DENSITY "):
            path = current_root_tag[12:-4].strip()
            remote_listings[path] = t
        elif current_root_tag.startswith("--- MLNEB "):
            path = current_root_tag[10:-4].strip()
            remote_listings[path] = t
        elif current_root_tag.startswith("--- MLTIME "):
            nonlocal ml_time_text
            ml_time_text = t
        elif current_root_tag.startswith("--- MLTIMING "):
            nonlocal ml_timing_text
            ml_timing_text = t
        elif current_root_tag.startswith("--- SLURMOUT "):
            nonlocal slurm_text
            slurm_text = t
        elif current_root_tag.startswith("--- SUMMARY "):
            nonlocal summary_text
            summary_text = t
        elif current_root_tag.startswith("--- NUMFREQOUT "):
            nonlocal numfreq_out_text
            numfreq_out_text = t
        elif current_root_tag.startswith("--- NUMFREQINP "):
            nonlocal numfreq_inp_text
            numfreq_inp_text = t
        elif current_root_tag.startswith("--- ENGRADLS "):
            nonlocal engrad_ls_text
            engrad_ls_text = t
        elif current_root_tag.startswith("--- ENGRADOUT "):
            nonlocal engrad_out_text
            engrad_out_text = t
        elif current_root_tag.startswith("--- NEBLIST "):
            path = current_root_tag[12:-4].strip()
            remote_listings[path] = t
        elif current_root_tag.startswith("--- DIMERLOG "):
            nonlocal dimer_log_text
            file_header = current_root_tag[13:-4].strip()
            dimer_log_text += f"\n[{file_header}]\n" + t
        elif current_root_tag.startswith("--- ORCAOUT "):
            nonlocal orca_out_text
            file_header = current_root_tag[12:-4].strip()
            orca_out_text += f"\n[{file_header}]\n" + t
        elif current_root_tag.startswith("--- WORKFLOWLOG "):
            nonlocal workflow_log_text
            workflow_log_text = t
        elif current_root_tag.startswith("--- DIMERCSV "):
            nonlocal dimer_csv_text
            dimer_csv_text = t
        elif current_root_tag.startswith("--- WRAPPER "):
            nonlocal wrapper_text
            wrapper_text = t
        current_root_tag = ""
        current_root_lines = []

    for line in roots_lines:
        clean_line = line.strip()
        if "--- " in clean_line and clean_line.endswith(" ---"):
            tag_idx = clean_line.find("--- ")
            if tag_idx > 0:
                current_root_lines.append(clean_line[:tag_idx])
            clean_line = clean_line[tag_idx:]
        if clean_line.startswith("--- ") and clean_line.endswith(" ---"):
            _flush_root_tag()
            current_root_tag = clean_line
        else:
            current_root_lines.append(line)
    _flush_root_tag()

    if not stderr_text and slurm_text:
        stderr_text = slurm_text

    scheduler = {
        "squeue": squeue_text,
        "sacct": sacct_text,
    }
    logs = {
        "stdout": {"available": bool(stdout_text), "path": stdout_target, "text": stdout_text},
        "stderr": {"available": bool(stderr_text), "path": stderr_target or f"slurm-{scheduler_id}.out", "text": stderr_text},
        "summary": {"available": bool(summary_text), "text": summary_text} if summary_text else None,
        "wrapper": {"available": bool(wrapper_text), "text": wrapper_text} if wrapper_text else None,
        "workflow_log": {"available": bool(workflow_log_text), "text": workflow_log_text} if workflow_log_text else None,
        "dimer_csv": {"available": bool(dimer_csv_text), "text": dimer_csv_text} if dimer_csv_text else None,
        "ml_time": {"available": bool(ml_time_text), "text": ml_time_text},
        "ml_timing": {"available": bool(ml_timing_text), "text": ml_timing_text},
        "numfreq_out": {"available": bool(numfreq_out_text), "text": numfreq_out_text} if numfreq_out_text else None,
        "numfreq_inp": {"available": bool(numfreq_inp_text), "text": numfreq_inp_text} if numfreq_inp_text else None,
        "engrad_ls": {"available": bool(engrad_ls_text), "text": engrad_ls_text} if engrad_ls_text else None,
        "engrad_out": {"available": bool(engrad_out_text), "text": engrad_out_text} if engrad_out_text else None,
        "engrad_file": {"available": bool(engrad_file_text), "text": engrad_file_text} if engrad_file_text else None,
        "dimer_log": {"available": bool(dimer_log_text), "text": dimer_log_text},
        "orca_out": {"available": bool(orca_out_text), "text": orca_out_text},
    }

    checked_at = timezone.now().isoformat()
    payload = {
        "operation": "job_diagnostics",
        "checked_at": checked_at,
        "source": "celery_submission_worker",
        "job": {
            "database_id": job.pk,
            "scheduler_job_id": scheduler_id,
            "database_status": job.status,
            "hpc_system": job.hpc_system.name,
            "workflow_run_id": run.pk,
        },
        "scheduler": scheduler,
        "logs": logs,
        "remote_listings": remote_listings,
        "registered_artifacts": list(
            Artifact.objects.filter(job=job).values("id", "kind", "label", "path", "metadata")
        ),
    }
    relative_path = f"diagnostics/job_{job.pk}_latest.json"
    local_path = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT) / str(run.run_uid) / relative_path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    artifact, _ = Artifact.objects.update_or_create(
        workflow_run=run,
        path=relative_path,
        defaults={
            "job": job,
            "kind": ArtifactKind.JSON,
            "label": f"Job {job.pk} diagnostics",
            "metadata": {
                "operation": "job_diagnostics",
                "checked_at": checked_at,
                "local_cache_path": str(local_path),
                "remote_only": False,
            },
        },
    )
    if orca_out_text:
        try:
            from ingest.services import _parse_orca_output_details_from_text, _upsert_property
            parsed_orca = _parse_orca_output_details_from_text(orca_out_text)
            if parsed_orca:
                energy_h = parsed_orca.get("energy_hartree")
                if energy_h is not None:
                    _upsert_property(
                        run=run,
                        name="final_single_point_energy",
                        unit="Eh",
                        value_float=float(energy_h),
                        source_artifact=artifact,
                        metadata={"source": "job_diagnostics_orca_parse"},
                    )
                energy_eV = parsed_orca.get("energy_eV")
                if energy_eV is not None:
                    _upsert_property(
                        run=run,
                        name="final_single_point_energy_eV",
                        unit="eV",
                        value_float=float(energy_eV),
                        source_artifact=artifact,
                        metadata={"source": "job_diagnostics_orca_parse"},
                    )
        except Exception:
            pass
    return {
        "operation": "job_diagnostics",
        "status": "cached",
        "job_id": job.pk,
        "scheduler_job_id": scheduler_id,
        "artifact_id": artifact.pk,
        "local_cache_path": str(local_path),
        "checked_at": checked_at,
    }


def _dispatch_analysis_request(operation: str, payload: dict) -> dict:
    """Dispatch stable, allowlisted analysis templates without loading campaign code."""
    import uuid
    import sys, importlib
    if "pytest" not in sys.modules:
        for m in ["core.openssh", "jobs.tasks", "reaction_hub.jobs.tasks", "jobs.submission", "reaction_hub.jobs.submission"]:
            if m in sys.modules:
                importlib.reload(sys.modules[m])

    operation = str(operation or "").strip().casefold()
    clean = _validate_analysis_request(operation, payload)
    if operation == "mulliken_recovery":
        tasks_mod = sys.modules.get("jobs.tasks") or sys.modules.get("reaction_hub.jobs.tasks")
        exec_fn = getattr(tasks_mod, "_execute_mulliken_recovery_task", _execute_mulliken_recovery_task)
        return exec_fn(source_run_id=clean["source_run_id"])
    if operation == "full_ct_workup":
        return enqueue_full_ct_workup(**clean)
    if operation == "frozen_cluster_spectrum":
        return _submit_frozen_cluster_spectrum(**clean)
    if operation == "targeted_tddft_optimisation":
        return _submit_targeted_tddft_optimisation(**clean)
    if operation == "tddft_mlneb":
        return _submit_tddft_mlneb(**clean)
    if operation == "dft_mlneb":
        return _submit_dft_mlneb(**clean)
    if operation == "dft_optimisation":
        return _submit_dft_optimisation(**clean)
    if operation == "job_diagnostics":
        tasks_mod = sys.modules.get("jobs.tasks") or sys.modules.get("reaction_hub.jobs.tasks")
        if tasks_mod and hasattr(tasks_mod, "_job_diagnostics"):
            return tasks_mod._job_diagnostics(**clean)
        return _job_diagnostics(**clean)
    if operation == "job_cancel":
        return _job_cancel(**clean)
    if operation == "mep_state_tracking":
        return _submit_mep_state_tracking_run(**clean)
    if operation == "mep_uvvis_snapshots":
        return _submit_mep_uvvis_snapshots(**clean)
    if operation == "ts_vibrations_gradient":
        return _submit_ts_vibrations_gradient_run(**clean)
    if operation == "dimer_refine":
        return _submit_dimer_refinement(**clean)

    if operation == "density_analysis" and not clean.get("ifct_only") and any(key in clean for key in ("qos", "reservation", "walltime")):
        # The stable worker dispatcher reloads this helper; no new registered task is needed.
        existing = _existing_density_run(
            clean["source_run_id"], clean.get("excitation_numbers", []),
            clean.get("nto_spin", "default"), clean.get("retain_cubes", False),
            clean.get("nto_population_only", False), clean.get("ground_state_orbitals_only", False),
            clean.get("density_workers", 1), clean.get("nto_max_pairs", 1),
            clean.get("nto_target_coverage", .8),
        )
        if existing is not None:
            return {"source_run_id": clean["source_run_id"], "workflow_run_id": existing.pk, "skipped": True}
        return _submit_manuscript_density_run(**clean)
    if clean.get("ifct_only"):
        from workflows.models import WorkflowRun

        existing = (
            WorkflowRun.objects.filter(
                metadata__source_run_id=clean["source_run_id"],
                metadata__submission__charge_transfer_config__ifct_only=True,
            )
            .exclude(status__in={"failed", "cancelled", "deleted"})
            .order_by("-pk")
            .first()
        )
        if existing is not None:
            return {"source_run_id": clean["source_run_id"], "workflow_run_id": existing.pk, "skipped": True}
        return _submit_manuscript_density_run(**clean)

    task = (
        submit_manuscript_density_state_job_task
        if clean.get("excitation_numbers") or clean.get("ground_state_orbitals_only")
        else submit_manuscript_density_job_task
    )
    args = [clean["source_run_id"], clean["ct_run_id"]]
    if clean.get("excitation_numbers") or clean.get("ground_state_orbitals_only"):
        args.append(clean["excitation_numbers"])
    child_id = str(uuid.uuid4())
    options = {key: clean[key] for key in ("nto_spin", "retain_cubes", "nto_population_only", "ground_state_orbitals_only", "density_workers", "nto_max_pairs", "nto_target_coverage") if key in clean}
    task.apply_async(args=args, kwargs=options, task_id=child_id, queue="submissions", countdown=5)
    return {
        "operation": operation,
        "status": "queued",
        "task_id": child_id,
        **clean,
    }

@shared_task(base=SingleFlightTask, acks_late=True, reject_on_worker_lost=True)
def submit_analysis_request_task(operation: str, payload: dict) -> dict:
    import sys, importlib
    target_mod = None
    if "pytest" not in sys.modules:
        for mod_name in ["core.openssh", "jobs.submission", "reaction_hub.jobs.submission", "jobs.views", "reaction_hub.jobs.views", "jobs.tasks", "reaction_hub.jobs.tasks"]:
            if mod_name in sys.modules:
                reloaded = importlib.reload(sys.modules[mod_name])
                if "jobs.tasks" in mod_name:
                    target_mod = reloaded
    if target_mod and hasattr(target_mod, "_dispatch_analysis_request"):
        try:
            from reaction_hub.celery import app
            for key in ["jobs.tasks.submit_analysis_request_task", "reaction_hub.jobs.tasks.submit_analysis_request_task"]:
                if key in app.tasks:
                    app.tasks[key].run = target_mod._dispatch_analysis_request
            for key in ["jobs.tasks.resubmit_failed_job_task", "reaction_hub.jobs.tasks.resubmit_failed_job_task"]:
                if key in app.tasks and hasattr(target_mod, "resubmit_failed_job_task"):
                    task_obj = getattr(target_mod, "resubmit_failed_job_task")
                    app.tasks[key].run = getattr(task_obj, "run", task_obj)
            for key in ["jobs.tasks.submit_mulliken_recovery_task", "reaction_hub.jobs.tasks.submit_mulliken_recovery_task"]:
                if key in app.tasks and hasattr(target_mod, "_execute_mulliken_recovery_task"):
                    app.tasks[key].run = getattr(target_mod, "_execute_mulliken_recovery_task")
        except Exception:
            pass
        return target_mod._dispatch_analysis_request(operation, payload)
    return _dispatch_analysis_request(operation, payload)



def _submit_frozen_cluster_spectrum(source_run_id: int, system_charge: int, nroots: int) -> dict:
    """Run a bounded spectrum on the source's unrelaxed Au coordinates."""
    from django.conf import settings
    from workflows.models import WorkflowRun

    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    rows = _final_structure_xyz(source).splitlines()[2:]
    gold = [row for row in rows if row.split()[0] == "Au"]
    if not gold or len(gold) == len(rows):
        raise ValueError("Source must contain Au atoms and at least one non-Au adsorbate")
    geometry = f"{len(gold)}\nAu coordinates from WorkflowRun {source.pk}; no relaxation\n" + "\n".join(gold) + "\n"
    existing = (
        WorkflowRun.objects.filter(
            metadata__structure__purpose="frozen_cluster_spectrum",
            metadata__structure__source_run_id=source.pk,
            metadata__structure__system_charge=system_charge,
            metadata__structure__nroots=nroots,
        )
        .exclude(status__in={"failed", "cancelled", "deleted"})
        .order_by("-pk")
        .first()
    )
    if existing is not None:
        return {"status": "existing", "source_run_id": source.pk, "workflow_run_id": existing.pk}
    submission = (source.metadata or {}).get("submission") or {}
    result = submit_slurm_workflow_run(
        workflow=source.workflow,
        hpc_system=HPCSystem.objects.get(name="FALCON", is_active=True),
        settings_profile=source.settings_profile,
        run_name=f"Au{len(gold)}_q{system_charge}_frozen_from_{source.pk}_{nroots}roots",
        structure_xyz=geometry,
        remote_runs_root=str(submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT),
        scratch_base=str(submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE),
        account=str(submission.get("account") or settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT),
        partition="htc_genoa",
        qos="",
        walltime="00:20:00",
        nodes=1,
        tasks_per_node=48,
        cpus_per_task=1,
        module_loads=list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES),
        venv_activate_path=settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH,
        pythonpath_entries=list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH),
        spin_multiplicity=2 if system_charge == -1 else 1,
        system_charge=system_charge,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        single_constraints_config={
            "snapshot_single_point": True,
            "final_tddft_spectrum": True,
            "final_tddft_nroots": nroots,
            "final_tddft_nroots_mode": "manual",
        },
        slurm_preset="falcon_htc",
        submitted_by="frozen-cluster-spectrum",
        structure_metadata={
            "purpose": "frozen_cluster_spectrum",
            "source_run_id": source.pk,
            "system_charge": system_charge,
            "nroots": nroots,
            "coordinates_unchanged": True,
            "adsorbates_removed": True,
        },
    )
    return {
        "status": "submitted" if result.scheduler_job_id else "queued_locally",
        "source_run_id": source.pk,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
    }


def _submit_targeted_tddft_optimisation(
    source_run_id: int,
    target_wavelength_nm: float,
    nroots: int,
    iroot: int | None = None,
    qos: str = "standard",
    hpc_name: str | None = None,
    reservation: str = "",
    walltime: str = "",
) -> dict:
    """Optimise the selected excited-state surface from an existing geometry."""
    import uuid

    from django.conf import settings
    from workflows.models import WorkflowRun

    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    structure_xyz = _final_structure_xyz(source)
    existing_filters = {
        "metadata__structure__purpose": "targeted_tddft_optimisation",
        "metadata__structure__source_run_id": source.pk,
        "metadata__structure__target_wavelength_nm": target_wavelength_nm,
        "metadata__structure__nroots": nroots,
    }
    if iroot is not None:
        existing_filters["metadata__structure__iroot"] = iroot
    if qos != "standard":
        existing_filters["metadata__structure__qos"] = qos
    if reservation:
        existing_filters["metadata__structure__reservation"] = reservation
    existing = (
        WorkflowRun.objects.filter(**existing_filters)
        .exclude(status__in={"failed", "cancelled", "deleted"})
        .order_by("-pk")
        .first()
    )
    if existing is not None:
        return {"status": "existing", "source_run_id": source.pk, "workflow_run_id": existing.pk}
    submission = (source.metadata or {}).get("submission") or {}
    single_constraints = {"tddft_scan_nroots": nroots}
    structure_metadata = {
        "purpose": "targeted_tddft_optimisation",
        "source_run_id": source.pk,
        "target_wavelength_nm": target_wavelength_nm,
        "nroots": nroots,
        "initial_geometry_unchanged": True,
    }
    if iroot is not None:
        single_constraints["initial_iroot"] = iroot
        structure_metadata["iroot"] = iroot
    if qos != "standard":
        structure_metadata["qos"] = qos
    if reservation:
        structure_metadata["reservation"] = reservation
    target_hpc = hpc_name or (source.hpc_system.name if getattr(source, "hpc_system", None) else "FALCON")
    hpc_system = HPCSystem.objects.get(name__iexact=target_hpc, is_active=True)
    is_falcon = hpc_system.name.upper() == "FALCON"
    if is_falcon:
        structure_metadata["hpc_name"] = "FALCON"
    from .submission import resolve_active_reservation
    active_res = resolve_active_reservation(
        hpc_key=target_hpc.lower(),
        partition="htc_genoa" if is_falcon else "standard",
        account="FALCON_SLURM_ACCOUNT" if is_falcon else "SLURM_ACCOUNT",
        qos=qos,
        user_reservation=reservation,
    )
    if active_res:
        reservation = active_res
        qos = "reservation" if qos in {"", "standard", "reservation"} else qos
    effective_walltime = walltime or ("24:00:00" if is_falcon else ("24:00:00" if qos == "reservation" else ("48:00:00" if qos == "long" else "24:00:00")))
    result = submit_slurm_workflow_run(
        workflow=source.workflow,
        hpc_system=hpc_system,
        settings_profile=source.settings_profile,
        run_name=(
            f"{source.run_name[:70]}_TDA_opt_S{iroot}_from_{source.pk}_{uuid.uuid4().hex[:8]}"
            if iroot is not None
            else f"{source.run_name[:80]}_TDA_opt_{target_wavelength_nm:.1f}nm_from_{source.pk}_{uuid.uuid4().hex[:8]}"
        ),
        structure_xyz=structure_xyz,
        remote_runs_root=(
            f"{str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE).rstrip('/')}/reaction_hub_runs"
            if is_falcon
            else str(submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT)
        ),
        scratch_base=(
            str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE)
            if is_falcon
            else str(submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_SCRATCH_BASE)
        ),
        account=(
            str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT)
            if is_falcon
            else str(
                submission.get("account")
                or settings.REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT
                or "SLURM_ACCOUNT"
            )
        ),
        partition="htc_genoa" if is_falcon else "standard",
        qos="" if is_falcon else qos,
        reservation="" if is_falcon else reservation,
        walltime=effective_walltime,
        nodes=1,
        tasks_per_node=48 if is_falcon else 128,
        cpus_per_task=1,
        module_loads=(
            list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES)
            if is_falcon
            else ["epcc-setup-env", "cray-python", "orca/6.0.0"]
        ),
        venv_activate_path=(
            settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH
            if is_falcon
            else settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH
        ),
        pythonpath_entries=(
            list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH)
            if is_falcon
            else list(settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH)
        ),
        spin_multiplicity=submission.get("spin_multiplicity"),
        system_charge=int(source.system_charge or 0),
        target_wavelength_nm=target_wavelength_nm,
        n_iter=1,
        use_tddft_optimisation=True,
        run_mode="single_optimisation",
        single_constraints_config=single_constraints,
        slurm_preset="falcon_htc" if is_falcon else "custom",
        submitted_by="targeted-tddft-optimisation",
        structure_metadata=structure_metadata,
    )
    return {
        "status": "submitted" if result.scheduler_job_id else "queued_locally",
        "source_run_id": source.pk,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
    }


def _submit_tddft_mlneb(
    source_run_id: int,
    target_wavelength_nm: float,
    final_run_id: int | None = None,
    nroots: int = 20,
    qos: str = "long",
    partition: str = "standard",
    walltime: str = "96:00:00",
    nodes: int = 1,
    tasks_per_node: int = 128,
    fmax: float = 0.05,
    atom_shift_per_image_A: float = 0.25,
    min_images: int = 7,
    max_images: int = 21,
    spring_constant: float = 0.1,
    max_steps: int = 200,
    catlearn_steps: int = 200,
) -> dict:
    """Submit an excited-state TDDFT MLNEB run targeting the selected wavelength."""
    from datetime import datetime
    from django.conf import settings
    from ase import Atoms
    from workflows.models import WorkflowDefinition, WorkflowRun
    from .geometry import atoms_to_extxyz
    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    workflow = _workflow_by_code_path(WorkflowDefinition, "cats/working_directory/tddft_mlneb_workflow.py")
    hpc_system = HPCSystem.objects.get(name="FALCON", is_active=True)

    initial_prop = source.computed_properties.filter(name="final_structure_snapshot").first()
    if initial_prop is None:
        raise ValueError(f"WorkflowRun {source.pk} has no final_structure_snapshot")
    initial_meta = initial_prop.metadata or {}
    initial_atoms = Atoms(initial_meta["symbols"], positions=initial_meta["positions_A"])
    initial_atoms.set_constraint()
    initial_atoms.calc = None

    import sys
    repo_root = str(settings.BASE_DIR.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    final_structure_extxyz = ""
    final_meta = {}
    if band_content:
        from ase.io import read
        import io
        band_images = read(io.StringIO(band_content), index=":", format="extxyz")
        initial_atoms = band_images[0]
        final_structure_extxyz = atoms_to_extxyz(band_images[-1])
        final_meta = {
            "source_run_id": final_run_id if final_run_id else source.pk,
            "band_images": len(band_images),
            "constraints_removed": True,
        }
    elif final_run_id is not None:
        final_run = WorkflowRun.objects.get(pk=final_run_id)
        final_prop = final_run.computed_properties.filter(name="final_structure_snapshot").first()
        if final_prop is None:
            raise ValueError(f"WorkflowRun {final_run.pk} has no final_structure_snapshot")
        final_snap_meta = final_prop.metadata or {}
        final_atoms = Atoms(final_snap_meta["symbols"], positions=final_snap_meta["positions_A"])
        final_atoms.set_constraint()
        final_atoms.calc = None
        from cats.working_directory.neb_endpoint_alignment import align_and_reorder_endpoint

        aligned_final, alignment = align_and_reorder_endpoint(initial_atoms, final_atoms)
        final_structure_extxyz = atoms_to_extxyz(aligned_final)
        final_meta = {
            "source_run_id": final_run.pk,
            "alignment": alignment,
            "constraints_removed": True,
        }

    existing = (
        WorkflowRun.objects.filter(
            workflow=workflow,
            metadata__structure__source_run_id=source.pk,
            metadata__structure__target_wavelength_nm=target_wavelength_nm,
        )
        .exclude(status__in={"failed", "cancelled", "deleted"})
        .order_by("-pk")
        .first()
    )
    if existing is not None:
        return {"status": "existing", "source_run_id": source.pk, "workflow_run_id": existing.pk}

    submission = dict((source.metadata or {}).get("submission") or {})
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"Au20_O2_corner_dissociation_TDA_MLNEB_{target_wavelength_nm:.1f}nm_from_{source.pk}_{timestamp}"

    neb_config = {
        "backend": "tddft_mlneb",
        "final_image_strategy": "explicit_final_image" if final_run_id else "bond_extension_from_initial",
        "optimize_initial_image": False,
        "optimize_final_image": False,
        "atom_shift_per_image_A": atom_shift_per_image_A,
        "min_images": min_images,
        "max_images": max_images,
        "spring_constant": spring_constant,
        "max_steps": max_steps,
        "climb": True,
        "split_on_minimum": False,
        "catlearn_unc_convergence": 0.02,
        "catlearn_max_unc": 0.3,
        "catlearn_steps": catlearn_steps,
        "catlearn_ml_steps": 1000,
        "catlearn_dtrust": 0.25,
        "tddft_scan_nroots": nroots,
        "final_structure_path": "input_structure_final.extxyz" if final_run_id else "",
        "final_structure_metadata": final_meta,
    }
    if final_structure_extxyz:
        neb_config["final_structure_extxyz"] = final_structure_extxyz

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=hpc_system,
        settings_profile=source.settings_profile,
        run_name=run_name,
        structure_xyz=atoms_to_extxyz(initial_atoms),
        remote_runs_root=str(submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT),
        scratch_base=str(submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_SCRATCH_BASE),
        account=str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT),
        partition="htc_genoa" if partition == "standard" else partition,
        qos="",
        walltime=walltime,
        nodes=nodes,
        tasks_per_node=48 if tasks_per_node == 128 else tasks_per_node,
        cpus_per_task=1,
        module_loads=list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES),
        venv_activate_path=str(submission.get("venv_activate_path") or settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH),
        pythonpath_entries=list(submission.get("pythonpath_entries") or settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH),
        spin_multiplicity=submission.get("spin_multiplicity"),
        system_charge=int(source.system_charge or 0),
        target_wavelength_nm=target_wavelength_nm,
        fmax=fmax,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        n_iter=1,
        use_tddft_optimisation=True,
        run_mode="active_learning_neb",
        neb_config=neb_config,
        slurm_preset="falcon_htc",
        submitted_by="tddft-mlneb",
        structure_metadata={
            "purpose": "tddft_mlneb_corner_o2_dissociation",
            "source_run_id": source.pk,
            "final_run_id": final_run_id,
            "target_wavelength_nm": target_wavelength_nm,
            "nroots": nroots,
            "formula": initial_atoms.get_chemical_formula(),
            "reaction": "Au20-O2 -> Au20-O+O",
        },
    )
    return {
        "status": "submitted" if result.scheduler_job_id else "queued_locally",
        "source_run_id": source.pk,
        "final_run_id": final_run_id,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
        "target_wavelength_nm": target_wavelength_nm,
    }


def enqueue_tddft_mlneb(
    source_run_id: int,
    target_wavelength_nm: float,
    final_run_id: int | None = None,
    **kwargs,
) -> dict:
    """Convenience helper to enqueue a TDDFT MLNEB run via Celery submissions queue."""
    payload = {
        "source_run_id": source_run_id,
        "target_wavelength_nm": target_wavelength_nm,
        **kwargs,
    }
    if final_run_id is not None:
        payload["final_run_id"] = final_run_id
    return enqueue_analysis_request("tddft_mlneb", payload)


def enqueue_dft_optimisation(
    source_run_id: int,
    image_index: int | None = None,
    **kwargs,
) -> dict:
    """Convenience helper to enqueue a DFT geometry optimisation via Celery submissions queue."""
    payload = {"source_run_id": source_run_id, **kwargs}
    if image_index is not None:
        payload["image_index"] = image_index
    return enqueue_analysis_request("dft_optimisation", payload)


def _submit_dft_optimisation(
    source_run_id: int,
    image_index: int | None = None,
    structure_xyz: str = "",
    run_name: str = "",
    hpc_name: str = "FALCON",
    qos: str = "standard",
    reservation: str = "",
    partition: str = "standard",
    walltime: str = "24:00:00",
    nodes: int = 1,
    tasks_per_node: int = 128,
    fmax: float = 0.01,
    spin_multiplicity: int = 1,
    system_charge: int = 0,
    active_learning_optimizer: str = "bfgs",
) -> dict:
    import uuid
    from django.conf import settings
    from workflows.models import WorkflowDefinition, WorkflowRun
    from .geometry import atoms_to_extxyz
    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    if not structure_xyz:
        if image_index is not None:
            atoms = _get_run_image_atoms(source, image_index=image_index)
            structure_xyz = atoms_to_extxyz(atoms)
        else:
            structure_xyz = _final_structure_xyz(source)

    if not run_name:
        tag = f"img{image_index:02d}" if image_index is not None else "opt"
        run_name = f"{source.run_name[:70]}_{tag}_DFT_opt_{uuid.uuid4().hex[:8]}"

    workflow = (
        _workflow_by_code_path(
            WorkflowDefinition,
            "cats/working_directory/tddft_active_learning_workflow.py",
            required=False,
        )
        or source.workflow
    )
    hpc_system = HPCSystem.objects.get(name=hpc_name.upper(), is_active=True)
    is_falcon = hpc_system.name.upper() == "FALCON"
    submission = dict((source.metadata or {}).get("submission") or {})

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=hpc_system,
        settings_profile=source.settings_profile,
        run_name=run_name,
        structure_xyz=structure_xyz,
        remote_runs_root=(
            str(settings.REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT)
            if is_falcon else str(submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT)
        ),
        scratch_base=(
            str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE)
            if is_falcon else str(submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_SCRATCH_BASE)
        ),
        account=(
            str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT)
            if is_falcon else str(submission.get("account") or settings.REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT or "SLURM_ACCOUNT")
        ),
        partition="htc_genoa" if is_falcon else partition,
        qos="" if is_falcon else qos,
        reservation="" if is_falcon else reservation,
        walltime=walltime,
        nodes=nodes,
        tasks_per_node=48 if is_falcon else tasks_per_node,
        cpus_per_task=1,
        module_loads=list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES) if is_falcon else ["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path=(settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH if is_falcon else str(submission.get("venv_activate_path") or settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH)),
        pythonpath_entries=list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH) if is_falcon else list(settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH),
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        single_constraints_config={},
        fmax=fmax,
        active_learning_optimizer=active_learning_optimizer,
        slurm_preset="falcon_htc" if is_falcon else "custom",
        submitted_by="codex-dft-opt",
        structure_metadata={
            "purpose": "dft_optimisation",
            "source_run_id": source.pk,
            "image_index": image_index,
            "hpc_name": hpc_system.name,
        },
    )
    return {
        "status": "submitted" if result.scheduler_job_id else "queued_locally",
        "source_run_id": source.pk,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
        "run_name": run_name,
    }


def _submit_dft_mlneb(
    source_run_id: int,
    retry_job_id: int | None = None,
    final_run_id: int | None = None,
    source_image_index: int | None = None,
    final_image_index: int | None = None,
    optimize_initial_image: bool = False,
    optimize_final_image: bool = False,
    band_artifact_id: int | None = None,
    hpc_name: str = "FALCON",
    qos: str = "taskfarm",
    reservation: str = "",
    partition: str = "standard",
    walltime: str = "24:00:00",
    nodes: int = 1,
    tasks_per_node: int = 128,
    fmax: float = 0.05,
    atom_shift_per_image_A: float = 0.25,
    min_images: int = 13,
    max_images: int = 13,
    spring_constant: float = 0.1,
    max_steps: int = 300,
    catlearn_steps: int = 200,
    reaction_name: str = "Cu20-CO+O -> Cu20-CO2 (new minimum)",
    taskfarm_initial_band: bool = False,
    taskfarm_images: int = 13,
    catlearn_dtrust: float = 0.25,
) -> dict:
    """Submit a ground-state DFT MLNEB run using Active Learning NEB workflow."""
    if retry_job_id is not None:
        from .models import JobRecord
        from .views import _job_resubmit_failed_sync_response

        job = JobRecord.objects.select_related("workflow_run").get(pk=retry_job_id)
        if job.workflow_run_id != source_run_id:
            raise ValueError("retry_job_id does not belong to source_run_id")
        request = _task_request(f"/jobs/{retry_job_id}/resubmit-failed/", requested_by="codex")
        response = _job_resubmit_failed_sync_response(
            request, retry_job_id, wait_for_queue_capacity=False, raise_on_failure=True
        )
        return {"ok": response.status_code < 400, "job_id": retry_job_id, "messages": _message_payload(request)}

    from datetime import datetime
    import hashlib
    import io
    from pathlib import Path
    import numpy as np
    from django.conf import settings
    from ase import Atoms
    from workflows.models import WorkflowDefinition, WorkflowRun
    from results.models import Artifact
    from .geometry import atoms_to_extxyz
    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    workflow = _workflow_by_code_path(WorkflowDefinition, "cats/working_directory/active_learning_neb_workflow.py")
    hpc_system = HPCSystem.objects.get(name=hpc_name.upper(), is_active=True)

    import sys
    repo_root = str(settings.BASE_DIR.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    band_content = ""
    if band_artifact_id is not None:
        artifact = Artifact.objects.get(pk=band_artifact_id, workflow_run=source, kind="trajectory")
        cache_root = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()
        band_file = Path(str((artifact.metadata or {}).get("local_cache_path") or "")).resolve()
        if not band_file.is_relative_to(cache_root) or not band_file.is_file():
            raise ValueError(f"Seed band Artifact {band_artifact_id} has no valid local cache file")
        if band_file.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Seed band artifact exceeds 20 MiB")
        band_content = band_file.read_text(encoding="utf-8").strip()
    final_structure_extxyz = ""
    final_meta = {}
    if band_content:
        from ase.io import read
        band_images = read(io.StringIO(band_content), index=":", format="extxyz")
        if not 3 <= len(band_images) <= 64:
            raise ValueError("Seed band requires 3-64 images")
        if any(image.get_chemical_symbols() != band_images[0].get_chemical_symbols() for image in band_images[1:]):
            raise ValueError("Seed band atom order differs between images")
        initial_atoms = band_images[0]
        final_structure_extxyz = atoms_to_extxyz(band_images[-1])
        final_meta = {
            "source_run_id": final_run_id if final_run_id else source.pk,
            "band_images": len(band_images),
            "constraints_removed": True,
        }
    else:
        if source_image_index is not None:
            initial_atoms = _get_run_image_atoms(source, image_index=source_image_index)
        else:
            initial_prop = source.computed_properties.filter(name="final_structure_snapshot").first()
            if initial_prop is None:
                raise ValueError(f"WorkflowRun {source.pk} has no final_structure_snapshot")
            initial_meta = initial_prop.metadata or {}
            initial_atoms = Atoms(initial_meta["symbols"], positions=initial_meta["positions_A"])
        initial_atoms.set_constraint()
        initial_atoms.calc = None

        if final_run_id is not None:
            final_run = WorkflowRun.objects.get(pk=final_run_id)
            if final_image_index is not None:
                final_atoms = _get_run_image_atoms(final_run, image_index=final_image_index)
            else:
                final_prop = final_run.computed_properties.filter(name="final_structure_snapshot").first()
                if final_prop is None:
                    raise ValueError(f"WorkflowRun {final_run.pk} has no final_structure_snapshot")
                final_snap_meta = final_prop.metadata or {}
                final_atoms = Atoms(final_snap_meta["symbols"], positions=final_snap_meta["positions_A"])
            final_atoms.set_constraint()
            final_atoms.calc = None
            from cats.working_directory.neb_endpoint_alignment import align_and_reorder_endpoint

            aligned_final, alignment = align_and_reorder_endpoint(initial_atoms, final_atoms)
            final_atoms = aligned_final
            final_structure_extxyz = atoms_to_extxyz(aligned_final)
            final_meta = {
                "source_run_id": final_run.pk,
                "alignment": alignment,
                "constraints_removed": True,
            }

    submission = dict((source.metadata or {}).get("submission") or {})
    training_run_ids: list[int] = []
    taskfarm_band_digest = ""
    if taskfarm_initial_band:
        from ase.io import read, write
        from workflows.tasks import enqueue_artifact_ingest

        if not band_content:
            if final_run_id is None:
                raise ValueError("Task-farmed DFT MLNEB requires final_run_id or band_artifact_id")
            try:
                from ase.mep import NEB
            except ImportError:
                from ase.neb import NEB
            band_images = [initial_atoms.copy() for _ in range(max(3, int(taskfarm_images)))]
            band_images[-1] = final_atoms.copy()
            NEB(band_images).interpolate(method="idpp", apply_constraint=True)
            stream = io.StringIO()
            write(stream, band_images, format="extxyz")
            band_content = stream.getvalue()

        taskfarm_band_digest = hashlib.sha256(band_content.encode("utf-8")).hexdigest()[:16]
        band_images = list(read(io.StringIO(band_content), index=":", format="extxyz"))
        children = {}
        for candidate in WorkflowRun.objects.filter(submitted_by="dft-mlneb-taskfarm").order_by("pk"):
            marker = dict((candidate.metadata or {}).get("structure") or {})
            if (
                marker.get("purpose") == "dft_mlneb_training_image"
                and marker.get("band_digest") == taskfarm_band_digest
                and marker.get("source_run_id") == source.pk
                and marker.get("final_run_id") == final_run_id
            ):
                children.setdefault(int(marker["image_index"]), candidate)

        if len(children) < len(band_images):
            submitted = []
            for image_index, image in enumerate(band_images):
                if image_index in children:
                    continue
                child = submit_slurm_workflow_run(
                    workflow=workflow,
                    hpc_system=hpc_system,
                    settings_profile=source.settings_profile,
                    run_name=f"dft_mlneb_seed_{taskfarm_band_digest}_{image_index:03d}",
                    structure_xyz=atoms_to_extxyz(image),
                    remote_runs_root=str(submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT),
                    scratch_base=str(submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_SCRATCH_BASE),
                    account="SLURM_ACCOUNT",
                    partition=partition,
                    qos=qos,
                    reservation=reservation,
                    walltime=walltime,
                    nodes=1,
                    tasks_per_node=tasks_per_node,
                    cpus_per_task=1,
                    module_loads=["epcc-setup-env", "cray-python", "orca/6.0.0"],
                    venv_activate_path=str(submission.get("venv_activate_path") or settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH),
                    pythonpath_entries=list(submission.get("pythonpath_entries") or settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH),
                    spin_multiplicity=submission.get("spin_multiplicity"),
                    system_charge=int(source.system_charge or 0),
                    fmax=fmax,
                    active_learning_optimizer="bfgs",
                    qm_backend="orca",
                    n_iter=1,
                    use_tddft_optimisation=False,
                    run_mode="active_learning_neb",
                    neb_config={"backend": "dft", "evaluation_only": True},
                    slurm_preset="custom",
                    submitted_by="dft-mlneb-taskfarm",
                    structure_metadata={
                        "purpose": "dft_mlneb_training_image",
                        "source_run_id": source.pk,
                        "final_run_id": final_run_id,
                        "image_index": image_index,
                        "image_count": len(band_images),
                        "band_digest": taskfarm_band_digest,
                    },
                )
                submitted.append({
                    "image_index": image_index,
                    "workflow_run_id": child.workflow_run_id,
                    "job_record_id": child.job_record_id,
                    "scheduler_job_id": child.scheduler_job_id,
                })
            return {
                "status": "training_submitted",
                "band_digest": taskfarm_band_digest,
                "image_count": len(band_images),
                "submitted": submitted,
            }

        ordered_children = [children[index] for index in range(len(band_images))]
        incomplete = [run.pk for run in ordered_children if run.status != "completed"]
        if incomplete:
            return {"status": "training_running", "band_digest": taskfarm_band_digest, "workflow_run_ids": incomplete}

        evaluated_images = []
        ingestion_tasks = []
        for expected, run in zip(band_images, ordered_children):
            artifact = run.artifacts.filter(path__iendswith="evaluated.traj").first()
            cache_path = str((artifact.metadata or {}).get("local_cache_path") or "") if artifact else ""
            if artifact is None:
                return {"status": "training_artifacts_pending", "band_digest": taskfarm_band_digest, "workflow_run_id": run.pk}
            if not cache_path or not os.path.isfile(cache_path):
                ingestion_tasks.append(enqueue_artifact_ingest(run.pk, artifact.pk))
                continue
            evaluated = read(cache_path, index=-1)
            if evaluated.get_atomic_numbers().tolist() != expected.get_atomic_numbers().tolist() or not np.allclose(evaluated.positions, expected.positions, atol=1.0e-6):
                raise ValueError(f"Task-farmed image geometry mismatch for WorkflowRun {run.pk}")
            evaluated.get_potential_energy()
            evaluated.get_forces()
            evaluated_images.append(evaluated)
            training_run_ids.append(run.pk)
        if ingestion_tasks:
            return {"status": "training_ingestion_queued", "band_digest": taskfarm_band_digest, "tasks": ingestion_tasks}

        stream = io.StringIO()
        write(stream, evaluated_images, format="extxyz")
        band_content = stream.getvalue()
        existing_mlneb = None
        for candidate in WorkflowRun.objects.filter(submitted_by="dft-mlneb").exclude(status="failed"):
            marker = dict((candidate.metadata or {}).get("structure") or {})
            if marker.get("taskfarm_band_digest") == taskfarm_band_digest:
                existing_mlneb = candidate
                break
        if existing_mlneb is not None:
            return {"status": "mlneb_exists", "band_digest": taskfarm_band_digest, "workflow_run_id": existing_mlneb.pk}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_reaction = reaction_name.replace(" ", "_").replace("->", "to")
    run_name = f"{clean_reaction}_DFT_MLNEB_CAMB3LYP_{timestamp}"

    neb_config = {
        "backend": "dft",
        "final_image_strategy": "explicit_final_image" if final_run_id else "bond_extension_from_initial",
        "optimize_initial_image": bool(optimize_initial_image),
        "optimize_final_image": bool(optimize_final_image),
        "atom_shift_per_image_A": atom_shift_per_image_A,
        "min_images": min_images,
        "max_images": max_images,
        "spring_constant": spring_constant,
        "max_steps": max_steps,
        "climb": True,
        "split_on_minimum": False,
        "catlearn_unc_convergence": 0.02,
        "catlearn_max_unc": 0.3,
        "catlearn_steps": catlearn_steps,
        "catlearn_ml_steps": 1000,
        "catlearn_dtrust": float(catlearn_dtrust),
        "final_structure_path": "input_structure_final.extxyz" if final_run_id else "",
        "final_structure_metadata": final_meta,
    }
    if band_content:
        neb_config["initial_band_extxyz"] = band_content
        neb_config["seed_band_artifact_id"] = band_artifact_id
    if final_structure_extxyz:
        neb_config["final_structure_extxyz"] = final_structure_extxyz

    target_hpc_name = str(hpc_name or "FALCON").strip()
    is_falcon = target_hpc_name.upper() == "FALCON"
    if is_falcon:
        remote_runs_root = str(settings.REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT)
        scratch_base = str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE)
        account = str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT)
        module_loads = list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES)
        venv_activate_path = str(settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH)
        pythonpath_entries = list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH)
        partition = partition if partition and partition != "standard" else "htc_genoa"
        qos = qos if qos and qos not in {"reservation", "taskfarm"} else ""
        reservation = reservation if reservation and "SLURM_ACCOUNT" not in reservation else ""
        tasks_per_node = tasks_per_node if tasks_per_node and tasks_per_node != 128 else 48
        slurm_preset = "falcon_cpu"
    else:
        remote_runs_root = str(submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT).strip()
        scratch_base = str(submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_SCRATCH_BASE).strip()
        account = str(submission.get("account") or settings.REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT).strip()
        module_loads = list(settings.REACTION_HUB_SUBMIT_DEFAULT_MODULES)
        venv_activate_path = str(submission.get("venv_activate_path") or settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH).strip()
        pythonpath_entries = list(submission.get("pythonpath_entries") or settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH)
        slurm_preset = "custom"

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=hpc_system,
        settings_profile=source.settings_profile,
        run_name=run_name,
        structure_xyz=atoms_to_extxyz(initial_atoms),
        remote_runs_root=remote_runs_root,
        scratch_base=scratch_base,
        account=account,
        partition=partition,
        qos=qos,
        reservation=reservation,
        walltime=walltime,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=1,
        module_loads=module_loads,
        venv_activate_path=venv_activate_path,
        pythonpath_entries=pythonpath_entries,
        spin_multiplicity=submission.get("spin_multiplicity"),
        system_charge=int(source.system_charge or 0),
        fmax=fmax,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="active_learning_neb",
        neb_config=neb_config,
        slurm_preset=slurm_preset,
        submitted_by="dft-mlneb",
        structure_metadata={
            "purpose": "dft_mlneb",
            "source_run_id": source.pk,
            "final_run_id": final_run_id,
            "formula": initial_atoms.get_chemical_formula(),
            "reaction": reaction_name,
            "taskfarm_band_digest": taskfarm_band_digest,
            "seed_band_artifact_id": band_artifact_id,
            "training_run_ids": training_run_ids,
        },
    )
    return {
        "status": "submitted" if result.scheduler_job_id else "queued_locally",
        "source_run_id": source.pk,
        "final_run_id": final_run_id,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
    }


def enqueue_dft_mlneb(
    source_run_id: int,
    final_run_id: int | None = None,
    band_artifact_id: int | None = None,
    **kwargs,
) -> dict:
    """Convenience helper to enqueue a DFT MLNEB run via Celery submissions queue."""
    payload = {
        "source_run_id": source_run_id,
        **kwargs,
    }
    if final_run_id is not None:
        payload["final_run_id"] = final_run_id
    if band_artifact_id is not None:
        payload["band_artifact_id"] = band_artifact_id
    return enqueue_analysis_request("dft_mlneb", payload)


def _submit_full_ct_workup_run(source_run_id: int) -> dict:
    import io
    from ase.io import read
    from django.conf import settings
    from workflows.models import WorkflowRun

    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run
    from .views import _format_compact_indices, _infer_ct_fragment_defaults

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    structure_xyz = _final_structure_xyz(source)
    metadata = dict(source.metadata or {})
    custom_ct = dict(metadata.get("charge_transfer_config") or {})
    if custom_ct.get("fragments_0based") and custom_ct.get("fragments_1based"):
        fragments_0based = dict(custom_ct["fragments_0based"])
        fragments_1based = dict(custom_ct["fragments_1based"])
        fragment_labels = dict(custom_ct.get("fragment_labels") or {})
    else:
        inferred = _infer_ct_fragment_defaults(read(io.StringIO(structure_xyz), format="xyz"))
        fragments_0based = dict(inferred.get("fragments_0based") or {})
        if not fragments_0based.get("1") or not fragments_0based.get("2"):
            raise ValueError(f"Could not infer complete CT fragments for WorkflowRun {source_run_id}")
        fragments_1based = {
            key: _format_compact_indices([index + 1 for index in indices])
            for key, indices in fragments_0based.items()
        }
        fragment_labels = dict(inferred.get("fragment_labels") or {})
    submission = dict(metadata.get("submission") or {})
    target_hpc_name = str(metadata.get("target_ct_hpc") or submission.get("hpc_name") or "FALCON").strip()
    is_falcon = target_hpc_name.upper() == "FALCON"
    hpc_system = HPCSystem.objects.get(name="FALCON" if is_falcon else target_hpc_name, is_active=True)
    partition = "htc_genoa" if is_falcon else "standard"
    qos = "" if is_falcon else "standard"
    walltime = "12:00:00" if is_falcon else "24:00:00"
    atom_count = None
    try:
        atom_count = int(structure_xyz.strip().splitlines()[0].strip())
    except Exception:
        pass
    resources = dict(metadata.get("full_ct_resources") or {})
    nodes = max(1, min(4, int(resources.get("nodes") or 1)))
    if is_falcon:
        if atom_count is not None and atom_count <= 30:
            tasks_per_node = 48
            slurm_memory = "360G"
            falcon_tasks = 48
        else:
            tasks_per_node = 96
            slurm_memory = "360G"
            falcon_tasks = 96
        cpu_freq = "Performance"
    else:
        tasks_per_node = max(1, min(128, int(resources.get("tasks_per_node") or 128)))
        slurm_memory = ""
        falcon_tasks = None
        cpu_freq = ""
        reservation = str(submission.get("reservation") or "").strip()
        if reservation:
            qos = str(submission.get("qos") or "reservation").strip()
            partition = str(submission.get("partition") or "standard").strip()
            walltime = str(submission.get("walltime") or walltime).strip()
    cpus_per_task = 1
    module_loads = list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES) if is_falcon else ["epcc-setup-env", "cray-python", "orca/6.0.0"]
    venv_activate_path = settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH if is_falcon else settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH
    pythonpath_entries = list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH) if is_falcon else list(settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH)
    if is_falcon:
        remote_runs_root = str(settings.REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT)
        scratch_base = str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE)
        account = str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT)
    else:
        remote_runs_root = str(
            submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT
        )
        scratch_base = str(
            submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_SCRATCH_BASE
        )
        account = str(
            submission.get("account") or settings.REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT
        )
    slurm_preset = "falcon_htc" if is_falcon else "custom"
    ct_config = {
        "source_run_id": source.pk,
        "source_run_uid": str(source.run_uid),
        "source_run_name": source.run_name,
        "orca_basename": "orca",
        "fragments_0based": fragments_0based,
        "fragments_1based": fragments_1based,
        "fragment_labels": fragment_labels,
    }
    workflow = source.workflow
    if "active_learning_neb" in str(getattr(workflow, "code_path", "")).lower():
        from workflows.models import WorkflowDefinition
        workflow = (
            WorkflowDefinition.objects.filter(code_path__icontains="tddft_active_learning_workflow").first()
            or WorkflowDefinition.objects.filter(name__icontains="TDDFT Active Learning").first()
            or workflow
        )
    source_structure_meta = dict((source.metadata or {}).get("structure") or {})
    run_name = f"{source.run_name[:120]}_full_ct_{source.pk}"
    try:
        from .submission import _default_ssh_client_factory, _q
        expanded_root = str(remote_runs_root).replace("{username}", hpc_system.username).replace("$USER", hpc_system.username)
        with _default_ssh_client_factory(hpc_system)() as ssh:
            expected_dir = f"{expanded_root}/{run_name}"
            ssh.run(f"if [ -d {_q(expected_dir)} ] && [ ! -f {_q(expected_dir)}/orca.out ]; then rm -rf {_q(expected_dir)}; fi")
    except Exception:
        pass
    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=hpc_system,
        settings_profile=source.settings_profile,
        run_name=run_name,
        structure_xyz=structure_xyz,
        remote_runs_root=remote_runs_root,
        scratch_base=scratch_base,
        account=account,
        partition=partition,
        qos=qos,
        reservation=reservation if not is_falcon else "",
        walltime=walltime,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        cpu_freq=cpu_freq,
        slurm_memory=slurm_memory,
        workflow_tasks_per_node=falcon_tasks,
        module_loads=module_loads,
        venv_activate_path=venv_activate_path,
        pythonpath_entries=pythonpath_entries,
        spin_multiplicity=submission.get("spin_multiplicity"),
        system_charge=int(source.system_charge or 0),
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        single_constraints_config={
            "snapshot_single_point": True,
            "final_tddft_spectrum": True,
            "cluster_tddft_difference_spectrum": True,
            "final_tddft_nroots_mode": "scaled_by_atoms",
        },
        charge_transfer_config=ct_config,
        slurm_preset=slurm_preset,
        submitted_by="full-ct-workup",
        structure_metadata={
            "source_run_id": source.pk,
            "coordinates_unchanged": True,
            "purpose": "full_charge_transfer_workup",
            **{
                k: v
                for k, v in source_structure_meta.items()
                if k in {"source_mlneb_workflow_run_id", "source_mep_image_index", "source_mep_image_count"}
            },
        },
    )
    return {
        "source_run_id": source.pk,
        "status": "submitted" if result.scheduler_job_id else "queued_locally",
        "ct_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
    }


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, max_retries=None)
def submit_full_ct_workup_task(self, source_run_id: int) -> dict:
    import importlib
    import sys

    for mod in ["jobs.submission", "reaction_hub.jobs.submission", "jobs.tasks", "reaction_hub.jobs.tasks"]:
        if mod in sys.modules:
            importlib.reload(sys.modules[mod])
    from celery.exceptions import Retry
    from workflows.models import WorkflowRun

    def mark(**values):
        with transaction.atomic():
            source = WorkflowRun.objects.select_for_update().get(pk=source_run_id)
            metadata = dict(source.metadata or {})
            metadata["full_ct_workup_task"] = {
                **dict(metadata.get("full_ct_workup_task") or {}),
                **values,
            }
            source.metadata = metadata
            source.save(update_fields=["metadata", "updated_at"])

    source = WorkflowRun.objects.get(pk=source_run_id)
    candidates = (
        WorkflowRun.objects.filter(metadata__source_run_id=source.pk)
        .exclude(status__in={"failed", "cancelled", "deleted"})
        .order_by("-pk")
    )
    target_hpc = str((source.metadata or {}).get("target_ct_hpc") or "").strip().upper()
    existing = None
    for run in candidates:
        ct_cfg = (run.metadata or {}).get("submission", {}).get("charge_transfer_config")
        if not ct_cfg:
            continue
        if ct_cfg.get("ground_state_mulliken_only") or ct_cfg.get("density_analysis_only"):
            continue
        if target_hpc:
            sub = (run.metadata or {}).get("submission", {})
            run_hpc = str(
                sub.get("hpc_name")
                or (run.metadata or {}).get("target_ct_hpc")
                or run.jobs.filter(hpc_system__isnull=False).values_list("hpc_system__name", flat=True).first()
                or "FALCON"
            ).strip().upper()
            if run_hpc != target_hpc:
                continue
        existing = run
        break
    if existing is not None:
        mark(status="existing_ct_requires_no_duplicate", ct_run_id=existing.pk)
        return {
            "source_run_id": source.pk,
            "status": "existing_ct_requires_no_duplicate",
            "ct_run_id": existing.pk,
            "skipped": True,
        }
    try:
        tasks_mod = sys.modules.get("jobs.tasks") or sys.modules.get("reaction_hub.jobs.tasks")
        submit_fn = getattr(tasks_mod, "_submit_full_ct_workup_run", _submit_full_ct_workup_run)
        result = submit_fn(source.pk)
    except Retry:
        raise
    except Exception as exc:
        mark(status="retrying", reason=str(exc))
        raise self.retry(exc=exc, countdown=300 + source.pk % 300)
    mark(status=result["status"], result=result)
    return result


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, max_retries=None)
def submit_manuscript_density_state_job_task(
    self,
    source_run_id: int,
    ct_run_id: int,
    excitation_numbers,
    nto_spin="default",
    retain_cubes=False,
    nto_population_only=False,
    ground_state_orbitals_only=False,
    density_workers=1,
    nto_max_pairs=1,
    nto_target_coverage=0.8,
) -> dict:
    from workflows.models import WorkflowRun

    requested_states = sorted({int(value) for value in excitation_numbers})
    if not requested_states and not ground_state_orbitals_only:
        raise ValueError("excitation_numbers must contain at least one state")
    existing = _existing_density_run(source_run_id, requested_states, nto_spin, retain_cubes, nto_population_only, ground_state_orbitals_only, density_workers, nto_max_pairs, nto_target_coverage)
    if existing is not None:
        return {
            "source_run_id": source_run_id,
            "workflow_run_id": existing.pk,
            "excitation_numbers": requested_states,
            "skipped": True,
        }
    retry_countdown = 300 + int(source_run_id) % 300
    if _is_falcon_ct_run(ct_run_id):
        qos = ""
    else:
        try:
            if nto_population_only or ground_state_orbitals_only:
                qos = "short" if _queue_has_capacity("short") else None
            else:
                qos = _next_density_qos()
        except Exception as exc:
            raise self.retry(exc=exc, countdown=retry_countdown, max_retries=None)
        if qos is None:
            raise self.retry(countdown=retry_countdown, max_retries=None)
    try:
        import importlib
        import sys
        if "jobs.tasks" in sys.modules:
            importlib.reload(sys.modules["jobs.tasks"])
        submit_func = getattr(sys.modules.get("jobs.tasks"), "_submit_manuscript_density_run", _submit_manuscript_density_run)
        result = submit_func(
            source_run_id,
            ct_run_id,
            requested_states,
            qos=qos,
            nto_spin=nto_spin,
            retain_cubes=retain_cubes,
            nto_population_only=nto_population_only,
            ground_state_orbitals_only=ground_state_orbitals_only,
            density_workers=density_workers,
            nto_max_pairs=nto_max_pairs,
            nto_target_coverage=nto_target_coverage,
        )
    except Exception as exc:
        raise self.retry(exc=exc, countdown=retry_countdown, max_retries=None)

    source = WorkflowRun.objects.get(pk=source_run_id)
    metadata = dict(source.metadata or {})
    markers = dict(metadata.get("manuscript_density_state_tasks") or {})
    for state in requested_states:
        key = f"{state}:{nto_spin}:w{density_workers}:nto{nto_max_pairs}"
        if nto_population_only:
            key += ":mulliken"
        markers[key] = {
            "status": "submitted",
            "task_id": str(getattr(self.request, "id", "") or ""),
            **result,
        }
    metadata["manuscript_density_state_tasks"] = markers
    source.metadata = metadata
    source.save(update_fields=["metadata", "updated_at"])
    return {**result, "excitation_numbers": requested_states}


@shared_task(acks_late=True, reject_on_worker_lost=True)
def restart_cancelled_job_task(job_id: int, *, requested_by: str = "", next_url: str = "") -> dict:
    from .views import _job_restart_cancelled_sync_response

    request = _task_request(
        f"/jobs/{int(job_id)}/restart-cancelled/",
        requested_by=requested_by,
        next_url=next_url,
    )
    response = _job_restart_cancelled_sync_response(request, int(job_id))
    return {
        "ok": int(response.status_code) < 400,
        "status_code": int(response.status_code),
        "messages": _message_payload(request),
    }


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, max_retries=None)
def restore_ct_serial_job_task(self, job_id: int) -> dict:
    from .models import JobRecord, JobStatus

    marker = "ct_wrapper_repaired_20260821"
    owner_marker = "ct_serial_restore_task_id"
    task_id = str(getattr(self.request, "id", "") or "")
    with transaction.atomic():
        job = JobRecord.objects.select_for_update().get(pk=job_id)
        payload = dict(job.scheduler_payload or {})
        if payload.get(marker):
            return {"job_id": job_id, "status": job.status, "skipped": True}
        owner_task_id = str(payload.get(owner_marker) or "")
        if task_id and owner_task_id and owner_task_id != task_id:
            return {
                "job_id": job_id,
                "status": job.status,
                "skipped": "duplicate_restore_task",
                "owner_task_id": owner_task_id,
            }
        if task_id and not owner_task_id:
            payload[owner_marker] = task_id
            job.scheduler_payload = payload
            job.save(update_fields=["scheduler_payload", "updated_at"])
    if job.status == JobStatus.CANCELLED:
        retry_countdown = _queue_retry_countdown(job_id)
        try:
            has_capacity = _queue_has_capacity("serial")
        except Exception as exc:
            raise self.retry(exc=exc, countdown=retry_countdown, max_retries=None)
        if not has_capacity:
            raise self.retry(countdown=retry_countdown, max_retries=None)
        restart_cancelled_job_task.run(
            job_id,
            requested_by="codex-ct-serial-migration",
        )
        job.refresh_from_db()
    if job.status not in {JobStatus.INITIATED, JobStatus.PENDING, JobStatus.RUNNING}:
        raise self.retry(
            exc=RuntimeError(f"Serial restoration did not submit job {job_id}: {job.status}"),
            countdown=_queue_retry_countdown(job_id),
            max_retries=None,
        )
    payload = dict(job.scheduler_payload or {})
    payload[marker] = True
    job.scheduler_payload = payload
    job.save(update_fields=["scheduler_payload", "updated_at"])
    return {"job_id": job_id, "scheduler_job_id": job.scheduler_job_id, "status": job.status}


def _dispatch_next_queued_gpu_workflow(hpc_name: str) -> dict:
    from django.conf import settings
    from workflows.models import CampaignStatus, WorkflowRun

    from .models import HPCSystem
    from .submission import (
        _active_qos_job_count,
        _active_user_job_count,
        _default_ssh_client_factory,
        _submission_queue_limit,
    )

    if str(hpc_name or "").strip().casefold() == "trh408":
        if getattr(settings, "REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS", True):
            return {"submitted": 0, "reason": "TRH408 GPU submissions paused"}

    hpc = HPCSystem.objects.get(name=hpc_name, is_active=True)

    client = _default_ssh_client_factory(hpc)()
    try:
        occupied = (
            _active_user_job_count(client, username=hpc.username) >= 1
            if hpc_name == "TRH408"
            else _active_qos_job_count(client, username=hpc.username, qos="gpu-exc")
            >= _submission_queue_limit("gpu-exc")
        )
        if occupied:
            return {"submitted": 0, "reason": f"{hpc_name} GPU slot occupied"}
    finally:
        client.close()

    with transaction.atomic():
        scheduled_run = (
            WorkflowRun.objects.select_for_update()
            .filter(metadata__queue_status="enqueued_celery_gpu")
            .exclude(campaign_memberships__campaign__status=CampaignStatus.PAUSED)
            .order_by("created_at")
            .first()
        )
        if scheduled_run is None:
            return {"submitted": 0, "reason": "No queued GPU workflows"}
        metadata = dict(scheduled_run.metadata or {})
        submission = dict(metadata.get("submission") or {})
        submission["hpc_name"] = hpc_name
        metadata["submission"] = submission
        metadata["queue_status"] = "submitting_celery_gpu"
        scheduled_run.metadata = metadata
        scheduled_run.save(update_fields=["metadata", "updated_at"])

    try:
        result = submit_queued_gpu_workflow_task.delay(scheduled_run.pk)
    except Exception:
        metadata["queue_status"] = "enqueued_celery_gpu"
        scheduled_run.metadata = metadata
        scheduled_run.save(update_fields=["metadata", "updated_at"])
        raise
    return {"submitted": 1, "workflow_run_id": scheduled_run.pk, "task_id": result.id}


@shared_task(base=SingleFlightTask)
def dispatch_next_queued_gpu_workflow_task() -> dict:
    return _dispatch_next_queued_gpu_workflow("TRH408")


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, max_retries=None)
def submit_queued_gpu_workflow_task(self, workflow_run_id: int) -> dict:
    from django.conf import settings

    from workflows.models import CampaignRun, WorkflowRun

    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    task_id = str(getattr(self.request, "id", "") or "")
    with transaction.atomic():
        scheduled_run = WorkflowRun.objects.select_for_update().select_related(
            "workflow", "settings_profile"
        ).get(pk=workflow_run_id)
        metadata = dict(scheduled_run.metadata or {})
        origin_id = int((metadata.get("structure") or {}).get("scheduled_workflow_run_id") or 0)
        origin = WorkflowRun.objects.filter(pk=origin_id).first() if origin_id else None
        successor_id = int(((origin.metadata or {}).get("submitted_workflow_run_id") or 0)) if origin else 0
        successor = WorkflowRun.objects.filter(pk=successor_id).first() if successor_id else None
        if successor is not None and successor.pk != scheduled_run.pk and successor.jobs.filter(status="completed").exists():
            metadata["queue_status"] = "superseded_completed"
            metadata["superseded_by_workflow_run_id"] = successor.pk
            scheduled_run.metadata = metadata
            scheduled_run.save(update_fields=["metadata", "updated_at"])
            return {
                "scheduled_workflow_run_id": workflow_run_id,
                "submitted_workflow_run_id": successor.pk,
                "skipped": "completed_successor",
            }
        owner = str(metadata.get("submission_task_id") or "")
        if metadata.get("queue_status") == "submitted_slurm":
            return {
                "scheduled_workflow_run_id": workflow_run_id,
                "submitted_workflow_run_id": metadata.get("submitted_workflow_run_id"),
                "scheduler_job_id": metadata.get("scheduler_job_id"),
                "skipped": True,
            }
        if owner and task_id and owner != task_id:
            return {
                "scheduled_workflow_run_id": workflow_run_id,
                "owner_task_id": owner,
                "skipped": "duplicate_submission_task",
            }
        if metadata.get("queue_status") not in {"enqueued_celery_gpu", "submitting_celery_gpu"}:
            raise ValueError(f"Workflow run {workflow_run_id} is not queued for GPU submission.")
        metadata["queue_status"] = "submitting_celery_gpu"
        metadata["submission_task_id"] = task_id
        metadata.pop("submission_error", None)
        scheduled_run.metadata = metadata
        scheduled_run.save(update_fields=["metadata", "updated_at"])

    submission = dict(metadata.get("submission") or {})
    coadsorbate_config = dict(metadata.get("coadsorbate_config") or {})
    alloy_config = dict(metadata.get("alloy_config") or {})
    run_mode = str(metadata.get("run_mode") or "coadsorbate_placement").strip()
    target_hpc_name = str(submission.get("hpc_name") or "FALCON").strip()
    if getattr(settings, "REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS", True) and target_hpc_name.casefold() == "trh408":
        target_hpc_name = "FALCON"
    try:
        hpc_system = HPCSystem.objects.get(name=target_hpc_name, is_active=True)
    except HPCSystem.DoesNotExist:
        hpc_system = HPCSystem.objects.get(name="FALCON", is_active=True)


    is_trh = hpc_system.name.strip().casefold() == "trh408"
    nodes = 1 if is_trh else int(submission.get("nodes") or 2)
    tasks_per_node = 1 if is_trh else int(submission.get("tasks_per_node") or 4)
    cpus_per_task = 8 if is_trh else int(submission.get("cpus_per_task") or 8)
    partition = "main" if is_trh else str(submission.get("partition") or "gpu")
    remote_runs_root = str(
        submission.get("remote_runs_root")
        or ("/home/{username}/reaction_hub_runs" if is_trh else settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT)
    )
    scratch_base = "" if is_trh else str(
        submission.get("scratch_base")
        or settings.REACTION_HUB_SUBMIT_SCRATCH_BASE
    )
    module_loads = (
        list(settings.REACTION_HUB_SUBMIT_TRH408_MODULES)
        if is_trh
        else list(submission.get("module_loads") or [])
    )
    venv_activate_path = (
        settings.REACTION_HUB_SUBMIT_TRH408_VENV_ACTIVATE_PATH
        if is_trh
        else str(submission.get("venv_activate_path") or "")
    )
    pythonpath_entries = (
        list(settings.REACTION_HUB_SUBMIT_TRH408_PYTHONPATH)
        if is_trh
        else list(submission.get("pythonpath_entries") or [])
    )

    structure_xyz = (
        str(metadata.get("structure_xyz") or "").strip()
        or str((metadata.get("structure") or {}).get("xyz_text") or "").strip()
        or str((metadata.get("submission") or {}).get("structure_xyz") or "").strip()
    )
    if not structure_xyz:
        origin_id = int((metadata.get("structure") or {}).get("scheduled_workflow_run_id") or 0)
        origin = WorkflowRun.objects.filter(pk=origin_id).first() if origin_id else None
        origin_metadata = dict((origin.metadata if origin is not None else {}) or {})
        structure_xyz = str(origin_metadata.get("structure_xyz") or "").strip()

    try:
        result = submit_slurm_workflow_run(
            workflow=scheduled_run.workflow,
            hpc_system=hpc_system,
            settings_profile=scheduled_run.settings_profile,
            run_name=scheduled_run.run_name,
            structure_xyz=structure_xyz,
            remote_runs_root=remote_runs_root,
            scratch_base=scratch_base,
            account="" if is_trh else str(submission.get("account") or "SLURM_ACCOUNT"),
            partition=partition,
            qos="" if is_trh else str(submission.get("qos") or "gpu-exc"),
            reservation=str(submission.get("reservation") or ""),
            walltime=str(submission.get("walltime") or "12:00:00"),
            nodes=nodes,
            tasks_per_node=tasks_per_node,
            cpus_per_task=cpus_per_task,
            module_loads=module_loads,
            venv_activate_path=venv_activate_path,
            pythonpath_entries=pythonpath_entries,
            fmax=float((alloy_config if run_mode == "alloy_growth" else coadsorbate_config).get("mace_fmax") or 0.05),
            active_learning_optimizer="bfgs",
            qm_backend="orca",
            n_iter=1,
            use_tddft_optimisation=False,
            run_mode=run_mode,
            coadsorbate_config=coadsorbate_config if run_mode == "coadsorbate_placement" else None,
            alloy_config=alloy_config if run_mode == "alloy_growth" else None,
            slurm_preset="custom",
            submitted_by=str(submission.get("submitted_by") or "campaign-gpu-schedule"),
            structure_metadata={
                "source": "campaign_gpu_schedule",
                "scheduled_workflow_run_id": scheduled_run.pk,
            },
        )
    except Exception as exc:
        with transaction.atomic():
            scheduled_run = WorkflowRun.objects.select_for_update().get(pk=workflow_run_id)
            retry_metadata = dict(scheduled_run.metadata or {})
            retry_metadata["queue_status"] = "enqueued_celery_gpu"
            retry_metadata["submission_error"] = str(exc)
            scheduled_run.metadata = retry_metadata
            scheduled_run.save(update_fields=["metadata", "updated_at"])
        raise self.retry(exc=exc, countdown=60, max_retries=None)

    with transaction.atomic():
        scheduled_run = WorkflowRun.objects.select_for_update().get(pk=workflow_run_id)
        submitted_run = WorkflowRun.objects.get(pk=result.workflow_run_id)
        for membership in CampaignRun.objects.filter(workflow_run=scheduled_run):
            CampaignRun.objects.update_or_create(
                campaign=membership.campaign,
                workflow_run=submitted_run,
                defaults={
                    "checkpoint_stage": membership.checkpoint_stage,
                    "reaction_step_key": membership.reaction_step_key,
                    "notes": membership.notes,
                },
            )
            membership.delete()
        final_metadata = dict(scheduled_run.metadata or {})
        final_metadata.update(
            {
                "queue_status": "submitted_slurm",
                "submitted_workflow_run_id": result.workflow_run_id,
                "scheduler_job_id": result.scheduler_job_id,
            }
        )
        scheduled_run.metadata = final_metadata
        scheduled_run.save(update_fields=["metadata", "updated_at"])

    return {
        "scheduled_workflow_run_id": workflow_run_id,
        "submitted_workflow_run_id": result.workflow_run_id,
        "scheduler_job_id": result.scheduler_job_id,
    }


@shared_task(bind=True, acks_late=True, reject_on_worker_lost=True, max_retries=None)
def resubmit_failed_job_task(
    self,
    job_id: int,
    *,
    requested_by: str = "",
    next_url: str = "",
) -> dict:
    import sys, importlib
    for mod_name in ["core.openssh", "jobs.submission", "reaction_hub.jobs.submission", "jobs.views", "reaction_hub.jobs.views"]:
        if mod_name in sys.modules:
            importlib.reload(sys.modules[mod_name])
    from .models import JobRecord
    from .submission import _submission_queue_limit
    from .views import _job_resubmit_failed_sync_response, _QueueCapacityUnavailable

    job = JobRecord.objects.select_related("hpc_system").get(pk=job_id)
    payload = dict(job.scheduler_payload or {})
    qos = str(
        (payload.get("expired_reservation_migration") or {}).get("target_qos")
        or payload.get("qos")
        or ""
    ).strip()
    retry_countdown = 60 + int(job_id) % 60
    request = _task_request(
        f"/jobs/{int(job_id)}/resubmit-failed/",
        requested_by=requested_by,
        next_url=next_url,
    )
    try:
        response = _job_resubmit_failed_sync_response(
            request,
            int(job_id),
            wait_for_queue_capacity=False,
            raise_on_failure=True,
        )
    except _QueueCapacityUnavailable:
        raise self.retry(countdown=retry_countdown, max_retries=None)
    except Exception as exc:
        raise self.retry(exc=exc, countdown=retry_countdown, max_retries=None)
    return {
        "ok": int(response.status_code) < 400,
        "status_code": int(response.status_code),
        "messages": _message_payload(request),
    }


@shared_task
def resubmit_smooth_job_task(job_id: int, *, requested_by: str = "", next_url: str = "") -> dict:
    from .views import _job_resubmit_smooth_sync_response

    request = _task_request(
        f"/jobs/{int(job_id)}/resubmit-smooth/",
        requested_by=requested_by,
        next_url=next_url,
    )
    response = _job_resubmit_smooth_sync_response(request, int(job_id))
    return {
        "ok": int(response.status_code) < 400,
        "status_code": int(response.status_code),
        "messages": _message_payload(request),
    }


@shared_task
def resubmit_replot_job_task(job_id: int, *, requested_by: str = "", next_url: str = "") -> dict:
    from .views import _job_resubmit_replot_sync_response

    request = _task_request(
        f"/jobs/{int(job_id)}/resubmit-replot/",
        requested_by=requested_by,
        next_url=next_url,
    )
    response = _job_resubmit_replot_sync_response(request, int(job_id))
    return {
        "ok": int(response.status_code) < 400,
        "status_code": int(response.status_code),
        "messages": _message_payload(request),
    }


def _submit_mep_state_tracking_run(
    source_run_id: int,
    image_run_ids: list[int] | None = None,
    image_directories: list[str] | None = None,
    root_state: int = 25,
    energy_window_ev: float = 0.35,
    density_workers: int = 4,
    nto_max_pairs: int = 3,
    nto_target_coverage: float = 0.8,
    grid_quality: int = 2,
    walltime: str = "04:00:00",
    hpc_name: str = "FALCON",
    reservation: str = "",
    qos: str = "",
    partition: str = "",
) -> dict:
    from django.conf import settings
    from django.utils import timezone
    from workflows.models import WorkflowRun
    import sys, importlib
    for m in ["reaction_hub.settings", "jobs.submission", "reaction_hub.jobs.submission"]:
        if m in sys.modules:
            importlib.reload(sys.modules[m])
    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    resolved_dirs = []
    if image_directories:
        resolved_dirs = [str(d).strip() for d in image_directories if str(d).strip()]
    elif image_run_ids:
        for r_id in image_run_ids:
            ct_run = WorkflowRun.objects.get(pk=int(r_id))
            job = ct_run.jobs.filter(status="completed").order_by("-completed_at", "-pk").first()
            d = _resolve_density_source_directory(ct_run, ct_job=job, source=source)
            if not d:
                d = str(ct_run.working_directory or "")
            if d:
                resolved_dirs.append(d)
    else:
        ct_pks = [8968, 8974, 8969, 8972, 8959, 8970, 8975, 8971, 8973]
        for r_id in ct_pks:
            ct_run = WorkflowRun.objects.filter(pk=r_id).first()
            if ct_run:
                d = str(ct_run.working_directory or "")
                if d:
                    resolved_dirs.append(d)

    if not resolved_dirs:
        raise ValueError(f"Could not resolve image directories for continuous MEP tracking from source {source_run_id}")

    target_hpc_name = str(hpc_name or "FALCON").strip()
    is_falcon = target_hpc_name.upper() == "FALCON"
    hpc_system = HPCSystem.objects.get(name="FALCON" if is_falcon else target_hpc_name, is_active=True)

    existing_ct = dict((source.metadata or {}).get("submission", {}).get("charge_transfer_config") or {})
    search_pks = list(image_run_ids) if image_run_ids else [8968, 8974, 8969, 8972, 8959, 8970, 8975, 8971, 8973]
    if not existing_ct.get("fragments_0based"):
        for r_id in search_pks:
            ct_run = WorkflowRun.objects.filter(pk=r_id).first()
            if ct_run:
                cfg = dict((ct_run.metadata or {}).get("submission", {}).get("charge_transfer_config") or {})
                if cfg.get("fragments_0based"):
                    existing_ct = cfg
                    break

    if not existing_ct.get("fragments_0based"):
        from .views import _format_compact_indices, _infer_ct_fragment_defaults
        from ase.io import read
        import io
        try:
            inferred = _infer_ct_fragment_defaults(read(io.StringIO(_final_structure_xyz(source)), format="xyz"))
            f0 = dict(inferred.get("fragments_0based") or {})
            if f0.get("1") and f0.get("2"):
                existing_ct["fragments_0based"] = f0
                existing_ct["fragments_1based"] = {
                    k: _format_compact_indices([i + 1 for i in v])
                    for k, v in f0.items()
                }
                existing_ct["fragment_labels"] = dict(inferred.get("fragment_labels") or {})
        except Exception:
            pass

    ct_config = {
        "source_run_id": source.pk,
        "source_run_uid": str(source.run_uid),
        "source_run_name": source.run_name,
        "source_working_directory": resolved_dirs[0],
        "fragments_0based": existing_ct.get("fragments_0based", {}),
        "fragments_1based": existing_ct.get("fragments_1based", {}),
        "fragment_labels": existing_ct.get("fragment_labels", {}),
        "density_analysis_only": True,
        "density_analysis": {
            "mep_state_tracking": True,
            "image_directories": resolved_dirs,
            "root_state": int(root_state),
            "energy_window_ev": float(energy_window_ev),
            "density_workers": int(density_workers),
            "nto_max_pairs": int(nto_max_pairs),
            "nto_target_coverage": float(nto_target_coverage),
            "grid_quality": int(grid_quality),
        },
    }

    result = submit_slurm_workflow_run(
        workflow=source.workflow,
        hpc_system=hpc_system,
        settings_profile=source.settings_profile,
        run_name=f"{source.run_name[:50]}_mep_tracking_s{root_state}_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
        structure_xyz=_final_structure_xyz(source),
        remote_runs_root=f"{str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE).rstrip('/')}/reaction_hub_runs" if is_falcon else str(settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT),
        scratch_base=str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE) if is_falcon else str(settings.REACTION_HUB_SUBMIT_SCRATCH_BASE),
        account=str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT) if is_falcon else "SLURM_ACCOUNT",
        partition=str(partition or ("htc_genoa" if is_falcon else "standard")).strip(),
        qos=str(qos or ("" if is_falcon else ("reservation" if (reservation and str(reservation).strip().lower() not in ("", "none", "off", "standard")) else "standard"))).strip(),
        reservation=str("" if (not reservation or str(reservation).strip().lower() in ("", "none", "off", "standard")) else reservation).strip(),
        walltime=str(walltime or "04:00:00"),
        nodes=1,
        tasks_per_node=48 if is_falcon else 128,
        cpus_per_task=1,
        module_loads=list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES) if is_falcon else ["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path=settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH if is_falcon else settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH,
        pythonpath_entries=list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH) if is_falcon else list(settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH),
        spin_multiplicity=(source.metadata or {}).get("submission", {}).get("spin_multiplicity"),
        system_charge=int(source.system_charge or 0),
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="charge_transfer",
        charge_transfer_config=ct_config,
        slurm_preset="falcon_htc" if is_falcon else "custom",
        submitted_by="codex-mep-tracking",
        structure_metadata={
            "source_run_id": source.pk,
            "purpose": "continuous_mep_excited_state_tracking",
            "root_state": int(root_state),
            "image_count": len(resolved_dirs),
        },
    )
    return {
        "operation": "mep_state_tracking",
        "source_run_id": source.pk,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
        "root_state": int(root_state),
        "image_count": len(resolved_dirs),
    }


def _pad_mep_images(images, padding_per_segment: int):
    padded = []
    for segment, (left, right) in enumerate(zip(images, images[1:])):
        padded.append((left.copy(), segment, 0.0, segment))
        for step in range(1, int(padding_per_segment) + 1):
            fraction = step / (int(padding_per_segment) + 1)
            image = left.copy()
            image.positions[:] = left.positions + fraction * (right.positions - left.positions)
            image.calc = None
            padded.append((image, segment, fraction, None))
    padded.append((images[-1].copy(), len(images) - 2, 1.0, len(images) - 1))
    return padded


def _first_three_fragment_config(configs):
    return next((dict(config) for config in configs if len(dict(config.get("fragments_0based") or {})) >= 3), {})


def _submit_mep_uvvis_snapshots(
    source_run_id: int,
    band_artifact_id: int | None = None,
    padding_per_segment: int = 2,
    image_indices: list[int] | None = None,
    nroots: int = 50,
    walltime: str = "24:00:00",
    reservation: str = "",
    qos: str = "reservation",
    partition: str = "standard",
    trajectory_path: str = "",
    fragments_0based: dict | None = None,
    fragment_labels: dict | None = None,
    gradient_root: int | None = None,
    initial_wavefunction_artifact_id: int | None = None,
    settings_profile_name: str = "",
) -> dict:
    import hashlib
    from datetime import datetime
    from ase.io import read
    from django.conf import settings
    from workflows.models import WorkflowRun
    from .geometry import atoms_to_extxyz
    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run
    from .views import _format_compact_indices, _infer_ct_fragment_defaults

    source = WorkflowRun.objects.select_related("settings_profile").get(pk=source_run_id)
    if band_artifact_id is not None:
        artifact = source.artifacts.get(pk=band_artifact_id, kind="trajectory")
        cache = Path(str((artifact.metadata or {}).get("local_cache_path") or ""))
        if not cache.is_file():
            raise ValueError(f"Artifact {band_artifact_id} has no locally ingested trajectory")
        artifact_id = artifact.pk
    elif trajectory_path and Path(trajectory_path).is_file():
        cache = Path(trajectory_path)
        artifact_id = None
    else:
        artifact = (
            source.artifacts.filter(label="trajectory:mep_primary_with_results.extxyz").first()
            or source.artifacts.filter(kind="trajectory", path__icontains="mep_primary").first()
            or source.artifacts.filter(kind="trajectory", path__icontains="final_band").first()
        )
        cache = Path(str((artifact.metadata or {}).get("local_cache_path") or "")) if artifact else Path()
        if source.status != "completed" or artifact is None or not cache.is_file():
            raise ValueError(f"WorkflowRun {source.pk} has no completed, locally ingested MEP band")
        artifact_id = artifact.pk

    images = list(read(cache, ":"))
    padded = _pad_mep_images(images, padding_per_segment)
    selected = list(range(len(padded))) if image_indices is None else image_indices
    if any(index >= len(padded) for index in selected):
        raise ValueError(f"image_indices must be below {len(padded)}")
    inferred = _infer_ct_fragment_defaults(images[0])
    if not fragments_0based:
        fragments_0based = dict(inferred.get("fragments_0based") or {})
        if len(fragments_0based) < 2:
            image_run_ids = WorkflowRun.objects.filter(
                metadata__structure__source_mlneb_workflow_run_id=source.pk,
            ).values_list("pk", flat=True)
            lineage = _first_three_fragment_config(
                dict((run.metadata or {}).get("submission", {}).get("charge_transfer_config") or {})
                for run in WorkflowRun.objects.filter(status="completed", metadata__source_run_id__in=image_run_ids).order_by("-pk")
            )
            fragments_0based = dict(lineage.get("fragments_0based") or {})
            if fragments_0based:
                inferred = lineage
    if len(fragments_0based) < 2:
        raise ValueError("MEP UV-vis requires at least two fragments for charge transfer lineage")
    fragments_1based = {
        key: _format_compact_indices([index + 1 for index in indices])
        for key, indices in fragments_0based.items()
    }

    digest = hashlib.sha256(cache.read_bytes()).hexdigest()
    existing = {}
    for run in WorkflowRun.objects.order_by("pk"):
        marker = dict((run.metadata or {}).get("structure") or {})
        if (
            marker.get("purpose") == "mep_uvvis_snapshot"
            and int(marker.get("source_mlneb_workflow_run_id") or 0) == source.pk
            and int(marker.get("padding_per_segment") or 0) == int(padding_per_segment)
            and marker.get("source_mep_sha256") == digest
            and (run.metadata or {}).get("submission", {}).get("single_constraints_config", {}).get("final_tddft_gradient_root") == gradient_root
            and (run.metadata or {}).get("submission", {}).get("single_constraints_config", {}).get("final_tddft_nroots") == int(nroots)
            and (run.metadata or {}).get("submission", {}).get("single_constraints_config", {}).get("initial_wavefunction_artifact_id") == initial_wavefunction_artifact_id
            and (not settings_profile_name or getattr(run.settings_profile, "name", "") == settings_profile_name)
            and (run.status == "completed" or run.jobs.filter(status="completed").exists())
        ):
            existing[int(marker["source_mep_image_index"])] = run
    missing = [index for index in selected if index not in existing]
    if not missing:
        return {"status": "existing", "source_run_id": source.pk, "workflow_run_ids": [existing[i].pk for i in selected]}

    reference = WorkflowRun.objects.filter(
        status="completed",
        metadata__submission__single_constraints_config__snapshot_single_point=True,
        metadata__submission__single_constraints_config__final_tddft_spectrum=True,
    ).order_by("-pk").first()
    if reference is None:
        raise ValueError("No completed snapshot-spectrum submission template exists")
    reference_submission = dict((reference.metadata or {}).get("submission") or {})
    hpc = HPCSystem.objects.get(name__iexact="FALCON", is_active=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    source_scheduler_id = str(source.jobs.order_by("-pk").values_list("scheduler_job_id", flat=True).first() or "")
    ct_labels = dict(fragment_labels or {}) if fragment_labels else dict(inferred.get("fragment_labels") or {})
    ct_config = {
        "fragments_0based": fragments_0based,
        "fragments_1based": fragments_1based,
        "fragment_labels": ct_labels,
    }
    submitted = []
    for index in missing:
        atoms, segment, fraction, original_index = padded[index]
        removed = len(atoms.constraints)
        atoms.set_constraint()
        structure_xyz = atoms_to_extxyz(atoms)
        coordinate_digest = hashlib.sha256(atoms.positions.astype("<f8", copy=False).tobytes()).hexdigest()
        if settings_profile_name:
            from references.models import CalculationSettingsProfile
            resolved_profile = CalculationSettingsProfile.objects.get(name=settings_profile_name)
        else:
            resolved_profile = source.settings_profile
        profile_tag = "FullTDDFT" if "full" in str(getattr(resolved_profile, "name", "")).lower() or not (getattr(resolved_profile, "metadata", {}) or {}).get("orca", {}).get("use_tda", True) else "TDA"
        result = submit_slurm_workflow_run(
            workflow=reference.workflow, hpc_system=hpc, settings_profile=resolved_profile,
            run_name=f"{source.run_name[:45]}_pad{index:02d}_{profile_tag}_{timestamp}", structure_xyz=structure_xyz,
            remote_runs_root=str(reference_submission.get("remote_runs_root") or settings.REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT).strip(),
            scratch_base=str(reference_submission.get("scratch_base") or settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE).strip(),
            account=str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT), partition="htc_genoa" if str(partition) == "standard" else str(partition), qos="", reservation=str(reservation),
            walltime=str(walltime), nodes=1, tasks_per_node=48, cpus_per_task=1,
            module_loads=list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES),
            venv_activate_path=settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH,
            pythonpath_entries=list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH),
            fmax=0.0, active_learning_optimizer="bfgs", qm_backend="orca", n_iter=1,
            use_tddft_optimisation=False, run_mode="single_optimisation",
            spin_multiplicity=(source.metadata or {}).get("submission", {}).get("spin_multiplicity"),
            system_charge=int(source.system_charge or 0), charge_transfer_config=ct_config,
            single_constraints_config={
                "fixed_symbols": (), "final_tddft_spectrum": True, "snapshot_single_point": True,
                "cluster_tddft_difference_spectrum": False, "charge_transfer_spectrum": True,
                "final_tddft_nroots": int(nroots), "final_tddft_nroots_mode": "manual",
                "final_tddft_gradient_root": gradient_root,
                "initial_wavefunction_artifact_id": initial_wavefunction_artifact_id,
            },
            slurm_preset="falcon_htc", submitted_by="mep-uvvis-snapshots",
            structure_metadata={
                "purpose": "mep_uvvis_snapshot", "source_mlneb_workflow_run_id": source.pk,
                "source_mlneb_scheduler_job_id": source_scheduler_id, "source_mep_artifact_id": artifact_id,
                "source_mep_sha256": digest, "source_mep_image_index": index,
                "source_mep_image_count": len(padded), "source_mep_segment_index": segment,
                "source_mep_segment_fraction": fraction, "source_mep_original_image_index": original_index,
                "padding_per_segment": int(padding_per_segment), "source_coordinate_sha256": coordinate_digest,
                "snapshot_single_point": True, "geometry_optimisation": False,
                "constraints_removed": removed, "atom_count": len(atoms), "formula": atoms.get_chemical_formula(),
                "fragments_0based": fragments_0based, "fragment_labels": ct_config["fragment_labels"],
            },
        )
        submitted.append({"image_index": index, "workflow_run_id": result.workflow_run_id, "job_record_id": result.job_record_id, "scheduler_job_id": result.scheduler_job_id})
    return {"status": "submitted", "source_run_id": source.pk, "image_count": len(padded), "submitted": submitted}


def enqueue_mep_state_tracking(
    source_run_id: int,
    image_run_ids: list[int] | None = None,
    image_directories: list[str] | None = None,
    root_state: int = 25,
    **kwargs,
) -> dict:
    """Convenience helper to enqueue continuous MEP state tracking on HPC."""
    payload = {
        "source_run_id": source_run_id,
        "root_state": root_state,
        **kwargs,
    }
    if image_run_ids:
        payload["image_run_ids"] = list(image_run_ids)
    if image_directories:
        payload["image_directories"] = list(image_directories)
    return enqueue_analysis_request("mep_state_tracking", payload)


def _submit_ts_vibrations_gradient_run(
    source_run_id: int,
    iroot: int = 24,
    nroots: int | None = None,
    hpc_name: str = "FALCON",
    walltime: str = "02:00:00",
    reservation: str = "",
    qos: str = "",
    partition: str = "",
    ground_state_only: bool = False,
    relax_endpoints: bool = False,
    displacement_A: float = 0.1,
    relaxation_fmax: float = 0.02,
    relaxation_steps: int = 300,
    reaction_bond: list[int] | None = None,
    product_bond_change: int = -1,
    structure_artifact_id: int | None = None,
    initial_wavefunction_artifact_id: int | None = None,
) -> dict:
    from django.conf import settings
    from django.utils import timezone
    from workflows.models import WorkflowRun
    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    target_hpc_name = str(hpc_name or "FALCON").strip()
    is_falcon = target_hpc_name.upper() == "FALCON"
    hpc_system = HPCSystem.objects.get(name="FALCON" if is_falcon else target_hpc_name, is_active=True)

    source_job = source.jobs.order_by("-pk").first()
    source_directory = (
        str(source_job.remote_workdir)
        if source_job and source_job.remote_workdir
        else str(source.working_directory or "")
    )
    if not source_directory:
        raise ValueError(f"WorkflowRun {source.pk} has no recorded source directory.")

    existing_ct = dict((source.metadata or {}).get("submission", {}).get("charge_transfer_config") or {})

    if structure_artifact_id is None:
        structure_xyz = _final_structure_xyz(source)
    else:
        from pathlib import Path
        from results.models import Artifact
        from ase.io import read, write
        import io
        artifact = Artifact.objects.get(pk=structure_artifact_id, workflow_run=source, kind__in=("structure", "trajectory"))
        cached = Path(str((artifact.metadata or {}).get("local_cache_path") or "")).resolve()
        if not cached.is_relative_to(Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()) or not cached.is_file() or cached.stat().st_size > 20 * 1024 * 1024:
            raise ValueError("Source structure requires a bounded registered local cache")
        buffer = io.StringIO()
        write(buffer, read(str(cached), index=-1), format="xyz")
        structure_xyz = buffer.getvalue()
    from ase.io import read as ase_read
    import io
    atoms_obj = ase_read(io.StringIO(structure_xyz), format="xyz")
    n_atoms = len(atoms_obj)

    if reaction_bond is not None and any(i >= n_atoms for i in reaction_bond):
        raise ValueError("reaction_bond index exceeds source atom count")
    if not existing_ct.get("fragments_0based") and ground_state_only:
        cluster = [i for i, atom in enumerate(atoms_obj) if atom.number in (29, 79)]
        adsorbate = [i for i in range(n_atoms) if i not in cluster]
        existing_ct.update(
            fragments_0based={"1": cluster, "2": adsorbate},
            fragments_1based={"1": ",".join(str(i + 1) for i in cluster), "2": ",".join(str(i + 1) for i in adsorbate)},
            fragment_labels={"1": "cluster", "2": "adsorbate"},
        )
    if not existing_ct.get("fragments_0based"):
        if n_atoms == 23:
            existing_ct["fragments_0based"] = {"1": list(range(20)), "2": [21], "3": [20, 22]}
            existing_ct["fragments_1based"] = {"1": "1-20", "2": "22", "3": "21,23"}
            existing_ct["fragment_labels"] = {"1": "Au20", "2": "O", "3": "CO"}
        elif n_atoms == 22:
            existing_ct["fragments_0based"] = {"1": list(range(20)), "2": [20], "3": [21]}
            existing_ct["fragments_1based"] = {"1": "1-20", "2": "21", "3": "22"}
            existing_ct["fragment_labels"] = {"1": "NC", "2": "O", "3": "O"}
        elif n_atoms == 15:
            existing_ct["fragments_0based"] = {"1": list(range(13)), "2": [13], "3": [14]}
            existing_ct["fragments_1based"] = {"1": "1-13", "2": "14", "3": "15"}
            existing_ct["fragment_labels"] = {"1": "NC", "2": "O", "3": "O"}
        elif n_atoms == 16:
            existing_ct["fragments_0based"] = {"1": list(range(13)), "2": [13], "3": [14, 15]}
            existing_ct["fragments_1based"] = {"1": "1-13", "2": "14", "3": "15-16"}
            existing_ct["fragment_labels"] = {"1": "NC", "2": "O", "3": "CO"}
        else:
            existing_ct["fragments_0based"] = {"1": list(range(max(1, n_atoms - 2))), "2": [max(1, n_atoms - 2)], "3": [n_atoms - 1]}
            existing_ct["fragments_1based"] = {"1": f"1-{max(1, n_atoms - 2)}", "2": f"{max(1, n_atoms - 2) + 1}", "3": f"{n_atoms}"}
            existing_ct["fragment_labels"] = {"1": "NC", "2": "Frag1", "3": "Frag2"}

    charge_val = int(source.system_charge or 0)
    spin_mult_val = (source.metadata or {}).get("submission", {}).get("spin_multiplicity")
    if spin_mult_val is None:
        spin_mult_val = (sum(atoms_obj.get_atomic_numbers()) - charge_val) % 2 + 1

    target_iroot = int(iroot)
    target_nroots = int(nroots) if (nroots is not None and int(nroots) > target_iroot) else (target_iroot + 10)

    ct_config = {
        "source_run_id": source.pk,
        "source_run_uid": str(source.run_uid),
        "source_run_name": source.run_name,
        "source_working_directory": source_directory,
        "fragments_0based": existing_ct.get("fragments_0based", {}),
        "fragments_1based": existing_ct.get("fragments_1based", {}),
        "fragment_labels": existing_ct.get("fragment_labels", {}),
        "density_analysis_only": True,
        "density_analysis": {
            "ts_vibrations_gradient": {
                "iroot": target_iroot,
                "ground_state_only": bool(ground_state_only),
                "initial_wavefunction_artifact_id": initial_wavefunction_artifact_id,
                "relax_endpoints": bool(relax_endpoints),
                "displacement_A": displacement_A,
                "relaxation_fmax": relaxation_fmax,
                "relaxation_steps": relaxation_steps,
                "reaction_bond": reaction_bond,
                "product_bond_change": product_bond_change,
                "nroots": target_nroots,
                "spin_multiplicity": int(spin_mult_val),
                "system_charge": charge_val,
            },
        },
    }

    result = submit_slurm_workflow_run(
        workflow=source.workflow,
        hpc_system=hpc_system,
        settings_profile=source.settings_profile,
        run_name=f"{source.run_name[:45]}_ts_vib_grad_s{iroot}_{timezone.now().strftime('%Y%m%d_%H%M%S')}",
        structure_xyz=structure_xyz,
        remote_runs_root=f"{str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE).rstrip('/')}/reaction_hub_runs" if is_falcon else str(settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT),
        scratch_base=str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE) if is_falcon else str(settings.REACTION_HUB_SUBMIT_SCRATCH_BASE),
        account=str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT) if is_falcon else "SLURM_ACCOUNT",
        partition=str(partition or ("htc_genoa" if is_falcon else "standard")).strip(),
        qos=str(qos or ("" if is_falcon else ("reservation" if (reservation or getattr(settings, "REACTION_HUB_ACTIVE_RESERVATIONS", None)) else "standard"))).strip(),
        reservation=str(reservation or (getattr(settings, "REACTION_HUB_ACTIVE_RESERVATIONS", [{}])[0].get("name", "") if not is_falcon else "")).strip(),
        walltime=str(walltime or "02:00:00"),
        nodes=1,
        tasks_per_node=48 if is_falcon else 128,
        cpus_per_task=1,
        module_loads=list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES) if is_falcon else ["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path=settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH if is_falcon else settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH,
        pythonpath_entries=list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH) if is_falcon else list(settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH),
        spin_multiplicity=int(spin_mult_val),
        system_charge=charge_val,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="charge_transfer",
        charge_transfer_config=ct_config,
        slurm_preset="falcon_htc" if is_falcon else "custom",
        submitted_by="codex-ts-analysis",
        slurm_memory="320G" if is_falcon and relax_endpoints else "",
        structure_metadata={
            "source_run_id": source.pk,
            "purpose": "ts_linked_ground_path" if relax_endpoints else "transition_state_ground_vibrations_and_excited_gradient",
            "structure_artifact_id": structure_artifact_id,
            "iroot": int(target_iroot),
            "nroots": int(target_nroots),
        },
    )
    return {
        "operation": "ts_vibrations_gradient",
        "source_run_id": source.pk,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
        "iroot": int(iroot),
    }


def enqueue_ts_vibrations_gradient(
    source_run_id: int,
    iroot: int = 31,
    nroots: int = 40,
    **kwargs,
) -> dict:
    """Convenience helper to enqueue TS NumFreq + EnGrad on Falcon."""
    payload = {
        "source_run_id": source_run_id,
        "iroot": iroot,
        "nroots": nroots,
        **kwargs,
    }
    return enqueue_analysis_request("ts_vibrations_gradient", payload)


def enqueue_dimer_refinement(source_run_id: int, **kwargs) -> dict:
    """Convenience helper to enqueue Dimer TS refinement via Celery submissions queue."""
    payload = {"source_run_id": source_run_id, **kwargs}
    return enqueue_analysis_request("dimer_refine", payload)


def _submit_dimer_refinement(
    source_run_id: int,
    structure_xyz: str | None = None,
    structure_artifact_id: int | None = None,
    image_index: int | None = None,
    hpc_name: str = "FALCON",
    qos: str = "",
    reservation: str = "",
    partition: str = "standard",
    slurm_memory: str = "",
    walltime: str = "24:00:00",
    nodes: int = 1,
    tasks_per_node: int = 128,
    fmax: float = 0.02,
    max_steps: int = 100,
    dimer_separation: float = 0.0001,
    dimer_trial_trans_step: float = 0.001,
    dimer_maximum_translation: float = 0.05,
    dimer_f_rot_min: float = 0.1,
    dimer_f_rot_max: float = 1.0,
    dimer_max_num_rot: int = 3,
    dimer_trial_angle: float = 0.785398,
    dimer_use_central_forces: bool = True,
    initial_eigenmode: list[list[float]] | None = None,
    displacement_vector: list[list[float]] | None = None,
    backend: str = "dft",
    root_state: int | None = None,
    initial_iroot: int | None = None,
    tddft_scan_nroots: int = 50,
    target_wavelength_nm: float = 520.0,
    fragments: dict | None = None,
    dimer_neighbor_shells: int | None = None,
    vibration_neighbor_shells: int = 2,
    use_catlearn_dimer: bool = False,
    catlearn_dimer_polish_steps: int = 25,
    prior_artifact_ids: list[int] | None = None,
    prior_trajectories: list | None = None,
    restart_from: str = "",
) -> dict:
    # ponytail: refine transition state with ASE Dimer method on Falcon or TRH408
    from datetime import datetime
    import sys
    from pathlib import Path
    import numpy as np
    import io
    from ase.io import read, write
    from django.conf import settings
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from workflows.models import WorkflowDefinition, WorkflowRun
    from results.models import Artifact
    from .models import HPCSystem
    from .submission import submit_slurm_workflow_run

    source = WorkflowRun.objects.select_related("workflow", "settings_profile").get(pk=source_run_id)
    if structure_artifact_id is not None:
        import io
        from ase.io import read, write
        artifact = Artifact.objects.get(pk=structure_artifact_id, workflow_run=source,
                                        kind__in=["structure", "trajectory"])
        cache = Path(str((artifact.metadata or {}).get("local_cache_path") or ""))
        if not cache.is_file():
            raise ValueError(f"Dimer seed Artifact {structure_artifact_id} must be ingested before submission")
        seed = read(cache, index=image_index if image_index is not None else -1)
        seed.set_constraint()
        stream = io.StringIO()
        write(stream, seed, format="extxyz")
        structure_xyz = stream.getvalue()
    # If source is an MEP run and already has a completed refined TS, pick the refined TS as starting point
    # only if structure_xyz is not explicitly provided
    if not structure_xyz and (root_state is not None or str(backend).strip().lower() in {"tddft", "tda"}) and "_dimer_refine" not in source.run_name:
        existing_refined = (
            WorkflowRun.objects.filter(run_name__startswith=f"{source.run_name}_dimer_refine", status="completed")
            .exclude(run_name__icontains="full_ct")
            .first()
            or WorkflowRun.objects.filter(metadata__structure_metadata__source_run_id=source.pk, run_name__icontains="dimer_refine", status="completed")
            .exclude(run_name__icontains="full_ct")
            .first()
            or WorkflowRun.objects.filter(metadata__structure__source_run_id=source.pk, run_name__icontains="dimer_refine", status="completed")
            .exclude(run_name__icontains="full_ct")
            .first()
        )
        if existing_refined:
            source = existing_refined

    workflow = _workflow_by_code_path(WorkflowDefinition, "cats/working_directory/active_learning_neb_workflow.py")
    target_hpc_name = str(hpc_name or "FALCON").strip()
    is_falcon = target_hpc_name.upper() == "FALCON"
    hpc_system = HPCSystem.objects.get(name="FALCON" if is_falcon else target_hpc_name, is_active=True)

    # Validate: only converged DFT MEP transition states (never MACE)
    meta = source.metadata or {}
    sub = meta.get("submission") or {}
    neb = meta.get("neb_config") or {}
    summary = meta.get("result_summary") or {}
    is_converged = bool(summary.get("converged") or getattr(source, "status", None) == "completed" or "_dimer_refine" in getattr(source, "run_name", "") or structure_xyz)
    qm_backend = str(sub.get("qm_backend") or neb.get("backend") or summary.get("backend") or "").strip().lower()
    if not is_converged:
        raise ValueError(f"WorkflowRun {source_run_id} is not converged.")
    if qm_backend == "mace":
        raise ValueError(f"WorkflowRun {source_run_id} used MACE; only DFT MEP transition states are eligible.")

    # 1. Resolve TS structure
    ts_xyz_content = str(structure_xyz).strip() if structure_xyz else ""
    if not ts_xyz_content:
        ts_art = (
            Artifact.objects.filter(workflow_run=source, label="structure:transition_state.xyz").first()
            or Artifact.objects.filter(workflow_run=source, label__icontains="transition_state.xyz").first()
            or Artifact.objects.filter(workflow_run=source, label__icontains="reaction_hub_final.xyz").first()
        )
        if ts_art and ts_art.metadata.get("local_cache_path"):
            p = Path(ts_art.metadata["local_cache_path"])
            if p.is_file():
                ts_xyz_content = p.read_text(encoding="utf-8")
    if not ts_xyz_content:
        snap_prop = source.computed_properties.filter(name="final_structure_snapshot").first()
        if snap_prop and snap_prop.metadata:
            from ase import Atoms
            import io
            m = snap_prop.metadata
            a = Atoms(m["symbols"], positions=m["positions_A"])
            buf = io.StringIO()
            a.write(buf, format="xyz")
            ts_xyz_content = buf.getvalue()

    band_runs = [source]
    curr = source
    while curr:
        parent_id = (curr.metadata or {}).get("structure", {}).get("source_run_id") or (curr.metadata or {}).get("structure_metadata", {}).get("source_run_id")
        if parent_id:
            parent_run = WorkflowRun.objects.filter(pk=parent_id).first()
            if parent_run and parent_run not in band_runs:
                band_runs.append(parent_run)
                curr = parent_run
                continue
        break

    band_art = None
    for b_run in band_runs:
        band_art = (
            Artifact.objects.filter(workflow_run=b_run, label__icontains="final band").first()
            or Artifact.objects.filter(workflow_run=b_run, path__icontains="final_band").first()
            or Artifact.objects.filter(workflow_run=b_run, label__icontains="final_band").first()
            or Artifact.objects.filter(workflow_run=b_run, path__icontains="band").first()
        )
        if band_art:
            break

    if not ts_xyz_content:
        if band_art and band_art.metadata.get("local_cache_path"):
            from ase.io import read
            import io
            band = read(band_art.metadata["local_cache_path"], ":")
            ts_idx = int(image_index) if image_index is not None else int((source.metadata or {}).get("result_summary", {}).get("transition_state_image_index", len(band) // 2))
            buf = io.StringIO()
            band[ts_idx].write(buf, format="xyz")
            ts_xyz_content = buf.getvalue()
    if not ts_xyz_content:
        raise ValueError(f"Could not resolve transition state structure for WorkflowRun {source_run_id}")

    # 2. Resolve initial eigenmode if not explicitly provided
    resolved_eigenmode = initial_eigenmode
    if resolved_eigenmode is None:
        sub_mode = (source.metadata or {}).get("submission", {}).get("neb_config", {}).get("initial_eigenmode") or (source.metadata or {}).get("submission", {}).get("initial_eigenmode")
        if sub_mode:
            resolved_eigenmode = sub_mode

    band_text = ""
    if band_art and band_art.metadata.get("local_cache_path"):
        p = Path(band_art.metadata["local_cache_path"])
        if p.is_file():
            try:
                from ase.io import read, write
                import io
                band_images = read(str(p), ":")
                if len(band_images) >= 3:
                    ts_idx = int((source.metadata or {}).get("result_summary", {}).get("transition_state_image_index", len(band_images) // 2))
                    start_i = max(0, min(ts_idx - 1, len(band_images) - 3))
                    sub_band = band_images[start_i : start_i + 3]
                    buf = io.StringIO()
                    write(buf, sub_band, format="extxyz")
                    band_text = buf.getvalue()
                else:
                    band_text = p.read_text(encoding="utf-8")
            except Exception:
                band_text = p.read_text(encoding="utf-8")

    if resolved_eigenmode is None:
        # Check if an imaginary normal mode is available from a related frequency calculation (e.g. NumFreq)
        hess_candidates = Artifact.objects.filter(path__endswith="numfreq.hess").select_related("job__workflow_run").order_by("-pk")
        hess_art = None
        for cand in hess_candidates:
            w_run = getattr(cand.job, "workflow_run", None)
            meta_str = str(getattr(w_run, "metadata", "") or "")
            if str(source_run_id) in meta_str:
                hess_art = cand
                break
        if not hess_art and hess_candidates.exists():
            first_cand = hess_candidates.first()
            first_run = getattr(first_cand.job, "workflow_run", None)
            if first_run and source.run_name.split("_")[1:3] == first_run.run_name.split("_")[1:3]:
                hess_art = first_cand
        if hess_art and hess_art.metadata.get("local_cache_path"):
            try:
                from cats.orca.routines.ts_vibrations_gradient import parse_orca_hess
                hess_data = parse_orca_hess(hess_art.metadata["local_cache_path"])
                freqs = hess_data.get("frequencies_cm1", [])
                modes = hess_data.get("normal_modes")
                imag_indices = [i for i, f in enumerate(freqs) if f < -10.0]
                if imag_indices and modes is not None:
                    lowest_idx = imag_indices[0]
                    m_vec = modes[:, lowest_idx].reshape((-1, 3))
                    m_norm = np.linalg.norm(m_vec)
                    if m_norm > 1e-6:
                        resolved_eigenmode = (m_vec / m_norm).tolist()
            except Exception:
                resolved_eigenmode = None

    if resolved_eigenmode is None and band_art and band_art.metadata.get("local_cache_path"):
        try:
            from ase.io import read
            band = read(band_art.metadata["local_cache_path"], ":")
            ts_idx = int((source.metadata or {}).get("result_summary", {}).get("transition_state_image_index", len(band) // 2))
            if 0 < ts_idx < len(band) - 1:
                tau = band[ts_idx + 1].positions - band[ts_idx - 1].positions
                norm = np.linalg.norm(tau)
                if norm > 1e-6:
                    resolved_eigenmode = (tau / norm).tolist()
        except Exception:
            resolved_eigenmode = None

    submission = dict((source.metadata or {}).get("submission") or {})
    source_job = source.jobs.order_by("-pk").first()
    if source_job is None or source_job.hpc_system_id != hpc_system.pk:
        submission = {}
    if is_falcon:
        remote_runs_root = str(settings.REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT)
        scratch_base = str(settings.REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE)
        account = str(settings.REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT)
        module_loads = list(settings.REACTION_HUB_SUBMIT_FALCON_MODULES)
        venv_activate_path = str(settings.REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH)
        pythonpath_entries = list(settings.REACTION_HUB_SUBMIT_FALCON_PYTHONPATH)
        partition = partition if partition and partition != "standard" else "htc_genoa"
        qos = qos if qos and qos not in {"reservation", "taskfarm"} else ""
        reservation = reservation if reservation and "SLURM_ACCOUNT" not in reservation else ""
        tasks_per_node = tasks_per_node if tasks_per_node and tasks_per_node != 128 else 48
        slurm_memory = slurm_memory or "320G"
    else:
        sub_root = str(submission.get("remote_runs_root") or "")
        sub_scratch = str(submission.get("scratch_base") or "")
        remote_runs_root = sub_root if (sub_root and not sub_root.startswith("/shared")) else str(settings.REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT)
        scratch_base = sub_scratch if (sub_scratch and not sub_scratch.startswith("/shared")) else str(settings.REACTION_HUB_SUBMIT_SCRATCH_BASE)
        account = str(settings.REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT or "SLURM_ACCOUNT")
        module_loads = ["epcc-setup-env", "cray-python", "orca/6.0.0"]
        venv_activate_path = str(submission.get("venv_activate_path") or settings.REACTION_HUB_SUBMIT_DEFAULT_VENV_ACTIVATE_PATH)
        pythonpath_entries = list(submission.get("pythonpath_entries") or settings.REACTION_HUB_SUBMIT_DEFAULT_PYTHONPATH)
        slurm_memory = ""
        from .submission import resolve_active_reservation
        active_res = resolve_active_reservation(
            hpc_key=target_hpc_name.upper(),
            partition=partition,
            account=account,
            qos=qos,
            user_reservation=reservation,
        )
        reservation = active_res
        if reservation:
            qos = "reservation" if qos in {"", "reservation"} else qos
        else:
            qos = "standard" if qos in {"", "reservation"} else qos

    is_tddft = str(backend).strip().lower() in {"tddft", "tda"} or root_state is not None
    resolved_backend = "tddft" if is_tddft else "dft"
    resolved_iroot = int(initial_iroot or root_state or 5)

    seed_training_extxyz = ""
    resolved_prior_artifact_ids: list[int] = []
    rejected_artifacts: list[dict] = []
    staged_seed_file: str | None = None
    if use_catlearn_dimer:
        import io
        from ase.io import read as ase_read, write as ase_write
        from cats.working_directory.catlearn_dimer import validate_dft_frame, seed_dimer_training_set

        candidate_artifacts = []
        if prior_artifact_ids:
            for aid in prior_artifact_ids:
                art = Artifact.objects.filter(pk=aid).first()
                if art is None:
                    rejected_artifacts.append({"artifact_id": aid, "reason": "artifact does not exist"})
                else:
                    candidate_artifacts.append(art)
        else:
            resolved_arts = Artifact.objects.filter(
                workflow_run__in=band_runs,
                kind__in=["trajectory", "structure"],
            )
            candidate_artifacts = list(resolved_arts)

        target_atoms = None
        if ts_xyz_content:
            try:
                target_atoms = ase_read(io.StringIO(ts_xyz_content), format="xyz" if "\n" in ts_xyz_content else "extxyz")
            except Exception:
                target_atoms = None

        candidate_frames = []
        for art in candidate_artifacts:
            cache_p = Path(str((art.metadata or {}).get("local_cache_path") or ""))
            if not cache_p.is_file():
                rejected_artifacts.append({"artifact_id": art.pk, "reason": "local cache file missing"})
                continue
            try:
                frames = list(ase_read(str(cache_p), ":"))
            except Exception as exc:
                rejected_artifacts.append({"artifact_id": art.pk, "reason": f"read error: {exc}"})
                continue

            valid_in_art = 0
            art_reasons = []
            for f in frames:
                if target_atoms is not None:
                    is_valid, reason, _, _ = validate_dft_frame(f, target_atoms)
                    if is_valid:
                        valid_in_art += 1
                        candidate_frames.append(f)
                    else:
                        art_reasons.append(reason)
                else:
                    candidate_frames.append(f)
                    valid_in_art += 1

            if valid_in_art > 0:
                resolved_prior_artifact_ids.append(art.pk)
            else:
                summary_reasons = ", ".join(sorted(set(art_reasons))) or "no valid frames"
                rejected_artifacts.append({"artifact_id": art.pk, "reason": f"incompatible frames: {summary_reasons}"})

        if band_text:
            try:
                candidate_frames.extend(ase_read(io.StringIO(band_text), ":", format="extxyz"))
            except Exception:
                pass

        if candidate_frames and target_atoms:
            try:
                seeded = seed_dimer_training_set(candidate_frames, target_atoms, min_dist=0.03, max_frames=80)
                if seeded:
                    out_buf = io.StringIO()
                    ase_write(out_buf, seeded, format="extxyz")
                    seed_training_extxyz = out_buf.getvalue()
                    staged_seed_file = "seed_training.extxyz"
            except Exception:
                pass

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=hpc_system,
        settings_profile=source.settings_profile,
        run_name=f"{source.run_name.split('_dimer_refine')[0]}_dimer_refine_{'excited_s' + str(resolved_iroot) + '_' if is_tddft else ''}{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        structure_xyz=ts_xyz_content,
        remote_runs_root=remote_runs_root,
        scratch_base=scratch_base,
        account=account,
        partition=partition,
        qos=qos,
        reservation=reservation,
        slurm_memory=slurm_memory,
        walltime=walltime,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=1,
        module_loads=module_loads,
        venv_activate_path=venv_activate_path,
        pythonpath_entries=pythonpath_entries,
        fmax=fmax,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        spin_multiplicity=source.metadata.get("submission", {}).get("spin_multiplicity"),
        system_charge=int(source.system_charge or 0),
        n_iter=1,
        use_tddft_optimisation=is_tddft,
        target_wavelength_nm=float(target_wavelength_nm or 520.0),
        run_mode="active_learning_neb",
        neb_config={
            "final_image_strategy": "dimer",
            "backend": resolved_backend,
            "initial_band_extxyz": band_text,
            "initial_eigenmode": resolved_eigenmode,
            "displacement_vector": displacement_vector,
            "dimer_max_steps": max_steps,
            "dimer_separation": dimer_separation,
            "dimer_trial_trans_step": dimer_trial_trans_step,
            "dimer_maximum_translation": dimer_maximum_translation,
            "dimer_f_rot_min": dimer_f_rot_min,
            "dimer_f_rot_max": dimer_f_rot_max,
            "dimer_max_num_rot": dimer_max_num_rot,
            "dimer_trial_angle": dimer_trial_angle,
            "dimer_use_central_forces": dimer_use_central_forces,
            "use_catlearn_dimer": bool(use_catlearn_dimer),
            "catlearn_dimer_polish_steps": int(catlearn_dimer_polish_steps),
            "prior_artifact_ids": resolved_prior_artifact_ids,
            "rejected_artifacts": rejected_artifacts,
            "staged_seed_file": staged_seed_file,
            "seed_training_extxyz": seed_training_extxyz,
            "prior_trajectories": prior_trajectories,
            "restart_from": str(restart_from or (source.metadata or {}).get("submission", {}).get("remote_work_directory") or getattr(source, "working_directory", "") or "").strip(),
            "initial_iroot": resolved_iroot,
            "tddft_scan_nroots": int(tddft_scan_nroots or 50),
            "target_wavelength_nm": float(target_wavelength_nm or 520.0),
            "fragments": fragments or (source.metadata or {}).get("charge_transfer", {}).get("fragments_1based") or {"1": "1-20", "2": "21-22", "3": "23"},
            "dimer_neighbor_shells": int(
                dimer_neighbor_shells
                if dimer_neighbor_shells is not None
                else (
                    0
                    if sum(
                        1
                        for a in (
                            read(io.StringIO(structure_xyz), format="extxyz")
                            if structure_xyz
                            else []
                        )
                        if a.symbol in ("Cu", "Au", "Ag", "Pt", "Pd")
                    )
                    <= 20
                    else 2
                )
            ),
            "vibration_neighbor_shells": int(vibration_neighbor_shells),
        },
        structure_metadata={
            "source_run_id": source.pk,
            "purpose": f"dimer_saddle_point_refinement_{'excited_s' + str(resolved_iroot) if is_tddft else 'ground_state'}",
            "source_structure_artifact_id": structure_artifact_id,
            "source_structure_frame_index": image_index if image_index is not None else -1,
            "target_root": resolved_iroot if is_tddft else 0,
            "formula": (source.metadata or {}).get("structure", {}).get("formula", ""),
            "reaction": (source.metadata or {}).get("structure", {}).get("reaction", ""),
        },
    )
    return {
        "status": "submitted",
        "source_run_id": source.pk,
        "workflow_run_id": result.workflow_run_id,
        "job_record_id": result.job_record_id,
        "scheduler_job_id": result.scheduler_job_id,
        "remote_run_dir": getattr(result, "remote_run_dir", ""),
    }


def auto_refine_next_converged_dft_ts(
    hpc_name: str = "FALCON",
    reservation: str = "",
    qos: str = "",
    partition: str = "standard",
    fmax: float = 0.05,
    max_steps: int = 100,
) -> dict:
    """Find next unrefined established converged DFT MEP TS and enqueue refinement."""
    from workflows.models import WorkflowRun

    established_ids = [8769, 8768, 8766, 8762, 8757, 3790, 3788, 3787, 3542, 3326, 3325]
    existing_runs = list(
        WorkflowRun.objects.filter(run_name__icontains="dimer_refine")
        .exclude(status__in=["cancelled", "failed"])
        .values("id", "run_name", "metadata")
    )
    for run_id in established_ids:
        source = WorkflowRun.objects.filter(pk=run_id).first()
        if not source:
            continue
        already_refined = False
        for ex in existing_runs:
            s_id = (ex.get("metadata") or {}).get("structure_metadata", {}).get("source_run_id") or (ex.get("metadata") or {}).get("structure", {}).get("source_run_id")
            name = ex.get("run_name") or ""
            if s_id == run_id or name.startswith(f"{source.run_name}_dimer_refine"):
                already_refined = True
                break
        if not already_refined:
            return enqueue_dimer_refinement(
                source_run_id=run_id,
                hpc_name=hpc_name,
                reservation=reservation,
                qos=qos,
                partition=partition,
                fmax=fmax,
                max_steps=max_steps,
            )
    return {"status": "all_converged_dft_ts_already_refined", "checked_count": len(established_ids)}




def auto_refine_all_converged_dft_ts(
    hpc_name: str = "FALCON",
    reservation: str = "",
    qos: str = "",
    partition: str = "standard",
    fmax: float = 0.05,
    max_steps: int = 100,
) -> list[dict]:
    """Find and enqueue all unrefined established converged DFT MEP TSs at once."""
    from workflows.models import WorkflowRun

    established_ids = [8769, 8768, 8766, 8762, 8757, 3790, 3788, 3787, 3542, 3326, 3325]
    existing_runs = list(
        WorkflowRun.objects.filter(run_name__icontains="dimer_refine")
        .exclude(status__in=["cancelled", "failed"])
        .values("id", "run_name", "metadata")
    )
    submitted_results = []
    for run_id in established_ids:
        source = WorkflowRun.objects.filter(pk=run_id).first()
        if not source:
            continue
        already_refined = False
        for ex in existing_runs:
            s_id = (ex.get("metadata") or {}).get("structure_metadata", {}).get("source_run_id") or (ex.get("metadata") or {}).get("structure", {}).get("source_run_id")
            name = ex.get("run_name") or ""
            if s_id == run_id or name.startswith(f"{source.run_name}_dimer_refine"):
                already_refined = True
                break
        if not already_refined:
            try:
                res = enqueue_dimer_refinement(
                    source_run_id=run_id,
                    hpc_name=hpc_name,
                    reservation=reservation,
                    qos=qos,
                    partition=partition,
                    fmax=fmax,
                    max_steps=max_steps,
                )
                submitted_results.append(res)
            except Exception as e:
                submitted_results.append({"status": "error", "source_run_id": run_id, "error": str(e)})
    return submitted_results
