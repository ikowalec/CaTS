from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from collections import Counter
from io import StringIO
from pathlib import Path, PurePosixPath

from ase.data import covalent_radii
from ase.db import connect as ase_connect
from ase.io import read as ase_read
from django.conf import settings
from django.contrib import messages
from django.http import HttpResponseRedirect, JsonResponse, QueryDict
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.views.generic import DetailView
from django.views.generic.edit import FormView
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin

from core.openssh import OpenSSHClient
from core.table_pagination import FriendlyTablePaginationMixin
from references.models import ReferenceAdsorbate, ReferenceNanocluster
from results.models import Artifact, ArtifactKind
from results.services import _METAL_SYMBOLS
from ingest.spectral import load_spectral_dataset
from workflows.identity import canonical_run_path, normalize_run_path
from workflows.models import WorkflowDefinition, WorkflowRun, WorkflowStatus
from workflows.views import resolve_ase_gui_command

from .forms import GeometrySource, JobSubmitForm, ReferenceIntent, RunMode
from .geometry import (
    atoms_to_extxyz,
    atoms_to_xyz,
    structure_file_to_atoms_frames,
    structure_file_to_xyz_frames,
    structure_input_to_atoms,
    structure_path_to_atoms,
    structure_path_to_atoms_frames,
)
from .filters import JobRecordFilter
from .models import HPCSystem, JobRecord, JobStatus, JobStatusEvent
from .submission import (
    _active_qos_job_count,
    _build_submission_script,
    _build_wrapper_script,
    _module_path_from_code_path,
    _resolve_remote_workdir,
    _submission_queue_limit,
    _structure_artifact_rel_path_for_mode,
    submit_slurm_workflow_run,
)
from .tables import JobRecordTable
from .tasks import (
    restart_cancelled_job_task,
    resubmit_failed_job_task,
    resubmit_replot_job_task,
    resubmit_smooth_job_task,
)

_SBATCH_JOB_ID_RE = re.compile(r"Submitted\s+batch\s+job\s+(\d+)")
_RESTART_REASONABLE_FMAX_EVA = 1.0
_REMOTE_FAILED_CODE_RE = re.compile(r"Remote command failed \((\d+)")
_ANSI_CSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
_ANSI_OSC_RE = re.compile(r"\x1B\][^\x07\x1B]*(?:\x07|\x1B\\)")
_ABS_POSIX_PATH_RE = re.compile(r"/[^\s\"'`]+")
_SCAN_ENDPOINTS_DB_FILENAME = "scan_endpoints.db"
_REPLOT_RESUBMIT_WALLTIME = "00:20:00"
_REPLOT_RESUBMIT_PARTITION = "standard"
_REPLOT_RESUBMIT_QOS = "short"


class JobRecordListView(FriendlyTablePaginationMixin, SingleTableMixin, FilterView):
    model = JobRecord
    table_class = JobRecordTable
    template_name = "jobs/job_list.html"
    filterset_class = JobRecordFilter

    def get_queryset(self):
        return (
            super()
            .get_queryset()
            .select_related("workflow_run", "hpc_system")
            .prefetch_related("workflow_run__campaign_memberships__campaign")
        )

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["cancelled_jobs_total"] = JobRecord.objects.filter(status=JobStatus.CANCELLED).count()
        return context


class JobRecordDetailView(DetailView):
    model = JobRecord
    template_name = "jobs/job_detail.html"
    context_object_name = "job"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["events"] = self.object.events.all().order_by("-created_at")
        return context


def _q(text: str) -> str:
    return shlex.quote(str(text))


def _format_wavelength_suffix_nm(value: float) -> str:
    formatted = f"{float(value):.12g}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    safe = formatted.replace(".", "p").replace("-", "m")
    return f"{safe}nm"


def _submission_structure_artifact_path(submission_meta: dict[str, object] | None) -> str:
    rel_path = str((submission_meta or {}).get("structure_artifact_path") or "").strip().replace("\\", "/")
    rel_path = rel_path.lstrip("/")
    return rel_path or "orca.xyz"


def _atoms_periodicity_payload(atoms) -> dict[str, object]:
    cell = [[float(value) for value in row] for row in atoms.get_cell().array.tolist()]
    pbc = [bool(flag) for flag in atoms.get_pbc().tolist()]
    return {
        "cell_A": cell,
        "pbc": pbc,
        "is_periodic": bool(any(pbc)),
    }


def _structure_text_for_artifact_path(
    *,
    structure_xyz: str,
    structure_meta: dict[str, object] | None,
    artifact_rel_path: str,
) -> str:
    rel_path = str(artifact_rel_path or "").strip().lower()
    if not structure_xyz or not rel_path.endswith(".extxyz"):
        return structure_xyz
    metadata = structure_meta if isinstance(structure_meta, dict) else {}
    try:
        atoms = ase_read(StringIO(structure_xyz), format="extxyz")
        return atoms_to_extxyz(atoms)
    except Exception:
        pass
    try:
        atoms = ase_read(StringIO(structure_xyz), format="xyz")
        raw_cell = metadata.get("cell_A")
        if isinstance(raw_cell, (list, tuple)) and len(raw_cell) == 3:
            try:
                atoms.set_cell([[float(row[0]), float(row[1]), float(row[2])] for row in raw_cell])
            except Exception:
                pass
        raw_pbc = metadata.get("pbc")
        if isinstance(raw_pbc, bool):
            atoms.set_pbc([bool(raw_pbc)] * 3)
        elif isinstance(raw_pbc, (list, tuple)) and len(raw_pbc) == 3:
            atoms.set_pbc([bool(raw_pbc[0]), bool(raw_pbc[1]), bool(raw_pbc[2])])
        return atoms_to_extxyz(atoms)
    except Exception:
        return structure_xyz


def _safe_next_url(request) -> str:
    candidates = [
        str(request.POST.get("next") or "").strip(),
        str(request.GET.get("next") or "").strip(),
        str(request.META.get("HTTP_REFERER") or "").strip(),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if url_has_allowed_host_and_scheme(
            url=candidate,
            allowed_hosts={request.get_host()},
            require_https=request.is_secure(),
        ):
            return candidate
    return reverse("job-list")


def _safe_positive_int(value: object, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _format_compact_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    ranges: list[str] = []
    start = previous = int(indices[0])
    for value in (int(item) for item in indices[1:]):
        if value == previous + 1:
            previous = value
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = value
    ranges.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(ranges)


def _infer_ct_fragment_defaults(atoms) -> dict[str, object]:
    metal_indices = [int(atom.index) for atom in atoms if atom.symbol in _METAL_SYMBOLS]
    nonmetal_indices = [int(atom.index) for atom in atoms if atom.symbol not in _METAL_SYMBOLS]
    if not metal_indices or not nonmetal_indices:
        return {}
    unseen = set(nonmetal_indices)
    adsorbate_components: list[list[int]] = []
    while unseen:
        component = {unseen.pop()}
        pending = list(component)
        while pending:
            current = pending.pop()
            bonded = {
                candidate
                for candidate in unseen
                if atoms.get_distance(current, candidate, mic=True)
                <= 1.2
                * (
                    float(covalent_radii[atoms[current].number])
                    + float(covalent_radii[atoms[candidate].number])
                )
            }
            unseen.difference_update(bonded)
            component.update(bonded)
            pending.extend(bonded)
        adsorbate_components.append(sorted(component))
    adsorbate_components.sort(key=lambda component: component[0])
    label_counts: Counter[str] = Counter()
    fragment_labels = {"1": "NC"}
    fragments_0based = {"1": metal_indices}
    for fragment_number, component in enumerate(adsorbate_components, start=2):
        base_label = atoms[component].get_chemical_formula(mode="hill")
        label_counts[base_label] += 1
        suffix = label_counts[base_label]
        label = base_label if suffix == 1 else f"{base_label}_{suffix}"
        key = str(fragment_number)
        fragments_0based[key] = component
        fragment_labels[key] = label
    return {
        "nanoparticle_indices": _format_compact_indices(metal_indices),
        "adsorbate_indices": _format_compact_indices(nonmetal_indices),
        "fragments_0based": fragments_0based,
        "fragment_labels": fragment_labels,
    }


def _tddft_nroots_for_run(run: WorkflowRun, *, default: int = 400) -> int:
    candidates: list[int] = []
    datasets = load_spectral_dataset(run) or {}
    for dataset_name in ("final_tddft", "charge_transfer", "bond_scan"):
        dataset = datasets.get(dataset_name)
        if not isinstance(dataset, dict):
            continue
        for raw_value in (dataset.get("nroots"), dataset.get("transition_count")):
            try:
                value = int(raw_value)
            except (TypeError, ValueError):
                continue
            if value > 0:
                candidates.append(value)
        transitions = dataset.get("transitions")
        if isinstance(transitions, list) and transitions:
            candidates.append(len(transitions))
        states = dataset.get("ct_states")
        if isinstance(states, list):
            state_numbers = []
            for state in states:
                try:
                    state_numbers.append(int(state.get("state")))
                except (AttributeError, TypeError, ValueError):
                    continue
            if state_numbers:
                candidates.append(max(state_numbers))
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission") if isinstance(metadata.get("submission"), dict) else {}
    single = (
        submission.get("single_constraints_config")
        if isinstance(submission.get("single_constraints_config"), dict)
        else {}
    )
    scan = submission.get("scan_config") if isinstance(submission.get("scan_config"), dict) else {}
    for raw_value in (
        single.get("final_tddft_nroots"),
        submission.get("tddft_scan_nroots"),
        scan.get("tddft_scan_nroots"),
    ):
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            candidates.append(value)
    return max(candidates) if candidates else int(default)


def _strip_terminal_control_sequences(text: object) -> str:
    cleaned = str(text or "")
    cleaned = _ANSI_OSC_RE.sub("", cleaned)
    cleaned = _ANSI_CSI_RE.sub("", cleaned)
    # Strip any remaining bare ESC bytes to avoid leaking control chars into payloads/messages.
    return cleaned.replace("\x1b", "")


def _extract_remote_path_from_output(text: object) -> str:
    cleaned = _strip_terminal_control_sequences(text)
    candidates = _ABS_POSIX_PATH_RE.findall(cleaned)
    if not candidates:
        return ""
    return normalize_run_path(candidates[-1])


def _remote_command_failed_with_code(exc: Exception, expected_code: int) -> bool:
    match = _REMOTE_FAILED_CODE_RE.search(str(exc or ""))
    if match is None:
        return False
    try:
        return int(match.group(1)) == int(expected_code)
    except (TypeError, ValueError):
        return False


def _safe_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for raw in value:
        text = str(raw or "").strip()
        if text:
            items.append(text)
    return items


def _parse_remote_probe_bool(output: object) -> bool:
    text = str(output or "")
    for line in reversed(text.splitlines()):
        token = line.strip()
        if token == "1":
            return True
        if token == "0":
            return False
    token = text.strip()
    if token == "1":
        return True
    if token == "0":
        return False
    raise ValueError(f"No boolean probe token found in output: {text!r}")


def _remote_directory_exists(client: OpenSSHClient, candidate_path: str) -> bool:
    exists_cmd = (
        f"if [ -d {_q(candidate_path)} ]; then "
        "printf '1'; "
        "else "
        "printf '0'; "
        "fi"
    )
    probe_output = client.exec_checked(exists_cmd)
    try:
        return _parse_remote_probe_bool(probe_output)
    except ValueError:
        if not str(probe_output or "").strip():
            return False
        # Fallback to exit-code based check in case shell startup noise polluted output.
        try:
            client.exec_checked(f"test -d {_q(candidate_path)}")
            return True
        except Exception as exc:
            if _remote_command_failed_with_code(exc, 1):
                return False
            raise


def _leading_scheduler_job_id_from_artifact_path(path_value: object) -> str:
    text = str(path_value or "").strip()
    if not text:
        return ""
    normalized = text.replace("\\", "/").lstrip("./")
    parts = [part for part in normalized.split("/") if part and part != "."]
    if parts and parts[0].isdigit():
        return parts[0]
    return ""


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _extract_scheduler_job_id_from_path(path_text: object) -> str:
    normalized = normalize_run_path(path_text)
    if not normalized:
        return ""
    parts = [part for part in normalized.split("/") if part]
    if parts and parts[0].isdigit():
        return parts[0]
    for part in parts:
        if part.isdigit() and len(part) >= 4:
            return part
    return ""


def _artifact_source_scheduler_job_id(artifact: Artifact) -> str:
    if artifact.job is not None:
        scheduler_job_id = str(artifact.job.scheduler_job_id or "").strip()
        if scheduler_job_id:
            return scheduler_job_id

    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    for key in ("source_scheduler_job_id", "scheduler_job_id", "job_id"):
        value = str(metadata.get(key) or "").strip()
        if value:
            return value

    for candidate in (
        metadata.get("remote_path"),
        metadata.get("local_cache_path"),
        artifact.path,
    ):
        extracted = _extract_scheduler_job_id_from_path(candidate)
        if extracted:
            return extracted
    return ""


def _collect_previous_scheduler_job_ids(
    submission_meta: dict[str, object],
    *,
    previous_scheduler_job_id: str = "",
) -> list[str]:
    candidates: list[str] = []
    candidates.extend(_safe_string_list(submission_meta.get("previous_scheduler_job_ids")))
    restart_history = submission_meta.get("restart_history")
    if isinstance(restart_history, list):
        for entry in restart_history:
            if not isinstance(entry, dict):
                continue
            for key in ("previous_scheduler_job_id", "new_scheduler_job_id"):
                text = str(entry.get(key) or "").strip()
                if text:
                    candidates.append(text)
    previous = str(previous_scheduler_job_id or "").strip()
    if previous:
        candidates.append(previous)

    normalized: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item in seen:
            continue
        seen.add(item)
        normalized.append(item)
    return normalized


def _run_mode_for_job_action(run: WorkflowRun) -> str:
    run_meta = run.metadata if isinstance(run.metadata, dict) else {}
    submission_meta = run_meta.get("submission") if isinstance(run_meta.get("submission"), dict) else {}
    return str(submission_meta.get("run_mode") or "").strip()


def _supports_replot_resubmit_for_run(run: WorkflowRun) -> bool:
    run_mode = _run_mode_for_job_action(run)
    if run_mode in {RunMode.SINGLE_OPTIMISATION, RunMode.TDDFT_BOND_SCAN}:
        return True
    if run_mode:
        return False
    # Legacy runs may not include submission.run_mode metadata.
    code_path = str(getattr(run.workflow, "code_path", "") or "").strip().lower()
    return "coadsorbate" not in code_path


def _delete_marker_name(job_name: str) -> str:
    base = str(job_name or "").strip()
    if not base:
        base = "job"
    base = base.replace("/", "_").replace("\\", "_")
    return f"{base}_to_delete"


def _scancel_and_rename_remote_run(job: JobRecord) -> tuple[str, str]:
    if job.hpc_system is None:
        raise RuntimeError("Job has no HPC system configured.")

    scheduler_job_id = str(job.scheduler_job_id or "").strip()
    if not scheduler_job_id:
        raise RuntimeError("Job has no scheduler job id.")

    remote_run_dir = canonical_run_path(job.workflow_run.working_directory)
    if not remote_run_dir or not remote_run_dir.startswith("/"):
        raise RuntimeError(
            "Workflow run directory is not a valid remote POSIX path; refusing remote rename."
        )

    src_path = PurePosixPath(remote_run_dir)
    target_base = str(src_path.parent / _delete_marker_name(job.job_name))

    client = OpenSSHClient(
        host=job.hpc_system.host,
        port=job.hpc_system.port,
        username=job.hpc_system.username,
        ssh_key_path=job.hpc_system.ssh_key_path,
        timeout=settings.REACTION_HUB_SSH_TIMEOUT_SECONDS,
        force_tty=settings.REACTION_HUB_SSH_FORCE_TTY,
        batch_mode=settings.REACTION_HUB_SSH_BATCH_MODE,
        use_clean_bash=settings.REACTION_HUB_SSH_CLEAN_BASH,
        bash_path=settings.REACTION_HUB_SSH_BASH_PATH,
    )
    try:
        cancel_state = "cancelled"
        try:
            client.exec_checked(f"scancel {_q(scheduler_job_id)}")
        except Exception as exc:
            text = str(exc).lower()
            if "invalid job id" in text:
                cancel_state = "not_found"
            else:
                raise RuntimeError(f"Failed to scancel job {scheduler_job_id}: {exc}") from exc

        rename_cmd = (
            f"src={_q(remote_run_dir)}; "
            f"target_base={_q(target_base)}; "
            "if [ ! -e \"$src\" ]; then echo '__MISSING_SOURCE__'; exit 44; fi; "
            "target=\"$target_base\"; "
            "if [ \"$target\" = \"$src\" ]; then target=\"${target_base}_1\"; fi; "
            "if [ -e \"$target\" ]; then "
            "idx=1; "
            "while [ -e \"${target_base}_${idx}\" ]; do idx=$((idx+1)); done; "
            "target=\"${target_base}_${idx}\"; "
            "fi; "
            "mv \"$src\" \"$target\"; "
            "printf '%s' \"$target\""
        )
        rename_out = client.exec_checked(rename_cmd)
        renamed_run_dir = _extract_remote_path_from_output(rename_out).strip()
        if not renamed_run_dir or not renamed_run_dir.startswith("/"):
            raise RuntimeError("Remote directory rename produced an empty target path.")
        return cancel_state, renamed_run_dir
    finally:
        client.close()


def _rename_remote_run_for_deletion(job: JobRecord) -> str:
    if job.hpc_system is None:
        raise RuntimeError("Job has no HPC system configured.")

    remote_run_dir = canonical_run_path(job.workflow_run.working_directory)
    if not remote_run_dir or not remote_run_dir.startswith("/"):
        raise RuntimeError(
            "Workflow run directory is not a valid remote POSIX path; refusing remote rename."
        )

    src_path = PurePosixPath(remote_run_dir)
    target_base = str(src_path.parent / _delete_marker_name(job.job_name))

    client = OpenSSHClient(
        host=job.hpc_system.host,
        port=job.hpc_system.port,
        username=job.hpc_system.username,
        ssh_key_path=job.hpc_system.ssh_key_path,
        timeout=settings.REACTION_HUB_SSH_TIMEOUT_SECONDS,
        force_tty=settings.REACTION_HUB_SSH_FORCE_TTY,
        batch_mode=settings.REACTION_HUB_SSH_BATCH_MODE,
        use_clean_bash=settings.REACTION_HUB_SSH_CLEAN_BASH,
        bash_path=settings.REACTION_HUB_SSH_BASH_PATH,
    )
    try:
        rename_cmd = (
            f"src={_q(remote_run_dir)}; "
            f"target_base={_q(target_base)}; "
            "if [ ! -e \"$src\" ]; then echo '__MISSING_SOURCE__'; exit 44; fi; "
            "target=\"$target_base\"; "
            "if [ \"$target\" = \"$src\" ]; then target=\"${target_base}_1\"; fi; "
            "if [ -e \"$target\" ]; then "
            "idx=1; "
            "while [ -e \"${target_base}_${idx}\" ]; do idx=$((idx+1)); done; "
            "target=\"${target_base}_${idx}\"; "
            "fi; "
            "mv \"$src\" \"$target\"; "
            "printf '%s' \"$target\""
        )
        rename_out = client.exec_checked(rename_cmd)
        renamed_run_dir = _extract_remote_path_from_output(rename_out).strip()
        if not renamed_run_dir or not renamed_run_dir.startswith("/"):
            raise RuntimeError("Remote directory rename produced an empty target path.")
        return renamed_run_dir
    finally:
        client.close()


class _QueueCapacityUnavailable(RuntimeError):
    pass


def _restart_cancelled_job_in_place(
    job: JobRecord,
    *,
    wait_for_queue_capacity: bool = True,
) -> tuple[str, str, dict[str, object]]:
    if job.hpc_system is None:
        raise RuntimeError("Job has no HPC system configured.")

    remote_run_root = canonical_run_path(job.workflow_run.working_directory)
    if not remote_run_root or not remote_run_root.startswith("/"):
        raise RuntimeError("Workflow run directory is not a valid remote POSIX path.")

    submission_meta = {}
    run_meta = job.workflow_run.metadata if isinstance(job.workflow_run.metadata, dict) else {}
    job_payload = job.scheduler_payload if isinstance(job.scheduler_payload, dict) else {}
    if isinstance(run_meta, dict):
        candidate = run_meta.get("submission")
        if isinstance(candidate, dict):
            submission_meta = candidate
    run_mode = str(submission_meta.get("run_mode") or "").strip()
    submission_structure_rel_path = _submission_structure_artifact_path(submission_meta)
    hpc_key = str(getattr(job.hpc_system, "name", "") or "").strip().lower()
    run_in_place = bool(submission_meta.get("run_in_place"))
    include_cpu_freq_req = bool(submission_meta.get("include_cpu_freq_req", True))
    if hpc_key == "trh408":
        run_in_place = True
        include_cpu_freq_req = False
    elif hpc_key == "falcon":
        include_cpu_freq_req = False

    submit_script = str(submission_meta.get("submit_script") or "submit_workflow.sh").strip()
    if not submit_script:
        submit_script = "submit_workflow.sh"
    wrapper_script = str(submission_meta.get("wrapper_script") or "run_workflow_submit.py").strip()
    if not wrapper_script:
        wrapper_script = "run_workflow_submit.py"
    previous_remote_workdir = str(job.remote_workdir or "").strip()
    previous_scheduler_job_id = str(job.scheduler_job_id or "").strip()
    previous_scheduler_job_ids = _collect_previous_scheduler_job_ids(
        submission_meta,
        previous_scheduler_job_id=previous_scheduler_job_id,
    )
    previous_job_folder_candidates: list[str] = []

    def _append_job_folder_candidate(path_value: str) -> None:
        candidate = str(path_value or "").strip()
        if not candidate or not candidate.startswith("/"):
            return
        if candidate not in previous_job_folder_candidates:
            previous_job_folder_candidates.append(candidate)

    if previous_scheduler_job_id:
        _append_job_folder_candidate(str(PurePosixPath(remote_run_root) / previous_scheduler_job_id))
    for scheduler_job_id in previous_scheduler_job_ids:
        scheduler_job_id = str(scheduler_job_id or "").strip()
        if not scheduler_job_id:
            continue
        _append_job_folder_candidate(str(PurePosixPath(remote_run_root) / scheduler_job_id))

    structure_xyz, structure_meta = _latest_restart_structure_xyz_for_run(job.workflow_run)
    structure_text = _structure_text_for_artifact_path(
        structure_xyz=structure_xyz or "",
        structure_meta=structure_meta if isinstance(structure_meta, dict) else {},
        artifact_rel_path=submission_structure_rel_path,
    )
    band_text, band_meta = (
        _latest_restart_band_extxyz_for_run(job.workflow_run)
        if run_mode == RunMode.ACTIVE_LEARNING_NEB
        else (None, {})
    )
    seeded_artifact_job_id = _leading_scheduler_job_id_from_artifact_path(structure_meta.get("artifact_path"))
    if seeded_artifact_job_id:
        _append_job_folder_candidate(str(PurePosixPath(remote_run_root) / seeded_artifact_job_id))
    _append_job_folder_candidate(previous_remote_workdir)

    client = OpenSSHClient(
        host=job.hpc_system.host,
        port=job.hpc_system.port,
        username=job.hpc_system.username,
        ssh_key_path=job.hpc_system.ssh_key_path,
        timeout=settings.REACTION_HUB_SSH_TIMEOUT_SECONDS,
        force_tty=settings.REACTION_HUB_SSH_FORCE_TTY,
        batch_mode=settings.REACTION_HUB_SSH_BATCH_MODE,
        use_clean_bash=settings.REACTION_HUB_SSH_CLEAN_BASH,
        bash_path=settings.REACTION_HUB_SSH_BASH_PATH,
    )
    try:
        ensure_cmd = (
            f"if [ ! -d {_q(remote_run_root)} ]; then echo 'Run directory missing' >&2; exit 41; fi"
        )
        client.exec_checked(ensure_cmd)

        previous_remote_workdir_exists: bool | None = None
        previous_job_folder_found = ""
        restart_submit_dir = remote_run_root
        restart_backup_dir = ""
        restarted_from_scratch = False
        if previous_job_folder_candidates:
            for candidate_path in previous_job_folder_candidates:
                if _remote_directory_exists(client, candidate_path):
                    previous_job_folder_found = candidate_path
                    break
            previous_remote_workdir_exists = bool(previous_job_folder_found)
            if previous_job_folder_found and previous_job_folder_found.startswith(f"{remote_run_root}/"):
                restart_submit_dir = previous_job_folder_found
            if not previous_remote_workdir_exists:
                restarted_from_scratch = True
                archive_state_cmd = (
                    f"run_dir={_q(remote_run_root)}; "
                    "stamp=$(date +%Y%m%dT%H%M%S); "
                    'backup="$run_dir/restart_backup_${stamp}"; '
                    "mkdir -p \"$backup\"; "
                    "moved=0; "
                    "for entry in DFT TDDFT TDDFT_active_learning_shared scan_plots "
                    "iroot_scan_endpoints.csv iroot_geometry_log.csv scan_endpoints.db "
                    "workflow_root.txt reaction_hub_summary.json reaction_hub_final.xyz; do "
                    'if [ -e "$run_dir/$entry" ]; then mv "$run_dir/$entry" "$backup/"; moved=1; fi; '
                    "done; "
                    'if [ "$moved" -eq 0 ]; then rmdir "$backup" >/dev/null 2>&1 || true; '
                    "printf ''; "
                    "else "
                    "printf '%s' \"$backup\"; "
                    "fi"
                )
                restart_backup_dir = client.exec_checked(archive_state_cmd).strip()

        submit_script_path = str(PurePosixPath(restart_submit_dir) / submit_script)
        wrapper_script_path = str(PurePosixPath(restart_submit_dir) / wrapper_script)
        structure_remote_path = str(PurePosixPath(restart_submit_dir) / submission_structure_rel_path)

        if run_mode in {RunMode.CHARGE_TRANSFER, RunMode.ACTIVE_LEARNING_NEB}:
            neb_config = dict(submission_meta.get("neb_config") or {})
            if neb_config.get("initial_band_path"):
                neb_config["initial_band_extxyz"] = "staged"
            wrapper_text = _build_wrapper_script(
                workflow_module=(
                    ""
                    if run_mode == RunMode.CHARGE_TRANSFER
                    else _module_path_from_code_path(job.workflow_run.workflow.code_path)
                ),
                run_mode=run_mode,
                hpc_name=job.hpc_system.name,
                n_iter=_safe_positive_int(submission_meta.get("n_iter"), default=1),
                use_tddft_optimisation=bool(submission_meta.get("use_tddft_optimisation")),
                fmax=float(submission_meta.get("fmax") or 0.01),
                active_learning_optimizer=str(submission_meta.get("active_learning_optimizer") or "bfgs"),
                qm_backend=str(submission_meta.get("qm_backend") or "orca"),
                qm_backend_settings=submission_meta.get("qm_backend_settings") or {},
                target_wavelength_nm=float(submission_meta.get("target_wavelength_nm") or 520.0),
                basis_keyword=str(submission_meta.get("basis_set") or "lanl2dz"),
                xc_functional=str(submission_meta.get("xc_functional") or "cam-b3lyp"),
                solvent=str(submission_meta.get("solvent") or "") or None,
                spin_multiplicity=submission_meta.get("spin_multiplicity"),
                system_charge=int(submission_meta.get("system_charge") or 0),
                nodes=_safe_positive_int(submission_meta.get("nodes"), default=1),
                tasks_per_node=_safe_positive_int(submission_meta.get("tasks_per_node"), default=1),
                cpus_per_task=_safe_positive_int(submission_meta.get("cpus_per_task"), default=1),
                al_steps_per_iter=_safe_positive_int(submission_meta.get("al_steps_per_iter"), default=200),
                max_al_steps_per_iter=submission_meta.get("max_al_steps_per_iter"),
                scan_config=submission_meta.get("scan_config") or {},
                single_constraints_config=submission_meta.get("single_constraints_config") or {},
                charge_transfer_config=submission_meta.get("charge_transfer_config") or {},
                neb_config=neb_config,
                coadsorbate_config=submission_meta.get("coadsorbate_config") or {},
                alloy_config=submission_meta.get("alloy_config") or {},
            )
            if hasattr(client, "upload_text"):
                client.upload_text(wrapper_script_path, wrapper_text)
            else:
                client.exec_checked(f"printf %s {_q(wrapper_text)} > {_q(wrapper_script_path)}")
            client.exec_checked(
                f"python3 -m py_compile {_q(wrapper_script_path)}"
            )
            if run_mode == RunMode.ACTIVE_LEARNING_NEB:
                neb_wf_local = settings.BASE_DIR.parent / "cats" / "working_directory" / "active_learning_neb_workflow.py"
                if neb_wf_local.is_file():
                    remote_neb_wf = str(PurePosixPath(restart_submit_dir) / "active_learning_neb_workflow.py")
                    neb_wf_text = neb_wf_local.read_text(encoding="utf-8")
                    if hasattr(client, "upload_text"):
                        client.upload_text(remote_neb_wf, neb_wf_text)
                    else:
                        client.exec_checked(f"printf %s {_q(neb_wf_text)} > {_q(remote_neb_wf)}")
                remote_vib_dir = str(PurePosixPath(restart_submit_dir) / "neb" / "vibrations" / "vib")
                client.exec_checked(f"if [ -d {_q(remote_vib_dir)} ]; then timeout 5 find {_q(remote_vib_dir)} -maxdepth 1 -type f -name 'cache.*.json' -size 0 -delete; fi")

        if structure_text:
            if hasattr(client, "upload_text"):
                client.upload_text(structure_remote_path, structure_text)
            else:
                client.exec_checked(f"printf %s {_q(structure_text)} > {_q(structure_remote_path)}")
        if band_text:
            band_remote_path = str(PurePosixPath(restart_submit_dir) / "input_band.extxyz")
            if hasattr(client, "upload_text"):
                client.upload_text(band_remote_path, band_text)
            else:
                client.exec_checked(f"printf %s {_q(band_text)} > {_q(band_remote_path)}")
        elif run_mode == RunMode.ACTIVE_LEARNING_NEB:
            band_target = str(PurePosixPath(restart_submit_dir) / "input_band.extxyz")
            ensure_band_cmd = (
                f"if [ ! -f {_q(band_target)} ] && [ -f {_q(str(PurePosixPath(remote_run_root) / 'input_band.extxyz'))} ]; then "
                f"cp {_q(str(PurePosixPath(remote_run_root) / 'input_band.extxyz'))} {_q(band_target)}; "
                f"elif [ ! -f {_q(band_target)} ] && [ -n {_q(restart_backup_dir)} ] && [ -f {_q(str(PurePosixPath(restart_backup_dir) / 'input_band.extxyz'))} ]; then "
                f"cp {_q(str(PurePosixPath(restart_backup_dir) / 'input_band.extxyz'))} {_q(band_target)}; fi"
            )
            client.exec_checked(ensure_band_cmd)

        refreshed_submit_script = False
        try:
            module_loads = _safe_string_list(submission_meta.get("module_loads"))
            neb_backend = str((submission_meta.get("neb_config") or {}).get("backend") or "").lower()
            if run_mode == RunMode.ACTIVE_LEARNING_NEB and neb_backend in {"dft", "tddft_mlneb"}:
                gpu_modules = {
                    "prgenv-cray",
                    "rocm",
                    "craype-accel-amd-gfx90a",
                    "craype-x86-milan",
                }
                module_loads = [
                    module for module in module_loads if module.strip().lower() not in gpu_modules
                ]
            script_kwargs: dict[str, object] = {
                "run_python_filename": (
                    str(submission_meta.get("wrapper_script") or "run_workflow_submit.py").strip()
                    or "run_workflow_submit.py"
                ),
                "nodes": _safe_positive_int(submission_meta.get("nodes"), default=1),
                "tasks_per_node": _safe_positive_int(submission_meta.get("tasks_per_node"), default=1),
                "cpus_per_task": _safe_positive_int(submission_meta.get("cpus_per_task"), default=1),
                "gpus": _safe_positive_int(submission_meta.get("gpus"), default=0),
                "use_srun": bool(submission_meta.get("use_srun")),
                "srun_ntasks": _safe_positive_int(submission_meta.get("srun_ntasks"), default=1),
                "srun_gpus_per_task": _safe_positive_int(
                    submission_meta.get("srun_gpus_per_task"), default=0
                ),
                "walltime": (
                    str(
                        submission_meta.get("walltime")
                        or getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_WALLTIME", "24:00:00")
                    ).strip()
                    or "24:00:00"
                ),
                "account": str(submission_meta.get("account") or "").strip(),
                "partition": (
                    str(
                        submission_meta.get("partition")
                        or getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_PARTITION", "standard")
                    ).strip()
                    or "standard"
                ),
                "qos": (
                    ""
                    if hpc_key == "falcon"
                    else (
                        str(
                            submission_meta.get("qos")
                            or job_payload.get("qos")
                            or getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_QOS", "standard")
                        ).strip()
                        or "standard"
                    )
                ),
                "reservation": (
                    ""
                    if hpc_key == "falcon" and not (submission_meta.get("reservation") or job_payload.get("reservation"))
                    else str(
                        submission_meta.get("reservation") or job_payload.get("reservation") or ""
                    ).strip()
                ),
                "slurm_memory": str(
                    submission_meta.get("slurm_memory")
                    or job_payload.get("slurm_memory")
                    or ""
                ).strip(),
                "module_loads": module_loads,
                "venv_activate_path": str(submission_meta.get("venv_activate_path") or "").strip(),
                "pythonpath_entries": _safe_string_list(submission_meta.get("pythonpath_entries")),
                "scratch_base": (
                    str(
                        submission_meta.get("scratch_base")
                        or getattr(settings, "REACTION_HUB_SUBMIT_SCRATCH_BASE", "")
                    ).strip()
                ),
                "include_cpu_freq_req": include_cpu_freq_req,
                "previous_scheduler_job_ids": previous_scheduler_job_ids,
                "run_mode": run_mode,
                "run_in_place": run_in_place,
            }
            if hpc_key == "trh408":
                script_kwargs["job_name"] = (
                    str(submission_meta.get("job_name") or "testgpu").strip() or "testgpu"
                )
                if run_mode == RunMode.COADSORBATE_PLACEMENT:
                    script_kwargs.update(
                        {
                            "include_cpu_layout_directives": False,
                            "include_tasks_per_node": False,
                            "include_cpus_per_task": True,
                            "gpu_directive": "gres",
                            "exclusive": True,
                            "use_srun": False,
                            "srun_ntasks": 1,
                            "srun_gpus_per_task": 0,
                        }
                    )

            submission_script = _build_submission_script(**script_kwargs)
            if hasattr(client, "upload_text"):
                client.upload_text(submit_script_path, submission_script)
            else:
                client.exec_checked(f"printf %s {_q(submission_script)} > {_q(submit_script_path)}")
            client.exec_checked(f"chmod 755 {_q(submit_script_path)}")
            refreshed_submit_script = True
        except Exception:
            refreshed_submit_script = False

        queue_qos = str(submission_meta.get("qos") or "").strip()
        queue_limit = _submission_queue_limit(queue_qos)
        while queue_limit and _active_qos_job_count(
            client, username=job.hpc_system.username, qos=queue_qos
        ) >= queue_limit:
            if not wait_for_queue_capacity:
                raise _QueueCapacityUnavailable(queue_qos)
            time.sleep(
                max(
                    1.0,
                    float(getattr(settings, "REACTION_HUB_SHORT_QUEUE_POLL_SECONDS", 30.0)),
                )
            )

        export_flag_parts = ["ALL"]
        if include_cpu_freq_req:
            export_flag_parts.append("SLURM_CPU_FREQ_REQ=2250000")
        if previous_scheduler_job_ids:
            export_flag_parts.append(
                f"REACTION_HUB_RSYNC_EXCLUDE_JOB_IDS={_q(':'.join(previous_scheduler_job_ids))}"
            )
        export_flags = ",".join(export_flag_parts)
        submit_cmd = f"cd {_q(restart_submit_dir)} && sbatch --export={export_flags} {_q(submit_script)}"
        submit_export_fallback = False
        try:
            submit_out = client.exec_checked(submit_cmd)
        except Exception:
            if previous_scheduler_job_ids:
                fallback_export = "ALL,SLURM_CPU_FREQ_REQ=2250000" if include_cpu_freq_req else "ALL"
                fallback_cmd = (
                    f"cd {_q(restart_submit_dir)} && sbatch --export={fallback_export} {_q(submit_script)}"
                )
                submit_out = client.exec_checked(fallback_cmd)
                submit_export_fallback = True
            else:
                raise
    finally:
        client.close()

    match = _SBATCH_JOB_ID_RE.search(str(submit_out))
    if not match:
        raise RuntimeError(f"Unable to parse scheduler job id from sbatch output: {str(submit_out).strip()}")
    new_scheduler_job_id = match.group(1)

    scratch_base = str(submission_meta.get("scratch_base") or "").strip()
    remote_workdir = ""
    if run_in_place:
        remote_workdir = restart_submit_dir
    elif scratch_base:
        remote_workdir = _resolve_remote_workdir(
            scratch_base=scratch_base,
            username=job.hpc_system.username,
            scheduler_job_id=new_scheduler_job_id,
        )

    restart_meta: dict[str, object] = {
        "submit_script": submit_script,
        "remote_run_root": remote_run_root,
        "remote_run_dir": restart_submit_dir,
        "used_latest_structure": bool(structure_xyz),
        "used_latest_band": bool(band_text),
        "run_in_place": run_in_place,
        "remote_workdir": remote_workdir,
        "previous_remote_workdir": previous_remote_workdir,
        "previous_remote_workdir_exists": previous_remote_workdir_exists,
        "previous_job_folder_candidates": previous_job_folder_candidates,
        "previous_job_folder_found": previous_job_folder_found,
        "previous_scheduler_job_ids": previous_scheduler_job_ids,
        "submit_script_refreshed": refreshed_submit_script,
        "submit_export_fallback": submit_export_fallback,
        "restarted_from_scratch": restarted_from_scratch,
        "restart_backup_dir": restart_backup_dir,
        "submission_structure_path": submission_structure_rel_path,
    }
    restart_meta.update(structure_meta)
    restart_meta.update(band_meta)
    return new_scheduler_job_id, restart_submit_dir, restart_meta


def _resubmit_completed_scan_for_smoothing(job: JobRecord) -> tuple[str, str, dict[str, object]]:
    if job.hpc_system is None:
        raise RuntimeError("Job has no HPC system configured.")

    run = job.workflow_run
    run_meta = run.metadata if isinstance(run.metadata, dict) else {}
    submission_meta = run_meta.get("submission") if isinstance(run_meta.get("submission"), dict) else {}
    run_mode = str(submission_meta.get("run_mode") or "").strip()
    submission_structure_rel_path = _submission_structure_artifact_path(submission_meta)
    if run_mode != RunMode.TDDFT_BOND_SCAN:
        raise RuntimeError("Smooth resubmit is only supported for TDDFT/DFT bond-scan runs.")

    submission_root = str(submission_meta.get("remote_runs_root") or "").strip()
    remote_run_root_candidates: list[str] = []
    for candidate in (
        canonical_run_path(run.working_directory),
        str(PurePosixPath(submission_root) / run.run_name) if submission_root else "",
    ):
        candidate_text = canonical_run_path(candidate)
        if candidate_text and candidate_text.startswith("/") and candidate_text not in remote_run_root_candidates:
            remote_run_root_candidates.append(candidate_text)

    if not remote_run_root_candidates:
        raise RuntimeError("Workflow run directory is not a valid remote POSIX path.")

    hpc_key = str(getattr(job.hpc_system, "name", "") or "").strip().lower()
    submit_script = str(submission_meta.get("submit_script") or "submit_workflow.sh").strip() or "submit_workflow.sh"
    wrapper_script = str(submission_meta.get("wrapper_script") or "run_workflow_submit.py").strip() or "run_workflow_submit.py"
    previous_scheduler_job_id = str(job.scheduler_job_id or "").strip()
    previous_scheduler_job_ids = _collect_previous_scheduler_job_ids(
        submission_meta,
        previous_scheduler_job_id=previous_scheduler_job_id,
    )
    previous_job_folder_candidates: list[str] = []

    def _append_job_folder_candidate(path_value: str) -> None:
        candidate = str(path_value or "").strip()
        if not candidate or not candidate.startswith("/"):
            return
        if candidate not in previous_job_folder_candidates:
            previous_job_folder_candidates.append(candidate)

    for remote_run_root in remote_run_root_candidates:
        for scheduler_job_id in previous_scheduler_job_ids:
            scheduler_job_id = str(scheduler_job_id or "").strip()
            if not scheduler_job_id:
                continue
            _append_job_folder_candidate(str(PurePosixPath(remote_run_root) / scheduler_job_id))
        _append_job_folder_candidate(remote_run_root)

    structure_xyz, structure_meta = _latest_smooth_scan_structure_xyz_for_run(run)
    structure_text = _structure_text_for_artifact_path(
        structure_xyz=structure_xyz or "",
        structure_meta=structure_meta if isinstance(structure_meta, dict) else {},
        artifact_rel_path=submission_structure_rel_path,
    )
    seeded_artifact_job_id = _leading_scheduler_job_id_from_artifact_path(structure_meta.get("artifact_path"))
    if seeded_artifact_job_id:
        for remote_run_root in remote_run_root_candidates:
            _append_job_folder_candidate(str(PurePosixPath(remote_run_root) / seeded_artifact_job_id))

    client = OpenSSHClient(
        host=job.hpc_system.host,
        port=job.hpc_system.port,
        username=job.hpc_system.username,
        ssh_key_path=job.hpc_system.ssh_key_path,
        timeout=settings.REACTION_HUB_SSH_TIMEOUT_SECONDS,
        force_tty=settings.REACTION_HUB_SSH_FORCE_TTY,
        batch_mode=settings.REACTION_HUB_SSH_BATCH_MODE,
        use_clean_bash=settings.REACTION_HUB_SSH_CLEAN_BASH,
        bash_path=settings.REACTION_HUB_SSH_BASH_PATH,
    )
    try:
        previous_job_folder_found = ""
        for candidate_path in previous_job_folder_candidates:
            if _remote_directory_exists(client, candidate_path):
                previous_job_folder_found = candidate_path
                break
        if not previous_job_folder_found:
            joined_candidates = ", ".join(previous_job_folder_candidates[:10]) or "(none)"
            raise RuntimeError(
                "Unable to locate previous SLURM job-id folder for smooth resubmit. "
                f"Checked: {joined_candidates}"
            )

        restart_submit_dir = previous_job_folder_found
        wrapper_script_path = str(PurePosixPath(restart_submit_dir) / wrapper_script)
        submit_script_path = str(PurePosixPath(restart_submit_dir) / submit_script)
        structure_remote_path = str(PurePosixPath(restart_submit_dir) / submission_structure_rel_path)

        if structure_text:
            if hasattr(client, "upload_text"):
                client.upload_text(structure_remote_path, structure_text)
            else:
                client.exec_checked(f"printf %s {_q(structure_text)} > {_q(structure_remote_path)}")

        # Force wrapper smoothing on for both new and legacy wrapper variants.
        patch_wrapper_cmd = (
            f"wrapper={_q(wrapper_script_path)}; "
            "if grep -q 'enable_ts_smoothing=' \"$wrapper\"; then "
            "sed -i "
            "'s/enable_ts_smoothing[[:space:]]*=[[:space:]]*False/enable_ts_smoothing=True/g; "
            "s/enable_ts_smoothing[[:space:]]*=[[:space:]]*True/enable_ts_smoothing=True/g' "
            "\"$wrapper\"; "
            "elif grep -q 'BondScanSettings(' \"$wrapper\"; then "
            "sed -i '/BondScanSettings(/a\\                enable_ts_smoothing=True,' \"$wrapper\"; "
            "else "
            "echo 'Missing BondScanSettings in wrapper script' >&2; "
            "exit 47; "
            "fi"
        )
        client.exec_checked(patch_wrapper_cmd)

        submission_script = _build_submission_script(
            run_python_filename=wrapper_script,
            nodes=_safe_positive_int(submission_meta.get("nodes"), default=1),
            tasks_per_node=_safe_positive_int(submission_meta.get("tasks_per_node"), default=1),
            cpus_per_task=_safe_positive_int(submission_meta.get("cpus_per_task"), default=1),
            gpus=_safe_positive_int(submission_meta.get("gpus"), default=0),
            use_srun=bool(submission_meta.get("use_srun")),
            srun_ntasks=_safe_positive_int(submission_meta.get("srun_ntasks"), default=1),
            srun_gpus_per_task=_safe_positive_int(submission_meta.get("srun_gpus_per_task"), default=0),
            walltime=str(
                submission_meta.get("walltime")
                or getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_WALLTIME", "24:00:00")
            ).strip()
            or "24:00:00",
            account=str(submission_meta.get("account") or "").strip(),
            partition=str(
                submission_meta.get("partition")
                or getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_PARTITION", "standard")
            ).strip()
            or "standard",
            qos=(
                ""
                if hpc_key == "falcon"
                else (
                    str(
                        submission_meta.get("qos")
                        or getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_QOS", "standard")
                    ).strip()
                    or "standard"
                )
            ),
            module_loads=_safe_string_list(submission_meta.get("module_loads")),
            venv_activate_path=str(submission_meta.get("venv_activate_path") or "").strip(),
            pythonpath_entries=_safe_string_list(submission_meta.get("pythonpath_entries")),
            scratch_base=str(
                submission_meta.get("scratch_base")
                or getattr(settings, "REACTION_HUB_SUBMIT_SCRATCH_BASE", "")
            ).strip(),
            previous_scheduler_job_ids=previous_scheduler_job_ids,
            run_mode=run_mode,
        )
        if hasattr(client, "upload_text"):
            client.upload_text(submit_script_path, submission_script)
        else:
            client.exec_checked(f"printf %s {_q(submission_script)} > {_q(submit_script_path)}")
        client.exec_checked(f"chmod 755 {_q(submit_script_path)}")

        export_flags = "ALL" if hpc_key == "falcon" else "ALL,SLURM_CPU_FREQ_REQ=2250000"
        if previous_scheduler_job_ids:
            export_flags = (
                f"{export_flags},REACTION_HUB_RSYNC_EXCLUDE_JOB_IDS="
                f"{_q(':'.join(previous_scheduler_job_ids))}"
            )
        submit_out = client.exec_checked(
            f"cd {_q(restart_submit_dir)} && sbatch --export={export_flags} {_q(submit_script)}"
        )
    finally:
        client.close()

    match = _SBATCH_JOB_ID_RE.search(str(submit_out))
    if not match:
        raise RuntimeError(f"Unable to parse scheduler job id from sbatch output: {str(submit_out).strip()}")
    new_scheduler_job_id = match.group(1)

    scratch_base = str(submission_meta.get("scratch_base") or "").strip()
    remote_workdir = ""
    if scratch_base:
        remote_workdir = _resolve_remote_workdir(
            scratch_base=scratch_base,
            username=job.hpc_system.username,
            scheduler_job_id=new_scheduler_job_id,
        )

    smooth_meta: dict[str, object] = {
        "submit_script": submit_script,
        "wrapper_script": wrapper_script,
        "remote_run_dir": restart_submit_dir,
        "remote_run_root_candidates": remote_run_root_candidates,
        "previous_job_folder_candidates": previous_job_folder_candidates,
        "previous_job_folder_found": previous_job_folder_found,
        "previous_scheduler_job_ids": previous_scheduler_job_ids,
        "used_latest_structure": bool(structure_xyz),
        "remote_workdir": remote_workdir,
        "enable_ts_smoothing": True,
        "submission_structure_path": submission_structure_rel_path,
    }
    smooth_meta.update(structure_meta)
    return new_scheduler_job_id, restart_submit_dir, smooth_meta


def _resubmit_completed_job_for_replot(job: JobRecord) -> tuple[str, str, dict[str, object]]:
    if job.hpc_system is None:
        raise RuntimeError("Job has no HPC system configured.")

    run = job.workflow_run
    run_mode = _run_mode_for_job_action(run)
    if not _supports_replot_resubmit_for_run(run):
        raise RuntimeError("Replot resubmit is only supported for completed optimisation/bond-scan runs.")

    run_meta = run.metadata if isinstance(run.metadata, dict) else {}
    submission_meta = run_meta.get("submission") if isinstance(run_meta.get("submission"), dict) else {}
    submission_structure_rel_path = _submission_structure_artifact_path(submission_meta)
    submission_root = str(submission_meta.get("remote_runs_root") or "").strip()
    remote_run_root_candidates: list[str] = []
    for candidate in (
        canonical_run_path(run.working_directory),
        str(PurePosixPath(submission_root) / run.run_name) if submission_root else "",
    ):
        candidate_text = canonical_run_path(candidate)
        if candidate_text and candidate_text.startswith("/") and candidate_text not in remote_run_root_candidates:
            remote_run_root_candidates.append(candidate_text)

    if not remote_run_root_candidates:
        raise RuntimeError("Workflow run directory is not a valid remote POSIX path.")

    submit_script = str(submission_meta.get("submit_script") or "submit_workflow.sh").strip() or "submit_workflow.sh"
    wrapper_script = str(submission_meta.get("wrapper_script") or "run_workflow_submit.py").strip() or "run_workflow_submit.py"
    previous_scheduler_job_id = str(job.scheduler_job_id or "").strip()
    previous_scheduler_job_ids = _collect_previous_scheduler_job_ids(
        submission_meta,
        previous_scheduler_job_id=previous_scheduler_job_id,
    )
    previous_job_folder_candidates: list[str] = []

    def _append_job_folder_candidate(path_value: str) -> None:
        candidate = str(path_value or "").strip()
        if not candidate or not candidate.startswith("/"):
            return
        if candidate not in previous_job_folder_candidates:
            previous_job_folder_candidates.append(candidate)

    for remote_run_root in remote_run_root_candidates:
        for scheduler_job_id in previous_scheduler_job_ids:
            scheduler_job_id = str(scheduler_job_id or "").strip()
            if not scheduler_job_id:
                continue
            _append_job_folder_candidate(str(PurePosixPath(remote_run_root) / scheduler_job_id))
        _append_job_folder_candidate(remote_run_root)

    if run_mode == RunMode.TDDFT_BOND_SCAN:
        structure_xyz, structure_meta = _latest_smooth_scan_structure_xyz_for_run(run)
    else:
        structure_xyz, structure_meta = _latest_restart_structure_xyz_for_run(run)
    structure_text = _structure_text_for_artifact_path(
        structure_xyz=structure_xyz or "",
        structure_meta=structure_meta if isinstance(structure_meta, dict) else {},
        artifact_rel_path=submission_structure_rel_path,
    )

    seeded_artifact_job_id = _leading_scheduler_job_id_from_artifact_path(structure_meta.get("artifact_path"))
    if seeded_artifact_job_id:
        for remote_run_root in remote_run_root_candidates:
            _append_job_folder_candidate(str(PurePosixPath(remote_run_root) / seeded_artifact_job_id))

    client = OpenSSHClient(
        host=job.hpc_system.host,
        port=job.hpc_system.port,
        username=job.hpc_system.username,
        ssh_key_path=job.hpc_system.ssh_key_path,
        timeout=settings.REACTION_HUB_SSH_TIMEOUT_SECONDS,
        force_tty=settings.REACTION_HUB_SSH_FORCE_TTY,
        batch_mode=settings.REACTION_HUB_SSH_BATCH_MODE,
        use_clean_bash=settings.REACTION_HUB_SSH_CLEAN_BASH,
        bash_path=settings.REACTION_HUB_SSH_BASH_PATH,
    )
    try:
        previous_job_folder_found = ""
        for candidate_path in previous_job_folder_candidates:
            if _remote_directory_exists(client, candidate_path):
                previous_job_folder_found = candidate_path
                break
        if not previous_job_folder_found:
            joined_candidates = ", ".join(previous_job_folder_candidates[:10]) or "(none)"
            raise RuntimeError(
                "Unable to locate previous SLURM job-id folder for replot resubmit. "
                f"Checked: {joined_candidates}"
            )

        restart_submit_dir = previous_job_folder_found
        wrapper_script_path = str(PurePosixPath(restart_submit_dir) / wrapper_script)
        submit_script_path = str(PurePosixPath(restart_submit_dir) / submit_script)
        structure_remote_path = str(PurePosixPath(restart_submit_dir) / submission_structure_rel_path)

        if structure_text:
            if hasattr(client, "upload_text"):
                client.upload_text(structure_remote_path, structure_text)
            else:
                client.exec_checked(f"printf %s {_q(structure_text)} > {_q(structure_remote_path)}")

        if run_mode == RunMode.TDDFT_BOND_SCAN:
            # Replot must never trigger smoothing postprocess; smooth-resubmit may
            # have modified the wrapper in-place for a prior run.
            patch_wrapper_cmd = (
                f"wrapper={_q(wrapper_script_path)}; "
                "if grep -q 'enable_ts_smoothing=' \"$wrapper\"; then "
                "sed -i "
                "'s/enable_ts_smoothing[[:space:]]*=[[:space:]]*True/enable_ts_smoothing=False/g; "
                "s/enable_ts_smoothing[[:space:]]*=[[:space:]]*False/enable_ts_smoothing=False/g' "
                "\"$wrapper\"; "
                "elif grep -q 'BondScanSettings(' \"$wrapper\"; then "
                "sed -i '/BondScanSettings(/a\\                enable_ts_smoothing=False,' \"$wrapper\"; "
                "else "
                "echo 'Missing BondScanSettings in wrapper script' >&2; "
                "exit 47; "
                "fi"
            )
            client.exec_checked(patch_wrapper_cmd)

        submission_script = _build_submission_script(
            run_python_filename=wrapper_script,
            nodes=_safe_positive_int(submission_meta.get("nodes"), default=1),
            tasks_per_node=_safe_positive_int(submission_meta.get("tasks_per_node"), default=1),
            cpus_per_task=_safe_positive_int(submission_meta.get("cpus_per_task"), default=1),
            gpus=_safe_positive_int(submission_meta.get("gpus"), default=0),
            use_srun=bool(submission_meta.get("use_srun")),
            srun_ntasks=_safe_positive_int(submission_meta.get("srun_ntasks"), default=1),
            srun_gpus_per_task=_safe_positive_int(submission_meta.get("srun_gpus_per_task"), default=0),
            walltime=_REPLOT_RESUBMIT_WALLTIME,
            account=str(submission_meta.get("account") or "").strip(),
            partition=_REPLOT_RESUBMIT_PARTITION,
            qos=_REPLOT_RESUBMIT_QOS,
            module_loads=_safe_string_list(submission_meta.get("module_loads")),
            venv_activate_path=str(submission_meta.get("venv_activate_path") or "").strip(),
            pythonpath_entries=_safe_string_list(submission_meta.get("pythonpath_entries")),
            scratch_base=str(
                submission_meta.get("scratch_base")
                or getattr(settings, "REACTION_HUB_SUBMIT_SCRATCH_BASE", "")
            ).strip(),
            previous_scheduler_job_ids=previous_scheduler_job_ids,
            run_mode=run_mode,
            run_in_place=True,
        )
        if hasattr(client, "upload_text"):
            client.upload_text(submit_script_path, submission_script)
        else:
            client.exec_checked(f"printf %s {_q(submission_script)} > {_q(submit_script_path)}")
        client.exec_checked(f"chmod 755 {_q(submit_script_path)}")

        export_flags = "ALL,SLURM_CPU_FREQ_REQ=2250000,REACTION_HUB_REPLOT_ONLY=1"
        if previous_scheduler_job_ids:
            export_flags = (
                f"{export_flags},REACTION_HUB_RSYNC_EXCLUDE_JOB_IDS="
                f"{_q(':'.join(previous_scheduler_job_ids))}"
            )
        submit_out = client.exec_checked(
            f"cd {_q(restart_submit_dir)} && sbatch --export={export_flags} {_q(submit_script)}"
        )
    finally:
        client.close()

    match = _SBATCH_JOB_ID_RE.search(str(submit_out))
    if not match:
        raise RuntimeError(f"Unable to parse scheduler job id from sbatch output: {str(submit_out).strip()}")
    new_scheduler_job_id = match.group(1)

    scratch_base = str(submission_meta.get("scratch_base") or "").strip()
    remote_workdir = ""
    if scratch_base:
        remote_workdir = _resolve_remote_workdir(
            scratch_base=scratch_base,
            username=job.hpc_system.username,
            scheduler_job_id=new_scheduler_job_id,
        )

    replot_meta: dict[str, object] = {
        "submit_script": submit_script,
        "wrapper_script": wrapper_script,
        "remote_run_dir": restart_submit_dir,
        "remote_run_root_candidates": remote_run_root_candidates,
        "previous_job_folder_candidates": previous_job_folder_candidates,
        "previous_job_folder_found": previous_job_folder_found,
        "previous_scheduler_job_ids": previous_scheduler_job_ids,
        "used_latest_structure": bool(structure_xyz),
        "remote_workdir": remote_workdir,
        "run_mode": run_mode,
        "replot_only": True,
        "walltime": _REPLOT_RESUBMIT_WALLTIME,
        "partition": _REPLOT_RESUBMIT_PARTITION,
        "qos": _REPLOT_RESUBMIT_QOS,
        "enable_ts_smoothing": False,
        "run_in_place": True,
        "submission_structure_path": submission_structure_rel_path,
    }
    replot_meta.update(structure_meta)
    return new_scheduler_job_id, restart_submit_dir, replot_meta


@require_POST
def job_scancel_delete_view(request, pk: int):
    redirect_to = _safe_next_url(request)
    job = (
        JobRecord.objects.select_related("hpc_system", "workflow_run")
        .filter(pk=pk)
        .first()
    )
    if job is None:
        messages.error(request, "Job was not found.")
        return HttpResponseRedirect(redirect_to)

    if job.status not in {JobStatus.INITIATED, JobStatus.PENDING, JobStatus.RUNNING}:
        messages.error(
            request,
            f"Only initiated, pending, or running jobs can use Scancel + Delete (current status: {job.status}).",
        )
        return HttpResponseRedirect(redirect_to)

    try:
        cancel_state, renamed_run_dir = _scancel_and_rename_remote_run(job)
    except Exception as exc:
        messages.error(request, f"Failed to mark job for deletion: {exc}")
        return HttpResponseRedirect(redirect_to)

    payload = {
        "requested_by": request.user.get_username() if request.user.is_authenticated else "",
        "action": "scancel_and_mark_deleted",
        "scheduler_job_id": job.scheduler_job_id,
        "remote_run_dir_before": canonical_run_path(job.workflow_run.working_directory),
        "remote_run_dir_after": renamed_run_dir,
        "scancel_state": cancel_state,
    }
    job.transition_to(
        JobStatus.DELETED,
        message="User requested Scancel + Delete from /jobs/.",
        payload=payload,
    )
    reason = f"Marked for deletion by user; remote run directory renamed to {renamed_run_dir}."
    if job.failure_reason != reason:
        job.failure_reason = reason
        job.save(update_fields=["failure_reason", "updated_at"])

    cancel_note = "Scheduler job cancelled." if cancel_state == "cancelled" else "Scheduler job was already absent."
    messages.success(
        request,
        f"{cancel_note} Remote run directory renamed to {renamed_run_dir}. Job marked for deletion.",
    )
    return HttpResponseRedirect(redirect_to)


@require_POST
def job_mark_delete_view(request, pk: int):
    redirect_to = _safe_next_url(request)
    job = JobRecord.objects.filter(pk=pk).first()
    if job is None:
        messages.error(request, "Job was not found.")
    elif job.status == JobStatus.DELETED:
        messages.info(request, "Job is already marked for deletion.")
    elif job.status in {JobStatus.INITIATED, JobStatus.PENDING, JobStatus.RUNNING}:
        messages.error(request, "Active jobs must use Scancel + Delete.")
    else:
        job.transition_to(
            JobStatus.DELETED,
            message="User marked job for deletion from the workflow browser.",
            payload={
                "requested_by": request.user.get_username() if request.user.is_authenticated else "",
                "action": "mark_for_deletion",
            },
        )
        messages.success(request, f"Job {job.scheduler_job_id} marked for deletion.")
    return HttpResponseRedirect(redirect_to)


@require_POST
def job_mark_failed_delete_view(request, pk: int):
    redirect_to = _safe_next_url(request)
    job = (
        JobRecord.objects.select_related("hpc_system", "workflow_run")
        .filter(pk=pk)
        .first()
    )
    if job is None:
        messages.error(request, "Job was not found.")
        return HttpResponseRedirect(redirect_to)

    if job.status != JobStatus.FAILED:
        messages.error(
            request,
            f"Only failed jobs can be marked for deletion (current status: {job.status}).",
        )
        return HttpResponseRedirect(redirect_to)

    try:
        renamed_run_dir = _mark_failed_job_for_deletion(
            job,
            requested_by=request.user.get_username() if request.user.is_authenticated else "",
        )
    except Exception as exc:
        messages.error(request, f"Failed to mark failed job for deletion: {exc}")
        return HttpResponseRedirect(redirect_to)

    messages.success(
        request,
        f"Remote run directory renamed to {renamed_run_dir}. Failed job marked for deletion.",
    )
    return HttpResponseRedirect(redirect_to)


def _mark_failed_job_for_deletion(job: JobRecord, *, requested_by: str = "") -> str:
    renamed_run_dir = _rename_remote_run_for_deletion(job)
    payload = {
        "requested_by": requested_by,
        "action": "mark_failed_for_deletion",
        "scheduler_job_id": job.scheduler_job_id,
        "remote_run_dir_before": canonical_run_path(job.workflow_run.working_directory),
        "remote_run_dir_after": renamed_run_dir,
        "scancel_state": "not_requested",
    }
    job.transition_to(
        JobStatus.DELETED,
        message="User requested mark-failed-for-deletion from /jobs/.",
        payload=payload,
    )
    reason = f"Marked for deletion from failed state; remote run directory renamed to {renamed_run_dir}."
    if job.failure_reason != reason:
        job.failure_reason = reason
        job.save(update_fields=["failure_reason", "updated_at"])
    return renamed_run_dir


@require_POST
def job_bulk_failed_action_view(request):
    redirect_to = _safe_next_url(request)
    action = str(request.POST.get("bulk_action") or "").strip()
    job_ids = [int(pk) for pk in request.POST.getlist("job_ids") if str(pk).isdigit()]
    if not job_ids:
        messages.error(request, "Select at least one failed job.")
        return HttpResponseRedirect(redirect_to)

    jobs = JobRecord.objects.select_related("hpc_system", "workflow_run").filter(pk__in=job_ids).order_by("pk")
    requested_by = request.user.get_username() if request.user.is_authenticated else ""
    done = 0
    skipped = 0
    failed = 0
    for job in jobs:
        if job.status != JobStatus.FAILED:
            skipped += 1
            continue
        try:
            if action == "restart_failed":
                resubmit_failed_job_task.delay(int(job.pk), requested_by=requested_by, next_url=redirect_to)
            elif action == "mark_failed_delete":
                _mark_failed_job_for_deletion(job, requested_by=requested_by)
            else:
                messages.error(request, "Unknown bulk action.")
                return HttpResponseRedirect(redirect_to)
        except Exception as exc:
            failed += 1
            messages.error(request, f"{job.job_name or job.pk}: {exc}")
        else:
            done += 1

    label = "restart queued" if action == "restart_failed" else "marked for deletion"
    if done:
        messages.success(request, f"{done} failed job(s) {label}.")
    if skipped:
        messages.warning(request, f"{skipped} selected job(s) skipped because they are not failed.")
    if failed and not done:
        messages.error(request, "No selected jobs were updated.")
    return HttpResponseRedirect(redirect_to)


def _restart_job_in_place_response(
    request,
    *,
    pk: int,
    required_status: str,
    action_payload: str,
    user_action_text: str,
    action_verb: str,
    wait_for_queue_capacity: bool = True,
    raise_on_failure: bool = False,
):
    redirect_to = _safe_next_url(request)
    job = (
        JobRecord.objects.select_related("hpc_system", "workflow_run")
        .filter(pk=pk)
        .first()
    )
    if job is None:
        messages.error(request, "Job was not found.")
        return HttpResponseRedirect(redirect_to)

    if job.status != required_status:
        messages.error(
            request,
            f"Only {required_status} jobs can be {action_verb} in place (current status: {job.status}).",
        )
        return HttpResponseRedirect(redirect_to)

    previous_scheduler_job_id = str(job.scheduler_job_id or "").strip()
    try:
        new_scheduler_job_id, remote_run_dir, restart_meta = _restart_cancelled_job_in_place(
            job,
            wait_for_queue_capacity=wait_for_queue_capacity,
        )
    except _QueueCapacityUnavailable:
        raise
    except Exception as exc:
        if raise_on_failure:
            raise
        messages.error(request, f"Failed to {action_verb} {required_status} job: {exc}")
        return HttpResponseRedirect(redirect_to)

    now = timezone.now()
    previous_status = job.status
    requested_by = request.user.get_username() if request.user.is_authenticated else ""

    scheduler_payload = job.scheduler_payload if isinstance(job.scheduler_payload, dict) else {}
    scheduler_payload = dict(scheduler_payload)
    restart_history = scheduler_payload.get("restart_history")
    if not isinstance(restart_history, list):
        restart_history = []
    restart_entry = {
        "requested_by": requested_by,
        "requested_at": now.isoformat(),
        "previous_scheduler_job_id": previous_scheduler_job_id,
        "new_scheduler_job_id": new_scheduler_job_id,
        "remote_run_dir": remote_run_dir,
        "submit_script": str(restart_meta.get("submit_script") or "submit_workflow.sh"),
        "used_latest_structure": bool(restart_meta.get("used_latest_structure")),
        "used_latest_band": bool(restart_meta.get("used_latest_band")),
    }
    remote_run_root = str(restart_meta.get("remote_run_root") or "").strip()
    if remote_run_root:
        restart_entry["remote_run_root"] = remote_run_root
    previous_remote_workdir = str(restart_meta.get("previous_remote_workdir") or "").strip()
    if previous_remote_workdir:
        restart_entry["previous_remote_workdir"] = previous_remote_workdir
    previous_remote_workdir_exists = restart_meta.get("previous_remote_workdir_exists")
    if previous_remote_workdir_exists is not None:
        restart_entry["previous_remote_workdir_exists"] = bool(previous_remote_workdir_exists)
    previous_job_folder_candidates = restart_meta.get("previous_job_folder_candidates")
    if isinstance(previous_job_folder_candidates, list):
        normalized_candidates = [str(item).strip() for item in previous_job_folder_candidates if str(item).strip()]
        if normalized_candidates:
            restart_entry["previous_job_folder_candidates"] = normalized_candidates
    previous_job_folder_found = str(restart_meta.get("previous_job_folder_found") or "").strip()
    if previous_job_folder_found:
        restart_entry["previous_job_folder_found"] = previous_job_folder_found
    previous_scheduler_job_ids = restart_meta.get("previous_scheduler_job_ids")
    if isinstance(previous_scheduler_job_ids, list):
        normalized_prev_ids = [str(item).strip() for item in previous_scheduler_job_ids if str(item).strip()]
        if normalized_prev_ids:
            restart_entry["previous_scheduler_job_ids"] = normalized_prev_ids
    submit_script_refreshed = restart_meta.get("submit_script_refreshed")
    if submit_script_refreshed is not None:
        restart_entry["submit_script_refreshed"] = bool(submit_script_refreshed)
    submit_export_fallback = restart_meta.get("submit_export_fallback")
    if submit_export_fallback is not None:
        restart_entry["submit_export_fallback"] = bool(submit_export_fallback)
    restarted_from_scratch = bool(restart_meta.get("restarted_from_scratch"))
    if restarted_from_scratch:
        restart_entry["restarted_from_scratch"] = True
    restart_backup_dir = str(restart_meta.get("restart_backup_dir") or "").strip()
    if restart_backup_dir:
        restart_entry["restart_backup_dir"] = restart_backup_dir
    artifact_id = restart_meta.get("artifact_id")
    if artifact_id is not None:
        restart_entry["structure_artifact_id"] = artifact_id
    artifact_label = str(restart_meta.get("artifact_label") or "").strip()
    if artifact_label:
        restart_entry["structure_artifact_label"] = artifact_label
    artifact_path = str(restart_meta.get("artifact_path") or "").strip()
    if artifact_path:
        restart_entry["structure_artifact_path"] = artifact_path
    for key in (
        "band_artifact_id",
        "band_artifact_label",
        "band_artifact_path",
        "band_frame_count",
        "band_frame_selection",
    ):
        value = restart_meta.get(key)
        if value not in (None, ""):
            restart_entry[key] = value
    trajectory_frame_count = restart_meta.get("trajectory_frame_count")
    if trajectory_frame_count is not None:
        try:
            restart_entry["trajectory_frame_count"] = int(trajectory_frame_count)
        except (TypeError, ValueError):
            pass
    trajectory_frame_index = restart_meta.get("trajectory_frame_index")
    if trajectory_frame_index is not None:
        try:
            restart_entry["trajectory_frame_index"] = int(trajectory_frame_index)
        except (TypeError, ValueError):
            pass
    trajectory_frame_fmax = restart_meta.get("trajectory_frame_fmax")
    if trajectory_frame_fmax is not None:
        try:
            restart_entry["trajectory_frame_fmax"] = float(trajectory_frame_fmax)
        except (TypeError, ValueError):
            pass
    trajectory_reasonable_fmax = restart_meta.get("trajectory_reasonable_fmax")
    if trajectory_reasonable_fmax is not None:
        try:
            restart_entry["trajectory_reasonable_fmax"] = float(trajectory_reasonable_fmax)
        except (TypeError, ValueError):
            pass
    trajectory_frame_selection = str(restart_meta.get("trajectory_frame_selection") or "").strip()
    if trajectory_frame_selection:
        restart_entry["trajectory_frame_selection"] = trajectory_frame_selection
    warning_text = str(restart_meta.get("warning") or "").strip()
    if warning_text:
        restart_entry["structure_seed_warning"] = warning_text
    restart_history.append(restart_entry)
    scheduler_payload["restart_history"] = restart_history

    remote_workdir = str(restart_meta.get("remote_workdir") or "").strip()
    job.scheduler_job_id = new_scheduler_job_id
    job.status = JobStatus.INITIATED
    job.submitted_at = now
    job.started_at = None
    job.completed_at = None
    job.last_seen_at = now
    job.exit_code = None
    job.failure_reason = ""
    job.stdout_path = f"{remote_run_dir}/workflow.{new_scheduler_job_id}.log"
    job.stderr_path = ""
    job.remote_workdir = remote_workdir
    job.scheduler_payload = scheduler_payload
    update_fields = [
        "scheduler_job_id",
        "status",
        "submitted_at",
        "started_at",
        "completed_at",
        "last_seen_at",
        "exit_code",
        "failure_reason",
        "stdout_path",
        "stderr_path",
        "remote_workdir",
        "scheduler_payload",
        "updated_at",
    ]
    job.save(update_fields=update_fields)

    run = job.workflow_run
    run_metadata = dict(run.metadata) if isinstance(run.metadata, dict) else {}
    run_submission = run_metadata.get("submission")
    if not isinstance(run_submission, dict):
        run_submission = {}
    run_restart_history = run_submission.get("restart_history")
    if not isinstance(run_restart_history, list):
        run_restart_history = []
    run_restart_history.append(
        {
            "requested_by": requested_by,
            "requested_at": now.isoformat(),
            "job_record_id": int(job.pk),
            "job_status_before_restart": previous_status,
            "previous_scheduler_job_id": previous_scheduler_job_id,
            "new_scheduler_job_id": new_scheduler_job_id,
            "remote_run_dir": remote_run_dir,
        }
    )
    run_submission["restart_history"] = run_restart_history
    run_submission["previous_scheduler_job_ids"] = _collect_previous_scheduler_job_ids(
        run_submission,
        previous_scheduler_job_id=previous_scheduler_job_id,
    )
    run_metadata["submission"] = run_submission
    run.status = "initiated"
    run.started_at = None
    run.completed_at = None
    if remote_run_dir:
        run.working_directory = remote_run_dir
    run.metadata = run_metadata
    run.save(update_fields=["status", "started_at", "completed_at", "working_directory", "metadata", "updated_at"])

    event_payload = dict(restart_entry)
    event_payload["action"] = action_payload
    JobStatusEvent.objects.create(
        job=job,
        previous_status=previous_status,
        status=JobStatus.INITIATED,
        message=f"User {user_action_text} from /jobs/.",
        payload=event_payload,
    )

    scratch_note = (
        " Restarted from scratch because previous job-id folder was not found."
        if restarted_from_scratch
        else ""
    )
    if warning_text:
        messages.warning(
            request,
            (
                f"Job {action_verb} as scheduler id {new_scheduler_job_id}, "
                f"but latest structure seed failed: {warning_text}{scratch_note}"
            ),
        )
    else:
        seed_note = (
            " Seeded from latest available structure artifact."
            if bool(restart_meta.get("used_latest_structure"))
            else ""
        )
        messages.success(
            request,
            (
                f"Job {action_verb} in place from {remote_run_dir} as scheduler id "
                f"{new_scheduler_job_id}.{seed_note}{scratch_note}"
            ),
        )
    return HttpResponseRedirect(redirect_to)


@require_POST
def job_restart_cancelled_view(request, pk: int):
    redirect_to = _safe_next_url(request)
    try:
        result = restart_cancelled_job_task.delay(
            int(pk),
            requested_by=request.user.get_username() if request.user.is_authenticated else "",
            next_url=redirect_to,
        )
    except Exception as exc:
        messages.error(request, f"Failed to queue cancelled-job restart: {exc}")
    else:
        messages.success(request, f"Cancelled-job restart queued as Celery task {result.id}.")
    return HttpResponseRedirect(redirect_to)


def _job_restart_cancelled_sync_response(request, pk: int):
    return _restart_job_in_place_response(
        request,
        pk=pk,
        required_status=JobStatus.CANCELLED,
        action_payload="restart_cancelled_in_place",
        user_action_text="restarted cancelled job",
        action_verb="restarted",
    )


@require_POST
def job_resubmit_failed_view(request, pk: int):
    redirect_to = _safe_next_url(request)
    try:
        result = resubmit_failed_job_task.delay(
            int(pk),
            requested_by=request.user.get_username() if request.user.is_authenticated else "",
            next_url=redirect_to,
        )
    except Exception as exc:
        messages.error(request, f"Failed to queue failed-job resubmit: {exc}")
    else:
        messages.success(request, f"Failed-job resubmit queued as Celery task {result.id}.")
    return HttpResponseRedirect(redirect_to)


def _job_resubmit_failed_sync_response(
    request,
    pk: int,
    *,
    wait_for_queue_capacity: bool = True,
    raise_on_failure: bool = False,
):
    return _restart_job_in_place_response(
        request,
        pk=pk,
        required_status=JobStatus.FAILED,
        action_payload="resubmit_failed_in_place",
        user_action_text="resubmitted failed job",
        action_verb="resubmitted",
        wait_for_queue_capacity=wait_for_queue_capacity,
        raise_on_failure=raise_on_failure,
    )


@require_POST
def job_resubmit_smooth_view(request, pk: int):
    redirect_to = _safe_next_url(request)
    try:
        result = resubmit_smooth_job_task.delay(
            int(pk),
            requested_by=request.user.get_username() if request.user.is_authenticated else "",
            next_url=redirect_to,
        )
    except Exception as exc:
        messages.error(request, f"Failed to queue smoothing resubmit: {exc}")
    else:
        messages.success(request, f"Smoothing resubmit queued as Celery task {result.id}.")
    return HttpResponseRedirect(redirect_to)


def _job_resubmit_smooth_sync_response(request, pk: int):
    redirect_to = _safe_next_url(request)
    job = (
        JobRecord.objects.select_related("hpc_system", "workflow_run")
        .filter(pk=pk)
        .first()
    )
    if job is None:
        messages.error(request, "Job was not found.")
        return HttpResponseRedirect(redirect_to)

    if job.status != JobStatus.COMPLETED:
        messages.error(
            request,
            f"Only completed jobs can be resubmitted for smoothing (current status: {job.status}).",
        )
        return HttpResponseRedirect(redirect_to)

    run = job.workflow_run
    run_meta = run.metadata if isinstance(run.metadata, dict) else {}
    submission_meta = run_meta.get("submission") if isinstance(run_meta.get("submission"), dict) else {}
    if str(submission_meta.get("run_mode") or "").strip() != RunMode.TDDFT_BOND_SCAN:
        messages.error(request, "Smooth resubmit is only available for completed scan runs.")
        return HttpResponseRedirect(redirect_to)

    previous_scheduler_job_id = str(job.scheduler_job_id or "").strip()
    try:
        new_scheduler_job_id, remote_run_dir, smooth_meta = _resubmit_completed_scan_for_smoothing(job)
    except Exception as exc:
        messages.error(request, f"Failed to resubmit completed scan for smoothing: {exc}")
        return HttpResponseRedirect(redirect_to)

    now = timezone.now()
    requested_by = request.user.get_username() if request.user.is_authenticated else ""
    previous_status = job.status

    scheduler_payload = job.scheduler_payload if isinstance(job.scheduler_payload, dict) else {}
    scheduler_payload = dict(scheduler_payload)
    smooth_history = scheduler_payload.get("smooth_resubmit_history")
    if not isinstance(smooth_history, list):
        smooth_history = []
    smooth_entry = {
        "requested_by": requested_by,
        "requested_at": now.isoformat(),
        "previous_scheduler_job_id": previous_scheduler_job_id,
        "new_scheduler_job_id": new_scheduler_job_id,
        "remote_run_dir": remote_run_dir,
        "submit_script": str(smooth_meta.get("submit_script") or "submit_workflow.sh"),
        "wrapper_script": str(smooth_meta.get("wrapper_script") or "run_workflow_submit.py"),
        "enable_ts_smoothing": bool(smooth_meta.get("enable_ts_smoothing")),
        "used_latest_structure": bool(smooth_meta.get("used_latest_structure")),
    }
    previous_job_folder_candidates = smooth_meta.get("previous_job_folder_candidates")
    if isinstance(previous_job_folder_candidates, list):
        normalized_candidates = [str(item).strip() for item in previous_job_folder_candidates if str(item).strip()]
        if normalized_candidates:
            smooth_entry["previous_job_folder_candidates"] = normalized_candidates
    previous_job_folder_found = str(smooth_meta.get("previous_job_folder_found") or "").strip()
    if previous_job_folder_found:
        smooth_entry["previous_job_folder_found"] = previous_job_folder_found
    previous_scheduler_job_ids = smooth_meta.get("previous_scheduler_job_ids")
    if isinstance(previous_scheduler_job_ids, list):
        normalized_prev_ids = [str(item).strip() for item in previous_scheduler_job_ids if str(item).strip()]
        if normalized_prev_ids:
            smooth_entry["previous_scheduler_job_ids"] = normalized_prev_ids
    artifact_path = str(smooth_meta.get("artifact_path") or "").strip()
    if artifact_path:
        smooth_entry["structure_artifact_path"] = artifact_path
    artifact_id = smooth_meta.get("artifact_id")
    if artifact_id is not None:
        smooth_entry["structure_artifact_id"] = artifact_id
    artifact_label = str(smooth_meta.get("artifact_label") or "").strip()
    if artifact_label:
        smooth_entry["structure_artifact_label"] = artifact_label
    warning_text = str(smooth_meta.get("warning") or "").strip()
    if warning_text:
        smooth_entry["structure_seed_warning"] = warning_text

    smooth_history.append(smooth_entry)
    scheduler_payload["smooth_resubmit_history"] = smooth_history

    remote_workdir = str(smooth_meta.get("remote_workdir") or "").strip()
    job.scheduler_job_id = new_scheduler_job_id
    job.status = JobStatus.INITIATED
    job.submitted_at = now
    job.started_at = None
    job.completed_at = None
    job.last_seen_at = now
    job.exit_code = None
    job.failure_reason = ""
    job.stdout_path = f"{remote_run_dir}/workflow.{new_scheduler_job_id}.log"
    job.stderr_path = ""
    job.remote_workdir = remote_workdir
    job.scheduler_payload = scheduler_payload
    job.save(
        update_fields=[
            "scheduler_job_id",
            "status",
            "submitted_at",
            "started_at",
            "completed_at",
            "last_seen_at",
            "exit_code",
            "failure_reason",
            "stdout_path",
            "stderr_path",
            "remote_workdir",
            "scheduler_payload",
            "updated_at",
        ]
    )

    run_metadata = dict(run.metadata) if isinstance(run.metadata, dict) else {}
    run_submission = run_metadata.get("submission")
    if not isinstance(run_submission, dict):
        run_submission = {}
    scan_config = run_submission.get("scan_config")
    if not isinstance(scan_config, dict):
        scan_config = {}
    scan_config["enable_ts_smoothing"] = True
    run_submission["scan_config"] = scan_config
    run_submission["smooth_resubmit_at"] = now.isoformat()
    run_submission["smooth_resubmit_requested_by"] = requested_by
    run_submission["previous_scheduler_job_ids"] = _collect_previous_scheduler_job_ids(
        run_submission,
        previous_scheduler_job_id=previous_scheduler_job_id,
    )
    run_smooth_history = run_submission.get("smooth_resubmit_history")
    if not isinstance(run_smooth_history, list):
        run_smooth_history = []
    run_smooth_history.append(
        {
            "requested_by": requested_by,
            "requested_at": now.isoformat(),
            "job_record_id": int(job.pk),
            "previous_scheduler_job_id": previous_scheduler_job_id,
            "new_scheduler_job_id": new_scheduler_job_id,
            "remote_run_dir": remote_run_dir,
            "enable_ts_smoothing": True,
        }
    )
    run_submission["smooth_resubmit_history"] = run_smooth_history
    run_metadata["submission"] = run_submission
    run.status = WorkflowStatus.INITIATED
    run.started_at = None
    run.completed_at = None
    run.working_directory = remote_run_dir
    run.metadata = run_metadata
    run.save(update_fields=["status", "started_at", "completed_at", "working_directory", "metadata", "updated_at"])

    event_payload = dict(smooth_entry)
    event_payload["action"] = "resubmit_completed_smooth_in_place"
    JobStatusEvent.objects.create(
        job=job,
        previous_status=previous_status,
        status=JobStatus.INITIATED,
        message="User resubmitted completed scan for smoothing from /workflows/.",
        payload=event_payload,
    )

    if warning_text:
        messages.warning(
            request,
            (
                f"Scan resubmitted for smoothing as scheduler id {new_scheduler_job_id}, "
                f"but structure seed warning occurred: {warning_text}"
            ),
        )
    else:
        messages.success(
            request,
            (
                f"Scan resubmitted for smoothing from {remote_run_dir} "
                f"as scheduler id {new_scheduler_job_id}."
            ),
        )
    return HttpResponseRedirect(redirect_to)


@require_POST
def job_resubmit_replot_view(request, pk: int):
    redirect_to = _safe_next_url(request)
    try:
        result = resubmit_replot_job_task.delay(
            int(pk),
            requested_by=request.user.get_username() if request.user.is_authenticated else "",
            next_url=redirect_to,
        )
    except Exception as exc:
        messages.error(request, f"Failed to queue replot resubmit: {exc}")
    else:
        messages.success(request, f"Replot resubmit queued as Celery task {result.id}.")
    return HttpResponseRedirect(redirect_to)


def _job_resubmit_replot_sync_response(request, pk: int):
    redirect_to = _safe_next_url(request)
    job = (
        JobRecord.objects.select_related("hpc_system", "workflow_run")
        .filter(pk=pk)
        .first()
    )
    if job is None:
        messages.error(request, "Job was not found.")
        return HttpResponseRedirect(redirect_to)

    if job.status != JobStatus.COMPLETED:
        messages.error(
            request,
            f"Only completed jobs can be resubmitted for replotting (current status: {job.status}).",
        )
        return HttpResponseRedirect(redirect_to)

    run = job.workflow_run
    if not _supports_replot_resubmit_for_run(run):
        messages.error(request, "Replot resubmit is only available for optimisation and bond-scan runs.")
        return HttpResponseRedirect(redirect_to)

    previous_scheduler_job_id = str(job.scheduler_job_id or "").strip()
    try:
        new_scheduler_job_id, remote_run_dir, replot_meta = _resubmit_completed_job_for_replot(job)
    except Exception as exc:
        messages.error(request, f"Failed to resubmit completed job for replotting: {exc}")
        return HttpResponseRedirect(redirect_to)

    now = timezone.now()
    requested_by = request.user.get_username() if request.user.is_authenticated else ""
    previous_status = job.status

    scheduler_payload = job.scheduler_payload if isinstance(job.scheduler_payload, dict) else {}
    scheduler_payload = dict(scheduler_payload)
    replot_history = scheduler_payload.get("replot_resubmit_history")
    if not isinstance(replot_history, list):
        replot_history = []
    replot_entry = {
        "requested_by": requested_by,
        "requested_at": now.isoformat(),
        "previous_scheduler_job_id": previous_scheduler_job_id,
        "new_scheduler_job_id": new_scheduler_job_id,
        "remote_run_dir": remote_run_dir,
        "submit_script": str(replot_meta.get("submit_script") or "submit_workflow.sh"),
        "wrapper_script": str(replot_meta.get("wrapper_script") or "run_workflow_submit.py"),
        "used_latest_structure": bool(replot_meta.get("used_latest_structure")),
        "replot_only": True,
        "run_mode": str(replot_meta.get("run_mode") or ""),
        "walltime": str(replot_meta.get("walltime") or _REPLOT_RESUBMIT_WALLTIME),
        "partition": str(replot_meta.get("partition") or _REPLOT_RESUBMIT_PARTITION),
        "qos": str(replot_meta.get("qos") or _REPLOT_RESUBMIT_QOS),
        "run_in_place": bool(replot_meta.get("run_in_place")),
    }
    previous_job_folder_candidates = replot_meta.get("previous_job_folder_candidates")
    if isinstance(previous_job_folder_candidates, list):
        normalized_candidates = [str(item).strip() for item in previous_job_folder_candidates if str(item).strip()]
        if normalized_candidates:
            replot_entry["previous_job_folder_candidates"] = normalized_candidates
    previous_job_folder_found = str(replot_meta.get("previous_job_folder_found") or "").strip()
    if previous_job_folder_found:
        replot_entry["previous_job_folder_found"] = previous_job_folder_found
    previous_scheduler_job_ids = replot_meta.get("previous_scheduler_job_ids")
    if isinstance(previous_scheduler_job_ids, list):
        normalized_prev_ids = [str(item).strip() for item in previous_scheduler_job_ids if str(item).strip()]
        if normalized_prev_ids:
            replot_entry["previous_scheduler_job_ids"] = normalized_prev_ids
    artifact_path = str(replot_meta.get("artifact_path") or "").strip()
    if artifact_path:
        replot_entry["structure_artifact_path"] = artifact_path
    artifact_id = replot_meta.get("artifact_id")
    if artifact_id is not None:
        replot_entry["structure_artifact_id"] = artifact_id
    artifact_label = str(replot_meta.get("artifact_label") or "").strip()
    if artifact_label:
        replot_entry["structure_artifact_label"] = artifact_label
    warning_text = str(replot_meta.get("warning") or "").strip()
    if warning_text:
        replot_entry["structure_seed_warning"] = warning_text

    replot_history.append(replot_entry)
    scheduler_payload["replot_resubmit_history"] = replot_history

    remote_workdir = str(replot_meta.get("remote_workdir") or "").strip()
    job.scheduler_job_id = new_scheduler_job_id
    job.status = JobStatus.INITIATED
    job.submitted_at = now
    job.started_at = None
    job.completed_at = None
    job.last_seen_at = now
    job.exit_code = None
    job.failure_reason = ""
    job.stdout_path = f"{remote_run_dir}/workflow.{new_scheduler_job_id}.log"
    job.stderr_path = ""
    job.remote_workdir = remote_workdir
    job.scheduler_payload = scheduler_payload
    job.save(
        update_fields=[
            "scheduler_job_id",
            "status",
            "submitted_at",
            "started_at",
            "completed_at",
            "last_seen_at",
            "exit_code",
            "failure_reason",
            "stdout_path",
            "stderr_path",
            "remote_workdir",
            "scheduler_payload",
            "updated_at",
        ]
    )

    run_metadata = dict(run.metadata) if isinstance(run.metadata, dict) else {}
    run_submission = run_metadata.get("submission")
    if not isinstance(run_submission, dict):
        run_submission = {}
    if str(replot_meta.get("run_mode") or "") == RunMode.TDDFT_BOND_SCAN:
        scan_config = run_submission.get("scan_config")
        if not isinstance(scan_config, dict):
            scan_config = {}
        scan_config["enable_ts_smoothing"] = False
        run_submission["scan_config"] = scan_config
    run_submission["replot_resubmit_at"] = now.isoformat()
    run_submission["replot_resubmit_requested_by"] = requested_by
    run_submission["previous_scheduler_job_ids"] = _collect_previous_scheduler_job_ids(
        run_submission,
        previous_scheduler_job_id=previous_scheduler_job_id,
    )
    run_replot_history = run_submission.get("replot_resubmit_history")
    if not isinstance(run_replot_history, list):
        run_replot_history = []
    run_replot_history.append(
        {
            "requested_by": requested_by,
            "requested_at": now.isoformat(),
            "job_record_id": int(job.pk),
            "previous_scheduler_job_id": previous_scheduler_job_id,
            "new_scheduler_job_id": new_scheduler_job_id,
            "remote_run_dir": remote_run_dir,
            "walltime": str(replot_meta.get("walltime") or _REPLOT_RESUBMIT_WALLTIME),
            "partition": str(replot_meta.get("partition") or _REPLOT_RESUBMIT_PARTITION),
            "qos": str(replot_meta.get("qos") or _REPLOT_RESUBMIT_QOS),
            "run_in_place": bool(replot_meta.get("run_in_place")),
        }
    )
    run_submission["replot_resubmit_history"] = run_replot_history
    run_metadata["submission"] = run_submission
    run.status = WorkflowStatus.INITIATED
    run.started_at = None
    run.completed_at = None
    run.working_directory = remote_run_dir
    run.metadata = run_metadata
    run.save(update_fields=["status", "started_at", "completed_at", "working_directory", "metadata", "updated_at"])

    event_payload = dict(replot_entry)
    event_payload["action"] = "resubmit_completed_replot_in_place"
    JobStatusEvent.objects.create(
        job=job,
        previous_status=previous_status,
        status=JobStatus.INITIATED,
        message="User resubmitted completed job for replotting from /jobs/.",
        payload=event_payload,
    )

    if warning_text:
        messages.warning(
            request,
            (
                f"Completed job resubmitted for replotting as scheduler id {new_scheduler_job_id} "
                f"(short queue settings, 20 minutes), but structure seed warning occurred: {warning_text}"
            ),
        )
    else:
        messages.success(
            request,
            (
                f"Completed job resubmitted for replotting from {remote_run_dir} as scheduler id "
                f"{new_scheduler_job_id} (partition={_REPLOT_RESUBMIT_PARTITION}, "
                f"qos={_REPLOT_RESUBMIT_QOS}, walltime={_REPLOT_RESUBMIT_WALLTIME})."
            ),
        )
    return HttpResponseRedirect(redirect_to)


@require_POST
def job_mark_all_cancelled_deleted_view(request):
    cancelled_jobs = list(
        JobRecord.objects.select_related("workflow_run")
        .filter(status=JobStatus.CANCELLED)
    )
    if not cancelled_jobs:
        messages.info(request, "No cancelled jobs found to mark for deletion.")
        return HttpResponseRedirect(reverse("job-list"))

    requested_by = request.user.get_username() if request.user.is_authenticated else ""
    updated = 0
    for job in cancelled_jobs:
        payload = {
            "requested_by": requested_by,
            "action": "mark_all_cancelled_for_deletion",
            "scheduler_job_id": job.scheduler_job_id,
            "remote_run_dir": canonical_run_path(job.workflow_run.working_directory),
        }
        changed = job.transition_to(
            JobStatus.DELETED,
            message="User requested bulk mark-cancelled-for-deletion from /jobs/.",
            payload=payload,
        )
        reason = "Marked for deletion by bulk cancelled-jobs action."
        if job.failure_reason != reason:
            job.failure_reason = reason
            job.save(update_fields=["failure_reason", "updated_at"])
            changed = True
        if changed:
            updated += 1

    messages.success(
        request,
        f"Marked {updated} cancelled job(s) for deletion.",
    )
    return HttpResponseRedirect(reverse("job-list"))


def _safe_atom_count(structure_meta: dict[str, object]) -> int | None:
    raw = structure_meta.get("atom_count")
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _artifact_is_remote_only(artifact: Artifact) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    return bool(metadata.get("remote_only"))


def _artifact_is_remote_sourced(artifact: Artifact) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    source = str(metadata.get("source") or "").strip().lower()
    if source.startswith("remote_"):
        return True
    return bool(
        metadata.get("downloaded_from_hpc")
        or metadata.get("remote_path")
        or metadata.get("remote_only")
    )


def _safe_cache_relative_path(path_value: str | Path) -> Path:
    normalized = normalize_run_path(path_value).lstrip("/")
    parts = [part for part in Path(normalized).parts if part and part not in {".", ".."}]
    if not parts:
        return Path("artifact.dat")
    return Path(*parts)


def _inferred_artifact_cache_path(run: WorkflowRun, artifact: Artifact) -> Path:
    run_key = str(getattr(run, "run_uid", "") or f"run_{run.pk}")
    run_root_text = str(run.working_directory or "")
    path_text = str(artifact.path or "")
    relative_under_run = _relative_to_run_root(run_root_text, path_text)
    if relative_under_run == "":
        relative_path = _safe_cache_relative_path(Path(path_text).name or "artifact.dat")
    elif relative_under_run:
        relative_path = _safe_cache_relative_path(relative_under_run)
    else:
        relative_path = _safe_cache_relative_path(path_text)
    cache_root = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()
    return (cache_root / run_key / relative_path).resolve()


def _artifact_local_path(run: WorkflowRun, artifact: Artifact) -> Path:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    remote_sourced = _artifact_is_remote_sourced(artifact)
    cache_path = str(metadata.get("local_cache_path") or "").strip()
    if cache_path:
        local_cache = _runtime_path(cache_path)
        if local_cache.exists() and local_cache.is_file():
            return local_cache
        if remote_sourced:
            return local_cache
    if remote_sourced:
        path_text = str(artifact.path or "")
        run_root_text = str(run.working_directory or "")
        relative_under_run = _relative_to_run_root(run_root_text, path_text)
        if relative_under_run == "":
            fallback_path = _runtime_path(run_root_text)
        elif relative_under_run:
            fallback_path = (_runtime_path(run_root_text) / Path(relative_under_run)).resolve()
        else:
            path = _runtime_path(path_text)
            if path.is_absolute():
                fallback_path = path
            else:
                fallback_path = (_runtime_path(run_root_text) / path).resolve()
        if fallback_path.exists() and fallback_path.is_file():
            return fallback_path
        inferred_cache = _inferred_artifact_cache_path(run, artifact)
        if inferred_cache.exists() and inferred_cache.is_file():
            return inferred_cache
        return inferred_cache
    path_text = str(artifact.path or "")
    run_root_text = str(run.working_directory or "")
    relative_under_run = _relative_to_run_root(run_root_text, path_text)
    if relative_under_run == "":
        return _runtime_path(run_root_text)
    if relative_under_run:
        return (_runtime_path(run_root_text) / Path(relative_under_run)).resolve()
    path = _runtime_path(path_text)
    if path.is_absolute():
        return path
    return (_runtime_path(run_root_text) / path).resolve()


def _runtime_path(path_value: str | Path) -> Path:
    normalized = normalize_run_path(path_value)
    if not normalized:
        return Path("")
    direct = Path(normalized)
    if direct.is_absolute():
        return direct
    canonical = canonical_run_path(normalized)
    canonical_path = Path(canonical)
    if canonical_path.is_absolute():
        return canonical_path
    return direct


def _relative_to_run_root(run_dir_text: str | Path, artifact_path_text: str | Path) -> str | None:
    run_root = canonical_run_path(run_dir_text)
    artifact_path = canonical_run_path(artifact_path_text)
    if not run_root or not artifact_path:
        return None
    if artifact_path == run_root:
        return ""
    prefix = f"{run_root}/"
    if artifact_path.startswith(prefix):
        return artifact_path[len(prefix) :]
    return None


def _is_evaluated_traj_artifact(artifact: Artifact) -> bool:
    if artifact.kind != ArtifactKind.TRAJECTORY:
        return False
    path_name = Path(str(artifact.path or "")).name.lower()
    if path_name == "evaluated.traj":
        return True
    label_text = str(artifact.label or "").lower()
    return "evaluated.traj" in label_text


def _is_scan_endpoints_db_artifact(artifact: Artifact) -> bool:
    return Path(str(artifact.path or "")).name.lower() == _SCAN_ENDPOINTS_DB_FILENAME


def _select_latest_local_scan_endpoints_db_artifact_for_run(run: WorkflowRun) -> Artifact | None:
    candidates: list[tuple[int, int, int, str, int, int, Artifact]] = []
    for artifact in Artifact.objects.filter(workflow_run=run).select_related("job"):
        if not _is_scan_endpoints_db_artifact(artifact):
            continue
        local_path = _artifact_local_path(run, artifact)
        if not local_path.exists() or not local_path.is_file():
            continue
        source_job_id = str(_artifact_source_scheduler_job_id(artifact) or "").strip()
        source_job_id_numeric = _parse_optional_int(source_job_id)
        has_source_job_id = 1 if source_job_id else 0
        is_numeric_source_job_id = 1 if source_job_id_numeric is not None else 0
        try:
            mtime_ns = int(local_path.stat().st_mtime_ns)
        except OSError:
            mtime_ns = 0
        candidates.append(
            (
                has_source_job_id,
                is_numeric_source_job_id,
                int(source_job_id_numeric or -1),
                source_job_id.casefold(),
                mtime_ns,
                int(artifact.pk or 0),
                artifact,
            )
        )
    if not candidates:
        return None
    return max(candidates, key=lambda row: row[:-1])[-1]


def _select_latest_local_evaluated_traj_artifact_for_run(run: WorkflowRun) -> Artifact | None:
    candidates: list[tuple[int, int, Artifact]] = []
    for artifact in Artifact.objects.filter(workflow_run=run, kind=ArtifactKind.TRAJECTORY):
        if not _is_evaluated_traj_artifact(artifact):
            continue
        local_path = _artifact_local_path(run, artifact)
        if not local_path.exists() or not local_path.is_file():
            continue
        try:
            mtime_ns = int(local_path.stat().st_mtime_ns)
        except OSError:
            mtime_ns = 0
        candidates.append((mtime_ns, int(artifact.pk or 0), artifact))
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row[0], row[1]))[2]


def _latest_restart_band_extxyz_for_run(run: WorkflowRun) -> tuple[str | None, dict[str, object]]:
    candidates: list[tuple[int, int, Artifact]] = []
    for artifact in Artifact.objects.filter(workflow_run=run, kind=ArtifactKind.TRAJECTORY):
        if Path(str(artifact.path or "")).name.lower() != "predicted.traj":
            continue
        local_path = _artifact_local_path(run, artifact)
        if not local_path.is_file():
            continue
        try:
            mtime_ns = int(local_path.stat().st_mtime_ns)
        except OSError:
            mtime_ns = 0
        candidates.append((mtime_ns, int(artifact.pk or 0), artifact))

    submission = run.metadata.get("submission", {}) if isinstance(run.metadata, dict) else {}
    neb_candidate = submission.get("neb_config", {}) if isinstance(submission, dict) else {}
    neb_config = neb_candidate if isinstance(neb_candidate, dict) else {}
    band_size = max(3, int(neb_config.get("max_images") or 21))
    for _mtime_ns, _artifact_id, artifact in sorted(candidates, reverse=True, key=lambda row: row[:2]):
        try:
            frames = ase_read(str(_artifact_local_path(run, artifact)), index=":")
            if not isinstance(frames, (list, tuple)):
                frames = [frames]
            if len(frames) < band_size:
                continue
            band = frames[-band_size:]
            return atoms_to_extxyz(band), {
                "band_artifact_id": int(artifact.pk),
                "band_artifact_label": str(artifact.label or ""),
                "band_artifact_path": str(artifact.path or ""),
                "band_frame_count": len(band),
                "band_frame_selection": f"last_{band_size}_predicted_frames",
            }
        except Exception:
            continue
    return None, {}


def _restart_reasonable_fmax_for_run(run: WorkflowRun) -> float:
    del run
    return float(_RESTART_REASONABLE_FMAX_EVA)


def _safe_atoms_fmax(atoms) -> float | None:
    forces = None
    for kwargs in ({"apply_constraint": False}, {}):
        try:
            forces = atoms.get_forces(**kwargs)
            break
        except Exception:
            continue
    if forces is None:
        return None

    max_force = 0.0
    try:
        for vector in forces:
            fx = float(vector[0])
            fy = float(vector[1])
            fz = float(vector[2])
            magnitude = (fx * fx + fy * fy + fz * fz) ** 0.5
            if magnitude > max_force:
                max_force = magnitude
    except Exception:
            return None
    return max_force


def _scan_row_lookup(row: object, key: str) -> object:
    key_values = dict(getattr(row, "key_value_pairs", {}) or {})
    if key in key_values:
        return key_values[key]
    data = dict(getattr(row, "data", {}) or {})
    if key in data:
        return data[key]
    return getattr(row, key, None)


def _select_restart_atoms_from_scan_endpoints_db(
    *,
    run: WorkflowRun,
    artifact: Artifact,
) -> tuple[object, dict[str, object]]:
    path = _artifact_local_path(run, artifact)
    selected: dict[int, tuple[float | None, object]] = {}
    try:
        db = ase_connect(str(path))
        for row in db.select():
            iteration = _parse_optional_int(_scan_row_lookup(row, "iteration"))
            if iteration is None:
                continue
            try:
                atoms = row.toatoms()
            except Exception:
                continue

            energy_eV = _parse_optional_float(_scan_row_lookup(row, "total_energy_eV"))
            if energy_eV is None:
                energy_eV = _parse_optional_float(_scan_row_lookup(row, "scan_energy_ev"))
            if energy_eV is None:
                energy_eV = _parse_optional_float(_scan_row_lookup(row, "energy_eV"))

            existing = selected.get(int(iteration))
            if existing is None:
                selected[int(iteration)] = (energy_eV, atoms)
                continue
            existing_energy = existing[0]
            if (
                existing_energy is None
                or (energy_eV is not None and float(energy_eV) <= float(existing_energy))
            ):
                selected[int(iteration)] = (energy_eV, atoms)
    except Exception as exc:
        raise RuntimeError(f"Failed to read scan endpoint frames from database: {exc}") from exc

    ordered = sorted(selected.items(), key=lambda item: item[0])
    if not ordered:
        raise ValueError("No scan endpoint frames were found in scan_endpoints.db.")

    endpoint_iteration, (endpoint_energy, endpoint_atoms) = ordered[-1]
    payload: dict[str, object] = {
        "scan_endpoints_frame_count": len(ordered),
        "scan_endpoints_iteration": int(endpoint_iteration),
        "scan_endpoints_selection": "latest_iteration_lowest_energy_frame",
    }
    if endpoint_energy is not None:
        payload["scan_endpoints_energy_eV"] = float(endpoint_energy)
    return endpoint_atoms, payload


def _select_restart_atoms_from_evaluated_traj(
    *,
    run: WorkflowRun,
    artifact: Artifact,
) -> tuple[object, dict[str, object]]:
    path = _artifact_local_path(run, artifact)
    frames = ase_read(str(path), index=":")
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    if not frames:
        raise ValueError(f"Trajectory is empty: {path}")

    frame_count = len(frames)
    threshold = _restart_reasonable_fmax_for_run(run)
    selected_index: int | None = None
    selected_atoms = None
    selected_fmax: float | None = None

    for idx in range(frame_count - 1, -1, -1):
        candidate = frames[idx]
        candidate_fmax = _safe_atoms_fmax(candidate)
        if candidate_fmax is None:
            continue
        if candidate_fmax <= threshold:
            selected_index = idx
            selected_atoms = candidate
            selected_fmax = candidate_fmax
            break
    if selected_atoms is None or selected_index is None:
        raise ValueError(
            f"No evaluated.traj frame satisfies reasonable fmax <= {threshold:.6g} eV/A."
        )

    return selected_atoms, {
        "trajectory_frame_count": frame_count,
        "trajectory_frame_index": selected_index,
        "trajectory_frame_fmax": selected_fmax,
        "trajectory_reasonable_fmax": threshold,
        "trajectory_frame_selection": "last_reasonable_fmax_frame",
    }


def _select_source_structure_artifact_for_run(
    run: WorkflowRun,
    *,
    allowed_kinds: tuple[str, ...] | None = None,
    exclude_evaluated_traj: bool = False,
) -> Artifact | None:
    kinds = allowed_kinds or (ArtifactKind.STRUCTURE, ArtifactKind.TRAJECTORY)
    all_artifacts = list(
        Artifact.objects.filter(
            workflow_run=run,
            kind__in=list(kinds),
        )
    )
    if not all_artifacts:
        return None
    artifacts: list[Artifact] = []
    for artifact in all_artifacts:
        if exclude_evaluated_traj and _is_evaluated_traj_artifact(artifact):
            continue
        local_path = _artifact_local_path(run, artifact)
        if local_path.exists() and local_path.is_file():
            artifacts.append(artifact)
    if not artifacts:
        return None

    def _score(artifact: Artifact) -> tuple[int, int]:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        score = 0
        if bool(metadata.get("is_final_structure")):
            score += 100
        label_text = str(artifact.label or "").lower()
        path_text = str(artifact.path or "").lower()
        if "reaction_hub_final" in label_text or "reaction_hub_final" in path_text:
            score += 40
        if "final" in label_text:
            score += 10
        return (score, int(artifact.pk or 0))

    return max(artifacts, key=_score)


def _resolve_existing_run_structure_atoms(*, source_run_id: int, source_artifact_id: int):
    source_run = WorkflowRun.objects.filter(pk=source_run_id).first()
    if source_run is None:
        raise ValueError(f"Source workflow run {source_run_id} does not exist.")

    artifact = Artifact.objects.filter(pk=source_artifact_id, workflow_run=source_run).first()
    if artifact is None:
        raise ValueError(
            f"Source artifact {source_artifact_id} was not found on workflow run {source_run_id}."
        )
    if artifact.kind not in {ArtifactKind.STRUCTURE, ArtifactKind.TRAJECTORY, ArtifactKind.DATABASE}:
        raise ValueError("Selected source artifact is not a structure/trajectory/database file.")

    local_path = _artifact_local_path(source_run, artifact)
    if _artifact_is_remote_only(artifact) and not local_path.exists():
        raise ValueError(
            "Selected source structure is marked as remote-only and no local cache is available."
        )
    if not local_path.exists() or not local_path.is_file():
        raise ValueError(f"Selected source structure file is not available locally: {local_path}")

    atoms, metadata = structure_path_to_atoms(
        local_path,
        source="existing_run",
        source_run_id=source_run.pk,
        source_artifact_id=artifact.pk,
        source_label=source_run.run_name,
    )
    payload = dict(metadata)
    payload["source_run_name"] = source_run.run_name
    payload["source_workflow_name"] = source_run.workflow.name
    payload["source_artifact_label"] = artifact.label
    payload["source_artifact_path"] = artifact.path
    coadsorbate_hints = _coadsorbate_provenance_hints_from_run(source_run)
    if coadsorbate_hints:
        payload["adsorption_reference_hints"] = coadsorbate_hints
        payload["coadsorbate_provenance"] = coadsorbate_hints
    return atoms, payload


def _resolve_existing_run_structure_payload(*, source_run_id: int, source_artifact_id: int) -> tuple[str, dict[str, object]]:
    atoms, payload = _resolve_existing_run_structure_atoms(
        source_run_id=source_run_id,
        source_artifact_id=source_artifact_id,
    )
    return atoms_to_xyz(atoms), payload


def _resolve_existing_run_structure_frame_payloads(
    *,
    source_run_id: int,
    source_artifact_id: int,
) -> list[tuple[str, dict[str, object]]]:
    frame_atoms = _resolve_existing_run_structure_frame_atoms(
        source_run_id=source_run_id,
        source_artifact_id=source_artifact_id,
    )
    return [(atoms_to_xyz(atoms), metadata) for atoms, metadata in frame_atoms]


def _resolve_existing_run_structure_frame_atoms(
    *,
    source_run_id: int,
    source_artifact_id: int,
):
    source_run = WorkflowRun.objects.filter(pk=source_run_id).first()
    if source_run is None:
        raise ValueError(f"Source workflow run {source_run_id} does not exist.")

    artifact = Artifact.objects.filter(pk=source_artifact_id, workflow_run=source_run).first()
    if artifact is None:
        raise ValueError(
            f"Source artifact {source_artifact_id} was not found on workflow run {source_run_id}."
        )
    if artifact.kind not in {ArtifactKind.TRAJECTORY, ArtifactKind.DATABASE}:
        raise ValueError("Batch submission from existing run requires a trajectory/database artifact.")

    local_path = _artifact_local_path(source_run, artifact)
    if _artifact_is_remote_only(artifact) and not local_path.exists():
        raise ValueError(
            "Selected source trajectory/database artifact is remote-only and no local cache is available."
        )
    if not local_path.exists() or not local_path.is_file():
        raise ValueError(f"Selected source file is not available locally: {local_path}")
    if local_path.suffix.lower() not in {".traj", ".db"}:
        raise ValueError("Batch submission from existing run is only supported for .traj/.db artifacts.")

    return structure_path_to_atoms_frames(
        local_path,
        source="existing_run",
        source_run_id=source_run.pk,
        source_artifact_id=artifact.pk,
        source_label=source_run.run_name,
    )


def _resolve_reference_cluster_structure_atoms(*, reference_cluster_id: int):
    cluster = ReferenceNanocluster.objects.filter(pk=reference_cluster_id).first()
    if cluster is None:
        raise ValueError(f"Reference nanocluster {reference_cluster_id} does not exist.")

    cluster_runs = list(
        WorkflowRun.objects.select_related("workflow")
        .filter(reference_nanocluster=cluster)
        .order_by("-created_at", "-id")
    )
    for cluster_run in cluster_runs:
        artifact = _select_source_structure_artifact_for_run(
            cluster_run,
            allowed_kinds=(ArtifactKind.STRUCTURE, ArtifactKind.TRAJECTORY, ArtifactKind.DATABASE),
            exclude_evaluated_traj=True,
        )
        if artifact is None:
            continue
        try:
            atoms, payload = _resolve_existing_run_structure_atoms(
                source_run_id=int(cluster_run.pk),
                source_artifact_id=int(artifact.pk),
            )
        except Exception:
            continue
        payload = dict(payload)
        payload["source"] = "reference_cluster"
        payload["reference_cluster_id"] = int(cluster.pk)
        payload["reference_cluster_name"] = str(cluster.name)
        payload["reference_cluster_composition"] = str(cluster.composition)
        payload["reference_cluster_run_id"] = int(cluster_run.pk)
        payload["reference_cluster_run_name"] = str(cluster_run.run_name)
        return atoms, payload

    structure_path = str(cluster.structure_path or "").strip()
    if structure_path:
        path = Path(structure_path)
        if path.exists() and path.is_dir():
            for filename in ("reaction_hub_final.xyz", "input_structure.extxyz", "orca.xyz"):
                candidate = path / filename
                if candidate.exists() and candidate.is_file():
                    atoms, metadata = structure_path_to_atoms(
                        candidate,
                        source="reference_cluster",
                        source_label=str(cluster.name),
                    )
                    metadata["reference_cluster_id"] = int(cluster.pk)
                    metadata["reference_cluster_name"] = str(cluster.name)
                    metadata["reference_cluster_composition"] = str(cluster.composition)
                    return atoms, metadata
        elif path.exists() and path.is_file():
            atoms, metadata = structure_path_to_atoms(
                path,
                source="reference_cluster",
                source_label=str(cluster.name),
            )
            metadata["reference_cluster_id"] = int(cluster.pk)
            metadata["reference_cluster_name"] = str(cluster.name)
            metadata["reference_cluster_composition"] = str(cluster.composition)
            return atoms, metadata

    raise ValueError(
        "Unable to resolve a local structure for selected reference nanocluster. "
        "Run a cluster reference job first, or ensure reference structure_path points to a local file."
    )


def _resolve_reference_adsorbate_structure_atoms(
    *,
    reference_adsorbate_id: int,
    settings_profile_id: int | None = None,
):
    adsorbate = (
        ReferenceAdsorbate.objects.select_related("settings_profile")
        .filter(pk=reference_adsorbate_id)
        .first()
    )
    if adsorbate is None:
        raise ValueError(f"Reference adsorbate {reference_adsorbate_id} does not exist.")

    if settings_profile_id is not None and adsorbate.settings_profile_id != settings_profile_id:
        raise ValueError(
            f"Reference adsorbate '{adsorbate.name}' does not match selected settings profile."
        )

    reference_formula = str(adsorbate.formula or "").strip()
    adsorbate_runs = list(
        WorkflowRun.objects.select_related("workflow")
        .filter(
            reference_adsorbate=adsorbate,
            **({"settings_profile_id": settings_profile_id} if settings_profile_id is not None else {}),
        )
        .order_by("-created_at", "-id")
    )
    for adsorbate_run in adsorbate_runs:
        artifact = _select_source_structure_artifact_for_run(
            adsorbate_run,
            allowed_kinds=(ArtifactKind.STRUCTURE, ArtifactKind.TRAJECTORY, ArtifactKind.DATABASE),
            exclude_evaluated_traj=True,
        )
        if artifact is None:
            continue
        try:
            atoms, payload = _resolve_existing_run_structure_atoms(
                source_run_id=int(adsorbate_run.pk),
                source_artifact_id=int(artifact.pk),
            )
        except Exception:
            continue
        payload = dict(payload)
        payload["source"] = "reference_adsorbate"
        payload["reference_adsorbate_id"] = int(adsorbate.pk)
        payload["reference_adsorbate_name"] = str(adsorbate.name)
        payload["reference_adsorbate_settings_profile_id"] = int(adsorbate.settings_profile_id)
        payload["reference_adsorbate_settings_profile_name"] = str(adsorbate.settings_profile.name)
        payload["reference_adsorbate_run_id"] = int(adsorbate_run.pk)
        payload["reference_adsorbate_run_name"] = str(adsorbate_run.run_name)
        if reference_formula:
            payload["reference_adsorbate_formula"] = reference_formula
            if not str(payload.get("formula") or "").strip():
                payload["formula"] = reference_formula
        return atoms, payload

    structure_path = str(adsorbate.structure_path or "").strip()
    if structure_path:
        path = Path(structure_path)
        if path.exists() and path.is_dir():
            for filename in ("reaction_hub_final.xyz", "input_structure.extxyz", "orca.xyz"):
                candidate = path / filename
                if not candidate.exists() or not candidate.is_file():
                    continue
                atoms, metadata = structure_path_to_atoms(
                    candidate,
                    source="reference_adsorbate",
                    source_label=str(adsorbate.name),
                )
                metadata["reference_adsorbate_id"] = int(adsorbate.pk)
                metadata["reference_adsorbate_name"] = str(adsorbate.name)
                metadata["reference_adsorbate_settings_profile_id"] = int(adsorbate.settings_profile_id)
                metadata["reference_adsorbate_settings_profile_name"] = str(adsorbate.settings_profile.name)
                if reference_formula:
                    metadata["reference_adsorbate_formula"] = reference_formula
                return atoms, metadata
        elif path.exists() and path.is_file():
            atoms, metadata = structure_path_to_atoms(
                path,
                source="reference_adsorbate",
                source_label=str(adsorbate.name),
            )
            metadata["reference_adsorbate_id"] = int(adsorbate.pk)
            metadata["reference_adsorbate_name"] = str(adsorbate.name)
            metadata["reference_adsorbate_settings_profile_id"] = int(adsorbate.settings_profile_id)
            metadata["reference_adsorbate_settings_profile_name"] = str(adsorbate.settings_profile.name)
            if reference_formula:
                metadata["reference_adsorbate_formula"] = reference_formula
            return atoms, metadata

    raise ValueError(
        "Unable to resolve a local structure for selected reference adsorbate. "
        "Run an adsorbate reference job first, or ensure reference structure_path points to a local file."
    )


def _latest_smooth_scan_structure_xyz_for_run(run: WorkflowRun) -> tuple[str | None, dict[str, object]]:
    scan_warning = ""
    scan_artifact = _select_latest_local_scan_endpoints_db_artifact_for_run(run)
    if scan_artifact is not None:
        try:
            atoms, scan_meta = _select_restart_atoms_from_scan_endpoints_db(run=run, artifact=scan_artifact)
            return atoms_to_xyz(atoms), {
                "artifact_id": int(scan_artifact.pk),
                "artifact_label": str(scan_artifact.label or ""),
                "artifact_path": str(scan_artifact.path or ""),
                **scan_meta,
                **_atoms_periodicity_payload(atoms),
            }
        except Exception as exc:
            scan_warning = str(exc)

    artifact = _select_source_structure_artifact_for_run(
        run,
        allowed_kinds=(ArtifactKind.STRUCTURE,),
        exclude_evaluated_traj=True,
    )
    if artifact is None:
        artifact = _select_source_structure_artifact_for_run(
            run,
            exclude_evaluated_traj=True,
        )
    if artifact is None:
        if scan_warning:
            return None, {"warning": f"Unable to use scan_endpoints.db for restart seed: {scan_warning}"}
        return None, {}
    try:
        atoms, source_meta = _resolve_existing_run_structure_atoms(
            source_run_id=int(run.pk),
            source_artifact_id=int(artifact.pk),
        )
    except Exception as exc:
        warning = str(exc)
        if scan_warning:
            warning = (
                f"Unable to use scan_endpoints.db for restart seed: {scan_warning}. "
                f"Fallback structure also failed: {warning}"
            )
        return None, {
            "artifact_id": int(artifact.pk),
            "artifact_label": str(artifact.label or ""),
            "artifact_path": str(artifact.path or ""),
            "warning": warning,
        }

    payload = {
        "artifact_id": int(artifact.pk),
        "artifact_label": str(artifact.label or ""),
        "artifact_path": str(artifact.path or ""),
        **_atoms_periodicity_payload(atoms),
    }
    for key in ("cell_A", "pbc", "is_periodic"):
        if source_meta.get(key) is not None:
            payload[key] = source_meta.get(key)
    if scan_warning:
        payload["warning"] = f"Unable to use scan_endpoints.db for restart seed: {scan_warning}"
    return atoms_to_xyz(atoms), payload


def _latest_restart_structure_xyz_for_run(run: WorkflowRun) -> tuple[str | None, dict[str, object]]:
    trajectory_warning = ""
    traj_artifact = _select_latest_local_evaluated_traj_artifact_for_run(run)
    if traj_artifact is not None:
        try:
            atoms, traj_meta = _select_restart_atoms_from_evaluated_traj(run=run, artifact=traj_artifact)
            return atoms_to_xyz(atoms), {
                "artifact_id": int(traj_artifact.pk),
                "artifact_label": str(traj_artifact.label or ""),
                "artifact_path": str(traj_artifact.path or ""),
                **traj_meta,
                **_atoms_periodicity_payload(atoms),
            }
        except Exception as exc:
            trajectory_warning = str(exc)

    artifact = _select_source_structure_artifact_for_run(
        run,
        allowed_kinds=(ArtifactKind.STRUCTURE,),
    )
    if artifact is None:
        artifact = _select_source_structure_artifact_for_run(
            run,
            exclude_evaluated_traj=True,
        )
    if artifact is None:
        if trajectory_warning:
            return None, {"warning": f"Unable to use evaluated.traj for restart seed: {trajectory_warning}"}
        return None, {}
    try:
        atoms, source_meta = _resolve_existing_run_structure_atoms(
            source_run_id=int(run.pk),
            source_artifact_id=int(artifact.pk),
        )
    except Exception as exc:
        warning = str(exc)
        if trajectory_warning:
            warning = (
                f"Unable to use evaluated.traj for restart seed: {trajectory_warning}. "
                f"Fallback structure also failed: {warning}"
            )
        return None, {
            "artifact_id": int(artifact.pk),
            "artifact_label": str(artifact.label or ""),
            "artifact_path": str(artifact.path or ""),
            "warning": warning,
        }

    payload = {
        "artifact_id": int(artifact.pk),
        "artifact_label": str(artifact.label or ""),
        "artifact_path": str(artifact.path or ""),
        **_atoms_periodicity_payload(atoms),
    }
    for key in ("cell_A", "pbc", "is_periodic"):
        if source_meta.get(key) is not None:
            payload[key] = source_meta.get(key)
    if trajectory_warning:
        payload["warning"] = f"Unable to use evaluated.traj for restart seed: {trajectory_warning}"
    return atoms_to_xyz(atoms), payload


def _structure_payload_to_xyz(geometry_payload: dict[str, object]) -> tuple[str, dict[str, object]]:
    source = str(geometry_payload.get("geometry_source") or GeometrySource.UPLOAD_FILE).strip()
    if source == GeometrySource.EXISTING_RUN:
        source_run_id = geometry_payload.get("source_run_id")
        source_artifact_id = geometry_payload.get("source_artifact_id")
        if source_run_id is None or source_artifact_id is None:
            raise ValueError("Source run/artifact is required for existing-run geometry mode.")
        try:
            run_id = int(source_run_id)
            artifact_id = int(source_artifact_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid source run/artifact identifiers.") from exc
        return _resolve_existing_run_structure_payload(
            source_run_id=run_id,
            source_artifact_id=artifact_id,
        )
    atoms, metadata = structure_input_to_atoms(
        geometry_source=source,
        structure_file=geometry_payload.get("structure_file"),
        traj_frame=geometry_payload.get("traj_frame"),
        ase_molecule_formula=str(geometry_payload.get("ase_molecule_formula") or ""),
        pubchem_cid=geometry_payload.get("pubchem_cid"),
        pubchem_query=str(geometry_payload.get("pubchem_query") or ""),
    )
    return atoms_to_xyz(atoms), metadata


def _structure_payload_to_atoms(geometry_payload: dict[str, object]):
    source = str(geometry_payload.get("geometry_source") or GeometrySource.UPLOAD_FILE).strip()
    if source == GeometrySource.EXISTING_RUN:
        source_run_id = geometry_payload.get("source_run_id")
        source_artifact_id = geometry_payload.get("source_artifact_id")
        if source_run_id is None or source_artifact_id is None:
            raise ValueError("Source run/artifact is required for existing-run geometry mode.")
        try:
            run_id = int(source_run_id)
            artifact_id = int(source_artifact_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid source run/artifact identifiers.") from exc
        return _resolve_existing_run_structure_atoms(
            source_run_id=run_id,
            source_artifact_id=artifact_id,
        )
    return structure_input_to_atoms(
        geometry_source=source,
        structure_file=geometry_payload.get("structure_file"),
        traj_frame=geometry_payload.get("traj_frame"),
        ase_molecule_formula=str(geometry_payload.get("ase_molecule_formula") or ""),
        pubchem_cid=geometry_payload.get("pubchem_cid"),
        pubchem_query=str(geometry_payload.get("pubchem_query") or ""),
    )


def _coadsorbate_adsorbate_payloads_from_cleaned_data(cleaned_data: dict[str, object]) -> list[tuple[str, dict[str, object]]]:
    payloads: list[tuple[str, dict[str, object]]] = []

    settings_profile = cleaned_data.get("settings_profile")
    settings_profile_id = int(settings_profile.pk) if settings_profile is not None else None

    for reference_adsorbate in cleaned_data.get("coadsorbate_reference_adsorbates") or []:
        reference_adsorbate_id = getattr(reference_adsorbate, "pk", None)
        if reference_adsorbate_id is None:
            continue
        atoms, metadata = _resolve_reference_adsorbate_structure_atoms(
            reference_adsorbate_id=int(reference_adsorbate_id),
            settings_profile_id=settings_profile_id,
        )
        payloads.append((atoms_to_xyz(atoms), metadata))

    for formula in cleaned_data.get("coadsorbate_ase_molecules") or []:
        atoms, metadata = structure_input_to_atoms(
            geometry_source=GeometrySource.ASE_MOLECULE,
            ase_molecule_formula=str(formula),
        )
        payloads.append((atoms_to_xyz(atoms), metadata))

    for cid in cleaned_data.get("coadsorbate_pubchem_cids") or []:
        atoms, metadata = structure_input_to_atoms(
            geometry_source=GeometrySource.PUBCHEM,
            pubchem_cid=int(cid),
        )
        payloads.append((atoms_to_xyz(atoms), metadata))

    for query in cleaned_data.get("coadsorbate_pubchem_queries") or []:
        atoms, metadata = structure_input_to_atoms(
            geometry_source=GeometrySource.PUBCHEM,
            pubchem_query=str(query),
        )
        payloads.append((atoms_to_xyz(atoms), metadata))

    for uploaded in cleaned_data.get("coadsorbate_structure_files") or []:
        atoms, metadata = structure_input_to_atoms(
            geometry_source=GeometrySource.UPLOAD_FILE,
            structure_file=uploaded,
            traj_frame=-1,
        )
        payloads.append((atoms_to_xyz(atoms), metadata))

    return payloads


def _coadsorbate_provenance_hints_from_run(run: WorkflowRun | None) -> dict[str, object]:
    if run is None:
        return {}

    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission") if isinstance(metadata.get("submission"), dict) else {}
    coadsorbate_config = submission.get("coadsorbate_config") if isinstance(submission, dict) else {}
    if not isinstance(coadsorbate_config, dict):
        return {}

    raw_adsorbate_metadata = coadsorbate_config.get("adsorbate_metadata")
    if not isinstance(raw_adsorbate_metadata, list):
        return {}

    normalized_adsorbates: list[dict[str, object]] = []
    formula_counts: Counter[str] = Counter()
    for entry in raw_adsorbate_metadata:
        if not isinstance(entry, dict):
            continue
        formula = str(entry.get("formula") or "").strip()
        if not formula:
            continue
        normalized_entry: dict[str, object] = {
            "formula": formula,
            "source": str(entry.get("source") or "").strip(),
        }
        for key in (
            "ase_molecule_formula",
            "pubchem_cid",
            "pubchem_query",
            "pubchem_query_mode",
            "source_filename",
            "source_suffix",
            "reference_adsorbate_id",
            "reference_adsorbate_name",
            "reference_adsorbate_formula",
            "reference_adsorbate_run_id",
            "reference_adsorbate_run_name",
            "reference_adsorbate_settings_profile_id",
            "reference_adsorbate_settings_profile_name",
        ):
            value = entry.get(key)
            if value not in (None, ""):
                normalized_entry[key] = value
        normalized_adsorbates.append(normalized_entry)
        formula_counts[formula] += 1

    if not normalized_adsorbates:
        return {}

    cluster_id = coadsorbate_config.get("reference_cluster_id")
    try:
        cluster_id_int = int(cluster_id) if cluster_id is not None else None
    except (TypeError, ValueError):
        cluster_id_int = None
    cluster_name = ""
    if cluster_id_int:
        cluster = ReferenceNanocluster.objects.filter(pk=cluster_id_int).only("name").first()
        if cluster is not None:
            cluster_name = str(cluster.name)
    elif run.reference_nanocluster_id:
        cluster_id_int = int(run.reference_nanocluster_id)
        cluster_name = str(run.reference_nanocluster.name)

    hints: dict[str, object] = {
        "source_run_id": int(run.pk),
        "source_run_name": str(run.run_name),
        "adsorbates": normalized_adsorbates,
        "adsorbate_formula_counts": dict(formula_counts),
        "adsorbate_formulas": [str(entry["formula"]) for entry in normalized_adsorbates],
    }
    if cluster_id_int:
        hints["reference_cluster_id"] = int(cluster_id_int)
    if cluster_name:
        hints["reference_cluster_name"] = cluster_name
    return hints


def _apply_reference_selection_to_run(
    *,
    run: WorkflowRun,
    reference_intent: str,
    reference_name: str,
    structure_meta: dict[str, object],
) -> str:
    if reference_intent == ReferenceIntent.NONE:
        return ""

    formula = str(structure_meta.get("formula") or "").strip()
    structure_path = str(run.working_directory or "").strip()

    if reference_intent == ReferenceIntent.ADSORBATE:
        adsorbate, created = ReferenceAdsorbate.objects.get_or_create(
            name=reference_name,
            defaults={
                "formula": formula[:64],
                "settings_profile": run.settings_profile,
                "structure_path": structure_path,
            },
        )
        if adsorbate.settings_profile_id != run.settings_profile_id:
            raise ValueError(
                f"Reference adsorbate '{reference_name}' uses settings profile '{adsorbate.settings_profile}', "
                f"but this run uses '{run.settings_profile}'. Choose another reference name or matching profile."
            )
        updates: list[str] = []
        if formula and not adsorbate.formula:
            adsorbate.formula = formula[:64]
            updates.append("formula")
        if structure_path and not adsorbate.structure_path:
            adsorbate.structure_path = structure_path
            updates.append("structure_path")
        if updates:
            adsorbate.save(update_fields=[*updates, "updated_at"])

        run.reference_adsorbate = adsorbate
        run.reference_nanocluster = None
        status = "created" if created else "linked existing"
        return f"Tagged as adsorbate reference '{adsorbate.name}' ({status})."

    if reference_intent == ReferenceIntent.NANOCLUSTER:
        composition = formula[:120] if formula else ""
        atom_count = _safe_atom_count(structure_meta)
        cluster, created = ReferenceNanocluster.objects.get_or_create(
            name=reference_name,
            defaults={
                "composition": composition or reference_name[:120],
                "atom_count": atom_count,
                "settings_profile": run.settings_profile,
                "structure_path": structure_path,
            },
        )
        if cluster.settings_profile_id != run.settings_profile_id:
            raise ValueError(
                f"Reference nanocluster '{reference_name}' uses settings profile '{cluster.settings_profile}', "
                f"but this run uses '{run.settings_profile}'. Choose another reference name or matching profile."
            )
        updates: list[str] = []
        if composition and not cluster.composition:
            cluster.composition = composition
            updates.append("composition")
        if atom_count is not None and cluster.atom_count is None:
            cluster.atom_count = atom_count
            updates.append("atom_count")
        if structure_path and not cluster.structure_path:
            cluster.structure_path = structure_path
            updates.append("structure_path")
        if updates:
            cluster.save(update_fields=[*updates, "updated_at"])

        run.reference_adsorbate = None
        run.reference_nanocluster = cluster
        status = "created" if created else "linked existing"
        return f"Tagged as nanocluster reference '{cluster.name}' ({status})."

    raise ValueError("Unsupported reference intent.")


class JobSubmitView(FormView):
    template_name = "jobs/job_submit.html"
    form_class = JobSubmitForm

    @staticmethod
    def _mode_from_query(raw: str | None) -> str:
        mode = str(raw or "").strip()
        if mode in {
            RunMode.SINGLE_OPTIMISATION,
            RunMode.TDDFT_BOND_SCAN,
            RunMode.CHARGE_TRANSFER,
            RunMode.ACTIVE_LEARNING_NEB,
            RunMode.COADSORBATE_PLACEMENT,
        }:
            return mode
        return ""

    @staticmethod
    def _flag_from_query(raw: str | None) -> bool:
        return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}

    def post(self, request, *args, **kwargs):
        if not self._flag_from_query(request.POST.get("quick_charge_transfer")):
            return super().post(request, *args, **kwargs)

        try:
            source_run_id = int(str(request.GET.get("from_run") or "").strip())
        except ValueError:
            source_run_id = 0
        if self._mode_from_query(request.GET.get("mode")) != RunMode.CHARGE_TRANSFER or source_run_id < 1:
            messages.error(request, "A source workflow run is required for CT analysis.")
            return HttpResponseRedirect(reverse("job-list"))

        unbound = self.form_class(initial=self.get_initial())
        data = QueryDict(mutable=True)
        for name in unbound.fields:
            value = unbound[name].value()
            if value in (None, "", False):
                continue
            if value is True:
                data[name] = "on"
            elif isinstance(value, (list, tuple)):
                data.setlist(name, [str(getattr(item, "pk", item)) for item in value])
            else:
                data[name] = str(getattr(value, "pk", value))
        form = self.form_class(data=data)
        if not form.is_valid():
            messages.error(request, f"CT submission could not be prepared: {form.errors.as_text()}")
            return HttpResponseRedirect(reverse("workflow-run-detail", kwargs={"pk": source_run_id}))
        response = self.form_valid(form)
        if response.status_code == 200:
            messages.error(request, f"CT submission failed: {form.errors.as_text()}")
            return HttpResponseRedirect(reverse("workflow-run-detail", kwargs={"pk": source_run_id}))
        return response

    @staticmethod
    def _coadsorbate_mace_model_path_for_system(system_name: str) -> str:
        default_path = str(getattr(settings, "REACTION_HUB_COADSORBATE_MACE_MODEL_PATH", "") or "").strip()
        raw_mapping = getattr(settings, "REACTION_HUB_COADSORBATE_MACE_MODEL_PATH_BY_HPC", {})
        if not isinstance(raw_mapping, dict):
            return default_path
        key = str(system_name or "").strip().lower()
        mapped = str(raw_mapping.get(key) or "").strip()
        return mapped or default_path

    @staticmethod
    def _workflow_is_coadsorbate_definition(workflow: WorkflowDefinition | None) -> bool:
        if workflow is None:
            return False
        code_path = str(getattr(workflow, "code_path", "") or "").strip().lower()
        return "coadsorbate" in code_path

    @staticmethod
    def _workflow_is_neb_definition(workflow: WorkflowDefinition | None) -> bool:
        if workflow is None:
            return False
        code_path = str(getattr(workflow, "code_path", "") or "").strip().lower()
        return "active_learning_neb_workflow.py" in code_path or "tddft_mlneb_workflow.py" in code_path

    @classmethod
    def _workflow_mode_mappings(
        cls,
        workflows: list[WorkflowDefinition],
    ) -> tuple[dict[str, list[str]], dict[str, str]]:
        mode_map: dict[str, list[str]] = {
            RunMode.SINGLE_OPTIMISATION: [],
            RunMode.TDDFT_BOND_SCAN: [],
            RunMode.CHARGE_TRANSFER: [],
            RunMode.ACTIVE_LEARNING_NEB: [],
            RunMode.COADSORBATE_PLACEMENT: [],
        }
        preferred: dict[str, str] = {
            RunMode.SINGLE_OPTIMISATION: "",
            RunMode.TDDFT_BOND_SCAN: "",
            RunMode.CHARGE_TRANSFER: "",
            RunMode.ACTIVE_LEARNING_NEB: "",
            RunMode.COADSORBATE_PLACEMENT: "",
        }
        preferred_rank = {
            RunMode.SINGLE_OPTIMISATION: -1,
            RunMode.TDDFT_BOND_SCAN: -1,
            RunMode.CHARGE_TRANSFER: -1,
            RunMode.ACTIVE_LEARNING_NEB: -1,
            RunMode.COADSORBATE_PLACEMENT: -1,
        }

        for workflow in workflows:
            workflow_id = str(workflow.pk)
            name = str(workflow.name or "").strip().lower()
            is_coadsorbate = cls._workflow_is_coadsorbate_definition(workflow)
            is_neb = cls._workflow_is_neb_definition(workflow)
            if is_coadsorbate:
                mode_map[RunMode.COADSORBATE_PLACEMENT].append(workflow_id)
                rank = 1
                if name == "mace adsorbate screening":
                    rank = 2
                if rank > preferred_rank[RunMode.COADSORBATE_PLACEMENT]:
                    preferred_rank[RunMode.COADSORBATE_PLACEMENT] = rank
                    preferred[RunMode.COADSORBATE_PLACEMENT] = workflow_id
                continue
            if is_neb:
                mode_map[RunMode.ACTIVE_LEARNING_NEB].append(workflow_id)
                rank = 1
                if name == "active learning neb (mace)":
                    rank = 2
                if rank > preferred_rank[RunMode.ACTIVE_LEARNING_NEB]:
                    preferred_rank[RunMode.ACTIVE_LEARNING_NEB] = rank
                    preferred[RunMode.ACTIVE_LEARNING_NEB] = workflow_id
                continue

            mode_map[RunMode.SINGLE_OPTIMISATION].append(workflow_id)
            mode_map[RunMode.TDDFT_BOND_SCAN].append(workflow_id)
            mode_map[RunMode.CHARGE_TRANSFER].append(workflow_id)
            rank = 1
            if name == "tddft active learning":
                rank = 2
            if rank > preferred_rank[RunMode.SINGLE_OPTIMISATION]:
                preferred_rank[RunMode.SINGLE_OPTIMISATION] = rank
                preferred[RunMode.SINGLE_OPTIMISATION] = workflow_id
            if rank > preferred_rank[RunMode.TDDFT_BOND_SCAN]:
                preferred_rank[RunMode.TDDFT_BOND_SCAN] = rank
                preferred[RunMode.TDDFT_BOND_SCAN] = workflow_id
            if rank > preferred_rank[RunMode.CHARGE_TRANSFER]:
                preferred_rank[RunMode.CHARGE_TRANSFER] = rank
                preferred[RunMode.CHARGE_TRANSFER] = workflow_id

        return mode_map, preferred

    def _prefill_source_from_query(self):
        raw_run_id = str(self.request.GET.get("from_run") or "").strip()
        if not raw_run_id:
            return None
        try:
            run_id = int(raw_run_id)
        except ValueError:
            return None

        source_run = (
            WorkflowRun.objects.select_related("workflow", "settings_profile")
            .filter(pk=run_id)
            .first()
        )
        if source_run is None:
            return None

        artifact: Artifact | None = None
        raw_artifact_id = str(
            self.request.GET.get("source_artifact")
            or self.request.GET.get("from_artifact")
            or ""
        ).strip()
        if raw_artifact_id:
            try:
                requested_artifact_id = int(raw_artifact_id)
            except ValueError:
                requested_artifact_id = 0
            if requested_artifact_id > 0:
                candidate = Artifact.objects.filter(pk=requested_artifact_id, workflow_run=source_run).first()
                if candidate is not None and candidate.kind in {
                    ArtifactKind.STRUCTURE,
                    ArtifactKind.TRAJECTORY,
                    ArtifactKind.DATABASE,
                }:
                    local_path = _artifact_local_path(source_run, candidate)
                    if local_path.exists() and local_path.is_file():
                        artifact = candidate

        if artifact is None:
            artifact = _select_source_structure_artifact_for_run(source_run)
        if artifact is None:
            return None

        latest_job = source_run.jobs.select_related("hpc_system").order_by("-created_at", "-id").first()
        return {
            "run": source_run,
            "artifact": artifact,
            "job": latest_job,
        }

    def get_initial(self):
        initial = super().get_initial()

        mode = self._mode_from_query(self.request.GET.get("mode"))
        if mode:
            initial["run_mode"] = mode

        prefill = self._prefill_source_from_query()
        if prefill is None:
            return initial

        source_run = prefill["run"]
        artifact = prefill["artifact"]
        latest_job = prefill["job"]
        submission = source_run.metadata.get("submission", {}) if isinstance(source_run.metadata, dict) else {}
        if not isinstance(submission, dict):
            submission = {}

        initial.update(
            {
                "geometry_source": GeometrySource.EXISTING_RUN,
                "source_run_id": source_run.pk,
                "source_artifact_id": artifact.pk,
                "neb_final_source_run_id": source_run.pk,
                "neb_final_source_artifact_id": artifact.pk,
                "workflow": source_run.workflow_id,
                "settings_profile": source_run.settings_profile_id,
                "reference_intent": ReferenceIntent.NONE,
            }
        )
        if latest_job and latest_job.hpc_system_id:
            initial["hpc_system"] = latest_job.hpc_system_id

        batch_requested = self._flag_from_query(self.request.GET.get("batch"))
        artifact_suffix = Path(str(artifact.path or "")).suffix.lower()
        selected_artifact_supports_batch = (
            artifact.kind in {ArtifactKind.TRAJECTORY, ArtifactKind.DATABASE}
            or artifact_suffix in {".traj", ".db"}
        )
        is_batch_submission_prefill = bool(
            batch_requested and selected_artifact_supports_batch and mode in {"", RunMode.SINGLE_OPTIMISATION}
        )
        if is_batch_submission_prefill:
            initial["run_mode"] = RunMode.SINGLE_OPTIMISATION
            initial["run_name"] = f"{source_run.run_name}_batch"
            initial["submit_all_traj_frames"] = True
        elif mode == RunMode.TDDFT_BOND_SCAN:
            initial["run_name"] = f"{source_run.run_name}_blscan"
        elif mode == RunMode.CHARGE_TRANSFER:
            suffix = (
                f"_{timezone.now():%Y%m%d_%H%M%S}"
                if self._flag_from_query(self.request.POST.get("quick_charge_transfer"))
                else ""
            )
            initial["run_name"] = f"{source_run.run_name}_ct{suffix}"
        elif mode == RunMode.ACTIVE_LEARNING_NEB:
            initial["run_name"] = f"{source_run.run_name}_neb"
        elif mode == RunMode.COADSORBATE_PLACEMENT:
            initial["run_name"] = f"{source_run.run_name}_coadsorbate"
            if source_run.reference_nanocluster_id:
                initial["coadsorbate_reference_cluster"] = source_run.reference_nanocluster_id
        else:
            initial["run_name"] = f"{source_run.run_name}_followup"

        for field_name in (
            "nodes",
            "tasks_per_node",
            "cpus_per_task",
            "walltime",
            "account",
            "partition",
            "qos",
            "reservation",
            "fmax",
            "active_learning_optimizer",
            "qm_backend",
            "target_wavelength_nm",
            "spin_multiplicity",
            "n_iter",
            "use_tddft_optimisation",
        ):
            value = submission.get(field_name)
            if field_name == "active_learning_optimizer" and value is not None:
                key = str(value).strip().lower().replace("-", "").replace("_", "")
                if key == "fire":
                    key = "fire2"
                if key in {"bfgs", "fire2"}:
                    value = key
            if value is not None and value != "":
                initial[field_name] = value

        for field_name in ("remote_runs_root", "scratch_base", "venv_activate_path"):
            value = submission.get(field_name)
            if isinstance(value, str) and value.strip():
                initial[field_name] = value

        if mode == RunMode.CHARGE_TRANSFER:
            initial["account"] = str(
                initial.get("account")
                or getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT", "")
                or ""
            )
            initial.update(
                {
                    "nodes": 1,
                    "tasks_per_node": 8,
                    "cpus_per_task": 1,
                    "walltime": "24:00:00",
                    "reservation": "",
                    "ct_nroots": _tddft_nroots_for_run(source_run),
                    "ct_orca_basename": "orca",
                }
            )
            try:
                source_atoms, _ = _resolve_existing_run_structure_atoms(
                    source_run_id=source_run.pk,
                    source_artifact_id=artifact.pk,
                )
            except ValueError:
                pass
            else:
                fragments = _infer_ct_fragment_defaults(source_atoms)
                if fragments:
                    initial["ct_nanoparticle_indices"] = fragments["nanoparticle_indices"]
                    initial["ct_adsorbate_indices"] = fragments["adsorbate_indices"]

        if mode == RunMode.TDDFT_BOND_SCAN:
            scan_config = submission.get("scan_config")
            if isinstance(scan_config, dict):
                for field_name in (
                    "bond_i",
                    "bond_j",
                    "increment",
                    "fix",
                    "fixed_symbols",
                    "target_distance",
                    "distance_change_fraction",
                    "enable_ts_smoothing",
                    "refine_points_each_side",
                    "tddft_scan_nroots",
                ):
                    value = scan_config.get(field_name)
                    if field_name == "fixed_symbols" and isinstance(value, (list, tuple)):
                        value = ", ".join(str(entry).strip() for entry in value if str(entry).strip())
                    if value is not None and value != "":
                        initial[field_name] = value
                # Restart extension is restart-only and should be an explicit user choice
                # for each new GUI submission.
                initial["restart_extend_steps"] = 0
                initial["restart_extend_increment_A"] = ""
        elif mode == RunMode.SINGLE_OPTIMISATION:
            single_constraints = submission.get("single_constraints_config")
            if isinstance(single_constraints, dict):
                fixed_symbols = single_constraints.get("fixed_symbols")
                if isinstance(fixed_symbols, (list, tuple)):
                    fixed_symbols_text = ", ".join(
                        str(entry).strip() for entry in fixed_symbols if str(entry).strip()
                    )
                    if fixed_symbols_text:
                        initial["fixed_symbols"] = fixed_symbols_text

                bond_pair_indices = single_constraints.get("bond_pair_indices")
                if (
                    isinstance(bond_pair_indices, (list, tuple))
                    and len(bond_pair_indices) == 2
                ):
                    initial["bond_i"] = bond_pair_indices[0]
                    initial["bond_j"] = bond_pair_indices[1]

                for field_name in ("target_distance", "fix"):
                    value = single_constraints.get(field_name)
                    if value is not None and value != "":
                        initial[field_name] = value
                for field_name in (
                    "final_tddft_spectrum",
                    "cluster_tddft_difference_spectrum",
                    "charge_transfer_spectrum",
                    "final_tddft_nroots",
                    "final_tddft_nroots_mode",
                ):
                    value = single_constraints.get(field_name)
                    if value is None or value == "":
                        continue
                    initial[f"single_{field_name}"] = value
        elif mode == RunMode.ACTIVE_LEARNING_NEB:
            neb_config = submission.get("neb_config")
            if isinstance(neb_config, dict):
                for field_name in (
                    "final_image_strategy",
                    "optimize_initial_image",
                    "optimize_final_image",
                    "endpoint_opt_steps",
                    "endpoint_opt_fmax",
                    "break_bond_i",
                    "break_bond_j",
                    "set_bond_distance_A",
                    "break_bond_extension_A",
                    "break_bond_fix",
                    "break_assert_broken",
                    "break_cutoff_scale",
                    "atom_shift_per_image_A",
                    "min_images",
                    "max_images",
                    "spring_constant",
                    "max_steps",
                    "climb",
                    "split_on_minimum",
                    "minimum_energy_window_eV",
                    "mace_model_path",
                    "mace_device",
                    "mace_default_dtype",
                    "mace_dispersion",
                    "catlearn_min_data",
                    "catlearn_unc_convergence",
                    "catlearn_max_unc",
                    "catlearn_steps",
                    "catlearn_ml_steps",
                    "catlearn_dtrust",
                    "tddft_scan_nroots",
                    "backend",
                ):
                    value = neb_config.get(field_name)
                    if value is None or value == "":
                        continue
                    if field_name == "atom_shift_per_image_A":
                        initial["neb_atom_shift_per_image_A"] = value
                    elif field_name == "final_image_strategy":
                        if str(value).strip().lower() == "bond_extension_from_initial":
                            value = "set_bond_distance_from_initial"
                        initial["neb_final_image_strategy"] = value
                    elif field_name == "optimize_initial_image":
                        initial["neb_optimize_initial_image"] = value
                    elif field_name == "optimize_final_image":
                        initial["neb_optimize_final_image"] = value
                    elif field_name == "endpoint_opt_steps":
                        initial["neb_endpoint_opt_steps"] = value
                    elif field_name == "endpoint_opt_fmax":
                        initial["neb_endpoint_opt_fmax"] = value
                    elif field_name == "break_bond_i":
                        initial["neb_break_bond_i"] = value
                    elif field_name == "break_bond_j":
                        initial["neb_break_bond_j"] = value
                    elif field_name in {"set_bond_distance_A", "break_bond_extension_A"}:
                        initial["neb_break_bond_extension_A"] = value
                    elif field_name == "break_bond_fix":
                        initial["neb_break_bond_fix"] = value
                    elif field_name == "break_assert_broken":
                        initial["neb_break_assert_broken"] = value
                    elif field_name == "break_cutoff_scale":
                        initial["neb_break_cutoff_scale"] = value
                    elif field_name == "min_images":
                        initial["neb_min_images"] = value
                    elif field_name == "max_images":
                        initial["neb_max_images"] = value
                    elif field_name == "spring_constant":
                        initial["neb_spring_constant"] = value
                    elif field_name == "max_steps":
                        initial["neb_max_steps"] = value
                    elif field_name == "climb":
                        initial["neb_climb"] = value
                    elif field_name == "split_on_minimum":
                        initial["neb_split_on_minimum"] = value
                    elif field_name == "minimum_energy_window_eV":
                        initial["neb_minimum_energy_window_eV"] = value
                    elif field_name == "mace_model_path":
                        initial["neb_mace_model_path"] = value
                    elif field_name == "mace_device":
                        initial["neb_mace_device"] = value
                    elif field_name == "mace_default_dtype":
                        initial["neb_mace_default_dtype"] = value
                    elif field_name == "mace_dispersion":
                        initial["neb_mace_dispersion"] = value
                    elif field_name == "catlearn_min_data":
                        initial["neb_catlearn_min_data"] = value
                    elif field_name == "catlearn_unc_convergence":
                        initial["neb_catlearn_unc_convergence"] = value
                    elif field_name == "catlearn_max_unc":
                        initial["neb_catlearn_max_unc"] = value
                    elif field_name == "catlearn_steps":
                        initial["neb_catlearn_steps"] = value
                    elif field_name == "catlearn_ml_steps":
                        initial["neb_catlearn_ml_steps"] = value
                    elif field_name == "catlearn_dtrust":
                        initial["neb_catlearn_dtrust"] = value
                    elif field_name == "tddft_scan_nroots":
                        initial["tddft_scan_nroots"] = value
                    elif field_name == "backend":
                        initial["neb_backend"] = value

        charge_transfer = submission.get("charge_transfer_config")
        if isinstance(charge_transfer, dict) and charge_transfer:
            fragments = charge_transfer.get("fragments_0based")
            if isinstance(fragments, dict):
                nanoparticle = fragments.get("1")
                adsorbate = fragments.get("2")
                if isinstance(nanoparticle, (list, tuple)):
                    initial["ct_nanoparticle_indices"] = ", ".join(
                        str(value) for value in nanoparticle
                    )
                if isinstance(adsorbate, (list, tuple)):
                    initial["ct_adsorbate_indices"] = ", ".join(
                        str(value) for value in adsorbate
                    )
            initial["ct_nroots"] = charge_transfer.get("nroots") or 400
            initial["ct_orca_basename"] = charge_transfer.get("orca_basename") or "orca"

        module_loads = submission.get("module_loads")
        if isinstance(module_loads, list) and not is_batch_submission_prefill:
            initial["module_loads"] = "\n".join(str(entry) for entry in module_loads if str(entry).strip())

        pythonpath_entries = submission.get("pythonpath_entries")
        if isinstance(pythonpath_entries, list):
            initial["pythonpath_entries"] = "\n".join(str(entry) for entry in pythonpath_entries if str(entry).strip())

        slurm_preset = str(submission.get("slurm_preset") or "").strip()
        if slurm_preset:
            initial["slurm_preset"] = slurm_preset
        if mode == RunMode.CHARGE_TRANSFER:
            # A single Multiwfn analysis is a direct serial job. Do not inherit
            # the source workflow's taskfarm/long preset accidentally.
            initial["slurm_preset"] = "custom"

        return initial

    @staticmethod
    def _slurm_presets_for_system(system_name: str) -> dict[str, object]:
        presets_config = getattr(settings, "REACTION_HUB_SUBMIT_SLURM_PRESETS", {}) or {}
        if not isinstance(presets_config, dict):
            presets_config = {}
        default_bundle = presets_config.get("__default__", {}) if isinstance(presets_config.get("__default__", {}), dict) else {}
        candidate = presets_config.get(system_name.strip().lower(), {})
        bundle = candidate if isinstance(candidate, dict) and candidate else default_bundle

        presets = bundle.get("presets", []) if isinstance(bundle.get("presets", []), list) else []
        normalized: list[dict[str, object]] = []
        for raw in presets:
            if not isinstance(raw, dict):
                continue
            preset_id = str(raw.get("id", "")).strip()
            label = str(raw.get("label", "")).strip()
            values = raw.get("values", {})
            if not preset_id or not label:
                continue
            if not isinstance(values, dict):
                values = {}
            normalized.append({"id": preset_id, "label": label, "values": values})

        if not any(p.get("id") == "custom" for p in normalized):
            normalized.insert(0, {"id": "custom", "label": "Custom (manual)", "values": {}})

        default_id = str(bundle.get("default", "custom")).strip() or "custom"
        valid_ids = {str(p["id"]) for p in normalized}
        if default_id not in valid_ids:
            default_id = "custom"

        return {"default": default_id, "presets": normalized}

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        form = context["form"]
        workflow_options = list(form.fields["workflow"].queryset)
        raw_from_run = str(self.request.GET.get("from_run") or "").strip()
        prefill = self._prefill_source_from_query()
        if prefill is not None:
            source_run = prefill["run"]
            artifact = prefill["artifact"]
            context["prefill_source"] = {
                "run_id": source_run.pk,
                "run_name": source_run.run_name,
                "workflow_name": source_run.workflow.name,
                "artifact_id": artifact.pk,
                "artifact_label": artifact.label,
                "artifact_path": artifact.path,
            }
            context["prefill_source_error"] = ""
        else:
            context["prefill_source"] = None
            context["prefill_source_error"] = (
                "Could not find a local structure artifact for the selected source run."
                if raw_from_run
                else ""
            )
        trh408_pythonpath = [
            str(entry).strip()
            for entry in getattr(settings, "REACTION_HUB_SUBMIT_TRH408_PYTHONPATH", [])
            if str(entry).strip()
        ]
        if "/srv/research/github/carmm" not in trh408_pythonpath:
            trh408_pythonpath.append("/srv/research/github/carmm")
        trh408_prefill = {
            "venv_activate_path": getattr(settings, "REACTION_HUB_SUBMIT_TRH408_VENV_ACTIVATE_PATH", ""),
            "pythonpath_entries": "\n".join(trh408_pythonpath),
            "module_loads": "\n".join(getattr(settings, "REACTION_HUB_SUBMIT_TRH408_MODULES", [])),
            "remote_runs_root": getattr(
                settings,
                "REACTION_HUB_SUBMIT_TRH408_REMOTE_RUNS_ROOT",
                "/home/{username}/reaction_hub_runs",
            ),
            "scratch_base": getattr(settings, "REACTION_HUB_SUBMIT_TRH408_SCRATCH_BASE", ""),
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "8",
            "account": "",
            "partition": "",
            "qos": "",
            "reservation": "",
        }
        falcon_prefill = {
            "venv_activate_path": getattr(settings, "REACTION_HUB_SUBMIT_FALCON_VENV_ACTIVATE_PATH", ""),
            "pythonpath_entries": "\n".join(getattr(settings, "REACTION_HUB_SUBMIT_FALCON_PYTHONPATH", [])),
            "module_loads": "\n".join(getattr(settings, "REACTION_HUB_SUBMIT_FALCON_MODULES", [])),
            "remote_runs_root": getattr(
                settings,
                "REACTION_HUB_SUBMIT_FALCON_REMOTE_RUNS_ROOT",
                "/shared/home1/$USER/reaction_hub_runs",
            ),
            "scratch_base": getattr(
                settings, "REACTION_HUB_SUBMIT_FALCON_SCRATCH_BASE", "/scratch/PROJECT/$USER/"
            ),
            "nodes": "1",
            "tasks_per_node": "192",
            "cpus_per_task": "1",
            "account": getattr(
                settings,
                "REACTION_HUB_SUBMIT_FALCON_DEFAULT_ACCOUNT",
                "FALCON_SLURM_ACCOUNT",
            ),
            "partition": "compute",
            "qos": "",
            "reservation": "",
            "walltime": "01:00:00",
        }
        hpc_prefill: dict[str, dict[str, str]] = {}
        for system in form.fields["hpc_system"].queryset:
            prefill: dict[str, str] = {}
            system_key = system.name.strip().lower()
            if system_key == "trh408":
                prefill.update(trh408_prefill)
            elif system_key == "falcon":
                prefill.update(falcon_prefill)
            mace_model_path = self._coadsorbate_mace_model_path_for_system(system.name)
            if mace_model_path:
                prefill["coadsorbate_mace_model_path"] = mace_model_path
                prefill["neb_mace_model_path"] = mace_model_path
            if prefill:
                hpc_prefill[str(system.pk)] = prefill
        context["hpc_prefill_json"] = json.dumps(hpc_prefill)
        slurm_presets: dict[str, dict[str, object]] = {}
        for system in form.fields["hpc_system"].queryset:
            slurm_presets[str(system.pk)] = self._slurm_presets_for_system(system.name)
        context["slurm_presets_json"] = json.dumps(slurm_presets)
        context["tddft_slurm_defaults_json"] = json.dumps(
            {
                "qos": getattr(settings, "REACTION_HUB_SUBMIT_TDDFT_DEFAULT_QOS", "long"),
                "walltime": getattr(settings, "REACTION_HUB_SUBMIT_TDDFT_DEFAULT_WALLTIME", "96:00:00"),
            }
        )
        settings_profile_backend: dict[str, str] = {}
        for profile in form.fields["settings_profile"].queryset:
            metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
            backend = str(metadata.get("qm_backend") or metadata.get("calculator") or "orca").strip().lower()
            if backend not in {"orca", "nwchem"}:
                backend = "orca"
            settings_profile_backend[str(profile.pk)] = backend
        context["settings_profile_backend_json"] = json.dumps(settings_profile_backend)
        workflow_mode_map, workflow_preferred_by_mode = self._workflow_mode_mappings(workflow_options)
        context["workflow_mode_map_json"] = json.dumps(workflow_mode_map)
        context["workflow_preferred_by_mode_json"] = json.dumps(workflow_preferred_by_mode)
        return context

    def form_valid(self, form):
        run_mode = form.cleaned_data["run_mode"]
        submission_structure_rel_path = _structure_artifact_rel_path_for_mode(run_mode)
        scan_config = form.scan_config() if run_mode == RunMode.TDDFT_BOND_SCAN else None
        single_constraints_config = (
            form.single_constraint_config() if run_mode == RunMode.SINGLE_OPTIMISATION else None
        )
        neb_config = form.neb_config() if run_mode == RunMode.ACTIVE_LEARNING_NEB else None
        coadsorbate_config = form.coadsorbate_config() if run_mode == RunMode.COADSORBATE_PLACEMENT else None
        charge_transfer_config = form.charge_transfer_config() or None
        module_loads, pythonpath_entries = form.submission_lists()
        is_batch_submit = bool(form.cleaned_data.get("submit_all_traj_frames"))
        tddft_wavelength_batch = [float(value) for value in (form.cleaned_data.get("target_wavelengths_nm") or [])]
        submitted_by = self.request.user.get_username() if self.request.user.is_authenticated else ""
        effective_neb_config = dict(neb_config or {}) if neb_config is not None else None
        if run_mode == RunMode.CHARGE_TRANSFER and charge_transfer_config is not None:
            source_run = WorkflowRun.objects.filter(
                pk=int(charge_transfer_config.get("source_run_id") or 0)
            ).first()
            if source_run is None:
                form.add_error(None, "The source workflow run for CT analysis no longer exists.")
                return self.form_invalid(form)
            source_working_directory = str(source_run.working_directory or "").strip()
            if not source_working_directory:
                form.add_error(None, "The source workflow run has no working directory containing ORCA files.")
                return self.form_invalid(form)
            latest_source_job = source_run.jobs.select_related("hpc_system").order_by("-pk").first()
            if (
                latest_source_job is not None
                and latest_source_job.hpc_system_id is not None
                and latest_source_job.hpc_system_id != form.cleaned_data["hpc_system"].pk
            ):
                form.add_error(
                    "hpc_system",
                    "Submit CT analysis on the same HPC system as the source calculation.",
                )
                return self.form_invalid(form)
            try:
                source_atoms, _ = _resolve_existing_run_structure_atoms(
                    source_run_id=source_run.pk,
                    source_artifact_id=int(form.cleaned_data.get("source_artifact_id") or 0),
                )
            except ValueError as exc:
                form.add_error(None, str(exc))
                return self.form_invalid(form)
            inferred_fragments = _infer_ct_fragment_defaults(source_atoms)
            if inferred_fragments:
                fragments_0based = inferred_fragments["fragments_0based"]
                charge_transfer_config = {
                    **charge_transfer_config,
                    "fragments_0based": fragments_0based,
                    "fragments_1based": {
                        key: _format_compact_indices([index + 1 for index in indices])
                        for key, indices in fragments_0based.items()
                    },
                    "fragment_labels": inferred_fragments["fragment_labels"],
                }
            charge_transfer_config = {
                **charge_transfer_config,
                "source_working_directory": source_working_directory,
                "source_run_uid": str(source_run.run_uid),
                "source_run_name": source_run.run_name,
            }

        def _submit_single_run(
            *,
            run_name: str,
            structure_text: str,
            structure_meta: dict[str, object],
            target_wavelength_nm: float | None,
        ):
            if charge_transfer_config is not None:
                atom_count = _safe_atom_count(structure_meta)
                if atom_count is None:
                    try:
                        atom_count = int(str(structure_text).splitlines()[0].strip())
                    except (IndexError, TypeError, ValueError):
                        atom_count = None
                selected_indices = [
                    int(value)
                    for values in charge_transfer_config.get("fragments_0based", {}).values()
                    for value in values
                ]
                if atom_count is not None and selected_indices and max(selected_indices) >= atom_count:
                    raise ValueError(
                        f"CT fragment index {max(selected_indices)} is out of range for a {atom_count}-atom structure."
                    )
            result = submit_slurm_workflow_run(
                workflow=form.cleaned_data["workflow"],
                hpc_system=form.cleaned_data["hpc_system"],
                settings_profile=form.cleaned_data["settings_profile"],
                run_name=run_name,
                structure_xyz=structure_text,
                remote_runs_root=form.cleaned_data["remote_runs_root"],
                scratch_base=form.cleaned_data["scratch_base"],
                account=form.cleaned_data["account"],
                partition=form.cleaned_data["partition"],
                qos=form.cleaned_data["qos"],
                reservation=form.cleaned_data["reservation"],
                walltime=form.cleaned_data["walltime"],
                nodes=form.cleaned_data["nodes"],
                tasks_per_node=form.cleaned_data["tasks_per_node"],
                cpus_per_task=form.cleaned_data["cpus_per_task"],
                module_loads=module_loads,
                venv_activate_path=form.cleaned_data["venv_activate_path"],
                pythonpath_entries=pythonpath_entries,
                fmax=form.cleaned_data["fmax"],
                active_learning_optimizer=form.cleaned_data["active_learning_optimizer"],
                qm_backend=form.cleaned_data.get("qm_backend", "orca"),
                target_wavelength_nm=target_wavelength_nm,
                spin_multiplicity=form.cleaned_data.get("spin_multiplicity"),
                system_charge=int(form.cleaned_data.get("system_charge") or 0),
                n_iter=form.cleaned_data["n_iter"],
                use_tddft_optimisation=form.cleaned_data["use_tddft_optimisation"],
                run_mode=run_mode,
                scan_config=scan_config,
                single_constraints_config=single_constraints_config,
                neb_config=effective_neb_config,
                coadsorbate_config=coadsorbate_config,
                charge_transfer_config=charge_transfer_config,
                slurm_preset=form.cleaned_data.get("slurm_preset", "custom"),
                submitted_by=submitted_by,
                structure_metadata=structure_meta,
            )
            run = (
                WorkflowRun.objects.get(pk=result.workflow_run_id)
                if result.workflow_run_id is not None
                else None
            )
            if run is None:
                return result, None
            run_metadata = dict(run.metadata or {})
            run_metadata["submission_result"] = {
                "scheduler_job_id": result.scheduler_job_id,
                "job_record_id": result.job_record_id,
                "remote_run_dir": result.remote_run_dir,
            }
            run.metadata = run_metadata
            return result, run

        coadsorbate_base_structure: tuple[str, dict[str, object]] | None = None
        neb_base_structure: tuple[str, dict[str, object]] | None = None
        if run_mode == RunMode.COADSORBATE_PLACEMENT:
            cluster = form.cleaned_data.get("coadsorbate_reference_cluster")
            try:
                adsorbate_payloads = _coadsorbate_adsorbate_payloads_from_cleaned_data(form.cleaned_data)
            except Exception as exc:
                form.add_error(None, str(exc))
                return self.form_invalid(form)
            if not adsorbate_payloads:
                form.add_error(None, "No adsorbates were resolved for co-adsorbate placement.")
                return self.form_invalid(form)
            use_reference_cluster_geometry = bool(
                form.cleaned_data.get("coadsorbate_use_reference_cluster_geometry")
            )
            try:
                if use_reference_cluster_geometry:
                    if cluster is None:
                        form.add_error(
                            "coadsorbate_reference_cluster",
                            "Reference cluster is required when geometry source is empty.",
                        )
                        return self.form_invalid(form)
                    cluster_atoms, cluster_meta = _resolve_reference_cluster_structure_atoms(
                        reference_cluster_id=int(cluster.pk),
                    )
                    coadsorbate_base_structure = (atoms_to_extxyz(cluster_atoms), dict(cluster_meta))
                else:
                    atoms, structure_meta = _structure_payload_to_atoms(form.geometry_source_payload())
                    coadsorbate_base_structure = (atoms_to_extxyz(atoms), dict(structure_meta))
            except Exception as exc:
                form.add_error(None, str(exc))
                return self.form_invalid(form)
            if coadsorbate_config is None:
                coadsorbate_config = {}
            coadsorbate_config = dict(coadsorbate_config)
            coadsorbate_config["adsorbates"] = [
                {"xyz": xyz_text, "metadata": metadata} for xyz_text, metadata in adsorbate_payloads
            ]

        if run_mode == RunMode.ACTIVE_LEARNING_NEB:
            try:
                initial_atoms, initial_meta = _structure_payload_to_atoms(form.geometry_source_payload())
            except Exception as exc:
                form.add_error(None, str(exc))
                return self.form_invalid(form)

            final_image_strategy = str(
                form.cleaned_data.get("neb_final_image_strategy") or "explicit_final_image"
            ).strip()
            final_meta: dict[str, object] = {}
            if final_image_strategy == "explicit_final_image":
                try:
                    final_atoms, final_meta = _structure_payload_to_atoms(form.neb_final_geometry_source_payload())
                except Exception as exc:
                    form.add_error(None, str(exc))
                    return self.form_invalid(form)

                initial_numbers = tuple(int(number) for number in initial_atoms.get_atomic_numbers())
                final_numbers = tuple(int(number) for number in final_atoms.get_atomic_numbers())
                if initial_numbers != final_numbers:
                    form.add_error(
                        None,
                        "Initial and final NEB images must have the same atom ordering and element sequence.",
                    )
                    return self.form_invalid(form)
            else:
                break_bond_i = form.cleaned_data.get("neb_break_bond_i")
                break_bond_j = form.cleaned_data.get("neb_break_bond_j")
                atom_count = int(len(initial_atoms))
                for value, label in ((break_bond_i, "A"), (break_bond_j, "B")):
                    if value is None:
                        form.add_error(None, f"Missing set-bond atom index {label} for generated NEB final image.")
                        return self.form_invalid(form)
                    if int(value) < 0 or int(value) >= atom_count:
                        form.add_error(
                            None,
                            f"Set-bond atom index {label}={int(value)} is out of range for {atom_count} atoms.",
                        )
                        return self.form_invalid(form)
                final_meta = {"source": "generated_from_initial_set_bond_distance"}

            initial_meta_payload = dict(initial_meta or {})
            initial_meta_payload["neb_final_structure"] = dict(final_meta or {})
            neb_base_structure = (atoms_to_extxyz(initial_atoms), initial_meta_payload)
            if effective_neb_config is None:
                effective_neb_config = {}
            effective_neb_config = dict(effective_neb_config)
            if final_image_strategy == "explicit_final_image":
                effective_neb_config["final_structure_extxyz"] = atoms_to_extxyz(final_atoms)
            effective_neb_config["final_structure_metadata"] = dict(final_meta or {})

        if run_mode == RunMode.TDDFT_BOND_SCAN and len(tddft_wavelength_batch) > 1:
            try:
                structure_text, structure_meta = _structure_payload_to_xyz(form.geometry_source_payload())
                structure_text = _structure_text_for_artifact_path(
                    structure_xyz=structure_text,
                    structure_meta=structure_meta if isinstance(structure_meta, dict) else {},
                    artifact_rel_path=submission_structure_rel_path,
                )
            except Exception as exc:
                form.add_error(None, str(exc))
                return self.form_invalid(form)

            base_name = str(form.cleaned_data["run_name"]).strip()
            successes: list[tuple[object, WorkflowRun]] = []
            failures: list[str] = []
            batch_size = len(tddft_wavelength_batch)
            batch_values = [float(value) for value in tddft_wavelength_batch]

            for batch_index, target_wavelength_nm in enumerate(tddft_wavelength_batch):
                run_name = f"{base_name}_{_format_wavelength_suffix_nm(target_wavelength_nm)}"
                metadata = dict(structure_meta or {})
                metadata["wavelength_batch_parent_run_name"] = base_name
                metadata["wavelength_batch_index"] = int(batch_index)
                metadata["wavelength_batch_size"] = int(batch_size)
                metadata["target_wavelength_nm"] = float(target_wavelength_nm)
                try:
                    result, run = _submit_single_run(
                        run_name=run_name,
                        structure_text=structure_text,
                        structure_meta=metadata,
                        target_wavelength_nm=float(target_wavelength_nm),
                    )
                except Exception as exc:
                    failures.append(f"{run_name}: {exc}")
                    continue

                submission_meta = run.metadata.get("submission", {})
                if not isinstance(submission_meta, dict):
                    submission_meta = {}
                submission_meta["wavelength_batch_parent_run_name"] = base_name
                submission_meta["wavelength_batch_index"] = int(batch_index)
                submission_meta["wavelength_batch_size"] = int(batch_size)
                submission_meta["wavelength_batch_mode"] = "target_wavelengths"
                submission_meta["target_wavelengths_nm"] = batch_values
                run.metadata["submission"] = submission_meta
                run.save()
                successes.append((result, run))

            if not successes:
                details = "; ".join(failures[:5]) if failures else "No wavelength jobs were submitted."
                form.add_error(None, f"Wavelength batch submission failed. {details}")
                return self.form_invalid(form)

            submitted_job_ids = [entry[0].scheduler_job_id for entry in successes]
            messages.success(
                self.request,
                (
                    f"Submitted {len(successes)}/{batch_size} wavelength jobs on "
                    f"{form.cleaned_data['hpc_system'].name}. "
                    f"Scheduler IDs: {', '.join(submitted_job_ids[:10])}"
                    f"{' ...' if len(submitted_job_ids) > 10 else ''}"
                ),
            )
            if failures:
                messages.warning(
                    self.request,
                    f"{len(failures)} wavelength jobs failed: {'; '.join(failures[:3])}"
                    f"{' ...' if len(failures) > 3 else ''}",
                )

            if len(successes) == 1:
                only_result = successes[0][0]
                if only_result.job_record_id is None:
                    only_run = successes[0][1]
                    return HttpResponseRedirect(reverse("workflow-run-detail", kwargs={"pk": only_run.pk}))
                return HttpResponseRedirect(reverse("job-detail", kwargs={"pk": only_result.job_record_id}))
            return HttpResponseRedirect(reverse("job-list"))

        if is_batch_submit:
            try:
                if str(form.cleaned_data.get("geometry_source") or "") == GeometrySource.EXISTING_RUN:
                    source_run = WorkflowRun.objects.filter(
                        pk=int(form.cleaned_data.get("source_run_id") or 0)
                    ).first()
                    if run_mode == RunMode.COADSORBATE_PLACEMENT:
                        frame_atoms = _resolve_existing_run_structure_frame_atoms(
                            source_run_id=int(form.cleaned_data.get("source_run_id") or 0),
                            source_artifact_id=int(form.cleaned_data.get("source_artifact_id") or 0),
                        )
                        frame_payloads = [(atoms_to_extxyz(atoms), metadata) for atoms, metadata in frame_atoms]
                    else:
                        frame_payloads = _resolve_existing_run_structure_frame_payloads(
                            source_run_id=int(form.cleaned_data.get("source_run_id") or 0),
                            source_artifact_id=int(form.cleaned_data.get("source_artifact_id") or 0),
                        )
                else:
                    source_run = None
                    if run_mode == RunMode.COADSORBATE_PLACEMENT:
                        frame_atoms = structure_file_to_atoms_frames(form.cleaned_data.get("structure_file"))
                        frame_payloads = [(atoms_to_extxyz(atoms), metadata) for atoms, metadata in frame_atoms]
                    else:
                        frame_payloads = structure_file_to_xyz_frames(form.cleaned_data.get("structure_file"))
            except Exception as exc:
                form.add_error(None, str(exc))
                return self.form_invalid(form)

            coadsorbate_hints = _coadsorbate_provenance_hints_from_run(source_run)
            base_name = str(form.cleaned_data["run_name"]).strip()
            successes: list[tuple[object, WorkflowRun]] = []
            failures: list[str] = []
            frame_count = len(frame_payloads)

            for frame_index, (structure_text, structure_meta) in enumerate(frame_payloads):
                run_name = f"{base_name}_{frame_index}"
                metadata = dict(structure_meta or {})
                metadata["batch_parent_run_name"] = base_name
                metadata["batch_index"] = frame_index
                metadata["batch_size"] = frame_count
                structure_text = _structure_text_for_artifact_path(
                    structure_xyz=structure_text,
                    structure_meta=metadata,
                    artifact_rel_path=submission_structure_rel_path,
                )
                if coadsorbate_hints:
                    metadata["adsorption_reference_hints"] = coadsorbate_hints
                    metadata["coadsorbate_provenance"] = coadsorbate_hints
                try:
                    result, run = _submit_single_run(
                        run_name=run_name,
                        structure_text=structure_text,
                        structure_meta=metadata,
                        target_wavelength_nm=form.cleaned_data.get("target_wavelength_nm"),
                    )
                except Exception as exc:
                    failures.append(f"{run_name}: {exc}")
                    continue

                if run is not None:
                    submission_meta = run.metadata.get("submission", {})
                    if not isinstance(submission_meta, dict):
                        submission_meta = {}
                    submission_meta["batch_parent_run_name"] = base_name
                    submission_meta["batch_index"] = frame_index
                    submission_meta["batch_size"] = frame_count
                    submission_meta["batch_mode"] = "trajectory_frames"
                    run.metadata["submission"] = submission_meta
                    run.save()
                successes.append((result, run))

            if not successes:
                details = "; ".join(failures[:5]) if failures else "No frames were submitted."
                form.add_error(None, f"Batch submission failed. {details}")
                return self.form_invalid(form)

            submitted_job_ids = [entry[0].scheduler_job_id for entry in successes]
            messages.success(
                self.request,
                (
                    f"Submitted {len(successes)}/{frame_count} frames on "
                    f"{form.cleaned_data['hpc_system'].name}. "
                    f"Scheduler IDs: {', '.join(submitted_job_ids[:10])}"
                    f"{' ...' if len(submitted_job_ids) > 10 else ''}"
                ),
            )
            if failures:
                messages.warning(
                    self.request,
                    f"{len(failures)} frames failed: {'; '.join(failures[:3])}{' ...' if len(failures) > 3 else ''}",
                )

            if len(successes) == 1:
                only_result = successes[0][0]
                if only_result.job_record_id is None:
                    only_run = successes[0][1]
                    if only_run is not None:
                        return HttpResponseRedirect(reverse("workflow-run-detail", kwargs={"pk": only_run.pk}))
                    return HttpResponseRedirect(reverse("job-list"))
                return HttpResponseRedirect(reverse("job-detail", kwargs={"pk": only_result.job_record_id}))
            return HttpResponseRedirect(reverse("job-list"))

        try:
            if coadsorbate_base_structure is not None:
                structure_text, structure_meta = coadsorbate_base_structure
            elif neb_base_structure is not None:
                structure_text, structure_meta = neb_base_structure
            elif run_mode == RunMode.COADSORBATE_PLACEMENT:
                atoms, structure_meta = _structure_payload_to_atoms(form.geometry_source_payload())
                structure_text = atoms_to_extxyz(atoms)
            else:
                structure_text, structure_meta = _structure_payload_to_xyz(form.geometry_source_payload())
            structure_text = _structure_text_for_artifact_path(
                structure_xyz=structure_text,
                structure_meta=structure_meta if isinstance(structure_meta, dict) else {},
                artifact_rel_path=submission_structure_rel_path,
            )
            result, run = _submit_single_run(
                run_name=form.cleaned_data["run_name"],
                structure_text=structure_text,
                structure_meta=structure_meta,
                target_wavelength_nm=form.cleaned_data.get("target_wavelength_nm"),
            )
        except Exception as exc:
            form.add_error(None, str(exc))
            return self.form_invalid(form)

        reference_message = ""
        if run is not None:
            try:
                reference_message = _apply_reference_selection_to_run(
                    run=run,
                    reference_intent=str(form.cleaned_data.get("reference_intent") or ReferenceIntent.NONE),
                    reference_name=str(form.cleaned_data.get("reference_name") or run.run_name).strip() or run.run_name,
                    structure_meta=structure_meta,
                )
            except Exception as exc:
                messages.warning(self.request, f"Job submitted, but reference tagging failed: {exc}")

            submission_meta = run.metadata.get("submission", {})
            if not isinstance(submission_meta, dict):
                submission_meta = {}
            submission_meta["reference_intent"] = str(form.cleaned_data.get("reference_intent") or ReferenceIntent.NONE)
            submission_meta["reference_name"] = str(form.cleaned_data.get("reference_name") or "").strip()
            run.metadata["submission"] = submission_meta
            run.save()

        messages.success(
            self.request,
            f"Submitted job {result.scheduler_job_id} on {form.cleaned_data['hpc_system'].name}.",
        )
        if reference_message:
            messages.success(self.request, reference_message)
        if result.job_record_id is None:
            return HttpResponseRedirect(reverse("job-list"))
        return HttpResponseRedirect(reverse("job-detail", kwargs={"pk": result.job_record_id}))


def _traj_frame_from_request(raw: str | None) -> int:
    if raw is None:
        return -1
    try:
        return int(str(raw).strip())
    except ValueError:
        return -1


def _optional_int_from_request(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _optional_float_from_request(raw: str | None) -> float | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _bool_from_request(raw: str | None) -> bool:
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _generated_neb_preview_atoms_from_request(
    request,
    *,
    initial_atoms,
    initial_metadata: dict[str, object],
):
    run_mode = str(request.POST.get("run_mode") or "").strip()
    if run_mode != RunMode.ACTIVE_LEARNING_NEB:
        return initial_atoms, initial_metadata

    strategy = str(request.POST.get("neb_final_image_strategy") or "").strip().lower()
    if strategy == "bond_extension_from_initial":
        strategy = "set_bond_distance_from_initial"
    if strategy != "set_bond_distance_from_initial":
        return initial_atoms, initial_metadata

    bond_i = _optional_int_from_request(request.POST.get("neb_break_bond_i"))
    bond_j = _optional_int_from_request(request.POST.get("neb_break_bond_j"))
    if bond_i is None or bond_j is None:
        raise ValueError("Generated final-image preview requires both bond atom indices.")
    if bond_i == bond_j:
        raise ValueError("Generated final-image preview requires two different bond atom indices.")

    atom_count = len(initial_atoms)
    for value, label in ((bond_i, "A"), (bond_j, "B")):
        if int(value) < 0 or int(value) >= atom_count:
            raise ValueError(
                f"Generated final-image preview atom index {label}={int(value)} is out of range for {atom_count} atoms."
            )

    target_distance = _optional_float_from_request(request.POST.get("neb_set_bond_distance_A"))
    if target_distance is None:
        target_distance = _optional_float_from_request(request.POST.get("neb_break_bond_extension_A"))
    if target_distance is None or float(target_distance) <= 0.0:
        raise ValueError("Generated final-image preview requires positive target bond distance.")

    set_distance_fix = _optional_float_from_request(request.POST.get("neb_break_bond_fix"))
    if set_distance_fix is None:
        set_distance_fix = 0.5
    if float(set_distance_fix) < 0.0 or float(set_distance_fix) > 1.0:
        raise ValueError("Generated final-image preview set_distance fix must be in [0.0, 1.0].")

    cutoff_scale = _optional_float_from_request(request.POST.get("neb_break_cutoff_scale"))
    if cutoff_scale is None or float(cutoff_scale) <= 0.0:
        cutoff_scale = 1.2

    generated = initial_atoms.copy()
    mic = bool(any(bool(flag) for flag in generated.get_pbc().tolist()))
    distance_before = float(generated.get_distance(int(bond_i), int(bond_j), mic=mic))
    generated.set_distance(
        int(bond_i),
        int(bond_j),
        float(target_distance),
        fix=float(set_distance_fix),
        mic=mic,
    )
    distance_after = float(generated.get_distance(int(bond_i), int(bond_j), mic=mic))
    radii_sum = float(covalent_radii[int(generated[int(bond_i)].number)] + covalent_radii[int(generated[int(bond_j)].number)])
    cutoff = float(cutoff_scale) if radii_sum <= 0.0 else float(cutoff_scale) * radii_sum
    bond_present_after = bool(distance_after <= cutoff)
    assert_absent = _bool_from_request(request.POST.get("neb_break_assert_broken"))

    metadata = dict(initial_metadata or {})
    metadata.update(
        {
            "source": "generated_set_bond_distance_preview",
            "source_generation": "neb_set_bond_distance",
            "set_bond_i": int(bond_i),
            "set_bond_j": int(bond_j),
            "set_bond_distance_A": float(target_distance),
            "set_distance_fix": float(set_distance_fix),
            "distance_before_set_A": float(distance_before),
            "distance_after_set_A": float(distance_after),
            "bond_cutoff_A": float(cutoff),
            "bond_present_after_set": bool(bond_present_after),
            "assert_absent_requested": bool(assert_absent),
        }
    )
    return generated, metadata


def _geometry_payload_from_request(request) -> dict[str, object]:
    return {
        "geometry_source": str(request.POST.get("geometry_source") or GeometrySource.UPLOAD_FILE),
        "structure_file": request.FILES.get("structure_file"),
        "traj_frame": _traj_frame_from_request(request.POST.get("traj_frame")),
        "ase_molecule_formula": str(request.POST.get("ase_molecule_formula") or ""),
        "pubchem_cid": _optional_int_from_request(request.POST.get("pubchem_cid")),
        "pubchem_query": str(request.POST.get("pubchem_query") or ""),
        "source_run_id": _optional_int_from_request(request.POST.get("source_run_id")),
        "source_artifact_id": _optional_int_from_request(request.POST.get("source_artifact_id")),
    }


@require_POST
def structure_preview_view(request):
    geometry_payload = _geometry_payload_from_request(request)
    try:
        atoms, metadata = _structure_payload_to_atoms(geometry_payload)
        atoms, metadata = _generated_neb_preview_atoms_from_request(
            request,
            initial_atoms=atoms,
            initial_metadata=metadata if isinstance(metadata, dict) else {},
        )
        xyz_text = atoms_to_xyz(atoms)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": f"Unable to parse structure: {exc}"}, status=400)

    max_atoms = 600
    rows: list[dict[str, object]] = []
    for idx, atom in enumerate(atoms):
        if idx >= max_atoms:
            break
        x, y, z = atom.position
        rows.append(
            {
                "index": idx,
                "symbol": atom.symbol,
                "x": round(float(x), 6),
                "y": round(float(y), 6),
                "z": round(float(z), 6),
            }
        )

    xyz_preview = xyz_text
    xyz_preview_truncated = False
    if len(xyz_preview) > 20000:
        xyz_preview = xyz_preview[:20000]
        xyz_preview_truncated = True

    element_counts = Counter(atom.symbol for atom in atoms)
    ct_fragments = _infer_ct_fragment_defaults(atoms)
    ct_nroots = None
    source_run_id = metadata.get("source_run_id") if isinstance(metadata, dict) else None
    if source_run_id is not None:
        source_run = WorkflowRun.objects.filter(pk=source_run_id).first()
        if source_run is not None:
            ct_nroots = _tddft_nroots_for_run(source_run)
    element_max_z: dict[str, float] = {}
    global_max_z: float | None = None
    for atom in atoms:
        symbol = str(atom.symbol)
        z_value = float(atom.z)
        if symbol not in element_max_z or z_value > element_max_z[symbol]:
            element_max_z[symbol] = z_value
        if global_max_z is None or z_value > global_max_z:
            global_max_z = z_value

    return JsonResponse(
        {
            "ok": True,
            "metadata": metadata,
            "element_counts": dict(sorted(element_counts.items())),
            "ct_fragments": ct_fragments,
            "ct_nroots": ct_nroots,
            "element_max_z": {
                symbol: round(z_value, 6)
                for symbol, z_value in sorted(element_max_z.items())
            },
            "global_max_z": None if global_max_z is None else round(global_max_z, 6),
            "atoms": rows,
            "atoms_truncated": len(atoms) > max_atoms,
            "xyz_preview": xyz_preview,
            "xyz_preview_truncated": xyz_preview_truncated,
        }
    )


@require_POST
def structure_open_ase_gui_view(request):
    geometry_payload = _geometry_payload_from_request(request)
    try:
        atoms, metadata = _structure_payload_to_atoms(geometry_payload)
        atoms, metadata = _generated_neb_preview_atoms_from_request(
            request,
            initial_atoms=atoms,
            initial_metadata=metadata if isinstance(metadata, dict) else {},
        )
        structure_text = atoms_to_extxyz(atoms)
    except Exception as exc:
        return JsonResponse({"ok": False, "error": f"Unable to parse structure: {exc}"}, status=400)

    temp_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".extxyz",
            prefix="reaction_hub_preview_",
            delete=False,
        ) as handle:
            handle.write(structure_text)
            temp_path = handle.name

        command = resolve_ase_gui_command(temp_path)

        err_file = tempfile.TemporaryFile()
        try:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=subprocess.DEVNULL,
                stderr=err_file,
                start_new_session=True,
            )
        except Exception:
            err_file.close()
            raise

        immediate_code = None
        if hasattr(process, "wait") and callable(process.wait):
            try:
                immediate_code = process.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                immediate_code = None
            except TypeError:
                immediate_code = process.poll()
        else:
            immediate_code = process.poll()

        if immediate_code not in (None, 0):
            err_file.seek(0)
            err_msg = err_file.read().decode("utf-8", errors="replace").strip()
            err_file.close()
            details = f": {err_msg}" if err_msg else f" with status {immediate_code}."
            raise RuntimeError(f"ASE GUI command exited immediately{details}")
        err_file.close()
        return JsonResponse(
            {
                "ok": True,
                "message": (
                    f"ASE GUI launch command started for {metadata['formula']} ({metadata['atom_count']} atoms). "
                    f"Temporary file: {temp_path}"
                ),
            }
        )
    except Exception as exc:
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except OSError:
                pass
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    "Failed to launch ASE GUI. Check local GUI/display support and "
                    "REACTION_HUB_ASE_GUI_COMMAND setting. "
                    f"Details: {exc}"
                ),
            },
            status=400,
        )
