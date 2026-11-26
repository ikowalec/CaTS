from __future__ import annotations

import base64
from datetime import timezone as dt_timezone
import hashlib
import json
import math
import logging
import re
import shlex
import socket
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from ase.db import connect as ase_connect
from ase.io import read as ase_read
from ase.io import write as ase_write
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from core.xyzrender import (
    build_xyzrender_subprocess_env,
    cairo_runtime_install_hint,
    is_xyzrender_cairo_error,
    normalize_gif_canvas,
    parse_canvas_size,
    resolve_xyzrender_command,
)
from jobs.models import HPCSystem, JobRecord, JobStatus
from references.models import CalculationSettingsProfile
from results.models import Artifact, ArtifactKind, ComputedProperty
from results.services import upsert_adsorption_energy_for_run
from workflows.identity import canonical_run_path, deterministic_run_uuid, normalize_run_path
from workflows.models import WorkflowDefinition, WorkflowRun, WorkflowStatus

from .adapters.cplas_adapter import collect_cplas_artifacts, discover_cplas_tddft_runs
from .hpc import SlurmSSHClient
from .spectral import spectrum_curve_from_csv, update_spectral_dataset, write_ct_plot_artifacts

_REACTION_HUB_SUMMARY_FILENAME = "reaction_hub_summary.json"
_SCAN_ENDPOINTS_DB_FILENAME = "scan_endpoints.db"
_HARTREE_TO_EV = 27.211386245988
_BOHR_TO_ANGSTROM = 0.529177210903
_GRADIENT_TO_FORCE_EV_A = _HARTREE_TO_EV / _BOHR_TO_ANGSTROM

_FINAL_ENERGY_RE = re.compile(
    r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"
)
_CHARGE_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]{1,3})\s*:\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"
)
_GRADIENT_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z]{1,3})\s*:\s*"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)\s+"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)\s*$"
)

_KEY_PLOT_FILENAMES = {
    "stacked_uvvis.png": "plot:stacked_uvvis",
    "stacked_ct.png": "plot:stacked_ct",
    "energy_vs_reaction_coordinate.png": "plot:energy_vs_reaction_coordinate",
    "converged_state_wavelengths.png": "plot:converged_state_wavelengths",
    "final_tddft_spectrum.png": "plot:final_tddft_spectrum",
    "cluster_tddft_spectrum.png": "plot:cluster_tddft_spectrum",
    "cluster_adsorbate_difference_spectrum.png": "plot:cluster_adsorbate_difference_spectrum",
}


def _should_skip_scan_plot_image(path_value: str) -> bool:
    return Path(str(path_value or "")).name.startswith("uvvis_iter_")


def _is_legacy_nested_cluster_tddft_spectrum_artifact(path_value: str) -> bool:
    normalized = normalize_run_path(path_value).lower()
    name = Path(normalized).name
    if name not in {"cluster_tddft_spectrum.csv", "cluster_tddft_spectrum.png"}:
        return False
    return "/final_tddft_spectrum/cluster_tddft/" in normalized


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


def _xyzrender_gif_canvas_size() -> tuple[int, int] | None:
    width = _parse_optional_int(getattr(settings, "REACTION_HUB_XYZRENDER_GIF_WIDTH", None))
    height = _parse_optional_int(getattr(settings, "REACTION_HUB_XYZRENDER_GIF_HEIGHT", None))
    if width is not None and height is not None and width > 0 and height > 0:
        return int(width), int(height)
    return parse_canvas_size(getattr(settings, "REACTION_HUB_XYZRENDER_GIF_SIZE", "900x900"))


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


def _source_scheduler_job_id_from_metadata(
    metadata: dict[str, Any] | None,
    *,
    path_hint: str = "",
    fallback: str = "",
) -> str:
    meta = metadata if isinstance(metadata, dict) else {}
    for key in ("source_scheduler_job_id", "scheduler_job_id", "job_id"):
        value = str(meta.get(key) or "").strip()
        if value:
            return value
    for candidate in (
        meta.get("ingestion_path"),
        meta.get("remote_path"),
        meta.get("local_cache_path"),
        path_hint,
    ):
        extracted = _extract_scheduler_job_id_from_path(candidate)
        if extracted:
            return extracted
    return str(fallback or "").strip()


def _artifact_source_scheduler_job_id_for_ingest(
    artifact: Artifact,
    *,
    fallback: str = "",
) -> str:
    if artifact.job is not None:
        scheduler_job_id = str(artifact.job.scheduler_job_id or "").strip()
        if scheduler_job_id:
            return scheduler_job_id
    return _source_scheduler_job_id_from_metadata(
        artifact.metadata if isinstance(artifact.metadata, dict) else {},
        path_hint=str(artifact.path or ""),
        fallback=fallback,
    )


def _artifact_ingestion_path_for_ingest(artifact: Artifact) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    ingestion_path = normalize_run_path(str(metadata.get("ingestion_path") or "").strip())
    if ingestion_path:
        return ingestion_path
    remote_path = normalize_run_path(str(metadata.get("remote_path") or "").strip())
    if remote_path:
        return remote_path
    return normalize_run_path(str(artifact.path or "").strip())


def _normalize_ingestion_path_for_source(path_text: str, source_scheduler_job_id: str = "") -> str:
    normalized = normalize_run_path(path_text)
    if not normalized:
        return ""
    parts = [part for part in normalized.split("/") if part]
    source_id = str(source_scheduler_job_id or "").strip()
    if source_id and parts and parts[0] == source_id:
        parts = parts[1:]
    elif parts and parts[0].isdigit() and len(parts[0]) >= 4:
        # Normalize mixed layouts where files may appear under <job_id>/... or directly in run root.
        parts = parts[1:]
    return "/".join(parts)


def _artifact_semantic_key_for_ingest(
    *,
    label: str,
    ingestion_path: str,
    source_scheduler_job_id: str,
) -> tuple[str, str, str] | None:
    label_key = str(label or "").strip().casefold()
    source_id = str(source_scheduler_job_id or "").strip()
    if not label_key or not source_id:
        return None
    normalized_path = _normalize_ingestion_path_for_source(ingestion_path, source_id).casefold()
    if not normalized_path:
        return None
    return (label_key, source_id, normalized_path)


def _latest_scheduler_job_id_for_run(run: WorkflowRun) -> str:
    scheduler_job_id = (
        JobRecord.objects.filter(workflow_run=run)
        .exclude(scheduler_job_id__isnull=True)
        .exclude(scheduler_job_id="")
        .order_by("-created_at", "-pk")
        .values_list("scheduler_job_id", flat=True)
        .first()
    )
    return str(scheduler_job_id or "").strip()


def _run_mode_for_ingest(run: WorkflowRun) -> str:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission")
    if not isinstance(submission, dict):
        return ""
    return str(submission.get("run_mode") or "").strip()


def _is_ground_state_mulliken_only_run(run: WorkflowRun) -> bool:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission")
    if isinstance(submission, dict):
        config = submission.get("charge_transfer_config")
        if isinstance(config, dict) and bool(config.get("ground_state_mulliken_only")):
            return True
    summary = metadata.get("result_summary")
    charge_transfer = summary.get("charge_transfer") if isinstance(summary, dict) else None
    return bool(
        isinstance(charge_transfer, dict)
        and str(charge_transfer.get("analysis_scope") or "").strip().lower()
        == "ground_state_mulliken"
    )


def _is_density_analysis_only_run(run: WorkflowRun) -> bool:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission")
    if isinstance(submission, dict):
        config = submission.get("charge_transfer_config")
        if isinstance(config, dict) and bool(config.get("density_analysis_only")):
            return True
    summary = metadata.get("result_summary")
    if isinstance(summary, dict) and summary.get("run_type") in ("density_analysis", "ts_vibrations_gradient"):
        return True
    return False


def _is_single_optimisation_run(run: WorkflowRun) -> bool:
    return _run_mode_for_ingest(run) == "single_optimisation"


def _single_optimisation_final_tddft_spectrum_requested(run: WorkflowRun) -> bool:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission")
    if not isinstance(submission, dict):
        return False
    if str(submission.get("run_mode") or "").strip() != "single_optimisation":
        return False
    single_config = submission.get("single_constraints_config")
    if not isinstance(single_config, dict):
        return False
    return bool(single_config.get("final_tddft_spectrum"))


def _is_coadsorbate_screening_run(run: WorkflowRun) -> bool:
    run_mode = _run_mode_for_ingest(run)
    if run_mode == "coadsorbate_placement":
        return True
    code_path = str(getattr(run.workflow, "code_path", "") or "").strip().lower()
    return "coadsorbate" in code_path


def _single_optimisation_path_penalty(run_dir_text: str, remote_path: str) -> tuple[int, int, int]:
    rel_path = normalize_run_path(_relative_artifact_path(run_dir_text, remote_path))
    parts = [part for part in rel_path.split("/") if part]
    in_restart_backup = any(part.startswith("restart_backup_") for part in parts)
    starts_with_job_id = bool(parts and parts[0].isdigit())
    return (
        int(in_restart_backup),
        int(starts_with_job_id),
        len(parts),
    )


def _evaluated_traj_effective_suffix(run_dir_text: str, remote_path: str) -> str:
    rel_path = normalize_run_path(_relative_artifact_path(run_dir_text, remote_path))
    parts = [part for part in rel_path.split("/") if part]
    while parts and (parts[0].isdigit() or parts[0].startswith("restart_backup_")):
        parts = parts[1:]
    return "/".join(parts)


def _evaluated_traj_keeper_score(artifact: Artifact, *, run_dir_text: str) -> tuple[int, int, int, int, object, int]:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    ingestion_path = _artifact_ingestion_path_for_ingest(artifact)
    penalty = _single_optimisation_path_penalty(run_dir_text, ingestion_path)
    has_local_cache = int(bool(str(metadata.get("local_cache_path") or "").strip()))
    recency = artifact.updated_at or artifact.created_at
    return (
        -int(penalty[0]),
        -int(penalty[1]),
        -int(penalty[2]),
        has_local_cache,
        recency,
        int(artifact.pk or 0),
    )


def _scan_endpoints_source_rank_for_ingest(
    remote_path: str,
    *,
    expected_source_scheduler_job_id: str = "",
) -> int:
    expected_source_id = str(expected_source_scheduler_job_id or "").strip()
    if not expected_source_id:
        return 1
    extracted_source_id = _extract_scheduler_job_id_from_path(remote_path)
    if extracted_source_id == expected_source_id:
        return 2
    if not extracted_source_id:
        return 1
    return 0


def _scan_endpoints_profile_selection_score(
    *,
    run_dir_text: str,
    remote_path: str,
    point_count: int,
    expected_source_scheduler_job_id: str = "",
    order_index: int = 0,
) -> tuple[int, int, int, int, int, int, str]:
    path_penalty = _single_optimisation_path_penalty(run_dir_text, remote_path)
    source_rank = _scan_endpoints_source_rank_for_ingest(
        remote_path,
        expected_source_scheduler_job_id=expected_source_scheduler_job_id,
    )
    return (
        int(source_rank),
        int(point_count),
        -int(path_penalty[0]),
        -int(path_penalty[1]),
        -int(path_penalty[2]),
        -int(order_index),
        normalize_run_path(remote_path),
    )


def _prune_evaluated_traj_duplicates_for_source_job(
    run: WorkflowRun,
    *,
    source_scheduler_job_id: str,
    keep_artifact_id: int,
) -> int:
    source_id = str(source_scheduler_job_id or "").strip()
    if not source_id:
        return 0

    removed = 0
    for artifact in (
        Artifact.objects.filter(
            workflow_run=run,
            label="trajectory:evaluated.traj",
        )
        .select_related("job")
        .order_by("pk")
    ):
        if int(artifact.pk or 0) == int(keep_artifact_id):
            continue
        artifact_source_id = _artifact_source_scheduler_job_id_for_ingest(
            artifact,
            fallback=source_id,
        )
        if artifact_source_id != source_id:
            continue
        artifact.delete()
        removed += 1
    return removed


def _prune_evaluated_traj_duplicates_for_all_sources(run: WorkflowRun, *, run_dir_text: str) -> int:
    grouped: dict[str, list[Artifact]] = {}
    for artifact in (
        Artifact.objects.filter(
            workflow_run=run,
            label="trajectory:evaluated.traj",
        )
        .select_related("job")
        .order_by("pk")
    ):
        source_id = _artifact_source_scheduler_job_id_for_ingest(artifact, fallback="")
        if not source_id:
            continue
        grouped.setdefault(source_id, []).append(artifact)

    removed = 0
    for members in grouped.values():
        if len(members) < 2:
            continue
        keeper = max(
            members,
            key=lambda item: _evaluated_traj_keeper_score(item, run_dir_text=run_dir_text),
        )
        for artifact in members:
            if int(artifact.pk or 0) == int(keeper.pk or 0):
                continue
            artifact.delete()
            removed += 1
    return removed


def _is_scan_endpoints_path(path_text: str) -> bool:
    return Path(str(path_text or "").strip()).name.lower() == _SCAN_ENDPOINTS_DB_FILENAME


def _scan_endpoints_db_keeper_score(
    run: WorkflowRun,
    artifact: Artifact,
    *,
    run_dir_text: str,
    expected_source_scheduler_job_id: str = "",
) -> tuple[int, int, int, int, int, int, object, int]:
    ingestion_path = _artifact_ingestion_path_for_ingest(artifact)
    local_path = _artifact_local_path_for_ingest(run, artifact)
    point_count = 0
    if local_path is not None:
        profile = _parse_scan_profile_from_db(local_path)
        points = profile.get("points")
        if isinstance(points, list):
            point_count = len(points)
    profile_score = _scan_endpoints_profile_selection_score(
        run_dir_text=run_dir_text,
        remote_path=ingestion_path,
        point_count=point_count,
        expected_source_scheduler_job_id=expected_source_scheduler_job_id,
        order_index=0,
    )
    mtime_ns = 0
    if local_path is not None:
        try:
            mtime_ns = int(local_path.stat().st_mtime_ns)
        except OSError:
            mtime_ns = 0
    recency = artifact.updated_at or artifact.created_at
    return (
        int(profile_score[0]),
        int(profile_score[1]),
        int(profile_score[2]),
        int(profile_score[3]),
        int(profile_score[4]),
        int(mtime_ns),
        recency,
        int(artifact.pk or 0),
    )


def _prune_scan_endpoints_db_duplicates_for_run(
    run: WorkflowRun,
    *,
    run_dir_text: str,
    keep_artifact_id: int | None = None,
) -> int:
    candidates = [
        artifact
        for artifact in (
            Artifact.objects.filter(workflow_run=run, kind=ArtifactKind.DATABASE)
            .select_related("job")
            .order_by("pk")
        )
        if _is_scan_endpoints_path(str(artifact.path or ""))
        or _is_scan_endpoints_path(_artifact_ingestion_path_for_ingest(artifact))
    ]
    if len(candidates) < 2:
        return 0

    keeper: Artifact | None = None
    if keep_artifact_id is not None:
        for artifact in candidates:
            if int(artifact.pk or 0) == int(keep_artifact_id):
                keeper = artifact
                break
    if keeper is None:
        expected_source_scheduler_job_id = _latest_scheduler_job_id_for_run(run)
        keeper = max(
            candidates,
            key=lambda item: _scan_endpoints_db_keeper_score(
                run,
                item,
                run_dir_text=run_dir_text,
                expected_source_scheduler_job_id=expected_source_scheduler_job_id,
            ),
        )

    removed = 0
    keeper_id = int(keeper.pk or 0)
    for artifact in candidates:
        if int(artifact.pk or 0) == keeper_id:
            continue
        artifact.delete()
        removed += 1
    return removed


def _default_settings_profile() -> CalculationSettingsProfile:
    profile, _ = CalculationSettingsProfile.objects.get_or_create(
        name="default_pbe0_water",
        defaults={
            "approved": True,
            "xc_functional": "PBE0",
            "basis_set": "6-31g(d,p)",
            "solvent": "Water",
        },
    )
    return profile


def _infer_run_status(run_dir: Path) -> str:
    if list(run_dir.rglob("error.log")):
        return WorkflowStatus.FAILED
    markers = list(run_dir.rglob("iteration_complete.txt"))
    if markers:
        return WorkflowStatus.COMPLETED
    if list(run_dir.rglob("orca.out")):
        return WorkflowStatus.RUNNING
    return WorkflowStatus.INITIATED


def _artifact_kind_from_text(kind_text: str) -> str:
    valid = {choice for choice, _label in ArtifactKind.choices}
    if kind_text in valid:
        return kind_text
    return ArtifactKind.OTHER


def _merge_ingest_metadata(existing_metadata: dict | None, run_dir_text: str) -> dict:
    merged = dict(existing_metadata or {})
    merged["ingested_from"] = "cplas_adapter"
    merged["path"] = run_dir_text
    return merged


def _parse_final_energy_hartree_from_text(text: str) -> float | None:
    matches = list(_FINAL_ENERGY_RE.finditer(text))
    if not matches:
        return None
    try:
        return float(matches[-1].group(1))
    except ValueError:
        return None


def _parse_final_mulliken_charges_from_text(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    blocks: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(lines):
        line = lines[idx].upper()
        if "MULLIKEN ATOMIC CHARGES" not in line:
            idx += 1
            continue

        block: list[dict[str, Any]] = []
        idx += 1
        while idx < len(lines):
            match = _CHARGE_LINE_RE.match(lines[idx])
            if match:
                block.append(
                    {
                        "index": int(match.group(1)),
                        "symbol": str(match.group(2)),
                        "charge": float(match.group(3)),
                    }
                )
                idx += 1
                continue
            if block:
                break
            idx += 1

        if block:
            blocks.append(block)

    if not blocks:
        return []
    return blocks[-1]


def _parse_final_cartesian_gradient_from_text(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    blocks: list[list[dict[str, Any]]] = []
    idx = 0
    while idx < len(lines):
        if "CARTESIAN GRADIENT" not in lines[idx].upper():
            idx += 1
            continue

        block: list[dict[str, Any]] = []
        idx += 1
        while idx < len(lines):
            match = _GRADIENT_LINE_RE.match(lines[idx])
            if match:
                block.append(
                    {
                        "index": int(match.group(1)),
                        "symbol": str(match.group(2)),
                        "gradient": [
                            float(match.group(3)),
                            float(match.group(4)),
                            float(match.group(5)),
                        ],
                    }
                )
                idx += 1
                continue
            if block:
                break
            idx += 1

        if block:
            blocks.append(block)

    if not blocks:
        return []
    return blocks[-1]


def _parse_orca_output_details(orca_out_path: Path) -> dict[str, Any]:
    try:
        text = orca_out_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    return _parse_orca_output_details_from_text(text)


def _parse_orca_output_details_from_text(text: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    energy_h = _parse_final_energy_hartree_from_text(text)
    if energy_h is not None:
        payload["energy_hartree"] = float(energy_h)
        payload["energy_eV"] = float(energy_h * _HARTREE_TO_EV)

    charges = _parse_final_mulliken_charges_from_text(text)
    if charges:
        payload["mulliken_charges"] = charges

    gradient = _parse_final_cartesian_gradient_from_text(text)
    if gradient:
        forces_eh_bohr = []
        forces_eV_A = []
        for row in gradient:
            gx, gy, gz = row["gradient"]
            fx_h, fy_h, fz_h = -float(gx), -float(gy), -float(gz)
            forces_eh_bohr.append(
                {"index": row["index"], "symbol": row["symbol"], "force": [fx_h, fy_h, fz_h]}
            )
            forces_eV_A.append(
                {
                    "index": row["index"],
                    "symbol": row["symbol"],
                    "force": [
                        fx_h * _GRADIENT_TO_FORCE_EV_A,
                        fy_h * _GRADIENT_TO_FORCE_EV_A,
                        fz_h * _GRADIENT_TO_FORCE_EV_A,
                    ],
                }
            )
        payload["forces_hartree_bohr"] = forces_eh_bohr
        payload["forces_eV_A"] = forces_eV_A

    return payload


def _parse_json_payload(raw_text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(raw_text[start : end + 1])
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _scan_profile_from_points(points: list[dict[str, Any]]) -> dict[str, Any]:
    if not points:
        return {}
    sorted_points = sorted(points, key=lambda item: int(item.get("iteration", 0)))
    start = sorted_points[0]
    peak = max(sorted_points, key=lambda item: float(item.get("energy_eV", 0.0)))
    start_energy = float(start.get("energy_eV", 0.0))
    peak_energy = float(peak.get("energy_eV", 0.0))
    profile: dict[str, Any] = {
        "points": sorted_points,
        "start_iteration": int(start.get("iteration", 1)),
        "start_energy_eV": start_energy,
        "peak_iteration": int(peak.get("iteration", 1)),
        "peak_energy_eV": peak_energy,
        "activation_barrier_eV": float(peak_energy - start_energy),
    }
    if "bond_length_A" in start:
        try:
            profile["start_bond_length_A"] = float(start["bond_length_A"])
        except (TypeError, ValueError):
            pass
    if "bond_length_A" in peak:
        try:
            profile["peak_bond_length_A"] = float(peak["bond_length_A"])
        except (TypeError, ValueError):
            pass
    return profile


def _row_lookup(row: Any, key: str):
    key_values = dict(getattr(row, "key_value_pairs", {}) or {})
    if key in key_values:
        return key_values[key]
    data = dict(getattr(row, "data", {}) or {})
    if key in data:
        return data[key]
    return getattr(row, key, None)


def _parse_scan_profile_from_db(scan_db_path: Path) -> dict[str, Any]:
    if not scan_db_path.exists():
        return {}
    points: list[dict[str, Any]] = []
    try:
        db = ase_connect(str(scan_db_path))
        for row in db.select():
            iteration = _parse_optional_int(_row_lookup(row, "iteration"))
            if iteration is None:
                continue
            energy_eV = _parse_optional_float(_row_lookup(row, "total_energy_eV"))
            if energy_eV is None:
                energy_eV = _parse_optional_float(_row_lookup(row, "scan_energy_ev"))
            if energy_eV is None:
                energy_eV = _parse_optional_float(_row_lookup(row, "energy_eV"))
            if energy_eV is None:
                continue
            point: dict[str, Any] = {
                "iteration": int(iteration),
                "energy_eV": float(energy_eV),
            }
            bond_length = _parse_optional_float(_row_lookup(row, "scan_bond_length"))
            if bond_length is not None:
                point["bond_length_A"] = float(bond_length)
            excited_states = _row_lookup(row, "excited_states")
            if isinstance(excited_states, list) and excited_states:
                point["excited_states"] = excited_states
            chosen_iroot = _row_lookup(row, "chosen_iroot_data")
            if isinstance(chosen_iroot, dict) and chosen_iroot:
                point["chosen_iroot_data"] = chosen_iroot
            points.append(point)
    except Exception:
        return {}
    return _scan_profile_from_points(points)


def _scan_endpoint_frames_from_db(scan_db_path: Path) -> list[tuple[int, Any, float | None]]:
    if not scan_db_path.exists():
        return []

    selected: dict[int, tuple[float | None, Any]] = {}
    try:
        db = ase_connect(str(scan_db_path))
        for row in db.select():
            iteration = _parse_optional_int(_row_lookup(row, "iteration"))
            if iteration is None:
                continue
            try:
                atoms = row.toatoms()
            except Exception:
                continue

            energy_eV = _parse_optional_float(_row_lookup(row, "total_energy_eV"))
            if energy_eV is None:
                energy_eV = _parse_optional_float(_row_lookup(row, "scan_energy_ev"))
            if energy_eV is None:
                energy_eV = _parse_optional_float(_row_lookup(row, "energy_eV"))

            atoms.info["scan_iteration"] = int(iteration)
            if energy_eV is not None:
                atoms.info["scan_energy_eV"] = float(energy_eV)

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
    except Exception:
        return []

    ordered = sorted(selected.items(), key=lambda item: item[0])
    return [(iteration, atoms, energy) for iteration, (energy, atoms) in ordered]


def _build_ping_pong_trajectory_frames(
    frames: list[tuple[int, Any, float | None]],
) -> list[Any]:
    ordered_atoms = [atoms for _iteration, atoms, _energy in frames]
    if len(ordered_atoms) <= 2:
        return ordered_atoms
    # 0..N-1..1 (avoid duplicate first/last frame at the turning points)
    return ordered_atoms + ordered_atoms[-2:0:-1]


def _write_bond_scan_mep_xyz(
    *,
    run_dir: Path,
    scan_db_path: Path,
) -> Path | None:
    frames = _scan_endpoint_frames_from_db(scan_db_path)
    if not frames:
        return None

    out_dir = run_dir / "results" / "scan"
    out_dir.mkdir(parents=True, exist_ok=True)
    mep_xyz_path = out_dir / "bond_scan_mep.xyz"

    atoms_frames = _build_ping_pong_trajectory_frames(frames)
    try:
        ase_write(mep_xyz_path, atoms_frames, format="xyz")
    except Exception:
        return None
    return mep_xyz_path


def _render_xyz_with_xyzrender(
    *,
    xyz_path: Path,
    gif_trajectory_mode: bool = False,
    render_gif: bool = True,
) -> tuple[Path | None, Path | None]:
    if not xyz_path.exists() or not xyz_path.is_file():
        return None, None

    command_setting = str(getattr(settings, "REACTION_HUB_XYZRENDER_COMMAND", "") or "").strip()
    try:
        command, command_warning = resolve_xyzrender_command(command_setting)
    except ValueError:
        return None, None
    if not command:
        return None, None
    if command_warning:
        print(f"Warning: {command_warning}")

    gif_rot = str(getattr(settings, "REACTION_HUB_XYZRENDER_GIF_ROT", "xy") or "xy").strip() or "xy"
    rot_frames = max(1, _parse_optional_int(getattr(settings, "REACTION_HUB_XYZRENDER_ROT_FRAMES", 200)) or 200)

    svg_path = xyz_path.with_suffix(".svg")
    gif_path = xyz_path.with_suffix(".gif")
    render_command = [*command, str(xyz_path.name), "-o", str(svg_path.name)]
    preexisting_gifs: set[Path] = set()
    if render_gif:
        preexisting_gifs = {path.resolve() for path in xyz_path.parent.glob("*.gif")}
        if gif_trajectory_mode:
            render_command.extend(["--gif-trj", "-go", str(gif_path.name)])
        else:
            render_command.extend(["--gif-rot", gif_rot, "--rot-frames", str(rot_frames)])

    try:
        completed = subprocess.run(
            render_command,
            cwd=str(xyz_path.parent),
            check=False,
            shell=False,
            env=build_xyzrender_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError:
        return None, None

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        if is_xyzrender_cairo_error(detail) and render_gif:
            if not svg_path.exists():
                svg_only_command = [
                    *command,
                    str(xyz_path.name),
                    "-o",
                    str(svg_path.name),
                ]
                svg_only = subprocess.run(
                    svg_only_command,
                    cwd=str(xyz_path.parent),
                    check=False,
                    shell=False,
                    env=build_xyzrender_subprocess_env(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                if svg_only.returncode != 0:
                    detail_svg = (svg_only.stderr or svg_only.stdout or "").strip()
                    if detail_svg:
                        detail_svg = detail_svg[-500:]
                    print(
                        "Warning: xyzrender failed while ingesting bond-scan render artifacts "
                        "even after SVG-only retry. "
                        f"Details: {detail_svg or detail[-500:]}. "
                        f"{cairo_runtime_install_hint()}"
                    )
                    return None, None
            print(
                "Warning: xyzrender GIF generation skipped during ingest because Cairo is unavailable; "
                f"ingesting SVG artifact only. {cairo_runtime_install_hint()}"
            )
            return (svg_path if svg_path.exists() else None), None

        if detail:
            detail = detail[-500:]
            print(
                "Warning: xyzrender failed while ingesting bond-scan render artifacts. "
                f"Details: {detail}"
            )
        return None, None

    produced_svg = svg_path if svg_path.exists() else None
    produced_gif: Path | None = gif_path if render_gif and gif_path.exists() else None
    if render_gif and produced_gif is None:
        gif_candidates = sorted(
            (
                path
                for path in xyz_path.parent.glob("*.gif")
                if path.resolve() not in preexisting_gifs
            ),
            key=lambda path: (path.stat().st_mtime_ns if path.exists() else 0, str(path)),
            reverse=True,
        )
        if gif_candidates:
            produced_gif = gif_candidates[0]
    if produced_gif is not None and produced_gif.exists():
        canvas_size = _xyzrender_gif_canvas_size()
        normalized, note = normalize_gif_canvas(produced_gif, size=canvas_size)
        if note:
            print(f"Warning: {note}" if not normalized else note)
    return produced_svg, produced_gif


def _ingest_bond_scan_mep_artifacts(
    *,
    run: WorkflowRun,
    run_dir: Path,
    scan_db_artifact: Artifact,
    scan_db_path: Path,
) -> None:
    mep_xyz_path = _write_bond_scan_mep_xyz(run_dir=run_dir, scan_db_path=scan_db_path)
    if mep_xyz_path is None:
        return

    try:
        mep_rel = str(mep_xyz_path.relative_to(run_dir))
    except ValueError:
        mep_rel = str(mep_xyz_path)

    mep_artifact, _ = Artifact.objects.update_or_create(
        workflow_run=run,
        path=mep_rel,
        defaults={
            "kind": ArtifactKind.TRAJECTORY,
            "label": f"trajectory:{mep_xyz_path.name}",
            "metadata": {
                "source": "scan_endpoints_db",
                "source_artifact_id": scan_db_artifact.pk,
                "generated_by": "bond_scan_mep_export",
                "trajectory_pattern": "ping_pong",
            },
        },
    )

    svg_path = mep_xyz_path.with_suffix(".svg")
    gif_path = mep_xyz_path.with_suffix(".gif")
    render_gif = not gif_path.exists()
    rendered_svg, rendered_gif = _render_xyz_with_xyzrender(
        xyz_path=mep_xyz_path,
        gif_trajectory_mode=True,
        render_gif=render_gif,
    )
    if rendered_svg is not None:
        svg_path = rendered_svg
    if rendered_gif is not None:
        gif_path = rendered_gif

    gif_canvas_size = _xyzrender_gif_canvas_size()
    for image_path in (svg_path, gif_path):
        if not image_path.exists():
            continue
        is_gif = image_path.suffix.lower() == ".gif"
        canvas_metadata = {}
        if is_gif and gif_canvas_size is not None:
            canvas_metadata = {
                "canvas_width_px": int(gif_canvas_size[0]),
                "canvas_height_px": int(gif_canvas_size[1]),
                "canvas_normalized": True,
            }
        try:
            image_rel = str(image_path.relative_to(run_dir))
        except ValueError:
            image_rel = str(image_path)
        Artifact.objects.update_or_create(
            workflow_run=run,
            path=image_rel,
            defaults={
                "kind": ArtifactKind.IMAGE,
                "label": f"render:{image_path.name}",
                "metadata": {
                    "source": "xyzrender",
                    "source_xyz_artifact_id": mep_artifact.pk,
                    "source_xyz_path": mep_rel,
                    "render_output": "gif" if is_gif else "svg",
                    **(
                        {
                            "gif_mode": "trajectory",
                            "gif_rotation": str(
                                getattr(settings, "REACTION_HUB_XYZRENDER_GIF_ROT", "xy")
                            ),
                            "rot_frames": int(
                                max(
                                    1,
                                    _parse_optional_int(
                                        getattr(settings, "REACTION_HUB_XYZRENDER_ROT_FRAMES", 200)
                                    )
                                    or 200,
                                )
                            ),
                        }
                        if is_gif
                        else {}
                    ),
                    **canvas_metadata,
                },
            },
        )


def _upsert_property(
    *,
    run: WorkflowRun,
    name: str,
    unit: str = "",
    value_float: float | None = None,
    value_text: str = "",
    source_artifact: Artifact | None = None,
    metadata: dict | None = None,
) -> ComputedProperty:
    return ComputedProperty.objects.update_or_create(
        workflow_run=run,
        name=name,
        unit=unit,
        defaults={
            "value_float": value_float,
            "value_text": value_text,
            "source_artifact": source_artifact,
            "metadata": metadata or {},
        },
    )[0]


def _store_scan_profile(
    *,
    run: WorkflowRun,
    profile: dict[str, Any],
    source_artifact: Artifact | None,
) -> None:
    if not profile:
        return
    start_energy = _parse_optional_float(profile.get("start_energy_eV"))
    peak_energy = _parse_optional_float(profile.get("peak_energy_eV"))
    barrier = _parse_optional_float(profile.get("activation_barrier_eV"))
    if start_energy is not None:
        _upsert_property(
            run=run,
            name="bond_scan_start_energy_eV",
            unit="eV",
            value_float=float(start_energy),
            source_artifact=source_artifact,
            metadata={"source": "scan_profile"},
        )
    if peak_energy is not None:
        _upsert_property(
            run=run,
            name="bond_scan_peak_energy_eV",
            unit="eV",
            value_float=float(peak_energy),
            source_artifact=source_artifact,
            metadata={"source": "scan_profile"},
        )
    if barrier is not None:
        _upsert_property(
            run=run,
            name="bond_scan_activation_barrier_eV",
            unit="eV",
            value_float=float(barrier),
            source_artifact=source_artifact,
            metadata={"source": "scan_profile"},
        )
    points = profile.get("points")
    if isinstance(points, list):
        _upsert_property(
            run=run,
            name="bond_scan_energy_profile",
            unit="eV",
            value_text="path_profile",
            source_artifact=source_artifact,
            metadata={"points": points, "source": "scan_profile"},
        )
        spectral_points = [
            point
            for point in points
            if point.get("excited_states") or point.get("chosen_iroot_data")
        ]
        if spectral_points:
            update_spectral_dataset(
                run=run,
                dataset_name="bond_scan",
                source_artifact=source_artifact,
                payload={
                    "provenance": {"source": "scan_endpoints_db"},
                    "scan_endpoints": spectral_points,
                    "transition_units": {
                        "wavenumber_cm": "cm^-1",
                        "wavelength_nm": "nm",
                        "oscillator_strength": "dimensionless",
                    },
                    "ct_state_units": {
                        "wavelength_nm": "nm",
                        "ct_e": "e",
                        "ct_pct": "%",
                    },
                },
            )


def _store_forces_payload(
    *,
    run: WorkflowRun,
    vectors: list[list[float]],
    symbols: list[str] | None,
    source_artifact: Artifact | None,
) -> None:
    if not vectors:
        return
    norms = [math.sqrt(float(x) ** 2 + float(y) ** 2 + float(z) ** 2) for x, y, z in vectors]
    _upsert_property(
        run=run,
        name="final_force_max_abs",
        unit="eV/A",
        value_float=float(max(norms)),
        source_artifact=source_artifact,
        metadata={"source": "ingest"},
    )
    _upsert_property(
        run=run,
        name="final_force_rms",
        unit="eV/A",
        value_float=float(math.sqrt(sum(n * n for n in norms) / len(norms))),
        source_artifact=source_artifact,
        metadata={"source": "ingest"},
    )
    _upsert_property(
        run=run,
        name="final_forces",
        unit="eV/A",
        value_text="per_atom",
        source_artifact=source_artifact,
        metadata={
            "vectors": vectors,
            "symbols": symbols or [],
            "source": "ingest",
        },
    )


def _safe_atoms_fmax_for_ingest(atoms) -> float | None:
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
    return float(max_force)


def _artifact_local_path_for_ingest(run: WorkflowRun, artifact: Artifact) -> Path | None:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    local_cache_text = str(metadata.get("local_cache_path") or "").strip()
    if local_cache_text:
        local_cache_path = Path(local_cache_text)
        if local_cache_path.exists() and local_cache_path.is_file():
            return local_cache_path

    run_dir_text = canonical_run_path(run.working_directory)
    run_dir = Path(run_dir_text)
    path_text = normalize_run_path(str(artifact.path or ""))
    if not path_text:
        return None

    if path_text.startswith("/"):
        absolute_path = Path(path_text)
        if absolute_path.exists() and absolute_path.is_file():
            return absolute_path
        rel_path = _relative_artifact_path(run_dir_text, path_text)
        rel_candidate = (run_dir / rel_path).resolve()
        if rel_candidate.exists() and rel_candidate.is_file():
            return rel_candidate
        return None

    candidate = (run_dir / path_text).resolve()
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _retain_spectrum_csv_artifacts(run: WorkflowRun) -> int:
    retained = 0
    for artifact in Artifact.objects.filter(
        workflow_run=run,
        kind=ArtifactKind.CSV,
        label__startswith="spectrum:",
    ):
        local_path = _artifact_local_path_for_ingest(run, artifact)
        curve = spectrum_curve_from_csv(local_path) if local_path is not None else None
        if curve is None:
            continue
        stem = Path(str(artifact.path or local_path)).stem
        dataset_name = stem.removesuffix("_spectrum") or "spectrum"
        curve["source_artifact_path"] = str(artifact.path)
        update_spectral_dataset(
            run=run,
            dataset_name=dataset_name,
            source_artifact=artifact,
            payload={"broadened_spectra": {stem: curve}},
        )
        retained += 1
    return retained


def _latest_local_evaluated_traj_for_ingest(run: WorkflowRun) -> tuple[Artifact, Path] | None:
    candidates: list[tuple[int, int, Artifact, Path]] = []
    for artifact in (
        Artifact.objects.filter(
            workflow_run=run,
            label="trajectory:evaluated.traj",
            kind=ArtifactKind.TRAJECTORY,
        )
        .select_related("job")
        .order_by("pk")
    ):
        local_path = _artifact_local_path_for_ingest(run, artifact)
        if local_path is None:
            continue
        try:
            mtime_ns = int(local_path.stat().st_mtime_ns)
        except OSError:
            mtime_ns = 0
        candidates.append((mtime_ns, int(artifact.pk or 0), artifact, local_path))
    if not candidates:
        return None
    _mtime_ns, _artifact_pk, artifact, local_path = max(candidates, key=lambda row: (row[0], row[1]))
    return artifact, local_path


def _upsert_latest_evaluated_traj_fmax_property(run: WorkflowRun) -> None:
    property_name = "latest_evaluated_traj_fmax"
    property_unit = "eV/A"

    run_mode = _run_mode_for_ingest(run)
    if run_mode not in {"", "single_optimisation"}:
        ComputedProperty.objects.filter(
            workflow_run=run,
            name=property_name,
            unit=property_unit,
        ).delete()
        return

    selected = _latest_local_evaluated_traj_for_ingest(run)
    if selected is None:
        ComputedProperty.objects.filter(
            workflow_run=run,
            name=property_name,
            unit=property_unit,
        ).delete()
        return
    source_artifact, traj_path = selected

    try:
        frames = ase_read(str(traj_path), index=":")
    except Exception:
        ComputedProperty.objects.filter(
            workflow_run=run,
            name=property_name,
            unit=property_unit,
        ).delete()
        return
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    if not frames:
        ComputedProperty.objects.filter(
            workflow_run=run,
            name=property_name,
            unit=property_unit,
        ).delete()
        return

    selected_index: int | None = None
    selected_fmax: float | None = None
    for idx in range(len(frames) - 1, -1, -1):
        candidate_fmax = _safe_atoms_fmax_for_ingest(frames[idx])
        if candidate_fmax is None:
            continue
        selected_index = idx
        selected_fmax = float(candidate_fmax)
        break

    if selected_index is None or selected_fmax is None:
        ComputedProperty.objects.filter(
            workflow_run=run,
            name=property_name,
            unit=property_unit,
        ).delete()
        return

    _upsert_property(
        run=run,
        name=property_name,
        unit=property_unit,
        value_float=selected_fmax,
        source_artifact=source_artifact,
        metadata={
            "source": "evaluated_traj",
            "trajectory_frame_index": int(selected_index),
            "trajectory_frame_count": int(len(frames)),
            "trajectory_frame_selection": (
                "latest_frame"
                if int(selected_index) == int(len(frames) - 1)
                else "latest_frame_with_forces"
            ),
            "trajectory_path": normalize_run_path(str(source_artifact.path or "")),
        },
    )


def _store_charges_payload(
    *,
    run: WorkflowRun,
    charges: list[float],
    symbols: list[str] | None,
    source_artifact: Artifact | None,
    charge_method: str | None = None,
) -> None:
    if not charges:
        return
    method_text = str(charge_method or "").strip().lower()
    if method_text in {"lowdin", "loewdin"}:
        method_text = "loewdin"
    elif method_text in {"mulliken", "mulliken_atomic"}:
        method_text = "mulliken"
    else:
        method_text = re.sub(r"[^a-z0-9]+", "_", method_text).strip("_")
    if not method_text:
        method_text = "mulliken"

    method_prefix = f"final_{method_text}"
    charge_sum = float(sum(charges))
    charge_max_abs = float(max(abs(value) for value in charges))

    common_metadata = {"source": "ingest", "charge_method": method_text}
    _upsert_property(
        run=run,
        name=f"{method_prefix}_charge_sum",
        unit="e",
        value_float=charge_sum,
        source_artifact=source_artifact,
        metadata=common_metadata,
    )
    _upsert_property(
        run=run,
        name=f"{method_prefix}_charge_max_abs",
        unit="e",
        value_float=charge_max_abs,
        source_artifact=source_artifact,
        metadata=common_metadata,
    )
    charge_metadata = {
        "charges": charges,
        "symbols": symbols or [],
        "source": "ingest",
        "charge_method": method_text,
    }
    _upsert_property(
        run=run,
        name=f"{method_prefix}_charges",
        unit="e",
        value_text="per_atom",
        source_artifact=source_artifact,
        metadata=charge_metadata,
    )

    # Prefer Mulliken for downstream consumers; retain other methods separately.
    preferred = ComputedProperty.objects.filter(
        workflow_run=run, name="multiwfn_ground_mulliken_charges", unit="e"
    ).first()
    if preferred is None and method_text != "mulliken":
        preferred = ComputedProperty.objects.filter(
            workflow_run=run, name="final_mulliken_charges", unit="e"
        ).first()
    if preferred is not None:
        charge_metadata = dict(preferred.metadata or {})
        preferred_charges = charge_metadata.get("charges") or []
        if preferred_charges:
            charge_sum = float(sum(preferred_charges))
            charge_max_abs = float(max(abs(value) for value in preferred_charges))
            common_metadata = {key: value for key, value in charge_metadata.items() if key not in {"charges", "symbols"}}
            source_artifact = preferred.source_artifact

    # Method-agnostic aliases for downstream consumers.
    _upsert_property(
        run=run,
        name="final_charge_sum",
        unit="e",
        value_float=charge_sum,
        source_artifact=source_artifact,
        metadata=common_metadata,
    )
    _upsert_property(
        run=run,
        name="final_charge_max_abs",
        unit="e",
        value_float=charge_max_abs,
        source_artifact=source_artifact,
        metadata=common_metadata,
    )
    _upsert_property(
        run=run,
        name="final_charges",
        unit="e",
        value_text="per_atom",
        source_artifact=source_artifact,
        metadata=charge_metadata,
    )


def _store_multiwfn_ground_mulliken_payload(
    *,
    run: WorkflowRun,
    payload: dict[str, Any],
    fragment_labels: dict[str, Any],
    source_artifact: Artifact | None,
) -> None:
    atoms = [row for row in (payload.get("atoms") or []) if isinstance(row, dict)]
    charges = [_parse_optional_float(row.get("charge_e")) for row in atoms]
    if not atoms or any(value is None for value in charges):
        return
    numeric_charges = [float(value) for value in charges if value is not None]
    metadata = {
        "source": "reaction_hub_summary",
        "program": str(payload.get("program") or "Multiwfn"),
        "charge_method": "mulliken",
        "source_wavefunction": payload.get("source_wavefunction"),
        "symbols": [str(row.get("symbol") or "") for row in atoms],
        "charges": numeric_charges,
        "atom_indices_1based": [row.get("index_1based") for row in atoms],
        "fragment_charges_e": dict(payload.get("fragment_charges_e") or {}),
        "fragment_labels": dict(fragment_labels or {}),
    }
    _upsert_property(
        run=run,
        name="multiwfn_ground_mulliken_charges",
        unit="e",
        value_text="per_atom",
        source_artifact=source_artifact,
        metadata=metadata,
    )
    _upsert_property(
        run=run,
        name="multiwfn_ground_mulliken_charge_sum",
        unit="e",
        value_float=float(sum(numeric_charges)),
        source_artifact=source_artifact,
        metadata={key: metadata[key] for key in ("source", "program", "charge_method")},
    )
    _store_charges_payload(
        run=run, charges=numeric_charges, symbols=metadata["symbols"],
        source_artifact=source_artifact, charge_method="mulliken",
    )


def _store_multiwfn_excited_mulliken_payload(
    *,
    run: WorkflowRun,
    payloads: list[dict[str, Any]],
    fragment_labels: dict[str, Any],
    source_artifact: Artifact | None,
) -> None:
    if not isinstance(payloads, list):
        return
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        atoms = [row for row in (payload.get("atoms") or []) if isinstance(row, dict)]
        charges = [_parse_optional_float(row.get("charge_e")) for row in atoms]
        if not atoms or any(value is None for value in charges):
            continue
        numeric_charges = [float(value) for value in charges if value is not None]
        state = payload.get("excitation_number")
        metadata = {
            "source": "reaction_hub_summary",
            "program": str(payload.get("program") or "Multiwfn"),
            "charge_method": "mulliken",
            "excitation_number": int(state) if state is not None else None,
            "density": str(payload.get("density") or "unrelaxed excited state"),
            "source_wavefunction": payload.get("source_wavefunction"),
            "symbols": [str(row.get("symbol") or "") for row in atoms],
            "charges": numeric_charges,
            "atom_indices_1based": [row.get("index_1based") for row in atoms],
            "fragment_charges_e": dict(payload.get("fragment_charges_e") or {}),
            "fragment_labels": dict(fragment_labels or {}),
        }
        suffix = f"_root_{int(state)}" if state is not None else ""
        _upsert_property(
            run=run,
            name=f"multiwfn_excited_mulliken_charges{suffix}",
            unit="e",
            value_text="per_atom",
            source_artifact=source_artifact,
            metadata=metadata,
        )
        _upsert_property(
            run=run,
            name=f"multiwfn_excited_mulliken_charge_sum{suffix}",
            unit="e",
            value_float=float(sum(numeric_charges)),
            source_artifact=source_artifact,
            metadata={key: metadata[key] for key in ("source", "program", "charge_method", "excitation_number")},
        )


def _store_structure_payload(
    *,
    run: WorkflowRun,
    formula: str,
    atom_count: int,
    symbols: list[str] | None,
    positions_A: list[list[float]] | None,
    source_artifact: Artifact | None,
) -> None:
    if formula:
        _upsert_property(
            run=run,
            name="final_structure_formula",
            unit="",
            value_text=str(formula),
            source_artifact=source_artifact,
            metadata={"source": "ingest"},
        )
    _upsert_property(
        run=run,
        name="final_atom_count",
        unit="count",
        value_float=float(atom_count),
        source_artifact=source_artifact,
        metadata={"source": "ingest"},
    )
    if symbols and positions_A:
        normalized_positions = [
            [round(float(row[0]), 8), round(float(row[1]), 8), round(float(row[2]), 8)]
            for row in positions_A
        ]
        _upsert_property(
            run=run,
            name="final_structure_snapshot",
            unit="",
            value_text="per_atom",
            source_artifact=source_artifact,
            metadata={
                "symbols": symbols,
                "positions_A": normalized_positions,
                "source": "ingest",
            },
        )
        digest_payload = json.dumps(
            {"symbols": [str(symbol) for symbol in symbols], "positions_A": normalized_positions},
            sort_keys=True,
            separators=(",", ":"),
        )
        _upsert_property(
            run=run,
            name="final_structure_sha256",
            unit="",
            value_text=hashlib.sha256(digest_payload.encode("utf-8")).hexdigest(),
            source_artifact=source_artifact,
            metadata={"source": "ingest"},
        )


def _summary_final_payload(summary: dict[str, Any]) -> dict[str, Any]:
    final = summary.get("final")
    if isinstance(final, dict):
        return final

    synthesized: dict[str, Any] = {}
    for source_key, target_key in (
        ("final_energy_hartree", "energy_hartree"),
        ("final_energy_eV", "energy_eV"),
        ("ase_final_energy_eV", "energy_eV"),
    ):
        if target_key in synthesized:
            continue
        value = summary.get(source_key)
        if value is not None:
            synthesized[target_key] = value
    return synthesized


def _coerce_float_list(raw_value: object) -> list[float]:
    if not isinstance(raw_value, list):
        return []
    values: list[float] = []
    for value in raw_value:
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def _store_tddft_summary_payload(
    *,
    run: WorkflowRun,
    tddft: dict[str, Any],
    source_artifact: Artifact | None,
) -> None:
    wavelengths_nm = _coerce_float_list(tddft.get("wavelengths_nm"))
    wavenumbers_cm = _coerce_float_list(tddft.get("wavenumbers_cm"))
    oscillator_strengths = _coerce_float_list(tddft.get("oscillator_strengths"))
    transition_count = max(len(wavelengths_nm), len(wavenumbers_cm), len(oscillator_strengths))
    if transition_count <= 0:
        return

    transitions: list[dict[str, float | int]] = []
    for idx in range(transition_count):
        row: dict[str, float | int] = {"index": idx + 1}
        if idx < len(wavelengths_nm):
            row["wavelength_nm"] = wavelengths_nm[idx]
        if idx < len(wavenumbers_cm):
            row["wavenumber_cm"] = wavenumbers_cm[idx]
        if idx < len(oscillator_strengths):
            row["oscillator_strength"] = oscillator_strengths[idx]
        transitions.append(row)

    _upsert_property(
        run=run,
        name="final_tddft_transitions",
        unit="",
        value_text="transitions",
        source_artifact=source_artifact,
        metadata={
            "source": "reaction_hub_summary",
            "transitions": transitions,
        },
    )
    update_spectral_dataset(
        run=run,
        dataset_name="final_tddft",
        source_artifact=source_artifact,
        payload={
            "provenance": {"source": "reaction_hub_summary"},
            "transitions": transitions,
            "transition_units": {
                "wavenumber_cm": "cm^-1",
                "wavelength_nm": "nm",
                "oscillator_strength": "dimensionless",
            },
        },
    )

    if oscillator_strengths:
        strongest_index = max(
            range(len(oscillator_strengths)),
            key=lambda idx: oscillator_strengths[idx],
        )
        strongest_strength = oscillator_strengths[strongest_index]
        _upsert_property(
            run=run,
            name="final_tddft_strongest_oscillator_strength",
            unit="",
            value_float=float(strongest_strength),
            source_artifact=source_artifact,
            metadata={"source": "reaction_hub_summary"},
        )
        if strongest_index < len(wavelengths_nm):
            _upsert_property(
                run=run,
                name="final_tddft_strongest_wavelength_nm",
                unit="nm",
                value_float=float(wavelengths_nm[strongest_index]),
                source_artifact=source_artifact,
                metadata={"source": "reaction_hub_summary"},
            )


def _write_tddft_spectrum_artifacts_from_summary(
    *,
    run: WorkflowRun,
    tddft: dict[str, Any],
) -> None:
    run_dir_text = canonical_run_path(run.working_directory)
    if not run_dir_text:
        return
    run_dir = Path(run_dir_text)
    if not run_dir.exists() or not run_dir.is_dir():
        return

    wavelengths_nm = _coerce_float_list(tddft.get("wavelengths_nm"))
    wavenumbers_cm = _coerce_float_list(tddft.get("wavenumbers_cm"))
    oscillator_strengths = _coerce_float_list(tddft.get("oscillator_strengths"))
    if not wavelengths_nm and wavenumbers_cm:
        wavelengths_nm = [1.0e7 / value for value in wavenumbers_cm if value]
    point_count = min(len(wavelengths_nm), len(oscillator_strengths))
    if point_count <= 0:
        return

    try:
        import numpy as np
    except Exception:
        return

    wavelengths = np.asarray(wavelengths_nm[:point_count], dtype=float)
    intensities = np.asarray(oscillator_strengths[:point_count], dtype=float)
    finite_mask = np.isfinite(wavelengths) & np.isfinite(intensities)
    wavelengths = wavelengths[finite_mask]
    intensities = intensities[finite_mask]
    if wavelengths.size == 0 or intensities.size == 0:
        return

    spectrum_dir = run_dir / "DFT" / "final_tddft_spectrum"
    csv_path = spectrum_dir / "final_tddft_spectrum.csv"
    png_path = spectrum_dir / "final_tddft_spectrum.png"
    try:
        spectrum_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    try:
        from cats.orca.tddft.scan import UVVisPlotSettings

        uvvis_settings = UVVisPlotSettings()
        linewidth_nm = float(uvvis_settings.stacked_uvvis_linewidth_nm)
        step_nm = float(uvvis_settings.step)
        start_nm = float(uvvis_settings.stacked_uvvis_xmin)
        end_nm = float(uvvis_settings.stacked_uvvis_xmax)
        normalize_spectrum = bool(uvvis_settings.stacked_uvvis_normalize)
    except Exception:
        linewidth_nm = 20.0
        step_nm = 1.0
        start_nm = 300.0
        end_nm = 800.0
        normalize_spectrum = False

    range_fallback_used = False
    if end_nm > start_nm:
        in_requested_window = (wavelengths >= start_nm) & (wavelengths <= end_nm)
        if not bool(np.any(in_requested_window)):
            start_nm = 300.0
            end_nm = 800.0
            range_fallback_used = True
    if end_nm <= start_nm:
        end_nm = start_nm + 10.0 * linewidth_nm
    axis = np.arange(start_nm, end_nm + step_nm, step_nm, dtype=float)
    sigma = linewidth_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    profile = np.zeros_like(axis)
    for wavelength, intensity in zip(wavelengths, intensities):
        profile += float(intensity) * np.exp(-0.5 * ((axis - float(wavelength)) / sigma) ** 2)
    peak = float(np.nanmax(profile)) if profile.size else 0.0
    if normalize_spectrum and peak > 0.0:
        profile = profile / peak

    update_spectral_dataset(
        run=run,
        dataset_name="final_tddft",
        payload={
            "broadened_spectra": {
                "final_tddft_spectrum": {
                    "x_unit": "nm",
                    "intensity_unit": "a.u.",
                    "normalization": "maximum" if normalize_spectrum else "raw",
                    "linewidth_nm": float(linewidth_nm),
                    "linewidth_convention": "fwhm",
                    "points": [
                        {"x": float(x_value), "intensity": float(y_value)}
                        for x_value, y_value in zip(axis, profile)
                    ],
                }
            },
        },
    )

    try:
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("wavelength_nm,intensity\n")
            for x_value, y_value in zip(axis, profile):
                handle.write(f"{float(x_value):.8g},{float(y_value):.8g}\n")
    except OSError:
        return

    Artifact.objects.update_or_create(
        workflow_run=run,
        path="DFT/final_tddft_spectrum/final_tddft_spectrum.csv",
        defaults={
            "kind": ArtifactKind.CSV,
            "label": "spectrum:final_tddft_spectrum.csv",
            "metadata": {
                "source": "reaction_hub_summary",
                "generated_from_summary": True,
                "ingestion_path": normalize_run_path(str(csv_path)),
                "wavelength_range_nm": [float(axis[0]), float(axis[-1])],
                "wavelength_range_fallback": bool(range_fallback_used),
            },
        },
    )

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot(axis, profile)
        stick_scale = max(float(np.nanmax(intensities)), 1.0)
        ax.vlines(wavelengths, 0.0, intensities / stick_scale, alpha=0.35)
        ax.set_xlabel("Wavelength / nm")
        ax.set_ylabel("Intensity (a.u.)")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        return

    Artifact.objects.update_or_create(
        workflow_run=run,
        path="DFT/final_tddft_spectrum/final_tddft_spectrum.png",
        defaults={
            "kind": ArtifactKind.IMAGE,
            "label": _KEY_PLOT_FILENAMES["final_tddft_spectrum.png"],
            "metadata": {
                "source": "reaction_hub_summary",
                "generated_from_summary": True,
                "ingestion_path": normalize_run_path(str(png_path)),
                "wavelength_range_nm": [float(axis[0]), float(axis[-1])],
                "wavelength_range_fallback": bool(range_fallback_used),
            },
        },
    )


def _summary_requests_final_tddft_spectrum(summary: dict[str, Any]) -> bool:
    if bool(summary.get("final_tddft_spectrum_requested")):
        return True
    if bool(summary.get("final_tddft_spectrum")):
        return True
    settings_payload = summary.get("settings")
    if isinstance(settings_payload, dict) and bool(settings_payload.get("final_tddft_spectrum")):
        return True
    return False


def _tddft_output_candidates_for_summary(run_dir: Path, summary: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for key in ("tddft_output", "tddft_transition_output"):
        raw_path = str(summary.get(key) or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        candidates.append(path if path.is_absolute() else run_dir / path)

    candidates.extend(
        [
            run_dir / "DFT" / "final_tddft_spectrum" / "nwchem_tddft" / "nwchem_tddft.nwo",
            run_dir / "DFT" / "final_tddft_spectrum" / "nwchem_tddft.nwo",
            run_dir / "nwchem_tddft" / "nwchem_tddft.nwo",
            run_dir / "nwchem_tddft.nwo",
        ]
    )

    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalize_run_path(str(candidate))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _workflow_backend_key_for_summary(run: WorkflowRun, summary: dict[str, Any]) -> str:
    settings_payload = summary.get("settings")
    if isinstance(settings_payload, dict):
        backend_key = str(settings_payload.get("qm_backend") or "").strip().lower()
        if backend_key:
            return backend_key
    workflow = str(summary.get("workflow") or "").strip().lower()
    if "nwchem" in workflow:
        return "nwchem"
    for key in ("output", "nwchem_out", "tddft_output", "tddft_transition_output"):
        if str(summary.get(key) or "").strip().lower().endswith(".nwo"):
            return "nwchem"
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission")
    if isinstance(submission, dict):
        backend_key = str(submission.get("qm_backend") or "").strip().lower()
        if backend_key:
            return backend_key
    return "orca"


def _backfill_tddft_spectrum_artifacts_from_output(
    *,
    run: WorkflowRun,
    summary: dict[str, Any],
    source_artifact: Artifact | None = None,
) -> bool:
    run_dir_text = canonical_run_path(run.working_directory)
    if not run_dir_text:
        return False
    run_dir = Path(run_dir_text)
    if not run_dir.exists():
        return False

    output_candidates = _tddft_output_candidates_for_summary(run_dir, summary)
    output_path = next((path for path in output_candidates if path.exists()), None)
    if output_path is None:
        return False

    backend_key = _workflow_backend_key_for_summary(run, summary)
    if backend_key not in {"orca", "nwchem"}:
        return False

    try:
        from cats.qm_backends.workflow import get_workflow_backend

        transitions = get_workflow_backend(backend_key).extract_transitions(output_path)
    except Exception:
        return False

    wavelengths = (
        transitions.wavelengths_nm.tolist()
        if transitions.wavelengths_nm is not None
        else None
    )
    tddft_payload = {
        "wavenumbers_cm": transitions.wavenumbers_cm.tolist(),
        "wavelengths_nm": wavelengths,
        "oscillator_strengths": transitions.intensities.tolist(),
    }
    _store_tddft_summary_payload(
        run=run,
        tddft=tddft_payload,
        source_artifact=source_artifact,
    )
    _write_tddft_spectrum_artifacts_from_summary(
        run=run,
        tddft=tddft_payload,
    )
    return True


def _run_dir_has_nwchem_tddft_output(run: WorkflowRun) -> bool:
    run_dir_text = canonical_run_path(run.working_directory)
    if not run_dir_text:
        return False
    run_dir = Path(run_dir_text)
    return any(path.exists() for path in _tddft_output_candidates_for_summary(run_dir, {}))


def _ingest_summary_payload(
    *,
    run: WorkflowRun,
    summary: dict[str, Any],
    source_artifact: Artifact | None = None,
) -> None:
    final = _summary_final_payload(summary)

    energy_h = final.get("energy_hartree")
    if energy_h is not None:
        try:
            _upsert_property(
                run=run,
                name="final_single_point_energy",
                unit="Eh",
                value_float=float(energy_h),
                source_artifact=source_artifact,
                metadata={"source": "reaction_hub_summary"},
            )
        except (TypeError, ValueError):
            pass

    energy_eV = final.get("energy_eV")
    if energy_eV is not None:
        try:
            _upsert_property(
                run=run,
                name="final_single_point_energy_eV",
                unit="eV",
                value_float=float(energy_eV),
                source_artifact=source_artifact,
                metadata={"source": "reaction_hub_summary"},
            )
        except (TypeError, ValueError):
            pass

    cluster_fragment = final.get("cluster_fragment")
    if isinstance(cluster_fragment, dict):
        cluster_energy_h = cluster_fragment.get("energy_hartree")
        if cluster_energy_h is not None:
            try:
                _upsert_property(
                    run=run,
                    name="final_cluster_single_point_energy",
                    unit="Eh",
                    value_float=float(cluster_energy_h),
                    source_artifact=source_artifact,
                    metadata={"source": "reaction_hub_summary"},
                )
            except (TypeError, ValueError):
                pass
        cluster_energy_eV = cluster_fragment.get("energy_eV")
        if cluster_energy_eV is not None:
            try:
                _upsert_property(
                    run=run,
                    name="final_cluster_single_point_energy_eV",
                    unit="eV",
                    value_float=float(cluster_energy_eV),
                    source_artifact=source_artifact,
                    metadata={"source": "reaction_hub_summary"},
                )
            except (TypeError, ValueError):
                pass

    symbols = []
    raw_symbols = final.get("symbols")
    if isinstance(raw_symbols, list):
        symbols = [str(value) for value in raw_symbols]

    positions_A: list[list[float]] | None = None
    raw_positions = final.get("positions_A")
    if isinstance(raw_positions, list):
        parsed_positions: list[list[float]] = []
        for row in raw_positions:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                continue
            try:
                parsed_positions.append([float(row[0]), float(row[1]), float(row[2])])
            except (TypeError, ValueError):
                continue
        if parsed_positions:
            positions_A = parsed_positions

    atom_count = 0
    raw_atom_count = final.get("atom_count")
    try:
        atom_count = int(raw_atom_count)
    except (TypeError, ValueError):
        atom_count = len(symbols)

    formula = str(final.get("formula") or "").strip()
    if formula or atom_count > 0:
        _store_structure_payload(
            run=run,
            formula=formula,
            atom_count=max(atom_count, len(symbols)),
            symbols=symbols,
            positions_A=positions_A,
            source_artifact=source_artifact,
        )

    raw_forces = final.get("forces_eV_A")
    if isinstance(raw_forces, list):
        vectors: list[list[float]] = []
        for item in raw_forces:
            vector = item.get("force") if isinstance(item, dict) else item
            if not isinstance(vector, (list, tuple)) or len(vector) != 3:
                continue
            try:
                vectors.append([float(vector[0]), float(vector[1]), float(vector[2])])
            except (TypeError, ValueError):
                continue
        _store_forces_payload(
            run=run,
            vectors=vectors,
            symbols=symbols,
            source_artifact=source_artifact,
        )

    raw_charges = final.get("charges")
    charges: list[float] = []
    if isinstance(raw_charges, list):
        for value in raw_charges:
            try:
                charges.append(float(value))
            except (TypeError, ValueError):
                continue
    _store_charges_payload(
        run=run,
        charges=charges,
        symbols=symbols,
        source_artifact=source_artifact,
        charge_method=str(final.get("charge_method") or "").strip() or None,
    )

    scan_profile = summary.get("scan")
    if isinstance(scan_profile, dict):
        points = scan_profile.get("points")
        normalized_scan_profile = (
            _scan_profile_from_points(points)
            if isinstance(points, list) and points
            else scan_profile
        )
        _store_scan_profile(
            run=run,
            profile=normalized_scan_profile,
            source_artifact=source_artifact,
        )

    settings_payload = summary.get("settings")
    if isinstance(settings_payload, dict):
        run_mode = _run_mode_for_ingest(run)
        if run_mode in {"", "tddft_bond_scan"}:
            target_nm = _parse_optional_float(settings_payload.get("target_wavelength_nm"))
            if target_nm is not None:
                _upsert_property(
                    run=run,
                    name="target_wavelength_nm",
                    unit="nm",
                    value_float=float(target_nm),
                    source_artifact=source_artifact,
                    metadata={"source": "reaction_hub_summary"},
                )
        else:
            ComputedProperty.objects.filter(
                workflow_run=run,
                name="target_wavelength_nm",
                unit="nm",
            ).delete()

    tddft_payload = summary.get("tddft")
    if isinstance(tddft_payload, dict):
        _store_tddft_summary_payload(
            run=run,
            tddft=tddft_payload,
            source_artifact=source_artifact,
        )
        _write_tddft_spectrum_artifacts_from_summary(
            run=run,
            tddft=tddft_payload,
        )
    elif (
        _single_optimisation_final_tddft_spectrum_requested(run)
        or _summary_requests_final_tddft_spectrum(summary)
        or _run_dir_has_nwchem_tddft_output(run)
    ):
        _backfill_tddft_spectrum_artifacts_from_output(
            run=run,
            summary=summary,
            source_artifact=source_artifact,
        )

    charge_transfer = summary.get("charge_transfer")
    orbital_mulliken_only = (
        isinstance(charge_transfer, dict)
        and charge_transfer.get("analysis_scope") == "nto_orbital_mulliken"
    )
    ground_orbitals_only = (
        isinstance(charge_transfer, dict)
        and charge_transfer.get("analysis_scope") == "ground_state_orbitals"
    )
    if ground_orbitals_only:
        _upsert_property(
            run=run, name="ground_state_orbital_exports",
            value_text=json.dumps(charge_transfer.get("density_analysis") or {}, sort_keys=True),
            source_artifact=source_artifact,
            metadata={"source_run_id": summary.get("source_run_id"), "orbital_kind": "ground_state_canonical"},
        )
    if orbital_mulliken_only:
        _upsert_property(
            run=run,
            name="multiwfn_nto_orbital_mulliken",
            value_text=json.dumps(charge_transfer.get("density_analysis") or {}, sort_keys=True),
            source_artifact=source_artifact,
            metadata={"source_run_id":summary.get("source_run_id"),
                      "method":"Mulliken orbital composition", "not_ground_state_charges":True},
        )
    if isinstance(charge_transfer, dict) and not orbital_mulliken_only and not ground_orbitals_only:
        ct_states = []
        for raw_row in charge_transfer.get("ct_states") or []:
            row = dict(raw_row)
            row.setdefault("wavelength_nm", row.get("nm"))
            row.setdefault("oscillator_strength", row.get("f"))
            row.setdefault("transferred_charge_e", row.get("CT(e)"))
            row.setdefault("charge_transfer_pct", row.get("CT(%)"))
            ct_states.append(row)
        ct_terms = []
        for raw_row in charge_transfer.get("ct_terms") or []:
            row = dict(raw_row)
            row.setdefault("percentage", row.get("pct"))
            ct_terms.append(row)
        spectral_run = run
        spectral_dataset_name = (
            "final_tddft"
            if _run_mode_for_ingest(run) == "single_optimisation"
            else "charge_transfer"
        )
        source_run_id = summary.get("source_run_id")
        try:
            source_run_pk = int(source_run_id)
        except (TypeError, ValueError):
            source_run_pk = 0
        if source_run_pk and source_run_pk != run.pk:
            source_run = WorkflowRun.objects.filter(pk=source_run_pk).first()
            if source_run is not None:
                spectral_run = source_run
                spectral_dataset_name = "final_tddft"
        ground_state_mulliken = charge_transfer.get("ground_state_mulliken")
        ground_state_mulliken_only = (
            str(charge_transfer.get("analysis_scope") or "").strip().lower()
            == "ground_state_mulliken"
        )
        if isinstance(ground_state_mulliken, dict):
            _store_multiwfn_ground_mulliken_payload(
                run=spectral_run,
                payload=ground_state_mulliken,
                fragment_labels=dict(charge_transfer.get("fragment_labels") or {}),
                source_artifact=source_artifact if spectral_run.pk == run.pk else None,
            )
        excited_state_mulliken = charge_transfer.get("excited_state_mulliken")
        if isinstance(excited_state_mulliken, list):
            _store_multiwfn_excited_mulliken_payload(
                run=spectral_run,
                payloads=excited_state_mulliken,
                fragment_labels=dict(charge_transfer.get("fragment_labels") or {}),
                source_artifact=source_artifact if spectral_run.pk == run.pk else None,
            )
        if not ground_state_mulliken_only:
            update_spectral_dataset(
                run=spectral_run,
                dataset_name=spectral_dataset_name,
                source_artifact=source_artifact if spectral_run.pk == run.pk else None,
                payload={
                    "provenance": {
                        "source": "reaction_hub_summary",
                        "charge_transfer_run_id": run.pk,
                        "source_run_id": summary.get("source_run_id"),
                        "source_run_uid": summary.get("source_run_uid"),
                        "source_run_name": summary.get("source_run_name"),
                    },
                    "fragments": {
                        "fragments_0based": charge_transfer.get("fragments_0based") or {},
                        "fragments_1based": charge_transfer.get("fragments_1based") or {},
                        "fragment_labels": charge_transfer.get("fragment_labels") or {},
                    },
                    "ct_states": ct_states,
                    "ct_terms": ct_terms,
                    "ct_spectra": list(charge_transfer.get("ct_spectra") or []),
                    "density_analysis": dict(charge_transfer.get("density_analysis") or {}),
                    "nroots": charge_transfer.get("nroots"),
                    "warnings": list(charge_transfer.get("warnings") or []),
                },
            )
            try:
                write_ct_plot_artifacts(spectral_run)
            except Exception as exc:
                print(f"Warning: Failed to generate retained CT plots for {spectral_run.run_name}: {exc}")
        if spectral_run.pk != run.pk:
            source_metadata = dict(spectral_run.metadata or {})
            source_tags = source_metadata.get("tags") or []
            if isinstance(source_tags, str):
                source_tags = [source_tags]
            result_tag = "mulliken_population" if ground_state_mulliken_only else "charge_transfer"
            source_metadata["tags"] = sorted(
                {str(tag).strip() for tag in source_tags if str(tag).strip()}
                | {result_tag}
            )
            link_key = (
                "ground_state_mulliken_run_ids"
                if ground_state_mulliken_only
                else "charge_transfer_run_ids"
            )
            linked_jobs = {
                int(value)
                for value in source_metadata.get(link_key, [])
                if str(value).isdigit()
            }
            linked_jobs.add(run.pk)
            source_metadata[link_key] = sorted(linked_jobs)
            spectral_run.metadata = source_metadata
            spectral_run.save(update_fields=["metadata", "updated_at"])

    metadata = dict(run.metadata or {})
    existing_tags = metadata.get("tags") or []
    summary_tags = summary.get("tags") or []
    if isinstance(existing_tags, str):
        existing_tags = [existing_tags]
    if isinstance(summary_tags, str):
        summary_tags = [summary_tags]
    tags = {
        str(tag).strip()
        for tag in list(existing_tags) + list(summary_tags)
        if str(tag).strip()
    }
    if tags:
        metadata["tags"] = sorted(tags)
    summary_for_metadata = dict(summary)
    if isinstance(charge_transfer, dict):
        compact_ct = dict(charge_transfer)
        for key in ("ct_states", "ct_terms", "ct_spectra"):
            rows = list(compact_ct.pop(key, []) or [])
            compact_ct[f"{key}_count"] = len(rows)
        summary_for_metadata["charge_transfer"] = compact_ct
    metadata["result_summary"] = summary_for_metadata
    run.metadata = metadata
    run.save(update_fields=["metadata", "updated_at"])


def _read_structure_snapshot(path: Path) -> dict[str, Any] | None:
    try:
        atoms = ase_read(str(path), index=-1)
    except Exception:
        return None

    if isinstance(atoms, (list, tuple)):
        if not atoms:
            return None
        atoms = atoms[-1]

    try:
        symbols = [str(symbol) for symbol in atoms.get_chemical_symbols()]
        positions = [[float(x), float(y), float(z)] for x, y, z in atoms.get_positions()]
        formula = str(atoms.get_chemical_formula())
    except Exception:
        return None

    return {
        "formula": formula,
        "atom_count": len(symbols),
        "symbols": symbols,
        "positions_A": positions,
    }


def _read_structure_snapshot_from_xyz_text(xyz_text: str) -> dict[str, Any] | None:
    try:
        atoms = ase_read(StringIO(xyz_text), format="xyz")
    except Exception:
        return None

    if isinstance(atoms, (list, tuple)):
        if not atoms:
            return None
        atoms = atoms[-1]

    try:
        symbols = [str(symbol) for symbol in atoms.get_chemical_symbols()]
        positions = [[float(x), float(y), float(z)] for x, y, z in atoms.get_positions()]
        formula = str(atoms.get_chemical_formula())
    except Exception:
        return None

    return {
        "formula": formula,
        "atom_count": len(symbols),
        "symbols": symbols,
        "positions_A": positions,
    }


def _select_latest_candidate(
    candidates: list[tuple[Path, Artifact, str]],
) -> tuple[Path, Artifact, str] | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (row[0].stat().st_mtime_ns if row[0].exists() else 0, str(row[0])),
    )


def _relative_artifact_path(run_dir_text: str, absolute_path: str) -> str:
    run_path = canonical_run_path(run_dir_text)
    abs_path = canonical_run_path(absolute_path)
    if abs_path.startswith(f"{run_path}/"):
        return abs_path[len(run_path) + 1 :]
    return abs_path


def _resolved_remote_path(run_dir_text: str, maybe_path: str) -> str:
    text = normalize_run_path(maybe_path)
    if not text:
        return text
    if text.startswith("/"):
        return text
    root = normalize_run_path(run_dir_text)
    return normalize_run_path(f"{root}/{text.lstrip('./')}")


def _plot_label_for_path(path_value: str) -> str:
    name = Path(str(path_value)).name
    return _KEY_PLOT_FILENAMES.get(name, f"plot:{name}")


def _ingest_local_artifacts_for_run(run: WorkflowRun, run_dir: Path) -> None:
    summary_payload: dict[str, Any] | None = None
    summary_artifact: Artifact | None = None
    log_candidates: list[tuple[Path, Artifact, str]] = []
    structure_candidates: list[tuple[Path, Artifact, str]] = []
    scan_db_candidates: list[tuple[Path, Artifact]] = []
    fallback_source_scheduler_job_id = _latest_scheduler_job_id_for_run(run)
    semantic_index: dict[tuple[str, str, str], Artifact] = {}

    for existing_artifact in Artifact.objects.filter(workflow_run=run).select_related("job").order_by("pk"):
        existing_source_job_id = _artifact_source_scheduler_job_id_for_ingest(
            existing_artifact,
            fallback=fallback_source_scheduler_job_id,
        )
        existing_key = _artifact_semantic_key_for_ingest(
            label=str(existing_artifact.label or ""),
            ingestion_path=_artifact_ingestion_path_for_ingest(existing_artifact),
            source_scheduler_job_id=existing_source_job_id,
        )
        if existing_key is not None and existing_key not in semantic_index:
            semantic_index[existing_key] = existing_artifact

    for kind_text, artifact_path, label in collect_cplas_artifacts(run_dir):
        try:
            rel_path = artifact_path.relative_to(run_dir)
            stored_path = normalize_run_path(str(rel_path))
        except ValueError:
            stored_path = normalize_run_path(str(artifact_path))

        metadata: dict[str, Any] = {
            "source": "cplas_adapter",
            "ingestion_path": normalize_run_path(str(artifact_path)),
        }
        source_scheduler_job_id = _source_scheduler_job_id_from_metadata(
            metadata,
            path_hint=stored_path,
            fallback=fallback_source_scheduler_job_id,
        )
        if source_scheduler_job_id:
            metadata["source_scheduler_job_id"] = source_scheduler_job_id

        semantic_key = _artifact_semantic_key_for_ingest(
            label=label,
            ingestion_path=str(metadata.get("ingestion_path") or stored_path),
            source_scheduler_job_id=source_scheduler_job_id,
        )
        upsert_path = stored_path
        if semantic_key is not None and semantic_key in semantic_index:
            upsert_path = normalize_run_path(str(semantic_index[semantic_key].path or stored_path))

        artifact, _ = Artifact.objects.update_or_create(
            workflow_run=run,
            path=upsert_path,
            defaults={
                "kind": _artifact_kind_from_text(kind_text),
                "label": label,
                "metadata": metadata,
            },
        )
        stored_path = normalize_run_path(str(artifact.path or upsert_path))
        if semantic_key is not None:
            semantic_index[semantic_key] = artifact

        if artifact_path.name == _REACTION_HUB_SUMMARY_FILENAME:
            try:
                payload = _parse_json_payload(artifact_path.read_text(encoding="utf-8"))
            except OSError:
                payload = None
            if isinstance(payload, dict):
                summary_payload = payload
                summary_artifact = artifact

        if artifact_path.name == "orca.out":
            log_candidates.append((artifact_path, artifact, stored_path))
        if (
            artifact.kind in {ArtifactKind.STRUCTURE, ArtifactKind.TRAJECTORY}
            and artifact_path.name != "bond_scan_mep.xyz"
        ):
            structure_candidates.append((artifact_path, artifact, stored_path))
        if artifact_path.name == _SCAN_ENDPOINTS_DB_FILENAME:
            scan_db_candidates.append((artifact_path, artifact))

    if summary_payload is not None:
        _ingest_summary_payload(run=run, summary=summary_payload, source_artifact=summary_artifact)

    selected_log = _select_latest_candidate(log_candidates)
    if selected_log is not None and _run_mode_for_ingest(run) != "charge_transfer":
        log_path, log_artifact, stored_path = selected_log
        parsed = _parse_orca_output_details(log_path)
        energy_h = parsed.get("energy_hartree")
        if energy_h is not None:
            _upsert_property(
                run=run,
                name="final_single_point_energy",
                unit="Eh",
                value_float=float(energy_h),
                source_artifact=log_artifact,
                metadata={"parsed_from": stored_path},
            )
        energy_eV = parsed.get("energy_eV")
        if energy_eV is not None:
            _upsert_property(
                run=run,
                name="final_single_point_energy_eV",
                unit="eV",
                value_float=float(energy_eV),
                source_artifact=log_artifact,
                metadata={"parsed_from": stored_path},
            )

        charge_rows = parsed.get("mulliken_charges")
        if isinstance(charge_rows, list) and charge_rows:
            charges = [float(row["charge"]) for row in charge_rows if "charge" in row]
            symbols = [str(row.get("symbol", "")) for row in charge_rows]
            _store_charges_payload(
                run=run,
                charges=charges,
                symbols=symbols,
                source_artifact=log_artifact,
                charge_method="mulliken",
            )

        force_rows = parsed.get("forces_eV_A")
        if isinstance(force_rows, list) and force_rows:
            vectors = [list(map(float, row["force"])) for row in force_rows if "force" in row]
            symbols = [str(row.get("symbol", "")) for row in force_rows]
            _store_forces_payload(
                run=run,
                vectors=vectors,
                symbols=symbols,
                source_artifact=log_artifact,
            )

    selected_structure = _select_latest_candidate(structure_candidates)
    if selected_structure is not None:
        structure_path, structure_artifact, _stored_path = selected_structure
        snapshot = _read_structure_snapshot(structure_path)
        if snapshot is not None:
            _store_structure_payload(
                run=run,
                formula=str(snapshot.get("formula") or ""),
                atom_count=int(snapshot.get("atom_count") or 0),
                symbols=snapshot.get("symbols"),
                positions_A=snapshot.get("positions_A"),
                source_artifact=structure_artifact,
            )
            artifact_meta = dict(structure_artifact.metadata or {})
            artifact_meta["is_final_structure"] = True
            structure_artifact.metadata = artifact_meta
            structure_artifact.save(update_fields=["metadata", "updated_at"])

    kept_scan_db_artifact_id: int | None = None
    if scan_db_candidates:
        scan_db_path, scan_db_artifact = max(
            scan_db_candidates,
            key=lambda row: (
                row[0].stat().st_mtime_ns if row[0].exists() else 0,
                str(row[0]),
            ),
        )
        scan_profile = _parse_scan_profile_from_db(scan_db_path)
        _store_scan_profile(
            run=run,
            profile=scan_profile,
            source_artifact=scan_db_artifact,
        )
        _ingest_bond_scan_mep_artifacts(
            run=run,
            run_dir=run_dir,
            scan_db_artifact=scan_db_artifact,
            scan_db_path=scan_db_path,
        )
        kept_scan_db_artifact_id = int(scan_db_artifact.pk or 0)
    removed_scan_db = _prune_scan_endpoints_db_duplicates_for_run(
        run,
        run_dir_text=canonical_run_path(str(run_dir)),
        keep_artifact_id=kept_scan_db_artifact_id,
    )
    if removed_scan_db:
        print(
            "Info: Pruned duplicate scan_endpoints.db artifacts for run "
            f"{run.run_name}: removed {removed_scan_db}."
        )

    _upsert_latest_evaluated_traj_fmax_property(run)
    _retain_spectrum_csv_artifacts(run)


def _remote_summary_candidate_paths(run_dir_text: str, *, scheduler_job_id: str | None = None) -> list[str]:
    root = normalize_run_path(run_dir_text)
    candidates = [
        f"{root}/{_REACTION_HUB_SUMMARY_FILENAME}",
        f"{root}/TDDFT_active_learning_shared/{_REACTION_HUB_SUMMARY_FILENAME}",
    ]
    job_id = str(scheduler_job_id or "").strip()
    if job_id:
        # Legacy submit_workflow.sh wrote a duplicate per-job archive under
        # run_dir/<job_id>. Prefer root-level stage-out, but keep those paths
        # as a fallback for already-submitted jobs.
        candidates.extend(
            [
                f"{root}/{job_id}/{_REACTION_HUB_SUMMARY_FILENAME}",
                f"{root}/{job_id}/TDDFT_active_learning_shared/{_REACTION_HUB_SUMMARY_FILENAME}",
            ]
        )
    return candidates


def _find_latest_remote_file(client: object, run_dir_text: str, name_pattern: str) -> str:
    if not hasattr(client, "exec_checked"):
        return ""
    run_path_q = shlex.quote(str(run_dir_text))
    pattern_q = shlex.quote(str(name_pattern))
    cmd = (
        f"find {run_path_q} -type f -name {pattern_q} -printf '%T@|%p\\n' "
        "| sort -nr | head -n 1 | cut -d'|' -f2-"
    )
    try:
        out = str(client.exec_checked(cmd))
    except Exception:
        return ""
    for line in reversed(out.splitlines()):
        candidate = line.strip()
        if candidate.startswith("/"):
            return candidate
    return ""


def _find_remote_files_by_path_pattern(
    client: object,
    run_dir_text: str,
    path_pattern: str,
    *,
    newest_first: bool = False,
) -> list[str]:
    if not hasattr(client, "exec_checked"):
        return []
    run_path_q = shlex.quote(str(run_dir_text))
    pattern_q = shlex.quote(str(path_pattern))
    if newest_first:
        cmd = (
            f"find {run_path_q} -type f -path {pattern_q} -printf '%T@|%p\\n' "
            "| sort -nr | cut -d'|' -f2-"
        )
    else:
        cmd = f"find {run_path_q} -type f -path {pattern_q} | sort"
    try:
        out = str(client.exec_checked(cmd))
    except Exception:
        return []
    found: list[str] = []
    for line in out.splitlines():
        candidate = line.strip()
        if "|" in candidate:
            # Compatibility with callers/tests that may emit "timestamp|path".
            candidate = candidate.split("|", 1)[-1].strip()
        if candidate.startswith("/"):
            found.append(candidate)
    return found


def _find_remote_files_by_path_pattern_with_mtime(
    client: object,
    run_dir_text: str,
    path_pattern: str,
    *,
    newest_first: bool = False,
) -> list[tuple[float | None, str]]:
    if not hasattr(client, "exec_checked"):
        return []
    run_path_q = shlex.quote(str(run_dir_text))
    pattern_q = shlex.quote(str(path_pattern))
    cmd = (
        f"find {run_path_q} -type f -path {pattern_q} -printf '%T@|%p\\n' "
        + ("| sort -nr" if newest_first else "| sort")
    )
    try:
        out = str(client.exec_checked(cmd))
    except Exception:
        return []

    found: list[tuple[float | None, str]] = []
    for line in out.splitlines():
        candidate = line.strip()
        mtime_epoch: float | None = None
        if "|" in candidate:
            mtime_text, candidate = candidate.split("|", 1)
            candidate = candidate.strip()
            mtime_epoch = _parse_optional_float(mtime_text.strip())
        if candidate.startswith("/"):
            found.append((mtime_epoch, candidate))
    return found


def _safe_cache_relative_path(path_value: str) -> Path:
    raw = str(path_value or "").strip().replace("\\", "/").lstrip("/")
    parts = [part for part in Path(raw).parts if part and part not in {".", ".."}]
    if not parts:
        return Path("artifact.dat")
    return Path(*parts)


def _remote_cache_target_path(run: WorkflowRun, artifact_rel_path: str) -> Path:
    cache_root = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()
    run_key = str(getattr(run, "run_uid", "") or f"run_{run.pk}")
    return cache_root / run_key / _safe_cache_relative_path(artifact_rel_path)


def _read_remote_file_bytes(client: object, remote_path: str) -> bytes | None:
    if not hasattr(client, "exec_checked"):
        return None

    start_marker = "__REACTION_HUB_B64_START__"
    end_marker = "__REACTION_HUB_B64_END__"
    path_q = shlex.quote(str(remote_path))
    cmd = (
        f"printf '{start_marker}\\n'; "
        f"base64 < {path_q} | tr -d '\\n'; "
        f"printf '\\n{end_marker}\\n'"
    )
    try:
        out = str(client.exec_checked(cmd))
    except Exception:
        return None

    start_idx = out.find(start_marker)
    if start_idx < 0:
        return None
    body_start = start_idx + len(start_marker)
    end_idx = out.find(end_marker, body_start)
    if end_idx < 0:
        return None

    encoded = out[body_start:end_idx].strip()
    if not encoded:
        return b""
    try:
        return base64.b64decode(encoded.encode("ascii"), validate=False)
    except Exception:
        return None


def _download_remote_evaluated_traj_artifacts(
    run: WorkflowRun,
    run_dir_text: str,
    client: object,
    *,
    download_missing_only: bool = False,
    fallback_source_scheduler_job_id: str = "",
    retain_latest_only: bool = False,
) -> int:
    if not hasattr(client, "exec_checked"):
        return 0

    remote_paths = _find_remote_files_by_path_pattern(
        client,
        run_dir_text,
        "*/active_learning/evaluated.traj",
        newest_first=retain_latest_only,
    )
    if not remote_paths:
        return 0
    if not retain_latest_only and len(remote_paths) > 1:
        effective_suffixes = {
            _evaluated_traj_effective_suffix(run_dir_text, remote_path)
            for remote_path in remote_paths
        }
        effective_suffixes.discard("")
        # Older single-optimisation runs may miss submission.run_mode metadata.
        # If all discovered evaluated.traj paths map to the same effective suffix,
        # treat it as single-result ingestion and keep only the final copy.
        if len(effective_suffixes) == 1:
            retain_latest_only = True

    if retain_latest_only and len(remote_paths) > 1:
        indexed_paths = list(enumerate(remote_paths))
        selected_idx, selected_path = min(
            indexed_paths,
            key=lambda row: (
                *_single_optimisation_path_penalty(run_dir_text, row[1]),
                int(row[0]),
                normalize_run_path(row[1]),
            ),
        )
        del selected_idx
        remote_paths = [selected_path]

    semantic_index: dict[tuple[str, str, str], Artifact] = {}
    for existing_artifact in Artifact.objects.filter(workflow_run=run).select_related("job").order_by("pk"):
        existing_source_job_id = _artifact_source_scheduler_job_id_for_ingest(
            existing_artifact,
            fallback=fallback_source_scheduler_job_id,
        )
        existing_key = _artifact_semantic_key_for_ingest(
            label=str(existing_artifact.label or ""),
            ingestion_path=_artifact_ingestion_path_for_ingest(existing_artifact),
            source_scheduler_job_id=existing_source_job_id,
        )
        if existing_key is not None and existing_key not in semantic_index:
            semantic_index[existing_key] = existing_artifact

    available_count = 0
    seen_semantic_keys: set[tuple[str, str, str]] = set()
    kept_artifact: Artifact | None = None
    kept_source_scheduler_job_id = ""
    for remote_path in remote_paths:
        rel_path = normalize_run_path(_relative_artifact_path(run_dir_text, remote_path))
        remote_path_normalized = normalize_run_path(remote_path)
        label = f"trajectory:{Path(rel_path).name}"
        metadata: dict[str, Any] = {
            "source": "remote_download",
            "downloaded_from_hpc": True,
            "remote_path": remote_path_normalized,
            "ingestion_path": remote_path_normalized,
        }
        source_scheduler_job_id = _source_scheduler_job_id_from_metadata(
            metadata,
            path_hint=rel_path,
            fallback=fallback_source_scheduler_job_id,
        )
        if source_scheduler_job_id:
            metadata["source_scheduler_job_id"] = source_scheduler_job_id

        semantic_key = _artifact_semantic_key_for_ingest(
            label=label,
            ingestion_path=remote_path_normalized,
            source_scheduler_job_id=source_scheduler_job_id,
        )
        if semantic_key is not None and semantic_key in seen_semantic_keys:
            continue
        upsert_path = rel_path
        if semantic_key is not None and semantic_key in semantic_index:
            upsert_path = normalize_run_path(str(semantic_index[semantic_key].path or rel_path))

        local_cache_path = _remote_cache_target_path(run, rel_path)
        has_local_copy = (
            download_missing_only
            and local_cache_path.exists()
            and local_cache_path.is_file()
        )
        if not has_local_copy:
            traj_bytes = _read_remote_file_bytes(client, remote_path)
            if traj_bytes is None:
                continue
            try:
                local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                local_cache_path.write_bytes(traj_bytes)
            except Exception:
                continue

        metadata["local_cache_path"] = str(local_cache_path)
        artifact, _ = Artifact.objects.update_or_create(
            workflow_run=run,
            path=upsert_path,
            defaults={
                "kind": ArtifactKind.TRAJECTORY,
                "label": label,
                "metadata": metadata,
            },
        )
        if semantic_key is not None:
            semantic_index[semantic_key] = artifact
            seen_semantic_keys.add(semantic_key)
        if retain_latest_only:
            kept_artifact = artifact
            kept_source_scheduler_job_id = str(source_scheduler_job_id or fallback_source_scheduler_job_id or "").strip()
        available_count += 1
    if retain_latest_only and kept_artifact is not None and kept_source_scheduler_job_id:
        removed = _prune_evaluated_traj_duplicates_for_source_job(
            run,
            source_scheduler_job_id=kept_source_scheduler_job_id,
            keep_artifact_id=int(kept_artifact.pk or 0),
        )
        if removed:
            print(
                "Info: Pruned duplicate evaluated.traj artifacts for single optimisation "
                f"(source job {kept_source_scheduler_job_id}): removed {removed}."
            )
    removed_all = _prune_evaluated_traj_duplicates_for_all_sources(run, run_dir_text=run_dir_text)
    if removed_all:
        print(
            "Info: Pruned duplicate evaluated.traj artifacts grouped by source job id: "
            f"removed {removed_all}."
        )
    return available_count


def _download_remote_scan_endpoints_db_artifacts(
    run: WorkflowRun,
    run_dir_text: str,
    client: object,
    *,
    download_missing_only: bool = False,
    fallback_source_scheduler_job_id: str = "",
) -> int:
    if not hasattr(client, "exec_checked"):
        return 0

    remote_paths = _find_remote_files_by_path_pattern(
        client,
        run_dir_text,
        "*/scan_endpoints.db",
        newest_first=True,
    )
    if not remote_paths:
        return 0

    available_count = 0
    best_profile: dict[str, Any] | None = None
    best_profile_artifact: Artifact | None = None
    best_profile_score: tuple[int, int, int, int, int, int, str] | None = None
    keep_artifact: Artifact | None = None
    keep_score: tuple[int, int, int, int, int, int, str] | None = None

    for order_index, remote_path in enumerate(remote_paths):
        rel_path = normalize_run_path(_relative_artifact_path(run_dir_text, remote_path))
        remote_path_normalized = normalize_run_path(remote_path)
        metadata: dict[str, Any] = {
            "source": "remote_download",
            "downloaded_from_hpc": True,
            "remote_path": remote_path_normalized,
            "ingestion_path": remote_path_normalized,
        }
        source_scheduler_job_id = _source_scheduler_job_id_from_metadata(
            metadata,
            path_hint=rel_path,
            fallback=fallback_source_scheduler_job_id,
        )
        if source_scheduler_job_id:
            metadata["source_scheduler_job_id"] = source_scheduler_job_id

        local_cache_path = _remote_cache_target_path(run, rel_path)
        has_local_copy = (
            download_missing_only
            and local_cache_path.exists()
            and local_cache_path.is_file()
        )
        if not has_local_copy:
            db_bytes = _read_remote_file_bytes(client, remote_path)
            if db_bytes is None:
                continue
            try:
                local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                local_cache_path.write_bytes(db_bytes)
            except Exception:
                continue

        db_artifact, _ = Artifact.objects.update_or_create(
            workflow_run=run,
            path=rel_path,
            defaults={
                "kind": ArtifactKind.DATABASE,
                "label": "scan_endpoints_db",
                "metadata": {**metadata, "local_cache_path": str(local_cache_path)},
            },
        )
        available_count += 1

        point_count = 0
        profile: dict[str, Any] | None = None
        if not local_cache_path.exists() or not local_cache_path.is_file():
            profile = None
        else:
            profile = _parse_scan_profile_from_db(local_cache_path)
            if profile:
                points = profile.get("points")
                point_count = len(points) if isinstance(points, list) else 0
        score = _scan_endpoints_profile_selection_score(
            run_dir_text=run_dir_text,
            remote_path=remote_path_normalized,
            point_count=point_count,
            expected_source_scheduler_job_id=fallback_source_scheduler_job_id,
            order_index=order_index,
        )
        if keep_score is None or score > keep_score:
            keep_artifact = db_artifact
            keep_score = score
        if not profile:
            continue
        if best_profile_score is None or score > best_profile_score:
            best_profile = profile
            best_profile_artifact = db_artifact
            best_profile_score = score

    if best_profile is not None:
        _store_scan_profile(
            run=run,
            profile=best_profile,
            source_artifact=best_profile_artifact,
        )
    keep_artifact_id = (
        int(best_profile_artifact.pk or 0)
        if best_profile_artifact is not None
        else (int(keep_artifact.pk or 0) if keep_artifact is not None else None)
    )
    removed_scan_db = _prune_scan_endpoints_db_duplicates_for_run(
        run,
        run_dir_text=run_dir_text,
        keep_artifact_id=keep_artifact_id,
    )
    if removed_scan_db:
        print(
            "Info: Pruned duplicate remote scan_endpoints.db artifacts for run "
            f"{run.run_name}: removed {removed_scan_db}."
        )
    return available_count


def _download_remote_scan_plot_artifacts(
    run: WorkflowRun,
    run_dir_text: str,
    client: object,
    *,
    download_missing_only: bool = False,
    fallback_source_scheduler_job_id: str = "",
) -> int:
    if not hasattr(client, "exec_checked"):
        return 0

    remote_paths: list[str] = []
    for ext in ("png", "jpg", "jpeg", "svg"):
        for pattern in (
            f"*/scan_plots/*.{ext}",
            f"*/final_tddft_spectrum/*.{ext}",
        ):
            remote_paths.extend(
                _find_remote_files_by_path_pattern(
                    client,
                    run_dir_text,
                    pattern,
                    newest_first=True,
                )
            )
    remote_paths.extend(
        _find_remote_files_by_path_pattern(
            client,
            run_dir_text,
            "*/final_tddft_spectrum/*.csv",
            newest_first=True,
        )
    )
    if not remote_paths:
        return 0

    available_count = 0
    seen: set[str] = set()
    for remote_path in remote_paths:
        normalized_remote_path = normalize_run_path(remote_path)
        if not normalized_remote_path or normalized_remote_path in seen:
            continue
        if _is_legacy_nested_cluster_tddft_spectrum_artifact(normalized_remote_path):
            continue
        seen.add(normalized_remote_path)
        path_suffix = Path(normalized_remote_path).suffix.lower()
        is_csv = path_suffix == ".csv"
        if not is_csv and _should_skip_scan_plot_image(normalized_remote_path):
            continue

        rel_path = normalize_run_path(_relative_artifact_path(run_dir_text, normalized_remote_path))
        local_cache_path = _remote_cache_target_path(run, rel_path)
        has_local_copy = (
            download_missing_only
            and local_cache_path.exists()
            and local_cache_path.is_file()
        )
        image_bytes = None
        if not has_local_copy:
            image_bytes = _read_remote_file_bytes(client, normalized_remote_path)
            if image_bytes is not None:
                try:
                    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    local_cache_path.write_bytes(image_bytes)
                    has_local_copy = True
                except Exception:
                    image_bytes = None

        metadata: dict[str, Any] = {
            "source": "remote_scan_plots",
            "remote_path": normalized_remote_path,
            "ingestion_path": normalized_remote_path,
        }
        source_scheduler_job_id = str(fallback_source_scheduler_job_id or "").strip()
        if source_scheduler_job_id:
            metadata["source_scheduler_job_id"] = source_scheduler_job_id
        if has_local_copy or image_bytes is not None:
            metadata["downloaded_from_hpc"] = True
            metadata["local_cache_path"] = str(local_cache_path)
        else:
            metadata["remote_only"] = True
            metadata["retained_on_hpc"] = True
            metadata["raw_fetch_disabled"] = True

        Artifact.objects.update_or_create(
            workflow_run=run,
            path=rel_path,
            defaults={
                "kind": ArtifactKind.CSV if is_csv else ArtifactKind.IMAGE,
                "label": (
                    f"spectrum:{Path(normalized_remote_path).name}"
                    if is_csv
                    else _plot_label_for_path(normalized_remote_path)
                ),
                "metadata": metadata,
            },
        )
        available_count += 1

    return available_count


def _ingest_remote_summary_for_job(
    job: JobRecord,
    client: object,
    *,
    download_missing_only: bool = False,
) -> bool:
    run = job.workflow_run

    def _return_with_latest_evaluated_traj_fmax(result: bool) -> bool:
        try:
            _upsert_latest_evaluated_traj_fmax_property(run)
        except Exception:
            pass
        return bool(result)

    if not hasattr(client, "file_exists") or not hasattr(client, "read_text"):
        return _return_with_latest_evaluated_traj_fmax(False)

    run_dir_text = canonical_run_path(run.working_directory)
    ground_state_mulliken_only = _is_ground_state_mulliken_only_run(run)
    retain_latest_evaluated_only = _is_single_optimisation_run(run)
    skip_evaluated_traj_download = False
    if _run_mode_for_ingest(run) == "tddft_bond_scan":
        latest_scan_db_rows = _find_remote_files_by_path_pattern_with_mtime(
            client,
            run_dir_text,
            "*/scan_endpoints.db",
            newest_first=True,
        )
        latest_scan_db_mtime = latest_scan_db_rows[0][0] if latest_scan_db_rows else None
        submitted_at = job.submitted_at
        submitted_at_epoch: float | None = None
        if submitted_at is not None:
            try:
                if submitted_at.tzinfo is None:
                    submitted_at = submitted_at.replace(tzinfo=dt_timezone.utc)
                submitted_at_epoch = float(submitted_at.timestamp())
            except Exception:
                submitted_at_epoch = None
        skip_evaluated_traj_download = (
            submitted_at_epoch is not None
            and latest_scan_db_mtime is not None
            and float(latest_scan_db_mtime) > float(submitted_at_epoch)
        )

    is_ct_run = _run_mode_for_ingest(run) == "charge_transfer"
    available_evaluated_count = 0
    if not ground_state_mulliken_only and not is_ct_run and not skip_evaluated_traj_download:
        available_evaluated_count = _download_remote_evaluated_traj_artifacts(
            run,
            run_dir_text,
            client,
            download_missing_only=download_missing_only,
            fallback_source_scheduler_job_id=str(job.scheduler_job_id or "").strip(),
            retain_latest_only=retain_latest_evaluated_only,
        )
    available_scan_db_count = 0
    available_scan_plot_count = 0
    if not ground_state_mulliken_only and not is_ct_run:
        available_scan_db_count = _download_remote_scan_endpoints_db_artifacts(
            run,
            run_dir_text,
            client,
            download_missing_only=download_missing_only,
            fallback_source_scheduler_job_id=str(job.scheduler_job_id or "").strip(),
        )
        available_scan_plot_count = _download_remote_scan_plot_artifacts(
            run,
            run_dir_text,
            client,
            download_missing_only=download_missing_only,
            fallback_source_scheduler_job_id=str(job.scheduler_job_id or "").strip(),
        )
        _retain_spectrum_csv_artifacts(run)
    coadsorbate_screening_run = _is_coadsorbate_screening_run(run)

    def _allow_local_download_for_remote_summary_image() -> bool:
        # For co-adsorbate screening we keep plots remote-only and only cache the final DB.
        return not coadsorbate_screening_run

    def _allow_local_download_for_remote_summary_database(normalized_remote_path: str) -> bool:
        if not coadsorbate_screening_run:
            return True
        return Path(normalized_remote_path).name.lower() == "coadsorption_final_for_dft.db"

    def _ingest_summary_from_remote_path(remote_summary_path: str, *, require_exists_check: bool) -> bool:
        if require_exists_check:
            try:
                exists = bool(client.file_exists(remote_summary_path))
            except Exception:
                return False
            if not exists:
                return False

        try:
            raw_summary = client.read_text(remote_summary_path)
            payload = _parse_json_payload(raw_summary)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False

        def _upsert_remote_summary_image_artifact(raw_image_path: str, *, plot_key: str = "") -> None:
            image_text = str(raw_image_path or "").strip()
            if not image_text:
                return
            if _should_skip_scan_plot_image(image_text):
                return
            resolved_remote_path = _resolved_remote_path(run_dir_text, image_text)
            normalized_remote_path = normalize_run_path(resolved_remote_path)
            stored_path = _relative_artifact_path(run_dir_text, normalized_remote_path)
            allow_local_download = _allow_local_download_for_remote_summary_image()
            metadata: dict[str, Any] = {
                "source": "remote_summary",
                "remote_path": normalized_remote_path,
                "ingestion_path": normalized_remote_path,
            }
            if plot_key:
                metadata["plot_key"] = plot_key
            source_scheduler_job_id = str(job.scheduler_job_id or "").strip()
            if source_scheduler_job_id:
                metadata["source_scheduler_job_id"] = source_scheduler_job_id

            if not allow_local_download:
                metadata["remote_only"] = True
                metadata["retained_on_hpc"] = True
                metadata["raw_fetch_disabled"] = True
                Artifact.objects.update_or_create(
                    workflow_run=run,
                    path=stored_path,
                    defaults={
                        "kind": ArtifactKind.IMAGE,
                        "label": _plot_label_for_path(image_text),
                        "metadata": metadata,
                    },
                )
                return

            local_cache_path = _remote_cache_target_path(run, stored_path)
            existing_artifact = (
                Artifact.objects.filter(workflow_run=run, path=stored_path).only("metadata").first()
            )
            existing_metadata = (
                existing_artifact.metadata
                if existing_artifact is not None and isinstance(existing_artifact.metadata, dict)
                else {}
            )
            existing_cache_path_text = str(existing_metadata.get("local_cache_path") or "").strip()
            has_existing_local_copy = False
            if existing_cache_path_text:
                existing_cache_path = Path(existing_cache_path_text)
                has_existing_local_copy = existing_cache_path.exists() and existing_cache_path.is_file()
                if has_existing_local_copy:
                    local_cache_path = existing_cache_path
            if not has_existing_local_copy and local_cache_path.exists() and local_cache_path.is_file():
                has_existing_local_copy = True

            image_bytes = None
            should_fetch_remote = not (download_missing_only and has_existing_local_copy)
            if should_fetch_remote:
                image_bytes = _read_remote_file_bytes(client, normalized_remote_path)
            if image_bytes is not None:
                try:
                    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    local_cache_path.write_bytes(image_bytes)
                except Exception:
                    image_bytes = None
            has_local_copy = bool(image_bytes is not None or has_existing_local_copy)
            if has_local_copy:
                metadata["downloaded_from_hpc"] = True
                metadata["local_cache_path"] = str(local_cache_path)
            else:
                metadata["remote_only"] = True
                metadata["retained_on_hpc"] = True
                metadata["raw_fetch_disabled"] = True

            Artifact.objects.update_or_create(
                workflow_run=run,
                path=stored_path,
                defaults={
                    "kind": ArtifactKind.IMAGE,
                    "label": _plot_label_for_path(image_text),
                    "metadata": metadata,
                },
            )

        def _upsert_remote_summary_database_artifact(raw_db_path: str, *, label_prefix: str = "database") -> None:
            db_text = str(raw_db_path or "").strip()
            if not db_text:
                return
            resolved_remote_path = _resolved_remote_path(run_dir_text, db_text)
            normalized_remote_path = normalize_run_path(resolved_remote_path)
            stored_path = _relative_artifact_path(run_dir_text, normalized_remote_path)
            allow_local_download = _allow_local_download_for_remote_summary_database(normalized_remote_path)
            metadata: dict[str, Any] = {
                "source": "remote_summary",
                "remote_path": normalized_remote_path,
                "ingestion_path": normalized_remote_path,
            }
            source_scheduler_job_id = str(job.scheduler_job_id or "").strip()
            if source_scheduler_job_id:
                metadata["source_scheduler_job_id"] = source_scheduler_job_id

            if not allow_local_download:
                metadata["remote_only"] = True
                metadata["retained_on_hpc"] = True
                metadata["raw_fetch_disabled"] = True
                Artifact.objects.update_or_create(
                    workflow_run=run,
                    path=stored_path,
                    defaults={
                        "kind": ArtifactKind.DATABASE,
                        "label": f"{label_prefix}:{Path(db_text).name}",
                        "metadata": metadata,
                    },
                )
                return

            local_cache_path = _remote_cache_target_path(run, stored_path)
            existing_artifact = (
                Artifact.objects.filter(workflow_run=run, path=stored_path).only("metadata").first()
            )
            existing_metadata = (
                existing_artifact.metadata
                if existing_artifact is not None and isinstance(existing_artifact.metadata, dict)
                else {}
            )
            existing_cache_path_text = str(existing_metadata.get("local_cache_path") or "").strip()
            has_existing_local_copy = False
            if existing_cache_path_text:
                existing_cache_path = Path(existing_cache_path_text)
                has_existing_local_copy = existing_cache_path.exists() and existing_cache_path.is_file()
                if has_existing_local_copy:
                    local_cache_path = existing_cache_path
            if not has_existing_local_copy and local_cache_path.exists() and local_cache_path.is_file():
                has_existing_local_copy = True

            db_bytes = None
            should_fetch_remote = not (download_missing_only and has_existing_local_copy)
            if should_fetch_remote:
                db_bytes = _read_remote_file_bytes(client, normalized_remote_path)
            if db_bytes is not None:
                try:
                    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    local_cache_path.write_bytes(db_bytes)
                except Exception:
                    db_bytes = None
            has_local_copy = bool(db_bytes is not None or has_existing_local_copy)
            if has_local_copy:
                metadata["downloaded_from_hpc"] = True
                metadata["local_cache_path"] = str(local_cache_path)
            else:
                metadata["remote_only"] = True
                metadata["retained_on_hpc"] = True
                metadata["raw_fetch_disabled"] = True

            Artifact.objects.update_or_create(
                workflow_run=run,
                path=stored_path,
                defaults={
                    "kind": ArtifactKind.DATABASE,
                    "label": f"{label_prefix}:{Path(db_text).name}",
                    "metadata": metadata,
                },
            )

        def _iter_summary_db_paths(raw_value: object) -> list[str]:
            if isinstance(raw_value, str):
                value = raw_value.strip()
                return [value] if value else []
            if isinstance(raw_value, list):
                values: list[str] = []
                for entry in raw_value:
                    if not isinstance(entry, str):
                        continue
                    value = entry.strip()
                    if value:
                        values.append(value)
                return values
            return []

        def _upsert_declared_artifact(raw_path: str) -> None:
            path_text = str(raw_path or "").strip()
            if not path_text or Path(path_text).name == _REACTION_HUB_SUMMARY_FILENAME:
                return
            normalized_remote_path = normalize_run_path(
                _resolved_remote_path(run_dir_text, path_text)
            )
            normalized_run_dir = normalize_run_path(run_dir_text).rstrip("/")
            if not normalized_remote_path.startswith(f"{normalized_run_dir}/"):
                return

            stored_path = _relative_artifact_path(run_dir_text, normalized_remote_path)
            suffix = Path(path_text).suffix.lower()
            if suffix in {".png", ".jpg", ".jpeg", ".svg", ".gif"}:
                _upsert_remote_summary_image_artifact(path_text)
                return
            if suffix == ".db":
                _upsert_remote_summary_database_artifact(path_text)
                return

            kind = {
                ".csv": ArtifactKind.CSV,
                ".json": ArtifactKind.JSON,
                ".xyz": ArtifactKind.STRUCTURE,
                ".extxyz": ArtifactKind.STRUCTURE,
                ".traj": ArtifactKind.TRAJECTORY,
                ".log": ArtifactKind.LOG,
                ".out": ArtifactKind.LOG,
            }.get(suffix, ArtifactKind.OTHER)
            name = Path(path_text).name
            if kind == ArtifactKind.STRUCTURE:
                label = f"structure:{name}"
            elif name.endswith(".molden.input"):
                label = f"molden:{name}"
            elif suffix == ".pov":
                label = f"povray_scene:{name}"
            elif suffix == ".ini":
                label = f"povray_ini:{name}"
            elif suffix in {".cub", ".cube"}:
                label = f"density_cube:{name}"
            elif "CT_multiple" in Path(path_text).parts or name in {
                "IFCTdata.txt",
                "IFCTmajor.txt",
                "config.txt",
            }:
                label = f"ct_data:{name}"
            else:
                label = f"artifact:{name}"

            metadata: dict[str, Any] = {
                "source": "remote_summary",
                "remote_path": normalized_remote_path,
                "ingestion_path": normalized_remote_path,
            }
            source_scheduler_job_id = str(job.scheduler_job_id or "").strip()
            if source_scheduler_job_id:
                metadata["source_scheduler_job_id"] = source_scheduler_job_id

            local_cache_path = _remote_cache_target_path(run, stored_path)
            existing = Artifact.objects.filter(workflow_run=run, path=stored_path).first()
            existing_metadata = existing.metadata if existing and isinstance(existing.metadata, dict) else {}
            existing_cache = Path(str(existing_metadata.get("local_cache_path") or ""))
            has_local_copy = existing_cache.is_file() if str(existing_cache) not in {"", "."} else False
            if has_local_copy:
                local_cache_path = existing_cache

            download_cubes = bool(getattr(settings, "REACTION_HUB_INGEST_DOWNLOAD_CUBES", False))
            retained_suffixes = {
                ".txt", ".csv", ".json", ".png", ".jpg", ".jpeg", ".svg",
                ".extxyz", ".xyz", ".pov", ".ini", ".py", ".sh",
            }
            if download_cubes:
                retained_suffixes.update({".cub", ".cube"})
            skip_download_names = {"log.txt", "orca.out", "orca.molden.input"}
            is_scratch_intermediate = (
                "step_" in Path(stored_path).parts
                or any(p.startswith("step_") for p in Path(stored_path).parts)
                or Path(stored_path).name.endswith(("_config.txt", "_log.txt", ".mwfn"))
            )
            content = None
            if (
                suffix in retained_suffixes
                and Path(stored_path).name.lower() not in skip_download_names
                and not is_scratch_intermediate
                and not (download_missing_only and has_local_copy)
            ):
                content = _read_remote_file_bytes(client, normalized_remote_path)


            if content is not None:
                try:
                    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
                    local_cache_path.write_bytes(content)
                    has_local_copy = True
                except OSError:
                    pass
            if has_local_copy:
                metadata["downloaded_from_hpc"] = True
                metadata["local_cache_path"] = str(local_cache_path)
            else:
                metadata.update({"remote_only": True, "retained_on_hpc": True})

            Artifact.objects.update_or_create(
                workflow_run=run,
                path=stored_path,
                defaults={"kind": kind, "label": label, "metadata": metadata},
            )

        artifact_path = _relative_artifact_path(run_dir_text, remote_summary_path)
        summary_cache_path = _remote_cache_target_path(run, artifact_path)
        summary_metadata = {
            "source": "remote_summary",
            "remote_path": normalize_run_path(remote_summary_path),
        }
        try:
            summary_cache_path.parent.mkdir(parents=True, exist_ok=True)
            summary_cache_path.write_text(raw_summary, encoding="utf-8")
            summary_metadata.update(
                {"downloaded_from_hpc": True, "local_cache_path": str(summary_cache_path)}
            )
        except OSError:
            summary_metadata.update({"remote_only": True, "retained_on_hpc": True})
        summary_artifact, _ = Artifact.objects.update_or_create(
            workflow_run=run,
            path=artifact_path,
            defaults={
                "kind": ArtifactKind.JSON,
                "label": "reaction_hub_summary",
                "metadata": summary_metadata,
            },
        )
        _ingest_summary_payload(run=run, summary=payload, source_artifact=summary_artifact)
        if ground_state_mulliken_only:
            return True

        final = payload.get("final")
        if isinstance(final, dict):
            for key, label, kind in (
                ("orca_out", "orca_out:final", ArtifactKind.LOG),
                ("opt_dir", "opt_dir:final", ArtifactKind.OTHER),
            ):
                raw_path = str(final.get(key) or "").strip()
                if not raw_path:
                    continue
                resolved_remote_path = _resolved_remote_path(run_dir_text, raw_path)
                artifact_metadata = {"source": "remote_summary", "remote_path": resolved_remote_path}
                if key == "orca_out":
                    artifact_metadata.update(
                        {
                            "remote_only": True,
                            "retained_on_hpc": True,
                            "raw_fetch_disabled": True,
                        }
                    )
                Artifact.objects.update_or_create(
                    workflow_run=run,
                    path=_relative_artifact_path(run_dir_text, resolved_remote_path),
                    defaults={
                        "kind": kind,
                        "label": label,
                        "metadata": artifact_metadata,
                    },
                )

            xyz_paths = final.get("final_xyz_paths")
            if isinstance(xyz_paths, list):
                for xyz_path in xyz_paths:
                    xyz_text = str(xyz_path or "").strip()
                    if not xyz_text:
                        continue
                    Artifact.objects.update_or_create(
                        workflow_run=run,
                        path=_relative_artifact_path(run_dir_text, xyz_text),
                        defaults={
                            "kind": ArtifactKind.STRUCTURE,
                            "label": "structure:reaction_hub_final.xyz",
                            "metadata": {"source": "remote_summary", "is_final_structure": True},
                    },
                )

            cluster_fragment = final.get("cluster_fragment")
            if isinstance(cluster_fragment, dict):
                for key, label, kind in (
                    ("orca_out", "orca_out:final_cluster_fragment", ArtifactKind.LOG),
                    ("directory", "opt_dir:final_cluster_fragment", ArtifactKind.OTHER),
                ):
                    raw_path = str(cluster_fragment.get(key) or "").strip()
                    if not raw_path:
                        continue
                    resolved_remote_path = _resolved_remote_path(run_dir_text, raw_path)
                    artifact_metadata = {
                        "source": "remote_summary",
                        "remote_path": resolved_remote_path,
                        "cluster_fragment": True,
                    }
                    if key == "orca_out":
                        artifact_metadata.update(
                            {
                                "remote_only": True,
                                "retained_on_hpc": True,
                                "raw_fetch_disabled": True,
                            }
                        )
                    Artifact.objects.update_or_create(
                        workflow_run=run,
                        path=_relative_artifact_path(run_dir_text, resolved_remote_path),
                        defaults={
                            "kind": kind,
                            "label": label,
                            "metadata": artifact_metadata,
                        },
                    )

                for raw_traj_path in _iter_summary_db_paths(cluster_fragment.get("traj_paths")):
                    traj_text = str(raw_traj_path or "").strip()
                    if not traj_text:
                        continue
                    resolved_remote_path = _resolved_remote_path(run_dir_text, traj_text)
                    Artifact.objects.update_or_create(
                        workflow_run=run,
                        path=_relative_artifact_path(run_dir_text, resolved_remote_path),
                        defaults={
                            "kind": ArtifactKind.TRAJECTORY,
                            "label": f"trajectory:{Path(traj_text).name}",
                            "metadata": {
                                "source": "remote_summary",
                                "remote_path": resolved_remote_path,
                                "cluster_fragment": True,
                            },
                        },
                    )

            for db_key in (
                "dft_db_paths",
                "database_paths",
                "final_db_paths",
                "dft_db_path",
                "database_path",
                "final_db_path",
            ):
                for raw_db_path in _iter_summary_db_paths(final.get(db_key)):
                    _upsert_remote_summary_database_artifact(
                        raw_db_path,
                        label_prefix="database",
                    )

        plots = payload.get("plots")
        if isinstance(plots, dict):
            for key in (
                "stacked_uvvis",
                "stacked_ct",
                "energy_vs_reaction_coordinate",
                "converged_state_wavelengths",
                "final_tddft_spectrum",
            ):
                raw_plot = str(plots.get(key) or "").strip()
                if not raw_plot:
                    continue
                _upsert_remote_summary_image_artifact(raw_plot, plot_key=key)
            raw_images = plots.get("images")
            if isinstance(raw_images, list):
                for raw_image in raw_images:
                    _upsert_remote_summary_image_artifact(str(raw_image or "").strip())

        artifacts_payload = payload.get("artifacts")
        if isinstance(artifacts_payload, list):
            for raw_artifact_path in artifacts_payload:
                if isinstance(raw_artifact_path, str):
                    _upsert_declared_artifact(raw_artifact_path)
        elif isinstance(artifacts_payload, dict):
            for db_key in ("db_paths", "database_paths", "db_path", "database_path"):
                for raw_db_path in _iter_summary_db_paths(artifacts_payload.get(db_key)):
                    _upsert_remote_summary_database_artifact(
                        raw_db_path,
                        label_prefix="database",
                    )

        for db_key in ("dft_db_paths", "database_paths", "db_paths", "dft_db_path", "database_path", "db_path"):
            for raw_db_path in _iter_summary_db_paths(payload.get(db_key)):
                _upsert_remote_summary_database_artifact(
                    raw_db_path,
                    label_prefix="database",
                )
        return True

    for remote_summary_path in _remote_summary_candidate_paths(
        run_dir_text,
        scheduler_job_id=job.scheduler_job_id,
    ):
        if _ingest_summary_from_remote_path(remote_summary_path, require_exists_check=True):
            return _return_with_latest_evaluated_traj_fmax(True)

    latest_summary_path = _find_latest_remote_file(client, run_dir_text, _REACTION_HUB_SUMMARY_FILENAME)
    if latest_summary_path and _ingest_summary_from_remote_path(
        latest_summary_path, require_exists_check=False
    ):
        return _return_with_latest_evaluated_traj_fmax(True)

    remote_orca_out = _find_latest_remote_file(client, run_dir_text, "orca.out")
    if not remote_orca_out:
        return _return_with_latest_evaluated_traj_fmax(
            (available_evaluated_count > 0)
            or (available_scan_db_count > 0)
            or (available_scan_plot_count > 0)
        )

    log_artifact, _ = Artifact.objects.update_or_create(
        workflow_run=run,
        path=_relative_artifact_path(run_dir_text, remote_orca_out),
        defaults={
            "kind": ArtifactKind.LOG,
            "label": "orca_out:remote_final",
            "metadata": {
                "source": "remote_reference_only",
                "remote_only": True,
                "retained_on_hpc": True,
                "raw_fetch_disabled": True,
                "remote_path": normalize_run_path(remote_orca_out),
            },
        },
    )
    if not settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ:
        return _return_with_latest_evaluated_traj_fmax(True)

    if not hasattr(client, "read_text"):
        return _return_with_latest_evaluated_traj_fmax(True)
    try:
        if hasattr(client, "exec_checked"):
            orca_text = client.exec_checked(f"tail -n 25000 {shlex.quote(remote_orca_out)}")
        else:
            orca_text = client.read_text(remote_orca_out)
    except Exception:
        return _return_with_latest_evaluated_traj_fmax(True)
    parsed = _parse_orca_output_details_from_text(orca_text)
    if not parsed:
        return _return_with_latest_evaluated_traj_fmax(True)

    energy_h = parsed.get("energy_hartree")
    if energy_h is not None:
        _upsert_property(
            run=run,
            name="final_single_point_energy",
            unit="Eh",
            value_float=float(energy_h),
            source_artifact=log_artifact,
            metadata={"source": "remote_orca_parse"},
        )
    energy_eV = parsed.get("energy_eV")
    if energy_eV is not None:
        _upsert_property(
            run=run,
            name="final_single_point_energy_eV",
            unit="eV",
            value_float=float(energy_eV),
            source_artifact=log_artifact,
            metadata={"source": "remote_orca_parse"},
        )

    charge_rows = parsed.get("mulliken_charges")
    if isinstance(charge_rows, list) and charge_rows:
        charges = [float(row["charge"]) for row in charge_rows if "charge" in row]
        symbols = [str(row.get("symbol", "")) for row in charge_rows]
        _store_charges_payload(
            run=run,
            charges=charges,
            symbols=symbols,
            source_artifact=log_artifact,
            charge_method="mulliken",
        )

    force_rows = parsed.get("forces_eV_A")
    if isinstance(force_rows, list) and force_rows:
        vectors = [list(map(float, row["force"])) for row in force_rows if "force" in row]
        symbols = [str(row.get("symbol", "")) for row in force_rows]
        _store_forces_payload(
            run=run,
            vectors=vectors,
            symbols=symbols,
            source_artifact=log_artifact,
        )

    return _return_with_latest_evaluated_traj_fmax(True)


def _resolve_workflow_definition_for_ingest(*, workflow_name: str, code_path: str, description: str) -> WorkflowDefinition:
    normalized_code_path = str(code_path or "").strip()
    if normalized_code_path:
        existing_by_code_path = (
            WorkflowDefinition.objects.filter(code_path=normalized_code_path).order_by("name", "pk").first()
        )
        if existing_by_code_path is not None:
            if description and not str(existing_by_code_path.description or "").strip():
                existing_by_code_path.description = description
                existing_by_code_path.save(update_fields=["description"])
            return existing_by_code_path

    workflow, _ = WorkflowDefinition.objects.get_or_create(
        name=workflow_name,
        defaults={
            "code_path": normalized_code_path,
            "description": description,
        },
    )
    if normalized_code_path and not str(workflow.code_path or "").strip():
        workflow.code_path = normalized_code_path
        if description and not str(workflow.description or "").strip():
            workflow.description = description
            workflow.save(update_fields=["code_path", "description"])
        else:
            workflow.save(update_fields=["code_path"])
    return workflow


@transaction.atomic
def ingest_single_run(run_dir: Path, workflow_name: str = "cplas_tddft_active_learning") -> WorkflowRun:
    run_dir_text = canonical_run_path(run_dir)
    run_dir = Path(run_dir_text)
    workflow = _resolve_workflow_definition_for_ingest(
        workflow_name=workflow_name,
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
        description="TDDFT active learning workflow imported from cplas outputs.",
    )
    run_status = _infer_run_status(run_dir)
    profile = _default_settings_profile()
    run_uid = deterministic_run_uuid(run_dir_text)

    run = WorkflowRun.objects.filter(run_uid=run_uid).first()
    if run is None:
        run = WorkflowRun.objects.filter(working_directory=run_dir_text).order_by("-updated_at", "-created_at").first()

    created = run is None
    if created:
        run = WorkflowRun.objects.create(
            workflow=workflow,
            run_uid=run_uid,
            run_name=run_dir.name,
            status=run_status,
            working_directory=run_dir_text,
            settings_profile=profile,
            started_at=timezone.now() if run_status != WorkflowStatus.INITIATED else None,
            completed_at=timezone.now() if run_status == WorkflowStatus.COMPLETED else None,
            metadata=_merge_ingest_metadata(None, run_dir_text),
        )
    else:
        run.run_name = run_dir.name
        if run.status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED} and run_status == WorkflowStatus.INITIATED:
            pass
        else:
            run.status = run_status
        run.working_directory = run_dir_text
        if run.settings_profile_id is None:
            run.settings_profile = profile
        if run.started_at is None and run_status != WorkflowStatus.INITIATED:
            run.started_at = timezone.now()
        if run.completed_at is None and run_status == WorkflowStatus.COMPLETED:
            run.completed_at = timezone.now()
        run.metadata = _merge_ingest_metadata(run.metadata, run_dir_text)
        if run.run_uid != run_uid and not WorkflowRun.objects.exclude(pk=run.pk).filter(run_uid=run_uid).exists():
            run.run_uid = run_uid
        run.save()

    _ingest_local_artifacts_for_run(run, run_dir)
    try:
        upsert_adsorption_energy_for_run(run)
    except Exception:
        pass
    return run


@transaction.atomic
def ingest_workflow_outputs(workflow_root: Path | None = None) -> dict:
    root = Path(workflow_root or settings.REACTION_HUB_INGEST_ROOT).resolve()
    candidates = discover_cplas_tddft_runs(root)
    ingested = 0
    for candidate in candidates:
        ingest_single_run(candidate.run_dir, workflow_name=candidate.workflow_name)
        ingested += 1
    return {"root": str(root), "discovered": len(candidates), "ingested": ingested}


def _normalize_scheduler_state(state: str) -> str:
    s = (state or "").strip().upper()
    if not s:
        return JobStatus.UNKNOWN

    primary = s.split()[0]

    if primary in {"PD", "PENDING", "CONFIGURING"}:
        return JobStatus.PENDING
    if primary in {"R", "RUNNING", "COMPLETING"}:
        return JobStatus.RUNNING
    if primary in {"CD", "COMPLETED"}:
        return JobStatus.COMPLETED
    if primary in {"F", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}:
        return JobStatus.FAILED
    if primary in {"CA", "CANCELLED", "CANCELLED+"} or primary.startswith("CANCELLED"):
        return JobStatus.CANCELLED
    if primary in {"S", "SUSPENDED"}:
        return JobStatus.PENDING
    return JobStatus.UNKNOWN


def _workflow_status_from_job_status(job_status: str) -> str | None:
    if job_status in {JobStatus.INITIATED, JobStatus.PENDING}:
        return WorkflowStatus.INITIATED
    if job_status == JobStatus.RUNNING:
        return WorkflowStatus.RUNNING
    if job_status == JobStatus.COMPLETED:
        return WorkflowStatus.COMPLETED
    if job_status in {JobStatus.FAILED, JobStatus.CANCELLED}:
        return WorkflowStatus.FAILED
    if job_status == JobStatus.DELETED:
        # Keep existing workflow run status as-is. Run data may still be valid
        # locally even if remote HPC directory has been removed.
        return None
    return None


def _run_has_result_summary(run: WorkflowRun) -> bool:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    return isinstance(metadata.get("result_summary"), dict)


def _run_has_key_plot_artifacts(run: WorkflowRun) -> bool:
    return Artifact.objects.filter(
        workflow_run=run,
        kind=ArtifactKind.IMAGE,
        label__in=set(_KEY_PLOT_FILENAMES.values()),
    ).exists()


def _run_has_computed_properties(run: WorkflowRun) -> bool:
    return ComputedProperty.objects.filter(workflow_run=run).exists()


def _job_marked_failed_for_missing_properties(job: JobRecord) -> bool:
    reason_text = str(job.failure_reason or "").strip().lower()
    if "no computed properties were ingested" in reason_text:
        return True
    latest_failed_event = (
        job.events.filter(status=JobStatus.FAILED)
        .order_by("-created_at", "-id")
        .first()
    )
    if latest_failed_event is None:
        return False
    payload = latest_failed_event.payload if isinstance(latest_failed_event.payload, dict) else {}
    return str(payload.get("rule") or "").strip() == "completed_requires_computed_properties"


def _mark_completed_job_failed_if_no_properties(job: JobRecord) -> bool:
    if job.status != JobStatus.COMPLETED:
        return False
    if not bool(getattr(settings, "REACTION_HUB_SYNC_STRICT_COMPLETED_REQUIRES_PROPERTIES", False)):
        return False
    if _run_has_computed_properties(job.workflow_run):
        return False

    reason = (
        "Scheduler reported completed, but no computed properties were ingested; "
        "treating job as failed."
    )
    status_changed = job.transition_to(
        JobStatus.FAILED,
        message="Scheduler completed but run has no computed properties; marked failed.",
        payload={
            "detected_by": "sync_hpc_jobs",
            "rule": "completed_requires_computed_properties",
        },
    )
    reason_changed = False
    if job.failure_reason != reason:
        job.failure_reason = reason
        job.save(update_fields=["failure_reason", "updated_at"])
        reason_changed = True

    run_changed = _sync_workflow_run_from_job(job)
    return status_changed or reason_changed or run_changed


def _recover_failed_completed_job_if_properties_available(job: JobRecord) -> bool:
    if job.status != JobStatus.FAILED:
        return False
    if not _job_marked_failed_for_missing_properties(job):
        return False
    if not _run_has_computed_properties(job.workflow_run):
        return False

    status_changed = job.transition_to(
        JobStatus.COMPLETED,
        message="Recovered completed job after computed properties became available.",
        payload={
            "detected_by": "sync_hpc_jobs",
            "rule": "recover_completed_from_properties",
        },
    )
    reason_changed = False
    if job.failure_reason:
        job.failure_reason = ""
        job.save(update_fields=["failure_reason", "updated_at"])
        reason_changed = True

    run_changed = _sync_workflow_run_from_job(job)
    return status_changed or reason_changed or run_changed


def _job_remote_run_root(job: JobRecord) -> str | None:
    run_dir_text = canonical_run_path(job.workflow_run.working_directory)
    if not run_dir_text:
        return None
    # Remote checks only make sense for POSIX-like absolute HPC paths.
    if not run_dir_text.startswith("/"):
        return None
    return run_dir_text


def _normalize_sync_lineage_root(path_text: str) -> str:
    normalized = canonical_run_path(path_text)
    if not normalized.startswith("/"):
        return ""
    parts = [part for part in normalized.split("/") if part]
    # Ignore scheduler-id/restart suffixes so a restart chain maps to one lineage.
    while parts and (
        parts[-1].startswith("restart_backup_")
        or (parts[-1].isdigit() and len(parts[-1]) >= 4)
    ):
        parts.pop()
    return f"/{'/'.join(parts)}" if parts else normalized


def _sync_tddft_retry_family_key(job: JobRecord) -> str:
    if _run_mode_for_ingest(job.workflow_run) != "tddft_bond_scan":
        return ""

    workflow_id = int(job.workflow_run.workflow_id or 0)
    run_name = str(job.workflow_run.run_name or "").strip().casefold()
    if not run_name:
        return ""

    tokens = [token for token in run_name.split("_") if token]
    if len(tokens) >= 3 and tokens[-1].isdigit():
        try:
            retry_index = int(tokens[-1])
        except (TypeError, ValueError):
            retry_index = -1
        if 0 <= retry_index <= 99:
            try:
                float(tokens[-2])
            except (TypeError, ValueError):
                pass
            else:
                run_name = "_".join(tokens[:-1]).strip() or run_name

    metadata = job.workflow_run.metadata if isinstance(job.workflow_run.metadata, dict) else {}
    submission = metadata.get("submission")
    if not isinstance(submission, dict):
        submission = {}
    target_text = ""
    try:
        target_text = f"{float(submission.get('target_wavelength_nm')):.6f}"
    except (TypeError, ValueError):
        target_text = ""

    if target_text:
        return f"{workflow_id}|tddft:{run_name}|target:{target_text}"
    return f"{workflow_id}|tddft:{run_name}"


def _sync_terminal_lineage_key(job: JobRecord) -> str:
    workflow_id = int(job.workflow_run.workflow_id or 0)
    run_name = str(job.workflow_run.run_name or "").strip().casefold()
    tddft_family = _sync_tddft_retry_family_key(job)
    if tddft_family:
        return tddft_family
    lineage_root = _normalize_sync_lineage_root(str(job.workflow_run.working_directory or ""))
    if lineage_root:
        return f"{workflow_id}|root:{lineage_root}"
    if run_name:
        return f"{workflow_id}|name:{run_name}"
    return f"{workflow_id}|run:{int(job.workflow_run_id)}"


def _sync_lineage_keys(job: JobRecord) -> set[str]:
    workflow_id = int(job.workflow_run.workflow_id or 0)
    run_name = str(job.workflow_run.run_name or "").strip().casefold()
    lineage_root = _normalize_sync_lineage_root(str(job.workflow_run.working_directory or ""))

    keys = {_sync_terminal_lineage_key(job)}
    if lineage_root:
        keys.add(f"{workflow_id}|root:{lineage_root}")
    if run_name:
        keys.add(f"{workflow_id}|name:{run_name}")
    return {key for key in keys if key}


def _sync_job_recency_score(job: JobRecord) -> tuple[float, float, int, int]:
    def _ts(value) -> float:
        if value is None:
            return 0.0
        try:
            return float(value.timestamp())
        except Exception:
            return 0.0

    return (
        _ts(job.last_seen_at or job.completed_at or job.started_at or job.submitted_at),
        _ts(job.updated_at or job.created_at),
        int(job.workflow_run_id or 0),
        int(job.pk or 0),
    )


def _query_remote_run_root_exists(client: object, jobs: list[JobRecord]) -> dict[str, bool]:
    run_roots = sorted({root for job in jobs if (root := _job_remote_run_root(job))})
    if not run_roots:
        return {}
    if hasattr(client, "directories_exist"):
        try:
            raw = client.directories_exist(run_roots)
        except Exception:
            raw = {}
        return {
            str(path): bool(raw.get(path))
            for path in run_roots
            if str(path) in raw
        }
    if hasattr(client, "directory_exists"):
        results: dict[str, bool] = {}
        for path in run_roots:
            try:
                results[path] = bool(client.directory_exists(path))
            except Exception:
                continue
        return results
    return {}


def _sync_workflow_run_from_job(job: JobRecord) -> bool:
    run = job.workflow_run
    update_fields: list[str] = []
    target_status = _workflow_status_from_job_status(job.status)

    if target_status and run.status != target_status:
        run.status = target_status
        update_fields.append("status")

    if job.started_at is not None and run.started_at is None:
        run.started_at = job.started_at
        update_fields.append("started_at")
    if job.completed_at is not None and run.completed_at is None:
        run.completed_at = job.completed_at
        update_fields.append("completed_at")

    if run.status == WorkflowStatus.RUNNING and run.started_at is None:
        run.started_at = timezone.now()
        update_fields.append("started_at")
    if run.status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED} and run.completed_at is None:
        run.completed_at = timezone.now()
        update_fields.append("completed_at")
    if run.status in {WorkflowStatus.COMPLETED, WorkflowStatus.FAILED} and run.started_at is None:
        run.started_at = job.started_at or job.submitted_at or run.completed_at or timezone.now()
        update_fields.append("started_at")

    if not update_fields:
        return False

    run.save(update_fields=[*update_fields, "updated_at"])
    return True


def _is_hpc_login_host_reachable(host: str, port: int, timeout_seconds: float) -> bool:
    host_text = str(host or "").strip()
    if not host_text:
        return False
    try:
        port_value = int(port)
    except (TypeError, ValueError):
        port_value = 22
    if port_value <= 0:
        port_value = 22
    try:
        timeout_value = max(float(timeout_seconds), 0.1)
    except (TypeError, ValueError):
        timeout_value = 1.5

    try:
        with socket.create_connection((host_text, int(port_value)), timeout=timeout_value):
            return True
    except Exception:
        return False


def _slurm_runtime_seconds(value: object) -> int | None:
    try:
        day_text, separator, clock_text = str(value or "").strip().partition("-")
        parts = [int(part) for part in (clock_text if separator else day_text).split(":")]
        days = int(day_text) if separator else 0
    except (TypeError, ValueError):
        return None
    if len(parts) == 2:
        hours = 0
        minutes, seconds = parts
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    else:
        return None
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def sync_hpc_jobs(
    *,
    client_factory: Callable[[HPCSystem], object] | None = None,
    force_reingestion: bool = False,
    retry_failed_ingestion: bool = False,
    debug: bool = False,
) -> dict:
    debug_enabled = bool(debug)
    retry_failed_enabled = bool(retry_failed_ingestion)

    def _debug(message: str) -> None:
        logger.info(f"[sync_hpc_jobs] {message}")
        if debug_enabled:
            print(f"[sync_hpc_jobs] {message}")

    factory = client_factory or (
        lambda system: SlurmSSHClient(
            host=system.host,
            port=system.port,
            username=system.username,
            ssh_key_path=system.ssh_key_path,
        )
    )

    systems = list(HPCSystem.objects.filter(is_active=True))
    checked = 0
    updated = 0
    ct_plot_backfills_queued = 0
    skipped_unreachable = 0
    preflight_enabled = bool(getattr(settings, "REACTION_HUB_SYNC_PREFLIGHT_REACHABILITY", True))
    preflight_timeout = float(
        getattr(settings, "REACTION_HUB_SYNC_PREFLIGHT_TIMEOUT_SECONDS", 1.5)
    )
    # Test and integration callers often provide a custom client factory. Keep
    # existing behaviour unless they explicitly exercise the default SSH path.
    if client_factory is not None:
        preflight_enabled = False

    _debug(
        "Starting sync across "
        f"{len(systems)} active systems; force_reingestion={bool(force_reingestion)}, "
        f"retry_failed_ingestion={bool(retry_failed_enabled)}, "
        f"preflight={bool(preflight_enabled)} timeout={float(preflight_timeout):.2f}s."
    )

    for system in systems:
        jobs_all = list(
            JobRecord.objects.filter(hpc_system=system)
            .exclude(status=JobStatus.DELETED)
            .select_related("workflow_run")
        )
        if not jobs_all:
            continue

        # Keep only the latest scheduler job record per workflow run. Older
        # restart/resubmit records are historical and should not be re-polled or
        # re-ingested during routine sync.
        latest_jobs_by_run: dict[int, JobRecord] = {}
        for job in jobs_all:
            run_id = int(job.workflow_run_id)
            current = latest_jobs_by_run.get(run_id)
            if current is None or int(job.pk or 0) > int(current.pk or 0):
                latest_jobs_by_run[run_id] = job
        jobs = list(latest_jobs_by_run.values())

        active_jobs = [
            job
            for job in jobs
            if job.status
            not in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.DELETED}
        ]
        active_lineage_keys: set[str] = set()
        for job in active_jobs:
            active_lineage_keys.update(_sync_lineage_keys(job))
        terminal_jobs = [
            job
            for job in jobs
            if job.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}
            and _sync_lineage_keys(job).isdisjoint(active_lineage_keys)
        ]
        # Keep only the latest terminal job in each logical lineage. This
        # avoids repeatedly reprocessing stale predecessor runs after restarts
        # or resubmissions that reuse the same run-root/run-name lineage.
        latest_terminal_by_lineage: dict[str, JobRecord] = {}
        for job in terminal_jobs:
            lineage_key = _sync_terminal_lineage_key(job)
            current = latest_terminal_by_lineage.get(lineage_key)
            if current is None or _sync_job_recency_score(job) > _sync_job_recency_score(current):
                latest_terminal_by_lineage[lineage_key] = job
        terminal_jobs = list(latest_terminal_by_lineage.values())
        terminal_job_ids = {int(job.pk) for job in terminal_jobs}

        if force_reingestion:
            recovery_jobs = [job for job in terminal_jobs if job.status == JobStatus.COMPLETED]
        else:
            recovery_jobs = []
        failed_ingest_jobs: list[JobRecord] = []
        failed_ingest_jobs = [
            job
            for job in terminal_jobs
            if job.status == JobStatus.FAILED
            and retry_failed_enabled
        ]
        # Running/queued jobs should be status-polled only. Remote filesystem
        # checks are limited to terminal states to avoid retrieving data from
        # in-flight runs.
        deletion_watch_jobs = [
            job
            for job in terminal_jobs
            if job.status in {JobStatus.FAILED, JobStatus.CANCELLED}
            and _job_remote_run_root(job) is not None
        ]
        _debug(
            f"System {system.name}: total_jobs={len(jobs_all)} latest_jobs={len(jobs)} "
            f"terminal_jobs={len(terminal_jobs)} terminal_skipped={len(jobs) - len(active_jobs) - len(terminal_job_ids)} "
            f"active={len(active_jobs)} "
            f"recovery={len(recovery_jobs)} failed_ingest={len(failed_ingest_jobs)} "
            f"deletion_watch={len(deletion_watch_jobs)}."
        )
        if not active_jobs and not recovery_jobs and not failed_ingest_jobs and not deletion_watch_jobs:
            continue

        if preflight_enabled and not _is_hpc_login_host_reachable(
            str(system.host or "").strip(),
            system.port or 22,
            preflight_timeout,
        ):
            skipped_unreachable += 1
            _debug(
                f"System {system.name}: skipped after reachability precheck failed "
                f"for {system.host}:{system.port}."
            )
            continue

        try:
            with factory(system) as client:
                snapshots: dict[str, dict] = {}
                if active_jobs:
                    job_ids = [job.scheduler_job_id for job in active_jobs]
                    _debug(f"System {system.name}: polling scheduler states for {len(job_ids)} active jobs.")
                    snapshots = client.query_jobs(job_ids)
                    _debug(
                        f"System {system.name}: received {len(snapshots)} scheduler snapshots."
                    )
                remote_root_exists = _query_remote_run_root_exists(client, deletion_watch_jobs)
                if deletion_watch_jobs:
                    _debug(
                        f"System {system.name}: checked remote run-root existence for "
                        f"{len(remote_root_exists)} paths."
                    )
                ingested_job_ids: set[int] = set()

                def _ingest_outputs_for_job(job: JobRecord) -> bool:
                    run_path = Path(job.workflow_run.working_directory)
                    if run_path.exists():
                        try:
                            ingest_single_run(run_path)
                            return True
                        except Exception:
                            return False
                    try:
                        return bool(
                            _ingest_remote_summary_for_job(
                                job,
                                client,
                                download_missing_only=force_reingestion,
                            )
                        )
                    except Exception:
                        return False

                for job in active_jobs:
                    checked += 1
                    snapshot = snapshots.get(job.scheduler_job_id)
                    if snapshot is None:
                        _debug(
                            f"System {system.name}: no scheduler snapshot for job "
                            f"{job.scheduler_job_id} (db status={job.status})."
                        )
                        continue
                    new_status = _normalize_scheduler_state(snapshot.get("state", ""))
                    _debug(
                        f"System {system.name}: job {job.scheduler_job_id} "
                        f"{job.status} -> {new_status}."
                    )
                    status_changed = job.transition_to(
                        new_status,
                        message=f"Scheduler update: {snapshot.get('state', '')}",
                        payload=snapshot,
                    )
                    run_changed = _sync_workflow_run_from_job(job)
                    if status_changed or run_changed:
                        updated += 1

                    if new_status == JobStatus.FAILED:
                        runtime_seconds = _slurm_runtime_seconds(snapshot.get("runtime"))
                        if runtime_seconds is not None and runtime_seconds < 3 * 60:
                            _debug(
                                f"System {system.name}: skipping artifact recovery for failed job "
                                f"{job.scheduler_job_id}; Slurm runtime was {snapshot.get('runtime')}."
                            )
                            continue
                        _debug(
                            f"System {system.name}: ingesting failed job "
                            f"{job.scheduler_job_id}."
                        )
                        ingested = _ingest_outputs_for_job(job)
                        if ingested:
                            try:
                                upsert_adsorption_energy_for_run(job.workflow_run)
                            except Exception:
                                pass
                            updated += 1
                        continue

                    if new_status != JobStatus.COMPLETED:
                        continue

                    ingested_job_ids.add(job.pk)
                    _debug(
                        f"System {system.name}: ingesting completed job "
                        f"{job.scheduler_job_id}."
                    )
                    ingested = _ingest_outputs_for_job(job)
                    if ingested:
                        try:
                            upsert_adsorption_energy_for_run(job.workflow_run)
                        except Exception:
                            pass
                        if (
                            _run_mode_for_ingest(job.workflow_run) == "charge_transfer"
                            and not _is_ground_state_mulliken_only_run(job.workflow_run)
                            and not _is_density_analysis_only_run(job.workflow_run)
                        ):
                            try:
                                from .tasks import backfill_ct_plots_task

                                backfill_ct_plots_task.delay(job.workflow_run_id)
                                ct_plot_backfills_queued += 1
                            except Exception as exc:
                                _debug(
                                    f"Could not queue CT plot backfill for run "
                                    f"{job.workflow_run_id}: {exc}"
                                )
                    if _mark_completed_job_failed_if_no_properties(job):
                        updated += 1

                # Re-ingest already-completed jobs only when explicitly requested.
                # Routine sync ingests active jobs when they transition to completed,
                # but does not revisit terminal completed jobs.
                for job in recovery_jobs:
                    if job.pk in ingested_job_ids:
                        continue
                    _debug(
                        f"System {system.name}: retrying ingestion for completed job "
                        f"{job.scheduler_job_id}."
                    )
                    ingested = _ingest_outputs_for_job(job)
                    if not ingested:
                        if _mark_completed_job_failed_if_no_properties(job):
                            updated += 1
                        continue
                    updated += 1
                    try:
                        upsert_adsorption_energy_for_run(job.workflow_run)
                    except Exception:
                        pass
                    if _mark_completed_job_failed_if_no_properties(job):
                        updated += 1

                for job in failed_ingest_jobs:
                    _debug(
                        f"System {system.name}: retrying ingestion for failed job "
                        f"{job.scheduler_job_id}."
                    )
                    ingested = _ingest_outputs_for_job(job)
                    if ingested:
                        try:
                            upsert_adsorption_energy_for_run(job.workflow_run)
                        except Exception:
                            pass
                        updated += 1

                    recovered = False
                    if _job_marked_failed_for_missing_properties(job):
                        recovered = _recover_failed_completed_job_if_properties_available(job)
                    if recovered:
                        updated += 1

                for job in deletion_watch_jobs:
                    remote_root = _job_remote_run_root(job)
                    if not remote_root:
                        continue
                    exists = remote_root_exists.get(remote_root)
                    if exists is not False:
                        continue
                    reason = f"Remote run directory missing on HPC: {remote_root}"
                    _debug(
                        f"System {system.name}: marking job {job.scheduler_job_id} deleted; "
                        f"missing remote root {remote_root}."
                    )
                    status_changed = job.transition_to(
                        JobStatus.DELETED,
                        message="Remote run directory missing on HPC; marked deleted.",
                        payload={
                            "remote_run_root": remote_root,
                            "detected_by": "sync_hpc_jobs",
                        },
                    )
                    reason_changed = False
                    if job.failure_reason != reason:
                        job.failure_reason = reason
                        job.save(update_fields=["failure_reason", "updated_at"])
                        reason_changed = True
                    run_changed = _sync_workflow_run_from_job(job)
                    if status_changed or reason_changed or run_changed:
                        updated += 1
        except Exception as exc:
            logger.exception(f"[sync_hpc_jobs] System {system.name}: sync failed with error: {exc}")
            _debug(f"System {system.name}: sync failed with error: {exc}")
            continue

    _debug(f"Sync complete: checked={checked}, updated={updated}, skipped_unreachable={skipped_unreachable}.")

    return {
        "systems": len(systems),
        "checked": checked,
        "updated": updated,
        "ct_plot_backfills_queued": ct_plot_backfills_queued,
        "skipped_unreachable": skipped_unreachable,
        "reachability_precheck": bool(preflight_enabled),
        "reachability_timeout_seconds": float(preflight_timeout),
        "force_reingestion": bool(force_reingestion),
        "retry_failed_ingestion": bool(retry_failed_enabled),
        "debug": bool(debug_enabled),
    }
