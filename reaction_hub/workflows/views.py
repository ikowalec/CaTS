from __future__ import annotations

import json
import shutil
import shlex
import subprocess
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlencode

from ase import Atoms
from ase.db import connect as ase_connect
from ase.io import read as ase_read
from ase.io import write as ase_write
from django.conf import settings
from django.db.models import Q
from django.http import JsonResponse
from django.urls import reverse
from django.shortcuts import get_object_or_404
from django.views.decorators.http import require_POST
from django.views.generic import DetailView
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin

from core.table_pagination import FriendlyTablePaginationMixin
from core.xyzrender import (
    build_xyzrender_subprocess_env,
    cairo_runtime_install_hint,
    is_xyzrender_cairo_error,
    normalize_gif_canvas,
    parse_canvas_size,
    resolve_xyzrender_command,
)
from jobs.models import JobRecord, JobStatus
from results.models import AdsorptionEnergyResult, Artifact, ComputedProperty
from results.services import _metadata_interaction_energy_eint_eV

from .filters import WorkflowRunFilter
from .identity import canonical_run_path, normalize_run_path
from .models import WorkflowRun, WorkflowStatus
from .tasks import ingest_artifact_from_hpc_task, render_artifact_xyz_task
from .tables import WorkflowRunTable

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
_STRUCTURE_EXTENSIONS = {".xyz", ".traj", ".cif", ".pdb"}
_DATABASE_EXTENSIONS = {".db"}
_KEY_PLOT_FILES = {
    "stacked_uvvis.png",
    "stacked_ct.png",
    "energy_vs_reaction_coordinate.png",
    "final_tddft_spectrum.png",
    "mep_pingpong.gif",
}
_SCAN_ENDPOINTS_DB_FILENAME = "scan_endpoints.db"
# Keep "xy" accepted for backward compatibility with existing API callers.
_XYZRENDER_ALLOWED_GIF_ROTATIONS = {"x", "y", "z", "xyz", "xy", "xz", "yz"}
_SCAN_PROPERTY_UNITS = {
    "bond_scan_start_energy_eV": "eV",
    "bond_scan_peak_energy_eV": "eV",
    "bond_scan_activation_barrier_eV": "eV",
    "target_wavelength_nm": "nm",
    "final_structure_formula": "",
    "final_structure_sha256": "",
}


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _ensure_dense_sparse_shim() -> None:
    try:
        import sparse  # noqa: F401
    except ImportError as exc:
        message = str(exc)
        if "Numba needs NumPy" not in message:
            raise
        fake_sparse = types.ModuleType("sparse")

        class COO:
            @classmethod
            def from_numpy(cls, value):
                return value

            def tocsr(self):
                return self

        fake_sparse.COO = COO
        fake_sparse.concatenate = lambda values, axis=0: __import__("numpy").concatenate(values, axis=axis)
        sys.modules["sparse"] = fake_sparse


def _run_snapshot_atoms(run: WorkflowRun) -> Atoms:
    prop = (
        run.computed_properties.filter(name="final_structure_snapshot")
        .order_by("-updated_at", "-created_at", "-id")
        .first()
    )
    metadata = prop.metadata if prop and isinstance(prop.metadata, dict) else {}
    symbols = metadata.get("symbols")
    positions = metadata.get("positions_A")
    if symbols and positions:
        return Atoms(symbols=symbols, positions=positions)
    for cand_art in run.artifacts.filter(kind__in=["structure", "trajectory"]):
        lp = _artifact_local_path(run, cand_art)
        if lp.exists():
            try:
                from ase.io import read
                imgs = read(str(lp), index=":", format="extxyz" if str(lp).endswith(".extxyz") else None)
                if imgs:
                    return imgs[-1]
            except Exception:
                continue
    raise ValueError(f"Run {run.pk} has no final structure artifact or snapshot metadata.")


def _rematch_global_similarity(local_kernel, *, alpha: float = 1.0, threshold: float = 1.0e-6) -> float:
    import numpy as np

    n, m = local_kernel.shape
    kernel = np.exp(-(1 - local_kernel) / alpha)
    u = np.ones(n) / n
    v = np.ones(m) / m
    en = np.ones(n) / float(n)
    em = np.ones(m) / float(m)
    error = 1.0
    iteration = 0
    while error > threshold:
        previous_u = u
        previous_v = v
        v = np.divide(em, np.dot(kernel.T, u))
        u = np.divide(en, np.dot(kernel, v))
        if iteration % 5:
            error = np.sum((u - previous_u) ** 2) / np.sum(u**2) + np.sum((v - previous_v) ** 2) / np.sum(v**2)
        iteration += 1
    transport = np.multiply(np.multiply(kernel, u.reshape((-1, 1))), v)
    return float(np.sum(np.multiply(transport, local_kernel)))


def _compare_run_structure_similarity(run: WorkflowRun, other_run: WorkflowRun) -> dict[str, object]:
    _ensure_dense_sparse_shim()
    from dscribe.descriptors import SOAP
    import numpy as np
    from sklearn.metrics.pairwise import pairwise_kernels
    from sklearn.preprocessing import normalize

    atoms = [_run_snapshot_atoms(run), _run_snapshot_atoms(other_run)]
    species = sorted({symbol for item in atoms for symbol in item.get_chemical_symbols()})
    soap = SOAP(species=species, periodic=any(any(item.get_pbc()) for item in atoms), r_cut=6.0, n_max=8, l_max=6)
    features = [normalize(soap.create(item)) for item in atoms]
    raw = np.array(
        [
            [
                _rematch_global_similarity(
                    pairwise_kernels(
                        features[i],
                        features[j],
                        metric="rbf",
                        filter_params=True,
                        gamma=1.0,
                        degree=3,
                        coef0=1,
                    )
                )
                for j in range(2)
            ]
            for i in range(2)
        ]
    )
    normalized = raw / np.outer(np.sqrt(np.diag(raw)), np.sqrt(np.diag(raw)))
    return {
        "other_run": other_run,
        "score": float(normalized[0, 1]),
        "raw_score": float(raw[0, 1]),
        "self_score": float(raw[0, 0]),
        "other_self_score": float(raw[1, 1]),
        "formula": _latest_property_text(run, "final_structure_formula"),
        "other_formula": _latest_property_text(other_run, "final_structure_formula"),
    }


def _latest_property_text(run: WorkflowRun, name: str) -> str:
    prop = (
        run.computed_properties.filter(name=name)
        .order_by("-updated_at", "-created_at", "-id")
        .first()
    )
    return str(prop.value_text or "").strip() if prop else ""


def _is_image_artifact(artifact: Artifact) -> bool:
    if artifact.kind == "image":
        return True
    return Path(str(artifact.path or "")).suffix.lower() in _IMAGE_EXTENSIONS


def _is_remote_only_artifact(artifact: Artifact) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    return bool(metadata.get("remote_only"))


def _is_remote_sourced_artifact(artifact: Artifact) -> bool:
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


def _inferred_local_cache_path(run: WorkflowRun, artifact: Artifact) -> Path:
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


def _artifact_content_available(run: WorkflowRun, artifact: Artifact) -> bool:
    if _is_remote_only_artifact(artifact):
        return False
    path = _artifact_local_path(run, artifact)
    return path.exists() and path.is_file()


def _artifact_local_path(run: WorkflowRun, artifact: Artifact) -> Path:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    remote_sourced = _is_remote_sourced_artifact(artifact)
    local_cache = str(metadata.get("local_cache_path") or "").strip()
    if local_cache:
        local_cache_path = _runtime_path(local_cache)
        if local_cache_path.exists() and local_cache_path.is_file():
            return local_cache_path
        if remote_sourced:
            return local_cache_path
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
        inferred_cache = _inferred_local_cache_path(run, artifact)
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


def _is_structure_artifact(artifact: Artifact) -> bool:
    if artifact.kind in {"structure", "trajectory"}:
        return True
    suffix = Path(str(artifact.path or "")).suffix.lower()
    return suffix in _STRUCTURE_EXTENSIONS


def _is_database_artifact(artifact: Artifact) -> bool:
    if artifact.kind == "database":
        return True
    suffix = Path(str(artifact.path or "")).suffix.lower()
    return suffix in _DATABASE_EXTENSIONS


def _is_ase_gui_supported_artifact(artifact: Artifact) -> bool:
    return (
        _is_structure_artifact(artifact)
        or _is_trajectory_artifact(artifact)
        or _is_database_artifact(artifact)
    )


def resolve_ase_gui_command(target_path: Path | str | None = None) -> list[str]:
    command_text = getattr(settings, "REACTION_HUB_ASE_GUI_COMMAND", "python -m ase gui").strip()
    command = shlex.split(command_text)
    if not command:
        raise RuntimeError("REACTION_HUB_ASE_GUI_COMMAND is empty.")

    launcher = command[0].strip().lower()
    if launcher in {"python", "python3", "python.exe", "python3.exe", "py", "py.exe"}:
        command[0] = sys.executable
    elif launcher in {"ase", "ase.exe"}:
        venv_ase = Path(sys.executable).parent / ("ase.exe" if sys.platform == "win32" else "ase")
        if venv_ase.exists():
            command[0] = str(venv_ase)

    if target_path is not None:
        path_text = str(Path(target_path).resolve())
        if "{file}" in command:
            command = [path_text if token == "{file}" else token for token in command]
        else:
            command.append(path_text)
    return command


def _is_scan_endpoints_db_artifact(artifact: Artifact) -> bool:
    return Path(str(artifact.path or "")).name.lower() == _SCAN_ENDPOINTS_DB_FILENAME


def _is_trajectory_artifact(artifact: Artifact) -> bool:
    if _is_scan_endpoints_db_artifact(artifact):
        return False
    if artifact.kind == "trajectory":
        return True
    return Path(str(artifact.path or "")).suffix.lower() == ".traj"


def _is_batch_submission_source_artifact(run: WorkflowRun, artifact: Artifact) -> bool:
    if not _artifact_content_available(run, artifact):
        return False
    suffix = Path(str(artifact.path or "")).suffix.lower()
    if suffix in {".traj", ".db"}:
        return True
    return artifact.kind in {"trajectory", "database"}


def _is_xyzrender_renderable_artifact(artifact: Artifact) -> bool:
    return _is_structure_artifact(artifact) or _is_scan_endpoints_db_artifact(artifact)


def _ase_gui_openable(run: WorkflowRun, artifact: Artifact) -> bool:
    if not _is_ase_gui_supported_artifact(artifact):
        return False
    if _is_remote_only_artifact(artifact):
        return False
    return _artifact_content_available(run, artifact)


def _xyzrender_actionable(run: WorkflowRun, artifact: Artifact) -> bool:
    if not _is_xyzrender_renderable_artifact(artifact):
        return False
    if _is_remote_only_artifact(artifact):
        return False
    return _artifact_content_available(run, artifact)


def _is_key_plot_artifact(artifact: Artifact) -> bool:
    name = Path(str(artifact.path or "")).name.lower()
    if name in _KEY_PLOT_FILES:
        return True
    return str(artifact.label or "").startswith("plot:")


def _is_render_image_artifact(artifact: Artifact) -> bool:
    return str(artifact.label or "").startswith("render:")


def _xyzrender_gif_canvas_size() -> tuple[int, int] | None:
    width = _parse_optional_int(getattr(settings, "REACTION_HUB_XYZRENDER_GIF_WIDTH", None))
    height = _parse_optional_int(getattr(settings, "REACTION_HUB_XYZRENDER_GIF_HEIGHT", None))
    if width is not None and height is not None and width > 0 and height > 0:
        return int(width), int(height)
    return parse_canvas_size(getattr(settings, "REACTION_HUB_XYZRENDER_GIF_SIZE", "900x900"))


def _xyzrender_fixed_scale_anchor_symbol() -> str:
    symbol = str(getattr(settings, "REACTION_HUB_XYZRENDER_FIXED_SCALE_ANCHOR_SYMBOL", "H") or "H").strip()
    if not symbol.isalpha() or len(symbol) > 2:
        return "H"
    return symbol[0].upper() + symbol[1:].lower()


def _fixed_scale_anchor_frames(frames: list[object]) -> list[object]:
    if len(frames) <= 1:
        return frames

    positions: list[tuple[float, float, float]] = []
    for atoms in frames:
        try:
            for position in atoms.get_positions():
                positions.append((float(position[0]), float(position[1]), float(position[2])))
        except Exception:
            return frames
    if not positions:
        return frames

    min_xyz = [min(position[axis] for position in positions) for axis in range(3)]
    max_xyz = [max(position[axis] for position in positions) for axis in range(3)]
    center = [(min_xyz[axis] + max_xyz[axis]) / 2.0 for axis in range(3)]
    max_span = max(max_xyz[axis] - min_xyz[axis] for axis in range(3))
    if max_span <= 0.0:
        max_span = 1.0
    padding = max(1.5, max_span * 0.30)
    half_extent = (max_span / 2.0) + padding
    anchor_positions = [
        (
            center[0] + sx * half_extent,
            center[1] + sy * half_extent,
            center[2] + sz * half_extent,
        )
        for sx in (-1.0, 1.0)
        for sy in (-1.0, 1.0)
        for sz in (-1.0, 1.0)
    ]
    anchor_symbol = _xyzrender_fixed_scale_anchor_symbol()
    anchor_atoms = Atoms([anchor_symbol] * len(anchor_positions), positions=anchor_positions)

    anchored_frames = []
    for atoms in frames:
        try:
            anchored = atoms.copy()
            anchored.extend(anchor_atoms)
        except Exception:
            return frames
        anchored_frames.append(anchored)
    return anchored_frames


_STATIC_SVG_ROTATIONS: dict[str, tuple[tuple[float, str], ...]] = {
    "x": ((90.0, "x"),),
    "y": ((90.0, "y"),),
    "z": ((90.0, "z"),),
    "xy": ((45.0, "x"), (45.0, "y")),
    "xz": ((45.0, "x"), (45.0, "z")),
    "yz": ((45.0, "y"), (45.0, "z")),
}


def _orient_frames_for_static_svg(frames: list[object], orientation: str) -> list[object]:
    rotations = _STATIC_SVG_ROTATIONS.get(str(orientation or "").strip().lower(), ())
    if not rotations:
        return frames

    oriented_frames = []
    for atoms in frames:
        try:
            oriented = atoms.copy()
            for angle, axis in rotations:
                oriented.rotate(angle, axis, center="COP")
            oriented_frames.append(oriented)
        except Exception:
            return frames
    return oriented_frames


def _artifact_stored_path(run: WorkflowRun, file_path: Path) -> str:
    resolved = file_path.resolve()
    run_root = Path(run.working_directory).resolve()
    try:
        return str(resolved.relative_to(run_root))
    except ValueError:
        return str(resolved)


def _prepare_xyzrender_input_xyz(
    source_path: Path,
    output_base: str,
    *,
    source_artifact: Artifact | None = None,
    use_full_trajectory: bool = False,
    scan_endpoint_mode: str = "trajectory",
    add_fixed_scale_anchors: bool = False,
    static_svg_orientation: str = "",
) -> Path:
    render_input = source_path.parent / f".reaction_hub_render_input_{output_base}.xyz"
    if source_artifact is not None and _is_scan_endpoints_db_artifact(source_artifact):
        records = _scan_endpoint_records_from_db(source_path)
        if not records:
            raise RuntimeError("No scan endpoint frames were found in the selected scan_endpoints.db file.")
        if scan_endpoint_mode == "transition_state":
            selected = _transition_state_record_from_scan_records(records)
            if selected is None:
                raise RuntimeError(
                    "No endpoint with an energy value was found in scan_endpoints.db; "
                    "cannot select a transition-state candidate."
                )
            _iteration, _energy, atoms = selected
            atoms.info["scan_render_mode"] = "transition_state"
            frames = [atoms]
        elif scan_endpoint_mode in {"first", "last"}:
            selected = records[0] if scan_endpoint_mode == "first" else records[-1]
            iteration, energy, atoms = selected
            atoms.info["scan_render_mode"] = scan_endpoint_mode
            atoms.info["scan_iteration"] = int(iteration)
            if energy is not None:
                atoms.info["scan_energy_eV"] = float(energy)
            frames = [atoms]
        else:
            frames = [atoms for _iteration, _energy, atoms in records]
            frames = _build_ping_pong_trajectory_frames(frames)
    else:
        try:
            if use_full_trajectory:
                atoms_obj = ase_read(str(source_path), index=":")
            else:
                # For regular structures/trajectories, render the converged geometry (last frame).
                atoms_obj = ase_read(str(source_path), index=-1)
        except Exception as exc:
            raise RuntimeError(f"Failed to read source artifact with ASE: {exc}") from exc

        if isinstance(atoms_obj, (list, tuple)):
            frames = list(atoms_obj)
        else:
            frames = [atoms_obj]
        if not frames:
            raise RuntimeError("No readable structure frames were found in the selected artifact.")
        if use_full_trajectory:
            frames = _build_ping_pong_trajectory_frames(frames)
        else:
            frames = [frames[-1]]
    if add_fixed_scale_anchors:
        frames = _fixed_scale_anchor_frames(frames)
    if static_svg_orientation:
        frames = _orient_frames_for_static_svg(frames, static_svg_orientation)
    try:
        ase_write(str(render_input), frames, format="xyz")
    except Exception as exc:
        raise RuntimeError(f"Failed to convert artifact to xyz for rendering: {exc}") from exc
    return render_input


def _summarize_xyz_input(path: Path, *, max_lines: int = 8) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        return f"could not read generated XYZ ({exc})"

    if not lines:
        return "generated XYZ is empty"

    atom_count_text = lines[0].strip() if lines else ""
    try:
        atom_count = int(atom_count_text)
    except ValueError:
        atom_count = None

    start = 2 if len(lines) > 2 else 0
    sample = lines[start : start + max_lines]
    coord_cols: list[int] = []
    malformed = 0
    for line in sample:
        parts = line.split()
        if not parts:
            continue
        cols = len(parts) - 1
        if cols < 0:
            malformed += 1
            continue
        coord_cols.append(cols)
    if coord_cols:
        col_counts = sorted(set(coord_cols))
        cols_text = ",".join(str(v) for v in col_counts)
    else:
        cols_text = "none"
    return (
        f"header_atoms={atom_count if atom_count is not None else atom_count_text!r}; "
        f"sample_coord_columns={cols_text}; "
        f"sampled_lines={len(sample)}; malformed={malformed}"
    )


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


def _sorted_artifacts_for_run(run: WorkflowRun) -> list[Artifact]:
    artifacts = list(Artifact.objects.filter(workflow_run=run).select_related("job"))
    for artifact in artifacts:
        artifact.source_scheduler_job_id = _artifact_source_scheduler_job_id(artifact)

    def _sort_key(artifact: Artifact) -> tuple[int, int, int, str, str, str, int]:
        source_job_id = str(getattr(artifact, "source_scheduler_job_id", "") or "").strip()
        job_id_numeric = _parse_optional_int(source_job_id)
        has_job_id = 1 if source_job_id else 0
        is_numeric = 1 if job_id_numeric is not None else 0
        return (
            -has_job_id,
            -is_numeric,
            -(job_id_numeric if job_id_numeric is not None else -1),
            source_job_id.casefold(),
            str(artifact.kind or "").casefold(),
            str(artifact.label or "").casefold(),
            -int(artifact.pk or 0),
        )

    artifacts.sort(key=_sort_key)
    return artifacts


def _artifact_ingestion_path(artifact: Artifact) -> str:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    ingestion_path = normalize_run_path(str(metadata.get("ingestion_path") or "").strip())
    if ingestion_path:
        return ingestion_path
    remote_path = normalize_run_path(str(metadata.get("remote_path") or "").strip())
    if remote_path:
        return remote_path
    return normalize_run_path(str(artifact.path or "").strip())


def _normalize_ingestion_path_for_duplicate_group(path_text: str, source_scheduler_job_id: str) -> str:
    normalized = normalize_run_path(path_text)
    if not normalized:
        return ""
    parts = [part for part in normalized.split("/") if part]
    source_id = str(source_scheduler_job_id or "").strip()
    if source_id and parts and parts[0] == source_id:
        parts = parts[1:]
    elif parts and parts[0].isdigit() and len(parts[0]) >= 4:
        # Normalize mixed layouts (<job_id>/... vs run-root/...).
        parts = parts[1:]
    return "/".join(parts)


def _artifact_duplicate_group_key(artifact: Artifact) -> tuple[str, str, str] | None:
    label = str(artifact.label or "").strip()
    ingestion_path = _artifact_ingestion_path(artifact)
    submission_id = str(_artifact_source_scheduler_job_id(artifact) or "").strip()
    if not label or not ingestion_path or not submission_id:
        return None
    normalized_ingestion_path = _normalize_ingestion_path_for_duplicate_group(ingestion_path, submission_id)
    if not normalized_ingestion_path:
        return None
    return (
        label.casefold(),
        normalized_ingestion_path.casefold(),
        submission_id,
    )


def _artifact_duplicate_keep_score(artifact: Artifact, referenced_artifact_ids: set[int]) -> tuple[int, int, int, object, int]:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    has_reference = int(int(artifact.pk or 0) in referenced_artifact_ids)
    has_job_link = int(artifact.job_id is not None)
    is_local_or_downloaded = int(
        (not bool(metadata.get("remote_only")))
        or bool(metadata.get("downloaded_from_hpc"))
        or bool(str(metadata.get("local_cache_path") or "").strip())
    )
    recency = artifact.updated_at or artifact.created_at
    return (
        has_reference,
        has_job_link,
        is_local_or_downloaded,
        recency,
        int(artifact.pk or 0),
    )


def _delete_duplicate_artifacts_for_run(run: WorkflowRun) -> dict[str, object]:
    artifacts = list(Artifact.objects.filter(workflow_run=run).select_related("job").order_by("pk"))
    referenced_artifact_ids = set(
        ComputedProperty.objects.filter(
            workflow_run=run,
            source_artifact__isnull=False,
        ).values_list("source_artifact_id", flat=True)
    )

    grouped: dict[tuple[str, str, str], list[Artifact]] = defaultdict(list)
    for artifact in artifacts:
        key = _artifact_duplicate_group_key(artifact)
        if key is None:
            continue
        grouped[key].append(artifact)

    removed_ids: list[int] = []
    duplicate_groups = 0
    for members in grouped.values():
        if len(members) < 2:
            continue
        duplicate_groups += 1
        keeper = max(members, key=lambda item: _artifact_duplicate_keep_score(item, referenced_artifact_ids))
        for artifact in members:
            artifact_id = int(artifact.pk or 0)
            if artifact_id == int(keeper.pk or 0):
                continue
            artifact.delete()
            if artifact_id:
                removed_ids.append(artifact_id)

    return {
        "duplicate_groups": int(duplicate_groups),
        "removed_count": len(removed_ids),
        "removed_artifact_ids": sorted(removed_ids),
    }


def _sorted_computed_properties_for_run(run: WorkflowRun) -> list[ComputedProperty]:
    properties = list(ComputedProperty.objects.filter(workflow_run=run))

    def _sort_key(prop: ComputedProperty) -> tuple[int, str, str, int]:
        # Text-valued properties first, then numeric-valued properties.
        value_type_rank = 1 if prop.value_float is not None else 0
        name_folded = str(prop.name or "").casefold()
        unit_folded = str(prop.unit or "").casefold()
        return (value_type_rank, name_folded, unit_folded, int(prop.pk))

    properties.sort(key=_sort_key)
    return properties


def _scan_row_lookup(row: object, key: str) -> object:
    key_values = dict(getattr(row, "key_value_pairs", {}) or {})
    if key in key_values:
        return key_values[key]
    data = dict(getattr(row, "data", {}) or {})
    if key in data:
        return data[key]
    return getattr(row, key, None)


def _scan_endpoint_records_from_db(scan_db_path: Path) -> list[tuple[int, float | None, object]]:
    selected: dict[int, tuple[float | None, object]] = {}
    try:
        db = ase_connect(str(scan_db_path))
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
    except Exception as exc:
        raise RuntimeError(f"Failed to read scan endpoint frames from database: {exc}") from exc

    ordered = sorted(selected.items(), key=lambda item: item[0])
    return [(int(iteration), energy, atoms) for iteration, (energy, atoms) in ordered]


def _scan_endpoint_frames_from_db(scan_db_path: Path) -> list[object]:
    return [atoms for _iteration, _energy, atoms in _scan_endpoint_records_from_db(scan_db_path)]


def _transition_state_record_from_scan_records(
    records: list[tuple[int, float | None, object]],
) -> tuple[int, float, object] | None:
    candidates = [
        (int(iteration), float(energy), atoms)
        for iteration, energy, atoms in records
        if energy is not None
    ]
    if not candidates:
        return None
    # The scan transition-state candidate is the peak-energy endpoint in the
    # exported reaction coordinate profile.
    return max(candidates, key=lambda item: (item[1], item[0]))


def _build_ping_pong_trajectory_frames(frames: list[object]) -> list[object]:
    if len(frames) <= 2:
        return list(frames)
    return [*frames, *frames[-2:0:-1]]


def _latest_scan_property_maps(run_ids: list[int]) -> dict[int, dict[str, object]]:
    if not run_ids:
        return {}
    props = (
        ComputedProperty.objects.filter(
            workflow_run_id__in=run_ids,
            name__in=list(_SCAN_PROPERTY_UNITS.keys()),
        )
        .order_by("workflow_run_id", "name", "unit", "-updated_at", "-created_at", "-id")
    )
    per_run: dict[int, dict[str, object]] = {}
    seen: set[tuple[int, str, str]] = set()
    for prop in props:
        key = (int(prop.workflow_run_id), str(prop.name), str(prop.unit))
        expected_unit = _SCAN_PROPERTY_UNITS.get(str(prop.name))
        if expected_unit is None or str(prop.unit) != expected_unit:
            continue
        if key in seen:
            continue
        seen.add(key)
        payload = per_run.setdefault(int(prop.workflow_run_id), {})
        if prop.value_float is not None:
            payload[str(prop.name)] = float(prop.value_float)
        elif prop.value_text:
            payload[str(prop.name)] = str(prop.value_text).strip()
    return per_run


def _run_formula_and_hash(run: WorkflowRun, prop_map: dict[str, object]) -> tuple[str, str]:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    structure = metadata.get("structure")
    formula = ""
    structure_hash = ""
    if isinstance(structure, dict):
        formula = str(structure.get("formula") or "").strip()
        structure_hash = str(structure.get("structure_sha256") or "").strip().lower()
    if not formula:
        formula = str(prop_map.get("final_structure_formula") or "").strip()
    if not structure_hash:
        structure_hash = str(prop_map.get("final_structure_sha256") or "").strip().lower()
    return formula, structure_hash


def _run_target_wavelength_nm(run: WorkflowRun, prop_map: dict[str, object]) -> float | None:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission")
    run_mode = ""
    if isinstance(submission, dict):
        run_mode = str(submission.get("run_mode") or "").strip()
    if run_mode and run_mode != "tddft_bond_scan":
        return None
    if isinstance(submission, dict):
        parsed = _parse_optional_float(submission.get("target_wavelength_nm"))
        if parsed is not None:
            return float(parsed)
    result_summary = metadata.get("result_summary")
    if isinstance(result_summary, dict):
        settings_payload = result_summary.get("settings")
        if isinstance(settings_payload, dict):
            parsed = _parse_optional_float(settings_payload.get("target_wavelength_nm"))
            if parsed is not None:
                return float(parsed)
    parsed = _parse_optional_float(prop_map.get("target_wavelength_nm"))
    if parsed is not None:
        return float(parsed)
    return None


def _build_bond_scan_rows(run: WorkflowRun, *, target_filter_nm: float | None) -> dict[str, object]:
    candidates = list(
        WorkflowRun.objects.filter(status=WorkflowStatus.COMPLETED)
        .select_related("settings_profile")
        .order_by("-created_at", "-id")
    )
    run_ids = [int(candidate.pk) for candidate in candidates]
    prop_maps = _latest_scan_property_maps(run_ids)
    current_props = prop_maps.get(int(run.pk), {})
    current_formula, current_hash = _run_formula_and_hash(run, current_props)

    unfiltered_rows: list[dict[str, object]] = []
    for candidate in candidates:
        props = prop_maps.get(int(candidate.pk), {})
        formula, structure_hash = _run_formula_and_hash(candidate, props)
        target_nm = _run_target_wavelength_nm(candidate, props)
        start_e = _parse_optional_float(props.get("bond_scan_start_energy_eV"))
        peak_e = _parse_optional_float(props.get("bond_scan_peak_energy_eV"))
        barrier_e = _parse_optional_float(props.get("bond_scan_activation_barrier_eV"))
        candidate_meta = candidate.metadata if isinstance(candidate.metadata, dict) else {}
        submission = candidate_meta.get("submission")
        run_mode = str(submission.get("run_mode") or "").strip() if isinstance(submission, dict) else ""
        has_scan_payload = (
            run_mode == "tddft_bond_scan"
            or start_e is not None
            or peak_e is not None
            or barrier_e is not None
        )
        if not has_scan_payload:
            continue

        if current_hash and structure_hash and structure_hash == current_hash:
            linked = True
            link_reason = "hash"
        elif current_formula and formula and formula == current_formula:
            linked = True
            link_reason = "formula"
        elif not current_hash and not current_formula:
            linked = True
            link_reason = "global"
        else:
            linked = False
            link_reason = ""
        if not linked:
            continue

        unfiltered_rows.append(
            {
                "run_id": int(candidate.pk),
                "run_name": candidate.run_name,
                "status": candidate.status,
                "settings_profile": (
                    candidate.settings_profile.name if candidate.settings_profile_id else "-"
                ),
                "formula": formula,
                "structure_sha256": structure_hash,
                "target_wavelength_nm": target_nm,
                "bond_scan_start_energy_eV": start_e,
                "bond_scan_peak_energy_eV": peak_e,
                "bond_scan_activation_barrier_eV": barrier_e,
                "created_at": candidate.created_at,
                "is_current_run": int(candidate.pk) == int(run.pk),
                "link_reason": link_reason,
            }
        )

    available_targets = sorted(
        {float(row["target_wavelength_nm"]) for row in unfiltered_rows if row["target_wavelength_nm"] is not None}
    )
    group_counts: dict[str, int] = defaultdict(int)
    for row in unfiltered_rows:
        structure_hash = str(row.get("structure_sha256") or "").strip()
        formula = str(row.get("formula") or "").strip()
        group_key = structure_hash or formula or f"run:{row['run_id']}"
        group_counts[group_key] += 1

    filtered_rows: list[dict[str, object]] = []
    for row in unfiltered_rows:
        if target_filter_nm is not None:
            row_target = _parse_optional_float(row.get("target_wavelength_nm"))
            if row_target is None or abs(row_target - target_filter_nm) > 1.0e-9:
                continue
        structure_hash = str(row.get("structure_sha256") or "").strip()
        formula = str(row.get("formula") or "").strip()
        group_key = structure_hash or formula or f"run:{row['run_id']}"
        row["link_group_size"] = int(group_counts.get(group_key, 1))
        filtered_rows.append(row)

    filtered_rows.sort(
        key=lambda item: (
            0 if item.get("is_current_run") else 1,
            -(item["created_at"].timestamp() if item.get("created_at") else 0.0),
            -int(item["run_id"]),
        )
    )

    return {
        "rows": filtered_rows,
        "available_targets_nm": available_targets,
        "current_formula": current_formula,
        "current_structure_hash": current_hash,
        "selected_target_nm": target_filter_nm,
    }


class WorkflowRunListView(FriendlyTablePaginationMixin, SingleTableMixin, FilterView):
    model = WorkflowRun
    table_class = WorkflowRunTable
    template_name = "workflows/run_list.html"
    filterset_class = WorkflowRunFilter

    _STATUS_RANK = {
        "initiated": 0,
        "running": 1,
        "failed": 2,
        "completed": 3,
    }

    def get_queryset(self):
        return (
            WorkflowRun.objects.select_related("workflow", "settings_profile")
            .prefetch_related("jobs")
            .order_by("-created_at", "-id")
        )

    @classmethod
    def _record_rank(cls, record: WorkflowRun) -> tuple:
        raw_path = normalize_run_path(record.working_directory)
        canonical_path = canonical_run_path(record.working_directory)
        has_posix_path = int(raw_path == canonical_path)
        return (
            has_posix_path,
            cls._STATUS_RANK.get(record.status, -1),
            int(record.completed_at is not None),
            int(record.started_at is not None),
            record.updated_at or record.created_at,
            record.id,
        )

    def get_table_data(self):
        rows = list(self.object_list)
        selected_by_key: dict[tuple[str, str], WorkflowRun] = {}

        for row in rows:
            key = (row.run_name, canonical_run_path(row.working_directory))
            existing = selected_by_key.get(key)
            if existing is None or self._record_rank(row) > self._record_rank(existing):
                selected_by_key[key] = row

        selected = list(selected_by_key.values())
        selected.sort(key=lambda row: (row.created_at, row.id), reverse=True)
        return selected


def _extract_mep_data(run: WorkflowRun) -> dict[str, object]:
    from ase.io import read
    from core.mep_render import analyze_structure_connectivity, compute_cumulative_atomic_shifts
    import json
    import numpy as np

    band_art = run.artifacts.filter(kind="trajectory", path__icontains="final_band").first()
    if not band_art:
        band_art = run.artifacts.filter(kind="trajectory", path__icontains="final").first()
    if not band_art:
        raise ValueError(f"Run #{run.pk} has no converged MEP trajectory artifact (final_band.extxyz).")

    lp = _artifact_local_path(run, band_art)
    if not lp.exists():
        raise FileNotFoundError(f"MEP band file not found in local cache for Run #{run.pk}.")

    images = read(str(lp), index=":", format="extxyz" if str(lp).endswith(".extxyz") else None)
    if not images or len(images) < 2:
        raise ValueError(f"Run #{run.pk} trajectory contains fewer than 2 images.")

    summary_art = run.artifacts.filter(kind="json", path__icontains="summary").first()
    energies = None
    if summary_art:
        slp = _artifact_local_path(run, summary_art)
        if slp.exists():
            try:
                with open(slp, "rb") as f:
                    s_data = json.loads(f.read().decode("utf-8", errors="replace"))
                    energies = s_data.get("energies_eV")
            except Exception:
                pass

    if not energies or len(energies) != len(images):
        energies = [
            img.get_potential_energy() if hasattr(img, "calc") and img.calc else img.info.get("energy", 0.0)
            for img in images
        ]

    shifts = compute_cumulative_atomic_shifts(images)
    rel_e = [float(e - energies[0]) for e in energies]

    ts_idx = int(np.argmax(rel_e))
    forward_barrier = float(rel_e[ts_idx])
    reaction_energy = float(rel_e[-1])
    reverse_barrier = float(forward_barrier - reaction_energy)

    cluster_formula_hint = (
        run.computed_properties.filter(name="nanocluster_formula")
        .values_list("value_text", flat=True)
        .first()
        or str(getattr(run.reference_nanocluster, "composition", "") or "")
    )
    cluster_formula, ads_init = analyze_structure_connectivity(images[0], cluster_formula_hint)
    _, ads_final = analyze_structure_connectivity(images[-1], cluster_formula_hint)

    ts_art = run.artifacts.filter(kind="structure", path__icontains="transition_state").first()

    return {
        "run": run,
        "run_id": run.pk,
        "run_name": run.run_name,
        "n_images": len(images),
        "shifts": shifts,
        "energies": energies,
        "rel_e": rel_e,
        "ts_idx": ts_idx,
        "ts_shift": shifts[ts_idx],
        "forward_barrier_eV": forward_barrier,
        "forward_barrier_kj": forward_barrier * 96.4853075,
        "forward_barrier_kcal": forward_barrier * 23.0605419,
        "reverse_barrier_eV": reverse_barrier,
        "reverse_barrier_kj": reverse_barrier * 96.4853075,
        "reaction_energy_eV": reaction_energy,
        "reaction_energy_kj": reaction_energy * 96.4853075,
        "start_energy_eV": float(energies[0]),
        "ts_energy_eV": float(energies[ts_idx]),
        "final_energy_eV": float(energies[-1]),
        "cluster_formula": cluster_formula,
        "ads_init": ads_init,
        "ads_final": ads_final,
        "ts_artifact_pk": ts_art.pk if ts_art else None,
        "ts_atoms": images[ts_idx],
    }


def _generate_mlneb_comparison_plot(mep_a: dict, mep_b: dict) -> str:
    import io
    import base64
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    import numpy as np

    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=140)
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#f8fafc")

    # Run A
    s_a = np.array(mep_a["shifts"])
    e_a = np.array(mep_a["rel_e"])
    if len(s_a) > 1 and np.all(np.diff(s_a) > 0):
        s_fine_a = np.linspace(s_a[0], s_a[-1], 300)
        e_fine_a = make_interp_spline(s_a, e_a, k=min(3, len(s_a) - 1))(s_fine_a)
    else:
        s_fine_a, e_fine_a = s_a, e_a

    label_a = f"#{mep_a['run_id']} (Ea_fwd: {mep_a['forward_barrier_eV']:.2f} eV | Ea_rev: {mep_a['reverse_barrier_eV']:.2f} eV)"
    ax.plot(s_fine_a, e_fine_a, color="#2563eb", linewidth=2.8, label=label_a)
    ax.scatter(s_a, e_a, color="#3b82f6", s=40, edgecolors="#1d4ed8", zorder=3)
    ax.scatter(
        [mep_a["ts_shift"]], [mep_a["forward_barrier_eV"]],
        color="#1d4ed8", marker="s", s=140, edgecolors="#0f172a", linewidths=1.5, zorder=6,
    )

    # Run B
    s_b = np.array(mep_b["shifts"])
    e_b = np.array(mep_b["rel_e"])
    if len(s_b) > 1 and np.all(np.diff(s_b) > 0):
        s_fine_b = np.linspace(s_b[0], s_b[-1], 300)
        e_fine_b = make_interp_spline(s_b, e_b, k=min(3, len(s_b) - 1))(s_fine_b)
    else:
        s_fine_b, e_fine_b = s_b, e_b

    label_b = f"#{mep_b['run_id']} (Ea_fwd: {mep_b['forward_barrier_eV']:.2f} eV | Ea_rev: {mep_b['reverse_barrier_eV']:.2f} eV)"
    ax.plot(s_fine_b, e_fine_b, color="#d97706", linewidth=2.8, linestyle="--", label=label_b)
    ax.scatter(s_b, e_b, color="#f59e0b", s=40, edgecolors="#b45309", zorder=3)
    ax.scatter(
        [mep_b["ts_shift"]], [mep_b["forward_barrier_eV"]],
        color="#b45309", marker="s", s=140, edgecolors="#0f172a", linewidths=1.5, zorder=6,
    )

    ax.set_title("Minimum Energy Path (MEP) Comparison", fontsize=13, fontweight="bold", pad=14, color="#0f172a")
    ax.set_xlabel("Cumulative Atomic Shift (Å)", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Relative Energy ΔE (eV)", fontsize=11, fontweight="bold", labelpad=8)
    ax.grid(True, linestyle="--", alpha=0.55, color="#cbd5e1")
    ax.legend(frameon=True, facecolor="#ffffff", edgecolor="#cbd5e1", fontsize=9.5, loc="lower right")

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _calculate_ts_similarity(run_a: WorkflowRun, run_b: WorkflowRun) -> dict[str, object] | None:
    _ensure_dense_sparse_shim()
    from dscribe.descriptors import SOAP
    from sklearn.metrics.pairwise import pairwise_kernels
    from sklearn.preprocessing import normalize
    import numpy as np
    from ase.io import read

    art_a = run_a.artifacts.filter(kind="structure", path__icontains="transition_state").first()
    art_b = run_b.artifacts.filter(kind="structure", path__icontains="transition_state").first()
    if not art_a or not art_b:
        return None

    lp_a = _artifact_local_path(run_a, art_a)
    lp_b = _artifact_local_path(run_b, art_b)
    if not lp_a.exists() or not lp_b.exists():
        return None

    atoms_a = read(str(lp_a))
    atoms_b = read(str(lp_b))

    atoms = [atoms_a, atoms_b]
    species = sorted({s for a in atoms for s in a.get_chemical_symbols()})
    soap = SOAP(species=species, periodic=False, r_cut=6.0, n_max=8, l_max=6)
    features = [normalize(soap.create(a)) for a in atoms]

    raw = np.array([
        [
            _rematch_global_similarity(pairwise_kernels(features[i], features[j], metric="linear", n_jobs=1))
            for j in range(2)
        ]
        for i in range(2)
    ])

    diag_sqrt = np.sqrt(np.diag(raw))
    norm_matrix = raw / np.outer(diag_sqrt, diag_sqrt)

    return {
        "score": float(norm_matrix[0, 1]),
        "raw_score": float(raw[0, 1]),
        "self_a": float(raw[0, 0]),
        "self_b": float(raw[1, 1]),
        "ts_art_a_pk": art_a.pk,
        "ts_art_b_pk": art_b.pk,
    }


def _build_mlneb_compare_context(run: WorkflowRun, compare_run_id: int | None = None) -> dict[str, object]:
    is_mlneb = "MLNEB" in run.run_name.upper() or "NEB" in run.run_name.upper() or "neb" in run.workflow.name.lower()

    if not is_mlneb:
        return {"is_mlneb_run": False, "candidates": [], "has_comparison": False}

    formula = run.computed_properties.filter(name="nanocluster_formula").values_list("value_text", flat=True).first()
    if not formula:
        formula = str(run.reference_nanocluster or "").strip()

    candidate_qs = (
        WorkflowRun.objects.filter(
            Q(run_name__icontains="MLNEB") | Q(run_name__icontains="NEB") | Q(workflow__name__icontains="NEB"),
            artifacts__kind="trajectory",
            artifacts__path__icontains="final_band",
        )
        .exclude(pk=run.pk)
        .distinct()
        .order_by("-id")
    )

    candidates = []
    for cand in candidate_qs:
        band_art = cand.artifacts.filter(kind="trajectory", path__icontains="final_band").first()
        if not band_art or not _artifact_local_path(cand, band_art).exists():
            continue
        cand_formula = cand.computed_properties.filter(name="nanocluster_formula").values_list("value_text", flat=True).first() or str(cand.reference_nanocluster or "").strip()
        candidates.append({
            "run": cand,
            "id": cand.id,
            "name": cand.run_name,
            "formula": cand_formula,
            "is_matching_formula": bool(cand_formula and formula and cand_formula == formula),
        })

    # Sort matching formulas to the top
    candidates.sort(key=lambda c: (c["is_matching_formula"], c["id"]), reverse=True)

    selected_candidate = None
    if compare_run_id:
        selected_candidate = next(
            (candidate["run"] for candidate in candidates if candidate["id"] == compare_run_id),
            None,
        )
    elif candidates:
        matching = [c["run"] for c in candidates if c["is_matching_formula"]]
        selected_candidate = matching[0] if matching else candidates[0]["run"]

    context = {
        "is_mlneb_run": is_mlneb,
        "current_formula": formula,
        "candidates": candidates,
        "selected_run_id": selected_candidate.pk if selected_candidate else None,
        "has_comparison": False,
    }

    if not selected_candidate:
        return context

    try:
        mep_a = _extract_mep_data(run)
        mep_b = _extract_mep_data(selected_candidate)
        b64_plot = _generate_mlneb_comparison_plot(mep_a, mep_b)
        ts_sim = _calculate_ts_similarity(run, selected_candidate)

        delta_ea_fwd = abs(mep_a["forward_barrier_eV"] - mep_b["forward_barrier_eV"])
        delta_ea_rev = abs(mep_a["reverse_barrier_eV"] - mep_b["reverse_barrier_eV"])
        delta_e_ts = abs(mep_a["ts_energy_eV"] - mep_b["ts_energy_eV"])
        delta_erxn = abs(mep_a["reaction_energy_eV"] - mep_b["reaction_energy_eV"])

        context.update({
            "has_comparison": True,
            "mep_a": mep_a,
            "mep_b": mep_b,
            "compared_run": selected_candidate,
            "comparison_plot_b64": b64_plot,
            "ts_similarity": ts_sim,
            "delta_ea_fwd_eV": delta_ea_fwd,
            "delta_ea_fwd_kj": delta_ea_fwd * 96.4853075,
            "delta_ea_fwd_kcal": delta_ea_fwd * 23.0605419,
            "delta_ea_rev_eV": delta_ea_rev,
            "delta_ea_rev_kj": delta_ea_rev * 96.4853075,
            "delta_ea_rev_kcal": delta_ea_rev * 23.0605419,
            "delta_e_ts_eV": delta_e_ts,
            "delta_e_ts_kj": delta_e_ts * 96.4853075,
            "delta_e_ts_kcal": delta_e_ts * 23.0605419,
            "delta_erxn_eV": delta_erxn,
            "delta_erxn_kj": delta_erxn * 96.4853075,
            "delta_erxn_kcal": delta_erxn * 23.0605419,
        })
    except Exception as exc:
        context["error"] = str(exc)

    return context


class WorkflowRunDetailView(DetailView):
    model = WorkflowRun
    template_name = "workflows/run_detail.html"
    context_object_name = "run"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        run = self.object
        tab = str(self.request.GET.get("tab") or "overview").strip().lower()
        if tab not in {"overview", "bond_scan", "structure_similarity", "mlneb_compare"}:
            tab = "overview"
        target_filter_nm = _parse_optional_float(self.request.GET.get("scan_target_nm"))
        context["selected_tab"] = tab
        context["structure_similarity"] = {"compare_run_id": str(self.request.GET.get("compare_run_id") or "").strip()}
        if tab == "structure_similarity" and context["structure_similarity"]["compare_run_id"]:
            try:
                other_run_id = int(context["structure_similarity"]["compare_run_id"])
                other_run = WorkflowRun.objects.get(pk=other_run_id)
                context["structure_similarity"].update(_compare_run_structure_similarity(run, other_run))
                context["structure_similarity"]["has_result"] = True
            except WorkflowRun.DoesNotExist:
                context["structure_similarity"]["error"] = "No run found with that id."
            except (TypeError, ValueError) as exc:
                context["structure_similarity"]["error"] = str(exc)
            except Exception as exc:
                context["structure_similarity"]["error"] = f"Could not compute similarity: {exc}"
        
        is_mlneb_run = "NEB" in run.run_name.upper() or "neb" in run.workflow.name.lower()
        if tab == "mlneb_compare" and is_mlneb_run:
            mlneb_compare_id = _parse_optional_int(self.request.GET.get("mlneb_compare_run_id"))
            context["mlneb_compare"] = _build_mlneb_compare_context(run, compare_run_id=mlneb_compare_id)
        else:
            context["mlneb_compare"] = {
                "is_mlneb_run": is_mlneb_run,
                "candidates": [],
                "has_comparison": False,
            }
        jobs = list(JobRecord.objects.filter(workflow_run=run).order_by("-created_at"))
        context["jobs"] = jobs
        latest_job = jobs[0] if jobs else None
        context["deletion_job"] = latest_job if latest_job and latest_job.status != JobStatus.DELETED else None
        run_metadata = run.metadata if isinstance(run.metadata, dict) else {}
        run_submission = run_metadata.get("submission") if isinstance(run_metadata.get("submission"), dict) else {}
        run_mode = str(run_submission.get("run_mode") or "").strip()
        try:
            source_run_id = int(run_metadata.get("source_run_id") or 0)
        except (TypeError, ValueError):
            source_run_id = 0
        context["source_run"] = (
            WorkflowRun.objects.filter(pk=source_run_id).first()
            if source_run_id and source_run_id != run.pk
            else None
        )
        context["smooth_resubmit_job"] = (
            latest_job
            if (
                latest_job is not None
                and latest_job.status == JobStatus.COMPLETED
                and run.status == WorkflowStatus.COMPLETED
                and run_mode == "tddft_bond_scan"
            )
            else None
        )
        context["replot_resubmit_job"] = (
            latest_job
            if (
                latest_job is not None
                and latest_job.status == JobStatus.COMPLETED
                and run.status == WorkflowStatus.COMPLETED
                and bool(getattr(latest_job, "supports_replot_resubmit", False))
            )
            else None
        )
        artifacts = _sorted_artifacts_for_run(run)
        for artifact in artifacts:
            batch_submit_url = ""
            if _is_batch_submission_source_artifact(run, artifact):
                query = urlencode(
                    {
                        "from_run": int(run.pk),
                        "source_artifact": int(artifact.pk),
                        "mode": "single_optimisation",
                        "batch": 1,
                    }
                )
                batch_submit_url = f"{reverse('job-submit')}?{query}"
            setattr(artifact, "batch_submit_url", batch_submit_url)
        image_artifacts = [
            artifact
            for artifact in artifacts
            if _is_image_artifact(artifact) and _artifact_content_available(run, artifact)
        ]
        key_plot_artifacts = [artifact for artifact in image_artifacts if _is_key_plot_artifact(artifact)]
        render_image_artifacts = [
            artifact for artifact in image_artifacts if _is_render_image_artifact(artifact)
        ]
        context["artifacts"] = artifacts
        context["image_artifacts"] = image_artifacts
        context["image_artifact_ids"] = [artifact.pk for artifact in image_artifacts]
        context["key_plot_artifacts"] = key_plot_artifacts
        context["render_image_artifacts"] = render_image_artifacts
        context["content_available_artifact_ids"] = [
            artifact.pk for artifact in artifacts if _artifact_content_available(run, artifact)
        ]
        context["renderable_3d_artifact_ids"] = [
            artifact.pk
            for artifact in artifacts
            if (_is_structure_artifact(artifact) or _is_trajectory_artifact(artifact))
            and _artifact_content_available(run, artifact)
        ]
        context["ase_openable_artifact_ids"] = [artifact.pk for artifact in artifacts if _ase_gui_openable(run, artifact)]
        context["xyzrender_artifact_ids"] = [artifact.pk for artifact in artifacts if _xyzrender_actionable(run, artifact)]
        context["xyzrender_trajectory_artifact_ids"] = [
            artifact.pk
            for artifact in artifacts
            if _xyzrender_actionable(run, artifact) and _is_trajectory_artifact(artifact)
        ]
        context["xyzrender_scan_endpoints_db_artifact_ids"] = [
            artifact.pk
            for artifact in artifacts
            if _xyzrender_actionable(run, artifact) and _is_scan_endpoints_db_artifact(artifact)
        ]
        adsorption_energy_result = (
            AdsorptionEnergyResult.objects.select_related(
                "adsorbate",
                "nanocluster",
                "settings_profile",
            )
            .filter(workflow_run=run)
            .first()
        )
        context["adsorption_energy_result"] = adsorption_energy_result
        eint_energy_eV = (
            _metadata_interaction_energy_eint_eV(adsorption_energy_result)
            if adsorption_energy_result
            else None
        )
        context["eint_energy_eV"] = eint_energy_eV
        context["has_eint_energy"] = eint_energy_eV is not None
        properties = _sorted_computed_properties_for_run(run)
        context["properties"] = properties
        
        # Build dynamic hero metrics
        props_map = {p.name: p for p in properties}
        hero_metrics = []
        if adsorption_energy_result:
            hero_metrics.append({
                "label": "Adsorption energy",
                "value": f"{adsorption_energy_result.adsorption_energy_eV:.2f} eV",
                "accent": "blue",
            })
            if eint_energy_eV is not None:
                hero_metrics.append({
                    "label": "Interaction energy · Eint",
                    "value": f"{eint_energy_eV:.2f} eV",
                    "accent": "indigo",
                })
        if "activation_barrier_eV" in props_map:
            p_fwd = props_map["activation_barrier_eV"]
            hero_metrics.append({
                "label": "Forward Barrier (ΔE‡)",
                "value": f"{p_fwd.value_float:.3f} eV" if p_fwd.value_float is not None else p_fwd.value_text,
                "accent": "red",
            })
            if "reaction_energy_eV" in props_map:
                p_rxn = props_map["reaction_energy_eV"]
                hero_metrics.append({
                    "label": "Reaction Energy (ΔE)",
                    "value": f"{p_rxn.value_float:+.3f} eV" if p_rxn.value_float is not None else p_rxn.value_text,
                    "accent": "blue",
                })
            if "activation_barrier_reverse_eV" in props_map:
                p_rev = props_map["activation_barrier_reverse_eV"]
                hero_metrics.append({
                    "label": "Reverse Barrier (ΔE‡_rev)",
                    "value": f"{p_rev.value_float:.3f} eV" if p_rev.value_float is not None else p_rev.value_text,
                    "accent": "amber",
                })
            if "transition_state_energy_eV" in props_map:
                p_ts = props_map["transition_state_energy_eV"]
                hero_metrics.append({
                    "label": "TS Energy",
                    "value": f"{p_ts.value_float:.2f} eV" if p_ts.value_float is not None else p_ts.value_text,
                    "accent": "purple",
                })
        elif not adsorption_energy_result and "final_single_point_energy_eV" in props_map:
            p_sp = props_map["final_single_point_energy_eV"]
            hero_metrics.append({
                "label": "Total Energy",
                "value": f"{p_sp.value_float:.4f} eV" if p_sp.value_float is not None else p_sp.value_text,
                "accent": "blue",
            })
            if "atomization_energy_per_atom_eV" in props_map:
                p_atom = props_map["atomization_energy_per_atom_eV"]
                hero_metrics.append({
                    "label": "Atomization Energy",
                    "value": f"{p_atom.value_float:.3f} eV/atom" if p_atom.value_float is not None else p_atom.value_text,
                    "accent": "indigo",
                })
        # Resolve nanocluster and adsorbates from properties or references
        resolved_nanocluster = str(run.reference_nanocluster or "").strip()
        if not resolved_nanocluster and "nanocluster_formula" in props_map:
            resolved_nanocluster = props_map["nanocluster_formula"].value_text or ""
            
        resolved_initial_adsorbate = str(run.reference_adsorbate or "").strip()
        if not resolved_initial_adsorbate and "initial_adsorbate" in props_map:
            resolved_initial_adsorbate = props_map["initial_adsorbate"].value_text or ""
            
        resolved_final_adsorbate = ""
        if "final_adsorbate" in props_map:
            resolved_final_adsorbate = props_map["final_adsorbate"].value_text or ""

        context["resolved_nanocluster"] = resolved_nanocluster
        context["resolved_initial_adsorbate"] = resolved_initial_adsorbate
        context["resolved_final_adsorbate"] = resolved_final_adsorbate
        context["hero_metrics"] = hero_metrics
        context["bond_scan_compare"] = _build_bond_scan_rows(run, target_filter_nm=target_filter_nm)
        return context


@require_POST
def workflow_open_artifact_ase_gui_view(request, pk: int, artifact_pk: int):
    run = get_object_or_404(WorkflowRun, pk=pk)
    artifact = get_object_or_404(Artifact, pk=artifact_pk, workflow_run=run)

    if not _is_ase_gui_supported_artifact(artifact):
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    "ASE GUI opening is only supported for structure/trajectory/database artifacts."
                ),
            },
            status=400,
        )
    if _is_remote_only_artifact(artifact):
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    "Artifact is marked as remote-only and retained on HPC. "
                    "ASE GUI can only open local files."
                ),
            },
            status=400,
        )

    file_path = _artifact_local_path(run, artifact)
    if not file_path.exists() or not file_path.is_file():
        return JsonResponse(
            {"ok": False, "error": f"Artifact file does not exist locally: {file_path}"},
            status=400,
        )

    try:
        command = resolve_ase_gui_command(file_path)

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
    except Exception as exc:
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

    return JsonResponse(
        {
            "ok": True,
            "message": f"ASE GUI launch command started for artifact '{artifact.label}' ({file_path.name}).",
        }
    )


@require_POST
def workflow_delete_image_artifact_view(request, pk: int, artifact_pk: int):
    run = get_object_or_404(WorkflowRun, pk=pk)
    artifact = get_object_or_404(Artifact, pk=artifact_pk, workflow_run=run)

    if not _is_image_artifact(artifact):
        return JsonResponse(
            {"ok": False, "error": "Only image artifacts can be deleted from the gallery."},
            status=400,
        )

    file_path = None
    removed_file = False
    if not _is_remote_only_artifact(artifact):
        file_path = _artifact_local_path(run, artifact)
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                removed_file = True
            except OSError as exc:
                return JsonResponse(
                    {"ok": False, "error": f"Failed to delete local file '{file_path}': {exc}"},
                    status=400,
                )

    label = str(artifact.label or artifact.path)
    artifact.delete()
    return JsonResponse(
        {
            "ok": True,
            "message": (
                f"Deleted image artifact '{label}'."
                + (" Local file removed." if removed_file else " Record removed.")
            ),
        }
    )


@require_POST
def workflow_delete_duplicate_artifacts_view(request, pk: int):
    run = get_object_or_404(WorkflowRun, pk=pk)
    summary = _delete_duplicate_artifacts_for_run(run)
    removed_count = int(summary.get("removed_count") or 0)
    duplicate_groups = int(summary.get("duplicate_groups") or 0)

    if removed_count <= 0:
        return JsonResponse(
            {
                "ok": True,
                "removed_count": 0,
                "duplicate_groups": duplicate_groups,
                "message": (
                    "No duplicate artifacts matched (label, ingestion path, submission id)."
                ),
            }
        )

    return JsonResponse(
        {
            "ok": True,
            "removed_count": removed_count,
            "duplicate_groups": duplicate_groups,
            "removed_artifact_ids": summary.get("removed_artifact_ids", []),
            "message": (
                "Deleted "
                f"{removed_count} duplicate artifact record(s) across {duplicate_groups} duplicate group(s)."
            ),
        }
    )


@require_POST
def workflow_render_artifact_xyz_view(request, pk: int, artifact_pk: int):
    payload = {}
    if request.body:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception:
            payload = {}
    if request.POST:
        payload.update(request.POST.dict())
    try:
        result = render_artifact_xyz_task.delay(int(pk), int(artifact_pk), payload)
    except Exception as exc:
        return JsonResponse(
            {"ok": False, "error": f"Failed to queue artifact rendering task: {exc}"},
            status=500,
        )
    return JsonResponse(
        {
            "ok": True,
            "queued": True,
            "task_id": result.id,
            "message": f"Artifact rendering queued as Celery task {result.id}.",
        }
    )


def _workflow_render_artifact_xyz_sync_response(request, pk: int, artifact_pk: int):
    run = get_object_or_404(WorkflowRun, pk=pk)
    artifact = get_object_or_404(Artifact, pk=artifact_pk, workflow_run=run)

    if not _is_xyzrender_renderable_artifact(artifact):
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    "Rendering is only supported for structure/trajectory artifacts "
                    "and scan_endpoints.db database artifacts."
                ),
            },
            status=400,
        )
    if _is_remote_only_artifact(artifact):
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    "Artifact is marked as remote-only and retained on HPC. "
                    "Rendering requires a local file."
                ),
            },
            status=400,
        )

    source_path = _artifact_local_path(run, artifact)
    if not source_path.exists() or not source_path.is_file():
        return JsonResponse(
            {"ok": False, "error": f"Artifact file does not exist locally: {source_path}"},
            status=400,
        )

    payload = {}
    if request.body:
        try:
            payload = json.loads(request.body.decode("utf-8"))
        except Exception:
            payload = {}
    raw_gif_mode = str(
        request.POST.get("gif_mode")
        or payload.get("gif_mode")
        or "converged"
    ).strip().lower()
    requested_scan_endpoint_frame_mode = ""
    if raw_gif_mode in {"", "converged", "rotation"}:
        requested_trajectory_mode = False
        requested_scan_ts_mode = False
    elif raw_gif_mode == "trajectory":
        requested_trajectory_mode = True
        requested_scan_ts_mode = False
    elif raw_gif_mode in {"transition_state", "ts"}:
        requested_trajectory_mode = False
        requested_scan_ts_mode = True
    elif raw_gif_mode in {"first", "first_endpoint", "scan_first"}:
        requested_trajectory_mode = False
        requested_scan_ts_mode = False
        requested_scan_endpoint_frame_mode = "first"
    elif raw_gif_mode in {"last", "last_endpoint", "scan_last"}:
        requested_trajectory_mode = False
        requested_scan_ts_mode = False
        requested_scan_endpoint_frame_mode = "last"
    else:
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    f"Unsupported GIF mode '{raw_gif_mode}'. "
                    "Choose 'converged', 'trajectory', 'transition_state', 'first', or 'last'."
                ),
            },
            status=400,
        )

    is_scan_endpoints_db = _is_scan_endpoints_db_artifact(artifact)
    trajectory_mode_available = _is_trajectory_artifact(artifact)
    if requested_trajectory_mode and not (trajectory_mode_available or is_scan_endpoints_db):
        return JsonResponse(
            {"ok": False, "error": "Trajectory GIF mode is only available for trajectory (.traj) artifacts."},
            status=400,
        )
    if requested_scan_ts_mode and not is_scan_endpoints_db:
        return JsonResponse(
            {"ok": False, "error": "Transition-state image mode is only available for scan_endpoints.db artifacts."},
            status=400,
        )
    if requested_scan_endpoint_frame_mode and not is_scan_endpoints_db:
        return JsonResponse(
            {"ok": False, "error": "First/last endpoint image modes are only available for scan_endpoints.db artifacts."},
            status=400,
        )
    raw_render_output = str(
        request.POST.get("render_output")
        or payload.get("render_output")
        or "both"
    ).strip().lower()
    if raw_render_output not in {"svg", "gif", "both"}:
        return JsonResponse(
            {
                "ok": False,
                "error": (
                    f"Unsupported render output '{raw_render_output}'. "
                    "Choose 'svg', 'gif', or 'both'."
                ),
            },
            status=400,
        )
    render_svg = raw_render_output in {"svg", "both"}
    render_gif = raw_render_output in {"gif", "both"}
    use_trajectory_gif = (
        is_scan_endpoints_db
        and not requested_scan_ts_mode
        and not requested_scan_endpoint_frame_mode
    ) or (
        trajectory_mode_available and requested_trajectory_mode
    )
    use_full_artifact_trajectory = use_trajectory_gif and (not is_scan_endpoints_db)

    raw_rotation = str(
        request.POST.get("gif_rot")
        or payload.get("gif_rot")
        or "x"
    ).strip().lower()
    if (render_svg or not use_trajectory_gif) and raw_rotation not in _XYZRENDER_ALLOWED_GIF_ROTATIONS:
        allowed = ", ".join(sorted(_XYZRENDER_ALLOWED_GIF_ROTATIONS))
        return JsonResponse(
            {
                "ok": False,
                "error": f"Unsupported rotation '{raw_rotation}'. Choose one of: {allowed}.",
            },
            status=400,
        )
    rotation_tag = "".join(ch for ch in raw_rotation if ch.isalnum()) or "x"

    try:
        rot_frames = max(1, int(getattr(settings, "REACTION_HUB_XYZRENDER_ROT_FRAMES", 200)))
    except (TypeError, ValueError):
        rot_frames = 200

    command_warning = ""
    command_setting = str(getattr(settings, "REACTION_HUB_XYZRENDER_COMMAND", "") or "").strip()
    try:
        command, command_warning = resolve_xyzrender_command(command_setting)
    except ValueError as exc:
        return JsonResponse(
            {"ok": False, "error": str(exc)},
            status=400,
        )
    if not command:
        return JsonResponse(
            {"ok": False, "error": "REACTION_HUB_XYZRENDER_COMMAND is empty."},
            status=400,
        )

    if requested_scan_endpoint_frame_mode:
        render_input_base = f"{source_path.stem}_{requested_scan_endpoint_frame_mode}_endpoint_rot_{raw_rotation}"
        output_svg_base = (
            f"{source_path.stem}_{requested_scan_endpoint_frame_mode}_endpoint_rot_{rotation_tag}"
            if render_svg and not render_gif
            else f"{source_path.stem}_{requested_scan_endpoint_frame_mode}_endpoint"
        )
        expected_gif_base = render_input_base
    elif requested_scan_ts_mode:
        render_input_base = f"{source_path.stem}_transition_state_rot_{raw_rotation}"
        output_svg_base = (
            f"{source_path.stem}_transition_state_rot_{rotation_tag}"
            if render_svg and not render_gif
            else f"{source_path.stem}_transition_state"
        )
        expected_gif_base = render_input_base
    elif use_trajectory_gif:
        render_input_base = f"{source_path.stem}_ping_pong"
        output_svg_base = (
            f"{render_input_base}_view_{rotation_tag}"
            if render_svg and not render_gif
            else render_input_base
        )
        expected_gif_base = render_input_base
    else:
        render_input_base = f"{source_path.stem}_rot_{rotation_tag}"
        output_svg_base = (
            f"{source_path.stem}_render_rot_{rotation_tag}"
            if render_svg and not render_gif
            else f"{source_path.stem}_render"
        )
        expected_gif_base = render_input_base
    render_input: Path | None = None
    cleanup_files: list[Path] = []
    keep_files: set[Path] = set()
    render_notes: list[str] = []
    fixed_scale_render_input: Path | None = None
    try:
        render_input = _prepare_xyzrender_input_xyz(
            source_path,
            render_input_base,
            source_artifact=artifact,
            use_full_trajectory=use_full_artifact_trajectory,
            scan_endpoint_mode=(
                requested_scan_endpoint_frame_mode
                or ("transition_state" if requested_scan_ts_mode else "trajectory")
            ),
            static_svg_orientation=(raw_rotation if render_svg and not render_gif else ""),
        )
        cleanup_files.append(render_input)
        command_input = render_input
        if render_gif and use_trajectory_gif and is_scan_endpoints_db:
            fixed_scale_render_input = _prepare_xyzrender_input_xyz(
                source_path,
                f"{render_input_base}_fixed_scale",
                source_artifact=artifact,
                use_full_trajectory=use_full_artifact_trajectory,
                scan_endpoint_mode="trajectory",
                add_fixed_scale_anchors=True,
            )
            cleanup_files.append(fixed_scale_render_input)
            command_input = fixed_scale_render_input
        render_dir = render_input.parent
        output_svg = render_dir / (
            f"{output_svg_base}.svg" if render_svg else f"{output_svg_base}_gif_tmp.svg"
        )
        expected_gif = render_dir / f"{expected_gif_base}.gif"
        preexisting_gifs: set[Path] = set()
        if render_gif:
            preexisting_gifs = {path.resolve() for path in render_dir.glob("*.gif")}
        if not render_svg:
            cleanup_files.append(output_svg)

        render_command = [
            *command,
            command_input.name,
            "-o",
            output_svg.name,
        ]
        if render_gif and use_trajectory_gif:
            render_command.extend(["-go", expected_gif.name])
            # Use the regular trajectory renderer for scan_endpoints.db too.
            # The transition-state GIF path can autoscale frames independently,
            # which makes atom sizes vary across the ping-pong scan.
            render_command.append("--gif-trj")
        elif render_gif:
            render_command.extend(["--gif-rot", raw_rotation, "--rot-frames", str(rot_frames)])

        completed = subprocess.run(  # noqa: S603
            render_command,
            cwd=str(render_dir),
            check=False,
            shell=False,
            env=build_xyzrender_subprocess_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        def _failure_response(detail: str) -> JsonResponse:
            detail_text = (detail or "").strip()
            if detail_text:
                detail_text = detail_text[-500:]
            detail_lower = detail_text.lower()
            hints: list[str] = []
            if render_input is not None and render_input.exists():
                keep_files.add(render_input)
                hints.append(f"Preserved generated XYZ for inspection: {render_input}")
                hints.append(f"Generated XYZ summary: {_summarize_xyz_input(render_input)}")
            if (
                "no module named xyzrender.__main__" in detail_lower
                or "cannot be directly executed" in detail_lower
            ):
                hints.append(
                    "Installed xyzrender package has no runnable module entrypoint; "
                    "use the xyzrender executable (this is now preferred automatically)."
                )
            elif "no module named xyzrender" in detail_lower:
                hints.append(
                    "Install xyzrender in the active Python environment "
                    "(for example: python -m pip install xyzrender)."
                )
            if is_xyzrender_cairo_error(detail_text):
                hints.append(cairo_runtime_install_hint())
            if command_warning:
                hints.append(command_warning)
            return JsonResponse(
                {
                    "ok": False,
                    "error": (
                        f"xyzrender failed with exit code {completed.returncode}."
                        + (f" Details: {detail_text}" if detail_text else "")
                        + (f" {' '.join(hints)}" if hints else "")
                    ),
                },
                status=400,
            )

        cairo_runtime_missing = False
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()
            if render_gif and is_xyzrender_cairo_error(detail):
                cairo_runtime_missing = True
                render_notes.append(
                    "GIF generation skipped because the Cairo runtime is unavailable. "
                    f"{cairo_runtime_install_hint()}"
                )
                if not output_svg.exists():
                    svg_only_command = [
                        *command,
                        render_input.name,
                        "-o",
                        output_svg.name,
                    ]
                    svg_only = subprocess.run(  # noqa: S603
                        svg_only_command,
                        cwd=str(render_dir),
                        check=False,
                        shell=False,
                        env=build_xyzrender_subprocess_env(),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    if svg_only.returncode != 0:
                        retry_detail = (svg_only.stderr or svg_only.stdout or "").strip()
                        combined_detail = detail
                        if retry_detail:
                            combined_detail = (
                                f"{combined_detail} SVG-only retry failed: {retry_detail}"
                                if combined_detail
                                else f"SVG-only retry failed: {retry_detail}"
                            )
                        return _failure_response(combined_detail)
            else:
                return _failure_response(detail)

        output_gif = expected_gif if render_gif and expected_gif.exists() else None
        if render_gif and output_gif is None and not cairo_runtime_missing:
            new_gif_candidates = sorted(
                (
                    path
                    for path in render_dir.glob("*.gif")
                    if path.resolve() not in preexisting_gifs
                ),
                key=lambda path: (path.stat().st_mtime_ns if path.exists() else 0, str(path)),
                reverse=True,
            )
            if new_gif_candidates:
                output_gif = new_gif_candidates[0]
        if render_gif and output_gif is not None and output_gif.resolve() != expected_gif.resolve():
            shutil.copyfile(output_gif, expected_gif)
            cleanup_files.append(output_gif)
            output_gif = expected_gif

        if render_svg and not output_svg.exists():
            return JsonResponse(
                {"ok": False, "error": "xyzrender finished but did not produce the expected SVG output."},
                status=400,
            )
        if render_gif and not cairo_runtime_missing and (output_gif is None or not output_gif.exists()):
            return JsonResponse(
                {"ok": False, "error": "xyzrender finished but did not produce a GIF output."},
                status=400,
            )
        if render_svg and fixed_scale_render_input is not None and render_input is not None:
            svg_clean = subprocess.run(  # noqa: S603
                [
                    *command,
                    render_input.name,
                    "-o",
                    output_svg.name,
                ],
                cwd=str(render_dir),
                check=False,
                shell=False,
                env=build_xyzrender_subprocess_env(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if svg_clean.returncode != 0:
                detail = (svg_clean.stderr or svg_clean.stdout or "").strip()
                if detail:
                    render_notes.append(
                        f"Clean SVG refresh failed after fixed-scale GIF render: {detail[-300:]}"
                    )
        gif_canvas_size = _xyzrender_gif_canvas_size()
        gif_canvas_normalized = False
        if render_gif and output_gif is not None and output_gif.exists():
            gif_canvas_normalized, gif_canvas_note = normalize_gif_canvas(
                output_gif,
                size=gif_canvas_size,
            )
            if gif_canvas_note:
                render_notes.append(gif_canvas_note)
            if (
                gif_canvas_size is not None
                and not gif_canvas_normalized
                and gif_canvas_note.startswith("Pillow unavailable")
            ):
                return JsonResponse(
                    {
                        "ok": False,
                        "error": (
                            "GIF canvas normalization is enabled but Pillow is not installed "
                            "in the Reaction Hub environment. Install/update reaction_hub "
                            "requirements and render the GIF again."
                        ),
                    },
                    status=500,
                )

        generated: list[Artifact] = []
        image_paths: list[Path | None] = []
        if render_svg:
            image_paths.append(output_svg)
        if render_gif:
            image_paths.append(output_gif)
        for image_path in image_paths:
            if image_path is None:
                continue
            is_gif_output = image_path.suffix.lower() == ".gif"
            stored_path = _artifact_stored_path(run, image_path)
            if requested_scan_ts_mode or requested_scan_endpoint_frame_mode:
                metadata = {
                    "source": "xyzrender",
                    "generated_by": "workflow_render_action",
                    "source_artifact_id": artifact.pk,
                    "source_artifact_label": str(artifact.label),
                    "render_output": "gif" if is_gif_output else "svg",
                    "scan_endpoint_render_mode": requested_scan_endpoint_frame_mode or "transition_state",
                    "trajectory_source": "scan_endpoints_db",
                }
                if is_gif_output:
                    metadata["gif_rotation"] = raw_rotation
                    metadata["rot_frames"] = int(rot_frames)
                    if gif_canvas_size is not None:
                        metadata["canvas_width_px"] = int(gif_canvas_size[0])
                        metadata["canvas_height_px"] = int(gif_canvas_size[1])
                        metadata["canvas_normalized"] = bool(gif_canvas_normalized)
            elif use_trajectory_gif:
                metadata = {
                    "source": "xyzrender",
                    "generated_by": "workflow_render_action",
                    "source_artifact_id": artifact.pk,
                    "source_artifact_label": str(artifact.label),
                    "render_output": "gif" if is_gif_output else "svg",
                    "gif_mode": "trajectory",
                    "trajectory_pattern": "ping_pong",
                }
                if use_full_artifact_trajectory:
                    metadata["trajectory_source"] = "artifact_frames"
                elif is_scan_endpoints_db:
                    metadata["trajectory_source"] = "scan_endpoints_db"
                    metadata["xyzrender_gif_flag"] = "--gif-trj"
                    metadata["fixed_scale_anchors"] = True
                    metadata["fixed_scale_anchor_symbol"] = _xyzrender_fixed_scale_anchor_symbol()
                if is_gif_output and gif_canvas_size is not None:
                    metadata["canvas_width_px"] = int(gif_canvas_size[0])
                    metadata["canvas_height_px"] = int(gif_canvas_size[1])
                    metadata["canvas_normalized"] = bool(gif_canvas_normalized)
            else:
                metadata = {
                    "source": "xyzrender",
                    "generated_by": "workflow_render_action",
                    "source_artifact_id": artifact.pk,
                    "source_artifact_label": str(artifact.label),
                    "render_output": "gif" if is_gif_output else "svg",
                }
                if is_gif_output:
                    metadata["gif_rotation"] = raw_rotation
                    metadata["rot_frames"] = int(rot_frames)
                    if gif_canvas_size is not None:
                        metadata["canvas_width_px"] = int(gif_canvas_size[0])
                        metadata["canvas_height_px"] = int(gif_canvas_size[1])
                        metadata["canvas_normalized"] = bool(gif_canvas_normalized)
            if not is_gif_output:
                metadata["svg_orientation"] = raw_rotation
            generated_artifact, _ = Artifact.objects.update_or_create(
                workflow_run=run,
                path=stored_path,
                defaults={
                    "kind": "image",
                    "label": f"render:{image_path.name}",
                    "metadata": metadata,
                },
            )
            generated.append(generated_artifact)
    except Exception as exc:
        return JsonResponse(
            {"ok": False, "error": f"Failed to render artifact with xyzrender: {exc}"},
            status=400,
        )
    finally:
        for temp_path in cleanup_files:
            if temp_path in keep_files:
                continue
            try:
                if temp_path.exists():
                    temp_path.unlink()
            except OSError:
                continue

    return JsonResponse(
        {
            "ok": True,
            "artifacts": [
                {
                    "id": int(item.pk),
                    "label": str(item.label or ""),
                    "path": str(item.path or ""),
                    "content_url": reverse("artifact-content", kwargs={"pk": item.pk}),
                    "file_url": reverse("artifact-file", kwargs={"pk": item.pk}),
                }
                for item in generated
            ],
            "message": (
                f"Rendered {len(generated)} artifact(s) for '{artifact.label}' "
                + (
                    f"as a {requested_scan_endpoint_frame_mode} endpoint image with rotation '{raw_rotation}'."
                    if requested_scan_endpoint_frame_mode
                    else f"as a transition-state image with rotation '{raw_rotation}'."
                    if requested_scan_ts_mode
                    else (
                        "as a ping-pong trajectory."
                        if use_trajectory_gif
                        else f"with rotation '{raw_rotation}'."
                    )
                )
                + (f" {' '.join(render_notes)}" if render_notes else "")
                + (f" {command_warning}" if command_warning else "")
            ).strip(),
            "generated_artifact_ids": [item.pk for item in generated],
        }
    )


def _fetch_remote_artifact_bytes(client, remote_path: str) -> bytes:
    import base64
    import gzip
    import shlex

    start_marker = "__REACTION_HUB_B64_START__"
    end_marker = "__REACTION_HUB_B64_END__"
    path_q = shlex.quote(str(remote_path))
    cmd_gzip = (
        f"printf '{start_marker}\\n'; "
        f"gzip -c < {path_q} | base64 | tr -d '\\n'; "
        f"printf '\\n{end_marker}\\n'"
    )
    try:
        out = str(client.exec_checked(cmd_gzip))
        if start_marker in out and end_marker in out:
            b64_data = out.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
            if b64_data:
                compressed = base64.b64decode(b64_data.encode("ascii"), validate=False)
                return gzip.decompress(compressed)
            return b""
    except Exception:
        pass

    cmd_fallback = (
        f"printf '{start_marker}\\n'; "
        f"base64 < {path_q} | tr -d '\\n'; "
        f"printf '\\n{end_marker}\\n'"
    )
    out = str(client.exec_checked(cmd_fallback))
    if start_marker in out and end_marker in out:
        b64_data = out.split(start_marker, 1)[1].split(end_marker, 1)[0].strip()
        if b64_data:
            return base64.b64decode(b64_data.encode("ascii"), validate=False)
        return b""
    raise RuntimeError(f"Failed to read remote file content: {remote_path}")


@require_POST
def workflow_ingest_artifact_view(request, pk: int, artifact_pk: int):
    run = get_object_or_404(WorkflowRun, pk=pk)
    artifact = get_object_or_404(Artifact, pk=artifact_pk, workflow_run=run)

    try:
        result = ingest_artifact_from_hpc_task.apply_async(
            args=[int(pk), int(artifact_pk)],
            queue="submissions",
        )
    except Exception as exc:
        return JsonResponse(
            {"ok": False, "error": f"Failed to queue artifact ingestion task: {exc}"},
            status=500,
        )
    return JsonResponse(
        {
            "ok": True,
            "queued": True,
            "task_id": result.id,
            "message": f"Artifact ingestion queued as Celery task {result.id}.",
        }
    )


def _workflow_ingest_artifact_sync_response(request, pk: int, artifact_pk: int):
    import shlex
    from core.openssh import OpenSSHClient
    from jobs.models import HPCSystem

    run = get_object_or_404(WorkflowRun, pk=pk)
    artifact = get_object_or_404(Artifact, pk=artifact_pk, workflow_run=run)

    target_cache_path = _inferred_local_cache_path(run, artifact)
    metadata = dict(artifact.metadata or {})

    existing_cache_str = str(metadata.get("local_cache_path") or "").strip()
    if existing_cache_str:
        existing_path = _runtime_path(existing_cache_str)
        if existing_path.exists() and existing_path.is_file():
            if metadata.get("remote_only"):
                metadata["remote_only"] = False
                metadata["downloaded_from_hpc"] = True
                artifact.metadata = metadata
                artifact.save(update_fields=["metadata", "updated_at"])
            return JsonResponse(
                {
                    "ok": True,
                    "already_cached": True,
                    "artifact_id": artifact.pk,
                    "local_cache_path": str(existing_path),
                    "message": f"Artifact '{artifact.label}' is already available locally.",
                }
            )

    remote_path = str(metadata.get("remote_path") or "").strip()
    if not remote_path:
        remote_path = str(artifact.path or "").strip()
    if run.working_directory:
        workdir = str(run.working_directory).strip().rstrip("/")
        if not remote_path.startswith("/") and not remote_path.startswith("~"):
            remote_path = f"{workdir}/{remote_path.lstrip('/')}"
    remote_path = normalize_run_path(remote_path)

    system = None
    if artifact.job and artifact.job.hpc_system:
        system = artifact.job.hpc_system
    if not system and metadata.get("source_scheduler_job_id"):
        job = run.jobs.filter(scheduler_job_id=metadata.get("source_scheduler_job_id")).first()
        if job and job.hpc_system:
            system = job.hpc_system
    if not system and metadata.get("scheduler_job_id"):
        job = run.jobs.filter(scheduler_job_id=metadata.get("scheduler_job_id")).first()
        if job and job.hpc_system:
            system = job.hpc_system
    if not system:
        job = run.jobs.exclude(hpc_system=None).order_by("-created_at").first()
        if job and job.hpc_system:
            system = job.hpc_system
    if not system:
        system = HPCSystem.objects.filter(is_active=True).first()
    if not system:
        return JsonResponse(
            {"ok": False, "error": "No active HPC system configured."},
            status=400,
        )

    try:
        with OpenSSHClient(
            host=system.host,
            port=system.port,
            username=system.username,
            ssh_key_path=system.ssh_key_path,
            timeout=120,
            force_tty=getattr(settings, "REACTION_HUB_SSH_FORCE_TTY", False),
            batch_mode=getattr(settings, "REACTION_HUB_SSH_BATCH_MODE", True),
            use_clean_bash=getattr(settings, "REACTION_HUB_SSH_CLEAN_BASH", False),
            bash_path=getattr(settings, "REACTION_HUB_SSH_BASH_PATH", "bash"),
        ) as client:
            path_q = shlex.quote(remote_path)
            exists_out = client.exec_checked(f"if [ -f {path_q} ]; then echo 1; else echo 0; fi").strip()
            if exists_out != "1":
                candidate_alt = f"{str(run.working_directory).rstrip('/')}/{Path(remote_path).name}"
                alt_q = shlex.quote(candidate_alt)
                if client.exec_checked(f"if [ -f {alt_q} ]; then echo 1; else echo 0; fi").strip() == "1":
                    remote_path = candidate_alt
                else:
                    return JsonResponse(
                        {"ok": False, "error": f"Remote file does not exist on {system.name}: {remote_path}"},
                        status=404,
                    )
            content = _fetch_remote_artifact_bytes(client, remote_path)
    except Exception as exc:
        return JsonResponse(
            {"ok": False, "error": f"Failed to retrieve artifact from {system.name}: {exc}"},
            status=500,
        )

    try:
        target_cache_path.parent.mkdir(parents=True, exist_ok=True)
        target_cache_path.write_bytes(content)
    except OSError as exc:
        return JsonResponse(
            {"ok": False, "error": f"Failed to write local cache file: {exc}"},
            status=500,
        )

    metadata["local_cache_path"] = str(target_cache_path)
    metadata["downloaded_from_hpc"] = True
    metadata["remote_only"] = False
    metadata["remote_path"] = remote_path
    artifact.metadata = metadata
    artifact.save(update_fields=["metadata", "updated_at"])

    return JsonResponse(
        {
            "ok": True,
            "artifact_id": artifact.pk,
            "local_cache_path": str(target_cache_path),
            "bytes": len(content),
            "message": f"Successfully ingested '{artifact.label}' ({len(content)} bytes) from {system.name}.",
        }
    )
