from __future__ import annotations

from datetime import datetime
import os
import re
import shlex
import textwrap
import uuid
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from core.openssh import OpenSSHClient
from references.models import CalculationSettingsProfile
from results.models import Artifact, ArtifactKind
from workflows.identity import canonical_run_path, normalize_run_path
from workflows.models import WorkflowDefinition, WorkflowRun

from .models import HPCSystem, JobRecord, JobStatus

_SBATCH_JOB_ID_RE = re.compile(r"Submitted\s+batch\s+job\s+(\d+)")
_FALCON_DEFAULT_SLURM_MEMORY = "720G"
_NWCHEM_HSE06_XC = "xpbe96 1.0 xcampbe96 -0.25 cpbe96 1.0 srhfexch 0.25"
_NWCHEM_HSE06_CAM = "0.11 cam_alpha 0.0 cam_beta 1.0"


def _active_gpu_job_count(ssh_client: Any, *, username: str, partition: str = "gpu") -> int:
    output = _exec_checked(
        ssh_client,
        f"squeue --noheader --user {_q(username)} --partition {_q(partition)} --format=%i",
    )
    return sum(1 for line in output.splitlines() if line.strip())


def _active_qos_job_count(ssh_client: Any, *, username: str, qos: str, partition: str = "") -> int:
    target_qos = str(qos or "").strip().casefold()
    target_part = str(partition or "").strip().casefold()
    if target_part in {"htc_genoa", "dev"}:
        output = _exec_checked(
            ssh_client,
            f"squeue --noheader --user {_q(username)} --partition {_q(partition)} --format=%i",
        )
        return sum(1 for line in output.splitlines() if line.strip())
    if target_qos in {"gpu", "gpu-exc", "gpu-shd"} or target_part == "gpu":
        return _active_gpu_job_count(ssh_client, username=username, partition="gpu")
    output = _exec_checked(
        ssh_client,
        f"squeue --noheader --user {_q(username)} --format=%q",
    )
    return sum(1 for line in output.splitlines() if line.strip().casefold() == target_qos)


def _active_user_job_count(ssh_client: Any, *, username: str) -> int:
    output = _exec_checked(
        ssh_client,
        f"squeue --noheader --user {_q(username)} --format=%i",
    )
    return sum(1 for line in output.splitlines() if line.strip())


def _active_reservation_node_count(
    ssh_client: Any,
    *,
    reservation: str,
    username: str,
) -> int:
    output = _exec_checked(
        ssh_client,
        f"squeue --noheader --reservation {_q(reservation)} --user {_q(username)} --format=%D",
    )
    return sum(int(line.strip()) for line in output.splitlines() if line.strip())


def _reservation_has_capacity(
    ssh_client: Any,
    *,
    reservation: str,
    username: str,
    requested_nodes: int,
    node_quota: int,
) -> bool:
    return (
        _active_reservation_node_count(
            ssh_client,
            reservation=reservation,
            username=username,
        )
        + requested_nodes
        <= node_quota
    )


def _submission_queue_limit(qos: str, partition: str = "") -> int:
    part = str(partition or "").strip().casefold()
    queue = str(qos or "").strip().casefold()
    if part == "htc_genoa" or queue == "htc_genoa":
        return min(10, max(1, int(getattr(settings, "REACTION_HUB_FALCON_HTC_QUEUE_MAX_JOBS", 10))))
    if part == "dev" or queue == "dev":
        return min(2, max(1, int(getattr(settings, "REACTION_HUB_FALCON_DEV_QUEUE_MAX_JOBS", 2))))
    if queue == "short":
        return min(4, max(1, int(getattr(settings, "REACTION_HUB_SHORT_QUEUE_MAX_JOBS", 4))))
    if queue == "serial":
        return min(8, max(1, int(getattr(settings, "REACTION_HUB_SERIAL_QUEUE_MAX_JOBS", 8))))
    if queue == "long":
        return min(16, max(1, int(getattr(settings, "REACTION_HUB_LONG_QUEUE_MAX_JOBS", 16))))
    if queue == "gpu-exc":
        return min(2, max(1, int(getattr(settings, "REACTION_HUB_GPU_QUEUE_MAX_JOBS", 2))))
    if queue == "taskfarm":
        return min(128, max(1, int(getattr(settings, "REACTION_HUB_TASKFARM_QUEUE_MAX_JOBS", 128))))
    if queue == "standard":
        return min(64, max(1, int(getattr(settings, "REACTION_HUB_STANDARD_QUEUE_MAX_JOBS", 64))))
    return 0


@dataclass(frozen=True)
class SubmissionResult:
    workflow_run_id: int | None
    job_record_id: int | None
    scheduler_job_id: str
    remote_run_dir: str


def _structure_artifact_rel_path_for_mode(_run_mode: str) -> str:
    return "input_structure.extxyz"


def _positive_int(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return int(default)
    return parsed if parsed > 0 else int(default)


def _atom_count_from_xyz_text(structure_xyz: str) -> int | None:
    lines = [line.strip() for line in str(structure_xyz or "").splitlines() if line.strip()]
    if not lines:
        return None
    try:
        atom_count = int(lines[0])
    except ValueError:
        return None
    return atom_count if atom_count > 0 else None


def _scaled_final_tddft_nroots(structure_xyz: str, *, default: int = 50) -> int:
    atom_count = _atom_count_from_xyz_text(structure_xyz)
    if atom_count is None:
        return int(default)
    roots = max(int(default), 8 * atom_count - 54)
    return ((roots + 4) // 5) * 5


def _optional_positive_int(value: Any, *, default: int | None = None) -> int | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    if text.lower() in {"none", "null"}:
        return None
    try:
        parsed = int(text)
    except (TypeError, ValueError):
        return default
    if parsed <= 0:
        return None
    return int(parsed)


def _slurm_memory_mb(memory: str) -> int | None:
    match = re.fullmatch(r"\s*(\d+)\s*([GMK]?)(?:B)?\s*", str(memory or ""), re.IGNORECASE)
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2).upper()
    if unit == "G":
        return value * 1024
    if unit == "K":
        return max(1, value // 1024)
    return value


def _normalize_xc_for_workflow(xc_functional: str) -> str:
    key = str(xc_functional or "").strip().lower().replace("_", "-")
    if key in {"cam-b3lyp", "camb3lyp"}:
        return "cam-b3lyp"
    if key == "pbe0":
        return "pbe0"
    raise ValueError("Unsupported xc_functional for submit flow. Use CAM-B3LYP or PBE0.")


def _normalize_basis_for_workflow(basis_set: str) -> str:
    key = str(basis_set or "").strip().lower().replace("_", "-")
    if "lanl2dz" in key:
        return "lanl2dz"
    if key in {"def2-svp", "def2svp", "def2"}:
        return "def2-svp"
    raise ValueError("Unsupported basis_set for submit flow. Use LANL2DZ or def2-svp.")


def _normalize_basis_for_backend(basis_set: str, qm_backend: str) -> str:
    if str(qm_backend or "").strip().lower() == "nwchem":
        text = str(basis_set or "").strip()
        return text or "def2-SVP"
    return _normalize_basis_for_workflow(basis_set)


def _normalize_solvent_for_workflow(solvent: str) -> str | None:
    text = str(solvent or "").strip()
    return text or None


def _normalize_xc_for_backend(xc_functional: str, qm_backend: str) -> str:
    if str(qm_backend or "").strip().lower() == "nwchem":
        text = str(xc_functional or "").strip() or "PBE0"
        return _NWCHEM_HSE06_XC if text.lower().replace("_", "-") == "hse06" else text
    return _normalize_xc_for_workflow(xc_functional)


def _default_nwchem_cam(xc_functional: str) -> str | None:
    if xc_functional == _NWCHEM_HSE06_XC:
        return _NWCHEM_HSE06_CAM
    if str(xc_functional).strip().lower().replace("_", "-") in {"cam-b3lyp", "camb3lyp"}:
        return "0.11 cam_alpha 0.0 cam_beta 1.0"
    return None


def _normalize_active_learning_optimizer_for_workflow(active_learning_optimizer: str) -> str:
    key = str(active_learning_optimizer or "").strip().lower().replace("-", "").replace("_", "")
    if not key:
        return "bfgs"
    if key == "bfgs":
        return "bfgs"
    if key in {"fire2", "fire"}:
        return "fire2"
    raise ValueError("Unsupported active_learning_optimizer for submit flow. Use BFGS or FIRE2.")


def _settings_profile_backend_settings(
    settings_profile: CalculationSettingsProfile,
    qm_backend: str,
) -> dict[str, Any]:
    metadata = settings_profile.metadata if isinstance(settings_profile.metadata, dict) else {}
    backend_key = str(qm_backend or "").strip().lower()
    backend_settings = metadata.get(backend_key)
    if isinstance(backend_settings, dict):
        return dict(backend_settings)
    return {}


def _module_name_key(module_name: str) -> str:
    return str(module_name or "").strip().lower()


def _is_orca_module(module_name: str) -> bool:
    key = _module_name_key(module_name)
    return key == "orca" or key.startswith("orca/")


def _is_nwchem_module(module_name: str) -> bool:
    key = _module_name_key(module_name)
    return key == "nwchem" or key.startswith("nwchem/")


def _append_module_once(module_loads: list[str], module_name: str) -> None:
    key = _module_name_key(module_name)
    if not key:
        return
    if any(_module_name_key(existing) == key for existing in module_loads):
        return
    module_loads.append(module_name)


def _module_path_from_code_path(code_path: str) -> str:
    path = str(code_path or "").strip()
    if not path.endswith(".py"):
        raise ValueError("Workflow code_path must be a Python file path ending with .py.")
    module_path = path[:-3].replace("/", ".").replace("\\", ".")
    if module_path.startswith("cplas."):
        return "cats." + module_path[len("cplas.") :]
    return module_path


def _module_name_is_orca(module_name: str) -> bool:
    segments = [segment for segment in str(module_name or "").strip().lower().split("/") if segment]
    return any(segment == "orca" for segment in segments)


def _module_name_is_tddft_mlneb(module_name: str) -> bool:
    normalized = str(module_name or "").strip().lower().replace("\\", "/")
    return normalized.endswith("tddft_mlneb_workflow") or "tddft_mlneb_workflow" in normalized


def _scratch_base_has_user_token(scratch_base: str) -> bool:
    text = str(scratch_base or "")
    return any(token in text for token in ("$USER", "${USER}", "{username}"))


def _temp_directory_expr_from_scratch_base(scratch_base: str) -> str:
    base = str(scratch_base or "").strip().rstrip("/")
    if not base:
        return "/tmp/$USER/$SLURM_JOB_ID"
    if _scratch_base_has_user_token(base):
        return f"{base}/$SLURM_JOB_ID"
    return f"{base}/$USER/$SLURM_JOB_ID"


def _resolve_remote_workdir(
    *,
    scratch_base: str,
    username: str,
    scheduler_job_id: str,
) -> str:
    base = str(scratch_base or "").strip().rstrip("/")
    user = str(username or "").strip()
    job_id = str(scheduler_job_id or "").strip()
    if not base or not user or not job_id:
        return ""

    if _scratch_base_has_user_token(base):
        base = (
            base.replace("${USER}", user)
            .replace("$USER", user)
            .replace("{username}", user)
        )
        return f"{base}/{job_id}"

    return f"{base}/{user}/{job_id}"


def _build_wrapper_script(
    *,
    workflow_module: str,
    run_mode: str,
    hpc_name: str,
    n_iter: int,
    use_tddft_optimisation: bool,
    fmax: float,
    active_learning_optimizer: str,
    qm_backend: str,
    qm_backend_settings: dict[str, Any] | None,
    target_wavelength_nm: float,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    spin_multiplicity: int | None,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
    al_steps_per_iter: int,
    max_al_steps_per_iter: int | None,
    system_charge: int = 0,
    scan_config: dict[str, Any] | None = None,
    single_constraints_config: dict[str, Any] | None = None,
    charge_transfer_config: dict[str, Any] | None = None,
    neb_config: dict[str, Any] | None = None,
    coadsorbate_config: dict[str, Any] | None = None,
    alloy_config: dict[str, Any] | None = None,
    workflow_tasks_per_node: int | None = None,
) -> str:
    qm_backend = str(qm_backend or "orca").strip().lower() or "orca"
    qm_backend_label = qm_backend.upper()
    qm_backend_settings = dict(qm_backend_settings or {})
    default_workflow_tasks_per_node = (
        int(workflow_tasks_per_node)
        if workflow_tasks_per_node is not None
        else (96 if int(tasks_per_node) * int(cpus_per_task) >= 192 else int(tasks_per_node))
    )
    solvent_expr = "None" if solvent is None else repr(solvent)
    spin_expr = "None" if spin_multiplicity is None else str(int(spin_multiplicity))
    charge_expr = str(int(system_charge))
    max_al_steps_expr = (
        "None" if max_al_steps_per_iter is None else str(int(max_al_steps_per_iter))
    )
    if run_mode == "charge_transfer":
        cfg = dict(charge_transfer_config or {})
        source_working_directory = str(cfg.get("source_working_directory") or "").strip()
        orca_basename = str(cfg.get("orca_basename") or "orca").strip() or "orca"
        fragments_0based = dict(cfg.get("fragments_0based") or {})
        fragments_1based = dict(cfg.get("fragments_1based") or {})
        fragment_labels = dict(cfg.get("fragment_labels") or {})
        nroots = _positive_int(cfg.get("nroots"), default=400)
        ground_state_mulliken_only = bool(cfg.get("ground_state_mulliken_only"))
        density_analysis_only = bool(cfg.get("density_analysis_only"))
        ifct_only = bool(cfg.get("ifct_only"))
        return textwrap.dedent(
            f"""\
            import json
            import os
            import shutil
            import subprocess
            from pathlib import Path

            from cats.multiwfn.density_analysis import generate_excitation_density_analysis
            from cats.multiwfn.plot_uvvis import get_full_ct_spectrum
            from cats.multiwfn.tools import mulliken_population_routine
            from cats.orca.legacy_catalog import _parse_ct_multiple, _parse_ifct_files
            from cats.orca.routines.orca_uv_extractor import extract_uv_transitions

            source_root = Path({source_working_directory!r})
            ground_state_mulliken_only = {ground_state_mulliken_only!r}
            density_analysis_only = {density_analysis_only!r}
            ifct_only = {ifct_only!r}
            density_analysis_config = {dict(cfg.get('density_analysis') or {})!r}
            if density_analysis_only and density_analysis_config.get("nto_population_only"):
                from cats.multiwfn.orbital_composition import mulliken_orbital_composition
                output_dir = Path("density_analysis")
                output_dir.mkdir(exist_ok=True)
                selector = density_analysis_config.get("hole_selector", "h")
                spin = {{"h":"default", "ha":"alpha", "hb":"beta"}}[selector]
                states = density_analysis_config.get("excitation_numbers") or []
                if not states:
                    raise ValueError("Population-only analysis requires explicit states")
                populations = []
                for state in states:
                    source_nto = source_root / "density_analysis" / f"state_{{int(state):03d}}_nto.mwfn"
                    if not source_nto.is_file():
                        raise FileNotFoundError(source_nto)
                    local_nto = output_dir / source_nto.name
                    shutil.copy2(source_nto, local_nto)
                    pair = {{"excitation_number":int(state), "spin":spin}}
                    for role in ("hole", "electron"):
                        pair[role] = mulliken_orbital_composition(
                            local_nto, spin=spin, role=role,
                            output_prefix=str(output_dir / f"state_{{int(state):03d}}_{{spin}}_{{role}}_mulliken"),
                        )
                    populations.append(pair)
                summary = {{
                    "status":"completed", "run_type":"density_analysis", "tags":["nto_orbital_mulliken"],
                    "source_run_id":{cfg.get('source_run_id')!r},
                    "source_run_uid":{str(cfg.get('source_run_uid') or '')!r},
                    "source_working_directory":str(source_root),
                    "charge_transfer":{{"analysis_scope":"nto_orbital_mulliken", "density_analysis":{{"states":populations}}}},
                    "artifacts":sorted(str(path.relative_to(Path.cwd())) for path in Path.cwd().rglob("*") if path.is_file()),
                }}
                Path("reaction_hub_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
                raise SystemExit(0)
            if density_analysis_only and density_analysis_config.get("mep_state_tracking"):
                from mep_state_tracker_runtime import ContinuousMEPTracker
                image_dirs = density_analysis_config.get("image_directories") or []
                if not image_dirs:
                    raise ValueError("mep_state_tracking requires image_directories")
                tracker = ContinuousMEPTracker(
                    image_dirs,
                    root_state=int(density_analysis_config.get("root_state") or 25),
                    energy_window_ev=float(density_analysis_config.get("energy_window_ev") or 0.35),
                    output_directory=Path("density_analysis"),
                    density_workers=int(density_analysis_config.get("density_workers") or 4),
                    nto_max_pairs=int(density_analysis_config.get("nto_max_pairs") or 3),
                    nto_target_coverage=float(density_analysis_config.get("nto_target_coverage") or 0.8),
                    grid_quality=int(density_analysis_config.get("grid_quality") or 2),
                )
                tracking_summary = tracker.run_tracking()
                summary = {{
                    "status": "completed",
                    "run_type": "mep_state_tracking",
                    "tags": ["mep_state_tracking"],
                    "source_run_id": {cfg.get('source_run_id')!r},
                    "source_run_uid": {str(cfg.get('source_run_uid') or '')!r},
                    "source_working_directory": str(source_root),
                    "charge_transfer": {{
                        "analysis_scope": "mep_state_tracking",
                        "tracking_results": tracking_summary,
                    }},
                    "artifacts": sorted(str(path.relative_to(Path.cwd())) for path in Path.cwd().rglob("*") if path.is_file()),
                }}
                Path("reaction_hub_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                raise SystemExit(0)
            if density_analysis_only and density_analysis_config.get("ts_vibrations_gradient"):
                from ts_vibrations_gradient_runtime import run_ts_vibrations_gradient
                ts_cfg = dict(density_analysis_config.get("ts_vibrations_gradient") or {{}})
                run_ts_vibrations_gradient(
                    structure_path="input_structure.extxyz",
                    output_dir=Path.cwd(),
                    iroot=int(ts_cfg.get("iroot") or 31),
                    nroots=int(ts_cfg.get("nroots") or 40),
                    spin_multiplicity=int(ts_cfg.get("spin_multiplicity")) if ts_cfg.get("spin_multiplicity") is not None else None,
                    system_charge=int(ts_cfg.get("system_charge") or 0),
                    nprocs=int(os.environ.get("WORKFLOW_TASKS_PER_NODE", 48)),
                    memory_per_cpu_mb=6000 if ts_cfg.get("relax_endpoints") else 10500,
                    hpc={str(hpc_name).lower()!r},
                    ground_state_only=bool(ts_cfg.get("ground_state_only", False)),
                    read_initial_wavefunction=bool(ts_cfg.get("initial_wavefunction_artifact_id")),
                    relax_endpoints=bool(ts_cfg.get("relax_endpoints", False)),
                    displacement_A=float(ts_cfg.get("displacement_A", 0.1)),
                    relaxation_fmax=float(ts_cfg.get("relaxation_fmax", 0.02)),
                    relaxation_steps=int(ts_cfg.get("relaxation_steps", 300)),
                    reaction_bond=ts_cfg.get("reaction_bond"),
                    product_bond_change=int(ts_cfg.get("product_bond_change", -1)),
                    basis_set={basis_keyword!r},
                    xc_functional={xc_functional!r},
                )
                raise SystemExit(0)
            if not ground_state_mulliken_only:
                source_candidates = (
                    source_root if source_root.name == "final_tddft_spectrum" else source_root / "final_tddft_spectrum",
                    source_root / "DFT" / "final_tddft_spectrum",
                    source_root.parent / "final_tddft_spectrum",
                    source_root.parent.parent / "DFT" / "final_tddft_spectrum",
                    source_root,
                )
            else:
                source_candidates = (
                    source_root / "DFT" / "final_tddft_spectrum",
                    source_root,
                )
            source_directory = next(
                (
                    directory
                    for directory in source_candidates
                    if (directory / {f'{orca_basename}.out'!r}).is_file()
                    or (
                        ground_state_mulliken_only
                        and any(
                            (directory / name).is_file()
                            for name in (
                                {f'{orca_basename}.molden.input'!r},
                                {f'{orca_basename}.molden'!r},
                                {f'{orca_basename}.gbw'!r},
                                "initial.gbw",
                            )
                        )
                    )
                ),
                None,
            )
            if source_directory is None:
                searched = ", ".join(
                    str(directory / {f'{orca_basename}.out'!r})
                    for directory in source_candidates
                )
                raise FileNotFoundError(f"Missing source ORCA output; searched: {{searched}}")
            source_output = source_directory / {f'{orca_basename}.out'!r}
            if source_output.is_file():
                shutil.copy2(source_output, "orca.out")

            molden_input = Path("orca.molden.input")
            source_molden_candidates = (
                source_directory / {f'{orca_basename}.molden.input'!r},
                source_directory / {f'{orca_basename}.molden'!r},
            )
            source_molden = next((path for path in source_molden_candidates if path.is_file()), None)
            if source_molden is not None:
                shutil.copy2(source_molden, molden_input)
            else:
                source_gbw = next(
                    (
                        path
                        for path in (
                            source_directory / {f'{orca_basename}.gbw'!r},
                            source_directory / "initial.gbw",
                        )
                        if path.is_file()
                    ),
                    source_directory / {f'{orca_basename}.gbw'!r},
                )
                if not source_gbw.is_file():
                    raise FileNotFoundError(
                        f"Missing source Molden file and ORCA wavefunction: {{source_gbw}}"
                    )
                shutil.copy2(source_gbw, "orca.gbw")
                orca_2mkl = shutil.which("orca_2mkl")
                if not orca_2mkl:
                    orcadir = str(os.environ.get("ORCADIR") or "").strip()
                    candidate = Path(orcadir) / "orca_2mkl" if orcadir else None
                    if candidate is not None and candidate.is_file():
                        orca_2mkl = str(candidate)
                if not orca_2mkl:
                    raise FileNotFoundError("orca_2mkl was not found in PATH or ORCADIR.")
                subprocess.run([orca_2mkl, "orca", "-molden"], check=True)

            fragments = {fragments_1based!r}
            if density_analysis_only and density_analysis_config.get("ground_state_orbitals_only"):
                from cats.multiwfn.nto import ground_state_orbital_cubes
                orbital_exports = ground_state_orbital_cubes(str(molden_input))
                summary = {{
                    "status": "completed", "run_type": "density_analysis",
                    "tags": ["ground_state_orbitals"],
                    "source_run_id": {cfg.get('source_run_id')!r},
                    "source_run_uid": {str(cfg.get('source_run_uid') or '')!r},
                    "source_working_directory": str(source_directory),
                    "charge_transfer": {{"analysis_scope": "ground_state_orbitals", "density_analysis": orbital_exports}},
                    "artifacts": sorted(str(path.relative_to(Path.cwd())) for path in Path.cwd().rglob("*") if path.is_file()),
                }}
                Path("reaction_hub_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
                raise SystemExit(0)
            ground_state_mulliken = {{}}
            excited_state_mulliken = []
            if not density_analysis_only:
                ground_state_mulliken = mulliken_population_routine(
                    str(molden_input),
                    fragments_0based={fragments_0based!r},
                )
            for state in {list(cfg.get('mulliken_excitation_numbers') or [])!r}:
                from cats.multiwfn.tools import run_multiwfn, save_routine_config
                state = int(state)
                if not 1 <= state <= {int(nroots)}:
                    raise ValueError("Mulliken excited-state root is outside calculated roots")
                state_dir = Path(f"mulliken_state_{{state:04d}}")
                state_dir.mkdir(exist_ok=True)
                analysis_root = Path.cwd()
                try:
                    os.chdir(state_dir)
                    save_routine_config([
                        str(analysis_root / molden_input), "18", "13",
                        str(analysis_root / "orca.out"), str(state), "0", "q",
                    ], "excited_NO_config.txt")
                    run_multiwfn("excited_NO_config.txt", "excited_NO_log.txt")
                    wavefunction = Path(f"NO_{{state:04d}}.mwfn")
                    if not wavefunction.is_file():
                        raise RuntimeError(f"Multiwfn did not export {{wavefunction}}")
                    population = mulliken_population_routine(
                        str(wavefunction), fragments_0based={fragments_0based!r},
                    )
                    if len(population["atoms"]) != len(ground_state_mulliken["atoms"]):
                        raise ValueError("Ground/excited Mulliken atom counts differ")
                    if abs(population["total_charge_e"] - ground_state_mulliken["total_charge_e"]) > 0.001:
                        raise ValueError("Ground/excited Mulliken total charge is not conserved")
                    population.update(excitation_number=state, density="unrelaxed excited state",
                                      source_wavefunction=str(state_dir / wavefunction))
                    excited_state_mulliken.append(population)
                finally:
                    os.chdir(analysis_root)
            if density_analysis_only:
                transitions = extract_uv_transitions("orca.out")
                wavelengths = (
                    transitions.wavelengths_nm
                    if transitions.wavelengths_nm is not None
                    else 1.0e7 / transitions.wavenumbers_cm
                )
                ct_states = [
                    {{
                        "state": index,
                        "wavelength_nm": float(wavelength),
                        "oscillator_strength": float(strength),
                    }}
                    for index, (wavelength, strength) in enumerate(
                        zip(wavelengths, transitions.intensities), start=1
                    )
                ]
                if ifct_only:
                    get_full_ct_spectrum("orca.out", str(molden_input), fragments=fragments)
                    ct_rows, ct_index_files, ct_warnings = _parse_ct_multiple(
                        "charge_transfer", Path.cwd(), Path.cwd(), -1
                    )
                    ct_states, ct_terms, ifct_files, ifct_warnings = _parse_ifct_files(
                        "charge_transfer", Path.cwd(), Path.cwd(), -1
                    )
                else:
                    ct_rows, ct_index_files, ct_warnings = [], [], []
                    ct_terms, ifct_files, ifct_warnings = [], [], []
            elif ground_state_mulliken_only:
                ct_rows, ct_index_files, ct_warnings = [], [], []
                ct_states, ct_terms, ifct_files, ifct_warnings = [], [], [], []
            else:
                get_full_ct_spectrum("orca.out", str(molden_input), fragments=fragments)
                ct_rows, ct_index_files, ct_warnings = _parse_ct_multiple(
                    "charge_transfer", Path.cwd(), Path.cwd(), -1
                )
                ct_states, ct_terms, ifct_files, ifct_warnings = _parse_ifct_files(
                    "charge_transfer", Path.cwd(), Path.cwd(), -1
                )
            density_analysis = {{}}
            if not ground_state_mulliken_only and density_analysis_config.get("enabled"):
                density_kwargs = {{
                    "output_directory": "density_analysis",
                    "excitation_numbers": density_analysis_config.get("excitation_numbers") or (),
                    "grid_quality": int(density_analysis_config.get("grid_quality") or 3),
                    "include_nto": bool(density_analysis_config.get("include_nto", True)),
                    "export_pov_scenes": bool(density_analysis_config.get("export_pov_scenes", True)),
                    "delete_cubes_after_export": bool(density_analysis_config.get("delete_cubes_after_export", True)),
                    "hole_selector": density_analysis_config.get("hole_selector", "h"),
                    "particle_selector": density_analysis_config.get("particle_selector", "l"),
                    "max_workers": int(density_analysis_config.get("max_workers") or 1),
                    "nto_max_pairs": int(density_analysis_config.get("nto_max_pairs") or 1),
                    "nto_target_coverage": float(density_analysis_config.get("nto_target_coverage") or 0.8),
                }}
                if density_analysis_config.get("wavelength_bands_nm"):
                    density_kwargs["wavelength_bands_nm"] = density_analysis_config["wavelength_bands_nm"]
                density_analysis = generate_excitation_density_analysis(
                    molden_input,
                    Path("orca.out"),
                    ct_states,
                    **density_kwargs,
                )
            grouped_spectra = {{}}
            for row in ct_rows:
                key = (str(row.get("source") or ""), str(row.get("property") or ""))
                grouped_spectra.setdefault(key, []).append(
                    {{
                        "energy_eV": row.get("energy_eV"),
                        "wavelength_nm": row.get("wavelength_nm"),
                        "intensity": row.get("intensity"),
                    }}
                )
            ct_spectra = [
                {{"source": source, "property": property_name, "points": points}}
                for (source, property_name), points in grouped_spectra.items()
            ]
            for rows in (ct_states, ct_terms):
                for row in rows:
                    row.pop("calculation_id", None)

            Path("orca.gbw").unlink(missing_ok=True)
            summary = {{
                "status": "completed",
                "run_type": "density_analysis" if density_analysis_only else ("ground_state_mulliken" if ground_state_mulliken_only else "charge_transfer"),
                "tags": ["density_analysis"] if density_analysis_only else (["mulliken_population"] if ground_state_mulliken_only else ["charge_transfer"]),
                "source_run_id": {cfg.get('source_run_id')!r},
                "source_run_uid": {str(cfg.get('source_run_uid') or '')!r},
                "source_run_name": {str(cfg.get('source_run_name') or '')!r},
                "source_working_directory": str(source_directory),
                "charge_transfer": {{
                    "analysis_scope": "full_ifct" if ifct_only else ("density_analysis" if density_analysis_only else ("ground_state_mulliken" if ground_state_mulliken_only else "full_ifct")),
                    "orca_basename": {orca_basename!r},
                    "nroots": {int(nroots)},
                    "fragments_0based": {fragments_0based!r},
                    "fragments_1based": fragments,
                    "fragment_labels": {fragment_labels!r},
                    "ground_state_mulliken": ground_state_mulliken,
                    "excited_state_mulliken": excited_state_mulliken,
                    "ct_states": ct_states,
                    "ct_terms": ct_terms,
                    "ct_spectra": ct_spectra,
                    "density_analysis": density_analysis,
                    "ct_index_files": ct_index_files,
                    "ifct_files": ifct_files,
                    "warnings": ct_warnings + ifct_warnings,
                }},
                "artifacts": sorted(
                    str(path.relative_to(Path.cwd()))
                    for path in Path.cwd().rglob("*")
                    if path.is_file()
                ),
            }}
            Path("reaction_hub_summary.json").write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            """
        )
    if run_mode == "single_optimisation":
        single_cfg = single_constraints_config or {}
        fixed_symbols = tuple(
            str(symbol).strip()
            for symbol in single_cfg.get("fixed_symbols", ())
            if str(symbol).strip()
        )
        fixed_symbols_expr = repr(fixed_symbols if fixed_symbols else ())
        bond_pair_indices = single_cfg.get("bond_pair_indices")
        bond_pair_indices_expr = "None"
        if isinstance(bond_pair_indices, (list, tuple)) and len(bond_pair_indices) == 2:
            bond_pair_indices_expr = f"({int(bond_pair_indices[0])}, {int(bond_pair_indices[1])})"
        target_distance = single_cfg.get("target_distance")
        target_distance_expr = "None" if target_distance is None else repr(float(target_distance))
        single_fix = float(single_cfg.get("fix", 0.5))
        final_tddft_spectrum = bool(single_cfg.get("final_tddft_spectrum", False))
        snapshot_single_point = bool(single_cfg.get("snapshot_single_point", False))
        final_tddft_gradient_root = single_cfg.get("final_tddft_gradient_root")
        gradient_root_argument = "" if final_tddft_gradient_root is None else f"final_tddft_gradient_root={final_tddft_gradient_root},"
        snapshot_workflow_module = "snapshot_tddft_workflow" if final_tddft_gradient_root is not None else workflow_module
        cluster_tddft_difference_spectrum = bool(
            single_cfg.get("cluster_tddft_difference_spectrum", False)
        )
        final_tddft_nroots = _positive_int(single_cfg.get("final_tddft_nroots"), default=50)
        scan_nroots = _positive_int(single_cfg.get("tddft_scan_nroots"), default=12)
        initial_iroot = single_cfg.get("initial_iroot")
        initial_iroot_expr = "None" if initial_iroot is None else str(int(initial_iroot))
        charge_transfer_postprocess = ""
        if charge_transfer_config:
            ct_cfg = dict(charge_transfer_config)
            ct_fragments_0based = dict(ct_cfg.get("fragments_0based") or {})
            ct_fragments_1based = dict(ct_cfg.get("fragments_1based") or {})
            ct_fragment_labels = dict(ct_cfg.get("fragment_labels") or {})
            ct_nroots = _positive_int(ct_cfg.get("nroots"), default=400)
            charge_transfer_postprocess = textwrap.dedent(
                f"""

                from cats.multiwfn.plot_uvvis import get_full_ct_spectrum
                from cats.multiwfn.plot_uvvis import plot_charge_transfer_spectrum_from_folders
                from cats.orca.legacy_catalog import _parse_ct_multiple, _parse_ifct_files

                workflow_root = Path.cwd()
                ct_directory = workflow_root / "DFT" / "final_tddft_spectrum"
                orca_output = ct_directory / "orca.out"
                if not orca_output.is_file():
                    raise FileNotFoundError(f"Missing final TDDFT ORCA output: {{orca_output}}")
                molden_input = ct_directory / "orca.molden.input"
                if not molden_input.is_file():
                    orca_gbw = ct_directory / "orca.gbw"
                    if not orca_gbw.is_file():
                        raise FileNotFoundError(
                            f"Missing final TDDFT Molden file and ORCA wavefunction: {{orca_gbw}}"
                        )
                    orca_2mkl = shutil.which("orca_2mkl")
                    if not orca_2mkl:
                        orcadir = str(os.environ.get("ORCADIR") or "").strip()
                        candidate = Path(orcadir) / "orca_2mkl" if orcadir else None
                        if candidate is not None and candidate.is_file():
                            orca_2mkl = str(candidate)
                    if not orca_2mkl:
                        raise FileNotFoundError("orca_2mkl was not found in PATH or ORCADIR.")
                    subprocess.run([orca_2mkl, "orca", "-molden"], cwd=ct_directory, check=True)

                fragments = {ct_fragments_1based!r}
                os.chdir(ct_directory)
                try:
                    get_full_ct_spectrum("orca.out", "orca.molden.input", fragments=fragments)
                    plot_charge_transfer_spectrum_from_folders(
                        str(Path.cwd()), ["."], nroots={int(ct_nroots)}
                    )
                    ct_rows, ct_index_files, ct_warnings = _parse_ct_multiple(
                        "final_tddft", Path.cwd(), Path.cwd(), -1
                    )
                    ct_states, ct_terms, ifct_files, ifct_warnings = _parse_ifct_files(
                        "final_tddft", Path.cwd(), Path.cwd(), -1
                    )
                finally:
                    os.chdir(workflow_root)

                grouped_spectra = {{}}
                for row in ct_rows:
                    key = (str(row.get("source") or ""), str(row.get("property") or ""))
                    grouped_spectra.setdefault(key, []).append(
                        {{
                            "energy_eV": row.get("energy_eV"),
                            "wavelength_nm": row.get("wavelength_nm"),
                            "intensity": row.get("intensity"),
                        }}
                    )
                ct_spectra = [
                    {{"source": source, "property": property_name, "points": points}}
                    for (source, property_name), points in grouped_spectra.items()
                ]
                for rows in (ct_states, ct_terms):
                    for row in rows:
                        row.pop("calculation_id", None)

                summary_path = workflow_root / "reaction_hub_summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                summary["tags"] = sorted(set(summary.get("tags") or []) | {{"charge_transfer"}})
                summary["charge_transfer"] = {{
                    "orca_basename": "orca",
                    "nroots": {int(ct_nroots)},
                    "system_charge": {charge_expr},
                    "fragments_0based": {ct_fragments_0based!r},
                    "fragments_1based": fragments,
                    "fragment_labels": {ct_fragment_labels!r},
                    "ct_states": ct_states,
                    "ct_terms": ct_terms,
                    "ct_spectra": ct_spectra,
                    "ct_index_files": ct_index_files,
                    "ifct_files": ifct_files,
                    "warnings": ct_warnings + ifct_warnings,
                }}
                summary["artifacts"] = sorted(
                    str(path.relative_to(workflow_root))
                    for path in workflow_root.rglob("*")
                    if path.is_file()
                )
                summary_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                """
            )
        if qm_backend == "nwchem":
            nwchem_memory_mb = int(qm_backend_settings.get("memory_mb") or 4000)
            nwchem_molden = bool(qm_backend_settings.get("molden", True))
            nwchem_version = str(qm_backend_settings.get("version") or "7.2.2")
            nwchem_cam = qm_backend_settings.get("cam")
            if nwchem_cam is None:
                nwchem_cam = _default_nwchem_cam(xc_functional)
            nwchem_settings_expr = repr(
                {
                    **qm_backend_settings,
                    "memory_mb": nwchem_memory_mb,
                    "molden": nwchem_molden,
                    "version": nwchem_version,
                }
            )
            return textwrap.dedent(
                f"""\
                import os

                from cats.working_directory.nwchem_single_workflow import run_nwchem_single_optimisation

                os.environ.setdefault("OMP_NUM_THREADS", "1")
                os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])
                os.environ.setdefault("MKL_DYNAMIC", "FALSE")

                run_nwchem_single_optimisation(
                    input_path="input_structure.extxyz",
                    hpc_name={str(hpc_name or "").strip().lower()!r},
                    nodes={int(nodes)},
                    tasks_per_node={int(tasks_per_node)},
                    cpus_per_task={int(cpus_per_task)},
                    backend_settings={nwchem_settings_expr},
                    basis_keyword={basis_keyword!r},
                    xc_functional={xc_functional!r},
                    cam={nwchem_cam!r},
                    solvent={solvent_expr},
                    spin_multiplicity={spin_expr},
                    bond_pair_indices={bond_pair_indices_expr},
                    target_distance={target_distance_expr},
                    bond_fix={single_fix},
                    final_tddft_spectrum={bool(final_tddft_spectrum)},
                    final_cluster_tddft_difference_spectrum={bool(cluster_tddft_difference_spectrum)},
                    final_tddft_nroots={int(final_tddft_nroots)},
                )
                """
            )
        wrapper_script = textwrap.dedent(
            f"""\
            import json
            import os
            import shutil
            import subprocess
            from pathlib import Path

            from ase.constraints import FixAtoms, FixBondLength
            from ase.io import read
            from {snapshot_workflow_module} import BondScanSettings, run_workflow

            workflow_nodes = int(os.environ.get("WORKFLOW_NODES", {int(nodes)}))
            workflow_tasks_per_node = int(os.environ.get("WORKFLOW_TASKS_PER_NODE", {default_workflow_tasks_per_node}))
            workflow_cpus_per_task = int(os.environ.get("WORKFLOW_CPUS_PER_TASK", {int(cpus_per_task)}))
            workflow_cpus_per_node = workflow_tasks_per_node * workflow_cpus_per_task
            if workflow_nodes <= 0 or workflow_cpus_per_node <= 0:
                raise ValueError(
                    "Invalid workflow resource mapping: "
                    f"nodes={{workflow_nodes}}, cpus_per_node={{workflow_cpus_per_node}}"
                )
            print(
                "Workflow/{qm_backend_label} resources: "
                f"nodes={{workflow_nodes}}, tasks_per_node={{workflow_tasks_per_node}}, "
                f"cpus_per_task={{workflow_cpus_per_task}}, cpus_per_node={{workflow_cpus_per_node}}, "
                f"total_cpus={{workflow_nodes * workflow_cpus_per_node}}"
            )

            try:
                atoms = read("input_structure.extxyz")
            except Exception:
                atoms = read("input_structure.extxyz", format="xyz")
            constraints = []
            fixed_symbols = {fixed_symbols_expr}
            if fixed_symbols:
                fixed_indices = [atom.index for atom in atoms if atom.symbol in fixed_symbols]
                if fixed_indices:
                    constraints.append(FixAtoms(indices=fixed_indices))

            bond_pair_indices = {bond_pair_indices_expr}
            if bond_pair_indices is not None:
                target_distance = {target_distance_expr}
                if target_distance is not None:
                    atoms.set_distance(
                        int(bond_pair_indices[0]),
                        int(bond_pair_indices[1]),
                        float(target_distance),
                        fix={single_fix},
                    )
                constraints.append(
                    FixBondLength(
                        int(bond_pair_indices[0]),
                        int(bond_pair_indices[1]),
                    )
                )
            if constraints:
                atoms.set_constraint(constraints)

            run_workflow(
                atoms,
                n_iter=1,
                bond_scan=BondScanSettings(enabled=False),
                use_tddft_optimisation={bool(use_tddft_optimisation)},
                ensure_initial_dft_converged={bool(use_tddft_optimisation)},
                al_steps_per_iter={int(al_steps_per_iter)},
                max_al_steps_per_iter={max_al_steps_expr},
                fmax={float(fmax)},
                active_learning_optimizer={active_learning_optimizer!r},
                target_wavelength_nm={float(target_wavelength_nm)},
                tddft_scan_nroots={int(scan_nroots)},
                initial_iroot={initial_iroot_expr},
                basis_keyword={basis_keyword!r},
                xc_functional={xc_functional!r},
                solvent={solvent_expr},
                spin_multiplicity={spin_expr},
                system_charge={charge_expr},
                final_tddft_spectrum={bool(final_tddft_spectrum)},
                snapshot_single_point={bool(snapshot_single_point)},
                {gradient_root_argument}
                final_cluster_tddft_difference_spectrum={bool(cluster_tddft_difference_spectrum)},
                final_tddft_nroots={int(final_tddft_nroots)},
                qm_backend={qm_backend!r},
                qm_backend_settings={qm_backend_settings!r},
                nodes=workflow_nodes,
                cpus_per_node=workflow_cpus_per_node,
            )
            """
        )
        return wrapper_script + charge_transfer_postprocess

    if run_mode == "tddft_bond_scan":
        cfg = scan_config or {}
        fixed_symbols = tuple(str(symbol).strip() for symbol in cfg.get("fixed_symbols", ()) if str(symbol).strip())
        fixed_symbols_expr = repr(fixed_symbols if fixed_symbols else ())
        scan_nroots = _positive_int(cfg.get("tddft_scan_nroots"), default=50)
        target_distance = cfg.get("target_distance")
        target_distance_expr = "None" if target_distance is None else repr(float(target_distance))
        restart_extend_steps = int(cfg.get("restart_extend_steps", 0) or 0)
        restart_extend_increment = cfg.get("restart_extend_increment_A")
        restart_extend_increment_expr = (
            "None"
            if restart_extend_increment is None or restart_extend_increment == ""
            else repr(float(restart_extend_increment))
        )
        return textwrap.dedent(
            f"""\
            import os

            from ase.io import read
            from {workflow_module} import BondScanSettings, run_workflow

            workflow_nodes = int(os.environ.get("WORKFLOW_NODES", {int(nodes)}))
            workflow_tasks_per_node = int(os.environ.get("WORKFLOW_TASKS_PER_NODE", {default_workflow_tasks_per_node}))
            workflow_cpus_per_task = int(os.environ.get("WORKFLOW_CPUS_PER_TASK", {int(cpus_per_task)}))
            workflow_cpus_per_node = workflow_tasks_per_node * workflow_cpus_per_task
            if workflow_nodes <= 0 or workflow_cpus_per_node <= 0:
                raise ValueError(
                    "Invalid workflow resource mapping: "
                    f"nodes={{workflow_nodes}}, cpus_per_node={{workflow_cpus_per_node}}"
                )
            print(
                "Workflow/{qm_backend_label} resources: "
                f"nodes={{workflow_nodes}}, tasks_per_node={{workflow_tasks_per_node}}, "
                f"cpus_per_task={{workflow_cpus_per_task}}, cpus_per_node={{workflow_cpus_per_node}}, "
                f"total_cpus={{workflow_nodes * workflow_cpus_per_node}}"
            )

            try:
                atoms = read("input_structure.extxyz")
            except Exception:
                atoms = read("input_structure.extxyz", format="xyz")
            bond_scan = BondScanSettings(
                enabled=True,
                increment={float(cfg.get("increment", 0.05))},
                fix={float(cfg.get("fix", 0.5))},
                fixed_symbols={fixed_symbols_expr},
                bond_pair_indices=({int(cfg.get("bond_i", 0))}, {int(cfg.get("bond_j", 1))}),
                target_distance={target_distance_expr},
                distance_change_fraction={float(cfg.get("distance_change_fraction", 1.0))},
                enable_ts_smoothing={bool(cfg.get("enable_ts_smoothing", False))},
                refine_points_each_side={int(cfg.get("refine_points_each_side", 0))},
                restart_extend_steps={int(restart_extend_steps)},
                restart_extend_increment_A={restart_extend_increment_expr},
            )
            run_workflow(
                atoms,
                n_iter={int(n_iter)},
                bond_scan=bond_scan,
                use_tddft_optimisation={bool(use_tddft_optimisation)},
                al_steps_per_iter={int(al_steps_per_iter)},
                max_al_steps_per_iter={max_al_steps_expr},
                fmax={float(fmax)},
                active_learning_optimizer={active_learning_optimizer!r},
                target_wavelength_nm={float(target_wavelength_nm)},
                tddft_scan_nroots={int(scan_nroots)},
                basis_keyword={basis_keyword!r},
                xc_functional={xc_functional!r},
                solvent={solvent_expr},
                spin_multiplicity={spin_expr},
                system_charge={charge_expr},
                qm_backend={qm_backend!r},
                qm_backend_settings={qm_backend_settings!r},
                nodes=workflow_nodes,
                cpus_per_node=workflow_cpus_per_node,
            )
            """
        )

    if run_mode == "active_learning_neb":
        cfg = dict(neb_config or {})
        is_tddft_mlneb = _module_name_is_tddft_mlneb(workflow_module)
        fixed_symbols = tuple(
            str(symbol).strip()
            for symbol in cfg.get("fixed_symbols", ())
            if str(symbol).strip()
        )
        final_image_strategy = str(cfg.get("final_image_strategy") or "explicit_final_image").strip().lower()
        if final_image_strategy == "bond_extension_from_initial":
            final_image_strategy = "set_bond_distance_from_initial"
        initial_band_text = str(cfg.get("initial_band_extxyz") or "").strip()
        initial_loader = 'initial = read("input_structure.extxyz")'
        final_loader = "final = read(\"input_structure_final.extxyz\")"
        if bool(cfg.get("evaluation_only", False)):
            final_loader = "final = None"
        elif final_image_strategy in {"dimer", "dimer_refine"}:
            initial_loader = 'initial = read("input_structure.extxyz")'
            if initial_band_text:
                initial_loader = 'initial_band = read("input_band.extxyz", index=":")\n            initial = read("input_structure.extxyz")'
            final_loader = "final = None"
        elif initial_band_text:
            initial_loader = 'initial_band = read("input_band.extxyz", index=":")\n            initial = None'
            final_loader = "final = None"
        elif final_image_strategy in {"set_bond_distance_from_initial"}:
            final_loader = "final = None"
        workflow_function = "run_tddft_mlneb_workflow" if is_tddft_mlneb else "run_active_learning_neb_workflow"
        initial_band_kwarg = "initial_band_atoms=initial_band,\n" if initial_band_text else ""
        workflow_specific_kwargs = ""
        if is_tddft_mlneb:
            workflow_specific_kwargs = textwrap.indent(
                textwrap.dedent(
                    f"""\
                    target_wavelength_nm={float(target_wavelength_nm)},
                    tddft_scan_nroots={int(cfg.get("tddft_scan_nroots") or 50)},
                    basis_keyword={basis_keyword!r},
                    xc_functional={xc_functional!r},
                    solvent={solvent_expr},
                    spin_multiplicity={spin_expr},
                    nodes={int(nodes)},
                    cpus_per_node={int(tasks_per_node) * int(cpus_per_task)},
                    {initial_band_kwarg}"""
                ),
                "                ",
            )
        elif str(cfg.get("backend") or "mace").strip().lower() == "dft":
            workflow_specific_kwargs = textwrap.indent(
                textwrap.dedent(
                    f"""\
                    basis_keyword={basis_keyword!r},
                    xc_functional={xc_functional!r},
                    solvent={solvent_expr},
                    spin_multiplicity={spin_expr},
                    nodes={int(nodes)},
                    cpus_per_node={int(tasks_per_node) * int(cpus_per_task)},
                    {initial_band_kwarg}"""
                ),
                "                ",
            )
        elif str(cfg.get("backend") or "mace").strip().lower() == "tddft":
            target_iroot_val = int(cfg.get("initial_iroot") or cfg.get("root_state") or 5)
            scan_nroots_val = int(cfg.get("tddft_scan_nroots") or (target_iroot_val + 10))
            workflow_specific_kwargs = textwrap.indent(
                textwrap.dedent(
                    f"""\
                    basis_keyword={basis_keyword!r},
                    xc_functional={xc_functional!r},
                    solvent={solvent_expr},
                    spin_multiplicity={spin_expr},
                    system_charge={charge_expr},
                    target_wavelength_nm={float(cfg.get("target_wavelength_nm") or target_wavelength_nm or 520.0)},
                    tddft_scan_nroots={scan_nroots_val},
                    initial_iroot={target_iroot_val},
                    fragments={cfg.get("fragments")!r},
                    nodes={int(nodes)},
                    cpus_per_node={int(tasks_per_node) * int(cpus_per_task)},
                    {initial_band_kwarg}"""
                ),
                "                ",
            )
        else:
            workflow_specific_kwargs = textwrap.indent(
                textwrap.dedent(
                    f"""\
                    mace_model_path={str(cfg.get("mace_model_path") or "").strip()!r},
                    mace_device={str(cfg.get("mace_device") or "cuda").strip()!r},
                    mace_default_dtype={str(cfg.get("mace_default_dtype") or "float64").strip()!r},
                    mace_dispersion={bool(cfg.get("mace_dispersion", False))},
                    use_regular_endpoint_optimisation_for_mace={bool(cfg.get("use_regular_endpoint_optimisation_for_mace", True))},
                    """
                ),
                "                ",
            )
        return textwrap.dedent(
            f"""\
            import json
            import sys
            from pathlib import Path

            _here = Path(__file__).resolve().parent
            if str(_here) not in sys.path:
                sys.path.insert(0, str(_here))

            from ase.io import read
            try:
                from active_learning_neb_workflow import {workflow_function}
            except ImportError:
                from {workflow_module} import {workflow_function}

            {initial_loader}
            {final_loader}

            result = {workflow_function}(
                initial_atoms=initial,
                final_atoms=final,
                fmax={float(fmax)},
                optimizer={active_learning_optimizer!r},
                final_image_strategy={final_image_strategy!r},
                optimize_initial_image={bool(cfg.get("optimize_initial_image", False))},
                optimize_final_image={bool(cfg.get("optimize_final_image", False))},
                endpoint_opt_steps={int(cfg.get("endpoint_opt_steps", 200))},
                endpoint_opt_fmax={"None" if cfg.get("endpoint_opt_fmax") is None else float(cfg.get("endpoint_opt_fmax"))},
                break_bond_i={"None" if cfg.get("break_bond_i") is None else int(cfg.get("break_bond_i"))},
                break_bond_j={"None" if cfg.get("break_bond_j") is None else int(cfg.get("break_bond_j"))},
                set_bond_distance_A={float(cfg.get("set_bond_distance_A", cfg.get("break_bond_extension_A", 2.5)))},
                break_bond_fix={float(cfg.get("break_bond_fix", 0.5))},
                break_assert_broken={bool(cfg.get("break_assert_broken", True))},
                break_cutoff_scale={float(cfg.get("break_cutoff_scale", 1.2))},
                backend={str(cfg.get("backend") or "mace").strip().lower()!r},
                atom_shift_per_image_A={float(cfg.get("atom_shift_per_image_A", 0.2))},
                min_images={int(cfg.get("min_images", 7))},
                max_images={int(cfg.get("max_images", 21))},
                spring_constant={float(cfg.get("spring_constant", 0.1))},
                max_steps={int(cfg.get("max_steps", 300))},
                climb={bool(cfg.get("climb", True))},
                split_on_minimum={bool(cfg.get("split_on_minimum", True))},
                minimum_energy_window_eV={float(cfg.get("minimum_energy_window_eV", 0.02))},
                fixed_symbols={repr(fixed_symbols)},
                catlearn_min_data={int(cfg.get("catlearn_min_data", 3))},
                catlearn_unc_convergence={float(cfg.get("catlearn_unc_convergence", 0.05))},
                catlearn_max_unc={float(cfg.get("catlearn_max_unc", 0.03))},
                catlearn_steps={int(cfg.get("catlearn_steps", 200))},
                catlearn_ml_steps={int(cfg.get("catlearn_ml_steps", 500))},
                catlearn_dtrust={float(cfg.get("catlearn_dtrust", 2.0))},
                catlearn_guesses_per_cycle={int(cfg.get("catlearn_guesses_per_cycle", 10))},
                dft_evaluation_timeout_seconds={int(cfg.get("dft_evaluation_timeout_seconds", 21600))},
                dft_min_distance_scale={float(cfg.get("dft_min_distance_scale", 0.55))},
                dft_max_force_eV_A={float(cfg.get("dft_max_force_eV_A", 50.0))},
                dft_max_energy_drop_eV={float(cfg.get("dft_max_energy_drop_eV", 20.0))},
                evaluation_only={bool(cfg.get("evaluation_only", False))},
                restart_from={repr(str(cfg.get("restart_from") or cfg.get("old_calculation_dir") or cfg.get("restart_dir")).strip()) if (cfg.get("restart_from") or cfg.get("old_calculation_dir") or cfg.get("restart_dir")) else "None"},
                refine_ts_with_dimer={bool(cfg.get("refine_ts_with_dimer", False))},
                dimer_max_steps={int(cfg.get("dimer_max_steps", 100))},
                dimer_separation={float(cfg.get("dimer_separation", 0.0001))},
                dimer_trial_trans_step={float(cfg.get("dimer_trial_trans_step", 0.001))},
                dimer_maximum_translation={float(cfg.get("dimer_maximum_translation", 0.05))},
                dimer_f_rot_min={float(cfg.get("dimer_f_rot_min", 0.1))},
                dimer_f_rot_max={float(cfg.get("dimer_f_rot_max", 1.0))},
                dimer_max_num_rot={int(cfg.get("dimer_max_num_rot", 3))},
                dimer_trial_angle={float(cfg.get("dimer_trial_angle", 0.785398))},
                dimer_use_central_forces={bool(cfg.get("dimer_use_central_forces", True))},
                use_catlearn_dimer={bool(cfg.get("use_catlearn_dimer", False))},
                catlearn_dimer_polish_steps={int(cfg.get("catlearn_dimer_polish_steps", 25))},
                prior_trajectories={repr(cfg.get("prior_trajectories"))},
                prior_artifact_ids={repr(cfg.get("prior_artifact_ids"))},
                staged_seed_file={repr(cfg.get("staged_seed_file") or ("seed_training.extxyz" if cfg.get("seed_training_extxyz") else None))},
                initial_eigenmode={repr(cfg.get("initial_eigenmode"))},
                displacement_vector={repr(cfg.get("displacement_vector"))},
                dimer_neighbor_shells={repr(cfg.get("dimer_neighbor_shells", 2))},
                vibration_neighbor_shells={repr(cfg.get("vibration_neighbor_shells", 2))},
{workflow_specific_kwargs}
                output_dir=Path("neb"),
            )

            summary_path = Path("reaction_hub_summary.json")
            summary_path.write_text(
                json.dumps(result, indent=2, sort_keys=True),
                encoding="utf-8",
            )

            final_xyz_path = Path(str(result.get("final_xyz_path") or ""))
            if final_xyz_path.exists():
                Path("reaction_hub_final.xyz").write_text(
                    final_xyz_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

            workflow_log = Path(str(result.get("workflow_log_path") or ""))
            if workflow_log.exists():
                Path("reaction_hub_workflow.log").write_text(
                    workflow_log.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
            """
        )

    if run_mode == "alloy_growth":
        cfg = dict(alloy_config or {})
        optimize_seed_only = bool(cfg.pop("optimize_seed_only", False))
        callable_name = (
            "optimize_alloy_nanocluster"
            if optimize_seed_only
            else "grow_alloy_nanocluster_by_atom_addition"
        )
        return textwrap.dedent(
            f"""\
            import json
            from pathlib import Path

            from ase.io import read
            from {workflow_module} import {callable_name}

            result = {callable_name}(
                read("input_structure.extxyz", index={':' if optimize_seed_only else '-1'!r}),
                **{cfg!r},
            )
            Path("reaction_hub_summary.json").write_text(
                json.dumps({{key: str(value) for key, value in result.items()}}, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            """
        )

    if run_mode == "coadsorbate_placement":
        cfg = dict(coadsorbate_config or {})
        adsorbate_paths = [str(path).strip() for path in cfg.get("adsorbate_paths") or [] if str(path).strip()]
        if not adsorbate_paths:
            raise ValueError("coadsorbate_placement requires at least one adsorbate path.")

        mace_model_path = str(cfg.get("mace_model_path") or "").strip()
        if not mace_model_path:
            mace_model_path = str(getattr(settings, "REACTION_HUB_COADSORBATE_MACE_MODEL_PATH", "") or "").strip()

        generation_site_symbols = tuple(
            str(symbol).strip()
            for symbol in (cfg.get("generation_site_symbols") or ("Ti", "Au", "Cu"))
            if str(symbol).strip()
        )
        generation_passive_symbols = tuple(
            str(symbol).strip()
            for symbol in (cfg.get("generation_passive_symbols") or ("Au", "Ce", "O", "Cu"))
            if str(symbol).strip()
        )
        similarity_support_species = tuple(
            str(symbol).strip()
            for symbol in (cfg.get("similarity_support_species") or ("Zn", "Au", "Cu", "Ce", "Ti", "O"))
            if str(symbol).strip()
        )
        if str(cfg.get("placement_mode") or "standard").strip() == "silicate_pore":
            silicate_heights = tuple(float(value) for value in (cfg.get("silicate_heights") or (1.45, 1.70, 1.95, 2.5, 3.0, 3.5)))
            silicate_triangle = tuple(int(value) for value in (cfg.get("silicate_pore_triangle_indices") or (1, 2, 3)))
            silicate_site_indices = tuple(int(value) for value in (cfg.get("silicate_site_atom_indices") or ()))
            return textwrap.dedent(
                f"""\
                import json
                from datetime import datetime, timezone
                from pathlib import Path

                from ase.io import read
                from {workflow_module} import run_silicate_pore_adsorbate_workflow

                structure = read("input_structure.extxyz")
                adsorbate_paths = {repr(adsorbate_paths)}
                adsorbates = [read(path) for path in adsorbate_paths]
                print(
                    f"[ReactionHub] Loaded silicate pore inputs: {{len(structure)}} framework atoms, "
                    f"{{len(adsorbates)}} adsorbates",
                    flush=True,
                )

                candidates_db_path = Path("scratch/silicate_pore_candidates.db")
                pre_relax_db_path = Path("scratch/silicate_pore_prefilter.db")
                relaxed_db_path = Path("scratch/silicate_pore_relaxed.db")
                final_db_path = Path("scratch/silicate_pore_final_for_dft.db")
                candidate_report_csv_path = Path("scratch/silicate_pore_candidate_similarity.csv")
                relaxed_minimum_report_csv_path = Path("scratch/silicate_pore_relaxed_similarity.csv")
                relaxation_report_csv_path = Path("scratch/silicate_pore_relaxation_report.csv")
                summary_csv_path = Path("scratch/silicate_pore_summary.csv")
                summary_txt_path = Path("scratch/silicate_pore_summary.txt")

                print("[ReactionHub] Entering silicate pore placement workflow", flush=True)
                result = run_silicate_pore_adsorbate_workflow(
                    structure,
                    adsorbates=adsorbates,
                    candidates_db_path=candidates_db_path,
                    pre_relax_db_path=pre_relax_db_path,
                    relaxed_db_path=relaxed_db_path,
                    final_db_path=final_db_path,
                    candidate_report_csv_path=candidate_report_csv_path,
                    relaxed_minimum_report_csv_path=relaxed_minimum_report_csv_path,
                    relaxation_report_csv_path=relaxation_report_csv_path,
                    summary_csv_path=summary_csv_path,
                    summary_txt_path=summary_txt_path,
                    candidate_reference_row_id={int(cfg.get("candidate_reference_row_id", 1))},
                    relaxed_minimum_reference_row_id={int(cfg.get("relaxed_minimum_reference_row_id", 1))},
                    candidate_duplicate_cutoff={repr(cfg.get("candidate_duplicate_cutoff"))},
                    relaxed_minimum_duplicate_cutoff={repr(cfg.get("relaxed_minimum_duplicate_cutoff"))},
                    candidate_suggested_cutoff_scale={float(cfg.get("candidate_suggested_cutoff_scale", 1.0))},
                    relaxed_minimum_suggested_cutoff_scale={float(cfg.get("relaxed_minimum_suggested_cutoff_scale", 1.0))},
                    mace_model_path={mace_model_path!r},
                    mace_fmax={float(cfg.get("mace_fmax", 0.05))},
                    mace_max_steps={int(cfg.get("mace_max_steps", 200))},
                    mace_device={str(cfg.get("mace_device") or "cuda").strip()!r},
                    mace_dispersion={bool(cfg.get("mace_dispersion", False))},
                    mace_default_dtype={str(cfg.get("mace_default_dtype") or "float64").strip()!r},
                    mace_enable_cueq={bool(cfg.get("mace_enable_cueq", True))},
                    require_converged={bool(cfg.get("require_converged", True))},
                    generation_kwargs={{
                        "origin_atom_index": {int(cfg.get("silicate_origin_atom_index", 0))},
                        "pore_triangle_indices": {repr(silicate_triangle)},
                        "site_atom_indices": {repr(silicate_site_indices or None)},
                        "heights": {repr(silicate_heights)},
                        "rotation_step_degrees": {float(cfg.get("silicate_rotation_step_degrees", 60.0))},
                        "clash_scale": {float(cfg.get("silicate_clash_scale", 0.8))},
                        "push_step": {float(cfg.get("silicate_push_step", 0.1))},
                        "max_push_steps": {int(cfg.get("silicate_max_push_steps", 8))},
                    }},
                    similarity_kwargs={{
                        "support_species": {repr(tuple(symbol for symbol in (similarity_support_species or ("Si", "O", "Al", "H"))))},
                        "slab_shave_distance": {repr(cfg.get("similarity_slab_shave_distance"))},
                        "kernel_gamma": {float(cfg.get("similarity_kernel_gamma", 1.0))},
                        "kernel_alpha": {float(cfg.get("similarity_kernel_alpha", 1.0))},
                        "kernel_threshold": {float(cfg.get("similarity_kernel_threshold", 1.0e-6))},
                    }},
                    show_final={bool(cfg.get("show_final", False))},
                )
                summary_df = result.get("summary_df")
                if summary_df is not None:
                    print(summary_df)

                summary_payload = {{
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "mode": "coadsorbate_placement",
                    "placement_mode": "silicate_pore",
                    "coadsorbate": {{
                        "adsorbate_count": len(adsorbates),
                        "adsorbate_paths": adsorbate_paths,
                        "mace_model_path": {mace_model_path!r},
                        "mace_device": {str(cfg.get("mace_device") or "cuda").strip()!r},
                        "mace_enable_cueq": {bool(cfg.get("mace_enable_cueq", True))},
                    }},
                    "final": {{
                        "dft_db_paths": [str(final_db_path)] if final_db_path.exists() else [],
                        "summary_txt_paths": [str(summary_txt_path)],
                    }},
                    "artifacts": {{
                        "db_paths": [str(candidates_db_path), str(pre_relax_db_path), str(relaxed_db_path), str(final_db_path)],
                        "csv_paths": [str(candidate_report_csv_path), str(relaxed_minimum_report_csv_path), str(relaxation_report_csv_path), str(summary_csv_path)],
                        "txt_paths": [str(summary_txt_path)],
                    }},
                }}
                Path("reaction_hub_summary.json").write_text(
                    json.dumps(summary_payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                """
            )
        return textwrap.dedent(
            f"""\
            import json
            from datetime import datetime, timezone
            from pathlib import Path

            from ase.io import read
            from {workflow_module} import run_coadsorbate_similarity_workflow

            structure = read("input_structure.extxyz")
            adsorbate_paths = {repr(adsorbate_paths)}
            adsorbates = [read(path) for path in adsorbate_paths]

            candidates_db_path = Path("scratch/coadsorption_candidates.db")
            pre_relax_db_path = Path("scratch/coadsorption_prefilter.db")
            relaxed_db_path = Path("scratch/coadsorption_relaxed.db")
            final_db_path = Path("scratch/coadsorption_final_for_dft.db")
            single_a_final_db_path = Path("scratch/coadsorption_single_a_final_for_dft.db")
            single_b_final_db_path = Path("scratch/coadsorption_single_b_final_for_dft.db")
            site_probe_report_csv_path = Path("scratch/coadsorption_site_probe_similarity.csv")
            candidate_report_csv_path = Path("scratch/coadsorption_candidate_similarity.csv")
            relaxed_minimum_report_csv_path = Path("scratch/coadsorption_relaxed_minimum_similarity.csv")
            relaxation_report_csv_path = Path("scratch/coadsorption_relaxation_report.csv")
            summary_csv_path = Path("scratch/coadsorption_workflow_summary.csv")
            summary_txt_path = Path("scratch/coadsorption_workflow_summary.txt")

            result = run_coadsorbate_similarity_workflow(
                structure,
                adsorbates=adsorbates,
                candidates_db_path=candidates_db_path,
                pre_relax_db_path=pre_relax_db_path,
                relaxed_db_path=relaxed_db_path,
                final_db_path=final_db_path,
                single_a_final_db_path=single_a_final_db_path,
                single_b_final_db_path=single_b_final_db_path,
                site_probe_report_csv_path=site_probe_report_csv_path,
                candidate_report_csv_path=candidate_report_csv_path,
                relaxed_minimum_report_csv_path=relaxed_minimum_report_csv_path,
                relaxation_report_csv_path=relaxation_report_csv_path,
                summary_csv_path=summary_csv_path,
                summary_txt_path=summary_txt_path,
                candidate_reference_row_id={int(cfg.get("candidate_reference_row_id", 1))},
                relaxed_minimum_reference_row_id={int(cfg.get("relaxed_minimum_reference_row_id", 1))},
                site_probe_symbol={str(cfg.get("site_probe_symbol") or "H")!r},
                site_probe_similarity_cutoff={float(cfg.get("site_probe_similarity_cutoff", 0.99995))},
                single_mode_duplicate_cutoff={float(cfg.get("single_mode_duplicate_cutoff", 0.9999))},
                candidate_duplicate_cutoff={repr(cfg.get("candidate_duplicate_cutoff", 0.9999))},
                candidate_suggested_cutoff_scale={float(cfg.get("candidate_suggested_cutoff_scale", 1.0))},
                relaxed_minimum_duplicate_cutoff={repr(cfg.get("relaxed_minimum_duplicate_cutoff", 0.99999))},
                relaxed_minimum_suggested_cutoff_scale={float(cfg.get("relaxed_minimum_suggested_cutoff_scale", 1.0))},
                similarity_batch_size={int(cfg.get("similarity_batch_size", 100))},
                mace_model_path={mace_model_path!r},
                mace_fmax={float(cfg.get("mace_fmax", 0.05))},
                mace_max_steps={int(cfg.get("mace_max_steps", 200))},
                mace_device={str(cfg.get("mace_device") or "cuda").strip()!r},
                mace_dispersion={bool(cfg.get("mace_dispersion", True))},
                mace_default_dtype={str(cfg.get("mace_default_dtype") or "float64").strip()!r},
                mace_enable_cueq={bool(cfg.get("mace_enable_cueq", True))},
                relax_support={bool(cfg.get("relax_support", False))},
                support_relaxation_origin_atom_index={repr(cfg.get("support_relaxation_origin_atom_index"))},
                support_relaxation_neighbor_shells={repr(cfg.get("support_relaxation_neighbor_shells"))},
                keep_bound_only={bool(cfg.get("keep_bound_only", True))},
                require_converged={bool(cfg.get("require_converged", True))},
                generation_kwargs={{
                    "site_symbols": {repr(generation_site_symbols)},
                    "passive_symbols": {repr(generation_passive_symbols)},
                    "height": {float(cfg.get("generation_height", 2.0))},
                    "contact_radius_scale": {repr(cfg.get("generation_contact_radius_scale", 1.0))},
                    "z_cutoff": {repr(cfg.get("generation_z_cutoff"))},
                    "z_buffer": {repr(cfg.get("generation_z_buffer"))},
                    "origin_atom_index": {repr(cfg.get("generation_origin_atom_index"))},
                    "site_cutoff": {repr(cfg.get("generation_site_cutoff"))},
                    "include_below_surface": {bool(cfg.get("generation_include_below_surface", False))},
                    "force_triangulated_com_fallback": {bool(cfg.get("generation_force_triangulated_com_fallback", True))},
                    "rotation_step_degrees": {float(cfg.get("generation_rotation_step_degrees", 30.0))},
                    "include_flipped_orientations": {bool(cfg.get("generation_include_flipped_orientations", True))},
                    "skip_geometry_screening": {bool(cfg.get("generation_skip_geometry_screening", False))},
                }},
                similarity_kwargs={{
                    "support_species": {repr(similarity_support_species)},
                    "slab_shave_distance": {repr(cfg.get("similarity_slab_shave_distance"))},
                    "kernel_gamma": {float(cfg.get("similarity_kernel_gamma", 1.0))},
                    "kernel_alpha": {float(cfg.get("similarity_kernel_alpha", 1.0))},
                    "kernel_threshold": {float(cfg.get("similarity_kernel_threshold", 1.0e-6))},
                }},
                show_final={bool(cfg.get("show_final", False))},
            )
            summary_df = result.get("summary_df")
            if summary_df is not None:
                print(summary_df)

            summary_payload = {{
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "mode": "coadsorbate_placement",
                "coadsorbate": {{
                    "adsorbate_count": len(adsorbates),
                    "adsorbate_paths": adsorbate_paths,
                    "mace_model_path": {mace_model_path!r},
                    "mace_device": {str(cfg.get("mace_device") or "cuda").strip()!r},
                    "mace_enable_cueq": {bool(cfg.get("mace_enable_cueq", True))},
                }},
                "final": {{
                    "dft_db_paths": [
                        str(path)
                        for path in (
                            single_a_final_db_path,
                            single_b_final_db_path,
                            final_db_path,
                        )
                        if path.exists()
                    ],
                    "summary_txt_paths": [str(summary_txt_path)],
                }},
                "artifacts": {{
                    "db_paths": [
                        str(candidates_db_path),
                        str(pre_relax_db_path),
                        str(relaxed_db_path),
                        str(single_a_final_db_path),
                        str(single_b_final_db_path),
                        str(final_db_path),
                    ],
                    "csv_paths": [
                        str(site_probe_report_csv_path),
                        str(candidate_report_csv_path),
                        str(relaxed_minimum_report_csv_path),
                        str(relaxation_report_csv_path),
                        str(summary_csv_path),
                    ],
                    "txt_paths": [str(summary_txt_path)],
                }},
            }}
            Path("reaction_hub_summary.json").write_text(
                json.dumps(summary_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            """
        )

    raise ValueError(
        "Unsupported run_mode. Choose 'single_optimisation', 'tddft_bond_scan', "
        "'active_learning_neb', 'coadsorbate_placement', or 'alloy_growth'."
    )


def resolve_active_reservation(
    hpc_key: str,
    partition: str = "",
    account: str = "",
    qos: str = "",
    user_reservation: str = "",
) -> str:
    user_res = str(user_reservation or "").strip()
    if user_res.casefold() in {"SLURM_ACCOUNT_RES_1", "SLURM_ACCOUNT_RES_2"}:
        user_res = ""

    reservations = getattr(settings, "REACTION_HUB_ACTIVE_RESERVATIONS", []) or []
    now = timezone.now()
    if user_res:
        for item in reservations:
            if not isinstance(item, dict):
                continue
            if str(item.get("name") or "").strip().casefold() == user_res.casefold():
                start_text = str(item.get("start_utc") or "").strip()
                end_text = str(item.get("end_utc") or "").strip()
                try:
                    start_dt = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
                    if not (start_dt <= now < end_dt):
                        return ""
                except Exception:
                    pass
                return user_res
        return user_res

    for item in reservations:
        if not isinstance(item, dict):
            continue
        res_hpc = str(item.get("hpc_system") or "").strip().lower()
        res_partition = str(item.get("partition") or "").strip().lower()
        res_account = str(item.get("account") or "").strip().lower()
        res_qos = str(item.get("qos") or "").strip().lower()
        res_name = str(item.get("name") or "").strip()

        if res_hpc and res_hpc != str(hpc_key or "").strip().lower():
            continue
        if res_partition and res_partition != str(partition or "").strip().lower():
            continue
        if res_account and res_account != str(account or "").strip().lower():
            continue
        if res_qos and res_qos != str(qos or "").strip().lower():
            continue

        start_text = str(item.get("start_utc") or "").strip()
        end_text = str(item.get("end_utc") or "").strip()

        try:
            start_dt = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
            if start_dt <= now < end_dt:
                return res_name
        except Exception:
            continue

    return ""


def _reservation_node_quota(reservation: str) -> int:
    target = str(reservation or "").strip().casefold()
    if not target or target == "SLURM_ACCOUNT_RES_1":
        return 0
    now = timezone.now()
    for item in getattr(settings, "REACTION_HUB_ACTIVE_RESERVATIONS", []) or []:
        if isinstance(item, dict) and str(item.get("name") or "").strip().casefold() == target:
            start_text = str(item.get("start_utc") or "").strip()
            end_text = str(item.get("end_utc") or "").strip()
            try:
                start_dt = datetime.fromisoformat(start_text.replace("Z", "+00:00"))
                end_dt = datetime.fromisoformat(end_text.replace("Z", "+00:00"))
                if not (start_dt <= now < end_dt):
                    return 0
            except Exception:
                pass
            allocated = int(item.get("nodes_quota") or item.get("nodes_allocated") or 0)
            reserved = int(item.get("nodes_reserved") or 0)
            return max(0, allocated - reserved)
    return 0


def _build_submission_script(
    *,
    job_name: str = "workflow",
    run_python_filename: str,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
    gpus: int = 0,
    include_cpu_layout_directives: bool | None = None,
    include_ntasks: bool | None = None,
    include_tasks_per_node: bool | None = None,
    include_cpus_per_task: bool | None = None,
    gpu_directive: str = "gpus",
    exclusive: bool = False,
    use_srun: bool = False,
    srun_ntasks: int = 1,
    srun_gpus_per_task: int = 0,
    walltime: str,
    account: str,
    partition: str,
    qos: str,
    reservation: str = "",
    slurm_memory: str = "",
    module_loads: list[str],
    venv_activate_path: str,
    pythonpath_entries: list[str],
    scratch_base: str,
    cpu_freq: str = "",
    include_cpu_freq_req: bool = True,
    extra_env_exports: dict[str, str] | None = None,
    previous_scheduler_job_ids: list[str] | None = None,
    run_mode: str = "",
    run_in_place: bool = False,
    workflow_tasks_per_node: int | None = None,
) -> str:
    normalized_gpus = int(gpus)
    job_name_text = str(job_name or "").strip() or "workflow"
    if include_cpu_layout_directives is None:
        include_cpu_layout_directives = normalized_gpus <= 0
    if include_ntasks is None:
        include_ntasks = False
    if include_tasks_per_node is None:
        include_tasks_per_node = bool(include_cpu_layout_directives)
    if include_cpus_per_task is None:
        include_cpus_per_task = bool(include_cpu_layout_directives)
    normalized_gpu_directive = str(gpu_directive or "gpus").strip().lower()
    if normalized_gpu_directive not in {"gpus", "gres"}:
        normalized_gpu_directive = "gpus"
    walltime_text = str(walltime or "").strip()
    memory_text = str(slurm_memory or "").strip()
    account_text = str(account or "").strip()
    partition_text = str(partition or "").strip()
    qos_text = str(qos or "").strip()
    reservation_text = str(reservation or "").strip()
    if not walltime_text:
        raise ValueError("walltime is required for Slurm submission.")
    normalized_run_mode = str(run_mode or "").strip()
    normalized_run_in_place = bool(run_in_place)
    temp_directory_expr = _temp_directory_expr_from_scratch_base(scratch_base)

    sanitized_previous_ids: list[str] = []
    seen_previous_ids: set[str] = set()
    for raw in previous_scheduler_job_ids or []:
        value = str(raw or "").strip()
        if not value or value in seen_previous_ids:
            continue
        seen_previous_ids.add(value)
        sanitized_previous_ids.append(value)
    previous_ids_csv = ",".join(sanitized_previous_ids)
    run_python_cmd = (
        'printf "[ReactionHub] Starting %s at " "$python_input" > "$HOME_DIRECTORY/$output"; '
        'date -Is >> "$HOME_DIRECTORY/$output"; '
        'time python3 -u "$python_input" >> "$HOME_DIRECTORY/$output" 2>&1'
    )
    if use_srun:
        run_parts = [f"time srun --ntasks={int(srun_ntasks)}"]
        if int(srun_gpus_per_task) > 0:
            run_parts.append(f"--gpus-per-task={int(srun_gpus_per_task)}")
        run_parts.append('python3 -u "$python_input" >> "$HOME_DIRECTORY/$output" 2>&1')
        run_python_cmd = (
            'printf "[ReactionHub] Starting %s at " "$python_input" > "$HOME_DIRECTORY/$output"; '
            'date -Is >> "$HOME_DIRECTORY/$output"; '
            + " ".join(run_parts)
        )

    lines = [
        "#!/bin/bash --login",
        "",
        f"#SBATCH --job-name={job_name_text}",
        f"#SBATCH --nodes={int(nodes)}",
    ]
    if bool(include_ntasks):
        lines.append(f"#SBATCH --ntasks={int(nodes) * int(tasks_per_node)}")
    if bool(include_tasks_per_node):
        lines.append(f"#SBATCH --tasks-per-node={int(tasks_per_node)}")
    if bool(include_cpus_per_task):
        lines.append(f"#SBATCH --cpus-per-task={int(cpus_per_task)}")
    if normalized_gpus > 0:
        if normalized_gpu_directive == "gres":
            lines.append(f"#SBATCH --gres=gpu:{normalized_gpus}")
        else:
            lines.append(f"#SBATCH --gpus={normalized_gpus}")
    if memory_text:
        lines.append(f"#SBATCH --mem={memory_text}")
    if bool(exclusive):
        lines.append("#SBATCH --exclusive")
    if str(cpu_freq or "").strip():
        lines.append(f"#SBATCH --cpu-freq={str(cpu_freq).strip()}")
    lines.append(f"#SBATCH --time={walltime_text}")
    if account_text:
        lines.append(f"#SBATCH --account={account_text}")
    if partition_text:
        lines.append(f"#SBATCH --partition={partition_text}")
    if qos_text:
        lines.append(f"#SBATCH --qos={qos_text}")
    if reservation_text:
        lines.append(f"#SBATCH --reservation={reservation_text}")
    derived_cpus_per_node = int(tasks_per_node) * int(cpus_per_task)
    if workflow_tasks_per_node is not None:
        export_tasks_per_node = int(workflow_tasks_per_node)
    else:
        export_tasks_per_node = 96 if derived_cpus_per_node >= 192 else int(tasks_per_node)
    lines.extend(
        [
            "",
            "set -euo pipefail",
            "ulimit -u unlimited 2>/dev/null || true",
            f"export WORKFLOW_NODES={int(nodes)}",
            f"export WORKFLOW_TASKS_PER_NODE={int(export_tasks_per_node)}",
            f"export WORKFLOW_CPUS_PER_TASK={int(cpus_per_task)}",
        ]
    )
    mem_mb = _slurm_memory_mb(memory_text) if memory_text else None
    if mem_mb and derived_cpus_per_node > 0:
        gb_match = re.fullmatch(r"\s*(\d+)\s*G(?:B)?\s*", str(memory_text or ""), re.IGNORECASE)
        mem_target = int(gb_match.group(1)) * 1000 if gb_match else mem_mb
        calc_cpus = export_tasks_per_node * int(cpus_per_task)
        lines.append(f"export WORKFLOW_MEM_PER_CPU={max(1000, int(round(mem_target * 0.70) // calc_cpus))}")
    if normalized_run_mode == "charge_transfer":
        lines.append("ulimit -s unlimited 2>/dev/null || true")
    if bool(include_cpu_freq_req):
        lines.append("export SLURM_CPU_FREQ_REQ=2250000")
    for key, value in (extra_env_exports or {}).items():
        key_text = str(key or "").strip()
        if key_text:
            lines.append(f"export {key_text}={_q(str(value))}")
    lines.append("")

    for module_name in module_loads:
        module_name = str(module_name or "").strip()
        if module_name:
            if normalized_run_mode == "coadsorbate_placement" and _module_name_is_orca(module_name):
                continue
            lines.append(f"module load {module_name}")

    activate_path = str(venv_activate_path or "").strip()
    if activate_path:
        lines.append(f". {activate_path}")

    for entry in pythonpath_entries:
        entry = str(entry or "").strip()
        if entry:
            lines.append(f"export PYTHONPATH={entry}${{PYTHONPATH:+:$PYTHONPATH}}")
    lines.append('if [ -z "${ORCADIR:-}" ] && command -v orca_2mkl >/dev/null 2>&1; then export ORCADIR="$(dirname "$(command -v orca_2mkl)")"; fi')

    lines.extend(
        [
            "",
            f"python_input={run_python_filename}",
            'HOME_DIRECTORY="$PWD"',
            f'REACTION_HUB_RUN_IN_PLACE={"1" if normalized_run_in_place else "0"}',
            'if [ "$REACTION_HUB_RUN_IN_PLACE" = "1" ]; then',
            '    TEMP_DIRECTORY="$HOME_DIRECTORY"',
            "else",
            f'    TEMP_DIRECTORY="{temp_directory_expr}"',
            "fi",
            'output="workflow.$SLURM_JOB_ID.log"',
            f'REACTION_HUB_PREVIOUS_JOB_IDS_CSV="{previous_ids_csv}"',
            f'REACTION_HUB_FULL_RSYNC="${{REACTION_HUB_FULL_RSYNC:-{1 if normalized_run_mode == "charge_transfer" else 0}}}"',
            "",
            "build_rsync_excludes() {",
            '    RSYNC_EXCLUDES=(--exclude="*.tmp*" --exclude="*.bas*" --exclude="dft_neb_calc_*/" --exclude="$output" --exclude="slurm-$SLURM_JOB_ID.out")',
            '    ids_csv="${REACTION_HUB_RSYNC_EXCLUDE_JOB_IDS:-$REACTION_HUB_PREVIOUS_JOB_IDS_CSV}"',
            '    ids_csv="${ids_csv//:/,}"',
            '    if [ -n "$ids_csv" ]; then',
            '        IFS=\",\" read -r -a ids <<< "$ids_csv"',
            '        for id in "${ids[@]}"; do',
            '            id="${id//[[:space:]]/}"',
            '            if [ -n "$id" ]; then',
            '                RSYNC_EXCLUDES+=(--exclude="$id" --exclude="$id/" --exclude="$id/*")',
            "            fi",
            "        done",
            "    fi",
            "}",
            "",
            "sync_item_to_temp() {",
            '    item="$1"',
            '    src="$HOME_DIRECTORY/$item"',
            '    dst="$TEMP_DIRECTORY/$item"',
            '    if [ ! -e "$src" ]; then',
            "        return",
            "    fi",
            '    dst_parent="${dst%/*}"',
            '    if [ "$dst_parent" != "$dst" ]; then',
            '        mkdir -p "$dst_parent"',
            "    fi",
            '    if [ -d "$src" ]; then',
            '        mkdir -p "$dst"',
            '        rsync -av "$src/" "$dst/" "${RSYNC_EXCLUDES[@]}"',
            "    else",
            '        rsync -av "$src" "$dst"',
            "    fi",
            "}",
            "",
            "sync_item_back_home() {",
            '    item="$1"',
            '    src="$TEMP_DIRECTORY/$item"',
            '    dst="$HOME_DIRECTORY/$item"',
            '    if [ ! -e "$src" ]; then',
            "        return",
            "    fi",
            '    dst_parent="${dst%/*}"',
            '    if [ "$dst_parent" != "$dst" ]; then',
            '        mkdir -p "$dst_parent"',
            "    fi",
            '    if [ -d "$src" ]; then',
            '        mkdir -p "$dst"',
            '        rsync -av "$src/" "$dst/" "${RSYNC_EXCLUDES[@]}"',
            "    else",
            '        rsync -av "$src" "$dst"',
            "    fi",
            "}",
            "",
            "sync_root_nwchem_items_to_temp() {",
            '    for path in "$HOME_DIRECTORY"/nwchem*; do',
            '        [ -e "$path" ] || continue',
            '        sync_item_to_temp "${path##*/}"',
            "    done",
            "}",
            "",
            "sync_root_nwchem_items_back_home() {",
            '    for path in "$TEMP_DIRECTORY"/nwchem*; do',
            '        [ -e "$path" ] || continue',
            '        sync_item_back_home "${path##*/}"',
            "    done",
            "}",
            "",
            "sync_minimal_inputs_to_temp() {",
            "    SYNC_INPUT_ITEMS=(",
            '        "$python_input"',
            '        "gpu-bind.sh"',
            '        "active_learning_neb_workflow.py"',
            '        "snapshot_tddft_workflow.py"',
            '        "cats"',
        '        "cplas"',
            '        "orca.xyz"',
            '        "input_structure.extxyz"',
            '        "input_structure_resolved.extxyz"',
            '        "input_structure_final.extxyz"',
            '        "input_band.extxyz"',
            '        "nwchem_final.extxyz"',
            '        "adsorbates"',
            '        "reaction_hub_final.xyz"',
            '        "reaction_hub_summary.json"',
            '        "workflow_root.txt"',
            '        "scratch"',
            '        "neb"',
            '        "scan_endpoints.db"',
            '        "iroot_scan_endpoints.csv"',
            '        "iroot_geometry_log.csv"',
            '        "DFT"',
            '        "TDDFT"',
            '        "TDDFT_active_learning_shared"',
            '        "scan_plots"',
            '        "nwchem"',
            '        "nwchem_property"',
            '        "nwchem_tddft"',
            "    )",
            '    for item in "${SYNC_INPUT_ITEMS[@]}"; do',
            '        sync_item_to_temp "$item"',
            "    done",
            "    sync_root_nwchem_items_to_temp",
            "}",
            "",
            "sync_minimal_outputs_back_home() {",
            "    SYNC_OUTPUT_ITEMS=(",
            '        "orca.xyz"',
            '        "input_structure.extxyz"',
            '        "input_structure_resolved.extxyz"',
            '        "input_structure_final.extxyz"',
            '        "nwchem_final.extxyz"',
            '        "adsorbates"',
            '        "reaction_hub_final.xyz"',
            '        "reaction_hub_summary.json"',
            '        "workflow_root.txt"',
            '        "scratch"',
            '        "neb"',
            '        "scan_endpoints.db"',
            '        "iroot_scan_endpoints.csv"',
            '        "iroot_geometry_log.csv"',
            '        "DFT"',
            '        "TDDFT"',
            '        "TDDFT_active_learning_shared"',
            '        "scan_plots"',
            '        "nwchem"',
            '        "nwchem_property"',
            '        "nwchem_tddft"',
            "    )",
            '    for item in "${SYNC_OUTPUT_ITEMS[@]}"; do',
            '        sync_item_back_home "$item"',
            "    done",
            "    sync_root_nwchem_items_back_home",
            "}",
            "",
            "cleanup() {",
            '    if [ "$REACTION_HUB_RUN_IN_PLACE" = "1" ]; then',
            "        return",
            "    fi",
            "    build_rsync_excludes",
            '    if [ "${REACTION_HUB_STAGED_SOURCE_DIRECTORY:-}" = "archived_source" ]; then',
            '        RSYNC_EXCLUDES+=(--exclude="archived_source" --exclude="archived_source/")',
            "    fi",
            '    if [ "$REACTION_HUB_FULL_RSYNC" = "1" ]; then',
            '        rsync -av "$TEMP_DIRECTORY/" "$HOME_DIRECTORY/" "${RSYNC_EXCLUDES[@]}" || true',
            "    else",
            "        sync_minimal_outputs_back_home || true",
            "    fi",
            '    if [ "${REACTION_HUB_STAGED_SOURCE_DIRECTORY:-}" = "archived_source" ]; then',
            '        rm -rf -- "$HOME_DIRECTORY/archived_source"',
            "    fi",
            "}",
            "trap cleanup EXIT",
            "",
            'if [ "$REACTION_HUB_RUN_IN_PLACE" != "1" ]; then',
            '    mkdir -p "$TEMP_DIRECTORY"',
            "    build_rsync_excludes",
            '    if [ "$REACTION_HUB_FULL_RSYNC" = "1" ]; then',
            '        rsync -av "$HOME_DIRECTORY/" "$TEMP_DIRECTORY/" "${RSYNC_EXCLUDES[@]}"',
            "    else",
            "        sync_minimal_inputs_to_temp",
            "    fi",
            "fi",
            'cd "$TEMP_DIRECTORY"',
            "",
            run_python_cmd,
        ]
    )

    return "\n".join(lines)


def _exec_checked(ssh_client: Any, command: str) -> str:
    if hasattr(ssh_client, "exec_checked"):
        return ssh_client.exec_checked(command)

    stdin, stdout, stderr = ssh_client.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"Remote command failed ({code}): {command}\n{err.strip() or out.strip()}")
    return out


def _upload_text(ssh_client: Any, remote_path: str, text: str):
    if hasattr(ssh_client, "upload_text"):
        ssh_client.upload_text(remote_path, text)
        return

    if hasattr(ssh_client, "run"):
        import base64
        import shlex

        path_q = shlex.quote(str(remote_path))
        if not text:
            ssh_client.run(f": > {path_q}")
            return
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        res = ssh_client.run(f"base64 -d > {path_q}", input_text=b64, force_tty=False)
        if res.returncode != 0:
            raise RuntimeError(
                f"Remote upload failed ({res.returncode}): {res.stderr.strip() or res.stdout.strip()}"
            )
        return

    if hasattr(ssh_client, "exec_checked"):
        import base64
        import shlex
        path_q = shlex.quote(str(remote_path))
        if not text:
            ssh_client.exec_checked(f": > {path_q}")
            return
        b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
        if len(b64) <= 24000:
            ssh_client.exec_checked(f"echo {shlex.quote(b64)} | base64 -d > {path_q}")
        else:
            chunk_size = 24000
            tmp_b64 = f"{remote_path}.b64.tmp"
            tmp_q = shlex.quote(tmp_b64)
            ssh_client.exec_checked(f": > {tmp_q}")
            for i in range(0, len(b64), chunk_size):
                chunk = b64[i : i + chunk_size]
                ssh_client.exec_checked(f"printf %s {shlex.quote(chunk)} >> {tmp_q}")
            ssh_client.exec_checked(f"base64 -d {tmp_q} > {path_q} && rm -f {tmp_q}")
        return

    if hasattr(ssh_client, "open_sftp"):
        sftp = ssh_client.open_sftp()
        try:
            with sftp.file(remote_path, "w") as handle:
                handle.write(text)
        finally:
            sftp.close()
        return

    raise TypeError("Unsupported SSH client interface; expected upload_text() or open_sftp().")


def _supports_batched_remote_stage(ssh_client: Any) -> bool:
    """Use a two-call submit path for OpenSSH-style clients."""
    return (
        hasattr(ssh_client, "exec_checked")
        and hasattr(ssh_client, "upload_text")
        and not hasattr(ssh_client, "open_sftp")
        and not bool(getattr(ssh_client, "force_tty", False))
    )


def _stage_remote_run_files_batched(
    *,
    ssh_client: Any,
    remote_run_dir: str,
    wrapper_filename: str,
    submit_filename: str,
    structure_text: str,
    structure_rel_path: str,
    wrapper_script: str,
    submission_script: str,
    extra_remote_files: dict[str, str] | None = None,
) -> None:
    """Create run dir + write all files + chmod in one remote command."""
    structure_path = normalize_run_path(f"{remote_run_dir}/{str(structure_rel_path or '').lstrip('/')}")
    wrapper_path = f"{remote_run_dir}/{wrapper_filename}"
    submit_path = f"{remote_run_dir}/{submit_filename}"
    stage_parts = [
        f"if [ -e {_q(remote_run_dir)} ]; then echo 'Run directory already exists' >&2; exit 32; fi",
        f"mkdir -p {_q(remote_run_dir)}",
    ]
    current_batch: list[str] = list(stage_parts)
    current_len: int = sum(len(p) for p in stage_parts)
    use_tty = bool(getattr(ssh_client, "force_tty", False))
    max_batch_len: int = 3000 if use_tty else 12000
    chunk_size: int = 2500 if use_tty else 10000

    def flush_batch() -> None:
        nonlocal current_batch, current_len
        if current_batch:
            _exec_checked(ssh_client, " && ".join(current_batch))
            current_batch = []
            current_len = 0

    core_files = [
        (structure_path, structure_text),
        (wrapper_path, wrapper_script),
        (submit_path, submission_script),
    ]
    for target_path, content in core_files:
        if len(content) > chunk_size:
            flush_batch()
            _upload_text(ssh_client, target_path, content)
        else:
            item_part = f"printf %s {_q(content)} > {_q(target_path)}"
            if current_len + len(item_part) > max_batch_len:
                flush_batch()
            current_batch.append(item_part)
            current_len += len(item_part)

    for rel_path, content in (extra_remote_files or {}).items():
        normalized_rel = normalize_run_path(rel_path).lstrip("/")
        if not normalized_rel:
            continue
        remote_path = str(PurePosixPath(remote_run_dir) / normalized_rel)
        parent_part = f"mkdir -p {_q(str(PurePosixPath(remote_path).parent))}"

        if len(content) > chunk_size:
            flush_batch()
            _exec_checked(ssh_client, parent_part)
            _upload_text(ssh_client, remote_path, content)
        else:
            item_parts = [
                parent_part,
                f"printf %s {_q(content)} > {_q(remote_path)}",
            ]
            item_len = sum(len(p) for p in item_parts)
            if current_len + item_len > max_batch_len:
                flush_batch()
            current_batch.extend(item_parts)
            current_len += item_len

    chmod_part = f"chmod 755 {_q(submit_path)}"
    if current_len + len(chmod_part) > max_batch_len:
        flush_batch()
    current_batch.append(chmod_part)
    flush_batch()


def _default_ssh_client_factory(system: HPCSystem) -> Callable[[], Any]:
    def _factory():
        timeout = settings.REACTION_HUB_SSH_TIMEOUT_SECONDS
        if str(system.name).upper() == "FALCON":
            timeout = max(120, timeout)
        return OpenSSHClient(
            host=system.host,
            port=system.port,
            username=system.username,
            ssh_key_path=system.ssh_key_path,
            timeout=timeout,
            force_tty=settings.REACTION_HUB_SSH_FORCE_TTY,
            batch_mode=settings.REACTION_HUB_SSH_BATCH_MODE,
            use_clean_bash=settings.REACTION_HUB_SSH_CLEAN_BASH,
            bash_path=settings.REACTION_HUB_SSH_BASH_PATH,
        )

    return _factory


def _q(path_or_text: str) -> str:
    return shlex.quote(str(path_or_text))


def _ssh_runtime_context(system: HPCSystem) -> str:
    key_info = system.ssh_key_path or "<agent/default>"
    return (
        f"hpc_system={system.name} target={system.username}@{system.host}:{system.port} "
        f"key={key_info} force_tty={settings.REACTION_HUB_SSH_FORCE_TTY} "
        f"batch_mode={settings.REACTION_HUB_SSH_BATCH_MODE} "
        f"clean_bash={settings.REACTION_HUB_SSH_CLEAN_BASH} "
        f"bash_path={settings.REACTION_HUB_SSH_BASH_PATH} "
        f"timeout={settings.REACTION_HUB_SSH_TIMEOUT_SECONDS}s"
    )


def _submitted_structure_cache_path(*, run: WorkflowRun, artifact_rel_path: str) -> Path:
    cache_root = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()
    run_key = str(getattr(run, "run_uid", "") or f"run_{run.pk}")
    rel = str(artifact_rel_path or "").replace("\\", "/").lstrip("/")
    if not rel:
        rel = "submitted_structure"
    return cache_root / run_key / Path(rel)


def _store_initial_structure_artifact(
    *,
    run: WorkflowRun,
    job: JobRecord | None,
    structure_text: str,
    remote_run_dir: str,
    artifact_rel_path: str,
) -> None:
    normalized_rel_path = normalize_run_path(str(artifact_rel_path or "").replace("\\", "/").lstrip("/"))
    if not normalized_rel_path:
        normalized_rel_path = "input_structure.extxyz"
    remote_path = normalize_run_path(f"{remote_run_dir}/{normalized_rel_path}")
    metadata: dict[str, Any] = {
        "source": "submission",
        "is_initial_structure": True,
        "remote_path": remote_path,
    }

    try:
        cache_path = _submitted_structure_cache_path(run=run, artifact_rel_path=normalized_rel_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(str(structure_text), encoding="utf-8")
        metadata["local_cache_path"] = str(cache_path)
    except Exception:
        # Submission should not fail solely because local cache mirroring failed.
        pass

    Artifact.objects.update_or_create(
        workflow_run=run,
        path=normalized_rel_path,
        defaults={
            "job": job,
            "kind": ArtifactKind.STRUCTURE,
            "label": f"structure:{normalized_rel_path}",
            "metadata": metadata,
        },
    )


def submit_slurm_workflow_run(
    *,
    workflow: WorkflowDefinition,
    hpc_system: HPCSystem,
    settings_profile: CalculationSettingsProfile,
    run_name: str,
    structure_xyz: str,
    remote_runs_root: str,
    scratch_base: str,
    account: str,
    partition: str,
    qos: str,
    reservation: str = "",
    dependency_job_id: str = "",
    walltime: str,
    nodes: int,
    tasks_per_node: int,
    cpus_per_task: int,
    module_loads: list[str],
    venv_activate_path: str,
    pythonpath_entries: list[str],
    fmax: float = 0.01,
    active_learning_optimizer: str = "bfgs",
    qm_backend: str = "orca",
    target_wavelength_nm: float | None = None,
    spin_multiplicity: int | None = None,
    system_charge: int = 0,
    n_iter: int,
    use_tddft_optimisation: bool,
    run_mode: str,
    scan_config: dict[str, Any] | None = None,
    single_constraints_config: dict[str, Any] | None = None,
    charge_transfer_config: dict[str, Any] | None = None,
    neb_config: dict[str, Any] | None = None,
    coadsorbate_config: dict[str, Any] | None = None,
    alloy_config: dict[str, Any] | None = None,
    slurm_preset: str = "custom",
    submitted_by: str = "",
    cpu_freq: str = "",
    slurm_memory: str = "",
    structure_metadata: dict[str, Any] | None = None,
    ssh_client_factory: Callable[[], Any] | None = None,
    workflow_tasks_per_node: int | None = None,
) -> SubmissionResult:
    system_charge = int(system_charge)
    dependency_job_id = str(dependency_job_id or "").strip()
    if dependency_job_id and not dependency_job_id.isdigit():
        raise ValueError("dependency_job_id must be a Slurm numeric job id.")
    if hpc_system.scheduler_type != "slurm":
        raise ValueError("Only slurm scheduler type is supported by submit flow.")
    if not workflow.code_path:
        raise ValueError("Selected workflow has empty code_path.")
    if not structure_xyz.strip():
        raise ValueError("structure_xyz is empty.")
    if run_mode not in {"single_optimisation", "tddft_bond_scan", "charge_transfer", "active_learning_neb", "coadsorbate_placement", "alloy_growth"}:
        raise ValueError("Unsupported run_mode.")
    if hpc_system.name.strip().casefold() == "trh408":
        if not str(venv_activate_path or "").strip():
            venv_activate_path = getattr(settings, "REACTION_HUB_SUBMIT_TRH408_VENV_ACTIVATE_PATH", "/home/hpc_user/gpu_venv/mace_gpu/bin/activate")
        if not any(str(entry or "").strip() for entry in pythonpath_entries):
            pythonpath_entries = list(getattr(settings, "REACTION_HUB_SUBMIT_TRH408_PYTHONPATH", ["/home/hpc_user/ORCA/cplas", "/home/hpc_user/ORCA/ase", "/home/hpc_user/ORCA/carmm"]))
    normalized_qm_backend = str(qm_backend or "orca").strip().lower() or "orca"
    if normalized_qm_backend not in {"orca", "nwchem"}:
        raise ValueError("Unsupported QM backend for Slurm submission. Choose ORCA or NWChem.")
    if normalized_qm_backend == "nwchem" and run_mode != "single_optimisation":
        raise ValueError("NWChem Slurm submission is currently limited to single optimisation test jobs.")
    structure_artifact_rel_path = _structure_artifact_rel_path_for_mode(run_mode)

    workflow_module = _module_path_from_code_path(workflow.code_path)
    workflow_is_tddft_mlneb = _module_name_is_tddft_mlneb(workflow_module)
    requested_neb_backend = str((neb_config or {}).get("backend") or "").strip().lower()
    workflow_is_dft_neb = run_mode == "active_learning_neb" and requested_neb_backend in {"dft", "tddft"}
    if run_mode in {"coadsorbate_placement", "alloy_growth"}:
        basis_keyword = "lanl2dz"
        xc_functional = "pbe0"
        solvent = _normalize_solvent_for_workflow(settings_profile.solvent)
        active_learning_optimizer = "bfgs"
    elif run_mode == "active_learning_neb":
        if workflow_is_tddft_mlneb or workflow_is_dft_neb:
            basis_keyword = _normalize_basis_for_backend(settings_profile.basis_set, normalized_qm_backend)
            xc_functional = _normalize_xc_for_backend(settings_profile.xc_functional, normalized_qm_backend)
            solvent = _normalize_solvent_for_workflow(settings_profile.solvent)
        else:
            basis_keyword = "def2-svp"
            xc_functional = "pbe0"
            solvent = None
        active_learning_optimizer = _normalize_active_learning_optimizer_for_workflow(active_learning_optimizer)
    else:
        basis_keyword = _normalize_basis_for_backend(settings_profile.basis_set, normalized_qm_backend)
        xc_functional = _normalize_xc_for_backend(settings_profile.xc_functional, normalized_qm_backend)
        solvent = _normalize_solvent_for_workflow(settings_profile.solvent)
        active_learning_optimizer = _normalize_active_learning_optimizer_for_workflow(active_learning_optimizer)
    qm_backend_settings = _settings_profile_backend_settings(settings_profile, normalized_qm_backend)
    hpc_key = str(getattr(hpc_system, "name", "") or "").strip().lower()
    submission_slurm_memory = ""
    submission_nwchem_mpi_ranks: int | None = None
    if slurm_memory:
        submission_slurm_memory = str(slurm_memory).strip()
    elif hpc_key == "falcon":
        submission_slurm_memory = str(
            getattr(settings, "REACTION_HUB_SUBMIT_FALCON_MEMORY", _FALCON_DEFAULT_SLURM_MEMORY)
            or _FALCON_DEFAULT_SLURM_MEMORY
        ).strip()
    falcon_workflow_tasks: int | None = None
    if workflow_tasks_per_node is not None:
        falcon_workflow_tasks = int(workflow_tasks_per_node)
    elif hpc_key == "falcon":
        atom_count = _atom_count_from_xyz_text(structure_xyz)
        cpus_per_node = int(tasks_per_node) * int(cpus_per_task)
        if atom_count is not None and atom_count <= 30:
            falcon_workflow_tasks = min(cpus_per_node, 48)
        elif cpus_per_node >= 192:
            falcon_workflow_tasks = 96
        else:
            falcon_workflow_tasks = int(tasks_per_node)
    if normalized_qm_backend == "nwchem" and hpc_key == "falcon":
        memory_mb = _slurm_memory_mb(submission_slurm_memory) or _slurm_memory_mb(_FALCON_DEFAULT_SLURM_MEMORY)
        requested_ranks = int(os.environ.get("NWCHEM_MPI_RANKS") or 0)
        if requested_ranks <= 0:
            requested_ranks = int(nodes) * int(tasks_per_node) * int(cpus_per_task)
        submission_nwchem_mpi_ranks = max(1, requested_ranks)
        qm_backend_settings["memory_mb"] = max(1, int(memory_mb or 102400) // submission_nwchem_mpi_ranks)
    elif normalized_qm_backend == "orca" and hpc_key == "falcon":
        gb_match = re.fullmatch(r"\s*(\d+)\s*G(?:B)?\s*", str(submission_slurm_memory or ""), re.IGNORECASE)
        mem_target = int(gb_match.group(1)) * 1000 if gb_match else (_slurm_memory_mb(submission_slurm_memory) or _slurm_memory_mb(_FALCON_DEFAULT_SLURM_MEMORY))
        orca_cpus = falcon_workflow_tasks if falcon_workflow_tasks is not None else (
            96 if cpus_per_node >= 192 else cpus_per_node
        )
        if mem_target and orca_cpus > 0:
            qm_backend_settings["memory_per_cpu_mb"] = max(1000, int(round(mem_target * 0.70) // orca_cpus))
    effective_target_wavelength_nm = (
        float(target_wavelength_nm) if target_wavelength_nm is not None else 520.0
    )
    al_steps_per_iter = _positive_int(
        getattr(settings, "REACTION_HUB_SUBMIT_AL_STEPS_PER_ITER", 200),
        default=200,
    )
    if run_mode == "single_optimisation":
        # Keep single optimisations uncapped by default even when scan workflows
        # retain a finite global cap.
        max_al_steps_per_iter = _optional_positive_int(
            getattr(
                settings,
                "REACTION_HUB_SUBMIT_MAX_AL_STEPS_PER_ITER_SINGLE_OPTIMISATION",
                None,
            ),
            default=None,
        )
    else:
        max_al_steps_per_iter = _optional_positive_int(
            getattr(settings, "REACTION_HUB_SUBMIT_MAX_AL_STEPS_PER_ITER", None),
            default=None,
        )
    effective_module_loads = list(module_loads or [])
    if normalized_qm_backend == "nwchem":
        effective_module_loads = [
            module_name
            for module_name in effective_module_loads
            if not _is_orca_module(module_name)
            and not _is_nwchem_module(module_name)
            and _module_name_key(module_name) != "other-software"
        ]
        if hpc_key == "falcon":
            _append_module_once(effective_module_loads, "intel")
    if hpc_key == "trh408":
        effective_module_loads = []
    extra_remote_files: dict[str, str] = {}
    if ((charge_transfer_config or {}).get("density_analysis") or {}).get("mep_state_tracking"):
        module_root = Path(settings.BASE_DIR).parent / "cats/multiwfn"
        extra_remote_files["state_tracker_runtime.py"] = (module_root / "state_tracker.py").read_text(encoding="utf-8")
        extra_remote_files["mep_state_tracker_runtime.py"] = (module_root / "mep_state_tracker.py").read_text(encoding="utf-8").replace(
            "from cats.multiwfn.state_tracker import (", "from state_tracker_runtime import (", 1
        )
    if ((charge_transfer_config or {}).get("density_analysis") or {}).get("ts_vibrations_gradient"):
        helper = Path(settings.BASE_DIR).parent / "cats/orca/routines/ts_vibrations_gradient.py"
        extra_remote_files["ts_vibrations_gradient_runtime.py"] = helper.read_text(encoding="utf-8")
    normalized_coadsorbate_config: dict[str, Any] | None = None
    normalized_alloy_config: dict[str, Any] | None = None
    normalized_scan_config: dict[str, Any] | None = None
    normalized_single_constraints_config: dict[str, Any] | None = None
    normalized_charge_transfer_config: dict[str, Any] | None = None
    normalized_neb_config: dict[str, Any] | None = None
    submission_gpus = 0
    submission_include_cpu_layout_directives: bool | None = None
    submission_include_ntasks: bool | None = None
    submission_include_tasks_per_node: bool | None = None
    submission_include_cpus_per_task: bool | None = None
    submission_gpu_directive = "gpus"
    submission_exclusive = False
    submission_job_name = "workflow"
    submission_include_cpu_freq_req = True
    submission_use_srun = False
    submission_srun_ntasks = 1
    submission_srun_gpus_per_task = 0
    submission_run_in_place = False
    submission_extra_env_exports: dict[str, str] = {}
    if hpc_key == "falcon":
        submission_include_cpu_freq_req = False
    if normalized_qm_backend == "nwchem" and hpc_key == "falcon":
        submission_include_ntasks = True
        submission_extra_env_exports["I_MPI_FABRICS"] = "ofi"
        if submission_nwchem_mpi_ranks is not None:
            submission_extra_env_exports["NWCHEM_MPI_RANKS"] = str(submission_nwchem_mpi_ranks)

    if charge_transfer_config:
        cfg = dict(charge_transfer_config or {})
        if run_mode not in {"single_optimisation", "charge_transfer"}:
            raise ValueError("Charge-transfer postprocessing is only supported for single optimisation jobs.")
        if normalized_qm_backend != "orca":
            raise ValueError("Charge-transfer analysis requires the ORCA backend.")
        source_working_directory = ""
        source_archive_directory = ""
        source_archive_directories: tuple[str, ...] = ()
        if run_mode == "charge_transfer":
            source_working_directory = normalize_run_path(
                str(cfg.get("source_working_directory") or "")
            )
            if not source_working_directory:
                raise ValueError("Charge-transfer analysis requires the source run working directory.")
            if re.match(r"^[A-Za-z]:/", source_working_directory):
                remapped = canonical_run_path(source_working_directory)
                if remapped.startswith(("/mnt/", "/work/", "/scratch/")):
                    source_working_directory = remapped
                else:
                    raise ValueError(
                        "The source run is stored at a local Windows path, not an HPC-visible directory. "
                        "Use a modern HPC run or remap the legacy source root before submitting CT analysis."
                    )
        fragments_0based = dict(cfg.get("fragments_0based") or {})
        fragments_1based = dict(cfg.get("fragments_1based") or {})
        needs_fragments = not cfg.get("ground_state_mulliken_only") and (
            not cfg.get("density_analysis_only") or cfg.get("ifct_only")
        )
        if needs_fragments and (not fragments_0based.get("1") or not fragments_0based.get("2")):
            raise ValueError("Charge-transfer analysis requires both nanoparticle and adsorbate fragments.")
        if needs_fragments and (not fragments_1based.get("1") or not fragments_1based.get("2")):
            raise ValueError("Charge-transfer Multiwfn fragment conversion is incomplete.")
        normalized_charge_transfer_config = {
            **cfg,
            "orca_basename": str(cfg.get("orca_basename") or "orca").strip() or "orca",
            "nroots": _positive_int(cfg.get("nroots"), default=400),
            "fragments_0based": fragments_0based,
            "fragments_1based": fragments_1based,
        }
        density_analysis = dict(cfg.get("density_analysis") or {})
        if density_analysis.get("enabled"):
            density_grid_quality = int(density_analysis.get("grid_quality") or 3)
            if density_grid_quality not in {1, 2, 3}:
                raise ValueError("Density-analysis grid_quality must be 1, 2, or 3.")
            density_states = [int(value) for value in density_analysis.get("excitation_numbers") or []]
            if any(value < 1 for value in density_states):
                raise ValueError("Density-analysis excitation numbers must be positive.")
            normalized_charge_transfer_config["density_analysis"] = {
                **density_analysis,
                "enabled": True,
                "grid_quality": density_grid_quality,
                "excitation_numbers": density_states,
                "include_nto": bool(density_analysis.get("include_nto", True)),
            }
        if source_working_directory:
            normalized_charge_transfer_config["source_working_directory"] = source_working_directory
        if source_archive_directory:
            normalized_charge_transfer_config["source_archive_directory"] = source_archive_directory
            normalized_charge_transfer_config["source_archive_directories"] = list(
                source_archive_directories
            )
            submission_extra_env_exports["REACTION_HUB_STAGED_SOURCE_DIRECTORY"] = "archived_source"
        if run_mode == "charge_transfer":
            submission_job_name = "multiwfn"
        submission_extra_env_exports.update(
            {
                "OMP_NUM_THREADS": "1",
                "OMP_STACKSIZE": "1500M",
                "MPLBACKEND": "Agg",
            }
        )

    if run_mode == "coadsorbate_placement":
        cfg = dict(coadsorbate_config or {})

        raw_adsorbates = list(cfg.pop("adsorbates", []) or [])
        adsorbate_paths: list[str] = []
        adsorbate_metadata: list[dict[str, Any]] = []
        for idx, entry in enumerate(raw_adsorbates, start=1):
            if not isinstance(entry, dict):
                continue
            xyz_text = str(entry.get("xyz") or "").strip()
            if not xyz_text:
                continue
            rel_path = normalize_run_path(f"adsorbates/adsorbate_{idx:03d}.xyz")
            extra_remote_files[rel_path] = xyz_text
            adsorbate_paths.append(rel_path)
            ads_meta = entry.get("metadata")
            adsorbate_metadata.append(ads_meta if isinstance(ads_meta, dict) else {})
        if not adsorbate_paths:
            raise ValueError("Co-adsorbate submission requires at least one resolved adsorbate structure.")
        mace_model_path = str(
            cfg.get("mace_model_path")
            or getattr(settings, "REACTION_HUB_COADSORBATE_MACE_MODEL_PATH", "")
        ).strip()
        if hpc_system.name.strip().casefold() == "trh408":
            trh_model_path = str(getattr(settings, "REACTION_HUB_SUBMIT_TRH408_MACE_MODEL_PATH", "/srv/research/mace_models/mace-omat-0-medium.model") or "/srv/research/mace_models/mace-omat-0-medium.model").strip()
            if not mace_model_path or "/work/PROJECT/GROUP/" in mace_model_path:
                mace_model_path = trh_model_path
            cfg["mace_enable_cueq"] = True
        else:
            if not mace_model_path:
                mace_model_path = str(getattr(settings, "REACTION_HUB_COADSORBATE_MACE_MODEL_PATH", "") or "").strip()
            if "mace_enable_cueq" not in cfg:
                cfg["mace_enable_cueq"] = True

        cfg["mace_model_path"] = mace_model_path
        cfg["adsorbate_paths"] = adsorbate_paths
        cfg["adsorbate_metadata"] = adsorbate_metadata
        normalized_coadsorbate_config = cfg

        submission_gpus = 1
        submission_use_srun = True
        submission_srun_ntasks = 1
        submission_srun_gpus_per_task = 1

        # TRH408 uses `--gres` plus explicit CPU/GPU allocation and direct Python launch.
        if hpc_key == "trh408":
            submission_include_cpu_layout_directives = False
            submission_include_tasks_per_node = False
            submission_include_cpus_per_task = True
            submission_gpu_directive = "gres"
            submission_exclusive = True
            submission_use_srun = False
            submission_srun_ntasks = 1
            submission_srun_gpus_per_task = 0

    if run_mode == "alloy_growth":
        normalized_alloy_config = dict(alloy_config or {})
        optimize_seed_only = bool(normalized_alloy_config.get("optimize_seed_only"))
        if not optimize_seed_only and not normalized_alloy_config.get("add_symbols"):
            raise ValueError("Alloy growth requires add_symbols.")
        if not optimize_seed_only and int(normalized_alloy_config.get("n_additions", 0)) <= 0:
            raise ValueError("Alloy growth requires positive n_additions.")
        submission_gpus = 1
        if hpc_key == "trh408":
            submission_include_cpu_layout_directives = False
            submission_include_tasks_per_node = False
            submission_include_cpus_per_task = True
            submission_gpu_directive = "gres"
            submission_exclusive = True

    if run_mode == "active_learning_neb":
        cfg = dict(neb_config or {})
        is_tddft_mlneb = bool(workflow_is_tddft_mlneb)
        final_image_strategy = str(cfg.get("final_image_strategy") or "explicit_final_image").strip().lower()
        if final_image_strategy == "bond_extension_from_initial":
            final_image_strategy = "set_bond_distance_from_initial"
        if final_image_strategy not in {"explicit_final_image", "set_bond_distance_from_initial", "dimer", "dimer_refine"}:
            raise ValueError("active_learning_neb received unsupported final_image_strategy.")
        if final_image_strategy == "dimer_refine":
            final_image_strategy = "dimer" 
        initial_band_text = str(cfg.get("initial_band_extxyz") or "").strip()
        if initial_band_text:
            extra_remote_files["input_band.extxyz"] = initial_band_text
            cfg["initial_band_path"] = "input_band.extxyz"
        final_structure_text = str(cfg.pop("final_structure_extxyz", "") or "").strip()
        evaluation_only = bool(cfg.get("evaluation_only", False))
        if final_image_strategy == "explicit_final_image" and not initial_band_text and not evaluation_only:
            if not final_structure_text:
                raise ValueError("active_learning_neb explicit final-image mode requires final_structure_extxyz payload.")
            extra_remote_files["input_structure_final.extxyz"] = final_structure_text
        elif final_structure_text:
            extra_remote_files["input_structure_final.extxyz"] = final_structure_text

        repo_root = Path(settings.BASE_DIR).parent
        neb_wf_path = repo_root / "cats" / "working_directory" / "active_learning_neb_workflow.py"
        if neb_wf_path.is_file():
            extra_remote_files["active_learning_neb_workflow.py"] = neb_wf_path.read_text(encoding="utf-8")
        catlearn_dimer_path = repo_root / "cats" / "working_directory" / "catlearn_dimer.py"
        if catlearn_dimer_path.is_file():
            extra_remote_files["catlearn_dimer.py"] = catlearn_dimer_path.read_text(encoding="utf-8")
        if cfg.get("seed_training_extxyz"):
            extra_remote_files["seed_training.extxyz"] = cfg["seed_training_extxyz"]

        endpoint_opt_fmax_raw = cfg.get("endpoint_opt_fmax")
        endpoint_opt_fmax = (
            None if endpoint_opt_fmax_raw is None or endpoint_opt_fmax_raw == "" else float(endpoint_opt_fmax_raw)
        )
        break_bond_i_raw = cfg.get("break_bond_i")
        break_bond_j_raw = cfg.get("break_bond_j")
        break_bond_i = None if break_bond_i_raw is None or break_bond_i_raw == "" else int(break_bond_i_raw)
        break_bond_j = None if break_bond_j_raw is None or break_bond_j_raw == "" else int(break_bond_j_raw)

        normalized_neb_config = {
            "initial_band_extxyz": initial_band_text,
            "initial_band_path": "input_band.extxyz" if initial_band_text else None,
            "evaluation_only": evaluation_only,
            "final_image_strategy": final_image_strategy,
            "optimize_initial_image": bool(cfg.get("optimize_initial_image", False)),
            "optimize_final_image": bool(cfg.get("optimize_final_image", False)),
            "endpoint_opt_steps": int(cfg.get("endpoint_opt_steps", 200)),
            "endpoint_opt_fmax": endpoint_opt_fmax,
            "break_bond_i": break_bond_i,
            "break_bond_j": break_bond_j,
            "set_bond_distance_A": float(cfg.get("set_bond_distance_A", cfg.get("break_bond_extension_A", 2.5))),
            "break_bond_fix": float(cfg.get("break_bond_fix", 0.5)),
            "break_assert_broken": bool(cfg.get("break_assert_broken", True)),
            "break_cutoff_scale": float(cfg.get("break_cutoff_scale", 1.2)),
            "backend": str(cfg.get("backend") or ("tddft_mlneb" if is_tddft_mlneb else "mace")).strip().lower(),
            "atom_shift_per_image_A": float(cfg.get("atom_shift_per_image_A", 0.2)),
            "min_images": int(cfg.get("min_images", 7)),
            "max_images": int(cfg.get("max_images", 21)),
            "spring_constant": float(cfg.get("spring_constant", 0.1)),
            "max_steps": int(cfg.get("max_steps", 300)),
            "climb": bool(cfg.get("climb", True)),
            "split_on_minimum": bool(cfg.get("split_on_minimum", True)),
            "minimum_energy_window_eV": float(cfg.get("minimum_energy_window_eV", 0.02)),
            "fixed_symbols": [
                str(symbol).strip()
                for symbol in cfg.get("fixed_symbols", ())
                if str(symbol).strip()
            ],
            "mace_model_path": str(cfg.get("mace_model_path") or "").strip(),
            "mace_device": str(cfg.get("mace_device") or "cuda").strip() or "cuda",
            "mace_default_dtype": str(cfg.get("mace_default_dtype") or "float64").strip() or "float64",
            "mace_dispersion": bool(cfg.get("mace_dispersion", False)),
            "use_regular_endpoint_optimisation_for_mace": bool(
                cfg.get("use_regular_endpoint_optimisation_for_mace", True)
            ),
            "catlearn_min_data": int(cfg.get("catlearn_min_data", 3)),
            "catlearn_unc_convergence": float(cfg.get("catlearn_unc_convergence", 0.05)),
            "catlearn_max_unc": float(cfg.get("catlearn_max_unc", 0.03)),
            "catlearn_steps": int(cfg.get("catlearn_steps", 200)),
            "catlearn_ml_steps": int(cfg.get("catlearn_ml_steps", 500)),
            "catlearn_dtrust": float(cfg.get("catlearn_dtrust", 2.0)),
            "catlearn_guesses_per_cycle": int(cfg.get("catlearn_guesses_per_cycle", 10)),
            "dft_evaluation_timeout_seconds": int(cfg.get("dft_evaluation_timeout_seconds", 21600)),
            "dft_min_distance_scale": float(cfg.get("dft_min_distance_scale", 0.55)),
            "dft_max_force_eV_A": float(cfg.get("dft_max_force_eV_A", 50.0)),
            "dft_max_energy_drop_eV": float(cfg.get("dft_max_energy_drop_eV", 20.0)),
            "tddft_scan_nroots": int(cfg.get("tddft_scan_nroots", 50) or 50),
            "final_structure_path": (
                "input_structure_final.extxyz"
                if final_image_strategy == "explicit_final_image"
                else ""
            ),
            "final_structure_metadata": cfg.get("final_structure_metadata", {}),
            "refine_ts_with_dimer": bool(cfg.get("refine_ts_with_dimer", False)),
            "dimer_max_steps": int(cfg.get("dimer_max_steps", 100)),
            "dimer_separation": float(cfg.get("dimer_separation", 0.0001)),
            "dimer_trial_trans_step": float(cfg.get("dimer_trial_trans_step", 0.001)),
            "dimer_maximum_translation": float(cfg.get("dimer_maximum_translation", 0.05)),
            "dimer_f_rot_min": float(cfg.get("dimer_f_rot_min", 0.1)),
            "dimer_f_rot_max": float(cfg.get("dimer_f_rot_max", 1.0)),
            "dimer_max_num_rot": int(cfg.get("dimer_max_num_rot", 3)),
            "dimer_trial_angle": float(cfg.get("dimer_trial_angle", 0.785398)),
            "dimer_use_central_forces": bool(cfg.get("dimer_use_central_forces", True)),
            "initial_eigenmode": cfg.get("initial_eigenmode"),
            "displacement_vector": cfg.get("displacement_vector"),
            "dimer_neighbor_shells": cfg.get("dimer_neighbor_shells"),
            "vibration_neighbor_shells": cfg.get("vibration_neighbor_shells"),
            "use_catlearn_dimer": bool(cfg.get("use_catlearn_dimer", False)),
            "catlearn_dimer_polish_steps": int(cfg.get("catlearn_dimer_polish_steps", 25)),
            "prior_artifact_ids": cfg.get("prior_artifact_ids"),
            "rejected_artifacts": cfg.get("rejected_artifacts", []),
            "staged_seed_file": cfg.get("staged_seed_file") or ("seed_training.extxyz" if cfg.get("seed_training_extxyz") else None),
            "seed_training_extxyz": cfg.get("seed_training_extxyz"),
            "prior_trajectories": cfg.get("prior_trajectories"),
            "restart_from": cfg.get("restart_from"),
        }

        if normalized_neb_config["backend"] == "tddft" and not is_tddft_mlneb:
            if normalized_neb_config["final_image_strategy"] in {"dimer", "dimer_refine"}:
                root = cfg.get("initial_iroot", cfg.get("root_state", 5))
                if type(root) is not int or not 1 <= root <= normalized_neb_config["tddft_scan_nroots"]:
                    raise ValueError("initial_iroot must be between 1 and tddft_scan_nroots")
                normalized_neb_config.update(
                    initial_iroot=root,
                    target_wavelength_nm=float(cfg.get("target_wavelength_nm") or target_wavelength_nm or 520.0),
                    fragments=cfg.get("fragments"),
                )
            else:
                normalized_neb_config["backend"] = "tddft_mlneb"
        if is_tddft_mlneb and normalized_neb_config["backend"] != "tddft_mlneb":
            raise ValueError("TDDFT MLNEB workflow requires the 'tddft_mlneb' backend.")
        if (not is_tddft_mlneb) and normalized_neb_config["backend"] not in {"mace", "dft", "tddft"}:
            raise ValueError("active_learning_neb workflow supports only 'mace', 'dft', or 'tddft' backend.")
        if normalized_neb_config["final_image_strategy"] == "set_bond_distance_from_initial":
            if normalized_neb_config["break_bond_i"] is None or normalized_neb_config["break_bond_j"] is None:
                raise ValueError(
                    "active_learning_neb set-bond-distance mode requires break_bond_i and break_bond_j."
                )

        if normalized_neb_config["backend"] == "mace":
            effective_module_loads = [
                module for module in effective_module_loads if not _is_orca_module(module)
            ]
            submission_gpus = 1
            submission_use_srun = True
            submission_srun_ntasks = 1
            submission_srun_gpus_per_task = 1

        if normalized_neb_config["backend"] == "mace" and hpc_key == "trh408":
            submission_include_cpu_layout_directives = False
            submission_include_tasks_per_node = False
            submission_include_cpus_per_task = True
            submission_gpu_directive = "gres"
            submission_exclusive = True
            submission_use_srun = False
            submission_srun_ntasks = 1
            submission_srun_gpus_per_task = 0

    if run_mode == "single_optimisation":
        cfg = dict(single_constraints_config or {})
        fixed_symbols = [
            str(symbol).strip()
            for symbol in cfg.get("fixed_symbols", ())
            if str(symbol).strip()
        ]
        bond_pair_indices = cfg.get("bond_pair_indices")
        normalized_bond_pair_indices = None
        if isinstance(bond_pair_indices, (list, tuple)) and len(bond_pair_indices) == 2:
            normalized_bond_pair_indices = [
                int(bond_pair_indices[0]),
                int(bond_pair_indices[1]),
            ]
        target_distance = cfg.get("target_distance")
        normalized_target_distance = (
            None if target_distance is None else float(target_distance)
        )
        single_fix = cfg.get("fix")
        normalized_fix = 0.5 if single_fix is None else float(single_fix)
        final_tddft_nroots_mode = str(
            cfg.get("final_tddft_nroots_mode") or "manual"
        ).strip().lower()
        final_tddft_nroots = _positive_int(cfg.get("final_tddft_nroots"), default=50)
        if final_tddft_nroots_mode == "scaled_by_atoms":
            final_tddft_nroots = _scaled_final_tddft_nroots(structure_xyz, default=50)
        else:
            final_tddft_nroots_mode = "manual"
        normalized_single_constraints_config = {
            "fixed_symbols": fixed_symbols,
            "bond_pair_indices": normalized_bond_pair_indices,
            "target_distance": normalized_target_distance,
            "fix": normalized_fix,
            "final_tddft_spectrum": bool(
                cfg.get("final_tddft_spectrum", False)
                or normalized_charge_transfer_config
            ),
            "snapshot_single_point": bool(cfg.get("snapshot_single_point", False)),
            "cluster_tddft_difference_spectrum": bool(
                cfg.get("cluster_tddft_difference_spectrum", False)
            ),
            "charge_transfer_spectrum": bool(normalized_charge_transfer_config),
            "final_tddft_nroots": final_tddft_nroots,
            "final_tddft_nroots_mode": final_tddft_nroots_mode,
        }
        if use_tddft_optimisation:
            normalized_single_constraints_config["tddft_scan_nroots"] = _positive_int(
                cfg.get("tddft_scan_nroots"), default=12
            )
            initial_iroot = cfg.get("initial_iroot")
            if initial_iroot is not None:
                initial_iroot = int(initial_iroot)
                if not 1 <= initial_iroot <= normalized_single_constraints_config["tddft_scan_nroots"]:
                    raise ValueError("initial_iroot must be between 1 and tddft_scan_nroots")
                normalized_single_constraints_config["initial_iroot"] = initial_iroot
        gradient_root = cfg.get("final_tddft_gradient_root")
        initial_wavefunction_artifact_id = cfg.get("initial_wavefunction_artifact_id")
        if initial_wavefunction_artifact_id is not None:
            if type(initial_wavefunction_artifact_id) is not int or initial_wavefunction_artifact_id <= 0:
                raise ValueError("initial_wavefunction_artifact_id must be a positive integer")
            if not normalized_single_constraints_config["snapshot_single_point"] or qm_backend != "orca":
                raise ValueError("Initial wavefunction requires an ORCA single-point snapshot")
            seed = Artifact.objects.select_related("job").get(pk=initial_wavefunction_artifact_id)
            relative = PurePosixPath(seed.path)
            if relative.is_absolute() or ".." in relative.parts or relative.name != "orca.gbw":
                raise ValueError("Initial wavefunction must be a registered relative orca.gbw artifact")
            if not seed.job or seed.job.workflow_run_id != seed.workflow_run_id or seed.job.hpc_system_id != hpc_system.pk:
                raise ValueError("Initial wavefunction must belong to its recorded job on the target HPC")
            normalized_single_constraints_config["initial_wavefunction_artifact_id"] = seed.pk
        if gradient_root is not None:
            if type(gradient_root) is not int or not 1 <= gradient_root <= final_tddft_nroots:
                raise ValueError("final_tddft_gradient_root must be between 1 and final_tddft_nroots")
            if not normalized_single_constraints_config["snapshot_single_point"] or qm_backend != "orca":
                raise ValueError("final_tddft_gradient_root requires an ORCA single-point snapshot")
            normalized_single_constraints_config["final_tddft_gradient_root"] = gradient_root
            helper = Path(settings.BASE_DIR).parent / "cats/working_directory/tddft_active_learning_workflow.py"
            extra_remote_files["snapshot_tddft_workflow.py"] = helper.read_text(encoding="utf-8")
        if normalized_charge_transfer_config:
            # The CT analysis consumes this job's final TDDFT states, so both
            # stages must describe the same number of roots.
            normalized_charge_transfer_config["nroots"] = final_tddft_nroots

    if run_mode == "tddft_bond_scan":
        cfg = dict(scan_config or {})
        target_distance_raw = cfg.get("target_distance")
        normalized_target_distance = (
            None if target_distance_raw is None or target_distance_raw == "" else float(target_distance_raw)
        )
        restart_extend_increment_raw = cfg.get("restart_extend_increment_A")
        normalized_restart_extend_increment = (
            None
            if restart_extend_increment_raw is None or restart_extend_increment_raw == ""
            else float(restart_extend_increment_raw)
        )
        normalized_scan_config = {
            "bond_i": int(cfg.get("bond_i", 0)),
            "bond_j": int(cfg.get("bond_j", 1)),
            "increment": float(cfg.get("increment", 0.05)),
            "fix": float(cfg.get("fix", 0.5)),
            "fixed_symbols": [
                str(symbol).strip()
                for symbol in cfg.get("fixed_symbols", ())
                if str(symbol).strip()
            ],
            "target_distance": normalized_target_distance,
            "distance_change_fraction": float(cfg.get("distance_change_fraction", 1.0)),
            "enable_ts_smoothing": bool(cfg.get("enable_ts_smoothing", False)),
            "refine_points_each_side": int(cfg.get("refine_points_each_side", 0)),
            "restart_extend_steps": int(cfg.get("restart_extend_steps", 0) or 0),
            "restart_extend_increment_A": normalized_restart_extend_increment,
            "tddft_scan_nroots": int(cfg.get("tddft_scan_nroots", 50)),
        }

    if hpc_key == "trh408":
        submission_run_in_place = True
        submission_job_name = "testgpu"
        submission_include_cpu_freq_req = False

    remote_runs_root_text = str(remote_runs_root or "").strip()
    default_remote_runs_root = str(
        getattr(settings, "REACTION_HUB_SUBMIT_REMOTE_RUNS_ROOT", "/work/PROJECT/GROUP/{username}/reaction_hub_runs")
        or "/work/PROJECT/GROUP/{username}/reaction_hub_runs"
    ).strip()
    trh408_remote_runs_root = str(
        getattr(settings, "REACTION_HUB_SUBMIT_TRH408_REMOTE_RUNS_ROOT", "/home/{username}/reaction_hub_runs")
        or "/home/{username}/reaction_hub_runs"
    ).strip() or "/home/{username}/reaction_hub_runs"
    if hpc_key == "trh408":
        # Keep TRH408 usable even when browser-side HPC prefill is unavailable.
        if (
            (not remote_runs_root_text)
            or (remote_runs_root_text == default_remote_runs_root)
            or remote_runs_root_text.startswith("/work/PROJECT/GROUP/")
        ):
            remote_runs_root_text = trh408_remote_runs_root
    if not remote_runs_root_text:
        raise ValueError("remote_runs_root is required for Slurm submission.")
    expanded_root = remote_runs_root_text.replace("{username}", hpc_system.username).replace("$USER", hpc_system.username)
    effective_account = str(account or "").strip()
    effective_qos = str(qos or "").strip()
    effective_partition = str(partition or "").strip()
    reservations = getattr(settings, "REACTION_HUB_ACTIVE_RESERVATIONS", []) or []
    effective_reservation = resolve_active_reservation(
        hpc_key=hpc_key,
        partition=effective_partition,
        account=effective_account,
        qos=effective_qos,
        user_reservation=reservation,
    )
    reservation_entry = next(
        (
            entry for entry in reservations
            if isinstance(entry, dict) and str(entry.get("name") or "").casefold() == effective_reservation.casefold()
        ),
        None,
    )
    if reservation_entry and reservation_entry.get("default_for_cpu") and (
        hpc_key != reservation_entry["hpc_system"].casefold()
        or effective_partition.casefold() != reservation_entry["partition"].casefold()
        or effective_account.casefold() != reservation_entry["account"].casefold()
        or submission_gpus
    ):
        raise ValueError("Reservation requires eligible CPU resources.")
    if not effective_reservation and effective_qos.casefold() == "reservation":
        raise ValueError("Reservation QoS requires an active reservation.")
    if hpc_key == "trh408":
        if effective_partition == "gpu":
            effective_partition = str(getattr(settings, "REACTION_HUB_SUBMIT_TRH408_DEFAULT_PARTITION", "main") or "main").strip()
        default_account = str(getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_ACCOUNT", "") or "").strip()
        default_qos = str(getattr(settings, "REACTION_HUB_SUBMIT_DEFAULT_QOS", "") or "").strip()
        if effective_account and effective_account == default_account:
            effective_account = str(getattr(settings, "REACTION_HUB_SUBMIT_TRH408_DEFAULT_ACCOUNT", "") or "").strip()
        if effective_qos and effective_qos == default_qos:
            effective_qos = str(getattr(settings, "REACTION_HUB_SUBMIT_TRH408_DEFAULT_QOS", "") or "").strip()
    if run_mode == "charge_transfer":
        ground_state_mulliken_only = bool(
            (normalized_charge_transfer_config or {}).get("ground_state_mulliken_only")
        )
        if hpc_key == "falcon":
            nodes = 1
            tasks_per_node = int(tasks_per_node or 192)
            cpus_per_task = 1
            effective_partition = effective_partition or "htc_genoa"
            effective_qos = effective_qos or ""
            walltime = walltime or "12:00:00"
        else:
            requested_tasks_per_node = tasks_per_node
            nodes, tasks_per_node, cpus_per_task = 1, (1 if ground_state_mulliken_only else 8), 1
            if ground_state_mulliken_only:
                effective_partition = "standard"
                effective_qos = "reservation" if (effective_qos.casefold() == "reservation" or bool(effective_reservation)) else "short"
            elif effective_qos.casefold() == "short" and (
                (normalized_charge_transfer_config or {}).get("density_analysis", {}).get("mep_state_tracking")
            ):
                effective_partition = "standard"
                tasks_per_node = int(requested_tasks_per_node or 128)
            elif effective_qos.casefold() == "taskfarm":
                effective_partition = "standard"
                effective_qos = "taskfarm"
                tasks_per_node = 8
            elif effective_qos.casefold() == "standard":
                effective_partition = "standard"
                effective_qos = "standard"
                tasks_per_node = int(requested_tasks_per_node or 128)
            elif effective_qos.casefold() == "reservation" or bool(effective_reservation):
                effective_partition = "standard"
                effective_qos = "reservation"
                tasks_per_node = int(requested_tasks_per_node or 128)
            else:
                effective_partition = "serial"
                effective_qos = "serial"
                tasks_per_node = 8
            walltime = (
                ("00:45:00" if (effective_qos.casefold() == "reservation" or bool(effective_reservation)) else "00:20:00")
                if ground_state_mulliken_only
                else (
                    str(walltime or "04:00:00").strip()
                    if (effective_qos.casefold() in {"reservation", "short"} or bool(effective_reservation))
                    else str(
                        getattr(settings, "REACTION_HUB_CT_SERIAL_WALLTIME", "24:00:00")
                        or "24:00:00"
                    ).strip()
                )
            )
    effective_scratch_base = str(scratch_base or "").strip()
    if not effective_scratch_base:
        effective_scratch_base = "/tmp"
    if (not submission_run_in_place) and (not str(scratch_base or "").strip()):
        raise ValueError("scratch_base is required unless run_in_place is enabled.")
    remote_run_dir = str(PurePosixPath(expanded_root) / run_name)
    normalized_remote_run_dir = normalize_run_path(remote_run_dir)
    wrapper_filename = "run_workflow_submit.py"
    submit_filename = "submit_workflow.sh"
    wrapper_script = _build_wrapper_script(
        workflow_module=workflow_module,
        run_mode=run_mode,
        hpc_name=str(getattr(hpc_system, "name", "") or ""),
        n_iter=n_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        fmax=fmax,
        active_learning_optimizer=active_learning_optimizer,
        qm_backend=normalized_qm_backend,
        qm_backend_settings=qm_backend_settings,
        target_wavelength_nm=effective_target_wavelength_nm,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        al_steps_per_iter=al_steps_per_iter,
        max_al_steps_per_iter=max_al_steps_per_iter,
        scan_config=normalized_scan_config,
        single_constraints_config=normalized_single_constraints_config,
        charge_transfer_config=normalized_charge_transfer_config,
        neb_config=normalized_neb_config,
        coadsorbate_config=normalized_coadsorbate_config,
        alloy_config=normalized_alloy_config,
        workflow_tasks_per_node=falcon_workflow_tasks,
    )
    reservation_end = None
    if reservation_entry and reservation_entry.get("end_utc"):
        reservation_end = datetime.fromisoformat(reservation_entry["end_utc"].replace("Z", "+00:00"))
    submission_script = _build_submission_script(
        job_name=submission_job_name,
        run_python_filename=wrapper_filename,
        nodes=nodes,
        tasks_per_node=tasks_per_node,
        cpus_per_task=cpus_per_task,
        gpus=submission_gpus,
        include_cpu_layout_directives=submission_include_cpu_layout_directives,
        include_ntasks=submission_include_ntasks,
        include_tasks_per_node=submission_include_tasks_per_node,
        include_cpus_per_task=submission_include_cpus_per_task,
        gpu_directive=submission_gpu_directive,
        exclusive=submission_exclusive,
        use_srun=submission_use_srun,
        srun_ntasks=submission_srun_ntasks,
        srun_gpus_per_task=submission_srun_gpus_per_task,
        walltime=walltime,
        slurm_memory=submission_slurm_memory,
        account=effective_account,
        partition=effective_partition,
        qos=effective_qos,
        reservation=effective_reservation,
        module_loads=effective_module_loads,
        venv_activate_path=venv_activate_path,
        pythonpath_entries=pythonpath_entries,
        scratch_base=effective_scratch_base,
        cpu_freq=cpu_freq,
        include_cpu_freq_req=submission_include_cpu_freq_req,
        extra_env_exports=submission_extra_env_exports,
        run_mode=run_mode,
        run_in_place=submission_run_in_place,
        workflow_tasks_per_node=falcon_workflow_tasks,
    )

    create_client = ssh_client_factory or _default_ssh_client_factory(hpc_system)
    reservation_node_quota = _reservation_node_quota(effective_reservation)
    if reservation_node_quota and nodes > reservation_node_quota:
        raise ValueError(
            f"Job requests {nodes} nodes, exceeding reservation quota {reservation_node_quota}."
        )

    def stage_and_submit(client):
        nonlocal walltime
        if _supports_batched_remote_stage(client):
            _stage_remote_run_files_batched(
                ssh_client=client,
                remote_run_dir=remote_run_dir,
                wrapper_filename=wrapper_filename,
                submit_filename=submit_filename,
                structure_text=structure_xyz,
                structure_rel_path=structure_artifact_rel_path,
                wrapper_script=wrapper_script,
                submission_script=submission_script,
                extra_remote_files=extra_remote_files,
            )
        else:
            _exec_checked(
                client,
                f"if [ -e {_q(remote_run_dir)} ]; then echo 'Run directory already exists' >&2; exit 32; fi",
            )
            _exec_checked(client, f"mkdir -p {_q(remote_run_dir)}")
            _upload_text(client, f"{remote_run_dir}/{structure_artifact_rel_path}", structure_xyz)
            _upload_text(client, f"{remote_run_dir}/{wrapper_filename}", wrapper_script)
            _upload_text(client, f"{remote_run_dir}/{submit_filename}", submission_script)
            _exec_checked(client, f"chmod 755 {_q(f'{remote_run_dir}/{submit_filename}')}")

        if not _supports_batched_remote_stage(client):
            for rel_path, content in extra_remote_files.items():
                normalized_rel = normalize_run_path(rel_path).lstrip("/")
                if not normalized_rel:
                    continue
                remote_path = str(PurePosixPath(remote_run_dir) / normalized_rel)
                remote_parent = str(PurePosixPath(remote_path).parent)
                _exec_checked(client, f"mkdir -p {_q(remote_parent)}")
                _upload_text(client, remote_path, content)

        ct = normalized_charge_transfer_config or {}
        ts = (ct.get("density_analysis") or {}).get("ts_vibrations_gradient") or {}
        frequency_seed_id = ts.get("initial_wavefunction_artifact_id")
        initial_wavefunction_id = frequency_seed_id or (normalized_single_constraints_config or {}).get("initial_wavefunction_artifact_id")
        if initial_wavefunction_id:
            seed = Artifact.objects.select_related("job").get(pk=initial_wavefunction_id)
            relative = PurePosixPath(seed.path)
            job = seed.job
            if frequency_seed_id:
                if (
                    not ts.get("ground_state_only") or seed.workflow_run_id != ct.get("source_run_id")
                    or relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".gbw"
                ):
                    raise ValueError("NumFreq requires a registered GBW from the source run")
                if job is None:
                    scheduler_id = (seed.metadata or {}).get("source_scheduler_job_id")
                    if scheduler_id:
                        job = seed.workflow_run.jobs.filter(scheduler_job_id=scheduler_id).first()
                if not job or job.workflow_run_id != seed.workflow_run_id or job.hpc_system_id != hpc_system.pk:
                    raise ValueError("NumFreq wavefunction source job must be on the target HPC")
                roots = (seed.workflow_run.working_directory, job.remote_workdir)
                seed_paths = [PurePosixPath(root) / relative for root in roots if root and PurePosixPath(root).is_absolute()]
                if not seed_paths:
                    raise ValueError("NumFreq wavefunction source has no recorded HPC directory")
            else:
                seed_paths = [PurePosixPath(job.remote_workdir) / relative]
            target_directory = str(PurePosixPath(remote_run_dir) / ("s0_numfreq" if frequency_seed_id else "DFT/final_tddft_spectrum"))
            target_path = str(PurePosixPath(target_directory) / "initial.gbw")
            copy_commands = " || ".join(
                f"(test -s {_q(str(candidate))} && cp -p -- {_q(str(candidate))} {_q(target_path)})"
                for candidate in seed_paths
            )
            _exec_checked(client, f"mkdir -p {_q(target_directory)} && ({copy_commands})")

        source_archive_directory = str(
            (normalized_charge_transfer_config or {}).get("source_archive_directory") or ""
        ).strip()
        if source_archive_directory:
            source_archive_directories = list(
                (normalized_charge_transfer_config or {}).get("source_archive_directories")
                or [source_archive_directory]
            )
            staged_source_directory = str(PurePosixPath(remote_run_dir) / "archived_source")
            orca_basename = str(
                (normalized_charge_transfer_config or {}).get("orca_basename") or "orca"
            ).strip() or "orca"
            archive_stage_command = "\n".join(
                [
                    "set -e",
                    "archive_roots=("
                    + " ".join(_q(str(path)) for path in source_archive_directories)
                    + ")",
                    f"staged_root={_q(staged_source_directory)}",
                    f"orca_basename={_q(orca_basename)}",
                    'source_root=""',
                    'for archive_root in "${archive_roots[@]}"; do',
                    '    for candidate in "$archive_root/DFT/final_tddft_spectrum" "$archive_root"; do',
                    '        if [ -f "$candidate/$orca_basename.out" ]; then source_root="$candidate"; break 2; fi',
                    "    done",
                    "done",
                    'if [ -z "$source_root" ]; then echo "Missing archived ORCA output" >&2; exit 34; fi',
                    'mkdir -p "$staged_root"',
                    'cp -p -- "$source_root/$orca_basename.out" "$staged_root/$orca_basename.out"',
                    'for name in "$orca_basename.molden.input" "$orca_basename.molden" "$orca_basename.gbw" "initial.gbw"; do',
                    '    if [ -f "$source_root/$name" ]; then cp -p -- "$source_root/$name" "$staged_root/$name"; fi',
                    "done",
                    'if [ ! -f "$staged_root/$orca_basename.molden.input" ] && [ ! -f "$staged_root/$orca_basename.molden" ] && [ ! -f "$staged_root/$orca_basename.gbw" ] && [ ! -f "$staged_root/initial.gbw" ]; then',
                    '    echo "Missing archived Molden file or ORCA wavefunction under $source_root" >&2; exit 35',
                    "fi",
                ]
            )
            _exec_checked(client, archive_stage_command)

        dependency_arg = f" --dependency=afterok:{dependency_job_id}" if dependency_job_id else ""
        time_arg = ""
        if reservation_end is not None:
            remaining_minutes = int((reservation_end - timezone.now()).total_seconds() // 60)
            if remaining_minutes < 1:
                raise ValueError("Reservation has less than one minute remaining.")
            hours, minutes, seconds = map(int, str(walltime).split(":"))
            if hours * 60 + minutes + bool(seconds) > remaining_minutes:
                walltime = f"{remaining_minutes // 60:02d}:{remaining_minutes % 60:02d}:00"
                time_arg = f" --time={walltime}"
        return _exec_checked(
            client,
            f"cd {_q(remote_run_dir)} && sbatch{dependency_arg}{time_arg} {_q(submit_filename)}",
        )

    submit_out = ""
    while not submit_out:
        should_wait = False
        client = None
        try:
            queue_limit = _submission_queue_limit(effective_qos, effective_partition)
            lock_context = transaction.atomic() if queue_limit or reservation_node_quota else nullcontext()
            with lock_context:
                if queue_limit or reservation_node_quota:
                    HPCSystem.objects.select_for_update().get(pk=hpc_system.pk)
                if reservation_end is not None and (reservation_end - timezone.now()).total_seconds() < 60:
                    raise ValueError("Reservation has less than one minute remaining.")
                client = create_client()
                if reservation_node_quota:
                    should_wait = not _reservation_has_capacity(
                        client,
                        reservation=effective_reservation,
                        username=hpc_system.username,
                        requested_nodes=nodes,
                        node_quota=reservation_node_quota,
                    )
                elif queue_limit:
                    should_wait = _active_qos_job_count(
                        client, username=hpc_system.username, qos=effective_qos, partition=effective_partition
                    ) >= queue_limit
                if not should_wait:
                    submit_out = stage_and_submit(client)
        except Exception as exc:
            raise RuntimeError(f"{exc}\nSubmission SSH context: {_ssh_runtime_context(hpc_system)}") from exc
        finally:
            if client is not None:
                client.close()
        if should_wait:
            time.sleep(
                max(
                    1.0,
                    float(getattr(settings, "REACTION_HUB_SHORT_QUEUE_POLL_SECONDS", 30.0)),
                )
            )

    match = _SBATCH_JOB_ID_RE.search(submit_out)
    if not match:
        raise RuntimeError(f"Unable to parse scheduler job id from sbatch output: {submit_out.strip()}")
    scheduler_job_id = match.group(1)
    submitted_at = timezone.now()
    submission_payload: dict[str, Any] = {
        "remote_runs_root": expanded_root,
        "scratch_base": effective_scratch_base,
        "nodes": nodes,
        "tasks_per_node": tasks_per_node,
        "cpus_per_task": cpus_per_task,
        "workflow_cpus_per_node": int((falcon_workflow_tasks if falcon_workflow_tasks is not None else tasks_per_node) * cpus_per_task),
        "workflow_total_cpus": int(nodes * (falcon_workflow_tasks if falcon_workflow_tasks is not None else tasks_per_node) * cpus_per_task),
        "walltime": walltime,
        "slurm_memory": submission_slurm_memory,
        "account": effective_account,
        "partition": effective_partition,
        "qos": effective_qos,
        "reservation": effective_reservation,
        "dependency": (
            {"type": "afterok", "scheduler_job_id": dependency_job_id}
            if dependency_job_id
            else {}
        ),
        "gpus": int(submission_gpus),
        "use_srun": bool(submission_use_srun),
        "srun_ntasks": int(submission_srun_ntasks),
        "srun_gpus_per_task": int(submission_srun_gpus_per_task),
        "run_in_place": bool(submission_run_in_place),
        "settings_profile_id": settings_profile.pk,
        "settings_profile_name": settings_profile.name,
        "xc_functional": settings_profile.xc_functional,
        "basis_set": settings_profile.basis_set,
        "solvent": settings_profile.solvent,
        "module_loads": effective_module_loads,
        "venv_activate_path": venv_activate_path,
        "pythonpath_entries": pythonpath_entries,
        "wrapper_script": wrapper_filename,
        "submit_script": submit_filename,
        "structure_artifact_path": structure_artifact_rel_path,
        "run_mode": run_mode,
        "n_iter": int(n_iter),
        "use_tddft_optimisation": bool(use_tddft_optimisation),
        "tags": ["charge_transfer"] if normalized_charge_transfer_config else [],
        "qm_backend": normalized_qm_backend,
        "qm_backend_settings": qm_backend_settings,
        "scan_config": normalized_scan_config or {},
        "single_constraints_config": normalized_single_constraints_config or {},
        "charge_transfer_config": normalized_charge_transfer_config or {},
        "neb_config": normalized_neb_config or {},
        "coadsorbate_config": normalized_coadsorbate_config or {},
        "alloy_config": normalized_alloy_config or {},
        "fmax": float(fmax),
        "active_learning_optimizer": active_learning_optimizer,
        "spin_multiplicity": None if spin_multiplicity is None else int(spin_multiplicity),
        "system_charge": system_charge,
        "al_steps_per_iter": int(al_steps_per_iter),
        "max_al_steps_per_iter": (
            None if max_al_steps_per_iter is None else int(max_al_steps_per_iter)
        ),
        "previous_scheduler_job_ids": [],
        "slurm_preset": str(slurm_preset or "custom"),
    }
    if run_mode == "tddft_bond_scan" or (
        run_mode == "active_learning_neb"
        and normalized_neb_config
        and normalized_neb_config.get("backend") == "tddft_mlneb"
    ):
        submission_payload["target_wavelength_nm"] = float(effective_target_wavelength_nm)

    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_uid=uuid.uuid4(),
        run_name=run_name,
        status="initiated",
        started_at=None,
        completed_at=None,
        submitted_by=submitted_by,
        working_directory=normalized_remote_run_dir,
        system_charge=system_charge,
        settings_profile=settings_profile,
        metadata={
            "submission": submission_payload,
            "structure": structure_metadata or {},
            "tags": ["charge_transfer"] if normalized_charge_transfer_config else [],
            "source_run_id": (
                normalized_charge_transfer_config.get("source_run_id")
                if normalized_charge_transfer_config
                else None
            ),
        },
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=hpc_system,
        scheduler_job_id=scheduler_job_id,
        job_name=run_name,
        status=JobStatus.INITIATED,
        submitted_at=submitted_at,
        remote_workdir=(
            normalized_remote_run_dir
            if submission_run_in_place
            else _resolve_remote_workdir(
                scratch_base=effective_scratch_base,
                username=hpc_system.username,
                scheduler_job_id=scheduler_job_id,
            )
        ),
        stdout_path=f"{normalized_remote_run_dir}/workflow.{scheduler_job_id}.log",
        scheduler_payload=submission_payload,
    )
    _store_initial_structure_artifact(
        run=run,
        job=job,
        structure_text=structure_xyz,
        remote_run_dir=normalized_remote_run_dir,
        artifact_rel_path=structure_artifact_rel_path,
    )

    return SubmissionResult(
        workflow_run_id=run.pk,
        job_record_id=job.pk,
        scheduler_job_id=scheduler_job_id,
        remote_run_dir=remote_run_dir,
    )


def _link_backfilled_dft_run(*, campaign_name: str, original_run_id: int, new_run: WorkflowRun) -> None:
    from workflows.models import Campaign, CampaignRun

    campaign = Campaign.objects.filter(name=campaign_name).first()
    if not campaign:
        return
    source = CampaignRun.objects.filter(campaign=campaign, workflow_run_id=original_run_id).first()
    CampaignRun.objects.update_or_create(
        campaign=campaign,
        workflow_run=new_run,
        defaults={
            "checkpoint_stage": 4,
            "reaction_step_key": source.reaction_step_key if source else "",
        },
    )

