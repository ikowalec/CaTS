from __future__ import annotations

import copy
from dataclasses import dataclass
from functools import partial
import json
from math import isfinite
import os
from pathlib import Path
import re
import shutil
from datetime import datetime, timezone
from typing import Optional

from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import read, write
from ase.units import Hartree

from cats.multiwfn.iroot_scoring import IRootCandidate
from cats.qm_backends.workflow import QMWorkflowBackend, get_workflow_backend
from cats.orca.run.orca_calculator import get_orca_calculator, get_orcablocks, get_orca_input_header
from cats.orca.run.orca_property import get_charges_from_orca_output

from cats.orca.tddft.active_learning import (
    AL_DTRUST,
    AL_LOCAL_OPTIMIZER_DEFAULT,
    AL_MIN_DATA,
    AL_ML_STEPS,
    AL_PREDICTION_MAX_UNC,
    ProxyContext,
    ScanThenEngradProxy,
    active_learning_step,
    append_training_set,
    ensure_training_seed,
    is_converged,
    normalize_local_optimizer_name,
)
from cats.orca.tddft.ct import (
    load_last_geometry_state,
    load_scan_log,
    record_scan_choice,
    record_scan_log_row,
    recover_scan_choice,
    summarize_geometry_log,
    summarize_scan_log,
)
from cats.orca.tddft.io import (
    is_valid_trainingset,
    resolve_restart_gbw_path,
    stage_restart_gbw_path,
    validate_orca_output,
)
from cats.orca.tddft.scan import (
    BondScanSettings,
    apply_bond_constraints,
    increment_bond_length,
    plot_scan_summaries,
    ScanSummaryPlotter,
    UVVisPlotSettings,
    resolve_bond_indices,
    resolve_bond_increment,
)
from cats.working_directory.tddft_workflow import scan_log as scan_log_helpers

# --- CONFIG ---
nodes = 2
cpus_per_node = 32
number_of_cpus = int(nodes * cpus_per_node)
memory_per_cpu = int(1300 * 128 / cpus_per_node)
ORCA_HPC = os.environ.get("CATS_ORCA_HPC") or os.environ.get("CPLAS_ORCA_HPC", "falcon")
ORCA_VERSION = os.environ.get("CATS_ORCA_VERSION") or os.environ.get("CPLAS_ORCA_VERSION", "6_1_0")

TARGET_WAVELENGTH_NM = 520.0
TDDFT_SCAN_ROOTS = 50
IROOT_SCORE_WEIGHTS = {
    "intensity": 0.1,
    "ct": 0.2,
    "wavelength": 0.7,
    "similarity": 1.0,
}
IROOT_FOLLOWUP_SCORE_WEIGHTS = {
    "intensity": 0.15,
    "ct": 0.20,
    "wavelength": 0.75,
    "similarity": 0.40,
}
IROOT_WAVELENGTH_SIGMA_NM = 20.0
IROOT_SIMILARITY_SIGMA_NM = 15.0
IROOT_SIMILARITY_SIGMA_CT = 0.40
IROOT_MIN_SIMILARITY = 0.10
IROOT_MIN_OSCILLATOR_STRENGTH = 5.0e-4
IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY = 0.10
IROOT_INTENSITY_REFERENCE_WINDOW_NM = 200.0
IROOT_MIN_TRANSITION_OVERLAP = 0.05
IROOT_MIN_CT_SIMILARITY = 0.15

CLUSTER_SYMBOLS = ("Cu", "Au")
# Non-metal + metalloid atomic numbers. Any other real element is treated as
# metal-like for distortion-energy eligibility checks.
NON_METAL_LIKE_ATOMIC_NUMBERS = {
    1,   # H
    2,   # He
    5,   # B
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    10,  # Ne
    14,  # Si
    15,  # P
    16,  # S
    17,  # Cl
    18,  # Ar
    32,  # Ge
    33,  # As
    34,  # Se
    35,  # Br
    36,  # Kr
    51,  # Sb
    52,  # Te
    53,  # I
    54,  # Xe
    84,  # Po
    85,  # At
    86,  # Rn
    117, # Ts
    118, # Og
}
ADSORBATE_BASIS_SET = "6-31g(d,p)"
DEF2_BASIS_SET = "def2-svp"
DEFAULT_BASIS_KEYWORD = "lanl2dz_mixed"
DEFAULT_XC_FUNCTIONAL = "pbe0"
DEFAULT_SOLVENT = "Water"
SHARED_TRAINING_DIR = Path("TDDFT_active_learning_shared")
WORKFLOW_ROOT_FILENAME = "workflow_root.txt"
IROOT_SCAN_LOG_FILENAME = "iroot_scan_endpoints.csv"
IROOT_GEOMETRY_LOG_FILENAME = "iroot_geometry_log.csv"
SCAN_PLOTS_DIRNAME = "scan_plots"
SCAN_ENDPOINTS_DB_FILENAME = "scan_endpoints.db"
SCAN_POSTPROCESS_DIRNAME = "scan_postprocess"
SCAN_POSTPROCESS_SMOOTHING_ITERATION_BASE = 10_000
SCAN_POSTPROCESS_REFINEMENT_ITERATION_BASE = 20_000
ITERATION_COMPLETE_FILENAME = "iteration_complete.txt"
REACTION_HUB_SUMMARY_FILENAME = "reaction_hub_summary.json"
REACTION_HUB_FINAL_XYZ_FILENAME = "reaction_hub_final.xyz"
REACTION_HUB_FINAL_CLUSTER_SP_DIRNAME = "cluster_single_point_final"
REACTION_HUB_FINAL_CLUSTER_TRAJ_FILENAME = "cluster_final_single_point.traj"
REACTION_HUB_FINAL_ADSORBATE_SP_DIRNAME = "adsorbate_single_point_final"
REACTION_HUB_FINAL_ADSORBATE_TRAJ_FILENAME = "adsorbate_final_single_point.traj"
REACTION_HUB_FINAL_TDDFT_SPECTRUM_DIRNAME = "final_tddft_spectrum"
REACTION_HUB_FINAL_TDDFT_SPECTRUM_PNG = "final_tddft_spectrum.png"
REACTION_HUB_FINAL_TDDFT_SPECTRUM_CSV = "final_tddft_spectrum.csv"
REACTION_HUB_CLUSTER_TDDFT_SPECTRUM_PNG = "cluster_tddft_spectrum.png"
REACTION_HUB_CLUSTER_TDDFT_SPECTRUM_CSV = "cluster_tddft_spectrum.csv"
REACTION_HUB_INTERACTION_DIFFERENCE_SPECTRUM_PNG = "cluster_adsorbate_difference_spectrum.png"
REACTION_HUB_INTERACTION_DIFFERENCE_SPECTRUM_CSV = "cluster_adsorbate_difference_spectrum.csv"
REACTION_HUB_REPLOT_ONLY_ENV = "REACTION_HUB_REPLOT_ONLY"
MAX_ADSORBATE_SINGLE_POINT_RANKS = 1
# Soft continuation cycles allow further optimisation when the AL step budget
# for a single pass is exhausted without hard failure.
MAX_ITERATION_SOFT_CONTINUATIONS = 64


class RequestedSpectrumError(RuntimeError):
    pass


# Hard guard for non-convergence retry loops caused by persistent runtime errors.
MAX_ITERATION_RETRY_ATTEMPTS = 8
DFT_ROOT_DIR = Path("DFT")
TDDFT_ROOT_DIR = Path("TDDFT")
INITIAL_DFT_DIR = DFT_ROOT_DIR / "iter_0"
INITIAL_SCAN_DIR = TDDFT_ROOT_DIR / "scan_iter_0"
EV_PER_WAVENUMBER_CM = 1.239841984e-4
_ORCA_FINAL_SP_ENERGY_RE = re.compile(
    r"FINAL\s+SINGLE\s+POINT\s+ENERGY\s+([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"
)
_ORCA_TOTAL_ENERGY_RE = re.compile(
    r"Total\s+Energy\s*[:=]\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)\s*Eh",
    re.IGNORECASE,
)
_AL_SUMMARY_TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|")


def _default_orca_workflow_backend() -> QMWorkflowBackend:
    return get_workflow_backend("orca")


def _workflow_backend_or_default(
    workflow_backend: QMWorkflowBackend | None = None,
) -> QMWorkflowBackend:
    return workflow_backend or _default_orca_workflow_backend()


def _workflow_output_path(
    directory: Path,
    workflow_backend: QMWorkflowBackend | None = None,
) -> Path:
    return _workflow_backend_or_default(workflow_backend).output_path(directory)


def _validate_workflow_output(
    output_path: Path,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
    warn: bool = False,
) -> bool:
    return _workflow_backend_or_default(workflow_backend).validate_output(output_path, warn=warn)


def _ensure_restart_gbw(
    target_dir: Path,
    *,
    fallback_dirs: list[Path] | None = None,
    workflow_backend: QMWorkflowBackend | None = None,
) -> bool:
    return _workflow_backend_or_default(workflow_backend).ensure_restart(
        target_dir,
        fallback_dirs=fallback_dirs or [],
    )


def _resolve_restart_gbw_name(directory: Path) -> str | None:
    restart = stage_restart_gbw_path(directory)
    return str(restart) if restart is not None else None


def _recover_restart_from_seed(
    directory: Path,
    seed_dir: Path,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> bool:
    backend = _workflow_backend_or_default(workflow_backend)
    if backend.has_restart(directory):
        return True
    seed_restart = backend.resolve_restart_path(seed_dir)
    if seed_restart is None:
        return False
    try:
        backend.copy_restart(seed_dir, directory, verbose=False)
    except OSError as exc:
        print(
            f"Warning: Failed to recover restart data in {directory} "
            f"from seed restart file {seed_restart}: {exc}"
        )
        return False
    print(f"Recovered restart data in {directory} from seed restart file {seed_restart}.")
    return True


def _recover_orca_gbw_from_seed(directory: Path, seed_dir: Path) -> bool:
    return _recover_restart_from_seed(directory, seed_dir)


def _stage_seed_single_point_as_converged_output(
    *,
    seed_dir: Path,
    output_dir: Path,
    label: str,
    workflow_backend: QMWorkflowBackend | None = None,
) -> bool:
    backend = _workflow_backend_or_default(workflow_backend)
    seed_out = backend.output_path(seed_dir)
    if not backend.validate_output(seed_out, warn=False):
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    copied_any = False
    filenames = [backend.output_filename, *backend.restart_filenames]
    if backend.gradient_filename:
        filenames.append(backend.gradient_filename)
    for filename in dict.fromkeys(filenames):
        source = seed_dir / filename
        if not source.exists():
            continue
        try:
            shutil.copy2(source, output_dir / filename)
            copied_any = True
        except OSError as exc:
            print(
                f"Warning: Failed to stage converged seed {source} "
                f"to {output_dir / filename} for {label}: {exc}"
            )
    return copied_any


def _parse_optional_int(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_optional_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _attach_charges_from_orca_output(atoms, orca_out: Path, *, label: str) -> None:
    try:
        get_charges_from_orca_output(atoms, orca_out, method="mulliken")
    except Exception as exc:
        print(f"Warning: Failed to recover charges for {label} from {orca_out}: {exc}")


def _read_orca_output_with_charges(
    orca_out: Path,
    *,
    label: str,
    workflow_backend: QMWorkflowBackend | None = None,
):
    return _workflow_backend_or_default(workflow_backend).read_output_atoms(orca_out, label=label)


def _resolve_last_iroot_from_candidates(
    candidate_dirs: list[Path | None],
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> tuple[int | None, Path | None]:
    backend = _workflow_backend_or_default(workflow_backend)
    checked_outputs: set[Path] = set()
    first_existing_output: Path | None = None
    for directory in candidate_dirs:
        if directory is None:
            continue
        output_path = backend.output_path(Path(directory))
        if output_path in checked_outputs:
            continue
        checked_outputs.add(output_path)
        if not output_path.exists():
            continue
        if first_existing_output is None:
            first_existing_output = output_path
        iroot = backend.get_followed_state(output_path)
        if iroot is not None:
            return int(iroot), output_path
    return None, first_existing_output


def _copy_constraints_between_atoms(source_atoms, target_atoms) -> None:
    try:
        source_constraints = getattr(source_atoms, "constraints", None)
    except Exception:
        source_constraints = None
    if source_constraints is None:
        return
    try:
        target_atoms.set_constraint(copy.deepcopy(source_constraints))
    except Exception:
        try:
            if isinstance(source_constraints, (list, tuple)):
                target_atoms.set_constraint(list(source_constraints))
            else:
                target_atoms.set_constraint(source_constraints)
        except Exception:
            pass


def _parse_total_energy_eV_from_orca_output(
    orca_out: Path,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> float | None:
    return _workflow_backend_or_default(workflow_backend).parse_total_energy_eV(orca_out)


def _resolve_total_energy_eV(
    endpoint_atoms,
    *,
    orca_out: Path,
    workflow_backend: QMWorkflowBackend | None = None,
) -> float | None:
    try:
        return float(endpoint_atoms.get_potential_energy())
    except Exception:
        pass
    return _parse_total_energy_eV_from_orca_output(
        orca_out,
        workflow_backend=workflow_backend,
    )


def _resolve_forces_eV_A(
    endpoint_atoms,
    *,
    opt_dir: Path,
    fallback_atoms=None,
    workflow_backend: QMWorkflowBackend | None = None,
):
    backend = _workflow_backend_or_default(workflow_backend)

    def _cached_forces(atoms_obj):
        if atoms_obj is None:
            return None
        calc = getattr(atoms_obj, "calc", None)
        if calc is None:
            return None
        try:
            # Only use already-available force data. Never trigger a fresh
            # calculator call during convergence checks.
            if calc.calculation_required(atoms_obj, ["forces"]):
                return None
            forces_local = atoms_obj.get_forces()
            if len(forces_local) == len(atoms_obj):
                return forces_local
        except Exception:
            return None
        return None

    backend_forces = backend.read_forces_eV_A(opt_dir, reference_atoms=endpoint_atoms)
    if backend_forces is not None:
        return backend_forces

    forces = _cached_forces(endpoint_atoms)
    if forces is not None:
        return forces

    fallback_forces = _cached_forces(fallback_atoms)
    if fallback_forces is not None:
        return fallback_forces

    return None


def _max_force_norm(forces) -> float | None:
    try:
        max_norm = 0.0
        for vec in forces:
            norm = (float(vec[0]) ** 2 + float(vec[1]) ** 2 + float(vec[2]) ** 2) ** 0.5
            if norm > max_norm:
                max_norm = norm
        return float(max_norm)
    except Exception:
        return None


def _is_converged_with_force_fallback(
    *,
    atoms,
    fmax: float,
    opt_dir: Path,
    fallback_atoms=None,
    workflow_backend: QMWorkflowBackend | None = None,
) -> bool:
    """Check convergence using the standard predicate, then ORCA force fallbacks."""
    if is_converged(atoms, fmax=fmax):
        return True
    forces_eV_A = _resolve_forces_eV_A(
        atoms,
        opt_dir=opt_dir,
        fallback_atoms=fallback_atoms,
        workflow_backend=workflow_backend,
    )
    max_force = _max_force_norm(forces_eV_A) if forces_eV_A is not None else None
    return bool(max_force is not None and max_force <= fmax)


def _is_active_learning_nonconvergence_error(exc: Exception) -> bool:
    text = str(exc or "").strip().lower()
    return "active learning did not converge" in text


def _last_true_fmax_from_al_summary(summary_path: Path) -> float | None:
    try:
        lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line or "true fmax" in line.lower():
            continue
        if not _AL_SUMMARY_TABLE_ROW_RE.match(line):
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if not parts:
            continue
        try:
            value = float(parts[-1])
        except ValueError:
            continue
        if isfinite(value):
            return float(value)
    return None


def _last_uncertainty_from_al_summary(summary_path: Path) -> float | None:
    try:
        lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line or "uncertainty" in line.lower():
            continue
        if not _AL_SUMMARY_TABLE_ROW_RE.match(line):
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) < 4:
            continue
        try:
            value = float(parts[-3])
        except ValueError:
            continue
        if isfinite(value):
            return float(value)
    return None


def _rollback_iteration_retry_attempt(iteration_attempts: dict[int, int], iteration: int) -> None:
    current = int(iteration_attempts.get(iteration, 0))
    if current <= 1:
        iteration_attempts.pop(iteration, None)
        return
    iteration_attempts[iteration] = current - 1


def _record_soft_iteration_continuation(
    *,
    iteration_attempts: dict[int, int],
    iteration_soft_continuations: dict[int, int],
    iteration: int,
) -> int:
    _rollback_iteration_retry_attempt(iteration_attempts, iteration)
    count = int(iteration_soft_continuations.get(iteration, 0)) + 1
    iteration_soft_continuations[iteration] = count
    if count > MAX_ITERATION_SOFT_CONTINUATIONS:
        raise RuntimeError(
            f"Iteration {iteration} exceeded soft continuation limit "
            f"({MAX_ITERATION_SOFT_CONTINUATIONS}) without reaching convergence. "
            "Consider increasing local-step budget or relaxing fmax."
        )
    return count


def _soft_continuation_status_text(
    *,
    cycle: int,
) -> str:
    return f"cycle {cycle}/{MAX_ITERATION_SOFT_CONTINUATIONS}"


def _scan_log_by_iteration(scan_log_path: Path) -> dict[int, dict[str, str]]:
    rows = load_scan_log(scan_log_path)
    per_iteration: dict[int, dict[str, str]] = {}
    for key, row in rows.items():
        iteration = _parse_optional_int(row.get("iteration")) or _parse_optional_int(key)
        if iteration is None or iteration < 1:
            continue
        per_iteration[iteration] = row
    return per_iteration


def _scan_metadata_from_row(row: dict[str, str]) -> dict[str, int | float | str]:
    metadata: dict[str, int | float | str] = {}
    scan_dir = str(row.get("scan_dir") or "").strip()
    if scan_dir:
        metadata["scan_dir"] = scan_dir

    for key in ("bond_i", "bond_j", "iroot", "source_iteration", "reference_iteration"):
        parsed = _parse_optional_int(row.get(key))
        if parsed is not None:
            metadata[f"scan_{key}"] = parsed

    for key in (
        "bond_length",
        "wavelength_nm",
        "intensity",
        "energy_ev",
        "ct_e",
        "ct_pct",
        "ct_net",
        "ct_magnitude",
        "score",
        "similarity",
        "score_intensity",
        "score_ct",
        "score_wavelength",
        "score_similarity",
        "target_wavelength_nm",
    ):
        parsed = _parse_optional_float(row.get(key))
        if parsed is not None:
            metadata[f"scan_{key}"] = parsed

    point_kind = str(row.get("point_kind") or "").strip()
    if point_kind:
        metadata["scan_point_kind"] = point_kind
    point_label = str(row.get("point_label") or "").strip()
    if point_label:
        metadata["scan_point_label"] = point_label
    return metadata


def _extract_excited_states(
    orca_out: Path,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> list[dict[str, int | float]]:
    backend = _workflow_backend_or_default(workflow_backend)
    try:
        transitions = backend.extract_transitions(orca_out)
    except Exception as exc:
        print(f"Warning: Failed to extract excited states from {orca_out}: {exc}")
        return []

    wavenumbers = [float(value) for value in transitions.wavenumbers_cm.tolist()]
    intensities = [float(value) for value in transitions.intensities.tolist()]
    wavelengths = (
        [float(value) for value in transitions.wavelengths_nm.tolist()]
        if transitions.wavelengths_nm is not None
        else None
    )

    states: list[dict[str, int | float]] = []
    for index, (wavenumber, intensity) in enumerate(zip(wavenumbers, intensities), start=1):
        wavelength_nm = (
            wavelengths[index - 1]
            if wavelengths is not None and len(wavelengths) >= index
            else (1.0e7 / wavenumber if wavenumber > 0.0 else 0.0)
        )
        states.append(
            {
                "state": index,
                "wavenumber_cm": wavenumber,
                "energy_ev": wavenumber * EV_PER_WAVENUMBER_CM,
                "wavelength_nm": wavelength_nm,
                "oscillator_strength": intensity,
            }
        )
    return states


def _chosen_iroot_data(
    iroot: int | None,
    row: dict[str, str] | None,
    excited_states: list[dict[str, int | float]],
) -> dict[str, int | float]:
    if iroot is None:
        return {}

    chosen: dict[str, int | float] = {"state": iroot}
    if 1 <= iroot <= len(excited_states):
        chosen.update(excited_states[iroot - 1])

    if row is not None:
        for key in (
            "wavelength_nm",
            "intensity",
            "energy_ev",
            "ct_e",
            "ct_pct",
            "ct_net",
            "ct_magnitude",
            "score",
            "similarity",
        ):
            parsed = _parse_optional_float(row.get(key))
            if parsed is not None:
                chosen[key] = parsed
    return chosen


def _iroot_candidate_from_scan_row(row: dict[str, str] | None) -> IRootCandidate | None:
    if row is None:
        return None
    state = _parse_optional_int(row.get("iroot"))
    wavelength_nm = _parse_optional_float(row.get("wavelength_nm"))
    intensity = _parse_optional_float(row.get("intensity"))
    if state is None or wavelength_nm is None or intensity is None:
        return None

    score_components: dict[str, float] = {}
    for key in ("score_intensity", "score_ct", "score_wavelength", "score_similarity"):
        parsed = _parse_optional_float(row.get(key))
        if parsed is not None:
            score_components[key.split("_", 1)[1]] = float(parsed)

    return IRootCandidate(
        state=int(state),
        wavelength_nm=float(wavelength_nm),
        intensity=float(intensity),
        energy_ev=_parse_optional_float(row.get("energy_ev")),
        ct_e=_parse_optional_float(row.get("ct_e")),
        ct_pct=_parse_optional_float(row.get("ct_pct")),
        ct_net=_parse_optional_float(row.get("ct_net")),
        score=_parse_optional_float(row.get("score")),
        similarity=_parse_optional_float(row.get("similarity")),
        score_components=score_components if score_components else None,
    )


def _resolve_ts_smoothing_replacement_dirs(
    *,
    max_iter: int,
    use_tddft_optimisation: bool,
    scan_rows: dict[int, dict[str, str]],
    workflow_backend: QMWorkflowBackend | None = None,
) -> dict[int, Path]:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    """Select TS-smoothing geometries that should replace primary TS points.

    Replacement rule:
    - candidate row must be a ts_smoothing postprocess row
    - candidate must reference the primary iteration via source_iteration
    - candidate total energy must be lower than the original primary endpoint
    """
    replacements: dict[int, Path] = {}
    best_candidate: dict[int, tuple[float, Path]] = {}

    for row in scan_rows.values():
        if str(row.get("point_kind") or "").strip() != "ts_smoothing":
            continue
        source_iteration = _parse_optional_int(row.get("source_iteration"))
        if source_iteration is None or source_iteration < 1 or source_iteration > int(max_iter):
            continue
        scan_dir_text = str(row.get("scan_dir") or "").strip()
        if not scan_dir_text:
            continue
        candidate_dir = Path(scan_dir_text)
        candidate_out = workflow_backend.output_path(candidate_dir)
        if not workflow_backend.validate_output(candidate_out, warn=False):
            continue
        candidate_energy = _parse_total_energy_eV_from_orca_output(
            candidate_out,
            workflow_backend=workflow_backend,
        )
        if candidate_energy is None:
            continue
        existing = best_candidate.get(int(source_iteration))
        if existing is None or float(candidate_energy) < float(existing[0]):
            best_candidate[int(source_iteration)] = (float(candidate_energy), candidate_dir)

    for iteration in range(1, int(max_iter) + 1):
        candidate = best_candidate.get(int(iteration))
        if candidate is None:
            continue
        primary_opt_dir = _resolve_opt_dir_for_mode(
            int(iteration),
            use_tddft_optimisation=use_tddft_optimisation,
        )
        primary_out = workflow_backend.output_path(primary_opt_dir)
        if not workflow_backend.validate_output(primary_out, warn=False):
            continue
        primary_energy = _parse_total_energy_eV_from_orca_output(
            primary_out,
            workflow_backend=workflow_backend,
        )
        if primary_energy is None:
            continue
        candidate_energy, candidate_dir = candidate
        if float(candidate_energy) < float(primary_energy) - 1.0e-8:
            replacements[int(iteration)] = candidate_dir
            print(
                "Info: Replacing TS endpoint iteration "
                f"{iteration} in scan_endpoints.db with smoothed geometry "
                f"({candidate_energy:.6f} eV < {primary_energy:.6f} eV)."
            )
    return replacements


def export_scan_endpoints_db(
    *,
    max_iter: int,
    use_tddft_optimisation: bool,
    scan_log_path: Path,
    output_db_path: Path,
    scan_nroots: int = TDDFT_SCAN_ROOTS,
    workflow_backend: QMWorkflowBackend | None = None,
) -> None:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    scan_rows = _scan_log_by_iteration(scan_log_path)
    if max_iter < 1 and not scan_rows:
        return
    output_db_path.parent.mkdir(parents=True, exist_ok=True)
    if output_db_path.exists():
        try:
            output_db_path.unlink()
        except OSError as exc:
            print(f"Warning: Failed to replace endpoints DB {output_db_path}: {exc}")
            return

    db = connect(str(output_db_path))
    written = 0
    replacement_dirs = _resolve_ts_smoothing_replacement_dirs(
        max_iter=max_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        scan_rows=scan_rows,
        workflow_backend=workflow_backend,
    )

    def _write_endpoint(
        *,
        iteration: int,
        opt_dir: Path,
        row: dict[str, str] | None,
    ) -> bool:
        opt_out = workflow_backend.output_path(opt_dir)
        if not workflow_backend.validate_output(opt_out, warn=False):
            return False
        try:
            endpoint_atoms = _read_orca_output_with_charges(
                opt_out,
                label=f"scan endpoint iter {iteration}",
                workflow_backend=workflow_backend,
            )
        except Exception as exc:
            print(f"Warning: Failed to read endpoint geometry from {opt_out}: {exc}")
            return False

        status = "tddft" if _has_tddft_state_tag(opt_out, workflow_backend=workflow_backend) else "dft"
        metadata: dict[str, object] = {
            "iteration": int(iteration),
            "status": status,
            "opt_dir": str(opt_dir),
        }
        if iteration > max_iter:
            metadata["postprocess"] = True
        charge_method = endpoint_atoms.info.get("charge_method")
        if charge_method:
            metadata["charge_method"] = str(charge_method)
        total_energy_eV = _resolve_total_energy_eV(
            endpoint_atoms,
            orca_out=opt_out,
            workflow_backend=workflow_backend,
        )
        if total_energy_eV is not None:
            metadata["total_energy_eV"] = float(total_energy_eV)
        forces_eV_A = _resolve_forces_eV_A(
            endpoint_atoms,
            opt_dir=opt_dir,
            workflow_backend=workflow_backend,
        )
        if forces_eV_A is not None:
            metadata["has_forces"] = True
            metadata["forces_eV_A"] = _serialize_vectors(forces_eV_A)
            max_force = _max_force_norm(forces_eV_A)
            if max_force is not None:
                metadata["max_force_eV_A"] = float(max_force)
        else:
            metadata["has_forces"] = False

        if total_energy_eV is not None or forces_eV_A is not None:
            sp_kwargs: dict[str, object] = {}
            if total_energy_eV is not None:
                sp_kwargs["energy"] = float(total_energy_eV)
            if forces_eV_A is not None:
                sp_kwargs["forces"] = forces_eV_A
            try:
                endpoint_atoms.calc = SinglePointCalculator(endpoint_atoms, **sp_kwargs)
            except Exception as exc:
                print(
                    "Warning: Failed to attach endpoint SinglePoint results "
                    f"for iter {iteration} ({opt_out}): {exc}"
                )
        if row is not None:
            metadata.update(_scan_metadata_from_row(row))
            metadata["scan_nroots"] = int(scan_nroots)

        excited_states: list[dict[str, int | float]] = []
        if row is not None or status == "tddft":
            scan_dir_text = str(row.get("scan_dir") or "").strip() if row is not None else ""
            transition_out = (
                workflow_backend.output_path(Path(scan_dir_text))
                if scan_dir_text
                else opt_out
            )
            if not transition_out.exists():
                transition_out = opt_out
            excited_states = _extract_excited_states(
                transition_out,
                workflow_backend=workflow_backend,
            )
            if excited_states:
                metadata["excited_states"] = excited_states
                metadata["n_excited_states"] = len(excited_states)

            iroot = _parse_optional_int(row.get("iroot")) if row is not None else None
            chosen = _chosen_iroot_data(iroot, row, excited_states)
            if chosen:
                metadata["chosen_iroot_data"] = chosen

        write_kwargs: dict[str, object] = {
            "iteration": int(iteration),
            "calc_status": status,
        }
        if total_energy_eV is not None:
            write_kwargs["total_energy_eV"] = float(total_energy_eV)
        if forces_eV_A is not None:
            max_force = _max_force_norm(forces_eV_A)
            if max_force is not None:
                write_kwargs["max_force_eV_A"] = float(max_force)

        db.write(endpoint_atoms, data=metadata, **write_kwargs)
        return True

    for i in range(1, max_iter + 1):
        opt_dir = replacement_dirs.get(
            int(i),
            _resolve_opt_dir_for_mode(
                i,
                use_tddft_optimisation=use_tddft_optimisation,
            ),
        )
        if _write_endpoint(iteration=i, opt_dir=opt_dir, row=scan_rows.get(i)):
            written += 1

    extra_iterations = sorted(
        iteration
        for iteration in scan_rows
        if int(iteration) > max_iter
    )
    for iteration in extra_iterations:
        row = scan_rows.get(iteration)
        if row is None:
            continue
        if str(row.get("point_kind") or "").strip() == "ts_smoothing":
            # TS smoothing rows are exported by replacing the primary TS endpoint
            # in-place; do not also append them as postprocess tail points.
            continue
        scan_dir_text = str(row.get("scan_dir") or "").strip()
        if not scan_dir_text:
            continue
        opt_dir = Path(scan_dir_text)
        if _write_endpoint(iteration=iteration, opt_dir=opt_dir, row=row):
            written += 1

    print(f"Saved {written} optimisation endpoints to {output_db_path}.")


def _build_metal_basis_block(atoms) -> list[str]:
    atomic_numbers = set(atoms.get_atomic_numbers())
    basis_lines: list[str] = []
    for atomic_number in (79, 29):
        if atomic_number in atomic_numbers:
            basis_lines.extend(
                [
                    f'newGTO {atomic_number} "LANL2DZ" end',
                    f'newECP {atomic_number} "HayWadt" end',
                ]
            )
    if not basis_lines:
        return []
    return ["%BASIS", *basis_lines, "END"]


def _normalize_basis_keyword(basis_keyword: str) -> str:
    key = str(basis_keyword).strip().lower()
    aliases = {
        "lanl2dz_mixed": "lanl2dz_mixed",
        "lanl2dz": "lanl2dz_mixed",
        "mixed": "lanl2dz_mixed",
        "mixed_lanl2dz": "lanl2dz_mixed",
        "def2-svp": "def2-svp",
        "def2svp": "def2-svp",
        "def2": "def2-svp",
    }
    resolved = aliases.get(key)
    if resolved is None:
        raise ValueError(
            "Unsupported basis_keyword. Choose 'lanl2dz_mixed' or 'def2-svp'."
        )
    return resolved


def _normalize_xc_functional(xc_functional: str) -> str:
    key = str(xc_functional).strip().lower().replace("_", "-")
    aliases = {
        "pbe0": "PBE0",
        "cam-b3lyp": "CAM-B3LYP",
    }
    resolved = aliases.get(key)
    if resolved is None:
        raise ValueError(
            "Unsupported xc_functional. Choose 'pbe0' or 'cam-b3lyp'."
        )
    return resolved


def _resolve_solvation_keyword(solvent: str | None) -> str | None:
    if solvent is None:
        return None
    text = str(solvent).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"none", "off", "gas", "gas-phase", "gas_phase", "vacuum"}:
        return None
    if "(" in text and ")" in text:
        return text
    return f"CPCM({text})"


def _resolve_orca_resources(*, nodes: int, cpus_per_node: int) -> tuple[int, int]:
    if nodes <= 0:
        raise ValueError("nodes must be positive.")
    if cpus_per_node <= 0:
        raise ValueError("cpus_per_node must be positive.")
    total_cpus = int(nodes * cpus_per_node)
    explicit_mem = os.environ.get("WORKFLOW_MEM_PER_CPU")
    if explicit_mem:
        mem_per_cpu = int(explicit_mem)
    elif os.environ.get("SLURM_MEM_PER_NODE"):
        try:
            mem_per_cpu = int(int(os.environ["SLURM_MEM_PER_NODE"]) * 0.70 / cpus_per_node)
        except (ValueError, TypeError):
            mem_per_cpu = int(1300 * 128 / cpus_per_node)
    else:
        mem_per_cpu = int(1300 * 128 / cpus_per_node)
    if mem_per_cpu <= 0:
        raise ValueError(
            "Derived memory_per_cpu is non-positive; reduce cpus_per_node."
        )
    return total_cpus, mem_per_cpu


def _resolve_spin_multiplicity(
    atoms,
    spin_multiplicity: int | None,
    system_charge: int = 0,
) -> int:
    if spin_multiplicity is None:
        return (sum(atoms.get_atomic_numbers()) - int(system_charge)) % 2 + 1
    try:
        value = int(spin_multiplicity)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid spin_multiplicity value: {spin_multiplicity!r}") from exc
    if value <= 0:
        raise ValueError(f"spin_multiplicity must be a positive integer, got {value}.")
    return value


def _resolve_basis_settings(atoms, basis_keyword: str) -> tuple[str, list[str]]:
    basis_mode = _normalize_basis_keyword(basis_keyword)
    if basis_mode == "def2-svp":
        return DEF2_BASIS_SET, []
    return ADSORBATE_BASIS_SET, _build_metal_basis_block(atoms)


def get_orca_tddft_calculator(
    atoms,
    *,
    cpus: int,
    mem: int,
    task: str,
    directory: Path,
    nroots: int,
    iroot: Optional[int] = None,
    basis_keyword: str = DEFAULT_BASIS_KEYWORD,
    xc_functional: str = DEFAULT_XC_FUNCTIONAL,
    solvent: str | None = DEFAULT_SOLVENT,
    spin_multiplicity: int | None = None,
    system_charge: int = 0,
    use_tda: bool = True,
):
    blocks_settings = []
    header_basis_set, basis_block = _resolve_basis_settings(atoms, basis_keyword)
    xc_keyword = _normalize_xc_functional(xc_functional)
    solvation_keyword = _resolve_solvation_keyword(solvent)
    restart_gbw = _resolve_restart_gbw_name(directory)
    if restart_gbw is not None:
        blocks_settings.append(f'%MOINP "{restart_gbw}"')
    blocks_settings += basis_block

    tddft = ["%TDDFT", "MaxIter 999", f"NRoots {nroots}"]
    if use_tda is False:
        tddft.append("TDA FALSE")
    else:
        tddft.append("TDA TRUE")
    if iroot is not None:
        tddft += [
            f"IRoot {iroot}",
            "FollowIRoot TRUE",
            # "FIRKEEPFIRSTREF TRUE",
        ]
    max_dim = 5 if nroots >= 80 else 10
    tddft += [
        f"MaxDim {max_dim}",
        "ETol 3.6e-4",
        "RTol 3.6e-4",
        "TPrint 1e-8",
        "END",
    ]
    blocks_settings += tddft

    header_kwargs = {
        "xc": xc_keyword,
        "cosx": "RIJCOSX",
        "basis_set": header_basis_set,
        "integration_grid": "DefGrid2",
        "scf_convergence_criteria": "TightSCF",
        "output_level": "NormalPrint",
        "task": task,
        "scf_convergence_strategy": "SlowConv",
        "restart": "MOREAD" if restart_gbw is not None else "",
    }
    if solvation_keyword is not None:
        header_kwargs["solvation"] = solvation_keyword
    orcasimpleinput = get_orca_input_header(**header_kwargs)
    orcablocks = get_orcablocks(blocks_settings, number_of_cpus=cpus, memory_per_cpu=mem)

    mult = _resolve_spin_multiplicity(atoms, spin_multiplicity, system_charge)
    return get_orca_calculator(
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        version=ORCA_VERSION,
        hpc=ORCA_HPC,
        charge=int(system_charge),
        mult=mult,
        directory=directory,
    )


def get_orca_dft_calculator(
    atoms,
    *,
    cpus: int,
    mem: int,
    task: str = "Opt",
    force_convergence_criteria: str = "TightOpt",
    directory: Path,
    nroots: int | None = None,
    iroot: int | None = None,
    basis_keyword: str = DEFAULT_BASIS_KEYWORD,
    xc_functional: str = DEFAULT_XC_FUNCTIONAL,
    solvent: str | None = DEFAULT_SOLVENT,
    spin_multiplicity: int | None = None,
    system_charge: int = 0,
):
    _ = (nroots, iroot)
    header_basis_set, basis_block = _resolve_basis_settings(atoms, basis_keyword)
    xc_keyword = _normalize_xc_functional(xc_functional)
    solvation_keyword = _resolve_solvation_keyword(solvent)
    restart_gbw = _resolve_restart_gbw_name(directory)
    header_kwargs = {
        "xc": xc_keyword,
        "cosx": "RIJCOSX",
        "basis_set": header_basis_set,
        "integration_grid": "DefGrid2",
        "scf_convergence_criteria": "TightSCF",
        "output_level": "NormalPrint",
        "task": task,
        "scf_convergence_strategy": "SlowConv",
        "restart": "MOREAD" if restart_gbw is not None else "",
    }
    if solvation_keyword is not None:
        header_kwargs["solvation"] = solvation_keyword
    if "opt" in task.lower():
        header_kwargs["force_convergence_criteria"] = force_convergence_criteria
    orcasimpleinput = get_orca_input_header(
        **header_kwargs,
    )
    blocks_settings: list[str] = []
    if restart_gbw is not None:
        blocks_settings.append(f'%MOINP "{restart_gbw}"')
    blocks_settings += basis_block
    orcablocks = get_orcablocks(
        blocks_settings,
        number_of_cpus=cpus,
        memory_per_cpu=mem,
    )
    mult = _resolve_spin_multiplicity(atoms, spin_multiplicity, system_charge)
    return get_orca_calculator(
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        version=ORCA_VERSION,
        hpc=ORCA_HPC,
        charge=int(system_charge),
        mult=mult,
        directory=directory,
    )


def get_orca_dft_engrad_calculator(
    atoms,
    *,
    cpus: int,
    mem: int,
    task: str = "engrad",
    directory: Path,
    nroots: int | None = None,
    iroot: int | None = None,
    basis_keyword: str = DEFAULT_BASIS_KEYWORD,
    xc_functional: str = DEFAULT_XC_FUNCTIONAL,
    solvent: str | None = DEFAULT_SOLVENT,
    spin_multiplicity: int | None = None,
    system_charge: int = 0,
):
    if task.lower() != "engrad":
        print(f"Warning: Overriding DFT task '{task}' with 'engrad' for ground-state forces.")
    return get_orca_dft_calculator(
        atoms,
        cpus=cpus,
        mem=mem,
        task="engrad",
        directory=directory,
        nroots=nroots,
        iroot=iroot,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
    )


def _resolve_qm_workflow_backend(
    qm_backend: str | QMWorkflowBackend | None = "orca",
    qm_backend_settings: dict[str, object] | None = None,
) -> QMWorkflowBackend:
    backend = get_workflow_backend(qm_backend, settings=dict(qm_backend_settings or {}))
    if backend.key == "orca" and hasattr(backend, "with_calculator_factories"):
        backend = backend.with_calculator_factories(
            tddft_calculator_factory=get_orca_tddft_calculator,
            dft_calculator_factory=get_orca_dft_calculator,
            dft_force_calculator_factory=get_orca_dft_engrad_calculator,
        )
    return backend


def _validate_qm_workflow_backend(
    backend: QMWorkflowBackend,
    *,
    use_tddft_optimisation: bool,
    bond_scan: BondScanSettings,
    final_tddft_spectrum: bool,
) -> None:
    capabilities = backend.capabilities
    if not capabilities.supports_active_learning_workflow:
        raise NotImplementedError(
            f"{backend.label} is registered as a workflow backend, but the active-learning "
            "TDDFT optimisation workflow is not wired for this backend yet. "
            "Use qm_backend='orca' for the current workflow path."
        )
    if use_tddft_optimisation and not capabilities.supports_tddft_gradients:
        raise NotImplementedError(
            f"{backend.label} does not support TDDFT gradients/state-following in this workflow yet."
        )
    if bond_scan.enabled and not capabilities.supports_tddft_scans:
        raise NotImplementedError(f"{backend.label} does not support TDDFT scan workflows.")
    if final_tddft_spectrum and not capabilities.supports_tddft_scans:
        raise NotImplementedError(f"{backend.label} does not support final TDDFT spectra.")
    if (use_tddft_optimisation or bond_scan.enabled) and not capabilities.supports_ct_vectors:
        raise NotImplementedError(
            f"{backend.label} does not support CT-vector based IROOT scoring in this workflow yet."
        )


def ensure_initial_dft(
    atoms,
    *,
    cpus: int,
    mem: int,
    directory: Path,
    fmax: float = 0.01,
    basis_keyword: str = DEFAULT_BASIS_KEYWORD,
    xc_functional: str = DEFAULT_XC_FUNCTIONAL,
    solvent: str | None = DEFAULT_SOLVENT,
    spin_multiplicity: int | None = None,
    workflow_backend: QMWorkflowBackend | None = None,
    dft_force_calc_factory=None,
) -> Path:
    # Keep legacy API, but route through the active-learning engrad path.
    _, resolved_dir = ensure_initial_dft_converged_atoms(
        atoms,
        cpus=cpus,
        mem=mem,
        directory=directory,
        run_if_missing=True,
        fmax=fmax,
        strict_fmax=fmax,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        active_learning_optimizer=AL_LOCAL_OPTIMIZER_DEFAULT,
        workflow_backend=workflow_backend,
        dft_force_calc_factory=dft_force_calc_factory,
    )
    return resolved_dir


def ensure_initial_dft_converged_atoms(
    atoms,
    *,
    cpus: int,
    mem: int,
    directory: Path,
    run_if_missing: bool = True,
    fmax: float = 0.01,
    strict_fmax: float | None = None,
    al_steps_per_call: int = 20,
    max_steps: int | None = None,
    basis_keyword: str = DEFAULT_BASIS_KEYWORD,
    xc_functional: str = DEFAULT_XC_FUNCTIONAL,
    solvent: str | None = DEFAULT_SOLVENT,
    spin_multiplicity: int | None = None,
    active_learning_optimizer: str = AL_LOCAL_OPTIMIZER_DEFAULT,
    workflow_backend: QMWorkflowBackend | None = None,
    dft_force_calc_factory=None,
) -> tuple[object, Path]:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    if dft_force_calc_factory is None:
        if workflow_backend.key == "orca":
            dft_force_calc_factory = partial(
                get_orca_dft_engrad_calculator,
                basis_keyword=basis_keyword,
                xc_functional=xc_functional,
                solvent=solvent,
                spin_multiplicity=spin_multiplicity,
            )
        else:
            dft_force_calc_factory = partial(
                workflow_backend.make_dft_force_calculator,
                basis_keyword=basis_keyword,
                xc_functional=xc_functional,
                solvent=solvent,
                spin_multiplicity=spin_multiplicity,
            )
    active_learning_optimizer = normalize_local_optimizer_name(active_learning_optimizer)
    fmax_check = strict_fmax if strict_fmax is not None else fmax
    out_path = workflow_backend.output_path(directory)
    if workflow_backend.validate_output(out_path, warn=False):
        try:
            opt_atoms = _read_orca_output_with_charges(
                out_path,
                label="initial DFT restart",
                workflow_backend=workflow_backend,
            )
            _copy_constraints_between_atoms(atoms, opt_atoms)
            if _is_converged_with_force_fallback(
                atoms=opt_atoms,
                fmax=fmax_check,
                opt_dir=directory,
                workflow_backend=workflow_backend,
            ):
                return opt_atoms, directory
            summary_path = directory / "active_learning" / "ml_summary.txt"
            summary_fmax = _last_true_fmax_from_al_summary(summary_path)
            summary_uncertainty = _last_uncertainty_from_al_summary(summary_path)
            if summary_fmax is not None and summary_fmax <= fmax_check:
                if (
                    summary_uncertainty is None
                    or summary_uncertainty <= AL_PREDICTION_MAX_UNC
                ):
                    print(
                        "Info: Accepting existing initial DFT output based on active-learning "
                        f"summary convergence evidence (True fmax={summary_fmax:.6f} <= "
                        f"{fmax_check:.6f} eV/A"
                        + (
                            ", uncertainty unavailable)."
                            if summary_uncertainty is None
                            else f", uncertainty={summary_uncertainty:.6f} <= {AL_PREDICTION_MAX_UNC:.6f})."
                        )
                    )
                    return opt_atoms, directory
            print(f"Warning: Existing initial DFT output not converged; rerunning ({out_path}).")
        except Exception as exc:
            print(f"Warning: Failed to read {out_path}; rerunning initial DFT: {exc}")
    elif not run_if_missing:
        raise FileNotFoundError(
            f"Missing initial DFT output at {out_path}; set run_if_missing=True to generate it."
        )

    directory.mkdir(parents=True, exist_ok=True)
    al_dir = directory / "active_learning"
    training_path = al_dir / "evaluated.traj"
    seed_dir = al_dir / "seed"
    atoms = ensure_dft_training_seed(
        atoms,
        training_path=training_path,
        seed_dir=seed_dir,
        cpus=cpus,
        mem=mem,
        calc_factory=dft_force_calc_factory,
        workflow_backend=workflow_backend,
    )
    seed_forces_eV_A = _resolve_forces_eV_A(
        atoms,
        opt_dir=seed_dir,
        workflow_backend=workflow_backend,
    )
    seed_max_force = _max_force_norm(seed_forces_eV_A) if seed_forces_eV_A is not None else None
    if seed_max_force is not None and seed_max_force <= fmax_check:
        if _stage_seed_single_point_as_converged_output(
            seed_dir=seed_dir,
            output_dir=directory,
            label="initial DFT",
            workflow_backend=workflow_backend,
        ):
            print(
                "Info: Initial DFT seed single-point is already converged "
                f"(max force={seed_max_force:.6f} <= {fmax_check:.6f} eV/A); "
                "skipping active learning."
            )
            return atoms, directory
    _ensure_restart_gbw(directory, fallback_dirs=[seed_dir], workflow_backend=workflow_backend)
    dft_calc = dft_force_calc_factory(
        atoms,
        cpus=cpus,
        mem=mem,
        directory=directory,
    )
    atoms = active_learning_step(
        atoms,
        dft_calc,
        fmax=fmax_check,
        shared_training_path=training_path,
        state_dir=al_dir,
        use_restart=is_valid_trainingset(training_path),
        steps_per_call=al_steps_per_call,
        optimize_to_convergence=True,
        max_steps=max_steps,
        max_unc=AL_PREDICTION_MAX_UNC,
        min_data=AL_MIN_DATA,
        ml_steps=AL_ML_STEPS,
        dtrust=AL_DTRUST,
        local_optimizer=active_learning_optimizer,
    )
    _recover_restart_from_seed(directory, seed_dir, workflow_backend=workflow_backend)
    al_converged = _is_converged_with_force_fallback(
        atoms=atoms,
        fmax=fmax_check,
        opt_dir=directory,
        workflow_backend=workflow_backend,
    )
    if not workflow_backend.validate_output(out_path, warn=False):
        if al_converged:
            print(
                f"Warning: Initial DFT output could not be validated at {out_path}, "
                "but active-learning convergence criteria were satisfied using the "
                "latest available force evaluation; continuing."
            )
            if (
                workflow_backend.capabilities.requires_restart_data
                and workflow_backend.resolve_restart_path(directory) is None
            ):
                raise RuntimeError(
                    f"Initial DFT run did not generate a {workflow_backend.label} restart file "
                    f"in {directory}."
                )
            return atoms, directory
        raise RuntimeError(f"Initial DFT output invalid or missing: {out_path}")
    try:
        opt_atoms = _read_orca_output_with_charges(
            out_path,
            label="initial DFT final output",
            workflow_backend=workflow_backend,
        )
        _copy_constraints_between_atoms(atoms, opt_atoms)
    except Exception as exc:
        raise RuntimeError(f"Failed to read initial DFT output {out_path}: {exc}") from exc
    if not _is_converged_with_force_fallback(
        atoms=opt_atoms,
        fmax=fmax_check,
        opt_dir=directory,
        workflow_backend=workflow_backend,
    ):
        if al_converged:
            print(
                "Warning: Initial DFT final ORCA parse did not satisfy direct convergence "
                f"check ({workflow_backend.label} output may not provide final-geometry forces), but "
                "active-learning convergence criteria were satisfied using the latest "
                "available force evaluation; continuing."
            )
            return atoms, directory
        raise RuntimeError(
            "Initial DFT geometry optimization did not converge; "
            f"check {out_path} and consider adjusting optimization settings."
        )
    if (
        workflow_backend.capabilities.requires_restart_data
        and workflow_backend.resolve_restart_path(directory) is None
    ):
        raise RuntimeError(
            f"Initial DFT run did not generate a {workflow_backend.label} restart file "
            f"in {directory}."
        )
    return opt_atoms, directory


def ensure_dft_training_seed(
    atoms,
    *,
    training_path: Path,
    seed_dir: Path,
    cpus: int,
    mem: int,
    calc_factory,
    workflow_backend: QMWorkflowBackend | None = None,
) -> object:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    seed_out = workflow_backend.output_path(seed_dir)
    if is_valid_trainingset(training_path) and workflow_backend.validate_output(seed_out, warn=False):
        return atoms
    seed_dir.mkdir(parents=True, exist_ok=True)
    seed_calc = calc_factory(
        atoms,
        cpus=cpus,
        mem=mem,
        task="engrad",
        directory=seed_dir,
    )
    atoms.calc = seed_calc
    atoms.get_forces()
    if not workflow_backend.validate_output(seed_out):
        raise RuntimeError(f"Failed to generate DFT engrad output: {seed_out}")
    append_training_set(training_path, atoms)
    return atoms


def _format_fragment_indices(indices: list[int]) -> str:
    if not indices:
        return ""
    indices = sorted(indices)
    ranges: list[str] = []
    start = indices[0]
    prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append(str(start) if start == prev else f"{start}-{prev}")
        start = idx
        prev = idx
    ranges.append(str(start) if start == prev else f"{start}-{prev}")
    return ", ".join(ranges)


def build_fragments(atoms, cluster_symbols: tuple[str, ...] = CLUSTER_SYMBOLS) -> dict[str, str]:
    cluster: list[int] = []
    others: list[int] = []
    for atom in atoms:
        idx = atom.index + 1
        if atom.symbol in cluster_symbols:
            cluster.append(idx)
        else:
            others.append(idx)
    if not cluster:
        raise ValueError(f"No cluster atoms found for fragment 1 (symbols={cluster_symbols}).")
    if not others:
        raise ValueError("No non-cluster atoms found for fragment 2.")
    return {
        "1": _format_fragment_indices(cluster),
        "2": _format_fragment_indices(others),
    }


def _split_cluster_adsorbate_indices(
    atoms,
    *,
    cluster_symbols: tuple[str, ...] = CLUSTER_SYMBOLS,
) -> tuple[list[int], list[int]]:
    cluster_indices: list[int] = []
    adsorbate_indices: list[int] = []
    for atom in atoms:
        if atom.symbol in cluster_symbols:
            cluster_indices.append(int(atom.index))
        else:
            adsorbate_indices.append(int(atom.index))
    return cluster_indices, adsorbate_indices


def _count_metal_like_atoms(atoms) -> int:
    count = 0
    for atomic_number in atoms.get_atomic_numbers():
        z = int(atomic_number)
        if z <= 0:
            continue
        if z not in NON_METAL_LIKE_ATOMIC_NUMBERS:
            count += 1
    return int(count)


def _run_isolated_single_point(
    *,
    fragment_atoms,
    sp_dir: Path,
    cpus: int,
    mem: int,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    spin_multiplicity: int | None,
    label: str,
    traj_filename: str,
    system_charge: int = 0,
    workflow_backend: QMWorkflowBackend | None = None,
) -> dict[str, object]:
    workflow_backend = workflow_backend or _resolve_qm_workflow_backend("orca")
    payload: dict[str, object] = {"status": "skipped"}
    sp_dir.mkdir(parents=True, exist_ok=True)
    sp_out = workflow_backend.output_path(sp_dir)

    energy_from_calc: float | None = None
    if not workflow_backend.validate_output(sp_out, warn=False):
        try:
            calc = workflow_backend.make_dft_calculator(
                fragment_atoms,
                cpus=cpus,
                mem=mem,
                task="energy",
                directory=sp_dir,
                basis_keyword=basis_keyword,
                xc_functional=xc_functional,
                solvent=solvent,
                spin_multiplicity=spin_multiplicity,
                system_charge=system_charge,
            )
            fragment_atoms.calc = calc
            energy_from_calc = float(fragment_atoms.get_potential_energy())
        except Exception as exc:
            payload["status"] = "error"
            payload["error"] = str(exc)
            payload["directory"] = str(sp_dir)
            return payload

    parsed_atoms = None
    if workflow_backend.validate_output(sp_out, warn=False):
        try:
            parsed_atoms = _read_orca_output_with_charges(
                sp_out,
                label=label,
                workflow_backend=workflow_backend,
            )
        except Exception:
            parsed_atoms = None

    energy_eV = None
    if parsed_atoms is not None:
        energy_eV = _resolve_total_energy_eV(
            parsed_atoms,
            orca_out=sp_out,
            workflow_backend=workflow_backend,
        )
    if energy_eV is None:
        energy_eV = energy_from_calc

    traj_atoms = parsed_atoms.copy() if parsed_atoms is not None else fragment_atoms.copy()
    if energy_eV is not None:
        try:
            traj_atoms.calc = SinglePointCalculator(traj_atoms, energy=float(energy_eV))
        except Exception:
            pass

    traj_paths: list[str] = []
    for traj_path in (
        sp_dir / traj_filename,
        SHARED_TRAINING_DIR / traj_filename,
    ):
        try:
            traj_path.parent.mkdir(parents=True, exist_ok=True)
            write(str(traj_path), traj_atoms, format="traj")
            traj_paths.append(str(traj_path))
        except Exception as exc:
            print(f"Warning: Failed to write isolated-fragment trajectory {traj_path}: {exc}")

    payload.update(
        {
            "status": "ok",
            "directory": str(sp_dir),
            "output": str(sp_out),
            "formula": str(traj_atoms.get_chemical_formula()),
            "symbols": [str(symbol) for symbol in traj_atoms.get_chemical_symbols()],
            "positions_A": _serialize_vectors(traj_atoms.get_positions()),
            "traj_paths": traj_paths,
        }
    )
    if workflow_backend.key == "orca":
        payload["orca_out"] = str(sp_out)
    elif workflow_backend.key == "nwchem":
        payload["nwchem_out"] = str(sp_out)
    if energy_eV is not None:
        payload["energy_eV"] = float(energy_eV)
        payload["energy_hartree"] = float(energy_eV / Hartree)
    return payload


def _run_final_cluster_single_point(
    *,
    final_atoms,
    final_opt_dir: Path,
    cpus: int,
    mem: int,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    spin_multiplicity: int | None,
    system_charge: int = 0,
    workflow_backend: QMWorkflowBackend | None = None,
) -> dict[str, object]:
    workflow_backend = workflow_backend or _resolve_qm_workflow_backend("orca")
    payload: dict[str, object] = {
        "status": "skipped",
        "cluster_symbols": [str(symbol) for symbol in CLUSTER_SYMBOLS],
    }
    metal_like_atom_count = _count_metal_like_atoms(final_atoms)
    payload["metal_atom_count"] = int(metal_like_atom_count)
    if metal_like_atom_count <= 0:
        payload["status"] = "skipped_no_metal_atoms"
        return payload

    cluster_indices, adsorbate_indices = _split_cluster_adsorbate_indices(final_atoms)
    payload["cluster_atom_count"] = int(len(cluster_indices))
    payload["adsorbate_atom_count"] = int(len(adsorbate_indices))

    if not cluster_indices:
        payload["status"] = "skipped_no_cluster_atoms"
        return payload
    if not adsorbate_indices:
        payload["status"] = "skipped_no_adsorbates"
        return payload

    cluster_atoms = final_atoms[cluster_indices].copy()
    cluster_payload = _run_isolated_single_point(
        fragment_atoms=cluster_atoms,
        sp_dir=final_opt_dir / REACTION_HUB_FINAL_CLUSTER_SP_DIRNAME,
        cpus=cpus,
        mem=mem,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=_resolve_spin_multiplicity(cluster_atoms, None, system_charge),
        label="final isolated cluster single-point",
        traj_filename=REACTION_HUB_FINAL_CLUSTER_TRAJ_FILENAME,
        system_charge=system_charge,
        workflow_backend=workflow_backend,
    )
    if cluster_payload.get("status") != "ok":
        payload.update(cluster_payload)
        return payload

    payload.update(cluster_payload)
    payload["cluster_indices_1based"] = [int(idx) + 1 for idx in cluster_indices]
    payload["adsorbate_indices_1based"] = [int(idx) + 1 for idx in adsorbate_indices]

    adsorbate_atoms = final_atoms[adsorbate_indices].copy()
    adsorbate_cpus = max(1, min(int(cpus), MAX_ADSORBATE_SINGLE_POINT_RANKS))
    adsorbate_payload = _run_isolated_single_point(
        fragment_atoms=adsorbate_atoms,
        sp_dir=final_opt_dir / REACTION_HUB_FINAL_ADSORBATE_SP_DIRNAME,
        cpus=adsorbate_cpus,
        mem=mem,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=_resolve_spin_multiplicity(adsorbate_atoms, None),
        label="final isolated adsorbate single-point",
        traj_filename=REACTION_HUB_FINAL_ADSORBATE_TRAJ_FILENAME,
        workflow_backend=workflow_backend,
    )
    payload["adsorbate"] = adsorbate_payload
    return payload


def reapply_bond_constraints(atoms, bond_scan: BondScanSettings, bond_pair: tuple[int, int] | None) -> None:
    if bond_scan.enabled and bond_pair is not None:
        apply_bond_constraints(atoms, bond_scan.fixed_symbols, bond_pair)


@dataclass
class IterationDirs:
    scan_dir: Path
    opt_dir: Path


def _tddft_opt_dir(i: int) -> Path:
    return TDDFT_ROOT_DIR / f"opt_iter_{i}"


def _dft_opt_dir(i: int) -> Path:
    return DFT_ROOT_DIR / f"opt_iter_{i}"


def _tddft_scan_dir(i: int) -> Path:
    return TDDFT_ROOT_DIR / f"scan_iter_{i}"


def _tddft_scan_geom_base(i: int) -> Path:
    return TDDFT_ROOT_DIR / f"scan_iter_{i}_geom"


def _tddft_scan_geom_dirs(i: int) -> list[Path]:
    return sorted(Path(".").glob(f"{_tddft_scan_geom_base(i)}_*"))


def _dft_seed_opt_dirs(i: int) -> list[Path]:
    return [_dft_opt_dir(i)]


def _resolve_opt_dir_for_mode(
    i: int,
    *,
    use_tddft_optimisation: bool,
) -> Path:
    if use_tddft_optimisation:
        return _tddft_opt_dir(i)
    return _dft_opt_dir(i)


def _postprocess_dft_restart_dirs(seed_out: Path) -> list[Path]:
    match = re.fullmatch(r"opt_iter_(\d+)", seed_out.parent.name)
    if match is None:
        return []
    iteration = _parse_optional_int(match.group(1))
    if iteration is None or iteration < 1:
        return []
    return [_dft_opt_dir(int(iteration))]


def _resolve_scan_plots_dir(*, use_tddft_optimisation: bool) -> Path:
    root = TDDFT_ROOT_DIR if use_tddft_optimisation else DFT_ROOT_DIR
    return root / SCAN_PLOTS_DIRNAME


def _resolve_scan_dir(i: int) -> Path:
    return _tddft_scan_dir(i)


def _resolve_scan_postprocess_dir(*, use_tddft_optimisation: bool) -> Path:
    root = TDDFT_ROOT_DIR if use_tddft_optimisation else DFT_ROOT_DIR
    return root / SCAN_POSTPROCESS_DIRNAME


def _max_scan_log_iteration(scan_log_path: Path) -> int:
    rows = load_scan_log(scan_log_path)
    max_iteration = 0
    for key, row in rows.items():
        # Postprocess rows (TS smoothing/refinement) must not redefine the
        # primary scan image count.
        if str(row.get("point_kind") or "").strip():
            continue
        iteration = _parse_optional_int(row.get("iteration"))
        if iteration is None:
            iteration = _parse_optional_int(key)
        if iteration is None or iteration < 1:
            continue
        max_iteration = max(max_iteration, int(iteration))
    return max_iteration


def _scan_dir_mtime(
    path: Path,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> float:
    backend = _workflow_backend_or_default(workflow_backend)
    return scan_log_helpers.scan_dir_mtime(
        path,
        output_path_for_dir=backend.output_path,
    )


def _scan_dir_has_valid_output(scan_dir: Path) -> bool:
    return scan_log_helpers.scan_dir_has_valid_output(
        scan_dir,
        validate_orca_output=lambda out: validate_orca_output(out, warn=False),
    )


def _scan_dir_has_valid_backend_output(
    scan_dir: Path,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> bool:
    backend = _workflow_backend_or_default(workflow_backend)
    return backend.validate_output(backend.output_path(scan_dir), warn=False)


def _resolve_scan_dir_from_hint(
    scan_dir_text: str,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> Path | None:
    backend = _workflow_backend_or_default(workflow_backend)
    return scan_log_helpers.resolve_scan_dir_from_hint(
        scan_dir_text,
        scan_dir_has_valid_output=lambda scan_dir: _scan_dir_has_valid_backend_output(
            scan_dir,
            workflow_backend=backend,
        ),
    )


def _resolve_scan_dir_for_iteration_for_log_repair(
    iteration: int,
    *,
    use_tddft_optimisation: bool,
    workflow_backend: QMWorkflowBackend | None = None,
) -> Path | None:
    backend = _workflow_backend_or_default(workflow_backend)
    return scan_log_helpers.resolve_scan_dir_for_iteration_for_log_repair(
        iteration,
        use_tddft_optimisation=use_tddft_optimisation,
        resolve_scan_dir=_resolve_scan_dir,
        tddft_scan_geom_dirs=_tddft_scan_geom_dirs,
        resolve_opt_dir_for_mode=_resolve_opt_dir_for_mode,
        scan_dir_mtime=lambda path: _scan_dir_mtime(path, workflow_backend=backend),
        output_path_for_dir=backend.output_path,
    )


def _score_components_from_scan_log_row(row: dict[str, str]) -> dict[str, float] | None:
    return scan_log_helpers.score_components_from_scan_log_row(
        row,
        parse_optional_float=_parse_optional_float,
    )


def _rewrite_scan_log_row_scan_dir(
    *,
    scan_log_path: Path,
    row: dict[str, str],
    scan_dir: Path,
) -> bool:
    return scan_log_helpers.rewrite_scan_log_row_scan_dir(
        scan_log_path=scan_log_path,
        row=row,
        scan_dir=scan_dir,
        parse_optional_int=_parse_optional_int,
        parse_optional_float=_parse_optional_float,
        score_components_from_scan_log_row=_score_components_from_scan_log_row,
        record_scan_log_row=record_scan_log_row,
    )


def _repair_scan_log_stale_paths(
    *,
    scan_log_path: Path,
    max_iter: int,
    use_tddft_optimisation: bool,
    workflow_backend: QMWorkflowBackend | None = None,
) -> int:
    backend = _workflow_backend_or_default(workflow_backend)
    return scan_log_helpers.repair_scan_log_stale_paths(
        scan_log_path=scan_log_path,
        max_iter=max_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        load_scan_log=load_scan_log,
        parse_optional_int=_parse_optional_int,
        scan_dir_has_valid_output=lambda scan_dir: _scan_dir_has_valid_backend_output(
            scan_dir,
            workflow_backend=backend,
        ),
        resolve_scan_dir_from_hint=lambda text: _resolve_scan_dir_from_hint(
            text,
            workflow_backend=backend,
        ),
        resolve_scan_dir_for_iteration_for_log_repair=lambda iteration, **kwargs: _resolve_scan_dir_for_iteration_for_log_repair(
            iteration,
            workflow_backend=backend,
            **kwargs,
        ),
        rewrite_scan_log_row_scan_dir=_rewrite_scan_log_row_scan_dir,
        summarize_scan_log=lambda path: summarize_scan_log(path),
    )


def _load_last_geometry_log_row_for_iteration(
    geometry_log_path: Path,
    *,
    iteration: int,
) -> dict[str, str] | None:
    return scan_log_helpers.load_last_geometry_log_row_for_iteration(
        geometry_log_path,
        iteration=iteration,
        parse_optional_int=_parse_optional_int,
    )


def _backfill_missing_primary_scan_log_rows(
    *,
    scan_log_path: Path,
    max_iter: int,
    use_tddft_optimisation: bool,
    bond_pair: tuple[int, int] | None,
    fragments: dict[str, str],
    target_wavelength_nm: float,
    workflow_backend: QMWorkflowBackend | None = None,
) -> int:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    return scan_log_helpers.backfill_missing_primary_scan_log_rows(
        scan_log_path=scan_log_path,
        max_iter=max_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        bond_pair=bond_pair,
        fragments=fragments,
        target_wavelength_nm=target_wavelength_nm,
        geometry_log_filename=IROOT_GEOMETRY_LOG_FILENAME,
        load_scan_log=load_scan_log,
        parse_optional_int=_parse_optional_int,
        parse_optional_float=_parse_optional_float,
        validate_orca_output=lambda out: workflow_backend.validate_output(
            workflow_backend.output_path(Path(out).parent),
            warn=False,
        ),
        resolve_opt_dir_for_mode=_resolve_opt_dir_for_mode,
        resolve_scan_dir_from_hint=lambda text: _resolve_scan_dir_from_hint(
            text,
            workflow_backend=workflow_backend,
        ),
        resolve_scan_dir_for_iteration_for_log_repair=lambda iteration, **kwargs: _resolve_scan_dir_for_iteration_for_log_repair(
            iteration,
            workflow_backend=workflow_backend,
            **kwargs,
        ),
        scan_dir_has_valid_output=lambda scan_dir: _scan_dir_has_valid_backend_output(
            scan_dir,
            workflow_backend=workflow_backend,
        ),
        record_scan_log_row=record_scan_log_row,
        score_components_from_scan_log_row=_score_components_from_scan_log_row,
        recover_scan_choice=recover_scan_choice,
        score_weights=IROOT_SCORE_WEIGHTS,
        wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
        similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
        similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
        min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
        bond_length_from_orca=workflow_backend.bond_length_from_output,
        record_scan_choice=record_scan_choice,
        get_last_iroot=lambda out: workflow_backend.get_followed_state(
            workflow_backend.output_path(Path(out).parent)
        ),
        summarize_scan_log=lambda path: summarize_scan_log(path),
        load_last_geometry_log_row_for_iteration=lambda path, iteration: _load_last_geometry_log_row_for_iteration(
            path,
            iteration=iteration,
        ),
    )


def _max_opt_iteration_on_disk(*, use_tddft_optimisation: bool) -> int:
    root = TDDFT_ROOT_DIR if use_tddft_optimisation else DFT_ROOT_DIR
    if not root.exists():
        return 0
    max_iteration = 0
    for candidate in root.glob("opt_iter_*"):
        if not candidate.is_dir():
            continue
        suffix = candidate.name.removeprefix("opt_iter_")
        iteration = _parse_optional_int(suffix)
        if iteration is None or iteration < 1:
            continue
        max_iteration = max(max_iteration, int(iteration))
    return max_iteration


def _postprocess_smoothing_iteration(ts_iteration: int) -> int:
    return int(SCAN_POSTPROCESS_SMOOTHING_ITERATION_BASE + ts_iteration)


def _postprocess_refinement_iteration(ts_iteration: int, point_index: int) -> int:
    return int(SCAN_POSTPROCESS_REFINEMENT_ITERATION_BASE + ts_iteration * 100 + point_index)


@dataclass
class _EndpointPoint:
    iteration: int
    opt_dir: Path
    opt_out: Path
    energy_eV: float
    bond_length_A: float | None


def _collect_endpoint_points_for_postprocess(
    *,
    max_iter: int,
    use_tddft_optimisation: bool,
    bond_pair: tuple[int, int],
    workflow_backend: QMWorkflowBackend | None = None,
) -> list[_EndpointPoint]:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    points: list[_EndpointPoint] = []
    for iteration in range(1, max_iter + 1):
        opt_dir = _resolve_opt_dir_for_mode(
            iteration,
            use_tddft_optimisation=use_tddft_optimisation,
        )
        opt_out = workflow_backend.output_path(opt_dir)
        if not workflow_backend.validate_output(opt_out, warn=False):
            continue
        try:
            endpoint_atoms = _read_orca_output_with_charges(
                opt_out,
                label=f"scan endpoint iter {iteration} for postprocess",
                workflow_backend=workflow_backend,
            )
        except Exception as exc:
            print(f"Warning: Failed to read endpoint geometry from {opt_out}: {exc}")
            continue
        energy_eV = _resolve_total_energy_eV(
            endpoint_atoms,
            orca_out=opt_out,
            workflow_backend=workflow_backend,
        )
        if energy_eV is None:
            continue
        try:
            bond_length = float(endpoint_atoms.get_distance(*bond_pair))
        except Exception:
            bond_length = None
        points.append(
            _EndpointPoint(
                iteration=int(iteration),
                opt_dir=opt_dir,
                opt_out=opt_out,
                energy_eV=float(energy_eV),
                bond_length_A=bond_length,
            )
        )
    points.sort(key=lambda item: int(item.iteration))
    return points


def _run_postprocess_optimisation(
    *,
    label: str,
    output_dir: Path,
    seed_out: Path,
    target_bond_length: float,
    bond_pair: tuple[int, int],
    bond_scan: BondScanSettings,
    use_tddft_optimisation: bool,
    iroot: int | None,
    cpus: int,
    mem: int,
    tddft_calc_factory,
    dft_calc_factory,
    fragments: dict[str, str] | None = None,
    target_wavelength_nm: float = TARGET_WAVELENGTH_NM,
    previous_choice: IRootCandidate | None = None,
    max_state_follow_attempts: int = 3,
    scan_nroots: int = TDDFT_SCAN_ROOTS,
    fmax: float = 0.01,
    active_learning_optimizer: str = AL_LOCAL_OPTIMIZER_DEFAULT,
    al_steps_per_call: int = 20,
    max_steps: int | None = None,
    dft_restart_iteration: int | None = None,
    enforce_dft_restart_iteration: bool = False,
    workflow_backend: QMWorkflowBackend | None = None,
):
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_out = workflow_backend.output_path(output_dir)
    fragments = fragments or {}
    requested_iroot = int(iroot) if iroot is not None else None
    preferred_dft_restart_dirs: list[Path] = []
    if dft_restart_iteration is not None and int(dft_restart_iteration) > 0:
        preferred_dft_restart_dirs.append(_dft_opt_dir(int(dft_restart_iteration)))
    for candidate in _postprocess_dft_restart_dirs(seed_out):
        if candidate not in preferred_dft_restart_dirs:
            preferred_dft_restart_dirs.append(candidate)
    primary_dft_restart_dir = preferred_dft_restart_dirs[0] if preferred_dft_restart_dirs else None
    if bool(enforce_dft_restart_iteration):
        if primary_dft_restart_dir is None:
            raise RuntimeError(
                "TS smoothing requires a corresponding DFT restart iteration, but none was "
                f"resolved for postprocess point {label}."
            )
        if (
            workflow_backend.capabilities.requires_restart_data
            and workflow_backend.resolve_restart_path(primary_dft_restart_dir) is None
        ):
            raise RuntimeError(
                f"TS smoothing requires {workflow_backend.label} restart data from "
                f"corresponding DFT iteration {primary_dft_restart_dir}, but no restart "
                "file was found "
                f"for postprocess point {label}."
            )
        # Strict smoothing mode: only allow the corresponding DFT restart dir.
        preferred_dft_restart_dirs = [primary_dft_restart_dir]
        try:
            workflow_backend.copy_restart(primary_dft_restart_dir, output_dir, verbose=False)
        except Exception as exc:
            raise RuntimeError(
                f"TS smoothing failed to stage {workflow_backend.label} restart data from corresponding DFT "
                f"iteration {primary_dft_restart_dir} for postprocess point {label}: {exc}"
            )

    if workflow_backend.validate_output(output_out, warn=False):
        if (not use_tddft_optimisation) or _has_tddft_state_tag(
            output_out,
            workflow_backend=workflow_backend,
        ):
            try:
                existing_atoms = _read_orca_output_with_charges(
                    output_out,
                    label=f"scan postprocess {label}",
                    workflow_backend=workflow_backend,
                )
                existing_choice = None
                existing_iroot = requested_iroot
                if use_tddft_optimisation:
                    existing_choice = _resolve_tddft_postprocess_choice(
                        scan_dir=output_dir,
                        fragments=fragments,
                        target_wavelength_nm=target_wavelength_nm,
                        previous_choice=previous_choice,
                        previous_state=requested_iroot,
                        workflow_backend=workflow_backend,
                    )
                    if existing_choice is not None:
                        existing_iroot = int(existing_choice.state)
                    elif existing_iroot is None:
                        existing_iroot = workflow_backend.get_followed_state(output_out)
                return existing_atoms, existing_iroot, existing_choice
            except Exception as exc:
                print(f"Warning: Failed to load existing postprocess output {output_out}: {exc}")
    follow_attempts = 1
    if use_tddft_optimisation:
        follow_attempts = max(1, int(max_state_follow_attempts))

    current_seed_out = seed_out
    last_atoms = None
    last_choice: IRootCandidate | None = None
    last_iroot = requested_iroot

    for attempt in range(1, follow_attempts + 1):
        try:
            seed_atoms = _read_orca_output_with_charges(
                current_seed_out,
                label=f"scan postprocess seed for {label}",
                workflow_backend=workflow_backend,
            )
        except Exception as exc:
            print(f"Warning: Failed to load seed geometry {current_seed_out} for {label}: {exc}")
            return None, None, None

        seed_atoms.set_constraint(None)
        seed_atoms.set_distance(
            bond_pair[0],
            bond_pair[1],
            float(target_bond_length),
            fix=float(bond_scan.fix),
        )
        apply_bond_constraints(seed_atoms, bond_scan.fixed_symbols, bond_pair)
        restart_fallback_dirs = [*preferred_dft_restart_dirs]
        if not bool(enforce_dft_restart_iteration):
            restart_fallback_dirs.extend(
                [
                    current_seed_out.parent,
                    seed_out.parent,
                    INITIAL_DFT_DIR,
                    DFT_ROOT_DIR,
                ]
            )
        _ensure_restart_gbw(
            output_dir,
            fallback_dirs=restart_fallback_dirs,
            workflow_backend=workflow_backend,
        )

        result_out = output_out
        try:
            al_dir = output_dir / "active_learning"
            training_path = al_dir / "evaluated.traj"
            seed_dir = al_dir / "seed"

            if use_tddft_optimisation:
                if requested_iroot is None:
                    print(f"Warning: Missing IROOT for TDDFT postprocess point {label}; skipping.")
                    return None, None, None
                preferred_seed_source = (
                    primary_dft_restart_dir
                    if primary_dft_restart_dir is not None
                    else current_seed_out.parent
                )
                seed_gbw_fallback_dirs = [*preferred_dft_restart_dirs]
                if not bool(enforce_dft_restart_iteration):
                    seed_gbw_fallback_dirs.extend(
                        [
                            current_seed_out.parent,
                            seed_out.parent,
                            output_dir,
                            INITIAL_DFT_DIR,
                            DFT_ROOT_DIR,
                        ]
                    )
                seed_atoms, gbw_source_dir = ensure_training_seed(
                    seed_atoms,
                    training_path=training_path,
                    seed_dir=seed_dir,
                    gbw_source_dir=preferred_seed_source,
                    gbw_fallback_dirs=seed_gbw_fallback_dirs,
                    iroot=int(requested_iroot),
                    cpus=cpus,
                    mem=mem,
                    calc_factory=tddft_calc_factory,
                )
                post_seed_restart_fallback_dirs = [*preferred_dft_restart_dirs]
                if not bool(enforce_dft_restart_iteration):
                    post_seed_restart_fallback_dirs.extend(
                        [
                            gbw_source_dir,
                            seed_dir,
                            current_seed_out.parent,
                            seed_out.parent,
                            INITIAL_DFT_DIR,
                            DFT_ROOT_DIR,
                        ]
                    )
                _ensure_restart_gbw(
                    output_dir,
                    fallback_dirs=post_seed_restart_fallback_dirs,
                    workflow_backend=workflow_backend,
                )
                calc = tddft_calc_factory(
                    seed_atoms,
                    cpus=cpus,
                    mem=mem,
                    task="engrad",
                    directory=output_dir,
                    nroots=int(scan_nroots),
                    iroot=int(requested_iroot),
                )
            else:
                seed_atoms = ensure_dft_training_seed(
                    seed_atoms,
                    training_path=training_path,
                    seed_dir=seed_dir,
                    cpus=cpus,
                    mem=mem,
                    calc_factory=dft_calc_factory,
                    workflow_backend=workflow_backend,
                )
                _ensure_restart_gbw(
                    output_dir,
                    fallback_dirs=[
                        seed_dir,
                        *preferred_dft_restart_dirs,
                        current_seed_out.parent,
                        seed_out.parent,
                        INITIAL_DFT_DIR,
                        DFT_ROOT_DIR,
                    ],
                    workflow_backend=workflow_backend,
                )
                calc = dft_calc_factory(
                    seed_atoms,
                    cpus=cpus,
                    mem=mem,
                    task="engrad",
                    directory=output_dir,
                )

            seed_atoms = active_learning_step(
                seed_atoms,
                calc,
                fmax=fmax,
                shared_training_path=training_path,
                state_dir=al_dir,
                use_restart=is_valid_trainingset(training_path),
                steps_per_call=max(1, int(al_steps_per_call)),
                optimize_to_convergence=True,
                max_steps=(max(1, int(max_steps)) if max_steps is not None else None),
                max_unc=AL_PREDICTION_MAX_UNC,
                min_data=AL_MIN_DATA,
                ml_steps=AL_ML_STEPS,
                dtrust=AL_DTRUST,
                raise_on_not_converged=True,
                local_optimizer=active_learning_optimizer,
            )
            _recover_restart_from_seed(
                output_dir,
                seed_dir,
                workflow_backend=workflow_backend,
            )
        except Exception as exc:
            fallback_seed_out = workflow_backend.output_path(seed_dir)
            if (
                "no information about the structure in the orca output file"
                in str(exc).lower()
                and workflow_backend.validate_output(fallback_seed_out, warn=False)
            ):
                result_out = fallback_seed_out
                if output_out != fallback_seed_out:
                    try:
                        shutil.copy2(fallback_seed_out, output_out)
                        fallback_seed_gradient = workflow_backend.gradient_path(seed_dir)
                        output_gradient = workflow_backend.gradient_path(output_dir)
                        if (
                            fallback_seed_gradient is not None
                            and output_gradient is not None
                            and fallback_seed_gradient.exists()
                        ):
                            shutil.copy2(fallback_seed_gradient, output_gradient)
                        result_out = output_out
                    except OSError as copy_exc:
                        print(
                            "Warning: Failed to stage fallback postprocess output "
                            f"from {fallback_seed_out} to {output_out}: {copy_exc}"
                        )
                print(
                    "Warning: Postprocess active-learning optimisation for "
                    f"{label} could not parse structure from {output_out} ({exc}); "
                    f"using seed output {fallback_seed_out}."
                )
            else:
                print(f"Warning: Failed postprocess active-learning optimisation for {label}: {exc}")
                return None, None, None

        if not workflow_backend.validate_output(result_out, warn=False):
            print(f"Warning: Postprocess active-learning optimisation did not converge for {label} ({result_out}).")
            return None, None, None

        try:
            last_atoms = _read_orca_output_with_charges(
                result_out,
                label=f"scan postprocess result {label}",
                workflow_backend=workflow_backend,
            )
        except Exception as exc:
            print(f"Warning: Failed to read postprocess output {result_out}: {exc}")
            return None, None, None

        if not use_tddft_optimisation:
            return last_atoms, None, None

        follow_choice = _resolve_tddft_postprocess_choice(
            scan_dir=output_dir,
            fragments=fragments,
            target_wavelength_nm=target_wavelength_nm,
            previous_choice=previous_choice,
            previous_state=requested_iroot,
            workflow_backend=workflow_backend,
        )
        if follow_choice is not None:
            last_choice = follow_choice
            last_iroot = int(follow_choice.state)
        else:
            last_choice = None
            fallback_iroot = workflow_backend.get_followed_state(result_out)
            if fallback_iroot is not None:
                last_iroot = int(fallback_iroot)
            else:
                last_iroot = requested_iroot
            return last_atoms, last_iroot, last_choice

        if (
            requested_iroot is not None
            and int(follow_choice.state) != int(requested_iroot)
            and attempt < follow_attempts
        ):
            print(
                "Info: Postprocess TDDFT state following for "
                f"{label} switched IROOT {requested_iroot} -> {follow_choice.state}; rerunning."
            )
            requested_iroot = int(follow_choice.state)
            current_seed_out = result_out
            continue

        return last_atoms, last_iroot, last_choice

    return last_atoms, last_iroot, last_choice


def _extract_state_payload_for_root(
    orca_out: Path,
    iroot: int | None,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> dict[str, float | int]:
    if iroot is None:
        return {}
    if iroot < 1:
        return {"iroot": int(iroot)}
    states = _extract_excited_states(orca_out, workflow_backend=workflow_backend)
    if not states or iroot > len(states):
        return {"iroot": int(iroot)}
    state = states[iroot - 1]
    wavelength_nm = _parse_optional_float(state.get("wavelength_nm"))
    intensity = _parse_optional_float(state.get("oscillator_strength"))
    energy_ev = _parse_optional_float(state.get("energy_ev"))
    payload: dict[str, float | int] = {"iroot": int(iroot)}
    if wavelength_nm is not None:
        payload["wavelength_nm"] = float(wavelength_nm)
    if intensity is not None:
        payload["intensity"] = float(intensity)
    if energy_ev is not None:
        payload["energy_ev"] = float(energy_ev)
    return payload


def _resolve_tddft_postprocess_choice(
    *,
    scan_dir: Path,
    fragments: dict[str, str],
    target_wavelength_nm: float,
    previous_choice: IRootCandidate | None,
    previous_state: int | None,
    workflow_backend: QMWorkflowBackend | None = None,
) -> IRootCandidate | None:
    backend = _workflow_backend_or_default(workflow_backend)
    if not fragments:
        return None
    try:
        df = backend.get_ct_info(
            scan_dir,
            fragments,
            min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
        )
    except Exception as exc:
        print(f"Warning: Failed TDDFT state extraction in postprocess {scan_dir}: {exc}")
        return None

    for score_weights, min_similarity in (
        (IROOT_FOLLOWUP_SCORE_WEIGHTS, IROOT_MIN_SIMILARITY),
        (IROOT_SCORE_WEIGHTS, 0.0),
    ):
        enforce_overlap = (
            score_weights == IROOT_FOLLOWUP_SCORE_WEIGHTS
            and (previous_choice is not None or previous_state is not None)
        )
        try:
            choice, _ = backend.pick_iroot(
                df,
                target_wavelength_nm=target_wavelength_nm,
                previous_choice=previous_choice,
                previous_state=previous_state,
                score_weights=score_weights,
                wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
                similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
                similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
                min_similarity=min_similarity,
                min_relative_intensity=IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY,
                intensity_reference_window_nm=IROOT_INTENSITY_REFERENCE_WINDOW_NM,
                min_transition_overlap=(
                    IROOT_MIN_TRANSITION_OVERLAP if enforce_overlap else 0.0
                ),
                min_ct_similarity=IROOT_MIN_CT_SIMILARITY if enforce_overlap else 0.0,
                require_transition_overlap=enforce_overlap,
                state_vectors=backend.extract_state_transition_vectors(
                    backend.output_path(scan_dir)
                ),
            )
        except Exception as exc:
            print(f"Warning: Failed TDDFT state scoring in postprocess {scan_dir}: {exc}")
            continue
        if choice is not None:
            return choice
    return None


def _record_postprocess_scan_log_entry(
    *,
    scan_log_path: Path,
    iteration: int,
    scan_dir: Path,
    bond_pair: tuple[int, int] | None,
    bond_length: float | None,
    target_wavelength_nm: float,
    point_kind: str,
    source_iteration: int | None,
    reference_iteration: int | None,
    point_label: str,
    state_payload: dict[str, float | int] | None = None,
    choice: IRootCandidate | None = None,
) -> None:
    state_payload = state_payload or {}
    iroot = _parse_optional_int(state_payload.get("iroot"))
    wavelength_nm = _parse_optional_float(state_payload.get("wavelength_nm"))
    intensity = _parse_optional_float(state_payload.get("intensity"))
    energy_ev = _parse_optional_float(state_payload.get("energy_ev"))
    if choice is not None:
        if iroot is None:
            iroot = int(choice.state)
        if wavelength_nm is None:
            wavelength_nm = float(choice.wavelength_nm)
        if intensity is None:
            intensity = float(choice.intensity)
        if energy_ev is None:
            energy_ev = _parse_optional_float(choice.energy_ev)

    record_scan_log_row(
        scan_log_path,
        iteration=int(iteration),
        scan_dir=scan_dir,
        bond_pair=bond_pair,
        bond_length=bond_length,
        iroot=iroot,
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        energy_ev=energy_ev,
        ct_e=_parse_optional_float(choice.ct_e) if choice is not None else None,
        ct_pct=_parse_optional_float(choice.ct_pct) if choice is not None else None,
        ct_net=_parse_optional_float(choice.ct_net) if choice is not None else None,
        ct_magnitude=float(choice.ct_magnitude) if choice is not None else None,
        score=_parse_optional_float(choice.score) if choice is not None else None,
        similarity=_parse_optional_float(choice.similarity) if choice is not None else None,
        score_components=dict(choice.score_components or {}) if choice is not None and choice.score_components else None,
        target_wavelength_nm=target_wavelength_nm,
        point_kind=point_kind,
        source_iteration=source_iteration,
        reference_iteration=reference_iteration,
        point_label=point_label,
    )


def _run_scan_smoothing_postprocess(
    *,
    max_iter: int,
    use_tddft_optimisation: bool,
    bond_scan: BondScanSettings,
    bond_pair: tuple[int, int] | None,
    scan_log_path: Path,
    target_wavelength_nm: float,
    cpus: int,
    mem: int,
    tddft_calc_factory,
    dft_calc_factory,
    fragments: dict[str, str] | None = None,
    scan_nroots: int = TDDFT_SCAN_ROOTS,
    fmax: float = 0.01,
    active_learning_optimizer: str = AL_LOCAL_OPTIMIZER_DEFAULT,
    al_steps_per_call: int = 20,
    max_steps: int | None = None,
    workflow_backend: QMWorkflowBackend | None = None,
) -> list[int]:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    fragments = fragments or {}
    refine_points_each_side = max(int(bond_scan.refine_points_each_side), 0)
    if (
        not bond_scan.enabled
        or bond_pair is None
        or (
            not bool(bond_scan.enable_ts_smoothing)
            and refine_points_each_side <= 0
        )
    ):
        return []

    points = _collect_endpoint_points_for_postprocess(
        max_iter=max_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        bond_pair=bond_pair,
        workflow_backend=workflow_backend,
    )
    if len(points) < 2:
        print("Warning: Not enough converged points to run scan smoothing postprocess.")
        return []

    candidate_indices = [idx for idx in range(len(points) - 1)]
    if not candidate_indices:
        print("Warning: Unable to locate a transition-state candidate for scan smoothing.")
        return []

    initial_ts_index = max(
        candidate_indices,
        key=lambda idx: (float(points[idx].energy_eV), -int(points[idx].iteration)),
    )
    scan_rows = _scan_log_by_iteration(scan_log_path) if use_tddft_optimisation else {}

    def _resolve_ts_iroot(point: _EndpointPoint) -> int | None:
        if not use_tddft_optimisation:
            return None
        ts_row = scan_rows.get(point.iteration)
        ts_iroot = _parse_optional_int(ts_row.get("iroot")) if ts_row is not None else None
        if ts_iroot is None:
            ts_iroot = workflow_backend.get_followed_state(point.opt_out)
        return ts_iroot

    def _resolve_scan_choice(iteration: int) -> IRootCandidate | None:
        if not use_tddft_optimisation:
            return None
        return _iroot_candidate_from_scan_row(scan_rows.get(int(iteration)))

    created_iterations: list[int] = []
    postprocess_root = _resolve_scan_postprocess_dir(
        use_tddft_optimisation=use_tddft_optimisation
    )
    postprocess_root.mkdir(parents=True, exist_ok=True)

    resolved_ts_index: int | None = None
    resolved_ts_iroot: int | None = None
    resolved_ts_choice: IRootCandidate | None = None
    max_allowed_ts_drop_eV = 0.05

    if bool(bond_scan.enable_ts_smoothing):
        search_index = int(initial_ts_index)
        while search_index < len(points) - 1:
            ts_point = points[search_index]
            next_point = points[search_index + 1]
            if ts_point.bond_length_A is None:
                print(
                    "Warning: Missing TS bond length for iteration "
                    f"{ts_point.iteration}; checking next candidate."
                )
                search_index += 1
                continue

            ts_iroot = _resolve_ts_iroot(ts_point)
            if use_tddft_optimisation and ts_iroot is None:
                print(
                    "Warning: Unable to resolve TS IROOT for TDDFT scan smoothing at "
                    f"iteration {ts_point.iteration}; checking next candidate."
                )
                search_index += 1
                continue

            ts_choice = _resolve_scan_choice(ts_point.iteration)
            ts_bond = float(ts_point.bond_length_A)
            smoothing_iteration = _postprocess_smoothing_iteration(ts_point.iteration)
            smoothing_label = (
                f"smooth_ts_iter_{ts_point.iteration:03d}_from_iter_{next_point.iteration:03d}"
            )
            smoothing_dir = postprocess_root / smoothing_label
            smoothing_atoms, smoothing_iroot, smoothing_choice = _run_postprocess_optimisation(
                label=smoothing_label,
                output_dir=smoothing_dir,
                seed_out=next_point.opt_out,
                target_bond_length=ts_bond,
                bond_pair=bond_pair,
                bond_scan=bond_scan,
                use_tddft_optimisation=use_tddft_optimisation,
                iroot=ts_iroot,
                cpus=cpus,
                mem=mem,
                tddft_calc_factory=tddft_calc_factory,
                dft_calc_factory=dft_calc_factory,
                fragments=fragments,
                target_wavelength_nm=target_wavelength_nm,
                previous_choice=ts_choice,
                scan_nroots=scan_nroots,
                fmax=fmax,
                active_learning_optimizer=active_learning_optimizer,
                al_steps_per_call=al_steps_per_call,
                max_steps=max_steps,
                dft_restart_iteration=ts_point.iteration,
                enforce_dft_restart_iteration=True,
                workflow_backend=workflow_backend,
            )
            if smoothing_atoms is None:
                print(
                    "Warning: TS smoothing failed for iteration "
                    f"{ts_point.iteration}; checking next candidate."
                )
                search_index += 1
                continue

            smoothing_out = workflow_backend.output_path(smoothing_dir)
            smoothed_total_energy = _resolve_total_energy_eV(
                smoothing_atoms,
                orca_out=smoothing_out,
                workflow_backend=workflow_backend,
            )
            if smoothed_total_energy is None:
                print(
                    "Warning: Missing smoothed TS energy for iteration "
                    f"{ts_point.iteration}; checking next candidate."
                )
                search_index += 1
                continue

            original_ts_energy = float(ts_point.energy_eV)
            ts_energy = float(smoothed_total_energy)
            energy_drop = float(original_ts_energy - float(ts_energy))
            if energy_drop > float(max_allowed_ts_drop_eV) + 1.0e-8:
                print(
                    "Info: TS candidate iteration "
                    f"{ts_point.iteration} lowered by {energy_drop:.6f} eV "
                    f"(threshold {max_allowed_ts_drop_eV:.2f} eV); checking next candidate."
                )
                search_index += 1
                continue

            bond_length = _parse_optional_float(
                smoothing_atoms.get_distance(*bond_pair)
            )
            state_payload = (
                    _extract_state_payload_for_root(
                        smoothing_out,
                        smoothing_iroot,
                        workflow_backend=workflow_backend,
                    )
                if use_tddft_optimisation
                else {}
            )
            _record_postprocess_scan_log_entry(
                scan_log_path=scan_log_path,
                iteration=smoothing_iteration,
                scan_dir=smoothing_dir,
                bond_pair=bond_pair,
                bond_length=bond_length,
                target_wavelength_nm=target_wavelength_nm,
                point_kind="ts_smoothing",
                source_iteration=ts_point.iteration,
                reference_iteration=next_point.iteration,
                point_label=smoothing_label,
                state_payload=state_payload,
                choice=smoothing_choice,
            )
            created_iterations.append(int(smoothing_iteration))

            resolved_ts_index = search_index
            resolved_ts_iroot = smoothing_iroot if smoothing_iroot is not None else ts_iroot
            resolved_ts_choice = smoothing_choice or ts_choice
            break
    else:
        resolved_ts_index = int(initial_ts_index)
        ts_point = points[resolved_ts_index]
        resolved_ts_iroot = _resolve_ts_iroot(ts_point)
        resolved_ts_choice = _resolve_scan_choice(ts_point.iteration)
        if use_tddft_optimisation and resolved_ts_iroot is None:
            print(
                "Warning: Unable to resolve TS IROOT for TDDFT scan smoothing; "
                "skipping postprocess points."
            )
            return []

    if resolved_ts_index is None:
        print(
            "Warning: Unable to locate a transition-state candidate on the return branch "
            "after TS smoothing attempts."
        )
        if created_iterations:
            summarize_scan_log(scan_log_path)
        return created_iterations

    ts_point = points[resolved_ts_index]
    next_point = points[resolved_ts_index + 1] if resolved_ts_index + 1 < len(points) else None
    if next_point is None:
        print("Warning: No point after resolved TS candidate; skipping refinement points.")
        if created_iterations:
            summarize_scan_log(scan_log_path)
        return created_iterations
    if ts_point.bond_length_A is None:
        print("Warning: Missing resolved TS bond length; skipping refinement points.")
        if created_iterations:
            summarize_scan_log(scan_log_path)
        return created_iterations

    ts_bond = float(ts_point.bond_length_A)
    if refine_points_each_side > 0:
        left_neighbor = points[resolved_ts_index - 1] if resolved_ts_index > 0 else None
        right_neighbor = next_point
        refinement_index = 0
        for side, neighbor in (("left", left_neighbor), ("right", right_neighbor)):
            if neighbor is None or neighbor.bond_length_A is None:
                continue
            delta = float(neighbor.bond_length_A) - ts_bond
            if abs(delta) <= 1.0e-8:
                continue
            side_iroot = resolved_ts_iroot
            side_previous_choice = resolved_ts_choice
            for rank in range(1, refine_points_each_side + 1):
                refinement_index += 1
                fraction = rank / float(refine_points_each_side + 1)
                target_distance = ts_bond + delta * fraction
                refinement_iteration = _postprocess_refinement_iteration(
                    ts_point.iteration,
                    refinement_index,
                )
                refinement_label = (
                    f"refine_ts_iter_{ts_point.iteration:03d}_{side}_{rank:02d}"
                )
                refinement_dir = postprocess_root / refinement_label
                refinement_atoms, refinement_iroot, refinement_choice = _run_postprocess_optimisation(
                    label=refinement_label,
                    output_dir=refinement_dir,
                    seed_out=neighbor.opt_out,
                    target_bond_length=target_distance,
                    bond_pair=bond_pair,
                    bond_scan=bond_scan,
                    use_tddft_optimisation=use_tddft_optimisation,
                    iroot=side_iroot,
                    cpus=cpus,
                    mem=mem,
                    tddft_calc_factory=tddft_calc_factory,
                    dft_calc_factory=dft_calc_factory,
                    fragments=fragments,
                    target_wavelength_nm=target_wavelength_nm,
                    previous_choice=side_previous_choice,
                    scan_nroots=scan_nroots,
                    fmax=fmax,
                    active_learning_optimizer=active_learning_optimizer,
                    al_steps_per_call=al_steps_per_call,
                    max_steps=max_steps,
                    workflow_backend=workflow_backend,
                )
                if refinement_atoms is None:
                    continue
                refinement_out = workflow_backend.output_path(refinement_dir)
                bond_length = _parse_optional_float(
                    refinement_atoms.get_distance(*bond_pair)
                )
                state_payload = (
                    _extract_state_payload_for_root(
                        refinement_out,
                        refinement_iroot,
                        workflow_backend=workflow_backend,
                    )
                    if use_tddft_optimisation
                    else {}
                )
                _record_postprocess_scan_log_entry(
                    scan_log_path=scan_log_path,
                    iteration=refinement_iteration,
                    scan_dir=refinement_dir,
                    bond_pair=bond_pair,
                    bond_length=bond_length,
                    target_wavelength_nm=target_wavelength_nm,
                    point_kind=f"ts_refine_{side}",
                    source_iteration=ts_point.iteration,
                    reference_iteration=neighbor.iteration,
                    point_label=refinement_label,
                    state_payload=state_payload,
                    choice=refinement_choice,
                )
                created_iterations.append(int(refinement_iteration))
                if use_tddft_optimisation:
                    if refinement_iroot is not None:
                        side_iroot = int(refinement_iroot)
                    if refinement_choice is not None:
                        side_previous_choice = refinement_choice

    if created_iterations:
        summarize_scan_log(scan_log_path)
        print(
            "Scan smoothing postprocess completed: TS iteration "
            f"{ts_point.iteration}, created {len(created_iterations)} point(s)."
        )
    return created_iterations


def _select_last_completed_opt_dir(
    max_iter: int,
    *,
    use_tddft_optimisation: bool,
    workflow_backend: QMWorkflowBackend | None = None,
) -> Path | None:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    for iteration in range(max_iter, 0, -1):
        candidate = _resolve_opt_dir_for_mode(
            iteration,
            use_tddft_optimisation=use_tddft_optimisation,
        )
        if workflow_backend.validate_output(workflow_backend.output_path(candidate), warn=False):
            return candidate
    return None


def _scan_energy_profile_from_db(scan_db_path: Path) -> dict[str, object]:
    if not scan_db_path.exists():
        return {}

    def _row_lookup(row, key: str):
        key_values = dict(getattr(row, "key_value_pairs", {}) or {})
        if key in key_values:
            return key_values[key]
        data = dict(getattr(row, "data", {}) or {})
        if key in data:
            return data[key]
        return getattr(row, key, None)

    points: list[dict[str, object]] = []
    try:
        db = connect(str(scan_db_path))
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
            bond_length_A = _parse_optional_float(_row_lookup(row, "scan_bond_length"))
            max_force_eV_A = _parse_optional_float(_row_lookup(row, "max_force_eV_A"))
            point: dict[str, object] = {
                "iteration": int(iteration),
                "energy_eV": float(energy_eV),
            }
            if bond_length_A is not None:
                point["bond_length_A"] = float(bond_length_A)
            if max_force_eV_A is not None:
                point["max_force_eV_A"] = float(max_force_eV_A)
            points.append(point)
    except Exception as exc:
        print(f"Warning: Failed to read scan profile from {scan_db_path}: {exc}")
        return {}

    if not points:
        return {}

    points.sort(key=lambda item: int(item.get("iteration", 0)))
    start = points[0]
    peak = max(points, key=lambda item: float(item.get("energy_eV", 0.0)))
    start_energy = float(start.get("energy_eV", 0.0))
    peak_energy = float(peak.get("energy_eV", 0.0))
    profile: dict[str, object] = {
        "points": points,
        "start_iteration": int(start.get("iteration", 1)),
        "start_energy_eV": start_energy,
        "peak_iteration": int(peak.get("iteration", 1)),
        "peak_energy_eV": peak_energy,
        "activation_barrier_eV": float(peak_energy - start_energy),
    }
    if "bond_length_A" in start:
        profile["start_bond_length_A"] = float(start["bond_length_A"])  # type: ignore[index]
    if "bond_length_A" in peak:
        profile["peak_bond_length_A"] = float(peak["bond_length_A"])  # type: ignore[index]
    return profile


def _collect_key_plot_paths(*, use_tddft_optimisation: bool) -> dict[str, object]:
    plots_dir = _resolve_scan_plots_dir(use_tddft_optimisation=use_tddft_optimisation)
    payload: dict[str, object] = {"plots_dir": str(plots_dir)}
    if not plots_dir.exists():
        return payload

    key_files = {
        "stacked_uvvis": "stacked_uvvis.png",
        "stacked_ct": "stacked_ct.png",
        "energy_vs_reaction_coordinate": "energy_vs_reaction_coordinate.png",
    }
    if use_tddft_optimisation:
        key_files["converged_state_wavelengths"] = "converged_state_wavelengths.png"
    for key, filename in key_files.items():
        path = plots_dir / filename
        if path.exists():
            payload[key] = str(path)

    image_paths: list[str] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.svg"):
        for image in sorted(plots_dir.glob(ext)):
            if image.is_file():
                # Per-iteration UV-vis images are intentionally omitted from
                # Reaction Hub summary payloads.
                if image.name.startswith("uvvis_iter_"):
                    continue
                image_paths.append(str(image))
    if image_paths:
        payload["images"] = image_paths
    final_spectrum_dir = DFT_ROOT_DIR / REACTION_HUB_FINAL_TDDFT_SPECTRUM_DIRNAME
    final_spectrum_png = final_spectrum_dir / REACTION_HUB_FINAL_TDDFT_SPECTRUM_PNG
    if final_spectrum_png.exists() and final_spectrum_png.is_file():
        payload["final_tddft_spectrum"] = str(final_spectrum_png)
        existing_images = list(payload.get("images", [])) if isinstance(payload.get("images"), list) else []
        if str(final_spectrum_png) not in existing_images:
            existing_images.append(str(final_spectrum_png))
        payload["images"] = existing_images
    cluster_spectrum_png = final_spectrum_dir / REACTION_HUB_CLUSTER_TDDFT_SPECTRUM_PNG
    if cluster_spectrum_png.exists() and cluster_spectrum_png.is_file():
        payload["cluster_tddft_spectrum"] = str(cluster_spectrum_png)
        existing_images = list(payload.get("images", [])) if isinstance(payload.get("images"), list) else []
        if str(cluster_spectrum_png) not in existing_images:
            existing_images.append(str(cluster_spectrum_png))
        payload["images"] = existing_images
    difference_spectrum_png = final_spectrum_dir / REACTION_HUB_INTERACTION_DIFFERENCE_SPECTRUM_PNG
    if difference_spectrum_png.exists() and difference_spectrum_png.is_file():
        payload["cluster_adsorbate_difference_spectrum"] = str(difference_spectrum_png)
        existing_images = list(payload.get("images", [])) if isinstance(payload.get("images"), list) else []
        if str(difference_spectrum_png) not in existing_images:
            existing_images.append(str(difference_spectrum_png))
        payload["images"] = existing_images
    return payload


def _reaction_hub_replot_only() -> bool:
    value = str(os.environ.get(REACTION_HUB_REPLOT_ONLY_ENV, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _run_final_tddft_spectrum(
    *,
    final_atoms,
    final_opt_dir: Path,
    cpus: int,
    mem: int,
    nroots: int,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    spin_multiplicity: int | None,
    system_charge: int = 0,
    workflow_backend: QMWorkflowBackend | None = None,
    spectrum_dirname: str = REACTION_HUB_FINAL_TDDFT_SPECTRUM_DIRNAME,
    csv_filename: str = REACTION_HUB_FINAL_TDDFT_SPECTRUM_CSV,
    png_filename: str = REACTION_HUB_FINAL_TDDFT_SPECTRUM_PNG,
    plot_title: str = "Final TDDFT spectrum",
    plot_output_dir: Path | None = None,
    gradient_root: int | None = None,
) -> dict[str, object]:
    workflow_backend = workflow_backend or _resolve_qm_workflow_backend("orca")
    spectrum_dir = DFT_ROOT_DIR / spectrum_dirname
    spectrum_dir.mkdir(parents=True, exist_ok=True)
    workflow_backend.ensure_restart(spectrum_dir, fallback_dirs=[final_opt_dir, DFT_ROOT_DIR])

    atoms = final_atoms.copy()
    atoms.calc = workflow_backend.make_tddft_calculator(
        atoms,
        cpus=cpus,
        mem=mem,
        task="EnGrad" if gradient_root is not None else "SP",
        directory=spectrum_dir,
        nroots=int(nroots),
        iroot=gradient_root,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
    )
    atoms.get_potential_energy()
    if gradient_root is not None:
        import numpy as np
        forces = atoms.get_forces(apply_constraint=False)
        if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
            raise ValueError("Invalid single-point excited-state forces")
        (spectrum_dir / "root_gradient_results.json").write_text(json.dumps({
            "iroot": gradient_root, "nroots": int(nroots),
            "symbols": atoms.get_chemical_symbols(), "positions_A": atoms.positions.tolist(),
            "energy_eV": float(atoms.get_potential_energy()), "atom_forces_eV_A": forces.tolist(),
            "fmax_eV_A": float(np.linalg.norm(forces, axis=1).max()),
            "force_norm_eV_A": float(np.linalg.norm(forces)),
        }, indent=2), encoding="utf-8")

    out_path = workflow_backend.output_path(spectrum_dir)
    if not workflow_backend.validate_output(out_path):
        raise RuntimeError(f"Final TDDFT spectrum output invalid or missing: {out_path}")

    result: dict[str, object] = {
        "directory": str(spectrum_dir),
        "output": str(out_path),
        "nroots": int(nroots),
    }
    if workflow_backend.key == "orca":
        result["orca_out"] = str(out_path)
    elif workflow_backend.key == "nwchem":
        result["nwchem_out"] = str(out_path)
    result.update(
        _export_final_tddft_spectrum_plot_data(
            spectrum_dir,
            workflow_backend=workflow_backend,
            csv_filename=csv_filename,
            png_filename=png_filename,
            plot_title=plot_title,
            plot_output_dir=plot_output_dir,
            expected_nroots=int(nroots),
        )
    )
    return result


def _export_final_tddft_spectrum_plot_data(
    spectrum_dir: Path,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
    csv_filename: str = REACTION_HUB_FINAL_TDDFT_SPECTRUM_CSV,
    png_filename: str = REACTION_HUB_FINAL_TDDFT_SPECTRUM_PNG,
    plot_title: str = "Final TDDFT spectrum",
    plot_output_dir: Path | None = None,
    transitions=None,
    expected_nroots: int | None = None,
) -> dict[str, object]:
    workflow_backend = workflow_backend or _resolve_qm_workflow_backend("orca")
    out_path = workflow_backend.output_path(spectrum_dir)
    if transitions is None:
        transitions = _extract_valid_tddft_transitions(
            out_path,
            workflow_backend,
            expected_nroots=expected_nroots,
        )

    result: dict[str, object] = {
        "directory": str(spectrum_dir),
        "output": str(out_path),
        "intensity_units": "broadened_oscillator_strength",
        "normalization_mode": "raw",
        "transition_count": int(len(transitions.wavenumbers_cm)),
    }
    if workflow_backend.key == "orca":
        result["orca_out"] = str(out_path)
    elif workflow_backend.key == "nwchem":
        result["nwchem_out"] = str(out_path)
    output_dir = Path(plot_output_dir) if plot_output_dir is not None else spectrum_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / csv_filename
    png_path = output_dir / png_filename
    from cats.visualise.uvvis import plot_uv_spectrum_nm

    uvvis_settings = UVVisPlotSettings()
    result["gaussian_fwhm_nm"] = float(uvvis_settings.stacked_uvvis_linewidth_nm)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    try:
        _, spectrum = plot_uv_spectrum_nm(
            transitions,
            linewidth_nm=float(uvvis_settings.stacked_uvvis_linewidth_nm),
            start_nm=uvvis_settings.stacked_uvvis_xmin,
            end_nm=uvvis_settings.stacked_uvvis_xmax,
            step=float(uvvis_settings.step),
            normalize=False,
            normalize_factor=1.0,
            ax=ax,
        )
        with csv_path.open("w", encoding="utf-8") as handle:
            handle.write("wavelength_nm,intensity\n")
            for x_value, y_value in zip(spectrum.x, spectrum.y):
                handle.write(f"{float(x_value):.8g},{float(y_value):.8g}\n")

        ax.set_xlabel("Wavelength / nm")
        ax.set_ylabel("Broadened oscillator strength")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)
    result["spectrum_csv"] = str(csv_path)
    result["spectrum_png"] = str(png_path)
    return result


def _extract_valid_tddft_transitions(
    out_path: Path,
    workflow_backend: QMWorkflowBackend,
    *,
    expected_nroots: int | None = None,
):
    if not workflow_backend.validate_output(out_path, warn=False):
        raise RequestedSpectrumError(f"TDDFT spectrum output invalid or missing: {out_path}")
    try:
        transitions = workflow_backend.extract_transitions(out_path)
        wavenumbers = transitions.wavenumbers_cm
        intensities = transitions.intensities
    except Exception as exc:
        raise RequestedSpectrumError(
            f"TDDFT spectrum output contains no extractable transitions: {out_path}"
        ) from exc
    if len(wavenumbers) == 0 or len(wavenumbers) != len(intensities):
        raise RequestedSpectrumError(
            f"TDDFT spectrum output contains incomplete transition data: {out_path}"
        )
    if expected_nroots is not None and len(wavenumbers) < int(expected_nroots):
        raise RequestedSpectrumError(
            f"TDDFT spectrum output contains {len(wavenumbers)} of "
            f"{int(expected_nroots)} requested transitions: {out_path}"
        )
    return transitions


def _ensure_final_tddft_spectrum_outputs(
    *,
    final_atoms,
    final_opt_dir: Path,
    cpus: int,
    mem: int,
    nroots: int,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    spin_multiplicity: int | None,
    system_charge: int = 0,
    workflow_backend: QMWorkflowBackend | None = None,
    gradient_root: int | None = None,
) -> dict[str, object]:
    workflow_backend = workflow_backend or _resolve_qm_workflow_backend("orca")
    spectrum_dir = DFT_ROOT_DIR / REACTION_HUB_FINAL_TDDFT_SPECTRUM_DIRNAME
    existing_out = workflow_backend.output_path(spectrum_dir)
    if gradient_root is None and workflow_backend.validate_output(existing_out, warn=False):
        try:
            transitions = _extract_valid_tddft_transitions(
                existing_out,
                workflow_backend,
                expected_nroots=int(nroots),
            )
        except RequestedSpectrumError:
            if _reaction_hub_replot_only():
                raise
            print(f"Info: Existing TDDFT output is incomplete; rerunning spectrum: {existing_out}")
        else:
            return _export_final_tddft_spectrum_plot_data(
                spectrum_dir,
                workflow_backend=workflow_backend,
                transitions=transitions,
                expected_nroots=int(nroots),
            )
    if _reaction_hub_replot_only():
        raise RequestedSpectrumError(
            f"Cannot replot requested TDDFT spectrum; no complete output exists: {existing_out}"
        )

    return _run_final_tddft_spectrum(
        final_atoms=final_atoms,
        final_opt_dir=final_opt_dir,
        cpus=cpus,
        mem=mem,
        nroots=nroots,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
        workflow_backend=workflow_backend,
        gradient_root=gradient_root,
    )


def _read_spectrum_csv(csv_path: Path) -> tuple[list[float], list[float]]:
    wavelengths: list[float] = []
    intensities: list[float] = []
    try:
        lines = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return wavelengths, intensities
    for raw_line in lines[1:]:
        parts = [part.strip() for part in raw_line.split(",")]
        if len(parts) < 2:
            continue
        try:
            wavelengths.append(float(parts[0]))
            intensities.append(float(parts[1]))
        except ValueError:
            continue
    return wavelengths, intensities


def _transition_wavelengths_and_strengths(transitions) -> tuple[list[float], list[float]]:
    try:
        raw_wavelengths = getattr(transitions, "wavelengths_nm", None)
        if raw_wavelengths is None:
            raw_wavenumbers = getattr(transitions, "wavenumbers_cm", [])
            wavelengths = [
                1.0e7 / float(value)
                for value in raw_wavenumbers
                if isfinite(float(value)) and float(value) > 0.0
            ]
        else:
            wavelengths = [float(value) for value in raw_wavelengths if isfinite(float(value))]
        strengths = [float(value) for value in getattr(transitions, "intensities", []) if isfinite(float(value))]
    except Exception:
        return [], []
    count = min(len(wavelengths), len(strengths))
    return wavelengths[:count], strengths[:count]


def _leftmost_signal_wavelength_nm(
    transitions,
    *,
    configured_xmin: float,
    configured_xmax: float,
    min_oscillator_strength: float = IROOT_MIN_OSCILLATOR_STRENGTH,
) -> float | None:
    wavelengths, strengths = _transition_wavelengths_and_strengths(transitions)
    candidates = [
        float(wavelength)
        for wavelength, strength in zip(wavelengths, strengths)
        if (
            configured_xmin <= float(wavelength) <= configured_xmax
            and float(strength) >= float(min_oscillator_strength)
        )
    ]
    return min(candidates) if candidates else None


def _resolve_difference_spectrum_plot_window(
    *,
    full_transitions,
    cluster_transitions,
    configured_xmin: float,
    configured_xmax: float,
    min_oscillator_strength: float = IROOT_MIN_OSCILLATOR_STRENGTH,
) -> tuple[float, float]:
    left_candidates = [
        value
        for value in (
            _leftmost_signal_wavelength_nm(
                full_transitions,
                configured_xmin=configured_xmin,
                configured_xmax=configured_xmax,
                min_oscillator_strength=min_oscillator_strength,
            ),
            _leftmost_signal_wavelength_nm(
                cluster_transitions,
                configured_xmin=configured_xmin,
                configured_xmax=configured_xmax,
                min_oscillator_strength=min_oscillator_strength,
            ),
        )
        if value is not None
    ]
    left = max([float(configured_xmin), *[float(value) for value in left_candidates]])
    right = float(configured_xmax)
    if right <= left:
        return float(configured_xmin), float(configured_xmax)
    return left, right


def _write_cluster_adsorbate_difference_spectrum(
    *,
    full_csv: Path,
    cluster_csv: Path,
    output_dir: Path,
    full_transitions=None,
    cluster_transitions=None,
) -> dict[str, object]:
    full_x, full_y = _read_spectrum_csv(full_csv)
    cluster_x, cluster_y = _read_spectrum_csv(cluster_csv)
    if not full_x or not full_y or not cluster_x or not cluster_y:
        return {"status": "skipped_missing_spectrum_data"}
    if len(full_x) != len(cluster_x):
        return {"status": "skipped_grid_mismatch", "full_points": len(full_x), "cluster_points": len(cluster_x)}

    diffs: list[float] = []
    for x_full, x_cluster, y_full, y_cluster in zip(full_x, cluster_x, full_y, cluster_y):
        if abs(float(x_full) - float(x_cluster)) > 1.0e-6:
            return {
                "status": "skipped_grid_mismatch",
                "full_wavelength_nm": float(x_full),
                "cluster_wavelength_nm": float(x_cluster),
            }
        diffs.append(float(y_full) - float(y_cluster))

    uvvis_settings = UVVisPlotSettings()
    configured_xmin = float(uvvis_settings.stacked_uvvis_xmin if uvvis_settings.stacked_uvvis_xmin is not None else full_x[0])
    configured_xmax = float(uvvis_settings.stacked_uvvis_xmax if uvvis_settings.stacked_uvvis_xmax is not None else full_x[-1])
    plot_left_nm, plot_right_nm = _resolve_difference_spectrum_plot_window(
        full_transitions=full_transitions,
        cluster_transitions=cluster_transitions,
        configured_xmin=configured_xmin,
        configured_xmax=configured_xmax,
        min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
    )
    fair_indices = [
        index
        for index, x_value in enumerate(full_x)
        if plot_left_nm <= float(x_value) <= plot_right_nm
    ]
    fair_full = [full_y[index] for index in fair_indices]
    fair_cluster = [cluster_y[index] for index in fair_indices]
    normalization_factor = 0.0
    if fair_full or fair_cluster:
        normalization_factor = max([0.0, *[float(value) for value in fair_full], *[float(value) for value in fair_cluster]])
    use_normalized = isfinite(float(normalization_factor)) and float(normalization_factor) > 0.0
    normalization_mode = "common_peak" if use_normalized else "raw_only"

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / REACTION_HUB_INTERACTION_DIFFERENCE_SPECTRUM_CSV
    png_path = output_dir / REACTION_HUB_INTERACTION_DIFFERENCE_SPECTRUM_PNG
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "wavelength_nm,delta_intensity,full_intensity,cluster_intensity,"
            "delta_intensity_raw,full_intensity_raw,cluster_intensity_raw,"
            "delta_intensity_common_peak,full_intensity_common_peak,cluster_intensity_common_peak,"
            "normalization_mode,normalization_factor,plot_left_nm,plot_right_nm\n"
        )
        for x_value, diff_value, full_value, cluster_value in zip(full_x, diffs, full_y, cluster_y):
            if use_normalized:
                diff_norm = float(diff_value) / float(normalization_factor)
                full_norm = float(full_value) / float(normalization_factor)
                cluster_norm = float(cluster_value) / float(normalization_factor)
            else:
                diff_norm = full_norm = cluster_norm = float("nan")
            handle.write(
                f"{float(x_value):.8g},{float(diff_value):.8g},"
                f"{float(full_value):.8g},{float(cluster_value):.8g},"
                f"{float(diff_value):.8g},{float(full_value):.8g},{float(cluster_value):.8g},"
                f"{diff_norm:.8g},{full_norm:.8g},{cluster_norm:.8g},"
                f"{normalization_mode},{float(normalization_factor):.8g},"
                f"{float(plot_left_nm):.8g},{float(plot_right_nm):.8g}\n"
            )

    try:
        import matplotlib.pyplot as plt

        plot_x = [full_x[index] for index in fair_indices]
        plot_y = [
            (diffs[index] / float(normalization_factor)) if use_normalized else diffs[index]
            for index in fair_indices
        ]
        if not plot_x:
            plot_x = full_x
            plot_y = [(value / float(normalization_factor)) if use_normalized else value for value in diffs]
        fig, ax = plt.subplots()
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.plot(plot_x, plot_y, color="#b91c1c", linewidth=1.4)
        ax.set_xlim(left=plot_left_nm, right=plot_right_nm)
        ax.set_xlabel("Wavelength / nm")
        if use_normalized:
            ax.set_ylabel("Common-peak normalized delta intensity")
        else:
            ax.set_ylabel("Delta intensity (cluster+adsorbate - cluster)")
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:
        return {
            "status": "partial",
            "spectrum_csv": str(csv_path),
            "plot_error": str(exc),
            "normalization_mode": normalization_mode,
            "normalization_factor": float(normalization_factor),
            "plot_left_nm": float(plot_left_nm),
            "plot_right_nm": float(plot_right_nm),
        }

    return {
        "status": "ok",
        "spectrum_csv": str(csv_path),
        "spectrum_png": str(png_path),
        "normalization_mode": normalization_mode,
        "normalization_factor": float(normalization_factor),
        "plot_left_nm": float(plot_left_nm),
        "plot_right_nm": float(plot_right_nm),
    }


def _ensure_cluster_tddft_difference_spectrum_outputs(
    *,
    final_atoms,
    final_opt_dir: Path,
    cpus: int,
    mem: int,
    nroots: int,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    spin_multiplicity: int | None,
    full_spectrum: dict[str, object],
    system_charge: int = 0,
    workflow_backend: QMWorkflowBackend | None = None,
) -> dict[str, object]:
    workflow_backend = workflow_backend or _resolve_qm_workflow_backend("orca")
    cluster_indices, adsorbate_indices = _split_cluster_adsorbate_indices(final_atoms)
    payload: dict[str, object] = {
        "status": "skipped",
        "cluster_atom_count": int(len(cluster_indices)),
        "adsorbate_atom_count": int(len(adsorbate_indices)),
    }
    if not cluster_indices:
        raise RequestedSpectrumError("Cannot calculate a difference spectrum without cluster atoms.")
    if not adsorbate_indices:
        raise RequestedSpectrumError("Cannot calculate a difference spectrum without adsorbate atoms.")

    spectrum_dir = DFT_ROOT_DIR / REACTION_HUB_FINAL_TDDFT_SPECTRUM_DIRNAME
    cluster_scf_dir = final_opt_dir / REACTION_HUB_FINAL_CLUSTER_SP_DIRNAME
    cluster_tddft_dir = spectrum_dir / "cluster_tddft"
    cluster_atoms = final_atoms[cluster_indices].copy()
    cluster_spin_multiplicity = _resolve_spin_multiplicity(cluster_atoms, None, system_charge)
    cluster_tddft_out = workflow_backend.output_path(cluster_tddft_dir)
    if workflow_backend.validate_output(cluster_tddft_out, warn=False):
        try:
            cluster_transitions = _extract_valid_tddft_transitions(
                cluster_tddft_out,
                workflow_backend,
                expected_nroots=int(nroots),
            )
        except RequestedSpectrumError:
            if _reaction_hub_replot_only():
                raise
            cluster_result = None
            print(
                "Info: Existing isolated-cluster TDDFT output is incomplete; "
                f"rerunning spectrum: {cluster_tddft_out}"
            )
        else:
            cluster_result = _export_final_tddft_spectrum_plot_data(
                cluster_tddft_dir,
                workflow_backend=workflow_backend,
                csv_filename=REACTION_HUB_CLUSTER_TDDFT_SPECTRUM_CSV,
                png_filename=REACTION_HUB_CLUSTER_TDDFT_SPECTRUM_PNG,
                plot_title="Isolated cluster TDDFT spectrum",
                plot_output_dir=spectrum_dir,
                transitions=cluster_transitions,
                expected_nroots=int(nroots),
            )
    elif _reaction_hub_replot_only():
        raise RequestedSpectrumError(
            "Cannot replot requested difference spectrum; no complete isolated-cluster "
            f"TDDFT output exists: {cluster_tddft_out}"
        )
    else:
        cluster_result = None

    cluster_scf_out = workflow_backend.output_path(cluster_scf_dir)
    if cluster_result is None and not workflow_backend.validate_output(cluster_scf_out, warn=False):
        payload["cluster_fragment_single_point"] = _run_final_cluster_single_point(
            final_atoms=final_atoms,
            final_opt_dir=final_opt_dir,
            cpus=cpus,
            mem=mem,
            basis_keyword=basis_keyword,
            xc_functional=xc_functional,
            solvent=solvent,
            spin_multiplicity=cluster_spin_multiplicity,
            system_charge=system_charge,
            workflow_backend=workflow_backend,
        )
    if cluster_result is None:
        cluster_result = _run_final_tddft_spectrum(
            final_atoms=cluster_atoms,
            final_opt_dir=cluster_scf_dir if cluster_scf_dir.exists() else final_opt_dir,
            cpus=cpus,
            mem=mem,
            nroots=nroots,
            basis_keyword=basis_keyword,
            xc_functional=xc_functional,
            solvent=solvent,
            spin_multiplicity=cluster_spin_multiplicity,
            system_charge=system_charge,
            workflow_backend=workflow_backend,
            spectrum_dirname=str(cluster_tddft_dir.relative_to(DFT_ROOT_DIR)),
            csv_filename=REACTION_HUB_CLUSTER_TDDFT_SPECTRUM_CSV,
            png_filename=REACTION_HUB_CLUSTER_TDDFT_SPECTRUM_PNG,
            plot_title="Isolated cluster TDDFT spectrum",
            plot_output_dir=spectrum_dir,
        )

    full_csv_text = str(full_spectrum.get("spectrum_csv") or "")
    cluster_csv_text = str(cluster_result.get("spectrum_csv") or "")
    full_transition_output = Path(str(full_spectrum.get("output") or workflow_backend.output_path(spectrum_dir)))
    cluster_transition_output = Path(str(cluster_result.get("output") or workflow_backend.output_path(cluster_tddft_dir)))
    try:
        full_transitions = workflow_backend.extract_transitions(full_transition_output)
    except Exception:
        full_transitions = None
    try:
        cluster_transitions = workflow_backend.extract_transitions(cluster_transition_output)
    except Exception:
        cluster_transitions = None
    difference_result = _write_cluster_adsorbate_difference_spectrum(
        full_csv=Path(full_csv_text),
        cluster_csv=Path(cluster_csv_text),
        output_dir=spectrum_dir,
        full_transitions=full_transitions,
        cluster_transitions=cluster_transitions,
    )
    if difference_result.get("status") != "ok":
        raise RequestedSpectrumError(
            "Failed to produce requested cluster-adsorbate difference spectrum: "
            f"{difference_result}"
        )
    payload.update(
        {
            "status": str(difference_result["status"]),
            "cluster": cluster_result,
            "difference": difference_result,
            "cluster_indices_1based": [int(idx) + 1 for idx in cluster_indices],
            "adsorbate_indices_1based": [int(idx) + 1 for idx in adsorbate_indices],
        }
    )
    return payload


def _serialize_vectors(vectors) -> list[list[float]]:
    serialized: list[list[float]] = []
    for row in vectors:
        serialized.append([float(row[0]), float(row[1]), float(row[2])])
    return serialized


def _write_reaction_hub_summary(
    *,
    max_iter: int,
    completed_iterations: int,
    use_tddft_optimisation: bool,
    bond_scan_enabled: bool,
    compute_cluster_fragment_single_point: bool,
    target_wavelength_nm: float,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    tddft_scan_nroots: int,
    final_cluster_tddft_difference_spectrum: bool,
    spin_multiplicity: int | None,
    system_charge: int = 0,
    active_learning_optimizer: str,
    nodes: int,
    cpus_per_node: int,
    workflow_backend: QMWorkflowBackend | None = None,
) -> None:
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    summary: dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_root": str(Path.cwd().resolve()),
        "mode": "tddft" if use_tddft_optimisation else "dft",
        "completed_iterations": int(completed_iterations),
        "settings": {
            "qm_backend": workflow_backend.key,
            "basis_keyword": basis_keyword,
            "xc_functional": xc_functional,
            "solvent": solvent,
            "spin_multiplicity": spin_multiplicity,
            "system_charge": int(system_charge),
            "active_learning_optimizer": str(active_learning_optimizer),
            "target_wavelength_nm": float(target_wavelength_nm),
            "tddft_scan_nroots": int(tddft_scan_nroots),
            "final_cluster_tddft_difference_spectrum": bool(final_cluster_tddft_difference_spectrum),
            "nodes": int(nodes),
            "cpus_per_node": int(cpus_per_node),
            "total_cpus": int(nodes * cpus_per_node),
        },
    }
    scan_profile = _scan_energy_profile_from_db(
        SHARED_TRAINING_DIR / SCAN_ENDPOINTS_DB_FILENAME
    )
    if scan_profile:
        summary["scan"] = scan_profile
    key_plots = _collect_key_plot_paths(use_tddft_optimisation=use_tddft_optimisation)
    if key_plots:
        summary["plots"] = key_plots

    final_opt_dir = _select_last_completed_opt_dir(
        max_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        workflow_backend=workflow_backend,
    )
    if final_opt_dir is None and completed_iterations > 0:
        final_opt_dir = _resolve_opt_dir_for_mode(
            completed_iterations,
            use_tddft_optimisation=use_tddft_optimisation,
        )

    if final_opt_dir is not None:
        final_out = workflow_backend.output_path(final_opt_dir)
        if workflow_backend.validate_output(final_out, warn=False):
            try:
                final_atoms = _read_orca_output_with_charges(
                    final_out,
                    label="reaction_hub summary export",
                    workflow_backend=workflow_backend,
                )
            except Exception as exc:
                print(
                    "Warning: Failed to read final "
                    f"{workflow_backend.label} output for summary export: {exc}"
                )
            else:
                final_data: dict[str, object] = {
                    "opt_dir": str(final_opt_dir),
                    "output": str(final_out),
                    "atom_count": len(final_atoms),
                    "formula": final_atoms.get_chemical_formula(),
                    "symbols": [str(symbol) for symbol in final_atoms.get_chemical_symbols()],
                    "positions_A": _serialize_vectors(final_atoms.get_positions()),
                }
                if workflow_backend.key == "orca":
                    final_data["orca_out"] = str(final_out)
                total_energy_eV = _resolve_total_energy_eV(
                    final_atoms,
                    orca_out=final_out,
                    workflow_backend=workflow_backend,
                )
                if total_energy_eV is not None:
                    final_data["energy_eV"] = float(total_energy_eV)
                    final_data["energy_hartree"] = float(total_energy_eV / Hartree)

                forces_eV_A = _resolve_forces_eV_A(
                    final_atoms,
                    opt_dir=final_opt_dir,
                    workflow_backend=workflow_backend,
                )
                if forces_eV_A is not None:
                    final_data["forces_eV_A"] = _serialize_vectors(forces_eV_A)
                    max_force = _max_force_norm(forces_eV_A)
                    if max_force is not None:
                        final_data["max_force_eV_A"] = float(max_force)

                charges = []
                try:
                    charges = [float(value) for value in final_atoms.get_initial_charges()]
                except Exception:
                    charges = []
                if charges:
                    final_data["charges"] = charges
                    final_data["charge_sum"] = float(sum(charges))
                    final_data["charge_max_abs"] = float(max(abs(value) for value in charges))
                    charge_method = final_atoms.info.get("charge_method")
                    if charge_method:
                        final_data["charge_method"] = str(charge_method)

                final_xyz_paths: list[str] = []
                for xyz_path in (
                    Path(REACTION_HUB_FINAL_XYZ_FILENAME),
                    SHARED_TRAINING_DIR / REACTION_HUB_FINAL_XYZ_FILENAME,
                ):
                    try:
                        xyz_path.parent.mkdir(parents=True, exist_ok=True)
                        write(str(xyz_path), final_atoms, format="xyz")
                        final_xyz_paths.append(str(xyz_path))
                    except Exception as exc:
                        print(f"Warning: Failed to write {xyz_path}: {exc}")
                if final_xyz_paths:
                    final_data["final_xyz_paths"] = final_xyz_paths

                if (
                    compute_cluster_fragment_single_point
                    and (not bond_scan_enabled)
                ):
                    # Run for any single-optimisation mode (DFT or TDDFT). This
                    # keeps replot/backfill resubmissions of older wrappers from
                    # skipping distortion-energy data.
                    try:
                        workflow_resources = workflow_backend.resolve_resources(
                            nodes=nodes,
                            cpus_per_node=cpus_per_node,
                        )
                        workflow_cpus = int(workflow_resources.cpus)
                        workflow_mem_per_cpu = int(workflow_resources.memory_per_cpu_mb)
                        final_data["cluster_fragment"] = _run_final_cluster_single_point(
                            final_atoms=final_atoms,
                            final_opt_dir=final_opt_dir,
                            cpus=workflow_cpus,
                            mem=workflow_mem_per_cpu,
                            basis_keyword=basis_keyword,
                            xc_functional=xc_functional,
                            solvent=solvent,
                            spin_multiplicity=spin_multiplicity,
                            system_charge=system_charge,
                            workflow_backend=workflow_backend,
                        )
                    except Exception as exc:
                        print(
                            "Warning: Failed to run final isolated-cluster single-point "
                            f"in {final_opt_dir}: {exc}"
                        )
                cluster_fragment = final_data.get("cluster_fragment", {})
                complex_energy = _parse_optional_float(final_data.get("energy_eV"))
                cluster_energy = _parse_optional_float(cluster_fragment.get("energy_eV"))
                adsorbate_energy = _parse_optional_float(
                    cluster_fragment.get("adsorbate", {}).get("energy_eV")
                )
                if (
                    complex_energy is not None
                    and cluster_energy is not None
                    and adsorbate_energy is not None
                ):
                    interaction_energy = float(complex_energy - cluster_energy - adsorbate_energy)
                    final_data["interaction_energy_eV"] = interaction_energy
                    final_data["interaction_energy_hartree"] = float(interaction_energy / Hartree)
                    final_data["adsorption_energy_eV"] = interaction_energy
                    final_data["adsorption_energy_hartree"] = float(interaction_energy / Hartree)

                summary["final"] = final_data

    summary_paths = [
        Path(REACTION_HUB_SUMMARY_FILENAME),
        SHARED_TRAINING_DIR / REACTION_HUB_SUMMARY_FILENAME,
    ]
    for summary_path in summary_paths:
        try:
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(
                json.dumps(summary, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"Warning: Failed to write workflow summary {summary_path}: {exc}")


def iter_dirs(i: int, *, use_tddft_optimisation: bool) -> IterationDirs:
    return IterationDirs(
        scan_dir=_tddft_scan_dir(i),
        opt_dir=_resolve_opt_dir_for_mode(i, use_tddft_optimisation=use_tddft_optimisation),
    )


def _has_tddft_state_tag(
    orca_out: Path,
    *,
    workflow_backend: QMWorkflowBackend | None = None,
) -> bool:
    try:
        return _workflow_backend_or_default(workflow_backend).has_tddft_state_tag(orca_out)
    except Exception:
        return False


def count_completed_iterations(
    max_iter: int,
    *,
    require_tddft_optimisation: bool = False,
    workflow_backend: QMWorkflowBackend | None = None,
) -> int:
    backend = _workflow_backend_or_default(workflow_backend)
    scan_log = load_scan_log(SHARED_TRAINING_DIR / IROOT_SCAN_LOG_FILENAME)
    completed = 0
    for i in range(1, max_iter + 1):
        opt_dir = _resolve_opt_dir_for_mode(
            i,
            use_tddft_optimisation=require_tddft_optimisation,
        )
        marker = opt_dir / ITERATION_COMPLETE_FILENAME
        out_path = backend.output_path(opt_dir)
        out_valid = backend.validate_output(out_path, warn=False)
        out_has_tddft = _has_tddft_state_tag(out_path, workflow_backend=backend) if out_valid else False
        mode_ok = (not require_tddft_optimisation) or out_has_tddft

        if marker.exists():
            if out_valid and mode_ok:
                completed = i
                continue
            break
        geom_dirs = _tddft_scan_geom_dirs(i)
        if str(i) in scan_log and out_valid and mode_ok:
            completed = i
            continue
        # A valid output can be the final true-force evaluation from an
        # unconverged active-learning cycle.  Only the marker or scan log
        # proves that the workflow accepted the optimisation endpoint.
        break
    return completed


def warn_if_iteration_gap(
    max_iter: int,
    completed: int,
    *,
    require_tddft_optimisation: bool = False,
    workflow_backend: QMWorkflowBackend | None = None,
) -> None:
    backend = _workflow_backend_or_default(workflow_backend)
    first_pending_iteration = completed + 1
    for i in range(completed + 1, max_iter + 1):
        if require_tddft_optimisation:
            opt_dir = _resolve_opt_dir_for_mode(i, use_tddft_optimisation=True)
            out_path = backend.output_path(opt_dir)
            if out_path.exists() and _has_tddft_state_tag(out_path, workflow_backend=backend):
                print(
                    f"Warning: Found {out_path} beyond first invalid iteration; "
                    "restart will ignore later iterations."
                )
                break
            if out_path.exists() and not backend.validate_output(out_path, warn=False) and i == first_pending_iteration:
                print(
                    f"Info: Found incomplete {out_path} at restart iteration {i}; "
                    "continuing from available restart seed/active-learning data."
                )
                break
            seed_dir = None
            for candidate in _dft_seed_opt_dirs(i):
                if backend.output_path(candidate).exists():
                    seed_dir = candidate
                    break
            if seed_dir is None:
                continue
            print(
                f"Info: Found DFT-only output in {seed_dir}; TDDFT mode will restart from "
                f"iteration {i} using existing geometries as starting points."
            )
            break
        opt_dir = _resolve_opt_dir_for_mode(i, use_tddft_optimisation=False)
        out_path = backend.output_path(opt_dir)
        if not out_path.exists():
            continue
        if not backend.validate_output(out_path, warn=False) and i == first_pending_iteration:
            print(
                f"Info: Found incomplete {out_path} at restart iteration {i}; "
                "continuing from available restart seed/active-learning data."
            )
            break
        print(
            f"Warning: Found {out_path} beyond first invalid iteration; "
            "restart will ignore later iterations."
        )
        break


def mark_iteration_complete(opt_dir: Path, iteration: int, *, mode: str) -> None:
    try:
        opt_dir.mkdir(parents=True, exist_ok=True)
        marker = opt_dir / ITERATION_COMPLETE_FILENAME
        marker.write_text(
            f"completed_iteration={iteration}\n"
            f"mode={mode}\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"Warning: Failed to write iteration marker in {opt_dir}: {exc}")


def _run_dft_only_endpoint_scans(
    *,
    atoms,
    effective_n_iter: int,
    bond_pair: tuple[int, int] | None,
    scan_log_path: Path,
    target_wavelength_nm: float,
    tddft_scan_nroots: int,
    fragments: dict[str, str],
    workflow_cpus: int,
    workflow_mem_per_cpu: int,
    run_initial_dft_if_missing: bool,
    initial_dft: Path,
    fmax: float,
    initial_fmax: float | None,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    spin_multiplicity: int | None,
    tddft_calc_factory,
    dft_force_calc_factory=None,
    workflow_backend: QMWorkflowBackend | None = None,
):
    workflow_backend = _workflow_backend_or_default(workflow_backend)
    # For DFT-only bond scans, run lightweight TDDFT scans at each endpoint
    # to recover excitation metadata. A plain single optimisation should not
    # trigger this post-processing path.
    for scan_iter in range(1, effective_n_iter + 1):
        opt_dir = _resolve_opt_dir_for_mode(
            scan_iter,
            use_tddft_optimisation=False,
        )
        opt_out = workflow_backend.output_path(opt_dir)
        if not workflow_backend.validate_output(opt_out, warn=False):
            print(f"Warning: Missing or invalid {opt_out}; skipping TDDFT scan.")
            continue
        try:
            scan_atoms = _read_orca_output_with_charges(
                opt_out,
                label=f"DFT-only iter {scan_iter} endpoint",
                workflow_backend=workflow_backend,
            )
        except Exception as exc:
            print(f"Warning: Failed to read {opt_out} for TDDFT scan: {exc}")
            continue

        scan_dir = _resolve_scan_dir(scan_iter)
        scan_choice = recover_scan_choice(
            scan_dir,
            target_wavelength_nm=target_wavelength_nm,
            fragments=fragments,
            score_weights=IROOT_SCORE_WEIGHTS,
            wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
            similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
            similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
            min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
        )
        if scan_choice is None:
            scan_dir = _resolve_scan_dir(scan_iter)
            fallback_dirs = [d for d in (opt_dir, initial_dft, Path("DFT")) if d is not None]
            gbw_source = workflow_backend.select_restart_source(opt_dir, fallback_dirs)
            if (
                workflow_backend.capabilities.requires_restart_data
                and workflow_backend.resolve_restart_path(gbw_source) is None
            ):
                if not run_initial_dft_if_missing:
                    raise FileNotFoundError(
                        f"Missing {workflow_backend.label} restart file for TDDFT scan; "
                        "provide restart data in DFT/iter_0 or DFT, or enable "
                        "run_initial_dft_if_missing."
                    )
                print(
                    f"No {workflow_backend.label} restart found; running initial DFT "
                    "active-learning engrad optimisation to generate DFT/iter_0 restart "
                    "data for TDDFT scans."
                )
                gbw_source = ensure_initial_dft(
                    scan_atoms,
                    cpus=workflow_cpus,
                    mem=workflow_mem_per_cpu,
                    directory=initial_dft,
                    fmax=initial_fmax if initial_fmax is not None else fmax,
                    basis_keyword=basis_keyword,
                    xc_functional=xc_functional,
                    solvent=solvent,
                    spin_multiplicity=spin_multiplicity,
                    workflow_backend=workflow_backend,
                    dft_force_calc_factory=dft_force_calc_factory,
                )
            workflow_backend.copy_restart(gbw_source, scan_dir)
            scan_calc = tddft_calc_factory(
                scan_atoms,
                cpus=workflow_cpus,
                mem=workflow_mem_per_cpu,
                task="energy",
                directory=scan_dir,
                nroots=tddft_scan_nroots,
            )
            scan_atoms.calc = scan_calc
            scan_atoms.get_potential_energy()
            df = workflow_backend.get_ct_info(
                scan_dir,
                fragments,
                min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
            )
            scan_choice, _ = workflow_backend.pick_iroot(
                df,
                target_wavelength_nm=target_wavelength_nm,
                score_weights=IROOT_SCORE_WEIGHTS,
                wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
                similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
                similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
                min_similarity=IROOT_MIN_SIMILARITY,
                min_relative_intensity=IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY,
                intensity_reference_window_nm=IROOT_INTENSITY_REFERENCE_WINDOW_NM,
                state_vectors=workflow_backend.extract_state_transition_vectors(
                    workflow_backend.output_path(scan_dir)
                ),
            )
            if scan_choice is None:
                raise RuntimeError(
                    f"No valid IROOT candidates found in TDDFT scan for iter {scan_iter}."
                )

        bond_length = (
            scan_atoms.get_distance(*bond_pair) if bond_pair is not None else None
        )
        record_scan_choice(
            scan_log_path,
            iteration=scan_iter,
            scan_dir=scan_dir,
            bond_pair=bond_pair,
            bond_length=bond_length,
            choice=scan_choice,
            target_wavelength_nm=target_wavelength_nm,
        )
        summarize_scan_log(scan_log_path)


def _finalize_workflow_outputs(
    *,
    atoms,
    effective_n_iter: int,
    completed_iterations: int | None = None,
    use_tddft_optimisation: bool,
    bond_scan: BondScanSettings,
    bond_pair: tuple[int, int] | None,
    scan_log_path: Path,
    target_wavelength_nm: float,
    workflow_cpus: int,
    workflow_mem_per_cpu: int,
    tddft_calc_factory,
    dft_calc_factory,
    fragments: dict[str, str],
    tddft_scan_nroots: int,
    final_tddft_spectrum: bool,
    final_tddft_nroots: int,
    final_cluster_tddft_difference_spectrum: bool,
    fmax: float,
    active_learning_optimizer: str,
    al_steps_per_iter: int,
    max_steps: int | None,
    plots_dir: Path,
    plotter: ScanSummaryPlotter | None,
    endpoints_db_path: Path,
    compute_cluster_fragment_single_point: bool,
    basis_keyword: str,
    xc_functional: str,
    solvent: str | None,
    spin_multiplicity: int | None,
    system_charge: int = 0,
    nodes: int,
    cpus_per_node: int,
    workflow_backend: QMWorkflowBackend | None = None,
):
    workflow_backend = workflow_backend or _resolve_qm_workflow_backend("orca")
    completed_for_outputs = (
        int(effective_n_iter)
        if completed_iterations is None
        else max(0, int(completed_iterations))
    )
    _run_scan_smoothing_postprocess(
        max_iter=completed_for_outputs,
        use_tddft_optimisation=use_tddft_optimisation,
        bond_scan=bond_scan,
        bond_pair=bond_pair,
        scan_log_path=scan_log_path,
        target_wavelength_nm=target_wavelength_nm,
        cpus=workflow_cpus,
        mem=workflow_mem_per_cpu,
        tddft_calc_factory=tddft_calc_factory,
        dft_calc_factory=dft_calc_factory,
        fragments=fragments,
        scan_nroots=tddft_scan_nroots,
        fmax=fmax,
        active_learning_optimizer=active_learning_optimizer,
        al_steps_per_call=al_steps_per_iter,
        max_steps=max_steps,
        workflow_backend=workflow_backend,
    )

    export_scan_endpoints_db(
        max_iter=completed_for_outputs,
        use_tddft_optimisation=use_tddft_optimisation,
        scan_log_path=scan_log_path,
        output_db_path=endpoints_db_path,
        scan_nroots=tddft_scan_nroots,
        workflow_backend=workflow_backend,
    )
    plot_max_iter = max(completed_for_outputs, _max_scan_log_iteration(scan_log_path))
    try:
        plot_scan_summaries(
            max_iter=plot_max_iter,
            bond_pair=bond_pair,
            scan_log_path=scan_log_path,
            output_dir=plots_dir,
            plotter=plotter,
            use_tddft_optimisation=use_tddft_optimisation,
            scan_endpoints_db_path=endpoints_db_path,
        )
    except Exception as exc:
        print(f"Warning: Failed to generate scan plots: {exc}")
    if (final_tddft_spectrum or final_cluster_tddft_difference_spectrum) and not bond_scan.enabled:
        final_opt_dir = _select_last_completed_opt_dir(
            completed_for_outputs,
            use_tddft_optimisation=use_tddft_optimisation,
            workflow_backend=workflow_backend,
        )
        if final_opt_dir is None and completed_for_outputs > 0:
            final_opt_dir = _resolve_opt_dir_for_mode(
                completed_for_outputs,
                use_tddft_optimisation=use_tddft_optimisation,
            )
        if final_opt_dir is None:
            raise RequestedSpectrumError(
                "Cannot calculate requested TDDFT spectrum; no completed optimisation directory found."
            )
        else:
            final_out = workflow_backend.output_path(final_opt_dir)
            try:
                final_atoms = _read_orca_output_with_charges(
                    final_out,
                    label="final TDDFT spectrum geometry",
                    workflow_backend=workflow_backend,
                )
            except Exception as exc:
                raise RequestedSpectrumError(
                    f"Cannot read geometry for requested TDDFT spectrum from {final_out}"
                ) from exc
            else:
                spectrum_result: dict[str, object] = {}
                try:
                    spectrum_result = _ensure_final_tddft_spectrum_outputs(
                        final_atoms=final_atoms,
                        final_opt_dir=final_opt_dir,
                        cpus=workflow_cpus,
                        mem=workflow_mem_per_cpu,
                        nroots=final_tddft_nroots,
                        basis_keyword=basis_keyword,
                        xc_functional=xc_functional,
                        solvent=solvent,
                        spin_multiplicity=spin_multiplicity,
                        system_charge=system_charge,
                        workflow_backend=workflow_backend,
                    )
                    print(
                        "Final TDDFT spectrum completed: "
                        f"{spectrum_result.get('output')} (NRoots={int(final_tddft_nroots)})."
                    )
                except Exception as exc:
                    raise RequestedSpectrumError(
                        f"Final TDDFT spectrum calculation failed: {exc}"
                    ) from exc
                if final_cluster_tddft_difference_spectrum and spectrum_result:
                    try:
                        cluster_spectrum_result = _ensure_cluster_tddft_difference_spectrum_outputs(
                            final_atoms=final_atoms,
                            final_opt_dir=final_opt_dir,
                            cpus=workflow_cpus,
                            mem=workflow_mem_per_cpu,
                            nroots=final_tddft_nroots,
                            basis_keyword=basis_keyword,
                            xc_functional=xc_functional,
                            solvent=solvent,
                            spin_multiplicity=spin_multiplicity,
                            full_spectrum=spectrum_result,
                            system_charge=system_charge,
                            workflow_backend=workflow_backend,
                        )
                        print(
                            "Cluster TDDFT difference spectrum completed: "
                            f"{cluster_spectrum_result.get('difference', {})}."
                        )
                    except Exception as exc:
                        raise RequestedSpectrumError(
                            f"Cluster TDDFT difference spectrum failed: {exc}"
                        ) from exc
    _write_reaction_hub_summary(
        max_iter=effective_n_iter,
        completed_iterations=completed_for_outputs,
        use_tddft_optimisation=use_tddft_optimisation,
        bond_scan_enabled=bool(bond_scan.enabled),
        compute_cluster_fragment_single_point=bool(
            compute_cluster_fragment_single_point
            or final_cluster_tddft_difference_spectrum
        ),
        target_wavelength_nm=target_wavelength_nm,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        tddft_scan_nroots=tddft_scan_nroots,
        final_cluster_tddft_difference_spectrum=bool(final_cluster_tddft_difference_spectrum),
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
        active_learning_optimizer=active_learning_optimizer,
        nodes=nodes,
        cpus_per_node=cpus_per_node,
        workflow_backend=workflow_backend,
    )

    final_dir = _select_last_completed_opt_dir(
        effective_n_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        workflow_backend=workflow_backend,
    )
    if final_dir is not None and workflow_backend.validate_output(
        workflow_backend.output_path(final_dir),
        warn=False,
    ):
        try:
            final_out = workflow_backend.output_path(final_dir)
            parsed_final_atoms = _read_orca_output_with_charges(
                final_out,
                label="workflow final iteration",
                workflow_backend=workflow_backend,
            )
            _copy_constraints_between_atoms(atoms, parsed_final_atoms)
            return parsed_final_atoms
        except Exception:
            pass
    return atoms


def run_workflow(
    atoms,
    n_iter: int = 10,
    bond_scan: BondScanSettings | None = None,
    al_steps_per_iter: int = 100,
    optimize_to_convergence: bool = True,
    max_al_steps_per_iter: int | None = None,
    fmax: float = 0.01,
    initial_fmax: float | None = None,
    run_initial_dft_if_missing: bool = True,
    use_tddft_optimisation: bool = True,
    ensure_initial_dft_converged: bool = True,
    use_scan_geometry: bool = False,
    plotter: ScanSummaryPlotter | None = None,
    basis_keyword: str = DEFAULT_BASIS_KEYWORD,
    xc_functional: str = DEFAULT_XC_FUNCTIONAL,
    solvent: str | None = DEFAULT_SOLVENT,
    tddft_scan_nroots: int = TDDFT_SCAN_ROOTS,
    initial_iroot: int | None = None,
    final_tddft_spectrum: bool = False,
    final_tddft_nroots: int = TDDFT_SCAN_ROOTS,
    final_cluster_tddft_difference_spectrum: bool = False,
    snapshot_single_point: bool = False,
    final_tddft_gradient_root: int | None = None,
    target_wavelength_nm: float = TARGET_WAVELENGTH_NM,
    spin_multiplicity: int | None = None,
    system_charge: int = 0,
    active_learning_optimizer: str = AL_LOCAL_OPTIMIZER_DEFAULT,
    compute_cluster_fragment_single_point: bool = True,
    nodes: int = nodes,
    cpus_per_node: int = cpus_per_node,
    qm_backend: str | QMWorkflowBackend = "orca",
    qm_backend_settings: dict[str, object] | None = None,
):
    system_charge = int(system_charge)
    workflow_backend = _resolve_qm_workflow_backend(qm_backend, qm_backend_settings)
    basis_keyword = workflow_backend.normalize_basis_keyword(basis_keyword)
    xc_functional = workflow_backend.normalize_xc_functional(xc_functional)
    solvent = workflow_backend.resolve_solvation_keyword(solvent)
    active_learning_optimizer = normalize_local_optimizer_name(active_learning_optimizer)
    workflow_resources = workflow_backend.resolve_resources(
        nodes=nodes,
        cpus_per_node=cpus_per_node,
    )
    workflow_cpus = int(workflow_resources.cpus)
    workflow_mem_per_cpu = int(workflow_resources.memory_per_cpu_mb)
    previous_choice: Optional[IRootCandidate] = None
    active_choice: Optional[IRootCandidate] = None
    dft_only = not use_tddft_optimisation
    bond_scan = bond_scan or BondScanSettings()
    _validate_qm_workflow_backend(
        workflow_backend,
        use_tddft_optimisation=use_tddft_optimisation,
        bond_scan=bond_scan,
        final_tddft_spectrum=bool(final_tddft_spectrum or final_cluster_tddft_difference_spectrum),
    )
    try:
        restart_extend_steps = int(getattr(bond_scan, "restart_extend_steps", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid restart_extend_steps value: {getattr(bond_scan, 'restart_extend_steps', None)!r}"
        ) from exc
    if restart_extend_steps < 0:
        raise ValueError("restart_extend_steps cannot be negative.")
    restart_extend_increment_raw = getattr(bond_scan, "restart_extend_increment_A", None)
    restart_extend_increment_A: float | None = None
    if restart_extend_increment_raw is not None and str(restart_extend_increment_raw).strip() != "":
        try:
            restart_extend_increment_A = float(restart_extend_increment_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid restart_extend_increment_A value: {restart_extend_increment_raw!r}"
            ) from exc
        if not isfinite(restart_extend_increment_A):
            raise ValueError("restart_extend_increment_A must be finite.")
    if restart_extend_steps > 0:
        effective_restart_increment = (
            float(restart_extend_increment_A)
            if restart_extend_increment_A is not None
            else float(bond_scan.increment)
        )
        if abs(effective_restart_increment) <= 1.0e-12:
            raise ValueError(
                "restart_extend_increment_A (or increment fallback) cannot be zero "
                "when restart_extend_steps is positive."
            )
    require_fragments = (not dft_only) or bond_scan.enabled
    effective_n_iter = n_iter if bond_scan.enabled else 1
    base_effective_n_iter = int(effective_n_iter)
    if not bond_scan.enabled and n_iter != 1:
        print(
            "Info: bond_scan is disabled; running a single optimisation iteration "
            "(n_iter=1)."
        )
    try:
        al_steps_per_iter = int(al_steps_per_iter)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid al_steps_per_iter value: {al_steps_per_iter!r}") from exc
    if al_steps_per_iter <= 0:
        raise ValueError("al_steps_per_iter must be positive.")
    # Defensive initialization: keep max_steps always bound even if this
    # branching logic is edited later.
    max_steps: int | None = None
    if max_al_steps_per_iter is None:
        max_steps = None if optimize_to_convergence else al_steps_per_iter
    else:
        try:
            max_al_steps_per_iter = int(max_al_steps_per_iter)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid max_al_steps_per_iter value: {max_al_steps_per_iter!r}") from exc
        if max_al_steps_per_iter <= 0:
            max_steps = None if optimize_to_convergence else al_steps_per_iter
        else:
            max_steps = max_al_steps_per_iter if optimize_to_convergence else al_steps_per_iter
    if optimize_to_convergence and max_steps is None:
        print(
            "Info: max_al_steps_per_iter is disabled; each optimisation iteration "
            "runs until convergence."
        )
    target_wavelength_nm = float(target_wavelength_nm)
    if target_wavelength_nm <= 0.0:
        raise ValueError("target_wavelength_nm must be positive.")
    try:
        tddft_scan_nroots = int(tddft_scan_nroots)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid tddft_scan_nroots value: {tddft_scan_nroots!r}") from exc
    if tddft_scan_nroots <= 0:
        raise ValueError("tddft_scan_nroots must be a positive integer.")
    if initial_iroot is not None:
        initial_iroot = int(initial_iroot)
        if not 1 <= initial_iroot <= tddft_scan_nroots:
            raise ValueError("initial_iroot must be between 1 and tddft_scan_nroots.")
    try:
        final_tddft_nroots = int(final_tddft_nroots)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid final_tddft_nroots value: {final_tddft_nroots!r}") from exc
    if final_tddft_nroots <= 0:
        raise ValueError("final_tddft_nroots must be a positive integer.")
    tddft_calc_factory = partial(
        workflow_backend.make_tddft_calculator,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
    )
    dft_calc_factory = partial(
        workflow_backend.make_dft_calculator,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
    )
    dft_engrad_calc_factory = partial(
        workflow_backend.make_dft_force_calculator,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
    )
    if snapshot_single_point:
        if final_tddft_gradient_root is not None and (
            workflow_backend.key != "orca" or type(final_tddft_gradient_root) is not int
            or not 1 <= final_tddft_gradient_root <= final_tddft_nroots
        ):
            raise ValueError("Invalid ORCA snapshot gradient root")
        if bond_scan.enabled or use_tddft_optimisation:
            raise ValueError(
                "snapshot_single_point requires bond_scan disabled and "
                "use_tddft_optimisation=False."
            )
        if not (final_tddft_spectrum or final_cluster_tddft_difference_spectrum):
            raise ValueError("snapshot_single_point requires a TDDFT spectrum request.")
        spectrum_result = _ensure_final_tddft_spectrum_outputs(
            final_atoms=atoms,
            final_opt_dir=DFT_ROOT_DIR,
            cpus=workflow_cpus,
            mem=workflow_mem_per_cpu,
            nroots=final_tddft_nroots,
            basis_keyword=basis_keyword,
            xc_functional=xc_functional,
            solvent=solvent,
            spin_multiplicity=spin_multiplicity,
            system_charge=system_charge,
            workflow_backend=workflow_backend,
            gradient_root=final_tddft_gradient_root,
        )
        if final_cluster_tddft_difference_spectrum:
            _ensure_cluster_tddft_difference_spectrum_outputs(
                final_atoms=atoms,
                final_opt_dir=DFT_ROOT_DIR,
                cpus=workflow_cpus,
                mem=workflow_mem_per_cpu,
                nroots=final_tddft_nroots,
                basis_keyword=basis_keyword,
                xc_functional=xc_functional,
                solvent=solvent,
                spin_multiplicity=spin_multiplicity,
                full_spectrum=spectrum_result,
                system_charge=system_charge,
                workflow_backend=workflow_backend,
            )
        _write_reaction_hub_summary(
            max_iter=0,
            completed_iterations=0,
            use_tddft_optimisation=False,
            bond_scan_enabled=False,
            compute_cluster_fragment_single_point=bool(
                compute_cluster_fragment_single_point
                or final_cluster_tddft_difference_spectrum
            ),
            target_wavelength_nm=target_wavelength_nm,
            basis_keyword=basis_keyword,
            xc_functional=xc_functional,
            solvent=solvent,
            tddft_scan_nroots=tddft_scan_nroots,
            final_cluster_tddft_difference_spectrum=bool(
                final_cluster_tddft_difference_spectrum
            ),
            spin_multiplicity=spin_multiplicity,
            system_charge=system_charge,
            active_learning_optimizer=active_learning_optimizer,
            nodes=nodes,
            cpus_per_node=cpus_per_node,
            workflow_backend=workflow_backend,
        )
        print("TDDFT single-point spectrum completed on the unchanged input snapshot.")
        return atoms
    bond_pair: tuple[int, int] | None = None
    shared_training_dir = SHARED_TRAINING_DIR
    shared_training_dir.mkdir(parents=True, exist_ok=True)
    scan_log_path = shared_training_dir / IROOT_SCAN_LOG_FILENAME
    endpoints_db_path = shared_training_dir / SCAN_ENDPOINTS_DB_FILENAME
    plots_dir = _resolve_scan_plots_dir(use_tddft_optimisation=use_tddft_optimisation)
    initial_dft = INITIAL_DFT_DIR
    workflow_root_path = shared_training_dir / WORKFLOW_ROOT_FILENAME
    current_root = Path.cwd().resolve()
    if workflow_root_path.exists():
        try:
            recorded_root = workflow_root_path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as exc:
            raise RuntimeError(f"Unable to read workflow root marker: {workflow_root_path}") from exc
        if recorded_root and recorded_root != str(current_root):
            print(
                "Info: Workflow root mismatch detected; continuing restart. "
                f"Recorded root: {recorded_root}, current root: {current_root}."
            )
    else:
        try:
            workflow_root_path.write_text(str(current_root), encoding="utf-8")
        except OSError as exc:
            print(f"Warning: Failed to write workflow root marker {workflow_root_path}: {exc}")
    repaired_scan_paths = _repair_scan_log_stale_paths(
        scan_log_path=scan_log_path,
        max_iter=effective_n_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        workflow_backend=workflow_backend,
    )
    if repaired_scan_paths:
        print(
            "Info: Repaired stale scan_dir entries in "
            f"{scan_log_path}: {repaired_scan_paths} row(s) updated."
        )
    completed_iters = count_completed_iterations(
        effective_n_iter,
        require_tddft_optimisation=use_tddft_optimisation,
        workflow_backend=workflow_backend,
    )
    restart_extension_applied = False
    if bond_scan.enabled and restart_extend_steps > 0:
        if completed_iters > 0 and completed_iters >= base_effective_n_iter:
            base_iter = max(int(effective_n_iter), int(completed_iters))
            extended_iter = base_iter + int(restart_extend_steps)
            if extended_iter > int(effective_n_iter):
                print(
                    "Info: Applying restart bond-scan extension: "
                    f"completed={completed_iters}, base_n_iter={effective_n_iter}, "
                    f"extra_steps={restart_extend_steps} -> n_iter={extended_iter}."
                )
                effective_n_iter = int(extended_iter)
                completed_iters = count_completed_iterations(
                    effective_n_iter,
                    require_tddft_optimisation=use_tddft_optimisation,
                    workflow_backend=workflow_backend,
                )
                restart_extension_applied = True
        elif completed_iters > 0:
            print(
                "Info: restart_extend_steps was requested, but the base scan is not yet "
                f"complete (completed={completed_iters}, requested={base_effective_n_iter}); "
                "continuing without extension."
            )
        else:
            print(
                "Info: restart_extend_steps was requested, but no completed scan "
                "iterations were found; ignoring extension for fresh run."
            )
    warn_if_iteration_gap(
        effective_n_iter,
        completed_iters,
        require_tddft_optimisation=use_tddft_optimisation,
        workflow_backend=workflow_backend,
    )
    # Single DFT optimisation special case:
    # keep robust precheck-style convergence behavior, but avoid duplicated
    # iter_0 + iter_1 runs by converging directly in iter_1.
    if (
        completed_iters == 0
        and ensure_initial_dft_converged
        and dft_only
        and not bond_scan.enabled
    ):
        single_opt_dir = _dft_opt_dir(1)
        try:
            atoms, _ = ensure_initial_dft_converged_atoms(
                atoms,
                cpus=workflow_cpus,
                mem=workflow_mem_per_cpu,
                directory=single_opt_dir,
                run_if_missing=run_initial_dft_if_missing,
                al_steps_per_call=al_steps_per_iter,
                max_steps=max_steps,
                fmax=fmax,
                strict_fmax=initial_fmax,
                basis_keyword=basis_keyword,
                xc_functional=xc_functional,
                solvent=solvent,
                spin_multiplicity=spin_multiplicity,
                active_learning_optimizer=active_learning_optimizer,
                workflow_backend=workflow_backend,
                dft_force_calc_factory=dft_engrad_calc_factory,
            )
            mark_iteration_complete(single_opt_dir, 1, mode="dft")
            completed_iters = 1
            print(
                "Info: Single DFT optimisation converged in DFT/iter_1 via precheck path; "
                "skipping duplicated iter_0 + iter_1 sequence."
            )
            if (final_tddft_spectrum or final_cluster_tddft_difference_spectrum) and not bond_scan.enabled:
                spectrum_result: dict[str, object] = {}
                try:
                    spectrum_result = _ensure_final_tddft_spectrum_outputs(
                        final_atoms=atoms,
                        final_opt_dir=single_opt_dir,
                        cpus=workflow_cpus,
                        mem=workflow_mem_per_cpu,
                        nroots=final_tddft_nroots,
                        basis_keyword=basis_keyword,
                        xc_functional=xc_functional,
                        solvent=solvent,
                        spin_multiplicity=spin_multiplicity,
                        system_charge=system_charge,
                        workflow_backend=workflow_backend,
                    )
                    print(
                        "Final TDDFT spectrum completed after converged DFT precheck: "
                        f"{spectrum_result.get('output')} (NRoots={int(final_tddft_nroots)})."
                    )
                except Exception as exc:
                    raise RequestedSpectrumError(
                        "Final TDDFT spectrum calculation failed after converged DFT "
                        f"precheck: {exc}"
                    ) from exc
                if final_cluster_tddft_difference_spectrum and spectrum_result:
                    try:
                        cluster_spectrum_result = _ensure_cluster_tddft_difference_spectrum_outputs(
                            final_atoms=atoms,
                            final_opt_dir=single_opt_dir,
                            cpus=workflow_cpus,
                            mem=workflow_mem_per_cpu,
                            nroots=final_tddft_nroots,
                            basis_keyword=basis_keyword,
                            xc_functional=xc_functional,
                            solvent=solvent,
                            spin_multiplicity=spin_multiplicity,
                            full_spectrum=spectrum_result,
                            system_charge=system_charge,
                            workflow_backend=workflow_backend,
                        )
                        print(
                            "Cluster TDDFT difference spectrum completed after converged DFT precheck: "
                            f"{cluster_spectrum_result.get('difference', {})}."
                        )
                    except Exception as exc:
                        raise RequestedSpectrumError(
                            "Cluster TDDFT difference spectrum failed after converged "
                            f"DFT precheck: {exc}"
                        ) from exc
        except RequestedSpectrumError:
            raise
        except Exception as exc:
            print(
                "Warning: Single DFT precheck path in DFT/iter_1 did not converge cleanly "
                f"({exc}); falling back to iterative retry mode."
            )
    run_initial_dft_precheck = (
        completed_iters == 0
        and ensure_initial_dft_converged
        and ((not dft_only) or bond_scan.enabled)
    )
    if run_initial_dft_precheck:
        atoms, _ = ensure_initial_dft_converged_atoms(
            atoms,
            cpus=workflow_cpus,
            mem=workflow_mem_per_cpu,
            directory=initial_dft,
            run_if_missing=run_initial_dft_if_missing,
            al_steps_per_call=al_steps_per_iter,
            max_steps=max_steps,
            fmax=fmax,
            strict_fmax=initial_fmax,
            basis_keyword=basis_keyword,
            xc_functional=xc_functional,
            solvent=solvent,
            spin_multiplicity=spin_multiplicity,
            active_learning_optimizer=active_learning_optimizer,
            workflow_backend=workflow_backend,
            dft_force_calc_factory=dft_engrad_calc_factory,
        )
    fragments: dict[str, str] = {}
    if require_fragments:
        fragments = build_fragments(atoms)
    if bond_scan.enabled:
        bond_pair = resolve_bond_indices(atoms, bond_scan)
    if bond_scan.enabled and bond_pair is not None:
        backfilled_rows = _backfill_missing_primary_scan_log_rows(
            scan_log_path=scan_log_path,
            max_iter=effective_n_iter,
            use_tddft_optimisation=use_tddft_optimisation,
            bond_pair=bond_pair,
            fragments=fragments,
            target_wavelength_nm=target_wavelength_nm,
            workflow_backend=workflow_backend,
        )
        if backfilled_rows:
            print(
                "Info: Reconstructed missing primary scan rows in "
                f"{scan_log_path}: {backfilled_rows} row(s) added."
            )
    if completed_iters >= effective_n_iter:
        completed_for_postprocess = int(completed_iters)
        max_opt_iter_on_disk = _max_opt_iteration_on_disk(
            use_tddft_optimisation=use_tddft_optimisation
        )
        if max_opt_iter_on_disk > completed_iters:
            discovered_completed = count_completed_iterations(
                max_opt_iter_on_disk,
                require_tddft_optimisation=use_tddft_optimisation,
                workflow_backend=workflow_backend,
            )
            if discovered_completed > completed_for_postprocess:
                completed_for_postprocess = int(discovered_completed)
                print(
                    "Info: Resubmission detected additional completed iterations on disk "
                    f"({completed_for_postprocess}); running smoothing/postprocess over "
                    "the full completed scan."
                )
        print(f"All {effective_n_iter} requested iterations already completed.")
        _run_scan_smoothing_postprocess(
            max_iter=completed_for_postprocess,
            use_tddft_optimisation=use_tddft_optimisation,
            bond_scan=bond_scan,
            bond_pair=bond_pair,
            scan_log_path=scan_log_path,
            target_wavelength_nm=target_wavelength_nm,
            cpus=workflow_cpus,
            mem=workflow_mem_per_cpu,
            tddft_calc_factory=tddft_calc_factory,
            dft_calc_factory=dft_calc_factory,
            fragments=fragments,
            scan_nroots=tddft_scan_nroots,
            fmax=fmax,
            active_learning_optimizer=active_learning_optimizer,
            al_steps_per_call=al_steps_per_iter,
            max_steps=max_steps,
            workflow_backend=workflow_backend,
        )
        export_scan_endpoints_db(
            max_iter=completed_for_postprocess,
            use_tddft_optimisation=use_tddft_optimisation,
            scan_log_path=scan_log_path,
            output_db_path=endpoints_db_path,
            scan_nroots=tddft_scan_nroots,
            workflow_backend=workflow_backend,
        )
        plot_max_iter = max(completed_for_postprocess, _max_scan_log_iteration(scan_log_path))
        try:
            plot_scan_summaries(
                max_iter=plot_max_iter,
                bond_pair=bond_pair,
                scan_log_path=scan_log_path,
                output_dir=plots_dir,
                plotter=plotter,
                use_tddft_optimisation=use_tddft_optimisation,
                scan_endpoints_db_path=endpoints_db_path,
            )
        except Exception as exc:
            print(f"Warning: Failed to generate scan plots: {exc}")
        final_atoms = atoms
        last_dir = (
            _resolve_opt_dir_for_mode(
                completed_for_postprocess,
                use_tddft_optimisation=use_tddft_optimisation,
            )
            if completed_for_postprocess > 0
            else None
        )
        if last_dir:
            last_out = workflow_backend.output_path(last_dir)
        else:
            last_out = None
        if last_out is not None and last_out.exists():
            parsed_final_atoms = _read_orca_output_with_charges(
                last_out,
                label="last completed iteration",
                workflow_backend=workflow_backend,
            )
            _copy_constraints_between_atoms(atoms, parsed_final_atoms)
            final_atoms = parsed_final_atoms
        if (
            (final_tddft_spectrum or final_cluster_tddft_difference_spectrum)
            and not bond_scan.enabled
            and last_dir is not None
        ):
            spectrum_result: dict[str, object] = {}
            try:
                spectrum_result = _ensure_final_tddft_spectrum_outputs(
                    final_atoms=final_atoms,
                    final_opt_dir=last_dir,
                    cpus=workflow_cpus,
                    mem=workflow_mem_per_cpu,
                    nroots=final_tddft_nroots,
                    basis_keyword=basis_keyword,
                    xc_functional=xc_functional,
                    solvent=solvent,
                    spin_multiplicity=spin_multiplicity,
                    system_charge=system_charge,
                    workflow_backend=workflow_backend,
                )
                print(
                    "Final TDDFT spectrum refreshed: "
                    f"{spectrum_result.get('output')} (NRoots={int(final_tddft_nroots)})."
                )
            except Exception as exc:
                raise RequestedSpectrumError(
                    f"Final TDDFT spectrum refresh failed: {exc}"
                ) from exc
            if final_cluster_tddft_difference_spectrum and spectrum_result:
                try:
                    cluster_spectrum_result = _ensure_cluster_tddft_difference_spectrum_outputs(
                        final_atoms=final_atoms,
                        final_opt_dir=last_dir,
                        cpus=workflow_cpus,
                        mem=workflow_mem_per_cpu,
                        nroots=final_tddft_nroots,
                        basis_keyword=basis_keyword,
                        xc_functional=xc_functional,
                        solvent=solvent,
                        spin_multiplicity=spin_multiplicity,
                        full_spectrum=spectrum_result,
                        system_charge=system_charge,
                        workflow_backend=workflow_backend,
                    )
                    print(
                        "Cluster TDDFT difference spectrum refreshed: "
                        f"{cluster_spectrum_result.get('difference', {})}."
                    )
                except Exception as exc:
                    raise RequestedSpectrumError(
                        f"Cluster TDDFT difference spectrum refresh failed: {exc}"
                    ) from exc
        _write_reaction_hub_summary(
            max_iter=completed_for_postprocess,
            completed_iterations=completed_for_postprocess,
            use_tddft_optimisation=use_tddft_optimisation,
            bond_scan_enabled=bool(bond_scan.enabled),
            compute_cluster_fragment_single_point=bool(
                compute_cluster_fragment_single_point
                or final_cluster_tddft_difference_spectrum
            ),
            target_wavelength_nm=target_wavelength_nm,
            basis_keyword=basis_keyword,
            xc_functional=xc_functional,
            solvent=solvent,
            tddft_scan_nroots=tddft_scan_nroots,
            final_cluster_tddft_difference_spectrum=bool(final_cluster_tddft_difference_spectrum),
            spin_multiplicity=spin_multiplicity,
            system_charge=system_charge,
            active_learning_optimizer=active_learning_optimizer,
            nodes=nodes,
            cpus_per_node=cpus_per_node,
            workflow_backend=workflow_backend,
        )
        return final_atoms

    initial_scan = _resolve_scan_dir(0)
    previous_tracking_dir: Path | None = initial_scan

    if completed_iters == 0:
        if not dft_only:
            def _recover_choice(scan_dir: Path):
                return recover_scan_choice(
                    scan_dir,
                    target_wavelength_nm=target_wavelength_nm,
                    fragments=fragments,
                    score_weights=IROOT_SCORE_WEIGHTS,
                    wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
                    similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
                    similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
                    min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
                )

            recovered_choice = None
            existing_scan_log = load_scan_log(scan_log_path)
            if existing_scan_log:
                def _iter_sort_key(value: str) -> int:
                    try:
                        return int(value)
                    except ValueError:
                        return 10**9

                for iteration in sorted(existing_scan_log, key=_iter_sort_key):
                    iter_num = _iter_sort_key(iteration)
                    if iter_num <= 0 or iter_num >= 10**9:
                        continue
                    row = existing_scan_log[iteration]
                    # Ignore postprocess rows when recovering the startup scan
                    # state; only primary scan iterations should seed a fresh run.
                    if str(row.get("point_kind") or "").strip():
                        continue
                    scan_dir_text = row.get("scan_dir") or str(_tddft_scan_dir(iter_num))
                    scan_dir = Path(scan_dir_text)
                    recovered_choice = _recover_choice(scan_dir)
                    if recovered_choice is not None:
                        print(
                            f"Recovered IROOT from existing scan {scan_dir}; "
                            "skipping iteration-0 TDDFT scan."
                        )
                        break

            if recovered_choice is None:
                recovered_choice = _recover_choice(initial_scan)
            if recovered_choice is None:
                gbw_source = workflow_backend.select_restart_source(initial_dft, [Path("DFT")])
                if (
                    workflow_backend.capabilities.requires_restart_data
                    and workflow_backend.resolve_restart_path(gbw_source) is None
                ):
                    if not run_initial_dft_if_missing:
                        raise FileNotFoundError(
                            f"Missing {workflow_backend.label} restart file; provide "
                            "restart data in DFT/iter_0 or DFT, or enable "
                            "run_initial_dft_if_missing."
                        )
                    print(
                        f"No {workflow_backend.label} restart found; running initial DFT "
                        "active-learning engrad optimisation to generate DFT/iter_0 restart data."
                    )
                    gbw_source = ensure_initial_dft(
                        atoms,
                        cpus=workflow_cpus,
                        mem=workflow_mem_per_cpu,
                        directory=initial_dft,
                        fmax=initial_fmax if initial_fmax is not None else fmax,
                        basis_keyword=basis_keyword,
                        xc_functional=xc_functional,
                        solvent=solvent,
                        spin_multiplicity=spin_multiplicity,
                        workflow_backend=workflow_backend,
                        dft_force_calc_factory=dft_engrad_calc_factory,
                    )
                workflow_backend.copy_restart(gbw_source, initial_scan)
                scan_calc = tddft_calc_factory(
                    atoms,
                    cpus=workflow_cpus,
                    mem=workflow_mem_per_cpu,
                    task="energy",
                    directory=initial_scan,
                    nroots=tddft_scan_nroots,
                )
                atoms.calc = scan_calc
                atoms.get_potential_energy()

                df = workflow_backend.get_ct_info(
                    initial_scan,
                    fragments,
                    min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
                )
                active_choice, scored_choices = workflow_backend.pick_iroot(
                    df,
                    target_wavelength_nm=target_wavelength_nm,
                    score_weights=IROOT_SCORE_WEIGHTS,
                    wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
                    similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
                    similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
                    min_similarity=IROOT_MIN_SIMILARITY,
                    min_relative_intensity=IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY,
                    intensity_reference_window_nm=IROOT_INTENSITY_REFERENCE_WINDOW_NM,
                    state_vectors=workflow_backend.extract_state_transition_vectors(
                        workflow_backend.output_path(initial_scan)
                    ),
                )
                if initial_iroot is not None:
                    active_choice = next(
                        (choice for choice in scored_choices if choice.state == initial_iroot),
                        None,
                    )
                    if active_choice is None:
                        raise RuntimeError(
                            f"Requested initial IROOT={initial_iroot} is absent from TDDFT scan."
                        )
                if active_choice is None:
                    raise RuntimeError("No valid IROOT candidates found in initial scan.")
            else:
                active_choice = recovered_choice
                print("Recovered initial TDDFT scan results; skipping re-run.")
            previous_choice = active_choice
            if use_scan_geometry:
                workflow_backend.update_atoms_from_output(
                    atoms,
                    workflow_backend.output_path(initial_scan),
                    label="initial scan",
                )
            iter0_opt_dir = _tddft_opt_dir(0)
            iter0_out = workflow_backend.output_path(iter0_opt_dir)
            iter0_tddft_ready = (
                workflow_backend.validate_output(iter0_out, warn=False)
                and _has_tddft_state_tag(iter0_out, workflow_backend=workflow_backend)
            )
            if not iter0_tddft_ready:
                if workflow_backend.validate_output(iter0_out, warn=False):
                    print(
                        f"Iteration 0: Existing output in {iter0_out} is DFT-only; "
                        "rerunning TDDFT optimisation bootstrap."
                    )
                else:
                    print(
                        f"Iteration 0: Running TDDFT optimisation bootstrap in {iter0_opt_dir}."
                    )
                reapply_bond_constraints(atoms, bond_scan, bond_pair)
                iter0_al_dir = iter0_opt_dir / "active_learning"
                iter0_al_dir.mkdir(parents=True, exist_ok=True)
                iter0_training_path = iter0_al_dir / "evaluated.traj"
                iter0_seed_dir = iter0_al_dir / "seed"
                iter0_gbw_source = workflow_backend.select_restart_source(
                    initial_scan,
                    [initial_dft, DFT_ROOT_DIR],
                )
                if (
                    workflow_backend.capabilities.requires_restart_data
                    and workflow_backend.resolve_restart_path(iter0_gbw_source) is None
                ):
                    raise FileNotFoundError(
                        f"Missing {workflow_backend.label} restart file for initial TDDFT "
                        "optimisation; expected restart data in TDDFT/scan_iter_0, "
                        "DFT/iter_0, or DFT."
                    )
                atoms, iter0_gbw_source = ensure_training_seed(
                    atoms,
                    training_path=iter0_training_path,
                    seed_dir=iter0_seed_dir,
                    gbw_source_dir=iter0_gbw_source,
                    gbw_fallback_dirs=[initial_scan, initial_dft, DFT_ROOT_DIR],
                    iroot=active_choice.state,
                    cpus=workflow_cpus,
                    mem=workflow_mem_per_cpu,
                    calc_factory=tddft_calc_factory,
                )
                _ensure_restart_gbw(
                    iter0_opt_dir,
                    fallback_dirs=[
                        iter0_seed_dir,
                        iter0_gbw_source,
                        initial_scan,
                        initial_dft,
                        DFT_ROOT_DIR,
                    ],
                    workflow_backend=workflow_backend,
                )
                iter0_context = ProxyContext(
                    target_wavelength_nm=target_wavelength_nm,
                    fragments=fragments,
                    initial_score_weights=IROOT_SCORE_WEIGHTS,
                    followup_score_weights=IROOT_FOLLOWUP_SCORE_WEIGHTS,
                    min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
                    wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
                    similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
                    similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
                    min_similarity=IROOT_MIN_SIMILARITY,
                    min_relative_intensity=IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY,
                    intensity_reference_window_nm=IROOT_INTENSITY_REFERENCE_WINDOW_NM,
                    min_transition_overlap=IROOT_MIN_TRANSITION_OVERLAP,
                    min_ct_similarity=IROOT_MIN_CT_SIMILARITY,
                    nroots=tddft_scan_nroots,
                )
                iter0_calc = ScanThenEngradProxy(
                    cpus=workflow_cpus,
                    mem=workflow_mem_per_cpu,
                    scan_base=_tddft_scan_geom_base(0),
                    opt_dir=iter0_opt_dir,
                    gbw_source_dir=iter0_gbw_source,
                    previous_choice=active_choice,
                    previous_state=active_choice.state,
                    iteration=0,
                    geometry_step_start=1,
                    bond_pair=bond_pair,
                    geometry_log_path=iter0_al_dir / IROOT_GEOMETRY_LOG_FILENAME,
                    context=iter0_context,
                    calc_factory=tddft_calc_factory,
                    engrad_calc_factory=tddft_calc_factory,
                    use_scan_geometry=use_scan_geometry,
                    reference_dir=initial_scan,
                )
                iter0_steps_per_call = 1 if optimize_to_convergence else max(1, int(al_steps_per_iter))
                iter0_max_steps = max_steps if optimize_to_convergence else max(1, int(al_steps_per_iter))
                iter0_soft_cycles = 0
                while True:
                    try:
                        atoms = active_learning_step(
                            atoms,
                            iter0_calc,
                            fmax=fmax,
                            shared_training_path=iter0_training_path,
                            state_dir=iter0_al_dir,
                            use_restart=is_valid_trainingset(iter0_training_path),
                            steps_per_call=iter0_steps_per_call,
                            optimize_to_convergence=optimize_to_convergence,
                            max_steps=iter0_max_steps,
                            max_unc=AL_PREDICTION_MAX_UNC,
                            min_data=AL_MIN_DATA,
                            ml_steps=AL_ML_STEPS,
                            dtrust=AL_DTRUST,
                            raise_on_not_converged=True,
                            local_optimizer=active_learning_optimizer,
                        )
                        break
                    except RuntimeError as exc:
                        if not _is_active_learning_nonconvergence_error(exc):
                            raise
                        iter0_soft_cycles += 1
                        if iter0_soft_cycles > MAX_ITERATION_SOFT_CONTINUATIONS:
                            raise RuntimeError(
                                "Iteration 0 exceeded soft continuation limit "
                                f"({MAX_ITERATION_SOFT_CONTINUATIONS}) during TDDFT bootstrap."
                            ) from exc
                        print(
                            f"Iteration 0: {exc} Continuing active learning "
                            f"({_soft_continuation_status_text(cycle=iter0_soft_cycles)})."
                        )
                        continue
                if iter0_calc.last_scan_choice is not None:
                    active_choice = iter0_calc.last_scan_choice
                    previous_choice = active_choice
                    previous_tracking_dir = iter0_calc.last_scan_dir
                if not workflow_backend.validate_output(iter0_out, warn=False):
                    raise RuntimeError(
                        f"Initial TDDFT optimisation bootstrap did not produce valid output: {iter0_out}"
                    )
            else:
                print(
                    f"Iteration 0: Reusing existing TDDFT-optimised geometry from {iter0_out}."
                )
            if workflow_backend.validate_output(iter0_out, warn=False):
                try:
                    iter0_atoms = _read_orca_output_with_charges(
                        iter0_out,
                        label="iteration 0 TDDFT bootstrap geometry",
                        workflow_backend=workflow_backend,
                    )
                    _copy_constraints_between_atoms(atoms, iter0_atoms)
                    atoms = iter0_atoms
                    reapply_bond_constraints(atoms, bond_scan, bond_pair)
                except Exception as exc:
                    print(f"Warning: Failed to read {iter0_out}: {exc}")
                if previous_tracking_dir == initial_scan:
                    previous_tracking_dir = iter0_opt_dir
        start_iter = 1
    else:
        last_dir = _resolve_opt_dir_for_mode(
            completed_iters,
            use_tddft_optimisation=use_tddft_optimisation,
        )
        recovered_out = workflow_backend.output_path(last_dir)
        previous_tracking_dir = last_dir
        if not recovered_out.exists():
            raise RuntimeError(f"Expected {recovered_out} for restart, but file is missing.")
        recovered_atoms = _read_orca_output_with_charges(
            recovered_out,
            label="restart geometry",
            workflow_backend=workflow_backend,
        )
        _copy_constraints_between_atoms(atoms, recovered_atoms)
        atoms = recovered_atoms
        if require_fragments:
            fragments = build_fragments(atoms)
        reapply_bond_constraints(atoms, bond_scan, bond_pair)
        if not dft_only:
            try:
                df = workflow_backend.get_ct_info(
                    last_dir,
                    fragments,
                    min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
                )
            except (FileNotFoundError, RuntimeError) as exc:
                raise RuntimeError(f"Restart failed while reading CT data: {exc}") from exc
            prev_state = workflow_backend.get_followed_state(recovered_out)
            if prev_state is None:
                print(f"Warning: No IROOT found in {recovered_out}; continuity may be reduced.")
            recover_score_weights = (
                IROOT_FOLLOWUP_SCORE_WEIGHTS
                if prev_state is not None
                else IROOT_SCORE_WEIGHTS
            )
            enforce_overlap = prev_state is not None
            active_choice, _ = workflow_backend.pick_iroot(
                df,
                target_wavelength_nm=target_wavelength_nm,
                previous_choice=None,
                previous_state=prev_state,
                score_weights=recover_score_weights,
                wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
                similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
                similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
                min_similarity=IROOT_MIN_SIMILARITY if enforce_overlap else 0.0,
                min_relative_intensity=IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY,
                intensity_reference_window_nm=IROOT_INTENSITY_REFERENCE_WINDOW_NM,
                min_transition_overlap=IROOT_MIN_TRANSITION_OVERLAP if enforce_overlap else 0.0,
                min_ct_similarity=IROOT_MIN_CT_SIMILARITY if enforce_overlap else 0.0,
                require_transition_overlap=enforce_overlap,
                state_vectors=workflow_backend.extract_state_transition_vectors(recovered_out),
            )
            if active_choice is None:
                raise RuntimeError("Unable to recover IROOT from last completed iteration.")
            previous_choice = active_choice
        start_iter = completed_iters + 1
        print(f"Resuming workflow from iteration {start_iter} (completed {completed_iters} iterations).")

    if bond_scan.enabled:
        if bond_pair is None:
            bond_pair = resolve_bond_indices(atoms, bond_scan)
        apply_bond_constraints(atoms, bond_scan.fixed_symbols, bond_pair)
        if completed_iters == 0 and active_choice is not None and not dft_only:
            record_scan_choice(
                scan_log_path,
                iteration=0,
                scan_dir=initial_scan,
                bond_pair=bond_pair,
                bond_length=atoms.get_distance(*bond_pair),
                choice=active_choice,
                target_wavelength_nm=target_wavelength_nm,
            )

    restart_extension_active = (
        bool(bond_scan.enabled)
        and bool(bond_pair is not None)
        and bool(restart_extension_applied)
    )
    bond_increment = bond_scan.increment
    if bond_scan.enabled and bond_pair is not None:
        remaining_iters = effective_n_iter - completed_iters
        if restart_extension_active:
            extension_increment = (
                float(restart_extend_increment_A)
                if restart_extend_increment_A is not None
                else float(bond_scan.increment)
            )
            current_distance = float(atoms.get_distance(*bond_pair))
            if bond_scan.target_distance is not None:
                bond_scan.target_distance = (
                    current_distance + extension_increment * float(restart_extend_steps)
                )
                print(
                    "Bond scan restart extension: "
                    f"target reset to {bond_scan.target_distance:.4f} Angstrom from "
                    f"restart distance {current_distance:.4f} Angstrom "
                    f"using increment={extension_increment:.4f} Angstrom "
                    f"for {int(restart_extend_steps)} additional step(s)."
                )
            else:
                print(
                    "Bond scan restart extension: "
                    f"continuing from {current_distance:.4f} Angstrom with "
                    f"increment={extension_increment:.4f} Angstrom for "
                    f"{int(restart_extend_steps)} additional step(s)."
                )
            bond_increment = extension_increment
        else:
            bond_increment = resolve_bond_increment(
                atoms,
                bond_pair,
                bond_scan,
                remaining_iterations=remaining_iters,
            )
            if bond_scan.target_distance is not None:
                current_distance = atoms.get_distance(*bond_pair)
                steps = max(remaining_iters - 1, 1)
                print(
                    "Bond scan: target="
                    f"{bond_scan.target_distance:.4f} Angstrom, "
                    f"start={current_distance:.4f} Angstrom, "
                    f"increment={bond_increment:.4f} Angstrom "
                    f"over {steps} steps "
                    f"(fraction={bond_scan.distance_change_fraction:.3f})."
                )

    i = start_iter
    iteration_attempts: dict[int, int] = {}
    iteration_soft_continuations: dict[int, int] = {}
    iteration_last_error: dict[int, str] = {}
    while i <= effective_n_iter:
        attempt = iteration_attempts.get(i, 0) + 1
        iteration_attempts[i] = attempt
        if attempt > MAX_ITERATION_RETRY_ATTEMPTS:
            last_error = str(iteration_last_error.get(i, "")).strip()
            if last_error:
                raise RuntimeError(
                    f"Iteration {i} exceeded workflow retry limit "
                    f"({MAX_ITERATION_RETRY_ATTEMPTS}) without completing. "
                    f"Last retry error: {last_error}"
                )
            raise RuntimeError(
                f"Iteration {i} exceeded workflow retry limit "
                f"({MAX_ITERATION_RETRY_ATTEMPTS}) without completing."
            )
        if attempt > 1:
            print(
                f"Iteration {i}: workflow retry attempt {attempt} "
                "(same iteration)."
            )
        dirs = iter_dirs(i, use_tddft_optimisation=use_tddft_optimisation)
        prev_opt_dir = _resolve_opt_dir_for_mode(
            i - 1,
            use_tddft_optimisation=use_tddft_optimisation,
        )
        gbw_source_dir = prev_opt_dir

        if active_choice is None and not dft_only:
            raise RuntimeError("Missing active IROOT selection before entering iteration.")

        iteration_max_steps: int | None = max_steps if optimize_to_convergence else al_steps_per_iter
        if iteration_max_steps is not None and iteration_max_steps <= 0:
            raise ValueError("Number of local AL steps must be positive.")
        al_dir = dirs.opt_dir / "active_learning"
        al_dir.mkdir(parents=True, exist_ok=True)
        opt_out = workflow_backend.output_path(dirs.opt_dir)
        forced_reingestion = False
        training_path = al_dir / "evaluated.traj"
        if (
            not dft_only
            and use_tddft_optimisation
            and workflow_backend.validate_output(opt_out, warn=False)
            and not _has_tddft_state_tag(opt_out, workflow_backend=workflow_backend)
        ):
            training_path = al_dir / "evaluated_tddft.traj"
            print(
                f"Iteration {i}: Existing output is DFT-only; using {training_path.name} "
                "for TDDFT active-learning restart data."
            )
        geometry_log_path: Path | None = None
        scan_choice = None
        last_scan_dir: Path | None = None
        scan_point_kind = ""
        scan_point_label = ""
        if not dft_only:
            geometry_log_path = al_dir / IROOT_GEOMETRY_LOG_FILENAME
            last_geom_step, resume_choice, resume_scan_dir = load_last_geometry_state(
                geometry_log_path,
                i,
            )
            geometry_start = 1
            if last_geom_step > 0:
                geometry_start = last_geom_step + 1
                if resume_choice is not None:
                    previous_choice = resume_choice
                    active_choice = resume_choice
                if resume_scan_dir is not None:
                    last_scan_dir = resume_scan_dir
                opt_out = workflow_backend.output_path(dirs.opt_dir)
                if workflow_backend.validate_output(opt_out, warn=False):
                    try:
                        resumed_atoms = _read_orca_output_with_charges(
                            opt_out,
                            label=f"iter {i} resumed TDDFT geometry",
                            workflow_backend=workflow_backend,
                        )
                        _copy_constraints_between_atoms(atoms, resumed_atoms)
                        atoms = resumed_atoms
                        reapply_bond_constraints(atoms, bond_scan, bond_pair)
                    except Exception as exc:
                        print(f"Warning: Failed to resume atoms from {opt_out}: {exc}")
            elif use_scan_geometry:
                # Seed TDDFT optimisation from any available geometry in this iteration.
                # Prefer existing DFT/TDDFT geometry; otherwise create one with
                # DFT active-learning engrad optimisation.
                dft_seed_out: Path | None = None
                for candidate in _dft_seed_opt_dirs(i):
                    candidate_out = workflow_backend.output_path(candidate)
                    if workflow_backend.validate_output(candidate_out, warn=False):
                        dft_seed_out = candidate_out
                        break
                tddft_seed_out = workflow_backend.output_path(dirs.opt_dir)
                if dft_seed_out is not None:
                    if workflow_backend.update_atoms_from_output(
                        atoms,
                        dft_seed_out,
                        label=f"iter {i} DFT seed geometry",
                    ):
                        reapply_bond_constraints(atoms, bond_scan, bond_pair)
                        print(
                            f"Iteration {i}: Using DFT geometry from {dft_seed_out} as TDDFT start."
                        )
                elif workflow_backend.validate_output(tddft_seed_out, warn=False):
                    if workflow_backend.update_atoms_from_output(
                        atoms,
                        tddft_seed_out,
                        label=f"iter {i} TDDFT seed geometry",
                    ):
                        reapply_bond_constraints(atoms, bond_scan, bond_pair)
                        print(
                            f"Iteration {i}: Using TDDFT geometry from {tddft_seed_out} as TDDFT start."
                        )
                else:
                    dft_pre_dir = _dft_opt_dir(i)
                    dft_pre_out = workflow_backend.output_path(dft_pre_dir)
                    print(
                        f"Iteration {i}: Missing pre-optimised geometry in {dft_pre_out}; "
                        "running DFT active-learning engrad optimisation before TDDFT."
                    )
                    dft_pre_atoms, _ = ensure_initial_dft_converged_atoms(
                        atoms,
                        cpus=workflow_cpus,
                        mem=workflow_mem_per_cpu,
                        directory=dft_pre_dir,
                        run_if_missing=True,
                        fmax=fmax,
                        strict_fmax=initial_fmax,
                        al_steps_per_call=al_steps_per_iter,
                        max_steps=max_steps,
                        basis_keyword=basis_keyword,
                        xc_functional=xc_functional,
                        solvent=solvent,
                        spin_multiplicity=spin_multiplicity,
                        active_learning_optimizer=active_learning_optimizer,
                        workflow_backend=workflow_backend,
                        dft_force_calc_factory=dft_engrad_calc_factory,
                    )
                    atoms = dft_pre_atoms
                    reapply_bond_constraints(atoms, bond_scan, bond_pair)
            engrad_factory = (
                tddft_calc_factory if use_tddft_optimisation else dft_engrad_calc_factory
            )
            atoms, gbw_source_dir = ensure_training_seed(
                atoms,
                training_path=training_path,
                seed_dir=al_dir / "seed",
                gbw_source_dir=gbw_source_dir,
                gbw_fallback_dirs=[
                    d for d in (last_scan_dir, initial_scan, initial_dft, Path("DFT")) if d is not None
                ],
                iroot=active_choice.state,
                cpus=workflow_cpus,
                mem=workflow_mem_per_cpu,
                calc_factory=engrad_factory,
            )
            gbw_fallback_dirs = [al_dir / "seed"] + [
                d
                for d in (
                    last_scan_dir,
                    dirs.opt_dir,
                    prev_opt_dir,
                    initial_scan,
                    initial_dft,
                    DFT_ROOT_DIR,
                )
                if d is not None
            ]
            gbw_source_dir = workflow_backend.select_restart_source(
                gbw_source_dir,
                gbw_fallback_dirs,
            )
            if (
                workflow_backend.capabilities.requires_restart_data
                and workflow_backend.resolve_restart_path(gbw_source_dir) is None
            ):
                raise FileNotFoundError(
                    f"Missing {workflow_backend.label} restart file for TDDFT optimisation; "
                    f"expected restart data in one of: {gbw_source_dir}, "
                    f"{al_dir / 'seed'}, {last_scan_dir}, {dirs.opt_dir}, {prev_opt_dir}, "
                    f"{initial_scan}, {initial_dft}, or {DFT_ROOT_DIR}."
                )
            step_previous_choice = previous_choice
            scan_choice = active_choice if last_geom_step > 0 and last_scan_dir is not None else None
            if optimize_to_convergence:
                if iteration_max_steps is None:
                    remaining_steps: int | None = None
                else:
                    remaining_steps = iteration_max_steps - (geometry_start - 1)
            else:
                remaining_steps = int(iteration_max_steps) - (geometry_start - 1)
            if remaining_steps is not None and remaining_steps <= 0:
                opt_out = workflow_backend.output_path(dirs.opt_dir)
                if not (last_scan_dir and workflow_backend.validate_output(opt_out, warn=False)):
                    geometry_start = int(iteration_max_steps) if iteration_max_steps is not None else 1
                    remaining_steps = 1
                else:
                    remaining_steps = 0

            if remaining_steps is None or remaining_steps > 0:
                prev_state, prev_state_source = _resolve_last_iroot_from_candidates(
                    [
                        gbw_source_dir,
                        dirs.opt_dir,
                        last_scan_dir,
                        prev_opt_dir,
                        initial_scan,
                        initial_dft,
                        DFT_ROOT_DIR,
                    ],
                    workflow_backend=workflow_backend,
                )
                if prev_state is None and step_previous_choice is not None:
                    prev_state = int(step_previous_choice.state)
                    print(
                        f"Warning: Falling back to previous scan IROOT={prev_state} "
                        "for continuity."
                    )
                elif prev_state is None:
                    if prev_state_source is not None:
                        print(
                            f"Warning: No IROOT found in {prev_state_source}; "
                            "continuity may be reduced."
                        )
                    else:
                        print(
                            f"Warning: No previous {workflow_backend.label} output with IROOT was found; "
                            "continuity may be reduced."
                        )
                context = ProxyContext(
                    target_wavelength_nm=target_wavelength_nm,
                    fragments=fragments,
                    initial_score_weights=IROOT_SCORE_WEIGHTS,
                    followup_score_weights=IROOT_FOLLOWUP_SCORE_WEIGHTS,
                    min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
                    wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
                    similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
                    similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
                    min_similarity=IROOT_MIN_SIMILARITY,
                    min_relative_intensity=IROOT_FOLLOWUP_MIN_RELATIVE_INTENSITY,
                    intensity_reference_window_nm=IROOT_INTENSITY_REFERENCE_WINDOW_NM,
                    min_transition_overlap=IROOT_MIN_TRANSITION_OVERLAP,
                    min_ct_similarity=IROOT_MIN_CT_SIMILARITY,
                    nroots=tddft_scan_nroots,
                )
                proxy_calc = ScanThenEngradProxy(
                    cpus=workflow_cpus,
                    mem=workflow_mem_per_cpu,
                    scan_base=_tddft_scan_geom_base(i),
                    opt_dir=dirs.opt_dir,
                    gbw_source_dir=gbw_source_dir,
                    previous_choice=step_previous_choice,
                    previous_state=prev_state,
                    iteration=i,
                    geometry_step_start=geometry_start,
                    bond_pair=bond_pair,
                    geometry_log_path=geometry_log_path,
                    context=context,
                    calc_factory=tddft_calc_factory,
                    engrad_calc_factory=engrad_factory,
                    use_scan_geometry=use_scan_geometry,
                    reference_dir=last_scan_dir or previous_tracking_dir or gbw_source_dir,
                )
                try:
                    tddft_steps_per_call = (
                        1
                        if optimize_to_convergence
                        else int(remaining_steps)
                    )
                    atoms = active_learning_step(
                        atoms,
                        proxy_calc,
                        fmax=fmax,
                        shared_training_path=training_path,
                        state_dir=al_dir,
                        use_restart=is_valid_trainingset(training_path),
                        steps_per_call=tddft_steps_per_call,
                        optimize_to_convergence=optimize_to_convergence,
                        max_steps=remaining_steps if optimize_to_convergence else int(remaining_steps),
                        max_unc=AL_PREDICTION_MAX_UNC,
                        min_data=AL_MIN_DATA,
                        ml_steps=AL_ML_STEPS,
                        dtrust=AL_DTRUST,
                        raise_on_not_converged=True,
                        local_optimizer=active_learning_optimizer,
                    )
                except RuntimeError as exc:
                    if _is_active_learning_nonconvergence_error(exc):
                        soft_cycle = _record_soft_iteration_continuation(
                            iteration_attempts=iteration_attempts,
                            iteration_soft_continuations=iteration_soft_continuations,
                            iteration=i,
                        )
                        print(
                            f"Iteration {i}: {exc} Continuing active learning "
                            f"({_soft_continuation_status_text(cycle=soft_cycle)})."
                        )
                    else:
                        iteration_last_error[i] = str(exc)
                        print(f"Iteration {i}: {exc}; repeating iteration.")
                    continue

                if proxy_calc.last_scan_choice is None:
                    atoms.calc = proxy_calc
                    atoms.get_forces()
                    append_training_set(training_path, atoms)

                scan_choice = proxy_calc.last_scan_choice
                last_scan_dir = proxy_calc.last_scan_dir
                previous_tracking_dir = last_scan_dir
                scan_point_kind = str(proxy_calc.last_scan_point_kind or "").strip()
                scan_point_label = str(proxy_calc.last_scan_point_label or "").strip()
                gbw_source_dir = proxy_calc.gbw_source_dir
                if scan_choice is not None:
                    active_choice = scan_choice
                    step_previous_choice = scan_choice

            if scan_choice is None or last_scan_dir is None:
                raise RuntimeError(f"No valid IROOT candidates found in scan iter {i}.")
        else:
            opt_out = workflow_backend.output_path(dirs.opt_dir)
            if workflow_backend.validate_output(opt_out, warn=False):
                try:
                    resumed_atoms = _read_orca_output_with_charges(
                        opt_out,
                        label=f"iter {i} resumed DFT geometry",
                        workflow_backend=workflow_backend,
                    )
                    _copy_constraints_between_atoms(atoms, resumed_atoms)
                    atoms = resumed_atoms
                    reapply_bond_constraints(atoms, bond_scan, bond_pair)
                except Exception as exc:
                    print(f"Warning: Failed to resume atoms from {opt_out}: {exc}")
            try:
                atoms = ensure_dft_training_seed(
                    atoms,
                    training_path=training_path,
                    seed_dir=al_dir / "seed",
                    cpus=workflow_cpus,
                    mem=workflow_mem_per_cpu,
                    calc_factory=dft_engrad_calc_factory,
                    workflow_backend=workflow_backend,
                )
            except Exception as exc:
                completed_before_failure = count_completed_iterations(
                    effective_n_iter,
                    require_tddft_optimisation=use_tddft_optimisation,
                    workflow_backend=workflow_backend,
                )
                if bond_scan.enabled and completed_before_failure > 0:
                    print(
                        f"Warning: Iteration {i} failed while preparing the DFT training seed "
                        f"({exc}). Finalizing plots and artifacts for the "
                        f"{completed_before_failure} completed iteration(s) instead of aborting "
                        "before post-processing."
                    )
                    if dft_only:
                        _run_dft_only_endpoint_scans(
                            atoms=atoms,
                            effective_n_iter=completed_before_failure,
                            bond_pair=bond_pair,
                            scan_log_path=scan_log_path,
                            target_wavelength_nm=target_wavelength_nm,
                            tddft_scan_nroots=tddft_scan_nroots,
                            fragments=fragments,
                            workflow_cpus=workflow_cpus,
                            workflow_mem_per_cpu=workflow_mem_per_cpu,
                            run_initial_dft_if_missing=run_initial_dft_if_missing,
                            initial_dft=initial_dft,
                            fmax=fmax,
                            initial_fmax=initial_fmax,
                            basis_keyword=basis_keyword,
                            xc_functional=xc_functional,
                            solvent=solvent,
                            spin_multiplicity=spin_multiplicity,
                            tddft_calc_factory=tddft_calc_factory,
                            dft_force_calc_factory=dft_engrad_calc_factory,
                            workflow_backend=workflow_backend,
                        )
                    return _finalize_workflow_outputs(
                        atoms=atoms,
                        effective_n_iter=effective_n_iter,
                        completed_iterations=completed_before_failure,
                        use_tddft_optimisation=use_tddft_optimisation,
                        bond_scan=bond_scan,
                        bond_pair=bond_pair,
                        scan_log_path=scan_log_path,
                        target_wavelength_nm=target_wavelength_nm,
                        workflow_cpus=workflow_cpus,
                        workflow_mem_per_cpu=workflow_mem_per_cpu,
                        tddft_calc_factory=tddft_calc_factory,
                        dft_calc_factory=dft_calc_factory,
                        fragments=fragments,
                        tddft_scan_nroots=tddft_scan_nroots,
                        final_tddft_spectrum=bool(final_tddft_spectrum),
                        final_tddft_nroots=int(final_tddft_nroots),
                        final_cluster_tddft_difference_spectrum=bool(
                            final_cluster_tddft_difference_spectrum
                        ),
                        fmax=fmax,
                        active_learning_optimizer=active_learning_optimizer,
                        al_steps_per_iter=al_steps_per_iter,
                        max_steps=max_steps,
                        plots_dir=plots_dir,
                        plotter=plotter,
                        endpoints_db_path=endpoints_db_path,
                        compute_cluster_fragment_single_point=compute_cluster_fragment_single_point,
                        basis_keyword=basis_keyword,
                        xc_functional=xc_functional,
                        solvent=solvent,
                        spin_multiplicity=spin_multiplicity,
                        nodes=nodes,
                        cpus_per_node=cpus_per_node,
                        workflow_backend=workflow_backend,
                    )
                raise
            seed_dir = al_dir / "seed"
            seed_forces_eV_A = _resolve_forces_eV_A(
                atoms,
                opt_dir=seed_dir,
                workflow_backend=workflow_backend,
            )
            seed_max_force = (
                _max_force_norm(seed_forces_eV_A)
                if seed_forces_eV_A is not None
                else None
            )
            skip_dft_active_learning = bool(
                optimize_to_convergence
                and seed_max_force is not None
                and seed_max_force <= float(fmax)
                and _stage_seed_single_point_as_converged_output(
                    seed_dir=seed_dir,
                    output_dir=dirs.opt_dir,
                    label=f"iteration {i} DFT",
                    workflow_backend=workflow_backend,
                )
            )
            if skip_dft_active_learning:
                print(
                    f"Iteration {i}: DFT seed single-point is already converged "
                    f"(max force={seed_max_force:.6f} <= {float(fmax):.6f} eV/A); "
                    "skipping active learning."
                )
            _ensure_restart_gbw(
                dirs.opt_dir,
                fallback_dirs=[seed_dir, prev_opt_dir, initial_dft, DFT_ROOT_DIR],
                workflow_backend=workflow_backend,
            )
            dft_calc = None
            if not skip_dft_active_learning:
                dft_calc = dft_engrad_calc_factory(
                    atoms,
                    cpus=workflow_cpus,
                    mem=workflow_mem_per_cpu,
                    directory=dirs.opt_dir,
                )
                dft_steps_per_call = (
                    al_steps_per_iter
                    if optimize_to_convergence
                    else int(iteration_max_steps)
                )
                if optimize_to_convergence and not bond_scan.enabled:
                    # Single DFT-only optimisation: check convergence after each AL step
                    # to avoid unnecessary extra evaluations once fmax is reached.
                    dft_steps_per_call = 1
                try:
                    atoms = active_learning_step(
                        atoms,
                        dft_calc,
                        fmax=fmax,
                        shared_training_path=training_path,
                        state_dir=al_dir,
                        use_restart=is_valid_trainingset(training_path),
                        steps_per_call=dft_steps_per_call,
                        optimize_to_convergence=optimize_to_convergence,
                        max_steps=iteration_max_steps,
                        max_unc=AL_PREDICTION_MAX_UNC,
                        min_data=AL_MIN_DATA,
                        ml_steps=AL_ML_STEPS,
                        dtrust=AL_DTRUST,
                        raise_on_not_converged=True,
                        local_optimizer=active_learning_optimizer,
                    )
                except RuntimeError as exc:
                    if _is_active_learning_nonconvergence_error(exc):
                        soft_cycle = _record_soft_iteration_continuation(
                            iteration_attempts=iteration_attempts,
                            iteration_soft_continuations=iteration_soft_continuations,
                            iteration=i,
                        )
                        print(
                            f"Iteration {i}: {exc} Continuing active learning "
                            f"({_soft_continuation_status_text(cycle=soft_cycle)})."
                        )
                    else:
                        iteration_last_error[i] = str(exc)
                        print(f"Iteration {i}: {exc}; repeating iteration.")
                    continue
            opt_out = workflow_backend.output_path(dirs.opt_dir)
            if not opt_out.exists():
                print(
                    f"Iteration {i}: {workflow_backend.label} output missing after active learning; forcing engrad run."
                )
                print(
                    f"Iteration {i}: Forced reingestion requested; recomputing outputs in {dirs.opt_dir} "
                    f"so {opt_out} can be re-read."
                )
                forced_reingestion = True
                try:
                    if dft_calc is None:
                        dft_calc = dft_engrad_calc_factory(
                            atoms,
                            cpus=workflow_cpus,
                            mem=workflow_mem_per_cpu,
                            directory=dirs.opt_dir,
                        )
                    atoms.calc = dft_calc
                    atoms.get_forces()
                except Exception as exc:
                    iteration_last_error[i] = f"Forced engrad failed: {exc}"
                    print(f"Iteration {i}: Forced engrad failed: {exc}; repeating iteration.")
                    continue
                if not opt_out.exists():
                    raise RuntimeError(
                        f"Iteration {i}: {workflow_backend.label} output still missing after forced engrad ({opt_out})."
                    )
                print(
                    f"Iteration {i}: Forced reingestion succeeded; regenerated {opt_out}."
                )

        opt_out = workflow_backend.output_path(dirs.opt_dir)
        if dft_only:
            if not opt_out.exists():
                iteration_last_error[i] = (
                    f"{workflow_backend.label} output missing after iteration {i}: {opt_out}"
                )
                print(f"Iteration {i}: {workflow_backend.label} output missing, repeating iteration.")
                continue
        else:
            if not workflow_backend.validate_output(opt_out, warn=False):
                iteration_last_error[i] = (
                    f"{workflow_backend.label} output invalid or unconverged after iteration {i}: {opt_out}"
                )
                print(
                    f"Iteration {i}: {workflow_backend.label} output invalid or unconverged, "
                    "repeating iteration."
                )
                continue
        # Keep pre-parse atoms as fallback for force-based convergence checks.
        # Reading from orca.out can drop force metadata for the final geometry.
        pre_parse_atoms = atoms
        if forced_reingestion:
            print(
                f"Iteration {i}: Reingesting regenerated {workflow_backend.label} output from {opt_out}."
            )
        try:
            parsed_opt_atoms = _read_orca_output_with_charges(
                opt_out,
                label=f"iteration {i} optimised geometry",
                workflow_backend=workflow_backend,
            )
            _copy_constraints_between_atoms(atoms, parsed_opt_atoms)
            if len(parsed_opt_atoms) == len(atoms):
                atoms = parsed_opt_atoms
                if forced_reingestion:
                    print(
                        f"Iteration {i}: Forced reingestion loaded {len(parsed_opt_atoms)} atoms from {opt_out}."
                    )
            else:
                if forced_reingestion:
                    print(
                        f"Iteration {i}: Forced reingestion found atom-count mismatch "
                        f"({len(parsed_opt_atoms)} vs {len(atoms)}); retaining in-memory geometry and "
                        f"reapplying charges from {opt_out}."
                    )
                _attach_charges_from_orca_output(
                    atoms,
                    opt_out,
                    label=f"iteration {i} optimised geometry",
                )
        except Exception as exc:
            if forced_reingestion:
                print(
                    f"Iteration {i}: Forced reingestion could not parse geometry from {opt_out} ({exc}); "
                    "falling back to charge-only attachment."
                )
            _attach_charges_from_orca_output(
                atoms,
                opt_out,
                label=f"iteration {i} optimised geometry",
            )
        if optimize_to_convergence:
            converged = _is_converged_with_force_fallback(
                atoms=atoms,
                fmax=fmax,
                opt_dir=dirs.opt_dir,
                fallback_atoms=pre_parse_atoms,
                workflow_backend=workflow_backend,
            )
            summary_fmax = None
            if not converged:
                summary_fmax = _last_true_fmax_from_al_summary(al_dir / "ml_summary.txt")
                if summary_fmax is not None and summary_fmax <= fmax:
                    converged = True
                    print(
                        f"Iteration {i}: Using active-learning summary convergence evidence "
                        f"(True fmax={summary_fmax:.6f} <= {fmax:.6f} eV/A)."
                    )
            if not converged:
                forces_eV_A = _resolve_forces_eV_A(
                    atoms,
                    opt_dir=dirs.opt_dir,
                    fallback_atoms=pre_parse_atoms,
                    workflow_backend=workflow_backend,
                )
                max_force = _max_force_norm(forces_eV_A) if forces_eV_A is not None else None
                if max_force is None:
                    if summary_fmax is not None:
                        force_status = (
                            f"True fmax={summary_fmax:.6f} eV/A "
                            f"(target <= {fmax:.6f} eV/A)"
                        )
                    else:
                        force_status = (
                            f"force unavailable (target <= {fmax:.6f} eV/A)"
                        )
                else:
                    force_status = (
                        f"max force={max_force:.6f} eV/A "
                        f"(target <= {fmax:.6f} eV/A)"
                    )
                soft_cycle = _record_soft_iteration_continuation(
                    iteration_attempts=iteration_attempts,
                    iteration_soft_continuations=iteration_soft_continuations,
                    iteration=i,
                )
                print(
                    f"Iteration {i}: geometry not converged; {force_status}. "
                    f"Continuing active learning ({_soft_continuation_status_text(cycle=soft_cycle)})."
                )
                continue
        if not dft_only:
            previous_choice = scan_choice
            active_choice = scan_choice

            if bond_scan.enabled and bond_pair is not None:
                record_scan_choice(
                    scan_log_path,
                    iteration=i,
                    scan_dir=last_scan_dir,
                    bond_pair=bond_pair,
                    bond_length=atoms.get_distance(*bond_pair),
                    choice=scan_choice,
                    target_wavelength_nm=target_wavelength_nm,
                    point_kind=scan_point_kind,
                    point_label=scan_point_label,
                )
                summarize_scan_log(scan_log_path)
            if geometry_log_path is not None:
                summarize_geometry_log(geometry_log_path)
        if bond_scan.enabled and bond_pair is not None:
            increment_bond_length(
                atoms,
                bond_pair,
                bond_increment,
                bond_scan.fix,
                target_distance=bond_scan.target_distance,
            )
            apply_bond_constraints(atoms, bond_scan.fixed_symbols, bond_pair)

        mark_iteration_complete(dirs.opt_dir, i, mode="tddft" if not dft_only else "dft")
        iteration_attempts.pop(i, None)
        iteration_soft_continuations.pop(i, None)
        iteration_last_error.pop(i, None)
        i += 1

    if dft_only and bond_scan.enabled:
        _run_dft_only_endpoint_scans(
            atoms=atoms,
            effective_n_iter=effective_n_iter,
            bond_pair=bond_pair,
            scan_log_path=scan_log_path,
            target_wavelength_nm=target_wavelength_nm,
            tddft_scan_nroots=tddft_scan_nroots,
            fragments=fragments,
            workflow_cpus=workflow_cpus,
            workflow_mem_per_cpu=workflow_mem_per_cpu,
            run_initial_dft_if_missing=run_initial_dft_if_missing,
            initial_dft=initial_dft,
            fmax=fmax,
            initial_fmax=initial_fmax,
            basis_keyword=basis_keyword,
            xc_functional=xc_functional,
            solvent=solvent,
            spin_multiplicity=spin_multiplicity,
            tddft_calc_factory=tddft_calc_factory,
            dft_force_calc_factory=dft_engrad_calc_factory,
            workflow_backend=workflow_backend,
        )

    return _finalize_workflow_outputs(
        atoms=atoms,
        effective_n_iter=effective_n_iter,
        use_tddft_optimisation=use_tddft_optimisation,
        bond_scan=bond_scan,
        bond_pair=bond_pair,
        scan_log_path=scan_log_path,
        target_wavelength_nm=target_wavelength_nm,
        workflow_cpus=workflow_cpus,
        workflow_mem_per_cpu=workflow_mem_per_cpu,
        tddft_calc_factory=tddft_calc_factory,
        dft_calc_factory=dft_calc_factory,
        fragments=fragments,
        tddft_scan_nroots=tddft_scan_nroots,
        final_tddft_spectrum=bool(final_tddft_spectrum),
        final_tddft_nroots=int(final_tddft_nroots),
        final_cluster_tddft_difference_spectrum=bool(
            final_cluster_tddft_difference_spectrum
        ),
        fmax=fmax,
        active_learning_optimizer=active_learning_optimizer,
        al_steps_per_iter=al_steps_per_iter,
        max_steps=max_steps,
        plots_dir=plots_dir,
        plotter=plotter,
        endpoints_db_path=endpoints_db_path,
        compute_cluster_fragment_single_point=compute_cluster_fragment_single_point,
        basis_keyword=basis_keyword,
        xc_functional=xc_functional,
        solvent=solvent,
        spin_multiplicity=spin_multiplicity,
        system_charge=system_charge,
        nodes=nodes,
        cpus_per_node=cpus_per_node,
        workflow_backend=workflow_backend,
    )


if __name__ == "__main__":
    atoms = read("orca.xyz")
    run_workflow(atoms, n_iter=2)
