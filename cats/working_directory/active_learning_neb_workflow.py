from __future__ import annotations

import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import inspect
import json
import math
import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixBondLength
from ase.data import covalent_radii
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate
from ase.mep.neb import NEB
from ase.optimize import BFGS, FIRE2


_OPTIMIZERS: dict[str, type] = {
    "bfgs": BFGS,
    "fire2": FIRE2,
    "fire": FIRE2,
}


class _ReferenceAnchoredMinModeAtoms(MinModeAtoms):
    def calculate_real_forces_and_energies(self, **kwargs):
        calculator = self.atoms.calc
        if not hasattr(calculator, "evaluating_dimer_center"):
            return super().calculate_real_forces_and_energies(**kwargs)
        calculator.evaluating_dimer_center = True
        # A trial may have cached these coordinates against the previous centre.
        calculator.reset()
        try:
            return super().calculate_real_forces_and_energies(**kwargs)
        finally:
            calculator.evaluating_dimer_center = False


_FINAL_IMAGE_STRATEGIES = {
    "explicit_final_image",
    "set_bond_distance_from_initial",
    "bond_extension_from_initial",  # legacy alias
    "dimer",
    "dimer_refine",
}


def _timestamp_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _append_workflow_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    try:
        line = f"[{_timestamp_utc()}] {str(message).rstrip()}\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
    except Exception:
        return


def _sanitize_stage_label(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "stage"
    chars: list[str] = []
    for char in text:
        if char.isalnum() or char in {"-", "_", "."}:
            chars.append(char)
        else:
            chars.append("_")
    sanitized = "".join(chars).strip("._")
    return sanitized or "stage"


def _catlearn_stage_root_from_logfile(logfile: str) -> Path:
    logfile_text = str(logfile or "").strip()
    if logfile_text and logfile_text != "-":
        try:
            return Path(logfile_text).resolve().parent / "catlearn_stages"
        except Exception:
            pass
    return Path.cwd().resolve() / "catlearn_stages"


def _allocate_catlearn_stage_dir(*, root: Path, stage_label: str) -> Path:
    safe_label = _sanitize_stage_label(stage_label)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / safe_label
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    index = 2
    while True:
        candidate = root / f"{safe_label}__{index:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        index += 1


def _resolve_catlearn_stage_dir(
    *,
    root: Path,
    stage_label: str,
    prefer_restart: bool = True,
) -> tuple[Path, bool]:
    safe_label = _sanitize_stage_label(stage_label)
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / safe_label
    if candidate.exists():
        if bool(prefer_restart):
            return candidate, True
        return _allocate_catlearn_stage_dir(root=root, stage_label=stage_label), True
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate, False


def _catlearn_restart_state_exists(stage_dir: Path) -> bool:
    markers = (
        "evaluated.traj",
        "ml_summary.txt",
        "predicted.traj",
        "predicted_evaluated.traj",
        "converged.traj",
        "initial_struc.traj",
    )
    for name in markers:
        if (stage_dir / name).exists():
            return True
    return False


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@dataclass(frozen=True)
class CalculatorFactoryConfig:
    backend: str = "mace"
    mace_model_path: str = ""
    mace_device: str = "cuda"
    mace_default_dtype: str = "float64"
    mace_dispersion: bool = False
    dft_cpus: int = 1
    dft_mem: int = 4000
    basis_keyword: str = "def2-svp"
    xc_functional: str = "pbe0"
    solvent: str | None = None
    spin_multiplicity: int | None = None
    dft_evaluation_timeout_seconds: int = 21600
    dft_min_distance_scale: float = 0.55
    dft_max_force_eV_A: float = 50.0
    dft_max_energy_drop_eV: float = 20.0
    target_wavelength_nm: float = 520.0
    tddft_scan_nroots: int = 50
    initial_iroot: int = 5
    system_charge: int = 0
    fragments: dict[str, Any] | None = None


@contextmanager
def _deadline(seconds: int, label: str):
    """Bound one Linux/HPC operation without adding another process layer."""
    seconds = int(seconds)
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, float(seconds))

    def _expired(_signum, _frame):
        raise TimeoutError(f"{label} exceeded {seconds} seconds")

    signal.signal(signal.SIGALRM, _expired)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous)


def _validate_dft_geometry(atoms: Atoms, minimum_scale: float) -> None:
    positions = np.asarray(atoms.positions, dtype=float)
    if not np.isfinite(positions).all():
        raise ValueError("DFT candidate contains non-finite coordinates")
    distances = atoms.get_all_distances(mic=bool(np.any(atoms.pbc)))
    numbers = np.asarray(atoms.numbers, dtype=int)
    limits = float(minimum_scale) * (
        covalent_radii[numbers][:, None] + covalent_radii[numbers][None, :]
    )
    mask = ~np.eye(len(atoms), dtype=bool)
    if np.any(distances[mask] < limits[mask]):
        i, j = np.argwhere(mask & (distances < limits))[0]
        raise ValueError(
            f"DFT candidate has implausible contact {int(i)}-{int(j)}: "
            f"{distances[i, j]:.3f} A < {limits[i, j]:.3f} A"
        )


def _normalize_optimizer_name(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if not key:
        return "bfgs"
    if key not in _OPTIMIZERS:
        allowed = ", ".join(sorted(_OPTIMIZERS))
        raise ValueError(f"Unsupported optimizer '{value}'. Choose one of: {allowed}.")
    return key


def _optimizer_class(value: str):
    return _OPTIMIZERS[_normalize_optimizer_name(value)]


def _normalize_final_image_strategy(value: str) -> str:
    key = str(value or "").strip().lower()
    if not key:
        return "explicit_final_image"
    if key == "bond_extension_from_initial":
        return "set_bond_distance_from_initial"
    if key in {"dimer", "dimer_refine"}:
        return "dimer"
    if key not in _FINAL_IMAGE_STRATEGIES:
        allowed = ", ".join(sorted(_FINAL_IMAGE_STRATEGIES))
        raise ValueError(
            f"Unsupported final_image_strategy '{value}'. Choose one of: {allowed}."
        )
    return key


def _safe_constraints_list(atoms: Atoms) -> list[Any]:
    constraints = getattr(atoms, "constraints", None)
    if constraints is None:
        return []
    if isinstance(constraints, (list, tuple)):
        return list(constraints)
    return [constraints]


def _fixed_indices_from_constraints(constraints: list[Any]) -> set[int]:
    indices: set[int] = set()
    for constraint in constraints:
        if constraint is None:
            continue
        name = constraint.__class__.__name__.lower()
        if "fixatoms" not in name:
            continue
        getter = getattr(constraint, "get_indices", None)
        if not callable(getter):
            continue
        try:
            resolved = getter()
        except Exception:
            continue
        for value in resolved:
            try:
                indices.add(int(value))
            except Exception:
                continue
    return indices


def _with_constraints(atoms: Atoms, constraints: list[Any]) -> Atoms:
    result = atoms.copy()
    if not constraints:
        result.set_constraint()
        return result
    try:
        result.set_constraint(copy.deepcopy(constraints))
    except Exception:
        result.set_constraint(constraints)
    return result


def _merged_constraints(
    initial_atoms: Atoms | None,
    final_atoms: Atoms,
    *,
    fixed_symbols: tuple[str, ...],
) -> list[Any]:
    initial_constraints = _safe_constraints_list(initial_atoms)
    final_constraints = _safe_constraints_list(final_atoms)
    fixed_indices = _fixed_indices_from_constraints(initial_constraints)
    fixed_indices.update(_fixed_indices_from_constraints(final_constraints))

    symbol_filter = {str(symbol).strip() for symbol in fixed_symbols if str(symbol).strip()}
    if symbol_filter:
        for atom in initial_atoms:
            if atom.symbol in symbol_filter:
                fixed_indices.add(int(atom.index))
        for atom in final_atoms:
            if atom.symbol in symbol_filter:
                fixed_indices.add(int(atom.index))

    merged: list[Any] = []
    for constraint in initial_constraints:
        if constraint is None:
            continue
        if "fixatoms" in constraint.__class__.__name__.lower():
            continue
        merged.append(constraint)

    if fixed_indices:
        merged.insert(0, FixAtoms(indices=sorted(fixed_indices)))
    return merged


def _max_atom_shift_A(initial_atoms: Atoms, final_atoms: Atoms) -> float:
    initial_positions = np.asarray(initial_atoms.get_positions(), dtype=float)
    final_positions = np.asarray(final_atoms.get_positions(), dtype=float)
    if initial_positions.shape != final_positions.shape:
        raise ValueError("Initial and final images must have matching atom count.")
    shifts = np.linalg.norm(final_positions - initial_positions, axis=1)
    if shifts.size == 0:
        return 0.0
    return float(np.max(shifts))


def _image_count_from_shift(
    *,
    max_shift_A: float,
    atom_shift_per_image_A: float,
    min_images: int,
    max_images: int,
) -> int:
    min_count = max(2, int(min_images))
    max_count = max(min_count, int(max_images))
    target_shift = float(atom_shift_per_image_A)
    if target_shift <= 0.0:
        return min_count
    estimated = int(math.ceil(float(max_shift_A) / target_shift)) + 1
    return max(min_count, min(max_count, estimated))


def _local_minima_indices(energies: list[float], *, minimum_energy_window_eV: float) -> list[int]:
    if len(energies) < 3:
        return []
    minima: list[int] = []
    threshold = float(minimum_energy_window_eV)
    for idx in range(1, len(energies) - 1):
        center = float(energies[idx])
        left = float(energies[idx - 1])
        right = float(energies[idx + 1])
        if center <= left - threshold and center <= right - threshold:
            minima.append(idx)
    return minima


def _max_force_from_atoms(atoms: Atoms) -> float | None:
    try:
        forces = np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        return None
    if forces.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(forces, axis=1)))


def _build_mace_factory(
    *,
    mace_model_path: str,
    mace_device: str,
    mace_default_dtype: str,
    mace_dispersion: bool,
) -> Callable[[], Any]:
    from mace.calculators import mace_mp

    kwargs: dict[str, Any] = {
        "model": str(mace_model_path).strip() or "small",
        "device": str(mace_device).strip() or "cuda",
        "default_dtype": str(mace_default_dtype).strip() or "float64",
        "dispersion": bool(mace_dispersion),
        "enable_cueq": True,
    }
    signature = inspect.signature(mace_mp)
    allowed = set(signature.parameters.keys())
    filtered = {key: value for key, value in kwargs.items() if key in allowed}

    def _factory():
        return mace_mp(**filtered)

    return _factory


def _build_dft_factory(config: CalculatorFactoryConfig) -> Callable[[], Any]:
    from cats.working_directory.tddft_active_learning_workflow import get_orca_dft_engrad_calculator

    counter = 0
    accepted_energies: list[float] = []

    class DFTNEBCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, *, directory: Path):
            super().__init__()
            self.directory = Path(directory)

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if atoms is None:
                atoms = self.atoms
            directory = Path(self.directory)
            directory.mkdir(parents=True, exist_ok=True)
            _validate_dft_geometry(atoms, config.dft_min_distance_scale)

            def run_orca():
                calc = get_orca_dft_engrad_calculator(
                    atoms,
                    cpus=int(config.dft_cpus),
                    mem=int(config.dft_mem),
                    directory=directory,
                    basis_keyword=config.basis_keyword,
                    xc_functional=config.xc_functional,
                    solvent=config.solvent,
                    spin_multiplicity=config.spin_multiplicity,
                )
                atoms.calc = calc
                with _deadline(config.dft_evaluation_timeout_seconds, f"ORCA evaluation in {directory}"):
                    return atoms.get_forces(), float(atoms.get_potential_energy())

            try:
                try:
                    forces, energy = run_orca()
                except Exception:
                    output = directory / "orca.out"
                    error_tail = b""
                    if output.is_file():
                        with output.open("rb") as handle:
                            handle.seek(max(0, output.stat().st_size - 65536))
                            error_tail = handle.read().lower()
                    if b"no orbitals were found in the gbw file" not in error_tail:
                        raise
                    for filename in ("initial.gbw", "orca.gbw"):
                        (directory / filename).unlink(missing_ok=True)
                    print(f"ORCA rejected the restart orbitals in {directory}; retrying once without MOREAD.")
                    forces, energy = run_orca()
            finally:
                atoms.calc = self
            force_max = float(np.max(np.linalg.norm(np.asarray(forces, dtype=float), axis=1)))
            if not math.isfinite(energy) or not np.isfinite(forces).all():
                raise RuntimeError(f"ORCA returned non-finite results in {directory}")
            if force_max > float(config.dft_max_force_eV_A):
                raise RuntimeError(
                    f"ORCA candidate force {force_max:.3f} eV/A exceeds "
                    f"{float(config.dft_max_force_eV_A):.3f} eV/A in {directory}"
                )
            if accepted_energies and energy < min(accepted_energies) - float(config.dft_max_energy_drop_eV):
                raise RuntimeError(
                    f"ORCA candidate energy dropped {min(accepted_energies) - energy:.3f} eV below "
                    f"previous evaluations in {directory}"
                )
            accepted_energies.append(energy)
            self.results["energy"] = energy
            self.results["forces"] = forces

    def _factory():
        nonlocal counter
        counter += 1
        return DFTNEBCalculator(directory=Path(f"dft_neb_calc_{counter:05d}"))

    return _factory


def _build_tddft_dimer_factory(config: CalculatorFactoryConfig) -> Callable[[], Any]:
    from functools import partial
    import json
    from pathlib import Path
    import numpy as np
    from ase.calculators.calculator import Calculator, all_changes
    from cats.orca.tddft.active_learning import (
        ProxyContext,
        ScanThenEngradProxy,
        copy_initial_gbw,
        generate_state_cubes,
        get_ct_info,
    )
    from cats.working_directory.tddft_active_learning_workflow import (
        IROOT_FOLLOWUP_SCORE_WEIGHTS,
        IROOT_INTENSITY_REFERENCE_WINDOW_NM,
        IROOT_MIN_CT_SIMILARITY,
        IROOT_MIN_OSCILLATOR_STRENGTH,
        IROOT_MIN_SIMILARITY,
        IROOT_MIN_TRANSITION_OVERLAP,
        IROOT_SCORE_WEIGHTS,
        IROOT_SIMILARITY_SIGMA_CT,
        IROOT_SIMILARITY_SIGMA_NM,
        IROOT_WAVELENGTH_SIGMA_NM,
        get_orca_tddft_calculator,
    )
    from cats.multiwfn.iroot_scoring import IRootCandidate

    calc_factory = partial(
        get_orca_tddft_calculator,
        basis_keyword=config.basis_keyword,
        xc_functional=config.xc_functional,
        solvent=config.solvent,
        spin_multiplicity=config.spin_multiplicity,
        system_charge=int(getattr(config, "system_charge", 0) or 0),
    )

    class TDDFTDimerProxy(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self):
            super().__init__()
            self.proxy: ScanThenEngradProxy | None = None
            self.initial_positions: np.ndarray | None = None
            self.initial_energy: float | None = None
            self.initial_forces: np.ndarray | None = None
            self.evaluating_dimer_center = False

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if atoms is None:
                atoms = self.atoms

            if self.proxy is None:
                iroot = int(getattr(config, "initial_iroot", 5) or 5)
                nroots = int(getattr(config, "tddft_scan_nroots", 50) or 50)
                cpus = int(config.dft_cpus)
                mem = int(config.dft_mem)
                fragments = getattr(config, "fragments", None) or {"1": "1-20", "2": "21-22", "3": "23"}

                init_scan_dir = Path("dimer_init_scan")
                init_scan_dir.mkdir(parents=True, exist_ok=True)
                scan_calc = calc_factory(
                    atoms,
                    cpus=cpus,
                    mem=mem,
                    task="energy",
                    directory=init_scan_dir,
                    nroots=nroots,
                )
                atoms.calc = scan_calc
                atoms.get_potential_energy()

                tracking_df = get_ct_info(init_scan_dir, fragments, min_oscillator_strength=0.0)
                all_states = ScanThenEngradProxy._track_states(tracking_df)
                s_cand = next((s for s in all_states if s["state"] == iroot), None)
                wl = float(s_cand["wavelength_nm"]) if s_cand else float(config.target_wavelength_nm)
                ev = float(s_cand["energy_ev"]) if s_cand else 1239.841984 / max(1.0, wl)
                f_val = float(s_cand["oscillator_strength"]) if s_cand else 0.0
                ct_val = float((s_cand.get("ct_metrics") or {}).get("q_ct", 0.0)) if s_cand else 0.0

                active_choice = IRootCandidate(
                    state=iroot,
                    wavelength_nm=wl,
                    intensity=f_val,
                    energy_ev=ev,
                    ct_e=ct_val,
                    score=1.0,
                )

                generate_state_cubes(init_scan_dir, [iroot], init_scan_dir / "state_tracking")

                init_engrad_dir = Path("dimer_init_engrad")
                init_engrad_dir.mkdir(parents=True, exist_ok=True)
                copy_initial_gbw(init_scan_dir, init_engrad_dir)
                engrad_calc = calc_factory(
                    atoms,
                    cpus=cpus,
                    mem=mem,
                    task="engrad",
                    directory=init_engrad_dir,
                    nroots=iroot + 5,
                    iroot=iroot,
                )
                atoms.calc = engrad_calc
                forces = atoms.get_forces()
                energy = float(atoms.get_potential_energy())

                self.initial_positions = atoms.get_positions().copy()
                self.initial_energy = energy
                self.initial_forces = np.asarray(forces, dtype=float).copy()
                self.results["energy"] = energy
                self.results["forces"] = self.initial_forces

                context = ProxyContext(
                    target_wavelength_nm=float(config.target_wavelength_nm),
                    fragments=fragments,
                    initial_score_weights=IROOT_SCORE_WEIGHTS,
                    followup_score_weights=IROOT_FOLLOWUP_SCORE_WEIGHTS,
                    min_oscillator_strength=IROOT_MIN_OSCILLATOR_STRENGTH,
                    wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
                    similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
                    similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
                    min_similarity=IROOT_MIN_SIMILARITY,
                    min_relative_intensity=0.10,
                    intensity_reference_window_nm=IROOT_INTENSITY_REFERENCE_WINDOW_NM,
                    min_transition_overlap=IROOT_MIN_TRANSITION_OVERLAP,
                    min_ct_similarity=IROOT_MIN_CT_SIMILARITY,
                    nroots=nroots,
                )
                self.proxy = ScanThenEngradProxy(
                    cpus=cpus,
                    mem=mem,
                    scan_base=Path("dimer_scan"),
                    opt_dir=Path("dimer_opt"),
                    gbw_source_dir=init_engrad_dir,
                    previous_choice=active_choice,
                    previous_state=iroot,
                    iteration=0,
                    geometry_step_start=1,
                    bond_pair=None,
                    geometry_log_path=Path("dimer_geometry.csv"),
                    context=context,
                    calc_factory=calc_factory,
                    engrad_calc_factory=calc_factory,
                    use_scan_geometry=False,
                    reference_dir=init_scan_dir,
                )
                atoms.calc = self
                return

            if (
                self.initial_positions is not None
                and self.proxy.reference_dir == Path("dimer_init_scan")
                and np.array_equal(atoms.get_positions(), self.initial_positions)
            ):
                self.results["energy"] = self.initial_energy
                self.results["forces"] = self.initial_forces.copy()
                return

            self.proxy._load_reference()
            reference = {name: getattr(self.proxy, name) for name in (
                "reference_point", "reference_cubes", "reference_dir", "previous_choice", "previous_state",
            )}
            # Restart from the immutable accepted-centre scan, not a preceding probe.
            self.proxy.gbw_source_dir = self.proxy.reference_dir
            self.proxy.reset()
            atoms.calc = self.proxy
            succeeded = False
            try:
                self.results["forces"] = atoms.get_forces()
                self.results["energy"] = float(atoms.get_potential_energy())
                tracking_path = self.proxy.last_scan_dir / "state_tracking.json"
                tracking = json.loads(tracking_path.read_text(encoding="utf-8"))
                tracking.update(
                    reference_step_id=str(reference["reference_dir"]),
                    reference_state=reference["previous_state"],
                    evaluation_kind="accepted_center" if self.evaluating_dimer_center else "probe",
                )
                tracking_path.write_text(json.dumps(tracking, indent=2), encoding="utf-8")
                succeeded = True
            finally:
                if not succeeded or not self.evaluating_dimer_center:
                    for name, value in reference.items():
                        setattr(self.proxy, name, value)
                self.proxy.gbw_source_dir = self.proxy.reference_dir
                atoms.calc = self

    instance = TDDFTDimerProxy()
    return lambda: instance


def build_calculator_factory(config: CalculatorFactoryConfig) -> Callable[[], Any]:
    backend = str(config.backend or "").strip().lower()
    if backend == "mace":
        return _build_mace_factory(
            mace_model_path=config.mace_model_path,
            mace_device=config.mace_device,
            mace_default_dtype=config.mace_default_dtype,
            mace_dispersion=config.mace_dispersion,
        )
    if backend == "dft":
        return _build_dft_factory(config)
    if backend in {"tddft", "tda"}:
        return _build_tddft_dimer_factory(config)
    raise NotImplementedError(
        f"Backend '{backend}' is not implemented. Extend build_calculator_factory to add DFT/TDDFT backends."
    )


def _resolve_share_calculator_instance(*, backend: str, share_calculator_instance: bool | None) -> bool:
    if share_calculator_instance is not None:
        return bool(share_calculator_instance)
    # Default for MACE: keep a single CUDA-backed calculator object.
    return str(backend or "").strip().lower() == "mace"


def _calculator_getter_from_factory(
    *,
    factory: Callable[[], Any],
    share_calculator_instance: bool,
) -> Callable[[], Any]:
    if not bool(share_calculator_instance):
        return factory

    shared_calculator: Any | None = None

    def _shared_getter():
        nonlocal shared_calculator
        if shared_calculator is None:
            shared_calculator = factory()
        return shared_calculator

    return _shared_getter


def _attach_calculators(images: list[Atoms], calculator_getter: Callable[[], Any]) -> None:
    for image in images:
        image.calc = calculator_getter()


def _run_neb_pass(
    *,
    images: list[Atoms],
    calculator_getter: Callable[[], Any],
    share_calculator_instance: bool,
    fmax: float,
    max_steps: int,
    spring_constant: float,
    climb: bool,
    optimizer: str,
    workflow_log_path: Path | None,
    output_dir: Path,
    label: str,
) -> tuple[list[float], str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _attach_calculators(images, calculator_getter)
    stage_text = str(label or "unnamed_stage")
    neb_log_path = output_dir / f"{label}.log"
    neb_traj_path = output_dir / f"{label}.traj"
    _append_workflow_log(
        workflow_log_path,
        (
            f"{stage_text}: starting ASE climbing-image NEB "
            f"(fmax={float(fmax):.6f}, steps={int(max_steps)}, "
            "optimizer=bfgs, maxstep=1000.000000)."
        ),
    )
    neb_kwargs: dict[str, Any] = {
        "climb": bool(climb),
        "k": float(spring_constant),
    }
    neb_signature = inspect.signature(NEB)
    if "allow_shared_calculator" in neb_signature.parameters:
        neb_kwargs["allow_shared_calculator"] = bool(share_calculator_instance)
    neb = NEB(images, **neb_kwargs)
    neb.interpolate(method="idpp", apply_constraint=True)

    optimizer_cls = BFGS
    optimizer_kwargs = _filter_supported_kwargs(
        optimizer_cls,
        {
            "logfile": str(neb_log_path),
            "trajectory": str(neb_traj_path),
            "maxstep": 1000.0,
        },
    )
    dyn = optimizer_cls(neb, **optimizer_kwargs)
    if hasattr(dyn, "maxstep"):
        try:
            setattr(dyn, "maxstep", 1000.0)
        except Exception:
            pass
    dyn.run(fmax=float(fmax), steps=int(max_steps))

    energies = [float(image.get_potential_energy()) for image in images]
    write(output_dir / f"{label}.xyz", images, format="extxyz")
    _append_workflow_log(workflow_log_path, f"{stage_text}: ASE climbing-image NEB completed.")
    return energies, "ase_climbing_image_neb"


def _select_mlneb_training_frames(
    frames: list[Atoms], endpoints: tuple[Atoms, Atoms], limit: int
) -> list[Atoms]:
    """Retain endpoint anchors, the highest-energy sample, then geometric coverage."""
    if limit < 3:
        raise ValueError("The MLNEB training pool needs room for both endpoints and one interior image.")
    selected = list(endpoints)
    remaining = [
        frame for frame in frames
        if not any(np.allclose(frame.positions, endpoint.positions, atol=1.0e-6) for endpoint in endpoints)
    ]
    if remaining:
        peak = max(remaining, key=lambda frame: float(frame.get_potential_energy()))
        selected.append(peak)
        remaining = [frame for frame in remaining if frame is not peak]
    while remaining and len(selected) < limit:
        next_frame = max(
            remaining,
            key=lambda frame: min(
                np.linalg.norm((frame.positions - anchor.positions).ravel()) for anchor in selected
            ),
        )
        selected.append(next_frame)
        remaining = [frame for frame in remaining if frame is not next_frame]
    return selected


def _run_catlearn_mlneb_pass(
    *,
    start: Atoms,
    end: Atoms,
    calculator_getter: Callable[[], Any],
    n_images: int,
    fmax: float,
    max_steps: int,
    spring_constant: float,
    catlearn_unc_convergence: float,
    catlearn_max_unc: float,
    catlearn_ml_steps: int,
    catlearn_dtrust: float,
    workflow_log_path: Path | None,
    output_dir: Path,
    initial_band_atoms: list[Atoms] | None = None,
    initial_training_atoms: list[Atoms] | None = None,
    catlearn_guesses_per_cycle: int | None = 10,
    restart_from: Path | str | None = None,
) -> tuple[list[Atoms], list[float], bool, int]:
    """Run CatLearn's MLNEB directly on the expensive DFT calculator."""
    try:
        from catlearn.activelearning.mlneb import MLNEB

        avishart_api = True
    except Exception:
        try:
            from catlearn.optimize.mlneb import MLNEB

            avishart_api = False
        except Exception as exc:
            raise ImportError(
                "DFT MLNEB requires CatLearn's MLNEB implementation in the runtime environment."
            ) from exc

    stage_dir = output_dir / "catlearn_mlneb"
    trajectory_name = "converged.traj" if avishart_api else "MLNEB.traj"
    trajectory_path = stage_dir / trajectory_name
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Import old calculation state into stage_dir if restart_from provided.
    if restart_from is not None:
        src = Path(restart_from).resolve()
        if src.is_dir():
            nested = src / "neb" / "catlearn_mlneb"
            if not nested.exists():
                nested = src / "catlearn_mlneb"
            if nested.exists():
                src = nested
            import shutil
            for fname in ("all_evaluated.traj", "evaluated.traj", "evaluated_structures.traj", "predicted.traj", "last_predicted_path.traj", "all_predicted_paths.traj", "initial.traj", "final.traj", "ml_summary.txt", "ml_time.txt"):
                sf = src / fname
                df = stage_dir / fname
                if sf.exists() and not df.exists():
                    shutil.copy2(sf, df)
        elif src.is_file():
            import shutil
            known_names = {"evaluated.traj", "evaluated_structures.traj", "predicted.traj", "last_predicted_path.traj", "all_predicted_paths.traj"}
            target_name = src.name if src.name in known_names else "predicted.traj"
            if not (stage_dir / target_name).exists():
                shutil.copy2(src, stage_dir / target_name)

    _append_workflow_log(
        workflow_log_path,
        (
            "Starting CatLearn MLNEB "
            f"(images={int(n_images)}, fmax={float(fmax):.6f}, "
            f"steps={int(max_steps)}, ml_steps={int(catlearn_ml_steps)})."
        ),
    )

    with _pushd(stage_dir):
        if restart_from is not None and trajectory_path.exists():
            try:
                trajectory_path.unlink()
            except Exception:
                pass

        for atoms, filename in ((start, "initial.traj"), (end, "final.traj")):
            saved = Path(filename)
            if not saved.exists() or not np.allclose(read(filename).positions, atoms.positions, atol=1.0e-6):
                evaluated = atoms.copy()
                evaluated.calc = calculator_getter()
                evaluated.get_forces()
                write(filename, evaluated)

        if trajectory_path.exists():
            images = list(read(trajectory_name, ":"))
            dyn = None
            converged_result = True
            total_evaluations = 0
        else:
            fallback_trajectory = Path("predicted.traj" if avishart_api else "last_predicted_path.traj")
            if not avishart_api and not fallback_trajectory.exists():
                fallback_trajectory = Path("all_predicted_paths.traj")
            eval_path = Path("evaluated.traj" if avishart_api else "evaluated_structures.traj")
            all_eval_path = Path("all_evaluated.traj")
            training_limit = 30
            converged_result = False
            remaining_steps = int(max_steps)
            cycle_budget = (
                int(catlearn_guesses_per_cycle)
                if (catlearn_guesses_per_cycle is not None and int(catlearn_guesses_per_cycle) > 0)
                else remaining_steps
            )
            current_band = initial_band_atoms
            dyn = None
            total_evaluations = 0

            if initial_training_atoms:
                existing = list(read(str(eval_path), ":")) if eval_path.exists() else []
                seeded = _select_mlneb_training_frames(
                    existing + list(initial_training_atoms),
                    (read("initial.traj"), read("final.traj")),
                    training_limit,
                )
                write(str(eval_path), seeded, format="traj")
                _append_workflow_log(
                    workflow_log_path,
                    f"Seeded CatLearn with {max(0, len(seeded) - 2)} evaluated initial-band frames.",
                )

            # Recover evaluated frames before trimming the active training set.
            if not eval_path.exists() and all_eval_path.exists():
                archived = list(read(str(all_eval_path), ":"))
                write(
                    str(eval_path),
                    _select_mlneb_training_frames(
                        archived, (read("initial.traj"), read("final.traj")), training_limit
                    ),
                    format="traj",
                )
            if eval_path.exists():
                old_evals = list(read(str(eval_path), ":"))
                if not all_eval_path.exists():
                    write(str(all_eval_path), old_evals, format="traj")
                else:
                    archived = list(read(str(all_eval_path), ":"))
                    seen = {(frame.positions.tobytes(), float(frame.get_potential_energy())) for frame in archived}
                    with Trajectory(str(all_eval_path), "a") as archive_traj:
                        for frame in old_evals:
                            key = (frame.positions.tobytes(), float(frame.get_potential_energy()))
                            if key not in seen:
                                archive_traj.write(frame)
                                seen.add(key)

            mlcalc_file = Path("mlcalc.pkl")
            if mlcalc_file.exists():
                try:
                    mlcalc_file.unlink()
                except Exception:
                    pass

            if current_band is None and fallback_trajectory.exists():
                try:
                    all_pred = list(read(str(fallback_trajectory), ":"))
                    if len(all_pred) >= n_images:
                        current_band = all_pred[-n_images:]
                except Exception:
                    pass

            # Keep the training set at or below 30, including this cycle's evaluations.
            while remaining_steps > 0:
                cycle_started = time.monotonic()
                step_limit = min(cycle_budget, remaining_steps, training_limit - 2)
                with Path("ml_timing.jsonl").open("a", encoding="utf-8") as timing:
                    timing.write(json.dumps({
                        "timestamp_utc": _timestamp_utc(),
                        "event": "cycle_started",
                        "step_limit": int(step_limit),
                        "ml_steps": int(catlearn_ml_steps),
                    }, sort_keys=True) + "\n")
                pool_count = 0
                if eval_path.exists():
                    eval_frames = list(read(str(eval_path), ":"))
                    kept = _select_mlneb_training_frames(
                        eval_frames,
                        (read("initial.traj"), read("final.traj")),
                        training_limit - step_limit,
                    )
                    if len(kept) != len(eval_frames) or any(a is not b for a, b in zip(kept, eval_frames)):
                        write(str(eval_path), kept, format="traj")
                    pool_count = len(kept)
                if avishart_api and eval_path.exists() and not fallback_trajectory.exists() and current_band is not None:
                    write(str(fallback_trajectory), current_band, format="traj")
                restart = eval_path.exists() and (
                    not avishart_api
                    or all(Path(name).exists() for name in ("predicted.traj", "ml_summary.txt", "ml_time.txt"))
                )
                if avishart_api:
                    dyn = MLNEB(
                        start="initial.traj",
                        end="final.traj",
                        ase_calc=calculator_getter(),
                        n_images=max(3, int(n_images)),
                        neb_interpolation=(current_band if current_band is not None else "idpp"),
                        neb_kwargs={"k": float(spring_constant)},
                        climb=True,
                        unc_convergence=float(catlearn_unc_convergence),
                        use_restart=True,
                        restart=restart,
                        prev_calculations=list(read(str(eval_path), ":")) if eval_path.exists() and not restart else None,
                        verbose=True,
                    )
                    converged_result = bool(
                        dyn.run(
                            fmax=float(fmax),
                            steps=int(step_limit),
                            ml_steps=int(catlearn_ml_steps),
                            max_unc=float(catlearn_max_unc),
                            dtrust=float(catlearn_dtrust),
                        )
                    )
                else:
                    dyn = MLNEB(
                        start="initial.traj",
                        end="final.traj",
                        ase_calc=calculator_getter(),
                        n_images=max(3, int(n_images)),
                        interpolation=(current_band if current_band is not None else "idpp"),
                        restart=restart,
                        k=float(spring_constant),
                    )
                    dyn.run(
                        fmax=float(fmax),
                        unc_convergence=float(catlearn_unc_convergence),
                        steps=int(step_limit),
                        trajectory=trajectory_name,
                        ml_steps=int(catlearn_ml_steps),
                        full_output=True,
                    )
                    converged_result = Path(trajectory_name).exists()

                with Path("ml_timing.jsonl").open("a", encoding="utf-8") as timing:
                    timing.write(json.dumps({
                        "timestamp_utc": _timestamp_utc(),
                        "event": "cycle_completed",
                        "cycle_seconds": round(time.monotonic() - cycle_started, 3),
                        "step_limit": int(step_limit),
                        "ml_steps": int(catlearn_ml_steps),
                        "converged": bool(converged_result),
                    }, sort_keys=True) + "\n")

                remaining_steps -= step_limit

                # CatLearn appends new evaluations to the active trajectory.
                if eval_path.exists():
                    current_evals = list(read(str(eval_path), ":"))
                    if len(current_evals) < pool_count:
                        raise RuntimeError("CatLearn replaced the active evaluations; cannot safely update the archive.")
                    if not all_eval_path.exists():
                        write(str(all_eval_path), current_evals, format="traj")
                    else:
                        with Trajectory(str(all_eval_path), "a") as archive_traj:
                            for frame in current_evals[pool_count:]:
                                archive_traj.write(frame)
                    total_evaluations = max(0, len(read(str(all_eval_path), ":")) - 2)
                    if len(current_evals) > training_limit:
                        raise RuntimeError(f"CatLearn exceeded the {training_limit}-evaluation training limit.")

                if converged_result or Path(trajectory_name).exists() or remaining_steps <= 0:
                    break

                # Update band to the latest predicted MEP.
                if not avishart_api:
                    fallback_trajectory = Path("last_predicted_path.traj")
                    if not fallback_trajectory.exists():
                        fallback_trajectory = Path("all_predicted_paths.traj")
                if fallback_trajectory.exists():
                    try:
                        all_pred = list(read(str(fallback_trajectory), ":"))
                        if len(all_pred) >= n_images:
                            current_band = all_pred[-n_images:]
                    except Exception:
                        pass

                # Purge the covariance cache before rebuilding with the selected pool.
                mlcalc_file = Path("mlcalc.pkl")
                if mlcalc_file.exists():
                    try:
                        mlcalc_file.unlink()
                    except Exception:
                        pass

                _append_workflow_log(
                    workflow_log_path,
                    (
                        f"CatLearn MLNEB cycle complete: active pool has {len(current_evals) if eval_path.exists() else 0} evaluated frames, "
                        f"purged GPR cache, remaining step budget={remaining_steps}."
                    ),
                )

            if Path(trajectory_name).exists():
                images = list(read(trajectory_name, ":"))
            elif isinstance(getattr(dyn, "best_structures", None), (list, tuple)):
                images = [_copy_with_single_point_results(image) for image in dyn.best_structures]
            elif fallback_trajectory.exists():
                stored_images = list(read(fallback_trajectory, ":"))
                images = stored_images[-max(3, int(n_images)) :]
            else:
                images = [image.copy() for image in getattr(dyn, "images", ())]

    if len(images) < 3:
        raise RuntimeError("CatLearn MLNEB did not return a path with at least three images.")
    images = [_copy_with_single_point_results(image) for image in images]
    energies = [float(image.get_potential_energy()) for image in images]
    evaluated_count = total_evaluations if total_evaluations > 0 else (0 if dyn is None else max(0, len(getattr(dyn, "list_targets", ())) - 2))
    converged = trajectory_path.exists() if dyn is None else bool(converged_result or trajectory_path.exists())
    _append_workflow_log(
        workflow_log_path,
        f"CatLearn MLNEB finished (converged={converged}, evaluations={evaluated_count}).",
    )
    return images, energies, converged, evaluated_count


def _target_calculator_for_localal(target: Any) -> Any | None:
    calc = getattr(target, "calc", None)
    if calc is not None:
        return calc
    images = getattr(target, "images", None)
    if isinstance(images, (list, tuple)):
        for image in images:
            image_calc = getattr(image, "calc", None)
            if image_calc is not None:
                return image_calc
    return None


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        return kwargs
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_kwargs:
        return kwargs
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def _run_localal_or_fallback(
    *,
    target: Any,
    target_calc: Any | None,
    optimizer: str,
    fmax: float,
    steps: int,
    catlearn_min_data: int,
    catlearn_unc_convergence: float,
    catlearn_max_unc: float,
    catlearn_ml_steps: int,
    catlearn_dtrust: float,
    logfile: str = "-",
    trajectory: str | None = None,
    allow_optimizer_fallback: bool = False,
    stage_label: str = "",
    workflow_log_path: Path | None = None,
) -> str:
    optimizer_key = _normalize_optimizer_name(optimizer)
    stage_text = str(stage_label or "unnamed_stage")
    base_optimizer_cls = _OPTIMIZERS[optimizer_key]
    stage_root = _catlearn_stage_root_from_logfile(logfile)
    stage_dir, stage_preexisting = _resolve_catlearn_stage_dir(
        root=stage_root,
        stage_label=stage_text,
        prefer_restart=True,
    )
    restart_enabled = bool(stage_preexisting and _catlearn_restart_state_exists(stage_dir))
    _append_workflow_log(
        workflow_log_path,
        (
            f"{stage_text}: starting CatLearn LocalAL "
            f"(fmax={float(fmax):.6f}, steps={int(steps)}, "
            f"max_unc={float(catlearn_max_unc):.6f}, ml_steps={int(catlearn_ml_steps)}, "
            f"dtrust={float(catlearn_dtrust):.6f}, restart={restart_enabled})."
        ),
    )
    _append_workflow_log(
        workflow_log_path,
        f"{stage_text}: CatLearn workspace directory: {stage_dir}.",
    )
    try:
        with _pushd(stage_dir):
            from catlearn.activelearning.local import LocalAL

            localal_kwargs = _filter_supported_kwargs(
                LocalAL,
                {
                    "atoms": target,
                    "ase_calc": target_calc if target_calc is not None else _target_calculator_for_localal(target),
                    "unc_convergence": float(catlearn_unc_convergence),
                    "local_opt": base_optimizer_cls,
                    "local_opt_kwargs": {"maxstep": 0.1},
                    "min_data": int(catlearn_min_data),
                    "save_memory": False,
                    "use_restart": bool(restart_enabled),
                    "restart": bool(restart_enabled),
                    "verbose": False,
                },
            )
            dyn = LocalAL(**localal_kwargs)
            run_kwargs = _filter_supported_kwargs(
                dyn.run,
                {
                    "fmax": float(fmax),
                    "max_unc": float(catlearn_max_unc),
                    "steps": int(steps),
                    "ml_steps": int(catlearn_ml_steps),
                    "dtrust": float(catlearn_dtrust),
                },
            )
            dyn.run(**run_kwargs)
        _append_workflow_log(workflow_log_path, f"{stage_text}: CatLearn LocalAL completed.")
        return "catlearn_localal"
    except Exception as exc:
        _append_workflow_log(
            workflow_log_path,
            (
                f"{stage_text}: CatLearn LocalAL failed ({exc.__class__.__name__}: {exc}). "
                f"Workspace: {stage_dir}."
            ),
        )
        if not bool(allow_optimizer_fallback):
            raise RuntimeError(
                f"CatLearn LocalAL failed for stage '{stage_text}' with "
                f"{exc.__class__.__name__}: {exc}. "
                f"Workspace: {stage_dir}. "
                "Set allow_optimizer_fallback=True to use ASE fallback."
            ) from exc
        _append_workflow_log(
            workflow_log_path,
            f"{stage_text}: falling back to ASE optimizer '{optimizer_key}'.",
        )
        optimizer_kwargs: dict[str, Any] = {"logfile": str(logfile)}
        if trajectory is not None:
            optimizer_kwargs["trajectory"] = str(trajectory)
        dyn = base_optimizer_cls(target, **optimizer_kwargs)
        dyn.run(fmax=float(fmax), steps=int(steps))
        _append_workflow_log(
            workflow_log_path,
            f"{stage_text}: ASE optimizer fallback completed.",
        )
        return "ase_optimizer_fallback"


def _optimise_structure(
    *,
    atoms: Atoms,
    calculator_getter: Callable[[], Any],
    optimizer: str,
    fmax: float,
    steps: int,
    catlearn_min_data: int,
    catlearn_unc_convergence: float,
    catlearn_max_unc: float,
    catlearn_ml_steps: int,
    catlearn_dtrust: float,
    use_regular_optimizer: bool,
    allow_optimizer_fallback: bool,
    workflow_log_path: Path | None,
    output_dir: Path,
    label: str,
) -> tuple[Atoms, str]:
    candidate = atoms.copy()
    candidate.calc = calculator_getter()
    if bool(use_regular_optimizer):
        optimizer_cls = _optimizer_class(optimizer)
        dyn = optimizer_cls(
            candidate,
            logfile=str(output_dir / f"{label}.log"),
            trajectory=str(output_dir / f"{label}.traj"),
        )
        dyn.run(fmax=float(fmax), steps=int(steps))
        method = "ase_optimizer_endpoint"
    else:
        method = _run_localal_or_fallback(
            target=candidate,
            target_calc=candidate.calc,
            optimizer=optimizer,
            fmax=fmax,
            steps=steps,
            catlearn_min_data=catlearn_min_data,
            catlearn_unc_convergence=catlearn_unc_convergence,
            catlearn_max_unc=catlearn_max_unc,
            catlearn_ml_steps=catlearn_ml_steps,
            catlearn_dtrust=catlearn_dtrust,
            logfile=str(output_dir / f"{label}.log"),
            trajectory=str(output_dir / f"{label}.traj"),
            allow_optimizer_fallback=allow_optimizer_fallback,
            stage_label=label,
            workflow_log_path=workflow_log_path,
        )
    write(output_dir / f"{label}.xyz", candidate, format="xyz")
    return candidate, method


def _catlearn_refine_structure(
    *,
    atoms: Atoms,
    calculator_getter: Callable[[], Any],
    optimizer: str,
    fmax: float,
    catlearn_min_data: int,
    catlearn_unc_convergence: float,
    catlearn_max_unc: float,
    catlearn_steps: int,
    catlearn_ml_steps: int,
    catlearn_dtrust: float,
    allow_optimizer_fallback: bool,
    workflow_log_path: Path | None,
    output_dir: Path,
    label: str,
) -> tuple[Atoms, str]:
    candidate = atoms.copy()
    candidate.calc = calculator_getter()
    method = _run_localal_or_fallback(
        target=candidate,
        target_calc=candidate.calc,
        optimizer=optimizer,
        fmax=fmax,
        steps=catlearn_steps,
        catlearn_min_data=catlearn_min_data,
        catlearn_unc_convergence=catlearn_unc_convergence,
        catlearn_max_unc=catlearn_max_unc,
        catlearn_ml_steps=catlearn_ml_steps,
        catlearn_dtrust=catlearn_dtrust,
        logfile=str(output_dir / f"{label}.log"),
        trajectory=str(output_dir / f"{label}.traj"),
        allow_optimizer_fallback=allow_optimizer_fallback,
        stage_label=label,
        workflow_log_path=workflow_log_path,
    )
    return candidate, method


def _copy_with_constraint_set(atoms: Atoms, constraints: list[Any]) -> Atoms:
    candidate = atoms.copy()
    if constraints:
        try:
            candidate.set_constraint(copy.deepcopy(constraints))
        except Exception:
            candidate.set_constraint(constraints)
    else:
        candidate.set_constraint()
    return candidate


def _copy_with_single_point_results(atoms: Atoms) -> Atoms:
    """Copy atoms without losing evaluated energy/forces."""
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float).copy()
    candidate = atoms.copy()
    candidate.calc = SinglePointCalculator(candidate, energy=energy, forces=forces)
    return candidate


def _validate_endpoints(initial_atoms: Atoms, final_atoms: Atoms) -> None:
    if len(initial_atoms) != len(final_atoms):
        raise ValueError("Initial and final images must contain the same number of atoms.")
    initial_numbers = tuple(int(number) for number in initial_atoms.get_atomic_numbers())
    final_numbers = tuple(int(number) for number in final_atoms.get_atomic_numbers())
    if initial_numbers != final_numbers:
        raise ValueError(
            "Initial and final images must preserve atom ordering and atomic numbers for NEB interpolation."
        )


def _generate_bond_extension_final(
    *,
    initial_atoms: Atoms,
    bond_i: int | None,
    bond_j: int | None,
    target_distance_A: float,
    fix: float,
) -> tuple[Atoms, int, int, float, float]:
    if bond_i is None or bond_j is None:
        raise ValueError("set_bond_distance_from_initial requires break_bond_i and break_bond_j.")
    i = int(bond_i)
    j = int(bond_j)
    if i == j:
        raise ValueError("break_bond_i and break_bond_j must reference different atoms.")
    atom_count = len(initial_atoms)
    if i < 0 or i >= atom_count or j < 0 or j >= atom_count:
        raise ValueError(
            f"Bond indices out of range for {atom_count} atoms: break_bond_i={i}, break_bond_j={j}."
        )
    if float(target_distance_A) <= 0.0:
        raise ValueError("set_bond_distance_A must be > 0.")

    candidate = initial_atoms.copy()
    constraints = _safe_constraints_list(candidate)
    if constraints:
        candidate.set_constraint()
    mic = bool(np.any(candidate.get_pbc()))
    distance_initial = float(candidate.get_distance(i, j, mic=mic))
    distance_target = float(target_distance_A)
    candidate.set_distance(
        i,
        j,
        distance_target,
        fix=float(fix),
        mic=mic,
    )
    if constraints:
        try:
            candidate.set_constraint(copy.deepcopy(constraints))
        except Exception:
            candidate.set_constraint(constraints)
    return candidate, i, j, distance_initial, distance_target


def _bond_distance_and_cutoff(
    *,
    atoms: Atoms,
    bond_i: int,
    bond_j: int,
    cutoff_scale: float,
) -> tuple[float, float]:
    mic = bool(np.any(atoms.get_pbc()))
    distance = float(atoms.get_distance(int(bond_i), int(bond_j), mic=mic))
    z_i = int(atoms[int(bond_i)].number)
    z_j = int(atoms[int(bond_j)].number)
    radii_sum = float(covalent_radii[z_i] + covalent_radii[z_j])
    if radii_sum <= 0.0:
        cutoff = float(cutoff_scale)
    else:
        cutoff = float(cutoff_scale) * radii_sum
    return distance, cutoff


def _run_local_ts_vibrations(atoms: Atoms, *, shells: int, output_dir: Path) -> dict[str, Any]:
    from ase.vibrations import Vibrations
    if not isinstance(shells, (int, np.integer)) or isinstance(shells, bool) or not 0 <= shells <= 3:
        raise ValueError("vibration_neighbor_shells must be an integer from 0 to 3 (0 = all atoms)")
    shells = int(shells)
    if shells == 0:
        calculator = atoms.calc
        atoms = atoms.copy()
        atoms.set_constraint()
        atoms.calc = calculator
        indices = list(range(len(atoms)))
    else:
        from carmm.analyse.neighbours import neighbours
        centres = [a.index for a in atoms if a.symbol not in ("Cu", "Au")]
        if not centres:
            raise ValueError("Local TS vibrations require adsorbate atoms")
        indices, _ = neighbours(atoms, centre=centres, shell=shells)
        indices = [int(i) for i in sorted(set(centres) | set(indices))]
    output_dir.mkdir(parents=True, exist_ok=True)
    vib_dir = output_dir / "vib"
    if vib_dir.is_dir():
        for f in vib_dir.glob("cache.*.json"):
            if f.stat().st_size == 0:
                print(f"[vibrations] Removing 0-byte cache file: {f.name}")
                f.unlink()
    vibrations = Vibrations(atoms, indices=indices, name=str(output_dir / "vib"))
    vibrations.run()
    frequencies = vibrations.get_frequencies()
    signed = [float(-abs(f.imag) if abs(f.imag) > 1e-8 else f.real) for f in frequencies]
    result = {"method": "ASE finite-difference full Hessian" if shells == 0 else "ASE finite-difference partial Hessian", "neighbor_shells": shells,
              "active_atom_indices_0based": indices, "frequencies_cm1": signed,
              "normal_modes": [vibrations.get_mode(i).tolist() for i in range(len(signed))],
              "imaginary_count": sum(f < -0.01 for f in signed),
              "scope": "all atoms; constraints removed" if shells == 0 else "adsorbates plus neighbor shells; not a full-system saddle-order test"}
    (output_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


try:
    from catlearn_dimer import (
        PersistentDimerLocalOpt,
        seed_dimer_training_set,
        DimerRunState,
        run_catlearn_dimer_workflow,
        validate_dft_frame,
    )
except ImportError:
    try:
        from .catlearn_dimer import (
            PersistentDimerLocalOpt,
            seed_dimer_training_set,
            DimerRunState,
            run_catlearn_dimer_workflow,
            validate_dft_frame,
        )
    except ImportError:
        from cats.working_directory.catlearn_dimer import (
            PersistentDimerLocalOpt,
            seed_dimer_training_set,
            DimerRunState,
            run_catlearn_dimer_workflow,
            validate_dft_frame,
        )


def _run_dimer_refine_pass(
    *,
    atoms: Atoms,
    calculator_getter: Callable[[], Any],
    fmax: float = 0.02,
    max_steps: int = 100,
    dimer_separation: float = 0.0001,
    trial_trans_step: float = 0.001,
    maximum_translation: float = 0.05,
    f_rot_min: float = 0.1,
    f_rot_max: float = 1.0,
    max_num_rot: int = 3,
    trial_angle: float = math.pi / 4.0,
    use_central_forces: bool = True,
    initial_eigenmode: Any = None,
    displacement_vector: Any = None,
    workflow_log_path: Path | None = None,
    output_dir: Path,
    label: str = "dimer",
    neighbour_shell: int = 2,
    use_catlearn: bool = False,
    catlearn_polish_steps: int = 25,
    prior_trajectories: Sequence[str | Path] | None = None,
    prior_artifact_ids: Sequence[int] | None = None,
    staged_seed_file: str | Path | None = None,
    catlearn_dtrust: float = 0.20,
    catlearn_max_unc: float = 0.03,
    catlearn_ml_steps: int = 200,
) -> tuple[Atoms, float, float, list[list[float]], bool]:
    # ponytail: ASE Dimer saddle point refinement with optional eigenmode/displacement
    output_dir.mkdir(parents=True, exist_ok=True)
    dimer_atoms = atoms.copy()
    dimer_atoms.calc = calculator_getter()

    neighbour_shell = dimer_atoms.info.get("dimer_neighbor_shells", neighbour_shell)
    if not isinstance(neighbour_shell, (int, np.integer)) or isinstance(neighbour_shell, bool) or not 0 <= neighbour_shell <= 3:
        raise ValueError("dimer_neighbor_shells must be an integer from 0 to 3")
    neighbour_shell = int(neighbour_shell)

    # Constraints only make sense on big nanoclusters (Au55, Au63) where spectator side
    # does not participate. Small nanoclusters (<= 20 metal atoms: Cu20, Au13, Au20) must be unconstrained.
    metal_count = sum(1 for a in dimer_atoms if a.symbol in ("Cu", "Au", "Ag", "Pt", "Pd"))
    if metal_count <= 20:
        neighbour_shell = 0

    if neighbour_shell and not dimer_atoms.constraints:
        try:
            from carmm.analyse.neighbours import neighbours
            adsorbate_indices = [a.index for a in dimer_atoms if a.symbol in ("C", "O", "H", "N", "S")]
            if adsorbate_indices and len(adsorbate_indices) < len(dimer_atoms):
                active_indices, _ = neighbours(dimer_atoms, centre=adsorbate_indices, shell=int(neighbour_shell))
                fixed_indices = [i for i in range(len(dimer_atoms)) if i not in active_indices]
                if fixed_indices:
                    dimer_atoms.set_constraint(FixAtoms(indices=fixed_indices))
                    _append_workflow_log(
                        workflow_log_path,
                        (
                            f"[{label}] carmm.neighbours shell={neighbour_shell} constraint: "
                            f"{len(active_indices)} active atoms, {len(fixed_indices)} spectator atoms frozen with FixAtoms."
                        ),
                    )
        except Exception as e:
            _append_workflow_log(workflow_log_path, f"[{label}] carmm.neighbours constraint skipped: {e}")

    run_state = DimerRunState(
        source_artifact_ids=list(prior_artifact_ids or []),
        staged_seed_file=str(staged_seed_file) if staged_seed_file else None,
    )
    eigenmodes = None
    if initial_eigenmode is not None:
        try:
            mode_arr = np.asarray(initial_eigenmode, dtype=float).reshape((len(atoms), 3))
            norm = float(np.linalg.norm(mode_arr))
            if norm > 1e-6:
                eigenmodes = [mode_arr / norm]
                run_state.last_eigenmode = eigenmodes[0]
        except Exception:
            eigenmodes = None

    fallback_reason: str | None = None

    if use_catlearn:
        sources: list[Any] = []
        if staged_seed_file and Path(staged_seed_file).is_file():
            sources.append(Path(staged_seed_file))
        for s in (prior_trajectories or []):
            if isinstance(s, (str, Path)):
                p = Path(s)
                if p.is_file() and p not in sources:
                    sources.append(p)
            elif s not in sources:
                sources.append(s)
        for cand_p in (
            output_dir / "seed_training.extxyz",
            output_dir.parent / "seed_training.extxyz",
            output_dir / f"{label}.traj",
            output_dir.parent / "input_band.extxyz",
        ):
            if cand_p.is_file() and cand_p not in sources:
                sources.append(cand_p)

        al_dir = output_dir / "catlearn_dimer"
        al_dir.mkdir(parents=True, exist_ok=True)
        trainingset_path = al_dir / "trainingset.traj"

        seeded_frames = seed_dimer_training_set(
            sources,
            dimer_atoms,
            trainingset_path,
            state=run_state,
        )

        try:
            from catlearn.activelearning.local import LocalAL
            can_use_catlearn = True
        except ImportError:
            can_use_catlearn = False

        if run_state.seeded_count < 3:
            fallback_reason = f"insufficient training seeds ({run_state.seeded_count} < 3)"
            run_state.surrogate_status = "skipped_missing_seeds"
            run_state.fallback_reason = fallback_reason
            _append_workflow_log(
                workflow_log_path,
                f"[{label}] Insufficient training seeds ({run_state.seeded_count} < 3); proceeding with direct DFT Dimer.",
            )
        elif not can_use_catlearn:
            fallback_reason = "CatLearn not available in environment"
            run_state.surrogate_status = "skipped_no_catlearn"
            run_state.fallback_reason = fallback_reason
            _append_workflow_log(
                workflow_log_path,
                f"[{label}] CatLearn not available in environment; proceeding with direct DFT Dimer.",
            )
        else:
            pred_traj_path = al_dir / "predicted.traj"
            summary_txt_path = al_dir / "ml_summary.txt"
            can_restart_traj = pred_traj_path.is_file() and pred_traj_path.stat().st_size > 0
            can_restart_state = can_restart_traj and summary_txt_path.is_file() and summary_txt_path.stat().st_size > 0

            _append_workflow_log(
                workflow_log_path,
                (
                    f"[{label}] Starting CatLearn surrogate Dimer (seeded={len(seeded_frames)} frames, "
                    f"restart_traj={can_restart_traj}, restart_state={can_restart_state})."
                ),
            )
            try:
                dyn = LocalAL(
                    atoms=dimer_atoms,
                    ase_calc=dimer_atoms.calc,
                    local_opt=PersistentDimerLocalOpt,
                    local_opt_kwargs={
                        "dimer_separation": 0.005,
                        "use_central_forces": bool(use_central_forces),
                        "initial_eigenmode": run_state.last_eigenmode,
                        "state": run_state,
                    },
                    unc_convergence=float(catlearn_max_unc),
                    min_data=3,
                    use_restart=can_restart_state,
                    restart=can_restart_traj,
                    trajectory=str(pred_traj_path),
                    trainingset=str(trainingset_path),
                    tabletxt=str(summary_txt_path),
                )
                dyn.run(
                    fmax=float(fmax),
                    max_unc=float(catlearn_max_unc),
                    steps=int(max_steps),
                    ml_steps=int(catlearn_ml_steps),
                    dtrust=float(catlearn_dtrust),
                )
                run_state.surrogate_status = "completed"
                if run_state.last_eigenmode is not None:
                    norm = float(np.linalg.norm(run_state.last_eigenmode))
                    if norm > 1e-6:
                        eigenmodes = [run_state.last_eigenmode / norm]
                _append_workflow_log(
                    workflow_log_path,
                    f"[{label}] CatLearn surrogate Dimer finished; starting DFT polish pass (max_steps={catlearn_polish_steps}).",
                )
                max_steps = int(catlearn_polish_steps)
            except Exception as exc:
                fallback_reason = f"{exc.__class__.__name__}: {exc}"
                run_state.surrogate_status = "fallback_direct_dft"
                run_state.fallback_reason = fallback_reason
                _append_workflow_log(
                    workflow_log_path,
                    f"[{label}] CatLearn surrogate Dimer raised {exc.__class__.__name__}: {exc}; proceeding with direct DFT Dimer.",
                )
                if pred_traj_path.is_file() and pred_traj_path.stat().st_size > 0:
                    try:
                        last_pred = read(str(pred_traj_path), index=-1)
                        if len(last_pred) == len(dimer_atoms):
                            dimer_atoms.set_positions(last_pred.positions)
                    except Exception:
                        pass
                if run_state.last_eigenmode is not None:
                    norm = float(np.linalg.norm(run_state.last_eigenmode))
                    if norm > 1e-6:
                        eigenmodes = [run_state.last_eigenmode / norm]

    initial_method = "displacement" if (displacement_vector is not None and eigenmodes is None) else "gauss"
    disp_method = "vector" if (displacement_vector is not None and eigenmodes is None) else "gauss"

    with DimerControl(
        eigenmode_method="dimer",
        f_rot_min=float(f_rot_min),
        f_rot_max=float(f_rot_max),
        max_num_rot=int(max_num_rot),
        trial_angle=float(trial_angle),
        trial_trans_step=float(trial_trans_step),
        maximum_translation=float(maximum_translation),
        cg_translation=True,
        use_central_forces=bool(use_central_forces),
        dimer_separation=float(dimer_separation),
        initial_eigenmode_method=initial_method,
        displacement_method=disp_method,
        logfile=str(output_dir / f"{label}_control.log"),
        eigenmode_logfile=str(output_dir / f"{label}_eigenmode.log"),
    ) as d_control:
        min_mode_atoms = _ReferenceAnchoredMinModeAtoms(dimer_atoms, control=d_control, eigenmodes=eigenmodes)
        if displacement_vector is not None and eigenmodes is None:
            min_mode_atoms.displace(displacement_vector=displacement_vector)
        elif eigenmodes is None:
            min_mode_atoms.displace()

        traj_path = output_dir / f"{label}.traj"
        log_path = output_dir / f"{label}.log"
        _append_workflow_log(
            workflow_log_path,
            (
                f"{label}: starting ASE Dimer refinement "
                f"(fmax={float(fmax):.6f}, steps={int(max_steps)}, "
                f"separation={float(dimer_separation)}, central_forces={bool(use_central_forces)})."
            ),
        )
        with MinModeTranslate(min_mode_atoms, trajectory=str(traj_path), logfile=str(log_path)) as optimizer:
            def _step_callback() -> None:
                try:
                    step = int(optimizer.nsteps)
                    e = float(min_mode_atoms.atoms.get_potential_energy())
                    f = min_mode_atoms.atoms.get_forces()
                    f_norms = np.linalg.norm(f, axis=1)
                    max_f_idx = int(np.argmax(f_norms))
                    max_f = float(f_norms[max_f_idx])
                    curv = float(min_mode_atoms.get_curvature())
                    sym = min_mode_atoms.atoms[max_f_idx].symbol
                    status_tag = "VALLEY/MINIMUM (curv>0)" if curv > 0 else "SADDLE-RIDGE (curv<0)"
                    msg = (
                        f"[{label}] Step {step:3d}: E={e:.6f} eV, "
                        f"fmax={max_f:.4f} eV/Å (atom {max_f_idx}:{sym}), "
                        f"curvature={curv:+.4f} eV/Å² [{status_tag}]"
                    )
                    _append_workflow_log(workflow_log_path, msg)
                    print(msg, flush=True)
                except Exception:
                    pass

            if hasattr(optimizer, "attach"):
                optimizer.attach(_step_callback, interval=1)
            converged = bool(optimizer.run(fmax=float(fmax), steps=int(max_steps)))

        energy_eV = float(min_mode_atoms.atoms.get_potential_energy())
        forces = min_mode_atoms.atoms.get_forces()
        refined_atoms = min_mode_atoms.atoms.copy()
        refined_atoms.calc = SinglePointCalculator(refined_atoms, energy=energy_eV, forces=forces)
        refined_atoms.info["dimer_run_state"] = run_state.to_dict()
        curvature = float(min_mode_atoms.get_curvature())
        final_eigenmode = min_mode_atoms.get_eigenmode().tolist()
        return refined_atoms, energy_eV, curvature, final_eigenmode, converged


def run_active_learning_neb_workflow(
    *,
    initial_atoms: Atoms | None = None,
    final_atoms: Atoms | None = None,
    fmax: float = 0.05,
    optimizer: str = "bfgs",
    final_image_strategy: str = "explicit_final_image",
    optimize_initial_image: bool = False,
    optimize_final_image: bool = False,
    endpoint_opt_steps: int = 200,
    endpoint_opt_fmax: float | None = None,
    break_bond_i: int | None = None,
    break_bond_j: int | None = None,
    set_bond_distance_A: float | None = None,
    break_bond_extension_A: float | None = None,  # legacy alias
    break_bond_fix: float = 0.5,
    break_assert_broken: bool = True,
    break_cutoff_scale: float = 1.2,
    backend: str = "mace",
    atom_shift_per_image_A: float = 0.2,
    min_images: int = 7,
    max_images: int = 21,
    spring_constant: float = 0.1,
    max_steps: int = 300,
    climb: bool = True,
    split_on_minimum: bool = True,
    minimum_energy_window_eV: float = 0.02,
    fixed_symbols: tuple[str, ...] = (),
    mace_model_path: str = "",
    mace_device: str = "cuda",
    mace_default_dtype: str = "float64",
    mace_dispersion: bool = False,
    basis_keyword: str = "def2-svp",
    xc_functional: str = "pbe0",
    solvent: str | None = None,
    spin_multiplicity: int | None = None,
    nodes: int = 1,
    cpus_per_node: int = 1,
    dft_mem: int = 4000,
    use_regular_endpoint_optimisation_for_mace: bool = True,
    share_calculator_instance: bool | None = None,
    catlearn_min_data: int = 3,
    catlearn_unc_convergence: float = 0.05,
    catlearn_max_unc: float = 0.03,
    catlearn_steps: int = 200,
    catlearn_ml_steps: int = 500,
    catlearn_dtrust: float = 2.0,
    allow_optimizer_fallback: bool = False,
    output_dir: Path | str = Path("neb"),
    initial_band_atoms: list[Atoms] | None = None,
    catlearn_guesses_per_cycle: int | None = 10,
    restart_from: Path | str | None = None,
    dft_evaluation_timeout_seconds: int = 21600,
    dft_min_distance_scale: float = 0.55,
    dft_max_force_eV_A: float = 50.0,
    dft_max_energy_drop_eV: float = 20.0,
    evaluation_only: bool = False,
    refine_ts_with_dimer: bool = False,
    dimer_max_steps: int = 100,
    dimer_separation: float = 0.0001,
    dimer_trial_trans_step: float = 0.001,
    dimer_maximum_translation: float = 0.05,
    dimer_f_rot_min: float = 0.1,
    dimer_f_rot_max: float = 1.0,
    dimer_max_num_rot: int = 3,
    dimer_trial_angle: float = math.pi / 4.0,
    dimer_use_central_forces: bool = True,
    initial_eigenmode: Any = None,
    displacement_vector: Any = None,
    use_catlearn_dimer: bool = False,
    catlearn_dimer_polish_steps: int = 25,
    prior_trajectories: Any = None,
    staged_seed_file: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    # ponytail: auto-load band / endpoints from restart_from if provided and initial_band_atoms not given
    if restart_from is not None and initial_band_atoms is None:
        rf = Path(restart_from).resolve()
        rf_dir = rf if rf.is_dir() else rf.parent
        for cand_name in ("input_band.extxyz", "final_band.extxyz", "mep_primary.extxyz", "predicted.traj"):
            cand = rf_dir / cand_name if (rf_dir / cand_name).exists() else (rf_dir / "catlearn_mlneb" / cand_name if (rf_dir / "catlearn_mlneb" / cand_name).exists() else None)
            if cand is not None and cand.exists():
                try:
                    c_frames = list(read(str(cand), ":"))
                    if c_frames and len(c_frames) >= 2:
                        if cand_name == "predicted.traj" and len(c_frames) > int(max_images):
                            stage = cand.parent
                            if not (stage / "initial.traj").exists() or not (stage / "final.traj").exists():
                                raise ValueError("Cannot identify a complete final MEP in a concatenated predicted trajectory.")
                            first = read(stage / "initial.traj").positions
                            last = read(stage / "final.traj").positions
                            starts = [i for i, frame in enumerate(c_frames) if np.allclose(frame.positions, first, atol=1.0e-6)]
                            for i in reversed(starts):
                                j = next(
                                    (j for j in range(i + 1, len(c_frames)) if np.allclose(c_frames[j].positions, last, atol=1.0e-6)),
                                    None,
                                )
                                if j is not None:
                                    c_frames = c_frames[i:j + 1]
                                    break
                            else:
                                raise ValueError("No complete MEP was found in the predicted trajectory.")
                        initial_band_atoms = c_frames
                        break
                except ValueError:
                    raise
                except Exception:
                    pass

    strategy = _normalize_final_image_strategy(final_image_strategy)
    backend_key = str(backend).strip().lower()
    initial_training_atoms: list[Atoms] = []
    if initial_band_atoms is not None:
        if len(initial_band_atoms) < 2:
            raise ValueError("Initial DFT MLNEB band requires at least two images.")
        cleaned_band: list[Atoms] = []
        expected_numbers = np.asarray(
            (initial_atoms if initial_atoms is not None else initial_band_atoms[0]).get_atomic_numbers(),
            dtype=int,
        )
        for index, source in enumerate(initial_band_atoms):
            try:
                energy = float(source.get_potential_energy())
                forces = np.asarray(source.get_forces(), dtype=float)
                if (
                    math.isfinite(energy)
                    and forces.shape == (len(source), 3)
                    and np.isfinite(forces).all()
                ):
                    evaluated = source.copy()
                    evaluated.calc = SinglePointCalculator(evaluated, energy=energy, forces=forces)
                    initial_training_atoms.append(evaluated)
            except Exception:
                pass
            image = source.copy()
            image.calc = None
            image.set_constraint()
            if not np.array_equal(np.asarray(image.get_atomic_numbers(), dtype=int), expected_numbers):
                raise ValueError(f"Initial DFT MLNEB band atom order differs at image {index}.")
            cleaned_band.append(image)
        initial_band_atoms = cleaned_band
        if strategy not in {"dimer", "dimer_refine"}:
            initial_atoms = cleaned_band[0]
            final_atoms = cleaned_band[-1]
            strategy = "explicit_final_image"
    use_regular_endpoint_opt = bool(
        bool(use_regular_endpoint_optimisation_for_mace) and backend_key in {"mace", "dft"}
    )
    if strategy == "explicit_final_image" and not evaluation_only:
        if final_atoms is None:
            raise ValueError("Explicit NEB final-image mode requires final_atoms.")
        _validate_endpoints(initial_atoms, final_atoms)

    run_dir = Path(output_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    workflow_log_path = run_dir / "active_learning_neb.log"
    workflow_log_path.write_text("", encoding="utf-8")
    _append_workflow_log(
        workflow_log_path,
        (
            "Workflow start: active-learning NEB "
            f"(backend={str(backend).strip().lower()}, "
            f"allow_optimizer_fallback={bool(allow_optimizer_fallback)})."
        ),
    )

    endpoint_fmax = float(fmax)
    if endpoint_opt_fmax is not None:
        endpoint_fmax = float(endpoint_opt_fmax)
    if endpoint_fmax <= 0.0:
        endpoint_fmax = float(fmax)
    endpoint_steps = max(1, int(endpoint_opt_steps))

    final_constraint_source = final_atoms if final_atoms is not None else initial_atoms
    merged_constraints = _merged_constraints(
        initial_atoms=initial_atoms,
        final_atoms=final_constraint_source,
        fixed_symbols=tuple(fixed_symbols or ()),
    )
    factory = build_calculator_factory(
        CalculatorFactoryConfig(
            backend=backend,
            mace_model_path=mace_model_path,
            mace_device=mace_device,
            mace_default_dtype=mace_default_dtype,
            mace_dispersion=mace_dispersion,
            dft_cpus=max(1, int(nodes) * int(cpus_per_node)),
            dft_mem=int(dft_mem),
            basis_keyword=basis_keyword,
            xc_functional=xc_functional,
            solvent=solvent,
            spin_multiplicity=spin_multiplicity,
            dft_evaluation_timeout_seconds=int(dft_evaluation_timeout_seconds),
            dft_min_distance_scale=float(dft_min_distance_scale),
            dft_max_force_eV_A=float(dft_max_force_eV_A),
            dft_max_energy_drop_eV=float(dft_max_energy_drop_eV),
            target_wavelength_nm=float(kwargs.get("target_wavelength_nm", 520.0)),
            tddft_scan_nroots=int(kwargs.get("tddft_scan_nroots", 50)),
            initial_iroot=int(kwargs.get("initial_iroot", kwargs.get("root_state", 5))),
            system_charge=int(kwargs.get("system_charge", 0)),
            fragments=kwargs.get("fragments"),
        )
    )
    use_shared_calculator = _resolve_share_calculator_instance(
        backend=backend,
        share_calculator_instance=share_calculator_instance,
    )
    calculator_getter = _calculator_getter_from_factory(
        factory=factory,
        share_calculator_instance=use_shared_calculator,
    )
    start = _with_constraints(initial_atoms, merged_constraints)
    if "dimer_neighbor_shells" in kwargs and kwargs["dimer_neighbor_shells"] is not None:
        start.info["dimer_neighbor_shells"] = int(kwargs["dimer_neighbor_shells"])
    if "vibration_neighbor_shells" in kwargs and kwargs["vibration_neighbor_shells"] is not None:
        start.info["vibration_neighbor_shells"] = int(kwargs["vibration_neighbor_shells"])
    scan_pair = start.info.get("relaxed_scan_pair")
    if scan_pair is not None:
        if backend_key != "dft" or merged_constraints or not initial_band_atoms or not 2 <= len(initial_band_atoms) <= 64:
            raise ValueError("Relaxed bond scan requires DFT, 2-64 target frames and no frozen atoms")
        pair = np.asarray(scan_pair)
        if pair.shape != (2,) or pair.dtype.kind not in "iu" or len(set(pair.tolist())) != 2 or pair.min() < 0 or pair.max() >= len(start):
            raise ValueError("relaxed_scan_pair must contain two distinct valid atom indices")
        pair = [int(i) for i in pair]
        targets = []
        for image in initial_band_atoms:
            if not np.array_equal(image.info.get("relaxed_scan_pair"), pair):
                raise ValueError("Relaxed scan bond indices differ between target frames")
            target = float(image.info.get("relaxed_scan_target_A", float("nan")))
            if not math.isfinite(target) or not 0.5 <= target <= 6.0:
                raise ValueError("Relaxed scan targets must be finite distances from 0.5 to 6.0 Angstrom")
            targets.append(target)
        if not (np.all(np.diff(targets) > 0) or np.all(np.diff(targets) < 0)):
            raise ValueError("Relaxed scan targets must be strictly ordered")
        rows, frames = [], []
        scan_path = run_dir / "relaxed_scan.extxyz"
        for index, target in enumerate(targets):
            candidate = start.copy()
            candidate.set_constraint()
            candidate.set_distance(*pair, target, fix=0.5)
            candidate.set_constraint(FixBondLength(*pair))
            candidate, method = _optimise_structure(
                atoms=candidate, calculator_getter=calculator_getter, optimizer=optimizer,
                fmax=fmax, steps=max_steps, catlearn_min_data=catlearn_min_data,
                catlearn_unc_convergence=catlearn_unc_convergence, catlearn_max_unc=catlearn_max_unc,
                catlearn_ml_steps=catlearn_ml_steps, catlearn_dtrust=catlearn_dtrust,
                use_regular_optimizer=True, allow_optimizer_fallback=False,
                workflow_log_path=workflow_log_path, output_dir=run_dir, label=f"scan_{index:02d}",
            )
            residual = float(np.linalg.norm(candidate.get_forces(), axis=1).max())
            distance = float(candidate.get_distance(*pair))
            energy = float(candidate.get_potential_energy())
            row = {"image": index, "target_distance_A": target, "distance_A": distance,
                   "energy_eV": energy, "constrained_fmax_eV_A": residual,
                   "converged": residual <= float(fmax) and abs(distance - target) < 1e-6,
                   "method": method}
            stored = candidate.copy()
            stored.set_constraint()
            stored.info.update(relaxed_scan_pair=pair, relaxed_scan_target_A=target)
            stored.calc = SinglePointCalculator(stored, energy=energy, forces=candidate.get_forces(apply_constraint=False))
            rows.append(row)
            frames.append(stored)
            write(scan_path, frames, format="extxyz")
            (run_dir / "relaxed_scan_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            _append_workflow_log(workflow_log_path, f"Relaxed scan point {index}: {row}")
            if not row["converged"]:
                raise RuntimeError(f"Constrained scan point {index} did not converge; checkpoint retained")
            start = candidate
        crest = int(np.argmax([row["energy_eV"] for row in rows]))
        write(run_dir / "transition_state.xyz", frames[crest], format="xyz")
        write(run_dir / "reaction_hub_final.xyz", frames[-1], format="xyz")
        summary = {"mode": "relaxed_bond_scan", "backend": "dft", "converged": True,
                   "image_count": len(rows), "bond_pair_indices_0based": pair, "points": rows,
                   "scan_path": str(scan_path), "transition_state_image_index": crest,
                   "crest_is_interior": 0 < crest < len(rows) - 1,
                   "transition_state_path": str(run_dir / "transition_state.xyz"),
                   "final_xyz_path": str(run_dir / "reaction_hub_final.xyz"),
                   "workflow_log_path": str(workflow_log_path), "output_dir": str(run_dir),
                   "scope": "constrained scan; crest is not an unconstrained transition state"}
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
    if evaluation_only:
        start.calc = calculator_getter()
        forces = np.asarray(start.get_forces(), dtype=float)
        energy = float(start.get_potential_energy())
        evaluated = start.copy()
        evaluated.calc = SinglePointCalculator(evaluated, energy=energy, forces=forces)
        evaluated_path = run_dir / "evaluated.traj"
        final_xyz_path = run_dir / "reaction_hub_final.xyz"
        write(evaluated_path, evaluated)
        write(final_xyz_path, evaluated, format="xyz")
        summary = {
            "mode": "dft_image_evaluation",
            "backend": "dft",
            "converged": True,
            "energy_eV": energy,
            "max_force_eV_A": _max_force_from_atoms(evaluated),
            "evaluated_path": str(evaluated_path),
            "final_xyz_path": str(final_xyz_path),
            "workflow_log_path": str(workflow_log_path),
        }
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary
    if strategy in {"dimer", "dimer_refine"}:
        dimer_dir = run_dir / "dimer"
        dimer_dir.mkdir(parents=True, exist_ok=True)
        refined_ts_path = run_dir / "transition_state_refined.xyz"
        summary_path = run_dir / "summary.json"
        dimer_resumed = False
        if refined_ts_path.is_file() and summary_path.is_file():
            try:
                prev_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if prev_summary.get("converged") and prev_summary.get("mode") == "dimer":
                    refined_atoms = read(refined_ts_path)
                    energy_eV = float(prev_summary.get("energy_eV", 0.0))
                    curvature = float(prev_summary.get("curvature", 0.0))
                    converged = True
                    dimer_resumed = True
                    final_mode = prev_summary.get("final_mode")
                    _append_workflow_log(
                        workflow_log_path,
                        f"Dimer TS refinement already converged in previous run (energy={energy_eV:.6f} eV, curvature={curvature:.6f}). Skipping dimer search and resuming vibrations.",
                    )
            except Exception:
                dimer_resumed = False
        if not dimer_resumed:
            dimer_mode = None
            if initial_eigenmode is not None:
                dimer_mode = initial_eigenmode
            elif initial_band_atoms is not None and len(initial_band_atoms) >= 3:
                mid = len(initial_band_atoms) // 2
                tangent = initial_band_atoms[min(len(initial_band_atoms) - 1, mid + 1)].positions - initial_band_atoms[max(0, mid - 1)].positions
                t_norm = float(np.linalg.norm(tangent))
                if t_norm > 1e-6:
                    dimer_mode = tangent / t_norm
            priors = list(prior_trajectories or [])
            if restart_from is not None:
                rf = Path(restart_from).resolve()
                rf_dir = rf if rf.is_dir() else rf.parent
                for cand in (rf_dir / "dimer" / "dimer.traj", rf_dir / "dimer.traj", rf_dir / "input_band.extxyz"):
                    if cand.exists() and cand not in priors:
                        priors.append(cand)

            refined_atoms, energy_eV, curvature, final_mode, converged = _run_dimer_refine_pass(
                atoms=start,
                calculator_getter=calculator_getter,
                fmax=fmax,
                max_steps=dimer_max_steps if dimer_max_steps else max_steps,
                dimer_separation=dimer_separation,
                trial_trans_step=dimer_trial_trans_step,
                maximum_translation=dimer_maximum_translation,
                f_rot_min=dimer_f_rot_min,
                f_rot_max=dimer_f_rot_max,
                max_num_rot=dimer_max_num_rot,
                trial_angle=dimer_trial_angle,
                use_central_forces=dimer_use_central_forces,
                initial_eigenmode=dimer_mode,
                displacement_vector=displacement_vector,
                workflow_log_path=workflow_log_path,
                output_dir=dimer_dir,
                label="dimer",
                use_catlearn=bool(use_catlearn_dimer or kwargs.get("use_catlearn_dimer", False)),
                catlearn_polish_steps=int(catlearn_dimer_polish_steps if catlearn_dimer_polish_steps is not None else kwargs.get("catlearn_dimer_polish_steps", 25)),
                prior_trajectories=priors if priors else None,
                prior_artifact_ids=kwargs.get("prior_artifact_ids"),
                staged_seed_file=staged_seed_file or kwargs.get("staged_seed_file"),
            )
            write(run_dir / "transition_state.xyz", refined_atoms, format="xyz")
            write(run_dir / "transition_state_refined.xyz", refined_atoms, format="xyz")
        final_xyz_path = run_dir / "reaction_hub_final.xyz"
        write(final_xyz_path, refined_atoms, format="xyz")
        summary = {
            "mode": "dimer",
            "backend": backend_key,
            "converged": bool(converged),
            "fmax_eV_A": float(fmax),
            "refined_max_force_eV_A": _max_force_from_atoms(refined_atoms),
            "energy_eV": float(energy_eV),
            "curvature": float(curvature),
            "transition_state_path": str(run_dir / "transition_state.xyz"),
            "final_xyz_path": str(final_xyz_path),
            "workflow_log_path": str(workflow_log_path),
            "output_dir": str(run_dir),
            "dimer_refine_audit": refined_atoms.info.get("dimer_run_state"),
            "fallback_reason": (refined_atoms.info.get("dimer_run_state") or {}).get("fallback_reason"),
        }
        _append_workflow_log(
            workflow_log_path,
            f"Workflow complete: Dimer TS refinement converged={converged}, energy_eV={energy_eV:.6f}, curvature={curvature:.6f}.",
        )
        vibration_shells = start.info.get("vibration_neighbor_shells")
        if vibration_shells is not None:
            if not isinstance(vibration_shells, (int, np.integer)) or isinstance(vibration_shells, bool) or not 0 <= vibration_shells <= 3:
                raise ValueError("vibration_neighbor_shells must be an integer from 0 to 3 (0 = all atoms)")
            vibration_shells = int(vibration_shells)
            summary["vibrations_requested_neighbor_shells"] = vibration_shells
            if converged:
                (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                vibration_atoms = refined_atoms.copy()
                vibration_atoms.calc = calculator_getter()
                summary["vibrations"] = _run_local_ts_vibrations(
                    vibration_atoms, shells=vibration_shells, output_dir=run_dir / "vibrations"
                )
            else:
                summary["vibrations_status"] = "skipped_unconverged_dimer"
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary
    endpoint_initial_opt_method = "not_requested"
    if bool(optimize_initial_image):
        start, endpoint_initial_opt_method = _optimise_structure(
            atoms=start,
            calculator_getter=calculator_getter,
            optimizer=optimizer,
            fmax=endpoint_fmax,
            steps=endpoint_steps,
            catlearn_min_data=catlearn_min_data,
            catlearn_unc_convergence=catlearn_unc_convergence,
            catlearn_max_unc=catlearn_max_unc,
            catlearn_ml_steps=catlearn_ml_steps,
            catlearn_dtrust=catlearn_dtrust,
            use_regular_optimizer=use_regular_endpoint_opt,
            allow_optimizer_fallback=allow_optimizer_fallback,
            workflow_log_path=workflow_log_path,
            output_dir=run_dir,
            label="endpoint_initial_opt",
        )
        start = _with_constraints(start, merged_constraints)

    bond_break_record: dict[str, Any] | None = None
    if strategy == "explicit_final_image":
        if final_atoms is None:
            raise ValueError("Explicit NEB final-image mode requires final_atoms.")
        end = _with_constraints(final_atoms, merged_constraints)
    else:
        target_distance_A = (
            float(set_bond_distance_A)
            if set_bond_distance_A is not None
            else float(break_bond_extension_A) if break_bond_extension_A is not None else 2.5
        )
        end, bond_i, bond_j, distance_initial, distance_target = _generate_bond_extension_final(
            initial_atoms=start,
            bond_i=break_bond_i,
            bond_j=break_bond_j,
            target_distance_A=target_distance_A,
            fix=break_bond_fix,
        )
        end = _with_constraints(end, merged_constraints)
        bond_break_record = {
            "bond_i": int(bond_i),
            "bond_j": int(bond_j),
            "distance_initial_A": float(distance_initial),
            "distance_target_A": float(distance_target),
            "set_distance_fix": float(break_bond_fix),
        }
    endpoint_final_opt_method = "not_requested"
    if bool(optimize_final_image):
        end, endpoint_final_opt_method = _optimise_structure(
            atoms=end,
            calculator_getter=calculator_getter,
            optimizer=optimizer,
            fmax=endpoint_fmax,
            steps=endpoint_steps,
            catlearn_min_data=catlearn_min_data,
            catlearn_unc_convergence=catlearn_unc_convergence,
            catlearn_max_unc=catlearn_max_unc,
            catlearn_ml_steps=catlearn_ml_steps,
            catlearn_dtrust=catlearn_dtrust,
            use_regular_optimizer=use_regular_endpoint_opt,
            allow_optimizer_fallback=allow_optimizer_fallback,
            workflow_log_path=workflow_log_path,
            output_dir=run_dir,
            label="endpoint_final_opt",
        )
        end = _with_constraints(end, merged_constraints)

    _validate_endpoints(start, end)
    if strategy == "set_bond_distance_from_initial" and bond_break_record is not None:
        distance, cutoff = _bond_distance_and_cutoff(
            atoms=end,
            bond_i=int(bond_break_record["bond_i"]),
            bond_j=int(bond_break_record["bond_j"]),
            cutoff_scale=float(break_cutoff_scale),
        )
        still_bonded = bool(distance <= cutoff)
        bond_break_record["distance_after_final_prep_A"] = float(distance)
        bond_break_record["bond_cutoff_A"] = float(cutoff)
        bond_break_record["bond_present_after_final_prep"] = still_bonded
        bond_break_record["assert_broken"] = bool(break_assert_broken)
        if bool(break_assert_broken) and still_bonded:
            raise RuntimeError(
                "Final-image bond-absence assertion failed: "
                f"distance={distance:.4f} A <= cutoff={cutoff:.4f} A "
                f"for bond ({int(bond_break_record['bond_i'])}, {int(bond_break_record['bond_j'])})."
            )

    max_shift_A = _max_atom_shift_A(start, end)
    if initial_band_atoms is not None:
        prepared_band = [_with_constraints(image, merged_constraints) for image in initial_band_atoms]
        prepared_band[0] = start.copy()
        prepared_band[-1] = end.copy()
        initial_band_atoms = prepared_band
        n_images = len(prepared_band)
    else:
        n_images = _image_count_from_shift(
            max_shift_A=max_shift_A,
            atom_shift_per_image_A=atom_shift_per_image_A,
            min_images=min_images,
            max_images=max_images,
        )
    _append_workflow_log(
        workflow_log_path,
        (
            "Primary path prepared: "
            f"max_atom_shift_A={float(max_shift_A):.6f}, image_count={int(n_images)}."
        ),
    )

    if backend_key == "dft":
        images, energies, converged, evaluated_count = _run_catlearn_mlneb_pass(
            start=start,
            end=end,
            calculator_getter=calculator_getter,
            n_images=n_images,
            fmax=fmax,
            max_steps=max_steps,
            spring_constant=spring_constant,
            catlearn_unc_convergence=catlearn_unc_convergence,
            catlearn_max_unc=catlearn_max_unc,
            catlearn_ml_steps=catlearn_ml_steps,
            catlearn_dtrust=catlearn_dtrust,
            workflow_log_path=workflow_log_path,
            output_dir=run_dir,
            initial_band_atoms=initial_band_atoms,
            initial_training_atoms=initial_training_atoms,
            catlearn_guesses_per_cycle=catlearn_guesses_per_cycle,
            restart_from=restart_from,
        )
        finite_indices = [idx for idx, value in enumerate(energies) if math.isfinite(value)]
        ts_index = max(finite_indices, key=lambda idx: energies[idx]) if finite_indices else 0
        final_xyz_path = run_dir / "reaction_hub_final.xyz"
        write(run_dir / "mep_primary.extxyz", images, format="extxyz")
        write(run_dir / "final_band.extxyz", images, format="extxyz")
        write(run_dir / "transition_state.xyz", images[ts_index], format="xyz")
        if bool(refine_ts_with_dimer):
            _append_workflow_log(workflow_log_path, f"Refining TS candidate (image {ts_index}) using ASE Dimer method...")
            dimer_mode = None
            if len(images) > 2 and 0 < ts_index < len(images) - 1:
                tangent = images[ts_index + 1].positions - images[ts_index - 1].positions
                t_norm = float(np.linalg.norm(tangent))
                if t_norm > 1e-6:
                    dimer_mode = tangent / t_norm
            refined_ts, dimer_energy, dimer_curv, dimer_mode_out, dimer_conv = _run_dimer_refine_pass(
                atoms=images[ts_index],
                calculator_getter=calculator_getter,
                fmax=fmax,
                max_steps=dimer_max_steps,
                dimer_separation=dimer_separation,
                trial_trans_step=dimer_trial_trans_step,
                maximum_translation=dimer_maximum_translation,
                f_rot_min=dimer_f_rot_min,
                f_rot_max=dimer_f_rot_max,
                max_num_rot=dimer_max_num_rot,
                trial_angle=dimer_trial_angle,
                use_central_forces=dimer_use_central_forces,
                initial_eigenmode=dimer_mode,
                displacement_vector=displacement_vector,
                workflow_log_path=workflow_log_path,
                output_dir=run_dir / "dimer",
                label="dimer_ts_refine",
                use_catlearn=bool(use_catlearn_dimer or kwargs.get("use_catlearn_dimer", False)),
                catlearn_polish_steps=int(catlearn_dimer_polish_steps if catlearn_dimer_polish_steps is not None else kwargs.get("catlearn_dimer_polish_steps", 25)),
                prior_trajectories=prior_trajectories,
                prior_artifact_ids=kwargs.get("prior_artifact_ids"),
                staged_seed_file=staged_seed_file or kwargs.get("staged_seed_file"),
            )
            images[ts_index] = refined_ts
            write(run_dir / "transition_state.xyz", refined_ts, format="xyz")
            write(run_dir / "transition_state_refined.xyz", refined_ts, format="xyz")
            energies[ts_index] = dimer_energy
        write(run_dir / "initial_image.xyz", images[0], format="xyz")
        write(run_dir / "final_image.xyz", images[-1], format="xyz")
        write(final_xyz_path, images[-1], format="xyz")
        summary = {
            "mode": "active_learning_neb",
            "backend": "dft",
            "mep_primary_optimisation_method": "catlearn_mlneb",
            "converged": bool(converged),
            "catlearn_evaluations": int(evaluated_count),
            "fmax_eV_A": float(fmax),
            "image_count": int(len(images)),
            "max_atom_shift_A": float(max_shift_A),
            "energies_eV": [value if math.isfinite(value) else None for value in energies],
            "transition_state_image_index": int(ts_index),
            "transition_state_path": str(run_dir / "transition_state.xyz"),
            "mep_path": str(run_dir / "mep_primary.extxyz"),
            "final_image_strategy": strategy,
            "optimize_initial_image": bool(optimize_initial_image),
            "optimize_final_image": bool(optimize_final_image),
            "endpoint_initial_opt_method": endpoint_initial_opt_method,
            "endpoint_final_opt_method": endpoint_final_opt_method,
            "output_dir": str(run_dir),
            "final_xyz_path": str(final_xyz_path),
            "workflow_log_path": str(workflow_log_path),
        }
        _append_workflow_log(
            workflow_log_path,
            f"Workflow complete: CatLearn MLNEB TS image={ts_index}.",
        )
        (run_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        return summary

    images = [_copy_with_constraint_set(start, merged_constraints)]
    for _ in range(max(0, n_images - 2)):
        images.append(_copy_with_constraint_set(start, merged_constraints))
    images.append(_copy_with_constraint_set(end, merged_constraints))

    energies, mep_primary_optimisation_method = _run_neb_pass(
        images=images,
        calculator_getter=calculator_getter,
        share_calculator_instance=use_shared_calculator,
        fmax=fmax,
        max_steps=max_steps,
        spring_constant=spring_constant,
        climb=climb,
        optimizer=optimizer,
        workflow_log_path=workflow_log_path,
        output_dir=run_dir,
        label="mep_primary",
    )
    write(run_dir / "mep_primary.extxyz", images, format="extxyz")

    minima_indices = _local_minima_indices(energies, minimum_energy_window_eV=minimum_energy_window_eV)
    _append_workflow_log(
        workflow_log_path,
        f"Primary MEP minima candidates detected: {len(minima_indices)}.",
    )
    refined_minima: dict[int, Atoms] = {}
    minima_records: list[dict[str, Any]] = []
    for idx in minima_indices:
        minimum_label = f"minimum_candidate_{idx:03d}_refine"
        refined, method = _catlearn_refine_structure(
            atoms=images[idx],
            calculator_getter=calculator_getter,
            optimizer=optimizer,
            fmax=fmax,
            catlearn_min_data=catlearn_min_data,
            catlearn_unc_convergence=catlearn_unc_convergence,
            catlearn_max_unc=catlearn_max_unc,
            catlearn_steps=catlearn_steps,
            catlearn_ml_steps=catlearn_ml_steps,
            catlearn_dtrust=catlearn_dtrust,
            allow_optimizer_fallback=allow_optimizer_fallback,
            workflow_log_path=workflow_log_path,
            output_dir=run_dir,
            label=minimum_label,
        )
        refined_force = _max_force_from_atoms(refined)
        refined_energy = None
        try:
            refined_energy = float(refined.get_potential_energy())
        except Exception:
            refined_energy = None
        is_verified = bool(refined_force is not None and refined_force <= float(fmax))
        if is_verified:
            refined_minima[idx] = refined
        minima_records.append(
            {
                "index": int(idx),
                "energy_eV": float(energies[idx]),
                "refined_energy_eV": refined_energy,
                "refined_max_force_eV_A": refined_force,
                "verified_minimum": is_verified,
                "refine_method": method,
            }
        )
        write(run_dir / f"minimum_candidate_{idx:03d}.xyz", refined, format="xyz")

    segment_records: list[dict[str, Any]] = []
    if bool(split_on_minimum) and refined_minima:
        boundaries = [0, *sorted(refined_minima.keys()), len(images) - 1]
        unique_boundaries: list[int] = []
        seen: set[int] = set()
        for value in boundaries:
            if value in seen:
                continue
            seen.add(value)
            unique_boundaries.append(value)

        for segment_idx in range(len(unique_boundaries) - 1):
            left = unique_boundaries[segment_idx]
            right = unique_boundaries[segment_idx + 1]
            if right - left < 2:
                continue
            start_image = refined_minima.get(left, images[left])
            end_image = refined_minima.get(right, images[right])
            segment_shift_A = _max_atom_shift_A(start_image, end_image)
            segment_image_count = _image_count_from_shift(
                max_shift_A=segment_shift_A,
                atom_shift_per_image_A=atom_shift_per_image_A,
                min_images=min_images,
                max_images=max_images,
            )
            segment_images = [_copy_with_constraint_set(start_image, merged_constraints)]
            for _ in range(max(0, segment_image_count - 2)):
                segment_images.append(_copy_with_constraint_set(start_image, merged_constraints))
            segment_images.append(_copy_with_constraint_set(end_image, merged_constraints))
            label = f"segment_{segment_idx:02d}_{left:03d}_{right:03d}"
            segment_energies, segment_optimisation_method = _run_neb_pass(
                images=segment_images,
                calculator_getter=calculator_getter,
                share_calculator_instance=use_shared_calculator,
                fmax=fmax,
                max_steps=max_steps,
                spring_constant=spring_constant,
                climb=climb,
                optimizer=optimizer,
                workflow_log_path=workflow_log_path,
                output_dir=run_dir,
                label=label,
            )

            ts_index_local = int(np.argmax(np.asarray(segment_energies, dtype=float)))
            ts_label = f"{label}_ts_refine"
            ts_atoms_refined, ts_method = _catlearn_refine_structure(
                atoms=segment_images[ts_index_local],
                calculator_getter=calculator_getter,
                optimizer=optimizer,
                fmax=fmax,
                catlearn_min_data=catlearn_min_data,
                catlearn_unc_convergence=catlearn_unc_convergence,
                catlearn_max_unc=catlearn_max_unc,
                catlearn_steps=catlearn_steps,
                catlearn_ml_steps=catlearn_ml_steps,
                catlearn_dtrust=catlearn_dtrust,
                allow_optimizer_fallback=allow_optimizer_fallback,
                workflow_log_path=workflow_log_path,
                output_dir=run_dir,
                label=ts_label,
            )
            write(run_dir / f"{label}_ts_refined.xyz", ts_atoms_refined, format="xyz")
            segment_records.append(
                {
                    "segment_index": int(segment_idx),
                    "left_primary_index": int(left),
                    "right_primary_index": int(right),
                    "image_count": int(len(segment_images)),
                    "energies_eV": [float(value) for value in segment_energies],
                    "optimisation_method": segment_optimisation_method,
                    "ts_image_local_index": int(ts_index_local),
                    "ts_refine_method": ts_method,
                    "ts_refined_max_force_eV_A": _max_force_from_atoms(ts_atoms_refined),
                }
            )

    final_xyz_path = run_dir / "reaction_hub_final.xyz"
    write(final_xyz_path, images[-1], format="xyz")
    write(run_dir / "initial_image.xyz", images[0], format="xyz")
    write(run_dir / "final_image.xyz", images[-1], format="xyz")

    summary = {
        "mode": "active_learning_neb",
        "backend": str(backend).strip().lower(),
        "use_regular_endpoint_optimisation_for_mace": bool(use_regular_endpoint_optimisation_for_mace),
        "use_regular_endpoint_optimisation_effective": bool(use_regular_endpoint_opt),
        "share_calculator_instance": bool(use_shared_calculator),
        "allow_optimizer_fallback": bool(allow_optimizer_fallback),
        "workflow_log_path": str(workflow_log_path),
        "catlearn_stage_root": str(run_dir / "catlearn_stages"),
        "optimizer": _normalize_optimizer_name(optimizer),
        "final_image_strategy": strategy,
        "optimize_initial_image": bool(optimize_initial_image),
        "optimize_final_image": bool(optimize_final_image),
        "endpoint_initial_opt_method": endpoint_initial_opt_method,
        "endpoint_final_opt_method": endpoint_final_opt_method,
        "endpoint_opt_fmax_eV_A": float(endpoint_fmax),
        "endpoint_opt_steps": int(endpoint_steps),
        "mep_primary_optimisation_method": mep_primary_optimisation_method,
        "fmax_eV_A": float(fmax),
        "neb_target_image_count": int(len(images)),
        "atom_shift_per_image_A": float(atom_shift_per_image_A),
        "max_atom_shift_A": float(max_shift_A),
        "image_count": int(len(images)),
        "energies_eV": [float(value) for value in energies],
        "minimum_candidates": minima_records,
        "segments": segment_records,
        "bond_break": bond_break_record or {},
        "output_dir": str(run_dir),
        "final_xyz_path": str(final_xyz_path),
    }
    _append_workflow_log(
        workflow_log_path,
        (
            "Workflow complete: "
            f"mep_method={mep_primary_optimisation_method}, "
            f"minima_candidates={len(minima_records)}, segments={len(segment_records)}."
        ),
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary

def run_dimer_refinement_workflow(*args: Any, **kwargs: Any) -> dict[str, Any]:
    kwargs["final_image_strategy"] = "dimer"
    return run_active_learning_neb_workflow(*args, **kwargs)
