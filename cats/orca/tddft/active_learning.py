from __future__ import annotations

import copy
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
import re
import shutil
from typing import Optional, Sequence

from ase.calculators.calculator import Calculator, all_changes
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS, FIRE2

from cats.orca.run.orca_property import get_charges_from_orca_output

from .io import (
    copy_initial_gbw,
    is_valid_trainingset,
    is_valid_trajectory,
    select_gbw_source,
    validate_orca_output,
)
from cats.multiwfn.iroot_scoring import IRootCandidate
from cats.multiwfn.state_tracker import (
    StateTracker,
    StateTrackPoint,
    generate_state_cubes,
)

from .ct import (
    extract_state_transition_vectors,
    record_geometry_choice,
    pick_iroot,
    get_ct_info,
)


AL_PREDICTION_MAX_UNC = 0.03
AL_MIN_DATA = 3
AL_LOCAL_OPT_KWARGS = {"maxstep": 0.02}
AL_ML_STEPS = 1000
AL_DTRUST = 2.0
AL_RECOVERY_MAX_RETRIES = 3
AL_RECOVERY_DTRUST_SCALE = 0.5
AL_RECOVERY_MIN_DTRUST = 0.25
AL_RECOVERY_MAXSTEP_SCALE = 0.5
AL_RECOVERY_MIN_MAXSTEP = 5.0e-4
AL_RECOVERY_TRUE_FMAX_SANITY_LIMIT = 10.0
AL_RECOVERY_RESEED_MAX_FMAX = 1.0
AL_PLATEAU_WINDOW = 10
AL_PLATEAU_MAX_ENERGY_RANGE = 0.002
AL_PLATEAU_MAX_UNCERTAINTY = 1.0e-4
AL_PERIODIC_RESTART_EVERY = 30
AL_LOCAL_OPTIMIZER_DEFAULT = "bfgs"
AL_LOCAL_OPTIMIZERS = {
    "bfgs": BFGS,
    "fire2": FIRE2,
}
_LOCAL_OPTIMIZER_ALIASES = {
    "bfgs": "bfgs",
    "fire2": "fire2",
    "fire": "fire2",
}
_SUMMARY_TABLE_ROW_RE = re.compile(r"^\|\s*\d+\s*\|")
_LOCALAL_STATE_FILENAMES = (
    "predicted.traj",
    "predicted_evaluated.traj",
    "converged.traj",
    "initial_struc.traj",
    "ml_summary.txt",
    "ml_time.txt",
)


def normalize_local_optimizer_name(value: object) -> str:
    text = str(value or "").strip().lower().replace("-", "").replace("_", "")
    if not text:
        return AL_LOCAL_OPTIMIZER_DEFAULT
    normalized = _LOCAL_OPTIMIZER_ALIASES.get(text)
    if normalized is None:
        allowed = ", ".join(sorted(AL_LOCAL_OPTIMIZERS))
        raise ValueError(f"Unsupported active-learning optimizer '{value}'. Choose one of: {allowed}.")
    return normalized


def resolve_local_optimizer(value: object):
    normalized = normalize_local_optimizer_name(value)
    return normalized, AL_LOCAL_OPTIMIZERS[normalized]


def _looks_like_ml_instability(exc: Exception) -> bool:
    text = str(exc).lower()
    if isinstance(exc, FileNotFoundError) and (
        "ml_summary.txt" in text or "ml_time.txt" in text
    ):
        return True
    markers = (
        "outside of the trust distance",
        "trust distance",
        "force spike",
        "true fmax",
        "nan",
        "non-finite",
        "not finite",
        "property \"energy\" is not available",
        "property 'energy' is not available",
        "property \"forces\" is not available",
        "property 'forces' is not available",
        "results unavailable",
        "floating point",
        "overflow",
        "singular",
        "ill-conditioned",
        "xa and xb must have the same number of columns",
        "feature dimension",
        "same number of columns",
        "no information about the structure in the orca output file",
    )
    return any(marker in text for marker in markers)


def _atoms_positions_are_finite(atoms) -> bool:
    try:
        for pos in atoms.get_positions():
            if not (
                isfinite(float(pos[0]))
                and isfinite(float(pos[1]))
                and isfinite(float(pos[2]))
            ):
                return False
    except Exception:
        return False
    return True


def _snapshot_geometry(atoms):
    constraints_snapshot = None
    try:
        constraints_snapshot = copy.deepcopy(getattr(atoms, "constraints", None))
    except Exception:
        try:
            constraints_snapshot = list(getattr(atoms, "constraints", []))
        except Exception:
            constraints_snapshot = None
    return (
        atoms.get_positions().copy(),
        atoms.get_cell().copy(),
        atoms.get_pbc().copy(),
        constraints_snapshot,
    )


def _restore_geometry(atoms, snapshot) -> None:
    positions, cell, pbc, constraints_snapshot = snapshot
    atoms.set_positions(positions)
    atoms.set_cell(cell)
    atoms.set_pbc(pbc)
    if constraints_snapshot is not None:
        try:
            atoms.set_constraint(copy.deepcopy(constraints_snapshot))
        except Exception:
            try:
                atoms.set_constraint(constraints_snapshot)
            except Exception:
                pass


def _archive_path(path: Path, *, tag: str) -> Path | None:
    if not path.exists():
        return None
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = path.with_name(f"{path.name}.{tag}.{timestamp}")
    index = 1
    while archive.exists():
        index += 1
        archive = path.with_name(f"{path.name}.{tag}.{timestamp}.{index}")
    try:
        path.rename(archive)
        return archive
    except Exception as exc:
        print(f"Warning: Failed to archive {path}: {exc}")
        return None


def _reset_localal_state_files(state_dir: Path) -> None:
    for name in _LOCALAL_STATE_FILENAMES:
        _archive_path(state_dir / name, tag="failed")


def _archive_localal_state(
    state_dir: Path,
    trainingset_path: Path,
    *,
    reason: str,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = state_dir / f"{reason}_restarts" / timestamp
    index = 1
    while archive_dir.exists():
        index += 1
        archive_dir = state_dir / f"{reason}_restarts" / f"{timestamp}.{index}"
    archive_dir.mkdir(parents=True)
    paths = [state_dir / name for name in _LOCALAL_STATE_FILENAMES]
    if trainingset_path not in paths:
        paths.append(trainingset_path)
    for path in paths:
        if path.exists():
            shutil.move(str(path), str(archive_dir / path.name))
    return archive_dir


def _training_frame_count(trainingset_path: Path) -> int | None:
    if not trainingset_path.exists():
        return 0
    try:
        with Trajectory(str(trainingset_path), "r") as trajectory:
            return len(trajectory)
    except Exception as exc:
        print(f"Warning: Failed to count active-learning frames in {trainingset_path}: {exc}")
        return None


def _read_recent_training_frames(trainingset_path: Path, keep: int):
    try:
        with Trajectory(str(trainingset_path), "r") as trajectory:
            start = max(0, len(trajectory) - max(1, int(keep)))
            return [trajectory[index] for index in range(start, len(trajectory))]
    except Exception as exc:
        print(f"Warning: Failed to retain recent active-learning frames from {trainingset_path}: {exc}")
        return []


def _write_training_frames(trainingset_path: Path, frames) -> None:
    trainingset_path.parent.mkdir(parents=True, exist_ok=True)
    with Trajectory(str(trainingset_path), "w") as trajectory:
        for frame in frames:
            trajectory.write(frame)


def _cached_max_force(atoms) -> float | None:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return None
    try:
        if calc.calculation_required(atoms, ["forces"]):
            return None
        forces = atoms.get_forces()
    except Exception:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    try:
        max_force = float(np.amax([np.linalg.norm(forces[i]) for i in range(len(atoms))]))
    except Exception:
        return None
    if not np.isfinite(max_force):
        return None
    return max_force


def _cached_potential_energy(atoms) -> float | None:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return None
    try:
        energy = float(atoms.get_potential_energy())
    except Exception:
        return None
    if not isfinite(energy):
        return None
    return energy


def _forces_are_finite(forces) -> bool:
    try:
        for vec in forces:
            if not (
                isfinite(float(vec[0]))
                and isfinite(float(vec[1]))
                and isfinite(float(vec[2]))
            ):
                return False
    except Exception:
        return False
    return True


def _max_force_from_current_forces(atoms) -> float | None:
    try:
        forces = atoms.get_forces()
    except Exception:
        return None
    if not _forces_are_finite(forces):
        return None
    try:
        max_force = 0.0
        for vec in forces:
            fx = float(vec[0])
            fy = float(vec[1])
            fz = float(vec[2])
            norm = (fx * fx + fy * fy + fz * fz) ** 0.5
            if norm > max_force:
                max_force = norm
        return float(max_force)
    except Exception:
        return None


def _max_force_from_cached_forces(atoms) -> float | None:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return None
    try:
        if calc.calculation_required(atoms, ["forces"]):
            return None
    except Exception:
        return None
    try:
        forces = atoms.get_forces()
    except Exception:
        return None
    if not _forces_are_finite(forces):
        return None
    max_force = 0.0
    for vec in forces:
        fx = float(vec[0])
        fy = float(vec[1])
        fz = float(vec[2])
        norm = (fx * fx + fy * fy + fz * fz) ** 0.5
        if norm > max_force:
            max_force = norm
    return float(max_force)


def _apply_geometry_from_atoms(target_atoms, source_atoms) -> bool:
    try:
        if len(target_atoms) != len(source_atoms):
            return False
        target_atoms.set_positions(source_atoms.get_positions())
        target_atoms.set_cell(source_atoms.get_cell())
        target_atoms.set_pbc(source_atoms.get_pbc())
    except Exception:
        return False

    if "initial_charges" in getattr(source_atoms, "arrays", {}):
        try:
            target_atoms.set_initial_charges(source_atoms.get_initial_charges())
            charge_method = source_atoms.info.get("charge_method")
            if charge_method:
                target_atoms.info["charge_method"] = charge_method
        except Exception:
            pass
    return True


def _select_last_viable_training_frame(
    trainingset_path: Path,
    *,
    max_fmax: float,
):
    if not trainingset_path.exists() or max_fmax <= 0.0 or not isfinite(max_fmax):
        return None
    try:
        frames = read(str(trainingset_path), index=":", format="traj")
    except Exception:
        try:
            frames = read(str(trainingset_path), index=":")
        except Exception as exc:
            print(f"Warning: Failed to read recovery training set {trainingset_path}: {exc}")
            return None
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    if not frames:
        return None

    for idx in range(len(frames) - 1, -1, -1):
        frame = frames[idx]
        frame_fmax = _max_force_from_cached_forces(frame)
        if frame_fmax is None:
            continue
        if frame_fmax <= max_fmax:
            return frame, idx, frame_fmax, len(frames)
    return None


def _trainingset_matches_atoms(trainingset_path: Path, atoms) -> tuple[bool, str | None]:
    if not trainingset_path.exists():
        return False, f"{trainingset_path} does not exist"
    try:
        frames = read(str(trainingset_path), index=":", format="traj")
    except Exception:
        try:
            frames = read(str(trainingset_path), index=":")
        except Exception as exc:
            return False, f"failed to read {trainingset_path}: {exc}"
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    if not frames:
        return False, f"{trainingset_path} contains no frames"
    try:
        expected_numbers = tuple(int(z) for z in atoms.get_atomic_numbers())
    except Exception as exc:
        return False, f"failed to read current atomic numbers: {exc}"
    expected_count = len(expected_numbers)
    for idx, frame in enumerate(frames, start=1):
        if len(frame) != expected_count:
            return (
                False,
                f"frame {idx} has {len(frame)} atoms, expected {expected_count}",
            )
        try:
            frame_numbers = tuple(int(z) for z in frame.get_atomic_numbers())
        except Exception as exc:
            return False, f"failed to read atomic numbers from frame {idx}: {exc}"
        if frame_numbers != expected_numbers:
            return (
                False,
                f"frame {idx} atomic numbers do not match current structure",
            )
    return True, None


def _seed_restart_trajectory_from_trainingset(
    trainingset_path: Path,
    trajectory_path: Path,
) -> bool:
    if not trainingset_path.exists():
        return False
    try:
        seed_atoms = read(str(trainingset_path), index=-1, format="traj")
    except Exception:
        try:
            seed_atoms = read(str(trainingset_path), index=-1)
        except Exception as exc:
            print(
                "Warning: Failed to rebuild restart trajectory from "
                f"{trainingset_path}: {exc}"
            )
            return False
    try:
        trajectory_path.parent.mkdir(parents=True, exist_ok=True)
        with Trajectory(str(trajectory_path), "w") as traj:
            traj.write(seed_atoms)
    except Exception as exc:
        print(
            f"Warning: Failed to write restart trajectory {trajectory_path} "
            f"from {trainingset_path}: {exc}"
        )
        return False
    return True


def _last_true_fmax_from_summary(summary_path: Path) -> float | None:
    try:
        lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line or "true fmax" in line.lower():
            continue
        if not _SUMMARY_TABLE_ROW_RE.match(line):
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if not parts:
            continue
        last_col = parts[-1]
        try:
            value = float(last_col)
        except ValueError:
            continue
        if isfinite(value):
            return float(value)
    return None


def _last_uncertainty_from_summary(summary_path: Path) -> float | None:
    try:
        lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for raw_line in reversed(lines):
        line = raw_line.strip()
        if not line or "uncertainty" in line.lower():
            continue
        if not _SUMMARY_TABLE_ROW_RE.match(line):
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) < 4:
            continue
        # Expected row columns:
        # Step | Date | True energy | Uncertainty | True error | True fmax
        # so uncertainty is the 3rd column from the end.
        uncertainty_col = parts[-3]
        try:
            value = float(uncertainty_col)
        except ValueError:
            continue
        if isfinite(value):
            return float(value)
    return None


def _plateau_from_summary(summary_path: Path) -> tuple[float, float] | None:
    try:
        lines = summary_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return None
    rows = []
    for raw_line in lines:
        line = raw_line.strip()
        if not _SUMMARY_TABLE_ROW_RE.match(line):
            continue
        parts = [part.strip() for part in line.split("|") if part.strip()]
        if len(parts) < 6:
            continue
        try:
            energy = float(parts[-4])
            uncertainty = abs(float(parts[-3]))
        except ValueError:
            continue
        if isfinite(energy) and isfinite(uncertainty):
            rows.append((energy, uncertainty))
    if len(rows) < AL_PLATEAU_WINDOW:
        return None
    window = rows[-AL_PLATEAU_WINDOW:]
    energy_range = max(row[0] for row in window) - min(row[0] for row in window)
    max_uncertainty = max(row[1] for row in window)
    if (
        energy_range <= AL_PLATEAU_MAX_ENERGY_RANGE
        and max_uncertainty <= AL_PLATEAU_MAX_UNCERTAINTY
    ):
        return energy_range, max_uncertainty
    return None


def _attach_charges_from_orca_output(atoms, orca_out: Path, *, label: str) -> None:
    try:
        get_charges_from_orca_output(atoms, orca_out, method="mulliken")
    except Exception as exc:
        print(f"Warning: Failed to recover charges for {label} from {orca_out}: {exc}")


def active_learning_step(
    atoms,
    calc,
    fmax=0.01,
    local_opt_kwargs=None,
    previous_calculations=None,
    shared_training_path: Path | None = None,
    state_dir: Path | None = None,
    *,
    use_restart: bool = True,
    steps_per_call: int = 1,
    optimize_to_convergence: bool = False,
    max_steps: int | None = None,
    max_unc: float = AL_PREDICTION_MAX_UNC,
    min_data: int = AL_MIN_DATA,
    unc_convergence: float | None = None,
    ml_steps: int = AL_ML_STEPS,
    dtrust: float = AL_DTRUST,
    raise_on_not_converged: bool = False,
    true_fmax_sanity_limit: float | None = AL_RECOVERY_TRUE_FMAX_SANITY_LIMIT,
    local_optimizer: str = AL_LOCAL_OPTIMIZER_DEFAULT,
    restart_every: int | None = AL_PERIODIC_RESTART_EVERY,
):
    local_opt_kwargs = dict(local_opt_kwargs or AL_LOCAL_OPT_KWARGS)
    _, local_opt_class = resolve_local_optimizer(local_optimizer)
    if unc_convergence is None:
        unc_convergence = max_unc
    if restart_every is not None:
        restart_every = int(restart_every)
        if restart_every <= 0:
            raise ValueError("restart_every must be positive or None.")
    calc_dir = getattr(calc, "directory", None)
    if calc_dir is None:
        raise ValueError("Calculator directory is required for LocalAL bookkeeping.")
    directory = Path(calc_dir)
    directory.mkdir(parents=True, exist_ok=True)
    state_dir = Path(state_dir) if state_dir is not None else directory
    state_dir.mkdir(parents=True, exist_ok=True)
    trainingset_path = shared_training_path if shared_training_path is not None else state_dir / "evaluated.traj"
    trainingset_path = Path(trainingset_path)
    trainingset_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = state_dir / "ml_summary.txt"
    # Always anchor cached-force checks to the calculator driving this AL call.
    # This avoids false convergence from stale forces attached by previous stages.
    atoms.calc = calc
    try:
        run_constraints = copy.deepcopy(getattr(atoms, "constraints", None))
    except Exception:
        try:
            run_constraints = list(getattr(atoms, "constraints", []))
        except Exception:
            run_constraints = None

    def _reapply_run_constraints() -> None:
        if run_constraints is None:
            return
        try:
            atoms.set_constraint(copy.deepcopy(run_constraints))
        except Exception:
            try:
                atoms.set_constraint(run_constraints)
            except Exception:
                pass

    def _periodic_restart_if_due() -> Path | None:
        training_frame_count = _training_frame_count(trainingset_path)
        if (
            restart_every is None
            or training_frame_count is None
            or training_frame_count < restart_every
        ):
            return None
        retained_frames = _read_recent_training_frames(trainingset_path, min_data)
        if not retained_frames:
            return None
        archive_dir = _archive_localal_state(
            state_dir,
            trainingset_path,
            reason="periodic",
        )
        _write_training_frames(trainingset_path, retained_frames)
        _apply_geometry_from_atoms(atoms, retained_frames[-1])
        _reapply_run_constraints()
        print(
            "Active learning restart: "
            f"archived {training_frame_count} frames in {archive_dir}; "
            f"retained the latest {len(retained_frames)} DFT-evaluated frames."
        )
        return trainingset_path

    use_restart_flag = bool(use_restart)
    trajectory_restart_flag = False
    prev_calc_path: Path | None = None
    periodic_restart_path = _periodic_restart_if_due()
    if periodic_restart_path is not None:
        use_restart_flag = False
        prev_calc_path = periodic_restart_path
    elif use_restart_flag:
        candidate_paths: list[Path] = []
        if trainingset_path.exists():
            candidate_paths.append(trainingset_path)
        if previous_calculations is not None:
            candidate_paths.append(Path(previous_calculations))
        for candidate in candidate_paths:
            if not is_valid_trainingset(candidate):
                continue
            compatible, incompatibility_reason = _trainingset_matches_atoms(candidate, atoms)
            if compatible:
                prev_calc_path = candidate
                break
            print(
                f"Warning: Restart trainingset {candidate} is incompatible with current atoms "
                f"({incompatibility_reason}); ignoring it."
            )
            if candidate == trainingset_path:
                _archive_path(trainingset_path, tag="incompatible")
        if candidate_paths and prev_calc_path is None:
            print(
                f"Warning: Restart trainingset(s) {', '.join(str(p) for p in candidate_paths)} "
                "are missing or invalid; disabling restart."
            )
            use_restart_flag = False
        if prev_calc_path is None:
            use_restart_flag = False
        if use_restart_flag and not summary_path.exists():
            print(
                f"Info: Restart training set {prev_calc_path} is available, but "
                f"{summary_path.name} is missing in {state_dir}; starting LocalAL without CatLearn "
                "state restart (seeding from existing training set only)."
            )
            use_restart_flag = False
        trajectory_path = state_dir / "predicted.traj"
        if use_restart_flag:
            if trajectory_path.exists() and is_valid_trajectory(trajectory_path):
                trajectory_restart_flag = True
            else:
                if trajectory_path.exists():
                    print(
                        f"Warning: Restart trajectory {trajectory_path} is invalid; "
                        "rebuilding from latest training frame."
                    )
                if prev_calc_path is not None and _seed_restart_trajectory_from_trainingset(
                    prev_calc_path,
                    trajectory_path,
                ):
                    trajectory_restart_flag = True
                    print(
                        f"Info: Rebuilt restart trajectory {trajectory_path.name} from "
                        f"{prev_calc_path.name}."
                    )
                elif trajectory_path.exists():
                    print(
                        f"Warning: Could not rebuild restart trajectory {trajectory_path}; "
                        "continuing with training-set-only restart."
                    )

    def _build_localal(
        *,
        use_restart_flag: bool,
        trajectory_restart_flag: bool,
        previous_path: Path | None,
        local_kwargs: dict,
    ):
        from catlearn.activelearning.local import LocalAL

        return LocalAL(
            atoms=atoms,
            ase_calc=calc,
            unc_convergence=float(unc_convergence),
            local_opt=local_opt_class,
            local_opt_kwargs=local_kwargs,
            save_memory=False,
            use_restart=use_restart_flag,
            min_data=min_data,
            restart=trajectory_restart_flag,
            verbose=False,
            prev_calculations=str(previous_path) if previous_path is not None else None,
            trajectory=str(state_dir / "predicted.traj"),
            trainingset=str(trainingset_path),
            pred_evaluated=str(state_dir / "predicted_evaluated.traj"),
            converged_trajectory=str(state_dir / "converged.traj"),
            initial_traj=str(state_dir / "initial_struc.traj"),
            last_traj=None,
            tabletxt=str(state_dir / "ml_summary.txt"),
            timetxt=str(state_dir / "ml_time.txt"),
        )

    def _tighten_recovery_controls(
        current_dtrust: float,
        current_local_kwargs: dict,
    ) -> tuple[float, dict]:
        next_local_kwargs = dict(current_local_kwargs)
        maxstep_default = float(AL_LOCAL_OPT_KWARGS.get("maxstep", 0.02))
        try:
            maxstep_value = float(next_local_kwargs.get("maxstep", maxstep_default))
        except (TypeError, ValueError):
            maxstep_value = maxstep_default
        if maxstep_value <= 0.0:
            maxstep_value = maxstep_default
        next_local_kwargs["maxstep"] = max(
            maxstep_value * AL_RECOVERY_MAXSTEP_SCALE,
            AL_RECOVERY_MIN_MAXSTEP,
        )
        next_dtrust = max(float(current_dtrust) * AL_RECOVERY_DTRUST_SCALE, AL_RECOVERY_MIN_DTRUST)
        return next_dtrust, next_local_kwargs

    def _reseed_training_set() -> None:
        archived = _archive_path(trainingset_path, tag="failed")
        source_training = archived if archived is not None else trainingset_path
        recovery_seed = None
        if source_training.exists():
            recovery_seed = _select_last_viable_training_frame(
                source_training,
                max_fmax=AL_RECOVERY_RESEED_MAX_FMAX,
            )

        if archived is not None:
            if recovery_seed is not None:
                _, frame_idx, frame_fmax, frame_count = recovery_seed
                print(
                    f"Warning: Archived unstable AL training set {archived.name}; "
                    "rebuilding from last viable training frame "
                    f"{frame_idx + 1}/{frame_count} (fmax={frame_fmax:.6f} eV/A)."
                )
            else:
                print(
                    f"Warning: Archived unstable AL training set {archived.name}; "
                    "no viable recovery frame found, rebuilding from current geometry."
                )

        recovery_seed_atoms = None
        if recovery_seed is not None:
            seed_atoms, frame_idx, frame_fmax, frame_count = recovery_seed
            if _apply_geometry_from_atoms(atoms, seed_atoms):
                recovery_seed_atoms = seed_atoms
                print(
                    "Info: Recovery geometry set from training frame "
                    f"{frame_idx + 1}/{frame_count} with cached fmax={frame_fmax:.6f} eV/A "
                    f"(threshold {AL_RECOVERY_RESEED_MAX_FMAX:.6f} eV/A)."
                )
            else:
                print(
                    "Warning: Recovery seed geometry atom-count mismatch; "
                    "falling back to current geometry reseed."
                )

        try:
            atoms.calc = calc
            atoms.get_forces()
            reseed_fmax = _max_force_from_current_forces(atoms)
        except Exception as exc:
            if recovery_seed_atoms is not None:
                append_training_set(trainingset_path, recovery_seed_atoms)
                return
            print(f"Warning: Failed to reseed AL training set from current geometry: {exc}")
            return

        if (
            recovery_seed_atoms is not None
            and reseed_fmax is not None
            and reseed_fmax > AL_RECOVERY_RESEED_MAX_FMAX
        ):
            print(
                "Warning: Re-evaluated recovery seed geometry exceeded reseed guard "
                f"(fmax={reseed_fmax:.6f} eV/A > {AL_RECOVERY_RESEED_MAX_FMAX:.6f} eV/A); "
                "keeping cached viable training frame instead."
            )
            _apply_geometry_from_atoms(atoms, recovery_seed_atoms)
            append_training_set(trainingset_path, recovery_seed_atoms)
            return

        if reseed_fmax is not None and reseed_fmax > AL_RECOVERY_RESEED_MAX_FMAX:
            print(
                "Warning: Recovery reseed frame has high force "
                f"(fmax={reseed_fmax:.6f} eV/A > {AL_RECOVERY_RESEED_MAX_FMAX:.6f} eV/A)."
            )
        append_training_set(trainingset_path, atoms)

    run_dtrust = float(dtrust)
    run_local_opt_kwargs = dict(local_opt_kwargs)
    try:
        dyn = _build_localal(
            use_restart_flag=use_restart_flag,
            trajectory_restart_flag=trajectory_restart_flag,
            previous_path=prev_calc_path,
            local_kwargs=run_local_opt_kwargs,
        )
    except Exception as exc:
        if not _looks_like_ml_instability(exc):
            raise
        print(
            "Warning: Failed to initialize active-learning restart state "
            f"({exc}); rebuilding LocalAL state from a fresh reseed."
        )
        _reset_localal_state_files(state_dir)
        _reseed_training_set()
        run_dtrust, run_local_opt_kwargs = _tighten_recovery_controls(
            run_dtrust,
            run_local_opt_kwargs,
        )
        dyn = _build_localal(
            use_restart_flag=False,
            trajectory_restart_flag=False,
            previous_path=None,
            local_kwargs=run_local_opt_kwargs,
        )

    def _run_chunk_with_recovery(current_dyn, *, step_chunk: int):
        nonlocal run_dtrust, run_local_opt_kwargs
        attempts = 0
        while True:
            _reapply_run_constraints()
            snapshot = _snapshot_geometry(atoms)
            try:
                current_dyn.run(
                    fmax=fmax,
                    max_unc=max_unc,
                    steps=step_chunk,
                    ml_steps=ml_steps,
                    dtrust=run_dtrust,
                )
                if not _atoms_positions_are_finite(atoms):
                    raise RuntimeError(
                        "Active-learning optimiser produced non-finite atom coordinates."
                    )
                energy = _cached_potential_energy(atoms)
                if energy is None:
                    raise RuntimeError(
                        "Active-learning optimiser returned unavailable or non-finite energy."
                    )
                cached_max_force = _max_force_from_current_forces(atoms)
                if cached_max_force is None:
                    raise RuntimeError(
                        "Active-learning optimiser returned unavailable or non-finite forces."
                    )

                summary_fmax = _last_true_fmax_from_summary(summary_path)
                try:
                    sanity_limit = (
                        float(true_fmax_sanity_limit)
                        if true_fmax_sanity_limit is not None
                        else None
                    )
                except (TypeError, ValueError):
                    sanity_limit = None
                if (
                    sanity_limit is not None
                    and isfinite(sanity_limit)
                    and sanity_limit > 0.0
                    and summary_fmax is not None
                    and summary_fmax > sanity_limit
                ):
                    raise RuntimeError(
                        "Active-learning optimiser produced a True fmax force spike "
                        f"({summary_fmax:.6f} eV/A > sanity limit {sanity_limit:.6f} eV/A)."
                    )
                _reapply_run_constraints()
                return current_dyn
            except Exception as exc:
                recoverable = _looks_like_ml_instability(exc)
                if not recoverable:
                    raise
                if attempts >= AL_RECOVERY_MAX_RETRIES:
                    raise RuntimeError(
                        "Active-learning optimiser instability persisted after "
                        f"{AL_RECOVERY_MAX_RETRIES} recovery attempts: {exc}"
                    ) from exc
                attempts += 1
                _restore_geometry(atoms, snapshot)
                print(
                    "Warning: Active-learning optimiser became unstable "
                    f"({exc}). Recovery attempt {attempts}/{AL_RECOVERY_MAX_RETRIES}: "
                    "dropping restart state/training data and retrying with tighter trust settings."
                )
                _reset_localal_state_files(state_dir)
                _reseed_training_set()
                run_dtrust, run_local_opt_kwargs = _tighten_recovery_controls(
                    run_dtrust,
                    run_local_opt_kwargs,
                )
                current_dyn = _build_localal(
                    use_restart_flag=False,
                    trajectory_restart_flag=False,
                    previous_path=None,
                    local_kwargs=run_local_opt_kwargs,
                )

    if optimize_to_convergence and max_steps is not None and max_steps <= 0:
        raise ValueError("max_steps must be positive when optimize_to_convergence is enabled.")

    if optimize_to_convergence and is_converged(atoms, fmax=fmax):
        _reapply_run_constraints()
        atoms.calc.directory = calc_dir
        return atoms

    if optimize_to_convergence:
        if steps_per_call <= 0:
            raise ValueError("steps_per_call must be positive when optimize_to_convergence is enabled.")
        total_steps = 0
        training_frame_guard_disabled = False
        converged_training_frame_info: tuple[int, int, float] | None = None
        while True:
            if max_steps is not None and total_steps >= max_steps:
                break
            if max_steps is None:
                step_chunk = int(steps_per_call)
            else:
                remaining = max_steps - total_steps
                step_chunk = min(steps_per_call, remaining)
            dyn = _run_chunk_with_recovery(dyn, step_chunk=step_chunk)
            total_steps += step_chunk
            if is_converged(atoms, fmax=fmax):
                break
            plateau = _plateau_from_summary(summary_path)
            if plateau is not None:
                energy_range, max_summary_uncertainty = plateau
                archive_dir = _archive_localal_state(
                    state_dir,
                    trainingset_path,
                    reason="plateau",
                )
                print(
                    "Warning: Active-learning energy plateau detected "
                    f"over {AL_PLATEAU_WINDOW} evaluations "
                    f"(energy range={energy_range:.6f}, "
                    f"max uncertainty={max_summary_uncertainty:.6f}); "
                    f"archived old state in {archive_dir} and restarting from the endpoint."
                )
                run_dtrust = float(dtrust)
                run_local_opt_kwargs = dict(local_opt_kwargs)
                dyn = _build_localal(
                    use_restart_flag=False,
                    trajectory_restart_flag=False,
                    previous_path=None,
                    local_kwargs=run_local_opt_kwargs,
                )
                continue
            if not training_frame_guard_disabled:
                converged_training_frame = _select_last_viable_training_frame(
                    trainingset_path,
                    max_fmax=fmax,
                )
                if converged_training_frame is not None:
                    frame_atoms, frame_idx, frame_fmax, frame_count = converged_training_frame
                    if _apply_geometry_from_atoms(atoms, frame_atoms):
                        print(
                            "Info: Using converged training-set frame "
                            f"{frame_idx + 1}/{frame_count} (fmax={frame_fmax:.6f} eV/A) "
                            f"to stop active learning at target fmax={fmax:.6f}."
                        )
                        converged_training_frame_info = (
                            frame_idx + 1,
                            frame_count,
                            frame_fmax,
                        )
                        break
                    print(
                        "Warning: Converged training-set frame atom-count mismatch; "
                        "ignoring training-set convergence guard for this run."
                    )
                    training_frame_guard_disabled = True
            periodic_restart_path = _periodic_restart_if_due()
            if periodic_restart_path is not None:
                run_dtrust = float(dtrust)
                run_local_opt_kwargs = dict(local_opt_kwargs)
                dyn = _build_localal(
                    use_restart_flag=False,
                    trajectory_restart_flag=False,
                    previous_path=periodic_restart_path,
                    local_kwargs=run_local_opt_kwargs,
                )
                continue
        cached_max_force = _cached_max_force(atoms)
        cached_force_converged = bool(
            cached_max_force is not None and cached_max_force <= fmax
        )
        summary_fmax = _last_true_fmax_from_summary(summary_path)
        summary_uncertainty = _last_uncertainty_from_summary(summary_path)
        uncertainty_converged = True
        try:
            max_unc_target = float(max_unc)
        except (TypeError, ValueError):
            max_unc_target = None
        if max_unc_target is not None and isfinite(max_unc_target) and max_unc_target > 0.0:
            # If LocalAL does not provide a finite uncertainty yet (for example
            # very early iterations), do not block convergence solely on that.
            uncertainty_converged = (
                summary_uncertainty is None
                or summary_uncertainty <= max_unc_target
            )
        if converged_training_frame_info is not None:
            converged_now = bool(uncertainty_converged)
            if converged_now and summary_uncertainty is not None and max_unc_target is not None:
                print(
                    "Info: Training-frame force convergence accepted with summary uncertainty "
                    f"{summary_uncertainty:.6f} <= {max_unc_target:.6f}."
                )
        elif summary_fmax is not None:
            force_converged = bool(summary_fmax <= fmax)
            converged_now = bool(force_converged and uncertainty_converged)
            if converged_now and not cached_force_converged:
                print(
                    "Info: Using LocalAL summary convergence evidence "
                    f"(True fmax={summary_fmax:.6f} <= {fmax:.6f})."
                )
            if force_converged and (not uncertainty_converged) and summary_uncertainty is not None and max_unc_target is not None:
                print(
                    "Warning: Force threshold is met but uncertainty threshold is not met "
                    f"(uncertainty={summary_uncertainty:.6f} > {max_unc_target:.6f}); "
                    "treating as not converged."
                )
        else:
            converged_now = bool(cached_force_converged and uncertainty_converged)
        if not converged_now:
            if max_steps is None:
                message = "Active learning did not converge."
            else:
                message = (
                    f"Active learning did not converge within {max_steps} local steps."
                )
            if summary_fmax is not None:
                message = (
                    f"{message} True fmax={summary_fmax:.6f} "
                    f"(target <= {fmax:.6f})."
                )
            elif cached_max_force is not None:
                message = (
                    f"{message} Last cached max force={cached_max_force:.6f}."
                )
            if max_unc_target is not None and isfinite(max_unc_target) and max_unc_target > 0.0:
                if summary_uncertainty is not None:
                    message = (
                        f"{message} Uncertainty={summary_uncertainty:.6f} "
                        f"(target <= {max_unc_target:.6f})."
                    )
                else:
                    message = (
                        f"{message} Uncertainty target <= {max_unc_target:.6f} "
                        "could not be verified from summary output."
                    )
            if raise_on_not_converged:
                raise RuntimeError(message)
            print(f"Warning: {message}")
    else:
        dyn = _run_chunk_with_recovery(dyn, step_chunk=steps_per_call)
    _reapply_run_constraints()
    atoms.calc.directory = calc_dir
    return atoms


def is_converged(atoms, fmax: float = 0.01) -> bool:
    max_force = _cached_max_force(atoms)
    if max_force is None:
        return False
    return bool(max_force <= fmax)


def append_training_set(trainingset_path: Path, atoms) -> None:
    try:
        trainingset_path.parent.mkdir(parents=True, exist_ok=True)
        with Trajectory(str(trainingset_path), "a") as traj:
            traj.write(atoms)
    except Exception as exc:
        print(f"Warning: Failed to append training data to {trainingset_path}: {exc}")


def update_atoms_from_orca_output(atoms, orca_out: Path, *, label: str = "scan") -> bool:
    if not validate_orca_output(orca_out):
        return False
    try:
        scanned_atoms = read(orca_out, format="orca-output")
    except Exception as exc:
        print(f"Warning: Failed to read {label} geometry from {orca_out}: {exc}")
        return False
    _attach_charges_from_orca_output(scanned_atoms, orca_out, label=label)
    if len(scanned_atoms) != len(atoms):
        print(
            f"Warning: {label} geometry atom count mismatch in {orca_out}; "
            "keeping current geometry."
        )
        return False
    atoms.set_positions(scanned_atoms.get_positions())
    atoms.set_cell(scanned_atoms.get_cell())
    atoms.set_pbc(scanned_atoms.get_pbc())
    if "initial_charges" in scanned_atoms.arrays:
        atoms.set_initial_charges(scanned_atoms.get_initial_charges())
        charge_method = scanned_atoms.info.get("charge_method")
        if charge_method:
            atoms.info["charge_method"] = charge_method
    return True


def ensure_training_seed(
    atoms,
    *,
    training_path: Path,
    seed_dir: Path,
    gbw_source_dir: Path,
    gbw_fallback_dirs: Sequence[Path] | None = None,
    iroot: int,
    cpus: int,
    mem: int,
    calc_factory,
) -> tuple[object, Path]:
    if is_valid_trainingset(training_path):
        return atoms, gbw_source_dir

    seed_out = seed_dir / "orca.out"
    if validate_orca_output(seed_out):
        try:
            seeded_atoms = read(seed_out, format="orca-output")
            _attach_charges_from_orca_output(seeded_atoms, seed_out, label="seed geometry")
            append_training_set(training_path, seeded_atoms)
            return seeded_atoms, seed_dir
        except Exception as exc:
            print(f"Warning: Failed to reuse seed output {seed_out}: {exc}")

    gbw_source = select_gbw_source(gbw_source_dir, gbw_fallback_dirs)
    copy_initial_gbw(gbw_source, seed_dir)
    seed_calc = calc_factory(
        atoms,
        cpus=cpus,
        mem=mem,
        task="engrad",
        directory=seed_dir,
        nroots=iroot + 5,
        iroot=iroot,
    )
    atoms.calc = seed_calc
    atoms.get_forces()
    if not validate_orca_output(seed_out):
        raise RuntimeError(f"Failed to generate seed engrad output: {seed_out}")
    append_training_set(training_path, atoms)
    return atoms, seed_dir


@dataclass
class ProxyContext:
    target_wavelength_nm: float
    fragments: dict[str, str]
    initial_score_weights: dict[str, float]
    followup_score_weights: dict[str, float]
    min_oscillator_strength: float
    wavelength_sigma_nm: float
    similarity_sigma_nm: float
    similarity_sigma_ct: float
    nroots: int
    min_similarity: float = 0.0
    min_relative_intensity: float = 0.0
    intensity_reference_window_nm: float | None = 200.0
    min_transition_overlap: float = 0.0
    min_ct_similarity: float = 0.0


class ScanThenEngradProxy(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        *,
        cpus: int,
        mem: int,
        scan_base: Path,
        opt_dir: Path,
        gbw_source_dir: Path,
        previous_choice: Optional[IRootCandidate],
        previous_state: Optional[int],
        iteration: int,
        geometry_step_start: int,
        bond_pair: tuple[int, int] | None,
        geometry_log_path: Path,
        context: ProxyContext,
        calc_factory,
        engrad_calc_factory=None,
        use_scan_geometry: bool = False,
        reference_dir: Path | None = None,
    ):
        super().__init__()
        self.cpus = cpus
        self.mem = mem
        self.scan_base = Path(scan_base)
        self.opt_dir = Path(opt_dir)
        self.gbw_source_dir = Path(gbw_source_dir)
        self.previous_choice = previous_choice
        self.previous_state = previous_state
        self.iteration = iteration
        self.geometry_step = geometry_step_start
        self.bond_pair = bond_pair
        self.geometry_log_path = geometry_log_path
        self.context = context
        self.calc_factory = calc_factory
        self.engrad_calc_factory = engrad_calc_factory or calc_factory
        self.last_scan_choice: Optional[IRootCandidate] = None
        self.last_scan_dir: Optional[Path] = None
        self.last_geometry_step: Optional[int] = None
        self.last_scan_point_kind: str = ""
        self.last_scan_point_label: str = ""
        self.directory = self.opt_dir
        self.use_scan_geometry = use_scan_geometry
        self.reference_dir = Path(reference_dir) if reference_dir is not None else Path(gbw_source_dir)
        self.state_tracker = StateTracker(root_state=int(previous_state or 0), mode="opt")
        self.reference_point: StateTrackPoint | None = None
        self.reference_cubes = None

    @staticmethod
    def _track_states(df_data) -> list[dict]:
        states = []
        for _, row in df_data.iterrows():
            wavelength = float(row["nm"])
            ct_value = row.get("CT(e)")
            try:
                q_ct = abs(float(ct_value)) if ct_value is not None else 0.0
            except (TypeError, ValueError):
                q_ct = 0.0
            if not isfinite(q_ct):
                q_ct = 0.0
            fragment_q = {}
            for name, value in row.items():
                if not isinstance(name, str) or ("->" not in name and "<-" not in name):
                    continue
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    continue
                if isfinite(parsed):
                    fragment_q[name] = parsed
            states.append(
                {
                    "state": int(row["state"]),
                    "wavelength_nm": wavelength,
                    "energy_ev": 1239.841984 / wavelength,
                    "oscillator_strength": float(row["f"]),
                    "ct_metrics": {"q_ct": q_ct, "fragment_q": fragment_q},
                }
            )
        return states

    def _load_reference(self) -> None:
        if self.reference_cubes is not None or self.previous_choice is None:
            return
        state = int(self.previous_choice.state)
        cubes = generate_state_cubes(
            self.reference_dir,
            [state],
            self.reference_dir / "state_tracking",
        )
        if state not in cubes:
            raise RuntimeError(
                f"State tracking could not generate reference cubes for S{state} in {self.reference_dir}"
            )
        self.reference_cubes = cubes[state]
        self.reference_point = StateTrackPoint(
            step_index=max(0, self.geometry_step - 1),
            step_id=str(self.reference_dir),
            state_number=state,
            energy_ev=float(self.previous_choice.energy_ev or 0.0),
            wavelength_nm=float(self.previous_choice.wavelength_nm),
            oscillator_strength=float(self.previous_choice.intensity),
            ct_metrics={"q_ct": float(self.previous_choice.ct_magnitude)},
            confidence="REFERENCE",
        )

    @staticmethod
    def _clean_losing_cubes(cubes_map, winner_state: int) -> None:
        for state, cube_data in cubes_map.items():
            if int(state) == int(winner_state):
                continue
            paths = cube_data.get("all_pair_paths", []) + [cube_data.get("cdd_path")]
            for path in paths:
                if path:
                    Path(path).unlink(missing_ok=True)

    def _next_scan_dir(self) -> tuple[Path, int]:
        step = self.geometry_step
        scan_dir = Path(f"{self.scan_base}_{step}")
        self.geometry_step += 1
        return scan_dir, step

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            atoms = self.atoms
        self.last_scan_point_kind = ""
        self.last_scan_point_label = ""

        scan_dir, step = self._next_scan_dir()
        gbw_source = select_gbw_source(self.gbw_source_dir, [scan_dir])
        copy_initial_gbw(gbw_source, scan_dir)
        scan_calc = self.calc_factory(
            atoms,
            cpus=self.cpus,
            mem=self.mem,
            task="energy",
            directory=scan_dir,
            nroots=self.context.nroots,
        )
        atoms.calc = scan_calc
        atoms.get_potential_energy()
        if self.use_scan_geometry:
            update_atoms_from_orca_output(
                atoms,
                scan_dir / "orca.out",
                label=f"scan iter {self.iteration} step {step}",
            )

        tracking_df = get_ct_info(
            scan_dir,
            self.context.fragments,
            min_oscillator_strength=0.0,
        )
        df = tracking_df[tracking_df["f"] > float(self.context.min_oscillator_strength)]
        score_weights = (
            self.context.followup_score_weights
            if (self.previous_choice is not None or self.previous_state is not None)
            else self.context.initial_score_weights
        )
        state_vectors = extract_state_transition_vectors(scan_dir / "orca.out")

        def _pick(
            df_data,
            *,
            weights: dict[str, float],
            min_similarity: float,
            use_orbital_overlap: bool = True,
        ):
            has_previous = self.previous_choice is not None or self.previous_state is not None
            enforce_overlap_filters = bool(use_orbital_overlap and has_previous)
            choice, _ = pick_iroot(
                df_data,
                target_wavelength_nm=self.context.target_wavelength_nm,
                previous_choice=self.previous_choice,
                previous_state=self.previous_state,
                score_weights=weights,
                wavelength_sigma_nm=self.context.wavelength_sigma_nm,
                similarity_sigma_nm=self.context.similarity_sigma_nm,
                similarity_sigma_ct=self.context.similarity_sigma_ct,
                min_similarity=min_similarity,
                min_relative_intensity=self.context.min_relative_intensity,
                intensity_reference_window_nm=self.context.intensity_reference_window_nm,
                min_transition_overlap=(
                    self.context.min_transition_overlap if enforce_overlap_filters else 0.0
                ),
                min_ct_similarity=(
                    self.context.min_ct_similarity if enforce_overlap_filters else 0.0
                ),
                require_transition_overlap=enforce_overlap_filters,
                state_vectors=(state_vectors if use_orbital_overlap else None),
            )
            return choice

        fallback_point_kind = ""
        fallback_point_label = ""
        has_followup_history = self.previous_choice is not None or self.previous_state is not None
        scan_choice = None
        tracking_results = []
        winner_point = None
        candidate_cubes = {}
        if self.previous_choice is not None:
            self._load_reference()
            all_states = self._track_states(tracking_df)
            candidates = self.state_tracker.filter_energy_window(
                float(self.reference_point.energy_ev),
                all_states,
            )
            candidate_states = [int(item["state"]) for item in candidates]
            candidate_cubes = generate_state_cubes(
                scan_dir,
                candidate_states,
                scan_dir / "state_tracking",
            )
            winner_point, tracking_results = self.state_tracker.select_next_state(
                self.reference_point,
                self.reference_cubes,
                candidates,
                candidate_cubes,
                step_index=step,
                step_id=str(scan_dir),
            )
            spatial_overlaps = {
                int(result.cand_state): float(result.composite_score)
                for result in tracking_results
            }
            _, scored = pick_iroot(
                tracking_df,
                target_wavelength_nm=self.context.target_wavelength_nm,
                previous_choice=self.previous_choice,
                previous_state=self.previous_state,
                score_weights=score_weights,
                wavelength_sigma_nm=self.context.wavelength_sigma_nm,
                similarity_sigma_nm=self.context.similarity_sigma_nm,
                similarity_sigma_ct=self.context.similarity_sigma_ct,
                state_spatial_overlaps=spatial_overlaps,
            )
            base_choice = next(
                (candidate for candidate in scored if candidate.state == winner_point.state_number),
                None,
            )
            if base_choice is None:
                raise RuntimeError(
                    f"Tracked state S{winner_point.state_number} missing from scan candidates"
                )
            spatial_score = spatial_overlaps[winner_point.state_number]
            scan_choice = replace(
                base_choice,
                score=spatial_score,
                similarity=spatial_score,
                spatial_overlap=spatial_score,
            )
        else:
            scan_choice = _pick(
                df,
                weights=score_weights,
                min_similarity=self.context.min_similarity,
            )
        if scan_choice is None and score_weights != self.context.initial_score_weights:
            baseline_choice = _pick(
                df,
                weights=self.context.initial_score_weights,
                min_similarity=self.context.min_similarity,
            )
            if baseline_choice is not None:
                scan_choice = baseline_choice
                print(
                    "Info: Used baseline TDDFT scan scoring with overlap+CT gating for "
                    f"iter {self.iteration} geom {step}; selected IROOT={scan_choice.state}."
                )
        relaxed_df = None
        if scan_choice is None and self.context.min_oscillator_strength > 0.0:
            try:
                relaxed_df = get_ct_info(
                    scan_dir,
                    self.context.fragments,
                    min_oscillator_strength=0.0,
                )
            except Exception as exc:
                print(
                    "Warning: Failed to relax oscillator-strength threshold for "
                    f"iter {self.iteration} geom {step}: {exc}"
                )
            else:
                relaxed_choice = _pick(
                    relaxed_df,
                    weights=score_weights,
                    min_similarity=self.context.min_similarity,
                )
                if (
                    relaxed_choice is None
                    and score_weights != self.context.initial_score_weights
                ):
                    relaxed_choice = _pick(
                        relaxed_df,
                        weights=self.context.initial_score_weights,
                        min_similarity=self.context.min_similarity,
                    )
                if relaxed_choice is not None:
                    scan_choice = relaxed_choice
                    print(
                        "Info: Relaxed TDDFT scan oscillator-strength threshold from "
                        f"{self.context.min_oscillator_strength:.6f} to 0.000000 "
                        "while retaining overlap+CT suitability checks for "
                        f"iter {self.iteration} geom {step}; selected IROOT={scan_choice.state}."
                    )
        if scan_choice is None and has_followup_history:
            print(
                "Info: No overlap+CT-qualified IROOT candidates found in scan iter "
                f"{self.iteration} geom {step}; switching to no-overlap fallback scoring."
            )
            no_overlap_choice = _pick(
                df,
                weights=self.context.initial_score_weights,
                min_similarity=0.0,
                use_orbital_overlap=False,
            )
            no_overlap_source = "standard scan candidates"
            if no_overlap_choice is None and self.context.min_oscillator_strength > 0.0:
                if relaxed_df is None:
                    try:
                        relaxed_df = get_ct_info(
                            scan_dir,
                            self.context.fragments,
                            min_oscillator_strength=0.0,
                        )
                    except Exception as exc:
                        print(
                            "Warning: Failed to prepare no-overlap fallback candidates for "
                            f"iter {self.iteration} geom {step}: {exc}"
                        )
                        relaxed_df = None
                if relaxed_df is not None:
                    no_overlap_choice = _pick(
                        relaxed_df,
                        weights=self.context.initial_score_weights,
                        min_similarity=0.0,
                        use_orbital_overlap=False,
                    )
                    if no_overlap_choice is not None:
                        no_overlap_source = "relaxed oscillator-strength candidates"
            if no_overlap_choice is not None:
                scan_choice = no_overlap_choice
                fallback_point_kind = "fallback_no_orbital_overlap"
                fallback_point_label = "no_orbital_overlap_initial_scoring"
                print(
                    "Fallback IROOT mode: Disabled orbital-overlap state-following for "
                    f"iter {self.iteration} geom {step}; using initial scoring "
                    "(target wavelength + CT intensity + similarity). "
                    f"Source={no_overlap_source}; selected IROOT={scan_choice.state}."
                )
        if scan_choice is None:
            raise RuntimeError(
                "No valid IROOT candidates found in scan iter "
                f"{self.iteration} geom {step} "
                f"(min_similarity={self.context.min_similarity:.3f}, "
                f"min_oscillator_strength={self.context.min_oscillator_strength:.6f}, "
                f"min_relative_intensity={self.context.min_relative_intensity:.3f}, "
                f"min_transition_overlap={self.context.min_transition_overlap:.3f}, "
                f"min_ct_similarity={self.context.min_ct_similarity:.3f})."
            )

        bond_length = atoms.get_distance(*self.bond_pair) if self.bond_pair is not None else None
        record_geometry_choice(
            self.geometry_log_path,
            iteration=self.iteration,
            geometry_step=step,
            scan_dir=scan_dir,
            bond_pair=self.bond_pair,
            bond_length=bond_length,
            choice=scan_choice,
            target_wavelength_nm=self.context.target_wavelength_nm,
        )
        self.last_scan_choice = scan_choice
        self.last_scan_dir = scan_dir
        self.last_geometry_step = step
        self.last_scan_point_kind = fallback_point_kind
        self.last_scan_point_label = fallback_point_label

        if winner_point is not None:
            tracking_path = scan_dir / "state_tracking.json"
            tracking_path.write_text(
                json.dumps(
                    {
                        "winner": {
                            "state": winner_point.state_number,
                            "energy_ev": winner_point.energy_ev,
                            "wavelength_nm": winner_point.wavelength_nm,
                            "oscillator_strength": winner_point.oscillator_strength,
                            "confidence": winner_point.confidence,
                        },
                        "overlaps": [
                            {
                                "state": result.cand_state,
                                "paired": result.paired_score,
                                "hole": result.hole_overlap_sq,
                                "electron": result.electron_overlap_sq,
                                "cdd": result.cdd_overlap_sq,
                                "multi_pair_paired": result.multi_pair_paired_score,
                                "cluster_rmsd_A": result.cluster_rmsd_A,
                                "spatial": result.composite_score,
                            }
                            for result in tracking_results
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            self._clean_losing_cubes(candidate_cubes, winner_point.state_number)
            self.reference_point = winner_point
            self.reference_cubes = candidate_cubes[winner_point.state_number]
            self.reference_dir = scan_dir

        copy_initial_gbw(scan_dir, self.opt_dir)
        engrad_calc = self.engrad_calc_factory(
            atoms,
            cpus=self.cpus,
            mem=self.mem,
            task="engrad",
            directory=self.opt_dir,
            nroots=scan_choice.state + 5,
            iroot=scan_choice.state,
        )
        atoms.calc = engrad_calc
        try:
            forces = atoms.get_forces()
            energy = atoms.get_potential_energy()
        except Exception as exc:
            raise RuntimeError(f"Engrad calculator results unavailable: {exc}") from exc
        if not _forces_are_finite(forces):
            raise RuntimeError("Engrad calculator returned non-finite forces.")
        try:
            energy_float = float(energy)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Engrad calculator returned non-numeric energy: {energy!r}") from exc
        if not isfinite(energy_float):
            raise RuntimeError(f"Engrad calculator returned non-finite energy: {energy_float!r}")
        self.results["forces"] = forces
        self.results["energy"] = energy_float

        opt_out = self.opt_dir / "orca.out"
        if not validate_orca_output(opt_out):
            raise RuntimeError(f"ORCA engrad output invalid: {opt_out}")
        _attach_charges_from_orca_output(
            atoms,
            opt_out,
            label=f"iter {self.iteration} engrad geometry",
        )

        atoms.calc = self
        self.previous_choice = scan_choice
        self.previous_state = scan_choice.state
        self.gbw_source_dir = self.opt_dir
