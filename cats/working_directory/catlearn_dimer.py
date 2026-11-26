from __future__ import annotations

import copy
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.mep.dimer import DimerControl, MinModeAtoms, MinModeTranslate


@dataclass
class DimerRunState:
    """Isolated per-run state tracking eigenmode and active learning surrogate status."""
    last_eigenmode: np.ndarray | None = None
    seeding_status: str = "unseeded"
    surrogate_status: str = "not_started"
    fallback_reason: str | None = None
    seeded_count: int = 0
    rejected_count: int = 0
    rejection_details: dict[str, int] = field(default_factory=dict)
    source_artifact_ids: list[int] = field(default_factory=list)
    rejected_artifacts: list[dict[str, Any]] = field(default_factory=list)
    staged_seed_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seeding_status": self.seeding_status,
            "surrogate_status": self.surrogate_status,
            "fallback_reason": self.fallback_reason,
            "seeded_count": self.seeded_count,
            "rejected_count": self.rejected_count,
            "rejection_details": dict(self.rejection_details),
            "source_artifact_ids": list(self.source_artifact_ids),
            "rejected_artifacts": list(self.rejected_artifacts),
            "staged_seed_file": self.staged_seed_file,
            "has_eigenmode": self.last_eigenmode is not None,
        }


class PersistentDimerLocalOpt:
    """ASE Dimer optimizer preserving eigenmode per-run across CatLearn retrain cycles."""
    _last_eigenmode: np.ndarray | None = None

    def __init__(self, atoms: Atoms, trajectory=None, logfile=None, **kwargs):
        self.state: DimerRunState | None = kwargs.pop("state", None)
        d_control = kwargs.pop("control", None) or DimerControl(
            eigenmode_method="dimer",
            f_rot_min=kwargs.pop("f_rot_min", 0.1),
            f_rot_max=kwargs.pop("f_rot_max", 1.0),
            max_num_rot=kwargs.pop("max_num_rot", 3),
            trial_angle=kwargs.pop("trial_angle", 0.785398),
            trial_trans_step=kwargs.pop("trial_trans_step", 0.001),
            maximum_translation=kwargs.pop("maximum_translation", 0.05),
            cg_translation=True,
            use_central_forces=kwargs.pop("use_central_forces", True),
            dimer_separation=kwargs.pop("dimer_separation", 0.005),
        )
        mode = kwargs.pop("initial_eigenmode", None)
        if mode is None:
            if self.state is not None and self.state.last_eigenmode is not None:
                mode = self.state.last_eigenmode
            elif PersistentDimerLocalOpt._last_eigenmode is not None:
                mode = PersistentDimerLocalOpt._last_eigenmode

        eigenmodes = None
        if mode is not None:
            try:
                m_arr = np.asarray(mode, float).reshape((len(atoms), 3))
                norm = float(np.linalg.norm(m_arr))
                if norm > 1e-6:
                    eigenmodes = [m_arr / norm]
            except Exception:
                eigenmodes = None

        self.min_mode_atoms = MinModeAtoms(atoms, control=d_control, eigenmodes=eigenmodes)
        if eigenmodes is None:
            self.min_mode_atoms.displace()

        self.optimizer = MinModeTranslate(self.min_mode_atoms, trajectory=trajectory, logfile=logfile, **kwargs)
        self.nsteps = 0

    def run(self, fmax: float = 0.02, steps: int = 100) -> bool:
        res = bool(self.optimizer.run(fmax=fmax, steps=steps))
        self.nsteps = getattr(self.optimizer, "nsteps", steps)
        try:
            mode = self.min_mode_atoms.get_eigenmode()
            if mode is not None:
                m_arr = np.asarray(mode, float).copy()
                if self.state is not None:
                    self.state.last_eigenmode = m_arr
                PersistentDimerLocalOpt._last_eigenmode = m_arr
        except Exception:
            pass
        return res

    def attach(self, *args, **kwargs):
        return self.optimizer.attach(*args, **kwargs)


def validate_dft_frame(
    frame: Atoms,
    target_atoms: Atoms,
) -> tuple[bool, str, float | None, np.ndarray | None]:
    """Validate DFT provenance, atom count, atomic numbers, and finite energy/forces."""
    if len(frame) != len(target_atoms):
        return False, f"atom count mismatch: {len(frame)} != {len(target_atoms)}", None, None

    if not np.array_equal(frame.get_atomic_numbers(), target_atoms.get_atomic_numbers()):
        return False, "atomic numbers or atom ordering mismatch", None, None

    calc = getattr(frame, "calc", None)
    calc_name = str(getattr(calc, "__class__", type(None)).__name__).lower()
    prov = str(frame.info.get("calculator_name") or frame.info.get("backend") or frame.info.get("provenance") or "").lower()
    if any(k in calc_name or k in prov for k in ("mace", "surrogate", "mock_surrogate")):
        return False, "non-DFT surrogate provenance rejected", None, None

    try:
        e = float(frame.get_potential_energy())
        if not math.isfinite(e):
            return False, "non-finite potential energy", None, None
    except Exception as exc:
        return False, f"cannot read potential energy: {exc}", None, None

    try:
        if calc is not None and hasattr(calc, "results") and "forces" in calc.results:
            raw_forces = np.asarray(calc.results["forces"], dtype=float).copy()
        else:
            try:
                raw_forces = np.asarray(frame.get_forces(apply_constraint=False), dtype=float).copy()
            except TypeError:
                raw_forces = np.asarray(frame.get_forces(), dtype=float).copy()

        if raw_forces.shape != (len(target_atoms), 3):
            return False, f"forces shape mismatch: {raw_forces.shape}", None, None
        if not np.isfinite(raw_forces).all():
            return False, "non-finite forces", None, None
    except Exception as exc:
        return False, f"cannot read forces: {exc}", None, None

    return True, "valid", e, raw_forces


def seed_dimer_training_set(
    source_trajs: Sequence[Any],
    target_atoms: Atoms,
    trainingset_path: str | Path | None = None,
    min_dist: float = 0.03,
    max_frames: int = 80,
    state: DimerRunState | None = None,
) -> list[Atoms]:
    """Harvest evaluated DFT frames using Furthest-Point Sampling (FPS) with raw force preservation."""
    valid_frames: list[Atoms] = []
    rejection_counts: dict[str, int] = {}

    def _record_rejection(reason: str):
        key = reason.split(":")[0].strip()
        rejection_counts[key] = rejection_counts.get(key, 0) + 1

    candidates: list[Atoms] = []
    for src in source_trajs:
        if isinstance(src, Atoms):
            candidates.append(src)
        elif isinstance(src, (list, tuple)):
            for item in src:
                if isinstance(item, Atoms):
                    candidates.append(item)
        else:
            p = Path(src)
            if not p.exists():
                _record_rejection("file_not_found")
                continue
            try:
                frames = list(read(str(p), ":"))
                candidates.extend(frames)
            except Exception as exc:
                _record_rejection(f"read_error: {exc}")
                continue

    for f in candidates:
        is_valid, reason, energy, forces = validate_dft_frame(f, target_atoms)
        if not is_valid:
            _record_rejection(reason)
            continue
        cleaned = f.copy()
        # Preserve raw unconstrained forces before applying target constraints
        cleaned.calc = SinglePointCalculator(cleaned, energy=energy, forces=forces.copy())
        cleaned.arrays["raw_forces"] = forces.copy()
        if target_atoms.constraints:
            cleaned.set_constraint(copy.deepcopy(target_atoms.constraints))
        else:
            cleaned.set_constraint()
        valid_frames.append(cleaned)

    if state is not None:
        state.rejected_count = sum(rejection_counts.values())
        state.rejection_details = rejection_counts

    if not valid_frames:
        if state is not None:
            state.seeded_count = 0
            state.seeding_status = "incompatible_seeds" if state.rejected_count > 0 else "no_sources"
        return []

    seeded: list[Atoms] = [valid_frames[0]]
    remaining = valid_frames[1:]
    while remaining and len(seeded) < max_frames:
        distances = [min(float(np.max(np.abs(f.positions - s.positions))) for s in seeded) for f in remaining]
        best_idx = int(np.argmax(distances))
        if distances[best_idx] < min_dist:
            break
        seeded.append(remaining.pop(best_idx))

    if state is not None:
        state.seeded_count = len(seeded)
        state.seeding_status = "seeded" if len(seeded) >= 3 else "missing_seeds"

    if trainingset_path is not None:
        trainingset_path = Path(trainingset_path)
        trainingset_path.parent.mkdir(parents=True, exist_ok=True)
        write(str(trainingset_path), seeded, format="traj")

    return seeded


def run_catlearn_dimer_workflow(
    *,
    atoms: Atoms,
    calc: Any,
    output_dir: str | Path,
    prior_trajectories: Sequence[Any] | None = None,
    staged_seed_file: str | Path | None = None,
    initial_eigenmode: Any = None,
    fmax: float = 0.02,
    max_steps: int = 100,
    polish_steps: int = 25,
    dtrust: float = 0.20,
    max_unc: float = 0.03,
    ml_steps: int = 200,
    dimer_separation: float = 0.005,
    use_central_forces: bool = True,
    workflow_log_path: Path | None = None,
    state: DimerRunState | None = None,
) -> dict[str, Any]:
    """Active learning Dimer surrogate optimization + final true DFT polish."""
    run_state = state or DimerRunState()
    if staged_seed_file:
        run_state.staged_seed_file = str(staged_seed_file)
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    trainingset_path = out_p / "trainingset.traj"

    if initial_eigenmode is not None:
        try:
            m_arr = np.asarray(initial_eigenmode, float).reshape((len(atoms), 3))
            norm = float(np.linalg.norm(m_arr))
            if norm > 1e-6:
                run_state.last_eigenmode = m_arr / norm
        except Exception:
            pass

    sources: list[Any] = []
    if staged_seed_file and Path(staged_seed_file).is_file():
        sources.append(Path(staged_seed_file))
    for s in (prior_trajectories or []):
        if s not in sources:
            sources.append(s)
    for cand in (
        out_p / "seed_training.extxyz",
        out_p.parent / "seed_training.extxyz",
        out_p / "seed_training.traj",
        out_p / "input_band.extxyz",
        out_p.parent / "input_band.extxyz",
    ):
        if cand.is_file() and cand not in sources:
            sources.append(cand)

    seeded = seed_dimer_training_set(
        sources,
        atoms,
        trainingset_path,
        state=run_state,
    )

    try:
        from catlearn.activelearning.local import LocalAL
        catlearn_available = True
    except ImportError:
        catlearn_available = False

    def _log(msg: str):
        if workflow_log_path:
            try:
                with open(workflow_log_path, "a", encoding="utf-8") as h:
                    h.write(f"[{Path(__file__).stem}] {msg}\n")
            except Exception:
                pass

    pred_traj_path = out_p / "predicted.traj"
    summary_txt_path = out_p / "ml_summary.txt"

    if run_state.seeded_count < 3:
        run_state.surrogate_status = "skipped_missing_seeds"
        run_state.fallback_reason = f"insufficient training seeds ({run_state.seeded_count} < 3)"
        _log(f"Insufficient training seeds ({run_state.seeded_count} < 3); proceeding with direct DFT Dimer.")
    elif not catlearn_available:
        run_state.surrogate_status = "skipped_no_catlearn"
        run_state.fallback_reason = "CatLearn not installed in environment"
        _log("CatLearn not available in environment; proceeding with direct DFT Dimer.")
    else:
        can_restart_traj = pred_traj_path.is_file() and pred_traj_path.stat().st_size > 0
        can_restart_state = can_restart_traj and summary_txt_path.is_file() and summary_txt_path.stat().st_size > 0

        _log(
            f"Starting CatLearn surrogate Dimer (seeded={len(seeded)} frames, "
            f"restart_traj={can_restart_traj}, restart_state={can_restart_state})."
        )
        try:
            dyn = LocalAL(
                atoms=atoms,
                ase_calc=calc,
                local_opt=PersistentDimerLocalOpt,
                local_opt_kwargs={
                    "dimer_separation": dimer_separation,
                    "use_central_forces": use_central_forces,
                    "initial_eigenmode": run_state.last_eigenmode,
                    "state": run_state,
                },
                unc_convergence=float(max_unc),
                min_data=3,
                use_restart=can_restart_state,
                restart=can_restart_traj,
                trajectory=str(pred_traj_path),
                trainingset=str(trainingset_path),
                tabletxt=str(summary_txt_path),
            )
            dyn.run(fmax=fmax, max_unc=max_unc, steps=max_steps, ml_steps=ml_steps, dtrust=dtrust)
            run_state.surrogate_status = "completed"
            _log(f"CatLearn surrogate Dimer finished; starting DFT polish pass (max_steps={polish_steps}).")
        except Exception as exc:
            run_state.surrogate_status = "fallback_direct_dft"
            run_state.fallback_reason = f"{exc.__class__.__name__}: {exc}"
            _log(f"CatLearn surrogate Dimer raised {exc.__class__.__name__}: {exc}; proceeding with direct DFT Dimer.")
            if pred_traj_path.is_file() and pred_traj_path.stat().st_size > 0:
                try:
                    last_pred = read(str(pred_traj_path), index=-1)
                    if len(last_pred) == len(atoms):
                        atoms.set_positions(last_pred.positions)
                except Exception:
                    pass

    polish_steps_run = int(polish_steps if run_state.surrogate_status == "completed" else max_steps)
    polish_separation = float(0.0001 if run_state.surrogate_status == "completed" else dimer_separation)

    atoms.calc = calc
    d_control = DimerControl(
        eigenmode_method="dimer",
        use_central_forces=use_central_forces,
        dimer_separation=polish_separation,
        logfile=str(out_p / "dft_polish_control.log"),
    )
    eigenmodes = None
    if run_state.last_eigenmode is not None:
        try:
            m = np.asarray(run_state.last_eigenmode, float).reshape((len(atoms), 3))
            n = float(np.linalg.norm(m))
            if n > 1e-6:
                eigenmodes = [m / n]
        except Exception:
            eigenmodes = None

    min_mode = MinModeAtoms(atoms, control=d_control, eigenmodes=eigenmodes)
    if eigenmodes is None:
        min_mode.displace()

    opt = MinModeTranslate(min_mode, trajectory=str(out_p / "dft_polish.traj"), logfile=str(out_p / "dft_polish.log"))
    converged = bool(opt.run(fmax=fmax, steps=polish_steps_run))

    try:
        final_mode = min_mode.get_eigenmode()
        if final_mode is not None:
            run_state.last_eigenmode = np.asarray(final_mode, float).copy()
    except Exception:
        pass

    e = float(min_mode.atoms.get_potential_energy())
    f = min_mode.atoms.get_forces()
    curv = float(min_mode.get_curvature())
    final_mode_list = min_mode.get_eigenmode().tolist() if min_mode.get_eigenmode() is not None else []

    write(out_p / "transition_state_refined.xyz", min_mode.atoms, format="xyz")
    res = {
        "converged": converged,
        "energy_eV": e,
        "curvature": curv,
        "fmax": float(np.max(np.linalg.norm(f, axis=1))),
        "final_eigenmode": final_mode_list,
        "seeding_status": run_state.seeding_status,
        "seeded_count": run_state.seeded_count,
        "rejected_count": run_state.rejected_count,
        "rejection_details": run_state.rejection_details,
        "surrogate_status": run_state.surrogate_status,
        "fallback_reason": run_state.fallback_reason,
        "staged_seed_file": run_state.staged_seed_file,
    }
    min_mode.atoms.info["dimer_run_state"] = run_state.to_dict()
    return res
