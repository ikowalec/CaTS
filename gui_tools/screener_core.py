"""Core algorithm for trajectory partitioning, high-force retention, and similarity screening.

Curates combined .xyz / .extxyz trajectory datasets for MACE neural network potential
training, keeping trajectory endpoints, high-force barrier states, and geometrically
distinct intermediate frames while tagging standardized REF_energy and REF_forces.
"""
from __future__ import annotations

import csv
import os
import sys
from math import isfinite
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import iread, write


DEFAULT_HIGH_FORCE_EV_A = 5.0
DEFAULT_HIGH_ENERGY_PER_ATOM_EV = 0.5
DEFAULT_SIMILARITY_THRESHOLD = 0.98
DEFAULT_MIN_PAIR_DISTANCE_RMSD_A = 0.08
DEFAULT_MIN_ENERGY_CHANGE_PER_ATOM_EV = 0.02
DEFAULT_MAX_INTERMEDIATE_FRAMES = 5
DEFAULT_MAX_FRAME_FMAX_EV_A = 50.0
DEFAULT_MAX_ABS_ENERGY_PER_ATOM_EV = 10000.0
DEFAULT_MAX_ENERGY_OUTLIER_PER_ATOM_EV = 100.0
DEFAULT_KERNEL_GAMMA = 1.0
DEFAULT_KERNEL_ALPHA = 1.0
DEFAULT_KERNEL_THRESHOLD = 1e-6

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from sklearn.preprocessing import normalize
    from similarity.rematch.get_rematch import get_cached_rematch_kernel
    from similarity.rematch.memory_rematch import (
        _build_shared_soap_descriptor,
        get_soap,
        suggest_similarity_threshold,
    )
    HAS_REMATCH = True
except Exception:
    HAS_REMATCH = False


def frame_energy_ev(atoms: Atoms) -> float | None:
    """Extract potential energy in eV from calculator or info tags."""
    if atoms.calc is not None:
        try:
            val = float(atoms.get_potential_energy())
            if isfinite(val):
                return val
        except Exception:
            pass
    for key in ("energy", "free_energy", "REF_energy", "ref_energy"):
        if key in atoms.info:
            try:
                val = float(atoms.info[key])
                if isfinite(val):
                    return val
            except (TypeError, ValueError):
                pass
    return None


def frame_forces_array(atoms: Atoms) -> np.ndarray | None:
    """Extract Cartesian forces (N, 3) in eV/A from calculator or arrays."""
    if atoms.calc is not None:
        try:
            f = np.asarray(atoms.get_forces(), dtype=float)
            if f.shape == (len(atoms), 3) and np.all(np.isfinite(f)):
                return f
        except Exception:
            pass
    for key in ("forces", "REF_forces", "ref_forces"):
        if key in atoms.arrays:
            try:
                f = np.asarray(atoms.arrays[key], dtype=float)
                if f.shape == (len(atoms), 3) and np.all(np.isfinite(f)):
                    return f
            except Exception:
                pass
    return None


def frame_fmax_ev_a(atoms: Atoms, forces: np.ndarray | None = None) -> float | None:
    """Calculate maximum atomic force magnitude in eV/A."""
    if forces is None:
        forces = frame_forces_array(atoms)
    if forces is None or len(forces) == 0:
        return None
    try:
        norms = np.linalg.norm(forces, axis=1)
        max_val = float(np.max(norms))
        return max_val if isfinite(max_val) else float("inf")
    except Exception:
        return None


def attach_calc_and_ref(
    atoms: Atoms,
    energy: float | None = None,
    forces: np.ndarray | None = None,
    tag_ref: bool = True,
) -> None:
    """Ensure Atoms has SinglePointCalculator and REF tags with clean extxyz roundtrip."""
    if energy is None:
        energy = frame_energy_ev(atoms)
    if forces is None:
        forces = frame_forces_array(atoms)

    if tag_ref:
        if energy is not None:
            atoms.info["REF_energy"] = float(energy)
        if forces is not None:
            atoms.set_array("REF_forces", np.asarray(forces, dtype=float).copy())

    if energy is not None or forces is not None:
        atoms.info.pop("energy", None)
        if "forces" in atoms.arrays:
            del atoms.arrays["forces"]
        kwargs = {}
        if energy is not None:
            kwargs["energy"] = float(energy)
        if forces is not None:
            kwargs["forces"] = np.asarray(forces, dtype=float).copy()
        atoms.calc = SinglePointCalculator(atoms, **kwargs)


def pair_distance_rmsd(a: Atoms, b: Atoms) -> float:
    """Compute all-atom pair distance root-mean-square deviation between two frames."""
    if len(a) != len(b) or a.get_chemical_symbols() != b.get_chemical_symbols():
        return float("inf")
    if len(a) < 2:
        return 0.0
    pbc_a = any(a.get_pbc())
    pbc_b = any(b.get_pbc())
    dist_a = a.get_all_distances(mic=pbc_a)
    dist_b = b.get_all_distances(mic=pbc_b)
    diff = dist_a - dist_b
    return float(np.sqrt(np.mean(diff * diff)))


SUPPORTED_TRAJECTORY_EXTENSIONS: tuple[str, ...] = (".xyz", ".extxyz", ".traj", ".db")


def discover_trajectory_files(path: str | Path, exclude_path: str | Path | None = None) -> list[Path]:
    """Find all supported trajectory/database files (.xyz, .extxyz, .traj, .db) from file or directory."""
    p = Path(path).resolve()
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []

    exclude_res = Path(exclude_path).resolve() if exclude_path else None
    discovered: list[Path] = []
    for candidate in sorted(p.rglob("*")):
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_TRAJECTORY_EXTENSIONS:
            if exclude_res and candidate.resolve() == exclude_res:
                continue
            discovered.append(candidate.resolve())
    return discovered


def _inspect_single_file(path: Path, sample_limit: int = 50) -> dict:
    """Inspect properties of a single trajectory or database file."""
    file_size_bytes = path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)

    total_frames = 0
    all_symbols = set()
    sample_formulas = []
    suffix = path.suffix.lower()

    if suffix == ".traj":
        from ase.io.trajectory import Trajectory

        traj = Trajectory(str(path), mode="r")
        total_frames = len(traj)
        for idx in range(min(sample_limit, total_frames)):
            atoms = traj[idx]
            all_symbols.update(atoms.get_chemical_symbols())
            formula = atoms.get_chemical_formula()
            if formula not in sample_formulas and len(sample_formulas) < 5:
                sample_formulas.append(formula)
    elif suffix == ".db":
        import ase.db

        db = ase.db.connect(str(path))
        total_frames = len(db)
        for idx, row in enumerate(db.select(limit=sample_limit)):
            atoms = row.toatoms()
            all_symbols.update(atoms.get_chemical_symbols())
            formula = atoms.get_chemical_formula()
            if formula not in sample_formulas and len(sample_formulas) < 5:
                sample_formulas.append(formula)
    else:
        try:
            for idx, atoms in enumerate(iread(str(path), format="extxyz")):
                total_frames += 1
                if idx < sample_limit:
                    all_symbols.update(atoms.get_chemical_symbols())
                    formula = atoms.get_chemical_formula()
                    if formula not in sample_formulas and len(sample_formulas) < 5:
                        sample_formulas.append(formula)
        except Exception:
            total_frames = 0
            all_symbols = set()
            for idx, atoms in enumerate(iread(str(path))):
                total_frames += 1
                if idx < sample_limit:
                    all_symbols.update(atoms.get_chemical_symbols())
                    formula = atoms.get_chemical_formula()
                    if formula not in sample_formulas and len(sample_formulas) < 5:
                        sample_formulas.append(formula)

    return {
        "path": str(path),
        "file_name": path.name,
        "is_directory": False,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_mb,
        "total_frames": total_frames,
        "elements": sorted(all_symbols),
        "sample_formulas": sample_formulas,
    }


def inspect_dataset(path: str | Path) -> dict:
    """Fast inspection of input .xyz / .extxyz / .traj / .db file or folder filled with trajectories."""
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    if p.is_dir():
        files = discover_trajectory_files(p)
        total_size = sum(f.stat().st_size for f in files)
        file_size_mb = total_size / (1024 * 1024)
        total_frames = 0
        all_symbols = set()
        sample_formulas = []
        for f in files:
            try:
                single_info = _inspect_single_file(f, sample_limit=10)
                total_frames += single_info["total_frames"]
                all_symbols.update(single_info["elements"])
                for formula in single_info["sample_formulas"]:
                    if formula not in sample_formulas and len(sample_formulas) < 5:
                        sample_formulas.append(formula)
            except Exception:
                pass
        return {
            "path": str(p),
            "file_name": p.name,
            "is_directory": True,
            "file_count": len(files),
            "file_size_bytes": total_size,
            "file_size_mb": file_size_mb,
            "total_frames": total_frames,
            "elements": sorted(all_symbols),
            "sample_formulas": sample_formulas,
        }

    return _inspect_single_file(p)


def load_dataset_images(path: str | Path) -> list[Atoms]:
    """Load structures from .xyz, .extxyz, .traj, or .db format."""
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix == ".traj":
        from ase.io import read

        return list(read(str(path), index=":"))
    if suffix == ".db":
        import ase.db

        db = ase.db.connect(str(path))
        return [row.toatoms() for row in db.select()]
    try:
        return list(iread(str(path), format="extxyz"))
    except Exception:
        return list(iread(str(path)))


def write_dataset_images(path: str | Path, images: list[Atoms]) -> None:
    """Save screened structures to .xyz (extxyz), .traj, or .db format."""
    path = Path(path).resolve()
    suffix = path.suffix.lower()

    for atoms in images:
        if atoms.calc is None:
            e = frame_energy_ev(atoms)
            f = frame_forces_array(atoms)
            if e is not None or f is not None:
                attach_calc_and_ref(atoms, energy=e, forces=f, tag_ref=False)

    if suffix == ".traj":
        from ase.io.trajectory import Trajectory

        traj = Trajectory(str(path), mode="w")
        for atoms in images:
            traj.write(atoms)
        traj.close()
    elif suffix == ".db":
        import ase.db

        db = ase.db.connect(str(path))
        for atoms in images:
            db.write(atoms)
    else:
        write(str(path), images, format="extxyz")


def partition_trajectories(
    images: Iterable[Atoms],
    frame_key: str = "source_frame",
    start_frame: int = 0,
    discontinuity_rmsd_a: float = 2.5,
) -> Iterator[list[Atoms]]:
    """Partition contiguous stream of Atoms into discrete chemical reaction/relaxation trajectories."""
    current: list[Atoms] = []
    prev_frame_num: int | None = None

    for atoms in images:
        if not current:
            current.append(atoms)
            val = atoms.info.get(frame_key)
            try:
                prev_frame_num = int(val) if val is not None else None
            except Exception:
                prev_frame_num = None
            continue

        prev_atoms = current[-1]
        is_boundary = False

        # 1. Chemical identity boundary
        if prev_atoms.get_chemical_formula() != atoms.get_chemical_formula() or len(prev_atoms) != len(atoms):
            is_boundary = True

        # 2. Metadata key boundary (e.g. source_frame resets)
        if not is_boundary and frame_key in atoms.info:
            try:
                curr_frame_num = int(atoms.info[frame_key])
                if curr_frame_num == start_frame and len(current) > 1:
                    is_boundary = True
                elif prev_frame_num is not None and curr_frame_num <= prev_frame_num:
                    is_boundary = True
                prev_frame_num = curr_frame_num
            except Exception:
                pass

        # 3. Explicit trajectory ID metadata
        if not is_boundary:
            for tid_key in ("trajectory_id", "traj_id", "run_uid", "config_type"):
                if tid_key in atoms.info and tid_key in prev_atoms.info:
                    if atoms.info[tid_key] != prev_atoms.info[tid_key]:
                        is_boundary = True
                        break

        # 4. Large geometric discontinuity jump
        if not is_boundary and len(atoms) == len(prev_atoms):
            if pair_distance_rmsd(prev_atoms, atoms) >= discontinuity_rmsd_a:
                is_boundary = True

        if is_boundary:
            yield current
            current = [atoms]
            val = atoms.info.get(frame_key)
            try:
                prev_frame_num = int(val) if val is not None else None
            except Exception:
                prev_frame_num = None
        else:
            current.append(atoms)

    if current:
        yield current


def suggest_trajectory_threshold(
    soaps: dict[int, np.ndarray],
    sensible_indices: list[int],
    rematch_kernel,
    fallback_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> float:
    """Calculate trajectory-specific dynamic REMatch similarity threshold via elbow/percentile heuristic."""
    if len(sensible_indices) <= 2 or rematch_kernel is None:
        return fallback_threshold

    ref_idx = sensible_indices[-1]
    if ref_idx not in soaps:
        return fallback_threshold

    ref_soap = soaps[ref_idx]
    sims = []
    for idx in sensible_indices[:-1]:
        if idx in soaps:
            sim = float(rematch_kernel.create([ref_soap, soaps[idx]])[0, 1])
            sims.append(sim)

    if not sims:
        return fallback_threshold

    values = np.sort(np.asarray(sims, dtype=float))[::-1]
    if len(values) == 1:
        return float(values[0])

    # Elbow gap heuristic from similarity.rematch.memory_rematch.suggest_similarity_threshold
    search_end = min(len(values) - 1, max(3, int(np.ceil(0.5 * len(values)))))
    diffs = values[:search_end] - values[1 : search_end + 1]
    best_idx = int(np.argmax(diffs))
    best_gap = float(diffs[best_idx])
    suggested = float(0.5 * (values[best_idx] + values[best_idx + 1]))
    if best_gap < 1e-3:
        suggested = float(np.percentile(values, 95))

    return float(min(1.0 - 1e-6, max(0.0, suggested)))


def screen_trajectory(
    trajectory: list[Atoms],
    *,
    high_force_ev_a: float = DEFAULT_HIGH_FORCE_EV_A,
    high_energy_per_atom_ev: float = DEFAULT_HIGH_ENERGY_PER_ATOM_EV,
    similarity_threshold: float | str | None = "dynamic",
    min_pair_distance_rmsd_a: float = DEFAULT_MIN_PAIR_DISTANCE_RMSD_A,
    min_energy_change_per_atom_ev: float = DEFAULT_MIN_ENERGY_CHANGE_PER_ATOM_EV,
    max_intermediate_frames: int = DEFAULT_MAX_INTERMEDIATE_FRAMES,
    max_frame_fmax_ev_a: float = DEFAULT_MAX_FRAME_FMAX_EV_A,
    max_abs_energy_per_atom_ev: float = DEFAULT_MAX_ABS_ENERGY_PER_ATOM_EV,
    max_energy_outlier_per_atom_ev: float = DEFAULT_MAX_ENERGY_OUTLIER_PER_ATOM_EV,
    kernel_gamma: float = DEFAULT_KERNEL_GAMMA,
    kernel_alpha: float = DEFAULT_KERNEL_ALPHA,
    kernel_threshold: float = DEFAULT_KERNEL_THRESHOLD,
    shared_soap=None,
) -> tuple[list[int], list[str], dict]:
    """Select indices to keep from a trajectory based on force, similarity, and endpoints."""
    if not trajectory:
        return [], [], {
            "high_force": 0,
            "endpoints": 0,
            "intermediate": 0,
            "threshold_used": 0.98,
            "rejected_similarity": 0,
            "rejected_cap": 0,
            "rejected_energy": 0,
            "rejected_force": 0,
            "total_rejected": 0,
        }

    # 1. Filter sensible frames
    sensible_indices = []
    rejected_reasons = []
    energies_per_atom: dict[int, float | None] = {}
    rej_similarity = 0
    rej_cap = 0
    rej_energy = 0
    rej_force = 0

    # Calculate median energy per atom as reference
    valid_e_per_atom = []
    for atoms in trajectory:
        e = frame_energy_ev(atoms)
        if e is not None and len(atoms) > 0 and isfinite(e):
            val = e / len(atoms)
            if abs(val) <= max_abs_energy_per_atom_ev:
                valid_e_per_atom.append(val)
    ref_energy_per_atom = float(np.median(valid_e_per_atom)) if valid_e_per_atom else None

    for idx, atoms in enumerate(trajectory):
        fmax = frame_fmax_ev_a(atoms)
        if fmax is not None and (not isfinite(fmax) or fmax > max_frame_fmax_ev_a):
            rej_force += 1
            rejected_reasons.append(f"Frame {idx}: force diverged ({fmax:.1f} eV/A)")
            continue

        e = frame_energy_ev(atoms)
        if e is not None and len(atoms) > 0:
            if not isfinite(e):
                rej_energy += 1
                rejected_reasons.append(f"Frame {idx}: non-finite energy")
                continue
            e_pa = e / len(atoms)
            if abs(e_pa) > max_abs_energy_per_atom_ev:
                rej_energy += 1
                rejected_reasons.append(f"Frame {idx}: extreme energy {e_pa:.1f} eV/atom")
                continue
            if ref_energy_per_atom is not None and abs(e_pa - ref_energy_per_atom) > max_energy_outlier_per_atom_ev:
                rej_energy += 1
                rejected_reasons.append(f"Frame {idx}: energy outlier")
                continue
            energies_per_atom[idx] = e_pa
        else:
            energies_per_atom[idx] = None

        sensible_indices.append(idx)

    if not sensible_indices:
        return [], rejected_reasons, {
            "high_force": 0,
            "endpoints": 0,
            "intermediate": 0,
            "threshold_used": 0.98,
            "rejected_similarity": 0,
            "rejected_cap": 0,
            "rejected_energy": rej_energy,
            "rejected_force": rej_force,
            "total_rejected": rej_energy + rej_force,
        }

    # If 1 or 2 sensible frames, keep all
    if len(sensible_indices) <= 2:
        return sensible_indices, rejected_reasons, {
            "high_force": 0,
            "endpoints": len(sensible_indices),
            "intermediate": 0,
            "threshold_used": 0.98,
            "rejected_similarity": 0,
            "rejected_cap": 0,
            "rejected_energy": rej_energy,
            "rejected_force": rej_force,
            "total_rejected": rej_energy + rej_force,
        }

    # Precompute SOAPs for sensible frames if REMatch available
    soaps: dict[int, np.ndarray] = {}
    rematch_kernel = None
    if HAS_REMATCH and len(sensible_indices) > 2:
        try:
            if shared_soap is None:
                shared_soap = _build_shared_soap_descriptor([trajectory[i] for i in sensible_indices])
            for idx in sensible_indices:
                raw_soap = get_soap(trajectory[idx], soap_descriptor=shared_soap)
                soaps[idx] = normalize(raw_soap)
            rematch_kernel = get_cached_rematch_kernel(
                gamma=kernel_gamma, alpha=kernel_alpha, threshold=kernel_threshold
            )
        except Exception:
            soaps = {}
            rematch_kernel = None

    # Determine dynamic per-trajectory similarity threshold or use fixed value
    if similarity_threshold is None or similarity_threshold == "dynamic":
        traj_threshold = suggest_trajectory_threshold(
            soaps, sensible_indices, rematch_kernel, fallback_threshold=DEFAULT_SIMILARITY_THRESHOLD
        )
    else:
        traj_threshold = float(similarity_threshold)

    # Compute energy floor
    valid_sensible_energies = [
        energies_per_atom[i] for i in sensible_indices if energies_per_atom.get(i) is not None
    ]
    energy_floor = min(valid_sensible_energies) if valid_sensible_energies else None

    # Endpoints
    endpoints_set = {sensible_indices[0], sensible_indices[-1]}
    unusual_set = set()
    ordinary_list = []
    kept_so_far = [sensible_indices[0]]

    prev_idx = sensible_indices[0]
    prev_energy_pa = energies_per_atom.get(prev_idx)

    for idx in sensible_indices[1:-1]:
        atoms = trajectory[idx]
        e_pa = energies_per_atom.get(idx)
        fmax = frame_fmax_ev_a(atoms)

        # Check high-force or high-energy
        is_unusual = (fmax is not None and fmax >= high_force_ev_a) or (
            energy_floor is not None and e_pa is not None and (e_pa - energy_floor) >= high_energy_per_atom_ev
        )

        if is_unusual:
            unusual_set.add(idx)
            kept_so_far.append(idx)
            prev_idx = idx
            prev_energy_pa = e_pa
            continue

        # Similarity check against already-kept structures using dynamic per-trajectory threshold
        is_duplicate = False
        if rematch_kernel is not None and idx in soaps:
            max_sim = max(
                (float(rematch_kernel.create([soaps[idx], soaps[k]])[0, 1]) for k in kept_so_far if k in soaps),
                default=0.0,
            )
            if max_sim >= traj_threshold:
                is_duplicate = True
        else:
            # Fallback to pair-distance RMSD if REMatch is unavailable
            rmsd = pair_distance_rmsd(trajectory[prev_idx], atoms)
            if rmsd < min_pair_distance_rmsd_a:
                is_duplicate = True

        # Allow energy variation to retain state if electronic difference is significant
        if is_duplicate and prev_energy_pa is not None and e_pa is not None:
            if abs(e_pa - prev_energy_pa) >= min_energy_change_per_atom_ev:
                is_duplicate = False

        if not is_duplicate and len(ordinary_list) < max_intermediate_frames:
            ordinary_list.append(idx)
            kept_so_far.append(idx)
            prev_idx = idx
            prev_energy_pa = e_pa
        elif is_duplicate:
            rej_similarity += 1
            rejected_reasons.append(f"Frame {idx}: high similarity duplicate")
        else:
            rej_cap += 1
            rejected_reasons.append(f"Frame {idx}: max intermediate frames cap reached")

    kept_indices = sorted(endpoints_set | unusual_set | set(ordinary_list))
    stats = {
        "endpoints": len(endpoints_set),
        "high_force": len(unusual_set),
        "intermediate": len(ordinary_list),
        "threshold_used": traj_threshold,
        "rejected_similarity": rej_similarity,
        "rejected_cap": rej_cap,
        "rejected_energy": rej_energy,
        "rejected_force": rej_force,
        "total_rejected": rej_similarity + rej_cap + rej_energy + rej_force,
    }
    return kept_indices, rejected_reasons, stats


def run_screening_workflow(
    input_path: str | Path,
    output_path: str | Path,
    *,
    audit_csv_path: str | Path | None = None,
    high_force_ev_a: float = DEFAULT_HIGH_FORCE_EV_A,
    high_energy_per_atom_ev: float = DEFAULT_HIGH_ENERGY_PER_ATOM_EV,
    similarity_threshold: float | str | None = "dynamic",
    min_pair_distance_rmsd_a: float = DEFAULT_MIN_PAIR_DISTANCE_RMSD_A,
    min_energy_change_per_atom_ev: float = DEFAULT_MIN_ENERGY_CHANGE_PER_ATOM_EV,
    max_intermediate_frames: int = DEFAULT_MAX_INTERMEDIATE_FRAMES,
    frame_key: str = "source_frame",
    tag_ref_properties: bool = True,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict:
    """Execute complete screening pipeline on trajectory file or directory of trajectory files."""
    in_path = Path(input_path).resolve()
    out_path = Path(output_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        raise FileNotFoundError(f"Input path not found: {in_path}")

    files = discover_trajectory_files(in_path, exclude_path=out_path)
    if not files:
        raise ValueError(f"No trajectory files (.xyz, .extxyz, .traj, .db) found in {in_path}")

    dataset_info = inspect_dataset(in_path)
    total_input_frames = dataset_info["total_frames"]

    if progress_callback:
        progress_callback(0, total_input_frames, f"Discovered {len(files)} file(s). Starting screening...")

    kept_atoms_list: list[Atoms] = []
    audit_rows = []
    total_high_force = 0
    total_endpoints = 0
    total_intermediate = 0
    total_rej_similarity = 0
    total_rej_cap = 0
    total_rej_energy = 0
    total_rej_force = 0
    global_traj_idx = 0
    processed_frames = 0
    total_files = len(files)

    for file_idx, file_path in enumerate(files, start=1):
        try:
            images = load_dataset_images(file_path)
        except Exception as exc:
            audit_rows.append(
                {
                    "source_file": file_path.name,
                    "trajectory_id": "-",
                    "formula": "Error",
                    "input_frames": 0,
                    "kept_frames": 0,
                    "threshold_used": "N/A",
                    "high_force_kept": 0,
                    "endpoints_kept": 0,
                    "intermediate_kept": 0,
                    "rejected_similarity": 0,
                    "rejected_cap": 0,
                    "rejected_energy": 0,
                    "rejected_force": 0,
                    "kept_indices": "",
                    "rejected_reasons": f"Failed to load file: {exc}",
                }
            )
            continue

        if not images:
            continue

        # Partition this file's images into discrete optimization trajectories
        file_trajectories = list(partition_trajectories(images, frame_key=frame_key))
        num_file_trajectories = len(file_trajectories)

        for local_traj_idx, traj in enumerate(file_trajectories, start=1):
            global_traj_idx += 1
            if progress_callback:
                msg = (
                    f"File {file_idx}/{total_files} ({file_path.name}) | "
                    f"Trajectory bin {local_traj_idx}/{num_file_trajectories} ({len(traj)} frames)..."
                )
                progress_callback(processed_frames, total_input_frames, msg)

            # Trajectory-resolved screening: similarity & thresholding are strictly confined to this optimization trajectory
            indices, rejected, stats = screen_trajectory(
                traj,
                high_force_ev_a=high_force_ev_a,
                high_energy_per_atom_ev=high_energy_per_atom_ev,
                similarity_threshold=similarity_threshold,
                min_pair_distance_rmsd_a=min_pair_distance_rmsd_a,
                min_energy_change_per_atom_ev=min_energy_change_per_atom_ev,
                max_intermediate_frames=max_intermediate_frames,
            )

            total_high_force += stats["high_force"]
            total_endpoints += stats["endpoints"]
            total_intermediate += stats["intermediate"]
            total_rej_similarity += stats.get("rejected_similarity", 0)
            total_rej_cap += stats.get("rejected_cap", 0)
            total_rej_energy += stats.get("rejected_energy", 0)
            total_rej_force += stats.get("rejected_force", 0)

            for idx in indices:
                src_atoms = traj[idx]
                energy = frame_energy_ev(src_atoms)
                forces = frame_forces_array(src_atoms)

                atoms = src_atoms.copy()
                atoms.info["source_file"] = file_path.name
                atoms.info["trajectory_bin"] = global_traj_idx
                attach_calc_and_ref(atoms, energy=energy, forces=forces, tag_ref=tag_ref_properties)
                kept_atoms_list.append(atoms)

            formula = traj[0].get_chemical_formula() if traj else "Unknown"
            audit_rows.append(
                {
                    "source_file": file_path.name,
                    "trajectory_id": global_traj_idx,
                    "formula": formula,
                    "input_frames": len(traj),
                    "kept_frames": len(indices),
                    "threshold_used": f"{stats.get('threshold_used', 0.98):.6f}",
                    "high_force_kept": stats["high_force"],
                    "endpoints_kept": stats["endpoints"],
                    "intermediate_kept": stats["intermediate"],
                    "rejected_similarity": stats.get("rejected_similarity", 0),
                    "rejected_cap": stats.get("rejected_cap", 0),
                    "rejected_energy": stats.get("rejected_energy", 0),
                    "rejected_force": stats.get("rejected_force", 0),
                    "kept_indices": " ".join(map(str, indices)),
                    "rejected_reasons": "; ".join(rejected) if rejected else "none",
                }
            )
            processed_frames += len(traj)

    if progress_callback:
        progress_callback(total_input_frames, total_input_frames, f"Writing {len(kept_atoms_list)} structures...")

    # Write output structures
    write_dataset_images(out_path, kept_atoms_list)

    # Write audit CSV if requested
    if audit_csv_path:
        csv_p = Path(audit_csv_path).resolve()
        csv_p.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_p, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "source_file",
                    "trajectory_id",
                    "formula",
                    "input_frames",
                    "kept_frames",
                    "threshold_used",
                    "high_force_kept",
                    "endpoints_kept",
                    "intermediate_kept",
                    "rejected_similarity",
                    "rejected_cap",
                    "rejected_energy",
                    "rejected_force",
                    "kept_indices",
                    "rejected_reasons",
                ],
            )
            writer.writeheader()
            writer.writerows(audit_rows)

    # Compute savings metrics
    in_bytes = dataset_info["file_size_bytes"]
    out_bytes = out_path.stat().st_size if out_path.is_file() else 0
    in_mb = dataset_info["file_size_mb"]
    out_mb = out_bytes / (1024 * 1024)

    initial_count = total_input_frames
    final_count = len(kept_atoms_list)
    filtered_count = initial_count - final_count
    structure_savings_pct = ((initial_count - final_count) / initial_count * 100.0) if initial_count > 0 else 0.0
    file_savings_pct = ((in_bytes - out_bytes) / in_bytes * 100.0) if in_bytes > 0 else 0.0
    speedup_factor = (initial_count / max(1, final_count))

    return {
        "status": "success",
        "input_path": str(in_path),
        "output_path": str(out_path),
        "audit_csv_path": str(audit_csv_path) if audit_csv_path else None,
        "initial_count": initial_count,
        "final_count": final_count,
        "filtered_count": filtered_count,
        "structure_savings_pct": structure_savings_pct,
        "initial_size_mb": in_mb,
        "final_size_mb": out_mb,
        "file_savings_pct": file_savings_pct,
        "speedup_factor": speedup_factor,
        "files_count": len(files),
        "trajectories_count": global_traj_idx,
        "high_force_kept": total_high_force,
        "endpoints_kept": total_endpoints,
        "intermediate_kept": total_intermediate,
        "rejected_similarity": total_rej_similarity,
        "rejected_cap": total_rej_cap,
        "rejected_energy": total_rej_energy,
        "rejected_force": total_rej_force,
        "total_rejected": total_rej_similarity + total_rej_cap + total_rej_energy + total_rej_force,
    }
