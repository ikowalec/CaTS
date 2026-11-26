from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.data import atomic_numbers
from ase.db import connect
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.optimize import BFGS
from ase.visualize import view

from cats.working_directory.coadsorbate_generator import (
    adsorbate_integrity_preserved,
    combine_single_adsorbate_modes,
    adsorbates_bound_to_support,
    run_coadsorbate_generation,
)
from cats.working_directory.silicate_pore_adsorbate import (
    generate_silicate_pore_adsorbate_candidates,
)
try:
    from similarity.rematch.memory_rematch import (
        deduplicate_atoms_by_similarity,
        suggest_similarity_threshold,
    )
except Exception as _rematch_import_exc:
    deduplicate_atoms_by_similarity = None
    suggest_similarity_threshold = None
    _rematch_import_error = str(_rematch_import_exc)
else:
    _rematch_import_error = ""


DEFAULT_MACE_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "run" / "mace-omat-0-medium.model"
)
_MAX_SIMILARITY_THRESHOLD = 0.999999


def _support_relaxation_indices(
    atoms, origin_atom_index: int, neighbor_shells: int
) -> list[int]:
    origin_atom_index = int(origin_atom_index)
    neighbor_shells = int(neighbor_shells)
    if not 0 <= origin_atom_index < len(atoms):
        raise ValueError(f"Support relaxation origin index {origin_atom_index} is out of range.")
    if neighbor_shells < 0:
        raise ValueError("Support relaxation neighbor shells must be non-negative.")

    neighbors = NeighborList(
        natural_cutoffs(atoms, mult=1.15),
        skin=0.0,
        self_interaction=False,
        bothways=True,
    )
    neighbors.update(atoms)
    selected = {origin_atom_index}
    frontier = selected.copy()
    for _ in range(neighbor_shells):
        frontier = {
            int(neighbor)
            for atom_index in frontier
            for neighbor in neighbors.get_neighbors(atom_index)[0]
        } - selected
        selected.update(frontier)
    return sorted(selected)


def _coadsorption_interaction_energy(
    ab_energy_eV: float,
    *,
    support_energy_eV: float,
    a_complex_min_energy_eV: float,
    b_complex_min_energy_eV: float,
) -> float:
    return (
        float(ab_energy_eV)
        + float(support_energy_eV)
        - float(a_complex_min_energy_eV)
        - float(b_complex_min_energy_eV)
    )


def _coerce_db_value(value):
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _output_path(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_atoms_db(db_path: Path, atoms_list, metadata_rows=None) -> None:
    db_path = _output_path(db_path)
    with connect(str(db_path), serial=True) as db:
        for idx, atoms in enumerate(atoms_list):
            data = {}
            if metadata_rows is not None:
                row = metadata_rows[idx] if idx < len(metadata_rows) else {}
                data = {key: _coerce_db_value(value) for key, value in row.items()}
            db.write(atoms, data=data)


def _write_csv(df, path: str | Path) -> None:
    df.to_csv(_output_path(path), index=False)


def _log(verbose: bool, message: str) -> None:
    if verbose:
        print(message, flush=True)


def _fallback_similarity_report(atoms_list, *, threshold: float):
    rows = []
    kept_positions = []
    for idx, atoms in enumerate(atoms_list, start=1):
        best = None
        closest = None
        positions = np.asarray(atoms.positions, dtype=float)
        for kept_pos in kept_positions:
            other = np.asarray(atoms_list[kept_pos].positions, dtype=float)
            if len(other) != len(positions):
                continue
            rmsd = float(np.sqrt(np.mean(np.sum((positions - other) ** 2, axis=1))))
            similarity = 1.0 / (1.0 + rmsd)
            if best is None or similarity > best:
                best = similarity
                closest = kept_pos + 1
        keep = best is None or best < float(threshold)
        if keep:
            kept_positions.append(idx - 1)
        rows.append(
            {
                "row_id": idx,
                "kept": bool(keep),
                "max_similarity_to_kept": np.nan if best is None else best,
                "closest_kept_row_id": np.nan if closest is None else closest,
                "fallback_reason": _rematch_import_error,
            }
        )
    report = pd.DataFrame(rows)
    report.attrs["similarity_threshold"] = float(threshold)
    report.attrs["n_input"] = len(atoms_list)
    report.attrs["n_unique"] = len(kept_positions)
    return [atoms_list[pos] for pos in kept_positions], report


def _unfiltered_similarity_report(atoms_list):
    report = pd.DataFrame(
        {
            "row_id": range(1, len(atoms_list) + 1),
            "kept": True,
            "max_similarity_to_kept": np.nan,
            "closest_kept_row_id": np.nan,
        }
    )
    report.attrs.update(
        similarity_threshold=1.0,
        n_input=len(atoms_list),
        n_unique=len(atoms_list),
    )
    return list(atoms_list), report


def _deduplicate_atoms_safe(atoms_list, *, similarity_threshold: float, similarity_kwargs: dict):
    if float(similarity_threshold) == 1.0:
        return _unfiltered_similarity_report(atoms_list)
    if deduplicate_atoms_by_similarity is None:
        return _fallback_similarity_report(atoms_list, threshold=float(similarity_threshold))
    return deduplicate_atoms_by_similarity(
        atoms_list,
        similarity_threshold=float(similarity_threshold),
        **similarity_kwargs,
    )


def _deduplicate_batch_mpi_task(task, _calculator, settings):
    start, atoms_batch = task
    _, report = _deduplicate_atoms_safe(
        atoms_batch,
        similarity_threshold=float(settings["similarity_threshold"]),
        similarity_kwargs=dict(settings["similarity_kwargs"]),
    )
    return start, report


def _write_single_mode_dft_candidates(
    db_path: Path,
    atoms_list,
    metadata_rows,
    kept_row_ids,
    similarity_report,
    original_constraints,
) -> None:
    final_metadata = []
    for row_id in kept_row_ids:
        report_row = similarity_report.iloc[row_id - 1]
        meta = dict(metadata_rows[row_id - 1])
        meta["single_mode_max_similarity_to_kept"] = _coerce_db_value(
            report_row["max_similarity_to_kept"]
        )
        final_metadata.append(meta)
    dft_atoms = [atoms.copy() for atoms in atoms_list]
    for atoms in dft_atoms:
        atoms.set_constraint(deepcopy(original_constraints))
    _write_atoms_db(db_path, dft_atoms, final_metadata)


def _suggest_threshold_safe(atoms_list, reference_row_id: int, similarity_kwargs: dict):
    if suggest_similarity_threshold is None:
        df = pd.DataFrame(
            [{"row_id": idx, "similarity": 0.0} for idx in range(1, len(atoms_list) + 1)]
        )
        df.attrs["suggested_threshold"] = 0.999
        df.attrs["similarity_spread"] = {}
        df.attrs["largest_gap"] = {"index": 0, "gap": 0.0}
        df.attrs["reference_row_id"] = int(reference_row_id)
        return df
    try:
        return suggest_similarity_threshold(
            atoms_list=atoms_list,
            reference_row_id=reference_row_id,
            **similarity_kwargs,
        )
    except ValueError as exc:
        if "No comparison structures available" not in str(exc):
            raise
        fallback_kwargs = dict(similarity_kwargs)
        fallback_kwargs["exclude_self"] = False
        return suggest_similarity_threshold(
            atoms_list=atoms_list,
            reference_row_id=reference_row_id,
            **fallback_kwargs,
        )


def _relax_threshold_from_suggestion(suggested_threshold: float, relax_fraction: float) -> float:
    suggested = float(suggested_threshold)
    if not np.isfinite(suggested):
        raise ValueError(f"Suggested similarity threshold must be finite, got {suggested_threshold!r}.")

    suggested = max(0.0, suggested)
    relax = float(relax_fraction)
    if relax < 0.0:
        raise ValueError("Similarity relaxation must be non-negative.")
    relaxed = suggested * (1.0 + relax)
    return float(min(_MAX_SIMILARITY_THRESHOLD, relaxed))


def _resolve_duplicate_cutoff(
    atoms_list,
    *,
    reference_row_id: int,
    explicit_cutoff: float | None,
    suggested_cutoff_scale: float,
    similarity_kwargs: dict,
):
    if not 1 <= int(reference_row_id) <= len(atoms_list):
        raise ValueError(
            f"reference_row_id must be in the range 1..{len(atoms_list)}, "
            f"got {reference_row_id}."
        )
    if explicit_cutoff is not None and (
        not np.isfinite(float(explicit_cutoff))
        or not 0.0 <= float(explicit_cutoff) <= 1.0
    ):
        raise ValueError("Duplicate similarity cutoffs must be finite and in [0, 1].")
    if not np.isfinite(float(suggested_cutoff_scale)) or float(suggested_cutoff_scale) <= 0.0:
        raise ValueError("Suggested cutoff scales must be positive.")

    if explicit_cutoff is not None and float(explicit_cutoff) == 1.0:
        threshold_df = pd.DataFrame()
        threshold_df.attrs.update(suggested_threshold=1.0, similarity_spread={})
        return threshold_df, 1.0, 1.0, 1.0

    threshold_df = _suggest_threshold_safe(
        atoms_list,
        reference_row_id=int(reference_row_id),
        similarity_kwargs=similarity_kwargs,
    )
    suggested = float(threshold_df.attrs["suggested_threshold"])
    if not np.isfinite(suggested):
        raise ValueError("Suggested similarity cutoff must be finite.")
    suggested = max(0.0, suggested)
    uncapped = suggested * float(suggested_cutoff_scale)
    used = (
        float(explicit_cutoff)
        if explicit_cutoff is not None
        else min(_MAX_SIMILARITY_THRESHOLD, uncapped)
    )
    return threshold_df, suggested, float(uncapped), float(used)


def _deduplicate_atoms_in_batches(
    atoms_list,
    *,
    batch_size: int,
    similarity_threshold: float,
    similarity_kwargs: dict,
    verbose: bool = False,
    label: str = "Similarity",
):
    batch_size = int(batch_size)
    if float(similarity_threshold) == 1.0:
        return _unfiltered_similarity_report(atoms_list)
    if batch_size <= 0 or len(atoms_list) <= batch_size:
        return _deduplicate_atoms_safe(
            atoms_list,
            similarity_threshold=similarity_threshold,
            similarity_kwargs=similarity_kwargs,
        )

    from similarity.workflows.mace_pool import run_cpu_tasks

    tasks = [
        (start, atoms_list[start:min(start + batch_size, len(atoms_list))])
        for start in range(0, len(atoms_list), batch_size)
    ]
    batch_results = run_cpu_tasks(
        tasks,
        _deduplicate_batch_mpi_task,
        distributed_operation=(
            "similarity.workflows.coadsorbate_similarity:_deduplicate_batch_mpi_task"
        ),
        distributed_settings={
            "similarity_threshold": float(similarity_threshold),
            "similarity_kwargs": dict(similarity_kwargs),
        },
        label=label,
        verbose=verbose,
    )

    survivor_atoms = []
    survivor_row_ids = []
    batch_reports = []
    for start, report in sorted(batch_results, key=lambda result: result[0]):
        report = report.copy()
        report["batch_row_id"] = report["row_id"].astype(int)
        report["row_id"] = report["batch_row_id"] + start
        report["closest_kept_row_id"] = [
            np.nan if pd.isna(row_id) else int(row_id) + start
            for row_id in report["closest_kept_row_id"]
        ]
        report["batch_kept"] = report["kept"].astype(bool)
        batch_reports.append(report)
        for row_id in report.loc[report["batch_kept"], "row_id"].astype(int):
            survivor_row_ids.append(row_id)
            survivor_atoms.append(atoms_list[row_id - 1])

    if not survivor_atoms:
        report = pd.concat(batch_reports, ignore_index=True)
        report["kept"] = False
        report.attrs["similarity_threshold"] = float(similarity_threshold)
        report.attrs["n_input"] = len(atoms_list)
        report.attrs["n_unique"] = 0
        return [], report

    if len(survivor_atoms) > batch_size:
        _log(verbose, f"[{label}] {len(survivor_atoms)} survivors exceed batch size; skipping final sweep")
        report = pd.concat(batch_reports, ignore_index=True)
        report["recombine_row_id"] = np.nan
        report.loc[report["batch_kept"], "recombine_row_id"] = range(
            1, len(survivor_atoms) + 1
        )
        report["kept"] = report["batch_kept"]
        report.attrs["similarity_threshold"] = float(similarity_threshold)
        report.attrs["n_input"] = len(atoms_list)
        report.attrs["n_unique"] = len(survivor_atoms)
        return survivor_atoms, report

    _log(
        verbose,
        f"[{label}] Recombining {len(survivor_atoms)} batch survivors for final sweep",
    )
    final_atoms, final_report = _deduplicate_atoms_safe(
        survivor_atoms,
        similarity_threshold=similarity_threshold,
        similarity_kwargs=similarity_kwargs,
    )
    final_by_original = {}
    for _, row in final_report.iterrows():
        original_row_id = survivor_row_ids[int(row["row_id"]) - 1]
        final_by_original[original_row_id] = row

    report = pd.concat(batch_reports, ignore_index=True)
    report["recombine_row_id"] = np.nan
    report["kept"] = False
    for idx, original_row_id in enumerate(survivor_row_ids, start=1):
        final_row = final_by_original.get(original_row_id)
        if final_row is None:
            continue
        mask = report["row_id"].astype(int) == int(original_row_id)
        report.loc[mask, "recombine_row_id"] = idx
        report.loc[mask, "kept"] = bool(final_row["kept"])
        report.loc[mask, "max_similarity_to_kept"] = _coerce_db_value(
            final_row["max_similarity_to_kept"]
        )
        closest = final_row["closest_kept_row_id"]
        report.loc[mask, "closest_kept_row_id"] = (
            np.nan if pd.isna(closest) else survivor_row_ids[int(closest) - 1]
        )
    report.attrs["similarity_threshold"] = float(similarity_threshold)
    report.attrs["n_input"] = len(atoms_list)
    report.attrs["n_unique"] = len(final_atoms)
    return final_atoms, report


def _checkpoint_hash(atoms, settings: dict) -> str:
    payload = {
        "symbols": atoms.get_chemical_symbols(),
        "positions": np.asarray(atoms.positions, dtype=float).round(10).tolist(),
        "cell": np.asarray(atoms.cell, dtype=float).round(10).tolist(),
        "pbc": np.asarray(atoms.pbc, dtype=bool).tolist(),
        "settings": settings,
    }
    return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _adsorbate_groups(n_support_atoms: int, n_atoms: int, metadata: dict):
    a_count = metadata.get("a_adsorbate_n_atoms")
    b_count = metadata.get("b_adsorbate_n_atoms")
    if a_count is None or b_count is None:
        return None
    a_end = int(n_support_atoms) + int(a_count)
    b_end = a_end + int(b_count)
    if b_end != int(n_atoms):
        return None
    return range(int(n_support_atoms), a_end), range(a_end, b_end)


def _integrity_mpi_task(task, _calculator, settings):
    index, initial, relaxed, metadata = task
    n_support_atoms = int(settings["n_support_atoms"])
    groups = _adsorbate_groups(n_support_atoms, len(initial), metadata)
    return (
        index,
        adsorbate_integrity_preserved(
            initial, relaxed, n_support_atoms, adsorbate_groups=groups
        ),
        adsorbates_bound_to_support(
            relaxed, n_support_atoms, adsorbate_groups=groups
        ),
    )


def _filter_and_sort_relaxed(
    initial_atoms,
    relaxed_atoms,
    report_df,
    metadata,
    *,
    n_support_atoms: int,
    require_converged: bool,
    require_bound: bool,
    verbose: bool = False,
):
    sizes = {len(initial_atoms), len(relaxed_atoms), len(report_df), len(metadata)}
    if len(sizes) != 1:
        raise RuntimeError("Relaxed structures, reports, and metadata are misaligned.")
    from similarity.workflows.mace_pool import run_cpu_tasks

    integrity = {
        index: (intact, bound)
        for index, intact, bound in run_cpu_tasks(
            [
                (index, initial, relaxed, meta)
                for index, (initial, relaxed, meta) in enumerate(
                    zip(initial_atoms, relaxed_atoms, metadata)
                )
            ],
            _integrity_mpi_task,
            distributed_operation=(
                "similarity.workflows.coadsorbate_similarity:_integrity_mpi_task"
            ),
            distributed_settings={"n_support_atoms": int(n_support_atoms)},
            label="Adsorbate integrity",
            verbose=verbose,
        )
    }
    records = []
    for index, (initial, relaxed, (_, row), meta) in enumerate(
        zip(initial_atoms, relaxed_atoms, report_df.iterrows(), metadata)
    ):
        intact, bound = integrity[index]
        keep = intact and (not require_converged or bool(row["converged"])) and (
            not require_bound or bound
        )
        records.append((initial, relaxed, row.to_dict(), meta, intact, bound, keep))

    kept = [record for record in records if record[-1]]
    kept.sort(key=lambda record: float(record[1].get_potential_energy()))
    return records, kept


def _metadata_vector(metadata: dict, key: str) -> np.ndarray:
    value = metadata.get(key)
    if isinstance(value, str):
        vector = np.array([float(part) for part in value.split(",")], dtype=float)
    else:
        vector = np.asarray(value, dtype=float)
    if vector.shape != (3,):
        raise ValueError(f"{key} must contain three coordinates.")
    return vector


def _adsorbate_inside_original_pore(atoms, n_support_atoms: int, metadata: dict) -> tuple[bool, float]:
    adsorbate = atoms[int(n_support_atoms):]
    if not adsorbate:
        return False, float("nan")
    site = _metadata_vector(metadata, "site_position")
    pore_vector = _metadata_vector(metadata, "pore_vector")
    norm = float(np.linalg.norm(pore_vector))
    if norm <= 1.0e-12:
        return False, float("nan")
    pore_dot = float(np.dot(adsorbate.get_center_of_mass() - site, pore_vector / norm))
    return pore_dot > 0.0, pore_dot


def _prune_single_adsorbate_modes(
    atoms_list,
    *,
    reference_row_id: int,
    similarity_threshold: float | None,
    similarity_relaxation: float,
    similarity_kwargs: dict,
    batch_size: int = 200,
    verbose: bool = False,
    label: str = "Single-mode similarity",
):
    if similarity_threshold is not None:
        threshold = float(similarity_threshold)
    else:
        threshold_atoms = atoms_list[:batch_size] if int(batch_size) > 0 else atoms_list
        threshold_df = _suggest_threshold_safe(
            threshold_atoms,
            reference_row_id=min(int(reference_row_id), len(threshold_atoms)),
            similarity_kwargs=similarity_kwargs,
        )
        threshold = _relax_threshold_from_suggestion(
            float(threshold_df.attrs["suggested_threshold"]), similarity_relaxation
        )
    kept, report = _deduplicate_atoms_in_batches(
        atoms_list,
        batch_size=batch_size,
        similarity_threshold=threshold,
        similarity_kwargs=similarity_kwargs,
        verbose=verbose,
        label=label,
    )
    kept_row_ids = [
        int(row_id)
        for row_id in report.loc[report["kept"], "row_id"].tolist()
    ]
    return kept, threshold, kept_row_ids, report


def _probe_representative_rows(report) -> dict[int, int]:
    rows = {
        int(row["row_id"]): row
        for _, row in report.iterrows()
    }
    representatives = {}
    for row_id in rows:
        current = row_id
        visited = set()
        while not bool(rows[current]["kept"]):
            if current in visited:
                raise RuntimeError("Site-probe similarity report contains a cycle.")
            visited.add(current)
            closest = rows[current]["closest_kept_row_id"]
            if pd.isna(closest) or int(closest) not in rows:
                raise RuntimeError(
                    f"Site-probe row {row_id} has no retained representative."
                )
            current = int(closest)
        representatives[row_id] = current
    return representatives


def _expand_modes_to_probe_equivalents(
    modes,
    metadata_rows,
    *,
    n_support_atoms: int,
    representative_site_indices,
    probe_modes,
    probe_site_indices,
    probe_report,
):
    if len(modes) != len(metadata_rows):
        raise RuntimeError("Single modes and generation metadata are misaligned.")
    representatives = _probe_representative_rows(probe_report)
    equivalent_positions = {}
    for row_id, representative_row_id in representatives.items():
        representative_site = probe_site_indices[representative_row_id - 1]
        equivalent_positions.setdefault(representative_site, []).append(
            (
                probe_site_indices[row_id - 1],
                probe_modes[row_id - 1].positions[-1],
            )
        )

    expanded_modes = []
    for mode, metadata in zip(modes, metadata_rows):
        local_site_index = int(str(metadata["site_indices"]).split(",")[0])
        representative_site = representative_site_indices[local_site_index]
        for target_site, target_position in equivalent_positions.get(
            representative_site,
            [(representative_site, mode.positions[int(n_support_atoms)])],
        ):
            expanded = mode.copy()
            if target_site != representative_site:
                expanded.positions[int(n_support_atoms):] += (
                    target_position - expanded.positions[int(n_support_atoms)]
                )
            expanded.calc = None
            expanded_modes.append(expanded)
    return expanded_modes


def _screen_sites_with_probe(
    input_atoms,
    *,
    candidates_db_path,
    generation_kwargs,
    site_probe_symbol,
    site_probe_similarity_cutoff,
    similarity_kwargs,
    batch_size,
    report_path,
    verbose,
):
    probe_db = candidates_db_path.with_name(f"{candidates_db_path.stem}_site_probes.db")
    probe_db.unlink(missing_ok=True)
    single_generation_kwargs = dict(generation_kwargs)
    single_generation_kwargs.pop("allowed_site_indices", None)
    probe_modes = run_coadsorbate_generation(
        input_atoms,
        adsorbates=[Atoms(str(site_probe_symbol))],
        output_db=probe_db,
        **{
            **single_generation_kwargs,
            "rotation_step_degrees": 360.0,
            "include_flipped_orientations": False,
            "max_structures": None,
        },
    )
    if not probe_modes:
        raise RuntimeError("Site-probe screening found no bond-valid adsorption sites.")
    _, _, kept_rows, report = _prune_single_adsorbate_modes(
        probe_modes,
        reference_row_id=1,
        similarity_threshold=site_probe_similarity_cutoff,
        similarity_relaxation=0.0,
        similarity_kwargs=similarity_kwargs,
        batch_size=batch_size,
        verbose=verbose,
        label="Workflow site-probe similarity",
    )
    with connect(str(probe_db), serial=True) as db:
        db_rows = list(db.select())
    site_indices = [int(str(row.data.site_indices).split(",")[0]) for row in db_rows]
    report = report.copy()
    report["site_index"] = [site_indices[int(row_id) - 1] for row_id in report["row_id"]]
    report["placement"] = [
        str(db_rows[int(row_id) - 1].data.placements) for row_id in report["row_id"]
    ]
    report["closest_kept_site_index"] = [
        None if pd.isna(row_id) else site_indices[int(row_id) - 1]
        for row_id in report["closest_kept_row_id"]
    ]
    report["similarity_cutoff"] = float(site_probe_similarity_cutoff)
    _write_csv(report, report_path)
    kept_site_indices = [site_indices[row_id - 1] for row_id in kept_rows]
    if verbose:
        print(
            "[Workflow] Site-probe screening: "
            f"{len(probe_modes)} -> {len(kept_site_indices)} sites "
            f"(probe={site_probe_symbol}, cutoff={site_probe_similarity_cutoff:.6f})"
        )
        print(f"[Workflow] Site-probe report: {report_path.resolve()}")
    probe_db.unlink(missing_ok=True)
    return probe_modes, site_indices, report, kept_site_indices, single_generation_kwargs


def _relax_adsorbates_with_mace_serial(
    atoms_list,
    *,
    n_fixed_atoms: int,
    relaxed_support_indices=None,
    model_path: str,
    fmax: float,
    max_steps: int,
    device: str,
    dispersion: bool,
    default_dtype: str,
    enable_cueq: bool = True,
    verbose: bool = True,
    calculator=None,
):
    relaxed_atoms = []
    rows = []
    cueq_enabled = bool(enable_cueq)
    if calculator is None:
        from similarity.workflows.mace_pool import get_mace_calculators

        calculator = get_mace_calculators(
            model=str(model_path),
            dispersion=bool(dispersion),
            default_dtype=str(default_dtype),
            device=str(device),
            enable_cueq=cueq_enabled,
        )[0]
    if verbose:
        print(
            "[Workflow] Starting MACE relaxation "
            f"(model={Path(model_path)}, fmax={fmax}, max_steps={max_steps}, "
            f"device={device}, enable_cueq={cueq_enabled})"
        )
    n_structures = len(atoms_list)
    relaxed_support_indices = {
        int(index) for index in (relaxed_support_indices or ())
    }

    def _run_relaxation_loop():
        for row_id, atoms in enumerate(atoms_list, start=1):
            relaxed = atoms.copy()
            fixed_count = min(int(n_fixed_atoms), len(relaxed))
            fixed_indices = [
                index
                for index in range(fixed_count)
                if index not in relaxed_support_indices
            ]
            free_indices = sorted(relaxed_support_indices | set(range(fixed_count, len(relaxed))))
            relaxed.set_constraint(FixAtoms(indices=fixed_indices) if fixed_indices else [])

            relaxed.calc = calculator
            if verbose:
                print(
                    "[MACE] Relaxing structure "
                    f"{row_id}/{n_structures} (fixed={len(fixed_indices)}, free={len(free_indices)})"
                )

            started = time.perf_counter()
            opt = BFGS(relaxed, logfile=None)
            opt.run(fmax=fmax, steps=max_steps)

            energy = float(relaxed.get_potential_energy())
            forces = relaxed.get_forces()
            if free_indices:
                free_forces = forces[free_indices]
                max_force = float(np.max(np.linalg.norm(free_forces, axis=1)))
            else:
                max_force = 0.0

            nsteps = int(getattr(opt, "nsteps", 0))
            row_info = {
                "row_id": row_id,
                "n_atoms": len(relaxed),
                "n_fixed": len(fixed_indices),
                "n_relaxed": len(free_indices),
                "energy_eV": energy,
                "max_force_eV_A": max_force,
                "converged": bool(max_force <= fmax),
                "nsteps": nsteps,
                "force_evaluations": nsteps + 1,
                "elapsed_seconds": time.perf_counter() - started,
            }
            rows.append(row_info)

            # Freeze per-structure results before calculator is reused.
            frozen = relaxed.copy()
            frozen.calc = SinglePointCalculator(frozen, energy=energy, forces=forces)
            relaxed_atoms.append(frozen)

            if verbose:
                print(
                    "[MACE] Completed structure "
                    f"{row_id}/{n_structures}: energy={energy:.6f} eV, "
                    f"max_free_force={max_force:.6f} eV/A, steps={int(getattr(opt, 'nsteps', 0))}"
                )

    _run_relaxation_loop()
    return relaxed_atoms, pd.DataFrame(rows)


def _relax_adsorbate_mpi_task(task, calculator, settings):
    row_id, atoms = task
    relaxed, report = _relax_adsorbates_with_mace_serial(
        [atoms],
        n_fixed_atoms=int(settings["n_fixed_atoms"]),
        relaxed_support_indices=settings.get("relaxed_support_indices"),
        model_path=str(settings["model_path"]),
        fmax=float(settings["fmax"]),
        max_steps=int(settings["max_steps"]),
        device=str(settings["device"]),
        dispersion=bool(settings["dispersion"]),
        default_dtype=str(settings["default_dtype"]),
        enable_cueq=bool(settings["enable_cueq"]),
        verbose=False,
        calculator=calculator,
    )
    row = report.iloc[0].to_dict()
    row["row_id"] = row_id
    return row_id, relaxed[0], row


def _relax_adsorbates_with_mace(
    atoms_list,
    *,
    n_fixed_atoms: int,
    relaxed_support_indices=None,
    model_path: str,
    fmax: float,
    max_steps: int,
    device: str,
    dispersion: bool,
    default_dtype: str,
    enable_cueq: bool = True,
    output_db_path: Path | None = None,
    metadata_rows=None,
    verbose: bool = True,
):
    from similarity.workflows.mace_pool import (
        get_mace_calculators,
        run_mace_tasks,
        write_ase_database_atomically,
    )

    atoms_list = list(atoms_list)
    checkpoint_settings = {
        "model_path": str(Path(model_path)),
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "device": str(device),
        "dispersion": bool(dispersion),
        "default_dtype": str(default_dtype),
        "enable_cueq": bool(enable_cueq),
        "n_fixed_atoms": int(n_fixed_atoms),
        "relaxed_support_indices": sorted(
            int(index) for index in (relaxed_support_indices or ())
        ),
    }
    hashes = {
        row_id: _checkpoint_hash(atoms, checkpoint_settings)
        for row_id, atoms in enumerate(atoms_list, start=1)
    }
    checkpoints = {}
    checkpoint_records = []
    if output_db_path is not None:
        output_db_path = _output_path(output_db_path)
        if output_db_path.exists():
            with connect(str(output_db_path), serial=True) as checkpoint_db:
                checkpoint_rows = list(checkpoint_db.select())
                checkpoint_records = [
                    (row.toatoms(), dict(row.data)) for row in checkpoint_rows
                ]
                checkpoints = {
                    int(row.data.relaxed_row_id): row
                    for row in checkpoint_rows
                    if row.data.get("relaxed_row_id") is not None
                    and row.data.get("checkpoint_hash")
                    == hashes.get(int(row.data.relaxed_row_id))
                }

    results = {}
    for row_id, checkpoint in checkpoints.items():
        data = checkpoint.data
        checkpoint_atoms = checkpoint.toatoms()
        results[row_id] = (
            checkpoint_atoms,
            {
                "row_id": row_id,
                "n_atoms": len(checkpoint_atoms),
                "n_fixed": max(
                    0,
                    min(int(n_fixed_atoms), len(checkpoint_atoms))
                    - len(checkpoint_settings["relaxed_support_indices"]),
                ),
                "n_relaxed": max(0, len(checkpoint_atoms) - int(n_fixed_atoms))
                + len(checkpoint_settings["relaxed_support_indices"]),
                "energy_eV": float(data.energy_eV),
                "max_force_eV_A": float(data.max_force_eV_A),
                "converged": bool(data.converged),
                "nsteps": int(data.nsteps),
                "force_evaluations": int(
                    data.get("force_evaluations", int(data.nsteps) + 1)
                ),
                "elapsed_seconds": float(data.get("elapsed_seconds", 0.0)),
            },
        )

    pending = [
        (row_id, atoms)
        for row_id, atoms in enumerate(atoms_list, start=1)
        if row_id not in checkpoints
    ]
    calculators = get_mace_calculators(
        model=str(model_path),
        dispersion=bool(dispersion),
        default_dtype=str(default_dtype),
        device=str(device),
        enable_cueq=bool(enable_cueq),
    )

    worker_settings = {
        "n_fixed_atoms": int(n_fixed_atoms),
        "relaxed_support_indices": list(relaxed_support_indices or []),
        "model_path": str(model_path),
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "device": str(device),
        "dispersion": bool(dispersion),
        "default_dtype": str(default_dtype),
        "enable_cueq": bool(enable_cueq),
    }

    def relax_one(task, calculator):
        return _relax_adsorbate_mpi_task(task, calculator, worker_settings)

    checkpoint_interval = max(1, (len(atoms_list) + 9) // 10)
    saved_since_checkpoint = 0

    def publish_checkpoint():
        if output_db_path is not None:
            write_ase_database_atomically(
                output_db_path,
                checkpoint_records,
            )

    def save_result(result):
        nonlocal saved_since_checkpoint
        row_id, frozen, row = result
        results[row_id] = (frozen, row)
        if output_db_path is None:
            return
        data = {}
        if metadata_rows is not None and row_id - 1 < len(metadata_rows):
            data.update(
                {
                    key: _coerce_db_value(value)
                    for key, value in metadata_rows[row_id - 1].items()
                }
            )
        data.update(
            {
                "relaxed_row_id": row_id,
                "energy_eV": float(row["energy_eV"]),
                "max_force_eV_A": float(row["max_force_eV_A"]),
                "converged": bool(row["converged"]),
                "nsteps": int(row["nsteps"]),
                "force_evaluations": int(row["force_evaluations"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
                "checkpoint_hash": hashes[row_id],
            }
        )
        checkpoint_records.append((frozen, data))
        saved_since_checkpoint += 1
        if (
            saved_since_checkpoint >= checkpoint_interval
            or len(results) == len(atoms_list)
        ):
            publish_checkpoint()
            saved_since_checkpoint = 0

    run_mace_tasks(
        pending,
        calculators,
        relax_one,
        on_result=save_result,
        completed=len(checkpoints),
        total=len(atoms_list),
        label="MACE adsorbate relaxation",
        verbose=verbose,
        distributed_operation=(
            "similarity.workflows.coadsorbate_similarity:_relax_adsorbate_mpi_task"
        ),
        distributed_settings=worker_settings,
        calculator_settings={
            "model": str(model_path),
            "dispersion": bool(dispersion),
            "default_dtype": str(default_dtype),
            "device": str(device),
            "enable_cueq": bool(enable_cueq),
        },
    )
    if saved_since_checkpoint:
        publish_checkpoint()
    ordered = [results[row_id] for row_id in range(1, len(atoms_list) + 1)]
    return [atoms for atoms, _ in ordered], pd.DataFrame([row for _, row in ordered])


def run_silicate_pore_adsorbate_workflow(
    input_atoms=None,
    *,
    adsorbates,
    candidates_db_path: str | Path = "scratch/silicate_pore_candidates.db",
    pre_relax_db_path: str | Path = "scratch/silicate_pore_prefilter.db",
    relaxed_db_path: str | Path = "scratch/silicate_pore_relaxed.db",
    final_db_path: str | Path = "scratch/silicate_pore_final_for_dft.db",
    candidate_report_csv_path: str | Path = "scratch/silicate_pore_candidate_similarity.csv",
    relaxed_minimum_report_csv_path: str | Path = "scratch/silicate_pore_relaxed_similarity.csv",
    relaxation_report_csv_path: str | Path = "scratch/silicate_pore_relaxation_report.csv",
    summary_csv_path: str | Path = "scratch/silicate_pore_summary.csv",
    summary_txt_path: str | Path = "scratch/silicate_pore_summary.txt",
    candidate_reference_row_id: int = 1,
    relaxed_minimum_reference_row_id: int = 1,
    candidate_duplicate_cutoff: float | None = None,
    relaxed_minimum_duplicate_cutoff: float | None = None,
    candidate_suggested_cutoff_scale: float = 1.0,
    relaxed_minimum_suggested_cutoff_scale: float = 1.0,
    candidate_similarity_batch_size: int = 200,
    generation_kwargs: dict | None = None,
    similarity_kwargs: dict | None = None,
    mace_model_path: str | Path = DEFAULT_MACE_MODEL_PATH,
    mace_fmax: float = 0.05,
    mace_max_steps: int = 200,
    mace_device: str = "cpu",
    mace_dispersion: bool = False,
    mace_default_dtype: str = "float64",
    mace_enable_cueq: bool = True,
    require_converged: bool = True,
    show_final: bool = False,
    verbose: bool = True,
):
    if input_atoms is None:
        raise ValueError("Provide input_atoms.")
    generation_kwargs = dict(generation_kwargs or {})
    similarity_kwargs = dict(similarity_kwargs or {})
    similarity_kwargs.setdefault("support_species", ["Si", "O", "Al", "H"])
    original_constraints = deepcopy(input_atoms.constraints)

    _log(verbose, "[SilicatePore] Starting silicate pore adsorbate workflow")
    _log(verbose, f"[SilicatePore] Input atoms: {len(input_atoms)}")
    _log(verbose, f"[SilicatePore] Adsorbates requested: {len(adsorbates)}")
    _log(verbose, f"[SilicatePore] Candidates DB: {Path(candidates_db_path).resolve()}")
    _log(verbose, f"[SilicatePore] Generation options: {generation_kwargs}")
    _log(verbose, f"[SilicatePore] Similarity options: {similarity_kwargs}")

    generation_progress_kwargs = dict(generation_kwargs)
    if verbose and "progress_callback" not in generation_progress_kwargs:
        generation_progress_kwargs["progress_callback"] = lambda message: _log(True, message)

    candidates, candidate_metadata = generate_silicate_pore_adsorbate_candidates(
        input_atoms,
        adsorbates,
        **generation_progress_kwargs,
    )
    if not candidates:
        raise RuntimeError("Silicate pore placement produced no candidates.")
    _write_atoms_db(Path(candidates_db_path), candidates, candidate_metadata)
    _log(verbose, f"[SilicatePore] Generated candidates: {len(candidates)}")
    _log(verbose, f"[SilicatePore] Candidate DB written: {Path(candidates_db_path).resolve()}")

    batch_size = int(candidate_similarity_batch_size)
    threshold_candidates = candidates[:batch_size] if batch_size > 0 else candidates
    if candidate_duplicate_cutoff is None and len(threshold_candidates) < len(candidates):
        _log(
            verbose,
            "[SilicatePore] Suggesting candidate similarity cutoff from first "
            f"{len(threshold_candidates)} structures",
        )
    _log(
        verbose,
        "[SilicatePore] Screening generated candidates for similarity "
        f"in batches of {batch_size if batch_size > 0 else len(candidates)}",
    )
    primary_threshold_df, _suggested, _uncapped, primary_threshold = _resolve_duplicate_cutoff(
        threshold_candidates,
        reference_row_id=min(int(candidate_reference_row_id), len(threshold_candidates)),
        explicit_cutoff=candidate_duplicate_cutoff,
        suggested_cutoff_scale=float(candidate_suggested_cutoff_scale),
        similarity_kwargs=similarity_kwargs,
    )
    pre_relax_atoms, primary_report_df = _deduplicate_atoms_in_batches(
        candidates,
        batch_size=batch_size,
        similarity_threshold=primary_threshold,
        similarity_kwargs=similarity_kwargs,
        verbose=verbose,
        label="SilicatePore",
    )
    _write_csv(primary_report_df, candidate_report_csv_path)
    _log(
        verbose,
        "[SilicatePore] Candidate similarity complete: "
        f"{len(pre_relax_atoms)}/{len(candidates)} kept, threshold={primary_threshold:.6f}",
    )
    _log(verbose, f"[SilicatePore] Candidate report: {Path(candidate_report_csv_path).resolve()}")
    primary_kept_rows = primary_report_df[primary_report_df["kept"]].copy()
    pre_relax_metadata = []
    for _, row in primary_kept_rows.iterrows():
        source_row_id = int(row["row_id"])
        meta = dict(candidate_metadata[source_row_id - 1])
        meta["source_candidate_row_id"] = source_row_id
        meta["candidate_max_similarity_to_kept"] = _coerce_db_value(
            row["max_similarity_to_kept"]
        )
        pre_relax_metadata.append(meta)
    _write_atoms_db(Path(pre_relax_db_path), pre_relax_atoms, pre_relax_metadata)
    _log(verbose, f"[SilicatePore] Pre-relax DB: {Path(pre_relax_db_path).resolve()}")
    if not pre_relax_atoms:
        raise RuntimeError("Initial silicate-pore similarity screening removed all candidates.")

    relaxed_atoms, relaxation_report_df = _relax_adsorbates_with_mace(
        pre_relax_atoms,
        n_fixed_atoms=len(input_atoms),
        model_path=str(mace_model_path),
        fmax=float(mace_fmax),
        max_steps=int(mace_max_steps),
        device=str(mace_device),
        dispersion=bool(mace_dispersion),
        default_dtype=str(mace_default_dtype),
        enable_cueq=bool(mace_enable_cueq),
        output_db_path=Path(relaxed_db_path),
        metadata_rows=pre_relax_metadata,
        verbose=verbose,
    )
    relaxation_records, kept_relaxed_records = _filter_and_sort_relaxed(
        pre_relax_atoms,
        relaxed_atoms,
        relaxation_report_df,
        pre_relax_metadata,
        n_support_atoms=len(input_atoms),
        require_converged=bool(require_converged),
        require_bound=False,
        verbose=verbose,
    )
    relaxation_report_df["adsorbate_integrity_preserved"] = [
        record[4] for record in relaxation_records
    ]
    pore_status = [
        _adsorbate_inside_original_pore(record[1], len(input_atoms), record[3])
        for record in relaxation_records
    ]
    relaxation_report_df["inside_original_pore"] = [
        inside for inside, _ in pore_status
    ]
    relaxation_report_df["relaxed_pore_dot_com_minus_site"] = [
        pore_dot for _, pore_dot in pore_status
    ]
    relaxation_report_df["kept_for_final_similarity"] = [
        record[6] and pore_status[idx][0]
        for idx, record in enumerate(relaxation_records)
    ]
    _write_csv(relaxation_report_df, relaxation_report_csv_path)
    kept_relaxed_records = [
        record
        for idx, record in enumerate(relaxation_records)
        if record[6] and pore_status[idx][0]
    ]
    kept_relaxed_records.sort(key=lambda record: float(record[1].get_potential_energy()))
    if not kept_relaxed_records:
        raise RuntimeError("MACE relaxation removed all candidates after integrity/convergence/pore-side filtering.")
    _log(
        verbose,
        "[SilicatePore] MACE relaxation screen complete: "
        f"{len(kept_relaxed_records)}/{len(pre_relax_atoms)} intact in-pore candidates kept",
    )
    _log(verbose, f"[SilicatePore] Relaxation report: {Path(relaxation_report_csv_path).resolve()}")

    relaxed_atoms = [record[1] for record in kept_relaxed_records]
    relaxed_metadata = []
    for record in kept_relaxed_records:
        pore_inside, pore_dot = _adsorbate_inside_original_pore(
            record[1], len(input_atoms), record[3]
        )
        meta = dict(record[3])
        meta.update(
            {
                "relaxed_row_id": int(record[2]["row_id"]),
                "energy_eV": float(record[2]["energy_eV"]),
                "max_force_eV_A": float(record[2]["max_force_eV_A"]),
                "converged": bool(record[2]["converged"]),
                "nsteps": int(record[2]["nsteps"]),
                "force_evaluations": int(record[2]["force_evaluations"]),
                "elapsed_seconds": float(record[2]["elapsed_seconds"]),
                "adsorbate_integrity_preserved": bool(record[4]),
                "inside_original_pore": bool(pore_inside),
                "relaxed_pore_dot_com_minus_site": float(pore_dot),
            }
        )
        relaxed_metadata.append(meta)

    _log(verbose, "[SilicatePore] Screening relaxed minima for similarity")
    secondary_threshold_df, _suggested2, _uncapped2, secondary_threshold = _resolve_duplicate_cutoff(
        relaxed_atoms,
        reference_row_id=int(relaxed_minimum_reference_row_id),
        explicit_cutoff=relaxed_minimum_duplicate_cutoff,
        suggested_cutoff_scale=float(relaxed_minimum_suggested_cutoff_scale),
        similarity_kwargs=similarity_kwargs,
    )
    final_atoms, secondary_report_df = _deduplicate_atoms_safe(
        relaxed_atoms,
        similarity_threshold=secondary_threshold,
        similarity_kwargs=similarity_kwargs,
    )
    _write_csv(secondary_report_df, relaxed_minimum_report_csv_path)
    _log(
        verbose,
        "[SilicatePore] Relaxed-minimum similarity complete: "
        f"{len(final_atoms)}/{len(relaxed_atoms)} kept, threshold={secondary_threshold:.6f}",
    )
    _log(verbose, f"[SilicatePore] Relaxed-minimum report: {Path(relaxed_minimum_report_csv_path).resolve()}")
    secondary_kept_rows = secondary_report_df[secondary_report_df["kept"]].copy()
    final_metadata = []
    for _, row in secondary_kept_rows.iterrows():
        source_idx = int(row["row_id"]) - 1
        meta = dict(relaxed_metadata[source_idx])
        meta["dft_candidate"] = True
        meta["source_relaxed_row_id"] = int(row["row_id"])
        meta["relaxed_minimum_max_similarity_to_kept"] = _coerce_db_value(
            row["max_similarity_to_kept"]
        )
        final_metadata.append(meta)
    for atoms in final_atoms:
        atoms.set_constraint(deepcopy(original_constraints))
    _write_atoms_db(Path(final_db_path), final_atoms, final_metadata)

    summary_df = pd.DataFrame(
        [
            {
                "stage": "initial_generation",
                "n_input": len(candidates),
                "n_output": len(candidates),
            },
            {
                "stage": "candidate_similarity",
                "n_input": len(candidates),
                "n_output": len(pre_relax_atoms),
                "threshold_used": float(primary_threshold),
                "threshold_suggested": float(primary_threshold_df.attrs["suggested_threshold"]),
            },
            {
                "stage": "mace_relaxation_integrity",
                "n_input": len(pre_relax_atoms),
                "n_output": len(relaxed_atoms),
                "n_converged": int(relaxation_report_df["converged"].sum()),
                "n_integrity_preserved": int(relaxation_report_df["adsorbate_integrity_preserved"].sum()),
                "n_inside_original_pore": int(relaxation_report_df["inside_original_pore"].sum()),
            },
            {
                "stage": "dft_candidate_similarity",
                "n_input": len(relaxed_atoms),
                "n_output": len(final_atoms),
                "threshold_used": float(secondary_threshold),
                "threshold_suggested": float(secondary_threshold_df.attrs["suggested_threshold"]),
            },
        ]
    )
    _write_csv(summary_df, summary_csv_path)
    with _output_path(summary_txt_path).open("w", encoding="utf-8") as handle:
        handle.write("Silicate Pore Adsorbate Workflow Summary\n")
        handle.write("=======================================\n")
        handle.write(f"Generated candidates: {len(candidates)}\n")
        handle.write(f"Prefiltered candidates: {len(pre_relax_atoms)}\n")
        handle.write(f"Relaxed intact in-pore candidates: {len(relaxed_atoms)}\n")
        handle.write(f"DFT candidates: {len(final_atoms)}\n")
        handle.write(f"Final DB for DFT: {Path(final_db_path).resolve()}\n")
        handle.write(f"Relaxation report: {Path(relaxation_report_csv_path).resolve()}\n")
        handle.write(f"Final similarity report: {Path(relaxed_minimum_report_csv_path).resolve()}\n")
    if verbose:
        print(summary_df)
        print(f"[SilicatePore] Final DB for DFT: {Path(final_db_path).resolve()}")
    if show_final:
        view(final_atoms)
    return {
        "candidates": candidates,
        "candidate_report_df": primary_report_df,
        "pre_relax_atoms": pre_relax_atoms,
        "relaxed_atoms": relaxed_atoms,
        "relaxation_report_df": relaxation_report_df,
        "final_atoms": final_atoms,
        "relaxed_minimum_report_df": secondary_report_df,
        "summary_df": summary_df,
        "candidates_db_path": Path(candidates_db_path),
        "pre_relax_db_path": Path(pre_relax_db_path),
        "relaxed_db_path": Path(relaxed_db_path),
        "final_db_path": Path(final_db_path),
        "summary_txt_path": Path(summary_txt_path),
    }


def run_coadsorbate_similarity_workflow(
    input_atoms=None,
    *,
    adsorbates,
    candidates_db_path: str | Path = "scratch/coadsorption_candidates.db",
    pre_relax_db_path: str | Path = "scratch/coadsorption_prefilter.db",
    relaxed_db_path: str | Path = "scratch/coadsorption_relaxed.db",
    final_db_path: str | Path = "scratch/coadsorption_final_for_dft.db",
    single_a_final_db_path: str | Path = "scratch/coadsorption_single_a_final_for_dft.db",
    single_b_final_db_path: str | Path = "scratch/coadsorption_single_b_final_for_dft.db",
    site_probe_report_csv_path: str | Path = "scratch/coadsorption_site_probe_similarity.csv",
    candidate_report_csv_path: str | Path = "scratch/coadsorption_candidate_similarity.csv",
    relaxed_minimum_report_csv_path: str | Path = "scratch/coadsorption_relaxed_minimum_similarity.csv",
    relaxation_report_csv_path: str | Path = "scratch/coadsorption_relaxation_report.csv",
    summary_csv_path: str | Path = "scratch/coadsorption_workflow_summary.csv",
    summary_txt_path: str | Path = "scratch/coadsorption_workflow_summary.txt",
    candidate_reference_row_id: int = 1,
    relaxed_minimum_reference_row_id: int = 1,
    site_probe_symbol: str = "H",
    site_probe_similarity_cutoff: float = 0.99,
    single_mode_duplicate_cutoff: float = 0.99,
    candidate_duplicate_cutoff: float | None = None,
    candidate_suggested_cutoff_scale: float = 1.0,
    relaxed_minimum_duplicate_cutoff: float | None = None,
    relaxed_minimum_suggested_cutoff_scale: float = 1.0,
    similarity_batch_size: int = 100,
    generation_kwargs: dict | None = None,
    similarity_kwargs: dict | None = None,
    mace_model_path: str | Path = DEFAULT_MACE_MODEL_PATH,
    mace_fmax: float = 0.05,
    mace_max_steps: int = 200,
    mace_device: str = "cpu",
    mace_dispersion: bool = False,
    mace_default_dtype: str = "float64",
    mace_enable_cueq: bool = True,
    relax_support: bool = False,
    support_relaxation_origin_atom_index: int | None = None,
    support_relaxation_neighbor_shells: int | None = None,
    keep_bound_only: bool = False,
    require_converged: bool = True,
    sequential_pairing: bool = True,
    show_final: bool = False,
    verbose: bool = True,
):
    """
    Stage-2 workflow:
    1) probe and de-duplicate adsorption sites,
    2) relax, validate, energy-sort, and de-duplicate single-adsorbate modes,
    3) pair modes and de-duplicate unrelaxed candidates,
    4) relax, validate, energy-sort, and de-duplicate combined minima,
    5) write the final DB ready for DFT.

    Candidate-cutoff behavior:
    - if candidate_duplicate_cutoff is provided, it is used directly;
    - otherwise the estimated threshold is relaxed upward proportionally:
      suggested * candidate_suggested_cutoff_scale.

    Before generating single-adsorbate modes, a temporary atom probes every
    geometrically allowed site. Bonding-invalid probes are rejected, then
    similarity-equivalent probe environments are reduced using
    site_probe_similarity_cutoff. After representative single modes are
    relaxed, copies are translated to the discarded probe-equivalent sites
    for coadsorbate pairing.

    Relaxed-minimum-cutoff behavior:
    - if relaxed_minimum_duplicate_cutoff is provided, it is used directly;
    - otherwise the estimated threshold uses the same proportional rule with
      relaxed_minimum_suggested_cutoff_scale (default 1.0 => unchanged).
    """
    generation_kwargs = dict(generation_kwargs or {})
    generation_kwargs.setdefault("verbose", verbose)
    similarity_kwargs = similarity_kwargs or {}

    supported_similarity_keys = {
        "support_species",
        "slab_shave_distance",
        "kernel_gamma",
        "kernel_alpha",
        "kernel_threshold",
        "exclude_self",
    }
    similarity_common_kwargs = {
        key: value
        for key, value in similarity_kwargs.items()
        if key in supported_similarity_keys
    }

    if input_atoms is None:
        raise ValueError("Provide input_atoms.")
    if relax_support and (
        support_relaxation_origin_atom_index is not None
        or support_relaxation_neighbor_shells is not None
    ):
        raise ValueError("relax_support cannot be combined with shell-based support relaxation.")
    if (support_relaxation_origin_atom_index is None) != (
        support_relaxation_neighbor_shells is None
    ):
        raise ValueError(
            "Support relaxation origin atom index and neighbor shells must be provided together."
        )
    relaxed_support_indices = (
        list(range(len(input_atoms)))
        if relax_support
        else (
            _support_relaxation_indices(
                input_atoms,
                support_relaxation_origin_atom_index,
                support_relaxation_neighbor_shells,
            )
            if support_relaxation_origin_atom_index is not None
            else []
        )
    )
    if not 0.0 <= float(site_probe_similarity_cutoff) <= 1.0:
        raise ValueError("site_probe_similarity_cutoff must be in the range [0, 1].")
    if not 0.0 <= float(single_mode_duplicate_cutoff) <= 1.0:
        raise ValueError("single_mode_duplicate_cutoff must be in the range [0, 1].")
    batch_size = int(similarity_batch_size)
    site_probe_symbol = str(site_probe_symbol).strip()
    if site_probe_symbol not in atomic_numbers:
        raise ValueError("site_probe_symbol must identify one chemical element.")
    original_constraints = deepcopy(getattr(input_atoms, "constraints", []))

    candidates_db_path = Path(candidates_db_path)
    pre_relax_db_path = Path(pre_relax_db_path)
    relaxed_db_path = Path(relaxed_db_path)
    final_db_path = Path(final_db_path)
    single_a_final_db_path = Path(single_a_final_db_path)
    single_b_final_db_path = Path(single_b_final_db_path)
    site_probe_report_csv_path = Path(site_probe_report_csv_path)
    candidate_report_csv_path = Path(candidate_report_csv_path)
    relaxed_minimum_report_csv_path = Path(relaxed_minimum_report_csv_path)
    relaxation_report_csv_path = Path(relaxation_report_csv_path)
    summary_csv_path = Path(summary_csv_path)
    summary_txt_path = Path(summary_txt_path)
    single_a_relaxed_db = candidates_db_path.with_name(
        f"{candidates_db_path.stem}_a_relaxed.db"
    )
    single_b_relaxed_db = candidates_db_path.with_name(
        f"{candidates_db_path.stem}_b_relaxed.db"
    )

    output_paths = [
        candidates_db_path,
        pre_relax_db_path,
        relaxed_db_path,
        final_db_path,
        single_a_final_db_path,
        single_b_final_db_path,
        site_probe_report_csv_path,
        candidate_report_csv_path,
        relaxed_minimum_report_csv_path,
        relaxation_report_csv_path,
        summary_csv_path,
        summary_txt_path,
        single_a_relaxed_db,
        single_b_relaxed_db,
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path not in {relaxed_db_path, single_a_relaxed_db, single_b_relaxed_db} and path.exists() and path.is_file():
            path.unlink()
    if verbose:
        print("[Workflow] Starting co-adsorbate workflow")
        print(f"[Workflow] Input atoms: {len(input_atoms)}")
        print(f"[Workflow] Adsorbates requested: {len(adsorbates)}")
        print(f"[Workflow] Candidates DB: {candidates_db_path.resolve()}")
        print(f"[Workflow] Generation options: {generation_kwargs}")
        print(f"[Workflow] Similarity options: {similarity_kwargs}")

    single_a_modes = []
    single_b_modes = []
    candidate_metadata = None
    interaction_references = None
    if sequential_pairing and len(adsorbates) == 2:
        single_a_db = candidates_db_path.with_name(f"{candidates_db_path.stem}_a.db")
        single_b_db = candidates_db_path.with_name(f"{candidates_db_path.stem}_b.db")
        for path in (single_a_db, single_b_db):
            if path.exists():
                path.unlink()
        (
            probe_modes,
            probe_site_indices,
            probe_report_df,
            site_indices,
            single_generation_kwargs,
        ) = _screen_sites_with_probe(
            input_atoms,
            candidates_db_path=candidates_db_path,
            generation_kwargs=generation_kwargs,
            site_probe_symbol=site_probe_symbol,
            site_probe_similarity_cutoff=site_probe_similarity_cutoff,
            similarity_kwargs=similarity_common_kwargs,
            batch_size=batch_size,
            report_path=site_probe_report_csv_path,
            verbose=verbose,
        )
        for path in (single_a_db, single_b_db):
            path.unlink(missing_ok=True)
        single_a_modes = run_coadsorbate_generation(
            input_atoms,
            adsorbates=[adsorbates[0]],
            output_db=single_a_db,
            allowed_site_indices=site_indices,
            **single_generation_kwargs,
        )
        single_b_modes = run_coadsorbate_generation(
            input_atoms,
            adsorbates=[adsorbates[1]],
            output_db=single_b_db,
            allowed_site_indices=site_indices,
            **single_generation_kwargs,
        )
        with connect(str(single_a_db), serial=True) as db:
            single_a_generation_metadata = [dict(row.data) for row in db.select()]
        with connect(str(single_b_db), serial=True) as db:
            single_b_generation_metadata = [dict(row.data) for row in db.select()]
        support_relaxed, _ = _relax_adsorbates_with_mace(
            [input_atoms],
            n_fixed_atoms=len(input_atoms),
            relaxed_support_indices=relaxed_support_indices,
            model_path=str(mace_model_path),
            fmax=mace_fmax,
            max_steps=mace_max_steps if relaxed_support_indices else 0,
            device=mace_device,
            dispersion=mace_dispersion,
            default_dtype=mace_default_dtype,
            enable_cueq=mace_enable_cueq,
            verbose=False,
        )
        single_a_initial = single_a_modes
        single_b_initial = single_b_modes
        single_a_modes, single_a_report = _relax_adsorbates_with_mace(
            single_a_initial,
            n_fixed_atoms=len(input_atoms),
            relaxed_support_indices=relaxed_support_indices,
            model_path=str(mace_model_path),
            fmax=mace_fmax,
            max_steps=mace_max_steps,
            device=mace_device,
            dispersion=mace_dispersion,
            default_dtype=mace_default_dtype,
            enable_cueq=mace_enable_cueq,
            output_db_path=single_a_relaxed_db,
            verbose=verbose,
        )
        single_b_modes, single_b_report = _relax_adsorbates_with_mace(
            single_b_initial,
            n_fixed_atoms=len(input_atoms),
            relaxed_support_indices=relaxed_support_indices,
            model_path=str(mace_model_path),
            fmax=mace_fmax,
            max_steps=mace_max_steps,
            device=mace_device,
            dispersion=mace_dispersion,
            default_dtype=mace_default_dtype,
            enable_cueq=mace_enable_cueq,
            output_db_path=single_b_relaxed_db,
            verbose=verbose,
        )
        a_records, a_kept = _filter_and_sort_relaxed(
            single_a_initial,
            single_a_modes,
            single_a_report,
            single_a_generation_metadata,
            n_support_atoms=len(input_atoms),
            require_converged=require_converged,
            require_bound=True,
        )
        b_records, b_kept = _filter_and_sort_relaxed(
            single_b_initial,
            single_b_modes,
            single_b_report,
            single_b_generation_metadata,
            n_support_atoms=len(input_atoms),
            require_converged=require_converged,
            require_bound=True,
        )
        single_a_modes = [record[1] for record in a_kept]
        single_b_modes = [record[1] for record in b_kept]
        single_a_metadata = [
            {
                **record[2],
                **record[3],
                "adsorbate_integrity_preserved": record[4],
                "bound_to_support": record[5],
            }
            for record in a_kept
        ]
        single_b_metadata = [
            {
                **record[2],
                **record[3],
                "adsorbate_integrity_preserved": record[4],
                "bound_to_support": record[5],
            }
            for record in b_kept
        ]
        if not single_a_modes or not single_b_modes:
            raise RuntimeError("Single-adsorbate validity filtering removed all modes.")
        if verbose:
            print(
                "[Workflow] Single-mode validity filtering: "
                f"A {len(a_records)} -> {len(single_a_modes)}; "
                f"B {len(b_records)} -> {len(single_b_modes)} "
                f"(require_converged={require_converged}, require_bound=True)"
            )
            print(
                "[Workflow] Single-mode validity predicates: "
                f"A intact={sum(record[4] for record in a_records)}, "
                f"bound={sum(record[5] for record in a_records)}; "
                f"B intact={sum(record[4] for record in b_records)}, "
                f"bound={sum(record[5] for record in b_records)}"
            )
        interaction_references = {
            "support_energy_eV": float(support_relaxed[0].get_potential_energy()),
            "a_complex_min_energy_eV": min(
                float(atoms.get_potential_energy()) for atoms in single_a_modes
            ),
            "b_complex_min_energy_eV": min(
                float(atoms.get_potential_energy()) for atoms in single_b_modes
            ),
        }
        single_counts = (len(single_a_modes), len(single_b_modes))
        single_a_modes, a_threshold, a_kept_rows, a_similarity_report = _prune_single_adsorbate_modes(
            single_a_modes,
            reference_row_id=1,
            similarity_threshold=single_mode_duplicate_cutoff,
            similarity_relaxation=0.0,
            similarity_kwargs=similarity_common_kwargs,
            batch_size=batch_size,
            verbose=verbose,
            label="Workflow single-A similarity",
        )
        single_b_modes, b_threshold, b_kept_rows, b_similarity_report = _prune_single_adsorbate_modes(
            single_b_modes,
            reference_row_id=1,
            similarity_threshold=single_mode_duplicate_cutoff,
            similarity_relaxation=0.0,
            similarity_kwargs=similarity_common_kwargs,
            batch_size=batch_size,
            verbose=verbose,
            label="Workflow single-B similarity",
        )
        _write_single_mode_dft_candidates(
            single_a_final_db_path,
            single_a_modes,
            single_a_metadata,
            a_kept_rows,
            a_similarity_report,
            original_constraints,
        )
        _write_single_mode_dft_candidates(
            single_b_final_db_path,
            single_b_modes,
            single_b_metadata,
            b_kept_rows,
            b_similarity_report,
            original_constraints,
        )
        if verbose:
            print(
                "[Workflow] Pre-combination screening: "
                f"A {single_counts[0]} -> {len(single_a_modes)} "
                f"(threshold={a_threshold:.6f}); "
                f"B {single_counts[1]} -> {len(single_b_modes)} "
                f"(threshold={b_threshold:.6f})"
            )
            print(f"[Workflow] Single-A final DB for DFT: {single_a_final_db_path.resolve()}")
            print(f"[Workflow] Single-B final DB for DFT: {single_b_final_db_path.resolve()}")
        pairing_a_modes = _expand_modes_to_probe_equivalents(
            single_a_modes,
            [single_a_metadata[row_id - 1] for row_id in a_kept_rows],
            n_support_atoms=len(input_atoms),
            representative_site_indices=site_indices,
            probe_modes=probe_modes,
            probe_site_indices=probe_site_indices,
            probe_report=probe_report_df,
        )
        pairing_b_modes = _expand_modes_to_probe_equivalents(
            single_b_modes,
            [single_b_metadata[row_id - 1] for row_id in b_kept_rows],
            n_support_atoms=len(input_atoms),
            representative_site_indices=site_indices,
            probe_modes=probe_modes,
            probe_site_indices=probe_site_indices,
            probe_report=probe_report_df,
        )
        if verbose:
            print(
                "[Workflow] H-probe equivalent-site expansion: "
                f"A {len(single_a_modes)} -> {len(pairing_a_modes)}; "
                f"B {len(single_b_modes)} -> {len(pairing_b_modes)}"
            )
        candidates, candidate_metadata = combine_single_adsorbate_modes(
            pairing_a_modes,
            pairing_b_modes,
            n_support_atoms=len(input_atoms),
            max_structures=generation_kwargs.get("max_structures"),
            skip_abnormal_bond_screening=bool(
                generation_kwargs.get("skip_abnormal_bond_screening", False)
            ),
            verbose=verbose,
        )
        if verbose:
            n_spaced = sum(
                row.get("generation") == "sequential_pairing_with_lateral_spacing"
                for row in candidate_metadata
            )
            if n_spaced:
                print(
                    "[Workflow] Lateral-spacing fallback recovered "
                    f"{n_spaced} clashing co-adsorbate pairs"
                )
        _write_atoms_db(candidates_db_path, candidates, candidate_metadata)
    elif sequential_pairing and len(adsorbates) == 1:
        _, _, _, site_indices, single_generation_kwargs = _screen_sites_with_probe(
            input_atoms,
            candidates_db_path=candidates_db_path,
            generation_kwargs=generation_kwargs,
            site_probe_symbol=site_probe_symbol,
            site_probe_similarity_cutoff=site_probe_similarity_cutoff,
            similarity_kwargs=similarity_common_kwargs,
            batch_size=batch_size,
            report_path=site_probe_report_csv_path,
            verbose=verbose,
        )
        candidates = run_coadsorbate_generation(
            input_atoms,
            adsorbates=adsorbates,
            output_db=candidates_db_path,
            allowed_site_indices=site_indices,
            **single_generation_kwargs,
        )
    else:
        candidates = run_coadsorbate_generation(
            input_atoms,
            adsorbates=adsorbates,
            output_db=candidates_db_path,
            **generation_kwargs,
        )
    if not candidates:
        raise RuntimeError("Co-adsorbate generator produced no candidates.")
    if verbose:
        print(f"[Workflow] Generated candidates: {len(candidates)}")

    threshold_candidates = candidates[:batch_size] if batch_size > 0 else candidates
    primary_threshold_df, primary_suggested, primary_uncapped, primary_threshold = _resolve_duplicate_cutoff(
        threshold_candidates,
        reference_row_id=min(int(candidate_reference_row_id), len(threshold_candidates)),
        explicit_cutoff=candidate_duplicate_cutoff,
        suggested_cutoff_scale=candidate_suggested_cutoff_scale,
        similarity_kwargs=similarity_common_kwargs,
    )
    if verbose:
        spread = primary_threshold_df.attrs.get("similarity_spread", {})
        print(
            "[Workflow] Candidate duplicate cutoff "
            f"(suggested={primary_suggested:.6f}, scale={candidate_suggested_cutoff_scale:.6f}, "
            f"uncapped={primary_uncapped:.6f}, used={primary_threshold:.6f}, "
            f"range={spread.get('min', float('nan')):.6f}-{spread.get('max', float('nan')):.6f})"
        )
    pre_relax_atoms, primary_report_df = _deduplicate_atoms_in_batches(
        candidates,
        batch_size=batch_size,
        similarity_threshold=primary_threshold,
        similarity_kwargs=similarity_common_kwargs,
        verbose=verbose,
        label="Workflow candidate similarity",
    )
    _write_csv(primary_report_df, candidate_report_csv_path)
    if not pre_relax_atoms:
        raise RuntimeError("Primary similarity pruning removed all candidates.")
    if verbose:
        print(
            "[Workflow] Candidate pruning "
            f"kept {len(pre_relax_atoms)} / {len(candidates)} "
            f"(pruned {len(candidates) - len(pre_relax_atoms)})"
        )
        print(f"[Workflow] Candidate report: {candidate_report_csv_path.resolve()}")

    primary_kept_rows = primary_report_df[primary_report_df["kept"]].copy()
    pre_relax_metadata = []
    for _, row in primary_kept_rows.iterrows():
        source_row_id = int(row["row_id"])
        metadata = (
            dict(candidate_metadata[source_row_id - 1])
            if candidate_metadata is not None
            else {}
        )
        metadata.update(
            {
                "source_candidate_row_id": source_row_id,
                "candidate_max_similarity_to_kept": _coerce_db_value(
                    row["max_similarity_to_kept"]
                ),
            }
        )
        pre_relax_metadata.append(metadata)
    _write_atoms_db(pre_relax_db_path, pre_relax_atoms, pre_relax_metadata)
    if verbose:
        print(f"[Workflow] Pre-relax DB: {pre_relax_db_path.resolve()}")

    relaxed_atoms, relaxation_report_df = _relax_adsorbates_with_mace(
        pre_relax_atoms,
        n_fixed_atoms=len(input_atoms),
        relaxed_support_indices=relaxed_support_indices,
        model_path=str(mace_model_path),
        fmax=mace_fmax,
        max_steps=mace_max_steps,
        device=mace_device,
        dispersion=mace_dispersion,
        default_dtype=mace_default_dtype,
        enable_cueq=mace_enable_cueq,
        output_db_path=relaxed_db_path,
        metadata_rows=pre_relax_metadata,
        verbose=verbose,
    )
    relaxation_records, kept_relaxed_records = _filter_and_sort_relaxed(
        pre_relax_atoms,
        relaxed_atoms,
        relaxation_report_df,
        pre_relax_metadata,
        n_support_atoms=len(input_atoms),
        require_converged=require_converged,
        require_bound=keep_bound_only,
        verbose=verbose,
    )
    relaxation_report_df["adsorbate_integrity_preserved"] = [
        record[4] for record in relaxation_records
    ]
    relaxation_report_df["bound_to_support"] = [record[5] for record in relaxation_records]
    relaxation_report_df["kept_for_deduplication"] = [
        record[6] for record in relaxation_records
    ]
    _write_csv(relaxation_report_df, relaxation_report_csv_path)
    if verbose:
        n_converged = int(relaxation_report_df["converged"].sum())
        print(
            "[Workflow] MACE relaxation complete: "
            f"{n_converged}/{len(relaxed_atoms)} converged (fmax <= {mace_fmax})"
        )
        print(f"[Workflow] Relaxation report: {relaxation_report_csv_path.resolve()}")

    relaxed_atoms = [record[1] for record in kept_relaxed_records]
    pre_relax_metadata = [record[3] for record in kept_relaxed_records]
    relaxation_report_df = pd.DataFrame(
        [
            {
                **record[2],
                "adsorbate_integrity_preserved": record[4],
                "bound_to_support": record[5],
                "kept_for_deduplication": record[6],
            }
            for record in kept_relaxed_records
        ]
    )
    if verbose:
        print(
            "[Workflow] Relaxed-candidate validity filtering kept "
            f"{len(relaxed_atoms)} / {len(relaxation_records)} structures "
            f"(require_converged={require_converged}, require_bound={keep_bound_only})"
        )
        print(
            "[Workflow] Relaxed-candidate validity predicates: "
            f"intact={sum(record[4] for record in relaxation_records)}, "
            f"bound={sum(record[5] for record in relaxation_records)}, "
            f"converged={sum(bool(record[2]['converged']) for record in relaxation_records)}"
        )
    if not relaxed_atoms:
        raise RuntimeError("Relaxed-candidate validity filtering removed all structures.")

    relaxed_metadata = []
    for idx, row in relaxation_report_df.iterrows():
        meta = dict(pre_relax_metadata[idx])
        meta.update(
            {
                "relaxed_row_id": int(row["row_id"]),
                "energy_eV": float(row["energy_eV"]),
                "max_force_eV_A": float(row["max_force_eV_A"]),
                "converged": bool(row["converged"]),
                "nsteps": int(row["nsteps"]),
                "force_evaluations": int(row["force_evaluations"]),
                "elapsed_seconds": float(row["elapsed_seconds"]),
            }
        )
        relaxed_metadata.append(meta)
    if verbose:
        print(f"[Workflow] Relaxation checkpoint DB: {relaxed_db_path.resolve()}")

    threshold_relaxed = relaxed_atoms[:batch_size] if batch_size > 0 else relaxed_atoms
    secondary_threshold_df, secondary_suggested, secondary_uncapped, secondary_threshold = _resolve_duplicate_cutoff(
        threshold_relaxed,
        reference_row_id=min(int(relaxed_minimum_reference_row_id), len(threshold_relaxed)),
        explicit_cutoff=relaxed_minimum_duplicate_cutoff,
        suggested_cutoff_scale=relaxed_minimum_suggested_cutoff_scale,
        similarity_kwargs=similarity_common_kwargs,
    )
    if verbose:
        spread = secondary_threshold_df.attrs.get("similarity_spread", {})
        print(
            "[Workflow] Relaxed-minimum duplicate cutoff "
            f"(suggested={secondary_suggested:.6f}, scale={relaxed_minimum_suggested_cutoff_scale:.6f}, "
            f"uncapped={secondary_uncapped:.6f}, used={secondary_threshold:.6f}, "
            f"range={spread.get('min', float('nan')):.6f}-{spread.get('max', float('nan')):.6f})"
        )
    final_atoms, secondary_report_df = _deduplicate_atoms_in_batches(
        relaxed_atoms,
        batch_size=batch_size,
        similarity_threshold=secondary_threshold,
        similarity_kwargs=similarity_common_kwargs,
        verbose=verbose,
        label="Workflow relaxed-minimum similarity",
    )
    _write_csv(secondary_report_df, relaxed_minimum_report_csv_path)
    if not final_atoms:
        raise RuntimeError("Relaxed-minimum similarity pruning removed all structures.")
    if verbose:
        print(
            "[Workflow] Relaxed-minimum pruning "
            f"kept {len(final_atoms)} / {len(relaxed_atoms)} "
            f"(pruned {len(relaxed_atoms) - len(final_atoms)})"
        )
        print(f"[Workflow] Relaxed-minimum report: {relaxed_minimum_report_csv_path.resolve()}")

    secondary_kept_rows = secondary_report_df[secondary_report_df["kept"]].copy()
    final_metadata = []
    for _, row in secondary_kept_rows.iterrows():
        relaxed_row_id = int(row["row_id"])
        meta = dict(relaxed_metadata[relaxed_row_id - 1])
        meta.update(
            {
                "source_relaxed_row_id": relaxed_row_id,
                "relaxed_minimum_max_similarity_to_kept": _coerce_db_value(
                    row["max_similarity_to_kept"]
                ),
            }
        )
        if interaction_references is not None:
            meta.update(interaction_references)
            meta["coadsorption_interaction_energy_eV"] = _coadsorption_interaction_energy(
                float(meta["energy_eV"]),
                **interaction_references,
            )
        final_metadata.append(meta)

    # Ensure final structures for DFT carry the same constraints as the
    # original input surface (or clear constraints if original had none).
    for atoms in final_atoms:
        atoms.set_constraint(deepcopy(original_constraints))

    _write_atoms_db(final_db_path, final_atoms, final_metadata)

    summary_df = pd.DataFrame(
        [
            {
                "stage": "candidate_similarity",
                "n_input": len(candidates),
                "n_output": len(pre_relax_atoms),
                "threshold_suggested": float(primary_threshold_df.attrs["suggested_threshold"]),
                "threshold_used": float(primary_threshold),
            },
            {
                "stage": "mace_relaxation",
                "n_input": len(pre_relax_atoms),
                "n_output": len(relaxed_atoms),
                "n_converged": int(relaxation_report_df["converged"].sum()),
                "mean_energy_eV": float(relaxation_report_df["energy_eV"].mean()),
            },
            {
                "stage": "relaxed_minimum_similarity",
                "n_input": len(relaxed_atoms),
                "n_output": len(final_atoms),
                "threshold_suggested": float(secondary_threshold_df.attrs["suggested_threshold"]),
                "threshold_used": float(secondary_threshold),
            },
        ]
    )
    if interaction_references is not None:
        interaction_energies = [
            float(meta["coadsorption_interaction_energy_eV"])
            for meta in final_metadata
        ]
        summary_df = pd.concat(
            [
                summary_df,
                pd.DataFrame(
                    [
                        {
                            "stage": "coadsorption_interaction",
                            "n_input": len(final_atoms),
                            "n_output": len(final_atoms),
                            "min_interaction_energy_eV": min(interaction_energies),
                            **interaction_references,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    _write_csv(summary_df, summary_csv_path)

    primary_pruned = len(candidates) - len(pre_relax_atoms)
    secondary_pruned = len(relaxed_atoms) - len(final_atoms)
    n_converged = int(relaxation_report_df["converged"].sum())
    with summary_txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Co-adsorbate Similarity Workflow Summary\n")
        handle.write("=====================================\n")
        handle.write(f"Candidates generated: {len(candidates)}\n")
        handle.write(
            f"Candidate duplicate cutoff: {primary_threshold:.6f} "
            f"(suggested {primary_threshold_df.attrs['suggested_threshold']:.6f})\n"
        )
        handle.write(
            f"Candidate pruning: kept {len(pre_relax_atoms)}, pruned {primary_pruned}\n"
        )
        handle.write(
            f"MACE relaxation: {n_converged}/{len(relaxed_atoms)} converged "
            f"(fmax <= {mace_fmax})\n"
        )
        handle.write(
            f"Relaxed-minimum duplicate cutoff: {secondary_threshold:.6f} "
            f"(suggested {secondary_threshold_df.attrs['suggested_threshold']:.6f})\n"
        )
        handle.write(
            f"Relaxed-minimum pruning: kept {len(final_atoms)}, pruned {secondary_pruned}\n"
        )
        handle.write(f"Final DB for DFT: {final_db_path.resolve()}\n")
        if single_a_final_db_path.exists():
            handle.write(f"Single-A final DB for DFT: {single_a_final_db_path.resolve()}\n")
        if single_b_final_db_path.exists():
            handle.write(f"Single-B final DB for DFT: {single_b_final_db_path.resolve()}\n")
        handle.write(f"Candidate report CSV: {candidate_report_csv_path.resolve()}\n")
        handle.write(f"Site-probe report CSV: {site_probe_report_csv_path.resolve()}\n")
        handle.write(f"Relaxation report CSV: {relaxation_report_csv_path.resolve()}\n")
        handle.write(f"Relaxed-minimum report CSV: {relaxed_minimum_report_csv_path.resolve()}\n")
        if interaction_references is not None:
            handle.write(
                "Most stable co-adsorption interaction energy (MACE): "
                f"{min(interaction_energies):.6f} eV\n"
            )
        pruned_secondary = secondary_report_df[~secondary_report_df["kept"]]
        if not pruned_secondary.empty:
            handle.write("Pruned relaxed minima (row -> closest kept, similarity):\n")
            for _, row in pruned_secondary.iterrows():
                row_id = int(row["row_id"])
                closest = (
                    "None"
                    if pd.isna(row["closest_kept_row_id"])
                    else str(int(row["closest_kept_row_id"]))
                )
                similarity = (
                    "NaN"
                    if pd.isna(row["max_similarity_to_kept"])
                    else f"{float(row['max_similarity_to_kept']):.6f}"
                )
                handle.write(f"  {row_id} -> {closest}, {similarity}\n")

    if verbose:
        print(f"[Workflow] Final structures for DFT: {len(final_atoms)}")
        print(f"[Workflow] Final DB: {final_db_path.resolve()}")
        print(f"[Workflow] Summary CSV: {summary_csv_path.resolve()}")
        print(f"[Workflow] Summary TXT: {summary_txt_path.resolve()}")

    if show_final:
        view(final_atoms)

    return {
        "candidates": candidates,
        "candidate_cutoff_df": primary_threshold_df,
        "candidate_duplicate_cutoff_used": primary_threshold,
        "pre_relax_atoms": pre_relax_atoms,
        "candidate_report_df": primary_report_df,
        "relaxed_atoms": relaxed_atoms,
        "relaxation_report_df": relaxation_report_df,
        "relaxed_minimum_cutoff_df": secondary_threshold_df,
        "relaxed_minimum_duplicate_cutoff_used": secondary_threshold,
        "final_atoms": final_atoms,
        "relaxed_minimum_report_df": secondary_report_df,
        "summary_df": summary_df,
        "candidates_db_path": candidates_db_path,
        "pre_relax_db_path": pre_relax_db_path,
        "relaxed_db_path": relaxed_db_path,
        "final_db_path": final_db_path,
        "single_a_final_db_path": single_a_final_db_path,
        "single_b_final_db_path": single_b_final_db_path,
        "candidate_report_csv_path": candidate_report_csv_path,
        "site_probe_report_csv_path": site_probe_report_csv_path,
        "relaxed_minimum_report_csv_path": relaxed_minimum_report_csv_path,
        "relaxation_report_csv_path": relaxation_report_csv_path,
        "summary_csv_path": summary_csv_path,
        "summary_txt_path": summary_txt_path,
        "interaction_references": interaction_references,
    }
