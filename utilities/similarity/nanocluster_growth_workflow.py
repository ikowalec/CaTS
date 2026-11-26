from __future__ import annotations

import argparse
import csv
import sys
import time
from copy import deepcopy
from pathlib import Path
import numpy as np

import pandas as pd
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.db import connect
from ase.io import read, write
from ase.optimize import BFGS

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _is_bonded_cluster(atoms: Atoms, *, bond_scale: float = 1.35) -> bool:
    if len(atoms) < 2:
        return True

    mic = bool(np.any(atoms.get_pbc()))
    seen = {0}
    stack = [0]
    while stack:
        i = stack.pop()
        radius_i = float(covalent_radii[int(atoms[i].number)])
        for j in range(len(atoms)):
            if j in seen or i == j:
                continue
            cutoff = (radius_i + float(covalent_radii[int(atoms[j].number)])) * float(bond_scale)
            if float(atoms.get_distance(i, j, mic=mic)) <= cutoff:
                seen.add(j)
                stack.append(j)
    return len(seen) == len(atoms)


def _get_mace_calculator(settings):
    from similarity.workflows.mace_pool import get_mace_calculators

    return get_mace_calculators(
        model=str(settings.model_path),
        dispersion=bool(settings.dispersion),
        default_dtype=str(settings.default_dtype),
        device=str(settings.device),
        enable_cueq=bool(getattr(settings, "enable_cueq", True)),
    )[0]


def _relax_cluster_mpi_task(task, calculator, settings):
    started = time.perf_counter()
    row_id, atoms = task
    relaxed = atoms.copy()
    original_constraints = deepcopy(relaxed.constraints)
    fixed = set(settings.get("fixed_symbols") or [])
    fixed_indices = [i for i, atom in enumerate(relaxed) if atom.symbol in fixed]
    relaxed.set_constraint(FixAtoms(indices=fixed_indices) if fixed_indices else None)
    relaxed.calc = calculator
    opt = BFGS(relaxed, logfile=None)
    opt.run(fmax=float(settings["fmax"]), steps=int(settings["max_steps"]))
    energy = float(relaxed.get_potential_energy())
    forces = relaxed.get_forces()
    raw_forces = relaxed.get_forces(apply_constraint=False)
    max_force = float((forces**2).sum(axis=1).max() ** 0.5) if len(forces) else 0.0
    frozen = relaxed.copy()
    frozen.set_constraint(original_constraints)
    frozen.calc = SinglePointCalculator(frozen, energy=energy, forces=raw_forces)
    bonded_set = set(settings.get("bonded_symbols") or [])
    bonded_atoms = frozen[
        [i for i, atom in enumerate(frozen) if atom.symbol in bonded_set]
    ] if bonded_set else frozen
    nsteps = int(getattr(opt, "nsteps", 0))
    row = {
        "row_id": row_id,
        "n_atoms": len(frozen),
        "energy_eV": energy,
        "max_force_eV_A": max_force,
        "converged": bool(max_force <= float(settings["fmax"])),
        "bonded": _is_bonded_cluster(bonded_atoms),
        "nsteps": nsteps,
        "force_evaluations": nsteps + 1,
        "elapsed_seconds": time.perf_counter() - started,
    }
    return row_id, frozen, row

def _relax_full_cluster_with_mace(
    atoms_list: list[Atoms],
    *,
    model_path: str | Path,
    fmax: float,
    max_steps: int,
    device: str,
    dispersion: bool,
    default_dtype: str,
    output_db_path: Path,
    verbose: bool,
    fixed_symbols: list[str] | None = None,
    bonded_symbols: list[str] | None = None,
    enable_cueq: bool = True,
    calculators=None,
) -> tuple[list[Atoms], pd.DataFrame]:
    from similarity.workflows.mace_pool import (
        checkpoint_hash,
        get_mace_calculators,
        run_mace_tasks,
        write_ase_database_atomically,
    )

    checkpoint_settings = {
        "model_path": str(model_path),
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "device": str(device),
        "dispersion": bool(dispersion),
        "default_dtype": str(default_dtype),
        "enable_cueq": bool(enable_cueq),
        "fixed_symbols": sorted(fixed_symbols or []),
        "bonded_symbols": sorted(bonded_symbols or []),
    }
    hashes = {
        row_id: checkpoint_hash(atoms, checkpoint_settings)
        for row_id, atoms in enumerate(atoms_list, start=1)
    }
    if calculators is None:
        calculators = get_mace_calculators(
            model=str(model_path),
            dispersion=bool(dispersion),
            default_dtype=str(default_dtype),
            device=str(device),
            enable_cueq=bool(enable_cueq),
        )
    completed = {}
    if output_db_path.exists():
        with connect(str(output_db_path), serial=True) as db:
            completed = {
                int(row.data.relaxed_row_id): row
                for row in db.select()
                if row.data.get("relaxed_row_id") is not None
                and row.data.get("checkpoint_hash")
                == hashes.get(int(row.data.relaxed_row_id))
            }
        if not completed:
            output_db_path.unlink()

    results = {}
    checkpoint_records = {}
    for row_id, checkpoint in completed.items():
        data = checkpoint.data
        checkpoint_records[row_id] = (checkpoint.toatoms(), dict(data))
        results[row_id] = (
            checkpoint.toatoms(),
            {
                "row_id": row_id,
                "n_atoms": int(data.n_atoms),
                "energy_eV": float(data.energy_eV),
                "max_force_eV_A": float(data.max_force_eV_A),
                "converged": bool(data.converged),
                "bonded": bool(data.bonded),
                "nsteps": int(data.nsteps),
                "force_evaluations": int(data.get("force_evaluations", int(data.nsteps) + 1)),
                "elapsed_seconds": float(data.get("elapsed_seconds", 0.0)),
            },
        )

    pending = [
        (row_id, atoms)
        for row_id, atoms in enumerate(atoms_list, start=1)
        if row_id not in completed
    ]

    worker_settings = {
        "fmax": float(fmax),
        "max_steps": int(max_steps),
        "fixed_symbols": list(fixed_symbols or []),
        "bonded_symbols": list(bonded_symbols or []),
    }

    def relax_one(task, calculator):
        return _relax_cluster_mpi_task(task, calculator, worker_settings)

    checkpoint_interval = max(1, (len(atoms_list) + 9) // 10)
    saved_since_checkpoint = 0

    def publish_checkpoint():
        write_ase_database_atomically(
            output_db_path,
            [checkpoint_records[row_id] for row_id in sorted(checkpoint_records)],
        )

    def save_result(result):
        nonlocal saved_since_checkpoint
        row_id, frozen, row = result
        data = dict(row)
        data.update(relaxed_row_id=row_id, checkpoint_hash=hashes[row_id])
        checkpoint_records[row_id] = (frozen, data)
        results[row_id] = (frozen, row)
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
        completed=len(completed),
        total=len(atoms_list),
        label="MACE full relaxation",
        verbose=verbose,
        distributed_operation=(
            "utilities.similarity.nanocluster_growth_workflow:_relax_cluster_mpi_task"
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
    relaxed_atoms = [
        atoms for atoms, row in ordered if row["converged"] and row["bonded"]
    ]
    return relaxed_atoms, pd.DataFrame([row for _, row in ordered])


def _deduplicate(
    atoms_list: list[Atoms],
    *,
    similarity_threshold: float,
    report_csv_path: Path,
    support_species: list[str],
    slab_shave_distance: float | None = None,
    batch_size: int = 100,
    verbose: bool = False,
    label: str = "Nanocluster similarity",
) -> list[Atoms]:
    from similarity.workflows.coadsorbate_similarity import _deduplicate_atoms_in_batches

    unique_atoms, report_df = _deduplicate_atoms_in_batches(
        atoms_list,
        batch_size=int(batch_size),
        similarity_threshold=float(similarity_threshold),
        similarity_kwargs={
            "support_species": support_species,
            "slab_shave_distance": slab_shave_distance,
            "kernel_gamma": 1.0,
            "kernel_alpha": 1.0,
            "kernel_threshold": 1e-6,
        },
        verbose=verbose,
        label=label,
    )
    report_df.to_csv(report_csv_path, index=False)
    return unique_atoms


def _write_summary_txt(
    path: Path,
    *,
    n_candidates: int,
    n_pre_relax: int,
    n_relaxed: int,
    n_final: int,
    selected_row_id: int,
    selected_energy: float,
) -> None:
    path.write_text(
        "\n".join(
            [
                "Nanocluster Growth Generation Summary",
                "=====================================",
                f"Candidates generated: {n_candidates}",
                f"Primary unique candidates: {n_pre_relax}",
                f"Full-cluster MACE relaxed structures: {n_relaxed}",
                f"Final unique relaxed structures: {n_final}",
                f"Selected final row: {selected_row_id}",
                f"Selected energy / eV: {selected_energy:.10f}",
                "",
            ]
        ),
        encoding="utf-8",
)


def _single_atom_adsorbate(symbol: str) -> Atoms:
    return Atoms(symbol, positions=[[0.0, 0.0, 0.0]])


def _single_attachment_candidate(parent: Atoms, add_symbol: str) -> Atoms:
    candidate = parent.copy()
    if len(parent):
        distance = (
            covalent_radii[atomic_numbers[parent[0].symbol]]
            + covalent_radii[atomic_numbers[add_symbol]]
        )
        position = parent.positions[0] + [distance, 0.0, 0.0]
    else:
        position = [0.0, 0.0, 0.0]
    candidate += Atoms(add_symbol, positions=[position])
    return candidate


def _energy(atoms: Atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        calc = getattr(atoms, "calc", None)
        if calc is not None and hasattr(calc, "results") and "energy" in calc.results:
            return float(calc.results["energy"])
        raise


def _select_lowest_energy(atoms_list: list[Atoms]) -> tuple[int, Atoms, float]:
    selected = _select_lowest_energy_n(atoms_list, n=1)[0]
    return selected


def _select_lowest_energy_n(atoms_list: list[Atoms], *, n: int) -> list[tuple[int, Atoms, float]]:
    if not atoms_list:
        raise RuntimeError("No candidate structures were available for selection.")
    ranked = sorted(
        ((idx, atoms, _energy(atoms)) for idx, atoms in enumerate(atoms_list, start=1)),
        key=lambda item: item[2],
    )
    return [(idx, atoms.copy(), float(energy)) for idx, atoms, energy in ranked[: max(1, int(n))]]

def generate_radial_growth_candidates(
    cluster,
    new_symbol="Cu",
    bond_distance=2.45,
    max_candidates=50,
):
    if len(cluster) == 0:
        return [_single_attachment_candidate(cluster, new_symbol)]

    positions = cluster.get_positions()
    com = positions.mean(axis=0)

    dmat = cluster.get_all_distances(mic=False)
    dmat[dmat < 1e-8] = np.inf

    # crude coordination estimate
    cutoff = 3.0
    coordination = (dmat < cutoff).sum(axis=1)

    radial_dist = np.linalg.norm(positions - com, axis=1)

    # Prefer low-coordinate, outer atoms
    order = sorted(
        range(len(cluster)),
        key=lambda i: (coordination[i], -radial_dist[i])
    )

    candidates = []

    for i in order[:max_candidates]:
        direction = positions[i] - com
        norm = np.linalg.norm(direction)

        if norm < 1e-8:
            continue

        direction /= norm
        new_pos = positions[i] + bond_distance * direction

        candidate = cluster.copy()
        candidate += Atoms(new_symbol, positions=[new_pos])
        candidates.append(candidate)

    return candidates or [_single_attachment_candidate(cluster, new_symbol)]


def grow_nanocluster_by_atom_addition(
    seed_atoms: Atoms,
    *,
    add_symbol: str,
    n_additions: int,
    output_root: str | Path = "scratch/nanocluster_growth",
    mace_model_path: str | Path = "similarity/run/mace-omat-0-medium.model",
    mace_device: str = "cpu",
    mace_fmax: float = 0.03,
    mace_max_steps: int = 250,
    mace_default_dtype: str = "float64",
    site_symbols: list[str] | None = None,
    passive_symbols: list[str] | None = None,
    max_structures_per_generation: int | None = None,
    keep_parents_per_generation: int = 1,
    similarity_threshold: float | None = None,
    secondary_similarity_threshold: float | None = None,
    similarity_batch_size: int = 100,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Iteratively grow a nanocluster by adding one atom per generation.

    Each generation reuses the existing co-adsorbate candidate generator and
    REMatch uniqueness filters, then fully relaxes every candidate with MACE.
    The lowest-energy relaxed unique structure becomes the parent for the next
    generation.
    """
    if int(n_additions) <= 0:
        raise ValueError("n_additions must be positive.")
    add_symbol = str(add_symbol).strip()
    if not add_symbol:
        raise ValueError("add_symbol is required.")
    if mace_default_dtype not in {"float32", "float64"}:
        raise ValueError("mace_default_dtype must be 'float32' or 'float64'.")
    if int(keep_parents_per_generation) <= 0:
        raise ValueError("keep_parents_per_generation must be positive.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_csv = output_root / "growth_summary.csv"
    selected_db_path = output_root / "selected_growth_path.db"
    selected_traj_path = output_root / "selected_growth_path.traj"

    if site_symbols is None:
        site_symbols = sorted(set(seed_atoms.get_chemical_symbols()) | {add_symbol})
    if passive_symbols is None:
        passive_symbols = []

    parents = [seed_atoms.copy()]
    selected_records: list[tuple[Atoms, dict[str, object]]] = []
    selected_records.append(
        (
            seed_atoms.copy(),
            {
                "generation": 0,
                "added_symbol": "",
                "n_atoms": len(seed_atoms),
                "selected_final_row_id": "",
                "selected_energy_eV": "",
                "rank_in_generation": 1,
                "generation_dir": "",
                "final_db_path": "",
            },
        )
    )
    summary_rows: list[dict[str, object]] = [
        {
            "generation": 0,
            "added_symbol": "",
            "n_atoms": len(seed_atoms),
            "selected_final_row_id": "",
            "selected_energy_eV": "",
            "rank_in_generation": 1,
            "generation_dir": "",
            "final_db_path": "",
        }
    ]

    for generation in range(1, int(n_additions) + 1):
        generation_dir = output_root / f"generation_{generation:03d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(
                f"[Growth] Generation {generation}/{n_additions}: "
                f"adding {add_symbol} to {len(parents)} parent(s)."
            )

        candidates = []
        for parent_id, parent in enumerate(parents, start=1):
            try:
                from cats.working_directory.coadsorbate_generator import run_coadsorbate_generation

                parent_candidates = run_coadsorbate_generation(
                    parent,
                    adsorbates=[_single_atom_adsorbate(add_symbol)],
                    output_db=None,
                    site_symbols=site_symbols,
                    passive_symbols=passive_symbols,
                    max_structures=max_structures_per_generation,
                    cluster_symbols=[add_symbol],
                    allow_6fold=True,
                    skip_abnormal_bond_screening=True,
                )
                if not parent_candidates:
                    raise RuntimeError("coadsorbate generator returned no candidates")
            except Exception as e:
                print(f"[Growth] Error occurred while generating co-adsorbate candidates: {e}")
                print("[Growth] Falling back to radial growth.")
                parent_candidates = generate_radial_growth_candidates(
                    parent,
                    new_symbol=add_symbol,
                    bond_distance=2.45,
                    max_candidates=max_structures_per_generation or 50,
                )
            for atoms in parent_candidates:
                atoms.info["parent_id"] = parent_id
                candidates.append(atoms)
        if not candidates:
            raise RuntimeError(f"Generation {generation} produced no candidates.")
        from similarity.workflows.coadsorbate_similarity import _write_atoms_db

        _write_atoms_db(generation_dir / "candidates.db", candidates)

        primary_threshold = 0.9999 if similarity_threshold is None else float(similarity_threshold)
        pre_relax_atoms = _deduplicate(
            list(candidates),
            similarity_threshold=primary_threshold,
            report_csv_path=generation_dir / "primary_similarity.csv",
            support_species=passive_symbols,
            batch_size=similarity_batch_size,
            verbose=verbose,
            label="Primary nanocluster similarity",
        )
        if not pre_relax_atoms:
            raise RuntimeError(f"Generation {generation} primary pruning removed all candidates.")

        _write_atoms_db(generation_dir / "pre_relax.db", pre_relax_atoms)

        relaxed_atoms, relaxation_report_df = _relax_full_cluster_with_mace(
            pre_relax_atoms,
            model_path=mace_model_path,
            fmax=mace_fmax,
            max_steps=mace_max_steps,
            device=mace_device,
            dispersion=False,
            default_dtype=mace_default_dtype,
            output_db_path=generation_dir / "relaxed.db",
            verbose=verbose,
        )
        relaxation_report_df.to_csv(generation_dir / "relaxation.csv", index=False)
        if not relaxed_atoms:
            raise RuntimeError(f"Generation {generation} produced no converged relaxed candidates.")

        secondary_threshold = (
            primary_threshold
            if secondary_similarity_threshold is None
            else float(secondary_similarity_threshold)
        )
        final_atoms = _deduplicate(
            relaxed_atoms,
            similarity_threshold=secondary_threshold,
            report_csv_path=generation_dir / "secondary_similarity.csv",
            support_species=passive_symbols,
            batch_size=similarity_batch_size,
            verbose=verbose,
            label="Secondary nanocluster similarity",
        )
        if not final_atoms:
            raise RuntimeError(f"Generation {generation} secondary pruning removed all candidates.")

        _write_atoms_db(generation_dir / "final_unique.db", final_atoms)

        selected = _select_lowest_energy_n(
            list(final_atoms),
            n=keep_parents_per_generation,
        )
        parents = [atoms.copy() for _, atoms, _ in selected]
        write(generation_dir / "selected.xyz", parents, parallel=False)
        _write_summary_txt(
            generation_dir / "summary.txt",
            n_candidates=len(candidates),
            n_pre_relax=len(pre_relax_atoms),
            n_relaxed=len(relaxed_atoms),
            n_final=len(final_atoms),
            selected_row_id=selected[0][0],
            selected_energy=selected[0][2],
        )

        for rank, (selected_row_id, selected_atoms, selected_energy) in enumerate(selected, start=1):
            row = {
                "generation": generation,
                "added_symbol": add_symbol,
                "n_atoms": len(selected_atoms),
                "selected_final_row_id": selected_row_id,
                "selected_energy_eV": selected_energy,
                "rank_in_generation": rank,
                "generation_dir": str(generation_dir),
                "final_db_path": str(generation_dir / "final_unique.db"),
            }
            summary_rows.append(row)
            selected_records.append((selected_atoms.copy(), dict(row)))
        if verbose:
            print(
                f"[Growth] Kept {len(parents)} parent(s); best final row {selected[0][0]} "
                f"with energy {selected[0][2]:.6f} eV."
            )

    with connect(str(selected_db_path), serial=True) as db:
        for atoms, row in selected_records:
            data = dict(row)
            db.write(atoms, data=data, generation=int(row["generation"]))
    write(selected_traj_path, [atoms for atoms, _ in selected_records], parallel=False)

    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return {
        "output_root": output_root,
        "summary_csv": summary_csv,
        "selected_db_path": selected_db_path,
        "selected_traj_path": selected_traj_path,
    }


def _self_check() -> None:
    from ase.calculators.lj import LennardJones

    first_growth = generate_radial_growth_candidates(Atoms("Au", positions=[[0, 0, 0]]), "Au")
    assert len(first_growth) == 1
    assert len(first_growth[0]) == 2
    empty_growth = generate_radial_growth_candidates(Atoms(), "Au")
    assert len(empty_growth) == 1
    assert len(empty_growth[0]) == 1
    a = Atoms("Au", positions=[[0, 0, 0]])
    b = Atoms("Au", positions=[[1, 0, 0]])
    a.calc = SinglePointCalculator(a, energy=-2.0)
    b.calc = SinglePointCalculator(b, energy=-1.0)
    assert [idx for idx, _, _ in _select_lowest_energy_n([b, a], n=2)] == [2, 1]
    assert _is_bonded_cluster(Atoms("Au3", positions=[[0, 0, 0], [2.7, 0, 0], [5.4, 0, 0]]))
    assert not _is_bonded_cluster(Atoms("Au3", positions=[[0, 0, 0], [2.7, 0, 0], [9.0, 0, 0]]))
    supported_split = Atoms(
        "ZnOAu2", positions=[[0, 0, 0], [1.2, 0, 0], [0, 0, 2], [9, 0, 2]]
    )
    assert not _is_bonded_cluster(supported_split[[2, 3]])

    supported = Atoms("ZnOAu", positions=[[0, 0, 0], [1.2, 0, 0], [2.4, 0, 0]])
    supported.set_constraint(FixAtoms(indices=[0]))
    output_db = Path("scratch/constraint_restore_self_check.db")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    try:
        restored, _ = _relax_full_cluster_with_mace(
            [supported], model_path="unused", fmax=1e10, max_steps=1,
            device="cpu", dispersion=False, default_dtype="float64",
            output_db_path=output_db,
            verbose=False, fixed_symbols=["Zn", "O"],
            calculators=[LennardJones()],
        )
        with connect(str(output_db), serial=True) as db:
            saved = db.get(1).toatoms()
    finally:
        output_db.unlink(missing_ok=True)
    assert restored[0].constraints[0].get_indices().tolist() == [0]
    assert saved.constraints[0].get_indices().tolist() == [0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grow a nanocluster by adding one atom at a time with MACE ranking."
    )
    parser.add_argument("seed_structure", nargs="?", help="Initial cluster structure readable by ASE.")
    parser.add_argument("--add-symbol", help="Element symbol to add each generation.")
    parser.add_argument("--n-additions", type=int, help="Number of atoms to add.")
    parser.add_argument("--output-root", default="scratch/nanocluster_growth")
    parser.add_argument("--mace-model-path", default="similarity/run/mace-omat-0-medium.model")
    parser.add_argument("--mace-device", default="cpu")
    parser.add_argument("--mace-fmax", type=float, default=0.03)
    parser.add_argument("--mace-max-steps", type=int, default=250)
    parser.add_argument("--mace-default-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--max-structures-per-generation", type=int, default=None)
    parser.add_argument("--keep-parents-per-generation", type=int, default=1)
    parser.add_argument("--similarity-threshold", type=float, default=0.9999)
    parser.add_argument("--similarity-batch-size", type=int, default=100)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        print("self-check passed")
        return
    if not args.add_symbol:
        parser.error("--add-symbol is required unless --self-check is used.")
    if args.n_additions is None:
        parser.error("--n-additions is required unless --self-check is used.")

    seed_atoms = (
        read(args.seed_structure, parallel=False)
        if args.seed_structure
        else Atoms(args.add_symbol, positions=[[0.0, 0.0, 0.0]])
    )
    grow_nanocluster_by_atom_addition(
        seed_atoms,
        add_symbol=args.add_symbol,
        n_additions=args.n_additions,
        output_root=args.output_root,
        mace_model_path=args.mace_model_path,
        mace_device=args.mace_device,
        mace_fmax=args.mace_fmax,
        mace_max_steps=args.mace_max_steps,
        mace_default_dtype=args.mace_default_dtype,
        max_structures_per_generation=args.max_structures_per_generation,
        keep_parents_per_generation=args.keep_parents_per_generation,
        similarity_threshold=args.similarity_threshold,
        similarity_batch_size=args.similarity_batch_size,
    )


if __name__ == "__main__":
    main()
