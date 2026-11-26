from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.db import connect
from ase.geometry import wrap_positions
from ase.io import read, write

PARENT_RATTLES = 3

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _growth_helpers():
    try:
        from utilities.similarity.nanocluster_growth_workflow import (
            _deduplicate,
            _relax_full_cluster_with_mace,
            _single_atom_adsorbate,
            generate_radial_growth_candidates,
        )
    except ModuleNotFoundError:
        from nanocluster_growth_workflow import (
            _deduplicate,
            _relax_full_cluster_with_mace,
            _single_atom_adsorbate,
            generate_radial_growth_candidates,
        )
    return (
        _deduplicate,
        _relax_full_cluster_with_mace,
        _single_atom_adsorbate,
        generate_radial_growth_candidates,
    )


def optimize_alloy_nanocluster(
    seed_atoms: Atoms | list[Atoms],
    *,
    output_root: str | Path = "scratch/alloy_nanocluster_optimisation",
    mace_model_path: str | Path = "similarity/run/mace-omat-0-medium.model",
    mace_device: str = "cuda",
    mace_fmax: float = 0.05,
    mace_max_steps: int = 250,
    mace_default_dtype: str = "float64",
    mace_dispersion: bool = True,
    mace_enable_cueq: bool = True,
    verbose: bool = True,
) -> dict[str, Path]:
    """Relax one fixed-composition alloy seed through the shared MACE cache."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _, relax, _, _ = _growth_helpers()
    seeds = seed_atoms if isinstance(seed_atoms, list) else [seed_atoms]
    relaxed, _ = relax(
        [atoms.copy() for atoms in seeds],
        model_path=mace_model_path,
        fmax=float(mace_fmax),
        max_steps=int(mace_max_steps),
        device=mace_device,
        dispersion=bool(mace_dispersion),
        default_dtype=mace_default_dtype,
        output_db_path=output_root / "relaxations.db",
        verbose=bool(verbose),
        enable_cueq=bool(mace_enable_cueq),
    )
    final_path = output_root / "optimized.extxyz"
    write(final_path, relaxed, parallel=False)
    return {
        "output_root": output_root,
        "cache_db_path": output_root / "relaxations.db",
        "final_xyz_path": final_path,
    }


def _energy(atoms: Atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        calc = getattr(atoms, "calc", None)
        if calc is not None and hasattr(calc, "results") and "energy" in calc.results:
            return float(calc.results["energy"])
        raise


def _above_z_cutoff(atoms: Atoms, symbols: set[str], cutoff: float | None) -> bool:
    return cutoff is None or all(atom.z >= float(cutoff) for atom in atoms if atom.symbol in symbols)


def _copy_with_energy(atoms: Atoms) -> Atoms:
    copied = atoms.copy()
    copied.calc = SinglePointCalculator(copied, energy=_energy(atoms))
    return copied


def _composition_counts(atoms: Atoms, symbols: list[str]) -> dict[str, int]:
    counts = {symbol: 0 for symbol in symbols}
    for symbol in atoms.get_chemical_symbols():
        if symbol in counts:
            counts[symbol] += 1
    return counts


def _composition_key(atoms: Atoms, symbols: list[str]) -> str:
    counts = _composition_counts(atoms, symbols)
    return "".join(f"{symbol}{counts[symbol]}" for symbol in symbols)


def _growth_attachment_and_bonded_symbols(
    *,
    add_symbols: list[str],
    site_symbols: list[str],
    attachment_symbols: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    attachment_symbols = list(
        attachment_symbols if attachment_symbols is not None else site_symbols or add_symbols
    )
    bonded_symbols = sorted(set(add_symbols) | set(attachment_symbols))
    return attachment_symbols, bonded_symbols


def atomization_energy_eV(atoms: Atoms, reference_energies_eV: dict[str, float]) -> float:
    total = _energy(atoms)
    reference = sum(
        reference_energies_eV[symbol]
        for symbol in atoms.get_chemical_symbols()
        if symbol in reference_energies_eV
    )
    return float(reference - total)


def atomization_energy_per_atom_eV(atoms: Atoms, reference_energies_eV: dict[str, float]) -> float:
    return atomization_energy_eV(atoms, reference_energies_eV) / max(len(atoms), 1)


def _single_atom_reference_mpi_task(symbol, calculator, _settings):
    atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
    atoms.calc = calculator
    return symbol, float(atoms.get_potential_energy())


def mace_single_atom_reference_energies(
    symbols: list[str],
    *,
    model_path: str | Path,
    device: str,
    dispersion: bool = True,
    default_dtype: str = "float64",
    enable_cueq: bool = True,
) -> dict[str, float]:
    from similarity.workflows.mace_pool import get_mace_calculators, run_mace_tasks

    calculator_settings = {
        "model": str(model_path),
        "dispersion": bool(dispersion),
        "default_dtype": str(default_dtype),
        "device": str(device),
        "enable_cueq": bool(enable_cueq),
    }
    unique_symbols = sorted(set(symbols))
    print(
        f"[MACE reference energies] Initializing calculators for "
        f"{len(unique_symbols)} element(s): {', '.join(unique_symbols)}"
    )
    calculators = get_mace_calculators(**calculator_settings)
    references = run_mace_tasks(
        unique_symbols,
        calculators,
        lambda symbol, calculator: _single_atom_reference_mpi_task(
            symbol, calculator, {}
        ),
        label="MACE reference energies",
        distributed_operation=(
            "utilities.similarity.alloy_nanocluster_growth_workflow:"
            "_single_atom_reference_mpi_task"
        ),
        distributed_settings={},
        calculator_settings=calculator_settings,
    )
    return dict(references)


def _bulk_reference_mpi_task(symbol, calculator, settings):
    import numpy as np
    from ase.build import bulk
    from ase.calculators.calculator import PropertyNotImplementedError
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.data import atomic_numbers, reference_states
    from ase.filters import ExpCellFilter
    from ase.optimize import BFGS

    atoms = bulk(symbol)
    atoms.calc = calculator
    reference_state = reference_states[atomic_numbers[symbol]] or {}
    try:
        initial_stress = np.asarray(atoms.get_stress(), dtype=float)
    except PropertyNotImplementedError as exc:
        return symbol, None, {
            "symbol": symbol,
            "stress_supported": False,
            "reason": str(exc),
            "ase_reference_symmetry": reference_state.get("symmetry", ""),
        }
    if not np.all(np.isfinite(initial_stress)):
        raise RuntimeError(f"MACE returned non-finite stress for bulk {symbol}.")

    initial_energy = float(atoms.get_potential_energy())
    optimizer = BFGS(ExpCellFilter(atoms), logfile=None)
    converged = optimizer.run(
        fmax=float(settings["fmax"]),
        steps=int(settings["max_steps"]),
    )
    if not converged:
        raise RuntimeError(
            f"Bulk {symbol} optimization did not converge in {settings['max_steps']} steps."
        )
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    stress = np.asarray(atoms.get_stress(), dtype=float)
    optimized = atoms.copy()
    optimized.calc = SinglePointCalculator(
        optimized,
        energy=energy,
        forces=forces,
        stress=stress,
    )
    energy_per_atom = energy / len(optimized)
    optimized.info.update(
        bulk_reference_symbol=symbol,
        bulk_energy_eV=energy,
        bulk_energy_per_atom_eV=energy_per_atom,
        bulk_optimization_steps=optimizer.get_number_of_steps(),
    )
    return symbol, optimized, {
        "symbol": symbol,
        "stress_supported": True,
        "ase_reference_symmetry": reference_state.get("symmetry", ""),
        "initial_energy_eV_per_atom": initial_energy / len(atoms),
        "optimized_energy_eV": energy,
        "optimized_energy_eV_per_atom": energy_per_atom,
        "optimized_volume_A3_per_atom": optimized.get_volume() / len(optimized),
        "optimization_steps": optimizer.get_number_of_steps(),
        "max_abs_stress_eV_A3": float(np.max(np.abs(stress))),
    }


def mace_bulk_reference_energies(
    symbols: list[str],
    *,
    output_root: str | Path,
    model_path: str | Path,
    device: str,
    dispersion: bool = True,
    default_dtype: str = "float64",
    enable_cueq: bool = True,
    fmax: float = 0.01,
    max_steps: int = 500,
) -> dict[str, float] | None:
    from similarity.workflows.mace_pool import get_mace_calculators, run_mace_tasks

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "bulk_reference_status.json"
    summary_path = output_root / "bulk_reference_summary.csv"
    structures_path = output_root / "optimized_bulk_structures.db"
    unique_symbols = sorted(set(symbols))
    settings = {
        "symbols": unique_symbols,
        "model": str(model_path),
        "dispersion": bool(dispersion),
        "default_dtype": str(default_dtype),
        "device": str(device),
        "enable_cueq": bool(enable_cueq),
        "fmax": float(fmax),
        "max_steps": int(max_steps),
    }
    if status_path.exists():
        stored = json.loads(status_path.read_text(encoding="utf-8"))
        if stored.get("settings") != settings:
            raise ValueError(f"Cannot change bulk-reference settings in: {output_root}")
        if stored.get("status") == "stress_unsupported":
            print("[AlloyGrowth] MACE model does not provide stress; skipping bulk references")
            return None
        references = dict(stored.get("reference_energies_eV_per_atom") or {})
        if stored.get("status") == "complete" and all(
            symbol in references for symbol in unique_symbols
        ):
            print("[AlloyGrowth] Reusing optimized elemental bulk references")
            return {symbol: float(references[symbol]) for symbol in unique_symbols}

    calculator_settings = {
        "model": str(model_path),
        "dispersion": bool(dispersion),
        "default_dtype": str(default_dtype),
        "device": str(device),
        "enable_cueq": bool(enable_cueq),
    }
    calculators = get_mace_calculators(**calculator_settings)
    results = run_mace_tasks(
        unique_symbols,
        calculators,
        lambda symbol, calculator: _bulk_reference_mpi_task(symbol, calculator, settings),
        label="MACE elemental bulk references",
        distributed_operation=(
            "utilities.similarity.alloy_nanocluster_growth_workflow:_bulk_reference_mpi_task"
        ),
        distributed_settings=settings,
        calculator_settings=calculator_settings,
    )
    rows = [row for _symbol, _atoms, row in results]
    _write_csv(summary_path, rows)
    unsupported = [row for row in rows if not bool(row["stress_supported"])]
    if unsupported:
        _write_json(
            status_path,
            {"status": "stress_unsupported", "settings": settings, "rows": rows},
        )
        print("[AlloyGrowth] MACE model does not provide stress; skipping bulk references")
        return None

    structures = [atoms for _symbol, atoms, _row in results if atoms is not None]
    references = {
        str(row["symbol"]): float(row["optimized_energy_eV_per_atom"])
        for row in rows
    }
    _write_db(structures_path, structures)
    _write_json(
        status_path,
        {
            "status": "complete",
            "settings": settings,
            "reference_energies_eV_per_atom": references,
            "rows": rows,
        },
    )
    return references


def _rank_by_atomization_energy(
    atoms_list: list[Atoms],
    *,
    concentration_symbols: list[str],
    reference_energies_eV: dict[str, float],
    bulk_reference_energies_eV_per_atom: dict[str, float] | None = None,
    target_mixing_energy: bool = False,
) -> list[dict[str, object]]:
    rows = []
    for row_id, atoms in enumerate(atoms_list, start=1):
        e_atomization = atomization_energy_eV(atoms, reference_energies_eV)
        counts = _composition_counts(atoms, concentration_symbols)
        n_atoms = sum(counts.values())
        e_atomization_per_atom = e_atomization / max(len(atoms), 1)
        rows.append(
            {
                "row_id": row_id,
                "atoms": atoms,
                "composition": _composition_key(atoms, concentration_symbols),
                "n_atoms": n_atoms,
                "n_system_atoms": len(atoms),
                "energy_eV": _energy(atoms),
                "atomization_energy_eV": e_atomization,
                "atomization_energy_per_atom_eV": e_atomization_per_atom,
                **{f"n_{symbol}": counts[symbol] for symbol in concentration_symbols},
                **{
                    f"x_{symbol}": counts[symbol] / max(n_atoms, 1)
                    for symbol in concentration_symbols
                },
            }
        )
    rows = _bulk_referenced_mixing_rows(
        rows,
        concentration_symbols,
        bulk_reference_energies_eV_per_atom,
    )
    use_bulk_mixing = bool(target_mixing_energy) and bool(rows) and all(
        row["bulk_mixing_energy_per_atom_eV"] != "" for row in rows
    )
    criterion = (
        "bulk_mixing_energy_per_atom_eV"
        if use_bulk_mixing
        else "atomization_energy_per_atom_eV"
    )
    for row in rows:
        row["selection_criterion"] = criterion
        row["selection_score_eV_per_atom"] = row[criterion]
    return sorted(
        rows,
        key=lambda row: (
            str(row["composition"]),
            (
                float(row["bulk_mixing_energy_per_atom_eV"])
                if use_bulk_mixing
                else -float(row["atomization_energy_per_atom_eV"])
            ),
        ),
    )


def _select_per_composition(
    ranked_rows: list[dict[str, object]],
    *,
    keep_per_composition: int,
) -> tuple[list[Atoms], list[dict[str, object]]]:
    seen = defaultdict(int)
    selected = []
    output_rows = []
    for row in ranked_rows:
        row = dict(row)
        atoms = row.pop("atoms")
        composition = str(row["composition"])
        seen[composition] += 1
        keep = seen[composition] <= int(keep_per_composition)
        row["rank_in_composition"] = seen[composition]
        row["selected"] = keep
        output_rows.append(row)
        if keep:
            selected.append(_copy_with_energy(atoms))
    return selected, output_rows


def _alloy_atom_count(atoms: Atoms, concentration_symbols: list[str]) -> int:
    symbols = set(concentration_symbols)
    return sum(atom.symbol in symbols for atom in atoms)


def _size_tightened_similarity_threshold(
    baseline: float,
    alloy_atom_count: int,
) -> float:
    """Scale a one-atom REMatch threshold toward 1 as the alloy grows."""
    baseline = float(baseline)
    if not 0.0 <= baseline <= 1.0:
        raise ValueError("Similarity thresholds must be in [0, 1].")
    return 1.0 - (1.0 - baseline) / max(1, int(alloy_atom_count))


def _concentration(atoms: Atoms, symbol: str, concentration_symbols: list[str]) -> float:
    total = _alloy_atom_count(atoms, concentration_symbols)
    if total <= 0:
        raise ValueError("Cannot calculate concentration for a structure without alloy atoms.")
    return sum(atom.symbol == symbol for atom in atoms) / total


def _stage_generation_count(
    seed: Atoms,
    *,
    concentration_symbol: str,
    concentration_symbols: list[str],
    concentration_span: float,
) -> int:
    total = _alloy_atom_count(seed, concentration_symbols)
    center = _concentration(seed, concentration_symbol, concentration_symbols)
    span = float(concentration_span)
    lower = center - span
    upper = center + span
    if not 0 < span < 1:
        raise ValueError("concentration_span must be between 0 and 1.")
    if lower <= 0 or upper >= 1:
        raise ValueError(
            f"A +/-{span:.4f} concentration window around {center:.4f} reaches a pure-element "
            "boundary; choose a smaller span or a less extreme seed."
        )
    lower_steps = math.ceil(total * span / lower)
    upper_steps = math.ceil(total * span / (1.0 - upper))
    return max(1, lower_steps, upper_steps)


def _concentration_count_bounds(
    total: int,
    *,
    lower: float,
    upper: float,
    strict: bool,
) -> tuple[int, int]:
    if strict:
        return math.ceil(lower * total), math.floor(upper * total)
    return math.floor(lower * total + 0.5), math.floor(upper * total + 0.5)


def _filter_concentration_window(
    atoms_list: list[Atoms],
    *,
    concentration_symbol: str,
    concentration_symbols: list[str],
    lower: float,
    upper: float,
    strict: bool = False,
) -> list[Atoms]:
    filtered = []
    for atoms in atoms_list:
        total = _alloy_atom_count(atoms, concentration_symbols)
        minimum, maximum = _concentration_count_bounds(
            total,
            lower=lower,
            upper=upper,
            strict=strict,
        )
        count = sum(atom.symbol == concentration_symbol for atom in atoms)
        if minimum <= count <= maximum:
            filtered.append(atoms)
    return filtered


def _select_restart_beam(
    atoms_list: list[Atoms],
    *,
    beam_width: int,
    concentration_symbols: list[str],
    reference_energies_eV: dict[str, float],
    bulk_reference_energies_eV_per_atom: dict[str, float] | None = None,
    target_mixing_energy: bool = False,
    require_alloy: bool = True,
) -> tuple[list[Atoms], list[dict[str, object]]]:
    if int(beam_width) <= 0:
        raise ValueError("restart_beam_width must be positive.")
    ranked = _rank_by_atomization_energy(
        atoms_list,
        concentration_symbols=concentration_symbols,
        reference_energies_eV=reference_energies_eV,
        bulk_reference_energies_eV_per_atom=bulk_reference_energies_eV_per_atom,
        target_mixing_energy=target_mixing_energy,
    )
    use_bulk_mixing = bool(target_mixing_energy) and bool(ranked) and all(
        row["bulk_mixing_energy_per_atom_eV"] != "" for row in ranked
    )
    ranked = sorted(
        (
            row
            for row in ranked
            if not require_alloy
            or all(int(row.get(f"n_{symbol}", 0)) > 0 for symbol in concentration_symbols)
        ),
        key=lambda row: (
            (
                float(row["bulk_mixing_energy_per_atom_eV"])
                if use_bulk_mixing
                else -float(row["atomization_energy_per_atom_eV"])
            ),
            float(row["energy_eV"]),
            str(row["composition"]),
            int(row["row_id"]),
        ),
    )
    selected_rows = []
    selected_row_ids = set()
    seen_compositions = set()
    for unique_compositions_only in (True, False):
        for row in ranked:
            if int(row["row_id"]) in selected_row_ids:
                continue
            composition = str(row["composition"])
            if unique_compositions_only and composition in seen_compositions:
                continue
            selected_rows.append(row)
            selected_row_ids.add(int(row["row_id"]))
            seen_compositions.add(composition)
            if len(selected_rows) >= int(beam_width):
                break
        if len(selected_rows) >= int(beam_width):
            break

    selected = []
    output_rows = []
    for beam_rank, row in enumerate(selected_rows, start=1):
        atoms = _copy_with_energy(row["atoms"])
        atoms.info.update(
            restart_beam_rank=beam_rank,
            restart_composition=str(row["composition"]),
            atomization_energy_per_atom_eV=float(row["atomization_energy_per_atom_eV"]),
            selection_criterion=str(row["selection_criterion"]),
            selection_score_eV_per_atom=float(row["selection_score_eV_per_atom"]),
        )
        if row["bulk_mixing_energy_per_atom_eV"] != "":
            atoms.info["bulk_mixing_energy_per_atom_eV"] = float(
                row["bulk_mixing_energy_per_atom_eV"]
            )
        selected.append(atoms)
        output_row = {key: value for key, value in row.items() if key != "atoms"}
        output_row["restart_beam_rank"] = beam_rank
        output_rows.append(output_row)
    return selected, output_rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(dict.fromkeys(key for row in rows for key in row)),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _atoms_signature(atoms: Atoms) -> dict[str, object]:
    return {
        "symbols": atoms.get_chemical_symbols(),
        "positions": [[round(float(value), 10) for value in row] for row in atoms.positions],
        "cell": [[round(float(value), 10) for value in row] for row in atoms.cell],
        "pbc": [bool(value) for value in atoms.get_pbc()],
        "energy_eV": round(_energy(atoms), 10),
    }


def _write_db(path: Path, atoms_list: list[Atoms]) -> None:
    from similarity.workflows.mace_pool import write_ase_database_atomically

    path.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"[AlloyGrowth] Checkpointing {len(atoms_list)} structure(s) to {path}",
        flush=True,
    )
    write_ase_database_atomically(path, ((atoms, None) for atoms in atoms_list))


def _best_per_size_composition(
    ranked_rows: list[dict[str, object]],
    *,
    generation: int,
    start_id: int = 1,
) -> tuple[list[Atoms], list[dict[str, object]]]:
    best: dict[tuple[int, str], dict[str, object]] = {}
    for row in ranked_rows:
        key = (int(row["n_atoms"]), str(row["composition"]))
        if key not in best:
            best[key] = row

    atoms_list = []
    rows = []
    for stable_id, row in enumerate(best.values(), start=start_id):
        atoms = _copy_with_energy(row["atoms"])
        atoms.info["stable_structure_id"] = stable_id
        atoms.info["generation"] = generation
        atoms.info["composition"] = row["composition"]
        atoms.info["atomization_energy_per_atom_eV"] = row["atomization_energy_per_atom_eV"]
        atoms_list.append(atoms)

        csv_row = dict(row)
        csv_row.pop("atoms")
        csv_row["generation"] = generation
        csv_row["stable_structure_id"] = stable_id
        rows.append(csv_row)
    return atoms_list, rows


def _nanoparticle_referenced_rows(
    rows: list[dict[str, object]],
    concentration_symbols: list[str],
) -> list[dict[str, object]]:
    references: dict[int, dict[str, float]] = defaultdict(dict)
    for row in rows:
        size = int(row["n_atoms"])
        for symbol in concentration_symbols:
            if int(row.get(f"n_{symbol}", 0)) == size:
                energy = float(row["energy_eV"])
                references[size][symbol] = min(
                    energy,
                    references[size].get(symbol, energy),
                )

    output = []
    for row in rows:
        size = int(row["n_atoms"])
        output_row = dict(row)
        output_row["mixing_energy_eV"] = ""
        output_row["mixing_energy_per_atom_eV"] = ""
        if size > 0 and all(
            symbol in references[size] for symbol in concentration_symbols
        ):
            reference = sum(
                int(row.get(f"n_{symbol}", 0)) * references[size][symbol] / size
                for symbol in concentration_symbols
            )
            mixing_energy = float(row["energy_eV"]) - reference
            output_row["mixing_energy_eV"] = mixing_energy
            output_row["mixing_energy_per_atom_eV"] = mixing_energy / size
        output.append(output_row)
    return output


def _bulk_referenced_mixing_rows(
    rows: list[dict[str, object]],
    concentration_symbols: list[str],
    bulk_reference_energies_eV_per_atom: dict[str, float] | None,
) -> list[dict[str, object]]:
    output = []
    for row in rows:
        output_row = dict(row)
        output_row["bulk_mixing_energy_eV"] = ""
        output_row["bulk_mixing_energy_per_atom_eV"] = ""
        alloy_atoms = int(row["n_atoms"])
        system_atoms = int(row.get("n_system_atoms", alloy_atoms))
        if bulk_reference_energies_eV_per_atom and alloy_atoms == system_atoms:
            reference = sum(
                int(row[f"n_{symbol}"])
                * float(bulk_reference_energies_eV_per_atom[symbol])
                for symbol in concentration_symbols
            )
            mixing_energy = float(row["energy_eV"]) - reference
            output_row["bulk_mixing_energy_eV"] = mixing_energy
            output_row["bulk_mixing_energy_per_atom_eV"] = mixing_energy / max(alloy_atoms, 1)
        output.append(output_row)
    return output


def _write_phase_scatter(
    path: Path,
    rows: list[dict[str, object]],
    concentration_symbols: list[str],
    *,
    y_key: str = "atomization_energy_per_atom_eV",
    y_label: str = "Atomization energy / atom (eV)",
    title: str = "Most stable alloy structures by size and composition",
) -> None:
    if not rows or not concentration_symbols:
        return
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"[AlloyGrowth] Skipping phase scatter plot: {exc}")
        return

    x_symbol = concentration_symbols[-1]
    x_key = f"x_{x_symbol}"
    xs = [float(row[x_key]) for row in rows]
    ys = [float(row[y_key]) for row in rows]
    colour_key, colour_label = next(
        (
            (key, label)
            for key, label in (
                ("mixing_energy_per_atom_eV", "Mixing energy / atom (eV)"),
                ("bulk_mixing_energy_per_atom_eV", "Bulk mixing energy / atom (eV)"),
            )
            if all(row.get(key, "") != "" for row in rows)
        ),
        ("n_atoms", "Atoms"),
    )
    colours = [float(row[colour_key]) for row in rows]

    fig, ax = plt.subplots(figsize=(7, 5))
    points = ax.scatter(xs, ys, c=colours, cmap="viridis", edgecolors="black", linewidths=0.4)
    ax.set_xlabel(f"{x_symbol} concentration")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    fig.colorbar(points, ax=ax, label=colour_label)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_bulk_mixing_minima_plot(
    png_path: Path,
    csv_path: Path,
    rows: list[dict[str, object]],
    concentration_symbols: list[str],
) -> None:
    x_symbol = concentration_symbols[-1]
    x_key = f"x_{x_symbol}"
    best: dict[float, dict[str, object]] = {}
    for row in rows:
        concentration = round(float(row[x_key]), 12)
        current = best.get(concentration)
        if current is None or (
            float(row["bulk_mixing_energy_per_atom_eV"]),
            int(row["n_atoms"]),
        ) < (
            float(current["bulk_mixing_energy_per_atom_eV"]),
            int(current["n_atoms"]),
        ):
            best[concentration] = dict(row)
    minima = [best[concentration] for concentration in sorted(best)]
    _write_csv(csv_path, minima)
    if not minima:
        return

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"[AlloyGrowth] Skipping concentration-minimum phase plot: {exc}")
        return

    xs = [float(row[x_key]) for row in minima]
    ys = [float(row["bulk_mixing_energy_per_atom_eV"]) for row in minima]
    sizes = [int(row["n_atoms"]) for row in minima]
    fig, axis = plt.subplots(figsize=(8, 5.6), constrained_layout=True)
    axis.plot(xs, ys, color="0.55", linewidth=1.0, zorder=1)
    points = axis.scatter(
        xs,
        ys,
        c=sizes,
        cmap="viridis",
        edgecolors="black",
        linewidths=0.4,
        zorder=2,
    )
    for x_value, y_value, atom_count in zip(xs, ys, sizes):
        axis.annotate(
            f"N={atom_count}",
            (x_value, y_value),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    axis.set(
        xlabel=f"{x_symbol} concentration",
        ylabel="Bulk-referenced mixing energy / atom (eV)",
        title="Lowest mixing energy at each visited concentration",
    )
    axis.grid(alpha=0.2)
    fig.colorbar(points, ax=axis, label="Alloy atom count")
    fig.savefig(png_path, dpi=200)
    plt.close(fig)


def _write_staged_campaign_analysis(
    output_root: Path,
    stable_structures: list[Atoms],
    stable_rows: list[dict[str, object]],
    concentration_symbols: list[str],
    *,
    bond_scale: float = 1.35,
) -> dict[str, Path]:
    from ase.data import covalent_radii

    if len(stable_structures) != len(stable_rows):
        raise ValueError(
            "Combined stable structure DB and summary CSV contain different row counts: "
            f"{len(stable_structures)} != {len(stable_rows)}"
        )

    composition_csv = output_root / "explored_compositions.csv"
    composition_png = output_root / "explored_compositions.png"
    coordination_csv = output_root / "stable_coordination_summary.csv"
    use_bulk_mixing = bool(stable_rows) and all(
        row.get("bulk_mixing_energy_per_atom_eV", "") != "" for row in stable_rows
    )
    atomization_paths = (
        output_root / "stable_atomization_vs_cross_coordination.png",
        output_root / f"stable_atomization_vs_{concentration_symbols[0]}_coordination.png",
        output_root / f"stable_atomization_vs_{concentration_symbols[1]}_coordination.png",
    )
    bulk_mixing_paths = (
        output_root / "stable_bulk_mixing_energy_vs_cross_coordination.png",
        output_root
        / f"stable_bulk_mixing_energy_vs_{concentration_symbols[0]}_coordination.png",
        output_root
        / f"stable_bulk_mixing_energy_vs_{concentration_symbols[1]}_coordination.png",
    )
    bulk_phase_png = output_root / "bulk_mixing_energy_phase_diagram.png"
    bulk_minima_png = output_root / "bulk_mixing_energy_concentration_minima.png"
    bulk_minima_csv = output_root / "bulk_mixing_energy_concentration_minima.csv"
    obsolete_phase_lines = output_root / "bulk_mixing_energy_phase_lines.png"
    if obsolete_phase_lines.exists():
        obsolete_phase_lines.unlink()

    explored: dict[tuple[str, int, tuple[int, ...]], dict[str, object]] = {}
    for ranked_path in sorted(output_root.rglob("ranked.csv")):
        relative = ranked_path.relative_to(output_root)
        stage = relative.parts[0]
        if stage.startswith("prestage_"):
            stage = "prestage"
        elif stage.startswith("stage_"):
            stage = "_".join(stage.split("_")[:2])
        with ranked_path.open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                try:
                    counts = tuple(int(float(raw[f"n_{symbol}"])) for symbol in concentration_symbols)
                    atom_count = int(float(raw["n_atoms"]))
                except (KeyError, TypeError, ValueError):
                    continue
                key = (stage, atom_count, counts)
                row = explored.setdefault(
                    key,
                    {
                        "stage": stage,
                        "atom_count": atom_count,
                        "composition": "".join(
                            f"{symbol}{count}" for symbol, count in zip(concentration_symbols, counts)
                        ),
                        "structure_count": 0,
                        **{
                            f"n_{symbol}": count
                            for symbol, count in zip(concentration_symbols, counts)
                        },
                    },
                )
                row["structure_count"] = int(row["structure_count"]) + 1

    def stage_sort_key(value: str) -> tuple[int, int | str]:
        if value == "prestage":
            return (0, 0)
        try:
            return (1, int(value.split("_")[-1]))
        except ValueError:
            return (2, value)

    stage_names = sorted(
        {str(row["stage"]) for row in explored.values()},
        key=stage_sort_key,
    )
    stage_order = {name: index for index, name in enumerate(stage_names)}
    composition_rows = []
    concentration_symbol = concentration_symbols[-1]
    for row in sorted(
        explored.values(),
        key=lambda item: (
            int(item["atom_count"]),
            stage_order[str(item["stage"])],
            str(item["composition"]),
        ),
    ):
        row = dict(row)
        row["stage_index"] = stage_order[str(row["stage"])]
        row[f"x_{concentration_symbol}"] = (
            int(row[f"n_{concentration_symbol}"]) / max(int(row["atom_count"]), 1)
        )
        composition_rows.append(row)
    _write_csv(composition_csv, composition_rows)

    coordination_rows = []
    first_symbol, second_symbol = concentration_symbols
    for atoms, stable_row in zip(stable_structures, stable_rows):
        metal_indices = [
            index for index, atom in enumerate(atoms) if atom.symbol in concentration_symbols
        ]
        pair_counts = defaultdict(int)
        coordination_counts = defaultdict(int)
        for offset, atom_i in enumerate(metal_indices):
            for atom_j in metal_indices[offset + 1 :]:
                cutoff = float(bond_scale) * (
                    float(covalent_radii[int(atoms[atom_i].number)])
                    + float(covalent_radii[int(atoms[atom_j].number)])
                )
                if float(atoms.get_distance(atom_i, atom_j, mic=bool(any(atoms.get_pbc())))) <= cutoff:
                    pair_counts[tuple(sorted((atoms[atom_i].symbol, atoms[atom_j].symbol)))] += 1
                    coordination_counts[atom_i] += 1
                    coordination_counts[atom_j] += 1
        total_bonds = sum(pair_counts.values())
        cross_bonds = pair_counts[tuple(sorted((first_symbol, second_symbol)))]
        counts = _composition_counts(atoms, concentration_symbols)
        coordination_rows.append(
            {
                **dict(stable_row),
                "coordination_cutoff_scale": float(bond_scale),
                "total_metal_metal_bonds": total_bonds,
                "mean_total_metal_coordination": 2.0 * total_bonds / max(len(metal_indices), 1),
                f"{first_symbol}_{second_symbol}_bonds": cross_bonds,
                "mean_cross_metal_coordination": 2.0 * cross_bonds / max(len(metal_indices), 1),
                f"mean_{first_symbol}_to_{second_symbol}_coordination": cross_bonds
                / max(counts[first_symbol], 1),
                f"mean_{second_symbol}_to_{first_symbol}_coordination": cross_bonds
                / max(counts[second_symbol], 1),
                f"mean_{first_symbol}_metal_coordination": sum(
                    coordination_counts[index]
                    for index in metal_indices
                    if atoms[index].symbol == first_symbol
                )
                / max(counts[first_symbol], 1),
                f"mean_{second_symbol}_metal_coordination": sum(
                    coordination_counts[index]
                    for index in metal_indices
                    if atoms[index].symbol == second_symbol
                )
                / max(counts[second_symbol], 1),
            }
        )
    _write_csv(coordination_csv, coordination_rows)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"[AlloyGrowth] Skipping staged campaign plots: {exc}")
    else:
        if composition_rows:
            fig, axis = plt.subplots(figsize=(8, 5.4), constrained_layout=True)
            points = axis.scatter(
                [int(row["atom_count"]) for row in composition_rows],
                [float(row[f"x_{concentration_symbol}"]) for row in composition_rows],
                c=[int(row["stage_index"]) for row in composition_rows],
                s=[25 + 12 * math.log1p(int(row["structure_count"])) for row in composition_rows],
                cmap="viridis",
                edgecolors="black",
                linewidths=0.35,
            )
            axis.set(
                xlabel="Alloy atom count",
                ylabel=f"{concentration_symbol} atomic fraction",
                title=f"Compositions explored in {output_root.name}",
            )
            axis.grid(alpha=0.2)
            colorbar = fig.colorbar(points, ax=axis, label="Search segment")
            colorbar.set_ticks(range(len(stage_names)))
            colorbar.set_ticklabels(stage_names)
            fig.savefig(composition_png, dpi=200)
            plt.close(fig)

        coordination_axes = (
            (
                "mean_cross_metal_coordination",
                f"Mean {first_symbol}–{second_symbol} coordination number",
            ),
            (
                f"mean_{first_symbol}_metal_coordination",
                f"Mean coordination number of {first_symbol}",
            ),
            (
                f"mean_{second_symbol}_metal_coordination",
                f"Mean coordination number of {second_symbol}",
            ),
        )
        energy_series = [
            (
                "atomization_energy_per_atom_eV",
                "Atomisation energy / atom (eV; higher is more stable)",
                atomization_paths,
            )
        ]
        if use_bulk_mixing:
            energy_series.append(
                (
                    "bulk_mixing_energy_per_atom_eV",
                    "Bulk-referenced mixing energy / atom (eV; lower is more stable)",
                    bulk_mixing_paths,
                )
            )
        for energy_key, energy_label, plot_paths in energy_series:
            for path, (x_key, x_label) in zip(plot_paths, coordination_axes):
                if not coordination_rows:
                    continue
                fig, axis = plt.subplots(figsize=(7.2, 5.6), constrained_layout=True)
                points = axis.scatter(
                    [float(row[x_key]) for row in coordination_rows],
                    [float(row[energy_key]) for row in coordination_rows],
                    c=[float(row["n_atoms"]) for row in coordination_rows],
                    cmap="plasma",
                    edgecolors="black",
                    linewidths=0.35,
                )
                axis.set(
                    xlabel=x_label,
                    ylabel=energy_label,
                    title="Most stable structure per size and composition",
                )
                axis.grid(alpha=0.2)
                fig.colorbar(points, ax=axis, label="Alloy atom count")
                fig.savefig(path, dpi=200)
                plt.close(fig)

    if use_bulk_mixing:
        _write_phase_scatter(
            bulk_phase_png,
            coordination_rows,
            concentration_symbols,
            y_key="bulk_mixing_energy_per_atom_eV",
            y_label="Bulk-referenced mixing energy / atom (eV)",
            title="Bulk-referenced alloy phase diagram",
        )
        _write_bulk_mixing_minima_plot(
            bulk_minima_png,
            bulk_minima_csv,
            coordination_rows,
            concentration_symbols,
        )

    return {
        "explored_compositions_csv": composition_csv,
        "explored_compositions_png": composition_png,
        "stable_coordination_csv": coordination_csv,
        "stable_atomization_vs_cross_coordination_png": atomization_paths[0],
        "stable_atomization_vs_first_metal_coordination_png": atomization_paths[1],
        "stable_atomization_vs_second_metal_coordination_png": atomization_paths[2],
        **(
            {
                "stable_bulk_mixing_vs_cross_coordination_png": bulk_mixing_paths[0],
                "stable_bulk_mixing_vs_first_metal_coordination_png": bulk_mixing_paths[1],
                "stable_bulk_mixing_vs_second_metal_coordination_png": bulk_mixing_paths[2],
                "bulk_mixing_energy_phase_diagram_png": bulk_phase_png,
                "bulk_mixing_energy_concentration_minima_png": bulk_minima_png,
                "bulk_mixing_energy_concentration_minima_csv": bulk_minima_csv,
            }
            if use_bulk_mixing
            else {}
        ),
    }


def _write_all_staged_campaign_analyses(
    output_root: Path,
    stable_structures: list[Atoms],
    stable_rows: list[dict[str, object]],
    concentration_symbols: list[str],
    reference_energies_eV: dict[str, float],
    bulk_reference_energies_eV_per_atom: dict[str, float] | None,
) -> dict[str, Path]:
    outputs = _write_staged_campaign_analysis(
        output_root,
        stable_structures,
        stable_rows,
        concentration_symbols,
    )
    stage_dirs = sorted(
        path
        for path in output_root.iterdir()
        if path.is_dir() and path.name.startswith(("prestage_", "stage_"))
    )
    for stage_dir in stage_dirs:
        stage_structures = [
            atoms
            for database in sorted(stage_dir.rglob("stable_alloy_structures.db"))
            for atoms in _read_db(database)
        ]
        if not stage_structures:
            continue
        ranked = _rank_by_atomization_energy(
            stage_structures,
            concentration_symbols=concentration_symbols,
            reference_energies_eV=reference_energies_eV,
            bulk_reference_energies_eV_per_atom=bulk_reference_energies_eV_per_atom,
        )
        selected_structures, selected_rows = _best_per_size_composition(
            ranked,
            generation=0,
        )
        _write_staged_campaign_analysis(
            stage_dir,
            selected_structures,
            _bulk_referenced_mixing_rows(
                _nanoparticle_referenced_rows(selected_rows, concentration_symbols),
                concentration_symbols,
                bulk_reference_energies_eV_per_atom,
            ),
            concentration_symbols,
        )
    return outputs


def _write_phase_lines(
    png_path: Path,
    html_path: Path | None,
    rows: list[dict[str, object]],
    concentration_symbols: list[str],
    *,
    y_key: str = "atomization_energy_per_atom_eV",
    y_total_key: str = "atomization_energy_eV",
    y_label: str = "Atomization energy / atom (eV)",
    title: str = "Most stable alloy atomization energies by size",
) -> None:
    if not rows or not concentration_symbols:
        return
    x_symbol = concentration_symbols[-1]
    x_key = f"x_{x_symbol}"
    by_size: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_size[int(row["n_atoms"])].append(row)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"[AlloyGrowth] Skipping phase line plot: {exc}")
    else:
        fig, ax = plt.subplots(figsize=(7, 5))
        for size, size_rows in sorted(by_size.items()):
            points = sorted(size_rows, key=lambda row: float(row[x_key]))
            ax.plot(
                [float(row[x_key]) for row in points],
                [float(row[y_key]) for row in points],
                marker="o",
                linewidth=1.2,
                label=str(size),
            )
        ax.set_xlabel(f"{x_symbol} concentration")
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.legend(title="Atoms", fontsize="small", ncols=2)
        fig.tight_layout()
        fig.savefig(png_path, dpi=200)
        plt.close(fig)

    if html_path is None:
        return

    try:
        import plotly.graph_objects as go
    except Exception as exc:
        print(f"[AlloyGrowth] Skipping interactive phase line plot: {exc}")
        return

    fig = go.Figure()
    for size, size_rows in sorted(by_size.items()):
        points = sorted(size_rows, key=lambda row: float(row[x_key]))
        fig.add_trace(
            go.Scatter(
                x=[float(row[x_key]) for row in points],
                y=[float(row[y_key]) for row in points],
                mode="lines+markers",
                name=f"{size} atoms",
                customdata=[
                    [
                        row["stable_structure_id"],
                        row["composition"],
                        row["energy_eV"],
                        row[y_total_key],
                    ]
                    for row in points
                ],
                hovertemplate=(
                    f"{x_symbol} concentration=%{{x:.3f}}<br>"
                    "Energy/atom=%{y:.6f} eV<br>"
                    "stable_id=%{customdata[0]}<br>"
                    "composition=%{customdata[1]}<br>"
                    "energy=%{customdata[2]:.6f} eV<br>"
                    "Energy=%{customdata[3]:.6f} eV<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=f"{x_symbol} concentration",
        yaxis_title=y_label,
        legend_title_text="Cluster size",
    )
    fig.write_html(html_path, include_plotlyjs="cdn")


def _read_db(path: Path, *, only_converged: bool = False) -> list[Atoms]:
    with connect(str(path), serial=True) as db:
        return [
            row.toatoms()
            for row in db.select()
            if not only_converged
            or (
                bool(row.data.get("converged", True))
                and bool(row.data.get("bonded", True))
            )
        ]


def _deduplicate_by_composition(
    atoms_list: list[Atoms],
    *,
    concentration_symbols: list[str],
    similarity_threshold: float,
    report_csv_path: Path,
    support_species: list[str],
    slab_shave_distance: float | None = None,
    batch_size: int = 100,
    verbose: bool = False,
) -> list[Atoms]:
    from similarity.workflows.coadsorbate_similarity import _deduplicate_atoms_in_batches

    def deduplicate(group: list[Atoms], report_path: Path, label: str) -> list[Atoms]:
        unique, report = _deduplicate_atoms_in_batches(
            group,
            batch_size=batch_size,
            similarity_threshold=similarity_threshold,
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
        report["similarity_threshold"] = float(similarity_threshold)
        report.to_csv(report_path, index=False)
        return unique

    groups: dict[str, list[tuple[int, Atoms]]] = {}
    for row_id, atoms in enumerate(atoms_list, start=1):
        groups.setdefault(_composition_key(atoms, concentration_symbols), []).append((row_id, atoms))

    if len(atoms_list) <= 1:
        _write_csv(
            report_csv_path,
            [{
                "row_id": 1,
                "kept": True,
                "max_similarity_to_kept": "",
                "closest_kept_row_id": "",
                "similarity_threshold": float(similarity_threshold),
            }],
        )
        return atoms_list
    if len(groups) <= 1:
        return deduplicate(
            atoms_list,
            report_csv_path,
            report_csv_path.stem,
        )

    unique_atoms = []
    report_rows = []
    for composition, entries in groups.items():
        if len(entries) == 1:
            row_id, atoms = entries[0]
            unique_atoms.append(atoms)
            report_rows.append(
                {
                    "row_id": row_id,
                    "kept": True,
                    "max_similarity_to_kept": "",
                    "closest_kept_row_id": "",
                    "composition": composition,
                    "similarity_threshold": float(similarity_threshold),
                }
            )
            continue
        part_report = report_csv_path.with_name(
            f"{report_csv_path.stem}_{composition}{report_csv_path.suffix}"
        )
        unique_atoms.extend(
            deduplicate(
                [atoms for _, atoms in entries],
                part_report,
                f"{report_csv_path.stem} {composition}",
            )
        )
        with part_report.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                local_row_id = int(row["row_id"])
                closest = row.get("closest_kept_row_id", "")
                row["composition"] = composition
                row["row_id"] = entries[local_row_id - 1][0]
                row["closest_kept_row_id"] = (
                    "" if closest == "" else entries[int(float(closest)) - 1][0]
                )
                report_rows.append(row)

    _write_csv(report_csv_path, report_rows)
    return unique_atoms


def _single_attachment_candidate(parent: Atoms, add_symbol: str) -> Atoms:
    from ase.data import atomic_numbers, covalent_radii

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


def _wrap_added_atom(candidate: Atoms) -> None:
    if len(candidate) and any(candidate.get_pbc()):
        candidate.positions[-1:] = wrap_positions(
            candidate.positions[-1:], candidate.cell, pbc=candidate.get_pbc()
        )


def _gas_atom_candidates(
    add_symbols: list[str], reference_energies_eV: dict[str, float]
) -> list[Atoms]:
    candidates = []
    for symbol in dict.fromkeys(add_symbols):
        atoms = Atoms(symbol, positions=[[0.0, 0.0, 0.0]])
        atoms.info.update(parent_id=0, added_symbol=symbol)
        atoms.calc = SinglePointCalculator(
            atoms, energy=reference_energies_eV[symbol], forces=[[0.0, 0.0, 0.0]]
        )
        candidates.append(atoms)
    return candidates


def _lower_energy_frame(
    best: tuple[float, Atoms] | None,
    candidate: Atoms,
) -> tuple[float, Atoms]:
    energy = _energy(candidate)
    return (energy, candidate.copy()) if best is None or energy < best[0] else best


def _condition_parents_with_rattle(
    parents: list[Atoms],
    *,
    generation_dir: Path,
    stdev: float,
    model_path: str | Path,
    device: str,
    dispersion: bool,
    default_dtype: str,
    fmax: float,
    max_steps: int,
    verbose: bool,
    fixed_symbols: list[str] | None = None,
    bonded_symbols: list[str] | None = None,
    enable_cueq: bool = True,
) -> list[Atoms]:
    try:
        from utilities.similarity.nanocluster_growth_workflow import _get_mace_calculator
    except ModuleNotFoundError:
        from nanocluster_growth_workflow import _get_mace_calculator

    calculator = _get_mace_calculator(
        SimpleNamespace(
            model_path=model_path,
            dispersion=dispersion,
            default_dtype=default_dtype,
            device=device,
            enable_cueq=enable_cueq,
        )
    )
    rattle_dir = generation_dir / "parent_rattle"
    rattle_dir.mkdir(parents=True, exist_ok=True)
    rattled_frames = []
    candidate_parent_ids = []
    initial_energies = []
    original_parents = [parent.copy() for parent in parents]
    for parent_id, parent in enumerate(parents, start=1):
        atoms = parent.copy()
        fixed = set(fixed_symbols or [])
        fixed_indices = [i for i, atom in enumerate(atoms) if atom.symbol in fixed]
        atoms.set_constraint(FixAtoms(indices=fixed_indices) if fixed_indices else None)
        atoms.calc = calculator
        initial_energies.append(_energy(atoms))
        for rattle_id in range(1, PARENT_RATTLES + 1):
            rattled = atoms.copy()
            rattled.rattle(stdev=float(stdev), seed=parent_id * 1000 + rattle_id)
            rattled.info.update(parent_id=parent_id, rattle_id=rattle_id)
            rattled_frames.append(rattled)
            candidate_parent_ids.append(parent_id)

    _write_db(rattle_dir / "rattled_frames.db", rattled_frames)
    _, relax, _, _ = _growth_helpers()
    reoptimized, report = relax(
        rattled_frames,
        model_path=model_path,
        fmax=fmax,
        max_steps=max_steps,
        device=device,
        dispersion=dispersion,
        default_dtype=default_dtype,
        output_db_path=rattle_dir / "reoptimized.db",
        verbose=verbose,
        fixed_symbols=fixed_symbols,
        bonded_symbols=bonded_symbols,
        enable_cueq=enable_cueq,
    )
    report["parent_id"] = candidate_parent_ids
    report.to_csv(rattle_dir / "reoptimization.csv", index=False)
    accepted_ids = report.loc[report["converged"] & report["bonded"], "row_id"].astype(int)
    reoptimized_by_id = dict(zip(accepted_ids, reoptimized))
    rows = []
    selected = []
    for parent_id, (original, initial_energy) in enumerate(
        zip(original_parents, initial_energies), start=1
    ):
        candidates = [
            (row_id, reoptimized_by_id[row_id])
            for row_id, candidate_parent_id in enumerate(candidate_parent_ids, start=1)
            if candidate_parent_id == parent_id and row_id in reoptimized_by_id
        ]
        best = min(candidates, key=lambda item: _energy(item[1])) if candidates else None
        best_energy = _energy(best[1]) if best else None
        use_rattled = best is not None and best_energy < initial_energy
        selected.append(best[1].copy() if use_rattled else original.copy())
        rows.append({
            "parent_id": parent_id,
            "rattles": PARENT_RATTLES,
            "stdev_A": float(stdev),
            "initial_energy_eV": initial_energy,
            "best_rattle_id": ((best[0] - 1) % PARENT_RATTLES + 1) if best else None,
            "best_reoptimized_energy_eV": best_energy,
            "selected_source": "reoptimized_rattle" if use_rattled else "original_parent",
            "selected_energy_eV": best_energy if use_rattled else initial_energy,
        })
    _write_csv(rattle_dir / "summary.csv", rows)
    _write_db(rattle_dir / "selected_parents.db", selected)
    return selected


def _candidate_structures(
    parent: Atoms,
    *,
    parent_id: int,
    add_symbol: str,
    site_symbols: list[str],
    passive_symbols: list[str],
    max_structures: int | None,
    cluster_symbols: list[str],
    triangle_neighbor_cutoff: float,
    surface_top_z_tolerance: float,
    surface_max_cluster_distance: float,
    force_triangulated_fallback: bool,
    origin_atom_index: int | None,
    site_cutoff: float | None,
    z_cutoff: float | None,
) -> list[Atoms]:
    _, _, _single_atom_adsorbate, generate_radial_growth_candidates = _growth_helpers()
    from cats.working_directory.coadsorbate_generator import run_coadsorbate_generation

    try:
        candidates = run_coadsorbate_generation(
            parent,
            adsorbates=[_single_atom_adsorbate(add_symbol)],
            output_db=None,
            site_symbols=site_symbols,
            passive_symbols=passive_symbols,
            max_structures=max_structures,
            cluster_symbols=cluster_symbols,
            enforce_cluster_integrity=True,
            allow_6fold=True,
            skip_abnormal_bond_screening=True,
            triangle_neighbor_cutoff=triangle_neighbor_cutoff,
            surface_top_z_tolerance=surface_top_z_tolerance,
            max_new_atom_cluster_distance=surface_max_cluster_distance,
            force_triangulated_com_fallback=force_triangulated_fallback,
            origin_atom_index=origin_atom_index,
            site_cutoff=site_cutoff,
            z_cutoff=z_cutoff,
        )
        if not candidates:
            raise RuntimeError("coadsorbate generator returned no candidates")
    except Exception as exc:
        print(f"[AlloyGrowth] Falling back to radial growth for {add_symbol}: {exc}")
        candidates = generate_radial_growth_candidates(
            parent,
            new_symbol=add_symbol,
            max_candidates=max_structures or 50,
        )
        if not candidates:
            candidates = [_single_attachment_candidate(parent, add_symbol)]
    out = []
    cluster_symbol_set = set(cluster_symbols)
    for atoms in candidates:
        atoms = atoms.copy()
        _wrap_added_atom(atoms)
        if any(parent.get_pbc()):
            new_index = len(atoms) - 1
            existing_cluster = [
                i for i, atom in enumerate(atoms[:-1])
                if atom.symbol in cluster_symbol_set
            ]
            if existing_cluster and min(
                atoms.get_distance(new_index, i, mic=True) for i in existing_cluster
            ) > float(surface_max_cluster_distance):
                continue
        atoms.info["parent_id"] = parent_id
        atoms.info["added_symbol"] = add_symbol
        out.append(atoms)
    return out


def grow_alloy_nanocluster_by_atom_addition(
    seed_atoms: Atoms,
    *,
    add_symbols: list[str],
    n_additions: int,
    output_root: str | Path = "scratch/alloy_nanocluster_growth",
    keep_per_composition: int = 10,
    min_per_composition: int = 1,
    mace_model_path: str | Path = "similarity/run/mace-omat-0-medium.model",
    mace_device: str = "cpu",
    mace_fmax: float = 0.05,
    mace_max_steps: int = 250,
    mace_default_dtype: str = "float64",
    mace_dispersion: bool = True,
    mace_enable_cueq: bool = True,
    site_symbols: list[str] | None = None,
    attachment_symbols: list[str] | None = None,
    passive_symbols: list[str] | None = None,
    max_structures_per_parent_symbol: int | None = None,
    similarity_threshold: float = 0.9999,
    secondary_similarity_threshold: float | None = None,
    similarity_batch_size: int = 100,
    rattle_parent: bool = False,
    parent_rattle_stdev: float = 0.3,
    parent_rattle_fmax: float = 0.01,
    fix_support: bool = True,
    triangle_neighbor_cutoff: float = 3.2,
    surface_top_z_tolerance: float = 1.5,
    surface_max_cluster_distance: float = 5.0,
    force_triangulated_fallback: bool = False,
    origin_atom_index: int | None = None,
    site_cutoff: float | None = None,
    z_cutoff: float | None = None,
    similarity_slab_depth: float = 6.0,
    reference_energies_eV: dict[str, float] | None = None,
    bulk_reference_energies_eV_per_atom: dict[str, float] | None = None,
    target_mixing_energy: bool = False,
    concentration_window: tuple[str, float, float] | None = None,
    strict_concentration_span: bool = False,
    verbose: bool = True,
) -> dict[str, Path]:
    if int(n_additions) <= 0:
        raise ValueError("n_additions must be positive.")
    add_symbols = [str(symbol).strip() for symbol in add_symbols if str(symbol).strip()]
    if not add_symbols:
        raise ValueError("add_symbols is required.")
    if int(keep_per_composition) <= 0:
        raise ValueError("keep_per_composition must be positive.")
    if mace_default_dtype not in {"float32", "float64"}:
        raise ValueError("mace_default_dtype must be 'float32' or 'float64'.")
    if rattle_parent and float(parent_rattle_stdev) <= 0:
        raise ValueError("parent_rattle_stdev must be positive.")
    if rattle_parent and float(parent_rattle_fmax) <= 0:
        raise ValueError("parent_rattle_fmax must be positive.")
    if float(similarity_slab_depth) <= 0:
        raise ValueError("similarity_slab_depth must be positive.")
    if float(surface_max_cluster_distance) <= 0:
        raise ValueError("surface_max_cluster_distance must be positive.")
    _size_tightened_similarity_threshold(similarity_threshold, 1)
    if secondary_similarity_threshold is not None:
        _size_tightened_similarity_threshold(secondary_similarity_threshold, 1)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    window_settings = {
        "concentration_window": list(concentration_window) if concentration_window else None,
        "strict_concentration_span": bool(strict_concentration_span),
    }
    window_marker = output_root / "concentration_window.json"
    window_text = json.dumps(window_settings, indent=2, sort_keys=True)
    if (
        not window_marker.exists()
        and concentration_window is not None
        and any(output_root.glob("generation_*"))
    ):
        raise ValueError(
            f"Cannot apply a concentration window to an existing untagged output: {output_root}"
        )
    if window_marker.exists() and window_marker.read_text(encoding="utf-8") != window_text:
        raise ValueError(f"Cannot change concentration window in existing output: {output_root}")
    window_marker.write_text(window_text, encoding="utf-8")
    dispersion_marker = output_root / "mace_dispersion.txt"
    dispersion_text = str(bool(mace_dispersion))
    if dispersion_marker.exists() and dispersion_marker.read_text().strip() != dispersion_text:
        raise ValueError(f"Cannot mix dispersion settings in existing output: {output_root}")
    if not dispersion_marker.exists() and mace_dispersion and any(output_root.glob("generation_*")):
        raise ValueError(
            f"Existing untagged generations used dispersion=False; choose a new output root: {output_root}"
        )
    dispersion_marker.write_text(dispersion_text, encoding="utf-8")
    summary_csv = output_root / "alloy_growth_summary.csv"
    stable_summary_csv = output_root / "stable_alloy_summary.csv"
    selected_db_path = output_root / "selected_alloy_growth.db"
    selected_traj_path = output_root / "selected_alloy_growth.traj"
    stable_db_path = output_root / "stable_alloy_structures.db"
    stable_traj_path = output_root / "stable_alloy_structures.traj"
    phase_scatter_png = output_root / "atomization_energy_phase_scatter.png"
    phase_lines_png = output_root / "atomization_energy_phase_lines.png"
    phase_lines_html = output_root / "atomization_energy_phase_lines.html"
    mixing_lines_png = output_root / "mixing_energy_phase_lines.png"

    concentration_symbols = sorted(set(add_symbols))
    if concentration_window is not None:
        window_symbol, window_lower, window_upper = concentration_window
        if window_symbol not in concentration_symbols:
            raise ValueError("The concentration window symbol must be included in add_symbols.")
        if not 0 <= float(window_lower) <= float(window_upper) <= 1:
            raise ValueError("Concentration window bounds must satisfy 0 <= lower <= upper <= 1.")
    all_symbols = sorted(set(seed_atoms.get_chemical_symbols()) | set(add_symbols))
    support_symbols = sorted(set(seed_atoms.get_chemical_symbols()) - set(add_symbols))
    if bulk_reference_energies_eV_per_atom is not None:
        missing_bulk_references = set(concentration_symbols) - set(
            bulk_reference_energies_eV_per_atom
        )
        if missing_bulk_references:
            raise ValueError(
                f"Missing elemental bulk reference energies for: {sorted(missing_bulk_references)}"
            )
    use_bulk_selection = (
        bool(target_mixing_energy)
        and bool(bulk_reference_energies_eV_per_atom)
        and not support_symbols
    )
    selection_criterion = (
        "bulk_mixing_energy_per_atom_eV"
        if use_bulk_selection
        else "atomization_energy_per_atom_eV"
    )
    selection_reason = (
        "optimized elemental bulk references are available"
        if use_bulk_selection
        else (
            "the system contains non-alloy support atoms"
            if support_symbols
            else (
                "optimized elemental bulk references are unavailable"
                if target_mixing_energy
                else "atomization energy is the default target"
            )
        )
    )
    existing_generations = sorted(output_root.glob("generation_*"))
    _write_json(
        output_root / "selection_criterion.json",
        {
            "criterion_for_new_selections": selection_criterion,
            "reason": selection_reason,
            "lower_is_better": bool(use_bulk_selection),
            "preexisting_generations_reused_without_reselection": [
                path.name for path in existing_generations
            ],
        },
    )
    print(
        f"[AlloyGrowth] STRUCTURE SELECTION CRITERION: {selection_criterion} "
        f"({selection_reason})"
    )
    passive_symbols = support_symbols if passive_symbols is None else passive_symbols
    site_symbols = (passive_symbols or add_symbols) if site_symbols is None else site_symbols
    attachment_symbols, bonded_symbols = _growth_attachment_and_bonded_symbols(
        add_symbols=add_symbols,
        site_symbols=site_symbols,
        attachment_symbols=attachment_symbols,
    )
    fixed_symbols = passive_symbols if fix_support else []
    support_z = [atom.z for atom in seed_atoms if atom.symbol in set(passive_symbols)]
    slab_shave_distance = (
        max(support_z) - float(similarity_slab_depth)
        if any(seed_atoms.get_pbc()) and support_z
        else None
    )
    if verbose:
        settings = {
            "seed_formula": seed_atoms.get_chemical_formula(),
            "seed_atoms": len(seed_atoms),
            "seed_pbc": seed_atoms.get_pbc().tolist(),
            "growth_mode": "surface" if any(seed_atoms.get_pbc()) else "gas_cluster",
            "generation_one_uses_gas_references": len(seed_atoms) == 0,
            "add_symbols": add_symbols,
            "n_additions": n_additions,
            "output_root": str(output_root),
            "keep_per_composition": keep_per_composition,
            "min_per_composition": min_per_composition,
            "max_structures_per_parent_symbol": max_structures_per_parent_symbol,
            "site_symbols": site_symbols,
            "attachment_symbols": attachment_symbols,
            "passive_symbols": passive_symbols,
            "support_symbols": support_symbols,
            "fixed_symbols_during_optimization": fixed_symbols,
            "fix_support": fix_support,
            "mace_model_path": str(mace_model_path),
            "mace_device": mace_device,
            "mace_default_dtype": mace_default_dtype,
            "mace_dispersion": mace_dispersion,
            "mace_enable_cueq": mace_enable_cueq,
            "mace_fmax_eV_A": mace_fmax,
            "mace_max_steps": mace_max_steps,
            "similarity_threshold": similarity_threshold,
            "secondary_similarity_threshold": secondary_similarity_threshold,
            "similarity_threshold_schedule": "1 - (1 - baseline) / alloy_atom_count",
            "similarity_batch_size": similarity_batch_size,
            "secondary_similarity_threshold_baseline": (
                similarity_threshold
                if secondary_similarity_threshold is None
                else secondary_similarity_threshold
            ),
            "similarity_slab_depth_A": similarity_slab_depth,
            "similarity_slab_threshold_z_A": slab_shave_distance,
            "triangle_neighbor_cutoff_A": triangle_neighbor_cutoff,
            "surface_top_z_tolerance_A": surface_top_z_tolerance,
            "surface_max_cluster_distance_A": surface_max_cluster_distance,
            "force_triangulated_fallback": force_triangulated_fallback,
            "origin_atom_index": origin_atom_index,
            "site_cutoff_A": site_cutoff,
            "z_cutoff_A": z_cutoff,
            "adsorption_height_A": 2.0,
            "allow_6fold_sites": True,
            "skip_abnormal_bond_screening": True,
            "enforce_cluster_integrity": True,
            "connectivity_bond_scale": 1.35,
            "rattle_parent": rattle_parent,
            "parent_rattles": PARENT_RATTLES,
            "parent_rattle_stdev_A": parent_rattle_stdev,
            "parent_rattle_fmax_eV_A": parent_rattle_fmax,
            "verbose": verbose,
        }
        print("[AlloyGrowth] Settings:")
        for key, value in settings.items():
            print(f"[AlloyGrowth]   {key} = {value}")
        print(f"[AlloyGrowth] Calculating single-atom MACE references for: {', '.join(all_symbols)}")
    if reference_energies_eV is None:
        reference_energies_eV = mace_single_atom_reference_energies(
            all_symbols,
            model_path=mace_model_path,
            device=mace_device,
            dispersion=mace_dispersion,
            default_dtype=mace_default_dtype,
            enable_cueq=mace_enable_cueq,
        )
    else:
        missing_references = set(all_symbols) - set(reference_energies_eV)
        if missing_references:
            raise ValueError(
                f"Missing single-atom reference energies for: {sorted(missing_references)}"
            )
    if verbose:
        for symbol in all_symbols:
            print(f"[AlloyGrowth] E_ref({symbol}) = {reference_energies_eV[symbol]:.10f} eV")
    _deduplicate, _relax_full_cluster_with_mace, _, _ = _growth_helpers()
    parents = [seed_atoms.copy()]
    summary_rows = []
    selected_history = [] if len(seed_atoms) == 0 else [seed_atoms.copy()]
    stable_rows = []
    stable_structures = []

    for generation in range(1, int(n_additions) + 1):
        generation_dir = output_root / f"generation_{generation:03d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        alloy_atom_count = _alloy_atom_count(parents[0], concentration_symbols) + 1
        primary_threshold = _size_tightened_similarity_threshold(
            similarity_threshold,
            alloy_atom_count,
        )
        secondary_threshold = _size_tightened_similarity_threshold(
            similarity_threshold
            if secondary_similarity_threshold is None
            else secondary_similarity_threshold,
            alloy_atom_count,
        )
        threshold_marker = generation_dir / "similarity_thresholds.json"
        threshold_settings = {
            "alloy_atom_count": alloy_atom_count,
            "formula": "1 - (1 - baseline) / alloy_atom_count",
            "primary_baseline": float(similarity_threshold),
            "primary_effective": primary_threshold,
            "secondary_baseline": float(
                similarity_threshold
                if secondary_similarity_threshold is None
                else secondary_similarity_threshold
            ),
            "secondary_effective": secondary_threshold,
        }
        threshold_text = json.dumps(threshold_settings, indent=2, sort_keys=True)
        reuse_similarity_cache = (
            threshold_marker.exists()
            and threshold_marker.read_text(encoding="utf-8") == threshold_text
        )
        completed_paths = [
            generation_dir / "selected.xyz",
            generation_dir / "final_unique.db",
            generation_dir / "ranked.csv",
        ]
        if reuse_similarity_cache and all(path.exists() for path in completed_paths):
            final_unique = _read_db(generation_dir / "final_unique.db")
            ranked = _rank_by_atomization_energy(
                final_unique,
                concentration_symbols=concentration_symbols,
                reference_energies_eV=reference_energies_eV,
                bulk_reference_energies_eV_per_atom=(
                    bulk_reference_energies_eV_per_atom if use_bulk_selection else None
                ),
                target_mixing_energy=use_bulk_selection,
            )
            generation_stable_structures, generation_stable_rows = _best_per_size_composition(
                ranked,
                generation=generation,
                start_id=len(stable_structures) + 1,
            )
            stable_structures.extend(generation_stable_structures)
            stable_rows.extend(generation_stable_rows)
            _, ranked_rows = _select_per_composition(
                ranked,
                keep_per_composition=keep_per_composition,
            )
            summary_rows.extend(
                {"generation": generation, **row}
                for row in ranked_rows
                if row["selected"]
            )
            parents = read(generation_dir / "selected.xyz", index=":", parallel=False)
            selected_history.extend(atoms.copy() for atoms in parents)
            if verbose:
                print(f"[AlloyGrowth] Resuming after completed generation {generation}")
            continue
        if verbose:
            print(f"[AlloyGrowth] Generation {generation}: {len(parents)} parents")
            print(
                f"[AlloyGrowth] Similarity thresholds at {alloy_atom_count} alloy atoms: "
                f"primary={primary_threshold:.10f}, secondary={secondary_threshold:.10f}"
            )
        threshold_marker.write_text(threshold_text, encoding="utf-8")
        gas_seed_generation = generation == 1 and len(seed_atoms) == 0

        candidates_db = generation_dir / "candidates.db"
        if candidates_db.exists():
            candidates = _read_db(candidates_db)
            if verbose:
                print(f"[AlloyGrowth] Reusing {candidates_db}")
        else:
            if gas_seed_generation:
                candidates = _gas_atom_candidates(add_symbols, reference_energies_eV)
            else:
                candidates = []
                for parent_id, parent in enumerate(parents, start=1):
                    for symbol in add_symbols:
                        candidates.extend(
                            _candidate_structures(
                                parent,
                                parent_id=parent_id,
                                add_symbol=symbol,
                                site_symbols=site_symbols,
                                passive_symbols=passive_symbols,
                                max_structures=max_structures_per_parent_symbol,
                                cluster_symbols=attachment_symbols,
                                triangle_neighbor_cutoff=triangle_neighbor_cutoff,
                                surface_top_z_tolerance=surface_top_z_tolerance,
                                surface_max_cluster_distance=surface_max_cluster_distance,
                                force_triangulated_fallback=force_triangulated_fallback,
                                origin_atom_index=origin_atom_index,
                                site_cutoff=site_cutoff,
                                z_cutoff=z_cutoff,
                            )
                        )
        if concentration_window is not None:
            candidates = _filter_concentration_window(
                candidates,
                concentration_symbol=window_symbol,
                concentration_symbols=concentration_symbols,
                lower=float(window_lower),
                upper=float(window_upper),
                strict=bool(strict_concentration_span),
            )
        if not candidates:
            raise RuntimeError(f"Generation {generation} produced no candidates.")
        if not candidates_db.exists():
            _write_db(candidates_db, candidates)

        pre_relax_db = generation_dir / "pre_relax.db"
        if pre_relax_db.exists() and reuse_similarity_cache:
            pre_relax = _read_db(pre_relax_db)
            if verbose:
                print(f"[AlloyGrowth] Reusing {pre_relax_db}")
        elif gas_seed_generation:
            pre_relax = candidates
            _write_db(pre_relax_db, pre_relax)
            _write_csv(
                generation_dir / "primary_similarity.csv",
                [
                    {
                        "row_id": row_id,
                        "kept": True,
                        "max_similarity_to_kept": "",
                        "closest_kept_row_id": "",
                    }
                    for row_id, _ in enumerate(pre_relax, start=1)
                ],
            )
        else:
            pre_relax = _deduplicate_by_composition(
                candidates,
                concentration_symbols=concentration_symbols,
                similarity_threshold=primary_threshold,
                report_csv_path=generation_dir / "primary_similarity.csv",
                support_species=passive_symbols,
                slab_shave_distance=slab_shave_distance,
                batch_size=similarity_batch_size,
                verbose=verbose,
            )
            _write_db(pre_relax_db, pre_relax)

        relaxed_db = generation_dir / "relaxed.db"
        if gas_seed_generation:
            relaxed = _gas_atom_candidates(add_symbols, reference_energies_eV)
            if not relaxed_db.exists():
                _write_db(relaxed_db, relaxed)
                _write_csv(
                    generation_dir / "relaxation.csv",
                    [
                        {
                            "row_id": row_id,
                            "n_atoms": 1,
                            "energy_eV": _energy(atoms),
                            "max_force_eV_A": 0.0,
                            "converged": True,
                            "bonded": True,
                            "nsteps": 0,
                            "reused_gas_reference": True,
                        }
                        for row_id, atoms in enumerate(relaxed, start=1)
                    ],
                )
        else:
            relaxed, relaxation_df = _relax_full_cluster_with_mace(
                pre_relax,
                model_path=mace_model_path,
                fmax=mace_fmax,
                max_steps=mace_max_steps,
                device=mace_device,
                dispersion=mace_dispersion,
                default_dtype=mace_default_dtype,
                output_db_path=relaxed_db,
                verbose=verbose,
                fixed_symbols=fixed_symbols,
                bonded_symbols=bonded_symbols,
                enable_cueq=mace_enable_cueq,
            )
            relaxation_df.to_csv(generation_dir / "relaxation.csv", index=False)
        relaxed = [
            atoms for atoms in relaxed
            if _above_z_cutoff(atoms, set(add_symbols), z_cutoff)
        ]
        if not relaxed:
            raise RuntimeError(f"Generation {generation} produced no converged relaxed candidates.")

        relaxed.sort(key=_energy)
        final_unique_db = generation_dir / "final_unique.db"
        final_unique = _deduplicate_by_composition(
            relaxed,
            concentration_symbols=concentration_symbols,
            similarity_threshold=secondary_threshold,
            report_csv_path=generation_dir / "secondary_similarity.csv",
            support_species=passive_symbols,
            slab_shave_distance=slab_shave_distance,
            batch_size=similarity_batch_size,
            verbose=verbose,
        )
        ranked = _rank_by_atomization_energy(
            final_unique,
            concentration_symbols=concentration_symbols,
            reference_energies_eV=reference_energies_eV,
            bulk_reference_energies_eV_per_atom=(
                bulk_reference_energies_eV_per_atom if use_bulk_selection else None
            ),
            target_mixing_energy=use_bulk_selection,
        )
        generation_stable_structures, generation_stable_rows = _best_per_size_composition(
            ranked,
            generation=generation,
            start_id=len(stable_structures) + 1,
        )
        stable_structures.extend(generation_stable_structures)
        stable_rows.extend(generation_stable_rows)
        parents, ranked_rows = _select_per_composition(
            ranked,
            keep_per_composition=keep_per_composition,
        )
        _write_csv(generation_dir / "ranked.csv", ranked_rows)
        _write_db(final_unique_db, final_unique)
        rattle_requested = rattle_parent and generation < int(n_additions)
        if rattle_requested:
            write(generation_dir / "selected_pre_rattle.xyz", parents, parallel=False)
            parents = _condition_parents_with_rattle(
                parents,
                generation_dir=generation_dir,
                stdev=parent_rattle_stdev,
                model_path=mace_model_path,
                device=mace_device,
                dispersion=mace_dispersion,
                default_dtype=mace_default_dtype,
                fmax=parent_rattle_fmax,
                max_steps=mace_max_steps,
                verbose=verbose,
                fixed_symbols=fixed_symbols,
                bonded_symbols=bonded_symbols,
                enable_cueq=mace_enable_cueq,
            )
        write(generation_dir / "selected.xyz", parents, parallel=False)

        counts_by_composition = defaultdict(int)
        for row in ranked_rows:
            if row["selected"]:
                counts_by_composition[str(row["composition"])] += 1
        short = {
            comp: count
            for comp, count in counts_by_composition.items()
            if count < int(min_per_composition)
        }
        if short and verbose:
            print(f"[AlloyGrowth] Below min_per_composition: {dict(short)}")

        for row in ranked_rows:
            if row["selected"]:
                summary_rows.append({"generation": generation, **row})
        selected_history.extend(atoms.copy() for atoms in parents)
        if verbose:
            print(f"[AlloyGrowth] Kept {len(parents)} structures across {len(counts_by_composition)} compositions")

    stable_rows = _nanoparticle_referenced_rows(stable_rows, concentration_symbols)
    _write_csv(summary_csv, summary_rows)
    _write_csv(stable_summary_csv, stable_rows)
    _write_db(selected_db_path, selected_history)
    _write_db(stable_db_path, stable_structures)
    write(selected_traj_path, selected_history, parallel=False)
    write(stable_traj_path, stable_structures, parallel=False)
    _write_phase_scatter(phase_scatter_png, stable_rows, concentration_symbols)
    _write_phase_lines(phase_lines_png, phase_lines_html, stable_rows, concentration_symbols)
    _write_phase_lines(
        mixing_lines_png,
        None,
        [row for row in stable_rows if row["mixing_energy_eV"] != ""],
        concentration_symbols,
        y_key="mixing_energy_per_atom_eV",
        y_total_key="mixing_energy_eV",
        y_label="Mixing energy / atom (eV)",
        title="Mixing energy relative to monometallic nanoparticles",
    )

    return {
        "output_root": output_root,
        "summary_csv": summary_csv,
        "stable_summary_csv": stable_summary_csv,
        "selected_db_path": selected_db_path,
        "selected_traj_path": selected_traj_path,
        "stable_db_path": stable_db_path,
        "stable_traj_path": stable_traj_path,
        "phase_scatter_png": phase_scatter_png,
        "phase_lines_png": phase_lines_png,
        "phase_lines_html": phase_lines_html,
        "mixing_lines_png": mixing_lines_png,
    }


def grow_alloy_nanocluster_staged(
    seed_atoms: Atoms,
    *,
    add_symbols: list[str],
    max_atoms: int,
    staging_start_atoms: int,
    concentration_symbol: str,
    concentration_span: float = 0.05,
    restart_beam_width: int = 1,
    staging_require_alloy: bool = True,
    strict_concentration_span: bool = False,
    bulk_fmax: float = 0.01,
    bulk_max_steps: int = 500,
    target_mixing_energy: bool = False,
    output_root: str | Path = "scratch/alloy_nanocluster_staged",
    **growth_kwargs,
) -> dict[str, Path]:
    """Run unrestricted alloy growth followed by concentration-limited restart stages."""
    add_symbols = sorted(set(str(symbol).strip() for symbol in add_symbols if str(symbol).strip()))
    if len(add_symbols) != 2:
        raise ValueError("Staged concentration search currently requires exactly two add_symbols.")
    if concentration_symbol not in add_symbols:
        raise ValueError("concentration_symbol must be included in add_symbols.")
    if int(restart_beam_width) <= 0:
        raise ValueError("restart_beam_width must be positive.")
    if not 0 < float(concentration_span) < 0.5:
        raise ValueError("concentration_span must be between 0 and 0.5.")
    if float(bulk_fmax) <= 0 or int(bulk_max_steps) <= 0:
        raise ValueError("bulk_fmax and bulk_max_steps must be positive.")
    reserved = {
        "n_additions",
        "output_root",
        "reference_energies_eV",
        "bulk_reference_energies_eV_per_atom",
        "concentration_window",
        "strict_concentration_span",
    }
    unexpected = reserved.intersection(growth_kwargs)
    if unexpected:
        raise ValueError(f"Staged growth controls these arguments: {sorted(unexpected)}")

    initial_size = _alloy_atom_count(seed_atoms, add_symbols)
    staging_size = max(int(staging_start_atoms), initial_size)
    if staging_size <= 0:
        raise ValueError("staging_start_atoms must be positive.")
    if int(max_atoms) < staging_size:
        raise ValueError("max_atoms must be at least staging_start_atoms or the seed size.")

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = growth_kwargs.get("mace_model_path", "similarity/run/mace-omat-0-medium.model")
    device = str(growth_kwargs.get("mace_device", "cpu"))
    dispersion = bool(growth_kwargs.get("mace_dispersion", True))
    default_dtype = str(growth_kwargs.get("mace_default_dtype", "float64"))
    enable_cueq = bool(growth_kwargs.get("mace_enable_cueq", True))
    all_symbols = sorted(set(seed_atoms.get_chemical_symbols()) | set(add_symbols))
    config_path = output_root / "campaign_config.json"
    existing_config = (
        json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    )
    stored_references = dict(existing_config.get("reference_energies_eV") or {})
    if all(symbol in stored_references for symbol in all_symbols):
        reference_energies_eV = {
            symbol: float(stored_references[symbol]) for symbol in all_symbols
        }
        print("[AlloyGrowth] Reusing stored single-atom reference energies")
    else:
        reference_energies_eV = mace_single_atom_reference_energies(
            all_symbols,
            model_path=model_path,
            device=device,
            dispersion=dispersion,
            default_dtype=default_dtype,
            enable_cueq=enable_cueq,
        )
    campaign_config = {
        "schema_version": 2,
        "initial_seed_formula": seed_atoms.get_chemical_formula(),
        "initial_alloy_atoms": initial_size,
        "add_symbols": add_symbols,
        "max_atoms": int(max_atoms),
        "staging_start_atoms": staging_size,
        "concentration_symbol": concentration_symbol,
        "concentration_span": float(concentration_span),
        "restart_beam_width": int(restart_beam_width),
        "staging_require_alloy": bool(staging_require_alloy),
        "strict_concentration_span": bool(strict_concentration_span),
        "target_mixing_energy": bool(target_mixing_energy),
        "mace_model_path": str(model_path),
        "mace_device": device,
        "mace_dispersion": dispersion,
        "mace_default_dtype": default_dtype,
        "mace_enable_cueq": enable_cueq,
        "reference_energies_eV": reference_energies_eV,
        "similarity_threshold_schedule": "1 - (1 - baseline) / alloy_atom_count",
        "growth_kwargs": json.loads(json.dumps(growth_kwargs, default=str)),
    }
    config_text = json.dumps(campaign_config, indent=2, sort_keys=True)
    if config_path.exists() and config_path.read_text(encoding="utf-8") != config_text:
        raise ValueError(f"Cannot change staged campaign settings in: {output_root}")
    config_path.write_text(config_text, encoding="utf-8")
    bulk_reference_root = output_root / "bulk_references"
    bulk_reference_energies_eV_per_atom = mace_bulk_reference_energies(
        add_symbols,
        output_root=bulk_reference_root,
        model_path=model_path,
        device=device,
        dispersion=dispersion,
        default_dtype=default_dtype,
        enable_cueq=enable_cueq,
        fmax=bulk_fmax,
        max_steps=bulk_max_steps,
    )
    staged_has_support = any(
        atom.symbol not in set(add_symbols) for atom in seed_atoms
    )
    use_bulk_staging = (
        bool(target_mixing_energy)
        and bool(bulk_reference_energies_eV_per_atom)
        and not staged_has_support
    )
    staging_selection_criterion = (
        "bulk_mixing_energy_per_atom_eV"
        if use_bulk_staging
        else "atomization_energy_per_atom_eV"
    )
    staging_selection_reason = (
        "optimized elemental bulk references are available"
        if use_bulk_staging
        else (
            "the system contains non-alloy support atoms"
            if staged_has_support
            else (
                "the MACE model did not provide usable optimized bulk references"
                if target_mixing_energy
                else "atomization energy is the default target"
            )
        )
    )
    _write_json(
        output_root / "selection_criterion.json",
        {
            "criterion_for_new_staging_decisions": staging_selection_criterion,
            "reason": staging_selection_reason,
            "lower_is_better": bool(use_bulk_staging),
            "published_checkpoints_are_reused_without_reselection": True,
        },
    )
    print(
        f"[AlloyGrowth] STAGING SELECTION CRITERION: {staging_selection_criterion} "
        f"({staging_selection_reason})"
    )

    campaign_path = output_root / "campaign.json"
    combined_db = output_root / "combined_stable_structures.db"
    combined_traj = output_root / "combined_stable_structures.traj"
    combined_csv = output_root / "combined_stable_summary.csv"
    campaign = (
        json.loads(campaign_path.read_text(encoding="utf-8"))
        if campaign_path.exists()
        else {}
    )
    if (
        int(campaign.get("completed_atoms", 0)) >= int(max_atoms)
        and combined_db.exists()
        and combined_csv.exists()
    ):
        print(
            "[AlloyGrowth] Campaign is already complete; existing checkpoint choices are "
            "retained and no nanoclusters are reselected."
        )
        stable_structures = _read_db(combined_db)
        with combined_csv.open(newline="", encoding="utf-8") as handle:
            stable_rows = list(csv.DictReader(handle))
        stable_rows = _bulk_referenced_mixing_rows(
            stable_rows,
            add_symbols,
            bulk_reference_energies_eV_per_atom,
        )
        _write_csv(combined_csv, stable_rows)
        analysis = _write_all_staged_campaign_analyses(
            output_root,
            stable_structures,
            stable_rows,
            add_symbols,
            reference_energies_eV,
            bulk_reference_energies_eV_per_atom,
        )
        return {
            "output_root": output_root,
            "campaign_path": campaign_path,
            "current_restart_seeds_db": output_root / "current_restart_seeds.db",
            "combined_stable_db": combined_db,
            "combined_stable_traj": combined_traj,
            "combined_stable_summary": combined_csv,
            "bulk_reference_status": bulk_reference_root / "bulk_reference_status.json",
            "selection_criterion": output_root / "selection_criterion.json",
            **analysis,
        }

    similarity_threshold = float(growth_kwargs.get("secondary_similarity_threshold") or growth_kwargs.get("similarity_threshold", 0.9999))
    similarity_batch_size = int(growth_kwargs.get("similarity_batch_size", 100))
    passive_symbols = list(growth_kwargs.get("passive_symbols") or [])
    similarity_slab_depth = float(growth_kwargs.get("similarity_slab_depth", 6.0))
    support_z = [atom.z for atom in seed_atoms if atom.symbol in set(passive_symbols)]
    slab_shave_distance = (
        max(support_z) - similarity_slab_depth
        if any(seed_atoms.get_pbc()) and support_z
        else None
    )
    combined_stable_frames: list[Atoms] = []
    stage_records: list[dict[str, object]] = []

    def run_growth(seed: Atoms, additions: int, root: Path, window=None):
        result = grow_alloy_nanocluster_by_atom_addition(
            seed,
            add_symbols=add_symbols,
            n_additions=additions,
            output_root=root,
            reference_energies_eV=reference_energies_eV,
            bulk_reference_energies_eV_per_atom=(
                bulk_reference_energies_eV_per_atom if use_bulk_staging else None
            ),
            target_mixing_energy=target_mixing_energy,
            concentration_window=window,
            strict_concentration_span=strict_concentration_span,
            **growth_kwargs,
        )
        combined_stable_frames.extend(_read_db(result["stable_db_path"]))
        return result

    def write_checkpoint(
        size: int,
        candidates: list[Atoms],
        *,
        checkpoint_dir: Path,
    ) -> list[Atoms]:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        restart_seeds_db = checkpoint_dir / "restart_seeds.db"
        restart_ranking_csv = checkpoint_dir / "restart_ranking.csv"
        if restart_seeds_db.exists() and restart_ranking_csv.exists():
            seeds = _read_db(restart_seeds_db)
            with restart_ranking_csv.open(newline="", encoding="utf-8") as handle:
                stored_ranking = list(csv.DictReader(handle))
            stored_criterion = (
                str(stored_ranking[0].get("selection_criterion") or "atomization_energy_per_atom_eV")
                if stored_ranking
                else "atomization_energy_per_atom_eV"
            )
            print(
                f"[AlloyGrowth] Reusing published checkpoint N={size} selected with "
                f"{stored_criterion}; no retroactive reselection"
            )
            _write_db(output_root / "current_restart_seeds.db", seeds)
            write(output_root / "current_restart_seeds.traj", seeds, parallel=False)
            return seeds
        checkpoint_threshold = _size_tightened_similarity_threshold(
            similarity_threshold,
            _alloy_atom_count(candidates[0], add_symbols),
        )
        candidates = _deduplicate_by_composition(
            candidates,
            concentration_symbols=add_symbols,
            similarity_threshold=checkpoint_threshold,
            report_csv_path=checkpoint_dir / "similarity.csv",
            support_species=passive_symbols,
            slab_shave_distance=slab_shave_distance,
            batch_size=similarity_batch_size,
            verbose=bool(growth_kwargs.get("verbose", True)),
        )
        eligible_candidates = candidates
        if size < int(max_atoms):
            eligible_candidates = [
                atoms
                for atoms in candidates
                if float(concentration_span)
                < _concentration(atoms, concentration_symbol, add_symbols)
                < 1.0 - float(concentration_span)
            ]
        seeds, ranking = _select_restart_beam(
            eligible_candidates,
            beam_width=restart_beam_width,
            concentration_symbols=add_symbols,
            reference_energies_eV=reference_energies_eV,
            bulk_reference_energies_eV_per_atom=(
                bulk_reference_energies_eV_per_atom if use_bulk_staging else None
            ),
            target_mixing_energy=target_mixing_energy,
            require_alloy=staging_require_alloy,
        )
        if not seeds:
            raise RuntimeError(f"Checkpoint at {size} atoms produced no eligible restart seeds.")
        _write_db(checkpoint_dir / "terminal_candidates.db", candidates)
        _write_csv(restart_ranking_csv, ranking)
        _write_db(restart_seeds_db, seeds)
        write(checkpoint_dir / "restart_seeds.traj", seeds, parallel=False)
        _write_db(output_root / "current_restart_seeds.db", seeds)
        write(output_root / "current_restart_seeds.traj", seeds, parallel=False)
        return seeds

    if initial_size < staging_size:
        prestage_additions = staging_size - initial_size
        if initial_size == 0:
            prestage_additions = staging_size
        prestage_result = run_growth(
            seed_atoms,
            prestage_additions,
            output_root / f"prestage_N{initial_size:03d}_N{staging_size:03d}",
        )
        initial_candidates = _read_db(
            prestage_result["output_root"]
            / f"generation_{prestage_additions:03d}"
            / "final_unique.db"
        )
    else:
        _energy(seed_atoms)
        initial_candidates = [_copy_with_energy(seed_atoms)]
    combined_stable_frames.extend(_copy_with_energy(atoms) for atoms in initial_candidates)

    current_size = staging_size
    seeds = write_checkpoint(
        current_size,
        initial_candidates,
        checkpoint_dir=output_root / f"checkpoint_N{current_size:03d}",
    )
    _write_json(
        output_root / "campaign.json",
        {"config": campaign_config, "completed_atoms": current_size, "stages": stage_records},
    )
    stage_index = 0
    while current_size < int(max_atoms):
        requested_additions = [
            _stage_generation_count(
                seed,
                concentration_symbol=concentration_symbol,
                concentration_symbols=add_symbols,
                concentration_span=concentration_span,
            )
            for seed in seeds
        ]
        additions = min(max(requested_additions), int(max_atoms) - current_size)
        next_size = current_size + additions
        stage_dir = output_root / f"stage_{stage_index:03d}_N{current_size:03d}_N{next_size:03d}"
        terminal_candidates = []
        branch_records = []
        for branch_index, seed in enumerate(seeds, start=1):
            center = _concentration(seed, concentration_symbol, add_symbols)
            lower = center - float(concentration_span)
            upper = center + float(concentration_span)
            branch_dir = stage_dir / f"branch_{branch_index:03d}"
            branch_dir.mkdir(parents=True, exist_ok=True)
            branch_seed_path = branch_dir / "branch_seed.json"
            branch_seed = _atoms_signature(seed)
            branch_seed_text = json.dumps(branch_seed, indent=2, sort_keys=True)
            if (
                branch_seed_path.exists()
                and branch_seed_path.read_text(encoding="utf-8") != branch_seed_text
            ):
                raise ValueError(f"Restart seed changed for existing branch: {branch_dir}")
            branch_seed_path.write_text(branch_seed_text, encoding="utf-8")
            branch_result = run_growth(
                seed,
                additions,
                branch_dir,
                (concentration_symbol, lower, upper),
            )
            branch_terminal = _read_db(
                branch_result["output_root"]
                / f"generation_{additions:03d}"
                / "final_unique.db"
            )
            for atoms in branch_terminal:
                atoms.info.update(
                    staged_source_branch=branch_index,
                    staged_source_size=current_size,
                    staged_target_size=next_size,
                )
            terminal_candidates.extend(branch_terminal)
            minimum_count, maximum_count = _concentration_count_bounds(
                next_size,
                lower=lower,
                upper=upper,
                strict=strict_concentration_span,
            )
            branch_records.append(
                {
                    "branch": branch_index,
                    "seed_formula": seed.get_chemical_formula(),
                    "seed_concentration": center,
                    "requested_generations": requested_additions[branch_index - 1],
                    "window_lower": lower,
                    "window_upper": upper,
                    "final_minimum_count": minimum_count,
                    "final_maximum_count": maximum_count,
                    "effective_final_lower": minimum_count / next_size,
                    "effective_final_upper": maximum_count / next_size,
                    "terminal_candidates": len(branch_terminal),
                    "output_root": str(branch_dir),
                }
            )

        current_size = next_size
        seeds = write_checkpoint(
            current_size,
            terminal_candidates,
            checkpoint_dir=output_root / f"checkpoint_N{current_size:03d}",
        )
        stage_records.append(
            {
                "stage": stage_index,
                "start_atoms": current_size - additions,
                "end_atoms": current_size,
                "additions": additions,
                "truncated_by_max_atoms": additions < max(requested_additions),
                "branches": branch_records,
                "selected_seed_formulas": [seed.get_chemical_formula() for seed in seeds],
            }
        )
        _write_json(
            output_root / "campaign.json",
            {
                "config": campaign_config,
                "completed_atoms": current_size,
                "stages": stage_records,
            },
        )
        stage_index += 1

    combined_ranked = _rank_by_atomization_energy(
        combined_stable_frames,
        concentration_symbols=add_symbols,
        reference_energies_eV=reference_energies_eV,
        bulk_reference_energies_eV_per_atom=(
            bulk_reference_energies_eV_per_atom if use_bulk_staging else None
        ),
        target_mixing_energy=use_bulk_staging,
    )
    combined_structures, combined_rows = _best_per_size_composition(
        combined_ranked,
        generation=0,
    )
    combined_rows = _bulk_referenced_mixing_rows(
        _nanoparticle_referenced_rows(combined_rows, add_symbols),
        add_symbols,
        bulk_reference_energies_eV_per_atom,
    )
    _write_db(combined_db, combined_structures)
    write(combined_traj, combined_structures, parallel=False)
    _write_csv(combined_csv, combined_rows)
    if not (output_root / "campaign.json").exists():
        _write_json(
            output_root / "campaign.json",
            {"config": campaign_config, "completed_atoms": current_size, "stages": stage_records},
        )
    analysis = _write_all_staged_campaign_analyses(
        output_root,
        combined_structures,
        combined_rows,
        add_symbols,
        reference_energies_eV,
        bulk_reference_energies_eV_per_atom,
    )
    return {
        "output_root": output_root,
        "campaign_path": output_root / "campaign.json",
        "current_restart_seeds_db": output_root / "current_restart_seeds.db",
        "combined_stable_db": combined_db,
        "combined_stable_traj": combined_traj,
        "combined_stable_summary": combined_csv,
        "bulk_reference_status": bulk_reference_root / "bulk_reference_status.json",
        "selection_criterion": output_root / "selection_criterion.json",
        **analysis,
    }


def _self_check() -> None:
    atoms = Atoms("AuCu", positions=[[0, 0, 0], [2.5, 0, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-5.0)
    rattled = atoms.copy()
    rattled.set_constraint(FixAtoms(indices=[0]))
    rattled.rattle(stdev=0.1, seed=1)
    assert (rattled.positions[0] == atoms.positions[0]).all()
    assert (rattled.positions[1] != atoms.positions[1]).any()
    refs = {"Au": -2.0, "Cu": -1.0}
    assert abs(atomization_energy_eV(atoms, refs) - 2.0) < 1e-12
    assert abs(atomization_energy_per_atom_eV(atoms, refs) - 1.0) < 1e-12
    more_stable = atoms.copy()
    more_stable.calc = SinglePointCalculator(more_stable, energy=-6.0)
    rows = _rank_by_atomization_energy(
        [atoms, more_stable],
        concentration_symbols=["Au", "Cu"],
        reference_energies_eV=refs,
    )
    assert rows[0]["energy_eV"] == -6.0
    best_frame = _lower_energy_frame(_lower_energy_frame(None, atoms), more_stable)
    assert best_frame[0] == -6.0
    selected, out_rows = _select_per_composition(rows, keep_per_composition=1)
    assert len(selected) == 1
    assert out_rows[0]["composition"] == "Au1Cu1"
    assert out_rows[0]["x_Cu"] == 0.5
    assert out_rows[0]["selected"] is True
    stable_atoms, stable_rows = _best_per_size_composition(rows, generation=1)
    assert len(stable_atoms) == 1
    assert stable_rows[0]["stable_structure_id"] == 1
    nanoparticle_rows = _nanoparticle_referenced_rows(
        [
            {"n_atoms": 2, "energy_eV": -5.0, "n_Au": 2, "n_Cu": 0},
            {"n_atoms": 2, "energy_eV": -6.0, "n_Au": 2, "n_Cu": 0},
            {"n_atoms": 2, "energy_eV": -4.0, "n_Au": 0, "n_Cu": 2},
            {"n_atoms": 2, "energy_eV": -5.5, "n_Au": 1, "n_Cu": 1},
        ],
        ["Au", "Cu"],
    )
    assert nanoparticle_rows[1]["mixing_energy_per_atom_eV"] == 0.0
    assert nanoparticle_rows[2]["mixing_energy_per_atom_eV"] == 0.0
    assert nanoparticle_rows[3]["mixing_energy_eV"] == -0.5
    assert nanoparticle_rows[3]["mixing_energy_per_atom_eV"] == -0.25
    bulk_rows = _bulk_referenced_mixing_rows(
        [
            {
                "n_atoms": 2,
                "n_system_atoms": 2,
                "energy_eV": -5.5,
                "n_Au": 1,
                "n_Cu": 1,
            }
        ],
        ["Au", "Cu"],
        {"Au": -3.0, "Cu": -2.0},
    )
    assert bulk_rows[0]["bulk_mixing_energy_eV"] == -0.5
    assert bulk_rows[0]["bulk_mixing_energy_per_atom_eV"] == -0.25
    first_growth = _single_attachment_candidate(Atoms("Au", positions=[[0, 0, 0]]), "Cu")
    assert len(first_growth) == 2
    periodic_growth = Atoms(
        "TiCu",
        positions=[[0.2, 0.0, 1.0], [10.4, -0.3, 13.0]],
        cell=[10.0, 10.0, 12.0],
        pbc=[True, True, False],
    )
    _wrap_added_atom(periodic_growth)
    assert abs(periodic_growth.positions[-1, 0] - 0.4) < 1e-12
    assert abs(periodic_growth.positions[-1, 1] - 9.7) < 1e-12
    assert periodic_growth.positions[-1, 2] == 13.0
    gas_atoms = _gas_atom_candidates(["Au", "Cu", "Au"], {"Au": -2.0, "Cu": -1.0})
    assert [atom.symbol for candidate in gas_atoms for atom in candidate] == ["Au", "Cu"]
    assert [_energy(candidate) for candidate in gas_atoms] == [-2.0, -1.0]
    attachment_symbols, bonded_symbols = _growth_attachment_and_bonded_symbols(
        add_symbols=["O"],
        site_symbols=["Ti", "O", "Cu"],
        attachment_symbols=["Cu"],
    )
    assert attachment_symbols == ["Cu"]
    assert bonded_symbols == ["Cu", "O"]
    gas_rows = _rank_by_atomization_energy(
        gas_atoms,
        concentration_symbols=["Au", "Cu"],
        reference_energies_eV={"Au": -2.0, "Cu": -1.0},
    )
    generation_two_parents, _ = _select_per_composition(gas_rows, keep_per_composition=1)
    assert {candidate.get_chemical_symbols()[0] for candidate in generation_two_parents} == {"Au", "Cu"}
    assert _stage_generation_count(
        Atoms("Au10Cu20"),
        concentration_symbol="Au",
        concentration_symbols=["Au", "Cu"],
        concentration_span=0.05,
    ) == 6


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grow alloy nanoclusters with MACE and auditable energy-based ranking."
    )
    parser.add_argument("seed_structure", nargs="?", help="Initial cluster structure readable by ASE.")
    parser.add_argument("--add-symbols", nargs="+", required=False, help="Species allowed at each growth step.")
    parser.add_argument("--n-additions", type=int, default=1)
    parser.add_argument("--max-atoms", type=int, default=None)
    parser.add_argument("--staging-start-atoms", type=int, default=None)
    parser.add_argument("--concentration-symbol", default=None)
    parser.add_argument("--concentration-span", type=float, default=0.05)
    parser.add_argument("--restart-beam-width", type=int, default=1)
    parser.add_argument("--staging-allow-pure", action="store_true")
    parser.add_argument("--strict-concentration-span", action="store_true")
    parser.add_argument("--output-root", default="scratch/alloy_nanocluster_growth")
    parser.add_argument("--keep-per-composition", type=int, default=10)
    parser.add_argument("--min-per-composition", type=int, default=1)
    parser.add_argument("--mace-model-path", default="similarity/run/mace-omat-0-medium.model")
    parser.add_argument("--mace-device", default="cpu")
    parser.add_argument("--mace-fmax", type=float, default=0.05)
    parser.add_argument("--mace-max-steps", type=int, default=250)
    parser.add_argument("--bulk-fmax", type=float, default=0.01)
    parser.add_argument("--bulk-max-steps", type=int, default=500)
    parser.add_argument(
        "--target-mixing-energy",
        action="store_true",
        help="Opt in to bulk mixing energy as the growth and staging selection target.",
    )
    parser.add_argument("--mace-default-dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument(
        "--mace-dispersion",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mace-enable-cueq",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable MACE cuEquivariance acceleration; use --no-mace-enable-cueq to disable it.",
    )
    parser.add_argument("--max-structures-per-parent-symbol", type=int, default=None)
    parser.add_argument("--similarity-threshold", type=float, default=0.9999)
    parser.add_argument("--secondary-similarity-threshold", type=float, default=None)
    parser.add_argument(
        "--similarity-batch-size",
        type=int,
        default=100,
        help="REMatch structures per batch; use 0 to disable batching.",
    )
    parser.add_argument("--rattle-parent", action="store_true")
    parser.add_argument("--parent-rattle-stdev", type=float, default=0.3)
    parser.add_argument("--parent-rattle-fmax", type=float, default=0.01)
    parser.add_argument("--site-symbols", nargs="+", default=None)
    parser.add_argument(
        "--attachment-symbols",
        nargs="+",
        default=None,
        help="Species to which each added atom must remain attached; defaults to site symbols.",
    )
    parser.add_argument("--passive-symbols", nargs="+", default=None)
    parser.add_argument("--relax-support", action="store_true", help="Allow support atoms to move.")
    parser.add_argument("--triangle-neighbor-cutoff", type=float, default=3.2)
    parser.add_argument("--surface-top-z-tolerance", type=float, default=1.5)
    parser.add_argument("--surface-max-cluster-distance", type=float, default=5.0)
    parser.add_argument("--origin-atom-index", type=int, default=None)
    parser.add_argument("--site-cutoff", type=float, default=None)
    parser.add_argument("--z-cutoff", type=float, default=None)
    parser.add_argument(
        "--force-triangulated-fallback",
        action="store_true",
        help="Bypass ACAT and generate hollow, bridge, and atop sites by triangulation.",
    )
    parser.add_argument("--similarity-slab-depth", type=float, default=6.0)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()

    if args.self_check:
        _self_check()
        print("self-check passed")
        return
    if not args.add_symbols:
        parser.error("--add-symbols is required unless --self-check is used.")

    common_kwargs = dict(
        add_symbols=args.add_symbols,
        n_additions=args.n_additions,
        keep_per_composition=args.keep_per_composition,
        min_per_composition=args.min_per_composition,
        mace_model_path=args.mace_model_path,
        mace_device=args.mace_device,
        mace_fmax=args.mace_fmax,
        mace_max_steps=args.mace_max_steps,
        mace_default_dtype=args.mace_default_dtype,
        mace_dispersion=args.mace_dispersion,
        mace_enable_cueq=args.mace_enable_cueq,
        max_structures_per_parent_symbol=args.max_structures_per_parent_symbol,
        similarity_threshold=args.similarity_threshold,
        secondary_similarity_threshold=args.secondary_similarity_threshold,
        similarity_batch_size=args.similarity_batch_size,
        rattle_parent=args.rattle_parent,
        parent_rattle_stdev=args.parent_rattle_stdev,
        parent_rattle_fmax=args.parent_rattle_fmax,
        site_symbols=args.site_symbols,
        attachment_symbols=args.attachment_symbols,
        passive_symbols=args.passive_symbols,
        fix_support=not args.relax_support,
        triangle_neighbor_cutoff=args.triangle_neighbor_cutoff,
        surface_top_z_tolerance=args.surface_top_z_tolerance,
        surface_max_cluster_distance=args.surface_max_cluster_distance,
        force_triangulated_fallback=args.force_triangulated_fallback,
        origin_atom_index=args.origin_atom_index,
        site_cutoff=args.site_cutoff,
        z_cutoff=args.z_cutoff,
        similarity_slab_depth=args.similarity_slab_depth,
        target_mixing_energy=args.target_mixing_energy,
    )
    seed_atoms = read(args.seed_structure, parallel=False) if args.seed_structure else Atoms()
    if args.staging_start_atoms is not None:
        staged_growth_kwargs = dict(common_kwargs)
        staged_growth_kwargs.pop("n_additions")
        initial_size = _alloy_atom_count(seed_atoms, sorted(set(args.add_symbols)))
        target_atoms = args.max_atoms
        if target_atoms is None:
            target_atoms = args.n_additions if initial_size == 0 else initial_size + args.n_additions
        grow_alloy_nanocluster_staged(
            seed_atoms,
            max_atoms=target_atoms,
            staging_start_atoms=args.staging_start_atoms,
            concentration_symbol=args.concentration_symbol or sorted(set(args.add_symbols))[0],
            concentration_span=args.concentration_span,
            restart_beam_width=args.restart_beam_width,
            staging_require_alloy=not args.staging_allow_pure,
            strict_concentration_span=args.strict_concentration_span,
            bulk_fmax=args.bulk_fmax,
            bulk_max_steps=args.bulk_max_steps,
            output_root=args.output_root,
            **staged_growth_kwargs,
        )
        return

    grow_alloy_nanocluster_by_atom_addition(
        seed_atoms,
        output_root=args.output_root,
        **common_kwargs,
    )


if __name__ == "__main__":
    main()
