#!/usr/bin/env python3
"""Optimise every structure in an ASE-readable file with MACE."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import read
from ase.optimize import BFGS, FIRE, FIRE2, LBFGS, MDMin

from similarity.workflows.mace_pool import (
    get_mace_calculators,
    initialize_mpi_worker_pool,
    run_mace_tasks,
    shutdown_mpi_workers,
    write_ase_database_atomically,
)


OPTIMIZERS = {
    "bfgs": BFGS,
    "fire": FIRE,
    "fire2": FIRE2,
    "lbfgs": LBFGS,
    "mdmin": MDMin,
}
SUMMARY_FIELDS = [
    "input_index",
    "input_id",
    "name",
    "formula",
    "n_atoms",
    "energy_eV",
    "max_force_eV_A",
    "converged",
    "nsteps",
    "force_evaluations",
    "elapsed_seconds",
    "optimizer",
    "model",
]


def load_structures(path: Path):
    if path.suffix.lower() == ".db":
        with connect(str(path), serial=True) as database:
            return [
                (
                    index,
                    row.toatoms(),
                    {"input_id": row.id, "name": row.get("name", f"row-{row.id}")},
                )
                for index, row in enumerate(database.select(), start=1)
            ]

    atoms = read(str(path), index=":", parallel=False)
    atoms = atoms if isinstance(atoms, list) else [atoms]
    return [
        (index, structure, {"input_id": index, "name": f"structure-{index}"})
        for index, structure in enumerate(atoms, start=1)
    ]


def optimise_task(task, calculator, settings):
    index, atoms, source = task
    atoms = atoms.copy()
    atoms.calc = calculator
    trajectory_dir = settings.get("trajectory_dir")
    trajectory = (
        str(Path(trajectory_dir) / f"structure-{index}.traj")
        if trajectory_dir
        else None
    )
    optimizer = OPTIMIZERS[settings["optimizer"]](
        atoms,
        logfile=None,
        trajectory=trajectory,
        maxstep=float(settings["maxstep"]),
    )

    started = time.perf_counter()
    converged = bool(
        optimizer.run(fmax=float(settings["fmax"]), steps=int(settings["steps"]))
    )
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=float)
    max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
    nsteps = int(optimizer.nsteps)
    row = {
        "input_index": int(index),
        "input_id": source["input_id"],
        "name": source["name"],
        "formula": atoms.get_chemical_formula(),
        "n_atoms": len(atoms),
        "energy_eV": energy,
        "max_force_eV_A": max_force,
        "converged": converged,
        "nsteps": nsteps,
        "force_evaluations": nsteps + 1,
        "elapsed_seconds": time.perf_counter() - started,
        "optimizer": settings["optimizer"],
        "model": settings["model"],
    }
    frozen = atoms.copy()
    frozen.calc = SinglePointCalculator(frozen, energy=energy, forces=forces)
    return int(index), frozen, row


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staged = path.with_suffix(path.suffix + ".tmp")
    with staged.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    staged.replace(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Optimise every structure in an ASE-readable file with MACE, using all "
            "visible GPUs or MPI ranks, and write endpoint structures plus a CSV summary."
        )
    )
    parser.add_argument("input", type=Path, help="Input .db, .xyz, .extxyz, .traj, or other ASE-readable file")
    parser.add_argument("--output-db", type=Path, help="Endpoint database (default: INPUT_mace_optimized.db)")
    parser.add_argument("--summary-csv", type=Path, help="Summary CSV (default: INPUT_mace_summary.csv)")
    parser.add_argument("--model", default=os.environ.get("MACE_MODEL_PATH", "medium"), help="MACE-MP model name or model-file path")
    parser.add_argument("--device", default=os.environ.get("MACE_DEVICE", "cuda"), help="MACE device, usually cuda or cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64", help="Calculator floating-point precision")
    parser.add_argument("--dispersion", action="store_true", help="Enable the MACE dispersion correction")
    parser.add_argument("--no-cueq", action="store_true", help="Disable cuequivariance acceleration")
    parser.add_argument("--optimizer", choices=tuple(OPTIMIZERS), default="bfgs")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold in eV/angstrom")
    parser.add_argument("--steps", type=int, default=200, help="Maximum optimizer steps per structure")
    parser.add_argument("--maxstep", type=float, default=0.2, help="Maximum atomic displacement per optimizer step")
    parser.add_argument("--trajectory-dir", type=Path, help="Keep per-structure .traj files in this directory (off by default)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.is_file():
        raise SystemExit(f"Input file does not exist: {args.input}")
    if args.fmax <= 0 or args.steps < 0 or args.maxstep <= 0:
        raise SystemExit("--fmax and --maxstep must be positive; --steps cannot be negative")

    output_db = args.output_db or args.input.with_name(f"{args.input.stem}_mace_optimized.db")
    summary_csv = args.summary_csv or args.input.with_name(f"{args.input.stem}_mace_summary.csv")
    if output_db.resolve() == args.input.resolve():
        raise SystemExit("The output database must differ from the input file")

    tasks = load_structures(args.input)
    if not tasks:
        raise SystemExit(f"No structures found in {args.input}")
    output_db.parent.mkdir(parents=True, exist_ok=True)
    if args.trajectory_dir:
        args.trajectory_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "model": str(args.model),
        "optimizer": args.optimizer,
        "fmax": args.fmax,
        "steps": args.steps,
        "maxstep": args.maxstep,
        "trajectory_dir": str(args.trajectory_dir) if args.trajectory_dir else None,
    }
    calculator_settings = {
        "model": str(args.model),
        "dispersion": args.dispersion,
        "default_dtype": args.dtype,
        "device": args.device,
        "enable_cueq": not args.no_cueq,
    }
    calculators = get_mace_calculators(**calculator_settings)
    results = run_mace_tasks(
        tasks,
        calculators,
        lambda task, calculator: optimise_task(task, calculator, settings),
        label="MACE batch optimisation",
        distributed_operation="similarity.run.mace_batch_optimisation:optimise_task",
        distributed_settings=settings,
        calculator_settings=calculator_settings,
    )
    results.sort(key=lambda result: result[0])
    rows = [row for _, _, row in results]
    write_ase_database_atomically(
        output_db,
        [(atoms, row) for _, atoms, row in results],
    )
    write_csv(summary_csv, rows)
    print(f"Optimised {len(results)} structures")
    print(f"Structures: {output_db}")
    print(f"Summary:    {summary_csv}")
    if args.trajectory_dir:
        print(f"Trajectories: {args.trajectory_dir}")
    return 0


if __name__ == "__main__":
    if initialize_mpi_worker_pool():
        try:
            raise SystemExit(main())
        finally:
            shutdown_mpi_workers()
