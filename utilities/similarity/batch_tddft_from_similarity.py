from __future__ import annotations

import csv
import os
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

from ase.db import connect
from ase.io import read, write

sys.path.append(os.getcwd())

from cats.working_directory.tddft_active_learning_workflow import run_workflow


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _load_db_rows(db_path: Path, row_ids: list[int] | None, max_structures: int | None):
    with connect(str(db_path)) as db:
        if row_ids is None:
            rows = list(db.select())
        else:
            rows = []
            for row_id in row_ids:
                try:
                    rows.append(db.get(id=int(row_id)))
                except KeyError as exc:
                    raise ValueError(f"Row id {row_id} not found in {db_path}.") from exc
    if max_structures is not None:
        rows = rows[: max(0, int(max_structures))]
    return rows


def _safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if value != value:  # NaN check
        return None
    return value


def _get_initial_mace_energy(row, metadata: dict):
    # Prefer explicit metadata written by the co-adsorbate workflow.
    energy = _safe_float(metadata.get("energy_eV"))
    if energy is not None:
        return energy
    # Fallback to ASE DB stored calculator result if present.
    return _safe_float(getattr(row, "energy", None))


def _effective_iteration_count(workflow_kwargs: dict) -> int:
    n_iter = int(workflow_kwargs.get("n_iter", 1))
    bond_scan = workflow_kwargs.get("bond_scan")
    if bond_scan is None:
        return 1
    enabled = bool(getattr(bond_scan, "enabled", False))
    return n_iter if enabled else 1


def _expected_final_orca_out(run_dir: Path, workflow_kwargs: dict) -> Path:
    use_tddft = bool(workflow_kwargs.get("use_tddft_optimisation", True))
    root = "TDDFT" if use_tddft else "DFT"
    iter_idx = _effective_iteration_count(workflow_kwargs)
    return run_dir / root / f"opt_iter_{iter_idx}" / "orca.out"


def _get_final_energy_from_run(
    run_dir: Path,
    workflow_kwargs: dict,
    final_atoms=None,
):
    out_path = _expected_final_orca_out(run_dir, workflow_kwargs)
    if out_path.exists():
        try:
            out_atoms = read(out_path, format="orca-output")
            return _safe_float(out_atoms.get_potential_energy())
        except Exception:
            pass
    if final_atoms is not None:
        try:
            return _safe_float(final_atoms.get_potential_energy())
        except Exception:
            return None
    return None


def run_batch_tddft_from_similarity_db(
    similarity_db_path: str | Path = "scratch/coadsorption_final_for_dft.db",
    batch_root: str | Path = "tddft_batch",
    row_ids: list[int] | None = None,
    max_structures: int | None = None,
    skip_completed: bool = True,
    workflow_kwargs: dict | None = None,
    summary_csv_path: str | Path | None = None,
):
    """
    Run multiple single-structure TDDFT active-learning optimisations from
    a similarity workflow output DB.

    Each structure runs in its own working directory:
    `batch_root/row_<id>/`
    """
    similarity_db_path = Path(similarity_db_path)
    batch_root = Path(batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    summary_csv_path = (
        Path(summary_csv_path)
        if summary_csv_path is not None
        else (batch_root / "batch_summary.csv")
    )
    summary_csv_path.parent.mkdir(parents=True, exist_ok=True)

    default_workflow_kwargs = {
        "n_iter": 1,
        "bond_scan": None,
        "al_steps_per_iter": 100,
        "use_tddft_optimisation": True,
        "xc_functional": "pbe0",
        "solvent": "Water",
        "use_scan_geometry": True,
        "optimize_to_convergence": True,
        "max_al_steps_per_iter": None,
        "fmax": 0.05,
        "initial_fmax": 0.01,
        "run_initial_dft_if_missing": True,
        "ensure_initial_dft_converged": True,
    }
    effective_workflow_kwargs = {
        **default_workflow_kwargs,
        **(workflow_kwargs or {}),
    }

    rows = _load_db_rows(similarity_db_path, row_ids=row_ids, max_structures=max_structures)
    if not rows:
        raise ValueError(f"No rows selected from {similarity_db_path}.")

    run_mode_tddft = bool(effective_workflow_kwargs.get("use_tddft_optimisation", True))
    iter_idx = _effective_iteration_count(effective_workflow_kwargs)
    completed_marker = (
        Path(f"TDDFT/opt_iter_{iter_idx}/iteration_complete.txt")
        if run_mode_tddft
        else Path(f"DFT/opt_iter_{iter_idx}/iteration_complete.txt")
    )

    summary_rows = []
    n_total = len(rows)
    for idx, row in enumerate(rows, start=1):
        run_dir = batch_root / f"row_{int(row.id):04d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        metadata = dict(getattr(row, "data", {}) or {})
        initial_mace_energy = _get_initial_mace_energy(row, metadata)

        print(
            f"[Batch TDDFT] ({idx}/{n_total}) row_id={int(row.id)} -> {run_dir}"
        )

        if skip_completed and (run_dir / completed_marker).exists():
            print("[Batch TDDFT] Skipping: iteration_complete marker already found.")
            final_dft_energy = _get_final_energy_from_run(
                run_dir,
                workflow_kwargs=effective_workflow_kwargs,
            )
            summary_rows.append(
                {
                    "row_id": int(row.id),
                    "run_dir": str(run_dir.resolve()),
                    "status": "skipped_completed",
                    "error": "",
                    "initial_mace_energy_eV": initial_mace_energy,
                    "final_dft_energy_eV": final_dft_energy,
                    "source_candidate_row_id": metadata.get("source_candidate_row_id"),
                    "source_relaxed_row_id": metadata.get("source_relaxed_row_id"),
                }
            )
            continue

        atoms = row.toatoms()
        write(run_dir / "input.xyz", atoms)
        try:
            with _pushd(run_dir):
                final_atoms = run_workflow(atoms, **effective_workflow_kwargs)
            final_dft_energy = _get_final_energy_from_run(
                run_dir,
                workflow_kwargs=effective_workflow_kwargs,
                final_atoms=final_atoms,
            )
            write(run_dir / "final.xyz", final_atoms)
            summary_rows.append(
                {
                    "row_id": int(row.id),
                    "run_dir": str(run_dir.resolve()),
                    "status": "ok",
                    "error": "",
                    "initial_mace_energy_eV": initial_mace_energy,
                    "final_dft_energy_eV": final_dft_energy,
                    "source_candidate_row_id": metadata.get("source_candidate_row_id"),
                    "source_relaxed_row_id": metadata.get("source_relaxed_row_id"),
                }
            )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            (run_dir / "error.log").write_text(
                traceback.format_exc(),
                encoding="utf-8",
            )
            print(f"[Batch TDDFT] Failed: {error_text}")
            summary_rows.append(
                {
                    "row_id": int(row.id),
                    "run_dir": str(run_dir.resolve()),
                    "status": "failed",
                    "error": error_text,
                    "initial_mace_energy_eV": initial_mace_energy,
                    "final_dft_energy_eV": None,
                    "source_candidate_row_id": metadata.get("source_candidate_row_id"),
                    "source_relaxed_row_id": metadata.get("source_relaxed_row_id"),
                }
            )

    comparable_rows = [
        row
        for row in summary_rows
        if row["initial_mace_energy_eV"] is not None and row["final_dft_energy_eV"] is not None
    ]
    if comparable_rows:
        mace_ref = min(row["initial_mace_energy_eV"] for row in comparable_rows)
        dft_ref = min(row["final_dft_energy_eV"] for row in comparable_rows)
        for row in summary_rows:
            init_e = row.get("initial_mace_energy_eV")
            final_e = row.get("final_dft_energy_eV")
            if init_e is None or final_e is None:
                row["rel_mace_energy_eV"] = None
                row["rel_dft_energy_eV"] = None
                row["rel_energy_shift_eV"] = None
                continue
            rel_mace = float(init_e - mace_ref)
            rel_dft = float(final_e - dft_ref)
            row["rel_mace_energy_eV"] = rel_mace
            row["rel_dft_energy_eV"] = rel_dft
            row["rel_energy_shift_eV"] = float(rel_dft - rel_mace)
    else:
        for row in summary_rows:
            row["rel_mace_energy_eV"] = None
            row["rel_dft_energy_eV"] = None
            row["rel_energy_shift_eV"] = None

    fieldnames = [
        "row_id",
        "run_dir",
        "status",
        "error",
        "initial_mace_energy_eV",
        "final_dft_energy_eV",
        "rel_mace_energy_eV",
        "rel_dft_energy_eV",
        "rel_energy_shift_eV",
        "source_candidate_row_id",
        "source_relaxed_row_id",
    ]
    with summary_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    n_ok = sum(row["status"] == "ok" for row in summary_rows)
    n_failed = sum(row["status"] == "failed" for row in summary_rows)
    n_skipped = sum(row["status"] == "skipped_completed" for row in summary_rows)
    print(f"[Batch TDDFT] Completed: ok={n_ok}, failed={n_failed}, skipped={n_skipped}")
    print(
        "[Batch TDDFT] Energy comparison rows: "
        f"{sum(row['rel_dft_energy_eV'] is not None for row in summary_rows)}"
    )
    print(f"[Batch TDDFT] Summary: {summary_csv_path.resolve()}")
    return summary_rows


if __name__ == "__main__":
    # Minimal editable example to pair with co-adsorbate similarity output.
    run_batch_tddft_from_similarity_db(
        similarity_db_path="scratch/coadsorption_final_for_dft.db",
        batch_root="tddft_batch",
        max_structures=5,  # set None to run every structure in the DB
        skip_completed=True,
        workflow_kwargs={
            "n_iter": 1,
            "bond_scan": None,
            "al_steps_per_iter": 100,
            "use_tddft_optimisation": True,
            "use_scan_geometry": True,
            "optimize_to_convergence": True,
            "max_al_steps_per_iter": None,
            "fmax": 0.05,
            "initial_fmax": 0.01,
            "run_initial_dft_if_missing": True,
            "ensure_initial_dft_converged": True,
        },
    )
