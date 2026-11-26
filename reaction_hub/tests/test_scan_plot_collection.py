from __future__ import annotations

import csv
from pathlib import Path

from ase import Atoms
from ase.db import connect

from cats.orca.tddft.ct import SCAN_LOG_FIELDNAMES
from cats.orca.tddft.scan import (
    _collect_converged_state_wavelength_points,
    _energy_profile_entries_from_scan_db,
    _select_latest_entries_by_bond_length,
)


def _write_scan_log(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCAN_LOG_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            normalized = {name: str(row.get(name, "")) for name in SCAN_LOG_FIELDNAMES}
            writer.writerow(normalized)


def test_collect_converged_state_wavelength_points_includes_no_overlap_fallback_rows(
    tmp_path: Path,
) -> None:
    scan_log = tmp_path / "TDDFT_active_learning_shared" / "iroot_scan_endpoints.csv"
    _write_scan_log(
        scan_log,
        [
            {
                "iteration": "0",
                "scan_dir": "TDDFT/scan_iter_0",
                "wavelength_nm": "500.0",
                "iroot": "8",
                "point_kind": "",
                "point_label": "",
            },
            {
                "iteration": "1",
                "scan_dir": "TDDFT/scan_iter_1_geom_1",
                "wavelength_nm": "512.0",
                "iroot": "9",
                "point_kind": "fallback_no_orbital_overlap",
                "point_label": "no_orbital_overlap_initial_scoring",
            },
            {
                "iteration": "2",
                "scan_dir": "TDDFT/scan_iter_2_geom_1",
                "wavelength_nm": "526.0",
                "iroot": "10",
                "point_kind": "fallback_no_orbital_overlap",
                "point_label": "no_orbital_overlap_initial_scoring",
            },
        ],
    )

    points = _collect_converged_state_wavelength_points(
        max_iter=2,
        scan_log_path=scan_log,
    )

    assert [int(point["iteration"]) for point in points] == [1, 2]
    assert [int(point["image_number"]) for point in points] == [1, 2]
    assert [float(point["wavelength_nm"]) for point in points] == [512.0, 526.0]


def test_select_latest_entries_by_bond_length_collapses_near_duplicate_endpoints() -> None:
    entries: list[dict[str, object]] = [
        {"iteration": 1, "scan_dir": Path("DFT/scan_iter_1"), "bond_length": 1.10000},
        {"iteration": 2, "scan_dir": Path("DFT/scan_iter_2"), "bond_length": 1.10004},
        {"iteration": 3, "scan_dir": Path("DFT/scan_iter_3"), "bond_length": 1.30000},
        {"iteration": 4, "scan_dir": Path("DFT/scan_iter_4"), "bond_length": 1.49996},
        {"iteration": 5, "scan_dir": Path("DFT/scan_iter_5"), "bond_length": 1.50001},
    ]

    selected = _select_latest_entries_by_bond_length(entries)

    assert [int(entry["iteration"]) for entry in selected] == [2, 3, 5]
    assert [float(entry["bond_length"]) for entry in selected] == [1.10004, 1.3, 1.50001]


def test_energy_profile_entries_from_scan_db_use_exported_endpoint_values(tmp_path: Path) -> None:
    db_path = tmp_path / "scan_endpoints.db"
    db = connect(str(db_path))
    atoms = Atoms("OH", positions=[(0.0, 0.0, 0.0), (1.42, 0.0, 0.0)])
    db.write(
        atoms,
        iteration=4,
        total_energy_eV=-123.45,
        data={
            "scan_bond_length": 1.10,
            "scan_point_kind": "ts_smoothing",
            "scan_point_label": "smoothed",
        },
    )

    entries = _energy_profile_entries_from_scan_db(db_path, bond_pair=(0, 1))

    assert len(entries) == 1
    assert entries[0]["iteration"] == 4
    assert entries[0]["energy_eV"] == -123.45
    assert entries[0]["bond_length"] == 1.42
    assert entries[0]["point_kind"] == "ts_smoothing"
