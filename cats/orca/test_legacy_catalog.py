from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixBondLength
from ase.io import read as ase_read
from ase.io import write

import cats.orca.legacy_catalog as legacy_catalog_module

from cats.orca.legacy_catalog import (
    _parse_ct_multiple,
    catalog_legacy_orca,
    expand_fragment_selector,
    extract_embedded_orca_input,
    parse_multiwfn_config,
    qualify_orca_input,
)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _orca_output(*, cpcm: bool = False) -> str:
    solvent = " CPCM(Water)" if cpcm else ""
    return f"""
Program Version 5.0
================================================================================
                                       INPUT FILE
================================================================================
NAME = orca.inp
|  1> ! CAM-B3LYP 6-31g(d,p) DefGrid3 VeryTightSCF NormalPrint OPT SlowConv{solvent}
|  2> * xyz 0 1
|  3> *
================================================================================
CARTESIAN COORDINATES (ANGSTROEM)
---------------------------------
Au  0.000000  0.000000  0.000000
Au  2.500000  0.000000  0.000000
O   3.700000  0.000000  0.000000

FINAL SINGLE POINT ENERGY -101.250000000
MULLIKEN ATOMIC CHARGES
-----------------------
  0 Au :  0.100000
  1 Au : -0.050000
  2 O  : -0.050000
LOEWDIN ATOMIC CHARGES
----------------------
  0 Au :  0.080000
  1 Au : -0.030000
  2 O  : -0.050000
ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS
  1   20000.0   500.0   0.1000
  2   25000.0   400.0   0.0500
ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS
ORCA TERMINATED NORMALLY
"""


def test_input_and_fragment_parsers(tmp_path: Path):
    embedded = extract_embedded_orca_input(_orca_output())
    assert "CAM-B3LYP" in embedded
    assert qualify_orca_input(embedded) == []
    assert qualify_orca_input(extract_embedded_orca_input(_orca_output(cpcm=True))) == [
        "cpcm_present"
    ]
    assert expand_fragment_selector("1-3, 5") == [1, 2, 3, 5]

    config = tmp_path / "config.txt"
    config.write_text(
        "orca.molden.input\n18\n16\n2\n1-2\n3\norca.out\n1\nq\n",
        encoding="utf-8",
    )
    parsed = parse_multiwfn_config(config, atom_count=3)
    assert parsed is not None
    assert parsed["fragments_0based"] == [[0, 1], [2]]
    assert parsed["validation_errors"] == []


def test_ct_multiple_accepts_quoted_member_paths(tmp_path: Path):
    ct_dir = tmp_path / "CT_multiple"
    ct_dir.mkdir()
    (ct_dir / "ET_1to2.txt").write_text(
        "1 1\n2.0 0.5\n",
        encoding="utf-8",
    )
    (ct_dir / "CT_multiple.txt").write_text(
        '"CT_multiple/ET_1to2.txt" 1->2\n',
        encoding="utf-8",
    )

    rows, index_files, warnings = _parse_ct_multiple("ct", tmp_path, tmp_path, -1)

    assert warnings == []
    assert index_files == ["CT_multiple/CT_multiple.txt"]
    assert rows[0]["property"] == "1->2"
    assert rows[0]["wavelength_nm"] == pytest.approx(619.920992)


@pytest.mark.filterwarnings("ignore:.*could not create cache path.*")
def test_catalog_extracts_spectra_ct_and_constraints(tmp_path: Path, monkeypatch):
    root = tmp_path / "legacy"
    accepted = root / "accepted"
    rejected = root / "rejected"
    unrelated = root / "unrelated"
    accepted.mkdir(parents=True)
    rejected.mkdir(parents=True)
    unrelated.mkdir(parents=True)
    (accepted / "orca.out").write_text(_orca_output(), encoding="utf-8")
    (rejected / "orca.out").write_text(_orca_output(cpcm=True), encoding="utf-8")
    (accepted / "orca.molden.input").write_text("[Molden Format]\n", encoding="utf-8")
    (accepted / "orca.gbw").write_bytes(b"wavefunction")
    (accepted / "config.txt").write_text(
        "orca.molden.input\n18\n16\n2\n1-2\n3\norca.out\n1\nq\n",
        encoding="utf-8",
    )
    (accepted / "IFCTdata.txt").write_text(
        "1 0.100 500.0: 1->2 75.0% Redis(1) 25.0% CT(e) 0.40 CT(%) 40.0\n",
        encoding="utf-8",
    )

    trajectory = Atoms("Au2O", positions=[[0, 0, 0], [2.5, 0, 0], [3.7, 0, 0]])
    trajectory.set_constraint(FixBondLength(1, 2))
    trajectory.calc = SinglePointCalculator(
        trajectory,
        energy=-10.0,
        forces=[[0.01, 0.0, 0.0], [0.0, 0.02, 0.0], [0.0, 0.0, 0.03]],
    )
    write(accepted / "fbl.traj", [trajectory])
    (unrelated / "empty.traj").touch()

    output = tmp_path / "catalog"
    summary = catalog_legacy_orca(root, output)

    assert summary["accepted"] == 1
    assert summary["rejected"] == 1
    assert summary["tddft_parsed"] == 1
    assert summary["ct_parsed"] == 1
    assert summary["trajectory_frames"] == 1
    assert len(summary["source_tree_id"]) == 64
    assert "unrelated/empty.traj" not in {
        row["path"] for row in _rows(output / "audit.csv")
    }

    calculation = _rows(output / "calculations.csv")[0]
    assert calculation["formula"] == "Au2O"
    assert calculation["cluster_family"] == "Au2"
    assert calculation["adsorbate_formula"] == "O"
    assert calculation["tddft_status"] == "parsed"
    assert calculation["ct_status"] == "parsed"
    assert calculation["orca_version"] == "5.0"
    assert len(calculation["calculation_fingerprint"]) == 64
    assert len(calculation["parse_fingerprint"]) == 64
    assert calculation["charge"] == "0"
    assert calculation["multiplicity"] == "1"
    assert "VeryTightSCF" in calculation["input_text"]
    assert json.loads(calculation["simple_keywords"])[0].startswith("CAM-B3LYP")
    assert len(_rows(output / "transitions.csv")) == 2
    assert len(_rows(output / "spectra.csv")) > 2
    assert _rows(output / "ct_states.csv")[0]["state"] == "1"
    assert _rows(output / "ct_terms.csv")[0]["term"] == "1->2"
    charges = _rows(output / "atomic_charges.csv")
    assert len(charges) == 6
    assert {row["method"] for row in charges} == {"loewdin", "mulliken"}
    manifest = _rows(output / "artifacts.csv")
    gbw = next(row for row in manifest if row["path"].endswith("orca.gbw"))
    assert gbw["path_relative_to_calculation"] == "orca.gbw"
    assert len(gbw["sha256"]) == 64

    fragments = json.loads((output / "fragments.jsonl").read_text(encoding="utf-8"))
    assert fragments["fragments_0based"] == [[0, 1], [2]]

    scan = _rows(output / "scan_frames.csv")[0]
    assert json.loads(scan["bond_pairs_0based"]) == [[1, 2]]
    assert json.loads(scan["bond_pairs_1based"]) == [[2, 3]]
    assert json.loads(scan["symbols"]) == ["Au", "Au", "O"]
    assert json.loads(scan["positions_A"])[2] == [3.7, 0.0, 0.0]
    assert json.loads(scan["forces_eV_A"])[2] == [0.0, 0.0, 0.03]
    calculation_id = calculation["calculation_id"]
    structure_path = output / calculation["structure_file"]
    assert structure_path.exists()
    assert ase_read(structure_path).get_chemical_formula() == "Au2O"
    assert (output / "plots" / f"{calculation_id}_uvvis.png").exists()

    schema_two_summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    schema_two_summary["schema_version"] = 2
    (output / "summary.json").write_text(
        json.dumps(schema_two_summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    calculations_text = (output / "calculations.csv").read_text(encoding="utf-8")
    (output / "calculations.csv").write_text(
        calculations_text.replace(calculation["parse_fingerprint"], "", 1),
        encoding="utf-8",
    )

    relocated_parent = tmp_path / "new_hpc_prefix"
    relocated_parent.mkdir()
    relocated_root = relocated_parent / root.name
    shutil.move(root, relocated_root)
    monkeypatch.setattr(
        legacy_catalog_module,
        "_sha256",
        lambda _path: (_ for _ in ()).throw(AssertionError("unchanged files were rehashed")),
    )
    relocated_summary = catalog_legacy_orca(relocated_root, output)
    assert relocated_summary["source_tree_id"] == summary["source_tree_id"]
    assert relocated_summary["reused_calculations"] == 1
    assert _rows(output / "calculations.csv")[0]["calculation_fingerprint"] == calculation[
        "calculation_fingerprint"
    ]
    assert len(_rows(output / "scan_frames.csv")) == 1

    monkeypatch.undo()
    (relocated_root / "accepted" / "orca.gbw").write_bytes(b"changed wavefunction")
    changed_summary = catalog_legacy_orca(relocated_root, output)
    assert changed_summary["reused_calculations"] == 0
    changed_calculation = _rows(output / "calculations.csv")[0]
    assert changed_calculation["calculation_fingerprint"] == calculation["calculation_fingerprint"]
    assert changed_calculation["parse_fingerprint"] != calculation["parse_fingerprint"]
