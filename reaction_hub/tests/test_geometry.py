from __future__ import annotations

import pytest

from jobs.geometry import ase_molecule_to_atoms, structure_input_to_atoms


def test_ase_molecule_to_atoms_builds_single_atom_at_origin():
    atoms, metadata = ase_molecule_to_atoms("Cu")

    assert len(atoms) == 1
    assert atoms[0].symbol == "Cu"
    assert tuple(float(value) for value in atoms.positions[0]) == pytest.approx((0.0, 0.0, 0.0))
    assert metadata["source"] == "ase_molecule"
    assert metadata["ase_molecule_formula"] == "Cu"
    assert metadata["formula"] == "Cu"
    assert metadata["atom_count"] == 1


def test_ase_molecule_to_atoms_keeps_co_as_molecule():
    atoms, metadata = ase_molecule_to_atoms("CO")

    assert len(atoms) == 2
    assert sorted(atoms.get_chemical_symbols()) == ["C", "O"]
    assert metadata["formula"] == "CO"
    assert metadata["atom_count"] == 2


def test_structure_input_to_atoms_accepts_single_atom_adsorbate():
    atoms, metadata = structure_input_to_atoms(
        geometry_source="ase_molecule",
        ase_molecule_formula="Au",
    )

    assert len(atoms) == 1
    assert atoms[0].symbol == "Au"
    assert tuple(float(value) for value in atoms.positions[0]) == pytest.approx((0.0, 0.0, 0.0))
    assert metadata["formula"] == "Au"
