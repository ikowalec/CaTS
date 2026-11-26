import importlib.util
from pathlib import Path

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "cats"
    / "working_directory"
    / "split_xyz_dataset.py"
)
SPEC = importlib.util.spec_from_file_location("split_xyz_dataset", SCRIPT_PATH)
splitter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(splitter)


def _atoms(symbol: str, energy: float) -> Atoms:
    atoms = Atoms(symbol)
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    return atoms


def test_split_reserves_each_composition_lowest_energy_structure_for_training():
    images = [_atoms("Au", energy) for energy in range(5)]
    images += [_atoms("Cu", energy) for energy in range(5)]

    splits = splitter.split_images(images, seed=42)

    train = {(atoms.symbols[0], atoms.get_potential_energy()) for atoms in splits["train"]}
    assert {("Au", 0.0), ("Cu", 0.0)} <= train
    assert {name: len(split) for name, split in splits.items()} == {
        "train": 8,
        "validation": 1,
        "test": 1,
    }
