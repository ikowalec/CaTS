from __future__ import annotations

import argparse
import random
from pathlib import Path

from ase.io import read, write


def split_images(images, seed: int):
    images = list(images)
    lowest_energy_indices = {}
    for index, atoms in enumerate(images):
        composition = atoms.get_chemical_formula(mode="hill")
        energy = atoms.get_potential_energy()
        if (
            composition not in lowest_energy_indices
            or energy < lowest_energy_indices[composition][0]
        ):
            lowest_energy_indices[composition] = (energy, index)

    stable_indices = {index for _, index in lowest_energy_indices.values()}
    stable = [atoms for index, atoms in enumerate(images) if index in stable_indices]
    shuffled = [atoms for index, atoms in enumerate(images) if index not in stable_indices]
    random.Random(seed).shuffle(shuffled)

    train_fill = max(0, int(0.8 * len(images)) - len(stable))
    validation_end = train_fill + min(int(0.1 * len(images)), len(shuffled) - train_fill)
    return {
        "train": stable + shuffled[:train_fill],
        "validation": shuffled[train_fill:validation_end],
        "test": shuffled[validation_end:],
    }


def add_reference_properties(images) -> None:
    for atoms in images:
        atoms.info["REF_energy"] = atoms.get_potential_energy()
        atoms.set_array("REF_forces", atoms.get_forces().copy())


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split an XYZ/extxyz file into 80% train, 10% validation, 10% test, "
            "reserving each composition's lowest-energy structure for training."
        )
    )
    parser.add_argument("xyz", type=Path, help="Input XYZ/extxyz file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory (default: current directory)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    images = read(args.xyz, index=":")
    if not images:
        raise SystemExit(f"No images found in {args.xyz}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    splits = split_images(images, args.seed)
    add_reference_properties(images)

    for name, split in splits.items():
        if split:
            write(args.output_dir / f"{name}.xyz", split, format="extxyz")

    print(
        "Wrote "
        + ", ".join(f"{name}={len(split)}" for name, split in splits.items())
        + f" to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
