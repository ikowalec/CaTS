from __future__ import annotations

import argparse
from itertools import product
from math import sqrt
from pathlib import Path

from ase import Atoms
from ase.io import write


LATTICE_CONSTANTS_A = {"Cu": 3.615, "Au": 4.078}
TETRAHEDRON_LAYERS = (2, 3, 4, 5)
CUBOID_CELLS = ((1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 2, 1))


def make_111_tetrahedron(symbol: str, layers: int, lattice_constant: float) -> Atoms:
    nearest_neighbour = lattice_constant / sqrt(2)
    layer_spacing = sqrt(2 / 3) * nearest_neighbour
    positions = []
    for layer in range(layers):
        edge_atoms = layers - layer
        for j in range(edge_atoms):
            for i in range(edge_atoms - j):
                positions.append(
                    [
                        (i + 0.5 * j) * nearest_neighbour + layer * 0.5 * nearest_neighbour,
                        sqrt(3) / 2 * j * nearest_neighbour
                        + layer * sqrt(3) / 6 * nearest_neighbour,
                        layer * layer_spacing,
                    ]
                )
    return Atoms(symbol * len(positions), positions=positions)


def make_100_cuboid(
    symbol: str,
    cells: tuple[int, int, int],
    lattice_constant: float,
) -> Atoms:
    nx, ny, nz = cells
    positions = [
        [i * lattice_constant / 2, j * lattice_constant / 2, k * lattice_constant / 2]
        for i, j, k in product(range(2 * nx + 1), range(2 * ny + 1), range(2 * nz + 1))
        if (i + j + k) % 2 == 0
    ]
    return Atoms(symbol * len(positions), positions=positions)


def make_clusters(vacuum: float = 8.0):
    clusters = []
    for symbol, lattice_constant in LATTICE_CONSTANTS_A.items():
        element_clusters = []
        for layers in TETRAHEDRON_LAYERS:
            atoms = make_111_tetrahedron(symbol, layers, lattice_constant)
            atoms.info.update(
                {
                    "name": f"{symbol}{len(atoms)}_tetrahedron_111",
                    "element": symbol,
                    "shape": "tetrahedron",
                    "facets": "(111)",
                    "layers": layers,
                    "lattice_constant_A": lattice_constant,
                }
            )
            element_clusters.append(atoms)
        for cells in CUBOID_CELLS:
            atoms = make_100_cuboid(symbol, cells, lattice_constant)
            dimensions = "x".join(str(value) for value in cells)
            shape = "cube" if len(set(cells)) == 1 else "cuboid"
            atoms.info.update(
                {
                    "name": f"{symbol}{len(atoms)}_{shape}_100_{dimensions}",
                    "element": symbol,
                    "shape": shape,
                    "facets": "(100)",
                    "cells": dimensions,
                    "lattice_constant_A": lattice_constant,
                }
            )
            element_clusters.append(atoms)
        for atoms in sorted(element_clusters, key=len):
            assert len(atoms) <= 40
            atoms.center(vacuum=float(vacuum))
            clusters.append(atoms)
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate separate Cu/Au FCC clusters bounded only by (111) or (100)."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("Cu_Au_111_100_clusters.traj"),
    )
    parser.add_argument("--vacuum", type=float, default=8.0)
    args = parser.parse_args()

    clusters = make_clusters(args.vacuum)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, clusters)
    for atoms in clusters:
        print(f"{atoms.info['name']}: {len(atoms)} atoms")
    print(f"Wrote {len(clusters)} clusters to {args.output}")


if __name__ == "__main__":
    main()
