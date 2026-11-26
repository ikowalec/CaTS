"""Build the requested Au/Cu pentagonal nanorod and write an ASE trajectory."""

from argparse import ArgumentParser
from math import cos, pi, sin, sqrt

import numpy as np
from ase import Atoms
from ase.io import write


def pentagon(radius, z, phase=0.0):
    return [
        (radius * cos(phase + 2 * pi * i / 5), radius * sin(phase + 2 * pi * i / 5), z)
        for i in range(5)
    ]


def terminal_hollows(axis, ring, distance, top):
    """Return tetrahedral sites above the five equilateral terminal faces."""
    sites = []
    for i in range(5):
        face = np.array([axis, ring[i], ring[(i + 1) % 5]])
        centre = face.mean(axis=0)
        normal = np.cross(face[1] - face[0], face[2] - face[0])
        normal /= np.linalg.norm(normal)
        candidates = centre + np.array([-1, 1])[:, None] * sqrt(2 / 3) * distance * normal
        sites.append(candidates[np.argmax(candidates[:, 2]) if top else np.argmin(candidates[:, 2])])
    return sites


def build_nanorod(units=3, distance=2.56, vacuum=6.0):
    if units < 1:
        raise ValueError("units must be at least 1")

    inner_radius = distance / (2 * sin(pi / 5))
    half_repeat = sqrt(distance**2 - inner_radius**2)
    repeat = 2 * half_repeat
    bridge_radius = inner_radius * cos(pi / 5) + distance * sqrt(3) / 2
    outer_radius = 2 * inner_radius

    symbols, positions, tags = [], [], []

    def add(symbol, points, tag):
        symbols.extend([symbol] * len(points))
        positions.extend(points)
        tags.extend([tag] * len(points))

    # Axial Au atoms bracket each whole Cu-pentagon repeat unit.
    add("Au", [(0.0, 0.0, i * repeat) for i in range(units + 1)], 1)
    cu_rings = []
    for i in range(units):
        z = (i + 0.5) * repeat
        ring = pentagon(inner_radius, z)
        cu_rings.append(ring)
        add("Cu", ring, 2)
        # Edge-bridging Au atoms make the five points of each star.
        add("Au", pentagon(bridge_radius, z, pi / 5), 3)

    # These rings fill the gaps between stars and terminate both rod ends.
    for i in range(units + 1):
        add("Au", pentagon(outer_radius, i * repeat), 4)

    add("Au", terminal_hollows((0, 0, 0), cu_rings[0], distance, top=False), 5)
    add(
        "Au",
        terminal_hollows((0, 0, units * repeat), cu_rings[-1], distance, top=True),
        5,
    )

    atoms = Atoms(symbols, positions=positions, tags=tags, pbc=False)
    atoms.center(vacuum=vacuum)

    cu = atoms.positions[np.array(symbols) == "Cu"].reshape(units, 5, 3)
    edges = np.linalg.norm(cu - np.roll(cu, -1, axis=1), axis=2)
    assert np.allclose(edges, distance, atol=1e-10)
    assert len(atoms) == 16 * (units + 1)
    pair_distances = atoms.get_all_distances()
    assert pair_distances[pair_distances > 0].min() >= distance - 1e-10
    return atoms


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("output", nargs="?", default=f"au_cu_pentagonal_nanorod_3.traj")
    parser.add_argument("-n", "--units", type=int, default=3)
    args = parser.parse_args()
    cluster = build_nanorod(units=args.units)
    write(args.output, cluster)
    print(f"Wrote {args.output}: {cluster.get_chemical_formula()} ({len(cluster)} atoms)")
