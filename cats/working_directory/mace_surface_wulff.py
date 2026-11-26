"""MACE surface energies using the one-sided relaxation method from DOI: 10.1039/D2CP01019D."""

from __future__ import annotations

import argparse
import csv
from math import gcd
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.cluster.cluster import Cluster
from ase.cluster.cubic import (
    BodyCenteredCubicFactory,
    FaceCenteredCubicFactory,
    SimpleCubicFactory,
)
from ase.constraints import FixAtoms
from ase.filters import ExpCellFilter
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor


EV_A2_TO_J_M2 = 16.02176634


class CubicLayerDistances:
    def get_layer_distance(self, miller, layers=1, tol=1e-9, new=True):
        h, k, l = (int(value) for value in miller)
        # The gcd of basis-plane coordinates gives the exact occupied-plane spacing.
        values = [2 * h, 2 * k, 2 * l]
        symmetry = getattr(self, "xtal_name", None) or self.symmetry
        if symmetry == "fcc":
            values.extend((h + k, h + l, k + l))
        elif symmetry == "bcc":
            values.append(h + k + l)
        lattice_constant = float(np.linalg.norm(self.lattice_basis[0]))
        spacing = lattice_constant * gcd(*(abs(value) for value in values))
        return layers * spacing / (2.0 * np.sqrt(h * h + k * k + l * l))


class CubicCluster(CubicLayerDistances, Cluster):
    pass


class StableFCCFactory(CubicLayerDistances, FaceCenteredCubicFactory):
    Cluster = CubicCluster


class StableBCCFactory(CubicLayerDistances, BodyCenteredCubicFactory):
    Cluster = CubicCluster


class StableSCFactory(CubicLayerDistances, SimpleCubicFactory):
    Cluster = CubicCluster


WULFF_FACTORIES = {
    "fcc": StableFCCFactory(),
    "bcc": StableBCCFactory(),
    "sc": StableSCFactory(),
}


def wulff_cluster(
    symbol: str,
    surfaces: list[tuple[int, int, int]],
    energies: list[float],
    size: int,
    structure: str,
    lattice_constant_A: float,
) -> Atoms:
    factory = WULFF_FACTORIES[structure]
    factory.set_atomic_numbers(symbol)
    factory.lattice_constant = lattice_constant_A
    factory.set_basis()
    layer_energies = np.asarray(energies) / np.array(
        [factory.get_layer_distance(surface) for surface in surfaces]
    )

    candidates = {}

    def make(factor: float) -> Atoms:
        layers = np.rint(factor * layer_energies).astype(int)
        cluster = factory(symbol, surfaces, layers, latticeconstant=lattice_constant_A)
        candidates[len(cluster)] = cluster
        return cluster

    low = 0.0
    high = size ** (1.0 / 3.0) / max(layer_energies)
    while len(make(high)) < size:
        low = high
        high *= 2.0

    for _ in range(32):
        middle = (low + high) / 2.0
        if len(make(middle)) < size:
            low = middle
        else:
            high = middle

    return min(candidates.values(), key=lambda atoms: (abs(len(atoms) - size), len(atoms) > size))


def cubic_surface_families(max_index: int) -> tuple[tuple[int, int, int], ...]:
    if max_index < 1:
        raise ValueError("Maximum Miller index must be positive.")
    return tuple(
        (h, k, l)
        for h in range(1, max_index + 1)
        for k in range(h + 1)
        for l in range(k + 1)
        if gcd(gcd(h, k), l) == 1
    )


DEFAULT_SURFACES = cubic_surface_families(4)


def parse_hkl(value: str) -> tuple[int, int, int]:
    text = value.strip().strip("()[]")
    parts = text.replace(" ", ",").split(",") if "," in text or " " in text else list(text)
    parts = [part for part in parts if part]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Miller indices must be like 111 or 3,2,2.")
    try:
        hkl = tuple(int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid Miller index: {value}") from exc
    if hkl == (0, 0, 0):
        raise argparse.ArgumentTypeError("Miller index cannot be (0,0,0).")
    return hkl


def paper_surface_energy(
    *,
    unrelaxed_slab_eV: float,
    relaxed_slab_eV: float,
    bulk_eV_per_atom: float,
    atom_count: int,
    area_A2: float,
) -> tuple[float, float, float]:
    cleave = (unrelaxed_slab_eV - atom_count * bulk_eV_per_atom) / (2.0 * area_A2)
    relax = (relaxed_slab_eV - unrelaxed_slab_eV) / area_A2
    return cleave, relax, cleave + relax


def completed_surface_rows(path: Path) -> dict[tuple[int, int, int], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["h"]), int(row["k"]), int(row["l"])): row
            for row in csv.DictReader(handle)
        }


def append_csv_row(path: Path, row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def surface_heights(atoms: Atoms) -> np.ndarray:
    normal = np.cross(atoms.cell[0], atoms.cell[1])
    normal /= np.linalg.norm(normal)
    if np.dot(normal, atoms.cell[2]) < 0:
        normal *= -1
    return atoms.positions @ normal


def surface_layers(atoms: Atoms, tolerance_A: float) -> tuple[np.ndarray, np.ndarray]:
    heights = surface_heights(atoms)
    tags = np.empty(len(atoms), dtype=int)
    levels = []
    for atom_index in np.argsort(heights):
        if not levels or heights[atom_index] - levels[-1] > tolerance_A:
            levels.append(float(heights[atom_index]))
        tags[atom_index] = len(levels) - 1
    return tags, np.asarray(levels)


def make_slab(
    optimized_bulk: Atoms,
    hkl: tuple[int, int, int],
    *,
    layers: int,
    minimum_thickness_A: float | None,
    vacuum_A: float,
    repeat_xy: tuple[int, int],
    layer_tolerance_A: float,
) -> Atoms:
    if minimum_thickness_A is None:
        min_slab_size = float(2 * layers)
        in_unit_planes = True
    else:
        min_slab_size = float(minimum_thickness_A + max(optimized_bulk.cell.lengths()))
        in_unit_planes = False
    generator = SlabGenerator(
        AseAtomsAdaptor.get_structure(optimized_bulk),
        hkl,
        min_slab_size=min_slab_size,
        min_vacuum_size=1.0,
        in_unit_planes=in_unit_planes,
        primitive=True,
        center_slab=False,
        reorient_lattice=True,
    )
    candidates = generator.get_slabs(symmetrize=False)
    if not candidates:
        raise RuntimeError(f"pymatgen generated no slab for {hkl}.")

    slab = AseAtomsAdaptor.get_atoms(candidates[0])
    tags, levels = surface_layers(slab, layer_tolerance_A)
    order = np.argsort(levels)
    required_layers = layers
    if minimum_thickness_A is not None:
        bottom = levels[order[0]]
        for count, layer_tag in enumerate(order, start=1):
            if levels[layer_tag] - bottom >= minimum_thickness_A - layer_tolerance_A:
                required_layers = max(layers, count)
                break
    if len(levels) < required_layers:
        raise RuntimeError(f"{hkl} produced only {len(levels)} atomic layers.")
    keep_tags = set(order[:required_layers].tolist())
    slab = slab[np.array([tag in keep_tags for tag in tags], dtype=bool)]
    slab.set_pbc((True, True, True))
    slab.center(vacuum=vacuum_A, axis=2)
    slab = slab.repeat((repeat_xy[0], repeat_xy[1], 1))

    _, final_levels = surface_layers(slab, layer_tolerance_A)
    thickness_A = float(np.ptp(surface_heights(slab)))
    if len(final_levels) != required_layers:
        raise RuntimeError(
            f"{hkl} has {len(final_levels)} layers after cutting; expected {required_layers}."
        )
    if minimum_thickness_A is not None and thickness_A < minimum_thickness_A - layer_tolerance_A:
        raise RuntimeError(
            f"{hkl} thickness is {thickness_A:.6f} A; expected at least "
            f"{minimum_thickness_A:.6f} A."
        )
    return slab


def bottom_layer_depth(
    slab: Atoms,
    *,
    bottom_layers: int,
    layer_tolerance_A: float,
) -> float:
    _, levels = surface_layers(slab, layer_tolerance_A)
    order = np.argsort(levels)
    if len(order) < bottom_layers:
        raise RuntimeError(f"Slab has only {len(order)} layers; cannot select {bottom_layers}.")
    return float(levels[order[bottom_layers - 1]] - levels[order[0]])


def constrain_bottom_depth(
    slab: Atoms,
    *,
    minimum_depth_A: float,
    layer_tolerance_A: float,
) -> tuple[int, int, float]:
    tags, levels = surface_layers(slab, layer_tolerance_A)
    order = np.argsort(levels)
    bottom = levels[order[0]]
    fixed_layers = len(order)
    for count, layer_tag in enumerate(order, start=1):
        if levels[layer_tag] - bottom >= minimum_depth_A - layer_tolerance_A:
            fixed_layers = count
            break
    fixed_tags = set(order[:fixed_layers].tolist())
    mask = np.array([tag in fixed_tags for tag in tags], dtype=bool)
    slab.set_constraint(FixAtoms(mask=mask))
    fixed_depth_A = float(levels[order[fixed_layers - 1]] - bottom)
    return int(mask.sum()), fixed_layers, fixed_depth_A


def conventional_lattice_constant(atoms: Atoms, structure: str) -> float:
    atoms_per_cell = {"fcc": 4, "bcc": 2, "sc": 1}[structure]
    return float((atoms_per_cell * atoms.get_volume() / len(atoms)) ** (1.0 / 3.0))


def make_calculator(args):
    if not str(args.device).lower().startswith("cuda"):
        raise RuntimeError("cuEquivariance is mandatory; use a CUDA device.")
    try:
        import cuequivariance  # noqa: F401
        import cuequivariance_torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "cuEquivariance is mandatory. Install cuequivariance and "
            "cuequivariance_torch for this CUDA/PyTorch environment."
        ) from exc

    from mace.calculators import mace_mp

    return mace_mp(
        model=args.model,
        device=args.device,
        default_dtype=args.default_dtype,
        dispersion=args.dispersion,
        enable_cueq=True,
    )


def optimize(atoms: Atoms, target, *, fmax: float, max_steps: int, directory: Path) -> None:
    optimizer = BFGS(
        target,
        trajectory=str(directory / "optimization.traj"),
        logfile=str(directory / "optimization.log"),
    )
    if not optimizer.run(fmax=fmax, steps=max_steps):
        raise RuntimeError(f"Optimization did not converge in {max_steps} steps: {directory}")


def build_wulff_series(
    *,
    symbol: str,
    structure: str,
    lattice_constant_A: float,
    surfaces: list[tuple[int, int, int]],
    surface_energies_J_m2: list[float],
    min_atoms: int,
    max_atoms: int,
    size_step: int,
    calculator,
    trajectory_path: Path,
    csv_path: Path,
) -> list[dict[str, int | float]]:
    rows = []
    seen_sizes = set()
    target_sizes = list(range(min_atoms, max_atoms + 1, size_step))
    if target_sizes[-1] != max_atoms:
        target_sizes.append(max_atoms)
    fieldnames = [
        "target_atom_count",
        "actual_atom_count",
        "mace_energy_eV",
        "mace_energy_eV_per_atom",
    ]
    with (
        Trajectory(trajectory_path, "w") as trajectory,
        csv_path.open("w", newline="", encoding="utf-8") as handle,
    ):
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for target_size in target_sizes:
            print(
                f"Constructing Wulff target {target_size} atoms from "
                f"{len(surfaces)} surface families",
                flush=True,
            )
            try:
                cluster = wulff_cluster(
                    symbol,
                    surfaces,
                    surface_energies_J_m2,
                    target_size,
                    structure,
                    lattice_constant_A,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Wulff construction failed for target {target_size} atoms, "
                    f"structure={structure}, surfaces={surfaces}"
                ) from exc
            actual_size = len(cluster)
            if actual_size in seen_sizes or not min_atoms <= actual_size <= max_atoms:
                continue
            seen_sizes.add(actual_size)
            cluster.set_pbc(False)
            cluster.set_cell(np.zeros((3, 3)), scale_atoms=False)
            cluster.calc = calculator
            energy_eV = float(cluster.get_potential_energy())
            row = {
                "target_atom_count": target_size,
                "actual_atom_count": actual_size,
                "mace_energy_eV": energy_eV,
                "mace_energy_eV_per_atom": energy_eV / actual_size,
            }
            cluster.info.update({"name": f"{symbol}{actual_size}_wulff", **row})
            trajectory.write(cluster)
            writer.writerow(row)
            handle.flush()
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate MACE surface energies and construct Wulff nanoparticles."
    )
    parser.add_argument("symbol", nargs="?", default="Pd")
    parser.add_argument("--structure", choices=("fcc", "bcc", "sc"), default="fcc")
    parser.add_argument("--lattice-constant", type=float)
    parser.add_argument("--model", default="medium", help="MACE model name or model-file path.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--dispersion", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--surfaces",
        nargs="+",
        default=["4"],
        help="One digit generates all reduced cubic families up to that index; "
        "otherwise provide explicit indices such as 111 100 322.",
    )
    parser.add_argument("--layers", type=int, default=7)
    parser.add_argument("--fixed-bottom-layers", type=int, default=3)
    parser.add_argument(
        "--repeat",
        nargs=2,
        type=int,
        default=(1, 1),
        metavar=("NX", "NY"),
        help="Lateral repeat; 1 1 is sufficient for a pristine ML-potential slab.",
    )
    parser.add_argument(
        "--vacuum",
        type=float,
        default=20.0,
        help="Vacuum on each side of each slab in angstrom.",
    )
    parser.add_argument("--layer-tolerance", type=float, default=0.2)
    parser.add_argument("--fmax", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--min-atoms", type=int, default=1)
    parser.add_argument("--max-atoms", type=int, default=300)
    parser.add_argument("--size-step", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("mace_surface_wulff"))
    args = parser.parse_args()

    if len(args.surfaces) == 1 and len(args.surfaces[0]) == 1 and args.surfaces[0].isdigit():
        try:
            args.surfaces = list(cubic_surface_families(int(args.surfaces[0])))
        except ValueError as exc:
            parser.error(str(exc))
    else:
        try:
            args.surfaces = [parse_hkl(value) for value in args.surfaces]
        except argparse.ArgumentTypeError as exc:
            parser.error(str(exc))

    if args.layers < 1 or args.fixed_bottom_layers < 1:
        parser.error("--layers and --fixed-bottom-layers must be positive.")
    if args.layers <= args.fixed_bottom_layers:
        parser.error("--layers must exceed --fixed-bottom-layers.")
    if any(value < 1 for value in args.repeat):
        parser.error("--repeat values must be positive.")
    if args.vacuum < 0:
        parser.error("--vacuum cannot be negative.")
    if args.fmax <= 0 or args.max_steps < 1 or args.layer_tolerance <= 0:
        parser.error("--fmax, --max-steps and --layer-tolerance must be positive.")
    if args.min_atoms < 1 or args.max_atoms < args.min_atoms:
        parser.error("Require 1 <= --min-atoms <= --max-atoms.")
    if args.size_step < 1:
        parser.error("--size-step must be positive.")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    calculator = make_calculator(args)

    bulk_kwargs = {"cubic": True}
    if args.lattice_constant is not None:
        bulk_kwargs["a"] = args.lattice_constant
    optimized_bulk = bulk(args.symbol, args.structure, **bulk_kwargs)
    optimized_bulk.calc = calculator
    bulk_dir = output_dir / "bulk"
    bulk_dir.mkdir(exist_ok=True)
    optimize(
        optimized_bulk,
        ExpCellFilter(optimized_bulk, hydrostatic_strain=True),
        fmax=args.fmax,
        max_steps=args.max_steps,
        directory=bulk_dir,
    )
    optimized_bulk.wrap()
    bulk_eV_per_atom = optimized_bulk.get_potential_energy() / len(optimized_bulk)
    lattice_constant_A = conventional_lattice_constant(optimized_bulk, args.structure)
    optimized_bulk.info.update(
        {
            "bulk_eV_per_atom": bulk_eV_per_atom,
            "lattice_constant_A": lattice_constant_A,
            "mace_model": args.model,
        }
    )
    write(bulk_dir / "optimized_bulk.extxyz", optimized_bulk)

    reference_100 = make_slab(
        optimized_bulk,
        (1, 0, 0),
        layers=args.layers,
        minimum_thickness_A=None,
        vacuum_A=args.vacuum,
        repeat_xy=(1, 1),
        layer_tolerance_A=args.layer_tolerance,
    )
    reference_thickness_A = float(np.ptp(surface_heights(reference_100)))
    reference_fixed_depth_A = bottom_layer_depth(
        reference_100,
        bottom_layers=args.fixed_bottom_layers,
        layer_tolerance_A=args.layer_tolerance,
    )
    print(
        f"(100) reference: thickness={reference_thickness_A:.6f} A, "
        f"fixed depth={reference_fixed_depth_A:.6f} A",
        flush=True,
    )

    surface_csv = output_dir / "surface_energies.csv"
    completed = completed_surface_rows(surface_csv)
    surface_rows = []
    for hkl in args.surfaces:
        if hkl in completed:
            surface_rows.append(completed[hkl])
            print(f"{hkl}: using completed facet", flush=True)
            continue
        label = "".join(str(value) for value in hkl)
        facet_dir = output_dir / "surfaces" / label
        facet_dir.mkdir(parents=True, exist_ok=True)
        unrelaxed = make_slab(
            optimized_bulk,
            hkl,
            layers=args.layers,
            minimum_thickness_A=reference_thickness_A,
            vacuum_A=args.vacuum,
            repeat_xy=tuple(args.repeat),
            layer_tolerance_A=args.layer_tolerance,
        )
        unrelaxed.calc = calculator
        unrelaxed_eV = unrelaxed.get_potential_energy()
        write(facet_dir / "unrelaxed.extxyz", unrelaxed)

        relaxed = unrelaxed.copy()
        fixed_atoms, fixed_layers, fixed_depth_A = constrain_bottom_depth(
            relaxed,
            minimum_depth_A=reference_fixed_depth_A,
            layer_tolerance_A=args.layer_tolerance,
        )
        relaxed.calc = calculator
        optimize(
            relaxed,
            relaxed,
            fmax=args.fmax,
            max_steps=args.max_steps,
            directory=facet_dir,
        )
        relaxed_eV = relaxed.get_potential_energy()
        write(facet_dir / "relaxed.extxyz", relaxed)

        area_A2 = float(np.linalg.norm(np.cross(relaxed.cell[0], relaxed.cell[1])))
        cleave, relax, surface = paper_surface_energy(
            unrelaxed_slab_eV=unrelaxed_eV,
            relaxed_slab_eV=relaxed_eV,
            bulk_eV_per_atom=bulk_eV_per_atom,
            atom_count=len(relaxed),
            area_A2=area_A2,
        )
        if surface <= 0:
            raise RuntimeError(f"Non-positive surface energy for {hkl}: {surface} eV/A^2")
        row = {
            "h": hkl[0],
            "k": hkl[1],
            "l": hkl[2],
            "atoms": len(relaxed),
            "layers": len(surface_layers(unrelaxed, args.layer_tolerance)[1]),
            "slab_thickness_A": float(np.ptp(surface_heights(unrelaxed))),
            "fixed_atoms": fixed_atoms,
            "fixed_layers": fixed_layers,
            "fixed_depth_A": fixed_depth_A,
            "area_A2": area_A2,
            "bulk_eV_per_atom": bulk_eV_per_atom,
            "unrelaxed_slab_eV": unrelaxed_eV,
            "relaxed_slab_eV": relaxed_eV,
            "cleave_eV_A2": cleave,
            "relax_eV_A2": relax,
            "surface_eV_A2": surface,
            "surface_J_m2": surface * EV_A2_TO_J_M2,
        }
        append_csv_row(surface_csv, row)
        surface_rows.append(row)
        print(f"{hkl}: {surface * EV_A2_TO_J_M2:.6f} J/m^2", flush=True)

    size_rows = build_wulff_series(
        symbol=args.symbol,
        structure=args.structure,
        lattice_constant_A=lattice_constant_A,
        surfaces=list(args.surfaces),
        surface_energies_J_m2=[float(row["surface_J_m2"]) for row in surface_rows],
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        size_step=args.size_step,
        calculator=calculator,
        trajectory_path=output_dir / "wulff_clusters.traj",
        csv_path=output_dir / "wulff_sizes.csv",
    )
    for row in size_rows:
        print(
            f"Wulff {row['actual_atom_count']} atoms: "
            f"{row['mace_energy_eV']:.8f} eV "
            f"({row['mace_energy_eV_per_atom']:.8f} eV/atom)",
            flush=True,
        )
    print(f"Wrote {len(size_rows)} unique Wulff nanoparticles to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
