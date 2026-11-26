"""Tools for constructing constrained bond scans and approximate MEP estimates.

The module provides a workflow that prepares endpoint structures, populates
intermediate images at incrementally constrained bond lengths, smooths the path
with IDPP interpolation, and adaptively inserts additional images to reach a
requested energy sampling density.  Dynamic mode relaxes the product geometry
without an explicit bond constraint and back-fills the target distance from the
relaxed result, which is useful when the final separation is unknown a priori.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read
from ase.mep.neb import NEB
from ase.optimize import BFGS
from ase.visualize import view
from mace.calculators import mace_mp

WORKDIR = Path(__file__).resolve().parent
OUTPUT_PATH = WORKDIR / "orca.out"
MODEL_PATH = WORKDIR / "MACE-matpes-r2scan-omat-ft.model"
DEFAULT_FIXED_SYMBOLS: Tuple[str, ...] = ("Au",)
DEFAULT_BOND_INDICES: Tuple[int, int] = (63, 64)
DEFAULT_DISTANCE_FIX = 0.5


def _log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message)


def _resolve_model(model: str | Path | None) -> str:
    """Return a valid model identifier for MACE."""
    if model is None:
        return "small"
    model_path = Path(model)
    if model_path.exists():
        return str(model_path)
    return str(model)


MACE_PARAMS = {
    "model": _resolve_model(MODEL_PATH if MODEL_PATH.exists() else None),
    "default_dtype": "float64",
    "device": os.environ.get("MACE_DEVICE", "cpu"),
}


def make_calculator():
    """Instantiate a new MACE calculator with the configured parameters."""
    return mace_mp(**MACE_PARAMS)


def apply_constraints(
    atoms: Atoms,
    *,
    bond_indices: Tuple[int, int] | None = None,
    fixed_symbols: Sequence[str] = DEFAULT_FIXED_SYMBOLS,
) -> None:
    """Apply atom-fixing and optional bond-length constraints in-place."""
    fixed_set = set(fixed_symbols)
    constraints: List[Any] = [
        FixAtoms([atom.index for atom in atoms if atom.symbol in fixed_set])
    ]
    if bond_indices is not None:
        constraints.append(FixBondLength(*bond_indices))
    atoms.set_constraint(constraints)


def ensure_calculators(images: Iterable[Atoms]) -> None:
    """Attach calculators to any images that are missing one."""
    for atoms in images:
        if atoms.calc is None:
            atoms.calc = make_calculator()


def relax_structure(
    structure: Atoms,
    *,
    fmax: float,
    bond_indices: Tuple[int, int] | None = None,
    fixed_symbols: Sequence[str] = DEFAULT_FIXED_SYMBOLS,
    verbose: bool = False,
    label: str | None = None,
) -> Atoms:
    """Return a constrained relaxation of *structure* using BFGS."""
    atoms = structure.copy()
    apply_constraints(atoms, bond_indices=bond_indices, fixed_symbols=fixed_symbols)
    atoms.calc = make_calculator()
    name = label or "structure"
    _log(f"Relaxing {name} with fmax={fmax}", verbose=verbose)
    opt = BFGS(atoms, logfile="-")
    opt.run(fmax=fmax)
    steps = getattr(opt, "get_number_of_steps", lambda: None)()
    if steps is not None:
        _log(f"Converged {name} in {steps} BFGS steps", verbose=verbose)
    return atoms


def _bond_distance(atoms: Atoms, bond_indices: Tuple[int, int]) -> float:
    """Return the distance between the selected atom indices."""
    return float(atoms.get_distance(*bond_indices))


def _coerce_atoms(value: Atoms | str | Path, *, label: str) -> Atoms:
    """Return a defensive copy of an Atoms object resolved from *value*."""
    if isinstance(value, Atoms):
        return value.copy()
    if isinstance(value, (str, Path)):
        resolved = read(value)
        if not isinstance(resolved, Atoms):
            raise TypeError(f"Resolved {label} is not an ASE Atoms object")
        return resolved.copy()
    raise TypeError(f"Unsupported type for {label}: {type(value)!r}")


def _collect_fixed_indices(atoms: Atoms, fixed_symbols: Sequence[str]) -> np.ndarray:
    """Return indices of atoms constrained via FixAtoms."""
    fixed_set = set(fixed_symbols)
    return np.array([atom.index for atom in atoms if atom.symbol in fixed_set], dtype=int)


def _distance_fraction(distance: float, start: float, end: float) -> float:
    """Return normalised progress between start and end bond distances."""
    span = end - start
    if abs(span) < 1e-8:
        return 0.0
    return float(np.clip((distance - start) / span, 0.0, 1.0))


def _interpolate_fixed_positions(
    atoms: Atoms,
    *,
    fixed_indices: np.ndarray,
    initial_fixed_positions: np.ndarray,
    final_fixed_positions: np.ndarray,
    fraction: float,
) -> None:
    """Overwrite fixed-atom coordinates with an interpolated position."""
    if fixed_indices.size == 0:
        return
    fraction = float(np.clip(fraction, 0.0, 1.0))
    positions = atoms.get_positions()
    constraints = atoms.constraints
    atoms.set_constraint()
    positions[fixed_indices] = initial_fixed_positions + fraction * (
        final_fixed_positions - initial_fixed_positions
    )
    atoms.set_positions(positions)
    atoms.set_constraint(constraints)


def prepare_endpoints(
    raw_atoms: Atoms,
    *,
    target_distance: float | None,
    dynamic_target: bool,
    fmax: float,
    bond_indices: Tuple[int, int],
    fixed_symbols: Sequence[str],
    verbose: bool,
) -> Tuple[Atoms, Atoms, float]:
    """Relax the initial and final structures, optionally resolving the target distance dynamically."""
    initial = relax_structure(
        raw_atoms,
        fmax=fmax,
        bond_indices=None,
        fixed_symbols=fixed_symbols,
        verbose=verbose,
        label="initial",
    )
    initial_distance = _bond_distance(initial, bond_indices)
    _log(f"Initial bond distance: {initial_distance:.3f} A", verbose=verbose)

    resolved_distance = target_distance or initial_distance
    final_seed = raw_atoms.copy()

    if dynamic_target:
        guess_distance = target_distance or initial_distance
        _log(
            f"Dynamic target enabled; relaxing unconstrained product from guess {guess_distance:.3f} A",
            verbose=verbose,
        )
        if guess_distance is not None:
            final_seed.set_distance(*bond_indices, guess_distance, fix=DEFAULT_DISTANCE_FIX)
        unconstrained_final = relax_structure(
            final_seed,
            fmax=fmax,
            bond_indices=None,
            fixed_symbols=fixed_symbols,
            verbose=verbose,
            label="dynamic final (unconstrained)",
        )
        resolved_distance = _bond_distance(unconstrained_final, bond_indices)
        _log(
            f"Resolved bond distance from dynamic relaxation: {resolved_distance:.3f} A",
            verbose=verbose,
        )
        final = unconstrained_final
    else:
        if resolved_distance is None:
            resolved_distance = initial_distance
            _log(
                f"No target distance provided; defaulting to initial distance {resolved_distance:.3f} A",
                verbose=verbose,
            )
        final_seed.set_distance(*bond_indices, resolved_distance, fix=DEFAULT_DISTANCE_FIX)
        final = relax_structure(
            final_seed,
            fmax=fmax,
            bond_indices=bond_indices,
            fixed_symbols=fixed_symbols,
            verbose=verbose,
            label="final (constrained)",
        )

    _log(
        f"Final endpoint prepared with bond distance { _bond_distance(final, bond_indices):.3f} A",
        verbose=verbose,
    )
    return initial, final, resolved_distance


def _constrained_image(
    reference: Atoms,
    *,
    distance: float,
    fmax: float,
    bond_indices: Tuple[int, int],
    fixed_symbols: Sequence[str],
    verbose: bool,
    fixed_indices: np.ndarray,
    initial_fixed_positions: np.ndarray,
    final_fixed_positions: np.ndarray,
    fraction: float,
) -> Atoms:
    """Relax a new image at a specified bond distance starting from *reference*."""
    guess = reference.copy()
    _interpolate_fixed_positions(
        guess,
        fixed_indices=fixed_indices,
        initial_fixed_positions=initial_fixed_positions,
        final_fixed_positions=final_fixed_positions,
        fraction=fraction,
    )
    guess.set_distance(*bond_indices, distance, fix=DEFAULT_DISTANCE_FIX)
    return relax_structure(
        guess,
        fmax=fmax,
        bond_indices=bond_indices,
        fixed_symbols=fixed_symbols,
        verbose=verbose,
        label="intermediate",
    )


def build_constrained_band(
    initial: Atoms,
    final: Atoms,
    *,
    n_images: int,
    fmax: float,
    bond_indices: Tuple[int, int],
    fixed_symbols: Sequence[str],
    verbose: bool,
) -> List[Atoms]:
    """Populate intermediate images by incrementally relaxing constrained scans."""
    if n_images < 2:
        raise ValueError("n_images must be at least 2 to include endpoints")

    start_dist = _bond_distance(initial, bond_indices)
    end_dist = _bond_distance(final, bond_indices)
    _log(
        f"Building band with {n_images} images from {start_dist:.3f} A to {end_dist:.3f} A",
        verbose=verbose,
    )

    fixed_indices = _collect_fixed_indices(initial, fixed_symbols)
    initial_fixed_positions = initial.get_positions()[fixed_indices]
    final_fixed_positions = final.get_positions()[fixed_indices]

    distances = np.linspace(start_dist, end_dist, n_images)
    images: List[Atoms] = [initial.copy()]
    current = initial
    for distance in distances[1:-1]:
        fraction = _distance_fraction(distance, start_dist, end_dist)
        current = _constrained_image(
            current,
            distance=distance,
            fmax=fmax,
            bond_indices=bond_indices,
            fixed_symbols=fixed_symbols,
            verbose=verbose,
            fixed_indices=fixed_indices,
            initial_fixed_positions=initial_fixed_positions,
            final_fixed_positions=final_fixed_positions,
            fraction=fraction,
        )
        images.append(current)
    images.append(final.copy())
    return images


def smooth_with_idpp(images: Sequence[Atoms], *, verbose: bool) -> None:
    """Smooth the band in-place using IDPP interpolation."""
    _log("Applying IDPP interpolation to band", verbose=verbose)
    neb = NEB(images)
    neb.interpolate(method="idpp", apply_constraint=True)


def energy_profile(images: Sequence[Atoms]) -> List[float]:
    """Compute potential energies for all images in the path."""
    return [float(image.get_potential_energy()) for image in images]


def _midpoint_image(
    left: Atoms,
    right: Atoms,
    *,
    distance: float,
    fmax: float,
    bond_indices: Tuple[int, int],
    fixed_symbols: Sequence[str],
    verbose: bool,
    fixed_indices: np.ndarray,
    initial_fixed_positions: np.ndarray,
    final_fixed_positions: np.ndarray,
    fraction: float,
) -> Atoms:
    """Construct and relax a midpoint guess between two images."""
    guess = left.copy()
    midpoint_positions = 0.5 * (left.get_positions() + right.get_positions())
    guess.set_positions(midpoint_positions)
    _interpolate_fixed_positions(
        guess,
        fixed_indices=fixed_indices,
        initial_fixed_positions=initial_fixed_positions,
        final_fixed_positions=final_fixed_positions,
        fraction=fraction,
    )
    guess.set_distance(*bond_indices, distance, fix=DEFAULT_DISTANCE_FIX)
    return relax_structure(
        guess,
        fmax=fmax,
        bond_indices=bond_indices,
        fixed_symbols=fixed_symbols,
        verbose=verbose,
        label="refined midpoint",
    )


def refine_by_energy_density(
    images: List[Atoms],
    *,
    target_spacing: float,
    fmax: float,
    max_images: int,
    bond_indices: Tuple[int, int],
    fixed_symbols: Sequence[str],
    verbose: bool,
) -> List[float]:
    """Insert midpoint images until the energy spacing meets the requested density."""
    ensure_calculators(images)
    fixed_indices = _collect_fixed_indices(images[0], fixed_symbols)
    initial_fixed_positions = images[0].get_positions()[fixed_indices]
    final_fixed_positions = images[-1].get_positions()[fixed_indices]
    start_distance = _bond_distance(images[0], bond_indices)
    end_distance = _bond_distance(images[-1], bond_indices)
    while len(images) < max_images:
        energies = energy_profile(images)
        if len(energies) < 2:
            _log("Insufficient images to evaluate profile; returning early", verbose=verbose)
            return energies

        gaps = np.abs(np.diff(energies))
        peak_idx = int(np.argmax(energies))
        eligible = [i for i, gap in enumerate(gaps) if gap > target_spacing]
        if not eligible:
            _log("Energy spacing satisfied; no further insertions needed", verbose=verbose)
            return energies

        scores = [
            (min(abs(peak_idx - i), abs(peak_idx - (i + 1))), -gaps[i], i)
            for i in eligible
        ]
        _, _, idx = min(scores)
        _log(
            f"Inserting midpoint between images {idx} and {idx + 1} (gap {gaps[idx]:.3f} eV)",
            verbose=verbose,
        )

        left = images[idx]
        right = images[idx + 1]
        new_distance = 0.5 * (
            _bond_distance(left, bond_indices) + _bond_distance(right, bond_indices)
        )
        fraction = _distance_fraction(new_distance, start_distance, end_distance)
        images.insert(
            idx + 1,
            _midpoint_image(
                left,
                right,
                distance=new_distance,
                fmax=fmax,
                bond_indices=bond_indices,
                fixed_symbols=fixed_symbols,
                verbose=verbose,
                fixed_indices=fixed_indices,
                initial_fixed_positions=initial_fixed_positions,
                final_fixed_positions=final_fixed_positions,
                fraction=fraction,
            ),
        )
        smooth_with_idpp(images, verbose=verbose)
        midpoint = images[idx + 1]
        _interpolate_fixed_positions(
            midpoint,
            fixed_indices=fixed_indices,
            initial_fixed_positions=initial_fixed_positions,
            final_fixed_positions=final_fixed_positions,
            fraction=fraction,
        )
        midpoint.set_distance(*bond_indices, new_distance, fix=DEFAULT_DISTANCE_FIX)
        images[idx + 1] = relax_structure(
            midpoint,
            fmax=fmax,
            bond_indices=bond_indices,
            fixed_symbols=fixed_symbols,
            verbose=verbose,
            label="refined midpoint (post-idpp)",
        )
        ensure_calculators(images)

    _log("Reached maximum images; returning current energy profile", verbose=verbose)
    return energy_profile(images)


def estimate_mep(
    output_path: Path = OUTPUT_PATH,
    *,
    target_distance: float | None = 6,
    dynamic_target: bool = False,
    initial_images: int = 7,
    relax_fmax: float = 0.05,
    target_energy_spacing: float = 0.1,
    max_images: int = 15,
    bond_indices: Tuple[int, int] = DEFAULT_BOND_INDICES,
    fixed_symbols: Sequence[str] = DEFAULT_FIXED_SYMBOLS,
    initial_structure: Atoms | str | Path | None = None,
    final_structure: Atoms | str | Path | None = None,
    images: List[Atoms] | None = None,
    verbose: bool = True,
) -> Tuple[List[Atoms], List[float]]:
    """Execute the constrained bond-length workflow and return images plus energies."""
    raw_atoms: Atoms | None = None
    if (
        images is None
        and initial_structure is None
        and final_structure is None
    ):
        _log("Loading reference geometry for MEP estimation", verbose=verbose)
        raw_atoms = read(output_path, format="orca-output")

    if images is None:
        if (initial_structure is None) != (final_structure is None):
            raise ValueError("Provide both initial_structure and final_structure together")
        if initial_structure is not None and final_structure is not None:
            initial = _coerce_atoms(initial_structure, label="initial_structure")
            final = _coerce_atoms(final_structure, label="final_structure")
            resolved_distance = _bond_distance(final, bond_indices)
            _log(
                f"Using supplied endpoints; final bond distance {resolved_distance:.3f} A",
                verbose=verbose,
            )
        else:
            if raw_atoms is None:
                raise RuntimeError("Failed to load reference geometry for endpoint preparation")
            initial, final, resolved_distance = prepare_endpoints(
                raw_atoms,
                target_distance=target_distance,
                dynamic_target=dynamic_target,
                fmax=relax_fmax,
                bond_indices=bond_indices,
                fixed_symbols=fixed_symbols,
                verbose=verbose,
            )
        images = build_constrained_band(
            initial,
            final,
            n_images=initial_images,
            fmax=relax_fmax,
            bond_indices=bond_indices,
            fixed_symbols=fixed_symbols,
            verbose=verbose,
        )
        ensure_calculators(images)
    else:
        if initial_structure is not None or final_structure is not None:
            _log("Ignoring supplied endpoints because explicit images were provided", verbose=verbose)
        _log("Using pre-supplied images; clearing old calculators", verbose=verbose)
        for atoms in images:
            atoms.calc = None
        resolved_distance = _bond_distance(images[-1], bond_indices)

    _log(
        f"Resolved target bond distance for band construction: {resolved_distance:.3f} A",
        verbose=verbose,
    )

    energies = refine_by_energy_density(
        images,
        target_spacing=target_energy_spacing,
        fmax=relax_fmax,
        max_images=max_images,
        bond_indices=bond_indices,
        fixed_symbols=fixed_symbols,
        verbose=verbose,
    )
    _log(f"Finished MEP estimate with {len(images)} images", verbose=verbose)
    return images, energies


if __name__ == "__main__":

    initial = read(r"<research-data-root>\2025\9.September\FBL_SHORTER\initial.traj")
    final = read(r"<research-data-root>\2025\9.September\FBL_SHORTER\final.traj")

    band, energies = estimate_mep(initial_structure=initial,
                                  final_structure=final,
                                  # target_distance=2.5, 
                                  max_images=5, 
                                  dynamic_target=False,
                                  fixed_symbols=("Au",)
                                )
    print("Computed constrained path with", len(band), "images")
    for idx, energy in enumerate(energies):
        print(f"Image {idx:02d}: {energy:.6f} eV")

    from ase.io.trajectory import Trajectory

    writer = Trajectory("mep.traj", "w")
    for atoms in band:
        atoms.get_forces()
        writer.write(atoms)

    view(band)