"""Local spatial overlap of signed NTO amplitude cubes; no HPC execution.

The atom map is explicit and fixed: corresponding Au atoms in source order.
This is a geometry-conditioned orbital comparison, not an overlap of many-body
states or of alpha and beta spin functions. No symmetry permutation maximization
or adsorbate deletion/projection is silently applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from ase.io.cube import read_cube
from ase.units import Bohr
from scipy.ndimage import map_coordinates


def read_amplitude(path):
    with Path(path).open(encoding="utf-8") as handle:
        cube = read_cube(handle)
    if len(cube["datas"]) != 1:
        raise ValueError("Expected one signed orbital amplitude per cube")
    data = cube["data"]
    if not np.isfinite(data).all():
        raise ValueError("Expected a finite NTO amplitude")
    return cube


def align_cluster(reference, candidate, candidate_cluster_order=None):
    """Align corresponding Cu/Au cluster atoms without fitting moving adsorbates."""
    ref_ids = np.flatnonzero(np.isin(reference["atoms"].numbers, (29, 79)))
    cand_ids = np.flatnonzero(np.isin(candidate["atoms"].numbers, (29, 79)))
    if len(ref_ids) < 3 or len(ref_ids) != len(cand_ids):
        raise ValueError("Cubes must contain the same Cu/Au cluster with at least three atoms")
    count = len(ref_ids)
    order = list(range(count)) if candidate_cluster_order is None else list(candidate_cluster_order)
    if sorted(order) != list(range(count)):
        raise ValueError(f"candidate_cluster_order must be a permutation of 0..{count - 1}")
    cand_ids = cand_ids[order]
    if not np.array_equal(
        reference["atoms"].numbers[ref_ids], candidate["atoms"].numbers[cand_ids]
    ):
        raise ValueError("Mapped cluster elements differ between cubes")
    ref = reference["atoms"].positions[ref_ids]
    cand = candidate["atoms"].positions[cand_ids]
    rc, cc = ref.mean(axis=0), cand.mean(axis=0)
    u, _, vt = np.linalg.svd((cand - cc).T @ (ref - rc))
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    translation = rc - cc @ rotation
    rmsd = float(np.sqrt(np.mean(np.sum((cand @ rotation + translation - ref) ** 2, axis=1))))
    return rotation, translation, rmsd, ref_ids.tolist(), cand_ids.tolist()


def norm(cube):
    # Cube amplitudes are atomic units; ASE converts coordinates to angstrom.
    volume = abs(np.linalg.det(cube["spacing"])) / Bohr ** 3
    value = float(np.sum(cube["data"] ** 2) * volume)
    if value <= 0 or not np.isfinite(value):
        raise ValueError("Invalid orbital norm or grid spacing")
    return value


def integrate_on_grid(grid, other, rotation, translation):
    """Transform grid points into other's coordinates: x_other = x_grid R + t."""
    data = grid["data"]
    inverse_spacing = np.linalg.inv(other["spacing"])
    inner, resampled_norm = 0.0, 0.0
    yz = np.indices(data.shape[1:]).reshape(2, -1).T
    # ponytail: linear interpolation, finer retained cubes if convergence fails.
    for i in range(data.shape[0]):
        indices = np.column_stack((np.full(len(yz), i), yz))
        xyz = grid["origin"] + indices @ grid["spacing"]
        coords = ((xyz @ rotation + translation - other["origin"]) @ inverse_spacing).T
        # Roundoff from rigid transforms must not discard valid boundary nodes.
        for axis, size in enumerate(other["data"].shape):
            coords[axis, np.abs(coords[axis]) < 1e-8] = 0.0
            coords[axis, np.abs(coords[axis] - (size - 1)) < 1e-8] = size - 1
        values = map_coordinates(other["data"], coords, order=1, mode="constant", cval=0.0, prefilter=False)
        inner += float(np.dot(data[i].ravel(), values))
        resampled_norm += float(np.dot(values, values))
    volume = abs(np.linalg.det(grid["spacing"])) / Bohr ** 3
    return inner * volume, resampled_norm * volume


def compare(reference, candidate, candidate_cluster_order=None):
    rotation, translation, rmsd, ref_ids, cand_ids = align_cluster(
        reference, candidate, candidate_cluster_order
    )
    nr, nc = norm(reference), norm(candidate)
    # candidate -> reference: x_ref = x_candidate R + t
    ir, sampled_c = integrate_on_grid(reference, candidate, rotation.T, -translation @ rotation.T)
    ic, sampled_r = integrate_on_grid(candidate, reference, rotation, translation)
    sr, sc = ir / np.sqrt(nr * nc), ic / np.sqrt(nr * nc)
    flags = []
    if abs(sr - sc) > 0.005:
        flags.append("Overlap is grid-sensitive: obtain finer cubes before interpretation")
    if abs(sampled_c / nc - 1) > 0.02 or abs(sampled_r / nr - 1) > 0.02:
        flags.append("Resampled norm differs by >2%: check grid coverage/interpolation")
    if abs(nr - 1) > 0.02 or abs(nc - 1) > 0.02:
        flags.append("Native orbital norm differs from 1 by >2%: check cube coverage/units")
    if max(abs(sr), abs(sc)) > 1.005:
        flags.append("Numerical overlap exceeds unity: refine integration")
    if rmsd > 0.3:
        flags.append("Cluster alignment RMSD exceeds 0.3 angstrom: verify atom map/distortion")
    return {
        "status": "needs_grid_or_mapping_check" if flags else "numerical_checks_passed",
        "cluster_rmsd_A": rmsd,
        "au_rmsd_A": rmsd,  # Legacy key retained for existing Au20 result readers.
        "reference_cluster_indices_0based": ref_ids,
        "candidate_cluster_indices_0based": cand_ids,
        "candidate_to_reference_rotation": rotation.tolist(),
        "candidate_to_reference_translation_A": translation.tolist(),
        "reference_native_norm": nr, "candidate_native_norm": nc,
        "candidate_resampled_norm_ratio": sampled_c / nc,
        "reference_resampled_norm_ratio": sampled_r / nr,
        "signed_overlap_reference_grid": float(sr),
        "signed_overlap_candidate_grid": float(sc),
        "squared_overlap_reference_grid": float(sr ** 2),
        "squared_overlap_candidate_grid": float(sc ** 2),
        "squared_overlap_mean": float(((sr + sc) / 2) ** 2),
        "warnings": flags,
        "method": "full-space signed amplitude; fixed Cu/Au cluster map; proper Kabsch rotation; linear grid interpolation; native-norm normalization",
        "caveat": "Agreement between the two grids is a diagnostic, not a substitute for independent finer-grid convergence",
    }


def compare_nto_pair(reference_hole, reference_particle, candidate_hole, candidate_particle, candidate_cluster_order=None):
    """Score continuity only when both dominant hole and particle NTOs persist."""
    hole = compare(reference_hole, candidate_hole, candidate_cluster_order)
    particle = compare(reference_particle, candidate_particle, candidate_cluster_order)
    combined = float(np.sqrt(hole["squared_overlap_mean"] * particle["squared_overlap_mean"]))
    return {"paired_overlap": combined, "hole": hole, "particle": particle}


def weighted_nto_density(pairs, role):
    """Build phase-insensitive attachment/detachment density from weighted NTOs."""

    cubes = [(float(pair["weight"]), pair[role]) for pair in pairs]
    if not cubes or sum(weight for weight, _cube in cubes) <= 0:
        raise ValueError("At least one positively weighted NTO pair is required")
    reference = cubes[0][1]
    density = np.zeros_like(reference["data"])
    for weight, cube in cubes:
        if cube["data"].shape != density.shape or not np.allclose(cube["origin"], reference["origin"]) or not np.allclose(cube["spacing"], reference["spacing"]):
            raise ValueError("NTOs from one state must share a cube grid")
        density += weight * cube["data"] ** 2
    return {**reference, "data": density / sum(weight for weight, _cube in cubes)}


def compare_nto_density_sets(reference_pairs, candidate_pairs, candidate_cluster_order=None):
    """Compare weighted hole/electron densities for truncated multi-pair NTO states."""

    hole = compare(weighted_nto_density(reference_pairs, "hole"), weighted_nto_density(candidate_pairs, "hole"), candidate_cluster_order)
    particle = compare(weighted_nto_density(reference_pairs, "particle"), weighted_nto_density(candidate_pairs, "particle"), candidate_cluster_order)
    combined = float(np.sqrt(hole["squared_overlap_mean"] * particle["squared_overlap_mean"]))
    return {"paired_density_overlap": combined, "hole_density": hole, "particle_density": particle}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    base = args.manifest.resolve().parent
    entries = [manifest["reference"], *manifest["candidates"]]
    if any(entry.get("field_kind") != "orbital_amplitude" for entry in entries):
        parser.error("Each input must explicitly identify field_kind=orbital_amplitude; densities are not suitable")
    paths = [(base / entry["cube"]).resolve() for entry in entries]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        parser.error("Missing orbital amplitude inputs; no result written: " + ", ".join(missing))
    reference = read_amplitude(paths[0])
    results = [
        {
            "candidate": entry,
            **compare(
                reference,
                read_amplitude(path),
                manifest.get("candidate_cluster_order_0based", manifest.get("candidate_au_order_0based")),
            ),
        }
        for entry, path in zip(entries[1:], paths[1:])
    ]
    output = {
        "input_manifest": manifest,
        "input_checksums_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
