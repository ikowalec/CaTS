import os
import concurrent.futures
from dataclasses import dataclass
from itertools import combinations, permutations, product
from math import comb
from pathlib import Path
from typing import Sequence

import numpy as np
from acat.adsorption_sites import ClusterAdsorptionSites, SlabAdsorptionSites
from acat.build import add_adsorbate_to_site
from acat.settings import CustomSurface
from acat.utilities import atoms_too_close, atoms_too_close_after_addition
from ase.atoms import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.db import connect
from ase.geometry import find_mic
from ase.io import read
from carmm.analyse.bonds import search_abnormal_bonds


DEFAULT_PASSIVE_SYMBOLS = ("Ce", "O")
DEFAULT_Z_REFERENCE_SYMBOL = "Ti"
DEFAULT_SLAB_VACUUM_THRESHOLD = 4.0
FAST_CLASH_CUTOFF = 0.5
TRIANGLE_NEIGHBOR_CUTOFF = 3.2
CLUSTER_INTEGRITY_BOND_SCALE = 1.35
COADSORBATE_SPACING_STEP = 0.25
COADSORBATE_MAX_SPACING = 3.0
COADSORBATE_NONBOND_SCALE = 1.10


AdsorbateLike = str | Atoms


@dataclass(frozen=True)
class AdsorbateSpec:
    label: str
    key: str
    payload: AdsorbateLike
    rotation_z_degrees: float = 0.0
    flipped: bool = False


def _atoms_signature(atoms: Atoms) -> str:
    parts: list[str] = []
    for atom in atoms:
        x, y, z = atom.position
        parts.append(f"{atom.symbol}:{x:.6f}:{y:.6f}:{z:.6f}")
    return "|".join(parts)


def _normalize_adsorbates(adsorbates: Sequence[AdsorbateLike]) -> list[AdsorbateSpec]:
    specs: list[AdsorbateSpec] = []
    for idx, adsorbate in enumerate(adsorbates):
        if isinstance(adsorbate, str):
            label = adsorbate.strip()
            if not label:
                continue
            specs.append(
                AdsorbateSpec(
                    label=label,
                    key=f"str:{label}",
                    payload=label,
                )
            )
            continue
        if isinstance(adsorbate, Atoms):
            if len(adsorbate) == 0:
                raise ValueError("Empty ASE Atoms adsorbate is not valid.")
            label = str(
                adsorbate.info.get("label") or adsorbate.get_chemical_formula(mode="hill")
            )
            key = f"atoms:{_atoms_signature(adsorbate)}"
            specs.append(
                AdsorbateSpec(
                    label=label if label else f"atoms_{idx}",
                    key=key,
                    payload=adsorbate,
                )
            )
            continue
        raise TypeError(
            "Adsorbates must be strings or ASE Atoms objects, "
            f"got {type(adsorbate).__name__}."
        )
    return specs


def _unique_adsorbate_orders(adsorbates: Sequence[AdsorbateSpec]) -> list[tuple[AdsorbateSpec, ...]]:
    # Deduplicate by adsorbate identity signature so repeated species do not
    # generate duplicate permutations.
    seen: set[tuple[str, ...]] = set()
    unique_orders: list[tuple[AdsorbateSpec, ...]] = []
    for order in permutations(range(len(adsorbates))):
        key = tuple(adsorbates[i].key for i in order)
        if key in seen:
            continue
        seen.add(key)
        unique_orders.append(tuple(adsorbates[i] for i in order))
    return unique_orders


def _adsorbate_orientations(
    adsorbate: AdsorbateSpec,
    *,
    rotation_step_degrees: float,
    include_flip: bool,
) -> list[AdsorbateSpec]:
    if not isinstance(adsorbate.payload, Atoms):
        return [adsorbate]
    if len(adsorbate.payload) == 1:
        return [adsorbate]
    if not 0.0 < float(rotation_step_degrees) <= 360.0:
        raise ValueError("rotation_step_degrees must be in the range (0, 360].")

    variants = []
    anchor = adsorbate.payload.positions[0].copy()
    for flipped in range(2 if include_flip else 1):
        for rotation in np.arange(0.0, 360.0, float(rotation_step_degrees)):
            payload = adsorbate.payload.copy()
            if flipped:
                payload.rotate(180.0, "x", center=anchor)
            payload.rotate(float(rotation), "z", center=anchor)
            variants.append(
                AdsorbateSpec(
                    label=adsorbate.label,
                    key=adsorbate.key,
                    payload=payload,
                    rotation_z_degrees=float(rotation),
                    flipped=bool(flipped),
                )
            )
    return variants


def _build_host_and_passive(
    atoms: Atoms,
    *,
    site_symbols: Sequence[str] | None,
    passive_symbols: Sequence[str],
) -> tuple[Atoms, Atoms | None]:
    site_symbol_set = {sym for sym in (site_symbols or ()) if str(sym).strip()}
    passive_symbol_set = {sym for sym in passive_symbols if str(sym).strip()}

    if site_symbol_set:
        host_indices = [atom.index for atom in atoms if atom.symbol in site_symbol_set]
        passive_indices = [atom.index for atom in atoms if atom.symbol not in site_symbol_set]
    else:
        host_indices = [atom.index for atom in atoms if atom.symbol not in passive_symbol_set]
        passive_indices = [atom.index for atom in atoms if atom.symbol in passive_symbol_set]

    if not host_indices:
        raise ValueError("No host atoms found for adsorption-site detection.")

    host = atoms[host_indices]
    passive = atoms[passive_indices] if passive_indices else None
    return host, passive


def _build_cluster_host_and_passive(
    atoms: Atoms,
    *,
    cluster_symbols: Sequence[str] | None,
) -> tuple[Atoms, Atoms | None, bool]:
    cluster_symbol_set = {str(sym).strip() for sym in (cluster_symbols or ()) if str(sym).strip()}
    if not cluster_symbol_set:
        return atoms, None, False

    # ponytail: symbol split is enough for metal clusters on oxide supports; use tags/indices if support and cluster share metals.
    cluster_indices = [atom.index for atom in atoms if atom.symbol in cluster_symbol_set]
    if not cluster_indices:
        return atoms, None, False

    passive_indices = [atom.index for atom in atoms if atom.index not in cluster_indices]
    return atoms[cluster_indices], atoms[passive_indices] if passive_indices else None, True


def _is_slab_like(atoms: Atoms) -> bool:
    return bool(np.any(atoms.get_pbc()))


def _resolve_z_cutoff(
    host_atoms: Atoms,
    *,
    z_cutoff: float | None,
    z_reference_symbol: str | None,
    z_buffer: float | None,
) -> float | None:
    if z_cutoff is not None:
        return float(z_cutoff)

    if z_buffer is None:
        return None

    if z_buffer < 0:
        return None

    if z_reference_symbol:
        z_values = [atom.z for atom in host_atoms if atom.symbol == z_reference_symbol]
        if not z_values:
            # Be permissive across chemistries: if the requested reference species
            # is absent, compute the cutoff from all host atoms instead of failing.
            z_values = [atom.z for atom in host_atoms]
    else:
        z_values = [atom.z for atom in host_atoms]

    if not z_values:
        return None
    return float(max(z_values) + z_buffer)


def _cluster_sites(
    host_atoms: Atoms,
    *,
    allow_6fold: bool,
    composition_effect: bool,
    surrogate_metal: str,
):
    cas = ClusterAdsorptionSites(
        host_atoms,
        allow_6fold=allow_6fold,
        composition_effect=composition_effect,
        label_sites=False,
        surrogate_metal=surrogate_metal,
    )
    return cas.get_sites()


def _slab_sites(
    host_atoms: Atoms,
    *,
    composition_effect: bool,
    top_z_tolerance: float,
):
    slab_surface = CustomSurface(host_atoms)
    try:
        sas = SlabAdsorptionSites(
            host_atoms,
            surface=slab_surface,
            composition_effect=composition_effect,
            label_sites=False,
        )
        sites = sas.get_sites()
        mode = "slab"
    except KeyError as exc:
        print(
            f"Warning: SlabAdsorptionSites labeling failed ({exc}); "
            "retrying with composition_effect=False."
        )
        sas = SlabAdsorptionSites(
            host_atoms,
            surface=slab_surface,
            composition_effect=False,
            label_sites=False,
        )
        sites = sas.get_sites()
        mode = "slab-fallback-no-composition"
    top_z = max(float(atom.z) for atom in host_atoms)
    return [
        site for site in sites
        if float(np.asarray(site["position"])[2]) >= top_z - float(top_z_tolerance)
    ], mode


def _fallback_active_indices(
    host_atoms: Atoms,
    *,
    top_surface_only: bool,
    top_z_tolerance: float,
) -> list[int]:
    active_indices = list(range(len(host_atoms)))
    if not top_surface_only or not active_indices:
        return active_indices
    carbon_z = [atom.z for atom in host_atoms if atom.symbol == "C"]
    z_reference = (
        max(carbon_z)
        if len(carbon_z) > len(host_atoms) / 2
        else max(float(atom.z) for atom in host_atoms)
    )
    return [i for i in active_indices if host_atoms[i].z >= z_reference - float(top_z_tolerance)]


def _hollow_com_sites(
    host_atoms: Atoms,
    *,
    neighbor_cutoff: float = TRIANGLE_NEIGHBOR_CUTOFF,
    max_sites: int | None = None,
    top_surface_only: bool = False,
    top_z_tolerance: float = 1.5,
) -> list[dict[str, object]]:
    positions = host_atoms.get_positions()
    active_indices = _fallback_active_indices(
        host_atoms,
        top_surface_only=top_surface_only,
        top_z_tolerance=top_z_tolerance,
    )
    if len(active_indices) < 3:
        return []

    active_positions = positions[active_indices]
    com = active_positions.mean(axis=0)
    vectors = active_positions[:, None, :] - active_positions[None, :, :]
    mic = bool(np.any(host_atoms.get_pbc()))
    if mic:
        _, distances = find_mic(
            vectors.reshape(-1, 3), host_atoms.cell, host_atoms.get_pbc()
        )
        distances = distances.reshape(vectors.shape[:2])
    else:
        distances = np.linalg.norm(vectors, axis=2)
    distances[distances < 1.0e-8] = np.inf
    cutoff = float(neighbor_cutoff)
    coordination = (distances < cutoff).sum(axis=1)
    radial_vectors = active_positions - com
    if mic:
        radial_vectors = find_mic(
            radial_vectors, host_atoms.cell, host_atoms.get_pbc()
        )[0]
    radial_distance = np.linalg.norm(radial_vectors, axis=1)
    order = sorted(range(len(active_indices)), key=lambda i: (coordination[i], -radial_distance[i]))

    sites: list[dict[str, object]] = []
    seen: set[tuple[int, int, int]] = set()
    for i in order:
        neighbors = [int(j) for j in np.argsort(distances[i])[:6] if distances[i, j] <= cutoff]
        for j, k in combinations(neighbors, 2):
            if distances[j, k] > cutoff:
                continue
            point_indices = tuple(active_indices[n] for n in (int(i), int(j), int(k)))
            indices = tuple(sorted(point_indices))
            if indices in seen:
                continue
            points = positions[list(point_indices)].copy()
            if mic:
                points[1:] = points[0] + find_mic(
                    points[1:] - points[0], host_atoms.cell, host_atoms.get_pbc()
                )[0]
            normal = np.cross(points[1] - points[0], points[2] - points[0])
            norm = float(np.linalg.norm(normal))
            if norm < 1.0e-8:
                continue
            position = points.mean(axis=0)
            normal = normal / norm
            radial = position - com
            if mic:
                radial = find_mic(radial, host_atoms.cell, host_atoms.get_pbc())[0]
            outward = float(np.dot(normal, radial))
            if (top_surface_only and normal[2] < 0.0) or (
                not top_surface_only
                and (outward < 0.0 or (abs(outward) < 1.0e-8 and normal[2] < 0.0))
            ):
                normal = -normal
            seen.add(indices)
            sites.append(
                {
                    "site": "fcc",
                    "position": position,
                    "normal": normal,
                    "indices": indices,
                    "connectivity": 3,
                }
            )
            break
        if max_sites is not None and len(sites) >= int(max_sites):
            break
    return sites


def _bridged_com_sites(
    host_atoms: Atoms,
    *,
    neighbor_cutoff: float = TRIANGLE_NEIGHBOR_CUTOFF,
    max_sites: int | None = None,
    top_surface_only: bool = False,
    top_z_tolerance: float = 1.5,
) -> list[dict[str, object]]:
    positions = host_atoms.get_positions()
    active_indices = _fallback_active_indices(
        host_atoms,
        top_surface_only=top_surface_only,
        top_z_tolerance=top_z_tolerance,
    )
    if len(active_indices) < 2:
        return []

    com = positions[active_indices].mean(axis=0)
    mic = bool(np.any(host_atoms.get_pbc()))
    sites = []
    for i, j in combinations(active_indices, 2):
        bond = positions[j] - positions[i]
        if mic:
            bond, distance = find_mic(bond, host_atoms.cell, host_atoms.get_pbc())
            distance = float(distance)
        else:
            distance = float(np.linalg.norm(bond))
        if distance < 1.0e-8 or distance > float(neighbor_cutoff):
            continue
        bond /= distance
        position = positions[i] + 0.5 * distance * bond
        normal = position - com
        if mic:
            normal = find_mic(normal, host_atoms.cell, host_atoms.get_pbc())[0]
        normal -= np.dot(normal, bond) * bond
        if np.linalg.norm(normal) < 1.0e-8:
            axis = np.eye(3)[int(np.argmin(np.abs(bond)))]
            normal = np.cross(bond, axis)
        normal /= np.linalg.norm(normal)
        if top_surface_only and normal[2] < 0.0:
            normal = -normal
        sites.append({
            "site": "bridge",
            "position": position,
            "normal": normal,
            "indices": (i, j),
            "connectivity": 2,
        })
        if max_sites is not None and len(sites) >= int(max_sites):
            break
    return sites


def _atop_com_sites(
    host_atoms: Atoms,
    *,
    top_surface_only: bool = False,
    top_z_tolerance: float = 1.5,
) -> list[dict[str, object]]:
    positions = host_atoms.get_positions()
    active_indices = _fallback_active_indices(
        host_atoms,
        top_surface_only=top_surface_only,
        top_z_tolerance=top_z_tolerance,
    )
    com = positions[active_indices].mean(axis=0) if active_indices else np.zeros(3)
    mic = bool(np.any(host_atoms.get_pbc()))
    sites = []
    for i in active_indices:
        normal = np.array([0.0, 0.0, 1.0]) if top_surface_only else positions[i] - com
        if mic and not top_surface_only:
            normal = find_mic(normal, host_atoms.cell, host_atoms.get_pbc())[0]
        if np.linalg.norm(normal) < 1.0e-8:
            normal = np.array([0.0, 0.0, 1.0])
        normal = normal / np.linalg.norm(normal)
        sites.append({
            "site": "ontop",
            "position": positions[i].copy(),
            "normal": normal,
            "indices": (i,),
            "connectivity": 1,
        })
    return sites


def _triangulated_com_sites(
    host_atoms: Atoms,
    *,
    neighbor_cutoff: float = TRIANGLE_NEIGHBOR_CUTOFF,
    max_sites: int | None = None,
    top_surface_only: bool = False,
    top_z_tolerance: float = 1.5,
) -> list[dict[str, object]]:
    groups = [
        _hollow_com_sites(
            host_atoms,
            neighbor_cutoff=neighbor_cutoff,
            top_surface_only=top_surface_only,
            top_z_tolerance=top_z_tolerance,
        ),
        _bridged_com_sites(
            host_atoms,
            neighbor_cutoff=neighbor_cutoff,
            top_surface_only=top_surface_only,
            top_z_tolerance=top_z_tolerance,
        ),
        _atop_com_sites(
            host_atoms,
            top_surface_only=top_surface_only,
            top_z_tolerance=top_z_tolerance,
        ),
    ]
    # Interleave types so a caller-imposed cap does not silently remove bridge or atop sites.
    sites = []
    for i in range(max(map(len, groups), default=0)):
        sites.extend(group[i] for group in groups if i < len(group))
    return sites if max_sites is None else sites[:int(max_sites)]


def _include_below_surface_sites(sites):
    below = []
    for site in sites:
        mirrored = dict(site)
        mirrored["normal"] = -np.asarray(site["normal"], dtype=float)
        mirrored["site"] = f"{site.get('site', '')}_below"
        below.append(mirrored)
    return [*sites, *below]


def _is_carbonaceous_periodic_support(atoms: Atoms) -> bool:
    return bool(np.any(atoms.get_pbc())) and sum(atom.symbol == "C" for atom in atoms) > len(atoms) / 2


def _passes_z_filter(candidate: Atoms, new_indices: Sequence[int], z_cutoff: float | None) -> bool:
    if z_cutoff is None:
        return True
    return all(candidate[idx].z >= z_cutoff for idx in new_indices)


def _sites_near_origin(
    sites,
    atoms: Atoms,
    origin_atom_index: int | None,
    site_cutoff: float | None,
):
    if origin_atom_index is None and site_cutoff is None:
        return list(sites)
    if origin_atom_index is None or site_cutoff is None:
        raise ValueError("origin_atom_index and site_cutoff must be provided together.")
    if not 0 <= int(origin_atom_index) < len(atoms):
        raise IndexError(
            f"origin_atom_index {origin_atom_index} is outside the 0-based atom range "
            f"0..{len(atoms) - 1}."
        )
    if float(site_cutoff) <= 0.0:
        raise ValueError("site_cutoff must be positive.")

    origin = atoms.positions[int(origin_atom_index)]
    nearby = []
    for site in sites:
        vector = np.asarray(site["position"], dtype=float) - origin
        if np.any(atoms.get_pbc()):
            vector, _ = find_mic(vector, atoms.cell, atoms.get_pbc())
        if float(np.linalg.norm(vector)) <= float(site_cutoff):
            nearby.append(site)
    return nearby


def _passes_bonding_filter(
    candidate: Atoms,
    *,
    skip_abnormal_bond_screening: bool,
    n_added: int | None = None,
) -> bool:
    # Fast geometric rejection of obvious overlaps, then chemistry-aware check.
    too_close = (
        atoms_too_close_after_addition(
            candidate, n_added=1, cutoff=float(FAST_CLASH_CUTOFF), mic=False
        )
        if n_added == 1
        else atoms_too_close(candidate, cutoff=float(FAST_CLASH_CUTOFF), mic=False)
    )
    if too_close:
        return False
    if skip_abnormal_bond_screening:
        return True
    return bool(search_abnormal_bonds(candidate, verbose=False))


def _adjust_contact_distance(
    atoms: Atoms,
    *,
    adsorbate_indices: Sequence[int],
    n_support_atoms: int,
    site,
    radius_scale: float,
) -> dict[str, object] | None:
    """Move an oriented adsorbate along the site normal to a covalent contact distance."""
    normal = np.asarray(site.get("normal"), dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm < 1.0e-12 or not adsorbate_indices or n_support_atoms <= 0:
        return None
    normal /= norm
    mic = bool(np.any(atoms.get_pbc()))
    contact = min(
        (
            atoms.get_distance(support_idx, adsorbate_idx, mic=mic),
            support_idx,
            adsorbate_idx,
        )
        for support_idx in range(int(n_support_atoms))
        for adsorbate_idx in adsorbate_indices
    )
    _, support_idx, adsorbate_idx = contact
    vector = atoms.positions[adsorbate_idx] - atoms.positions[support_idx]
    if mic:
        vector, _ = find_mic(vector, atoms.cell, atoms.get_pbc())
    target = float(radius_scale) * (
        _covalent_radius(atoms[support_idx].symbol)
        + _covalent_radius(atoms[adsorbate_idx].symbol)
    )
    parallel = float(np.dot(vector, normal))
    lateral_sq = float(np.dot(vector, vector) - parallel**2)
    if lateral_sq > target**2:
        return None
    shift = -parallel + float(np.sqrt(max(0.0, target**2 - lateral_sq)))
    atoms.positions[list(adsorbate_indices)] += shift * normal
    return {
        "support_index": support_idx,
        "support_symbol": atoms[support_idx].symbol,
        "adsorbate_index": adsorbate_idx,
        "adsorbate_symbol": atoms[adsorbate_idx].symbol,
        "target_distance": target,
        "normal_shift": shift,
    }


def combine_single_adsorbate_modes(
    a_modes: Sequence[Atoms],
    b_modes: Sequence[Atoms],
    *,
    n_support_atoms: int,
    max_structures: int | None = None,
    skip_abnormal_bond_screening: bool = False,
    allow_lateral_spacing_fallback: bool = True,
    verbose: bool = False,
) -> tuple[list[Atoms], list[dict[str, object]]]:
    """Combine independently generated A and B modes on their common support."""
    combined = []
    metadata = []
    total = len(a_modes) * len(b_modes)
    progress_step = max(1, total // 20)
    for a_row_id, a_mode in enumerate(a_modes, start=1):
        for b_row_id, b_mode in enumerate(b_modes, start=1):
            screened = (a_row_id - 1) * len(b_modes) + b_row_id
            if verbose and (
                screened == 1 or screened % progress_step == 0 or screened == total
            ):
                print(
                    "[Workflow] Co-adsorbate pair screening: "
                    f"{screened}/{total} ({100.0 * screened / total:.1f}%)",
                    flush=True,
                )
            candidate = a_mode.copy()
            candidate += b_mode[n_support_atoms:]
            if not _passes_bonding_filter(
                candidate,
                skip_abnormal_bond_screening=skip_abnormal_bond_screening,
            ):
                if not allow_lateral_spacing_fallback:
                    continue
                spaced = _space_coadsorbates_apart(
                    a_mode,
                    b_mode,
                    n_support_atoms=n_support_atoms,
                    skip_abnormal_bond_screening=skip_abnormal_bond_screening,
                )
                if spaced is None:
                    continue
                candidate, a_shift, b_shift = spaced
            else:
                a_shift = b_shift = None
            combined.append(candidate)
            row = {
                "a_mode_row_id": a_row_id,
                "b_mode_row_id": b_row_id,
                "a_adsorbate_n_atoms": len(a_mode) - int(n_support_atoms),
                "b_adsorbate_n_atoms": len(b_mode) - int(n_support_atoms),
                "generation": "sequential_single_mode_pairing",
            }
            if a_shift is not None:
                row["generation"] = "sequential_pairing_with_lateral_spacing"
                row["a_lateral_shift_A"] = ",".join(f"{value:g}" for value in a_shift)
                row["b_lateral_shift_A"] = ",".join(f"{value:g}" for value in b_shift)
            metadata.append(row)
            if max_structures is not None and len(combined) >= max_structures:
                return combined, metadata
    return combined, metadata


def _lateral_unit(vector, fallback):
    lateral = np.asarray(vector, dtype=float).copy()
    lateral[2] = 0.0
    norm = float(np.linalg.norm(lateral))
    if norm < 1.0e-12:
        return np.asarray(fallback, dtype=float)
    return lateral / norm


def _space_coadsorbates_apart(
    a_mode: Atoms,
    b_mode: Atoms,
    *,
    n_support_atoms: int,
    skip_abnormal_bond_screening: bool,
):
    a_adsorbate = a_mode[n_support_atoms:]
    b_adsorbate = b_mode[n_support_atoms:]
    unshifted = a_mode.copy()
    unshifted += b_adsorbate
    system_com = unshifted.get_center_of_mass()
    a_com = a_adsorbate.get_center_of_mass()
    b_com = b_adsorbate.get_center_of_mass()
    radial_a = _lateral_unit(a_com - system_com, (1.0, 0.0, 0.0))
    radial_b = _lateral_unit(b_com - system_com, (1.0, 0.0, 0.0))
    separation = np.asarray(a_com - b_com, dtype=float)
    separation[2] = 0.0

    if float(np.linalg.norm(separation)) < 1.0e-12:
        tangent = np.array([-radial_a[1], radial_a[0], 0.0])
        direction_a = _lateral_unit(radial_a + tangent, radial_a)
        direction_b = _lateral_unit(radial_b - tangent, radial_b)
    else:
        separation /= np.linalg.norm(separation)
        direction_a = _lateral_unit(radial_a + separation, radial_a)
        direction_b = _lateral_unit(radial_b - separation, radial_b)

    for distance in np.arange(
        COADSORBATE_SPACING_STEP,
        COADSORBATE_MAX_SPACING + COADSORBATE_SPACING_STEP / 2.0,
        COADSORBATE_SPACING_STEP,
    ):
        a_shift = direction_a * float(distance)
        b_shift = direction_b * float(distance)
        spaced_a = a_mode.copy()
        spaced_a.positions[n_support_atoms:] += a_shift
        spaced_b = b_adsorbate.copy()
        spaced_b.positions += b_shift
        candidate = spaced_a
        candidate += spaced_b
        a_indices = range(n_support_atoms, len(a_mode))
        b_indices = range(len(a_mode), len(candidate))
        mic = bool(np.any(candidate.get_pbc()))
        mutually_bonded = any(
            candidate.get_distance(i, j, mic=mic)
            <= COADSORBATE_NONBOND_SCALE
            * (_covalent_radius(candidate[i].symbol) + _covalent_radius(candidate[j].symbol))
            for i in a_indices
            for j in b_indices
        )
        if not mutually_bonded and _passes_bonding_filter(
            candidate,
            skip_abnormal_bond_screening=skip_abnormal_bond_screening,
        ):
            return candidate, a_shift, b_shift
    return None


def _covalent_radius(symbol: str) -> float:
    number = atomic_numbers.get(symbol)
    if number is None:
        return 1.5
    return float(covalent_radii[number])


def _passes_cluster_integrity(
    candidate: Atoms,
    new_indices: Sequence[int],
    cluster_indices: Sequence[int],
    *,
    bond_scale: float,
) -> bool:
    if not cluster_indices:
        return True
    mic = bool(np.any(candidate.get_pbc()))
    for new_idx in new_indices:
        new_radius = _covalent_radius(candidate[int(new_idx)].symbol)
        if not any(
            float(candidate.get_distance(int(new_idx), int(cluster_idx), mic=mic))
            <= (new_radius + _covalent_radius(candidate[int(cluster_idx)].symbol)) * float(bond_scale)
            for cluster_idx in cluster_indices
        ):
            return False
    return True


def _passes_attachment_filter(
    atoms: Atoms,
    new_indices: Sequence[int],
    attachment_symbols: Sequence[str],
    max_distance: float | None,
) -> bool:
    if max_distance is None:
        return True
    attachment_set = set(attachment_symbols)
    existing = [
        i for i, atom in enumerate(atoms)
        if atom.symbol in attachment_set and i not in new_indices
    ]
    mic = bool(np.any(atoms.get_pbc()))
    return not existing or all(
        min(atoms.get_distance(i, j, mic=mic) for j in existing) <= float(max_distance)
        for i in new_indices
    )


def adsorbates_bound_to_support(
    atoms: Atoms,
    n_support_atoms: int,
    cutoff_scale: float = 1.2,
    adsorbate_groups: Sequence[Sequence[int]] | None = None,
) -> bool:
    adsorbate_indices = list(range(int(n_support_atoms), len(atoms)))
    if not adsorbate_indices or n_support_atoms <= 0:
        return False
    mic = bool(np.any(atoms.get_pbc()))
    if adsorbate_groups is not None:
        components = [{int(i) for i in group} for group in adsorbate_groups]
        if any(not component or not component <= set(adsorbate_indices) for component in components):
            return False
    else:
        components = []
        unseen = set(adsorbate_indices)
        while unseen:
            component = {unseen.pop()}
            pending = list(component)
            while pending:
                i = pending.pop()
                bonded = {
                    j for j in unseen
                    if atoms.get_distance(i, j, mic=mic)
                    <= cutoff_scale
                    * (covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number])
                }
                unseen -= bonded
                component |= bonded
                pending.extend(bonded)
            components.append(component)
    return all(
        any(
            atoms.get_distance(i, j, mic=mic)
            <= cutoff_scale * (covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number])
            for i in component
            for j in range(int(n_support_atoms))
        )
        for component in components
    )


def adsorbate_integrity_preserved(
    initial: Atoms,
    relaxed: Atoms,
    n_support_atoms: int,
    adsorbate_groups: Sequence[Sequence[int]] | None = None,
    cutoff_scale: float = 1.2,
    min_bond_ratio: float = 0.65,
    max_bond_ratio: float = 1.60,
) -> bool:
    """Return whether relaxation preserved the adsorbates' covalent bond graph."""
    start = int(n_support_atoms)
    if len(initial) != len(relaxed) or start < 0 or start > len(initial):
        return False

    groups = (
        [tuple(range(start, len(initial)))]
        if adsorbate_groups is None
        else [tuple(int(i) for i in group) for group in adsorbate_groups]
    )
    if any(i < start or i >= len(initial) for group in groups for i in group):
        return False

    def bonds(atoms: Atoms) -> set[tuple[int, int]]:
        mic = bool(np.any(atoms.get_pbc()))
        return {
            (i, j)
            for group in groups
            for pos, i in enumerate(group)
            for j in group[pos + 1:]
            if atoms.get_distance(i, j, mic=mic)
            <= cutoff_scale
            * (covalent_radii[atoms[i].number] + covalent_radii[atoms[j].number])
        }

    initial_bonds = bonds(initial)
    if (
        initial.get_chemical_symbols() != relaxed.get_chemical_symbols()
        or initial_bonds != bonds(relaxed)
    ):
        return False
    initial_mic = bool(np.any(initial.get_pbc()))
    relaxed_mic = bool(np.any(relaxed.get_pbc()))
    return all(
        min_bond_ratio
        <= relaxed.get_distance(i, j, mic=relaxed_mic)
        / initial.get_distance(i, j, mic=initial_mic)
        <= max_bond_ratio
        for i, j in initial_bonds
    )


def _resolve_geometry_concurrency_workers(n_jobs: int | None = None) -> int:
    if n_jobs is not None and int(n_jobs) > 0:
        return int(n_jobs)
    for env_var in (
        "WORKFLOW_CPUS_PER_TASK",
        "SLURM_CPUS_PER_TASK",
        "SLURM_NPROCS",
        "OMP_NUM_THREADS",
    ):
        val = os.environ.get(env_var)
        if val and val.isdigit():
            n = int(val)
            if n > 0:
                return n
    return max(1, os.cpu_count() or 1)


def _evaluate_candidate_task(task_args):
    (
        host_atoms,
        passive_atoms,
        sites,
        site_indices,
        oriented_order,
        height,
        contact_radius_scale,
        z_cutoff,
        cluster_indices,
        enforce_cluster_integrity,
        cluster_integrity_bond_scale,
        attachment_symbols,
        max_attachment_distance,
        skip_abnormal_bond_screening,
        skip_geometry_screening,
    ) = task_args

    temp = host_atoms.copy()
    new_indices: list[int] = []
    placements: list[str] = []
    contacts: list[dict[str, object]] = []
    for site_idx, adsorbate in zip(site_indices, oriented_order):
        prev_size = len(temp)
        adsorbate_payload = (
            adsorbate.payload.copy()
            if isinstance(adsorbate.payload, Atoms)
            else adsorbate.payload
        )
        add_adsorbate_to_site(
            temp,
            adsorbate=adsorbate_payload,
            height=height,
            site=sites[site_idx],
        )
        placed_indices = list(range(prev_size, len(temp)))
        new_indices.extend(placed_indices)
        if contact_radius_scale is not None:
            contact = _adjust_contact_distance(
                temp,
                adsorbate_indices=placed_indices,
                n_support_atoms=len(host_atoms),
                site=sites[site_idx],
                radius_scale=contact_radius_scale,
            )
            if contact is not None:
                contacts.append(contact)
        site_label = str(sites[site_idx].get("site", ""))
        placements.append(
            f"{adsorbate.label}@{site_idx}:{site_label}"
            f":flip={int(adsorbate.flipped)}:rz={adsorbate.rotation_z_degrees:g}"
        )

    return _screen_candidate(
        temp,
        passive_atoms,
        new_indices=new_indices,
        placements=placements,
        adsorbate_order=oriented_order,
        site_indices=site_indices,
        height=height,
        contacts=contacts,
        z_cutoff=z_cutoff,
        cluster_indices=cluster_indices,
        enforce_cluster_integrity=enforce_cluster_integrity,
        cluster_integrity_bond_scale=cluster_integrity_bond_scale,
        attachment_symbols=attachment_symbols,
        max_attachment_distance=max_attachment_distance,
        skip_abnormal_bond_screening=skip_abnormal_bond_screening,
        skip_geometry_screening=skip_geometry_screening,
        stats=None,
    )


def _iter_candidates(
    host_atoms: Atoms,
    passive_atoms: Atoms | None,
    *,
    sites: Sequence[dict[str, object]],
    adsorbates: Sequence[AdsorbateSpec],
    height: float = 2.0,
    contact_radius_scale: float | None = None,
    z_cutoff: float | None = None,
    max_structures: int | None = None,
    cluster_indices: Sequence[int] = (),
    enforce_cluster_integrity: bool = False,
    cluster_integrity_bond_scale: float = CLUSTER_INTEGRITY_BOND_SCALE,
    attachment_symbols: Sequence[str] = (),
    max_attachment_distance: float | None = None,
    skip_abnormal_bond_screening: bool = False,
    skip_geometry_screening: bool = False,
    rotation_step_degrees: float = 30.0,
    include_flipped_orientations: bool = True,
    n_jobs: int | None = None,
    stats: dict[str, int] | None = None,
):
    n_adsorbates = len(adsorbates)
    if len(sites) < n_adsorbates:
        raise ValueError(
            f"Not enough adsorption sites ({len(sites)}) for requested adsorbates ({n_adsorbates})."
        )

    adsorbate_orders = _unique_adsorbate_orders(adsorbates)
    all_tasks = []
    for site_indices in combinations(range(len(sites)), n_adsorbates):
        for adsorbate_order in adsorbate_orders:
            orientation_options = [
                _adsorbate_orientations(
                    adsorbate,
                    rotation_step_degrees=rotation_step_degrees,
                    include_flip=include_flipped_orientations,
                )
                for adsorbate in adsorbate_order
            ]
            for oriented_order in product(*orientation_options):
                if stats is not None:
                    stats["attempted"] += 1
                all_tasks.append((
                    host_atoms,
                    passive_atoms,
                    sites,
                    site_indices,
                    oriented_order,
                    height,
                    contact_radius_scale,
                    z_cutoff,
                    cluster_indices,
                    enforce_cluster_integrity,
                    cluster_integrity_bond_scale,
                    attachment_symbols,
                    max_attachment_distance,
                    skip_abnormal_bond_screening,
                    skip_geometry_screening,
                ))

    n_workers = _resolve_geometry_concurrency_workers(n_jobs=n_jobs)
    emitted = 0
    if n_workers > 1 and len(all_tasks) > 4:
        chunksize = max(1, len(all_tasks) // (n_workers * 4))
        if hasattr(os, "fork"):
            mp_ctx = concurrent.futures.process.mp.get_context("fork")
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_ctx) as executor:
                for screened in executor.map(_evaluate_candidate_task, all_tasks, chunksize=chunksize):
                    if screened is None:
                        continue
                    if stats is not None:
                        stats["accepted"] += 1
                    yield screened
                    emitted += 1
                    if max_structures is not None and emitted >= max_structures:
                        return
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                for screened in executor.map(_evaluate_candidate_task, all_tasks):
                    if screened is None:
                        continue
                    if stats is not None:
                        stats["accepted"] += 1
                    yield screened
                    emitted += 1
                    if max_structures is not None and emitted >= max_structures:
                        return
    else:
        for task in all_tasks:
            screened = _evaluate_candidate_task(task)
            if screened is None:
                continue
            if stats is not None:
                stats["accepted"] += 1
            yield screened
            emitted += 1
            if max_structures is not None and emitted >= max_structures:
                return


def _screen_candidate(
    temp,
    passive_atoms,
    *,
    new_indices,
    placements,
    adsorbate_order,
    site_indices,
    height,
    contacts,
    z_cutoff,
    cluster_indices,
    enforce_cluster_integrity,
    cluster_integrity_bond_scale,
    attachment_symbols,
    max_attachment_distance,
    skip_abnormal_bond_screening,
    skip_geometry_screening,
    stats,
):
    if not skip_geometry_screening and not _passes_z_filter(temp, new_indices, z_cutoff):
        if stats is not None:
            stats["z_filter"] += 1
        return None

    if not skip_geometry_screening and enforce_cluster_integrity and not _passes_cluster_integrity(
        temp,
        new_indices,
        cluster_indices,
        bond_scale=float(cluster_integrity_bond_scale),
    ):
        if stats is not None:
            stats["cluster_integrity"] += 1
        return None

    if not skip_geometry_screening and not _passes_bonding_filter(
        temp,
        skip_abnormal_bond_screening=skip_abnormal_bond_screening,
        n_added=len(new_indices),
    ):
        if stats is not None:
            stats["bonding"] += 1
        return None

    final = temp if passive_atoms is None else passive_atoms + temp
    offset = 0 if passive_atoms is None else len(passive_atoms)
    if not skip_geometry_screening and not _passes_attachment_filter(
        final,
        [offset + idx for idx in new_indices],
        attachment_symbols,
        max_attachment_distance,
    ):
        if stats is not None:
            stats["attachment"] += 1
        return None
    if stats is not None:
        stats["accepted"] += 1
    metadata = {
        "adsorbates": ",".join(spec.label for spec in adsorbate_order),
        "site_indices": ",".join(str(idx) for idx in site_indices),
        "placements": ";".join(placements),
        "height": float(height),
        "z_cutoff": "" if z_cutoff is None else float(z_cutoff),
    }
    if contacts:
        metadata["contacts"] = ";".join(
            f"{contact['adsorbate_symbol']}-{contact['support_symbol']}"
            f":{contact['target_distance']:.6f}A:shift={contact['normal_shift']:.6f}A"
            for contact in contacts
        )
    return final, metadata


def run_coadsorbate_generation(
    atoms: Atoms,
    *,
    adsorbates: Sequence[AdsorbateLike],
    output_db: str | Path | None = "coadsorption_candidates.db",
    site_symbols: Sequence[str] | None = None,
    passive_symbols: Sequence[str] = DEFAULT_PASSIVE_SYMBOLS,
    height: float = 2.0,
    contact_radius_scale: float | None = None,
    z_cutoff: float | None = None,
    z_reference_symbol: str | None = DEFAULT_Z_REFERENCE_SYMBOL,
    z_buffer: float | None = None,
    allow_6fold: bool = False,
    surrogate_metal: str = "Cu",
    composition_effect: bool = True,
    max_structures: int | None = None,
    cluster_symbols: Sequence[str] | None = None,
    enforce_cluster_integrity: bool = False,
    cluster_integrity_bond_scale: float = CLUSTER_INTEGRITY_BOND_SCALE,
    use_triangulated_com_fallback: bool = True,
    force_triangulated_com_fallback: bool = False,
    triangle_neighbor_cutoff: float = TRIANGLE_NEIGHBOR_CUTOFF,
    surface_top_z_tolerance: float = 1.5,
    include_below_surface: bool = False,
    max_new_atom_cluster_distance: float | None = None,
    skip_abnormal_bond_screening: bool = False,
    skip_geometry_screening: bool = False,
    rotation_step_degrees: float = 30.0,
    include_flipped_orientations: bool = True,
    allowed_site_indices: Sequence[int] | None = None,
    show: bool = False,
    origin_atom_index: int | None = None,
    site_cutoff: float | None = None,
    n_jobs: int | None = None,
    verbose: bool = True,
) -> list[Atoms]:
    """
    Generate single- and co-adsorption candidates and write them to an ASE DB.

    z_cutoff behavior:
    - set `z_cutoff` to an explicit float for a fixed filter;
    - leave `z_cutoff=None` and set `z_buffer` to compute cutoff from host geometry;
    - keep both as `None` to disable z filtering.

    bonding behavior:
    - default (`skip_abnormal_bond_screening=False`): keep quick clash check and
      radii-based abnormal-bond screening.
    - set `skip_abnormal_bond_screening=True`: keep only the quick clash check.
    - set `skip_geometry_screening=True` to defer all geometry validation until
      after optimisation.

    orientation behavior:
    - ASE Atoms adsorbates are rotated around their first (attachment) atom;
    - z-axis rotations use `rotation_step_degrees` and optionally include a
      180-degree x-axis flipped series.

    fallback behavior:
    - carbon-majority periodic supports bypass ACAT;
    - fallback detection always includes hollow, bridge, and atop sites.

    Periodic structures always use surface sites; non-periodic structures use
    cluster sites.

    Set `include_below_surface=True` on a carbonaceous periodic support to
    mirror each top-layer site below the same carbon layer.

    Set both `origin_atom_index` (0-based, in the input structure) and
    `site_cutoff` (Angstrom) to retain only adsorption sites near that atom.
    Use `allowed_site_indices` to generate orientations only for selected
    indices from the detected, origin-filtered site list.
    """
    if max_structures is not None and max_structures <= 0:
        max_structures = None
    if contact_radius_scale is not None and float(contact_radius_scale) <= 0.0:
        raise ValueError("contact_radius_scale must be positive or None.")

    adsorbates_specs = _normalize_adsorbates(adsorbates)
    if not adsorbates_specs:
        raise ValueError("At least one adsorbate (str or ASE Atoms) must be provided.")
    has_existing_cluster = any(
        atom.symbol in set(cluster_symbols or ()) for atom in atoms
    )
    fallback_site_cap = (
        None
        if max_new_atom_cluster_distance is not None and has_existing_cluster
        else max_structures if len(adsorbates_specs) == 1 else None
    )

    use_slab_sites = _is_slab_like(atoms)
    if use_slab_sites:
        host_atoms, passive_atoms = _build_host_and_passive(
            atoms,
            site_symbols=site_symbols,
            passive_symbols=passive_symbols,
        )
        cluster_host = False
    else:
        host_atoms, passive_atoms, cluster_host = _build_cluster_host_and_passive(
            atoms,
            cluster_symbols=cluster_symbols,
        )
        if not cluster_host:
            host_atoms, passive_atoms = _build_host_and_passive(
                atoms,
                site_symbols=site_symbols,
                passive_symbols=passive_symbols,
            )
    cluster_indices = tuple(range(len(host_atoms))) if cluster_host else ()
    z_cutoff = _resolve_z_cutoff(
        host_atoms,
        z_cutoff=z_cutoff,
        z_reference_symbol=z_reference_symbol,
        z_buffer=z_buffer,
    )

    mode = "cluster"
    carbonaceous_support = _is_carbonaceous_periodic_support(atoms)
    if verbose:
        detector = (
            "Delaunay triangulation"
            if force_triangulated_com_fallback or carbonaceous_support
            else "ACAT"
        )
        print(f"[Generation] Detecting adsorption sites with {detector}...")
    if force_triangulated_com_fallback:
        sites = _triangulated_com_sites(
            host_atoms,
            neighbor_cutoff=float(triangle_neighbor_cutoff),
            max_sites=fallback_site_cap,
            top_surface_only=use_slab_sites,
            top_z_tolerance=float(surface_top_z_tolerance),
        )
        mode = "forced-triangulated-com-fallback"
    elif carbonaceous_support:
        sites = _triangulated_com_sites(
            host_atoms,
            neighbor_cutoff=float(triangle_neighbor_cutoff),
            max_sites=fallback_site_cap,
            top_surface_only=True,
            top_z_tolerance=float(surface_top_z_tolerance),
        )
        mode = "carbonaceous-hollow-bridge-atop-fallback"
    elif use_slab_sites:
        try:
            sites, mode = _slab_sites(
                host_atoms,
                composition_effect=composition_effect,
                top_z_tolerance=float(surface_top_z_tolerance),
            )
        except ModuleNotFoundError as exc:
            if exc.name != "autoadsorbate":
                raise
            if not use_triangulated_com_fallback:
                raise
            print(
                "Warning: optional dependency 'autoadsorbate' is missing; "
                "using Delaunay triangulation."
            )
            sites = _triangulated_com_sites(
                host_atoms,
                neighbor_cutoff=float(triangle_neighbor_cutoff),
                max_sites=fallback_site_cap,
                top_surface_only=True,
                top_z_tolerance=float(surface_top_z_tolerance),
            )
            mode = "triangulated-com-fallback"
        except Exception as exc:
            if not use_triangulated_com_fallback:
                raise
            print(f"Warning: ACAT slab site detection failed ({exc}); using Delaunay triangulation.")
            sites = _triangulated_com_sites(
                host_atoms,
                neighbor_cutoff=float(triangle_neighbor_cutoff),
                max_sites=fallback_site_cap,
                top_surface_only=True,
                top_z_tolerance=float(surface_top_z_tolerance),
            )
            mode = "triangulated-com-fallback"
    else:
        try:
            sites = _cluster_sites(
                host_atoms,
                allow_6fold=allow_6fold,
                composition_effect=composition_effect,
                surrogate_metal=surrogate_metal,
            )
        except Exception as exc:
            if not use_triangulated_com_fallback:
                raise
            print(f"Warning: ACAT cluster site detection failed ({exc}); using Delaunay triangulation.")
            sites = _triangulated_com_sites(
                host_atoms,
                neighbor_cutoff=float(triangle_neighbor_cutoff),
                max_sites=fallback_site_cap,
            )
            mode = "triangulated-com-fallback"
    if not sites and use_triangulated_com_fallback:
        print("Warning: ACAT detected no adsorption sites; using Delaunay triangulation.")
        sites = _triangulated_com_sites(
            host_atoms,
            neighbor_cutoff=float(triangle_neighbor_cutoff),
            max_sites=fallback_site_cap,
            top_surface_only=use_slab_sites,
            top_z_tolerance=float(surface_top_z_tolerance),
        )
        mode = "triangulated-com-fallback"
    if include_below_surface:
        if not carbonaceous_support:
            raise ValueError(
                "include_below_surface requires a carbonaceous periodic support."
            )
        sites = _include_below_surface_sites(sites)
        mode = f"{mode}-both-sides"
    detected_site_count = len(sites)
    sites = _sites_near_origin(sites, atoms, origin_atom_index, site_cutoff)
    if allowed_site_indices is not None:
        allowed = {int(idx) for idx in allowed_site_indices}
        sites = [site for idx, site in enumerate(sites) if idx in allowed]
    if not sites:
        suffix = " within the requested origin cutoff" if origin_atom_index is not None else ""
        raise RuntimeError(f"No adsorption sites were detected{suffix}.")
    if verbose:
        print(
            f"[Generation] Host atoms: {len(host_atoms)}; "
            f"passive atoms: {0 if passive_atoms is None else len(passive_atoms)}"
        )
        uses_delaunay = mode.startswith(
            ("forced-triangulated", "triangulated", "carbonaceous-hollow")
        )
        display_mode = "Delaunay triangulation" if uses_delaunay else f"{mode} mode"
        print(f"[Generation] Adsorption sites: {len(sites)} / {detected_site_count} ({display_mode})")
        if origin_atom_index is not None:
            print(
                f"[Generation] Site origin: atom {int(origin_atom_index)} "
                f"({atoms[int(origin_atom_index)].symbol}); cutoff: {float(site_cutoff):.3f} A"
            )
        orientation_count = sum(
            int(
                np.prod(
                    [
                        len(
                            _adsorbate_orientations(
                                spec,
                                rotation_step_degrees=rotation_step_degrees,
                                include_flip=include_flipped_orientations,
                            )
                        )
                        for spec in order
                    ]
                )
            )
            for order in _unique_adsorbate_orders(adsorbates_specs)
        )
        print(
            f"[Generation] Candidate combinations before screening: "
            f"{comb(len(sites), len(adsorbates_specs)) * orientation_count}"
        )

    generated: list[Atoms] = []
    generated_metadata: list[dict[str, object]] = []
    written = 0
    stats = {key: 0 for key in ("attempted", "z_filter", "cluster_integrity", "bonding", "attachment", "accepted")}
    # TODO: Dispatch `_screen_candidate` through the existing MPI CPU-task pool
    # and add periodic generation progress.
    for candidate, metadata in _iter_candidates(
        host_atoms,
        passive_atoms,
        sites=sites,
        adsorbates=adsorbates_specs,
        height=height,
        contact_radius_scale=contact_radius_scale,
        z_cutoff=z_cutoff,
        max_structures=max_structures,
        cluster_indices=cluster_indices,
        enforce_cluster_integrity=bool(enforce_cluster_integrity),
        cluster_integrity_bond_scale=float(cluster_integrity_bond_scale),
        attachment_symbols=cluster_symbols or (),
        max_attachment_distance=(
            max_new_atom_cluster_distance if use_slab_sites else None
        ),
        skip_abnormal_bond_screening=bool(skip_abnormal_bond_screening),
        skip_geometry_screening=bool(skip_geometry_screening),
        rotation_step_degrees=float(rotation_step_degrees),
        include_flipped_orientations=bool(include_flipped_orientations),
        n_jobs=n_jobs,
        stats=stats,
    ):
        generated.append(candidate)
        generated_metadata.append(metadata)
        written += 1

    if written == 0 and use_triangulated_com_fallback and "fallback" not in mode:
        sites = _triangulated_com_sites(
            host_atoms,
            neighbor_cutoff=float(triangle_neighbor_cutoff),
            max_sites=fallback_site_cap,
            top_surface_only=use_slab_sites,
            top_z_tolerance=float(surface_top_z_tolerance),
        )
        if sites:
            sites = _sites_near_origin(sites, atoms, origin_atom_index, site_cutoff)
            if allowed_site_indices is not None:
                allowed = {int(idx) for idx in allowed_site_indices}
                sites = [site for idx, site in enumerate(sites) if idx in allowed]
        if sites:
            mode = "triangulated-com-fallback"
            print(
                f"No accepted ACAT placements; retrying with {len(sites)} "
                "Delaunay triangulation sites"
            )
            for candidate, metadata in _iter_candidates(
                host_atoms,
                passive_atoms,
                sites=sites,
                adsorbates=adsorbates_specs,
                height=height,
                contact_radius_scale=contact_radius_scale,
                z_cutoff=z_cutoff,
                max_structures=max_structures,
                cluster_indices=cluster_indices,
                enforce_cluster_integrity=bool(enforce_cluster_integrity),
                cluster_integrity_bond_scale=float(cluster_integrity_bond_scale),
                attachment_symbols=cluster_symbols or (),
                max_attachment_distance=(
                    max_new_atom_cluster_distance if use_slab_sites else None
                ),
                skip_abnormal_bond_screening=bool(skip_abnormal_bond_screening),
                skip_geometry_screening=bool(skip_geometry_screening),
                rotation_step_degrees=float(rotation_step_degrees),
                include_flipped_orientations=bool(include_flipped_orientations),
                stats=stats,
            ):
                generated.append(candidate)
                generated_metadata.append(metadata)
                written += 1

    if output_db is not None:
        with connect(str(output_db), serial=True) as db:
            for candidate, metadata in zip(generated, generated_metadata):
                db.write(candidate, data=metadata)

    if verbose:
        print(
            "[Generation] Screening: "
            f"attempted={stats['attempted']}, accepted={stats['accepted']}, "
            f"z={stats['z_filter']}, cluster_integrity={stats['cluster_integrity']}, "
            f"bonding={stats['bonding']}, attachment={stats['attachment']}"
        )
        if output_db is not None:
            print(f"[Generation] Output DB: {Path(output_db).resolve()}")

    if show and generated:
        from ase.visualize import view

        view(generated)
    return generated


def run_coadsorbate_generation_from_file(
    input_path: str | Path = "POSCAR",
    **kwargs,
) -> list[Atoms]:
    atoms = read(input_path, parallel=False)
    return run_coadsorbate_generation(atoms, **kwargs)


def _self_check() -> None:
    assert not _is_slab_like(Atoms("Au"))
    assert _is_slab_like(Atoms("ZnO", cell=[5, 5, 10], pbc=[True, True, False]))
    assert _passes_attachment_filter(
        Atoms("AuCu", positions=[[0, 0, 0], [5, 0, 0]]), [1], ["Au", "Cu"], 5.0
    )
    assert not _passes_attachment_filter(
        Atoms("AuCu", positions=[[0, 0, 0], [5.1, 0, 0]]), [1], ["Au", "Cu"], 5.0
    )

    atoms = Atoms("Cu3", positions=[[0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0]])
    sites = _triangulated_com_sites(atoms, neighbor_cutoff=3.0)
    assert {site["site"] for site in sites} == {"fcc", "bridge", "ontop"}
    assert sites[0]["site"] == "fcc"
    assert np.allclose(sites[0]["position"], atoms.get_positions().mean(axis=0))
    assert sites[0]["normal"][2] > 0.0
    both_sides = _include_below_surface_sites([sites[0]])
    assert both_sides[0] is sites[0]
    assert both_sides[1]["site"] == "fcc_below"
    assert np.allclose(both_sides[1]["normal"], -sites[0]["normal"])

    chain = Atoms("Cu3", positions=[[0, 0, 0], [2.5, 0, 0], [5.0, 0, 0]])
    bridge_sites = [
        site for site in _triangulated_com_sites(chain, neighbor_cutoff=3.0)
        if site["site"] == "bridge"
    ]
    assert bridge_sites
    assert all(abs(float(np.dot(site["normal"], [1, 0, 0]))) < 1.0e-12 for site in bridge_sites)

    slab = Atoms(
        "Cu6",
        positions=[
            [0, 0, 0], [2.5, 0, 0], [1.25, 2.1, 0],
            [0, 0, 3], [2.5, 0, 3], [1.25, 2.1, 3],
        ],
    )
    top_sites = _triangulated_com_sites(
        slab, neighbor_cutoff=3.0, top_surface_only=True, top_z_tolerance=0.5
    )
    assert top_sites and all(i >= 3 for i in top_sites[0]["indices"])
    carbon = Atoms("C3Au", positions=[[0, 0, 0], [2, 0, 0], [1, 1.7, 0], [1, 0.5, 2]], cell=[6, 6, 10], pbc=[True, True, False])
    assert _is_carbonaceous_periodic_support(carbon)

    supported = Atoms(
        "Ti2Au",
        positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 0.0, 2.0]],
        cell=[6.0, 6.0, 12.0],
        pbc=[True, True, False],
    )
    host, passive, cluster_host = _build_cluster_host_and_passive(supported, cluster_symbols=["Au"])
    assert cluster_host
    assert host.get_chemical_symbols() == ["Au"]
    assert passive is not None and passive.get_chemical_symbols() == ["Ti", "Ti"]

    close = Atoms("Au2", positions=[[0.0, 0.0, 0.0], [2.7, 0.0, 0.0]])
    far = Atoms("Au2", positions=[[0.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    assert _passes_cluster_integrity(close, [1], [0], bond_scale=CLUSTER_INTEGRITY_BOND_SCALE)
    assert not _passes_cluster_integrity(far, [1], [0], bond_scale=CLUSTER_INTEGRITY_BOND_SCALE)


if __name__ == "__main__":
    # Minimal editable example.
    import sys

    if "--self-check" in sys.argv:
        _self_check()
        print("self-check passed")
        raise SystemExit(0)

    surface = read("POSCAR", parallel=False)
    co = read("CO.xyz", parallel=False)
    no = read("NO.xyz", parallel=False)
    co.info["label"] = "CO_obj"
    no.info["label"] = "NO_obj"

    run_coadsorbate_generation(
        surface,
        adsorbates=[co, no],  # Atoms objects
        output_db="coadsorption_candidates.db",
    )
