from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from math import comb
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from ase.atoms import Atoms
from ase.build import molecule
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixInternals
from ase.data import covalent_radii
from ase.db import connect
from ase.optimize import BFGS
from ase.visualize import view

from similarity.rematch.memory_rematch import (
    deduplicate_atoms_by_similarity,
    suggest_similarity_threshold,
)


DEFAULT_MACE_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "run" / "mace-omat-0-medium.model"
)


AdsorbateLike = str | Atoms


def _fmt_optional(value: float | None, fmt: str = ".4f", suffix: str = "") -> str:
    if value is None:
        return "N/A"
    return f"{float(value):{fmt}}{suffix}"


@dataclass(frozen=True)
class ZeoliteAdsorbateSpec:
    label: str
    key: str
    template: Atoms


def _coerce_db_value(value):
    if value is None:
        return None
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _write_atoms_db(db_path: Path, atoms_list, metadata_rows=None) -> None:
    with connect(str(db_path)) as db:
        for idx, atoms in enumerate(atoms_list):
            data = {}
            if metadata_rows is not None:
                row = metadata_rows[idx] if idx < len(metadata_rows) else {}
                data = {key: _coerce_db_value(value) for key, value in row.items()}
            db.write(atoms, data=data)


def _suggest_threshold_safe(atoms_list, reference_row_id: int, similarity_kwargs: dict):
    try:
        return suggest_similarity_threshold(
            atoms_list=atoms_list,
            reference_row_id=reference_row_id,
            **similarity_kwargs,
        )
    except ValueError as exc:
        if "No comparison structures available" not in str(exc):
            raise
        fallback_kwargs = dict(similarity_kwargs)
        fallback_kwargs["exclude_self"] = False
        return suggest_similarity_threshold(
            atoms_list=atoms_list,
            reference_row_id=reference_row_id,
            **fallback_kwargs,
        )


def _atoms_signature(atoms: Atoms) -> str:
    parts: list[str] = []
    for atom in atoms:
        x, y, z = atom.position
        parts.append(f"{atom.symbol}:{x:.6f}:{y:.6f}:{z:.6f}")
    return "|".join(parts)


def _normalize_adsorbates(adsorbates: Sequence[AdsorbateLike]) -> list[ZeoliteAdsorbateSpec]:
    specs: list[ZeoliteAdsorbateSpec] = []
    for idx, adsorbate in enumerate(adsorbates):
        if isinstance(adsorbate, Atoms):
            if len(adsorbate) == 0:
                raise ValueError("Empty ASE Atoms adsorbate is not valid.")
            label = str(
                adsorbate.info.get("label") or adsorbate.get_chemical_formula(mode="hill")
            )
            specs.append(
                ZeoliteAdsorbateSpec(
                    label=label if label else f"ads_{idx}",
                    key=f"atoms:{_atoms_signature(adsorbate)}",
                    template=adsorbate.copy(),
                )
            )
            continue
        if isinstance(adsorbate, str):
            name = adsorbate.strip()
            if not name:
                continue
            try:
                template = molecule(name)
            except Exception as exc:
                raise ValueError(
                    f"Failed to build adsorbate '{name}' via ase.build.molecule: {exc}"
                ) from exc
            specs.append(
                ZeoliteAdsorbateSpec(
                    label=name,
                    key=f"str:{name}",
                    template=template,
                )
            )
            continue
        raise TypeError(
            "Adsorbates must be strings or ASE Atoms objects, "
            f"got {type(adsorbate).__name__}."
        )
    if not specs:
        raise ValueError("At least one adsorbate is required.")
    return specs


def _unique_adsorbate_orders(
    adsorbates: Sequence[ZeoliteAdsorbateSpec],
) -> list[tuple[ZeoliteAdsorbateSpec, ...]]:
    seen: set[tuple[str, ...]] = set()
    unique_orders: list[tuple[ZeoliteAdsorbateSpec, ...]] = []
    for order in permutations(range(len(adsorbates))):
        key = tuple(adsorbates[i].key for i in order)
        if key in seen:
            continue
        seen.add(key)
        unique_orders.append(tuple(adsorbates[i] for i in order))
    return unique_orders


def _ring_center_of_mass(atoms: Atoms, ring_indices: Sequence[int]) -> np.ndarray:
    if not ring_indices:
        raise ValueError("ring_indices must contain at least one atom index.")
    max_index = len(atoms) - 1
    for idx in ring_indices:
        if idx < 0 or idx > max_index:
            raise ValueError(
                f"Ring index {idx} out of bounds for structure with {len(atoms)} atoms."
            )
    ring_atoms = atoms[list(ring_indices)]
    return ring_atoms.get_center_of_mass()


def _normalize_com_point(cage_com: Sequence[float]) -> np.ndarray:
    array = np.asarray(cage_com, dtype=float).reshape(-1)
    if array.shape != (3,):
        raise ValueError(
            "cage_com must be a 3-element coordinate [x, y, z]. "
            f"Received shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("cage_com contains non-finite values.")
    return array


def _select_adsorption_sites_near_com(
    atoms: Atoms,
    *,
    cage_com: np.ndarray,
    site_symbols: Sequence[str],
    target_radius: float,
    max_site_candidates: int | None = None,
) -> tuple[list[int], list[float]]:
    if target_radius <= 0.0:
        raise ValueError("target_radius must be positive.")
    site_symbol_set = {str(symbol).strip() for symbol in site_symbols if str(symbol).strip()}
    if not site_symbol_set:
        raise ValueError("site_symbols must contain at least one chemical symbol.")

    def _distance(i: int) -> float:
        return float(np.linalg.norm(atoms[i].position - cage_com))

    ordered = sorted(
        [
            i
            for i, atom in enumerate(atoms)
            if atom.symbol in site_symbol_set and _distance(i) <= target_radius
        ],
        key=_distance,
    )

    if max_site_candidates is not None and max_site_candidates > 0:
        ordered = ordered[: max_site_candidates]
    distances = [_distance(i) for i in ordered]
    return ordered, distances


def _infer_cage_boundary_radius_from_framework(
    atoms: Atoms,
    *,
    cage_com: np.ndarray,
    n_nearest: int = 12,
) -> float:
    if n_nearest <= 0:
        raise ValueError("cage_boundary_n_nearest must be positive.")
    if len(atoms) == 0:
        raise ValueError("Cannot infer cage boundary radius from empty framework.")

    distances = np.linalg.norm(atoms.positions - cage_com, axis=1)
    if distances.size == 0:
        raise ValueError("Could not infer cage boundary radius: no framework distances.")
    k = min(int(n_nearest), int(distances.size))
    nearest = np.partition(distances, k - 1)[:k]
    inferred_radius = float(np.mean(nearest))
    if inferred_radius <= 0.0:
        raise ValueError(
            "Inferred cage boundary radius is non-positive. "
            f"Check cage_com: inferred={inferred_radius:.6f}"
        )
    return inferred_radius


def _direction_to_com(site_position: np.ndarray, cage_com: np.ndarray) -> np.ndarray:
    vec = cage_com - site_position
    norm = float(np.linalg.norm(vec))
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 1.0])
    return vec / norm


def _adsorbate_max_extent_from_com(template: Atoms) -> float:
    if len(template) == 0:
        return 0.0
    com = template.get_center_of_mass()
    distances = np.linalg.norm(template.positions - com, axis=1)
    if distances.size == 0:
        return 0.0
    return float(np.max(distances))


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=float)
    axis /= norm
    x, y, z = axis
    cos_a = float(np.cos(angle_rad))
    sin_a = float(np.sin(angle_rad))
    one_minus_cos = 1.0 - cos_a
    return np.array(
        [
            [cos_a + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_a, x * z * one_minus_cos + y * sin_a],
            [y * x * one_minus_cos + z * sin_a, cos_a + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_a],
            [z * x * one_minus_cos - y * sin_a, z * y * one_minus_cos + x * sin_a, cos_a + z * z * one_minus_cos],
        ],
        dtype=float,
    )


def _rotation_matrix_align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = np.asarray(source, dtype=float).reshape(3)
    tgt = np.asarray(target, dtype=float).reshape(3)
    src_norm = float(np.linalg.norm(src))
    tgt_norm = float(np.linalg.norm(tgt))
    if src_norm <= 1.0e-12 or tgt_norm <= 1.0e-12:
        return np.eye(3, dtype=float)
    src /= src_norm
    tgt /= tgt_norm

    dot = float(np.clip(np.dot(src, tgt), -1.0, 1.0))
    if dot >= 1.0 - 1.0e-12:
        return np.eye(3, dtype=float)
    if dot <= -1.0 + 1.0e-12:
        trial_axis = np.cross(src, np.array([1.0, 0.0, 0.0], dtype=float))
        if float(np.linalg.norm(trial_axis)) <= 1.0e-12:
            trial_axis = np.cross(src, np.array([0.0, 1.0, 0.0], dtype=float))
        return _rotation_matrix_from_axis_angle(trial_axis, np.pi)

    axis = np.cross(src, tgt)
    angle = float(np.arccos(dot))
    return _rotation_matrix_from_axis_angle(axis, angle)


def _fibonacci_sphere_directions(count: int, *, phase: float = 0.0) -> list[np.ndarray]:
    if int(count) <= 0:
        return []
    golden = np.pi * (3.0 - np.sqrt(5.0))
    vectors: list[np.ndarray] = []
    for idx in range(int(count)):
        z = 1.0 - 2.0 * ((idx + 0.5) / float(count))
        radial = float(np.sqrt(max(0.0, 1.0 - z * z)))
        theta = (float(idx) + float(phase)) * golden
        vectors.append(
            np.array([radial * np.cos(theta), radial * np.sin(theta), z], dtype=float)
        )
    return vectors


def _estimate_open_direction_from_framework(
    atoms: Atoms,
    *,
    cage_com: np.ndarray,
    n_nearest: int = 24,
) -> np.ndarray:
    if len(atoms) == 0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    deltas = cage_com - atoms.positions
    distances = np.linalg.norm(deltas, axis=1)
    if distances.size == 0:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    k = min(int(max(1, n_nearest)), int(distances.size))
    nearest_indices = np.argpartition(distances, k - 1)[:k]
    nearest_distances = np.maximum(distances[nearest_indices], 1.0e-3)
    weights = 1.0 / np.square(nearest_distances)
    vector = np.sum(deltas[nearest_indices] * weights[:, None], axis=0)
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return np.asarray(vector, dtype=float) / norm


def _base_separation_directions(n_adsorbates: int) -> list[np.ndarray]:
    n = int(n_adsorbates)
    if n <= 0:
        return []
    if n == 1:
        return [np.array([1.0, 0.0, 0.0], dtype=float)]
    if n == 2:
        return [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([-1.0, 0.0, 0.0], dtype=float),
        ]
    if n == 3:
        return [
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([-0.5, 0.8660254037844386, 0.0], dtype=float),
            np.array([-0.5, -0.8660254037844386, 0.0], dtype=float),
        ]
    if n == 4:
        tetra = np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=float,
        )
        tetra /= np.linalg.norm(tetra, axis=1)[:, None]
        return [np.asarray(vec, dtype=float) for vec in tetra]
    return _fibonacci_sphere_directions(n)


def _build_com_direction_trials(
    *,
    n_adsorbates: int,
    preferred_axis: np.ndarray,
    max_trials: int,
) -> list[list[np.ndarray]]:
    base = _base_separation_directions(n_adsorbates)
    if not base:
        return []

    align = _rotation_matrix_align_vectors(base[0], preferred_axis)
    euler_triplets = [
        (0.0, 0.0, 0.0),
        (35.0, 0.0, 0.0),
        (0.0, 35.0, 0.0),
        (0.0, 0.0, 35.0),
        (50.0, 25.0, 10.0),
        (80.0, 15.0, 45.0),
        (120.0, 0.0, 30.0),
        (150.0, 40.0, 70.0),
        (210.0, 35.0, 20.0),
        (260.0, 55.0, 15.0),
        (310.0, 20.0, 65.0),
        (330.0, 45.0, 80.0),
    ]
    trials: list[list[np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for alpha_deg, beta_deg, gamma_deg in euler_triplets:
        if len(trials) >= int(max_trials):
            break
        alpha = float(np.deg2rad(alpha_deg))
        beta = float(np.deg2rad(beta_deg))
        gamma = float(np.deg2rad(gamma_deg))
        rz = np.array(
            [
                [np.cos(alpha), -np.sin(alpha), 0.0],
                [np.sin(alpha), np.cos(alpha), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )
        ry = np.array(
            [
                [np.cos(beta), 0.0, np.sin(beta)],
                [0.0, 1.0, 0.0],
                [-np.sin(beta), 0.0, np.cos(beta)],
            ],
            dtype=float,
        )
        rx = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(gamma), -np.sin(gamma)],
                [0.0, np.sin(gamma), np.cos(gamma)],
            ],
            dtype=float,
        )
        rotation = rz @ ry @ rx @ align
        directions = []
        for vector in base:
            rotated = rotation @ vector
            norm = float(np.linalg.norm(rotated))
            if norm <= 1.0e-12:
                rotated = np.array([0.0, 0.0, 1.0], dtype=float)
                norm = 1.0
            directions.append(np.asarray(rotated, dtype=float) / norm)
        key = tuple(np.round(np.concatenate(directions), decimals=6).tolist())
        if key in seen:
            continue
        seen.add(key)
        trials.append(directions)
    if not trials:
        trials.append(base)
    return trials


def _shift_group_inside_radius(
    atoms: Atoms,
    group_indices: Sequence[int],
    *,
    center: np.ndarray,
    radius: float,
    padding: float,
) -> None:
    if not group_indices:
        return
    allowed = max(float(radius) - float(padding), 0.0)
    if allowed <= 0.0:
        return
    positions = atoms.positions[list(group_indices)]
    distances = np.linalg.norm(positions - center, axis=1)
    max_distance = float(np.max(distances))
    if max_distance <= allowed:
        return

    shift_magnitude = max_distance - allowed
    group_com = positions.mean(axis=0)
    direction = center - group_com
    norm = float(np.linalg.norm(direction))
    if norm <= 1.0e-12:
        return
    shift = (direction / norm) * shift_magnitude
    atoms.positions[list(group_indices)] += shift


def _minimum_framework_adsorbate_distance(
    atoms: Atoms,
    framework_indices: Sequence[int],
    adsorbate_indices: Sequence[int],
) -> float | None:
    if not framework_indices or not adsorbate_indices:
        return None
    framework_positions = atoms.positions[list(framework_indices)]
    ads_positions = atoms.positions[list(adsorbate_indices)]
    diffs = framework_positions[:, None, :] - ads_positions[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return float(np.min(distances))


def _maximum_adsorbate_distance_to_com(
    atoms: Atoms,
    adsorbate_indices: Sequence[int],
    *,
    cage_com: np.ndarray,
) -> float | None:
    if not adsorbate_indices:
        return None
    ads_positions = atoms.positions[list(adsorbate_indices)]
    distances = np.linalg.norm(ads_positions - cage_com, axis=1)
    return float(np.max(distances))


def _minimum_distance_between_groups(
    atoms: Atoms,
    group_a: Sequence[int],
    group_b: Sequence[int],
) -> float | None:
    if not group_a or not group_b:
        return None
    pos_a = atoms.positions[list(group_a)]
    pos_b = atoms.positions[list(group_b)]
    diffs = pos_a[:, None, :] - pos_b[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    return float(np.min(distances))


def _closest_pair_vector_between_groups(
    atoms: Atoms,
    group_a: Sequence[int],
    group_b: Sequence[int],
) -> tuple[float, np.ndarray] | None:
    if not group_a or not group_b:
        return None
    indices_a = list(group_a)
    indices_b = list(group_b)
    pos_a = atoms.positions[indices_a]
    pos_b = atoms.positions[indices_b]
    diffs = pos_a[:, None, :] - pos_b[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    if distances.size == 0:
        return None
    ia, ib = np.unravel_index(int(np.argmin(distances)), distances.shape)
    min_distance = float(distances[ia, ib])
    vector = atoms.positions[indices_a[ia]] - atoms.positions[indices_b[ib]]
    return min_distance, np.asarray(vector, dtype=float)


def _minimum_adsorbate_adsorbate_distance(
    atoms: Atoms,
    adsorbate_groups: Sequence[Sequence[int]],
) -> float | None:
    if len(adsorbate_groups) < 2:
        return None
    min_distance: float | None = None
    for idx_a in range(len(adsorbate_groups)):
        for idx_b in range(idx_a + 1, len(adsorbate_groups)):
            distance = _minimum_distance_between_groups(
                atoms,
                adsorbate_groups[idx_a],
                adsorbate_groups[idx_b],
            )
            if distance is None:
                continue
            if min_distance is None or distance < min_distance:
                min_distance = float(distance)
    return min_distance


def _minimum_inter_adsorbate_bond_ratio(
    atoms: Atoms,
    adsorbate_groups: Sequence[Sequence[int]],
) -> tuple[float | None, tuple[int, int, float, float, np.ndarray] | None]:
    """
    Return the minimum ratio distance/(r_cov_i + r_cov_j) across atoms belonging
    to different adsorbate groups, plus metadata for the most critical pair:
    (group_a_idx, group_b_idx, distance, covalent_cutoff, vector_i_minus_j).
    """
    if len(adsorbate_groups) < 2:
        return None, None

    numbers = atoms.get_atomic_numbers()
    min_ratio: float | None = None
    min_payload: tuple[int, int, float, float, np.ndarray] | None = None

    for group_a_idx in range(len(adsorbate_groups)):
        group_a = list(adsorbate_groups[group_a_idx])
        if not group_a:
            continue
        for group_b_idx in range(group_a_idx + 1, len(adsorbate_groups)):
            group_b = list(adsorbate_groups[group_b_idx])
            if not group_b:
                continue
            for atom_i in group_a:
                radius_i = float(covalent_radii[numbers[atom_i]])
                for atom_j in group_b:
                    radius_j = float(covalent_radii[numbers[atom_j]])
                    covalent_cutoff = radius_i + radius_j
                    if covalent_cutoff <= 1.0e-12:
                        continue
                    vector = atoms.positions[atom_i] - atoms.positions[atom_j]
                    distance = float(np.linalg.norm(vector))
                    ratio = distance / covalent_cutoff
                    if min_ratio is None or ratio < min_ratio:
                        min_ratio = float(ratio)
                        min_payload = (
                            int(group_a_idx),
                            int(group_b_idx),
                            float(distance),
                            float(covalent_cutoff),
                            np.asarray(vector, dtype=float),
                        )
    return min_ratio, min_payload


def _nearest_framework_vector_to_group(
    atoms: Atoms,
    framework_indices: Sequence[int],
    group_indices: Sequence[int],
) -> tuple[float, np.ndarray] | None:
    if not framework_indices or not group_indices:
        return None
    fw_indices = list(framework_indices)
    group = list(group_indices)
    fw_pos = atoms.positions[fw_indices]
    group_pos = atoms.positions[group]
    diffs = group_pos[:, None, :] - fw_pos[None, :, :]
    distances = np.linalg.norm(diffs, axis=2)
    if distances.size == 0:
        return None
    i_group, i_fw = np.unravel_index(int(np.argmin(distances)), distances.shape)
    min_distance = float(distances[i_group, i_fw])
    vector = atoms.positions[group[i_group]] - atoms.positions[fw_indices[i_fw]]
    return min_distance, np.asarray(vector, dtype=float)


def _resolve_adsorbate_overlaps(
    atoms: Atoms,
    *,
    framework_indices: Sequence[int],
    adsorbate_groups: Sequence[Sequence[int]],
    cage_com: np.ndarray,
    target_radius: float,
    cage_padding: float,
    min_framework_distance: float,
    min_adsorbate_distance: float | None = None,
    enforce_nonbonded_coadsorbates: bool = False,
    nonbonded_coadsorbate_scale: float = 1.10,
    max_steps: int = 24,
    verbose: bool = False,
    log_prefix: str = "[ZeoliteWorkflow]",
    candidate_label: str | None = None,
    progress_interval: int = 5,
) -> tuple[bool, float | None, float | None, int]:
    label = "" if candidate_label is None else f" {candidate_label}"
    progress_step = max(1, int(progress_interval))
    group_count = len([group for group in adsorbate_groups if group])
    if max_steps <= 0:
        fw_min = _minimum_framework_adsorbate_distance(
            atoms,
            framework_indices=framework_indices,
            adsorbate_indices=[idx for group in adsorbate_groups for idx in group],
        )
        aa_min = _minimum_adsorbate_adsorbate_distance(atoms, adsorbate_groups)
        min_bond_ratio, _ = _minimum_inter_adsorbate_bond_ratio(atoms, adsorbate_groups)
        fw_ok = fw_min is None or fw_min >= float(min_framework_distance)
        aa_ok = True
        if min_adsorbate_distance is not None:
            aa_ok = aa_min is None or aa_min >= float(min_adsorbate_distance)
        nonbond_ok = True
        if enforce_nonbonded_coadsorbates:
            nonbond_ok = (
                min_bond_ratio is None
                or min_bond_ratio >= float(nonbonded_coadsorbate_scale)
            )
        resolved = bool(fw_ok and aa_ok and nonbond_ok)
        if verbose:
            print(
                f"{log_prefix} overlap-resolve{label}: "
                f"steps=0, resolved={resolved}, "
                f"groups={group_count}, "
                f"min_fw={_fmt_optional(fw_min, '.4f', ' A')}, "
                f"min_ads={_fmt_optional(aa_min, '.4f', ' A')}, "
                f"min_inter_ratio={_fmt_optional(min_bond_ratio)}"
            )
        return resolved, fw_min, aa_min, 0

    groups = [list(group) for group in adsorbate_groups if group]
    if not groups:
        return True, None, None, 0

    fw_min: float | None = None
    aa_min: float | None = None
    steps_taken = 0
    for step_idx in range(1, int(max_steps) + 1):
        steps_taken = step_idx
        moved = False

        for group in groups:
            _shift_group_inside_radius(
                atoms,
                group,
                center=cage_com,
                radius=target_radius,
                padding=cage_padding,
            )

        for group in groups:
            nearest = _nearest_framework_vector_to_group(
                atoms,
                framework_indices=framework_indices,
                group_indices=group,
            )
            if nearest is None:
                continue
            distance_fw, away_vector = nearest
            if distance_fw >= float(min_framework_distance):
                continue

            away_norm = float(np.linalg.norm(away_vector))
            if away_norm <= 1.0e-12:
                group_com = atoms.positions[group].mean(axis=0)
                away_vector = group_com - cage_com
                away_norm = float(np.linalg.norm(away_vector))
            if away_norm <= 1.0e-12:
                away_vector = np.array([0.0, 0.0, 1.0], dtype=float)
                away_norm = 1.0
            away_unit = away_vector / away_norm

            group_com = atoms.positions[group].mean(axis=0)
            to_com = cage_com - group_com
            to_com_norm = float(np.linalg.norm(to_com))
            to_com_unit = away_unit if to_com_norm <= 1.0e-12 else (to_com / to_com_norm)

            direction = 0.75 * away_unit + 0.25 * to_com_unit
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm <= 1.0e-12:
                direction = away_unit
                direction_norm = 1.0
            direction /= direction_norm

            overlap = float(min_framework_distance) - float(distance_fw)
            shift = direction * (overlap + 1.0e-3)
            atoms.positions[group] += shift
            moved = True

        if min_adsorbate_distance is not None and len(groups) > 1:
            threshold = float(min_adsorbate_distance)
            for idx_a in range(len(groups)):
                for idx_b in range(idx_a + 1, len(groups)):
                    nearest = _closest_pair_vector_between_groups(
                        atoms,
                        groups[idx_a],
                        groups[idx_b],
                    )
                    if nearest is None:
                        continue
                    distance_ab, vec_ab = nearest
                    if distance_ab >= threshold:
                        continue
                    vec_norm = float(np.linalg.norm(vec_ab))
                    if vec_norm <= 1.0e-12:
                        com_a = atoms.positions[groups[idx_a]].mean(axis=0)
                        com_b = atoms.positions[groups[idx_b]].mean(axis=0)
                        vec_ab = com_a - com_b
                        vec_norm = float(np.linalg.norm(vec_ab))
                    if vec_norm <= 1.0e-12:
                        vec_ab = np.array([1.0, 0.0, 0.0], dtype=float)
                        vec_norm = 1.0
                    direction = vec_ab / vec_norm
                    overlap = threshold - float(distance_ab)
                    shift = direction * (0.5 * overlap + 1.0e-3)
                    atoms.positions[groups[idx_a]] += shift
                    atoms.positions[groups[idx_b]] -= shift
                    moved = True

        if enforce_nonbonded_coadsorbates and len(groups) > 1:
            min_bond_ratio, min_payload = _minimum_inter_adsorbate_bond_ratio(atoms, groups)
            if (
                min_bond_ratio is not None
                and min_payload is not None
                and min_bond_ratio < float(nonbonded_coadsorbate_scale)
            ):
                group_a_idx, group_b_idx, distance, covalent_cutoff, vector_ab = min_payload
                vec_norm = float(np.linalg.norm(vector_ab))
                if vec_norm <= 1.0e-12:
                    com_a = atoms.positions[groups[group_a_idx]].mean(axis=0)
                    com_b = atoms.positions[groups[group_b_idx]].mean(axis=0)
                    vector_ab = com_a - com_b
                    vec_norm = float(np.linalg.norm(vector_ab))
                if vec_norm <= 1.0e-12:
                    vector_ab = np.array([1.0, 0.0, 0.0], dtype=float)
                    vec_norm = 1.0
                direction = vector_ab / vec_norm
                target_distance = float(nonbonded_coadsorbate_scale) * float(covalent_cutoff)
                overlap = target_distance - float(distance)
                if overlap > 0.0:
                    shift = direction * (0.5 * overlap + 1.0e-3)
                    atoms.positions[groups[group_a_idx]] += shift
                    atoms.positions[groups[group_b_idx]] -= shift
                    moved = True

        for group in groups:
            _shift_group_inside_radius(
                atoms,
                group,
                center=cage_com,
                radius=target_radius,
                padding=cage_padding,
            )

        adsorbate_indices = [idx for group in groups for idx in group]
        fw_min = _minimum_framework_adsorbate_distance(
            atoms,
            framework_indices=framework_indices,
            adsorbate_indices=adsorbate_indices,
        )
        aa_min = _minimum_adsorbate_adsorbate_distance(atoms, groups)
        min_bond_ratio, _ = _minimum_inter_adsorbate_bond_ratio(atoms, groups)
        fw_ok = fw_min is None or fw_min >= float(min_framework_distance)
        aa_ok = True
        if min_adsorbate_distance is not None:
            aa_ok = aa_min is None or aa_min >= float(min_adsorbate_distance)
        nonbond_ok = True
        if enforce_nonbonded_coadsorbates:
            nonbond_ok = (
                min_bond_ratio is None
                or min_bond_ratio >= float(nonbonded_coadsorbate_scale)
            )
        if verbose and (
            step_idx == 1
            or step_idx % progress_step == 0
            or step_idx == int(max_steps)
            or (fw_ok and aa_ok and nonbond_ok)
        ):
            print(
                f"{log_prefix} overlap-resolve{label} step {step_idx}/{int(max_steps)}: "
                f"moved={moved}, fw_ok={fw_ok}, aa_ok={aa_ok}, nonbond_ok={nonbond_ok}, "
                f"groups={group_count}, "
                f"min_fw={_fmt_optional(fw_min, '.4f', ' A')}, "
                f"min_ads={_fmt_optional(aa_min, '.4f', ' A')}, "
                f"min_inter_ratio={_fmt_optional(min_bond_ratio)}"
            )

        if fw_ok and aa_ok and nonbond_ok:
            return True, fw_min, aa_min, steps_taken
        if not moved:
            break

    if verbose:
        final_ratio, _ = _minimum_inter_adsorbate_bond_ratio(atoms, groups)
        print(
            f"{log_prefix} overlap-resolve{label}: unresolved after {steps_taken} step(s), "
            f"groups={group_count}, "
            f"min_fw={_fmt_optional(fw_min, '.4f', ' A')}, "
            f"min_ads={_fmt_optional(aa_min, '.4f', ' A')}, "
            f"min_inter_ratio={_fmt_optional(final_ratio)}"
        )
    return False, fw_min, aa_min, steps_taken


def _infer_bonds(template: Atoms, *, scale: float = 1.20) -> list[tuple[int, int, float]]:
    bonds: list[tuple[int, int, float]] = []
    numbers = template.get_atomic_numbers()
    for i in range(len(template)):
        for j in range(i + 1, len(template)):
            distance = float(template.get_distance(i, j))
            cutoff = scale * (covalent_radii[numbers[i]] + covalent_radii[numbers[j]])
            if distance <= cutoff:
                bonds.append((i, j, distance))
    if not bonds and len(template) == 2:
        bonds.append((0, 1, float(template.get_distance(0, 1))))
    return bonds


def _adsorbate_integrity_check(
    atoms: Atoms,
    *,
    adsorbate_groups: Sequence[Sequence[int]],
    adsorbate_templates: Sequence[Atoms],
    bond_mode: str = "all",
    min_ratio: float = 0.65,
    max_ratio: float = 1.60,
) -> tuple[bool, float | None]:
    mode = str(bond_mode).strip().lower()
    if mode not in {"nearest", "all"}:
        raise ValueError(
            "bond_mode must be 'nearest' or 'all', "
            f"received '{bond_mode}'."
        )
    min_seen_ratio: float | None = None
    max_seen_ratio: float | None = None
    for group, template in zip(adsorbate_groups, adsorbate_templates):
        if len(group) != len(template) or len(group) <= 1:
            continue
        bonds = _infer_bonds(template)
        if mode == "nearest":
            bonds = _select_nearest_neighbor_bonds(template, bonds)
        for i_local, j_local, ref_distance in bonds:
            if ref_distance <= 1.0e-12:
                continue
            current = float(atoms.get_distance(group[i_local], group[j_local]))
            ratio = current / float(ref_distance)
            if min_seen_ratio is None or ratio < min_seen_ratio:
                min_seen_ratio = float(ratio)
            if max_seen_ratio is None or ratio > max_seen_ratio:
                max_seen_ratio = float(ratio)
    if min_seen_ratio is None or max_seen_ratio is None:
        return True, None
    integrity_ok = bool(
        float(min_ratio) <= float(min_seen_ratio) and float(max_seen_ratio) <= float(max_ratio)
    )
    worst_ratio = float(max(max_seen_ratio, 1.0 / max(min_seen_ratio, 1.0e-12)))
    return integrity_ok, worst_ratio


def _select_nearest_neighbor_bonds(
    template: Atoms,
    bonds: Sequence[tuple[int, int, float]],
) -> list[tuple[int, int, float]]:
    if not bonds:
        return []
    selected: dict[tuple[int, int], float] = {}
    for atom_idx in range(len(template)):
        local = [bond for bond in bonds if atom_idx in (bond[0], bond[1])]
        if not local:
            continue
        i, j, distance = min(local, key=lambda item: float(item[2]))
        key = (i, j) if i < j else (j, i)
        if key not in selected or float(distance) < float(selected[key]):
            selected[key] = float(distance)
    return [(i, j, dist) for (i, j), dist in sorted(selected.items())]


def _infer_angles_from_bonds(
    template: Atoms,
    bonds: Sequence[tuple[int, int, float]],
    *,
    min_angle_deg: float = 10.0,
    max_angle_deg: float = 170.0,
) -> list[tuple[int, int, int, float]]:
    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(template))}
    for i, j, _ in bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)

    angles: list[tuple[int, int, int, float]] = []
    for center, neighbors in adjacency.items():
        sorted_neighbors = sorted(neighbors)
        for left_idx, right_idx in combinations(sorted_neighbors, 2):
            angle = float(template.get_angle(left_idx, center, right_idx))
            # Avoid near-linear angle constraints that can make FixInternals unstable.
            if angle <= float(min_angle_deg) or angle >= float(max_angle_deg):
                continue
            angles.append((left_idx, center, right_idx, angle))
    return angles


def _build_fixinternals_constraint(
    bond_constraints: list[tuple[int, int, float]],
    angle_constraints: list[tuple[int, int, int, float]],
    *,
    fixinternals_epsilon: float | None = None,
    fixinternals_maxiter: int | None = None,
):
    if not bond_constraints and not angle_constraints:
        return None

    bond_defs = [[dist, [i, j]] for i, j, dist in bond_constraints]
    angle_defs = [[angle, [i, j, k]] for i, j, k, angle in angle_constraints]

    extras = {}
    if fixinternals_epsilon is not None:
        extras["epsilon"] = float(fixinternals_epsilon)
    if fixinternals_maxiter is not None:
        extras["maxiter"] = int(fixinternals_maxiter)

    signatures = (
        {"bonds": bond_defs, "angles_deg": angle_defs, **extras},
        {"bonds": bond_defs, "angles": angle_defs, **extras},
        {"bonds": bond_defs, "angles_deg": angle_defs},
        {"bonds": bond_defs, "angles": angle_defs},
    )
    for kwargs in signatures:
        try:
            clean_kwargs = {key: value for key, value in kwargs.items() if value}
            return FixInternals(**clean_kwargs)
        except TypeError:
            continue
        except Exception as exc:
            print(f"Warning: Failed to build FixInternals constraint: {exc}")
            return None

    print("Warning: Unable to construct FixInternals constraints with current ASE signature.")
    return None


def _build_adsorbate_internal_constraint(
    *,
    adsorbate_groups: Sequence[Sequence[int]],
    adsorbate_templates: Sequence[Atoms],
    constrain_angles: bool,
    bond_mode: str = "nearest",
    fixinternals_epsilon: float | None = None,
    fixinternals_maxiter: int | None = None,
):
    mode = str(bond_mode).strip().lower()
    if mode not in {"nearest", "all"}:
        raise ValueError(
            "bond_mode must be 'nearest' or 'all', "
            f"received '{bond_mode}'."
        )
    all_bonds: list[tuple[int, int, float]] = []
    all_angles: list[tuple[int, int, int, float]] = []
    for group, template in zip(adsorbate_groups, adsorbate_templates):
        if len(group) != len(template):
            print(
                "Warning: Adsorbate group/template size mismatch; skipping internal constraints "
                f"for group size {len(group)} and template size {len(template)}."
            )
            continue
        local_bonds = _infer_bonds(template)
        if mode == "nearest":
            local_bonds = _select_nearest_neighbor_bonds(template, local_bonds)
        for i_local, j_local, distance in local_bonds:
            all_bonds.append((group[i_local], group[j_local], distance))
        if constrain_angles:
            local_angles = _infer_angles_from_bonds(template, local_bonds)
            for i_local, j_local, k_local, angle in local_angles:
                all_angles.append((group[i_local], group[j_local], group[k_local], angle))
    return _build_fixinternals_constraint(
        all_bonds,
        all_angles,
        fixinternals_epsilon=fixinternals_epsilon,
        fixinternals_maxiter=fixinternals_maxiter,
    )


def _generate_com_separated_candidates(
    *,
    framework: Atoms,
    adsorbate_specs: Sequence[ZeoliteAdsorbateSpec],
    cage_com: np.ndarray,
    cage_boundary_radius: float,
    site_selection_radius: float,
    cage_padding: float,
    min_initial_adsorbate_framework_distance: float,
    min_initial_adsorbate_adsorbate_distance: float | None,
    resolve_initial_overlaps: bool,
    overlap_resolution_max_steps: int,
    enforce_nonbonded_coadsorbates: bool,
    nonbonded_coadsorbate_scale: float,
    max_structures: int | None,
    trial_count: int = 12,
    radial_steps: int = 6,
    force_emit_best_effort: bool = True,
    verbose: bool = False,
    placement_progress_interval: int = 25,
    overlap_detailed_logging: bool = False,
    overlap_progress_interval: int = 5,
):
    n_adsorbates = len(adsorbate_specs)
    if n_adsorbates == 0:
        return

    emitted = 0
    processed_candidates = 0
    rejected_overlap = 0
    rejected_initial_distances = 0
    rejected_nonbonded = 0
    adsorbate_orders = _unique_adsorbate_orders(adsorbate_specs)
    framework_indices = list(range(len(framework)))
    preferred_axis = _estimate_open_direction_from_framework(
        framework,
        cage_com=cage_com,
        n_nearest=24,
    )
    direction_trials = _build_com_direction_trials(
        n_adsorbates=n_adsorbates,
        preferred_axis=preferred_axis,
        max_trials=max(1, int(trial_count)),
    )
    radial_factors = np.linspace(0.20, 0.90, num=max(1, int(radial_steps))).tolist()
    if 0.0 not in radial_factors:
        radial_factors = [0.0] + radial_factors

    estimated_total_candidates = (
        len(adsorbate_orders) * len(direction_trials) * len(radial_factors)
    )
    progress_interval = max(1, int(placement_progress_interval))

    best_failed_payload = None
    best_failed_score = -float("inf")

    def _maybe_print_progress(force: bool = False) -> None:
        if not verbose:
            return
        if not force and (processed_candidates == 0 or processed_candidates % progress_interval != 0):
            return
        total_text = (
            f"/{estimated_total_candidates}"
            if estimated_total_candidates > 0
            else ""
        )
        print(
            "[ZeoliteWorkflow] COM placement progress: "
            f"processed={processed_candidates}{total_text}, "
            f"kept={emitted}, "
            f"rejected_overlap={rejected_overlap}, "
            f"rejected_distances={rejected_initial_distances}, "
            f"rejected_nonbonded={rejected_nonbonded}"
        )

    if verbose:
        print(
            "[ZeoliteWorkflow] COM placement stage started: "
            f"adsorbate_orders={len(adsorbate_orders)}, "
            f"direction_trials={len(direction_trials)}, "
            f"radial_steps={len(radial_factors)}, "
            f"estimated_candidates={estimated_total_candidates}"
        )

    for adsorbate_order in adsorbate_orders:
        max_radii = [
            max(
                float(cage_boundary_radius) - float(cage_padding) - _adsorbate_max_extent_from_com(spec.template),
                0.0,
            )
            for spec in adsorbate_order
        ]
        for direction_trial_idx, directions in enumerate(direction_trials, start=1):
            for radial_factor in radial_factors:
                processed_candidates += 1
                candidate = framework.copy()
                adsorbate_groups: list[list[int]] = []
                adsorbate_templates: list[Atoms] = []
                placement_labels: list[str] = []

                for local_idx, (adsorbate_spec, direction, max_radius) in enumerate(
                    zip(adsorbate_order, directions, max_radii)
                ):
                    placed_adsorbate = adsorbate_spec.template.copy()
                    radial_boost = 0.0
                    if n_adsorbates > 1 and max_radius > 0.0:
                        radial_boost = min(0.25, 0.10 * float(local_idx)) * float(max_radius)
                    radial_distance = min(
                        float(max_radius),
                        float(radial_factor) * float(max_radius) + radial_boost,
                    )
                    placement_anchor = cage_com + np.asarray(direction, dtype=float) * radial_distance

                    delta = placement_anchor - placed_adsorbate.get_center_of_mass()
                    placed_adsorbate.translate(delta)

                    start = len(candidate)
                    candidate += placed_adsorbate
                    group_indices = list(range(start, len(candidate)))
                    _shift_group_inside_radius(
                        candidate,
                        group_indices,
                        center=cage_com,
                        radius=cage_boundary_radius,
                        padding=cage_padding,
                    )

                    adsorbate_groups.append(group_indices)
                    adsorbate_templates.append(adsorbate_spec.template)
                    placement_labels.append(
                        f"{adsorbate_spec.label}@com_trial_{direction_trial_idx}_"
                        f"r{float(radial_factor):.3f}_slot_{local_idx}"
                    )

                adsorbate_indices = [idx for group in adsorbate_groups for idx in group]
                overlap_resolved = True
                min_distance = _minimum_framework_adsorbate_distance(
                    candidate,
                    framework_indices=framework_indices,
                    adsorbate_indices=adsorbate_indices,
                )
                min_adsorbate_distance = _minimum_adsorbate_adsorbate_distance(
                    candidate,
                    adsorbate_groups=adsorbate_groups,
                )
                if resolve_initial_overlaps:
                    (
                        overlap_resolved,
                        min_distance_resolved,
                        min_adsorbate_distance_resolved,
                        overlap_steps,
                    ) = _resolve_adsorbate_overlaps(
                        candidate,
                        framework_indices=framework_indices,
                        adsorbate_groups=adsorbate_groups,
                        cage_com=cage_com,
                        target_radius=float(cage_boundary_radius),
                        cage_padding=float(cage_padding),
                        min_framework_distance=float(min_initial_adsorbate_framework_distance),
                        min_adsorbate_distance=(
                            None
                            if min_initial_adsorbate_adsorbate_distance is None
                            else float(min_initial_adsorbate_adsorbate_distance)
                        ),
                        enforce_nonbonded_coadsorbates=bool(enforce_nonbonded_coadsorbates),
                        nonbonded_coadsorbate_scale=float(nonbonded_coadsorbate_scale),
                        max_steps=int(overlap_resolution_max_steps),
                        verbose=bool(verbose and overlap_detailed_logging),
                        candidate_label=f"com-candidate {processed_candidates}",
                        progress_interval=int(overlap_progress_interval),
                    )
                    min_distance = min_distance_resolved
                    min_adsorbate_distance = min_adsorbate_distance_resolved
                else:
                    overlap_steps = 0

                min_inter_ads_ratio, _ = _minimum_inter_adsorbate_bond_ratio(
                    candidate,
                    adsorbate_groups,
                )

                framework_ok = (
                    min_distance is None
                    or min_distance >= float(min_initial_adsorbate_framework_distance)
                )
                adsorbate_ok = True
                if min_initial_adsorbate_adsorbate_distance is not None:
                    adsorbate_ok = (
                        min_adsorbate_distance is None
                        or min_adsorbate_distance >= float(min_initial_adsorbate_adsorbate_distance)
                    )
                nonbonded_ok = True
                if enforce_nonbonded_coadsorbates:
                    nonbonded_ok = (
                        min_inter_ads_ratio is None
                        or min_inter_ads_ratio >= float(nonbonded_coadsorbate_scale)
                    )

                metadata = {
                    "adsorbates": ",".join(spec.label for spec in adsorbate_order),
                    "site_indices": "",
                    "site_distances_to_cage_com": "",
                    "placements": ";".join(placement_labels),
                    "placement_strategy": "com_separation_math",
                    "com_direction_trial": int(direction_trial_idx),
                    "com_radial_factor": float(radial_factor),
                    "com_preferred_axis_x": float(preferred_axis[0]),
                    "com_preferred_axis_y": float(preferred_axis[1]),
                    "com_preferred_axis_z": float(preferred_axis[2]),
                    "cage_com_x": float(cage_com[0]),
                    "cage_com_y": float(cage_com[1]),
                    "cage_com_z": float(cage_com[2]),
                    "target_radius": float(site_selection_radius),
                    "cage_boundary_radius": float(cage_boundary_radius),
                    "insertion_distance": None,
                    "cage_padding": float(cage_padding),
                    "min_site_separation": 0.0,
                    "initial_min_framework_adsorbate_distance": (
                        None if min_distance is None else float(min_distance)
                    ),
                    "initial_min_adsorbate_adsorbate_distance": (
                        None if min_adsorbate_distance is None else float(min_adsorbate_distance)
                    ),
                    "initial_overlap_resolved": bool(overlap_resolved),
                    "initial_overlap_resolution_steps": int(overlap_steps),
                    "initial_min_inter_adsorbate_bond_ratio": (
                        None if min_inter_ads_ratio is None else float(min_inter_ads_ratio)
                    ),
                    "initial_nonbonded_coadsorbates_ok": bool(nonbonded_ok),
                    "enforce_nonbonded_coadsorbates": bool(enforce_nonbonded_coadsorbates),
                    "nonbonded_coadsorbate_scale": float(nonbonded_coadsorbate_scale),
                    "initial_best_effort_seed": False,
                }

                if overlap_resolved and framework_ok and adsorbate_ok and nonbonded_ok:
                    yield candidate, metadata, adsorbate_groups, adsorbate_templates
                    emitted += 1
                    _maybe_print_progress(force=False)
                    if max_structures is not None and emitted >= max_structures:
                        _maybe_print_progress(force=True)
                        if verbose:
                            print(
                                "[ZeoliteWorkflow] COM placement stage completed early: "
                                f"reached max_structures={int(max_structures)}"
                            )
                        return
                    continue

                if not overlap_resolved:
                    rejected_overlap += 1
                elif not nonbonded_ok:
                    rejected_nonbonded += 1
                else:
                    rejected_initial_distances += 1
                _maybe_print_progress(force=False)

                if bool(force_emit_best_effort):
                    fw_gap = (
                        -1.0e6
                        if min_distance is None
                        else float(min_distance) - float(min_initial_adsorbate_framework_distance)
                    )
                    ads_gap = 0.0
                    if min_initial_adsorbate_adsorbate_distance is not None:
                        ads_gap = (
                            -1.0e6
                            if min_adsorbate_distance is None
                            else float(min_adsorbate_distance)
                            - float(min_initial_adsorbate_adsorbate_distance)
                        )
                    nonbond_gap = 0.0
                    if enforce_nonbonded_coadsorbates:
                        nonbond_gap = (
                            -1.0e6
                            if min_inter_ads_ratio is None
                            else float(min_inter_ads_ratio) - float(nonbonded_coadsorbate_scale)
                        )
                    score = fw_gap + 0.5 * ads_gap + 0.25 * nonbond_gap
                    if overlap_resolved:
                        score += 0.2
                    if score > best_failed_score:
                        best_failed_score = float(score)
                        best_failed_payload = (
                            candidate.copy(),
                            dict(metadata),
                            [list(group) for group in adsorbate_groups],
                            list(adsorbate_templates),
                        )

    if emitted == 0 and bool(force_emit_best_effort) and best_failed_payload is not None:
        candidate, metadata, groups, templates = best_failed_payload
        metadata["initial_best_effort_seed"] = True
        metadata["placement_strategy"] = "com_separation_math_best_effort"
        if verbose:
            print(
                "[ZeoliteWorkflow] Warning: no COM placement candidate passed all "
                "initial thresholds; emitting best-effort seed for downstream relaxation."
            )
        yield candidate, metadata, groups, templates
        emitted += 1

    _maybe_print_progress(force=True)
    if verbose:
        print(
            "[ZeoliteWorkflow] COM placement stage completed: "
            f"processed={processed_candidates}, kept={emitted}"
        )


def _generate_zeolite_ring_candidates(
    *,
    framework: Atoms,
    site_indices: Sequence[int],
    site_distances: Sequence[float],
    adsorbate_specs: Sequence[ZeoliteAdsorbateSpec],
    cage_com: np.ndarray,
    cage_boundary_radius: float,
    site_selection_radius: float,
    insertion_distance: float,
    cage_padding: float,
    min_site_separation: float,
    min_initial_adsorbate_framework_distance: float,
    min_initial_adsorbate_adsorbate_distance: float | None,
    resolve_initial_overlaps: bool,
    overlap_resolution_max_steps: int,
    enforce_nonbonded_coadsorbates: bool,
    nonbonded_coadsorbate_scale: float,
    max_structures: int | None,
    verbose: bool = False,
    placement_progress_interval: int = 25,
    overlap_detailed_logging: bool = False,
    overlap_progress_interval: int = 5,
):
    n_adsorbates = len(adsorbate_specs)
    if len(site_indices) < n_adsorbates:
        raise ValueError(
            f"Not enough sites ({len(site_indices)}) for requested adsorbates ({n_adsorbates})."
        )

    emitted = 0
    processed_candidates = 0
    rejected_site_separation = 0
    rejected_overlap = 0
    rejected_initial_distances = 0
    rejected_nonbonded = 0
    adsorbate_orders = _unique_adsorbate_orders(adsorbate_specs)
    framework_indices = list(range(len(framework)))
    site_distance_map = {
        site_index: float(site_distance)
        for site_index, site_distance in zip(site_indices, site_distances)
    }
    total_site_combinations = comb(len(site_indices), n_adsorbates)
    estimated_total_candidates = total_site_combinations * len(adsorbate_orders)
    progress_interval = max(1, int(placement_progress_interval))

    def _maybe_print_progress(force: bool = False) -> None:
        if not verbose:
            return
        if not force and (processed_candidates == 0 or processed_candidates % progress_interval != 0):
            return
        total_text = (
            f"/{estimated_total_candidates}"
            if estimated_total_candidates > 0
            else ""
        )
        print(
            "[ZeoliteWorkflow] Placement progress: "
            f"processed={processed_candidates}{total_text}, "
            f"kept={emitted}, "
            f"rejected_site_sep={rejected_site_separation}, "
            f"rejected_overlap={rejected_overlap}, "
            f"rejected_distances={rejected_initial_distances}, "
            f"rejected_nonbonded={rejected_nonbonded}"
        )

    if verbose:
        print(
            "[ZeoliteWorkflow] Placement stage started: "
            f"site_combinations={total_site_combinations}, "
            f"adsorbate_orders={len(adsorbate_orders)}, "
            f"estimated_candidates={estimated_total_candidates}"
        )

    for selected_sites in combinations(site_indices, n_adsorbates):
        if min_site_separation > 0.0 and len(selected_sites) > 1:
            min_site_distance = min(
                float(framework.get_distance(i, j))
                for i, j in combinations(selected_sites, 2)
            )
            if min_site_distance < float(min_site_separation):
                skipped_here = len(adsorbate_orders)
                processed_candidates += skipped_here
                rejected_site_separation += skipped_here
                _maybe_print_progress(force=False)
                continue
        for adsorbate_order in adsorbate_orders:
            processed_candidates += 1
            candidate = framework.copy()
            adsorbate_groups: list[list[int]] = []
            adsorbate_templates: list[Atoms] = []
            placement_labels: list[str] = []
            for site_idx, adsorbate_spec in zip(selected_sites, adsorbate_order):
                site_position = framework[site_idx].position.copy()
                direction = _direction_to_com(site_position, cage_com)
                placement_anchor = site_position + direction * float(insertion_distance)

                placed_adsorbate = adsorbate_spec.template.copy()
                delta = placement_anchor - placed_adsorbate.get_center_of_mass()
                placed_adsorbate.translate(delta)

                start = len(candidate)
                candidate += placed_adsorbate
                group_indices = list(range(start, len(candidate)))
                _shift_group_inside_radius(
                    candidate,
                    group_indices,
                    center=cage_com,
                    radius=cage_boundary_radius,
                    padding=cage_padding,
                )

                adsorbate_groups.append(group_indices)
                adsorbate_templates.append(adsorbate_spec.template)
                placement_labels.append(f"{adsorbate_spec.label}@site_{site_idx}")

            adsorbate_indices = [idx for group in adsorbate_groups for idx in group]
            overlap_resolved = True
            min_distance = _minimum_framework_adsorbate_distance(
                candidate,
                framework_indices=framework_indices,
                adsorbate_indices=adsorbate_indices,
            )
            min_adsorbate_distance = _minimum_adsorbate_adsorbate_distance(
                candidate,
                adsorbate_groups=adsorbate_groups,
            )
            if resolve_initial_overlaps:
                (
                    overlap_resolved,
                    min_distance_resolved,
                    min_adsorbate_distance_resolved,
                    overlap_steps,
                ) = _resolve_adsorbate_overlaps(
                    candidate,
                    framework_indices=framework_indices,
                    adsorbate_groups=adsorbate_groups,
                    cage_com=cage_com,
                    target_radius=float(cage_boundary_radius),
                    cage_padding=float(cage_padding),
                    min_framework_distance=float(min_initial_adsorbate_framework_distance),
                    min_adsorbate_distance=(
                        None
                        if min_initial_adsorbate_adsorbate_distance is None
                        else float(min_initial_adsorbate_adsorbate_distance)
                    ),
                    enforce_nonbonded_coadsorbates=bool(enforce_nonbonded_coadsorbates),
                    nonbonded_coadsorbate_scale=float(nonbonded_coadsorbate_scale),
                    max_steps=int(overlap_resolution_max_steps),
                    verbose=bool(verbose and overlap_detailed_logging),
                    candidate_label=f"candidate {processed_candidates}",
                    progress_interval=int(overlap_progress_interval),
                )
                min_distance = min_distance_resolved
                min_adsorbate_distance = min_adsorbate_distance_resolved
            else:
                overlap_steps = 0

            min_inter_ads_ratio, _ = _minimum_inter_adsorbate_bond_ratio(
                candidate,
                adsorbate_groups,
            )

            framework_ok = (
                min_distance is None
                or min_distance >= float(min_initial_adsorbate_framework_distance)
            )
            adsorbate_ok = True
            if min_initial_adsorbate_adsorbate_distance is not None:
                adsorbate_ok = (
                    min_adsorbate_distance is None
                    or min_adsorbate_distance >= float(min_initial_adsorbate_adsorbate_distance)
                )
            nonbonded_ok = True
            if enforce_nonbonded_coadsorbates:
                nonbonded_ok = (
                    min_inter_ads_ratio is None
                    or min_inter_ads_ratio >= float(nonbonded_coadsorbate_scale)
                )
            if not overlap_resolved or not framework_ok or not adsorbate_ok or not nonbonded_ok:
                if not overlap_resolved:
                    rejected_overlap += 1
                elif not nonbonded_ok:
                    rejected_nonbonded += 1
                else:
                    rejected_initial_distances += 1
                _maybe_print_progress(force=False)
                continue

            metadata = {
                "adsorbates": ",".join(spec.label for spec in adsorbate_order),
                "site_indices": ",".join(str(i) for i in selected_sites),
                "site_distances_to_cage_com": ",".join(
                    f"{site_distance_map[i]:.6f}" for i in selected_sites
                ),
                "placements": ";".join(placement_labels),
                "cage_com_x": float(cage_com[0]),
                "cage_com_y": float(cage_com[1]),
                "cage_com_z": float(cage_com[2]),
                "target_radius": float(site_selection_radius),
                "cage_boundary_radius": float(cage_boundary_radius),
                "insertion_distance": float(insertion_distance),
                "cage_padding": float(cage_padding),
                "min_site_separation": float(min_site_separation),
                "initial_min_framework_adsorbate_distance": (
                    None if min_distance is None else float(min_distance)
                ),
                "initial_min_adsorbate_adsorbate_distance": (
                    None if min_adsorbate_distance is None else float(min_adsorbate_distance)
                ),
                "initial_overlap_resolved": bool(overlap_resolved),
                "initial_overlap_resolution_steps": int(overlap_steps),
                "initial_min_inter_adsorbate_bond_ratio": (
                    None if min_inter_ads_ratio is None else float(min_inter_ads_ratio)
                ),
                "initial_nonbonded_coadsorbates_ok": bool(nonbonded_ok),
                "enforce_nonbonded_coadsorbates": bool(enforce_nonbonded_coadsorbates),
                "nonbonded_coadsorbate_scale": float(nonbonded_coadsorbate_scale),
            }

            yield candidate, metadata, adsorbate_groups, adsorbate_templates
            emitted += 1
            _maybe_print_progress(force=False)
            if max_structures is not None and emitted >= max_structures:
                _maybe_print_progress(force=True)
                if verbose:
                    print(
                        "[ZeoliteWorkflow] Placement stage completed early: "
                        f"reached max_structures={int(max_structures)}"
                    )
                return
    _maybe_print_progress(force=True)
    if verbose:
        print(
            "[ZeoliteWorkflow] Placement stage completed: "
            f"processed={processed_candidates}, kept={emitted}"
        )


def _relax_with_mace(
    atoms_list: Sequence[Atoms],
    *,
    framework_indices: Sequence[int],
    adsorbate_groups_list: Sequence[Sequence[Sequence[int]]],
    adsorbate_templates_list: Sequence[Sequence[Atoms]],
    model_path: str,
    fmax: float,
    max_steps: int,
    device: str,
    dispersion: bool,
    default_dtype: str,
    constrain_adsorbate_internals: bool,
    constrain_adsorbate_angles: bool,
    min_physical_adsorbate_framework_distance: float,
    adsorbate_bond_constraint_mode: str = "nearest",
    strict_adsorbate_integrity: bool = False,
    allow_internals_fallback_to_framework_only: bool = True,
    integrity_min_bond_ratio: float = 0.65,
    integrity_max_bond_ratio: float = 1.60,
    fixinternals_epsilon: float | None = None,
    fixinternals_maxiter: int | None = None,
    fixinternals_relaxed_retry: bool = True,
    fixinternals_relaxed_epsilon: float = 1.0e-4,
    fixinternals_relaxed_maxiter: int = 800,
    fixinternals_relaxed_step_scale: float = 0.5,
    cage_com: np.ndarray | None = None,
    target_radius: float | None = None,
    cage_padding: float = 0.2,
    inside_radius_tolerance: float = 0.1,
    resolve_overlaps_before_relax: bool = True,
    pre_relax_overlap_max_steps: int = 24,
    min_pre_relax_adsorbate_adsorbate_distance: float | None = None,
    enforce_nonbonded_coadsorbates: bool = True,
    nonbonded_coadsorbate_scale: float = 1.10,
    constrained_maxstep: float = 0.01,
    unconstrained_maxstep: float = 0.01,
    output_db_path: Path | None = None,
    metadata_rows: Sequence[dict] | None = None,
    trajectory_dir: Path | None = None,
    trajectory_prefix: str = "relax",
    verbose: bool = True,
):
    from mace.calculators import mace_mp

    try:
        calculator = mace_mp(
            model=model_path,
            dispersion=dispersion,
            default_dtype=default_dtype,
            device=device,
            enable_cueq=True,
        )
    except TypeError as exc:
        if "enable_cueq" not in str(exc):
            raise
        if verbose:
            print("[MACE] Warning: 'enable_cueq' not supported by installed MACE; continuing without it.")
        calculator = mace_mp(
            model=model_path,
            dispersion=dispersion,
            default_dtype=default_dtype,
            device=device,
        )

    relaxed_atoms: list[Atoms] = []
    rows: list[dict[str, object]] = []
    if trajectory_dir is not None:
        trajectory_dir.mkdir(parents=True, exist_ok=True)

    framework_indices_list = list(framework_indices)

    def _build_constraints_for_attempt(
        *,
        groups: Sequence[Sequence[int]],
        templates: Sequence[Atoms],
        include_internals: bool,
        include_angles: bool,
        attempt_fixinternals_epsilon: float | None,
        attempt_fixinternals_maxiter: int | None,
    ):
        constraints = [FixAtoms(indices=framework_indices_list)]
        internal_applied = False
        if include_internals:
            internal_constraint = _build_adsorbate_internal_constraint(
                adsorbate_groups=groups,
                adsorbate_templates=templates,
                constrain_angles=include_angles,
                bond_mode=adsorbate_bond_constraint_mode,
                fixinternals_epsilon=attempt_fixinternals_epsilon,
                fixinternals_maxiter=attempt_fixinternals_maxiter,
            )
            if internal_constraint is not None:
                constraints.append(internal_constraint)
                internal_applied = True
        return (
            constraints if len(constraints) > 1 else constraints[0],
            internal_applied,
        )

    def _run_relax(db_context=None):
        n_structures = len(atoms_list)
        for idx, atoms in enumerate(atoms_list, start=1):
            work_seed = atoms.copy()
            groups = adsorbate_groups_list[idx - 1] if idx - 1 < len(adsorbate_groups_list) else []
            templates = (
                adsorbate_templates_list[idx - 1] if idx - 1 < len(adsorbate_templates_list) else []
            )
            adsorbate_indices = [i for group in groups for i in group]

            pre_relax_overlap_resolved = None
            pre_relax_min_framework_adsorbate_distance = None
            pre_relax_min_adsorbate_adsorbate_distance = None
            pre_relax_overlap_steps = 0
            pre_relax_min_inter_adsorbate_bond_ratio = None
            if (
                resolve_overlaps_before_relax
                and adsorbate_indices
                and cage_com is not None
                and target_radius is not None
            ):
                (
                    pre_relax_overlap_resolved,
                    pre_relax_min_framework_adsorbate_distance,
                    pre_relax_min_adsorbate_adsorbate_distance,
                    pre_relax_overlap_steps,
                ) = _resolve_adsorbate_overlaps(
                    work_seed,
                    framework_indices=framework_indices_list,
                    adsorbate_groups=groups,
                    cage_com=cage_com,
                    target_radius=float(target_radius),
                    cage_padding=float(cage_padding),
                    min_framework_distance=float(min_physical_adsorbate_framework_distance),
                    min_adsorbate_distance=(
                        None
                        if min_pre_relax_adsorbate_adsorbate_distance is None
                        else float(min_pre_relax_adsorbate_adsorbate_distance)
                    ),
                    enforce_nonbonded_coadsorbates=bool(enforce_nonbonded_coadsorbates),
                    nonbonded_coadsorbate_scale=float(nonbonded_coadsorbate_scale),
                    max_steps=int(pre_relax_overlap_max_steps),
                )
                pre_relax_min_inter_adsorbate_bond_ratio, _ = _minimum_inter_adsorbate_bond_ratio(
                    work_seed,
                    groups,
                )
            if verbose:
                internals_text = (
                    "off"
                    if not constrain_adsorbate_internals
                    else (
                        "on("
                        f"mode={adsorbate_bond_constraint_mode}, "
                        f"angles={bool(constrain_adsorbate_angles)}, "
                        f"fi_eps={_fmt_optional(fixinternals_epsilon, '.1e') if fixinternals_epsilon is not None else 'default'}, "
                        f"fi_maxiter={int(fixinternals_maxiter) if fixinternals_maxiter is not None else 'default'}"
                        ")"
                    )
                )
                print(
                    f"[MACE] {trajectory_prefix} structure {idx}/{n_structures} "
                    f"(framework_fixed={len(framework_indices)}, ads_groups={len(groups)}, ads_atoms={len(adsorbate_indices)}, "
                    f"internals={internals_text})"
                )
                if pre_relax_overlap_resolved is not None:
                    print(
                        f"[MACE] {trajectory_prefix} pre-resolve {idx}/{n_structures}: "
                        f"resolved={pre_relax_overlap_resolved}, "
                        f"steps={int(pre_relax_overlap_steps)}, "
                        f"min_fw={_fmt_optional(pre_relax_min_framework_adsorbate_distance, '.4f', ' A')}, "
                        f"min_ads={_fmt_optional(pre_relax_min_adsorbate_adsorbate_distance, '.4f', ' A')}, "
                        f"min_inter_ratio={_fmt_optional(pre_relax_min_inter_adsorbate_bond_ratio)}"
                    )

            trajectory = None
            logfile = None
            if trajectory_dir is not None:
                trajectory = str(trajectory_dir / f"{trajectory_prefix}_{idx}.traj")
                logfile = str(trajectory_dir / f"{trajectory_prefix}_{idx}.log")

            attempt_specs: list[dict[str, object]] = []
            if constrain_adsorbate_internals:
                fallback_step_floor = 1.0e-3
                base_fi_epsilon = (
                    None if fixinternals_epsilon is None else float(fixinternals_epsilon)
                )
                base_fi_maxiter = (
                    None if fixinternals_maxiter is None else int(fixinternals_maxiter)
                )
                attempt_specs.append(
                    {
                        "mode": (
                            "internals+angles" if constrain_adsorbate_angles else "internals"
                        ),
                        "use_internals": True,
                        "use_angles": bool(constrain_adsorbate_angles),
                        "maxstep": float(constrained_maxstep),
                        "fixinternals_epsilon": base_fi_epsilon,
                        "fixinternals_maxiter": base_fi_maxiter,
                    }
                )
                relaxed_eps = None
                relaxed_maxiter = None
                relaxed_step = None
                if bool(fixinternals_relaxed_retry):
                    relaxed_eps = max(
                        float(fixinternals_relaxed_epsilon),
                        0.0 if base_fi_epsilon is None else float(base_fi_epsilon),
                    )
                    relaxed_maxiter = max(
                        int(fixinternals_relaxed_maxiter),
                        0 if base_fi_maxiter is None else int(base_fi_maxiter),
                    )
                    relaxed_step = max(
                        fallback_step_floor,
                        float(constrained_maxstep) * float(fixinternals_relaxed_step_scale),
                    )
                    if constrain_adsorbate_angles:
                        attempt_specs.append(
                            {
                                "mode": "internals+angles_relaxed_solver",
                                "use_internals": True,
                                "use_angles": True,
                                "maxstep": float(relaxed_step),
                                "fixinternals_epsilon": float(relaxed_eps),
                                "fixinternals_maxiter": int(relaxed_maxiter),
                            }
                        )
                if constrain_adsorbate_angles:
                    attempt_specs.append(
                        {
                            "mode": "internals_only",
                            "use_internals": True,
                            "use_angles": False,
                            "maxstep": max(
                                fallback_step_floor,
                                float(constrained_maxstep) * 0.8,
                            ),
                            "fixinternals_epsilon": base_fi_epsilon,
                            "fixinternals_maxiter": base_fi_maxiter,
                        }
                    )
                if bool(fixinternals_relaxed_retry):
                    attempt_specs.append(
                        {
                            "mode": "internals_relaxed_solver",
                            "use_internals": True,
                            "use_angles": False,
                            "maxstep": float(relaxed_step),
                            "fixinternals_epsilon": float(relaxed_eps),
                            "fixinternals_maxiter": int(relaxed_maxiter),
                        }
                    )
                if bool(allow_internals_fallback_to_framework_only):
                    attempt_specs.append(
                        {
                            "mode": "framework_only",
                            "use_internals": False,
                            "use_angles": False,
                            "maxstep": max(fallback_step_floor, float(constrained_maxstep) * 0.6),
                            "fixinternals_epsilon": None,
                            "fixinternals_maxiter": None,
                        }
                    )
            else:
                attempt_specs.append(
                    {
                        "mode": "framework_only",
                        "use_internals": False,
                        "use_angles": False,
                        "maxstep": float(unconstrained_maxstep),
                        "fixinternals_epsilon": None,
                        "fixinternals_maxiter": None,
                    }
                )

            work: Atoms | None = None
            nsteps = 0
            relax_status = "failed"
            used_adsorbate_internals = False
            used_adsorbate_angles = False
            used_fixinternals_epsilon = None
            used_fixinternals_maxiter = None
            used_optimizer_maxstep = None
            optimisation_error = None
            for attempt_idx, attempt_spec in enumerate(attempt_specs, start=1):
                mode = str(attempt_spec["mode"])
                use_internals = bool(attempt_spec["use_internals"])
                use_angles = bool(attempt_spec["use_angles"])
                maxstep = float(attempt_spec["maxstep"])
                attempt_fi_epsilon = attempt_spec["fixinternals_epsilon"]
                attempt_fi_maxiter = attempt_spec["fixinternals_maxiter"]
                attempt = work_seed.copy()
                opt = None
                try:
                    constraints_obj, internals_applied = _build_constraints_for_attempt(
                        groups=groups,
                        templates=templates,
                        include_internals=use_internals,
                        include_angles=use_angles,
                        attempt_fixinternals_epsilon=attempt_fi_epsilon,
                        attempt_fixinternals_maxiter=attempt_fi_maxiter,
                    )
                    if bool(strict_adsorbate_integrity) and use_internals and not internals_applied:
                        raise RuntimeError(
                            "Strict adsorbate integrity requested but FixInternals could not be built."
                        )
                    attempt.set_constraint(constraints_obj)
                    attempt.calc = calculator

                    attempt_trajectory = trajectory
                    attempt_logfile = logfile
                    if trajectory_dir is not None and attempt_idx > 1:
                        attempt_trajectory = str(
                            trajectory_dir / f"{trajectory_prefix}_{idx}_try{attempt_idx}.traj"
                        )
                        attempt_logfile = str(
                            trajectory_dir / f"{trajectory_prefix}_{idx}_try{attempt_idx}.log"
                        )

                    opt = BFGS(attempt, trajectory=attempt_trajectory, logfile=attempt_logfile)
                    opt.maxstep = float(maxstep)
                    opt.run(fmax=float(fmax), steps=int(max_steps))

                    nsteps = int(getattr(opt, "nsteps", 0))
                    work = attempt
                    relax_status = mode if attempt_idx == 1 else f"fallback:{mode}"
                    used_adsorbate_internals = bool(internals_applied)
                    used_adsorbate_angles = bool(internals_applied and use_angles)
                    used_fixinternals_epsilon = (
                        None
                        if not internals_applied or attempt_fi_epsilon is None
                        else float(attempt_fi_epsilon)
                    )
                    used_fixinternals_maxiter = (
                        None
                        if not internals_applied or attempt_fi_maxiter is None
                        else int(attempt_fi_maxiter)
                    )
                    used_optimizer_maxstep = float(maxstep)
                    optimisation_error = None
                    break
                except Exception as exc:
                    nsteps = int(getattr(opt, "nsteps", 0)) if opt is not None else 0
                    optimisation_error = str(exc)
                    if verbose:
                        fi_epsilon_text = (
                            "default"
                            if attempt_fi_epsilon is None
                            else f"{float(attempt_fi_epsilon):.1e}"
                        )
                        fi_maxiter_text = (
                            "default"
                            if attempt_fi_maxiter is None
                            else str(int(attempt_fi_maxiter))
                        )
                        print(
                            f"[MACE] {trajectory_prefix} warning {idx}/{n_structures}: "
                            f"attempt '{mode}' failed "
                            f"(maxstep={float(maxstep):.4f}, "
                            f"fi_eps={fi_epsilon_text}, fi_maxiter={fi_maxiter_text}): {exc}"
                        )
                    continue

            if work is None:
                work = work_seed.copy()
                work.calc = calculator

            try:
                energy = float(work.get_potential_energy())
                forces = work.get_forces()
            except Exception as exc:
                energy = float("nan")
                forces = np.full((len(work), 3), np.nan, dtype=float)
                if optimisation_error is None:
                    optimisation_error = str(exc)

            if adsorbate_indices:
                ads_force_norms = np.linalg.norm(forces[adsorbate_indices], axis=1)
                if np.any(np.isfinite(ads_force_norms)):
                    max_ads_force = float(np.nanmax(ads_force_norms))
                else:
                    max_ads_force = float("inf")
            else:
                max_ads_force = 0.0

            min_distance = _minimum_framework_adsorbate_distance(
                work,
                framework_indices=framework_indices_list,
                adsorbate_indices=adsorbate_indices,
            )
            integrity_ok, integrity_worst_ratio = _adsorbate_integrity_check(
                work,
                adsorbate_groups=groups,
                adsorbate_templates=templates,
                bond_mode=str(adsorbate_bond_constraint_mode),
                min_ratio=float(integrity_min_bond_ratio),
                max_ratio=float(integrity_max_bond_ratio),
            )
            physical_ok = bool(np.isfinite(max_ads_force))
            if min_distance is not None:
                physical_ok = bool(
                    physical_ok
                    and min_distance >= float(min_physical_adsorbate_framework_distance)
                )
            if bool(strict_adsorbate_integrity):
                physical_ok = bool(physical_ok and integrity_ok)
            if str(relax_status) == "failed":
                physical_ok = False

            max_distance_to_com = None
            inside_target_radius = True
            if cage_com is not None and target_radius is not None:
                max_distance_to_com = _maximum_adsorbate_distance_to_com(
                    work,
                    adsorbate_indices=adsorbate_indices,
                    cage_com=cage_com,
                )
                if max_distance_to_com is not None:
                    allowed = float(target_radius) + float(inside_radius_tolerance)
                    inside_target_radius = bool(max_distance_to_com <= allowed)

            row = {
                "row_id": idx,
                "energy_eV": energy,
                "max_ads_force_eV_A": max_ads_force,
                "converged": bool(np.isfinite(max_ads_force) and max_ads_force <= float(fmax)),
                "nsteps": int(nsteps),
                "relax_status": str(relax_status),
                "used_adsorbate_internals": bool(used_adsorbate_internals),
                "used_adsorbate_angles": bool(used_adsorbate_angles),
                "used_fixinternals_epsilon": (
                    None
                    if used_fixinternals_epsilon is None
                    else float(used_fixinternals_epsilon)
                ),
                "used_fixinternals_maxiter": (
                    None
                    if used_fixinternals_maxiter is None
                    else int(used_fixinternals_maxiter)
                ),
                "used_optimizer_maxstep": (
                    None if used_optimizer_maxstep is None else float(used_optimizer_maxstep)
                ),
                "min_framework_adsorbate_distance": (
                    None if min_distance is None else float(min_distance)
                ),
                "adsorbate_integrity_ok": bool(integrity_ok),
                "adsorbate_integrity_worst_bond_ratio": (
                    None if integrity_worst_ratio is None else float(integrity_worst_ratio)
                ),
                "physical_ok": physical_ok,
                "max_adsorbate_distance_to_com": (
                    None if max_distance_to_com is None else float(max_distance_to_com)
                ),
                "inside_target_radius": bool(inside_target_radius),
                "pre_relax_overlap_resolved": (
                    None
                    if pre_relax_overlap_resolved is None
                    else bool(pre_relax_overlap_resolved)
                ),
                "pre_relax_min_framework_adsorbate_distance": (
                    None
                    if pre_relax_min_framework_adsorbate_distance is None
                    else float(pre_relax_min_framework_adsorbate_distance)
                ),
                "pre_relax_min_adsorbate_adsorbate_distance": (
                    None
                    if pre_relax_min_adsorbate_adsorbate_distance is None
                    else float(pre_relax_min_adsorbate_adsorbate_distance)
                ),
                "pre_relax_overlap_resolution_steps": int(pre_relax_overlap_steps),
                "pre_relax_min_inter_adsorbate_bond_ratio": (
                    None
                    if pre_relax_min_inter_adsorbate_bond_ratio is None
                    else float(pre_relax_min_inter_adsorbate_bond_ratio)
                ),
                "optimisation_error": optimisation_error,
            }
            rows.append(row)

            frozen = work.copy()
            frozen.calc = SinglePointCalculator(frozen, energy=energy, forces=forces)
            relaxed_atoms.append(frozen)

            if db_context is not None:
                data = {}
                if metadata_rows is not None and idx - 1 < len(metadata_rows):
                    data.update(
                        {
                            key: _coerce_db_value(value)
                            for key, value in metadata_rows[idx - 1].items()
                        }
                    )
                data.update(
                    {
                        "relaxed_row_id": int(row["row_id"]),
                        "energy_eV": _coerce_db_value(row["energy_eV"]),
                        "max_ads_force_eV_A": _coerce_db_value(row["max_ads_force_eV_A"]),
                        "converged": bool(row["converged"]),
                        "nsteps": int(row["nsteps"]),
                        "relax_status": str(row["relax_status"]),
                        "used_adsorbate_internals": bool(row["used_adsorbate_internals"]),
                        "used_adsorbate_angles": bool(row["used_adsorbate_angles"]),
                        "used_fixinternals_epsilon": _coerce_db_value(
                            row["used_fixinternals_epsilon"]
                        ),
                        "used_fixinternals_maxiter": _coerce_db_value(
                            row["used_fixinternals_maxiter"]
                        ),
                        "used_optimizer_maxstep": _coerce_db_value(
                            row["used_optimizer_maxstep"]
                        ),
                        "min_framework_adsorbate_distance": _coerce_db_value(
                            row["min_framework_adsorbate_distance"]
                        ),
                        "adsorbate_integrity_ok": bool(row["adsorbate_integrity_ok"]),
                        "adsorbate_integrity_worst_bond_ratio": _coerce_db_value(
                            row["adsorbate_integrity_worst_bond_ratio"]
                        ),
                        "physical_ok": bool(row["physical_ok"]),
                        "max_adsorbate_distance_to_com": _coerce_db_value(
                            row["max_adsorbate_distance_to_com"]
                        ),
                        "inside_target_radius": bool(row["inside_target_radius"]),
                        "pre_relax_overlap_resolved": _coerce_db_value(
                            row["pre_relax_overlap_resolved"]
                        ),
                        "pre_relax_min_framework_adsorbate_distance": _coerce_db_value(
                            row["pre_relax_min_framework_adsorbate_distance"]
                        ),
                        "pre_relax_min_adsorbate_adsorbate_distance": _coerce_db_value(
                            row["pre_relax_min_adsorbate_adsorbate_distance"]
                        ),
                        "pre_relax_overlap_resolution_steps": int(
                            row["pre_relax_overlap_resolution_steps"]
                        ),
                        "pre_relax_min_inter_adsorbate_bond_ratio": _coerce_db_value(
                            row["pre_relax_min_inter_adsorbate_bond_ratio"]
                        ),
                        "optimisation_error": _coerce_db_value(row["optimisation_error"]),
                    }
                )
                db_context.write(frozen, data=data)

            if verbose:
                print(
                    f"[MACE] {trajectory_prefix} done {idx}/{n_structures}: "
                    f"E={energy:.6f} eV, max_ads_force={max_ads_force:.6f} eV/A, "
                    f"min_ads_fw_dist={float(min_distance) if min_distance is not None else float('nan'):.4f} A, "
                    f"integrity_ok={bool(integrity_ok)}, "
                    f"worst_bond_ratio={_fmt_optional(integrity_worst_ratio)}, "
                    f"inside_radius={inside_target_radius}, physical_ok={physical_ok}, "
                    f"status={relax_status}, "
                    f"maxstep={_fmt_optional(used_optimizer_maxstep)}, "
                    f"fi_eps={_fmt_optional(used_fixinternals_epsilon, '.1e') if used_fixinternals_epsilon is not None else 'default/off'}, "
                    f"fi_maxiter={used_fixinternals_maxiter if used_fixinternals_maxiter is not None else 'default/off'}"
                )

    if output_db_path is None:
        _run_relax(db_context=None)
    else:
        with connect(str(output_db_path)) as db_context:
            _run_relax(db_context=db_context)

    return relaxed_atoms, pd.DataFrame(rows)


def run_zeolite_ring_similarity_workflow(
    zeolite_atoms: Atoms,
    *,
    adsorbates: Sequence[AdsorbateLike],
    cage_com: Sequence[float] | None = None,
    ring_indices: Sequence[int] | None = None,
    site_symbols: Sequence[str] = ("H", "O"),
    target_radius: float = 3.2,
    use_framework_cage_boundary: bool = True,
    cage_boundary_n_nearest: int = 12,
    cage_boundary_margin: float = 0.0,
    relax_cage_boundary_if_empty: bool = True,
    relaxed_cage_boundary_radius: float | None = None,
    insertion_distance: float = 1.2,
    cage_padding: float = 0.2,
    max_site_candidates: int | None = None,
    max_structures: int | None = None,
    enable_site_based_placement: bool = True,
    enable_com_separation_placement: bool = False,
    com_separation_trial_count: int = 12,
    com_separation_radial_steps: int = 6,
    com_separation_force_emit_best_effort: bool = True,
    min_site_separation: float = 0.0,
    min_initial_adsorbate_framework_distance: float = 0.8,
    min_initial_adsorbate_adsorbate_distance: float | None = 0.8,
    enforce_nonbonded_coadsorbates: bool = True,
    nonbonded_coadsorbate_scale: float = 1.10,
    resolve_initial_overlaps: bool = True,
    overlap_resolution_max_steps: int = 24,
    placement_progress_interval: int = 25,
    overlap_detailed_logging: bool = False,
    overlap_progress_interval: int = 5,
    min_physical_adsorbate_framework_distance: float = 1.0,
    enforce_inside_target_radius: bool = True,
    relax_inside_radius_filter_if_empty: bool = True,
    inside_radius_tolerance: float = 0.1,
    similarity_threshold: float | None = None,
    reference_row_id: int = 1,
    similarity_kwargs: dict | None = None,
    candidates_db_path: str | Path = "scratch/zeolite_ring_candidates.db",
    prefilter_db_path: str | Path = "scratch/zeolite_ring_prefilter.db",
    constrained_relaxed_db_path: str | Path = "scratch/zeolite_ring_constrained_relaxed.db",
    final_relaxed_db_path: str | Path = "scratch/zeolite_ring_final_relaxed.db",
    similarity_report_csv_path: str | Path = "scratch/zeolite_ring_similarity_report.csv",
    constrained_report_csv_path: str | Path = "scratch/zeolite_ring_constrained_report.csv",
    final_report_csv_path: str | Path = "scratch/zeolite_ring_final_report.csv",
    summary_txt_path: str | Path = "scratch/zeolite_ring_summary.txt",
    mace_model_path: str | Path = DEFAULT_MACE_MODEL_PATH,
    mace_device: str = "cpu",
    mace_dispersion: bool = False,
    mace_default_dtype: str = "float64",
    crude_fmax: float = 0.20,
    crude_max_steps: int = 500,
    crude_mace_maxstep: float = 0.01,
    final_fmax: float = 0.05,
    final_max_steps: int = 1000,
    final_mace_maxstep: float = 0.01,
    constrain_adsorbate_angles: bool = False,
    adsorbate_bond_constraint_mode: str = "nearest",
    preserve_adsorbate_integrity: bool = False,
    integrity_min_bond_ratio: float = 0.65,
    integrity_max_bond_ratio: float = 1.60,
    fixinternals_epsilon: float | None = None,
    fixinternals_maxiter: int | None = None,
    fixinternals_relaxed_retry: bool = True,
    fixinternals_relaxed_epsilon: float = 1.0e-4,
    fixinternals_relaxed_maxiter: int = 800,
    fixinternals_relaxed_step_scale: float = 0.5,
    mace_trajectory_dir: str | Path | None = None,
    show_final: bool = False,
    verbose: bool = True,
):
    """
    Zeolite cage workflow:
    1) select a cage COM point (`cage_com`; optional fallback from `ring_indices`);
    2) infer cage boundary radius from nearest framework atoms (default behavior);
    3) place adsorbates using one or both strategies:
       - COM-separation math placement (COM-centered rigid seeds, then deterministic
         spatial separation + clash cleanup),
       - site-based placement (legacy behavior using nearby framework sites);
    4) iteratively resolve initial framework/adsorbate clashes before MACE;
    5) initial similarity pre-screening;
    6) constrained rough MACE relaxation (framework fixed, adsorbate internals fixed);
    7) keep physically plausible and (optionally) in-cage structures;
    8) final MACE relaxation (framework fixed, adsorbates fully relaxed).

    New robustness keywords:
    - enable_site_based_placement / enable_com_separation_placement:
      choose whether to generate initial seeds from site anchors, COM-separation, or both.
    - com_separation_trial_count / com_separation_radial_steps:
      number of deterministic direction/origin-radius trials for COM-separation placement.
    - com_separation_force_emit_best_effort:
      if True, emit the best COM-separation seed even when strict initial checks reject all.
    - min_initial_adsorbate_adsorbate_distance:
      optional minimum adsorbate-adsorbate separation during placement/pre-relax cleanup.
    - use_framework_cage_boundary / cage_boundary_n_nearest / cage_boundary_margin:
      derive cage boundary from nearest framework atoms instead of using target_radius directly.
    - relax_cage_boundary_if_empty / relaxed_cage_boundary_radius:
      if candidate generation keeps zero structures, optionally retry once with a relaxed boundary.
    - enforce_nonbonded_coadsorbates / nonbonded_coadsorbate_scale:
      require atoms from different adsorbates to remain beyond bond-like distances,
      using distance/(r_cov_i+r_cov_j) >= nonbonded_coadsorbate_scale.
    - resolve_initial_overlaps:
      run iterative overlap cleanup before screening and before each MACE stage.
    - overlap_resolution_max_steps:
      maximum iterations for overlap cleanup.
    - placement_progress_interval:
      print placement-stage progress every N processed candidates.
    - overlap_detailed_logging / overlap_progress_interval:
      optional per-step overlap-resolution heartbeat logging.
    - relax_inside_radius_filter_if_empty:
      if True and no row passes (physical_ok & inside_target_radius), fallback to physical_ok only.
    - constrain_adsorbate_angles:
      defaults to False for improved FixInternals stability.
    - adsorbate_bond_constraint_mode:
      'nearest' (recommended) constrains only nearest bonded pairs per atom.
      'all' constrains all inferred covalent bonds.
    - preserve_adsorbate_integrity:
      if True, enforce intramolecular constraints across both constrained and final
      stages, disable framework-only fallback when internals are requested, and
      include an integrity-based physical filter.
    - integrity_min_bond_ratio / integrity_max_bond_ratio:
      acceptable relaxed/template bond-length ratio bounds used by integrity checks.
    - crude_mace_maxstep / final_mace_maxstep:
      BFGS maximum displacement per optimization step in constrained/final stages.
      Defaults are intentionally small to reduce geometric instabilities.
    - fixinternals_epsilon / fixinternals_maxiter:
      optional solver controls passed to ASE FixInternals during constrained stage.
    - fixinternals_relaxed_retry:
      if True, adds a dedicated internals retry with relaxed FixInternals solver
      settings before dropping to framework_only fallback.
    - fixinternals_relaxed_epsilon / fixinternals_relaxed_maxiter:
      relaxed solver values used in that retry.
    - fixinternals_relaxed_step_scale:
      scales constrained maxstep for the relaxed-solver retry attempt.

    Constrained MACE automatically retries weaker constraints if FixInternals fails.
    With angle constraints enabled:
    internals+angles -> internals+angles_relaxed_solver -> internals_only
    -> internals_relaxed_solver -> framework_only.
    Without angle constraints:
    internals -> internals_relaxed_solver -> framework_only.
    """

    if zeolite_atoms is None:
        raise ValueError("Provide zeolite_atoms.")
    if len(zeolite_atoms) == 0:
        raise ValueError("zeolite_atoms is empty.")

    if cage_com is not None:
        cage_com_point = _normalize_com_point(cage_com)
        com_source = "cage_com"
    elif ring_indices is not None:
        cage_com_point = _ring_center_of_mass(zeolite_atoms, ring_indices)
        com_source = "ring_indices"
    else:
        raise ValueError(
            "Provide cage_com=[x, y, z]. "
            "ring_indices is supported only as an optional fallback."
        )

    similarity_kwargs = similarity_kwargs or {}
    adsorbate_bond_constraint_mode = str(adsorbate_bond_constraint_mode).strip().lower()
    if adsorbate_bond_constraint_mode not in {"nearest", "all"}:
        raise ValueError(
            "adsorbate_bond_constraint_mode must be 'nearest' or 'all', "
            f"received '{adsorbate_bond_constraint_mode}'."
        )
    if float(integrity_min_bond_ratio) <= 0.0:
        raise ValueError(
            "integrity_min_bond_ratio must be positive, "
            f"received {integrity_min_bond_ratio}."
        )
    if float(integrity_max_bond_ratio) <= 0.0:
        raise ValueError(
            "integrity_max_bond_ratio must be positive, "
            f"received {integrity_max_bond_ratio}."
        )
    if float(integrity_min_bond_ratio) >= float(integrity_max_bond_ratio):
        raise ValueError(
            "integrity_min_bond_ratio must be smaller than integrity_max_bond_ratio, "
            f"received {integrity_min_bond_ratio} >= {integrity_max_bond_ratio}."
        )
    if float(nonbonded_coadsorbate_scale) <= 0.0:
        raise ValueError(
            "nonbonded_coadsorbate_scale must be positive, "
            f"received {nonbonded_coadsorbate_scale}."
        )
    if not bool(enable_site_based_placement) and not bool(enable_com_separation_placement):
        raise ValueError(
            "At least one initial placement strategy must be enabled. "
            "Set enable_site_based_placement=True and/or enable_com_separation_placement=True."
        )
    if int(com_separation_trial_count) <= 0:
        raise ValueError(
            "com_separation_trial_count must be positive, "
            f"received {com_separation_trial_count}."
        )
    if int(com_separation_radial_steps) <= 0:
        raise ValueError(
            "com_separation_radial_steps must be positive, "
            f"received {com_separation_radial_steps}."
        )
    if float(crude_mace_maxstep) <= 0.0:
        raise ValueError(
            "crude_mace_maxstep must be positive, "
            f"received {crude_mace_maxstep}."
        )
    if float(final_mace_maxstep) <= 0.0:
        raise ValueError(
            "final_mace_maxstep must be positive, "
            f"received {final_mace_maxstep}."
        )
    if fixinternals_epsilon is not None and float(fixinternals_epsilon) <= 0.0:
        raise ValueError(
            "fixinternals_epsilon must be positive when provided, "
            f"received {fixinternals_epsilon}."
        )
    if fixinternals_maxiter is not None and int(fixinternals_maxiter) <= 0:
        raise ValueError(
            "fixinternals_maxiter must be positive when provided, "
            f"received {fixinternals_maxiter}."
        )
    if float(fixinternals_relaxed_epsilon) <= 0.0:
        raise ValueError(
            "fixinternals_relaxed_epsilon must be positive, "
            f"received {fixinternals_relaxed_epsilon}."
        )
    if int(fixinternals_relaxed_maxiter) <= 0:
        raise ValueError(
            "fixinternals_relaxed_maxiter must be positive, "
            f"received {fixinternals_relaxed_maxiter}."
        )
    if float(fixinternals_relaxed_step_scale) <= 0.0:
        raise ValueError(
            "fixinternals_relaxed_step_scale must be positive, "
            f"received {fixinternals_relaxed_step_scale}."
        )
    if int(cage_boundary_n_nearest) <= 0:
        raise ValueError(
            "cage_boundary_n_nearest must be positive, "
            f"received {cage_boundary_n_nearest}."
        )
    if (
        relaxed_cage_boundary_radius is not None
        and float(relaxed_cage_boundary_radius) <= 0.0
    ):
        raise ValueError(
            "relaxed_cage_boundary_radius must be positive when provided, "
            f"received {relaxed_cage_boundary_radius}."
        )
    effective_adsorbate_bond_constraint_mode = str(adsorbate_bond_constraint_mode)
    if bool(preserve_adsorbate_integrity):
        effective_adsorbate_bond_constraint_mode = "all"
        if (
            str(adsorbate_bond_constraint_mode).strip().lower() != "all"
            and bool(verbose)
        ):
            print(
                "[ZeoliteWorkflow] preserve_adsorbate_integrity=True: "
                "forcing adsorbate_bond_constraint_mode='all'."
            )
    supported_similarity_keys = {
        "support_species",
        "slab_shave_distance",
        "kernel_gamma",
        "kernel_alpha",
        "kernel_threshold",
        "exclude_self",
    }
    similarity_common_kwargs = {
        key: value
        for key, value in similarity_kwargs.items()
        if key in supported_similarity_keys
    }

    candidates_db_path = Path(candidates_db_path)
    prefilter_db_path = Path(prefilter_db_path)
    constrained_relaxed_db_path = Path(constrained_relaxed_db_path)
    final_relaxed_db_path = Path(final_relaxed_db_path)
    similarity_report_csv_path = Path(similarity_report_csv_path)
    constrained_report_csv_path = Path(constrained_report_csv_path)
    final_report_csv_path = Path(final_report_csv_path)
    summary_txt_path = Path(summary_txt_path)
    mace_trajectory_dir = None if mace_trajectory_dir is None else Path(mace_trajectory_dir)

    output_paths = [
        candidates_db_path,
        prefilter_db_path,
        constrained_relaxed_db_path,
        final_relaxed_db_path,
        similarity_report_csv_path,
        constrained_report_csv_path,
        final_report_csv_path,
        summary_txt_path,
    ]
    if mace_trajectory_dir is not None:
        output_paths.append(mace_trajectory_dir)
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_file():
            path.unlink()

    adsorbate_specs = _normalize_adsorbates(adsorbates)
    framework_indices = list(range(len(zeolite_atoms)))
    all_site_indices: list[int] = []
    all_site_distances: list[float] = []
    site_indices: list[int] = []
    site_distances: list[float] = []
    if bool(enable_site_based_placement):
        all_site_indices, all_site_distances = _select_adsorption_sites_near_com(
            zeolite_atoms,
            cage_com=cage_com_point,
            site_symbols=site_symbols,
            target_radius=float(target_radius),
            max_site_candidates=None,
        )
        if max_site_candidates is not None and int(max_site_candidates) > 0:
            effective_site_cap = min(int(max_site_candidates), len(all_site_indices))
        else:
            effective_site_cap = len(all_site_indices)
        site_indices = all_site_indices[:effective_site_cap]
        site_distances = all_site_distances[:effective_site_cap]

    if bool(use_framework_cage_boundary):
        inferred_cage_boundary_radius = _infer_cage_boundary_radius_from_framework(
            zeolite_atoms,
            cage_com=cage_com_point,
            n_nearest=int(cage_boundary_n_nearest),
        )
        cage_boundary_source = f"framework_nearest_{int(cage_boundary_n_nearest)}"
    else:
        inferred_cage_boundary_radius = float(target_radius)
        cage_boundary_source = "target_radius"
    cage_boundary_radius = float(inferred_cage_boundary_radius) - float(cage_boundary_margin)
    if cage_boundary_radius <= 0.0:
        raise ValueError(
            "Computed cage boundary radius is non-positive after applying margin. "
            f"inferred={float(inferred_cage_boundary_radius):.6f}, "
            f"margin={float(cage_boundary_margin):.6f}, "
            f"final={float(cage_boundary_radius):.6f}"
        )

    if verbose:
        print("[ZeoliteWorkflow] Starting zeolite COM workflow")
        print(f"[ZeoliteWorkflow] Framework atoms: {len(zeolite_atoms)}")
        print(
            "[ZeoliteWorkflow] Cage COM "
            f"({com_source}): ({cage_com_point[0]:.4f}, {cage_com_point[1]:.4f}, {cage_com_point[2]:.4f})"
        )
        if bool(enable_site_based_placement):
            print(
                "[ZeoliteWorkflow] Candidate site atoms: "
                f"{len(site_indices)} / {len(all_site_indices)} "
                f"(symbols={list(site_symbols)}, selection_radius={target_radius:.3f} A, "
                f"max_site_candidates={max_site_candidates})"
            )
        else:
            print("[ZeoliteWorkflow] Candidate site atoms: skipped (site-based placement disabled)")
        print(
            "[ZeoliteWorkflow] Cage boundary radius: "
            f"{float(cage_boundary_radius):.3f} A "
            f"(source={cage_boundary_source}, "
            f"inferred={float(inferred_cage_boundary_radius):.3f} A, "
            f"margin={float(cage_boundary_margin):.3f} A)"
        )

    generated_atoms: list[Atoms] = []
    generated_metadata: list[dict] = []
    generated_groups: list[list[list[int]]] = []
    generated_templates: list[list[Atoms]] = []

    def _collect_candidates(boundary_radius: float):
        atoms_acc: list[Atoms] = []
        metadata_acc: list[dict] = []
        groups_acc: list[list[list[int]]] = []
        templates_acc: list[list[Atoms]] = []
        remaining = None if max_structures is None else max(0, int(max_structures))

        if bool(enable_com_separation_placement) and (remaining is None or remaining > 0):
            for atoms_obj, metadata, groups, templates in _generate_com_separated_candidates(
                framework=zeolite_atoms,
                adsorbate_specs=adsorbate_specs,
                cage_com=cage_com_point,
                cage_boundary_radius=float(boundary_radius),
                site_selection_radius=float(target_radius),
                cage_padding=float(cage_padding),
                min_initial_adsorbate_framework_distance=float(
                    min_initial_adsorbate_framework_distance
                ),
                min_initial_adsorbate_adsorbate_distance=(
                    None
                    if min_initial_adsorbate_adsorbate_distance is None
                    else float(min_initial_adsorbate_adsorbate_distance)
                ),
                enforce_nonbonded_coadsorbates=bool(enforce_nonbonded_coadsorbates),
                nonbonded_coadsorbate_scale=float(nonbonded_coadsorbate_scale),
                resolve_initial_overlaps=bool(resolve_initial_overlaps),
                overlap_resolution_max_steps=int(overlap_resolution_max_steps),
                max_structures=remaining,
                trial_count=int(com_separation_trial_count),
                radial_steps=int(com_separation_radial_steps),
                force_emit_best_effort=bool(com_separation_force_emit_best_effort),
                verbose=bool(verbose),
                placement_progress_interval=int(placement_progress_interval),
                overlap_detailed_logging=bool(overlap_detailed_logging),
                overlap_progress_interval=int(overlap_progress_interval),
            ):
                atoms_acc.append(atoms_obj)
                metadata_acc.append(metadata)
                groups_acc.append(groups)
                templates_acc.append(templates)
                if remaining is not None:
                    remaining = max(0, remaining - 1)
                    if remaining == 0:
                        return atoms_acc, metadata_acc, groups_acc, templates_acc

        if bool(enable_site_based_placement) and (remaining is None or remaining > 0):
            if len(site_indices) < len(adsorbate_specs):
                if verbose:
                    print(
                        "[ZeoliteWorkflow] Warning: not enough site candidates for site-based "
                        f"placement (sites={len(site_indices)}, adsorbates={len(adsorbate_specs)}). "
                        "Skipping site-based generation."
                    )
            else:
                for atoms_obj, metadata, groups, templates in _generate_zeolite_ring_candidates(
                    framework=zeolite_atoms,
                    site_indices=site_indices,
                    site_distances=site_distances,
                    adsorbate_specs=adsorbate_specs,
                    cage_com=cage_com_point,
                    cage_boundary_radius=float(boundary_radius),
                    site_selection_radius=float(target_radius),
                    insertion_distance=float(insertion_distance),
                    cage_padding=float(cage_padding),
                    min_site_separation=float(min_site_separation),
                    min_initial_adsorbate_framework_distance=float(
                        min_initial_adsorbate_framework_distance
                    ),
                    min_initial_adsorbate_adsorbate_distance=(
                        None
                        if min_initial_adsorbate_adsorbate_distance is None
                        else float(min_initial_adsorbate_adsorbate_distance)
                    ),
                    enforce_nonbonded_coadsorbates=bool(enforce_nonbonded_coadsorbates),
                    nonbonded_coadsorbate_scale=float(nonbonded_coadsorbate_scale),
                    resolve_initial_overlaps=bool(resolve_initial_overlaps),
                    overlap_resolution_max_steps=int(overlap_resolution_max_steps),
                    max_structures=remaining,
                    verbose=bool(verbose),
                    placement_progress_interval=int(placement_progress_interval),
                    overlap_detailed_logging=bool(overlap_detailed_logging),
                    overlap_progress_interval=int(overlap_progress_interval),
                ):
                    atoms_acc.append(atoms_obj)
                    metadata_acc.append(metadata)
                    groups_acc.append(groups)
                    templates_acc.append(templates)
                    if remaining is not None:
                        remaining = max(0, remaining - 1)
                        if remaining == 0:
                            return atoms_acc, metadata_acc, groups_acc, templates_acc
        return atoms_acc, metadata_acc, groups_acc, templates_acc

    if verbose:
        print(
            "[ZeoliteWorkflow] Running placement + overlap cleanup before similarity pre-screen "
            f"(strategy_com={bool(enable_com_separation_placement)}, "
            f"strategy_site={bool(enable_site_based_placement)}, "
            f"progress interval={int(placement_progress_interval)}, "
            f"overlap_detailed_logging={bool(overlap_detailed_logging)}, "
            f"enforce_nonbonded_coadsorbates={bool(enforce_nonbonded_coadsorbates)}, "
            f"nonbonded_scale={float(nonbonded_coadsorbate_scale):.3f})"
        )

    (
        generated_atoms,
        generated_metadata,
        generated_groups,
        generated_templates,
    ) = _collect_candidates(float(cage_boundary_radius))

    placement_fallback_used = False
    if (
        not generated_atoms
        and bool(relax_cage_boundary_if_empty)
        and bool(use_framework_cage_boundary)
    ):
        fallback_boundary = (
            float(target_radius)
            if relaxed_cage_boundary_radius is None
            else float(relaxed_cage_boundary_radius)
        )
        if fallback_boundary > float(cage_boundary_radius) + 1.0e-6:
            if verbose:
                print(
                    "[ZeoliteWorkflow] Warning: zero candidates with inferred cage boundary; "
                    f"retrying placement with relaxed boundary={fallback_boundary:.3f} A"
                )
            (
                generated_atoms,
                generated_metadata,
                generated_groups,
                generated_templates,
            ) = _collect_candidates(fallback_boundary)
            if generated_atoms:
                placement_fallback_used = True
                cage_boundary_radius = float(fallback_boundary)
                cage_boundary_source = f"{cage_boundary_source}_fallback_relaxed"

    if not generated_atoms:
        raise RuntimeError(
            "No initial zeolite-cage adsorption structures were generated. "
            f"Strategies: com={bool(enable_com_separation_placement)}, "
            f"site={bool(enable_site_based_placement)}. "
            f"Site candidates used/available: {len(site_indices)}/{len(all_site_indices)}. "
            "Adjust cage boundary, clash thresholds, or placement strategy settings."
        )
    _write_atoms_db(candidates_db_path, generated_atoms, generated_metadata)
    if verbose:
        print(f"[ZeoliteWorkflow] Generated candidates: {len(generated_atoms)}")

    threshold_df = _suggest_threshold_safe(
        generated_atoms,
        reference_row_id=reference_row_id,
        similarity_kwargs=similarity_common_kwargs,
    )
    threshold = (
        float(similarity_threshold)
        if similarity_threshold is not None
        else float(threshold_df.attrs["suggested_threshold"])
    )
    if verbose:
        spread = threshold_df.attrs.get("similarity_spread", {})
        print(
            "[ZeoliteWorkflow] Similarity threshold "
            f"(used={threshold:.6f}, suggested={threshold_df.attrs['suggested_threshold']:.6f}, "
            f"range={spread.get('min', float('nan')):.6f}-{spread.get('max', float('nan')):.6f})"
        )
        print(
            f"[ZeoliteWorkflow] Running similarity pre-screen on {len(generated_atoms)} candidates"
        )

    prefilter_atoms, similarity_report_df = deduplicate_atoms_by_similarity(
        generated_atoms,
        similarity_threshold=threshold,
        **similarity_common_kwargs,
    )
    similarity_report_df.to_csv(similarity_report_csv_path, index=False)
    if not prefilter_atoms:
        raise RuntimeError("Similarity pre-screen removed all generated structures.")

    kept_rows = similarity_report_df[similarity_report_df["kept"]].copy()
    prefilter_metadata: list[dict] = []
    prefilter_groups: list[list[list[int]]] = []
    prefilter_templates: list[list[Atoms]] = []
    for _, row in kept_rows.iterrows():
        source_row_id = int(row["row_id"])
        source_idx = source_row_id - 1
        source_metadata = dict(generated_metadata[source_idx])
        source_metadata.update(
            {
                "source_candidate_row_id": source_row_id,
                "max_similarity_to_kept": _coerce_db_value(row["max_similarity_to_kept"]),
            }
        )
        prefilter_metadata.append(source_metadata)
        prefilter_groups.append(generated_groups[source_idx])
        prefilter_templates.append(generated_templates[source_idx])
    _write_atoms_db(prefilter_db_path, prefilter_atoms, prefilter_metadata)
    if verbose:
        print(
            "[ZeoliteWorkflow] Prefilter kept "
            f"{len(prefilter_atoms)} / {len(generated_atoms)} candidates"
        )

    trajectory_dir_constrained = (
        None if mace_trajectory_dir is None else mace_trajectory_dir / "constrained"
    )
    constrained_atoms, constrained_report_df = _relax_with_mace(
        prefilter_atoms,
        framework_indices=framework_indices,
        adsorbate_groups_list=prefilter_groups,
        adsorbate_templates_list=prefilter_templates,
        model_path=str(mace_model_path),
        fmax=float(crude_fmax),
        max_steps=int(crude_max_steps),
        device=mace_device,
        dispersion=mace_dispersion,
        default_dtype=mace_default_dtype,
        constrain_adsorbate_internals=True,
        constrain_adsorbate_angles=constrain_adsorbate_angles,
        adsorbate_bond_constraint_mode=str(effective_adsorbate_bond_constraint_mode),
        strict_adsorbate_integrity=bool(preserve_adsorbate_integrity),
        allow_internals_fallback_to_framework_only=not bool(preserve_adsorbate_integrity),
        integrity_min_bond_ratio=float(integrity_min_bond_ratio),
        integrity_max_bond_ratio=float(integrity_max_bond_ratio),
        fixinternals_epsilon=(
            None if fixinternals_epsilon is None else float(fixinternals_epsilon)
        ),
        fixinternals_maxiter=(
            None if fixinternals_maxiter is None else int(fixinternals_maxiter)
        ),
        fixinternals_relaxed_retry=bool(fixinternals_relaxed_retry),
        fixinternals_relaxed_epsilon=float(fixinternals_relaxed_epsilon),
        fixinternals_relaxed_maxiter=int(fixinternals_relaxed_maxiter),
        fixinternals_relaxed_step_scale=float(fixinternals_relaxed_step_scale),
        min_physical_adsorbate_framework_distance=float(min_physical_adsorbate_framework_distance),
        cage_com=cage_com_point,
        target_radius=float(cage_boundary_radius),
        cage_padding=float(cage_padding),
        inside_radius_tolerance=float(inside_radius_tolerance),
        resolve_overlaps_before_relax=bool(resolve_initial_overlaps),
        pre_relax_overlap_max_steps=int(overlap_resolution_max_steps),
        min_pre_relax_adsorbate_adsorbate_distance=(
            None
            if min_initial_adsorbate_adsorbate_distance is None
            else float(min_initial_adsorbate_adsorbate_distance)
        ),
        enforce_nonbonded_coadsorbates=bool(enforce_nonbonded_coadsorbates),
        nonbonded_coadsorbate_scale=float(nonbonded_coadsorbate_scale),
        constrained_maxstep=float(crude_mace_maxstep),
        unconstrained_maxstep=float(crude_mace_maxstep),
        output_db_path=constrained_relaxed_db_path,
        metadata_rows=prefilter_metadata,
        trajectory_dir=trajectory_dir_constrained,
        trajectory_prefix="constrained",
        verbose=verbose,
    )
    constrained_report_df.to_csv(constrained_report_csv_path, index=False)
    if constrained_report_df.empty:
        raise RuntimeError("Constrained MACE stage produced no structures.")

    constrained_total = int(len(constrained_report_df))
    constrained_physical_count = int(constrained_report_df["physical_ok"].sum())
    constrained_inside_count = int(constrained_report_df["inside_target_radius"].sum())
    constrained_both_count = int(
        (constrained_report_df["physical_ok"] & constrained_report_df["inside_target_radius"]).sum()
    )
    if verbose:
        print(
            "[ZeoliteWorkflow] Constrained filter stats: "
            f"total={constrained_total}, "
            f"physical_ok={constrained_physical_count}, "
            f"inside_radius={constrained_inside_count}, "
            f"physical_and_inside={constrained_both_count}"
        )

    constrained_filter_mode = (
        "physical_and_inside" if enforce_inside_target_radius else "physical_only"
    )
    if enforce_inside_target_radius:
        physical_rows = constrained_report_df[
            constrained_report_df["physical_ok"]
            & constrained_report_df["inside_target_radius"]
        ].copy()
        if (
            physical_rows.empty
            and bool(relax_inside_radius_filter_if_empty)
            and constrained_physical_count > 0
        ):
            physical_rows = constrained_report_df[constrained_report_df["physical_ok"]].copy()
            constrained_filter_mode = "fallback_physical_only"
            if verbose:
                print(
                    "[ZeoliteWorkflow] Warning: no structures passed "
                    "(physical_ok & inside_target_radius); falling back to physical_ok only."
                )
    else:
        physical_rows = constrained_report_df[constrained_report_df["physical_ok"]].copy()

    if physical_rows.empty:
        raise RuntimeError(
            "All constrained-relaxed structures were filtered out. "
            f"Counts: total={constrained_total}, "
            f"physical_ok={constrained_physical_count}, "
            f"inside_radius={constrained_inside_count}, "
            f"physical_and_inside={constrained_both_count}. "
            "Increase cage boundary (e.g. reduce cage_boundary_margin, adjust cage_boundary_n_nearest, "
            "or set use_framework_cage_boundary=False), increase insertion_distance, lower clash thresholds, "
            "or set enforce_inside_target_radius=False."
        )
    physical_indices = [int(row_id) - 1 for row_id in physical_rows["row_id"]]
    physical_atoms = [constrained_atoms[idx] for idx in physical_indices]
    physical_metadata = [dict(prefilter_metadata[idx]) for idx in physical_indices]
    physical_groups = [prefilter_groups[idx] for idx in physical_indices]
    physical_templates = [prefilter_templates[idx] for idx in physical_indices]
    if verbose:
        print(
            "[ZeoliteWorkflow] Physically plausible after constrained stage: "
            f"{len(physical_atoms)} / {len(constrained_atoms)}"
        )

    trajectory_dir_final = None if mace_trajectory_dir is None else mace_trajectory_dir / "final"
    final_atoms, final_report_df = _relax_with_mace(
        physical_atoms,
        framework_indices=framework_indices,
        adsorbate_groups_list=physical_groups,
        adsorbate_templates_list=physical_templates,
        model_path=str(mace_model_path),
        fmax=float(final_fmax),
        max_steps=int(final_max_steps),
        device=mace_device,
        dispersion=mace_dispersion,
        default_dtype=mace_default_dtype,
        constrain_adsorbate_internals=bool(preserve_adsorbate_integrity),
        constrain_adsorbate_angles=False,
        adsorbate_bond_constraint_mode=str(effective_adsorbate_bond_constraint_mode),
        strict_adsorbate_integrity=bool(preserve_adsorbate_integrity),
        allow_internals_fallback_to_framework_only=not bool(preserve_adsorbate_integrity),
        integrity_min_bond_ratio=float(integrity_min_bond_ratio),
        integrity_max_bond_ratio=float(integrity_max_bond_ratio),
        fixinternals_epsilon=(
            None if fixinternals_epsilon is None else float(fixinternals_epsilon)
        ),
        fixinternals_maxiter=(
            None if fixinternals_maxiter is None else int(fixinternals_maxiter)
        ),
        fixinternals_relaxed_retry=bool(fixinternals_relaxed_retry),
        fixinternals_relaxed_epsilon=float(fixinternals_relaxed_epsilon),
        fixinternals_relaxed_maxiter=int(fixinternals_relaxed_maxiter),
        fixinternals_relaxed_step_scale=float(fixinternals_relaxed_step_scale),
        min_physical_adsorbate_framework_distance=float(min_physical_adsorbate_framework_distance),
        cage_com=cage_com_point,
        target_radius=float(cage_boundary_radius),
        cage_padding=float(cage_padding),
        inside_radius_tolerance=float(inside_radius_tolerance),
        resolve_overlaps_before_relax=bool(resolve_initial_overlaps),
        pre_relax_overlap_max_steps=int(overlap_resolution_max_steps),
        min_pre_relax_adsorbate_adsorbate_distance=(
            None
            if min_initial_adsorbate_adsorbate_distance is None
            else float(min_initial_adsorbate_adsorbate_distance)
        ),
        enforce_nonbonded_coadsorbates=bool(enforce_nonbonded_coadsorbates),
        nonbonded_coadsorbate_scale=float(nonbonded_coadsorbate_scale),
        constrained_maxstep=float(final_mace_maxstep),
        unconstrained_maxstep=float(final_mace_maxstep),
        output_db_path=final_relaxed_db_path,
        metadata_rows=physical_metadata,
        trajectory_dir=trajectory_dir_final,
        trajectory_prefix="final",
        verbose=verbose,
    )
    if bool(preserve_adsorbate_integrity):
        final_keep_rows = final_report_df[final_report_df["physical_ok"]].copy()
        if final_keep_rows.empty:
            raise RuntimeError(
                "Final stage did not produce any physically valid intact adsorbates. "
                "Consider relaxing integrity bounds or increasing overlap-resolution effort."
            )
        final_keep_indices = [int(row_id) - 1 for row_id in final_keep_rows["row_id"]]
        final_atoms = [final_atoms[idx] for idx in final_keep_indices]
        final_metadata_filtered: list[dict] = []
        for source_idx, (_, row) in zip(final_keep_indices, final_keep_rows.iterrows()):
            payload = dict(physical_metadata[source_idx])
            payload.update({key: _coerce_db_value(value) for key, value in row.to_dict().items()})
            final_metadata_filtered.append(payload)
        if final_relaxed_db_path.exists() and final_relaxed_db_path.is_file():
            final_relaxed_db_path.unlink()
        _write_atoms_db(final_relaxed_db_path, final_atoms, final_metadata_filtered)
        final_report_df = final_keep_rows.reset_index(drop=True)
    final_report_df.to_csv(final_report_csv_path, index=False)
    if not final_atoms:
        raise RuntimeError("Final MACE stage produced no structures.")

    with summary_txt_path.open("w", encoding="utf-8") as handle:
        handle.write("Zeolite COM Adsorbate Workflow Summary\n")
        handle.write("=====================================\n")
        handle.write(
            f"Cage COM ({com_source}): ({cage_com_point[0]:.6f}, "
            f"{cage_com_point[1]:.6f}, {cage_com_point[2]:.6f})\n"
        )
        handle.write(
            f"Placement strategies (com/site): "
            f"{bool(enable_com_separation_placement)}/{bool(enable_site_based_placement)}\n"
        )
        handle.write(
            "COM separation settings: "
            f"trials={int(com_separation_trial_count)}, "
            f"radial_steps={int(com_separation_radial_steps)}, "
            f"force_best_effort={bool(com_separation_force_emit_best_effort)}\n"
        )
        if bool(enable_site_based_placement):
            handle.write(f"Site symbols: {','.join(site_symbols)}\n")
        else:
            handle.write("Site symbols: (site-based placement disabled)\n")
        handle.write(f"Target radius (site selection): {float(target_radius):.6f}\n")
        handle.write(
            "Site candidates (within/used): "
            f"{len(all_site_indices)}/{len(site_indices)}\n"
        )
        handle.write(
            f"Cage boundary radius: {float(cage_boundary_radius):.6f} "
            f"(source={cage_boundary_source}, inferred={float(inferred_cage_boundary_radius):.6f}, "
            f"margin={float(cage_boundary_margin):.6f})\n"
        )
        handle.write(
            "Placement boundary fallback: "
            f"{bool(placement_fallback_used)} "
            f"(enabled={bool(relax_cage_boundary_if_empty)}, "
            f"relaxed_radius={float(target_radius) if relaxed_cage_boundary_radius is None else float(relaxed_cage_boundary_radius):.6f})\n"
        )
        handle.write(f"Insertion distance: {float(insertion_distance):.6f}\n")
        handle.write(f"Cage padding: {float(cage_padding):.6f}\n")
        handle.write(f"Min site separation: {float(min_site_separation):.6f}\n")
        handle.write(
            f"Resolve initial overlaps: {bool(resolve_initial_overlaps)} "
            f"(steps={int(overlap_resolution_max_steps)})\n"
        )
        handle.write(
            f"Placement progress interval: {int(placement_progress_interval)}\n"
        )
        handle.write(
            f"Overlap detailed logging: {bool(overlap_detailed_logging)} "
            f"(step interval={int(overlap_progress_interval)})\n"
        )
        handle.write(
            "Min initial adsorbate-framework distance: "
            f"{float(min_initial_adsorbate_framework_distance):.6f}\n"
        )
        handle.write(
            (
                "Min initial adsorbate-adsorbate distance: "
                f"{float(min_initial_adsorbate_adsorbate_distance):.6f}\n"
            )
            if min_initial_adsorbate_adsorbate_distance is not None
            else "Min initial adsorbate-adsorbate distance: None\n"
        )
        handle.write(
            "Require non-bonded co-adsorbates: "
            f"{bool(enforce_nonbonded_coadsorbates)} "
            f"(min ratio={float(nonbonded_coadsorbate_scale):.4f})\n"
        )
        handle.write(
            "Min physical adsorbate-framework distance: "
            f"{float(min_physical_adsorbate_framework_distance):.6f}\n"
        )
        handle.write(
            "MACE maxstep (constrained/final): "
            f"{float(crude_mace_maxstep):.6f}/{float(final_mace_maxstep):.6f}\n"
        )
        handle.write(
            "Constrained internals mode: "
            f"bond_mode={str(effective_adsorbate_bond_constraint_mode)}, "
            f"angles={bool(constrain_adsorbate_angles)}, "
            f"preserve_integrity={bool(preserve_adsorbate_integrity)}\n"
        )
        handle.write(
            "Integrity bond-ratio window: "
            f"[{float(integrity_min_bond_ratio):.3f}, {float(integrity_max_bond_ratio):.3f}]\n"
        )
        handle.write(
            "Final-stage integrity filtering enabled: "
            f"{bool(preserve_adsorbate_integrity)}\n"
        )
        handle.write(
            "FixInternals base settings: "
            f"epsilon={('default' if fixinternals_epsilon is None else f'{float(fixinternals_epsilon):.1e}')}, "
            f"maxiter={('default' if fixinternals_maxiter is None else int(fixinternals_maxiter))}\n"
        )
        handle.write(
            "FixInternals relaxed retry: "
            f"{bool(fixinternals_relaxed_retry)} "
            f"(epsilon={float(fixinternals_relaxed_epsilon):.1e}, "
            f"maxiter={int(fixinternals_relaxed_maxiter)}, "
            f"step_scale={float(fixinternals_relaxed_step_scale):.3f})\n"
        )
        handle.write(
            f"Enforce inside radius after constrained stage: {bool(enforce_inside_target_radius)}\n"
        )
        handle.write(
            "Relax inside-radius filter if empty: "
            f"{bool(relax_inside_radius_filter_if_empty)}\n"
        )
        handle.write(f"Inside radius tolerance: {float(inside_radius_tolerance):.6f}\n")
        handle.write(
            f"Constrained filter mode used: {constrained_filter_mode}\n"
        )
        handle.write(
            "Constrained filter counts: "
            f"total={constrained_total}, "
            f"physical_ok={constrained_physical_count}, "
            f"inside_radius={constrained_inside_count}, "
            f"physical_and_inside={constrained_both_count}\n"
        )
        handle.write(f"Initial candidates: {len(generated_atoms)}\n")
        handle.write(f"Similarity-kept candidates: {len(prefilter_atoms)}\n")
        handle.write(f"Physical after constrained stage: {len(physical_atoms)}\n")
        handle.write(f"Final relaxed structures: {len(final_atoms)}\n")
        handle.write(
            f"Similarity threshold used: {float(threshold):.6f} "
            f"(suggested {float(threshold_df.attrs['suggested_threshold']):.6f})\n"
        )
        handle.write(f"Candidates DB: {candidates_db_path.resolve()}\n")
        handle.write(f"Prefilter DB: {prefilter_db_path.resolve()}\n")
        handle.write(f"Constrained relaxed DB: {constrained_relaxed_db_path.resolve()}\n")
        handle.write(f"Final relaxed DB: {final_relaxed_db_path.resolve()}\n")
        handle.write(f"Similarity report: {similarity_report_csv_path.resolve()}\n")
        handle.write(f"Constrained report: {constrained_report_csv_path.resolve()}\n")
        handle.write(f"Final report: {final_report_csv_path.resolve()}\n")

    if verbose:
        print(f"[ZeoliteWorkflow] Final structures: {len(final_atoms)}")
        print(f"[ZeoliteWorkflow] Final DB: {final_relaxed_db_path.resolve()}")
        print(f"[ZeoliteWorkflow] Summary: {summary_txt_path.resolve()}")

    if show_final:
        view(final_atoms)

    return {
        "cage_com": cage_com_point,
        "ring_com": cage_com_point,
        "com_source": com_source,
        "site_indices": site_indices,
        "site_distances": site_distances,
        "all_site_indices_within_target_radius": all_site_indices,
        "all_site_distances_within_target_radius": all_site_distances,
        "cage_boundary_radius": float(cage_boundary_radius),
        "cage_boundary_source": cage_boundary_source,
        "cage_boundary_inferred_radius": float(inferred_cage_boundary_radius),
        "cage_boundary_margin": float(cage_boundary_margin),
        "placement_boundary_fallback_used": bool(placement_fallback_used),
        "relax_cage_boundary_if_empty": bool(relax_cage_boundary_if_empty),
        "relaxed_cage_boundary_radius": (
            float(target_radius)
            if relaxed_cage_boundary_radius is None
            else float(relaxed_cage_boundary_radius)
        ),
        "enable_com_separation_placement": bool(enable_com_separation_placement),
        "enable_site_based_placement": bool(enable_site_based_placement),
        "com_separation_trial_count": int(com_separation_trial_count),
        "com_separation_radial_steps": int(com_separation_radial_steps),
        "com_separation_force_emit_best_effort": bool(
            com_separation_force_emit_best_effort
        ),
        "preserve_adsorbate_integrity": bool(preserve_adsorbate_integrity),
        "integrity_min_bond_ratio": float(integrity_min_bond_ratio),
        "integrity_max_bond_ratio": float(integrity_max_bond_ratio),
        "adsorbate_bond_constraint_mode_used": str(
            effective_adsorbate_bond_constraint_mode
        ),
        "candidates": generated_atoms,
        "similarity_threshold_df": threshold_df,
        "similarity_threshold_used": threshold,
        "prefilter_atoms": prefilter_atoms,
        "similarity_report_df": similarity_report_df,
        "constrained_atoms": constrained_atoms,
        "constrained_report_df": constrained_report_df,
        "physical_atoms": physical_atoms,
        "final_atoms": final_atoms,
        "final_report_df": final_report_df,
        "constrained_filter_mode": constrained_filter_mode,
        "constrained_filter_counts": {
            "total": constrained_total,
            "physical_ok": constrained_physical_count,
            "inside_radius": constrained_inside_count,
            "physical_and_inside": constrained_both_count,
        },
        "candidates_db_path": candidates_db_path,
        "prefilter_db_path": prefilter_db_path,
        "constrained_relaxed_db_path": constrained_relaxed_db_path,
        "final_relaxed_db_path": final_relaxed_db_path,
        "similarity_report_csv_path": similarity_report_csv_path,
        "constrained_report_csv_path": constrained_report_csv_path,
        "final_report_csv_path": final_report_csv_path,
        "summary_txt_path": summary_txt_path,
        "fixinternals_settings": {
            "epsilon": (
                None if fixinternals_epsilon is None else float(fixinternals_epsilon)
            ),
            "maxiter": (
                None if fixinternals_maxiter is None else int(fixinternals_maxiter)
            ),
            "relaxed_retry": bool(fixinternals_relaxed_retry),
            "relaxed_epsilon": float(fixinternals_relaxed_epsilon),
            "relaxed_maxiter": int(fixinternals_relaxed_maxiter),
            "relaxed_step_scale": float(fixinternals_relaxed_step_scale),
        },
    }


def run_zeolite_com_similarity_workflow(
    zeolite_atoms: Atoms,
    *,
    cage_com: Sequence[float],
    **kwargs,
):
    """
    COM-first entry point for zeolite adsorption screening.
    Defaults to deterministic COM-separation seed generation and disables
    site-based seeding unless explicitly re-enabled by the caller.
    Also defaults to preserving adsorbate integrity during both MACE stages.
    """
    workflow_kwargs = dict(kwargs)
    workflow_kwargs.setdefault("enable_com_separation_placement", True)
    workflow_kwargs.setdefault("enable_site_based_placement", False)
    workflow_kwargs.setdefault("com_separation_force_emit_best_effort", True)
    workflow_kwargs.setdefault("preserve_adsorbate_integrity", True)
    workflow_kwargs.setdefault("adsorbate_bond_constraint_mode", "all")
    return run_zeolite_ring_similarity_workflow(
        zeolite_atoms=zeolite_atoms,
        cage_com=cage_com,
        **workflow_kwargs,
    )
