from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def _kabsch(moving: np.ndarray, target: np.ndarray) -> np.ndarray:
    covariance = moving.T @ target
    left, _, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
    return rotation


def align_and_reorder_endpoint(
    initial: Atoms,
    final: Atoms,
    *,
    anchor_symbol: str = "Au",
    anchor_weight: float = 20.0,
    max_iterations: int = 20,
    rotation_starts: int = 512,
) -> tuple[Atoms, dict[str, float]]:
    """Rigidly align ``final`` and reorder like elements to minimize displacement."""
    initial_symbols = initial.get_chemical_symbols()
    final_symbols = final.get_chemical_symbols()
    if Counter(initial_symbols) != Counter(final_symbols):
        raise ValueError("Initial and final endpoints must have the same elemental composition.")

    if initial_symbols.count(anchor_symbol) < 3:
        counts = Counter(initial_symbols)
        candidates = [s for s, c in counts.most_common() if c >= 3 and s not in {"H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I"}]
        if candidates:
            anchor_symbol = candidates[0]

    initial_anchor = np.asarray(
        [initial.positions[i] for i, symbol in enumerate(initial_symbols) if symbol == anchor_symbol]
    )
    final_anchor = np.asarray(
        [final.positions[i] for i, symbol in enumerate(final_symbols) if symbol == anchor_symbol]
    )
    if len(initial_anchor) < 3:
        raise ValueError(f"At least three {anchor_symbol} atoms are required for rigid alignment.")

    initial_center = initial_anchor.mean(axis=0)
    final_center = final_anchor.mean(axis=0)
    fixed = initial_anchor - initial_center
    moving = final_anchor - final_center

    # Sorted intracluster distances are rotation/permutation invariant and give
    # a reliable initial correspondence for a slightly distorted nanoparticle.
    fixed_fingerprints = np.sort(cdist(fixed, fixed), axis=1)
    moving_fingerprints = np.sort(cdist(moving, moving), axis=1)
    rows, columns = linear_sum_assignment(cdist(fixed_fingerprints, moving_fingerprints))
    fingerprint_mapping = np.empty(len(rows), dtype=int)
    fingerprint_mapping[rows] = columns
    candidates = [_kabsch(moving[fingerprint_mapping], fixed)]
    if int(rotation_starts) > 0:
        from scipy.spatial.transform import Rotation

        candidates.extend(
            Rotation.random(int(rotation_starts), random_state=0).as_matrix()
        )

    initial_positions = np.asarray(initial.positions)
    final_positions = np.asarray(final.positions)
    anchor_mask = np.asarray([symbol == anchor_symbol for symbol in initial_symbols])
    weights = np.where(anchor_mask, max(1.0, float(anchor_weight)), 1.0)
    weighted_scale = np.sqrt(weights)[:, None]
    best: tuple[float, Atoms, dict[str, float]] | None = None
    for seed_rotation in candidates:
        rotation = np.asarray(seed_rotation)
        previous_order: np.ndarray | None = None
        for _ in range(max(1, int(max_iterations))):
            transformed = (final_positions - final_center) @ rotation + initial_center
            order_array = np.empty(len(initial_symbols), dtype=int)
            for symbol in set(initial_symbols):
                initial_group = [i for i, value in enumerate(initial_symbols) if value == symbol]
                final_group = [i for i, value in enumerate(final_symbols) if value == symbol]
                group_rows, group_columns = linear_sum_assignment(
                    cdist(initial_positions[initial_group], transformed[final_group])
                )
                for row, column in zip(group_rows.tolist(), group_columns.tolist()):
                    order_array[initial_group[row]] = final_group[column]
            new_rotation = _kabsch(
                (final_positions[order_array] - final_center) * weighted_scale,
                (initial_positions - initial_center) * weighted_scale,
            )
            if previous_order is not None and np.array_equal(order_array, previous_order):
                rotation = new_rotation
                break
            previous_order = order_array
            rotation = new_rotation

        transformed = (final_positions - final_center) @ rotation + initial_center
        order = order_array.tolist()

        aligned = final[order]
        aligned.positions = transformed[order]
        shifts = np.linalg.norm(initial_positions - np.asarray(aligned.positions), axis=1)
        metrics = {
            "rmsd_all_A": float(np.sqrt(np.mean(shifts**2))),
            "max_shift_all_A": float(np.max(shifts)),
            "rmsd_anchor_A": float(np.sqrt(np.mean(shifts[anchor_mask] ** 2))),
            "max_shift_anchor_A": float(np.max(shifts[anchor_mask])),
            "weighted_rmsd_A": float(np.sqrt(np.average(shifts**2, weights=weights))),
        }
        score = metrics["weighted_rmsd_A"]
        if best is None or score < best[0]:
            best = score, aligned, metrics

    assert best is not None
    return best[1], best[2]
