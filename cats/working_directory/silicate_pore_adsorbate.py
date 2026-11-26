from __future__ import annotations

from itertools import combinations
from typing import Callable, Sequence

import numpy as np
from ase import Atoms
from ase.data import covalent_radii


SILICATE_FRAMEWORK_SYMBOLS = {"Si", "O", "Al", "H"}
DEFAULT_HEIGHTS = (1.45, 1.70, 1.95, 2.50, 3.00, 3.50)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("Cannot normalise a zero vector.")
    return np.asarray(vector, dtype=float) / norm


def _rotation_matrix(axis: np.ndarray, angle_degrees: float) -> np.ndarray:
    axis = _unit(axis)
    angle = float(np.deg2rad(angle_degrees))
    kx, ky, kz = axis
    skew = np.array([[0.0, -kz, ky], [kz, 0.0, -kx], [-ky, kx, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def _rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = _unit(source)
    target = _unit(target)
    cross = np.cross(source, target)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot > 1.0 - 1.0e-12:
        return np.eye(3)
    if dot < -1.0 + 1.0e-12:
        axis = np.cross(source, [1.0, 0.0, 0.0])
        if float(np.linalg.norm(axis)) <= 1.0e-12:
            axis = np.cross(source, [0.0, 1.0, 0.0])
        return _rotation_matrix(axis, 180.0)
    return _rotation_matrix(cross, float(np.rad2deg(np.arccos(dot))))


def _lateral_axis(inward: np.ndarray) -> np.ndarray:
    trial = np.cross(inward, [0.0, 0.0, 1.0])
    if float(np.linalg.norm(trial)) <= 1.0e-12:
        trial = np.cross(inward, [0.0, 1.0, 0.0])
    return _unit(trial)


def _anchor_index(adsorbate: Atoms, anchor_symbol: str | None) -> int:
    if anchor_symbol:
        for idx, atom in enumerate(adsorbate):
            if atom.symbol == anchor_symbol:
                return idx
        raise ValueError(f"Adsorbate has no anchor atom with symbol {anchor_symbol!r}.")
    for symbol in ("O", "N", "S", "P", "C"):
        for idx, atom in enumerate(adsorbate):
            if atom.symbol == symbol:
                return idx
    return 0


def _clash_metrics(framework: Atoms, adsorbate: Atoms, clash_scale: float) -> dict[str, object]:
    distances = np.linalg.norm(
        framework.positions[:, None, :] - adsorbate.positions[None, :, :],
        axis=2,
    )
    framework_radii = np.array([covalent_radii[number] for number in framework.numbers])
    adsorbate_radii = np.array([covalent_radii[number] for number in adsorbate.numbers])
    radii_sum = framework_radii[:, None] + adsorbate_radii[None, :]
    ratios = distances / np.maximum(radii_sum, 1.0e-12)
    min_index = np.unravel_index(int(np.argmin(ratios)), ratios.shape)
    return {
        "ok": bool(float(ratios[min_index]) >= float(clash_scale)),
        "min_distance": float(distances[min_index]),
        "min_cutoff": float(radii_sum[min_index] * float(clash_scale)),
        "min_ratio": float(ratios[min_index]),
    }


def _resolve_clash(
    framework: Atoms,
    adsorbate: Atoms,
    push_direction: np.ndarray,
    *,
    clash_scale: float,
    push_step: float,
    max_push_steps: int,
) -> tuple[Atoms, dict[str, object], int, float]:
    metrics = _clash_metrics(framework, adsorbate, clash_scale)
    if float(np.linalg.norm(push_direction)) <= 1.0e-12:
        return adsorbate, metrics, 0, 0.0
    shift = _unit(push_direction) * float(push_step)
    for step in range(int(max_push_steps)):
        if bool(metrics["ok"]):
            return adsorbate, metrics, step, step * float(push_step)
        adsorbate.positions[:] += shift
        metrics = _clash_metrics(framework, adsorbate, clash_scale)
    return adsorbate, metrics, int(max_push_steps), int(max_push_steps) * float(push_step)


def _placed_adsorbate(
    template: Atoms,
    site: np.ndarray,
    inward: np.ndarray,
    height: float,
    *,
    rotation_degrees: float,
    flipped: bool,
    tilt_degrees: float,
    anchor_symbol: str | None,
) -> Atoms:
    adsorbate = template.copy()
    anchor = _anchor_index(adsorbate, anchor_symbol)
    anchor_to_com = adsorbate.get_center_of_mass() - adsorbate.positions[anchor]
    rotation = _rotation_between(anchor_to_com, inward)
    tilt = -float(tilt_degrees) if flipped else float(tilt_degrees)
    rotation = _rotation_matrix(_lateral_axis(inward), tilt) @ rotation
    rotation = _rotation_matrix(inward, rotation_degrees) @ rotation
    adsorbate.positions[:] = (adsorbate.positions - adsorbate.positions[anchor]) @ rotation.T
    adsorbate.positions[:] += site + inward * float(height)
    return adsorbate


def _silicate_sites(framework: Atoms, site_atom_indices: Sequence[int]):
    sites = []
    for idx in site_atom_indices:
        sites.append(("atop", (int(idx),), framework.positions[int(idx)].copy()))
    for i, j in combinations([int(idx) for idx in site_atom_indices], 2):
        sites.append(("bridge", (i, j), (framework.positions[i] + framework.positions[j]) / 2.0))
    return sites


def generate_silicate_pore_adsorbate_candidates(
    framework: Atoms,
    adsorbates: Sequence[Atoms],
    *,
    origin_atom_index: int = 0,
    pore_triangle_indices: Sequence[int] = (1, 2, 3),
    site_atom_indices: Sequence[int] | None = None,
    heights: Sequence[float] = DEFAULT_HEIGHTS,
    rotation_step_degrees: float = 60.0,
    include_flipped_orientations: bool = True,
    tilt_degrees: float = 25.0,
    anchor_symbol: str | None = None,
    clash_scale: float = 0.8,
    push_step: float = 0.1,
    max_push_steps: int = 8,
    max_structures: int | None = None,
    progress_callback: Callable[[str], None] | None = None,
    progress_interval: int = 500,
) -> tuple[list[Atoms], list[dict[str, object]]]:
    symbols = set(framework.get_chemical_symbols())
    if not symbols <= SILICATE_FRAMEWORK_SYMBOLS:
        raise ValueError(
            "Silicate pore placement only accepts Si/O/Al/H frameworks; "
            f"got {sorted(symbols)}."
        )
    max_index = len(framework) - 1
    indices_to_check = [int(origin_atom_index), *[int(i) for i in pore_triangle_indices]]
    if any(idx < 0 or idx > max_index for idx in indices_to_check):
        raise IndexError(f"Silicate pore indices must be in 0..{max_index}.")

    origin = framework.positions[int(origin_atom_index)]
    pore_center = framework.positions[[int(i) for i in pore_triangle_indices]].mean(axis=0)
    inward = _unit(pore_center - origin)
    site_indices = tuple(site_atom_indices or (int(origin_atom_index), *[int(i) for i in pore_triangle_indices]))
    sites = _silicate_sites(framework, site_indices)
    rotations = np.arange(0.0, 360.0, float(rotation_step_degrees))
    flip_options = (False, True) if include_flipped_orientations else (False,)
    total_trials = len(adsorbates) * len(sites) * len(heights) * len(flip_options) * len(rotations)
    if progress_callback is not None:
        progress_callback(
            f"[SilicatePore] Generating candidates: testing up to {total_trials} placements "
            f"({len(sites)} sites, {len(heights)} heights, {len(rotations)} rotations)"
        )

    candidates: list[Atoms] = []
    metadata: list[dict[str, object]] = []
    trials = 0
    for adsorbate_idx, template in enumerate(adsorbates, start=1):
        label = str(template.info.get("label") or template.get_chemical_formula(mode="hill"))
        for site_kind, site_indices_tuple, site in sites:
            if float(np.dot(site - origin, inward)) < -1.0e-6:
                continue
            for height in heights:
                for flipped in flip_options:
                    for rotation_degrees in rotations:
                        trials += 1
                        if (
                            progress_callback is not None
                            and progress_interval > 0
                            and trials % int(progress_interval) == 0
                        ):
                            progress_callback(
                                f"[SilicatePore] Candidate generation progress: "
                                f"{trials}/{total_trials} placements tested, {len(candidates)} accepted"
                            )
                        placed = _placed_adsorbate(
                            template,
                            site,
                            inward,
                            float(height),
                            rotation_degrees=float(rotation_degrees),
                            flipped=bool(flipped),
                            tilt_degrees=float(tilt_degrees),
                            anchor_symbol=anchor_symbol,
                        )
                        pore_dot = float(np.dot(placed.get_center_of_mass() - site, inward))
                        if pore_dot <= 0.0:
                            continue
                        placed, clash, push_steps, push_distance = _resolve_clash(
                            framework,
                            placed,
                            pore_center - site,
                            clash_scale=float(clash_scale),
                            push_step=float(push_step),
                            max_push_steps=int(max_push_steps),
                        )
                        pore_dot = float(np.dot(placed.get_center_of_mass() - site, inward))
                        if pore_dot <= 0.0 or not bool(clash["ok"]):
                            continue
                        candidate = framework + placed
                        candidates.append(candidate)
                        metadata.append(
                            {
                                "adsorbate_index": int(adsorbate_idx),
                                "adsorbate_label": label,
                                "adsorbate_n_atoms": len(template),
                                "placement": f"{site_kind}:{','.join(map(str, site_indices_tuple))}",
                                "height": float(height),
                                "rotation_degrees": float(rotation_degrees),
                                "flipped": bool(flipped),
                                "pore_origin_atom_index": int(origin_atom_index),
                                "pore_triangle_indices": ",".join(map(str, pore_triangle_indices)),
                                "pore_vector": ",".join(f"{value:.8f}" for value in inward),
                                "site_position": ",".join(f"{value:.8f}" for value in site),
                                "pore_dot_com_minus_site": pore_dot,
                                "min_framework_adsorbate_distance": float(clash["min_distance"]),
                                "min_framework_adsorbate_clash_cutoff": float(clash["min_cutoff"]),
                                "min_framework_adsorbate_radius_ratio": float(clash["min_ratio"]),
                                "clash_push_steps": int(push_steps),
                                "clash_push_distance": float(push_distance),
                            }
                        )
                        if max_structures is not None and len(candidates) >= int(max_structures):
                            if progress_callback is not None:
                                progress_callback(
                                    f"[SilicatePore] Candidate generation stopped at max_structures="
                                    f"{max_structures}: {len(candidates)} accepted"
                                )
                            return candidates, metadata
    if progress_callback is not None:
        progress_callback(
            f"[SilicatePore] Candidate generation complete: "
            f"{trials}/{total_trials} placements tested, {len(candidates)} accepted"
        )
    return candidates, metadata


def _self_check() -> None:
    framework = Atoms(
        "SiO3",
        positions=[[0, 0, 0], [-1, -1, 0], [1, -1, 0], [0, -1, -1]],
    )
    methanol = Atoms("COH4", positions=[[0, 0.7, 0], [0, -0.7, 0], [0, -1.6, 0], [1, 1, 0], [-1, 1, 0], [0, 1, 1]])
    candidates, rows = generate_silicate_pore_adsorbate_candidates(
        framework,
        [methanol],
        heights=(1.5,),
        rotation_step_degrees=180.0,
    )
    assert candidates
    assert rows
    assert "site_position" in rows[0]
    assert min(row["min_framework_adsorbate_radius_ratio"] for row in rows) >= 0.8


if __name__ == "__main__":
    _self_check()
    print("self-check passed")
