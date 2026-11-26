from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

_HC_EV_NM = 1239.841984


def _is_valid_number(value: Optional[float]) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _to_float_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _gaussian_score(delta: float, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    return math.exp(-0.5 * (delta / sigma) ** 2)


def energy_from_wavelength_nm(wavelength_nm: float) -> Optional[float]:
    if not _is_valid_number(wavelength_nm) or wavelength_nm == 0:
        return None
    return _HC_EV_NM / float(wavelength_nm)


@dataclass(frozen=True)
class IRootCandidate:
    state: int
    wavelength_nm: float
    intensity: float
    energy_ev: Optional[float]
    ct_e: Optional[float] = None
    ct_pct: Optional[float] = None
    ct_net: Optional[float] = None
    score: Optional[float] = None
    similarity: Optional[float] = None
    transition_overlap: Optional[float] = None
    score_components: Optional[Dict[str, float]] = None
    transition_vector: Optional[Dict[str, float]] = None
    spatial_overlap: Optional[float] = None

    @property
    def ct_magnitude(self) -> float:
        if _is_valid_number(self.ct_e):
            return abs(float(self.ct_e))
        if _is_valid_number(self.ct_pct):
            return abs(float(self.ct_pct))
        return 0.0


def _sum_ct_terms_from_row(row: pd.Series) -> Optional[float]:
    ct_terms = []
    for col in row.index:
        if not isinstance(col, str):
            continue
        if "->" not in col and "<-" not in col:
            continue
        value = _to_float_or_none(row.get(col))
        if _is_valid_number(value):
            ct_terms.append(abs(float(value)))
    return sum(ct_terms) if ct_terms else None


def _net_ct_terms_from_row(row: pd.Series) -> Optional[float]:
    """Return signed net CT from IFCT terms.

    Convention:
    - terms like ``1->2`` are negative (more electrons on adsorbate)
    - terms like ``1<-2`` are positive (fewer electrons on adsorbate)
    """
    net_ct = 0.0
    found = False
    for col in row.index:
        if not isinstance(col, str):
            continue
        value = _to_float_or_none(row.get(col))
        if not _is_valid_number(value):
            continue
        amount = abs(float(value))
        if "<-" in col:
            net_ct += amount
            found = True
            continue
        if "->" in col:
            net_ct -= amount
            found = True
            continue
    if not found:
        return None
    return float(net_ct)


def _normalise_transition_vector(
    vector: Optional[Dict[str, float]],
) -> Optional[Dict[str, float]]:
    if not vector:
        return None
    cleaned: Dict[str, float] = {}
    for raw_key, raw_value in vector.items():
        key = str(raw_key).strip()
        if not key:
            continue
        value = _to_float_or_none(raw_value)
        if value is None:
            continue
        # Use squared amplitudes as sign-invariant transition weights.
        cleaned[key] = float(value) * float(value)
    if not cleaned:
        return None
    norm = sum(cleaned.values())
    if norm <= 0.0:
        return None
    return {key: (value / norm) for key, value in cleaned.items()}


def _cosine_overlap(
    left: Optional[Dict[str, float]],
    right: Optional[Dict[str, float]],
) -> Optional[float]:
    if not left or not right:
        return None
    left_norm = math.sqrt(sum(float(v) * float(v) for v in left.values()))
    right_norm = math.sqrt(sum(float(v) * float(v) for v in right.values()))
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    overlap = 0.0
    if len(left) <= len(right):
        for key, value in left.items():
            overlap += float(value) * float(right.get(key, 0.0))
    else:
        for key, value in right.items():
            overlap += float(value) * float(left.get(key, 0.0))
    value = overlap / (left_norm * right_norm)
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def _row_to_candidate(
    row: pd.Series,
    *,
    state_vectors: Optional[Dict[int, Dict[str, float]]] = None,
    state_spatial_overlaps: Optional[Dict[int, float]] = None,
) -> IRootCandidate:
    wavelength_nm = float(row["nm"])
    intensity = float(row["f"])
    ct_e = _to_float_or_none(row.get("CT(e)"))
    ct_pct = _to_float_or_none(row.get("CT(%)"))
    ct_net = _net_ct_terms_from_row(row)
    if ct_e is None and ct_pct is None:
        ct_pct = _sum_ct_terms_from_row(row)
    energy_ev = energy_from_wavelength_nm(wavelength_nm)
    spatial_overlap = None
    if state_spatial_overlaps is not None:
        spatial_overlap = state_spatial_overlaps.get(int(row["state"]))
    return IRootCandidate(
        state=int(row["state"]),
        wavelength_nm=wavelength_nm,
        intensity=intensity,
        energy_ev=energy_ev,
        ct_e=ct_e,
        ct_pct=ct_pct,
        ct_net=ct_net,
        transition_vector=_normalise_transition_vector(
            (state_vectors or {}).get(int(row["state"]))
        ),
        spatial_overlap=spatial_overlap,
    )


def score_iroot_candidates(
    df: pd.DataFrame,
    *,
    target_wavelength_nm: float,
    previous_choice: Optional[IRootCandidate] = None,
    previous_state: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    wavelength_sigma_nm: float = 20.0,
    similarity_sigma_nm: float = 15.0,
    similarity_sigma_ct: float = 0.05,
    state_vectors: Optional[Dict[int, Dict[str, float]]] = None,
    state_spatial_overlaps: Optional[Dict[int, float]] = None,
) -> List[IRootCandidate]:
    if df is None or df.empty:
        return []

    candidates = [
        _row_to_candidate(
            row,
            state_vectors=state_vectors,
            state_spatial_overlaps=state_spatial_overlaps,
        )
        for _, row in df.iterrows()
    ]

    max_intensity = max((c.intensity for c in candidates), default=0.0)
    max_ct = max((c.ct_magnitude for c in candidates), default=0.0)

    prev = previous_choice
    if prev is None and previous_state is not None:
        prev_rows = df[df["state"] == previous_state]
        if not prev_rows.empty:
            prev = _row_to_candidate(
                prev_rows.iloc[0],
                state_vectors=state_vectors,
                state_spatial_overlaps=state_spatial_overlaps,
            )

    use_similarity = prev is not None
    ct_scale = 1.0
    if use_similarity and prev is not None:
        # Keep CT similarity scale robust across reaction coordinates by using
        # a relative delta instead of absolute CT units.
        ct_scale = max(max_ct, prev.ct_magnitude, 1.0)
    if weights is None:
        weights = {
            "intensity": 0.4,
            "ct": 0.4,
            "wavelength": 0.2,
            "similarity": 0.1,
        }

    scored: List[IRootCandidate] = []
    for cand in candidates:
        intensity_norm = cand.intensity / max_intensity if max_intensity > 0 else 0.0
        ct_norm = cand.ct_magnitude / max_ct if max_ct > 0 else 0.0
        wavelength_score = _gaussian_score(
            abs(cand.wavelength_nm - target_wavelength_nm), wavelength_sigma_nm
        )

        similarity_score = 0.0
        similarity_nm_score = 0.0
        similarity_ct_score = 1.0
        transition_overlap_score: Optional[float] = None
        spatial_overlap_score: Optional[float] = cand.spatial_overlap
        if use_similarity:
            similarity_nm_score = _gaussian_score(
                abs(cand.wavelength_nm - prev.wavelength_nm), similarity_sigma_nm
            )
            if prev is not None and (max_ct > 0.0 or prev.ct_magnitude > 0.0):
                ct_delta = abs(cand.ct_magnitude - prev.ct_magnitude) / ct_scale
                similarity_ct_score = _gaussian_score(ct_delta, similarity_sigma_ct)
            transition_overlap_score = _cosine_overlap(
                cand.transition_vector,
                prev.transition_vector if prev is not None else None,
            )
            if spatial_overlap_score is not None:
                # 3D spatial overlap provides direct orbital/density continuity
                similarity_score = spatial_overlap_score * similarity_ct_score
            elif transition_overlap_score is not None:
                # Require both overlap continuity and CT continuity when overlap
                # vectors are available.
                similarity_score = transition_overlap_score * similarity_ct_score
            else:
                # If overlap vectors are unavailable, rely on continuity in
                # wavelength and CT character.
                similarity_score = similarity_nm_score * similarity_ct_score

        total = (
            weights.get("intensity", 0.0) * intensity_norm
            + weights.get("ct", 0.0) * ct_norm
            + weights.get("wavelength", 0.0) * wavelength_score
        )
        if use_similarity:
            total += weights.get("similarity", 0.0) * similarity_score

        scored.append(
            replace(
                cand,
                score=total,
                similarity=similarity_score if use_similarity else None,
                transition_overlap=transition_overlap_score,
                score_components={
                    "intensity": intensity_norm,
                    "ct": ct_norm,
                    "wavelength": wavelength_score,
                    "similarity": similarity_score if use_similarity else 0.0,
                    "similarity_nm": similarity_nm_score if use_similarity else 0.0,
                    "similarity_ct": similarity_ct_score if use_similarity else 0.0,
                    "similarity_overlap": (
                        transition_overlap_score if transition_overlap_score is not None else 0.0
                    ),
                    "spatial_overlap": (
                        spatial_overlap_score if spatial_overlap_score is not None else 0.0
                    ),
                },
            )
        )

    scored.sort(
        key=lambda c: (
            -(
                float(c.score)
                if c.score is not None and math.isfinite(float(c.score))
                else -math.inf
            ),
            abs(float(c.wavelength_nm) - float(target_wavelength_nm)),
            -float(c.intensity),
            int(c.state),
        )
    )
    return scored


def select_iroot_by_score(
    df: pd.DataFrame,
    *,
    target_wavelength_nm: float,
    previous_choice: Optional[IRootCandidate] = None,
    previous_state: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
    wavelength_sigma_nm: float = 20.0,
    similarity_sigma_nm: float = 15.0,
    similarity_sigma_ct: float = 0.05,
    min_similarity: float = 0.0,
    min_relative_intensity: float = 0.0,
    min_transition_overlap: float = 0.0,
    min_spatial_overlap: float = 0.0,
    min_ct_similarity: float = 0.0,
    intensity_reference_window_nm: Optional[float] = 200.0,
    require_transition_overlap: bool = False,
    require_spatial_overlap: bool = False,
    state_vectors: Optional[Dict[int, Dict[str, float]]] = None,
    state_spatial_overlaps: Optional[Dict[int, float]] = None,
) -> Tuple[Optional[IRootCandidate], List[IRootCandidate]]:
    scored = score_iroot_candidates(
        df,
        target_wavelength_nm=target_wavelength_nm,
        previous_choice=previous_choice,
        previous_state=previous_state,
        weights=weights,
        wavelength_sigma_nm=wavelength_sigma_nm,
        similarity_sigma_nm=similarity_sigma_nm,
        similarity_sigma_ct=similarity_sigma_ct,
        state_vectors=state_vectors,
        state_spatial_overlaps=state_spatial_overlaps,
    )
    if not scored:
        return None, scored

    filtered = scored
    if min_relative_intensity > 0.0:
        reference_candidates = scored
        if intensity_reference_window_nm is not None and intensity_reference_window_nm > 0.0:
            window = float(intensity_reference_window_nm)
            local_candidates = [
                cand
                for cand in scored
                if abs(float(cand.wavelength_nm) - float(target_wavelength_nm)) <= window
            ]
            if local_candidates:
                reference_candidates = local_candidates
        strongest = max(float(cand.intensity) for cand in reference_candidates)
        cutoff = strongest * float(min_relative_intensity)
        filtered = [
            cand
            for cand in filtered
            if float(cand.intensity) + 1.0e-15 >= cutoff
        ]
        if not filtered:
            return None, scored

    if require_spatial_overlap:
        spatial_candidates = [cand for cand in filtered if cand.spatial_overlap is not None]
        if not spatial_candidates:
            return None, scored
        filtered = spatial_candidates

    if min_spatial_overlap > 0.0:
        filtered = [
            cand
            for cand in filtered
            if (cand.spatial_overlap or 0.0) >= float(min_spatial_overlap)
        ]
        if not filtered:
            return None, scored

    if require_transition_overlap:
        overlap_candidates = [cand for cand in filtered if cand.transition_overlap is not None]
        if not overlap_candidates:
            return None, scored
        filtered = overlap_candidates

    if min_transition_overlap > 0.0:
        filtered = [
            cand
            for cand in filtered
            if (cand.transition_overlap or 0.0) >= float(min_transition_overlap)
        ]
        if not filtered:
            return None, scored

    if min_ct_similarity > 0.0:
        threshold = float(min_ct_similarity)
        filtered = [
            cand
            for cand in filtered
            if (cand.score_components or {}).get("similarity_ct", 1.0) >= threshold
        ]
        if not filtered:
            return None, scored

    best = filtered[0]
    if min_similarity > 0.0:
        has_similarity = any(cand.similarity is not None for cand in filtered)
        if has_similarity:
            filtered_by_similarity = [
                cand for cand in filtered if (cand.similarity or 0.0) >= min_similarity
            ]
            if not filtered_by_similarity:
                return None, scored
            best = filtered_by_similarity[0]
    return best, scored


def compute_state_spatial_overlaps(
    ref_cubes: Dict[str, Any],
    candidate_cubes_map: Dict[int, Dict[str, Any]],
    *,
    tracker: Optional[Any] = None,
) -> Dict[int, float]:
    """Compute 3D spatial NTO+CDD overlaps between reference and candidate states."""
    if not ref_cubes or not candidate_cubes_map:
        return {}
    if tracker is None:
        from cats.multiwfn.state_tracker import StateTracker
        tracker = StateTracker(root_state=0)
    overlaps: Dict[int, float] = {}
    for state, cand_cubes in candidate_cubes_map.items():
        try:
            res = tracker.score_candidate(ref_cubes, cand_cubes, 0, int(state))
            overlaps[int(state)] = float(res.composite_score)
        except Exception:
            continue
    return overlaps

