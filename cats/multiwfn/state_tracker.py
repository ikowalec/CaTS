"""Generic Excited State Tracking and Diabatization Engine.

Tracks a specific excited state along an arbitrary coordinate:
- Ground-state MEP post-simulation analysis (image-to-image tracking)
- Excited-state geometry optimization (step-to-step state following / crossing detection)
- Excited-state ML-NEB (on-the-fly root state tracking to prevent root-flipping)

Uses 3D spatial NTO and CDD grid overlaps (cats.multiwfn.nto_overlap)
conditioned on Kabsch cluster alignment.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from cats.multiwfn.nto_overlap import (
    align_cluster,
    compare,
    compare_nto_density_sets,
    compare_nto_pair,
    read_amplitude,
)

logger = logging.getLogger(__name__)
MIN_ORBITAL_OVERLAP_SQ = 0.30


def _resolve_orca_file(directory: Path, names: List[str]) -> Path:
    candidates = [directory / name for name in names]
    for nested in ("DFT/final_tddft_spectrum", "final_tddft_spectrum"):
        candidates.extend(directory / nested / name for name in names)
    found = next((path for path in candidates if path.is_file()), None)
    if found is None:
        raise FileNotFoundError(f"None of {names!r} found in {directory}")
    return found


def _load_state_cubes(density_dir: Path, states: List[int]) -> Dict[int, Dict[str, Any]]:
    cubes_map: Dict[int, Dict[str, Any]] = {}
    for state in states:
        hole_paths = sorted(density_dir.glob(f"state_{state:03d}_nto*hole*.cub"))
        particle_paths = sorted(density_dir.glob(f"state_{state:03d}_nto*particle*.cub"))
        cdd_paths = sorted(density_dir.glob(f"state_{state:03d}*cdd*.cub"))
        pairs = []
        for hole_path, particle_path in zip(hole_paths, particle_paths):
            if hole_path.is_file() and particle_path.is_file():
                pairs.append(
                    {
                        "hole": read_amplitude(hole_path),
                        "particle": read_amplitude(particle_path),
                        "hole_path": hole_path,
                        "particle_path": particle_path,
                        "weight": 1.0,
                    }
                )
        if not pairs:
            continue
        cdd_path = cdd_paths[0] if cdd_paths else None
        cubes_map[state] = {
            "nto_hole": pairs[0]["hole"],
            "nto_particle": pairs[0]["particle"],
            "multi_pairs": pairs,
            "cdd": read_amplitude(cdd_path) if cdd_path and cdd_path.is_file() else None,
            "hole_path": pairs[0]["hole_path"],
            "particle_path": pairs[0]["particle_path"],
            "cdd_path": cdd_path,
            "all_pair_paths": [path for pair in pairs for path in (pair["hole_path"], pair["particle_path"])],
        }
    return cubes_map


def generate_state_cubes(
    directory: Path | str,
    states: List[int],
    working_dir: Path | str,
    *,
    density_workers: int = 4,
    nto_max_pairs: int = 3,
    nto_target_coverage: float = 0.8,
    grid_quality: int = 2,
) -> Dict[int, Dict[str, Any]]:
    """Generate or reload NTO/CDD cubes used by state tracking."""
    from cats.multiwfn.density_analysis import generate_excitation_density_analysis

    directory = Path(directory)
    working_dir = Path(working_dir)
    density_dir = working_dir / "densities"
    density_dir.mkdir(parents=True, exist_ok=True)
    state_numbers = sorted({int(state) for state in states})
    cubes_map = _load_state_cubes(density_dir, state_numbers)
    missing = [state for state in state_numbers if state not in cubes_map]
    if not missing:
        return cubes_map

    try:
        molden = _resolve_orca_file(directory, ["orca.molden.input", "orca.molden"])
    except FileNotFoundError:
        gbw = _resolve_orca_file(directory, ["orca.gbw"])
        converter = shutil.which("orca_2mkl")
        if not converter:
            orcadir = str(os.environ.get("ORCADIR") or "").strip()
            candidate = Path(orcadir) / "orca_2mkl" if orcadir else None
            converter = str(candidate) if candidate and candidate.is_file() else None
        if not converter:
            raise RuntimeError("orca_2mkl not found; cannot prepare state-tracking cubes")
        local_gbw = working_dir / "orca.gbw"
        shutil.copy2(gbw, local_gbw)
        import subprocess

        subprocess.run([converter, "orca", "-molden"], cwd=working_dir, check=True)
        molden = working_dir / "orca.molden.input"
    orca_out = _resolve_orca_file(directory, ["orca.out"])
    generate_excitation_density_analysis(
        molden,
        orca_out,
        [{"excitation_number": state, "state": state} for state in missing],
        output_directory=density_dir,
        excitation_numbers=missing,
        grid_quality=int(grid_quality),
        include_nto=True,
        export_pov_scenes=False,
        delete_cubes_after_export=False,
        max_workers=min(int(density_workers), len(missing)),
        nto_max_pairs=int(nto_max_pairs),
        nto_target_coverage=float(nto_target_coverage),
    )
    return _load_state_cubes(density_dir, state_numbers)


@dataclass
class StateTrackPoint:
    """Recorded state at one step along the path."""
    step_index: int
    step_id: str
    state_number: int
    energy_ev: float
    wavelength_nm: float
    oscillator_strength: float
    ct_metrics: Dict[str, float] = field(default_factory=dict)
    hole_overlap_sq: Optional[float] = None
    electron_overlap_sq: Optional[float] = None
    paired_score: Optional[float] = None
    cdd_overlap_sq: Optional[float] = None
    cluster_rmsd_A: Optional[float] = None
    confidence: str = "INITIAL"


@dataclass
class OverlapResult:
    """Pairwise overlap between two states."""
    ref_state: int
    cand_state: int
    hole_overlap_sq: float
    electron_overlap_sq: float
    paired_score: float
    cdd_overlap_sq: Optional[float] = None
    multi_pair_paired_score: Optional[float] = None
    cluster_rmsd_A: float = 0.0
    spatial_score: float = 0.0
    energy_score: float = 1.0
    intensity_score: float = 1.0
    composite_score: float = 0.0


class StateTracker:
    """Generic excited-state tracking pipeline."""

    def __init__(
        self,
        *,
        root_state: int,
        energy_window_ev: float = 0.35,
        min_paired_overlap: float = 0.20,
        weight_paired_nto: float = 0.70,
        weight_cdd: float = 0.30,
        mode: str = "mep",  # 'mep', 'opt', or 'mlneb'
    ):
        self.root_state = int(root_state)
        self.energy_window_ev = float(energy_window_ev)
        if not math.isfinite(self.energy_window_ev) or self.energy_window_ev <= 0:
            raise ValueError("energy_window_ev must be finite and positive")
        self.min_paired_overlap = float(min_paired_overlap)
        self.weight_paired_nto = float(weight_paired_nto)
        self.weight_cdd = float(weight_cdd)
        self.mode = mode
        self.history: List[StateTrackPoint] = []

    def score_candidate(
        self,
        ref_cubes: Dict[str, Any],
        cand_cubes: Dict[str, Any],
        ref_state: int,
        cand_state: int,
    ) -> OverlapResult:
        """Compute spatial NTO + CDD overlap between reference and candidate."""
        # 1. Leading NTO pair overlap
        pair_res = compare_nto_pair(
            ref_cubes["nto_hole"],
            ref_cubes["nto_particle"],
            cand_cubes["nto_hole"],
            cand_cubes["nto_particle"],
        )
        hole_sq = float(pair_res["hole"]["squared_overlap_mean"])
        elec_sq = float(pair_res["particle"]["squared_overlap_mean"])
        paired = float(pair_res["paired_overlap"])
        rmsd = float(pair_res["hole"]["cluster_rmsd_A"])

        # All-to-all cross-matching across multiple pairs to resolve pair order flips
        ref_pairs = ref_cubes.get("multi_pairs") or [{"hole": ref_cubes["nto_hole"], "particle": ref_cubes["nto_particle"]}]
        cand_pairs = cand_cubes.get("multi_pairs") or [{"hole": cand_cubes["nto_hole"], "particle": cand_cubes["nto_particle"]}]
        if len(ref_pairs) > 1 or len(cand_pairs) > 1:
            for r in ref_pairs:
                for c in cand_pairs:
                    match = compare_nto_pair(r["hole"], r["particle"], c["hole"], c["particle"])
                    if float(match["paired_overlap"]) > paired:
                        paired = float(match["paired_overlap"])
                        hole_sq = float(match["hole"]["squared_overlap_mean"])
                        elec_sq = float(match["particle"]["squared_overlap_mean"])
                        rmsd = float(match["hole"]["cluster_rmsd_A"])
                        pair_res = match

        # 2. CDD overlap (if available)
        cdd_sq = None
        if "cdd" in ref_cubes and "cdd" in cand_cubes and ref_cubes["cdd"] is not None and cand_cubes["cdd"] is not None:
            cdd_res = compare(ref_cubes["cdd"], cand_cubes["cdd"])
            cdd_sq = float(cdd_res["squared_overlap_mean"])

        # 3. Multi-pair density overlap (if multi-pair cubes provided)
        multi_paired = None
        if "multi_pairs" in ref_cubes and "multi_pairs" in cand_cubes:
            try:
                multi_res = compare_nto_density_sets(
                    ref_cubes["multi_pairs"],
                    cand_cubes["multi_pairs"],
                )
                multi_paired = float(multi_res.get("paired_density_overlap", multi_res.get("paired_overlap", 0.0)))
            except Exception:
                multi_paired = None

        # Composite match score: best matching pair or multi-pair density set
        effective_paired = max(paired, multi_paired) if multi_paired is not None else paired
        if cdd_sq is not None:
            spatial_score = self.weight_paired_nto * effective_paired + self.weight_cdd * cdd_sq
        else:
            spatial_score = effective_paired

        return OverlapResult(
            ref_state=ref_state,
            cand_state=cand_state,
            hole_overlap_sq=hole_sq,
            electron_overlap_sq=elec_sq,
            paired_score=effective_paired,
            cdd_overlap_sq=cdd_sq,
            multi_pair_paired_score=multi_paired,
            cluster_rmsd_A=rmsd,
            spatial_score=spatial_score,
            composite_score=spatial_score,
        )

    def select_next_state(
        self,
        ref_point: StateTrackPoint,
        ref_cubes: Dict[str, Any],
        candidate_states: List[Dict[str, Any]],
        candidate_cubes_map: Dict[int, Dict[str, Any]],
        step_index: int,
        step_id: str,
        ct_evaluator: Optional[Callable[[List[int]], Dict[int, Dict[str, float]]]] = None,
    ) -> Tuple[StateTrackPoint, List[OverlapResult]]:
        """Evaluate candidate states and select the adiabatic match."""
        results: List[OverlapResult] = []
        cand_map = {int(c["state"]): c for c in candidate_states}
        ref_ev = float(ref_point.energy_ev or 0.0)
        ref_f = float(ref_point.oscillator_strength or 0.0)

        for cand in candidate_states:
            st = int(cand["state"])
            if st not in candidate_cubes_map:
                continue
            cand_cubes = candidate_cubes_map[st]
            res = self.score_candidate(ref_cubes, cand_cubes, ref_point.state_number, st)

            # Intensity continuity: preserve optically active resonance, penalise collapse to dark state
            cand_info = cand_map.get(st, {})
            cand_f = float(cand_info.get("oscillator_strength") or 0.0)
            intensity_factor = 1.0
            if ref_f >= 0.001:
                # Downweight if candidate oscillator strength drops by orders of magnitude
                intensity_factor = min(1.0, (cand_f + 1e-4) / (0.4 * ref_f + 1e-4))

            res.energy_score = 1.0
            res.intensity_score = intensity_factor
            # Pure spatial overlap governs adiabatic state tracking (intensity factor advisory only)
            res.composite_score = res.spatial_score
            results.append(res)

        if not results:
            raise RuntimeError(f"No candidate cubes available at step {step_index} ({step_id})")

        # Rank by composite overlap score
        results.sort(key=lambda r: r.composite_score, reverse=True)
        # Density similarity must not override a lost matched hole or electron.
        eligible = [r for r in results if all(
            math.isfinite(value) and value >= MIN_ORBITAL_OVERLAP_SQ
            for value in (r.hole_overlap_sq, r.electron_overlap_sq)
        )]
        if not eligible:
            raise RuntimeError(
                f"State tracking unresolved at step {step_index} ({step_id}): "
                f"no candidate has both matched orbital overlaps squared >= {MIN_ORBITAL_OVERLAP_SQ:.2f}"
            )
        winner = eligible[0]

        # Top candidate tie-breaking: if runner-up is within 15% of leader, evaluate paired NTO & CT continuity
        top_tier = [r for r in eligible if r.composite_score >= 0.85 * winner.composite_score]
        if len(top_tier) > 1:
            # Query ad-hoc CT for key candidates if evaluator hook provided and CT metrics missing
            if ct_evaluator is not None:
                needed = [r.cand_state for r in top_tier if not cand_map.get(r.cand_state, {}).get("ct_metrics", {}).get("q_ct")]
                if needed:
                    adhoc_data = ct_evaluator(needed)
                    for st_id, m in adhoc_data.items():
                        if st_id in cand_map:
                            cand_map[st_id].setdefault("ct_metrics", {}).update(m)

            ref_ct = getattr(ref_point, "ct_metrics", {}) or {}
            ref_q_ct = float(ref_ct.get("q_ct", ref_ct.get("charge_transfer_e", 0.0)))
            ref_frag_q = ref_ct.get("fragment_q", {})

            def tie_break_key(r: OverlapResult):
                cand_info = cand_map.get(r.cand_state, {})
                c_ct = cand_info.get("ct_metrics") or {}
                c_q_ct = float(c_ct.get("q_ct", c_ct.get("charge_transfer_e", 0.0)))
                c_frag_q = c_ct.get("fragment_q", {})
                ct_diff = abs(c_q_ct - ref_q_ct) if (ref_q_ct > 0 and c_q_ct > 0) else 0.0

                # Generic L1 norm of fragment charge shift differences across all defined fragments
                frag_keys = set(ref_frag_q.keys()) & set(c_frag_q.keys())
                frag_diff = (
                    sum(abs(float(c_frag_q[k]) - float(ref_frag_q[k])) for k in frag_keys) / len(frag_keys)
                    if frag_keys else 0.0
                )
                # Primary within top-tier: highest paired NTO overlap, penalty for CT character divergence
                penalty = 0.25 * ct_diff + 0.20 * frag_diff
                return (r.paired_score - penalty, r.composite_score)

            winner = max(top_tier, key=tie_break_key)

        # Match confidence assessment
        cand_dict = next(c for c in candidate_states if int(c["state"]) == winner.cand_state)
        confidence = "CONFIRMED" if winner.paired_score >= self.min_paired_overlap else "PROVISIONAL"
        if any(r is not winner and r.composite_score > 0.8 * winner.composite_score for r in eligible):
            confidence = "AVOIDED_CROSSING_AMBIGUOUS"

        next_point = StateTrackPoint(
            step_index=step_index,
            step_id=str(step_id),
            state_number=winner.cand_state,
            energy_ev=float(cand_dict.get("energy_ev", 0.0)),
            wavelength_nm=float(cand_dict.get("wavelength_nm", 0.0)),
            oscillator_strength=float(cand_dict.get("oscillator_strength", 0.0)),
            ct_metrics=cand_dict.get("ct_metrics", {}),
            hole_overlap_sq=winner.hole_overlap_sq,
            electron_overlap_sq=winner.electron_overlap_sq,
            paired_score=winner.paired_score,
            cdd_overlap_sq=winner.cdd_overlap_sq,
            cluster_rmsd_A=winner.cluster_rmsd_A,
            confidence=confidence,
        )
        return next_point, results

    def filter_energy_window(
        self,
        ref_energy_ev: float,
        states: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Require valid excitation energies and stop when the energy window is empty."""
        if not math.isfinite(ref_energy_ev) or ref_energy_ev <= 0:
            raise ValueError("Reference excitation energy must be finite and positive")
        candidates = []
        for s in states:
            ev = float(s.get("energy_ev") or 0.0)
            if math.isfinite(ev) and ev > 0:
                if abs(ev - ref_energy_ev) <= self.energy_window_ev:
                    candidates.append(s)
        if not candidates:
            raise RuntimeError(
                f"State tracking stopped: no candidates within +/-{self.energy_window_ev:g} eV "
                f"of {ref_energy_ev:g} eV"
            )
        return candidates
