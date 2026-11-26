from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from cats.multiwfn.iroot_scoring import (
    IRootCandidate,
    compute_state_spatial_overlaps,
    score_iroot_candidates,
    select_iroot_by_score,
)
from cats.orca.tddft.ct import (
    load_last_geometry_state,
    pick_iroot,
    record_geometry_choice,
    record_scan_choice,
    load_scan_log,
)


@pytest.fixture
def sample_spectrum_df():
    return pd.DataFrame([
        {"state": 1, "nm": 500.0, "f": 0.05, "CT(e)": 0.20},
        {"state": 2, "nm": 510.0, "f": 0.12, "CT(e)": 0.85},
        {"state": 3, "nm": 520.0, "f": 0.10, "CT(e)": 0.80},
    ])


def test_score_iroot_candidates_with_spatial_overlap(sample_spectrum_df):
    prev_choice = IRootCandidate(
        state=2,
        wavelength_nm=512.0,
        intensity=0.11,
        energy_ev=2.42,
        ct_e=0.82,
        ct_pct=None,
        ct_net=0.82,
    )

    spatial_overlaps = {
        1: 0.10,
        2: 0.95,
        3: 0.40,
    }

    scored = score_iroot_candidates(
        sample_spectrum_df,
        target_wavelength_nm=510.0,
        previous_choice=prev_choice,
        state_spatial_overlaps=spatial_overlaps,
    )

    assert len(scored) == 3
    # State 2 should have spatial_overlap = 0.95 and top score
    state2 = next(c for c in scored if c.state == 2)
    assert state2.spatial_overlap == 0.95
    assert state2.score_components["spatial_overlap"] == 0.95
    assert scored[0].state == 2


def test_select_iroot_by_score_filters_spatial_overlap(sample_spectrum_df):
    spatial_overlaps = {
        1: 0.15,
        2: 0.88,
        3: 0.35,
    }

    best, scored = select_iroot_by_score(
        sample_spectrum_df,
        target_wavelength_nm=510.0,
        state_spatial_overlaps=spatial_overlaps,
        min_spatial_overlap=0.50,
    )

    assert best is not None
    assert best.state == 2
    assert best.spatial_overlap == 0.88

    # Cutoff higher than max available
    none_best, _ = select_iroot_by_score(
        sample_spectrum_df,
        target_wavelength_nm=510.0,
        state_spatial_overlaps=spatial_overlaps,
        min_spatial_overlap=0.99,
    )
    assert none_best is None


def test_pick_iroot_passes_spatial_overlap(sample_spectrum_df):
    spatial_overlaps = {
        1: 0.10,
        2: 0.92,
        3: 0.30,
    }

    best, scored = pick_iroot(
        sample_spectrum_df,
        target_wavelength_nm=510.0,
        score_weights={"intensity": 0.3, "ct": 0.3, "wavelength": 0.2, "similarity": 0.2},
        wavelength_sigma_nm=20.0,
        similarity_sigma_nm=15.0,
        similarity_sigma_ct=0.05,
        state_spatial_overlaps=spatial_overlaps,
        require_spatial_overlap=True,
    )

    assert best is not None
    assert best.state == 2
    assert best.spatial_overlap == 0.92


def test_scan_and_geometry_log_records_spatial_overlap(tmp_path: Path):
    scan_log = tmp_path / "scan.csv"
    geom_log = tmp_path / "geometry.csv"

    choice = IRootCandidate(
        state=2,
        wavelength_nm=510.0,
        intensity=0.15,
        energy_ev=2.43,
        ct_e=0.85,
        ct_pct=None,
        ct_net=0.85,
        score=0.91,
        similarity=0.95,
        spatial_overlap=0.945,
    )

    record_scan_choice(
        scan_log,
        iteration=1,
        scan_dir=tmp_path / "iter1",
        bond_pair=(0, 1),
        bond_length=1.45,
        choice=choice,
        target_wavelength_nm=510.0,
    )

    scan_data = load_scan_log(scan_log)
    assert "1" in scan_data
    assert float(scan_data["1"]["spatial_overlap"]) == pytest.approx(0.945)

    record_geometry_choice(
        geom_log,
        iteration=1,
        geometry_step=1,
        scan_dir=tmp_path / "iter1",
        bond_pair=(0, 1),
        bond_length=1.45,
        choice=choice,
        target_wavelength_nm=510.0,
    )

    step, recovered_choice, scan_dir = load_last_geometry_state(geom_log, iteration=1)
    assert step == 1
    assert recovered_choice is not None
    assert recovered_choice.state == 2
    assert recovered_choice.spatial_overlap == pytest.approx(0.945)
