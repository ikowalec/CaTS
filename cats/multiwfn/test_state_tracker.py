import pytest

from cats.multiwfn.state_tracker import StateTracker


def test_per_orbital_gate_cannot_be_overridden_by_density_or_ct(monkeypatch):
    from cats.multiwfn.state_tracker import OverlapResult, StateTrackPoint
    tracker = StateTracker(root_state=5)
    scores = {
        3: OverlapResult(5, 3, .861, .066, .779, spatial_score=.702),
        5: OverlapResult(5, 5, .693, .544, .735, spatial_score=.636),
        4: OverlapResult(5, 4, .748, .683, .715, spatial_score=.529),
    }
    monkeypatch.setattr(tracker, 'score_candidate', lambda _r, _c, _rs, cs: scores[cs])
    point = StateTrackPoint(26, 'previous', 5, 1.97, 629.1, .008)
    candidates = [dict(state=s, energy_ev=1.9, wavelength_nm=650, oscillator_strength=.007) for s in scores]
    winner, results = tracker.select_next_state(point, {}, candidates, dict.fromkeys(scores, {}), 27, 'dimer')
    assert winner.state_number == 5
    assert len(results) == 3  # Preserve rejected candidates for provenance.
    scores[5].hole_overlap_sq = .30
    scores[5].electron_overlap_sq = .30
    assert tracker.select_next_state(point, {}, candidates, dict.fromkeys(scores, {}), 27, 'dimer')[0].state_number == 5
    for score in scores.values():
        score.electron_overlap_sq = .299999
    with pytest.raises(RuntimeError, match='no candidate has both matched orbital overlaps squared >= 0.30'):
        tracker.select_next_state(point, {}, candidates, dict.fromkeys(scores, {}), 27, 'dimer')


def test_energy_window_never_uses_wavelength_or_nearest_roots():
    tracker = StateTracker(root_state=1)
    inside = {"state": 7, "energy_ev": 2.1, "wavelength_nm": 590.0}
    outside = {"state": 1, "energy_ev": 3.0, "wavelength_nm": 620.0}
    missing = {"state": 2, "wavelength_nm": 620.0}
    assert tracker.filter_energy_window(2.0, [outside, missing, inside]) == [inside]
    with pytest.raises(RuntimeError, match="no candidates"):
        tracker.filter_energy_window(2.0, [outside, missing])
    for energy in (0.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="Reference excitation energy"):
            tracker.filter_energy_window(energy, [inside])
        with pytest.raises(ValueError, match="energy_window_ev"):
            StateTracker(root_state=1, energy_window_ev=energy)


def test_candidate_ct_recovery_never_overwrites_preceding_geometry(monkeypatch):
    from cats.multiwfn.state_tracker import OverlapResult, StateTrackPoint
    tracker = StateTracker(root_state=2)
    scores = {2: OverlapResult(2,2,.6,.6,.6,spatial_score=.6),
              4: OverlapResult(2,4,.6,.6,.6,spatial_score=.59)}
    monkeypatch.setattr(tracker,'score_candidate',lambda _r,_c,_rs,cs:scores[cs])
    reference = StateTrackPoint(0,'previous',2,2,620,.01)
    candidates = [dict(state=s,energy_ev=2,oscillator_strength=.01) for s in scores]
    tracker.select_next_state(reference,{},candidates,dict.fromkeys(scores,{}),1,'next',
                              ct_evaluator=lambda states:{s:{'q_ct':.2} for s in states})
    assert reference.ct_metrics == {}


def test_tracking_and_visualisation_use_separate_grids(tmp_path, monkeypatch):
    from cats.multiwfn import density_analysis
    from cats.multiwfn.mep_state_tracker import ContinuousMEPTracker

    tracker = ContinuousMEPTracker([tmp_path], output_directory=tmp_path / "tracking")
    assert tracker.grid_quality == 2
    coarse = tmp_path / "coarse.cub"
    coarse.write_text("coarse")
    monkeypatch.setattr(tracker, "_resolve_molden", lambda directory: directory / "orca.molden.input")
    monkeypatch.setattr(tracker, "_resolve_orca_out", lambda directory: directory / "orca.out")

    def generate(*args, **kwargs):
        assert kwargs["grid_quality"] == 3
        assert kwargs["include_nto"] is False
        directory = kwargs["output_directory"]
        directory.mkdir(parents=True)
        (directory / "state_007_cdd.cub").write_text("fine")

    monkeypatch.setattr(density_analysis, "generate_excitation_density_analysis", generate)
    tracker._save_visualisation_cdd(tmp_path, 7, 0, coarse)
    assert (tracker.output_dir / "mep_cdd_image_00.cub").read_text() == "fine"
    assert coarse.read_text() == "coarse"
    tracker.grid_quality = 3
    tracker._save_visualisation_cdd(tmp_path, 7, 1, coarse)
    assert (tracker.output_dir / "mep_cdd_image_01.cub").read_text() == "coarse"
