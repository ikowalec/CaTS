import pytest
from ase import Atoms
from jobs.tasks import _first_three_fragment_config, _pad_mep_images, _validate_analysis_request


def test_validate_mep_state_tracking_payload():
    payload = {
        "source_run_id": 8968,
        "image_run_ids": [8968, 8974, 8969, 8972, 8959, 8970, 8975, 8971, 8973],
        "root_state": 25,
        "energy_window_ev": 0.35,
        "density_workers": 4,
        "nto_max_pairs": 3,
        "nto_target_coverage": 0.8,
        "grid_quality": 3,
        "walltime": "04:00:00",
        "hpc_name": "FALCON",
    }
    clean = _validate_analysis_request("mep_state_tracking", payload)
    assert clean["source_run_id"] == 8968
    assert clean["root_state"] == 25
    assert len(clean["image_run_ids"]) == 9
    assert clean["density_workers"] == 4
    assert clean["hpc_name"] == "FALCON"


def test_validate_mep_state_tracking_invalid():
    with pytest.raises(ValueError):
        _validate_analysis_request("mep_state_tracking", {"source_run_id": -1})

    with pytest.raises(ValueError):
        _validate_analysis_request("mep_state_tracking", {"source_run_id": 8968, "root_state": -5})


def test_padded_mep_uvvis_request_and_geometry():
    clean = _validate_analysis_request(
        "mep_uvvis_snapshots", {"source_run_id": 8763, "padding_per_segment": 2}
    )
    padded = _pad_mep_images(
        [Atoms("H", positions=[[0, 0, 0]]), Atoms("H", positions=[[3, 0, 0]])], 2
    )

    assert clean["nroots"] == 50
    assert [item[0].positions[0, 0] for item in padded] == [0, 1, 2, 3]


def test_mep_uvvis_selected_indices_are_bounded():
    payload = {"source_run_id": 9475, "padding_per_segment": 0, "image_indices": [23, 16, 16]}
    assert _validate_analysis_request("mep_uvvis_snapshots", payload)["image_indices"] == [16, 23]
    for indices in ([], [-1], ["16"], [True]):
        with pytest.raises(ValueError, match="image_indices"):
            _validate_analysis_request("mep_uvvis_snapshots", {**payload, "image_indices": indices})


def test_mep_uvvis_accepts_registered_band_id_only():
    payload = {"source_run_id": 9475, "band_artifact_id": 158401}
    assert _validate_analysis_request("mep_uvvis_snapshots", payload)["band_artifact_id"] == 158401
    for invalid in (0, -1, True, "158401"):
        with pytest.raises(ValueError, match="band_artifact_id"):
            _validate_analysis_request("mep_uvvis_snapshots", {**payload, "band_artifact_id": invalid})


def test_mep_uvvis_wavefunction_requires_database_id():
    payload = {"source_run_id": 9854, "nroots": 14, "gradient_root": 2, "initial_wavefunction_artifact_id": 123}
    assert _validate_analysis_request("mep_uvvis_snapshots", payload)["nroots"] == 14
    for value in [True, 0, -1, "/tmp/orca.gbw"]:
        with pytest.raises(ValueError, match="initial_wavefunction_artifact_id"):
            _validate_analysis_request("mep_uvvis_snapshots", {**payload, "initial_wavefunction_artifact_id": value})


def test_padded_mep_reuses_completed_three_fragment_lineage():
    expected = {"fragments_0based": {"1": [0, 1], "2": [2, 4], "3": [3]}}

    assert _first_three_fragment_config([{}, expected]) == expected
