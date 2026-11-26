from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from cats.multiwfn.iroot_scoring import IRootCandidate
from cats.multiwfn.state_tracker import OverlapResult, StateTrackPoint
from cats.orca.tddft import active_learning
from cats.orca.tddft.active_learning import ProxyContext, ScanThenEngradProxy


class _FakeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, directory: Path):
        super().__init__()
        self.directory = Path(directory)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        directory = Path(self.directory)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "orca.out").write_text("mock", encoding="utf-8")
        self.results = {
            "energy": 0.0,
            "forces": np.zeros((len(atoms), 3)),
        }


def test_local_tddft_engrad_uses_spatially_tracked_root(monkeypatch, tmp_path: Path):
    calls = []

    def calc_factory(atoms, **kwargs):
        calls.append(dict(kwargs))
        return _FakeCalculator(Path(kwargs["directory"]))

    spectrum = pd.DataFrame(
        [
            {"state": 1, "nm": 500.0, "f": 0.20, "CT(e)": 0.2},
            {"state": 2, "nm": 510.0, "f": 0.05, "CT(e)": 0.2},
        ]
    )
    monkeypatch.setattr(active_learning, "select_gbw_source", lambda source, fallback: Path(source))
    monkeypatch.setattr(active_learning, "copy_initial_gbw", lambda source, target: None)
    monkeypatch.setattr(active_learning, "get_ct_info", lambda *args, **kwargs: spectrum)
    monkeypatch.setattr(active_learning, "extract_state_transition_vectors", lambda path: {})
    monkeypatch.setattr(active_learning, "validate_orca_output", lambda path: True)
    monkeypatch.setattr(active_learning, "_attach_charges_from_orca_output", lambda *args, **kwargs: None)
    monkeypatch.setattr(active_learning, "record_geometry_choice", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        active_learning,
        "generate_state_cubes",
        lambda directory, states, working_dir: {
            state: {"all_pair_paths": [], "cdd_path": None} for state in states
        },
    )

    previous = IRootCandidate(1, 500.0, 0.20, 2.4797, ct_e=0.2)
    proxy = ScanThenEngradProxy(
        cpus=1,
        mem=1000,
        scan_base=tmp_path / "scan",
        opt_dir=tmp_path / "opt",
        gbw_source_dir=tmp_path / "reference",
        previous_choice=previous,
        previous_state=1,
        iteration=1,
        geometry_step_start=1,
        bond_pair=None,
        geometry_log_path=tmp_path / "geometry.csv",
        context=ProxyContext(
            target_wavelength_nm=500.0,
            fragments={},
            initial_score_weights={"wavelength": 1.0},
            followup_score_weights={"similarity": 1.0},
            min_oscillator_strength=0.0,
            wavelength_sigma_nm=20.0,
            similarity_sigma_nm=15.0,
            similarity_sigma_ct=0.4,
            nroots=5,
        ),
        calc_factory=calc_factory,
        reference_dir=tmp_path / "reference",
    )
    winner = StateTrackPoint(1, "scan", 2, 2.4311, 510.0, 0.05, confidence="CONFIRMED")
    overlap = OverlapResult(1, 2, 0.95, 0.94, 0.945, spatial_score=0.945, composite_score=0.945)
    proxy.state_tracker.select_next_state = lambda *args, **kwargs: (winner, [overlap])

    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = proxy
    atoms.get_forces()

    engrad_call = next(call for call in calls if call["task"] == "engrad")
    assert engrad_call["iroot"] == 2
    assert proxy.previous_state == 2
    assert (tmp_path / "scan_1" / "state_tracking.json").is_file()
