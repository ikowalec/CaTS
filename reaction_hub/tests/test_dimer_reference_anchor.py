import ast
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.mep.dimer import DimerControl, MinModeAtoms


def dimer_classes():
    path = Path(__file__).resolve().parents[2] / "cats/working_directory/active_learning_neb_workflow.py"
    tree = ast.parse(path.read_text())
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
               and node.name in {"TDDFTDimerProxy", "_ReferenceAnchoredMinModeAtoms"}]
    namespace = dict(Calculator=Calculator, all_changes=all_changes, MinModeAtoms=MinModeAtoms,
                     np=np, Path=Path, json=json)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *classes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["TDDFTDimerProxy"], namespace["_ReferenceAnchoredMinModeAtoms"]


class SequentialTracker(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, directory):
        super().__init__()
        self.reference_dir = directory
        self.reference_point = SimpleNamespace(state_number=5)
        self.reference_cubes = {"anchor": 5}
        self.previous_choice = SimpleNamespace(state=5)
        self.previous_state = 5
        self.history = []
        self.fail = False

    def _load_reference(self):
        pass

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = float(atoms.positions[0, 0])
        state = 4 if x >= .05 else (3 if x else 5)
        self.history.append((x, self.previous_state, state, self.gbw_source_dir))
        self.last_scan_dir = self.reference_dir.parent / f"scan{len(self.history)}"
        self.last_scan_dir.mkdir()
        (self.last_scan_dir / "state_tracking.json").write_text('{}')
        self.reference_dir = self.last_scan_dir
        self.reference_point = SimpleNamespace(state_number=state)
        self.reference_cubes = {"anchor": state}
        self.previous_choice = SimpleNamespace(state=state)
        self.previous_state = state
        if self.fail:
            raise RuntimeError("failed force evaluation")
        self.results = {"energy": -x, "forces": np.ones((len(atoms), 3))}


def test_probes_are_frozen_and_accepted_centres_advance_even_after_cached_trial(tmp_path):
    wrapper_type, centres_type = dimer_classes()
    proxy = SequentialTracker(tmp_path / "seed")
    calculator = wrapper_type()
    calculator.proxy = proxy
    atoms = Atoms("H", positions=[[0., 0., 0.]])
    atoms.calc = calculator
    with DimerControl(logfile=None, eigenmode_logfile=None) as control:
        centres = centres_type(atoms, control=control, eigenmodes=[np.array([[1., 0., 0.]])])
        centres.get_forces(real=True)
        anchor = proxy.reference_dir
        for x in [.0001, -.0001, .05]:
            centres.get_forces(real=True, pos=np.array([[x, 0., 0.]]))
            assert proxy.previous_state == 5 and proxy.reference_dir == anchor
            assert proxy.reference_cubes == {"anchor": 5}
        assert [entry[1] for entry in proxy.history] == [5, 5, 5, 5]
        assert all(entry[3] == anchor for entry in proxy.history[1:])
        # The .05 trial is cached, but accepting it must commit a fresh centre match.
        centres.set_positions([[.05, 0., 0.]])
        centres.get_forces(real=True)
        assert len(proxy.history) == 5 and proxy.history[-1][1] == 5
        assert proxy.previous_state == 4
        next_anchor = proxy.reference_dir
        centres.get_forces(real=True, pos=np.array([[.0501, 0., 0.]]))
        assert proxy.history[-1][1] == 4 and proxy.reference_dir == next_anchor
        audit = json.loads((proxy.last_scan_dir / "state_tracking.json").read_text())
        assert audit["evaluation_kind"] == "probe" and audit["reference_state"] == 4
        centres.set_positions([[.1, 0., 0.]])
        centres.get_forces(real=True)
        audit = json.loads((proxy.last_scan_dir / "state_tracking.json").read_text())
        assert audit["evaluation_kind"] == "accepted_center"
        assert audit["reference_step_id"] == str(next_anchor) and proxy.history[-1][1] == 4
        assert calculator.evaluating_dimer_center is False


def test_failed_centre_does_not_advance_reference(tmp_path):
    wrapper_type, centres_type = dimer_classes()
    calculator = wrapper_type()
    calculator.proxy = SequentialTracker(tmp_path / "seed")
    calculator.proxy.fail = True
    atoms = Atoms("H", positions=[[.1, 0., 0.]])
    atoms.calc = calculator
    with DimerControl(logfile=None, eigenmode_logfile=None) as control:
        centres = centres_type(atoms, control=control, eigenmodes=[np.array([[1., 0., 0.]])])
        with pytest.raises(RuntimeError, match="failed force"):
            centres.get_forces(real=True)
    assert calculator.proxy.previous_state == 5
    assert calculator.proxy.reference_dir == tmp_path / "seed"
    assert calculator.evaluating_dimer_center is False and atoms.calc is calculator


def test_small_initial_probe_is_not_mistaken_for_cached_seed(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wrapper_type, _ = dimer_classes()
    calculator = wrapper_type()
    calculator.proxy = SequentialTracker(Path("dimer_init_scan"))
    calculator.initial_positions = np.array([[10., 0., 0.]])
    calculator.initial_energy = 0.
    calculator.initial_forces = np.zeros((1, 3))
    atoms = Atoms("H", positions=[[10.0001, 0., 0.]])
    atoms.calc = calculator
    atoms.get_forces()
    assert len(calculator.proxy.history) == 1
    assert calculator.proxy.previous_state == 5
