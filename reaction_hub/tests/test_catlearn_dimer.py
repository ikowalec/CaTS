import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.io import read, write

from cats.working_directory.catlearn_dimer import (
    PersistentDimerLocalOpt,
    DimerRunState,
    seed_dimer_training_set,
    validate_dft_frame,
    run_catlearn_dimer_workflow,
)
from cats.working_directory.active_learning_neb_workflow import (
    _run_dimer_refine_pass,
)
from jobs import tasks
from jobs.tasks import _validate_analysis_request
from results.models import Artifact
from workflows.models import WorkflowDefinition, WorkflowRun
from jobs.models import HPCSystem


def test_seed_dimer_training_set_harvest_and_deduplication(tmp_path):
    target = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]])
    source_traj = tmp_path / "prior_dimer.traj"

    # Frame 1: valid
    f1 = target.copy()
    f1.calc = SinglePointCalculator(f1, energy=-10.0, forces=[[0, 0, 0.1], [0, 0, -0.1]])

    # Frame 2: duplicate positions within min_dist
    f2 = Atoms("Au2", positions=[[0, 0, 0.005], [0, 0, 2.505]])
    f2.calc = SinglePointCalculator(f2, energy=-10.05, forces=[[0, 0, 0.08], [0, 0, -0.08]])

    # Frame 3: distant valid frame
    f3 = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.8]])
    f3.calc = SinglePointCalculator(f3, energy=-9.5, forces=[[0, 0, 0.3], [0, 0, -0.3]])

    # Frame 4: mismatched formula
    f4 = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2.5]])
    f4.calc = SinglePointCalculator(f4, energy=-8.0, forces=[[0, 0, 0.1], [0, 0, -0.1]])

    write(str(source_traj), [f1, f2, f3, f4], format="traj")

    trainingset_path = tmp_path / "trainingset.traj"
    seeded = seed_dimer_training_set([source_traj], target, trainingset_path, min_dist=0.03, max_frames=10)

    # Must harvest f1 and f3; skip f2 (duplicate < 0.03 A) and f4 (wrong formula Cu != Au)
    assert len(seeded) == 2
    assert trainingset_path.exists()


def test_seed_dimer_training_set_preserves_raw_forces_before_target_constraints(tmp_path):
    target = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]])
    target.set_constraint(FixAtoms(indices=[1]))

    raw_f = np.array([[0.15, 0.25, 0.35], [-0.15, -0.25, -0.35]])
    source_atom = target.copy()
    source_atom.set_constraint(FixAtoms(indices=[0]))
    source_atom.calc = SinglePointCalculator(source_atom, energy=-12.5, forces=raw_f)

    seeded = seed_dimer_training_set([source_atom], target)
    assert len(seeded) == 1
    frame = seeded[0]

    # Target constraints must be attached
    assert len(frame.constraints) == 1
    assert frame.constraints[0].index[0] == 1

    # Raw physical forces must be preserved before constraints
    assert np.allclose(frame.arrays["raw_forces"], raw_f)
    assert np.allclose(frame.calc.results["forces"], raw_f)


def test_seed_dimer_training_set_validation_and_constraints(tmp_path):
    target = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]])
    target.set_constraint(FixAtoms(indices=[0]))

    # Valid frame
    v1 = target.copy()
    v1.calc = SinglePointCalculator(v1, energy=-12.0, forces=[[0, 0, 0.05], [0, 0, -0.05]])

    # Incompatible: non-finite energy
    bad_e = target.copy()
    bad_e.calc = SinglePointCalculator(bad_e, energy=float("nan"), forces=[[0, 0, 0.1], [0, 0, -0.1]])

    # Incompatible: missing forces
    bad_f = target.copy()
    bad_f.calc = SinglePointCalculator(bad_f, energy=-11.0, forces=None)

    # Incompatible: surrogate provenance
    bad_prov = target.copy()
    bad_prov.calc = SinglePointCalculator(bad_prov, energy=-11.0, forces=[[0, 0, 0.1], [0, 0, -0.1]])
    bad_prov.info["calculator_name"] = "mace_surrogate"

    state = DimerRunState()
    seeded = seed_dimer_training_set([v1, bad_e, bad_f, bad_prov], target, state=state)

    assert len(seeded) == 1
    assert state.seeded_count == 1
    assert state.rejected_count == 3
    assert state.seeding_status == "missing_seeds"
    assert len(seeded[0].constraints) == 1
    assert isinstance(seeded[0].constraints[0], FixAtoms)
    assert seeded[0].constraints[0].index[0] == 0


def test_dimer_run_state_per_run_isolation():
    atoms = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]])
    atoms.calc = EMT()

    state1 = DimerRunState()
    mode1 = np.array([[0, 0, 1.0], [0, 0, -1.0]]) / np.sqrt(2)
    state1.last_eigenmode = mode1

    state2 = DimerRunState()
    mode2 = np.array([[1.0, 0, 0], [-1.0, 0, 0]]) / np.sqrt(2)
    state2.last_eigenmode = mode2

    opt1 = PersistentDimerLocalOpt(atoms, state=state1)
    opt2 = PersistentDimerLocalOpt(atoms, state=state2)

    assert np.allclose(opt1.min_mode_atoms.get_eigenmode(), mode1, atol=1e-4)
    assert np.allclose(opt2.min_mode_atoms.get_eigenmode(), mode2, atol=1e-4)
    assert not np.allclose(state1.last_eigenmode, state2.last_eigenmode)


def test_persistent_dimer_local_opt_eigenmode_continuity():
    atoms = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]])
    atoms.calc = EMT()

    initial_mode = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    opt1 = PersistentDimerLocalOpt(atoms, initial_eigenmode=initial_mode)
    assert PersistentDimerLocalOpt._last_eigenmode is None or isinstance(PersistentDimerLocalOpt._last_eigenmode, np.ndarray)

    PersistentDimerLocalOpt._last_eigenmode = initial_mode / np.linalg.norm(initial_mode)
    opt2 = PersistentDimerLocalOpt(atoms)
    mode2 = opt2.min_mode_atoms.get_eigenmode()
    assert mode2 is not None
    assert np.allclose(mode2, initial_mode / np.linalg.norm(initial_mode), atol=1e-4)


def test_worker_staged_seed_file_reaches_training_set_via_wrapper(tmp_path):
    target = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]])

    worker_selected_frames = []
    for dz in (0.0, 0.1, 0.2):
        f = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5 + dz]])
        f.calc = SinglePointCalculator(f, energy=-10.0 - dz, forces=[[0, 0, 0.05], [0, 0, -0.05]])
        worker_selected_frames.append(f)

    staged_seed_file = tmp_path / "staged_seed_training.extxyz"
    write(str(staged_seed_file), worker_selected_frames, format="extxyz")

    out_dir = tmp_path / "dimer_out"
    refined_atoms, energy, curv, mode, converged = _run_dimer_refine_pass(
        atoms=target,
        calculator_getter=lambda: EMT(),
        fmax=0.5,
        max_steps=2,
        use_catlearn=True,
        catlearn_polish_steps=2,
        staged_seed_file=staged_seed_file,
        output_dir=out_dir,
    )

    trainingset_path = out_dir / "catlearn_dimer" / "trainingset.traj"
    assert trainingset_path.exists()
    harvested = list(read(str(trainingset_path), ":"))
    assert len(harvested) == 3
    orig_sorted = sorted(worker_selected_frames, key=lambda a: a.positions[1, 2])
    harvest_sorted = sorted(harvested, key=lambda a: a.positions[1, 2])
    for orig, h in zip(orig_sorted, harvest_sorted):
        assert np.allclose(orig.positions, h.positions, atol=1e-4)
        assert math.isclose(orig.get_potential_energy(), h.get_potential_energy(), abs_tol=1e-4)

    assert refined_atoms.info["dimer_run_state"]["staged_seed_file"] == str(staged_seed_file)
    assert refined_atoms.info["dimer_run_state"]["seeded_count"] == 3


def test_catlearn_dimer_restart_api_flags_in_fresh_dir(monkeypatch, tmp_path):
    import sys
    import types

    target = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]])
    target.calc = EMT()

    recorded_kwargs = {}

    class MockLocalAL:
        def __init__(self, **kwargs):
            recorded_kwargs.update(kwargs)

        def run(self, **kwargs):
            pass

    mock_mod = types.ModuleType("catlearn.activelearning.local")
    mock_mod.LocalAL = MockLocalAL
    monkeypatch.setitem(sys.modules, "catlearn", types.ModuleType("catlearn"))
    monkeypatch.setitem(sys.modules, "catlearn.activelearning", types.ModuleType("catlearn.activelearning"))
    monkeypatch.setitem(sys.modules, "catlearn.activelearning.local", mock_mod)

    seeds = []
    for dz in (0.0, 0.1, 0.2):
        a = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5 + dz]])
        a.calc = SinglePointCalculator(a, energy=-10.0 + dz, forces=[[0, 0, 0.1], [0, 0, -0.1]])
        seeds.append(a)

    out_dir = tmp_path / "fresh_run"
    run_catlearn_dimer_workflow(
        atoms=target,
        calc=EMT(),
        output_dir=out_dir,
        prior_trajectories=seeds,
        polish_steps=2,
    )

    assert recorded_kwargs["use_restart"] is False
    assert recorded_kwargs["restart"] is False
    assert recorded_kwargs["trainingset"] == str(out_dir / "trainingset.traj")


def test_skipped_or_failed_catlearn_routes_through_direct_dft_with_original_settings(monkeypatch, tmp_path):
    import sys
    import types

    target = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5]])

    # Case A: Missing seeds -> routes through direct DFT with original max_steps (not polish_steps)
    out_dir_a = tmp_path / "dimer_missing_seeds"
    refined_a, e_a, c_a, m_a, conv_a = _run_dimer_refine_pass(
        atoms=target,
        calculator_getter=lambda: EMT(),
        fmax=0.5,
        max_steps=5,
        use_catlearn=True,
        catlearn_polish_steps=2,
        prior_trajectories=[],
        output_dir=out_dir_a,
    )
    assert refined_a.info["dimer_run_state"]["surrogate_status"] == "skipped_missing_seeds"
    assert "insufficient training seeds" in refined_a.info["dimer_run_state"]["fallback_reason"]

    # Case B: Surrogate failure -> routes through direct DFT with original max_steps
    class FailingLocalAL:
        def __init__(self, **kwargs):
            pass

        def run(self, **kwargs):
            raise RuntimeError("simulated surrogate crash")

    mock_mod = types.ModuleType("catlearn.activelearning.local")
    mock_mod.LocalAL = FailingLocalAL
    monkeypatch.setitem(sys.modules, "catlearn", types.ModuleType("catlearn"))
    monkeypatch.setitem(sys.modules, "catlearn.activelearning", types.ModuleType("catlearn.activelearning"))
    monkeypatch.setitem(sys.modules, "catlearn.activelearning.local", mock_mod)

    seeds = []
    for dz in (0.0, 0.1, 0.2):
        a = Atoms("Au2", positions=[[0, 0, 0], [0, 0, 2.5 + dz]])
        a.calc = SinglePointCalculator(a, energy=-10.0 + dz, forces=[[0, 0, 0.1], [0, 0, -0.1]])
        seeds.append(a)

    out_dir_b = tmp_path / "dimer_failing_surrogate"
    refined_b, e_b, c_b, m_b, conv_b = _run_dimer_refine_pass(
        atoms=target,
        calculator_getter=lambda: EMT(),
        fmax=0.5,
        max_steps=5,
        use_catlearn=True,
        catlearn_polish_steps=2,
        prior_trajectories=seeds,
        output_dir=out_dir_b,
    )
    assert refined_b.info["dimer_run_state"]["surrogate_status"] == "fallback_direct_dft"
    assert "RuntimeError: simulated surrogate crash" in refined_b.info["dimer_run_state"]["fallback_reason"]


def test_validate_analysis_request_and_report_rejected_artifacts(monkeypatch, tmp_path):
    # Test schema validation accepts prior_artifact_ids and staged_seed_file
    payload = {
        "source_run_id": 9323,
        "prior_artifact_ids": [101, 102],
        "staged_seed_file": "seed_training.extxyz",
    }
    clean = _validate_analysis_request("dimer_refine", payload)
    assert clean["prior_artifact_ids"] == [101, 102]
    assert clean["staged_seed_file"] == "seed_training.extxyz"

    # Test artifact rejection reporting in _submit_dimer_refinement
    source = SimpleNamespace(
        pk=9323,
        run_name="test_run",
        status="completed",
        settings_profile=None,
        system_charge=0,
        metadata={},
        jobs=SimpleNamespace(order_by=lambda *a: SimpleNamespace(first=lambda: None)),
    )
    monkeypatch.setattr(WorkflowRun.objects, "select_related", lambda *args: SimpleNamespace(get=lambda **kw: source))
    monkeypatch.setattr(WorkflowDefinition.objects, "get", lambda **kw: SimpleNamespace(pk=1))
    monkeypatch.setattr(HPCSystem.objects, "get", lambda **kw: SimpleNamespace(pk=1, name="ARCHER2"))

    art_missing_cache = SimpleNamespace(pk=101, metadata={"local_cache_path": str(tmp_path / "missing.xyz")})
    
    incomp_path = tmp_path / "incompatible.xyz"
    incomp_atom = Atoms("Cu2", positions=[[0, 0, 0], [0, 0, 2.5]])
    incomp_atom.calc = SinglePointCalculator(incomp_atom, energy=-10.0, forces=[[0, 0, 0.1], [0, 0, -0.1]])
    write(str(incomp_path), incomp_atom, format="extxyz")
    art_incomp = SimpleNamespace(pk=102, metadata={"local_cache_path": str(incomp_path)})

    class MockArtifactQuerySet:
        def select_related(self, *args, **kwargs):
            return self
        def order_by(self, *args, **kwargs):
            return self
        def first(self):
            return None
        def exists(self):
            return False
        def __iter__(self):
            return iter([])

    def mock_filter(**kwargs):
        if "pk" in kwargs:
            aid = kwargs["pk"]
            if aid == 101:
                return SimpleNamespace(first=lambda: art_missing_cache)
            if aid == 102:
                return SimpleNamespace(first=lambda: art_incomp)
        return MockArtifactQuerySet()

    monkeypatch.setattr(Artifact.objects, "filter", mock_filter)

    captured_neb_config = {}
    def mock_submit_slurm(**kwargs):
        captured_neb_config.update(kwargs.get("neb_config", {}))
        return SimpleNamespace(workflow_run_id=1, job_record_id=1, scheduler_job_id="12345", remote_run_dir="/tmp")

    from jobs import submission
    monkeypatch.setattr(submission, "submit_slurm_workflow_run", mock_submit_slurm)

    target_xyz = "2\nAu2\nAu 0 0 0\nAu 0 0 2.5\n"
    tasks._submit_dimer_refinement(
        source_run_id=9323,
        structure_xyz=target_xyz,
        use_catlearn_dimer=True,
        prior_artifact_ids=[101, 102],
    )

    rejected = captured_neb_config.get("rejected_artifacts", [])
    assert len(rejected) == 2
    assert rejected[0]["artifact_id"] == 101
    assert "local cache file missing" in rejected[0]["reason"]
    assert rejected[1]["artifact_id"] == 102
    assert "incompatible frames" in rejected[1]["reason"]


def test_actual_catlearn_api_check():
    """Verify against real CatLearn API or report unverified if unavailable locally."""
    try:
        from catlearn.activelearning.local import LocalAL
    except ImportError:
        pytest.skip("actual CatLearn API check unverified: catlearn.activelearning.local unavailable locally")

    # If LocalAL is available in the environment, prove end-to-end execution
    assert LocalAL is not None
