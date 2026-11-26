from __future__ import annotations

import sys
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from ase.io.trajectory import Trajectory

_MODULE_NAME = "active_learning_neb_workflow_under_test"
_SPEC = importlib.util.spec_from_file_location(
    _MODULE_NAME,
    Path(__file__).resolve().parents[2] / "cats" / "working_directory" / "active_learning_neb_workflow.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_run_catlearn_mlneb_pass = _MODULE._run_catlearn_mlneb_pass
_run_dimer_refine_pass = _MODULE._run_dimer_refine_pass
_copy_with_single_point_results = _MODULE._copy_with_single_point_results
_build_dft_factory = _MODULE._build_dft_factory
_validate_dft_geometry = _MODULE._validate_dft_geometry
CalculatorFactoryConfig = _MODULE.CalculatorFactoryConfig
run_active_learning_neb_workflow = _MODULE.run_active_learning_neb_workflow

_ALIGN_SPEC = importlib.util.spec_from_file_location(
    "neb_endpoint_alignment_under_test",
    Path(__file__).resolve().parents[2] / "cats" / "working_directory" / "neb_endpoint_alignment.py",
)
assert _ALIGN_SPEC is not None and _ALIGN_SPEC.loader is not None
_ALIGN_MODULE = importlib.util.module_from_spec(_ALIGN_SPEC)
_ALIGN_SPEC.loader.exec_module(_ALIGN_MODULE)
align_and_reorder_endpoint = _ALIGN_MODULE.align_and_reorder_endpoint


class _EndpointCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(atoms), 3))}


def test_dft_image_evaluation_writes_reusable_training_frame(monkeypatch, tmp_path):
    monkeypatch.setattr(_MODULE, "build_calculator_factory", lambda _config: _EndpointCalculator)

    result = run_active_learning_neb_workflow(
        initial_atoms=Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
        backend="dft",
        evaluation_only=True,
        output_dir=tmp_path,
    )

    evaluated = read(tmp_path / "evaluated.traj")
    assert result["mode"] == "dft_image_evaluation"
    assert evaluated.get_potential_energy() == 0.0
    assert np.allclose(evaluated.get_forces(), 0.0)


def test_dimer_result_keeps_single_point_after_atoms_copy(monkeypatch, tmp_path):
    class Context:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class MinModeAtoms:
        def __init__(self, atoms, **kwargs):
            self.atoms = atoms

        def displace(self, **kwargs):
            pass

        def get_curvature(self):
            return -1.0

        def get_eigenmode(self):
            return np.zeros((len(self.atoms), 3))

    class Translate(Context):
        def run(self, **kwargs):
            return True

    monkeypatch.setattr(_MODULE, "DimerControl", Context)
    monkeypatch.setattr(_MODULE, "MinModeAtoms", MinModeAtoms)
    monkeypatch.setattr(_MODULE, "MinModeTranslate", Translate)

    refined, energy, *_ = _run_dimer_refine_pass(
        atoms=Atoms("H", positions=[[0.0, 0.0, 0.0]]),
        calculator_getter=_EndpointCalculator,
        output_dir=tmp_path,
    )

    assert energy == refined.get_potential_energy() == 0.0


def test_mlneb_result_copy_keeps_energy_and_forces(tmp_path):
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=1.25, forces=[[0.1, 0.2, 0.3]])

    copied = _copy_with_single_point_results(atoms)
    write(tmp_path / "band.extxyz", [copied], format="extxyz")
    restored = read(tmp_path / "band.extxyz")

    assert restored.get_potential_energy() == pytest.approx(1.25)
    assert np.allclose(restored.get_forces(), [[0.1, 0.2, 0.3]])


def test_dft_neb_calculator_converts_ase_directory_string_to_path(monkeypatch, tmp_path):
    seen = {}
    workflow = ModuleType("cats.working_directory.tddft_active_learning_workflow")

    def fake_orca_calculator(atoms, *, directory, **_kwargs):
        seen["directory"] = directory
        return SinglePointCalculator(
            atoms,
            energy=0.0,
            forces=np.zeros((len(atoms), 3)),
        )

    workflow.get_orca_dft_engrad_calculator = fake_orca_calculator
    monkeypatch.setitem(
        sys.modules,
        "cats.working_directory.tddft_active_learning_workflow",
        workflow,
    )
    monkeypatch.chdir(tmp_path)
    calculator = _build_dft_factory(CalculatorFactoryConfig(backend="dft"))()
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], calculator=calculator)

    atoms.get_forces()

    assert isinstance(calculator.directory, str)
    assert seen["directory"] == Path("dft_neb_calc_00001")
    assert (tmp_path / seen["directory"]).is_dir()


def test_dft_neb_retries_missing_gbw_orbitals_without_restart(monkeypatch, tmp_path):
    calls = []
    workflow = ModuleType("cats.working_directory.tddft_active_learning_workflow")

    class _FailingRestartCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def __init__(self, directory):
            super().__init__()
            self.directory = Path(directory)

        def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            directory = Path(self.directory)
            (directory / "orca.out").write_text(
                "No orbitals were found in the gbw file\n", encoding="utf-8"
            )
            (directory / "orca.gbw").write_bytes(b"truncated")
            raise RuntimeError("ORCA failed")

    def fake_orca_calculator(atoms, *, directory, **_kwargs):
        calls.append((directory / "initial.gbw").exists())
        if len(calls) == 1:
            (directory / "initial.gbw").write_bytes(b"bad restart")
            return _FailingRestartCalculator(directory)
        assert not (directory / "initial.gbw").exists()
        assert not (directory / "orca.gbw").exists()
        return SinglePointCalculator(
            atoms,
            energy=-1.0,
            forces=np.zeros((len(atoms), 3)),
        )

    workflow.get_orca_dft_engrad_calculator = fake_orca_calculator
    monkeypatch.setitem(
        sys.modules,
        "cats.working_directory.tddft_active_learning_workflow",
        workflow,
    )
    monkeypatch.chdir(tmp_path)
    calculator = _build_dft_factory(CalculatorFactoryConfig(backend="dft"))()
    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]], calculator=calculator)

    assert np.allclose(atoms.get_forces(), 0.0)
    assert calls == [False, False]


def test_dft_neb_rejects_collapsed_geometry_before_orca():
    atoms = Atoms("Au2", positions=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])

    with pytest.raises(ValueError, match="implausible contact"):
        _validate_dft_geometry(atoms, 0.55)


def test_catlearn_mlneb_uses_total_image_count_and_writes_mep(monkeypatch, tmp_path):
    seen = {}

    class _FakeMLNEB:
        def __init__(self, *, start, end, n_images, **kwargs):
            seen["n_images"] = n_images
            first = read(start)
            last = read(end)
            self.images = []
            for index in range(n_images):
                fraction = index / (n_images - 1)
                image = first.copy()
                image.positions = (1.0 - fraction) * first.positions + fraction * last.positions
                image.calc = SinglePointCalculator(
                    image,
                    energy=float(index),
                    forces=np.zeros((len(image), 3)),
                )
                self.images.append(image)
            self.list_targets = [0.0, 1.0, 2.0]

        def run(self, **kwargs):
            seen["run_kwargs"] = kwargs
            write("converged.traj", self.images)
            return True

    catlearn = ModuleType("catlearn")
    activelearning = ModuleType("catlearn.activelearning")
    mlneb = ModuleType("catlearn.activelearning.mlneb")
    mlneb.MLNEB = _FakeMLNEB
    monkeypatch.setitem(sys.modules, "catlearn", catlearn)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning", activelearning)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning.mlneb", mlneb)

    start = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    end = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4]])
    images, energies, converged, evaluations = _run_catlearn_mlneb_pass(
        start=start,
        end=end,
        calculator_getter=_EndpointCalculator,
        n_images=7,
        fmax=0.05,
        max_steps=20,
        spring_constant=0.1,
        catlearn_unc_convergence=0.05,
        catlearn_max_unc=0.3,
        catlearn_ml_steps=50,
        catlearn_dtrust=2.0,
        workflow_log_path=None,
        output_dir=tmp_path,
    )

    assert seen["n_images"] == 7
    assert len(images) == 7
    assert energies == list(map(float, range(7)))
    assert converged is True
    assert evaluations == 1
    assert seen["run_kwargs"]["max_unc"] == 0.3
    assert seen["run_kwargs"]["dtrust"] == 2.0
    timing = (tmp_path / "catlearn_mlneb" / "ml_timing.jsonl").read_text(encoding="utf-8")
    assert '"ml_steps": 50' in timing


def test_catlearn_mlneb_seeds_evaluated_initial_band(monkeypatch, tmp_path):
    seen = {}

    class _SeededMLNEB:
        def __init__(self, *, start, end, n_images, **kwargs):
            seen["evaluated"] = list(read("evaluated.traj", ":"))
            seen["restart"] = kwargs["restart"]
            seen["previous"] = kwargs["prev_calculations"]
            seen["predicted"] = list(read("predicted.traj", ":"))
            first = read(start)
            last = read(end)
            self.images = []
            for index in range(n_images):
                image = first.copy()
                image.positions = (
                    (1.0 - index / (n_images - 1)) * first.positions
                    + index / (n_images - 1) * last.positions
                )
                image.calc = SinglePointCalculator(
                    image, energy=float(index), forces=np.zeros((len(image), 3))
                )
                self.images.append(image)
            self.list_targets = list(range(len(seen["evaluated"])))

        def run(self, **kwargs):
            write("converged.traj", self.images)
            return True

    catlearn = ModuleType("catlearn")
    activelearning = ModuleType("catlearn.activelearning")
    mlneb = ModuleType("catlearn.activelearning.mlneb")
    mlneb.MLNEB = _SeededMLNEB
    monkeypatch.setitem(sys.modules, "catlearn", catlearn)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning", activelearning)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning.mlneb", mlneb)

    start = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    end = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4]])
    seeds = []
    for index, distance in enumerate((0.85, 1.0, 1.2), start=1):
        image = Atoms("H2", positions=[[0, 0, 0], [0, 0, distance]])
        image.calc = SinglePointCalculator(
            image, energy=float(index), forces=np.full((2, 3), float(index))
        )
        seeds.append(image)

    _run_catlearn_mlneb_pass(
        start=start,
        end=end,
        calculator_getter=_EndpointCalculator,
        n_images=5,
        fmax=0.05,
        max_steps=10,
        spring_constant=0.1,
        catlearn_unc_convergence=0.05,
        catlearn_max_unc=0.3,
        catlearn_ml_steps=50,
        catlearn_dtrust=2.0,
        workflow_log_path=None,
        output_dir=tmp_path,
        initial_band_atoms=[start, *seeds, end],
        initial_training_atoms=seeds,
    )

    evaluated = seen["evaluated"]
    assert seen["restart"] is False
    assert len(seen["previous"]) == 5
    assert len(seen["predicted"]) == 5
    assert len(evaluated) == 5
    assert np.allclose(evaluated[0].positions, start.positions)
    assert np.allclose(evaluated[1].positions, end.positions)
    assert sorted(frame.get_potential_energy() for frame in evaluated[2:]) == [1.0, 2.0, 3.0]
    assert len(read(str(tmp_path / "catlearn_mlneb" / "all_evaluated.traj"), ":")) == 5


def test_endpoint_alignment_undoes_rotation_and_like_atom_permutation():
    initial = Atoms(
        "Au5CO2",
        positions=[
            [0.0, 0.0, 0.0],
            [1.1, 0.2, 0.0],
            [0.1, 1.4, 0.3],
            [0.2, 0.3, 1.7],
            [1.6, 1.1, 0.8],
            [0.5, 0.5, 2.5],
            [0.4, 0.6, 3.6],
            [0.7, 0.4, 3.5],
        ],
    )
    angle = np.deg2rad(73.0)
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    permutation = [3, 0, 4, 1, 2, 5, 7, 6]
    final = initial[permutation]
    final.positions = np.asarray(initial.positions)[permutation] @ rotation + [4.0, -3.0, 2.0]

    aligned, metrics = align_and_reorder_endpoint(initial, final)

    assert aligned.get_chemical_symbols() == initial.get_chemical_symbols()
    assert np.allclose(aligned.positions, initial.positions, atol=1.0e-10)
    assert metrics["max_shift_all_A"] < 1.0e-10


def test_catlearn_mlneb_dynamic_cycle_pruning_and_restart(monkeypatch, tmp_path):
    cycle_calls = []
    restart_calls = []
    purged_mlcalc = []

    class _MockCyclingMLNEB:
        def __init__(self, *, start, end, n_images, neb_interpolation, **kwargs):
            restart_calls.append(kwargs["restart"])
            self.n_images = n_images
            self.interpolation = neb_interpolation
            first = read(start)
            last = read(end)
            self.images = [first.copy() for _ in range(n_images)]
            for idx, img in enumerate(self.images):
                img.calc = SinglePointCalculator(img, energy=float(idx), forces=np.zeros((len(img), 3)))
            self.list_targets = list(range(n_images + 2))

        def run(self, *, steps, **kwargs):
            cycle_calls.append(steps)
            # Create a mock mlcalc.pkl to verify it gets purged on un-converged cycle
            Path("mlcalc.pkl").write_bytes(b"mock_gpr_cache")
            # Write predicted band of dynamic length
            write("predicted.traj", self.images)

            if len(cycle_calls) == 1:
                # First cycle: not converged, produces evaluated structures (e.g. 5 band + 10 guesses = 15)
                eval_frames = [self.images[0].copy() for _ in range(self.n_images + 10)]
                for f in eval_frames:
                    f.calc = SinglePointCalculator(f, energy=0.5, forces=np.zeros((len(f), 3)))
                write("evaluated.traj", eval_frames)
                Path("ml_summary.txt").write_text("step\n", encoding="utf-8")
                Path("ml_time.txt").write_text("step\n", encoding="utf-8")
                return False

            # Second cycle: converges
            write("converged.traj", self.images)
            return True

    catlearn = ModuleType("catlearn")
    activelearning = ModuleType("catlearn.activelearning")
    mlneb = ModuleType("catlearn.activelearning.mlneb")
    mlneb.MLNEB = _MockCyclingMLNEB
    monkeypatch.setitem(sys.modules, "catlearn", catlearn)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning", activelearning)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning.mlneb", mlneb)

    start = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    end = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4]])

    images, energies, converged, evaluations = _run_catlearn_mlneb_pass(
        start=start,
        end=end,
        calculator_getter=_EndpointCalculator,
        n_images=5,
        fmax=0.05,
        max_steps=25,
        spring_constant=0.1,
        catlearn_unc_convergence=0.05,
        catlearn_max_unc=0.3,
        catlearn_ml_steps=50,
        catlearn_dtrust=2.0,
        workflow_log_path=None,
        output_dir=tmp_path,
        catlearn_guesses_per_cycle=10,
    )

    # 1. Verify dynamic cycle budgeting: 10 guesses on cycle 1, 10 guesses on cycle 2
    assert cycle_calls == [10, 10]
    assert restart_calls == [False, True]
    # 2. Dynamic length preserved
    assert len(images) == 5
    assert converged is True
    # 3. Trajectory and archive state
    assert (tmp_path / "catlearn_mlneb" / "all_evaluated.traj").exists()
    assert (tmp_path / "catlearn_mlneb" / "converged.traj").exists()
    assert len(read(str(tmp_path / "catlearn_mlneb" / "all_evaluated.traj"), ":")) == 15


def test_catlearn_mlneb_ingests_old_calculation_at_new_ruleset(monkeypatch, tmp_path):
    old_dir = tmp_path / "old_job_15227337"
    old_stage = old_dir / "catlearn_mlneb"
    old_stage.mkdir(parents=True)

    start = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    end = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4]])

    # Old calculation had 35 distinct evaluated structures.
    old_evals = []
    for i in range(35):
        img = start.copy()
        img.positions[1, 2] += i * 0.01
        img.calc = SinglePointCalculator(img, energy=float(i), forces=np.zeros((len(img), 3)))
        old_evals.append(img)
    write(old_stage / "evaluated.traj", old_evals)
    write(old_stage / "all_evaluated.traj", old_evals[:12])
    stale = start.copy()
    stale.positions += 10.0
    stale.calc = SinglePointCalculator(stale, energy=0.0, forces=np.zeros((len(stale), 3)))
    write(old_stage / "initial.traj", stale)
    write(old_stage / "final.traj", stale)

    # Old calculation had predicted bands of 5 images
    old_preds = []
    for i in range(15):
        img = start.copy()
        img.positions += i * 0.01
        old_preds.append(img)
    write(old_stage / "predicted.traj", old_preds)

    # Old bloated GPR cache
    (old_stage / "mlcalc.pkl").write_bytes(b"bloated_1.6gb_cache")

    seen_on_init = {}

    class _MockResumeMLNEB:
        def __init__(self, *, start, end, n_images, neb_interpolation, **kwargs):
            self.n_images = n_images
            self.interpolation = neb_interpolation
            seen_on_init["evaluated_count"] = len(read("evaluated.traj", ":")) if Path("evaluated.traj").exists() else 0
            seen_on_init["mlcalc_exists"] = Path("mlcalc.pkl").exists()
            seen_on_init["band_positions"] = [img.positions.copy() for img in neb_interpolation] if isinstance(neb_interpolation, list) else None
            seen_on_init["start_positions"] = read(start).positions.copy()
            seen_on_init["end_positions"] = read(end).positions.copy()

            first = read(start)
            self.images = [first.copy() for _ in range(n_images)]
            for idx, img in enumerate(self.images):
                img.calc = SinglePointCalculator(img, energy=float(idx), forces=np.zeros((len(img), 3)))
            self.list_targets = list(range(n_images + 2))

        def run(self, *, steps, **kwargs):
            seen_on_init["step_limit"] = steps
            write("converged.traj", self.images)
            return True

    catlearn = ModuleType("catlearn")
    activelearning = ModuleType("catlearn.activelearning")
    mlneb = ModuleType("catlearn.activelearning.mlneb")
    mlneb.MLNEB = _MockResumeMLNEB
    monkeypatch.setitem(sys.modules, "catlearn", catlearn)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning", activelearning)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning.mlneb", mlneb)

    new_run_dir = tmp_path / "new_run"
    images, energies, converged, evaluations = _run_catlearn_mlneb_pass(
        start=start,
        end=end,
        calculator_getter=_EndpointCalculator,
        n_images=5,
        fmax=0.05,
        max_steps=25,
        spring_constant=0.1,
        catlearn_unc_convergence=0.05,
        catlearn_max_unc=0.3,
        catlearn_ml_steps=50,
        catlearn_dtrust=2.0,
        workflow_log_path=None,
        output_dir=new_run_dir,
        catlearn_guesses_per_cycle=10,
        restart_from=old_dir,
    )

    # Reserve space for ten new evaluations within the 30-frame training cap.
    assert seen_on_init["evaluated_count"] == 20
    assert np.allclose(seen_on_init["start_positions"], start.positions)
    assert np.allclose(seen_on_init["end_positions"], end.positions)
    # 2. Bloated cache was purged before CatLearn initialized
    assert seen_on_init["mlcalc_exists"] is False
    # 3. Dynamic cycle budget was applied (10 guesses)
    assert seen_on_init["step_limit"] == 10
    # 4. MEP band was warm-started from old predicted.traj trailing 5 images
    assert seen_on_init["band_positions"] is not None
    assert np.allclose(seen_on_init["band_positions"][-1], old_preds[-1].positions)
    # 5. Complete archive preserved all 35 old evaluations
    all_evals = list(read(str(new_run_dir / "catlearn_mlneb" / "all_evaluated.traj"), ":"))
    assert len(all_evals) == 35
    assert len(images) == 5
    assert converged is True


def test_catlearn_mlneb_archives_every_cycle_with_strict_training_cap(monkeypatch, tmp_path):
    pool_sizes = []
    endpoint_positions = []
    peak_energies = []

    class _MockMLNEB:
        def __init__(self, *, start, end, n_images, neb_interpolation, **kwargs):
            self.images = [read(start).copy() for _ in range(n_images)]
            for image in self.images:
                image.calc = SinglePointCalculator(image, energy=0.0, forces=np.zeros((len(image), 3)))
            if not Path("evaluated.traj").exists():
                write("evaluated.traj", [read(start), read(end)])
            frames = list(read("evaluated.traj", ":"))
            pool_sizes.append(len(frames))
            endpoint_positions.append((frames[0].positions.copy(), frames[1].positions.copy()))
            peak_energies.append(max(frame.get_potential_energy() for frame in frames))
            self.list_targets = list(range(len(frames)))

        def run(self, *, steps, **kwargs):
            with Trajectory("evaluated.traj", "a") as trajectory:
                for step in range(steps):
                    image = self.images[0].copy()
                    image.positions[1, 2] = 0.8 + (len(pool_sizes) * 10 + step) * 0.01
                    image.calc = SinglePointCalculator(
                        image, energy=float(len(pool_sizes) * 10 + step), forces=np.zeros((len(image), 3))
                    )
                    trajectory.write(image)
            write("predicted.traj", self.images)
            return False

    catlearn = ModuleType("catlearn")
    activelearning = ModuleType("catlearn.activelearning")
    mlneb = ModuleType("catlearn.activelearning.mlneb")
    mlneb.MLNEB = _MockMLNEB
    monkeypatch.setitem(sys.modules, "catlearn", catlearn)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning", activelearning)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning.mlneb", mlneb)

    start = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    end = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4]])
    _, _, converged, evaluations = _run_catlearn_mlneb_pass(
        start=start, end=end, calculator_getter=_EndpointCalculator, n_images=5,
        fmax=0.05, max_steps=40, spring_constant=0.1, catlearn_unc_convergence=0.05,
        catlearn_max_unc=0.3, catlearn_ml_steps=50, catlearn_dtrust=2.0,
        workflow_log_path=None, output_dir=tmp_path, catlearn_guesses_per_cycle=10,
    )

    stage = tmp_path / "catlearn_mlneb"
    assert pool_sizes == [2, 12, 20, 20]
    assert len(read(str(stage / "evaluated.traj"), ":")) == 30
    assert len(read(str(stage / "all_evaluated.traj"), ":")) == 42
    assert evaluations == 40
    assert converged is False
    assert peak_energies[-1] == 39.0
    assert all(np.allclose(first, start.positions) and np.allclose(last, end.positions)
               for first, last in endpoint_positions)


def test_catlearn_fallback_reuses_last_predicted_path(monkeypatch, tmp_path):
    interpolations = []

    class _FallbackMLNEB:
        def __init__(self, *, start, end, n_images, interpolation, **kwargs):
            interpolations.append(interpolation)
            self.images = [read(start).copy() for _ in range(n_images)]
            for image in self.images:
                image.calc = SinglePointCalculator(image, energy=0.0, forces=np.zeros((len(image), 3)))
            if not Path("evaluated_structures.traj").exists():
                write("evaluated_structures.traj", [read(start), read(end)])
            self.list_targets = [0, 1]

        def run(self, *, trajectory, **kwargs):
            if len(interpolations) == 1:
                predicted = [image.copy() for image in self.images]
                for image in predicted:
                    image.positions += 0.2
                write("last_predicted_path.traj", predicted)
            else:
                write(trajectory, self.images)

    catlearn = ModuleType("catlearn")
    optimize = ModuleType("catlearn.optimize")
    mlneb = ModuleType("catlearn.optimize.mlneb")
    mlneb.MLNEB = _FallbackMLNEB
    monkeypatch.setitem(sys.modules, "catlearn", catlearn)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning", None)
    monkeypatch.setitem(sys.modules, "catlearn.activelearning.mlneb", None)
    monkeypatch.setitem(sys.modules, "catlearn.optimize", optimize)
    monkeypatch.setitem(sys.modules, "catlearn.optimize.mlneb", mlneb)

    start = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]])
    end = Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.4]])
    _, _, converged, _ = _run_catlearn_mlneb_pass(
        start=start, end=end, calculator_getter=_EndpointCalculator, n_images=5,
        fmax=0.05, max_steps=20, spring_constant=0.1, catlearn_unc_convergence=0.05,
        catlearn_max_unc=0.3, catlearn_ml_steps=50, catlearn_dtrust=2.0,
        workflow_log_path=None, output_dir=tmp_path, catlearn_guesses_per_cycle=10,
    )

    assert len(interpolations) == 2
    assert isinstance(interpolations[1], list)
    assert np.allclose(interpolations[1][0].positions, start.positions + 0.2)
    assert converged is True
