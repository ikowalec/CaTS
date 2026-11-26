import importlib.util
import json
from pathlib import Path

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "cats" / "working_directory" / "evaluated_traj_to_xyz.py"
SPEC = importlib.util.spec_from_file_location("evaluated_traj_to_xyz", SCRIPT_PATH)
exporter = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(exporter)


def test_selected_indices_does_not_duplicate_single_frame():
    assert exporter.selected_indices(1, sample_interval=10) == (0,)


def test_collect_images_includes_single_frame_once(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / ("a" * 32)
    run_dir.mkdir()
    traj_path = run_dir / "evaluated.traj"
    write(traj_path, [Atoms("AuCu")])

    monkeypatch.setattr(
        exporter,
        "load_run_metadata",
        lambda root: {"a" * 32: {"settings_profile": "cam-b3lyp-lanl2dz"}},
    )

    images, audit = exporter.collect_images(tmp_path)

    assert len(images) == 1
    assert (traj_path, "included", "1 frames") in audit


def test_collect_images_can_select_settings_profile_for_training(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / ("a" * 32)
    run_dir.mkdir()
    traj_path = run_dir / "evaluated.traj"
    write(traj_path, [Atoms("AuCu")])

    monkeypatch.setattr(
        exporter,
        "load_run_metadata",
        lambda root: {"a" * 32: {"settings_profile": "other-profile"}},
    )

    default_images, default_audit = exporter.collect_images(tmp_path)
    selected_images, selected_audit = exporter.collect_images(
        tmp_path,
        settings_profiles={"other-profile"},
    )

    assert default_images == []
    assert (traj_path, "skipped", "settings_missing") in default_audit
    assert len(selected_images) == 1
    assert (traj_path, "included", "1 frames") in selected_audit


def _atoms_with_results(energy: float, force_z: float = 0.0) -> Atoms:
    atoms = Atoms("AuCu")
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=float(energy),
        forces=[[0.0, 0.0, float(force_z)], [0.0, 0.0, float(-force_z)]],
    )
    return atoms


def test_collect_images_skips_frames_with_huge_forces(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / ("a" * 32)
    run_dir.mkdir()
    traj_path = run_dir / "evaluated.traj"
    good = _atoms_with_results(-10.0, force_z=1.0)
    bad = _atoms_with_results(-10.0, force_z=100.0)
    write(traj_path, [good, bad])

    monkeypatch.setattr(
        exporter,
        "load_run_metadata",
        lambda root: {"a" * 32: {"settings_profile": "cam-b3lyp-lanl2dz"}},
    )

    images, audit = exporter.collect_images(tmp_path)

    assert len(images) == 1
    assert images[0].get_potential_energy() == -10.0
    assert (traj_path, "included", "1 frames; skipped 1 nonsensical") in audit


def test_collect_images_skips_single_frame_with_absurd_energy(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / ("a" * 32)
    run_dir.mkdir()
    traj_path = run_dir / "evaluated.traj"
    write(traj_path, [_atoms_with_results(1.0e9)])

    monkeypatch.setattr(
        exporter,
        "load_run_metadata",
        lambda root: {"a" * 32: {"settings_profile": "cam-b3lyp-lanl2dz"}},
    )

    images, audit = exporter.collect_images(tmp_path)

    assert images == []
    assert (traj_path, "skipped", "no_sensible_frames") in audit


def test_collect_images_skips_energy_outlier_frame(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / ("a" * 32)
    run_dir.mkdir()
    traj_path = run_dir / "evaluated.traj"
    write(
        traj_path,
        [
            _atoms_with_results(-10.0),
            _atoms_with_results(-11.0),
            _atoms_with_results(1000.0),
        ],
    )

    monkeypatch.setattr(
        exporter,
        "load_run_metadata",
        lambda root: {"a" * 32: {"settings_profile": "cam-b3lyp-lanl2dz"}},
    )

    images, audit = exporter.collect_images(tmp_path, sample_interval=1)

    assert len(images) == 1
    assert images[0].get_potential_energy() == -10.0
    assert (traj_path, "included", "1 frames; skipped 1 nonsensical") in audit


def test_collect_images_species_filter_skips_reference_gas(monkeypatch, tmp_path: Path):
    metal_dir = tmp_path / ("a" * 32)
    gas_dir = tmp_path / ("b" * 32)
    metal_dir.mkdir()
    gas_dir.mkdir()
    write(metal_dir / "evaluated.traj", [Atoms("AuCu")])
    write(gas_dir / "evaluated.traj", [Atoms("CO")])

    def fake_metadata(root: Path):
        return {
            "a" * 32: {"settings_profile": "cam-b3lyp-lanl2dz"},
            "b" * 32: {"settings_profile": "cam-b3lyp-lanl2dz", "reference_adsorbate_id": 1},
        }

    monkeypatch.setattr(exporter, "load_run_metadata", fake_metadata)

    images, audit = exporter.collect_images(tmp_path, species={"Au", "Cu"})

    assert [atoms.get_chemical_formula() for atoms in images] == ["AuCu"]
    assert (gas_dir / "evaluated.traj", "skipped", "species_filter") in audit


def test_collect_images_skips_profile_match_with_conflicting_summary(monkeypatch, tmp_path: Path):
    run_dir = tmp_path / ("a" * 32)
    run_dir.mkdir()
    traj_path = run_dir / "evaluated.traj"
    write(traj_path, [Atoms("AuCu")])
    (run_dir / "reaction_hub_summary.json").write_text(
        json.dumps(
            {
                "settings": {
                    "basis_keyword": "def2-svp",
                    "xc_functional": "pbe0",
                }
            }
        ),
        encoding="utf-8",
    )

    def fake_metadata(root: Path):
        return {"a" * 32: {"settings_profile": "cam-b3lyp-lanl2dz"}}

    monkeypatch.setattr(exporter, "load_run_metadata", fake_metadata)

    images, audit = exporter.collect_images(tmp_path)

    assert images == []
    assert (traj_path, "skipped", "settings_conflict") in audit


def test_collect_images_skips_matching_parent_summary_with_conflicting_input(tmp_path: Path):
    parent = tmp_path / "batch"
    run_dir = parent / ("a" * 32)
    run_dir.mkdir(parents=True)
    traj_path = run_dir / "evaluated.traj"
    write(traj_path, [Atoms("AuCu")])
    (parent / "reaction_hub_summary.json").write_text(
        json.dumps(
            {
                "settings": {
                    "basis_keyword": "lanl2dz_mixed",
                    "xc_functional": "CAM-B3LYP",
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "run_workflow_submit.py").write_text(
        "run_workflow(basis_keyword='def2-svp', xc_functional='pbe0')",
        encoding="utf-8",
    )

    images, audit = exporter.collect_images(tmp_path)

    assert images == []
    assert (traj_path, "skipped", "settings_mismatch") in audit
