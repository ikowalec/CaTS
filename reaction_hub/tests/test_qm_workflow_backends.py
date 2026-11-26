from __future__ import annotations

import csv
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from ase import Atoms

from cats.qm_backends.workflow import get_workflow_backend, list_workflow_backends
from cats.orca.tddft.io import stage_restart_gbw_path
from cats.working_directory.tddft_workflow import scan_log as scan_log_helpers


def test_orca_charged_calculators_keep_charge_and_use_odd_electron_default(
    tmp_path: Path,
    monkeypatch,
):
    from cats.working_directory import tddft_active_learning_workflow as workflow

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        workflow,
        "get_orca_calculator",
        lambda **kwargs: captured.append(kwargs) or "calculator",
    )
    atoms = Atoms("Au20")

    workflow.get_orca_dft_calculator(
        atoms,
        cpus=1,
        mem=1000,
        directory=tmp_path / "cation",
        system_charge=1,
    )
    workflow.get_orca_tddft_calculator(
        atoms,
        cpus=1,
        mem=1000,
        task="SP",
        directory=tmp_path / "anion",
        nroots=10,
        system_charge=-1,
    )

    assert [(item["charge"], item["mult"]) for item in captured] == [(1, 2), (-1, 2)]


def test_orca_tddft_calculator_enables_tda_explicitly(tmp_path: Path, monkeypatch):
    from cats.working_directory import tddft_active_learning_workflow as workflow

    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        workflow,
        "get_orca_calculator",
        lambda **kwargs: captured.append(kwargs) or "calculator",
    )

    workflow.get_orca_tddft_calculator(
        Atoms("Au2"),
        cpus=1,
        mem=1000,
        task="engrad",
        directory=tmp_path / "tda",
        nroots=12,
        iroot=3,
        use_tda=True,
    )

    assert "TDA TRUE" in str(captured[0]["orcablocks"])


def test_orca_does_not_restart_from_gbw_after_failed_calculation(tmp_path: Path):
    (tmp_path / "orca.gbw").write_bytes(b"truncated")
    (tmp_path / "initial.gbw").write_bytes(b"stale copied orbitals")
    (tmp_path / "orca.out").write_text("ORCA finished by error termination", encoding="utf-8")

    assert get_workflow_backend("orca").resolve_restart_path(tmp_path) is None


def test_orca_restart_uses_only_fresh_seed_or_normally_completed_predecessor(tmp_path: Path):
    backend = get_workflow_backend("orca")
    failed = tmp_path / "failed"
    complete = tmp_path / "complete"
    fresh = tmp_path / "fresh"
    for directory in (failed, complete, fresh):
        directory.mkdir()

    (failed / "orca.gbw").write_bytes(b"bad orbitals")
    (failed / "orca.out").write_text("ORCA finished by error termination", encoding="utf-8")
    (complete / "orca.gbw").write_bytes(b"good orbitals")
    (complete / "orca.out").write_text("ORCA TERMINATED NORMALLY\n", encoding="utf-8")
    (fresh / "initial.gbw").write_bytes(b"staged good orbitals")

    assert backend.resolve_restart_path(failed) is None
    assert backend.resolve_restart_path(complete) == complete / "orca.gbw"
    assert backend.resolve_restart_path(fresh) == fresh / "initial.gbw"
    assert backend.select_restart_source(failed, [complete]) == complete


def test_orca_calculator_stages_completed_gbw_under_distinct_absolute_path(
    tmp_path: Path,
):
    directory = tmp_path / "iteration"
    directory.mkdir()
    (directory / "orca.gbw").write_bytes(b"good orbitals")
    (directory / "orca.out").write_text("ORCA TERMINATED NORMALLY\n", encoding="utf-8")

    staged = (directory / "initial.gbw").resolve()
    assert stage_restart_gbw_path(directory) == staged
    assert staged.read_bytes() == b"good orbitals"


def test_list_workflow_backends_includes_orca_and_nwchem():
    keys = {entry["key"] for entry in list_workflow_backends()}

    assert {"orca", "nwchem"}.issubset(keys)


def test_nwchem_does_not_require_restart_data():
    assert get_workflow_backend("orca").capabilities.requires_restart_data
    assert not get_workflow_backend("nwchem").capabilities.requires_restart_data


def test_orca_backend_uses_configured_calculator_factories(tmp_path: Path):
    backend = get_workflow_backend("orca")
    calls: list[tuple[str, Path]] = []

    def _factory(_atoms, **kwargs):
        calls.append((str(kwargs["task"]), Path(kwargs["directory"])))
        return "calculator"

    configured = backend.with_calculator_factories(
        tddft_calculator_factory=_factory,
        dft_calculator_factory=_factory,
        dft_force_calculator_factory=_factory,
    )

    atoms = Atoms("H2")
    assert configured.make_tddft_calculator(atoms, task="energy", directory=tmp_path) == "calculator"
    assert configured.make_dft_calculator(atoms, task="Opt", directory=tmp_path) == "calculator"
    assert configured.make_dft_force_calculator(atoms, task="engrad", directory=tmp_path) == "calculator"
    assert calls == [("energy", tmp_path), ("Opt", tmp_path), ("engrad", tmp_path)]


def test_orca_backend_parses_final_single_point_energy(tmp_path: Path):
    out_path = tmp_path / "orca.out"
    out_path.write_text("FINAL SINGLE POINT ENERGY     -75.000000\n", encoding="utf-8")

    backend = get_workflow_backend("orca")

    assert backend.parse_total_energy_eV(out_path) == pytest.approx(-75.0 * 27.211386245988)


def test_nwchem_backend_parses_energy_and_transitions(tmp_path: Path):
    out_path = tmp_path / "nwchem.nwo"
    out_path.write_text(
        "\n".join(
            [
                " Total DFT energy = -10.000000",
                " Root    1         a              0.116857103 a.u.                3.1798 eV",
                " Total Oscillator Strength                     0.0000271105",
                " Task times",
            ]
        ),
        encoding="utf-8",
    )

    backend = get_workflow_backend("nwchem")
    transitions = backend.extract_transitions(out_path)

    assert backend.validate_output(out_path, warn=False)
    assert backend.parse_total_energy_eV(out_path) == pytest.approx(-10.0 * 27.211386245988)
    assert transitions.intensities.tolist() == pytest.approx([0.0000271105])
    assert transitions.wavenumbers_cm.tolist() == pytest.approx([3.1798 / 1.239841984e-4])
    assert transitions.wavelengths_nm is not None
    assert np.isfinite(transitions.wavelengths_nm).all()


def test_nwchem_calculator_expands_hse06():
    backend = get_workflow_backend("nwchem")
    kwargs = backend._calculator_kwargs(
        Atoms("H"),
        task="energy",
        directory=Path("."),
        cpus=1,
        mem=1000,
        basis_keyword="def2-SVP",
        xc_functional="HSE06",
    )

    assert kwargs["xc"] == "xpbe96 1.0 xcampbe96 -0.25 cpbe96 1.0 srhfexch 0.25"
    assert kwargs["cam"] == "0.11 cam_alpha 0.0 cam_beta 1.0"


def test_nwchem_calculator_adds_def2_ecp_only_for_heavy_elements():
    backend = get_workflow_backend("nwchem")
    common = {
        "task": "energy",
        "directory": Path("."),
        "cpus": 1,
        "mem": 1000,
        "basis_keyword": "def2-SVP",
        "xc_functional": "HSE06",
    }

    with_ecp = backend._calculator_kwargs(Atoms("RbLaHfAuPb"), **common)
    without_ecp = backend._calculator_kwargs(Atoms("KrCuTiCeLu"), **common)

    assert with_ecp["ecp"] == {
        "Au": "library def2-ecp",
        "Hf": "library def2-ecp",
        "La": "library def2-ecp",
        "Pb": "library def2-ecp",
        "Rb": "library def2-ecp",
    }
    assert "ecp" not in without_ecp
    assert without_ecp["basispar"] == "bse"


def test_nwchem_backend_parses_spin_forbidden_triplet_roots(tmp_path: Path):
    out_path = tmp_path / "nwchem_tddft.nwo"
    out_path.write_text(
        "\n".join(
            [
                " ----------------------------------------------------------------------------",
                "  Root   1 triplet a              0.262572727 a.u.                7.1450 eV",
                "  ----------------------------------------------------------------------------",
                "     Transition Moments                    Spin forbidden",
                "     Oscillator Strength                   Spin forbidden",
                "",
                "     Occ.    5  a   ---  Virt.    6  a   -0.99739 X",
                "  Root   2 singlet a              0.120000000 a.u.                3.2654 eV",
                "     Oscillator Strength                   0.12345",
            ]
        ),
        encoding="utf-8",
    )

    transitions = get_workflow_backend("nwchem").extract_transitions(out_path)

    assert transitions.intensities.tolist() == pytest.approx([0.0, 0.12345])
    assert transitions.wavenumbers_cm.tolist() == pytest.approx(
        [7.1450 / 1.239841984e-4, 3.2654 / 1.239841984e-4]
    )
    assert transitions.wavelengths_nm is not None
    assert transitions.wavelengths_nm.tolist()[0] == pytest.approx(1.0e7 / (7.1450 / 1.239841984e-4))


def test_nwchem_backend_prefers_total_oscillator_strength(tmp_path: Path):
    out_path = tmp_path / "nwchem_tddft.nwo"
    out_path.write_text(
        "\n".join(
            [
                "  Root   1 singlet a              0.288529306 a.u.                7.8513 eV",
                "     Dipole Oscillator Strength                    0.0185001666",
                "     Electric Quadrupole                           0.0000000008",
                "     Magnetic Dipole                               0.0000000853",
                "     Total Oscillator Strength                     0.0185002527",
                "  Root   2 singlet a              0.359145819 a.u.                9.7729 eV",
                "     Dipole Oscillator Strength                    0.0000000000",
                "     Electric Quadrupole                           0.0000000219",
                "     Magnetic Dipole                               0.0000002683",
                "     Total Oscillator Strength                     0.0000002902",
            ]
        ),
        encoding="utf-8",
    )

    transitions = get_workflow_backend("nwchem").extract_transitions(out_path)

    assert transitions.intensities.tolist() == pytest.approx([0.0185002527, 0.0000002902])
    assert transitions.wavenumbers_cm.tolist() == pytest.approx(
        [7.8513 / 1.239841984e-4, 9.7729 / 1.239841984e-4]
    )


def test_partial_nwchem_tddft_output_is_rerun(tmp_path: Path, monkeypatch):
    workflow = pytest.importorskip("cats.working_directory.tddft_active_learning_workflow")
    spectrum_dir = tmp_path / workflow.REACTION_HUB_FINAL_TDDFT_SPECTRUM_DIRNAME
    spectrum_dir.mkdir()
    (spectrum_dir / "nwchem.nwo").write_text(
        "\n".join(
            [
                " Total DFT energy = -10.000000",
                " Root    1         a              0.116857103 a.u.                3.1798 eV",
                " Total Oscillator Strength                     0.0000271105",
            ]
        ),
        encoding="utf-8",
    )
    rerun = {}

    def _fake_run(**kwargs):
        rerun.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(workflow, "DFT_ROOT_DIR", tmp_path)
    monkeypatch.setattr(workflow, "_run_final_tddft_spectrum", _fake_run)

    result = workflow._ensure_final_tddft_spectrum_outputs(
        final_atoms=Atoms("H"),
        final_opt_dir=tmp_path,
        cpus=1,
        mem=1000,
        nroots=5,
        basis_keyword="def2-SVP",
        xc_functional="PBE0",
        solvent=None,
        spin_multiplicity=2,
        workflow_backend=get_workflow_backend("nwchem"),
    )

    assert result == {"status": "ok"}
    assert rerun["nroots"] == 5


def test_charged_fragment_single_points_put_charge_on_cluster(tmp_path: Path, monkeypatch):
    workflow = pytest.importorskip("cats.working_directory.tddft_active_learning_workflow")
    states = []

    def _fake_single_point(**kwargs):
        states.append((kwargs["spin_multiplicity"], kwargs.get("system_charge", 0)))
        return {"status": "ok"}

    monkeypatch.setattr(workflow, "_run_isolated_single_point", _fake_single_point)

    workflow._run_final_cluster_single_point(
        final_atoms=Atoms("CuH", positions=[(0, 0, 0), (0, 0, 1.5)]),
        final_opt_dir=tmp_path,
        cpus=1,
        mem=1000,
        basis_keyword="def2-SVP",
        xc_functional="PBE0",
        solvent=None,
        spin_multiplicity=2,
        system_charge=1,
        workflow_backend=get_workflow_backend("nwchem"),
    )

    assert states == [(1, 1), (2, 0)]


def test_nwchem_final_spectrum_range_falls_back_when_visible_window_is_empty():
    from cats.working_directory.nwchem_single_workflow import build_uv_spectrum_nm

    uv_transitions = SimpleNamespace(
        wavelengths_nm=np.asarray([157.9, 126.9]),
        wavenumbers_cm=np.asarray([63331.0, 78802.0]),
        intensities=np.asarray([0.0185, 0.2826]),
    )

    axis, _profile, fallback_used = build_uv_spectrum_nm(
        uv_transitions,
        linewidth_nm=10.0,
        start_nm=300.0,
        end_nm=800.0,
        step=1.0,
        normalize=False,
    )

    assert fallback_used is True
    assert axis[0] == pytest.approx(300.0)
    assert axis[-1] == pytest.approx(800.0)

    visible_transitions = SimpleNamespace(
        wavelengths_nm=np.asarray([520.0]),
        wavenumbers_cm=np.asarray([19230.77]),
        intensities=np.asarray([0.1]),
    )

    axis, _profile, fallback_used = build_uv_spectrum_nm(
        visible_transitions,
        linewidth_nm=10.0,
        start_nm=300.0,
        end_nm=800.0,
        step=1.0,
        normalize=False,
    )

    assert fallback_used is False
    assert axis[0] == pytest.approx(300.0)
    assert axis[-1] == pytest.approx(800.0)


def test_uvvis_linewidth_is_full_width_at_half_maximum():
    from cats.visualise.uvvis import plot_uv_spectrum_nm

    transitions = SimpleNamespace(
        wavelengths_nm=np.asarray([100.0]),
        wavenumbers_cm=np.asarray([100000.0]),
        intensities=np.asarray([1.0]),
    )

    axis, spectrum = plot_uv_spectrum_nm(
        transitions,
        linewidth_nm=20.0,
        start_nm=90.0,
        end_nm=110.0,
        step=10.0,
    )

    assert spectrum.y.tolist() == pytest.approx([0.5, 1.0, 0.5])
    axis.figure.clf()


def test_nwchem_single_optimisation_disables_bond_scan_for_final_spectrum(tmp_path: Path, monkeypatch):
    import sys
    from dataclasses import dataclass

    from ase.io import write

    from cats.working_directory import nwchem_single_workflow

    @dataclass
    class FakeBondScanSettings:
        enabled: bool = True

    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])
    input_path = tmp_path / "input.extxyz"
    write(input_path, atoms)
    calls: list[dict[str, object]] = []

    def fake_run_workflow(workflow_atoms, **kwargs):
        calls.append(kwargs)
        return workflow_atoms

    fake_workflow = ModuleType("cats.working_directory.tddft_active_learning_workflow")
    fake_workflow.BondScanSettings = FakeBondScanSettings
    fake_workflow.run_workflow = fake_run_workflow

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        nwchem_single_workflow,
        "_resolve_nwchem_command",
        lambda **_kwargs: "nwchem PREFIX.nwi > PREFIX.nwo",
    )
    monkeypatch.setitem(
        sys.modules,
        "cats.working_directory.tddft_active_learning_workflow",
        fake_workflow,
    )

    nwchem_single_workflow.run_nwchem_single_optimisation(
        input_path=str(input_path),
        hpc_name="archer2",
        nodes=2,
        tasks_per_node=64,
        cpus_per_task=1,
        backend_settings={"version": "7.2.2", "memory_mb": 1800, "molden": True},
        basis_keyword="def2-SVP",
        xc_functional="PBE0",
        final_tddft_spectrum=True,
        final_tddft_nroots=100,
    )

    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["use_tddft_optimisation"] is False
    assert kwargs["final_tddft_spectrum"] is True
    assert kwargs["final_tddft_nroots"] == 100
    bond_scan = kwargs["bond_scan"]
    assert isinstance(bond_scan, FakeBondScanSettings)
    assert bond_scan.enabled is False


def _write_test_spectrum_csv(path: Path, values: list[tuple[float, float]]) -> None:
    path.write_text(
        "wavelength_nm,intensity\n"
        + "".join(f"{float(x):.8g},{float(y):.8g}\n" for x, y in values),
        encoding="utf-8",
    )


def test_cluster_difference_spectrum_uses_raw_subtraction_and_common_peak_normalization(tmp_path: Path):
    workflow = pytest.importorskip("cats.working_directory.tddft_active_learning_workflow")
    _write_cluster_adsorbate_difference_spectrum = workflow._write_cluster_adsorbate_difference_spectrum

    full_csv = tmp_path / "full.csv"
    cluster_csv = tmp_path / "cluster.csv"
    _write_test_spectrum_csv(full_csv, [(300.0, 10.0), (310.0, 4.0), (360.0, 2.0), (800.0, 1.0)])
    _write_test_spectrum_csv(cluster_csv, [(300.0, 5.0), (310.0, 1.0), (360.0, 8.0), (800.0, 1.0)])

    result = _write_cluster_adsorbate_difference_spectrum(
        full_csv=full_csv,
        cluster_csv=cluster_csv,
        output_dir=tmp_path,
        full_transitions=SimpleNamespace(wavelengths_nm=np.asarray([310.0]), intensities=np.asarray([1.0e-3])),
        cluster_transitions=SimpleNamespace(wavelengths_nm=np.asarray([360.0]), intensities=np.asarray([1.0e-3])),
    )

    assert result["normalization_mode"] == "common_peak"
    assert result["normalization_factor"] == pytest.approx(8.0)
    assert result["plot_left_nm"] == pytest.approx(360.0)
    with (tmp_path / "cluster_adsorbate_difference_spectrum.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row_360 = next(row for row in rows if abs(float(row["wavelength_nm"]) - 360.0) < 1.0e-12)
    assert float(row_360["delta_intensity"]) == pytest.approx(-6.0)
    assert float(row_360["delta_intensity_raw"]) == pytest.approx(-6.0)
    assert float(row_360["full_intensity_common_peak"]) == pytest.approx(0.25)
    assert float(row_360["cluster_intensity_common_peak"]) == pytest.approx(1.0)
    assert float(row_360["delta_intensity_common_peak"]) == pytest.approx(-0.75)


def test_cluster_difference_spectrum_fair_left_limit_ignores_weak_transitions(tmp_path: Path):
    workflow = pytest.importorskip("cats.working_directory.tddft_active_learning_workflow")
    _write_cluster_adsorbate_difference_spectrum = workflow._write_cluster_adsorbate_difference_spectrum

    full_csv = tmp_path / "full.csv"
    cluster_csv = tmp_path / "cluster.csv"
    values = [(300.0, 1.0), (320.0, 2.0), (400.0, 3.0), (800.0, 1.0)]
    _write_test_spectrum_csv(full_csv, values)
    _write_test_spectrum_csv(cluster_csv, values)

    result = _write_cluster_adsorbate_difference_spectrum(
        full_csv=full_csv,
        cluster_csv=cluster_csv,
        output_dir=tmp_path,
        full_transitions=SimpleNamespace(
            wavelengths_nm=np.asarray([310.0, 320.0]),
            intensities=np.asarray([1.0e-5, 1.0e-3]),
        ),
        cluster_transitions=SimpleNamespace(wavelengths_nm=np.asarray([400.0]), intensities=np.asarray([1.0e-3])),
    )

    assert result["plot_left_nm"] == pytest.approx(400.0)


def test_cluster_difference_spectrum_identical_inputs_have_zero_delta(tmp_path: Path):
    workflow = pytest.importorskip("cats.working_directory.tddft_active_learning_workflow")
    _write_cluster_adsorbate_difference_spectrum = workflow._write_cluster_adsorbate_difference_spectrum

    full_csv = tmp_path / "full.csv"
    cluster_csv = tmp_path / "cluster.csv"
    values = [(300.0, 1.0), (500.0, 3.0), (800.0, 2.0)]
    _write_test_spectrum_csv(full_csv, values)
    _write_test_spectrum_csv(cluster_csv, values)

    _write_cluster_adsorbate_difference_spectrum(
        full_csv=full_csv,
        cluster_csv=cluster_csv,
        output_dir=tmp_path,
        full_transitions=SimpleNamespace(wavelengths_nm=np.asarray([500.0]), intensities=np.asarray([1.0e-3])),
        cluster_transitions=SimpleNamespace(wavelengths_nm=np.asarray([500.0]), intensities=np.asarray([1.0e-3])),
    )

    with (tmp_path / "cluster_adsorbate_difference_spectrum.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
    assert all(float(row["delta_intensity_raw"]) == pytest.approx(0.0) for row in rows)
    assert all(float(row["delta_intensity_common_peak"]) == pytest.approx(0.0) for row in rows)


def test_nwchem_backend_renders_tddft_input_file():
    backend = get_workflow_backend("nwchem")
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])

    files = backend.render_input_files(
        atoms,
        {
            "task": "optimize_tddft",
            "basis": "def2-SVP",
            "xc": "PBE0",
            "nroots": 10,
            "prefix": "h2",
            "molden": True,
        },
    )

    assert set(files) == {"h2.nwi"}
    text = files["h2.nwi"]
    assert "geometry units angstrom" in text
    assert "H" in text
    assert "* library def2-SVP" in text
    assert "xc PBE0" in text
    assert "task dft optimize" in text
    assert "nroots 10" in text
    assert "task tddft energy" in text
    assert "  moldenfile\n" in text
    assert "molden_norm none" in text
    assert "h2.molden" not in text


def test_nwchem_backend_renders_hse06_input_file():
    backend = get_workflow_backend("nwchem")
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])

    text = backend.render_input_files(
        atoms,
        {"prefix": "h2", "xc_functional": "hse06", "molden": False},
    )["h2.nwi"]

    assert "xc xpbe96 1.0 xcampbe96 -0.25 cpbe96 1.0 srhfexch 0.25" in text
    assert "cam 0.11 cam_alpha 0.0 cam_beta 1.0" in text


def test_nwchem_backend_sets_hse06_cam_for_calculator(tmp_path: Path):
    backend = get_workflow_backend("nwchem")
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])

    calc = backend.make_dft_calculator(
        atoms,
        task="energy",
        directory=tmp_path,
        cpus=1,
        mem=4000,
        basis_keyword="def2-SVP",
        xc_functional=backend.normalize_xc_functional("hse06"),
    )

    assert calc.parameters["dft"]["xc"] == "xpbe96 1.0 xcampbe96 -0.25 cpbe96 1.0 srhfexch 0.25"
    assert calc.parameters["dft"]["cam"] == "0.11 cam_alpha 0.0 cam_beta 1.0"


def test_nwchem_backend_omits_unset_solvent():
    backend = get_workflow_backend("nwchem")
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])

    files = backend.render_input_files(atoms, {"prefix": "h2", "solvent": None, "molden": False})

    text = files["h2.nwi"]
    assert "\nsolvent\n" not in text
    assert "\ncosmo\n" not in text


def test_nwchem_backend_renders_cosmo_solvent_block():
    backend = get_workflow_backend("nwchem")
    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])

    files = backend.render_input_files(atoms, {"prefix": "h2", "solvent": "Water", "molden": False})

    assert "\ncosmo\n  solvent h2o\nend\n" in files["h2.nwi"]


def test_nwchem_calculator_omits_unset_solvent_keyword():
    from cats.nwchem.run import get_nwchem_calculator

    calc = get_nwchem_calculator(
        task="energy",
        nwchem_command="nwchem PREFIX.nwi > PREFIX.nwo",
        mult=1,
        solvent=None,
    )

    assert "solvent" not in calc.parameters
    assert "cosmo" not in calc.parameters


def test_nwchem_calculator_uses_cosmo_for_solvent():
    from cats.nwchem.run import get_nwchem_calculator

    calc = get_nwchem_calculator(
        task="energy",
        nwchem_command="nwchem PREFIX.nwi > PREFIX.nwo",
        mult=1,
        solvent="h2o",
    )

    assert calc.parameters["cosmo"] == {"solvent": "h2o"}


def test_scan_log_helpers_accept_backend_output_filename(tmp_path: Path):
    scan_dir = tmp_path / "scan_iter_1"
    scan_dir.mkdir()
    output_path = scan_dir / "nwchem.nwo"
    output_path.write_text("Task times\n", encoding="utf-8")
    seen_paths: list[Path] = []

    def _output_path_for_dir(path: Path) -> Path:
        return path / "nwchem.nwo"

    def _validate(path: Path) -> bool:
        seen_paths.append(path)
        return path.exists()

    assert scan_log_helpers.scan_dir_has_valid_output(
        scan_dir,
        validate_orca_output=_validate,
        output_path_for_dir=_output_path_for_dir,
    )
    assert seen_paths == [output_path]

    resolved = scan_log_helpers.resolve_scan_dir_for_iteration_for_log_repair(
        1,
        use_tddft_optimisation=False,
        resolve_scan_dir=lambda _iteration: scan_dir,
        tddft_scan_geom_dirs=lambda _iteration: [],
        resolve_opt_dir_for_mode=lambda _iteration, **_kwargs: tmp_path / "missing",
        output_path_for_dir=_output_path_for_dir,
    )

    assert resolved == scan_dir


def test_run_workflow_accepts_backend_argument_and_rejects_unwired_nwchem():
    pytest.importorskip("catlearn.activelearning.local")

    from cats.orca.tddft.scan import BondScanSettings
    from cats.working_directory.tddft_active_learning_workflow import run_workflow

    atoms = Atoms("H2", positions=[(0, 0, 0), (0, 0, 0.74)])

    with pytest.raises(NotImplementedError, match="active-learning TDDFT optimisation workflow"):
        run_workflow(
            atoms,
            qm_backend="nwchem",
            bond_scan=BondScanSettings(enabled=False),
            use_tddft_optimisation=False,
            final_tddft_spectrum=False,
        )
