from __future__ import annotations

import base64
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect as ase_connect
from ase.io import read as ase_read, write as ase_write
from django.utils import timezone

from references.models import CalculationSettingsProfile
from jobs.models import HPCSystem, JobRecord, JobStatus
from results.models import Artifact, ArtifactKind, ComputedProperty
from workflows.models import WorkflowDefinition, WorkflowRun
from ingest.hpc import SlurmSSHClient
from ingest.spectral import load_spectral_dataset
from ingest.tasks import backfill_ct_plots_task
from ingest.services import (
    _ingest_remote_summary_for_job,
    _relative_artifact_path,
    ingest_single_run,
    ingest_workflow_outputs,
    sync_hpc_jobs,
)


@pytest.mark.django_db
def test_ingest_single_run_is_idempotent(tmp_path: Path):
    run_dir = tmp_path / "row_0001"
    (run_dir / "TDDFT_active_learning_shared").mkdir(parents=True)
    (run_dir / "TDDFT_active_learning_shared" / "workflow_root.txt").write_text(
        str(run_dir), encoding="utf-8"
    )
    opt_dir = run_dir / "TDDFT" / "opt_iter_1"
    opt_dir.mkdir(parents=True)
    (opt_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")
    (opt_dir / "orca.out").write_text(
        "...\nFINAL SINGLE POINT ENERGY   -123.456789\n", encoding="utf-8"
    )
    (run_dir / "input.xyz").write_text("2\ncomment\nH 0 0 0\nH 0 0 0.74\n", encoding="utf-8")

    ingest_single_run(run_dir)
    ingest_single_run(run_dir)

    run_count = WorkflowRun.objects.filter(working_directory=str(run_dir.resolve())).count()
    assert run_count == 1

    artifact_count = Artifact.objects.filter(workflow_run__working_directory=str(run_dir.resolve())).count()
    assert artifact_count >= 2

    prop_qs = ComputedProperty.objects.filter(
        workflow_run__working_directory=str(run_dir.resolve()),
        name="final_single_point_energy",
        unit="Eh",
    )
    assert prop_qs.count() == 1
    assert prop_qs.first().value_float == pytest.approx(-123.456789)


def test_relative_artifact_path_handles_windows_absolute_style_paths():
    run_dir = "C:/work/PROJECT/GROUP/user/reaction_hub_runs/Au13_O2_v2_1_blscan_dft_cAu"
    artifact_path = "C:/work/PROJECT/GROUP/user/reaction_hub_runs/Au13_O2_v2_1_blscan_dft_cAu/DFT/scan_plots/uvvis_iter_008.png"
    rel = _relative_artifact_path(run_dir, artifact_path)
    assert rel == "DFT/scan_plots/uvvis_iter_008.png"


@pytest.mark.django_db
def test_ingest_single_run_collects_xyzrender_artifacts(tmp_path: Path):
    run_dir = tmp_path / "row_render"
    render_dir = run_dir / "results" / "energy" / "1"
    render_dir.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")
    (render_dir / "test.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
        encoding="utf-8",
    )
    (render_dir / "orca.gif").write_bytes(b"GIF89a")

    run = ingest_single_run(run_dir)
    run = ingest_single_run(run_dir)

    svg_artifact = Artifact.objects.get(
        workflow_run=run,
        path="results/energy/1/test.svg",
    )
    gif_artifact = Artifact.objects.get(
        workflow_run=run,
        path="results/energy/1/orca.gif",
    )

    assert svg_artifact.kind == "image"
    assert gif_artifact.kind == "image"
    assert svg_artifact.label == "render:test.svg"
    assert gif_artifact.label == "render:orca.gif"


@pytest.mark.django_db
def test_ingest_single_run_skips_per_iteration_uvvis_images(tmp_path: Path):
    run_dir = tmp_path / "row_skip_uvvis_iter"
    (run_dir / "TDDFT_active_learning_shared").mkdir(parents=True)
    (run_dir / "TDDFT_active_learning_shared" / "workflow_root.txt").write_text(
        str(run_dir), encoding="utf-8"
    )
    plot_dir = run_dir / "DFT" / "scan_plots"
    plot_dir.mkdir(parents=True)
    (plot_dir / "uvvis_iter_008.png").write_bytes(b"uvvis_png")
    (plot_dir / "stacked_uvvis.png").write_bytes(b"stacked_png")

    run = ingest_single_run(run_dir)

    assert Artifact.objects.filter(
        workflow_run=run,
        path="DFT/scan_plots/stacked_uvvis.png",
        label="plot:stacked_uvvis",
    ).exists()
    assert not Artifact.objects.filter(
        workflow_run=run,
        path="DFT/scan_plots/uvvis_iter_008.png",
    ).exists()


@pytest.mark.django_db
def test_ingest_single_run_keeps_per_iteration_ct_images(tmp_path: Path):
    run_dir = tmp_path / "row_keep_ct_iter"
    (run_dir / "TDDFT_active_learning_shared").mkdir(parents=True)
    (run_dir / "TDDFT_active_learning_shared" / "workflow_root.txt").write_text(
        str(run_dir), encoding="utf-8"
    )
    plot_dir = run_dir / "DFT" / "scan_plots"
    plot_dir.mkdir(parents=True)
    ct_names = [
        "ct_spectrum_iter_000.png",
        "ct_spectrum_iter_000_overall.png",
        "ct_spectrum_iter_000_transfer_only.png",
    ]
    for name in ct_names:
        (plot_dir / name).write_bytes(f"{name}_png".encode("ascii"))

    run = ingest_single_run(run_dir)

    for name in ct_names:
        assert Artifact.objects.filter(
            workflow_run=run,
            path=f"DFT/scan_plots/{name}",
            label=f"plot:{name}",
            kind=ArtifactKind.IMAGE,
        ).exists()


def test_slurm_query_jobs_prefers_parent_sacct_state_over_substeps():
    client = SlurmSSHClient(host="example.org", port=22, username="user")
    sacct_output = "\n".join(
        [
            "12585898|FAILED|00:10:00|workflow|/work/PROJECT/GROUP/user/reaction_hub_runs/run_a|1:0",
            "12585898.batch|FAILED|00:10:00|batch||1:0",
            "12585898.extern|COMPLETED|00:10:00|extern||0:0",
            "12585898.0|COMPLETED|00:09:58|orted||0:0",
            "12585898.1|COMPLETED|00:09:58|orted||0:0",
        ]
    )

    def _fake_run(command: str) -> tuple[str, str, int]:
        if command.startswith("squeue "):
            return ("", "", 0)
        if command.startswith("sacct "):
            return (sacct_output, "", 0)
        raise RuntimeError(f"unexpected command: {command}")

    client._run = _fake_run  # type: ignore[method-assign]
    snapshots = client.query_jobs(["12585898"])

    assert snapshots["12585898"]["state"] == "FAILED"


def test_slurm_query_jobs_uses_sacct_when_squeue_reports_invalid_job_id():
    client = SlurmSSHClient(host="example.org", port=22, username="user")
    sacct_output = "12585899|COMPLETED|00:04:12|workflow|/work/PROJECT/GROUP/user/run_b|0:0"

    def _fake_run(command: str) -> tuple[str, str, int]:
        if command.startswith("squeue "):
            return ("", "slurm_load_jobs error: Invalid job id specified", 1)
        if command.startswith("sacct "):
            return (sacct_output, "", 0)
        raise RuntimeError(f"unexpected command: {command}")

    client._run = _fake_run  # type: ignore[method-assign]
    snapshots = client.query_jobs(["12585899"])

    assert snapshots["12585899"]["state"] == "COMPLETED"


def test_slurm_query_jobs_matches_heterogeneous_parent_job_id_rows():
    client = SlurmSSHClient(host="example.org", port=22, username="user")
    sacct_output = "12585900+0|COMPLETED|00:04:12|workflow|/work/PROJECT/GROUP/user/run_c|0:0"

    def _fake_run(command: str) -> tuple[str, str, int]:
        if command.startswith("squeue "):
            return ("", "", 0)
        if command.startswith("sacct "):
            return (sacct_output, "", 0)
        raise RuntimeError(f"unexpected command: {command}")

    client._run = _fake_run  # type: ignore[method-assign]
    snapshots = client.query_jobs(["12585900"])

    assert snapshots["12585900"]["state"] == "COMPLETED"


def test_slurm_query_jobs_falls_back_to_scontrol_when_sacct_unavailable():
    client = SlurmSSHClient(host="example.org", port=22, username="user")
    scontrol_output = (
        "JobId=12585901 JobName=workflow JobState=COMPLETED "
        "RunTime=00:02:01 ExitCode=0:0 WorkDir=/work/PROJECT/GROUP/user/run_d"
    )

    def _fake_run(command: str) -> tuple[str, str, int]:
        if command.startswith("squeue "):
            return ("", "slurm_load_jobs error: Invalid job id specified", 1)
        if command.startswith("sacct "):
            return ("", "sacct: command not found", 127)
        if command.startswith("scontrol -o show job "):
            return (scontrol_output, "", 0)
        raise RuntimeError(f"unexpected command: {command}")

    client._run = _fake_run  # type: ignore[method-assign]
    snapshots = client.query_jobs(["12585901"])

    assert snapshots["12585901"]["state"] == "COMPLETED"
    assert snapshots["12585901"]["exit_code"] == "0:0"


@pytest.mark.django_db
def test_ingest_single_run_reuses_existing_run_by_working_directory(tmp_path: Path):
    run_dir = tmp_path / "row_0002"
    run_dir.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")

    workflow = WorkflowDefinition.objects.create(
        name="wf_existing",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    profile = CalculationSettingsProfile.objects.create(name="profile_existing")
    existing = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="row_0002",
        status="initiated",
        working_directory=str(run_dir.resolve()),
        settings_profile=profile,
        metadata={"submission": {"slurm_preset": "archer2_dev_test_short"}},
    )

    run = ingest_single_run(run_dir, workflow_name="wf_other")

    assert run.pk == existing.pk
    assert WorkflowRun.objects.filter(working_directory=str(run_dir.resolve())).count() == 1
    assert run.workflow_id == workflow.id
    assert run.settings_profile_id == profile.id
    assert run.metadata.get("submission", {}).get("slurm_preset") == "archer2_dev_test_short"


@pytest.mark.django_db
def test_ingest_single_run_reuses_existing_workflow_by_code_path(tmp_path: Path):
    run_dir = tmp_path / "row_0003"
    run_dir.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")

    existing_workflow = WorkflowDefinition.objects.create(
        name="TDDFT Active Learning",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )

    run = ingest_single_run(run_dir)

    assert run.workflow_id == existing_workflow.id
    assert WorkflowDefinition.objects.filter(code_path="cats/working_directory/tddft_active_learning_workflow.py").count() == 1
    assert not WorkflowDefinition.objects.filter(name="cplas_tddft_active_learning").exists()


@pytest.mark.django_db
def test_ingest_single_run_reads_reaction_hub_summary(tmp_path: Path):
    run_dir = tmp_path / "row_summary"
    run_dir.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")
    (run_dir / "reaction_hub_final.xyz").write_text(
        "2\nCO\nC 0.0 0.0 0.0\nO 0.0 0.0 1.128\n",
        encoding="utf-8",
    )
    summary = {
        "generated_at_utc": "2026-02-20T16:00:00+00:00",
        "scan": {
            "points": [
                {"iteration": 1, "energy_eV": -100.0, "bond_length_A": 1.10},
                {"iteration": 2, "energy_eV": -99.5, "bond_length_A": 1.15},
                {"iteration": 3, "energy_eV": -99.7, "bond_length_A": 1.20},
            ],
            "start_energy_eV": -100.0,
            "peak_energy_eV": -99.5,
            "activation_barrier_eV": 0.5,
        },
        "final": {
            "formula": "CO",
            "atom_count": 2,
            "symbols": ["C", "O"],
            "positions_A": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]],
            "energy_hartree": -113.0,
            "energy_eV": -3074.886647796644,
            "forces_eV_A": [[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]],
            "charges": [-0.2, 0.2],
            "charge_method": "loewdin",
        },
    }
    (run_dir / "reaction_hub_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    run = ingest_single_run(run_dir)

    assert run.status == "completed"
    assert ComputedProperty.objects.filter(
        workflow_run=run, name="final_single_point_energy", unit="Eh"
    ).exists()
    assert ComputedProperty.objects.filter(
        workflow_run=run, name="final_single_point_energy_eV", unit="eV"
    ).exists()
    assert ComputedProperty.objects.filter(
        workflow_run=run, name="final_force_max_abs", unit="eV/A"
    ).exists()
    assert ComputedProperty.objects.filter(
        workflow_run=run, name="final_loewdin_charges", unit="e"
    ).exists()
    assert ComputedProperty.objects.filter(
        workflow_run=run, name="final_charges", unit="e"
    ).exists()
    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_start_energy_eV", unit="eV"
        ).value_float
        == pytest.approx(-100.0)
    )
    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_activation_barrier_eV", unit="eV"
        ).value_float
        == pytest.approx(0.5)
    )
    assert (
        ComputedProperty.objects.get(
            workflow_run=run,
            name="final_structure_formula",
            unit="",
        ).value_text
        == "CO"
    )


@pytest.mark.django_db
def test_ingest_single_run_reads_nwchem_flat_summary_and_artifacts(tmp_path: Path):
    run_dir = tmp_path / "nwchem_summary"
    run_dir.mkdir(parents=True)
    (run_dir / "nwchem.nwo").write_text("Total DFT energy = -1.25\n", encoding="utf-8")
    (run_dir / "nwchem.nwi").write_text("start nwchem\n", encoding="utf-8")
    (run_dir / "nwchem.db").write_text("db", encoding="utf-8")
    (run_dir / "nwchem.movecs").write_text("movecs", encoding="utf-8")
    (run_dir / "nwchem.molden").write_text("[Molden Format]\n", encoding="utf-8")
    (run_dir / "nwchem_tddft.nwo").write_text("TDDFT output\n", encoding="utf-8")
    (run_dir / "nwchem_property.nwo").write_text("Property output\n", encoding="utf-8")
    (run_dir / "nwchem_final.extxyz").write_text(
        "2\nProperties=species:S:1:pos:R:3\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n",
        encoding="utf-8",
    )
    summary = {
        "workflow": "nwchem_single_optimisation",
        "status": "completed",
        "final_energy_eV": -34.014232807485,
        "final_energy_hartree": -1.25,
        "output": "nwchem.nwo",
        "tddft_output": "nwchem_tddft.nwo",
        "tddft": {
            "wavenumbers_cm": [25000.0, 30000.0],
            "wavelengths_nm": [400.0, 333.333333],
            "oscillator_strengths": [0.1, 0.4],
        },
    }
    (run_dir / "reaction_hub_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    run = ingest_single_run(run_dir)

    assert ComputedProperty.objects.get(
        workflow_run=run,
        name="final_single_point_energy",
        unit="Eh",
    ).value_float == pytest.approx(-1.25)
    assert ComputedProperty.objects.get(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).value_float == pytest.approx(-34.014232807485)
    assert ComputedProperty.objects.get(
        workflow_run=run,
        name="final_tddft_strongest_wavelength_nm",
        unit="nm",
    ).value_float == pytest.approx(333.333333)
    transitions = ComputedProperty.objects.get(
        workflow_run=run,
        name="final_tddft_transitions",
        unit="",
    )
    assert transitions.metadata["transitions"][1]["oscillator_strength"] == pytest.approx(0.4)
    assert Artifact.objects.filter(
        workflow_run=run,
        path="nwchem.nwo",
        label="nwchem_out:nwchem",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="nwchem.nwi",
        label="nwchem_input:nwchem",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="nwchem.db",
        label="nwchem_db:nwchem.db",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="nwchem.movecs",
        label="nwchem_movecs:nwchem.movecs",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="nwchem.molden",
        label="molden:nwchem.molden",
    ).exists()
    spectrum_csv = run_dir / "DFT" / "final_tddft_spectrum" / "final_tddft_spectrum.csv"
    assert spectrum_csv.exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="DFT/final_tddft_spectrum/final_tddft_spectrum.csv",
        label="spectrum:final_tddft_spectrum.csv",
    ).exists()


@pytest.mark.django_db
def test_ingest_single_run_reads_scan_profile_from_scan_endpoints_db(tmp_path: Path):
    run_dir = tmp_path / "row_scan_profile"
    shared = run_dir / "TDDFT_active_learning_shared"
    shared.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")

    db_path = shared / "scan_endpoints.db"
    db = ase_connect(str(db_path))
    for iteration, energy, bond_length in (
        (1, -200.0, 1.10),
        (2, -199.2, 1.20),
        (3, -199.6, 1.30),
    ):
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, bond_length)]),
            data={"scan_bond_length": float(bond_length)},
            iteration=int(iteration),
            total_energy_eV=float(energy),
        )

    run = ingest_single_run(run_dir)

    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_start_energy_eV", unit="eV"
        ).value_float
        == pytest.approx(-200.0)
    )
    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_peak_energy_eV", unit="eV"
        ).value_float
        == pytest.approx(-199.2)
    )
    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_activation_barrier_eV", unit="eV"
        ).value_float
        == pytest.approx(0.8)
    )


@pytest.mark.django_db
def test_ingest_single_run_reads_scan_profile_from_nested_scan_endpoints_db(tmp_path: Path):
    run_dir = tmp_path / "row_scan_profile_nested"
    nested = run_dir / "12572882" / "TDDFT_active_learning_shared"
    nested.mkdir(parents=True)
    (run_dir / "12572882" / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")

    db_path = nested / "scan_endpoints.db"
    db = ase_connect(str(db_path))
    for iteration, energy, bond_length in (
        (1, -300.0, 1.05),
        (2, -299.1, 1.15),
        (3, -299.4, 1.25),
    ):
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, bond_length)]),
            data={"scan_bond_length": float(bond_length)},
            iteration=int(iteration),
            total_energy_eV=float(energy),
        )

    run = ingest_single_run(run_dir)

    assert Artifact.objects.filter(
        workflow_run=run,
        path="12572882/TDDFT_active_learning_shared/scan_endpoints.db",
        kind="database",
        label="scan_endpoints_db",
    ).exists()
    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_start_energy_eV", unit="eV"
        ).value_float
        == pytest.approx(-300.0)
    )
    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_peak_energy_eV", unit="eV"
        ).value_float
        == pytest.approx(-299.1)
    )
    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_activation_barrier_eV", unit="eV"
        ).value_float
        == pytest.approx(0.9)
    )


@pytest.mark.django_db
def test_ingest_single_run_prunes_duplicate_scan_endpoints_db_artifacts(tmp_path: Path):
    run_dir = tmp_path / "row_scan_profile_duplicate_prune"
    root_shared = run_dir / "TDDFT_active_learning_shared"
    nested_shared = run_dir / "12572882" / "TDDFT_active_learning_shared"
    root_shared.mkdir(parents=True)
    nested_shared.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")

    old_db_path = root_shared / "scan_endpoints.db"
    with ase_connect(str(old_db_path)) as db:
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.05)]),
            data={"scan_bond_length": 1.05},
            iteration=1,
            total_energy_eV=-100.0,
        )
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.15)]),
            data={"scan_bond_length": 1.15},
            iteration=2,
            total_energy_eV=-99.6,
        )
    os.utime(old_db_path, (1, 1))

    new_db_path = nested_shared / "scan_endpoints.db"
    with ase_connect(str(new_db_path)) as db:
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.10)]),
            data={"scan_bond_length": 1.10},
            iteration=1,
            total_energy_eV=-200.0,
        )
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.20)]),
            data={"scan_bond_length": 1.20},
            iteration=2,
            total_energy_eV=-199.5,
        )
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.30)]),
            data={"scan_bond_length": 1.30},
            iteration=3,
            total_energy_eV=-199.7,
        )

    run = ingest_single_run(run_dir)

    scan_db_artifacts = Artifact.objects.filter(
        workflow_run=run,
        kind=ArtifactKind.DATABASE,
        path__iendswith="scan_endpoints.db",
    ).order_by("pk")
    assert scan_db_artifacts.count() == 1
    kept = scan_db_artifacts.first()
    assert kept is not None
    assert kept.path == "12572882/TDDFT_active_learning_shared/scan_endpoints.db"

    profile_prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="bond_scan_energy_profile",
        unit="eV",
    )
    assert profile_prop.source_artifact_id == kept.pk
    assert (
        ComputedProperty.objects.get(
            workflow_run=run, name="bond_scan_start_energy_eV", unit="eV"
        ).value_float
        == pytest.approx(-200.0)
    )


@pytest.mark.django_db
def test_ingest_single_run_creates_bond_scan_mep_xyz_and_render_artifacts(
    tmp_path: Path,
    monkeypatch,
    settings,
):
    run_dir = tmp_path / "row_scan_mep"
    shared = run_dir / "TDDFT_active_learning_shared"
    shared.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")

    db_path = shared / "scan_endpoints.db"
    db = ase_connect(str(db_path))
    for iteration, energy, bond_length in (
        (1, -200.0, 1.10),
        (2, -199.2, 1.20),
        (3, -199.6, 1.30),
    ):
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, bond_length)]),
            data={"scan_bond_length": float(bond_length)},
            iteration=int(iteration),
            total_energy_eV=float(energy),
        )

    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"
    settings.REACTION_HUB_XYZRENDER_GIF_ROT = "xy"
    settings.REACTION_HUB_XYZRENDER_ROT_FRAMES = 200

    def _fake_xyzrender_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text
        assert "--gif-trj" in command
        assert "-go" in command
        assert "--gif-rot" not in command
        assert "--rot-frames" not in command
        assert "avogadro" not in str(env.get("PATH", "")).lower()
        out_dir = Path(str(cwd))
        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )
        gif_name = command[command.index("-go") + 1]
        (out_dir / gif_name).write_bytes(b"GIF89a")

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("ingest.services.subprocess.run", _fake_xyzrender_run)

    run = ingest_single_run(run_dir)

    mep_xyz_path = run_dir / "results" / "scan" / "bond_scan_mep.xyz"
    assert mep_xyz_path.exists()
    ping_pong_frames = ase_read(str(mep_xyz_path), index=":")
    assert len(ping_pong_frames) == 4
    assert [frame.get_distance(0, 1) for frame in ping_pong_frames] == pytest.approx(
        [1.10, 1.20, 1.30, 1.20]
    )
    mep_artifact = Artifact.objects.get(
        workflow_run=run,
        path="results/scan/bond_scan_mep.xyz",
    )
    assert mep_artifact.kind == "trajectory"
    assert mep_artifact.label == "trajectory:bond_scan_mep.xyz"
    assert Artifact.objects.filter(
        workflow_run=run,
        path="results/scan/bond_scan_mep.svg",
        kind="image",
        label="render:bond_scan_mep.svg",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="results/scan/bond_scan_mep.gif",
        kind="image",
        label="render:bond_scan_mep.gif",
    ).exists()


@pytest.mark.django_db
def test_ingest_single_run_does_not_rerender_existing_bond_scan_mep_gif(
    tmp_path: Path,
    monkeypatch,
    settings,
):
    run_dir = tmp_path / "row_scan_mep_existing_gif"
    shared = run_dir / "TDDFT_active_learning_shared"
    shared.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")

    db_path = shared / "scan_endpoints.db"
    db = ase_connect(str(db_path))
    for iteration, energy, bond_length in (
        (1, -200.0, 1.10),
        (2, -199.2, 1.20),
    ):
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, bond_length)]),
            data={"scan_bond_length": float(bond_length)},
            iteration=int(iteration),
            total_energy_eV=float(energy),
        )

    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"
    settings.REACTION_HUB_XYZRENDER_GIF_ROT = "xy"
    settings.REACTION_HUB_XYZRENDER_ROT_FRAMES = 200

    render_commands: list[list[str]] = []

    def _fake_xyzrender_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text, env
        render_commands.append(list(command))
        out_dir = Path(str(cwd))
        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )
        if "-go" in command:
            gif_name = command[command.index("-go") + 1]
            (out_dir / gif_name).write_bytes(b"GIF89a")

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("ingest.services.subprocess.run", _fake_xyzrender_run)

    run = ingest_single_run(run_dir)
    ingest_single_run(run_dir)

    assert len(render_commands) == 2
    first_command, second_command = render_commands
    assert "--gif-trj" in first_command
    assert "-go" in first_command
    assert "--gif-trj" not in second_command
    assert "-go" not in second_command

    svg_artifact = Artifact.objects.get(
        workflow_run=run,
        path="results/scan/bond_scan_mep.svg",
        kind="image",
        label="render:bond_scan_mep.svg",
    )
    gif_artifact = Artifact.objects.get(
        workflow_run=run,
        path="results/scan/bond_scan_mep.gif",
        kind="image",
        label="render:bond_scan_mep.gif",
    )
    assert svg_artifact.metadata.get("render_output") == "svg"
    assert "gif_mode" not in svg_artifact.metadata
    assert gif_artifact.metadata.get("render_output") == "gif"
    assert gif_artifact.metadata.get("gif_mode") == "trajectory"


@pytest.mark.django_db
def test_ingest_single_run_keeps_svg_when_cairo_missing(
    tmp_path: Path,
    monkeypatch,
    settings,
):
    run_dir = tmp_path / "row_scan_mep_cairo"
    shared = run_dir / "TDDFT_active_learning_shared"
    shared.mkdir(parents=True)
    (run_dir / "iteration_complete.txt").write_text("mode=tddft", encoding="utf-8")

    db_path = shared / "scan_endpoints.db"
    db = ase_connect(str(db_path))
    for iteration, energy, bond_length in (
        (1, -200.0, 1.10),
        (2, -199.2, 1.20),
    ):
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, bond_length)]),
            data={"scan_bond_length": float(bond_length)},
            iteration=int(iteration),
            total_energy_eV=float(energy),
        )

    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"
    settings.REACTION_HUB_XYZRENDER_GIF_ROT = "xy"
    settings.REACTION_HUB_XYZRENDER_ROT_FRAMES = 200

    def _fake_xyzrender_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text, env
        assert "--gif-trj" in command
        out_dir = Path(str(cwd))
        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )

        class _Completed:
            returncode = 1
            stdout = ""
            stderr = 'cannot load library "libcairo-2.dll"'

        return _Completed()

    monkeypatch.setattr("ingest.services.subprocess.run", _fake_xyzrender_run)

    run = ingest_single_run(run_dir)

    assert Artifact.objects.filter(
        workflow_run=run,
        path="results/scan/bond_scan_mep.svg",
        kind="image",
        label="render:bond_scan_mep.svg",
    ).exists()
    assert not Artifact.objects.filter(
        workflow_run=run,
        path="results/scan/bond_scan_mep.gif",
        kind="image",
        label="render:bond_scan_mep.gif",
    ).exists()


@pytest.mark.django_db
def test_ingest_remote_downloads_evaluated_traj_to_local_cache(tmp_path: Path, settings):
    profile = CalculationSettingsProfile.objects.create(name="remote_cache_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_cache_wf", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_cache_run",
        status="completed",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/remote_cache_run",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-cache-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="321",
        job_name="remote_cache_job",
        status="completed",
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_cache_run"
    remote_traj_1 = f"{run_root}/TDDFT/opt_iter_1/active_learning/evaluated.traj"
    remote_traj_2 = f"{run_root}/DFT/opt_iter_1/active_learning/evaluated.traj"
    remote_orca = f"{run_root}/TDDFT/opt_iter_1/orca.out"
    traj_bytes_1 = b"\x00\x01traj_payload_1\x9d"
    traj_bytes_2 = b"\x00\x02traj_payload_2"
    traj_b64_1 = base64.b64encode(traj_bytes_1).decode("ascii")
    traj_b64_2 = base64.b64encode(traj_bytes_2).decode("ascii")

    class _FakeRemoteClient:
        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "active_learning/evaluated.traj" in command:
                return f"{remote_traj_2}\n{remote_traj_1}\n"
            if "base64 < " in command and remote_traj_1 in command:
                return f"__REACTION_HUB_B64_START__\n{traj_b64_1}\n__REACTION_HUB_B64_END__\n"
            if "base64 < " in command and remote_traj_2 in command:
                return f"__REACTION_HUB_B64_START__\n{traj_b64_2}\n__REACTION_HUB_B64_END__\n"
            if " -name " in command and "orca.out" in command:
                return remote_orca
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteClient())
    assert fetched is True

    traj_artifact = Artifact.objects.get(
        workflow_run=run,
        path="TDDFT/opt_iter_1/active_learning/evaluated.traj",
    )
    assert traj_artifact.kind == "trajectory"
    assert traj_artifact.metadata.get("downloaded_from_hpc") is True
    assert traj_artifact.metadata.get("remote_path") == remote_traj_1
    local_cache_path = Path(str(traj_artifact.metadata.get("local_cache_path")))
    assert local_cache_path.exists()
    assert local_cache_path.read_bytes() == traj_bytes_1

    log_artifact = Artifact.objects.get(
        workflow_run=run,
        path="TDDFT/opt_iter_1/orca.out",
    )
    assert log_artifact.metadata.get("remote_only") is True


@pytest.mark.django_db
def test_ingest_remote_downloads_evaluated_traj_dedupes_mixed_jobid_layouts(tmp_path: Path, settings):
    profile = CalculationSettingsProfile.objects.create(name="remote_cache_dedupe_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_cache_dedupe_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_cache_dedupe_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_cache_dedupe_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-cache-dedupe-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="321",
        job_name="remote_cache_dedupe_job",
        status="completed",
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    nested_remote_traj = f"{run_root}/321/TDDFT/opt_iter_1/active_learning/evaluated.traj"
    root_remote_traj = f"{run_root}/TDDFT/opt_iter_1/active_learning/evaluated.traj"
    remote_orca = f"{run_root}/TDDFT/opt_iter_1/orca.out"
    traj_bytes = b"traj_payload"
    traj_b64 = base64.b64encode(traj_bytes).decode("ascii")

    class _FakeRemoteClient:
        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "active_learning/evaluated.traj" in command:
                return f"{nested_remote_traj}\n{root_remote_traj}\n"
            if "base64 < " in command and nested_remote_traj in command:
                return f"__REACTION_HUB_B64_START__\n{traj_b64}\n__REACTION_HUB_B64_END__\n"
            if "base64 < " in command and root_remote_traj in command:
                raise RuntimeError("duplicate trajectory path should be skipped after semantic dedupe")
            if " -name " in command and "orca.out" in command:
                return remote_orca
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteClient())
    assert fetched is True

    artifacts = list(
        Artifact.objects.filter(
            workflow_run=run,
            label="trajectory:evaluated.traj",
        ).order_by("path")
    )
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact.metadata.get("source_scheduler_job_id") == "321"
    assert artifact.metadata.get("remote_path") == nested_remote_traj


@pytest.mark.django_db
def test_ingest_remote_single_optimisation_keeps_only_latest_evaluated_traj_for_source_job(
    tmp_path: Path,
    settings,
):
    profile = CalculationSettingsProfile.objects.create(name="remote_single_opt_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_single_opt_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_single_opt_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_single_opt_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"submission": {"run_mode": "single_optimisation"}},
    )
    system = HPCSystem.objects.create(
        name="remote-single-opt-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="321",
        job_name="remote_single_opt_job",
        status=JobStatus.COMPLETED,
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="legacy/evaluated_old_a.traj",
        metadata={
            "source": "remote_download",
            "source_scheduler_job_id": "321",
            "remote_path": f"{run_root}/legacy/evaluated_old_a.traj",
        },
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="legacy/evaluated_old_b.traj",
        metadata={
            "source": "remote_download",
            "source_scheduler_job_id": "321",
            "remote_path": f"{run_root}/legacy/evaluated_old_b.traj",
        },
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="legacy/evaluated_old_other_source_a.traj",
        metadata={
            "source": "remote_download",
            "source_scheduler_job_id": "222",
            "remote_path": f"{run_root}/legacy/evaluated_old_other_source_a.traj",
        },
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="legacy/evaluated_old_other_source_b.traj",
        metadata={
            "source": "remote_download",
            "source_scheduler_job_id": "222",
            "remote_path": f"{run_root}/legacy/evaluated_old_other_source_b.traj",
        },
    )

    remote_traj_root = f"{run_root}/DFT/opt_iter_1/active_learning/evaluated.traj"
    remote_traj_job_dir = f"{run_root}/321/DFT/opt_iter_1/active_learning/evaluated.traj"
    remote_traj_backup = f"{run_root}/restart_backup_20260225T152000/DFT/opt_iter_1/active_learning/evaluated.traj"
    remote_orca = f"{run_root}/DFT/opt_iter_1/orca.out"
    traj_bytes = b"single_opt_final_traj"
    traj_b64 = base64.b64encode(traj_bytes).decode("ascii")

    class _FakeRemoteClient:
        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "active_learning/evaluated.traj" in command:
                return f"{remote_traj_backup}\n{remote_traj_job_dir}\n{remote_traj_root}\n"
            if "base64 < " in command and remote_traj_root in command:
                return f"__REACTION_HUB_B64_START__\n{traj_b64}\n__REACTION_HUB_B64_END__\n"
            if "base64 < " in command and remote_traj_job_dir in command:
                raise RuntimeError("single optimisation ingest should keep only one selected evaluated.traj")
            if "base64 < " in command and remote_traj_backup in command:
                raise RuntimeError("single optimisation ingest should ignore restart backup evaluated.traj")
            if " -path " in command and "scan_endpoints.db" in command:
                return ""
            if " -name " in command and "reaction_hub_summary.json" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return remote_orca
            return ""

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteClient())
    assert fetched is True

    artifacts = list(
        Artifact.objects.filter(
            workflow_run=run,
            label="trajectory:evaluated.traj",
        ).order_by("path")
    )
    assert len(artifacts) == 2
    by_source = {
        str(item.metadata.get("source_scheduler_job_id") or ""): item
        for item in artifacts
    }
    assert set(by_source) == {"321", "222"}
    current = by_source["321"]
    assert current.path == "DFT/opt_iter_1/active_learning/evaluated.traj"
    assert current.metadata.get("remote_path") == remote_traj_root


@pytest.mark.django_db
def test_ingest_remote_single_optimisation_records_latest_evaluated_traj_fmax(
    tmp_path: Path,
    settings,
):
    profile = CalculationSettingsProfile.objects.create(name="remote_single_opt_fmax_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_single_opt_fmax_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_single_opt_fmax_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_single_opt_fmax_run",
        status="failed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"submission": {"run_mode": "single_optimisation"}},
    )
    system = HPCSystem.objects.create(
        name="remote-single-opt-fmax-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="321",
        job_name="remote_single_opt_fmax_job",
        status=JobStatus.FAILED,
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    frame_1 = Atoms("H2", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.74)])
    frame_1.calc = SinglePointCalculator(
        frame_1,
        energy=-1.0,
        forces=[[0.10, 0.0, 0.0], [-0.10, 0.0, 0.0]],
    )
    frame_2 = Atoms("H2", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.78)])
    frame_2.calc = SinglePointCalculator(
        frame_2,
        energy=-1.1,
        forces=[[0.35, 0.0, 0.0], [-0.35, 0.0, 0.0]],
    )
    source_traj = tmp_path / "remote_single_opt_evaluated_source.traj"
    ase_write(str(source_traj), [frame_1, frame_2], format="traj")
    traj_b64 = base64.b64encode(source_traj.read_bytes()).decode("ascii")

    remote_traj_path = f"{run_root}/DFT/opt_iter_1/active_learning/evaluated.traj"

    class _FakeRemoteClient:
        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "active_learning/evaluated.traj" in command:
                return f"{remote_traj_path}\n"
            if "base64 < " in command and remote_traj_path in command:
                return f"__REACTION_HUB_B64_START__\n{traj_b64}\n__REACTION_HUB_B64_END__\n"
            if " -path " in command and "scan_endpoints.db" in command:
                return ""
            if " -name " in command and "reaction_hub_summary.json" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteClient())
    assert fetched is True

    prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="latest_evaluated_traj_fmax",
        unit="eV/A",
    )
    assert prop.value_float == pytest.approx(0.35)
    assert prop.source_artifact is not None
    assert prop.source_artifact.label == "trajectory:evaluated.traj"
    assert prop.source_artifact.path == "DFT/opt_iter_1/active_learning/evaluated.traj"
    assert prop.metadata.get("source") == "evaluated_traj"
    assert prop.metadata.get("trajectory_frame_index") == 1
    assert prop.metadata.get("trajectory_frame_count") == 2
    assert prop.metadata.get("trajectory_frame_selection") == "latest_frame"


@pytest.mark.django_db
def test_ingest_remote_summary_populates_computed_properties(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-summary-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="654",
        job_name="remote_summary_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    summary = {
        "generated_at_utc": "2026-02-20T16:00:00+00:00",
        "final": {
            "formula": "CO",
            "atom_count": 2,
            "symbols": ["C", "O"],
            "positions_A": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]],
            "energy_hartree": -113.0,
            "energy_eV": -3074.886647796644,
            "forces_eV_A": [[0.0, 0.0, 0.01], [0.0, 0.0, -0.01]],
            "charges": [-0.2, 0.2],
            "charge_method": "loewdin",
            "orca_out": f"{run_root}/TDDFT/opt_iter_1/orca.out",
            "opt_dir": f"{run_root}/TDDFT/opt_iter_1",
            "final_xyz_paths": [f"{run_root}/reaction_hub_final.xyz"],
            "cluster_fragment": {
                "energy_eV": -100.5,
                "energy_hartree": -3.6933245112,
                "orca_out": f"{run_root}/TDDFT/opt_iter_1/cluster_single_point_final/orca.out",
                "directory": f"{run_root}/TDDFT/opt_iter_1/cluster_single_point_final",
                "traj_paths": [
                    f"{run_root}/TDDFT/opt_iter_1/cluster_single_point_final/cluster_final_single_point.traj"
                ],
            },
        },
        "scan": {
            "points": [
                {"iteration": 1, "energy_eV": -100.0, "bond_length_A": 1.10},
                {"iteration": 2, "energy_eV": -99.6, "bond_length_A": 1.15},
            ],
            "start_energy_eV": -100.0,
            "peak_energy_eV": -99.6,
            "activation_barrier_eV": 0.4,
        },
    }

    class _FakeRemoteSummaryClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteSummaryClient())
    assert fetched is True

    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).exists()
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_cluster_single_point_energy_eV",
        unit="eV",
    ).exists()
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_structure_formula",
        unit="",
    ).exists()
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_loewdin_charges",
        unit="e",
    ).exists()
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_charges",
        unit="e",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="TDDFT/opt_iter_1/cluster_single_point_final/cluster_final_single_point.traj",
        kind=ArtifactKind.TRAJECTORY,
    ).exists()


@pytest.mark.django_db
def test_ingest_remote_summary_ignores_target_wavelength_for_non_tddft_run():
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_non_tddft_target_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_non_tddft_target_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_non_tddft_target_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_non_tddft_target_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"submission": {"run_mode": "single_optimisation"}},
    )
    system = HPCSystem.objects.create(
        name="remote-summary-non-tddft-target-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="754",
        job_name="remote_summary_non_tddft_target_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    summary = {
        "generated_at_utc": "2026-02-20T16:00:00+00:00",
        "settings": {"target_wavelength_nm": 520.0},
    }
    ComputedProperty.objects.create(
        workflow_run=run,
        name="target_wavelength_nm",
        unit="nm",
        value_float=520.0,
        metadata={"source": "seed"},
    )

    class _FakeRemoteSummaryNonTddftTargetClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteSummaryNonTddftTargetClient())
    assert fetched is True

    assert not ComputedProperty.objects.filter(
        workflow_run=run,
        name="target_wavelength_nm",
        unit="nm",
    ).exists()


@pytest.mark.django_db
def test_ingest_remote_summary_reads_nwchem_flat_summary():
    profile = CalculationSettingsProfile.objects.create(name="remote_nwchem_summary_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_nwchem_summary_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_nwchem_summary_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_nwchem_summary_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"submission": {"run_mode": "single_optimisation"}},
    )
    system = HPCSystem.objects.create(
        name="remote-nwchem-summary-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="954",
        job_name="remote_nwchem_summary_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    summary = {
        "workflow": "nwchem_single_optimisation",
        "status": "completed",
        "final_energy_eV": -54.422772491976,
        "final_energy_hartree": -2.0,
        "tddft": {
            "wavelengths_nm": [500.0, 450.0],
            "oscillator_strengths": [0.2, 0.7],
        },
    }

    class _FakeRemoteNwchemSummaryClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteNwchemSummaryClient())
    assert fetched is True
    assert ComputedProperty.objects.get(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).value_float == pytest.approx(-54.422772491976)
    assert ComputedProperty.objects.get(
        workflow_run=run,
        name="final_tddft_strongest_wavelength_nm",
        unit="nm",
    ).value_float == pytest.approx(450.0)


@pytest.mark.django_db
def test_ingest_remote_summary_downloads_plot_images_to_local_cache(tmp_path: Path, settings):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_plot_cache_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_plot_cache_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_plot_cache_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_plot_cache_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-summary-plot-cache-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="777",
        job_name="remote_summary_plot_cache_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    remote_plot_path = f"{run_root}/DFT/scan_plots/stacked_uvvis.png"
    png_bytes = b"fake_png"
    encoded_png = base64.b64encode(png_bytes).decode("ascii")
    summary = {
        "generated_at_utc": "2026-02-26T09:00:00+00:00",
        "plots": {
            "stacked_uvvis": remote_plot_path,
        },
    }

    class _FakeRemoteSummaryPlotClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if "base64 <" in command and remote_plot_path in command:
                return (
                    "__REACTION_HUB_B64_START__\n"
                    f"{encoded_png}\n"
                    "__REACTION_HUB_B64_END__\n"
                )
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteSummaryPlotClient())
    assert fetched is True

    artifact = Artifact.objects.get(
        workflow_run=run,
        path="DFT/scan_plots/stacked_uvvis.png",
        label="plot:stacked_uvvis",
    )
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    assert metadata.get("source") == "remote_summary"
    assert metadata.get("remote_path") == remote_plot_path
    assert metadata.get("downloaded_from_hpc") is True
    assert metadata.get("remote_only") is not True
    local_cache_path = Path(str(metadata.get("local_cache_path") or ""))
    assert local_cache_path.exists() and local_cache_path.is_file()
    assert local_cache_path.read_bytes() == png_bytes


@pytest.mark.django_db
def test_ingest_remote_summary_retains_declared_ct_artifacts(tmp_path: Path, settings):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_ct_artifacts_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_ct_artifacts_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/ct_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-summary-ct-artifacts-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="778",
        job_name="ct_artifacts_job",
        status="completed",
    )
    summary_path = f"{run_root}/reaction_hub_summary.json"
    ct_path = f"{run_root}/CT_multiple/ET_1to2.txt"
    ct_bytes = b"energy,intensity\n2.0,0.5\n"
    encoded_ct = base64.b64encode(ct_bytes).decode("ascii")
    summary = {
        "artifacts": [
            "reaction_hub_summary.json",
            "CT_multiple/ET_1to2.txt",
            "orca.molden.input",
        ]
    }

    class _FakeRemoteSummaryCtArtifactsClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            return json.dumps(summary)

        def exec_checked(self, command: str) -> str:
            if "base64 <" in command and ct_path in command:
                return (
                    "__REACTION_HUB_B64_START__\n"
                    f"{encoded_ct}\n"
                    "__REACTION_HUB_B64_END__\n"
                )
            return ""

    assert _ingest_remote_summary_for_job(job, _FakeRemoteSummaryCtArtifactsClient()) is True

    ct_artifact = Artifact.objects.get(workflow_run=run, path="CT_multiple/ET_1to2.txt")
    assert ct_artifact.label == "ct_data:ET_1to2.txt"
    assert Path(ct_artifact.metadata["local_cache_path"]).read_bytes() == ct_bytes
    molden_artifact = Artifact.objects.get(workflow_run=run, path="orca.molden.input")
    assert molden_artifact.label == "molden:orca.molden.input"
    assert molden_artifact.metadata["remote_only"] is True
    summary_artifact = Artifact.objects.get(workflow_run=run, label="reaction_hub_summary")
    assert json.loads(Path(summary_artifact.metadata["local_cache_path"]).read_text())["artifacts"]


@pytest.mark.django_db
def test_ingest_remote_summary_refreshes_existing_plot_cache_when_not_missing_only(
    tmp_path: Path, settings
):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_plot_refresh_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_plot_refresh_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_plot_refresh_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_plot_refresh_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-summary-plot-refresh-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="780",
        job_name="remote_summary_plot_refresh_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    remote_plot_path = f"{run_root}/DFT/scan_plots/stacked_uvvis.png"
    old_png_bytes = b"old_cached_png"
    new_png_bytes = b"new_remote_png"
    encoded_png = base64.b64encode(new_png_bytes).decode("ascii")
    summary = {
        "generated_at_utc": "2026-02-26T09:30:00+00:00",
        "plots": {
            "stacked_uvvis": remote_plot_path,
        },
    }

    existing_cache_path = tmp_path / "existing_plot_cache.png"
    existing_cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_cache_path.write_bytes(old_png_bytes)
    Artifact.objects.create(
        workflow_run=run,
        path="DFT/scan_plots/stacked_uvvis.png",
        kind=ArtifactKind.IMAGE,
        label="plot:stacked_uvvis",
        metadata={
            "source": "remote_summary",
            "remote_path": remote_plot_path,
            "local_cache_path": str(existing_cache_path),
            "downloaded_from_hpc": True,
        },
    )

    remote_base64_calls: list[str] = []

    class _FakeRemoteSummaryPlotRefreshClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if "base64 <" in command and remote_plot_path in command:
                remote_base64_calls.append(command)
                return (
                    "__REACTION_HUB_B64_START__\n"
                    f"{encoded_png}\n"
                    "__REACTION_HUB_B64_END__\n"
                )
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(
        job,
        _FakeRemoteSummaryPlotRefreshClient(),
        download_missing_only=False,
    )
    assert fetched is True
    assert remote_base64_calls

    artifact = Artifact.objects.get(
        workflow_run=run,
        path="DFT/scan_plots/stacked_uvvis.png",
        label="plot:stacked_uvvis",
    )
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    assert Path(str(metadata.get("local_cache_path") or "")) == existing_cache_path
    assert existing_cache_path.read_bytes() == new_png_bytes


@pytest.mark.django_db
def test_ingest_remote_summary_skips_per_iteration_uvvis_images(tmp_path: Path, settings):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_skip_uvvis_iter_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_skip_uvvis_iter_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_skip_uvvis_iter_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_skip_uvvis_iter_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-summary-skip-uvvis-iter-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="778",
        job_name="remote_summary_skip_uvvis_iter_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    uvvis_plot_path = f"{run_root}/DFT/scan_plots/uvvis_iter_008.png"
    stacked_plot_path = f"{run_root}/DFT/scan_plots/stacked_uvvis.png"
    png_bytes = b"stacked_png"
    encoded_png = base64.b64encode(png_bytes).decode("ascii")
    summary = {
        "generated_at_utc": "2026-02-26T10:00:00+00:00",
        "plots": {
            "images": [uvvis_plot_path, stacked_plot_path],
            "stacked_uvvis": stacked_plot_path,
        },
    }

    class _FakeRemoteSummarySkipUvvisClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if "base64 <" in command and stacked_plot_path in command:
                return (
                    "__REACTION_HUB_B64_START__\n"
                    f"{encoded_png}\n"
                    "__REACTION_HUB_B64_END__\n"
                )
            if "base64 <" in command and uvvis_plot_path in command:
                raise RuntimeError("uvvis_iter plot should be skipped during ingest")
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteSummarySkipUvvisClient())
    assert fetched is True

    assert Artifact.objects.filter(
        workflow_run=run,
        path="DFT/scan_plots/stacked_uvvis.png",
        label="plot:stacked_uvvis",
    ).exists()
    assert not Artifact.objects.filter(
        workflow_run=run,
        path="DFT/scan_plots/uvvis_iter_008.png",
    ).exists()


@pytest.mark.django_db
def test_ingest_remote_summary_downloads_per_iteration_ct_images(tmp_path: Path, settings):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_ct_iter_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_ct_iter_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_ct_iter_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_ct_iter_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-summary-ct-iter-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="779",
        job_name="remote_summary_ct_iter_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    ct_names = [
        "ct_spectrum_iter_000.png",
        "ct_spectrum_iter_000_overall.png",
        "ct_spectrum_iter_000_transfer_only.png",
    ]
    remote_ct_paths = [f"{run_root}/DFT/scan_plots/{name}" for name in ct_names]
    ct_payloads = {
        remote_path: f"{Path(remote_path).name}_png".encode("ascii")
        for remote_path in remote_ct_paths
    }
    summary = {
        "generated_at_utc": "2026-02-26T10:00:00+00:00",
        "plots": {
            "images": remote_ct_paths,
        },
    }

    class _FakeRemoteSummaryCtClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            if "base64 <" in command:
                for remote_path, png_bytes in ct_payloads.items():
                    if remote_path in command:
                        encoded_png = base64.b64encode(png_bytes).decode("ascii")
                        return (
                            "__REACTION_HUB_B64_START__\n"
                            f"{encoded_png}\n"
                            "__REACTION_HUB_B64_END__\n"
                        )
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteSummaryCtClient())
    assert fetched is True

    for name, remote_path in zip(ct_names, remote_ct_paths):
        artifact = Artifact.objects.get(
            workflow_run=run,
            path=f"DFT/scan_plots/{name}",
            label=f"plot:{name}",
            kind=ArtifactKind.IMAGE,
        )
        metadata = artifact.metadata
        assert metadata.get("remote_path") == remote_path
        assert metadata.get("downloaded_from_hpc") is True
        assert Path(str(metadata.get("local_cache_path"))).read_bytes() == ct_payloads[remote_path]


@pytest.mark.django_db
def test_ingest_remote_summary_downloads_dft_database_to_local_cache(tmp_path: Path, settings):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_dft_db_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_dft_db_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_dft_db_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_dft_db_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-summary-dft-db-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="778",
        job_name="remote_summary_dft_db_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    remote_db_path = f"{run_root}/scratch/coadsorption_final_for_dft.db"
    db_bytes = b"sqlite_db_bytes"
    encoded_db = base64.b64encode(db_bytes).decode("ascii")
    summary = {
        "generated_at_utc": "2026-02-26T10:00:00+00:00",
        "final": {
            "dft_db_paths": [
                "scratch/coadsorption_final_for_dft.db",
            ],
        },
    }

    class _FakeRemoteSummaryDftDbClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if "base64 <" in command and remote_db_path in command:
                return (
                    "__REACTION_HUB_B64_START__\n"
                    f"{encoded_db}\n"
                    "__REACTION_HUB_B64_END__\n"
                )
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteSummaryDftDbClient())
    assert fetched is True

    artifact = Artifact.objects.get(
        workflow_run=run,
        path="scratch/coadsorption_final_for_dft.db",
    )
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    assert artifact.kind == "database"
    assert metadata.get("source") == "remote_summary"
    assert metadata.get("remote_path") == remote_db_path
    assert metadata.get("downloaded_from_hpc") is True
    assert metadata.get("remote_only") is not True
    local_cache_path = Path(str(metadata.get("local_cache_path") or ""))
    assert local_cache_path.exists() and local_cache_path.is_file()
    assert local_cache_path.read_bytes() == db_bytes


@pytest.mark.django_db
def test_ingest_remote_summary_refreshes_existing_database_cache_when_not_missing_only(
    tmp_path: Path, settings
):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_dft_db_refresh_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_summary_dft_db_refresh_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_dft_db_refresh_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_dft_db_refresh_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-summary-dft-db-refresh-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="781",
        job_name="remote_summary_dft_db_refresh_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    remote_db_path = f"{run_root}/scratch/coadsorption_final_for_dft.db"
    old_db_bytes = b"old_db"
    new_db_bytes = b"new_db"
    encoded_db = base64.b64encode(new_db_bytes).decode("ascii")
    summary = {
        "generated_at_utc": "2026-02-26T10:15:00+00:00",
        "final": {
            "dft_db_paths": [
                "scratch/coadsorption_final_for_dft.db",
            ],
        },
    }

    existing_cache_path = tmp_path / "existing_database_cache.db"
    existing_cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing_cache_path.write_bytes(old_db_bytes)
    Artifact.objects.create(
        workflow_run=run,
        path="scratch/coadsorption_final_for_dft.db",
        kind=ArtifactKind.DATABASE,
        label="database:coadsorption_final_for_dft.db",
        metadata={
            "source": "remote_summary",
            "remote_path": remote_db_path,
            "local_cache_path": str(existing_cache_path),
            "downloaded_from_hpc": True,
        },
    )

    remote_base64_calls: list[str] = []

    class _FakeRemoteSummaryDftDbRefreshClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if "base64 <" in command and remote_db_path in command:
                remote_base64_calls.append(command)
                return (
                    "__REACTION_HUB_B64_START__\n"
                    f"{encoded_db}\n"
                    "__REACTION_HUB_B64_END__\n"
                )
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(
        job,
        _FakeRemoteSummaryDftDbRefreshClient(),
        download_missing_only=False,
    )
    assert fetched is True
    assert remote_base64_calls

    artifact = Artifact.objects.get(
        workflow_run=run,
        path="scratch/coadsorption_final_for_dft.db",
    )
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    assert Path(str(metadata.get("local_cache_path") or "")) == existing_cache_path
    assert existing_cache_path.read_bytes() == new_db_bytes


@pytest.mark.django_db
def test_ingest_remote_summary_for_coadsorbate_only_downloads_final_dft_db(tmp_path: Path, settings):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_coadsorbate_cache_policy_profile")
    workflow = WorkflowDefinition.objects.create(
        name="remote_summary_coadsorbate_cache_policy_wf",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_summary_coadsorbate_cache_policy_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_summary_coadsorbate_cache_policy_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"submission": {"run_mode": "coadsorbate_placement"}},
    )
    system = HPCSystem.objects.create(
        name="remote-summary-coadsorbate-cache-policy-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="779",
        job_name="remote_summary_coadsorbate_cache_policy_job",
        status="completed",
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    remote_final_db_path = f"{run_root}/scratch/coadsorption_final_for_dft.db"
    remote_candidates_db_path = f"{run_root}/scratch/coadsorption_candidates.db"
    remote_plot_path = f"{run_root}/DFT/scan_plots/stacked_uvvis.png"
    final_db_bytes = b"sqlite_final_db_bytes"
    encoded_final_db = base64.b64encode(final_db_bytes).decode("ascii")
    summary = {
        "generated_at_utc": "2026-02-26T10:30:00+00:00",
        "plots": {
            "stacked_uvvis": remote_plot_path,
        },
        "artifacts": {
            "db_paths": [
                "scratch/coadsorption_candidates.db",
                "scratch/coadsorption_final_for_dft.db",
            ],
        },
    }

    class _FakeRemoteSummaryCoadsorbateCachePolicyClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if "base64 <" in command and remote_final_db_path in command:
                return (
                    "__REACTION_HUB_B64_START__\n"
                    f"{encoded_final_db}\n"
                    "__REACTION_HUB_B64_END__\n"
                )
            if "base64 <" in command and (
                remote_candidates_db_path in command or remote_plot_path in command
            ):
                raise RuntimeError(f"unexpected remote byte download: {command}")
            if " -name " in command and "orca.out" in command:
                return ""
            raise RuntimeError(f"unexpected command: {command}")

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteSummaryCoadsorbateCachePolicyClient())
    assert fetched is True

    final_db_artifact = Artifact.objects.get(
        workflow_run=run,
        path="scratch/coadsorption_final_for_dft.db",
    )
    final_db_metadata = final_db_artifact.metadata if isinstance(final_db_artifact.metadata, dict) else {}
    assert final_db_metadata.get("downloaded_from_hpc") is True
    assert final_db_metadata.get("remote_only") is not True
    final_db_cache_path = Path(str(final_db_metadata.get("local_cache_path") or ""))
    assert final_db_cache_path.exists() and final_db_cache_path.is_file()
    assert final_db_cache_path.read_bytes() == final_db_bytes

    candidates_db_artifact = Artifact.objects.get(
        workflow_run=run,
        path="scratch/coadsorption_candidates.db",
    )
    candidates_db_metadata = (
        candidates_db_artifact.metadata if isinstance(candidates_db_artifact.metadata, dict) else {}
    )
    assert candidates_db_metadata.get("remote_only") is True
    assert candidates_db_metadata.get("downloaded_from_hpc") is not True
    assert not str(candidates_db_metadata.get("local_cache_path") or "").strip()

    plot_artifact = Artifact.objects.get(
        workflow_run=run,
        path="DFT/scan_plots/stacked_uvvis.png",
    )
    plot_metadata = plot_artifact.metadata if isinstance(plot_artifact.metadata, dict) else {}
    assert plot_metadata.get("remote_only") is True
    assert plot_metadata.get("downloaded_from_hpc") is not True
    assert not str(plot_metadata.get("local_cache_path") or "").strip()


@pytest.mark.django_db
def test_ingest_remote_downloads_scan_endpoints_db_and_stores_scan_profile(tmp_path: Path, settings):
    profile = CalculationSettingsProfile.objects.create(name="remote_scan_db_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_scan_db_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_scan_db_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_scan_db_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-scan-db-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="655",
        job_name="remote_scan_db_job",
        status=JobStatus.COMPLETED,
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    source_db_path = tmp_path / "scan_endpoints_source.db"
    with ase_connect(source_db_path) as db:
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.10)]),
            data={"scan_bond_length": 1.10},
            iteration=1,
            total_energy_eV=-200.0,
        )
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.20)]),
            data={"scan_bond_length": 1.20},
            iteration=2,
            total_energy_eV=-199.5,
        )
    db_bytes = source_db_path.read_bytes()
    db_b64 = base64.b64encode(db_bytes).decode("ascii")
    remote_scan_db = f"{run_root}/TDDFT_active_learning_shared/scan_endpoints.db"

    class _FakeRemoteScanDbClient:
        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "active_learning/evaluated.traj" in command:
                return ""
            if " -path " in command and "scan_endpoints.db" in command:
                return f"{remote_scan_db}\n"
            if "base64 < " in command and remote_scan_db in command:
                return f"__REACTION_HUB_B64_START__\n{db_b64}\n__REACTION_HUB_B64_END__\n"
            if " -name " in command and "reaction_hub_summary.json" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteScanDbClient())
    assert fetched is True

    scan_db_artifact = Artifact.objects.get(
        workflow_run=run,
        path="TDDFT_active_learning_shared/scan_endpoints.db",
    )
    assert scan_db_artifact.kind == "database"
    assert scan_db_artifact.metadata.get("downloaded_from_hpc") is True
    assert scan_db_artifact.metadata.get("remote_path") == remote_scan_db
    local_cache_path = Path(str(scan_db_artifact.metadata.get("local_cache_path")))
    assert local_cache_path.exists()
    assert local_cache_path.read_bytes() == db_bytes

    start_prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="bond_scan_start_energy_eV",
        unit="eV",
    )
    peak_prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="bond_scan_peak_energy_eV",
        unit="eV",
    )
    barrier_prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="bond_scan_activation_barrier_eV",
        unit="eV",
    )
    assert start_prop.value_float == pytest.approx(-200.0)
    assert peak_prop.value_float == pytest.approx(-199.5)
    assert barrier_prop.value_float == pytest.approx(0.5)


@pytest.mark.django_db
def test_ingest_remote_scan_endpoints_prefers_current_scheduler_job_profile_in_ties(tmp_path: Path, settings):
    profile = CalculationSettingsProfile.objects.create(name="remote_scan_db_tie_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_scan_db_tie_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_scan_db_tie_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_scan_db_tie_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"submission": {"run_mode": "tddft_bond_scan"}},
    )
    system = HPCSystem.objects.create(
        name="remote-scan-db-tie-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="900",
        job_name="remote_scan_db_tie_job",
        status=JobStatus.COMPLETED,
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    old_db_path = tmp_path / "scan_endpoints_old.db"
    with ase_connect(old_db_path) as db:
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.11)]),
            data={"scan_bond_length": 1.11},
            iteration=1,
            total_energy_eV=-210.0,
        )
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.21)]),
            data={"scan_bond_length": 1.21},
            iteration=2,
            total_energy_eV=-209.6,
        )

    new_db_path = tmp_path / "scan_endpoints_new.db"
    with ase_connect(new_db_path) as db:
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.11)]),
            data={"scan_bond_length": 1.11},
            iteration=1,
            total_energy_eV=-150.0,
        )
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.21)]),
            data={"scan_bond_length": 1.21},
            iteration=2,
            total_energy_eV=-149.6,
        )

    old_db_b64 = base64.b64encode(old_db_path.read_bytes()).decode("ascii")
    new_db_b64 = base64.b64encode(new_db_path.read_bytes()).decode("ascii")
    old_remote_scan_db = f"{run_root}/888/TDDFT_active_learning_shared/scan_endpoints.db"
    current_remote_scan_db = f"{run_root}/TDDFT_active_learning_shared/scan_endpoints.db"

    class _FakeRemoteScanDbTieClient:
        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "active_learning/evaluated.traj" in command:
                return ""
            if " -path " in command and "scan_endpoints.db" in command:
                return f"{old_remote_scan_db}\n{current_remote_scan_db}\n"
            if "base64 < " in command and old_remote_scan_db in command:
                return f"__REACTION_HUB_B64_START__\n{old_db_b64}\n__REACTION_HUB_B64_END__\n"
            if "base64 < " in command and current_remote_scan_db in command:
                return f"__REACTION_HUB_B64_START__\n{new_db_b64}\n__REACTION_HUB_B64_END__\n"
            if " -name " in command and "reaction_hub_summary.json" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteScanDbTieClient())
    assert fetched is True

    old_artifact = Artifact.objects.get(
        workflow_run=run,
        path="888/TDDFT_active_learning_shared/scan_endpoints.db",
    )
    current_artifact = Artifact.objects.get(
        workflow_run=run,
        path="TDDFT_active_learning_shared/scan_endpoints.db",
    )
    assert str(old_artifact.metadata.get("source_scheduler_job_id") or "") == "888"
    assert str(current_artifact.metadata.get("source_scheduler_job_id") or "") == "900"

    start_prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="bond_scan_start_energy_eV",
        unit="eV",
    )
    barrier_prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="bond_scan_activation_barrier_eV",
        unit="eV",
    )
    assert start_prop.value_float == pytest.approx(-150.0)
    assert barrier_prop.value_float == pytest.approx(0.4)


@pytest.mark.django_db
def test_ingest_remote_bond_scan_skips_evaluated_traj_when_newer_scan_db_exists(tmp_path: Path, settings):
    profile = CalculationSettingsProfile.objects.create(name="remote_bond_scan_skip_traj_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_bond_scan_skip_traj_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/remote_bond_scan_skip_traj_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="remote_bond_scan_skip_traj_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"submission": {"run_mode": "tddft_bond_scan"}},
    )
    system = HPCSystem.objects.create(
        name="remote-bond-scan-skip-traj-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    submitted_at = timezone.now()
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="931",
        job_name="remote_bond_scan_skip_traj_job",
        status=JobStatus.COMPLETED,
        submitted_at=submitted_at,
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    source_db_path = tmp_path / "scan_endpoints_skip_traj.db"
    with ase_connect(source_db_path) as db:
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.10)]),
            data={"scan_bond_length": 1.10},
            iteration=1,
            total_energy_eV=-200.0,
        )
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.20)]),
            data={"scan_bond_length": 1.20},
            iteration=2,
            total_energy_eV=-199.7,
        )
    db_b64 = base64.b64encode(source_db_path.read_bytes()).decode("ascii")
    remote_scan_db = f"{run_root}/TDDFT_active_learning_shared/scan_endpoints.db"

    class _FakeRemoteSkipTrajClient:
        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "scan_endpoints.db" in command:
                latest_epoch = submitted_at.timestamp() + 60.0
                return f"{latest_epoch}|{remote_scan_db}\n"
            if "base64 < " in command and remote_scan_db in command:
                return f"__REACTION_HUB_B64_START__\n{db_b64}\n__REACTION_HUB_B64_END__\n"
            if " -path " in command and "active_learning/evaluated.traj" in command:
                raise RuntimeError("evaluated.traj should be skipped for bond-scan ingest when newer scan DB exists")
            if " -name " in command and "reaction_hub_summary.json" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    fetched = _ingest_remote_summary_for_job(job, _FakeRemoteSkipTrajClient())
    assert fetched is True

    assert not Artifact.objects.filter(
        workflow_run=run,
        label="trajectory:evaluated.traj",
    ).exists()
    scan_db_artifact = Artifact.objects.get(
        workflow_run=run,
        path="TDDFT_active_learning_shared/scan_endpoints.db",
    )
    assert scan_db_artifact.kind == "database"
    assert scan_db_artifact.metadata.get("downloaded_from_hpc") is True


@pytest.mark.django_db
def test_ingest_single_run_reads_nested_reaction_hub_summary(tmp_path: Path):
    run_dir = tmp_path / "H2_3"
    nested = run_dir / "12572882"
    nested.mkdir(parents=True)
    summary = {
        "generated_at_utc": "2026-02-21T16:00:00+00:00",
        "final": {
            "formula": "H2",
            "atom_count": 2,
            "symbols": ["H", "H"],
            "positions_A": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            "energy_hartree": -1.1,
            "energy_eV": -29.9325248705868,
        },
    }
    (nested / "reaction_hub_summary.json").write_text(json.dumps(summary), encoding="utf-8")

    run = ingest_single_run(run_dir)
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).exists()


@pytest.mark.django_db
def test_ingest_remote_summary_falls_back_to_nested_summary_path(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="remote_nested_summary_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_nested_summary_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/H2_3"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="H2_3",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-nested-summary-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="755",
        job_name="remote_nested_summary_job",
        status=JobStatus.COMPLETED,
    )

    nested_summary_path = f"{run_root}/12572882/reaction_hub_summary.json"
    summary = {
        "generated_at_utc": "2026-02-21T16:00:00+00:00",
        "final": {
            "formula": "H2",
            "atom_count": 2,
            "symbols": ["H", "H"],
            "positions_A": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            "energy_hartree": -1.1,
            "energy_eV": -29.9325248705868,
        },
    }

    class _FakeNestedSummaryClient:
        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            if remote_path == nested_summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if " -name " in command and "reaction_hub_summary.json" in command:
                return nested_summary_path
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    fetched = _ingest_remote_summary_for_job(job, _FakeNestedSummaryClient())
    assert fetched is True
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).exists()


@pytest.mark.django_db
def test_ingest_remote_summary_falls_back_to_job_id_path(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="remote_jobid_summary_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_jobid_summary_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/H2_3"
    job_id = "12572882"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="H2_3",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-jobid-summary-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id=job_id,
        job_name="remote_jobid_summary_job",
        status=JobStatus.COMPLETED,
    )

    direct_summary_path = f"{run_root}/{job_id}/reaction_hub_summary.json"
    summary = {
        "generated_at_utc": "2026-02-21T16:00:00+00:00",
        "final": {
            "formula": "H2",
            "atom_count": 2,
            "symbols": ["H", "H"],
            "positions_A": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            "energy_hartree": -1.1,
            "energy_eV": -29.9325248705868,
        },
    }

    class _FakeJobIdSummaryClient:
        def __init__(self):
            self.find_called = False

        def file_exists(self, remote_path: str) -> bool:
            return remote_path == direct_summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == direct_summary_path:
                return json.dumps(summary)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if " -name " in command and "reaction_hub_summary.json" in command:
                self.find_called = True
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    fake_client = _FakeJobIdSummaryClient()
    fetched = _ingest_remote_summary_for_job(job, fake_client)
    assert fetched is True
    assert fake_client.find_called is False
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).exists()


@pytest.mark.django_db
def test_ingest_remote_summary_prefers_root_over_legacy_job_id_path():
    profile = CalculationSettingsProfile.objects.create(name="remote_root_preferred_summary_profile")
    workflow = WorkflowDefinition.objects.create(name="remote_root_preferred_summary_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/H2_root"
    job_id = "12572882"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="H2_root",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="remote-root-preferred-summary-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id=job_id,
        job_name="remote_root_preferred_summary_job",
        status=JobStatus.COMPLETED,
    )

    root_summary_path = f"{run_root}/reaction_hub_summary.json"
    legacy_summary_path = f"{run_root}/{job_id}/reaction_hub_summary.json"
    summaries = {
        root_summary_path: {
            "generated_at_utc": "2026-02-21T16:00:00+00:00",
            "final": {"energy_eV": -11.0},
        },
        legacy_summary_path: {
            "generated_at_utc": "2026-02-20T16:00:00+00:00",
            "final": {"energy_eV": -99.0},
        },
    }

    class _FakeRootPreferredSummaryClient:
        def file_exists(self, remote_path: str) -> bool:
            return remote_path in summaries

        def read_text(self, remote_path: str) -> str:
            if remote_path in summaries:
                return json.dumps(summaries[remote_path])
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if "*/scan_endpoints.db" in command:
                return ""
            if " -name " in command and "reaction_hub_summary.json" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    fetched = _ingest_remote_summary_for_job(job, _FakeRootPreferredSummaryClient())
    assert fetched is True
    prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    )
    assert prop.value_float == pytest.approx(-11.0)


@pytest.mark.django_db
def test_ingest_single_run_ignores_shadowed_legacy_job_archive(tmp_path: Path):
    run_dir = tmp_path / "shadowed_archive_run"
    legacy_dir = run_dir / "14005317"
    legacy_dir.mkdir(parents=True)
    root_summary = {
        "generated_at_utc": "2026-02-21T16:00:00+00:00",
        "final": {"energy_eV": -11.0},
    }
    legacy_summary = {
        "generated_at_utc": "2026-02-20T16:00:00+00:00",
        "final": {"energy_eV": -99.0},
    }
    (run_dir / "reaction_hub_summary.json").write_text(json.dumps(root_summary), encoding="utf-8")
    (legacy_dir / "reaction_hub_summary.json").write_text(json.dumps(legacy_summary), encoding="utf-8")
    (run_dir / "DFT").mkdir()
    (legacy_dir / "DFT").mkdir()

    run = ingest_single_run(run_dir)

    prop = ComputedProperty.objects.get(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    )
    assert prop.value_float == pytest.approx(-11.0)
    assert Artifact.objects.filter(
        workflow_run=run,
        path="reaction_hub_summary.json",
        label="reaction_hub_summary",
    ).exists()
    assert not Artifact.objects.filter(
        workflow_run=run,
        path="14005317/reaction_hub_summary.json",
    ).exists()


@pytest.mark.django_db
def test_ingest_workflow_outputs_maps_legacy_job_archive_summary_to_run_root(tmp_path: Path):
    run_dir = tmp_path / "legacy_archive_summary_run"
    legacy_dir = run_dir / "14005317"
    legacy_dir.mkdir(parents=True)
    (legacy_dir / "reaction_hub_summary.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-02-21T10:00:00+00:00",
                "final": {"energy_eV": -11.0},
            }
        ),
        encoding="utf-8",
    )

    result = ingest_workflow_outputs(tmp_path)

    assert result["discovered"] == 1
    run = WorkflowRun.objects.get(run_name="legacy_archive_summary_run")
    assert run.working_directory == str(run_dir.resolve())
    assert ComputedProperty.objects.get(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).value_float == pytest.approx(-11.0)


@pytest.mark.django_db
def test_ingest_workflow_outputs_discovers_summary_only_runs(tmp_path: Path):
    run_dir = tmp_path / "summary_only_run"
    run_dir.mkdir(parents=True)
    (run_dir / "reaction_hub_summary.json").write_text(
        json.dumps(
            {
                "generated_at_utc": "2026-02-21T10:00:00+00:00",
                "final": {
                    "formula": "H2",
                    "atom_count": 2,
                    "symbols": ["H", "H"],
                    "positions_A": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
                    "energy_hartree": -1.1,
                    "energy_eV": -29.9325248705868,
                },
            }
        ),
        encoding="utf-8",
    )

    summary = ingest_workflow_outputs(tmp_path)
    assert summary["discovered"] >= 1
    run = WorkflowRun.objects.get(working_directory=str(run_dir.resolve()))
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).exists()


@pytest.mark.django_db
def test_sync_hpc_jobs_skips_unreachable_systems_before_ssh_connect(tmp_path: Path, monkeypatch, settings):
    settings.REACTION_HUB_SYNC_PREFLIGHT_REACHABILITY = True
    settings.REACTION_HUB_SYNC_PREFLIGHT_TIMEOUT_SECONDS = 0.2

    profile = CalculationSettingsProfile.objects.create(name="sync_preflight_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_preflight_wf", code_path="x.py")

    run_reachable = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_preflight_run_reachable",
        status="running",
        working_directory=str((tmp_path / "sync_preflight_run_reachable").resolve()),
        settings_profile=profile,
        metadata={},
    )
    run_unreachable = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_preflight_run_unreachable",
        status="running",
        working_directory=str((tmp_path / "sync_preflight_run_unreachable").resolve()),
        settings_profile=profile,
        metadata={},
    )

    reachable = HPCSystem.objects.create(
        name="sync-preflight-reachable",
        scheduler_type="slurm",
        host="reachable.example.org",
        username="user",
        is_active=True,
    )
    unreachable = HPCSystem.objects.create(
        name="sync-preflight-unreachable",
        scheduler_type="slurm",
        host="unreachable.example.org",
        username="user",
        is_active=True,
    )

    JobRecord.objects.create(
        workflow_run=run_reachable,
        hpc_system=reachable,
        scheduler_job_id="901",
        job_name="sync_preflight_job_reachable",
        status=JobStatus.RUNNING,
    )
    JobRecord.objects.create(
        workflow_run=run_unreachable,
        hpc_system=unreachable,
        scheduler_job_id="902",
        job_name="sync_preflight_job_unreachable",
        status=JobStatus.RUNNING,
    )

    attempted_hosts: list[str] = []

    class _FakeSlurmClient:
        def __init__(self, host: str, port: int, username: str, ssh_key_path: str = ""):
            del port, username, ssh_key_path
            attempted_hosts.append(str(host))

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    def _fake_reachable(host: str, port: int, timeout_seconds: float) -> bool:
        del port, timeout_seconds
        return host == "reachable.example.org"

    monkeypatch.setattr("ingest.services.SlurmSSHClient", _FakeSlurmClient)
    monkeypatch.setattr("ingest.services._is_hpc_login_host_reachable", _fake_reachable)

    summary = sync_hpc_jobs()

    assert summary["systems"] == 2
    assert summary["reachability_precheck"] is True
    assert summary["skipped_unreachable"] == 1
    assert summary["checked"] == 1
    assert attempted_hosts == ["reachable.example.org"]


@pytest.mark.django_db
def test_sync_hpc_jobs_running_jobs_are_status_polled_only(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_running_status_only_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_running_status_only_wf", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_running_status_only_run",
        status="running",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/sync_running_status_only_run",
        settings_profile=profile,
        metadata={},
    )
    system = HPCSystem.objects.create(
        name="sync-running-status-only-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="903",
        job_name="sync_running_status_only_job",
        status=JobStatus.RUNNING,
    )

    directories_checked: list[list[str]] = []
    file_checks: list[str] = []
    text_reads: list[str] = []
    exec_calls: list[str] = []

    class _FakeStatusOnlyClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            return {str(job_id): {"state": "RUNNING"} for job_id in job_ids}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            directories_checked.append(list(remote_paths))
            return {path: True for path in remote_paths}

        def file_exists(self, remote_path: str) -> bool:
            file_checks.append(str(remote_path))
            return False

        def read_text(self, remote_path: str) -> str:
            text_reads.append(str(remote_path))
            return ""

        def exec_checked(self, command: str) -> str:
            exec_calls.append(str(command))
            return ""

    summary = sync_hpc_jobs(client_factory=lambda _system: _FakeStatusOnlyClient())
    job.refresh_from_db()
    run.refresh_from_db()

    assert summary["checked"] == 1
    assert job.status == JobStatus.RUNNING
    assert run.status == "running"
    assert directories_checked == []
    assert file_checks == []
    assert text_reads == []
    assert exec_calls == []


@pytest.mark.django_db
def test_completed_charge_transfer_job_queues_plot_backfill(tmp_path: Path, monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_ct_backfill_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_ct_backfill_wf", code_path="x.py")
    run_dir = tmp_path / "sync_ct_backfill_run"
    run_dir.mkdir()
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_ct_backfill_run",
        status="running",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={"submission": {"run_mode": "charge_transfer"}},
    )
    system = HPCSystem.objects.create(
        name="sync-ct-backfill-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="904",
        job_name=run.run_name,
        status=JobStatus.RUNNING,
    )
    queued = []
    monkeypatch.setattr("ingest.services.ingest_single_run", lambda _path: run)
    monkeypatch.setattr(
        "ingest.tasks.backfill_ct_plots_task.delay",
        lambda run_id: queued.append(run_id),
    )

    class _CompletedClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids):
            return {str(job_id): {"state": "COMPLETED"} for job_id in job_ids}

        def directories_exist(self, remote_paths):
            return {path: True for path in remote_paths}

    summary = sync_hpc_jobs(client_factory=lambda _system: _CompletedClient())

    assert queued == [run.pk]
    assert summary["ct_plot_backfills_queued"] == 1
    job.refresh_from_db()
    assert job.status == JobStatus.COMPLETED


@pytest.mark.django_db
def test_backfill_ct_plots_task_applies_retained_summary_to_source(monkeypatch, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="ct_task_profile")
    workflow = WorkflowDefinition.objects.create(name="ct_task_wf", code_path="x.py")
    source = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_task_source",
        status="completed",
        settings_profile=profile,
    )
    summary = {
        "status": "completed",
        "run_type": "charge_transfer",
        "source_run_id": source.pk,
        "charge_transfer": {
            "fragment_labels": {"1": "NC", "2": "CO", "3": "O"},
            "fragments_0based": {"1": [0, 1], "2": [2, 3], "3": [4]},
            "fragments_1based": {"1": "1-2", "2": "3-4", "3": "5"},
            "ct_states": [],
            "ct_terms": [],
            "ct_spectra": [],
        },
    }
    ct_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_task_followup",
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {"run_mode": "charge_transfer"},
            "result_summary": summary,
        },
    )
    summary_path = tmp_path / "reaction_hub_summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    Artifact.objects.create(
        workflow_run=ct_run,
        kind=ArtifactKind.JSON,
        label="reaction_hub_summary",
        path="reaction_hub_summary.json",
        metadata={"local_cache_path": str(summary_path)},
    )
    monkeypatch.setattr("ingest.spectral.write_ct_plot_artifacts", lambda run: [])

    result = backfill_ct_plots_task(ct_run.pk)
    repeated = backfill_ct_plots_task(ct_run.pk)

    source.refresh_from_db()
    dataset = load_spectral_dataset(source, "final_tddft")
    assert dataset["fragments"]["fragment_labels"] == {"1": "NC", "2": "CO", "3": "O"}
    assert source.metadata["charge_transfer_run_ids"] == [ct_run.pk]
    assert result["source_run_id"] == source.pk
    assert repeated["source_run_id"] == source.pk


@pytest.mark.django_db
def test_backfill_ct_plots_task_skips_density_analysis_only(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="density_task_profile")
    workflow = WorkflowDefinition.objects.create(name="density_task_wf", code_path="x.py")
    da_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ts_vib_grad_run",
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {
                "run_mode": "charge_transfer",
                "charge_transfer_config": {"density_analysis_only": True},
            },
            "result_summary": {
                "status": "completed",
                "run_type": "ts_vibrations_gradient",
            },
        },
    )
    result = backfill_ct_plots_task(da_run.pk)
    assert result["skipped"] == "density_analysis_only"
    assert result["artifact_ids"] == []


@pytest.mark.django_db
def test_sync_hpc_jobs_recovery_uses_latest_job_record_per_run(tmp_path: Path, monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_latest_job_recovery_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_latest_job_recovery_wf", code_path="x.py")
    run_dir = tmp_path / "sync_latest_job_recovery_run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_latest_job_recovery_run",
        status="completed",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={},
    )
    system = HPCSystem.objects.create(
        name="sync-latest-job-recovery-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    # Older historical completed record (e.g. restart/resubmit predecessor).
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="910",
        job_name="sync_latest_job_recovery_job_old",
        status=JobStatus.COMPLETED,
    )
    # Latest completed record for the same workflow run.
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="911",
        job_name="sync_latest_job_recovery_job_new",
        status=JobStatus.COMPLETED,
    )

    ingest_calls: list[str] = []

    def _fake_ingest_single_run(run_path: Path, workflow_name: str = "cplas_tddft_active_learning"):
        del workflow_name
        ingest_calls.append(str(run_path))
        return run

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    summary = sync_hpc_jobs(
        client_factory=lambda _system: _FakeSyncClient(),
        force_reingestion=True,
    )

    assert summary["updated"] >= 1
    # Regression check: only the latest completed record for the run should
    # trigger recovery ingestion.
    assert ingest_calls == [str(run_dir)]


@pytest.mark.django_db
def test_sync_hpc_jobs_skips_stale_completed_recovery_when_lineage_has_active_job(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_lineage_active_skip_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_lineage_active_skip_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/sync_lineage_active_skip"

    old_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_lineage_active_skip_500",
        status="completed",
        working_directory=f"{run_root}_500/13227642/13229045/13260412",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "tddft_bond_scan", "target_wavelength_nm": 500.0}},
    )
    active_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_lineage_active_skip_500_2",
        status="running",
        working_directory=f"{run_root}_500_2",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "tddft_bond_scan", "target_wavelength_nm": 500.0}},
    )

    system = HPCSystem.objects.create(
        name="sync-lineage-active-skip-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )

    JobRecord.objects.create(
        workflow_run=old_run,
        hpc_system=system,
        scheduler_job_id="13301034",
        job_name="sync_lineage_active_skip_old",
        status=JobStatus.COMPLETED,
    )
    active_job = JobRecord.objects.create(
        workflow_run=active_run,
        hpc_system=system,
        scheduler_job_id="13301627",
        job_name="sync_lineage_active_skip_active",
        status=JobStatus.INITIATED,
    )

    remote_ingest_attempts: list[str] = []

    def _fake_ingest_remote_summary_for_job(
        job,
        _client,
        *,
        download_missing_only: bool = False,
    ) -> bool:
        del download_missing_only
        remote_ingest_attempts.append(str(job.scheduler_job_id))
        return True

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            return {str(job_id): {"state": "RUNNING"} for job_id in job_ids}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services._ingest_remote_summary_for_job", _fake_ingest_remote_summary_for_job)

    summary = sync_hpc_jobs(client_factory=lambda _system: _FakeSyncClient())

    active_job.refresh_from_db()
    assert summary["checked"] == 1
    assert active_job.status == JobStatus.RUNNING
    assert remote_ingest_attempts == []


@pytest.mark.django_db
def test_sync_hpc_jobs_does_not_implicitly_retry_summary_ingest_for_completed_jobs(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="sync_retry_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_retry_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/sync_retry_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_retry_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={},
    )
    system = HPCSystem.objects.create(
        name="sync-retry-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="777",
        job_name="sync_retry_job",
        status=JobStatus.COMPLETED,
    )

    summary_path = f"{run_root}/reaction_hub_summary.json"
    summary_payload = {
        "generated_at_utc": "2026-02-21T10:00:00+00:00",
        "final": {
            "formula": "CO",
            "atom_count": 2,
            "symbols": ["C", "O"],
            "positions_A": [[0.0, 0.0, 0.0], [0.0, 0.0, 1.128]],
            "energy_hartree": -113.0,
            "energy_eV": -3074.886647796644,
            "charges": [-0.2, 0.2],
            "charge_method": "loewdin",
        },
    }

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def file_exists(self, remote_path: str) -> bool:
            return remote_path == summary_path

        def read_text(self, remote_path: str) -> str:
            if remote_path == summary_path:
                return json.dumps(summary_payload)
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if "active_learning/evaluated.traj" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    sync_summary = sync_hpc_jobs(client_factory=lambda _system: _FakeSyncClient())
    assert sync_summary["updated"] == 0

    run.refresh_from_db()
    assert "result_summary" not in run.metadata
    assert not ComputedProperty.objects.filter(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
    ).exists()


@pytest.mark.django_db
def test_sync_hpc_jobs_does_not_implicitly_retry_completed_single_optimisation_missing_requested_final_tddft_spectrum(
    tmp_path: Path,
    monkeypatch,
):
    profile = CalculationSettingsProfile.objects.create(name="sync_single_final_tddft_retry_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_single_final_tddft_retry_wf", code_path="x.py")
    run_dir = tmp_path / "sync_single_final_tddft_retry_run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_single_final_tddft_retry_run",
        status="completed",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={
            "result_summary": {"mode": "dft"},
            "submission": {
                "run_mode": "single_optimisation",
                "single_constraints_config": {
                    "final_tddft_spectrum": True,
                    "final_tddft_nroots": 50,
                },
            },
        },
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-1.0,
    )
    system = HPCSystem.objects.create(
        name="sync-single-final-tddft-retry-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="7811",
        job_name="sync_single_final_tddft_retry_job",
        status=JobStatus.COMPLETED,
    )

    ingest_calls: list[str] = []

    def _fake_ingest_single_run(run_path: Path, workflow_name: str = "cplas_tddft_active_learning"):
        del workflow_name
        ingest_calls.append(str(run_path))
        Artifact.objects.create(
            workflow_run=run,
            kind=ArtifactKind.IMAGE,
            label="plot:final_tddft_spectrum",
            path="DFT/final_tddft_spectrum/final_tddft_spectrum.png",
        )
        return run

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    sync_summary = sync_hpc_jobs(client_factory=lambda _system: _FakeSyncClient())

    assert ingest_calls == []
    assert sync_summary["updated"] == 0
    assert not Artifact.objects.filter(
        workflow_run=run,
        kind=ArtifactKind.IMAGE,
        label="plot:final_tddft_spectrum",
    ).exists()


@pytest.mark.django_db
def test_sync_hpc_jobs_skips_scan_profile_refresh_when_recent_result_summary_exists(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_scan_summary_skip_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_scan_summary_skip_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/sync_scan_summary_skip"
    submitted_at = timezone.now()
    generated_at = (submitted_at + timedelta(minutes=2)).isoformat()
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_scan_summary_skip_500",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={
            "submission": {"run_mode": "tddft_bond_scan", "target_wavelength_nm": 500.0},
            "result_summary": {"generated_at_utc": generated_at},
        },
    )
    system = HPCSystem.objects.create(
        name="sync-scan-summary-skip-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="99001",
        job_name="sync_scan_summary_skip_job",
        status=JobStatus.COMPLETED,
        submitted_at=submitted_at,
    )
    summary_artifact = Artifact.objects.create(
        workflow_run=run,
        kind=ArtifactKind.JSON,
        label="reaction_hub_summary",
        path="TDDFT_active_learning_shared/reaction_hub_summary.json",
        metadata={},
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        source_artifact=summary_artifact,
        name="bond_scan_energy_profile",
        unit="eV",
        value_text="[0.0, 0.2, 0.4]",
    )

    remote_ingest_attempts: list[str] = []

    def _fake_ingest_remote_summary_for_job(
        job,
        _client,
        *,
        download_missing_only: bool = False,
    ) -> bool:
        del download_missing_only
        remote_ingest_attempts.append(str(job.scheduler_job_id))
        return True

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services._ingest_remote_summary_for_job", _fake_ingest_remote_summary_for_job)
    summary = sync_hpc_jobs(client_factory=lambda _system: _FakeSyncClient())

    assert summary["updated"] == 0
    assert remote_ingest_attempts == []


@pytest.mark.django_db
def test_sync_hpc_jobs_does_not_implicitly_retry_completed_scan_jobs_when_scan_profile_is_stale(tmp_path: Path, monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_retry_scan_profile_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_retry_scan_profile_wf", code_path="x.py")
    run_dir = tmp_path / "sync_retry_scan_profile_run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_retry_scan_profile_run",
        status="completed",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={
            "result_summary": {"mode": "dft"},
            "submission": {"run_mode": "tddft_bond_scan"},
        },
    )
    system = HPCSystem.objects.create(
        name="sync-retry-scan-profile-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="900",
        job_name="sync_retry_scan_profile_job",
        status=JobStatus.COMPLETED,
    )

    stale_scan_artifact = Artifact.objects.create(
        workflow_run=run,
        kind="database",
        label="scan_endpoints_db",
        path="888/TDDFT_active_learning_shared/scan_endpoints.db",
        metadata={
            "source": "remote_download",
            "source_scheduler_job_id": "888",
            "remote_path": "/work/PROJECT/GROUP/user/reaction_hub_runs/sync_retry_scan_profile_run/888/TDDFT_active_learning_shared/scan_endpoints.db",
        },
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="bond_scan_energy_profile",
        unit="eV",
        value_text="path_profile",
        source_artifact=stale_scan_artifact,
        metadata={"source": "scan_profile", "points": []},
    )

    ingest_calls: list[str] = []

    def _fake_ingest_single_run(run_path: Path, workflow_name: str = "cplas_tddft_active_learning"):
        del workflow_name
        ingest_calls.append(str(run_path))
        return run

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    sync_summary = sync_hpc_jobs(client_factory=lambda _system: _FakeSyncClient())

    assert ingest_calls == []
    assert sync_summary["updated"] == 0


@pytest.mark.django_db
@pytest.mark.parametrize(("runtime", "should_ingest"), [("00:02:59", False), ("00:03:00", True)])
def test_sync_hpc_jobs_failed_job_artifact_recovery_depends_on_runtime(
    tmp_path: Path,
    monkeypatch,
    runtime: str,
    should_ingest: bool,
):
    profile = CalculationSettingsProfile.objects.create(name="sync_running_to_failed_ingest_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_running_to_failed_ingest_wf", code_path="x.py")
    run_dir = tmp_path / "sync_running_to_failed_ingest_run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_running_to_failed_ingest_run",
        status="running",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={},
    )
    system = HPCSystem.objects.create(
        name="sync-running-to-failed-ingest-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="778",
        job_name="sync_running_to_failed_ingest_job",
        status=JobStatus.RUNNING,
    )

    ingest_calls: list[str] = []

    def _fake_ingest_single_run(run_path: Path, workflow_name: str = "cplas_tddft_active_learning"):
        del workflow_name
        ingest_calls.append(str(run_path))
        return run

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            return {job_id: {"state": "FAILED", "runtime": runtime} for job_id in job_ids}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    sync_summary = sync_hpc_jobs(client_factory=lambda _system: _FakeSyncClient())

    job.refresh_from_db()
    run.refresh_from_db()
    assert job.status == JobStatus.FAILED
    assert run.status == "failed"
    assert ingest_calls == ([str(run_dir)] if should_ingest else [])
    assert sync_summary["updated"] >= 1


@pytest.mark.django_db
def test_sync_hpc_jobs_skips_failed_jobs_missing_plots_by_default(tmp_path: Path, monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_failed_reingest_missing_plots_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_failed_reingest_missing_plots_wf", code_path="x.py")
    run_dir = tmp_path / "sync_failed_reingest_missing_plots_run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_failed_reingest_missing_plots_run",
        status="failed",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={"result_summary": {"mode": "dft"}},
    )
    system = HPCSystem.objects.create(
        name="sync-failed-reingest-missing-plots-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="7820",
        job_name="sync_failed_reingest_missing_plots_job",
        status=JobStatus.FAILED,
        failure_reason="Active learning did not converge",
    )

    ingest_calls: list[str] = []

    def _fake_ingest_single_run(run_path: Path, workflow_name: str = "cplas_tddft_active_learning"):
        del workflow_name
        ingest_calls.append(str(run_path))
        return run

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    sync_summary = sync_hpc_jobs(client_factory=lambda _system: _FakeSyncClient())

    job.refresh_from_db()
    run.refresh_from_db()
    assert job.status == JobStatus.FAILED
    assert run.status == "failed"
    assert ingest_calls == []
    assert sync_summary["retry_failed_ingestion"] is False
    assert sync_summary["updated"] == 0


@pytest.mark.django_db
def test_sync_hpc_jobs_force_reingestion_does_not_retry_failed_jobs_without_flag(
    tmp_path: Path,
    monkeypatch,
):
    profile = CalculationSettingsProfile.objects.create(name="sync_failed_force_skip_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_failed_force_skip_wf", code_path="x.py")
    run_dir = tmp_path / "sync_failed_force_skip_run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_failed_force_skip_run",
        status="failed",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={},
    )
    system = HPCSystem.objects.create(
        name="sync-failed-force-skip-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="7821",
        job_name="sync_failed_force_skip_job",
        status=JobStatus.FAILED,
        failure_reason="Active learning did not converge",
    )

    ingest_calls: list[str] = []

    def _fake_ingest_single_run(run_path: Path, workflow_name: str = "cplas_tddft_active_learning"):
        del workflow_name
        ingest_calls.append(str(run_path))
        return run

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    sync_summary = sync_hpc_jobs(
        client_factory=lambda _system: _FakeSyncClient(),
        force_reingestion=True,
    )

    job.refresh_from_db()
    run.refresh_from_db()
    assert job.status == JobStatus.FAILED
    assert run.status == "failed"
    assert ingest_calls == []
    assert sync_summary["force_reingestion"] is True
    assert sync_summary["retry_failed_ingestion"] is False
    assert sync_summary["updated"] == 0


@pytest.mark.django_db
def test_sync_hpc_jobs_reingests_failed_jobs_without_changing_failed_status(tmp_path: Path, monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_failed_reingest_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_failed_reingest_wf", code_path="x.py")
    run_dir = tmp_path / "sync_failed_reingest_run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_failed_reingest_run",
        status="failed",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={},
    )
    system = HPCSystem.objects.create(
        name="sync-failed-reingest-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="782",
        job_name="sync_failed_reingest_job",
        status=JobStatus.FAILED,
        failure_reason="Active learning did not converge",
    )

    ingest_calls: list[str] = []

    def _fake_ingest_single_run(run_path: Path, workflow_name: str = "cplas_tddft_active_learning"):
        del workflow_name
        ingest_calls.append(str(run_path))
        return run

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    sync_summary = sync_hpc_jobs(
        client_factory=lambda _system: _FakeSyncClient(),
        retry_failed_ingestion=True,
    )

    job.refresh_from_db()
    run.refresh_from_db()
    assert job.status == JobStatus.FAILED
    assert run.status == "failed"
    assert ingest_calls == [str(run_dir)]
    assert sync_summary["retry_failed_ingestion"] is True
    assert sync_summary["updated"] >= 1


@pytest.mark.django_db
def test_sync_hpc_jobs_force_reingestion_reingests_completed_jobs(tmp_path: Path, monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="sync_force_reingest_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_force_reingest_wf", code_path="x.py")
    run_dir = tmp_path / "sync_force_reingest_run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_force_reingest_run",
        status="completed",
        working_directory=str(run_dir),
        settings_profile=profile,
        metadata={"result_summary": {"mode": "dft"}},
    )
    system = HPCSystem.objects.create(
        name="sync-force-reingest-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="779",
        job_name="sync_force_reingest_job",
        status=JobStatus.COMPLETED,
    )

    ingest_calls: list[str] = []

    def _fake_ingest_single_run(run_path: Path, workflow_name: str = "cplas_tddft_active_learning"):
        del workflow_name
        ingest_calls.append(str(run_path))
        return run

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    sync_summary = sync_hpc_jobs(
        client_factory=lambda _system: _FakeSyncClient(),
        force_reingestion=True,
    )

    assert sync_summary["force_reingestion"] is True
    assert ingest_calls == [str(run_dir)]


@pytest.mark.django_db
def test_sync_hpc_jobs_force_reingestion_downloads_only_missing_remote_files(tmp_path: Path, settings):
    profile = CalculationSettingsProfile.objects.create(name="sync_force_missing_only_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_force_missing_only_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/sync_force_missing_only_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_force_missing_only_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"result_summary": {"mode": "dft"}},
    )
    system = HPCSystem.objects.create(
        name="sync-force-missing-only-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="780",
        job_name="sync_force_missing_only_job",
        status=JobStatus.COMPLETED,
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    rel_traj_path = "TDDFT/opt_iter_1/active_learning/evaluated.traj"
    local_cache_path = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT) / str(run.run_uid) / rel_traj_path
    local_cache_path.parent.mkdir(parents=True, exist_ok=True)
    local_cache_path.write_bytes(b"already_cached")

    remote_traj_path = f"{run_root}/{rel_traj_path}"
    remote_base64_calls: list[str] = []

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "active_learning/evaluated.traj" in command:
                return f"{remote_traj_path}\n"
            if "base64 < " in command:
                remote_base64_calls.append(command)
                raise RuntimeError("base64 fetch should be skipped for pre-cached files")
            if " -name " in command and "reaction_hub_summary.json" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    sync_summary = sync_hpc_jobs(
        client_factory=lambda _system: _FakeSyncClient(),
        force_reingestion=True,
    )

    assert sync_summary["force_reingestion"] is True
    assert not remote_base64_calls
    traj_artifact = Artifact.objects.get(workflow_run=run, path=rel_traj_path)
    assert traj_artifact.metadata.get("downloaded_from_hpc") is True
    assert Path(str(traj_artifact.metadata.get("local_cache_path"))) == local_cache_path
    assert local_cache_path.read_bytes() == b"already_cached"


@pytest.mark.django_db
def test_sync_hpc_jobs_force_reingestion_ingests_remote_scan_endpoints_db(tmp_path: Path, settings):
    profile = CalculationSettingsProfile.objects.create(name="sync_force_scan_db_profile")
    workflow = WorkflowDefinition.objects.create(name="sync_force_scan_db_wf", code_path="x.py")
    run_root = "/work/PROJECT/GROUP/user/reaction_hub_runs/sync_force_scan_db_run"
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="sync_force_scan_db_run",
        status="completed",
        working_directory=run_root,
        settings_profile=profile,
        metadata={"result_summary": {"mode": "dft"}},
    )
    system = HPCSystem.objects.create(
        name="sync-force-scan-db-hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="781",
        job_name="sync_force_scan_db_job",
        status=JobStatus.COMPLETED,
    )

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    settings.REACTION_HUB_INGEST_REMOTE_ORCA_READ = False

    source_db_path = tmp_path / "sync_scan_endpoints_source.db"
    with ase_connect(source_db_path) as db:
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.11)]),
            data={"scan_bond_length": 1.11},
            iteration=1,
            total_energy_eV=-150.0,
        )
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.21)]),
            data={"scan_bond_length": 1.21},
            iteration=2,
            total_energy_eV=-149.6,
        )
    db_b64 = base64.b64encode(source_db_path.read_bytes()).decode("ascii")
    remote_scan_db_path = f"{run_root}/TDDFT_active_learning_shared/scan_endpoints.db"

    class _FakeSyncClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

        def query_jobs(self, job_ids: list[str]) -> dict[str, dict]:
            del job_ids
            return {}

        def directories_exist(self, remote_paths: list[str]) -> dict[str, bool]:
            return {path: True for path in remote_paths}

        def file_exists(self, remote_path: str) -> bool:
            del remote_path
            return False

        def read_text(self, remote_path: str) -> str:
            raise RuntimeError(f"unexpected read_text path: {remote_path}")

        def exec_checked(self, command: str) -> str:
            if " -path " in command and "active_learning/evaluated.traj" in command:
                return ""
            if " -path " in command and "scan_endpoints.db" in command:
                return f"{remote_scan_db_path}\n"
            if "base64 < " in command and remote_scan_db_path in command:
                return f"__REACTION_HUB_B64_START__\n{db_b64}\n__REACTION_HUB_B64_END__\n"
            if " -name " in command and "reaction_hub_summary.json" in command:
                return ""
            if " -name " in command and "orca.out" in command:
                return ""
            return ""

    sync_summary = sync_hpc_jobs(
        client_factory=lambda _system: _FakeSyncClient(),
        force_reingestion=True,
    )

    assert sync_summary["force_reingestion"] is True
    assert sync_summary["updated"] >= 1

    scan_db_artifact = Artifact.objects.get(
        workflow_run=run,
        path="TDDFT_active_learning_shared/scan_endpoints.db",
    )
    assert scan_db_artifact.kind == "database"
    assert scan_db_artifact.metadata.get("downloaded_from_hpc") is True
    assert ComputedProperty.objects.filter(
        workflow_run=run,
        name="bond_scan_activation_barrier_eV",
        unit="eV",
    ).exists()
