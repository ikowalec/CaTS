from __future__ import annotations

import json
from decimal import Decimal
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect as ase_connect
from ase.io import read as ase_read
from ase.io import write
from django.contrib.admin.sites import AdminSite
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory, override_settings

from jobs.admin import JobRecordAdmin
from jobs.forms import JobSubmitForm, ReferenceIntent
from jobs.views import (
    JobSubmitView,
    _apply_reference_selection_to_run,
    _infer_ct_fragment_defaults,
    _latest_restart_band_extxyz_for_run,
    _job_resubmit_failed_sync_response,
    _latest_restart_structure_xyz_for_run,
    _tddft_nroots_for_run,
)
from jobs.tasks import _task_request
from ingest.services import _ingest_summary_payload, sync_hpc_jobs
from ingest.spectral import load_spectral_dataset, update_spectral_dataset
from jobs.models import HPCSystem, JobRecord, JobStatus, JobStatusEvent
from references.models import CalculationSettingsProfile, ReferenceAdsorbate, ReferenceNanocluster
from results.models import AdsorptionEnergyResult, Artifact, ArtifactKind, ComputedProperty
from results.services import upsert_adsorption_energy_for_run
from workflows.models import Campaign, CampaignRun, WorkflowDefinition, WorkflowRun


class _FakeClient:
    def __init__(self, snapshots):
        self.snapshots = snapshots

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def query_jobs(self, job_ids):
        return {job_id: self.snapshots[job_id] for job_id in job_ids if job_id in self.snapshots}

    def directories_exist(self, remote_paths):
        return {path: True for path in remote_paths}


@pytest.mark.django_db
def test_job_list_tracks_coadsorbate_campaign_job_location_and_status(client):
    profile = CalculationSettingsProfile.objects.create(name="tracked-coadsorbate-profile")
    workflow = WorkflowDefinition.objects.create(
        name="tracked-coadsorbate-workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    run = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="Au20_GPU_survey_CO2_H_MACE_2node",
        metadata={"submission": {"run_mode": "coadsorbate_placement"}},
    )
    campaign = Campaign.objects.create(name="Tracked GPU campaign", slug="tracked-gpu-campaign")
    CampaignRun.objects.create(campaign=campaign, workflow_run=run, checkpoint_stage=3)
    JobRecord.objects.create(
        workflow_run=run,
        scheduler_job_id="14949999",
        job_name=run.run_name,
        status=JobStatus.PENDING,
        remote_workdir="/scratch/PROJECT/GROUP/user/14949999",
    )

    response = client.get("/jobs/")
    body = response.content.decode("utf-8")

    assert response.status_code == 200
    assert "14949999" in body
    assert "pending" in body
    assert "Tracked GPU campaign" in body
    assert "/scratch/PROJECT/GROUP/user/14949999" in body


@pytest.mark.django_db
def test_job_submit_form_workflow_queryset_deduplicates_by_code_path():
    WorkflowDefinition.objects.create(
        name="TDDFT Active Learning",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    WorkflowDefinition.objects.create(
        name="cplas_tddft_active_learning",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    WorkflowDefinition.objects.create(
        name="Another Workflow",
        code_path="cats/working_directory/another_workflow.py",
    )

    form = JobSubmitForm()
    workflows = list(form.fields["workflow"].queryset)
    tddft_entries = [
        workflow
        for workflow in workflows
        if workflow.code_path == "cats/working_directory/tddft_active_learning_workflow.py"
    ]
    assert len(tddft_entries) == 1
    assert tddft_entries[0].name == "TDDFT Active Learning"
    assert form.fields["active_learning_optimizer"].initial == "bfgs"
    assert ("fire2", "FIRE2") in tuple(form.fields["active_learning_optimizer"].choices)


@pytest.mark.django_db
def test_job_submit_form_ensures_canonical_coadsorbate_workflow_exists():
    WorkflowDefinition.objects.create(
        name="TDDFT Active Learning",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )

    form = JobSubmitForm()
    coadsorbate_workflows = [
        workflow
        for workflow in form.fields["workflow"].queryset
        if "coadsorbate" in str(workflow.code_path or "").strip().lower()
    ]

    assert len(coadsorbate_workflows) >= 1
    assert any(str(workflow.name) == "MACE Adsorbate Screening" for workflow in coadsorbate_workflows)


@pytest.mark.django_db
def test_job_submit_form_ensures_canonical_neb_workflow_exists():
    form = JobSubmitForm()
    neb_workflows = [
        workflow
        for workflow in form.fields["workflow"].queryset
        if "active_learning_neb_workflow.py" in str(workflow.code_path or "").strip().lower()
    ]

    assert len(neb_workflows) >= 1
    assert any(str(workflow.name) == "Active Learning NEB (MACE)" for workflow in neb_workflows)


@pytest.mark.django_db
def test_job_submit_form_ensures_canonical_tddft_mlneb_workflow_exists():
    form = JobSubmitForm()
    mlneb_workflows = [
        workflow
        for workflow in form.fields["workflow"].queryset
        if "tddft_mlneb_workflow.py" in str(workflow.code_path or "").strip().lower()
    ]

    assert len(mlneb_workflows) >= 1
    assert any(str(workflow.name) == "TDDFT MLNEB" for workflow in mlneb_workflows)


@pytest.mark.django_db
def test_job_submit_form_ensures_canonical_trh408_hpc_exists(settings):
    settings.REACTION_HUB_SUBMIT_TRH408_HOST = "trh408.example.org"
    settings.REACTION_HUB_SUBMIT_TRH408_PORT = 22
    settings.REACTION_HUB_SUBMIT_TRH408_USERNAME = "researcher"
    form = JobSubmitForm()
    trh408 = HPCSystem.objects.filter(name="TRH408").first()

    assert trh408 is not None
    assert trh408.scheduler_type == "slurm"
    assert trh408.is_active is True
    assert trh408.host == "trh408.example.org"
    assert trh408.username == "researcher"
    assert any(system.pk == trh408.pk for system in form.fields["hpc_system"].queryset)


@pytest.mark.django_db
def test_job_submit_form_ensures_canonical_falcon_hpc_exists():
    form = JobSubmitForm()
    falcon = HPCSystem.objects.filter(name="FALCON").first()

    assert falcon is not None
    assert falcon.scheduler_type == "slurm"
    assert falcon.is_active is True
    assert any(system.pk == falcon.pk for system in form.fields["hpc_system"].queryset)


@pytest.mark.django_db
def test_job_submit_form_ensures_nwchem_settings_profile_exists():
    form = JobSubmitForm()
    profile = CalculationSettingsProfile.objects.filter(name="nwchem_pbe0_def2svp").first()

    assert profile is not None
    assert profile.basis_set == "def2-SVP"
    assert profile.xc_functional == "PBE0"
    assert profile.metadata["qm_backend"] == "nwchem"
    assert ("nwchem", "NWChem") in tuple(form.fields["qm_backend"].choices)
    assert any(candidate.pk == profile.pk for candidate in form.fields["settings_profile"].queryset)


@pytest.mark.django_db
def test_job_submit_context_exposes_settings_profile_backend_map(client):
    resp = client.get("/jobs/submit/")

    assert resp.status_code == 200
    profile = CalculationSettingsProfile.objects.get(name="nwchem_pbe0_def2svp")
    backend_map = json.loads(resp.context["settings_profile_backend_json"])
    assert backend_map[str(profile.pk)] == "nwchem"


@pytest.mark.django_db
def test_job_submit_form_builds_one_based_charge_transfer_fragments():
    profile = CalculationSettingsProfile.objects.create(
        name="ct_form_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="",
    )
    workflow = WorkflowDefinition.objects.create(
        name="ct_form_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="ct_form_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_source",
        working_directory="/work/PROJECT/GROUP/user/ct_source",
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind=ArtifactKind.STRUCTURE,
        label="structure:source.xyz",
        path="source.xyz",
    )
    form = JobSubmitForm(
        data={
            "run_name": "ct_source_ct",
            "run_mode": "charge_transfer",
            "workflow": workflow.pk,
            "hpc_system": system.pk,
            "settings_profile": profile.pk,
            "geometry_source": "existing_run",
            "source_run_id": source_run.pk,
            "source_artifact_id": artifact.pk,
            "ct_nanoparticle_indices": "0-62",
            "ct_adsorbate_indices": "63, 64",
            "ct_nroots": "400",
            "ct_orca_basename": "orca",
            "nodes": "1",
            "tasks_per_node": "4",
            "cpus_per_task": "1",
            "walltime": "24:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "serial",
            "qos": "serial",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
        }
    )

    assert form.is_valid(), form.errors.as_json()
    config = form.charge_transfer_config()
    assert config["fragments_0based"] == {"1": list(range(63)), "2": [63, 64]}
    assert config["fragments_1based"] == {"1": "1-63", "2": "64-65"}
    assert config["nroots"] == 400


def test_charge_transfer_defaults_split_metals_from_nonmetals():
    defaults = _infer_ct_fragment_defaults(Atoms("AuCuOCH"))

    assert defaults["nanoparticle_indices"] == "0-1"
    assert defaults["adsorbate_indices"] == "2-4"
    assert defaults["fragments_0based"] == {"1": [0, 1], "2": [2, 3, 4]}


def test_charge_transfer_defaults_split_connected_adsorbates():
    atoms = Atoms(
        "AuAuCOO",
        positions=[(0, 0, 0), (0, 0, 2.8), (4, 0, 0), (5.13, 0, 0), (0, 4, 0)],
    )

    defaults = _infer_ct_fragment_defaults(atoms)

    assert defaults["fragments_0based"] == {"1": [0, 1], "2": [2, 3], "3": [4]}
    assert defaults["fragment_labels"] == {"1": "NC", "2": "CO", "3": "O"}


@pytest.mark.django_db
def test_charge_transfer_nroots_uses_retained_tddft_state_count():
    profile = CalculationSettingsProfile.objects.create(name="ct_nroots_profile")
    workflow = WorkflowDefinition.objects.create(name="ct_nroots_workflow", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_nroots_source",
        settings_profile=profile,
        metadata={
            "submission": {
                "single_constraints_config": {"final_tddft_nroots": 120}
            }
        },
    )
    update_spectral_dataset(
        run=run,
        dataset_name="final_tddft",
        payload={"transitions": [{"index": index} for index in range(1, 151)]},
    )

    assert _tddft_nroots_for_run(run) == 150

    run_without_retained_spectrum = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_nroots_modern_source",
        settings_profile=profile,
        metadata={
            "submission": {
                "single_constraints_config": {"final_tddft_nroots": 120}
            }
        },
    )
    assert _tddft_nroots_for_run(run_without_retained_spectrum) == 120


@pytest.mark.django_db
def test_charge_transfer_followup_prefills_roots_and_inferred_fragments(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="ct_followup_prefill_profile")
    workflow = WorkflowDefinition.objects.create(name="ct_followup_prefill_workflow", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_followup_prefill_source",
        settings_profile=profile,
        metadata={
            "submission": {
                "reservation": "old-expired-reservation",
                "single_constraints_config": {"final_tddft_nroots": 275}
            }
        },
    )
    archer2 = HPCSystem.objects.create(
        name="ARCHER2",
        scheduler_type="slurm",
        host="hpc.example.invalid",
        username="user",
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind=ArtifactKind.STRUCTURE,
        label="final",
        path="reaction_hub_final.xyz",
    )
    view = JobSubmitView()
    view.request = RequestFactory().get(
        f"/jobs/submit/?from_run={run.pk}&mode=charge_transfer"
    )
    monkeypatch.setattr(
        view,
        "_prefill_source_from_query",
        lambda: {"run": run, "artifact": artifact, "job": None},
    )
    monkeypatch.setattr(
        "jobs.views._resolve_existing_run_structure_atoms",
        lambda **_kwargs: (Atoms("AuAuCO"), {}),
    )

    initial = view.get_initial()

    assert initial["ct_nroots"] == 275
    assert initial["hpc_system"] == archer2.pk
    assert initial["account"] == "SLURM_ACCOUNT"
    assert initial["reservation"] == ""
    assert initial["ct_nanoparticle_indices"] == "0-1"
    assert initial["ct_adsorbate_indices"] == "2-3"


@pytest.mark.django_db
def test_charge_transfer_request_submits_directly_with_resolved_adsorbates(
    client, monkeypatch
):
    profile = CalculationSettingsProfile.objects.create(name="ct_quick_profile")
    workflow = WorkflowDefinition.objects.create(
        name="ct_quick_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="ct_quick_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_quick_source",
        status="completed",
        working_directory="/work/user/runs/source",
        settings_profile=profile,
        metadata={
            "submission": {
                "account": "account",
                "partition": "compute",
                "remote_runs_root": "/work/user/runs",
                "scratch_base": "/scratch/user",
            }
        },
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind=ArtifactKind.STRUCTURE,
        label="final",
        path="reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )
    JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="123",
        job_name=run.run_name,
        status=JobStatus.COMPLETED,
        remote_workdir="/work/user/runs/source",
        stdout_path="/work/user/runs/source/workflow.log",
    )
    captured = {}
    atoms = Atoms(
        "AuAuCOO",
        positions=[(0, 0, 0), (0, 0, 2.8), (4, 0, 0), (5.13, 0, 0), (0, 4, 0)],
    )
    monkeypatch.setattr(
        JobSubmitView,
        "_prefill_source_from_query",
        lambda self: {
            "run": run,
            "artifact": artifact,
            "job": run.jobs.select_related("hpc_system").latest("pk"),
        },
    )
    monkeypatch.setattr(
        "jobs.views._resolve_existing_run_structure_atoms",
        lambda **kwargs: (atoms, {}),
    )

    def fake_submit(**kwargs):
        captured.update(kwargs["charge_transfer_config"])
        captured["run_name"] = kwargs["run_name"]
        followup = WorkflowRun.objects.create(
            workflow=workflow,
            run_name=kwargs["run_name"],
            status="initiated",
            working_directory="/work/user/runs/ct",
            settings_profile=profile,
        )
        job = JobRecord.objects.create(
            workflow_run=followup,
            hpc_system=system,
            scheduler_job_id="124",
            job_name=followup.run_name,
            status=JobStatus.INITIATED,
            remote_workdir=followup.working_directory,
            stdout_path=f"{followup.working_directory}/workflow.log",
        )
        return SimpleNamespace(
            workflow_run_id=followup.pk,
            job_record_id=job.pk,
            scheduler_job_id=job.scheduler_job_id,
            remote_run_dir=followup.working_directory,
        )

    monkeypatch.setattr("jobs.views.submit_slurm_workflow_run", fake_submit)

    response = client.post(
        f"/jobs/submit/?from_run={run.pk}&mode=charge_transfer",
        {"quick_charge_transfer": "1"},
    )

    assert response.status_code == 302
    assert captured["run_name"].startswith(f"{run.run_name}_ct_")
    assert captured["fragments_0based"] == {"1": [0, 1], "2": [2, 3], "3": [4]}
    assert captured["fragments_1based"] == {"1": "1-2", "2": "3-4", "3": "5"}
    assert captured["fragment_labels"] == {"1": "NC", "2": "CO", "3": "O"}


@pytest.mark.django_db
def test_structure_preview_returns_charge_transfer_fragment_defaults(client):
    upload = SimpleUploadedFile(
        "au2co.xyz",
        b"4\nAu2CO\nAu 0 0 0\nAu 0 0 2.5\nC 0 0 4\nO 0 0 5.2\n",
        content_type="chemical/x-xyz",
    )

    response = client.post(
        "/jobs/submit/structure-preview/",
        data={"geometry_source": "upload_file", "structure_file": upload},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ct_fragments"]["nanoparticle_indices"] == "0-1"
    assert payload["ct_fragments"]["adsorbate_indices"] == "2-3"


@pytest.mark.django_db
def test_job_submit_form_builds_combined_optimisation_and_spectroscopy_config():
    profile = CalculationSettingsProfile.objects.create(
        name="combined_spectroscopy_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="",
    )
    workflow = WorkflowDefinition.objects.create(
        name="combined_spectroscopy_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="combined_spectroscopy_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    upload = SimpleUploadedFile(
        "auco.xyz",
        b"2\nAuCO\nAu 0 0 0\nC 0 0 2.0\n",
        content_type="chemical/x-xyz",
    )
    form = JobSubmitForm(
        data={
            "run_name": "combined_spectroscopy",
            "run_mode": "single_optimisation",
            "workflow": workflow.pk,
            "hpc_system": system.pk,
            "settings_profile": profile.pk,
            "geometry_source": "upload_file",
            "reference_intent": "none",
            "qm_backend": "orca",
            "single_cluster_tddft_difference_spectrum": "on",
            "single_charge_transfer_spectrum": "on",
            "single_final_tddft_nroots": "50",
            "ct_nanoparticle_indices": "0",
            "ct_adsorbate_indices": "1",
            "ct_nroots": "50",
            "nodes": "1",
            "tasks_per_node": "4",
            "cpus_per_task": "1",
            "walltime": "24:00:00",
            "account": "SLURM_ACCOUNT",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
        },
        files={"structure_file": upload},
    )

    assert form.is_valid(), form.errors.as_json()
    single_config = form.single_constraint_config()
    ct_config = form.charge_transfer_config()
    assert single_config["final_tddft_spectrum"] is True
    assert single_config["cluster_tddft_difference_spectrum"] is True
    assert single_config["charge_transfer_spectrum"] is True
    assert ct_config["fragments_1based"] == {"1": "1", "2": "2"}
    assert ct_config["nroots"] == 50
    assert "source_run_id" not in ct_config


@pytest.mark.django_db
def test_combined_charge_transfer_summary_merges_into_same_final_tddft_dataset(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="combined_summary_profile")
    workflow = WorkflowDefinition.objects.create(
        name="combined_summary_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="combined_summary",
        working_directory="/work/PROJECT/GROUP/user/combined_summary",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "single_optimisation"}},
    )
    plot_calls = []
    monkeypatch.setattr("ingest.services.write_ct_plot_artifacts", lambda selected: plot_calls.append(selected.pk))

    _ingest_summary_payload(
        run=run,
        summary={
            "tags": ["charge_transfer"],
            "charge_transfer": {
                "nroots": 50,
                "fragments_0based": {"1": [0], "2": [1]},
                "fragments_1based": {"1": "1", "2": "2"},
                "ct_states": [{"state": 1}],
                "ct_terms": [{"state": 1, "term": "1->2", "pct": 25.0}],
                "ct_spectra": [],
            },
        },
    )

    retained = load_spectral_dataset(run, "final_tddft")
    assert retained["ct_terms"][0]["term"] == "1->2"
    assert plot_calls == [run.pk]


@pytest.mark.django_db
def test_charge_transfer_summary_is_retained_for_future_replotting(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="ct_summary_profile")
    workflow = WorkflowDefinition.objects.create(
        name="ct_summary_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_summary",
        working_directory="/work/PROJECT/GROUP/user/ct_summary",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "charge_transfer"}},
    )
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="ct_summary_source",
        working_directory="/work/PROJECT/GROUP/user/ct_summary_source",
        settings_profile=profile,
    )
    plot_calls = []
    monkeypatch.setattr("ingest.services.write_ct_plot_artifacts", lambda selected: plot_calls.append(selected.pk))

    _ingest_summary_payload(
        run=run,
        summary={
            "tags": ["charge_transfer"],
            "source_run_id": source_run.pk,
            "charge_transfer": {
                "nroots": 400,
                "fragments_0based": {"1": [0, 1], "2": [2]},
                "fragments_1based": {"1": "1-2", "2": "3"},
                "ct_states": [{"state": 1, "nm": 500.0, "f": 0.2}],
                "ct_terms": [{"state": 1, "term": "1->2", "pct": 20.0}],
                "ct_spectra": [
                    {
                        "source": "CT_multiple/ET_1to2.txt",
                        "property": "1->2",
                        "points": [{"energy_eV": 2.48, "wavelength_nm": 500.0, "intensity": 0.2}],
                    }
                ],
            },
        },
    )

    retained = load_spectral_dataset(source_run, "final_tddft")
    assert retained["fragments"]["fragments_1based"] == {"1": "1-2", "2": "3"}
    assert retained["ct_states"][0]["wavelength_nm"] == 500.0
    assert retained["ct_states"][0]["oscillator_strength"] == 0.2
    assert retained["ct_terms"][0]["percentage"] == 20.0
    assert retained["ct_spectra"][0]["points"][0]["wavelength_nm"] == 500.0
    run.refresh_from_db()
    assert "charge_transfer" in run.metadata["tags"]
    assert run.metadata["result_summary"]["charge_transfer"]["ct_spectra_count"] == 1
    source_run.refresh_from_db()
    assert "charge_transfer" in source_run.metadata["tags"]
    assert source_run.metadata["charge_transfer_run_ids"] == [run.pk]
    assert plot_calls == [source_run.pk]


@pytest.mark.django_db
def test_job_submit_form_accepts_nwchem_single_optimisation():
    profile = CalculationSettingsProfile.objects.create(
        name="nwchem_form_profile",
        xc_functional="PBE0",
        basis_set="def2-SVP",
        solvent="",
        metadata={"qm_backend": "nwchem", "nwchem": {"memory_mb": 4000}},
    )
    workflow = WorkflowDefinition.objects.create(
        name="nwchem_form_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    upload = SimpleUploadedFile(
        "co.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "nwchem_submit_test",
            "run_mode": "single_optimisation",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "qm_backend": "nwchem",
            "single_final_tddft_spectrum": "on",
            "single_cluster_tddft_difference_spectrum": "on",
            "single_final_tddft_nroots": "10",
            "nodes": "1",
            "tasks_per_node": "4",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "01:00:00",
            "account": "FALCON_SLURM_ACCOUNT",
            "partition": "",
            "qos": "",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "/tmp",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={"structure_file": upload},
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["qm_backend"] == "nwchem"
    assert form.cleaned_data["single_cluster_tddft_difference_spectrum"] is True


@pytest.mark.django_db
def test_job_submit_form_rejects_nwchem_tddft_bond_scan():
    profile = CalculationSettingsProfile.objects.create(
        name="nwchem_reject_scan_profile",
        metadata={"qm_backend": "nwchem"},
    )
    workflow = WorkflowDefinition.objects.create(
        name="nwchem_reject_scan_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    upload = SimpleUploadedFile(
        "co.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "nwchem_scan_reject",
            "run_mode": "tddft_bond_scan",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "qm_backend": "nwchem",
            "target_wavelength_nm": "520",
            "tddft_scan_nroots": "20",
            "bond_i": "0",
            "bond_j": "1",
            "increment": "0.05",
            "fix": "0.5",
            "nodes": "1",
            "tasks_per_node": "4",
            "cpus_per_task": "1",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "/tmp",
        },
        files={"structure_file": upload},
    )

    assert not form.is_valid()
    assert "NWChem online submission is currently limited" in form.errors.as_text()


@pytest.mark.django_db
def test_job_submit_context_prefills_falcon_account_and_pythonpath(client):
    resp = client.get("/jobs/submit/")

    assert resp.status_code == 200
    falcon = HPCSystem.objects.filter(name="FALCON").first()
    assert falcon is not None
    hpc_prefill = json.loads(resp.context["hpc_prefill_json"])
    falcon_prefill = hpc_prefill.get(str(falcon.pk), {})
    assert falcon_prefill.get("account") == "FALCON_SLURM_ACCOUNT"
    assert falcon_prefill.get("partition") == "compute"
    assert "/scratch/PROJECT/shared/software/github/cplas" in str(
        falcon_prefill.get("pythonpath_entries", "")
    )


@pytest.mark.django_db
def test_job_submit_context_prefills_trh408_pythonpath_with_carmm(client):
    resp = client.get("/jobs/submit/")

    assert resp.status_code == 200
    trh408 = HPCSystem.objects.filter(name="TRH408").first()
    assert trh408 is not None
    hpc_prefill = json.loads(resp.context["hpc_prefill_json"])
    trh_prefill = hpc_prefill.get(str(trh408.pk), {})
    assert "/srv/research/github/carmm" in str(trh_prefill.get("pythonpath_entries", ""))


@pytest.mark.django_db
def test_job_submit_form_default_generation_site_symbols_include_oxygen():
    form = JobSubmitForm()

    assert str(form.fields["coadsorbate_site_symbols"].initial) == "Ti,Au,Cu,O"


@pytest.mark.django_db
def test_job_submit_form_active_learning_neb_mode_validates_initial_and_final_images():
    profile = CalculationSettingsProfile.objects.create(name="neb_form_profile")
    workflow = WorkflowDefinition.objects.create(
        name="neb_form_workflow",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    initial_upload = SimpleUploadedFile(
        "initial.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        content_type="chemical/x-xyz",
    )
    final_upload = SimpleUploadedFile(
        "final.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.3\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "neb_submit_test",
            "run_mode": "active_learning_neb",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "fixed_symbols": "Au,Cu",
            "neb_final_geometry_source": "upload_file",
            "neb_final_traj_frame": "-1",
            "neb_backend": "mace",
            "neb_atom_shift_per_image_A": "0.2",
            "neb_min_images": "7",
            "neb_max_images": "21",
            "neb_spring_constant": "0.1",
            "neb_max_steps": "300",
            "neb_climb": "on",
            "neb_split_on_minimum": "on",
            "neb_minimum_energy_window_eV": "0.02",
            "neb_mace_model_path": "/srv/research/models/mace.model",
            "neb_mace_device": "cuda",
            "neb_mace_default_dtype": "float64",
            "neb_catlearn_min_data": "3",
            "neb_catlearn_unc_convergence": "0.05",
            "neb_catlearn_max_unc": "0.03",
            "neb_catlearn_steps": "200",
            "neb_catlearn_ml_steps": "500",
            "neb_catlearn_dtrust": "2.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "8",
            "slurm_preset": "custom",
            "walltime": "08:00:00",
            "account": "",
            "partition": "",
            "qos": "",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={
            "structure_file": initial_upload,
            "neb_final_structure_file": final_upload,
        },
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["run_mode"] == "active_learning_neb"
    assert form.cleaned_data["submit_all_traj_frames"] is False
    neb_cfg = form.neb_config()
    assert neb_cfg["backend"] == "mace"
    assert neb_cfg["final_image_strategy"] == "explicit_final_image"
    assert neb_cfg["fixed_symbols"] == ("Au", "Cu")
    assert neb_cfg["atom_shift_per_image_A"] == 0.2


@pytest.mark.django_db
def test_job_submit_form_tddft_mlneb_accepts_tddft_backend_and_state_controls():
    profile = CalculationSettingsProfile.objects.create(name="tddft_mlneb_form_profile")
    workflow = WorkflowDefinition.objects.create(
        name="tddft_mlneb_form_workflow",
        code_path="cats/working_directory/tddft_mlneb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    initial_upload = SimpleUploadedFile(
        "initial.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        content_type="chemical/x-xyz",
    )
    final_upload = SimpleUploadedFile(
        "final.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.3\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "tddft_mlneb_submit_test",
            "run_mode": "active_learning_neb",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "tddft_scan_nroots": "50",
            "neb_final_geometry_source": "upload_file",
            "neb_final_traj_frame": "-1",
            "neb_backend": "tddft_mlneb",
            "neb_atom_shift_per_image_A": "0.2",
            "neb_min_images": "7",
            "neb_max_images": "21",
            "neb_spring_constant": "0.1",
            "neb_max_steps": "300",
            "neb_climb": "on",
            "neb_split_on_minimum": "on",
            "neb_minimum_energy_window_eV": "0.02",
            "neb_catlearn_min_data": "3",
            "neb_catlearn_unc_convergence": "0.05",
            "neb_catlearn_max_unc": "0.03",
            "neb_catlearn_steps": "200",
            "neb_catlearn_ml_steps": "500",
            "neb_catlearn_dtrust": "2.0",
            "nodes": "2",
            "tasks_per_node": "32",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "24:00:00",
            "account": "",
            "partition": "",
            "qos": "",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "/tmp",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={
            "structure_file": initial_upload,
            "neb_final_structure_file": final_upload,
        },
    )

    assert form.is_valid(), form.errors.as_json()
    neb_cfg = form.neb_config()
    assert neb_cfg["backend"] == "tddft_mlneb"
    assert neb_cfg["tddft_scan_nroots"] == 50
    assert form.cleaned_data["target_wavelength_nm"] == 520.0


@pytest.mark.django_db
def test_job_submit_form_rejects_tddft_backend_for_mace_neb_workflow():
    profile = CalculationSettingsProfile.objects.create(name="mace_neb_tddft_backend_profile")
    workflow = WorkflowDefinition.objects.create(
        name="mace_neb_tddft_backend_workflow",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    initial_upload = SimpleUploadedFile(
        "initial.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        content_type="chemical/x-xyz",
    )
    final_upload = SimpleUploadedFile(
        "final.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.3\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "reject_tddft_backend",
            "run_mode": "active_learning_neb",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "neb_final_geometry_source": "upload_file",
            "neb_final_traj_frame": "-1",
            "neb_backend": "tddft_mlneb",
            "neb_atom_shift_per_image_A": "0.2",
            "neb_min_images": "7",
            "neb_max_images": "21",
            "neb_spring_constant": "0.1",
            "neb_max_steps": "300",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "/tmp",
        },
        files={
            "structure_file": initial_upload,
            "neb_final_structure_file": final_upload,
        },
    )

    assert not form.is_valid()
    assert "TDDFT MLNEB backend requires the TDDFT MLNEB workflow" in form.errors.as_text()


@pytest.mark.django_db
def test_job_submit_form_allows_dft_backend_for_regular_neb_workflow():
    profile = CalculationSettingsProfile.objects.create(name="dft_neb_backend_profile")
    workflow = WorkflowDefinition.objects.create(
        name="dft_neb_backend_workflow",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    initial_upload = SimpleUploadedFile(
        "initial.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        content_type="chemical/x-xyz",
    )
    final_upload = SimpleUploadedFile(
        "final.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.3\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "dft_neb_submit_test",
            "run_mode": "active_learning_neb",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "neb_final_geometry_source": "upload_file",
            "neb_final_traj_frame": "-1",
            "neb_backend": "dft",
            "neb_atom_shift_per_image_A": "0.2",
            "neb_min_images": "7",
            "neb_max_images": "21",
            "neb_spring_constant": "0.1",
            "neb_max_steps": "300",
            "neb_catlearn_min_data": "3",
            "neb_catlearn_unc_convergence": "0.05",
            "neb_catlearn_max_unc": "0.03",
            "neb_catlearn_steps": "200",
            "neb_catlearn_ml_steps": "500",
            "neb_catlearn_dtrust": "2.0",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "/tmp",
        },
        files={
            "structure_file": initial_upload,
            "neb_final_structure_file": final_upload,
        },
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.neb_config()["backend"] == "dft"


@pytest.mark.django_db
def test_job_submit_form_active_learning_neb_mode_validates_bond_extension_final_strategy():
    profile = CalculationSettingsProfile.objects.create(name="neb_form_bond_extension_profile")
    workflow = WorkflowDefinition.objects.create(
        name="neb_form_bond_extension_workflow",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    initial_upload = SimpleUploadedFile(
        "initial.xyz",
        b"3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.92 0 0.0\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "neb_bond_extension_submit_test",
            "run_mode": "active_learning_neb",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "fixed_symbols": "",
            "neb_final_image_strategy": "set_bond_distance_from_initial",
            "neb_optimize_initial_image": "on",
            "neb_optimize_final_image": "on",
            "neb_endpoint_opt_steps": "150",
            "neb_endpoint_opt_fmax": "0.04",
            "neb_break_bond_i": "0",
            "neb_break_bond_j": "1",
            "neb_break_bond_extension_A": "1.8",
            "neb_break_bond_fix": "0.5",
            "neb_break_assert_broken": "on",
            "neb_break_cutoff_scale": "1.2",
            "neb_backend": "mace",
            "neb_atom_shift_per_image_A": "0.2",
            "neb_min_images": "7",
            "neb_max_images": "21",
            "neb_spring_constant": "0.1",
            "neb_max_steps": "300",
            "neb_climb": "on",
            "neb_split_on_minimum": "on",
            "neb_minimum_energy_window_eV": "0.02",
            "neb_mace_model_path": "/srv/research/models/mace.model",
            "neb_mace_device": "cuda",
            "neb_mace_default_dtype": "float64",
            "neb_catlearn_min_data": "3",
            "neb_catlearn_unc_convergence": "0.05",
            "neb_catlearn_max_unc": "0.03",
            "neb_catlearn_steps": "200",
            "neb_catlearn_ml_steps": "500",
            "neb_catlearn_dtrust": "2.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "8",
            "slurm_preset": "custom",
            "walltime": "08:00:00",
            "account": "",
            "partition": "",
            "qos": "",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={
            "structure_file": initial_upload,
        },
    )

    assert form.is_valid(), form.errors.as_json()
    neb_cfg = form.neb_config()
    assert neb_cfg["final_image_strategy"] == "set_bond_distance_from_initial"
    assert neb_cfg["optimize_initial_image"] is True
    assert neb_cfg["optimize_final_image"] is True
    assert neb_cfg["break_bond_i"] == 0
    assert neb_cfg["break_bond_j"] == 1
    assert neb_cfg["set_bond_distance_A"] == 1.8
    assert neb_cfg["endpoint_opt_steps"] == 150
    assert neb_cfg["endpoint_opt_fmax"] == 0.04


@pytest.mark.django_db
def test_job_submit_form_active_learning_neb_mode_allows_existing_run_initial_source(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="neb_existing_initial_profile")
    workflow = WorkflowDefinition.objects.create(
        name="neb_existing_initial_workflow",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    source_dir = tmp_path / "neb-existing-initial-source"
    source_dir.mkdir(parents=True)
    (source_dir / "initial.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.2\n", encoding="utf-8")
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="neb_existing_initial_source_run",
        working_directory=str(source_dir),
        status="completed",
        settings_profile=profile,
    )
    source_artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind=ArtifactKind.STRUCTURE,
        label="final",
        path="initial.xyz",
    )
    final_upload = SimpleUploadedFile(
        "final.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.5\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "neb_existing_initial_submit_test",
            "run_mode": "active_learning_neb",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "existing_run",
            "source_run_id": str(source_run.pk),
            "source_artifact_id": str(source_artifact.pk),
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "neb_final_image_strategy": "explicit_final_image",
            "neb_final_geometry_source": "upload_file",
            "neb_backend": "mace",
            "neb_atom_shift_per_image_A": "0.2",
            "neb_min_images": "7",
            "neb_max_images": "21",
            "neb_spring_constant": "0.1",
            "neb_max_steps": "300",
            "neb_climb": "on",
            "neb_split_on_minimum": "on",
            "neb_minimum_energy_window_eV": "0.02",
            "neb_mace_model_path": "/srv/research/models/mace.model",
            "neb_mace_device": "cuda",
            "neb_mace_default_dtype": "float64",
            "neb_catlearn_min_data": "3",
            "neb_catlearn_unc_convergence": "0.05",
            "neb_catlearn_max_unc": "0.03",
            "neb_catlearn_steps": "200",
            "neb_catlearn_ml_steps": "500",
            "neb_catlearn_dtrust": "2.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "8",
            "slurm_preset": "custom",
            "walltime": "08:00:00",
            "account": "",
            "partition": "",
            "qos": "",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={
            "neb_final_structure_file": final_upload,
        },
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["geometry_source"] == "existing_run"
    assert form.cleaned_data["source_run_id"] == source_run.pk
    assert form.cleaned_data["source_artifact_id"] == source_artifact.pk


@pytest.mark.django_db
def test_job_submit_form_active_learning_neb_mode_allows_existing_run_final_source_fallback(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="neb_existing_final_profile")
    workflow = WorkflowDefinition.objects.create(
        name="neb_existing_final_workflow",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    source_dir = tmp_path / "neb-existing-final-source"
    source_dir.mkdir(parents=True)
    (source_dir / "state.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.2\n", encoding="utf-8")
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="neb_existing_final_source_run",
        working_directory=str(source_dir),
        status="completed",
        settings_profile=profile,
    )
    source_artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind=ArtifactKind.STRUCTURE,
        label="state",
        path="state.xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "neb_existing_final_submit_test",
            "run_mode": "active_learning_neb",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "existing_run",
            "source_run_id": str(source_run.pk),
            "source_artifact_id": str(source_artifact.pk),
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "neb_final_image_strategy": "explicit_final_image",
            "neb_final_geometry_source": "existing_run",
            "neb_backend": "mace",
            "neb_atom_shift_per_image_A": "0.2",
            "neb_min_images": "7",
            "neb_max_images": "21",
            "neb_spring_constant": "0.1",
            "neb_max_steps": "300",
            "neb_climb": "on",
            "neb_split_on_minimum": "on",
            "neb_minimum_energy_window_eV": "0.02",
            "neb_mace_model_path": "/srv/research/models/mace.model",
            "neb_mace_device": "cuda",
            "neb_mace_default_dtype": "float64",
            "neb_catlearn_min_data": "3",
            "neb_catlearn_unc_convergence": "0.05",
            "neb_catlearn_max_unc": "0.03",
            "neb_catlearn_steps": "200",
            "neb_catlearn_ml_steps": "500",
            "neb_catlearn_dtrust": "2.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "8",
            "slurm_preset": "custom",
            "walltime": "08:00:00",
            "account": "",
            "partition": "",
            "qos": "",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "",
            "module_loads": "",
            "pythonpath_entries": "",
        },
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["neb_final_geometry_source"] == "existing_run"
    assert form.cleaned_data["neb_final_source_run_id"] == source_run.pk
    assert form.cleaned_data["neb_final_source_artifact_id"] == source_artifact.pk


@pytest.mark.django_db
def test_job_submit_form_active_learning_neb_mode_allows_existing_run_final_source_with_uploaded_initial(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="neb_existing_final_upload_initial_profile")
    workflow = WorkflowDefinition.objects.create(
        name="neb_existing_final_upload_initial_workflow",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    source_dir = tmp_path / "neb-existing-final-upload-initial-source"
    source_dir.mkdir(parents=True)
    (source_dir / "state.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.2\n", encoding="utf-8")
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="neb_existing_final_upload_initial_source_run",
        working_directory=str(source_dir),
        status="completed",
        settings_profile=profile,
    )
    source_artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind=ArtifactKind.STRUCTURE,
        label="state",
        path="state.xyz",
    )
    initial_upload = SimpleUploadedFile(
        "initial.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.4\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "neb_existing_final_upload_initial_submit_test",
            "run_mode": "active_learning_neb",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "source_run_id": str(source_run.pk),
            "source_artifact_id": str(source_artifact.pk),
            "traj_frame": "-1",
            "fmax": "0.05",
            "active_learning_optimizer": "bfgs",
            "neb_final_image_strategy": "explicit_final_image",
            "neb_final_geometry_source": "existing_run",
            "neb_backend": "mace",
            "neb_atom_shift_per_image_A": "0.2",
            "neb_min_images": "7",
            "neb_max_images": "21",
            "neb_spring_constant": "0.1",
            "neb_max_steps": "300",
            "neb_climb": "on",
            "neb_split_on_minimum": "on",
            "neb_minimum_energy_window_eV": "0.02",
            "neb_mace_model_path": "/srv/research/models/mace.model",
            "neb_mace_device": "cuda",
            "neb_mace_default_dtype": "float64",
            "neb_catlearn_min_data": "3",
            "neb_catlearn_unc_convergence": "0.05",
            "neb_catlearn_max_unc": "0.03",
            "neb_catlearn_steps": "200",
            "neb_catlearn_ml_steps": "500",
            "neb_catlearn_dtrust": "2.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "8",
            "slurm_preset": "custom",
            "walltime": "08:00:00",
            "account": "",
            "partition": "",
            "qos": "",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={
            "structure_file": initial_upload,
        },
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["geometry_source"] == "upload_file"
    assert form.cleaned_data["neb_final_geometry_source"] == "existing_run"
    assert form.cleaned_data["neb_final_source_run_id"] == source_run.pk
    assert form.cleaned_data["neb_final_source_artifact_id"] == source_artifact.pk


@pytest.mark.django_db
def test_job_submit_form_allows_existing_run_batch_from_database_artifact(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="existing_run_db_profile")
    workflow = WorkflowDefinition.objects.create(name="existing_run_db_workflow", code_path="x.py")
    system = HPCSystem.objects.create(
        name="existing_run_db_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    run_dir = tmp_path / "existing-run-db"
    run_dir.mkdir(parents=True)
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="existing-run-db-source",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind=ArtifactKind.DATABASE,
        label="coadsorption_db:coadsorption_final_for_dft.db",
        path="scratch/coadsorption_final_for_dft.db",
    )

    form = JobSubmitForm(
        data={
            "run_name": "batch_from_existing_db",
            "run_mode": "single_optimisation",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "existing_run",
            "source_run_id": str(source_run.pk),
            "source_artifact_id": str(artifact.pk),
            "submit_all_traj_frames": "on",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "01:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "standard",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        }
    )

    assert form.is_valid(), form.errors.as_json()
    assert bool(form.cleaned_data.get("submit_all_traj_frames")) is True


@pytest.mark.django_db
def test_job_submit_form_parses_tddft_target_wavelength_batch(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="tddft_wavelength_batch_profile")
    workflow = WorkflowDefinition.objects.create(
        name="tddft_wavelength_batch_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="tddft_wavelength_batch_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    upload = SimpleUploadedFile(
        "scan_input.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.2\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "co_scan_batch",
            "run_mode": "tddft_bond_scan",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "n_iter": "4",
            "use_tddft_optimisation": "on",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "target_wavelengths_nm": "500, 600, 700",
            "tddft_scan_nroots": "50",
            "bond_i": "0",
            "bond_j": "1",
            "increment": "0.05",
            "fix": "0.5",
            "fixed_symbols": "Cu",
            "distance_change_fraction": "1.0",
            "refine_points_each_side": "0",
            "restart_extend_steps": "5",
            "restart_extend_increment_A": "0.2",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "04:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "standard",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={"structure_file": upload},
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["target_wavelength_nm"] == 500.0
    assert form.cleaned_data["target_wavelengths_nm"] == [500.0, 600.0, 700.0]
    scan_config = form.scan_config()
    assert scan_config["restart_extend_steps"] == 5
    assert scan_config["restart_extend_increment_A"] == pytest.approx(0.2)


@pytest.mark.django_db
def test_job_submit_form_rejects_zero_restart_extension_increment(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="tddft_restart_extension_profile")
    workflow = WorkflowDefinition.objects.create(
        name="tddft_restart_extension_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="tddft_restart_extension_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    upload = SimpleUploadedFile(
        "scan_input.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.2\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "co_scan_restart_extension",
            "run_mode": "tddft_bond_scan",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "n_iter": "4",
            "use_tddft_optimisation": "on",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "tddft_scan_nroots": "50",
            "bond_i": "0",
            "bond_j": "1",
            "increment": "0.0",
            "fix": "0.5",
            "fixed_symbols": "Cu",
            "target_distance": "2.0",
            "distance_change_fraction": "1.0",
            "refine_points_each_side": "0",
            "restart_extend_steps": "3",
            "restart_extend_increment_A": "0.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "04:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "standard",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={"structure_file": upload},
    )

    assert not form.is_valid()
    assert "__all__" in form.errors
    assert "Restart extension increment cannot be zero" in form.errors["__all__"][0]


@pytest.mark.django_db
def test_job_submit_form_rejects_restart_extension_increment_without_steps(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="tddft_restart_extension_without_steps_profile")
    workflow = WorkflowDefinition.objects.create(
        name="tddft_restart_extension_without_steps_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="tddft_restart_extension_without_steps_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    upload = SimpleUploadedFile(
        "scan_input.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.2\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "co_scan_restart_extension_without_steps",
            "run_mode": "tddft_bond_scan",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "n_iter": "4",
            "use_tddft_optimisation": "on",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "tddft_scan_nroots": "50",
            "bond_i": "0",
            "bond_j": "1",
            "increment": "0.05",
            "fix": "0.5",
            "fixed_symbols": "Cu",
            "distance_change_fraction": "1.0",
            "refine_points_each_side": "0",
            "restart_extend_steps": "0",
            "restart_extend_increment_A": "0.2",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "04:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "standard",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={"structure_file": upload},
    )

    assert not form.is_valid()
    assert "__all__" in form.errors
    assert "Restart extension increment requires restart extension steps greater than zero." in form.errors["__all__"][0]


@pytest.mark.django_db
def test_job_submit_form_rejects_duplicate_tddft_target_wavelength_batch_values(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="tddft_wavelength_batch_duplicate_profile")
    workflow = WorkflowDefinition.objects.create(
        name="tddft_wavelength_batch_duplicate_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="tddft_wavelength_batch_duplicate_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    upload = SimpleUploadedFile(
        "scan_input.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.2\n",
        content_type="chemical/x-xyz",
    )

    form = JobSubmitForm(
        data={
            "run_name": "co_scan_batch_duplicate",
            "run_mode": "tddft_bond_scan",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "n_iter": "4",
            "use_tddft_optimisation": "on",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "target_wavelengths_nm": "500, 500.0",
            "tddft_scan_nroots": "50",
            "bond_i": "0",
            "bond_j": "1",
            "increment": "0.05",
            "fix": "0.5",
            "fixed_symbols": "Cu",
            "distance_change_fraction": "1.0",
            "refine_points_each_side": "0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "04:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "standard",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={"structure_file": upload},
    )

    assert not form.is_valid()
    assert "target_wavelengths_nm" in form.errors
    assert "Duplicate target wavelength value" in form.errors["target_wavelengths_nm"][0]


@pytest.mark.django_db
def test_job_submit_form_validates_coadsorbate_mode_with_reference_cluster_and_adsorbates():
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_form_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_form_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="coadsorbate_form_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    cluster = ReferenceNanocluster.objects.create(
        name="coadsorbate_cluster_ref",
        composition="Au13",
        settings_profile=profile,
    )

    form = JobSubmitForm(
        data={
            "run_name": "coadsorbate_submit",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "coadsorbate_reference_cluster": str(cluster.pk),
            "coadsorbate_ase_molecules": "CO\nNO2",
            "coadsorbate_pubchem_cids": "1805",
            "coadsorbate_pubchem_queries": "methanol",
            "coadsorbate_mace_model_path": "/work/PROJECT/GROUP/user/models/mace.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_mace_dispersion": "on",
            "coadsorbate_mace_enable_cueq": "on",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "gpu",
            "qos": "gpu-shd",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        }
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["n_iter"] == 1
    assert form.cleaned_data["use_tddft_optimisation"] is False
    assert form.cleaned_data["target_wavelength_nm"] is None
    assert form.cleaned_data["coadsorbate_ase_molecules"] == ["CO", "NO2"]
    assert form.cleaned_data["coadsorbate_pubchem_cids"] == [1805]
    assert form.cleaned_data["coadsorbate_pubchem_queries"] == ["methanol"]
    assert form.coadsorbate_config()["mace_dispersion"] is True
    assert form.coadsorbate_config()["mace_enable_cueq"] is True


@pytest.mark.django_db
def test_job_submit_form_accepts_poscar_filename_for_upload_geometry():
    profile = CalculationSettingsProfile.objects.create(name="upload_poscar_profile")
    workflow = WorkflowDefinition.objects.create(name="upload_poscar_workflow", code_path="x.py")
    system = HPCSystem.objects.create(
        name="upload_poscar_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    upload = SimpleUploadedFile(
        "POSCAR",
        b"placeholder",
        content_type="application/octet-stream",
    )

    form = JobSubmitForm(
        data={
            "run_name": "upload_poscar_submit",
            "run_mode": "single_optimisation",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "traj_frame": "-1",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "01:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "standard",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={"structure_file": upload},
    )

    assert form.is_valid(), form.errors.as_json()
    assert str(getattr(form.cleaned_data.get("structure_file"), "name", "")) == "POSCAR"


@pytest.mark.django_db
def test_job_submit_form_accepts_contcar_filename_for_coadsorbate_adsorbate_upload():
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_upload_contcar_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_upload_contcar_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="coadsorbate_upload_contcar_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    cluster = ReferenceNanocluster.objects.create(
        name="coadsorbate_upload_contcar_cluster",
        composition="Au13",
        settings_profile=profile,
    )
    upload = SimpleUploadedFile(
        "adsorbate_CONTCAR",
        b"placeholder",
        content_type="application/octet-stream",
    )

    form = JobSubmitForm(
        data={
            "run_name": "coadsorbate_upload_contcar_submit",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "coadsorbate_reference_cluster": str(cluster.pk),
            "coadsorbate_ase_molecules": "",
            "coadsorbate_pubchem_cids": "",
            "coadsorbate_pubchem_queries": "",
            "coadsorbate_mace_model_path": "/work/PROJECT/GROUP/user/models/mace.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "gpu",
            "qos": "gpu-shd",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        },
        files={"coadsorbate_structure_files": upload},
    )

    assert form.is_valid(), form.errors.as_json()
    uploaded_files = list(form.cleaned_data.get("coadsorbate_structure_files") or [])
    assert len(uploaded_files) == 1
    assert str(getattr(uploaded_files[0], "name", "")) == "adsorbate_CONTCAR"


@pytest.mark.django_db
def test_job_submit_form_coadsorbate_config_includes_periodic_surface_controls():
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_periodic_controls_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_periodic_controls_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="coadsorbate_periodic_controls_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    cluster = ReferenceNanocluster.objects.create(
        name="coadsorbate_periodic_controls_cluster",
        composition="Au13",
        settings_profile=profile,
    )

    form = JobSubmitForm(
        data={
            "run_name": "coadsorbate_periodic_controls_submit",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "coadsorbate_reference_cluster": str(cluster.pk),
            "coadsorbate_ase_molecules": "CO",
            "coadsorbate_pubchem_cids": "",
            "coadsorbate_pubchem_queries": "",
            "coadsorbate_mace_model_path": "/work/PROJECT/GROUP/user/models/mace.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_generation_z_cutoff": "4.5",
            "coadsorbate_include_below_surface": "on",
            "coadsorbate_origin_atom_index": "7",
            "coadsorbate_site_cutoff": "3.25",
            "coadsorbate_keep_bound_only": "on",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "coadsorbate_similarity_slab_shave_distance": "1.25",
            "coadsorbate_site_probe_symbol": "He",
            "coadsorbate_site_probe_similarity_cutoff": "0.97",
            "coadsorbate_single_mode_duplicate_cutoff": "0.96",
            "coadsorbate_candidate_duplicate_cutoff": "0.95",
            "coadsorbate_relaxed_minimum_duplicate_cutoff": "0.94",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "gpu",
            "qos": "gpu-shd",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        }
    )

    assert form.is_valid(), form.errors.as_json()
    config = form.coadsorbate_config()
    assert config.get("generation_z_cutoff") == 4.5
    assert config.get("generation_include_below_surface") is True
    assert config.get("generation_origin_atom_index") == 7
    assert config.get("generation_site_cutoff") == 3.25
    assert config.get("keep_bound_only") is True
    assert config.get("similarity_slab_shave_distance") == 1.25
    assert config.get("site_probe_symbol") == "He"
    assert config.get("site_probe_similarity_cutoff") == 0.97
    assert config.get("single_mode_duplicate_cutoff") == 0.96
    assert config.get("candidate_duplicate_cutoff") == 0.95
    assert config.get("relaxed_minimum_duplicate_cutoff") == 0.94


@pytest.mark.django_db
def test_job_submit_form_allows_blank_account_for_trh408():
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_form_trh408_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_form_trh408_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh408.example.invalid",
        username="user",
    )
    cluster = ReferenceNanocluster.objects.create(
        name="coadsorbate_cluster_trh408_ref",
        composition="Au13",
        settings_profile=profile,
    )

    form = JobSubmitForm(
        data={
            "run_name": "coadsorbate_submit_trh408",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "coadsorbate_reference_cluster": str(cluster.pk),
            "coadsorbate_ase_molecules": "CO",
            "coadsorbate_pubchem_cids": "",
            "coadsorbate_pubchem_queries": "",
            "coadsorbate_mace_model_path": "/srv/research/mace_models/mace-omat-0-medium.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "8",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "",
            "partition": "",
            "qos": "",
            "remote_runs_root": "/home/{username}/reaction_hub_runs",
            "scratch_base": "",
            "module_loads": "",
            "pythonpath_entries": "/srv/research/github/cplas",
        }
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["account"] == ""


@pytest.mark.django_db
def test_job_submit_form_accepts_reference_adsorbates_matching_settings_profile():
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_form_ref_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_form_ref_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="coadsorbate_form_ref_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    cluster = ReferenceNanocluster.objects.create(
        name="coadsorbate_form_ref_cluster",
        composition="Au13",
        settings_profile=profile,
    )
    adsorbate = ReferenceAdsorbate.objects.create(
        name="coadsorbate_form_ref_adsorbate",
        formula="CO",
        settings_profile=profile,
    )

    form = JobSubmitForm(
        data={
            "run_name": "coadsorbate_submit_ref_adsorbate",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "coadsorbate_reference_cluster": str(cluster.pk),
            "coadsorbate_reference_adsorbates": [str(adsorbate.pk)],
            "coadsorbate_ase_molecules": "",
            "coadsorbate_pubchem_cids": "",
            "coadsorbate_pubchem_queries": "",
            "coadsorbate_mace_model_path": "/work/PROJECT/GROUP/user/models/mace.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "gpu",
            "qos": "gpu-shd",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        }
    )

    assert form.is_valid(), form.errors.as_json()
    selected_adsorbates = list(form.cleaned_data.get("coadsorbate_reference_adsorbates") or [])
    assert [int(item.pk) for item in selected_adsorbates] == [adsorbate.pk]


@pytest.mark.django_db
def test_job_submit_form_rejects_reference_adsorbates_from_different_settings_profile():
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_form_ref_profile_main")
    other_profile = CalculationSettingsProfile.objects.create(name="coadsorbate_form_ref_profile_other")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_form_ref_profile_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="coadsorbate_form_ref_profile_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    cluster = ReferenceNanocluster.objects.create(
        name="coadsorbate_form_ref_profile_cluster",
        composition="Au13",
        settings_profile=profile,
    )
    mismatched_adsorbate = ReferenceAdsorbate.objects.create(
        name="coadsorbate_form_ref_profile_adsorbate",
        formula="CO",
        settings_profile=other_profile,
    )

    form = JobSubmitForm(
        data={
            "run_name": "coadsorbate_submit_ref_adsorbate_mismatch",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "coadsorbate_reference_cluster": str(cluster.pk),
            "coadsorbate_reference_adsorbates": [str(mismatched_adsorbate.pk)],
            "coadsorbate_ase_molecules": "",
            "coadsorbate_pubchem_cids": "",
            "coadsorbate_pubchem_queries": "",
            "coadsorbate_mace_model_path": "/work/PROJECT/GROUP/user/models/mace.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "gpu",
            "qos": "gpu-shd",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        }
    )

    assert not form.is_valid()
    assert "coadsorbate_reference_adsorbates" in form.errors
    assert "Select a valid choice" in form.errors["coadsorbate_reference_adsorbates"][0]


@pytest.mark.django_db
def test_job_submit_form_coadsorbate_forces_reference_intent_none():
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_ref_intent_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_ref_intent_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="coadsorbate_ref_intent_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    cluster = ReferenceNanocluster.objects.create(
        name="coadsorbate_ref_intent_cluster",
        composition="Au13",
        settings_profile=profile,
    )

    form = JobSubmitForm(
        data={
            "run_name": "coadsorbate_ref_intent_submit",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": ReferenceIntent.ADSORBATE,
            "reference_name": "should_not_be_used",
            "geometry_source": "upload_file",
            "coadsorbate_reference_cluster": str(cluster.pk),
            "coadsorbate_ase_molecules": "CO",
            "coadsorbate_pubchem_cids": "",
            "coadsorbate_pubchem_queries": "",
            "coadsorbate_mace_model_path": "/work/PROJECT/GROUP/user/models/mace.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "gpu",
            "qos": "gpu-shd",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        }
    )

    assert form.is_valid(), form.errors.as_json()
    assert form.cleaned_data["reference_intent"] == ReferenceIntent.NONE
    assert form.cleaned_data["reference_name"] == ""


@pytest.mark.django_db
@override_settings(
    REACTION_HUB_COADSORBATE_MACE_MODEL_PATH_BY_HPC={
        "archer2": "/work/PROJECT/GROUP/shared/user/mace-omat-0-medium.model",
        "hawk": "/home/user/mace.model",
    }
)
def test_job_submit_context_prefills_hpc_specific_coadsorbate_mace_model_path(client):
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_hpc_prefill_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_hpc_prefill_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    del profile, workflow

    archer = HPCSystem.objects.create(
        name="ARCHER2",
        scheduler_type="slurm",
        host="example.org",
        username="archer_user",
        is_active=True,
    )
    hawk = HPCSystem.objects.create(
        name="Hawk",
        scheduler_type="slurm",
        host="example.org",
        username="hawk_user",
        is_active=True,
    )

    resp = client.get("/jobs/submit/")

    assert resp.status_code == 200
    hpc_prefill = json.loads(resp.context["hpc_prefill_json"])
    assert hpc_prefill[str(archer.pk)]["coadsorbate_mace_model_path"] == (
        "/work/PROJECT/GROUP/shared/user/mace-omat-0-medium.model"
    )
    assert hpc_prefill[str(hawk.pk)]["coadsorbate_mace_model_path"] == "/home/user/mace.model"
    assert hpc_prefill[str(archer.pk)]["neb_mace_model_path"] == (
        "/work/PROJECT/GROUP/shared/user/mace-omat-0-medium.model"
    )
    assert hpc_prefill[str(hawk.pk)]["neb_mace_model_path"] == "/home/user/mace.model"
    workflow_mode_map = json.loads(resp.context["workflow_mode_map_json"])
    workflow_preferred = json.loads(resp.context["workflow_preferred_by_mode_json"])
    assert str(workflow_preferred.get("coadsorbate_placement") or "").strip() != ""
    assert str(workflow_preferred.get("tddft_bond_scan") or "").strip() != ""
    assert str(workflow_preferred.get("single_optimisation") or "").strip() != ""
    assert str(workflow_preferred.get("active_learning_neb") or "").strip() != ""
    assert len(workflow_mode_map.get("coadsorbate_placement", [])) >= 1
    assert len(workflow_mode_map.get("tddft_bond_scan", [])) >= 1
    assert len(workflow_mode_map.get("single_optimisation", [])) >= 1
    assert len(workflow_mode_map.get("active_learning_neb", [])) >= 1


@pytest.mark.django_db
def test_job_submit_coadsorbate_mode_resolves_reference_adsorbate_payload(
    client, monkeypatch, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_submit_ref_payload_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_submit_ref_payload_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="coadsorbate_submit_ref_payload_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )

    cluster_xyz = tmp_path / "cluster.xyz"
    cluster_xyz.write_text(
        "2\nAu2\nAu 0.0 0.0 0.0\nAu 0.0 0.0 2.5\n",
        encoding="utf-8",
    )
    cluster = ReferenceNanocluster.objects.create(
        name="coadsorbate_submit_ref_payload_cluster",
        composition="Au2",
        settings_profile=profile,
        structure_path=str(cluster_xyz),
    )

    reference_adsorbate = ReferenceAdsorbate.objects.create(
        name="coadsorbate_submit_ref_payload_adsorbate",
        formula="CO",
        settings_profile=profile,
    )
    reference_run_dir = tmp_path / "coadsorbate_submit_ref_payload_adsorbate_run"
    reference_run_dir.mkdir(parents=True)
    (reference_run_dir / "reaction_hub_final.xyz").write_text(
        "2\nCO\nC 0.0 0.0 0.0\nO 0.0 0.0 1.1\n",
        encoding="utf-8",
    )
    reference_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="coadsorbate_submit_ref_payload_adsorbate_run",
        status="completed",
        working_directory=str(reference_run_dir),
        settings_profile=profile,
        reference_adsorbate=reference_adsorbate,
    )
    Artifact.objects.create(
        workflow_run=reference_run,
        kind=ArtifactKind.STRUCTURE,
        label="reaction_hub_final",
        path="reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )

    captured_configs: list[dict[str, object]] = []

    def _fake_submit(**kwargs):
        captured_configs.append(dict(kwargs.get("coadsorbate_config") or {}))
        run_name = str(kwargs["run_name"])
        run = WorkflowRun.objects.create(
            workflow=kwargs["workflow"],
            run_name=run_name,
            status="initiated",
            working_directory=f"/work/runs/{run_name}",
            settings_profile=kwargs["settings_profile"],
            metadata={"submission": {}, "structure": kwargs.get("structure_metadata") or {}},
        )
        job = JobRecord.objects.create(
            workflow_run=run,
            hpc_system=kwargs["hpc_system"],
            scheduler_job_id="92001",
            job_name=run_name,
            status=JobStatus.INITIATED,
            remote_workdir=f"/scratch/{run_name}",
            stdout_path=f"/work/runs/{run_name}/workflow.log",
        )
        return SimpleNamespace(
            workflow_run_id=run.pk,
            job_record_id=job.pk,
            scheduler_job_id=job.scheduler_job_id,
            remote_run_dir=f"/work/runs/{run_name}",
        )

    monkeypatch.setattr("jobs.views.submit_slurm_workflow_run", _fake_submit)

    resp = client.post(
        "/jobs/submit/",
        data={
            "run_name": "coadsorbate_submit_ref_payload",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "coadsorbate_reference_cluster": str(cluster.pk),
            "coadsorbate_reference_adsorbates": [str(reference_adsorbate.pk)],
            "coadsorbate_ase_molecules": "",
            "coadsorbate_pubchem_cids": "",
            "coadsorbate_pubchem_queries": "",
            "coadsorbate_mace_model_path": "/work/PROJECT/GROUP/user/models/mace.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "gpu",
            "qos": "gpu-shd",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        },
    )

    assert resp.status_code == 302
    assert len(captured_configs) == 1
    coadsorbate_config = captured_configs[0]
    assert coadsorbate_config.get("reference_adsorbate_ids") == (reference_adsorbate.pk,)
    adsorbates = coadsorbate_config.get("adsorbates")
    assert isinstance(adsorbates, list)
    assert len(adsorbates) == 1
    payload = adsorbates[0]
    assert isinstance(payload, dict)
    metadata = payload.get("metadata")
    assert isinstance(metadata, dict)
    assert metadata.get("source") == "reference_adsorbate"
    assert metadata.get("reference_adsorbate_id") == reference_adsorbate.pk
    assert metadata.get("reference_adsorbate_name") == reference_adsorbate.name
    assert metadata.get("formula") == "CO"


@pytest.mark.django_db
def test_job_submit_coadsorbate_mode_accepts_uploaded_base_geometry_without_reference_cluster(
    client, monkeypatch
):
    profile = CalculationSettingsProfile.objects.create(name="coadsorbate_upload_base_profile")
    workflow = WorkflowDefinition.objects.create(
        name="coadsorbate_upload_base_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="coadsorbate_upload_base_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
        is_active=True,
    )

    captured_payloads: list[dict[str, object]] = []

    def _fake_submit(**kwargs):
        captured_payloads.append(
            {
                "structure_xyz": str(kwargs.get("structure_xyz") or ""),
                "structure_metadata": dict(kwargs.get("structure_metadata") or {}),
                "coadsorbate_config": dict(kwargs.get("coadsorbate_config") or {}),
            }
        )
        run_name = str(kwargs["run_name"])
        run = WorkflowRun.objects.create(
            workflow=kwargs["workflow"],
            run_name=run_name,
            status="initiated",
            working_directory=f"/work/runs/{run_name}",
            settings_profile=kwargs["settings_profile"],
            metadata={"submission": {}, "structure": kwargs.get("structure_metadata") or {}},
        )
        job = JobRecord.objects.create(
            workflow_run=run,
            hpc_system=kwargs["hpc_system"],
            scheduler_job_id="93001",
            job_name=run_name,
            status=JobStatus.INITIATED,
            remote_workdir=f"/scratch/{run_name}",
            stdout_path=f"/work/runs/{run_name}/workflow.log",
        )
        return SimpleNamespace(
            workflow_run_id=run.pk,
            job_record_id=job.pk,
            scheduler_job_id=job.scheduler_job_id,
            remote_run_dir=f"/work/runs/{run_name}",
        )

    monkeypatch.setattr("jobs.views.submit_slurm_workflow_run", _fake_submit)

    uploaded = SimpleUploadedFile(
        "base_structure.xyz",
        b"2\nAu2\nAu 0.0 0.0 0.0\nAu 0.0 0.0 2.2\n",
        content_type="chemical/x-xyz",
    )
    resp = client.post(
        "/jobs/submit/",
        data={
            "run_name": "coadsorbate_upload_base_submit",
            "run_mode": "coadsorbate_placement",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "structure_file": uploaded,
            "coadsorbate_reference_cluster": "",
            "coadsorbate_reference_adsorbates": [],
            "coadsorbate_ase_molecules": "CO",
            "coadsorbate_pubchem_cids": "",
            "coadsorbate_pubchem_queries": "",
            "coadsorbate_mace_model_path": "/work/PROJECT/GROUP/user/models/mace.model",
            "coadsorbate_mace_fmax": "0.05",
            "coadsorbate_mace_max_steps": "200",
            "coadsorbate_mace_device": "cuda",
            "coadsorbate_site_symbols": "Ti,Au,Cu",
            "coadsorbate_passive_symbols": "Au,Ce,O",
            "coadsorbate_height": "2.0",
            "coadsorbate_support_species": "Zn,Au,Ce,Ti,O",
            "n_iter": "1",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "12:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "gpu",
            "qos": "gpu-shd",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "pythonpath_entries": "",
        },
    )

    assert resp.status_code == 302
    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    structure_text = str(payload.get("structure_xyz") or "")
    assert structure_text.startswith("2\n")
    assert "Lattice=" in structure_text
    assert "Au" in structure_text
    assert "2.2" in structure_text
    structure_metadata = payload.get("structure_metadata")
    assert isinstance(structure_metadata, dict)
    assert structure_metadata.get("source") == "upload_file"
    coadsorbate_config = payload.get("coadsorbate_config")
    assert isinstance(coadsorbate_config, dict)
    assert coadsorbate_config.get("reference_cluster_id") is None


@pytest.mark.django_db
def test_dashboard_failures_table_shows_restart_in_place_actions(client):
    profile = CalculationSettingsProfile.objects.create(name="dashboard_actions_profile")
    wf = WorkflowDefinition.objects.create(name="dashboard_actions_wf")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="dashboard-actions-run",
        working_directory="/work/dashboard-actions-run",
        status="running",
        settings_profile=profile,
    )

    failed_job = JobRecord.objects.create(
        workflow_run=run,
        scheduler_job_id="801",
        job_name="failed-job",
        status=JobStatus.FAILED,
        failure_reason="failed reason",
    )
    cancelled_job = JobRecord.objects.create(
        workflow_run=run,
        scheduler_job_id="802",
        job_name="cancelled-job",
        status=JobStatus.CANCELLED,
        failure_reason="cancelled reason",
    )
    deleted_job = JobRecord.objects.create(
        workflow_run=run,
        scheduler_job_id="803",
        job_name="deleted-job",
        status=JobStatus.DELETED,
        failure_reason="deleted reason",
    )

    resp = client.get("/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert resp.context["page_size"] == 10
    assert resp.context["table"].paginator.per_page == 10
    assert 'name="page_size"' in body
    assert "Failures Needing Attention" in body
    assert "<th>Action</th>" in body
    assert "<details" in body
    assert 'action="/jobs/bulk-failed-action/"' in body
    assert f'name="job_ids" value="{failed_job.pk}"' in body
    assert f'name="job_ids" value="{cancelled_job.pk}"' not in body
    assert f'name="job_ids" value="{deleted_job.pk}"' not in body
    assert "Restart Selected In Place" in body
    assert "Mark Selected for Deletion" in body


@pytest.mark.django_db
def test_bulk_failed_action_queues_selected_restarts(monkeypatch, client):
    profile = CalculationSettingsProfile.objects.create(name="bulk_restart_profile")
    wf = WorkflowDefinition.objects.create(name="bulk_restart_wf")
    run = WorkflowRun.objects.create(workflow=wf, run_name="bulk-restart", settings_profile=profile)
    job_a = JobRecord.objects.create(workflow_run=run, scheduler_job_id="901", job_name="failed-a", status=JobStatus.FAILED)
    job_b = JobRecord.objects.create(workflow_run=run, scheduler_job_id="902", job_name="failed-b", status=JobStatus.FAILED)
    skipped = JobRecord.objects.create(workflow_run=run, scheduler_job_id="903", job_name="done", status=JobStatus.COMPLETED)
    queued = []

    class FakeTask:
        def delay(self, job_id, **kwargs):
            queued.append((job_id, kwargs))
            return SimpleNamespace(id=f"task-{job_id}")

    monkeypatch.setattr("jobs.views.resubmit_failed_job_task", FakeTask())

    resp = client.post(
        "/jobs/bulk-failed-action/",
        {"bulk_action": "restart_failed", "job_ids": [str(job_a.pk), str(job_b.pk), str(skipped.pk)], "next": "/"},
    )

    assert resp.status_code == 302
    assert [item[0] for item in queued] == [job_a.pk, job_b.pk]


@pytest.mark.django_db
def test_bulk_failed_action_marks_selected_failed_jobs_for_deletion(monkeypatch, client):
    profile = CalculationSettingsProfile.objects.create(name="bulk_delete_profile")
    wf = WorkflowDefinition.objects.create(name="bulk_delete_wf")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="bulk-delete",
        working_directory="/work/run",
        settings_profile=profile,
    )
    job_a = JobRecord.objects.create(workflow_run=run, scheduler_job_id="911", job_name="failed-a", status=JobStatus.FAILED)
    job_b = JobRecord.objects.create(workflow_run=run, scheduler_job_id="912", job_name="failed-b", status=JobStatus.FAILED)

    monkeypatch.setattr("jobs.views._rename_remote_run_for_deletion", lambda job: f"/work/{job.pk}_to_delete")

    resp = client.post(
        "/jobs/bulk-failed-action/",
        {"bulk_action": "mark_failed_delete", "job_ids": [str(job_a.pk), str(job_b.pk)], "next": "/"},
    )

    assert resp.status_code == 302
    job_a.refresh_from_db()
    job_b.refresh_from_db()
    assert job_a.status == JobStatus.DELETED
    assert job_b.status == JobStatus.DELETED


@pytest.mark.django_db
def test_sync_hpc_jobs_transitions_running_to_completed(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p1")
    wf = WorkflowDefinition.objects.create(name="wf")
    run_dir = tmp_path / "run-sync"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="sync-run",
        working_directory=str(run_dir),
        status="running",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="archer2",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="12345",
        job_name="calc",
        status=JobStatus.RUNNING,
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-1.23,
    )

    snapshots = {
        "12345": {
            "job_id": "12345",
            "state": "COMPLETED",
            "runtime": "00:05:00",
            "name": "calc",
            "workdir": str(run_dir),
            "exit_code": "0:0",
        }
    }

    summary = sync_hpc_jobs(client_factory=lambda _system: _FakeClient(snapshots))
    job.refresh_from_db()
    run.refresh_from_db()

    assert summary["updated"] == 1
    assert job.status == JobStatus.COMPLETED
    assert run.status == "completed"
    assert run.completed_at is not None
    assert JobStatusEvent.objects.filter(job=job, status=JobStatus.COMPLETED).exists()


@pytest.mark.django_db
def test_sync_hpc_jobs_transitions_running_to_cancelled_for_cancelled_by_state(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p_cancelled_by")
    wf = WorkflowDefinition.objects.create(name="wf_cancelled_by")
    run_dir = tmp_path / "run-cancelled-by"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="cancelled-by-run",
        working_directory=str(run_dir),
        status="running",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="archer2_cancelled_by",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="12346",
        job_name="cancelled-by-job",
        status=JobStatus.RUNNING,
    )

    snapshots = {
        "12346": {
            "job_id": "12346",
            "state": "CANCELLED by 12199",
            "runtime": "00:01:00",
            "name": "cancelled-by-job",
            "workdir": str(run_dir),
            "exit_code": "0:0",
        }
    }

    summary = sync_hpc_jobs(client_factory=lambda _system: _FakeClient(snapshots))
    job.refresh_from_db()
    run.refresh_from_db()

    assert summary["updated"] >= 1
    assert job.status == JobStatus.CANCELLED
    assert run.status == "failed"
    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.CANCELLED).first()
    assert event is not None
    assert "CANCELLED by 12199" in event.message


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("initial_status", "expected_status"),
    [
        (JobStatus.COMPLETED, JobStatus.COMPLETED),
        (JobStatus.FAILED, JobStatus.DELETED),
    ],
)
def test_sync_hpc_jobs_remote_run_root_missing_deletion_check(initial_status, expected_status):
    profile = CalculationSettingsProfile.objects.create(name="p_deleted")
    wf = WorkflowDefinition.objects.create(name="wf_deleted")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="missing-remote-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/missing-remote-run",
        status="completed",
        settings_profile=profile,
        metadata={"result_summary": {"mode": "dft"}},
    )
    system = HPCSystem.objects.create(
        name="archer2_deleted_check",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="99123",
        job_name="deleted-check",
        status=initial_status,
    )

    class _FakeDeletedClient:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def query_jobs(self, job_ids):
            del job_ids
            return {}

        def directories_exist(self, remote_paths):
            return {path: False for path in remote_paths}

    summary = sync_hpc_jobs(client_factory=lambda _system: _FakeDeletedClient())
    job.refresh_from_db()

    assert job.status == expected_status
    if expected_status == JobStatus.DELETED:
        assert summary["updated"] >= 1
        assert "Remote run directory missing on HPC" in job.failure_reason
        assert JobStatusEvent.objects.filter(job=job, status=JobStatus.DELETED).exists()
    else:
        assert summary["updated"] == 0
        assert job.failure_reason == ""
        assert not JobStatusEvent.objects.filter(job=job, status=JobStatus.DELETED).exists()


@pytest.mark.django_db
def test_sync_hpc_jobs_keeps_completed_status_when_no_computed_properties_by_default(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p_no_props")
    wf = WorkflowDefinition.objects.create(name="wf_no_props")
    run_dir = tmp_path / "run-no-props"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="run-no-props",
        working_directory=str(run_dir),
        status="running",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="archer2_no_props",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="54321",
        job_name="calc-no-props",
        status=JobStatus.RUNNING,
    )

    snapshots = {
        "54321": {
            "job_id": "54321",
            "state": "COMPLETED",
            "runtime": "00:02:00",
            "name": "calc-no-props",
            "workdir": str(run_dir),
            "exit_code": "0:0",
        }
    }

    summary = sync_hpc_jobs(client_factory=lambda _system: _FakeClient(snapshots))
    job.refresh_from_db()
    run.refresh_from_db()

    assert summary["updated"] >= 1
    assert job.status == JobStatus.COMPLETED
    assert run.status == "completed"
    assert JobStatusEvent.objects.filter(job=job, status=JobStatus.COMPLETED).exists()
    assert not JobStatusEvent.objects.filter(job=job, status=JobStatus.FAILED).exists()


@pytest.mark.django_db
def test_sync_hpc_jobs_marks_completed_job_failed_when_no_computed_properties_in_strict_mode(
    tmp_path: Path,
    settings,
):
    settings.REACTION_HUB_SYNC_STRICT_COMPLETED_REQUIRES_PROPERTIES = True
    profile = CalculationSettingsProfile.objects.create(name="p_no_props_strict")
    wf = WorkflowDefinition.objects.create(name="wf_no_props_strict")
    run_dir = tmp_path / "run-no-props-strict"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="run-no-props-strict",
        working_directory=str(run_dir),
        status="running",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="archer2_no_props_strict",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="54322",
        job_name="calc-no-props-strict",
        status=JobStatus.RUNNING,
    )

    snapshots = {
        "54322": {
            "job_id": "54322",
            "state": "COMPLETED",
            "runtime": "00:02:00",
            "name": "calc-no-props-strict",
            "workdir": str(run_dir),
            "exit_code": "0:0",
        }
    }

    summary = sync_hpc_jobs(client_factory=lambda _system: _FakeClient(snapshots))
    job.refresh_from_db()
    run.refresh_from_db()

    assert summary["updated"] >= 1
    assert job.status == JobStatus.FAILED
    assert run.status == "failed"
    assert "no computed properties" in job.failure_reason.lower()
    assert JobStatusEvent.objects.filter(job=job, status=JobStatus.FAILED).exists()


@pytest.mark.parametrize("initial_status", [JobStatus.INITIATED, JobStatus.PENDING, JobStatus.RUNNING])
@pytest.mark.django_db
def test_job_scancel_delete_view_marks_initiated_pending_or_running_job_deleted(monkeypatch, initial_status):
    profile = CalculationSettingsProfile.objects.create(name="p_scancel_delete")
    wf = WorkflowDefinition.objects.create(name="wf_scancel_delete")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="scancel-delete-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/scancel-delete-run",
        status="initiated",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="archer2_scancel_delete",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="55123",
        job_name="scancel-delete-run",
        status=initial_status,
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if command.startswith("scancel "):
                return ""
            if "mv \"$src\" \"$target\"" in command:
                return "/work/PROJECT/GROUP/user/reaction_hub_runs/scancel-delete-run_to_delete"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    from django.test import Client

    client = Client()
    resp = client.post(f"/jobs/{job.pk}/scancel-delete/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    assert job.status == JobStatus.DELETED
    assert "scancel-delete-run_to_delete" in job.failure_reason
    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.DELETED).first()
    assert event is not None
    assert event.payload["scheduler_job_id"] == "55123"
    assert event.payload["remote_run_dir_after"].endswith("_to_delete")
    assert any(command.startswith("scancel ") for command in remote_commands)


@pytest.mark.django_db
def test_job_mark_failed_delete_view_marks_failed_job_deleted(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="p_mark_failed_delete")
    wf = WorkflowDefinition.objects.create(name="wf_mark_failed_delete")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="mark-failed-delete-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/mark-failed-delete-run",
        status="failed",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="archer2_mark_failed_delete",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="77123",
        job_name="mark-failed-delete-run",
        status=JobStatus.FAILED,
        failure_reason="Failed due to runtime issue",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if "mv \"$src\" \"$target\"" in command:
                return "/work/PROJECT/GROUP/user/reaction_hub_runs/mark-failed-delete-run_to_delete"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    from django.test import Client

    client = Client()
    resp = client.post(f"/jobs/{job.pk}/mark-failed-delete/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    assert job.status == JobStatus.DELETED
    assert "mark-failed-delete-run_to_delete" in job.failure_reason
    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.DELETED).first()
    assert event is not None
    assert event.payload["action"] == "mark_failed_for_deletion"
    assert event.payload["scheduler_job_id"] == "77123"
    assert event.payload["remote_run_dir_after"].endswith("_to_delete")
    assert event.payload["scancel_state"] == "not_requested"
    assert not any(command.startswith("scancel ") for command in remote_commands)


@pytest.mark.django_db
def test_job_mark_failed_delete_view_strips_terminal_control_sequences_from_remote_path(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="p_mark_failed_delete_ansi")
    wf = WorkflowDefinition.objects.create(name="wf_mark_failed_delete_ansi")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="mark-failed-delete-ansi-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/mark-failed-delete-ansi-run",
        status="failed",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="archer2_mark_failed_delete_ansi",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="77124",
        job_name="mark-failed-delete-ansi-run",
        status=JobStatus.FAILED,
        failure_reason="Failed due to runtime issue",
    )

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            if "mv \"$src\" \"$target\"" in command:
                return (
                    "\x1b]P0000000\x1b]P1D16969\x1b]P2608B4E "
                    "/work/PROJECT/GROUP/user/reaction_hub_runs/mark-failed-delete-ansi-run_to_delete"
                )
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    from django.test import Client

    client = Client()
    resp = client.post(f"/jobs/{job.pk}/mark-failed-delete/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    assert job.status == JobStatus.DELETED
    assert job.failure_reason.endswith("mark-failed-delete-ansi-run_to_delete.")
    assert "\x1b" not in job.failure_reason
    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.DELETED).first()
    assert event is not None
    assert event.payload["remote_run_dir_after"].endswith("mark-failed-delete-ansi-run_to_delete")
    assert "\x1b" not in str(event.payload["remote_run_dir_after"])


@pytest.mark.django_db
def test_job_detail_shows_mark_failed_delete_action_for_failed_jobs(client):
    profile = CalculationSettingsProfile.objects.create(name="p_job_detail_failed_delete_action")
    wf = WorkflowDefinition.objects.create(name="wf_job_detail_failed_delete_action")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="failed-detail-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/failed-detail-run",
        status="failed",
        settings_profile=profile,
    )
    failed_job = JobRecord.objects.create(
        workflow_run=run,
        scheduler_job_id="99123",
        job_name="failed-detail-run",
        status=JobStatus.FAILED,
    )

    resp = client.get(f"/jobs/{failed_job.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert f'action="/jobs/{failed_job.pk}/resubmit-failed/"' in body
    assert f'action="/jobs/{failed_job.pk}/mark-failed-delete/"' in body
    assert "Mark Failed for Deletion" in body


@pytest.mark.django_db
def test_job_detail_shows_replot_resubmit_action_for_completed_supported_job(client):
    profile = CalculationSettingsProfile.objects.create(name="p_job_detail_replot_action")
    wf = WorkflowDefinition.objects.create(name="wf_job_detail_replot_action")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="completed-detail-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/completed-detail-run",
        status="completed",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "single_optimisation"}},
    )
    completed_job = JobRecord.objects.create(
        workflow_run=run,
        scheduler_job_id="99124",
        job_name="completed-detail-run",
        status=JobStatus.COMPLETED,
    )

    resp = client.get(f"/jobs/{completed_job.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert f'action="/jobs/{completed_job.pk}/resubmit-replot/"' in body
    assert "Resubmit Replot" in body


@pytest.mark.django_db
def test_job_list_shows_scancel_delete_button_for_initiated_pending_and_running_jobs(client):
    profile = CalculationSettingsProfile.objects.create(name="p_jobs_list_action")
    wf = WorkflowDefinition.objects.create(name="wf_jobs_list_action")
    system = HPCSystem.objects.create(
        name="archer2_jobs_list_action",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )

    run_initiated = WorkflowRun.objects.create(
        workflow=wf,
        run_name="initiated-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/initiated-run",
        status="initiated",
        settings_profile=profile,
    )
    run_completed = WorkflowRun.objects.create(
        workflow=wf,
        run_name="completed-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/completed-run",
        status="completed",
        settings_profile=profile,
    )
    initiated_job = JobRecord.objects.create(
        workflow_run=run_initiated,
        hpc_system=system,
        scheduler_job_id="66101",
        job_name="initiated-run",
        status=JobStatus.INITIATED,
    )
    run_pending = WorkflowRun.objects.create(
        workflow=wf,
        run_name="pending-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/pending-run",
        status="initiated",
        settings_profile=profile,
    )
    pending_job = JobRecord.objects.create(
        workflow_run=run_pending,
        hpc_system=system,
        scheduler_job_id="66103",
        job_name="pending-run",
        status=JobStatus.PENDING,
    )
    run_running = WorkflowRun.objects.create(
        workflow=wf,
        run_name="running-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/running-run",
        status="running",
        settings_profile=profile,
    )
    running_job = JobRecord.objects.create(
        workflow_run=run_running,
        hpc_system=system,
        scheduler_job_id="66104",
        job_name="running-run",
        status=JobStatus.RUNNING,
    )
    JobRecord.objects.create(
        workflow_run=run_completed,
        hpc_system=system,
        scheduler_job_id="66102",
        job_name="completed-run",
        status=JobStatus.COMPLETED,
    )

    resp = client.get("/jobs/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert f"/jobs/{initiated_job.pk}/scancel-delete/" in body
    assert f"/jobs/{pending_job.pk}/scancel-delete/" in body
    assert f"/jobs/{running_job.pk}/scancel-delete/" in body
    assert body.count("Scancel + Delete") == 3


@pytest.mark.django_db
def test_job_list_shows_restart_button_for_cancelled_jobs(client):
    profile = CalculationSettingsProfile.objects.create(name="p_jobs_cancelled_action")
    wf = WorkflowDefinition.objects.create(name="wf_jobs_cancelled_action")
    system = HPCSystem.objects.create(
        name="archer2_jobs_cancelled_action",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    run_cancelled = WorkflowRun.objects.create(
        workflow=wf,
        run_name="cancelled-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/cancelled-run",
        status="failed",
        settings_profile=profile,
    )
    cancelled_job = JobRecord.objects.create(
        workflow_run=run_cancelled,
        hpc_system=system,
        scheduler_job_id="66105",
        job_name="cancelled-run",
        status=JobStatus.CANCELLED,
    )

    resp = client.get("/jobs/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert f"/jobs/{cancelled_job.pk}/restart-cancelled/" in body
    assert "Restart In Place" in body
    assert "/jobs/mark-cancelled-deleted/" in body


@pytest.mark.django_db
def test_job_mark_all_cancelled_deleted_view_marks_only_cancelled_jobs(client):
    profile = CalculationSettingsProfile.objects.create(name="p_bulk_cancelled_delete")
    wf = WorkflowDefinition.objects.create(name="wf_bulk_cancelled_delete")
    system = HPCSystem.objects.create(
        name="archer2_bulk_cancelled_delete",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    run_cancelled_1 = WorkflowRun.objects.create(
        workflow=wf,
        run_name="cancelled-1",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/cancelled-1",
        status="failed",
        settings_profile=profile,
    )
    run_cancelled_2 = WorkflowRun.objects.create(
        workflow=wf,
        run_name="cancelled-2",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/cancelled-2",
        status="failed",
        settings_profile=profile,
    )
    run_failed = WorkflowRun.objects.create(
        workflow=wf,
        run_name="failed-1",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/failed-1",
        status="failed",
        settings_profile=profile,
    )
    cancelled_job_1 = JobRecord.objects.create(
        workflow_run=run_cancelled_1,
        hpc_system=system,
        scheduler_job_id="77101",
        job_name="cancelled-1",
        status=JobStatus.CANCELLED,
    )
    cancelled_job_2 = JobRecord.objects.create(
        workflow_run=run_cancelled_2,
        hpc_system=system,
        scheduler_job_id="77102",
        job_name="cancelled-2",
        status=JobStatus.CANCELLED,
    )
    failed_job = JobRecord.objects.create(
        workflow_run=run_failed,
        hpc_system=system,
        scheduler_job_id="77103",
        job_name="failed-1",
        status=JobStatus.FAILED,
    )

    resp = client.post("/jobs/mark-cancelled-deleted/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"

    cancelled_job_1.refresh_from_db()
    cancelled_job_2.refresh_from_db()
    failed_job.refresh_from_db()

    assert cancelled_job_1.status == JobStatus.DELETED
    assert cancelled_job_2.status == JobStatus.DELETED
    assert failed_job.status == JobStatus.FAILED
    assert "Marked for deletion by bulk cancelled-jobs action." in cancelled_job_1.failure_reason
    assert "Marked for deletion by bulk cancelled-jobs action." in cancelled_job_2.failure_reason
    assert JobStatusEvent.objects.filter(job=cancelled_job_1, status=JobStatus.DELETED).exists()
    assert JobStatusEvent.objects.filter(job=cancelled_job_2, status=JobStatus.DELETED).exists()
    assert not JobStatusEvent.objects.filter(job=failed_job, status=JobStatus.DELETED).exists()


@pytest.mark.django_db
def test_job_list_shows_resubmit_button_for_failed_jobs(client):
    profile = CalculationSettingsProfile.objects.create(name="p_jobs_failed_action")
    wf = WorkflowDefinition.objects.create(name="wf_jobs_failed_action")
    system = HPCSystem.objects.create(
        name="archer2_jobs_failed_action",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    run_failed = WorkflowRun.objects.create(
        workflow=wf,
        run_name="failed-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/failed-run",
        status="failed",
        settings_profile=profile,
    )
    failed_job = JobRecord.objects.create(
        workflow_run=run_failed,
        hpc_system=system,
        scheduler_job_id="66106",
        job_name="failed-run",
        status=JobStatus.FAILED,
    )

    resp = client.get("/jobs/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert f"/jobs/{failed_job.pk}/resubmit-failed/" in body
    assert "Resubmit Failed" in body


@pytest.mark.django_db
def test_job_list_shows_replot_resubmit_button_for_completed_supported_modes(client):
    profile = CalculationSettingsProfile.objects.create(name="p_jobs_replot_action")
    wf = WorkflowDefinition.objects.create(name="wf_jobs_replot_action")
    coads_wf = WorkflowDefinition.objects.create(
        name="wf_jobs_replot_action_coads",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="archer2_jobs_replot_action",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    run_single = WorkflowRun.objects.create(
        workflow=wf,
        run_name="completed-single-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/completed-single-run",
        status="completed",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "single_optimisation"}},
    )
    run_scan = WorkflowRun.objects.create(
        workflow=wf,
        run_name="completed-scan-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/completed-scan-run",
        status="completed",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "tddft_bond_scan"}},
    )
    run_coads = WorkflowRun.objects.create(
        workflow=coads_wf,
        run_name="completed-coads-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/completed-coads-run",
        status="completed",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "coadsorbate_placement"}},
    )
    job_single = JobRecord.objects.create(
        workflow_run=run_single,
        hpc_system=system,
        scheduler_job_id="66107",
        job_name="completed-single-run",
        status=JobStatus.COMPLETED,
    )
    job_scan = JobRecord.objects.create(
        workflow_run=run_scan,
        hpc_system=system,
        scheduler_job_id="66108",
        job_name="completed-scan-run",
        status=JobStatus.COMPLETED,
    )
    job_coads = JobRecord.objects.create(
        workflow_run=run_coads,
        hpc_system=system,
        scheduler_job_id="66109",
        job_name="completed-coads-run",
        status=JobStatus.COMPLETED,
    )

    resp = client.get("/jobs/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert f"/jobs/{job_single.pk}/resubmit-replot/" in body
    assert f"/jobs/{job_scan.pk}/resubmit-replot/" in body
    assert f"/jobs/{job_coads.pk}/resubmit-replot/" not in body
    assert "Resubmit Replot" in body


@pytest.mark.django_db
def test_job_restart_cancelled_view_resubmits_from_same_folder(monkeypatch, client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p_restart_cancelled")
    wf = WorkflowDefinition.objects.create(name="wf_restart_cancelled")
    run_dir = tmp_path / "restart-cancelled-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-cancelled-run",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={"submission": {"submit_script": "submit_workflow.sh", "scratch_base": "/tmp/scratch"}},
    )
    system = HPCSystem.objects.create(
        name="archer2_restart_cancelled",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88001",
        job_name="restart-cancelled-run",
        status=JobStatus.CANCELLED,
        stdout_path=f"{run_dir}/workflow.88001.log",
        failure_reason="Cancelled by scheduler",
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="structure",
        label="final",
        path="final.xyz",
    )
    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if "sbatch" in command:
                return "Submitted batch job 99009"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/restart-cancelled/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99009"
    assert job.submitted_at is not None
    assert job.started_at is None
    assert job.completed_at is None
    assert job.failure_reason == ""
    assert job.stdout_path.endswith("workflow.99009.log")
    assert run.status == "initiated"
    assert run.started_at is None
    assert run.completed_at is None

    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).first()
    assert event is not None
    assert event.previous_status == JobStatus.CANCELLED
    assert event.payload["previous_scheduler_job_id"] == "88001"
    assert event.payload["new_scheduler_job_id"] == "99009"
    assert event.payload["used_latest_structure"] is True
    assert event.payload["submit_script_refreshed"] is True
    assert "88001" in event.payload.get("previous_scheduler_job_ids", [])
    run_submission = run.metadata.get("submission", {}) if isinstance(run.metadata, dict) else {}
    assert "88001" in run_submission.get("previous_scheduler_job_ids", [])
    assert any("sbatch" in command for command in remote_commands)
    assert any("SLURM_CPU_FREQ_REQ=2250000" in command for command in remote_commands)
    assert any("REACTION_HUB_RSYNC_EXCLUDE_JOB_IDS='88001'" in command for command in remote_commands if "sbatch" in command)
    assert any("orca.xyz" in command and "printf %s" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_restart_cancelled_view_trh408_coadsorbate_keeps_in_place_runtime(monkeypatch, client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p_restart_cancelled_trh408")
    wf = WorkflowDefinition.objects.create(
        name="wf_restart_cancelled_trh408",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    run_dir = tmp_path / "restart-cancelled-trh408-run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-cancelled-trh408-run",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={
            "submission": {
                "run_mode": "coadsorbate_placement",
                "submit_script": "submit_workflow.sh",
                "wrapper_script": "run_workflow_submit.py",
                "scratch_base": "/tmp",
                "nodes": 1,
                "tasks_per_node": 1,
                "cpus_per_task": 8,
                "gpus": 1,
                "use_srun": False,
                "srun_ntasks": 1,
                "srun_gpus_per_task": 0,
                "walltime": "12:00:00",
                "partition": "main",
                "qos": "",
                "account": "",
                "module_loads": [],
                "venv_activate_path": "/srv/research/mace_venv/bin/activate",
                "pythonpath_entries": [],
                "run_in_place": True,
            }
        },
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh408.example.invalid",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88055",
        job_name="restart-cancelled-trh408-run",
        status=JobStatus.CANCELLED,
        remote_workdir=str(run_dir),
        stdout_path=f"{run_dir}/workflow.88055.log",
        failure_reason="Cancelled by scheduler",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if f"{run_dir}/88055" in command and "printf '1'" in command and "printf '0'" in command:
                return "0"
            if (
                f"{run_dir}" in command
                and "printf '1'" in command
                and "printf '0'" in command
                and f"{run_dir}/88055" not in command
            ):
                return "1"
            if "sbatch" in command:
                return "Submitted batch job 99055"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/restart-cancelled/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99055"
    assert job.remote_workdir == str(run_dir)
    assert run.working_directory == str(run_dir)

    sbatch_commands = [command for command in remote_commands if "sbatch --export=" in command]
    assert sbatch_commands
    assert all("SLURM_CPU_FREQ_REQ=2250000" not in command for command in sbatch_commands)
    assert any("REACTION_HUB_RUN_IN_PLACE=1" in command for command in remote_commands)
    assert any("#SBATCH --gres=gpu:1" in command for command in remote_commands)
    assert any("#SBATCH --exclusive" in command for command in remote_commands)
    assert any("#SBATCH --job-name=testgpu" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_restart_cancelled_view_retries_submit_without_restart_export_if_primary_submit_fails(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_restart_cancelled_export_retry")
    wf = WorkflowDefinition.objects.create(name="wf_restart_cancelled_export_retry")
    run_dir = tmp_path / "restart-cancelled-export-retry"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-cancelled-export-retry",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={"submission": {"submit_script": "submit_workflow.sh", "scratch_base": "/tmp/scratch"}},
    )
    system = HPCSystem.objects.create(
        name="archer2_restart_export_retry",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88111",
        job_name="restart-cancelled-export-retry",
        status=JobStatus.CANCELLED,
        stdout_path=f"{run_dir}/workflow.88111.log",
        failure_reason="Cancelled by scheduler",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if "sbatch --export=" in command and "REACTION_HUB_RSYNC_EXCLUDE_JOB_IDS" in command:
                raise RuntimeError("Remote command failed (1 (raw=0)).")
            if "sbatch --export=" in command:
                return "Submitted batch job 99111"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/restart-cancelled/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99111"

    sbatch_commands = [command for command in remote_commands if "sbatch --export=" in command]
    assert len(sbatch_commands) == 2
    assert "REACTION_HUB_RSYNC_EXCLUDE_JOB_IDS" in sbatch_commands[0]
    assert "REACTION_HUB_RSYNC_EXCLUDE_JOB_IDS" not in sbatch_commands[1]

    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).first()
    assert event is not None
    assert event.payload.get("submit_export_fallback") is True


@pytest.mark.django_db
def test_job_resubmit_failed_view_resubmits_from_same_folder(monkeypatch, client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p_resubmit_failed")
    wf = WorkflowDefinition.objects.create(name="wf_resubmit_failed")
    run_dir = tmp_path / "resubmit-failed-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="resubmit-failed-run",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={
            "submission": {
                "submit_script": "submit_workflow.sh",
                "scratch_base": "/tmp/scratch",
                "run_mode": "charge_transfer",
            }
        },
    )
    system = HPCSystem.objects.create(
        name="archer2_resubmit_failed",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88003",
        job_name="resubmit-failed-run",
        status=JobStatus.FAILED,
        stdout_path=f"{run_dir}/workflow.88003.log",
        failure_reason="Scheduler failed",
        scheduler_payload={"qos": "reservation", "reservation": "test-reservation"},
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="structure",
        label="final",
        path="final.xyz",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if "sbatch" in command:
                return "Submitted batch job 99011"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    request = _task_request(f"/jobs/{job.pk}/resubmit-failed/")
    resp = _job_resubmit_failed_sync_response(request, job.pk)

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED, [str(message) for message in request._messages]
    assert job.scheduler_job_id == "99011"
    assert job.failure_reason == ""
    assert run.status == "initiated"

    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).first()
    assert event is not None
    assert event.previous_status == JobStatus.FAILED
    assert event.payload["action"] == "resubmit_failed_in_place"
    assert event.payload["previous_scheduler_job_id"] == "88003"
    assert event.payload["new_scheduler_job_id"] == "99011"
    assert any("sbatch" in command for command in remote_commands)
    assert any("SLURM_CPU_FREQ_REQ=2250000" in command for command in remote_commands)
    assert any("python3 -m py_compile" in command for command in remote_commands)
    assert any("from cats.multiwfn" in command for command in remote_commands)
    assert any("#SBATCH --reservation=test-reservation" in command for command in remote_commands)
    assert not any("sed -i '1,/^import json$" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_resubmit_smooth_view_resubmits_completed_scan_from_previous_job_folder(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_resubmit_smooth")
    wf = WorkflowDefinition.objects.create(name="wf_resubmit_smooth")
    run_dir = tmp_path / "resubmit-smooth-run"
    run_dir.mkdir(parents=True)
    previous_job_dir = run_dir / "88030"
    previous_job_dir.mkdir(parents=True)
    (previous_job_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="resubmit-smooth-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {
                "run_mode": "tddft_bond_scan",
                "submit_script": "submit_workflow.sh",
                "wrapper_script": "run_workflow_submit.py",
                "scratch_base": "/tmp/scratch",
                "nodes": 2,
                "tasks_per_node": 8,
                "cpus_per_task": 1,
                "walltime": "08:00:00",
                "account": "SLURM_ACCOUNT",
                "partition": "standard",
                "qos": "long",
                "scan_config": {
                    "bond_i": 0,
                    "bond_j": 1,
                    "increment": 0.1,
                    "fix": 0.5,
                    "enable_ts_smoothing": False,
                },
            }
        },
    )
    system = HPCSystem.objects.create(
        name="archer2_resubmit_smooth",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88030",
        job_name="resubmit-smooth-run",
        status=JobStatus.COMPLETED,
        stdout_path=f"{run_dir}/workflow.88030.log",
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="88030/reaction_hub_final.xyz",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if f"{run_dir}/88030" in command and "printf '1'" in command and "printf '0'" in command:
                return "1"
            if "sbatch" in command:
                return "Submitted batch job 99030"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(
        f"/jobs/{job.pk}/resubmit-smooth/",
        data={"next": f"/workflows/{run.pk}/"},
    )

    assert resp.status_code == 302
    assert resp["Location"] == f"/workflows/{run.pk}/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99030"
    assert job.failure_reason == ""
    assert job.stdout_path.endswith("/88030/workflow.99030.log")
    assert run.status == "initiated"
    assert run.working_directory == f"{run_dir}/88030"

    run_submission = run.metadata.get("submission", {}) if isinstance(run.metadata, dict) else {}
    scan_config = run_submission.get("scan_config", {})
    assert isinstance(scan_config, dict)
    assert scan_config.get("enable_ts_smoothing") is True
    assert "88030" in run_submission.get("previous_scheduler_job_ids", [])

    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).order_by("-id").first()
    assert event is not None
    assert event.previous_status == JobStatus.COMPLETED
    assert event.payload["action"] == "resubmit_completed_smooth_in_place"
    assert event.payload["previous_scheduler_job_id"] == "88030"
    assert event.payload["new_scheduler_job_id"] == "99030"
    assert event.payload["previous_job_folder_found"] == f"{run_dir}/88030"
    assert event.payload["enable_ts_smoothing"] is True
    assert any("sed -i" in command and "enable_ts_smoothing=True" in command for command in remote_commands)
    assert any(f"cd '{run_dir}/88030' && sbatch" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_resubmit_replot_view_resubmits_completed_job_with_short_settings(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_resubmit_replot")
    wf = WorkflowDefinition.objects.create(name="wf_resubmit_replot")
    run_dir = tmp_path / "resubmit-replot-run"
    run_dir.mkdir(parents=True)
    previous_job_dir = run_dir / "88032"
    previous_job_dir.mkdir(parents=True)
    (previous_job_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="resubmit-replot-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {
                "run_mode": "single_optimisation",
                "submit_script": "submit_workflow.sh",
                "wrapper_script": "run_workflow_submit.py",
                "scratch_base": "/tmp/scratch",
                "nodes": 2,
                "tasks_per_node": 8,
                "cpus_per_task": 1,
                "walltime": "24:00:00",
                "account": "SLURM_ACCOUNT",
                "partition": "standard",
                "qos": "standard",
            }
        },
    )
    system = HPCSystem.objects.create(
        name="archer2_resubmit_replot",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88032",
        job_name="resubmit-replot-run",
        status=JobStatus.COMPLETED,
        stdout_path=f"{run_dir}/workflow.88032.log",
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="88032/reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if f"{run_dir}/88032" in command and "printf '1'" in command and "printf '0'" in command:
                return "1"
            if "sbatch" in command:
                return "Submitted batch job 99032"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/resubmit-replot/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99032"
    assert job.stdout_path.endswith("/88032/workflow.99032.log")
    assert run.status == "initiated"
    assert run.working_directory == f"{run_dir}/88032"

    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).order_by("-id").first()
    assert event is not None
    assert event.previous_status == JobStatus.COMPLETED
    assert event.payload["action"] == "resubmit_completed_replot_in_place"
    assert event.payload["walltime"] == "00:20:00"
    assert event.payload["partition"] == "standard"
    assert event.payload["qos"] == "short"
    assert event.payload.get("run_in_place") is True
    assert any("#SBATCH --time=00:20:00" in command for command in remote_commands)
    assert any("#SBATCH --partition=standard" in command for command in remote_commands)
    assert any("#SBATCH --qos=short" in command for command in remote_commands)
    assert any("REACTION_HUB_RUN_IN_PLACE=1" in command for command in remote_commands)
    assert any("REACTION_HUB_REPLOT_ONLY=1" in command for command in remote_commands)
    assert any(f"cd '{run_dir}/88032' && sbatch" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_resubmit_replot_view_disables_scan_smoothing_for_bond_scan_runs(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_resubmit_replot_scan_disable_smooth")
    wf = WorkflowDefinition.objects.create(name="wf_resubmit_replot_scan_disable_smooth")
    run_dir = tmp_path / "resubmit-replot-scan-disable-smooth-run"
    run_dir.mkdir(parents=True)
    previous_job_dir = run_dir / "88040"
    previous_job_dir.mkdir(parents=True)
    (previous_job_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="resubmit-replot-scan-disable-smooth-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {
                "run_mode": "tddft_bond_scan",
                "submit_script": "submit_workflow.sh",
                "wrapper_script": "run_workflow_submit.py",
                "scratch_base": "/tmp/scratch",
                "nodes": 2,
                "tasks_per_node": 8,
                "cpus_per_task": 1,
                "walltime": "24:00:00",
                "account": "SLURM_ACCOUNT",
                "partition": "standard",
                "qos": "standard",
                "scan_config": {
                    "bond_i": 0,
                    "bond_j": 1,
                    "increment": 0.1,
                    "fix": 0.5,
                    "enable_ts_smoothing": True,
                },
            }
        },
    )
    system = HPCSystem.objects.create(
        name="archer2_resubmit_replot_scan_disable_smooth",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88040",
        job_name="resubmit-replot-scan-disable-smooth-run",
        status=JobStatus.COMPLETED,
        stdout_path=f"{run_dir}/workflow.88040.log",
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="88040/reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if f"{run_dir}/88040" in command and "printf '1'" in command and "printf '0'" in command:
                return "1"
            if "sbatch" in command:
                return "Submitted batch job 99040"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/resubmit-replot/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99040"
    assert job.stdout_path.endswith("/88040/workflow.99040.log")
    assert run.status == "initiated"
    assert run.working_directory == f"{run_dir}/88040"

    run_submission = run.metadata.get("submission", {}) if isinstance(run.metadata, dict) else {}
    scan_config = run_submission.get("scan_config", {})
    assert isinstance(scan_config, dict)
    assert scan_config.get("enable_ts_smoothing") is False

    assert any("#SBATCH --time=00:20:00" in command for command in remote_commands)
    assert any("#SBATCH --partition=standard" in command for command in remote_commands)
    assert any("#SBATCH --qos=short" in command for command in remote_commands)
    assert any("REACTION_HUB_RUN_IN_PLACE=1" in command for command in remote_commands)
    assert any("REACTION_HUB_REPLOT_ONLY=1" in command for command in remote_commands)
    assert any("sed -i" in command and "enable_ts_smoothing=False" in command for command in remote_commands)
    assert not any("sed -i" in command and "enable_ts_smoothing=True" in command for command in remote_commands)
    assert any(f"cd '{run_dir}/88040' && sbatch" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_resubmit_smooth_view_prefers_latest_scan_endpoints_db_seed_over_evaluated_traj(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_resubmit_smooth_scan_db_seed")
    wf = WorkflowDefinition.objects.create(name="wf_resubmit_smooth_scan_db_seed")
    run_dir = tmp_path / "resubmit-smooth-scan-db-seed-run"
    run_dir.mkdir(parents=True)
    previous_job_dir = run_dir / "88031"
    previous_job_dir.mkdir(parents=True)

    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="resubmit-smooth-scan-db-seed-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {
                "run_mode": "tddft_bond_scan",
                "submit_script": "submit_workflow.sh",
                "wrapper_script": "run_workflow_submit.py",
                "scratch_base": "/tmp/scratch",
                "scan_config": {
                    "bond_i": 0,
                    "bond_j": 1,
                    "increment": 0.1,
                    "fix": 0.5,
                    "enable_ts_smoothing": False,
                },
            }
        },
    )
    system = HPCSystem.objects.create(
        name="archer2_resubmit_smooth_scan_db_seed",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88031",
        job_name="resubmit-smooth-scan-db-seed-run",
        status=JobStatus.COMPLETED,
        stdout_path=f"{run_dir}/workflow.88031.log",
    )

    old_db_path = run_dir / "88030" / "TDDFT_active_learning_shared" / "scan_endpoints.db"
    old_db_path.parent.mkdir(parents=True, exist_ok=True)
    old_db = ase_connect(str(old_db_path))
    old_db.write(
        Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.20)]),
        iteration=1,
        total_energy_eV=-100.0,
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="database",
        label="scan_endpoints_db:old",
        path="88030/TDDFT_active_learning_shared/scan_endpoints.db",
    )

    new_db_path = run_dir / "88031" / "TDDFT_active_learning_shared" / "scan_endpoints.db"
    new_db_path.parent.mkdir(parents=True, exist_ok=True)
    new_db = ase_connect(str(new_db_path))
    new_db.write(
        Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.40)]),
        iteration=1,
        total_energy_eV=-120.0,
    )
    new_db.write(
        Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.70)]),
        iteration=2,
        total_energy_eV=-119.0,
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="database",
        label="scan_endpoints_db:new",
        path="88031/TDDFT_active_learning_shared/scan_endpoints.db",
    )

    bad_traj_path = run_dir / "DFT" / "iter_1" / "active_learning" / "evaluated.traj"
    bad_traj_path.parent.mkdir(parents=True, exist_ok=True)
    bad_atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.90]])
    bad_atoms.calc = SinglePointCalculator(
        bad_atoms,
        energy=-1.0,
        forces=[[0.0, 0.0, 1.2], [0.0, 0.0, -1.2]],
    )
    write(bad_traj_path, [bad_atoms])
    Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="DFT/iter_1/active_learning/evaluated.traj",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if f"{run_dir}/88031" in command and "printf '1'" in command and "printf '0'" in command:
                return "1"
            if "sbatch" in command:
                return "Submitted batch job 99031"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(
        f"/jobs/{job.pk}/resubmit-smooth/",
        data={"next": f"/workflows/{run.pk}/"},
    )

    assert resp.status_code == 302
    assert resp["Location"] == f"/workflows/{run.pk}/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99031"
    assert run.status == "initiated"
    assert run.working_directory == f"{run_dir}/88031"

    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).order_by("-id").first()
    assert event is not None
    assert event.payload["action"] == "resubmit_completed_smooth_in_place"
    assert event.payload["structure_artifact_label"] == "scan_endpoints_db:new"
    assert event.payload["structure_artifact_path"] == "88031/TDDFT_active_learning_shared/scan_endpoints.db"
    assert "structure_seed_warning" not in event.payload
    assert any(f"cd '{run_dir}/88031' && sbatch" in command for command in remote_commands)

@pytest.mark.django_db
def test_job_restart_cancelled_view_restarts_from_scratch_when_previous_job_folder_missing(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_restart_cancelled_scratch")
    wf = WorkflowDefinition.objects.create(name="wf_restart_cancelled_scratch")
    run_dir = tmp_path / "restart-cancelled-scratch-run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-cancelled-scratch-run",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={"submission": {"submit_script": "submit_workflow.sh", "scratch_base": "/tmp/scratch"}},
    )
    system = HPCSystem.objects.create(
        name="archer2_restart_cancelled_scratch",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88002",
        job_name="restart-cancelled-scratch-run",
        status=JobStatus.CANCELLED,
        remote_workdir="/tmp/scratch/user/88002",
        stdout_path=f"{run_dir}/workflow.88002.log",
        failure_reason="Cancelled by scheduler",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if "/tmp/scratch/user/88002" in command and "printf '1'" in command and "printf '0'" in command:
                return "0"
            if "restart_backup_${stamp}" in command:
                return f"{run_dir}/restart_backup_20260224T120000"
            if "sbatch" in command:
                return "Submitted batch job 99010"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/restart-cancelled/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99010"
    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).first()
    assert event is not None
    assert event.payload["previous_scheduler_job_id"] == "88002"
    assert event.payload["new_scheduler_job_id"] == "99010"
    assert event.payload["previous_remote_workdir"] == "/tmp/scratch/user/88002"
    assert event.payload["previous_remote_workdir_exists"] is False
    assert event.payload["restarted_from_scratch"] is True
    assert "restart_backup_20260224T120000" in str(event.payload.get("restart_backup_dir", ""))
    assert any("restart_backup_${stamp}" in command for command in remote_commands)
    assert any("sbatch" in command for command in remote_commands)
    assert any("SLURM_CPU_FREQ_REQ=2250000" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_restart_cancelled_view_uses_run_jobid_folder_when_scratch_missing(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_restart_cancelled_run_jobid")
    wf = WorkflowDefinition.objects.create(name="wf_restart_cancelled_run_jobid")
    run_dir = tmp_path / "restart-cancelled-run-jobid-folder"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-cancelled-run-jobid-folder",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={"submission": {"submit_script": "submit_workflow.sh", "scratch_base": "/tmp/scratch"}},
    )
    system = HPCSystem.objects.create(
        name="archer2_restart_cancelled_run_jobid",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88012",
        job_name="restart-cancelled-run-jobid-folder",
        status=JobStatus.CANCELLED,
        remote_workdir="/tmp/scratch/user/88012",
        stdout_path=f"{run_dir}/workflow.88012.log",
        failure_reason="Cancelled by scheduler",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if "/tmp/scratch/user/88012" in command and "printf '1'" in command and "printf '0'" in command:
                return "0"
            if f"{run_dir}/88012" in command and "printf '1'" in command and "printf '0'" in command:
                return "1"
            if "sbatch" in command:
                return "Submitted batch job 99012"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/restart-cancelled/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99012"
    assert job.stdout_path.endswith("/88012/workflow.99012.log")
    assert run.working_directory == f"{run_dir}/88012"

    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).first()
    assert event is not None
    assert event.payload["previous_scheduler_job_id"] == "88012"
    assert event.payload["new_scheduler_job_id"] == "99012"
    assert event.payload["remote_run_dir"] == f"{run_dir}/88012"
    assert event.payload["remote_run_root"] == str(run_dir)
    assert event.payload["previous_remote_workdir"] == "/tmp/scratch/user/88012"
    assert event.payload["previous_remote_workdir_exists"] is True
    assert event.payload["previous_job_folder_found"] == f"{run_dir}/88012"
    assert "/tmp/scratch/user/88012" in event.payload["previous_job_folder_candidates"]
    assert f"{run_dir}/88012" in event.payload["previous_job_folder_candidates"]
    assert "restarted_from_scratch" not in event.payload
    assert not any("restart_backup_${stamp}" in command for command in remote_commands)
    assert any(f"{run_dir}/88012" in command for command in remote_commands)
    assert any(f"cd '{run_dir}/88012' && sbatch" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_restart_cancelled_view_uses_historical_run_jobid_folder_when_latest_missing(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_restart_cancelled_hist_jobid")
    wf = WorkflowDefinition.objects.create(name="wf_restart_cancelled_hist_jobid")
    run_dir = tmp_path / "restart-cancelled-historical-jobid-folder"
    run_dir.mkdir(parents=True)
    hist_dir = run_dir / "88013"
    hist_dir.mkdir(parents=True)
    (hist_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-cancelled-historical-jobid-folder",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={
            "submission": {
                "submit_script": "submit_workflow.sh",
                "scratch_base": "/tmp/scratch",
                "previous_scheduler_job_ids": ["88013"],
            }
        },
    )
    system = HPCSystem.objects.create(
        name="archer2_restart_cancelled_hist_jobid",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="99020",
        job_name="restart-cancelled-historical-jobid-folder",
        status=JobStatus.CANCELLED,
        remote_workdir="/tmp/scratch/user/99020",
        stdout_path=f"{run_dir}/workflow.99020.log",
        failure_reason="Cancelled by scheduler",
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="88013/reaction_hub_final.xyz",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if "/tmp/scratch/user/99020" in command and "printf '1'" in command and "printf '0'" in command:
                return "0"
            if f"{run_dir}/99020" in command and "printf '1'" in command and "printf '0'" in command:
                return "0"
            if f"{run_dir}/88013" in command and "printf '1'" in command and "printf '0'" in command:
                return "1"
            if "sbatch" in command:
                return "Submitted batch job 99021"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/restart-cancelled/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99021"
    assert job.stdout_path.endswith("/88013/workflow.99021.log")
    assert run.working_directory == f"{run_dir}/88013"

    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).first()
    assert event is not None
    assert event.payload["previous_scheduler_job_id"] == "99020"
    assert event.payload["new_scheduler_job_id"] == "99021"
    assert event.payload["remote_run_dir"] == f"{run_dir}/88013"
    assert event.payload["remote_run_root"] == str(run_dir)
    assert event.payload["previous_remote_workdir"] == "/tmp/scratch/user/99020"
    assert event.payload["previous_remote_workdir_exists"] is True
    assert event.payload["previous_job_folder_found"] == f"{run_dir}/88013"
    assert f"{run_dir}/88013" in event.payload["previous_job_folder_candidates"]
    assert "restarted_from_scratch" not in event.payload
    assert not any("restart_backup_${stamp}" in command for command in remote_commands)
    assert any(f"{run_dir}/88013" in command for command in remote_commands)
    assert any(f"cd '{run_dir}/88013' && sbatch" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_restart_cancelled_view_tolerates_noisy_exists_probe_output(
    monkeypatch, client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="p_restart_cancelled_noisy_probe")
    wf = WorkflowDefinition.objects.create(name="wf_restart_cancelled_noisy_probe")
    run_dir = tmp_path / "restart-cancelled-noisy-probe"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-cancelled-noisy-probe",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={"submission": {"submit_script": "submit_workflow.sh", "scratch_base": "/tmp/scratch"}},
    )
    system = HPCSystem.objects.create(
        name="archer2_restart_cancelled_noisy_probe",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88150",
        job_name="restart-cancelled-noisy-probe",
        status=JobStatus.CANCELLED,
        remote_workdir="/tmp/scratch/user/88150",
        stdout_path=f"{run_dir}/workflow.88150.log",
        failure_reason="Cancelled by scheduler",
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="final",
        path="final.xyz",
    )

    remote_commands: list[str] = []

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            remote_commands.append(command)
            if "/tmp/scratch/user/88150" in command and "printf '1'" in command and "printf '0'" in command:
                return "NOTICE: module list updated\n0"
            if f"{run_dir}/88150" in command and "printf '1'" in command and "printf '0'" in command:
                return "Welcome to ARCHER2\n1"
            if "sbatch" in command:
                return "Submitted batch job 99150"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(f"/jobs/{job.pk}/restart-cancelled/")

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    job.refresh_from_db()
    run.refresh_from_db()

    assert job.status == JobStatus.INITIATED
    assert job.scheduler_job_id == "99150"
    assert job.stdout_path.endswith("/88150/workflow.99150.log")
    assert run.working_directory == f"{run_dir}/88150"
    event = JobStatusEvent.objects.filter(job=job, status=JobStatus.INITIATED).first()
    assert event is not None
    assert event.payload["previous_remote_workdir_exists"] is True
    assert event.payload["previous_job_folder_found"] == f"{run_dir}/88150"
    assert event.payload["remote_run_dir"] == f"{run_dir}/88150"
    assert event.payload["remote_run_root"] == str(run_dir)
    assert "restarted_from_scratch" not in event.payload
    assert not any("restart_backup_${stamp}" in command for command in remote_commands)


@pytest.mark.django_db
def test_job_restart_cancelled_view_redirects_to_next_when_provided(monkeypatch, client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p_restart_redirect_next")
    wf = WorkflowDefinition.objects.create(name="wf_restart_redirect_next")
    run_dir = tmp_path / "restart-redirect-next"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-redirect-next",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={"submission": {"submit_script": "submit_workflow.sh", "scratch_base": "/tmp/scratch"}},
    )
    system = HPCSystem.objects.create(
        name="archer2_restart_redirect_next",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88101",
        job_name="restart-redirect-next",
        status=JobStatus.CANCELLED,
        stdout_path=f"{run_dir}/workflow.88101.log",
        failure_reason="Cancelled by scheduler",
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="structure",
        label="final",
        path="final.xyz",
    )

    class _FakeOpenSSHClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def exec_checked(self, command: str) -> str:
            if "sbatch" in command:
                return "Submitted batch job 99101"
            return ""

        def close(self):
            return None

    monkeypatch.setattr("jobs.views.OpenSSHClient", _FakeOpenSSHClient)

    resp = client.post(
        f"/jobs/{job.pk}/restart-cancelled/",
        data={"next": "/"},
    )

    assert resp.status_code == 302
    assert resp["Location"] == "/"


@pytest.mark.django_db
def test_latest_restart_band_uses_last_predicted_band_not_evaluated_samples(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="restart_band_profile")
    wf = WorkflowDefinition.objects.create(name="restart_band_wf")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-band-run",
        working_directory=str(tmp_path),
        settings_profile=profile,
        metadata={"submission": {"neb_config": {"max_images": 3}}},
    )
    predicted_path = tmp_path / "predicted.traj"
    write(
        predicted_path,
        [Atoms("CO", positions=[(0, 0, 0), (0, 0, distance)]) for distance in (0.9, 1.1, 1.3, 1.5)],
    )
    predicted = Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="artifact:predicted.traj",
        path="predicted.traj",
    )
    write(tmp_path / "predicted_evaluated.traj", [Atoms("CO"), Atoms("CO"), Atoms("CO")])
    Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="artifact:predicted_evaluated.traj",
        path="predicted_evaluated.traj",
    )

    band_text, meta = _latest_restart_band_extxyz_for_run(run)

    band = ase_read(StringIO(band_text), format="extxyz", index=":")
    assert [atoms.get_distance(0, 1) for atoms in band] == pytest.approx([1.1, 1.3, 1.5])
    assert meta["band_artifact_id"] == predicted.pk
    assert meta["band_frame_selection"] == "last_3_predicted_frames"


@pytest.mark.django_db
def test_latest_restart_structure_prefers_last_reasonable_frame_from_evaluated_traj(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="restart_structure_traj_profile")
    wf = WorkflowDefinition.objects.create(name="restart_structure_traj_wf")
    run_dir = tmp_path / "restart-structure-from-traj"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-structure-from-traj",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={"submission": {"fmax": 0.01}},
    )

    # Deliberately include a final structure artifact; restart should still seed from evaluated.traj.
    (run_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 0.95\n", encoding="utf-8")
    Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )

    def _frame(distance: float, force_z: float):
        atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, float(distance)]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-1.0,
            forces=[[0.0, 0.0, float(force_z)], [0.0, 0.0, float(-force_z)]],
        )
        return atoms

    traj_path = run_dir / "DFT" / "iter_1" / "active_learning" / "evaluated.traj"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    write(
        traj_path,
        [
            _frame(1.00, 0.20),
            _frame(1.10, 0.03),
            _frame(1.20, 0.02),
            _frame(1.30, 1.20),
        ],
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="DFT/iter_1/active_learning/evaluated.traj",
    )

    xyz_text, meta = _latest_restart_structure_xyz_for_run(run)

    assert xyz_text is not None
    seeded = ase_read(StringIO(xyz_text), format="xyz")
    assert seeded.get_distance(0, 1) == pytest.approx(1.20, abs=1.0e-8)
    assert str(meta.get("artifact_path", "")).endswith("evaluated.traj")
    assert int(meta.get("trajectory_frame_count")) == 4
    assert int(meta.get("trajectory_frame_index")) == 2
    assert int(meta.get("trajectory_frame_index")) != 0
    assert float(meta.get("trajectory_reasonable_fmax")) == pytest.approx(1.0, abs=1.0e-12)
    assert float(meta.get("trajectory_frame_fmax")) == pytest.approx(0.02, abs=1.0e-12)
    assert meta.get("trajectory_frame_selection") == "last_reasonable_fmax_frame"


@pytest.mark.django_db
def test_latest_restart_structure_falls_back_when_no_reasonable_traj_frame(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="restart_structure_traj_fallback_profile")
    wf = WorkflowDefinition.objects.create(name="restart_structure_traj_fallback_wf")
    run_dir = tmp_path / "restart-structure-traj-fallback"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="restart-structure-traj-fallback",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        metadata={"submission": {"fmax": 0.01}},
    )

    (run_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 0.95\n", encoding="utf-8")
    Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )

    def _frame(distance: float, force_z: float):
        atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, float(distance)]])
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=-1.0,
            forces=[[0.0, 0.0, float(force_z)], [0.0, 0.0, float(-force_z)]],
        )
        return atoms

    traj_path = run_dir / "DFT" / "iter_1" / "active_learning" / "evaluated.traj"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    write(
        traj_path,
        [
            _frame(1.10, 1.10),
            _frame(1.20, 1.09),
            _frame(1.30, 1.08),
        ],
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="DFT/iter_1/active_learning/evaluated.traj",
    )

    xyz_text, meta = _latest_restart_structure_xyz_for_run(run)

    assert xyz_text is not None
    seeded = ase_read(StringIO(xyz_text), format="xyz")
    assert seeded.get_distance(0, 1) == pytest.approx(0.95, abs=1.0e-8)
    assert meta.get("artifact_path") == "reaction_hub_final.xyz"
    assert "No evaluated.traj frame satisfies reasonable fmax <= 1 eV/A" in str(meta.get("warning", ""))
    assert meta.get("trajectory_frame_index") is None


@pytest.mark.django_db
def test_job_list_accepts_page_size_query_param(client):
    profile = CalculationSettingsProfile.objects.create(name="p_jobs_page_size")
    wf = WorkflowDefinition.objects.create(name="wf_jobs_page_size")
    system = HPCSystem.objects.create(
        name="archer2_jobs_page_size",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    for idx in range(12):
        run = WorkflowRun.objects.create(
            workflow=wf,
            run_name=f"paged-run-{idx}",
            working_directory=f"/work/PROJECT/GROUP/user/reaction_hub_runs/paged-run-{idx}",
            status="initiated",
            settings_profile=profile,
        )
        JobRecord.objects.create(
            workflow_run=run,
            hpc_system=system,
            scheduler_job_id=str(77000 + idx),
            job_name=f"paged-run-{idx}",
            status=JobStatus.INITIATED,
        )

    resp = client.get("/jobs/?page_size=10")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200


@pytest.mark.django_db
def test_job_list_filters_failed_jobs_by_formula_and_allowed_species(client):
    profile = CalculationSettingsProfile.objects.create(name="job_species_profile")
    workflow = WorkflowDefinition.objects.create(name="job_species_workflow", code_path="x.py")
    hpc = HPCSystem.objects.create(
        name="job_species_hpc",
        host="example.test",
        username="user",
    )
    for name, formula in (("pure_cu_failed", "Cu13"), ("alloy_failed", "Au6Cu7"), ("oxide_failed", "Cu13O")):
        run = WorkflowRun.objects.create(
            workflow=workflow,
            run_name=name,
            status="failed",
            settings_profile=profile,
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="final_structure_formula",
            value_text=formula,
        )
        JobRecord.objects.create(
            workflow_run=run,
            hpc_system=hpc,
            scheduler_job_id=name,
            job_name=name,
            status="failed",
        )

    formula_body = client.get("/jobs/?status=failed&formula=Au6Cu7").content.decode()
    assert "alloy_failed" in formula_body
    assert "pure_cu_failed" not in formula_body

    cu_body = client.get("/jobs/?status=failed&species=Cu").content.decode()
    assert "pure_cu_failed" in cu_body
    assert "alloy_failed" not in cu_body
    assert "oxide_failed" not in cu_body

    alloy_body = client.get("/jobs/?status=failed&species=Cu%2C+Au").content.decode()
    assert "pure_cu_failed" in alloy_body
    assert "alloy_failed" in alloy_body
    assert "oxide_failed" not in alloy_body


@pytest.mark.django_db
def test_result_detail_renders_with_structure_artifact(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p2")
    ads = ReferenceAdsorbate.objects.create(name="ads", formula="CO", settings_profile=profile)
    cluster = ReferenceNanocluster.objects.create(name="clus", composition="Cu13", settings_profile=profile)
    wf = WorkflowDefinition.objects.create(name="wf2")
    run_dir = tmp_path / "run-view"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="view-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        reference_adsorbate=ads,
        reference_nanocluster=cluster,
    )
    result = AdsorptionEnergyResult.objects.create(
        workflow_run=run,
        adsorbate=ads,
        nanocluster=cluster,
        settings_profile=profile,
        total_energy_eV=Decimal("-10.000000"),
        reference_energy_eV=Decimal("-9.500000"),
        adsorption_energy_eV=Decimal("-0.500000"),
        metadata={"ed_energy_eV": "0.500000", "eint_energy_eV": "-1.000000"},
    )
    xyz_path = run_dir / "final.xyz"
    xyz_path.write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    structure_artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="final",
        path="final.xyz",
    )

    resp = client.get(f"/results/energy/{result.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "Total Energy:</strong> -10.00 eV" in body
    assert "Reference Energy:</strong> -9.50 eV" in body
    assert "Adsorption Energy:</strong> -0.50 eV" in body
    assert "Adsorbate:</strong> CO" in body
    assert "Ed (Energy of Distortion):</strong> 0.50 eV" in body
    assert "Eint (Interaction Energy):</strong> -1.00 eV" in body
    assert "Structure Viewer" in body
    assert "final.xyz" in body
    assert f"/workflows/{run.pk}/artifacts/{structure_artifact.pk}/open-ase-gui/" in body
    assert f"/results/energy/{result.pk}/reprocess/" in body
    assert "Reprocess Result" in body


@pytest.mark.django_db
def test_result_reprocess_view_reingests_run_and_recomputes(monkeypatch, client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p2_reprocess")
    ads = ReferenceAdsorbate.objects.create(name="ads_reprocess", formula="CO", settings_profile=profile)
    cluster = ReferenceNanocluster.objects.create(name="clus_reprocess", composition="Cu13", settings_profile=profile)
    wf = WorkflowDefinition.objects.create(name="wf2_reprocess")
    run_dir = tmp_path / "run-reprocess"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="run-reprocess",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        reference_adsorbate=ads,
        reference_nanocluster=cluster,
    )
    result = AdsorptionEnergyResult.objects.create(
        workflow_run=run,
        adsorbate=ads,
        nanocluster=cluster,
        settings_profile=profile,
        total_energy_eV=Decimal("-10.000000"),
        reference_energy_eV=Decimal("-9.500000"),
        adsorption_energy_eV=Decimal("-0.500000"),
    )

    calls: dict[str, object] = {}

    def _fake_ingest_single_run(path, workflow_name="cplas_tddft_active_learning"):  # noqa: ANN001
        calls["ingest_path"] = str(path)
        calls["workflow_name"] = workflow_name
        return run

    def _fake_upsert(run_obj):  # noqa: ANN001
        calls["upsert_run_id"] = int(run_obj.pk)
        return result

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    monkeypatch.setattr("results.views.upsert_adsorption_energy_for_run", _fake_upsert)

    resp = client.post(f"/results/energy/{result.pk}/reprocess/")

    assert resp.status_code == 302
    assert resp["Location"] == f"/results/energy/{result.pk}/"
    assert calls["ingest_path"] == str(run_dir)
    assert calls["upsert_run_id"] == run.pk


@pytest.mark.django_db
def test_result_reprocess_view_recovers_false_failed_job_status(monkeypatch, client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p2_reprocess_recover")
    ads = ReferenceAdsorbate.objects.create(name="ads_reprocess_recover", formula="CO", settings_profile=profile)
    cluster = ReferenceNanocluster.objects.create(name="clus_reprocess_recover", composition="Cu13", settings_profile=profile)
    wf = WorkflowDefinition.objects.create(name="wf2_reprocess_recover")
    run_dir = tmp_path / "run-reprocess-recover"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="run-reprocess-recover",
        working_directory=str(run_dir),
        status="failed",
        settings_profile=profile,
        reference_adsorbate=ads,
        reference_nanocluster=cluster,
    )
    result = AdsorptionEnergyResult.objects.create(
        workflow_run=run,
        adsorbate=ads,
        nanocluster=cluster,
        settings_profile=profile,
        total_energy_eV=Decimal("-10.000000"),
        reference_energy_eV=Decimal("-9.500000"),
        adsorption_energy_eV=Decimal("-0.500000"),
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )
    system = HPCSystem.objects.create(
        name="archer2_reprocess_recover",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="77801",
        job_name="run-reprocess-recover",
        status=JobStatus.FAILED,
        failure_reason=(
            "Scheduler reported completed, but no computed properties were ingested; "
            "treating job as failed."
        ),
    )

    def _fake_ingest_single_run(path, workflow_name="cplas_tddft_active_learning"):  # noqa: ANN001
        del path, workflow_name
        return run

    monkeypatch.setattr("ingest.services.ingest_single_run", _fake_ingest_single_run)
    monkeypatch.setattr("results.views.upsert_adsorption_energy_for_run", lambda run_obj: result)

    resp = client.post(f"/results/energy/{result.pk}/reprocess/")

    assert resp.status_code == 302
    job.refresh_from_db()
    run.refresh_from_db()
    assert job.status == JobStatus.COMPLETED
    assert run.status == "completed"
    assert job.failure_reason == ""
    assert JobStatusEvent.objects.filter(job=job, status=JobStatus.COMPLETED).exists()


@pytest.mark.django_db
def test_result_detail_renders_open_ase_button_for_trajectory_artifact(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p2_traj")
    ads = ReferenceAdsorbate.objects.create(name="ads_traj", formula="CO", settings_profile=profile)
    cluster = ReferenceNanocluster.objects.create(name="clus_traj", composition="Cu13", settings_profile=profile)
    wf = WorkflowDefinition.objects.create(name="wf2_traj")
    run_dir = tmp_path / "run-view-traj"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="view-run-traj",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        reference_adsorbate=ads,
        reference_nanocluster=cluster,
    )
    result = AdsorptionEnergyResult.objects.create(
        workflow_run=run,
        adsorbate=ads,
        nanocluster=cluster,
        settings_profile=profile,
        total_energy_eV=Decimal("-10.000000"),
        reference_energy_eV=Decimal("-9.500000"),
        adsorption_energy_eV=Decimal("-0.500000"),
    )

    traj_path = run_dir / "evaluated.traj"
    write(
        traj_path,
        [
            Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]),
            Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]]),
        ],
    )
    traj_artifact = Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="evaluated.traj",
    )

    resp = client.get(f"/results/energy/{result.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "trajectory:evaluated.traj" in body
    assert f"/workflows/{run.pk}/artifacts/{traj_artifact.pk}/open-ase-gui/" in body


@pytest.mark.django_db
def test_result_detail_renders_xyzrender_image_artifacts(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="p2_render")
    ads = ReferenceAdsorbate.objects.create(name="ads_render", formula="CO", settings_profile=profile)
    cluster = ReferenceNanocluster.objects.create(name="clus_render", composition="Cu13", settings_profile=profile)
    wf = WorkflowDefinition.objects.create(name="wf2_render")
    run_dir = tmp_path / "run-view-render"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="view-run-render",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        reference_adsorbate=ads,
        reference_nanocluster=cluster,
    )
    result = AdsorptionEnergyResult.objects.create(
        workflow_run=run,
        adsorbate=ads,
        nanocluster=cluster,
        settings_profile=profile,
        total_energy_eV=Decimal("-10.000000"),
        reference_energy_eV=Decimal("-9.500000"),
        adsorption_energy_eV=Decimal("-0.500000"),
    )

    (run_dir / "test.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
        encoding="utf-8",
    )
    (run_dir / "orca.gif").write_bytes(b"GIF89a")
    Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="render:test.svg",
        path="test.svg",
    )
    gif_artifact = Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="render:orca.gif",
        path="orca.gif",
    )

    resp = client.get(f"/results/energy/{result.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "Rendered Structure Images" in body
    assert "render:test.svg" in body
    assert "render:orca.gif" in body
    assert f"/results/artifact/{gif_artifact.pk}/content/" in body
    assert f'href="/results/artifact/{gif_artifact.pk}/content/"' in body
    assert f'href="/results/artifact/{gif_artifact.pk}/"' not in body
    assert f"/workflows/{run.pk}/artifacts/{gif_artifact.pk}/delete-image/" in body


@pytest.mark.django_db
def test_apply_reference_selection_to_run_creates_adsorbate_reference(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="ref_profile")
    wf = WorkflowDefinition.objects.create(name="wf-ref")
    run_dir = tmp_path / "run-ref-ads"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="ref-ads-run",
        working_directory=str(run_dir),
        status="initiated",
        settings_profile=profile,
    )

    message = _apply_reference_selection_to_run(
        run=run,
        reference_intent=ReferenceIntent.ADSORBATE,
        reference_name="CO_reference",
        structure_meta={"formula": "CO", "atom_count": 2},
    )
    run.save()
    run.refresh_from_db()

    assert run.reference_adsorbate is not None
    assert run.reference_adsorbate.name == "CO_reference"
    assert run.reference_adsorbate.formula == "CO"
    assert "adsorbate reference" in message.lower()


@pytest.mark.django_db
def test_apply_reference_selection_to_run_creates_nanocluster_reference(tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="ref_profile_cluster")
    wf = WorkflowDefinition.objects.create(name="wf-ref-cluster")
    run_dir = tmp_path / "run-ref-cluster"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="ref-cluster-run",
        working_directory=str(run_dir),
        status="initiated",
        settings_profile=profile,
    )

    message = _apply_reference_selection_to_run(
        run=run,
        reference_intent=ReferenceIntent.NANOCLUSTER,
        reference_name="Au13_reference",
        structure_meta={"formula": "Au13", "atom_count": 13},
    )
    run.save()
    run.refresh_from_db()

    assert run.reference_nanocluster is not None
    assert run.reference_nanocluster.name == "Au13_reference"
    assert run.reference_nanocluster.composition == "Au13"
    assert run.reference_nanocluster.atom_count == 13
    assert "nanocluster reference" in message.lower()


@pytest.mark.django_db
def test_results_dashboard_renders_formula_based_match(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="dash_profile")
    wf = WorkflowDefinition.objects.create(name="wf-dash", code_path="x.py")

    ads = ReferenceAdsorbate.objects.create(name="CO_ref_dash", formula="CO", settings_profile=profile)
    clus = ReferenceNanocluster.objects.create(name="Cu13_ref_dash", composition="Cu13", settings_profile=profile)

    run_ads = WorkflowRun.objects.create(
        workflow=wf,
        run_name="co_ref_run",
        working_directory=str(tmp_path / "co_ref"),
        status="completed",
        settings_profile=profile,
        reference_adsorbate=ads,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=wf,
        run_name="cu13_ref_run",
        working_directory=str(tmp_path / "cu13_ref"),
        status="completed",
        settings_profile=profile,
        reference_nanocluster=clus,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=wf,
        run_name="cu13co_system_run",
        working_directory=str(tmp_path / "cu13co_system"),
        status="completed",
        settings_profile=profile,
        metadata={"structure": {"formula": "Cu13CO"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.0,
    )
    upsert_adsorption_energy_for_run(run_system)

    dashboard_resp = client.get("/results/dashboard/")
    dashboard_body = dashboard_resp.content.decode("utf-8", errors="replace")
    results_resp = client.get("/results/")
    results_body = results_resp.content.decode("utf-8", errors="replace")

    assert dashboard_resp.status_code == 200
    assert "Adsorption Dashboard" in dashboard_body
    assert "cu13co_system_run" in dashboard_body
    assert results_resp.status_code == 200
    assert "Adsorption Dashboard" not in results_body


@pytest.mark.django_db
def test_results_dashboard_csv_export(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="dash_csv_profile")
    wf = WorkflowDefinition.objects.create(name="wf-dash-csv", code_path="x.py")

    ads = ReferenceAdsorbate.objects.create(name="CO_ref_csv", formula="CO", settings_profile=profile)
    clus = ReferenceNanocluster.objects.create(name="Cu13_ref_csv", composition="Cu13", settings_profile=profile)

    run_ads = WorkflowRun.objects.create(
        workflow=wf,
        run_name="co_ref_csv_run",
        working_directory=str(tmp_path / "co_ref_csv"),
        status="completed",
        settings_profile=profile,
        reference_adsorbate=ads,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=wf,
        run_name="cu13_ref_csv_run",
        working_directory=str(tmp_path / "cu13_ref_csv"),
        status="completed",
        settings_profile=profile,
        reference_nanocluster=clus,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=wf,
        run_name="cu13co_csv_run",
        working_directory=str(tmp_path / "cu13co_csv"),
        status="completed",
        settings_profile=profile,
        metadata={"structure": {"formula": "Cu13CO"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.0,
    )
    upsert_adsorption_energy_for_run(run_system)

    resp = client.get("/results/export/dashboard.csv")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "text/csv" in resp["Content-Type"]
    assert "run_id,run_name,status,settings_profile" in body
    assert "cu13co_csv_run" in body


@pytest.mark.django_db
def test_run_energy_dataset_basket_and_csv_export(client):
    profile = CalculationSettingsProfile.objects.create(name="energy_basket_profile")
    other_profile = CalculationSettingsProfile.objects.create(name="energy_basket_other_profile")
    workflow = WorkflowDefinition.objects.create(name="energy_basket_workflow", code_path="x.py")
    runs = []
    for name, energy, formula in (
        ("scan_a_001", -10.0, "Cu13CO"),
        ("scan_a_002", -12.5, "Cu13N2"),
        ("other_001", -99.0, "Au13CO"),
    ):
        run = WorkflowRun.objects.create(
            workflow=workflow,
            run_name=name,
            working_directory=f"/tmp/{name}",
            status="completed",
            settings_profile=profile,
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="final_single_point_energy_eV",
            unit="eV",
            value_float=energy,
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="final_structure_formula",
            value_text=formula,
        )
        runs.append(run)
    other_profile_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="scan_a_other_settings",
        working_directory="/tmp/scan_a_other_settings",
        status="completed",
        settings_profile=other_profile,
    )
    ComputedProperty.objects.create(
        workflow_run=other_profile_run,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-8.0,
    )
    ComputedProperty.objects.create(
        workflow_run=other_profile_run,
        name="final_structure_formula",
        value_text="Cu13CO",
    )
    ComputedProperty.objects.create(
        workflow_run=runs[1],
        name="mixing_energy_eV",
        unit="eV",
        value_float=-0.25,
    )
    ComputedProperty.objects.create(
        workflow_run=runs[0],
        name="mixing_energy_eV",
        unit="eV",
        value_float=0.5,
    )
    ComputedProperty.objects.create(
        workflow_run=runs[0],
        name="final_tddft_transitions",
        value_text="transitions",
        metadata={"transitions": [{"state": 1}]},
    )
    ComputedProperty.objects.create(
        workflow_run=runs[1],
        name="spectral_dataset",
        value_text="schema:1",
        metadata={
            "datasets": {
                "final_tddft": {
                    "ct_states": [{"state": 1}],
                }
            }
        },
    )
    runs[1].metadata = {"legacy_catalog": {"calculation_id": "scan-a-002"}}
    runs[1].save(update_fields=["metadata"])
    for run, total, per_atom in ((runs[0], 0.5, 0.05), (runs[1], 1.25, 0.1)):
        ComputedProperty.objects.create(
            workflow_run=run,
            name="atomization_energy_eV",
            unit="eV",
            value_float=total,
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="atomization_energy_per_atom_eV",
            unit="eV/atom",
            value_float=per_atom,
        )
    adsorbate = ReferenceAdsorbate.objects.create(
        name="energy_basket_adsorbate",
        formula="CO",
        settings_profile=profile,
    )
    nanocluster = ReferenceNanocluster.objects.create(
        name="energy_basket_cluster",
        composition="Cu13",
        settings_profile=profile,
    )
    AdsorptionEnergyResult.objects.create(
        workflow_run=runs[0],
        adsorbate=adsorbate,
        nanocluster=nanocluster,
        settings_profile=profile,
        total_energy_eV=Decimal("-10.000000"),
        reference_energy_eV=Decimal("-9.000000"),
        adsorption_energy_eV=Decimal("-1.000000"),
    )
    AdsorptionEnergyResult.objects.create(
        workflow_run=runs[1],
        adsorbate=adsorbate,
        nanocluster=nanocluster,
        settings_profile=profile,
        total_energy_eV=Decimal("-12.500000"),
        reference_energy_eV=Decimal("-10.500000"),
        adsorption_energy_eV=Decimal("-2.000000"),
    )

    list_resp = client.get("/results/run-energies/?prefix=scan_a_&sort=energy")
    list_body = list_resp.content.decode("utf-8", errors="replace")

    assert list_resp.status_code == 200
    assert "scan_a_001" in list_body
    assert "scan_a_002" in list_body
    assert "other_001" not in list_body
    assert "select-all-run-energies" in list_body
    assert "Cu13CO" in list_body
    assert "scan_a_other_settings" in list_body

    settings_resp = client.get(f"/results/run-energies/?prefix=scan_a_&settings_profile={profile.pk}")
    settings_body = settings_resp.content.decode("utf-8", errors="replace")
    assert settings_resp.status_code == 200
    assert "scan_a_001" in settings_body
    assert "scan_a_002" in settings_body
    assert "scan_a_other_settings" not in settings_body

    formula_resp = client.get("/results/run-energies/?formula=Cu13CO")
    formula_body = formula_resp.content.decode("utf-8", errors="replace")
    assert formula_resp.status_code == 200
    assert "scan_a_001" in formula_body
    assert "scan_a_002" not in formula_body
    assert "other_001" not in formula_body

    tddft_body = client.get("/results/run-energies/?data=tddft").content.decode()
    assert "scan_a_001" in tddft_body
    assert "scan_a_002" not in tddft_body
    assert "TDDFT / spectrum" in tddft_body

    ct_body = client.get("/results/run-energies/?data=ct&origin=imported").content.decode()
    assert "scan_a_002" in ct_body
    assert "scan_a_001" not in ct_body
    assert "Charge transfer" in ct_body

    combined_body = client.get("/results/run-energies/?data=tddft&data=ct").content.decode()
    assert "scan_a_001" not in combined_body
    assert "scan_a_002" not in combined_body
    assert "No run energies match these filters" in combined_body

    add_resp = client.post(
        "/results/run-energies/",
        {
            "run_ids": [str(runs[0].pk), str(runs[1].pk)],
            "prefix": "scan_a_",
            "settings_profile": str(profile.pk),
        },
    )
    assert add_resp.status_code == 302
    assert f"settings_profile={profile.pk}" in add_resp["Location"]

    checkout_resp = client.get("/results/energy-dataset/?sort=energy")
    checkout_body = checkout_resp.content.decode("utf-8", errors="replace")
    assert checkout_resp.status_code == 200
    assert checkout_body.index("scan_a_002") < checkout_body.index("scan_a_001")
    assert "Cu13N2" in checkout_body
    assert "Mixing Energy (eV)" in checkout_body
    assert "Mixing Energy/Atom (eV)" in checkout_body
    assert "Atomization Energy (eV)" in checkout_body
    assert "Atomization Energy/Atom (eV)" in checkout_body
    assert "1.250000" in checkout_body
    assert "-0.250000" in checkout_body

    concentration_resp = client.get(
        "/results/energy-dataset/?element_a=Cu&element_a_min=80&element_b=N&element_b_max=10"
    )
    concentration_body = concentration_resp.content.decode("utf-8", errors="replace")
    assert concentration_resp.status_code == 200
    assert "scan_a_001" in concentration_body
    assert "scan_a_002" not in concentration_body
    assert "Element A Concentration (%)" in concentration_body
    assert "86.666667" in concentration_body
    assert "0.000000" in concentration_body

    concentration_sort_resp = client.get(
        "/results/energy-dataset/?element_b=N&sort=element_b_concentration"
    )
    concentration_sort_body = concentration_sort_resp.content.decode(
        "utf-8", errors="replace"
    )
    assert concentration_sort_resp.status_code == 200
    assert concentration_sort_body.index("scan_a_001") < concentration_sort_body.index(
        "scan_a_002"
    )
    assert "sort=-element_b_concentration" in concentration_sort_body

    custom_resp = client.get(
        "/results/energy-dataset/?sort=energy&columns=run_name&columns=mixing_energy_eV&columns=atomization_energy_eV&columns=adsorption_energy_eV"
    )
    custom_body = custom_resp.content.decode("utf-8", errors="replace")
    assert custom_resp.status_code == 200
    assert "Mixing Energy (eV)" in custom_body
    assert "Atomization Energy (eV)" in custom_body
    assert "Adsorption Energy (eV)" in custom_body
    assert "-0.250000" in custom_body
    assert "-1.000000" in custom_body
    assert "sort=mixing_energy_eV" in custom_body
    assert "sort=atomization_energy_eV" in custom_body
    assert "sort=adsorption_energy_eV" in custom_body

    mixing_sort_resp = client.get(
        "/results/energy-dataset/?sort=mixing_energy_eV&columns=run_name&columns=mixing_energy_eV"
    )
    mixing_sort_body = mixing_sort_resp.content.decode("utf-8", errors="replace")
    assert mixing_sort_resp.status_code == 200
    assert mixing_sort_body.index("scan_a_002") < mixing_sort_body.index("scan_a_001")

    atomization_sort_resp = client.get(
        "/results/energy-dataset/?sort=atomization_energy_eV&columns=run_name&columns=atomization_energy_eV"
    )
    atomization_sort_body = atomization_sort_resp.content.decode("utf-8", errors="replace")
    assert atomization_sort_resp.status_code == 200
    assert atomization_sort_body.index("scan_a_001") < atomization_sort_body.index("scan_a_002")

    adsorption_sort_resp = client.get(
        "/results/energy-dataset/?sort=adsorption_energy_eV&columns=run_name&columns=adsorption_energy_eV"
    )
    adsorption_sort_body = adsorption_sort_resp.content.decode("utf-8", errors="replace")
    assert adsorption_sort_resp.status_code == 200
    assert adsorption_sort_body.index("scan_a_002") < adsorption_sort_body.index("scan_a_001")

    csv_resp = client.get(
        "/results/export/energy-dataset.csv?sort=energy&columns=run_name&columns=mixing_energy_eV&columns=atomization_energy_eV&columns=adsorption_energy_eV"
    )
    csv_body = csv_resp.content.decode("utf-8", errors="replace")
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp["Content-Type"]
    assert "Run,Mixing Energy (eV),Atomization Energy (eV),Adsorption Energy (eV)" in csv_body
    assert "scan_a_002" in csv_body
    assert "-0.250000" in csv_body
    assert "other_001" not in csv_body


@pytest.mark.django_db
def test_run_energy_species_filter_is_additive_and_exclusive(client):
    profile = CalculationSettingsProfile.objects.create(name="energy_species_profile")
    workflow = WorkflowDefinition.objects.create(name="energy_species_workflow", code_path="x.py")
    for name, formula in (("pure_cu", "Cu13"), ("alloy", "Au6Cu7"), ("supported", "Cu13O")):
        run = WorkflowRun.objects.create(
            workflow=workflow,
            run_name=name,
            status="completed",
            settings_profile=profile,
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="final_single_point_energy_eV",
            unit="eV",
            value_float=-1.0,
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="final_structure_formula",
            value_text=formula,
        )

    cu_body = client.get("/results/run-energies/?species=Cu").content.decode()
    assert "pure_cu" in cu_body
    assert "alloy" not in cu_body
    assert "supported" not in cu_body

    alloy_body = client.get("/results/run-energies/?species=Cu&species=Au").content.decode()
    assert "pure_cu" in alloy_body
    assert "alloy" in alloy_body
    assert "supported" not in alloy_body


@pytest.mark.django_db
def test_results_list_hides_legacy_coadsorbate_screening_results(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="result_list_hide_coads_profile")
    workflow = WorkflowDefinition.objects.create(name="wf-result-list-hide-coads", code_path="x.py")
    coadsorbate_workflow = WorkflowDefinition.objects.create(
        name="wf-result-list-hide-coads-screening",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )

    ads = ReferenceAdsorbate.objects.create(
        name="CO_ref_result_list_hide_coads",
        formula="CO",
        settings_profile=profile,
    )
    clus = ReferenceNanocluster.objects.create(
        name="Cu13_ref_result_list_hide_coads",
        composition="Cu13",
        settings_profile=profile,
    )

    run_system_visible = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="result_list_visible_system_run",
        working_directory=str(tmp_path / "result_list_visible_system"),
        status="completed",
        settings_profile=profile,
        metadata={"structure": {"formula": "Cu13CO"}},
    )
    run_system_hidden = WorkflowRun.objects.create(
        workflow=coadsorbate_workflow,
        run_name="result_list_hidden_coads_screening_run",
        working_directory=str(tmp_path / "result_list_hidden_coads_screening"),
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {"run_mode": "coadsorbate_placement"},
            "structure": {"formula": "Cu13CO"},
        },
    )

    AdsorptionEnergyResult.objects.create(
        workflow_run=run_system_visible,
        adsorbate=ads,
        nanocluster=clus,
        settings_profile=profile,
        total_energy_eV=Decimal("-111.000000"),
        reference_energy_eV=Decimal("-110.000000"),
        adsorption_energy_eV=Decimal("-1.000000"),
    )
    AdsorptionEnergyResult.objects.create(
        workflow_run=run_system_hidden,
        adsorbate=ads,
        nanocluster=clus,
        settings_profile=profile,
        total_energy_eV=Decimal("-111.500000"),
        reference_energy_eV=Decimal("-110.000000"),
        adsorption_energy_eV=Decimal("-1.500000"),
    )

    resp = client.get("/results/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "result_list_visible_system_run" in body
    assert "result_list_hidden_coads_screening_run" not in body


@pytest.mark.django_db
def test_results_list_supports_ordering_by_eint(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="result_list_eint_order_profile")
    workflow = WorkflowDefinition.objects.create(name="wf-result-list-eint-order", code_path="x.py")
    ads = ReferenceAdsorbate.objects.create(
        name="CO_ref_result_list_eint_order",
        formula="CO",
        settings_profile=profile,
    )
    clus = ReferenceNanocluster.objects.create(
        name="Cu13_ref_result_list_eint_order",
        composition="Cu13",
        settings_profile=profile,
    )

    run_low = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="result_list_eint_low",
        working_directory=str(tmp_path / "result_list_eint_low"),
        status="completed",
        settings_profile=profile,
    )
    run_high = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="result_list_eint_high",
        working_directory=str(tmp_path / "result_list_eint_high"),
        status="completed",
        settings_profile=profile,
    )

    AdsorptionEnergyResult.objects.create(
        workflow_run=run_low,
        adsorbate=ads,
        nanocluster=clus,
        settings_profile=profile,
        total_energy_eV=Decimal("-111.000000"),
        reference_energy_eV=Decimal("-110.000000"),
        adsorption_energy_eV=Decimal("-1.000000"),
        metadata={"eint_energy_eV": "-1.500000"},
    )
    AdsorptionEnergyResult.objects.create(
        workflow_run=run_high,
        adsorbate=ads,
        nanocluster=clus,
        settings_profile=profile,
        total_energy_eV=Decimal("-110.400000"),
        reference_energy_eV=Decimal("-110.000000"),
        adsorption_energy_eV=Decimal("-0.400000"),
        metadata={"eint_energy_eV": "0.250000"},
    )

    resp = client.get("/results/?sort=-eint_energy_eV")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert body.index("result_list_eint_high") < body.index("result_list_eint_low")


@pytest.mark.django_db
def test_artifact_content_page_and_file_view_for_image(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="image_profile")
    wf = WorkflowDefinition.objects.create(name="wf-image", code_path="x.py")
    run_dir = tmp_path / "run-image"
    run_dir.mkdir(parents=True)
    plot_path = run_dir / "stacked_uvvis.png"
    plot_path.write_bytes(b"fake_png")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="image-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="plot:stacked_uvvis",
        path="stacked_uvvis.png",
    )

    page_resp = client.get(f"/results/artifact/{artifact.pk}/content/")
    file_resp = client.get(f"/results/artifact/{artifact.pk}/file/")
    body = page_resp.content.decode("utf-8", errors="replace")

    assert page_resp.status_code == 200
    assert "Image Preview" in body
    assert f"/results/artifact/{artifact.pk}/file/" in body
    assert "Download File" in body
    assert file_resp.status_code == 200
    assert "image/png" in file_resp["Content-Type"]


@pytest.mark.django_db
def test_artifact_content_page_previews_text_and_supports_download(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="text_preview_profile")
    wf = WorkflowDefinition.objects.create(name="wf-text-preview", code_path="x.py")
    run_dir = tmp_path / "run-text-preview"
    run_dir.mkdir(parents=True)
    json_path = run_dir / "reaction_hub_summary.json"
    json_path.write_text('{"scan":{"peak_energy_eV":1.23},"ok":true}', encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="text-preview-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="json",
        label="summary:reaction_hub_summary",
        path="reaction_hub_summary.json",
    )

    page_resp = client.get(f"/results/artifact/{artifact.pk}/content/")
    download_resp = client.get(f"/results/artifact/{artifact.pk}/file/?download=1")
    body = page_resp.content.decode("utf-8", errors="replace")

    assert page_resp.status_code == 200
    assert "Text Preview" in body
    assert "peak_energy_eV" in body
    assert "Download File" in body
    assert download_resp.status_code == 200
    assert "attachment;" in str(download_resp["Content-Disposition"])


@pytest.mark.django_db
def test_artifact_content_uses_inferred_local_cache_path_for_remote_summary_artifact(
    client, tmp_path: Path, settings
):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(name="remote_summary_cache_profile")
    wf = WorkflowDefinition.objects.create(name="wf-remote-summary-cache", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="remote-summary-cache-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/remote-summary-cache-run/12588039",
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="plot:converged_state_wavelengths",
        path="DFT/scan_plots/converged_state_wavelengths.png",
        metadata={"source": "remote_summary"},
    )
    inferred_cache_path = (
        Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()
        / str(run.run_uid)
        / "DFT"
        / "scan_plots"
        / "converged_state_wavelengths.png"
    )
    inferred_cache_path.parent.mkdir(parents=True, exist_ok=True)
    inferred_cache_path.write_bytes(b"fake_png")

    page_resp = client.get(f"/results/artifact/{artifact.pk}/content/")
    file_resp = client.get(f"/results/artifact/{artifact.pk}/file/")
    body = page_resp.content.decode("utf-8", errors="replace")

    assert page_resp.status_code == 200
    assert "Image Preview" in body
    assert str(inferred_cache_path) in body
    assert file_resp.status_code == 200
    assert "image/png" in file_resp["Content-Type"]


@pytest.mark.django_db
def test_artifact_file_resolves_windows_style_absolute_path_under_run_root(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="windows_style_path_profile")
    wf = WorkflowDefinition.objects.create(name="wf-windows-style-path", code_path="x.py")
    run_dir = tmp_path / "run-windows-style-path"
    image_dir = run_dir / "DFT" / "scan_plots"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "uvvis_iter_008.png"
    image_path.write_bytes(b"fake_png")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="windows-style-path-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    windows_like_path = f"C:{run_dir.as_posix()}/DFT/scan_plots/uvvis_iter_008.png"
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="plot:uvvis_iter_008",
        path=windows_like_path,
    )

    resp = client.get(f"/results/artifact/{artifact.pk}/file/")
    assert resp.status_code == 200
    assert "image/png" in resp["Content-Type"]


@pytest.mark.django_db
def test_remote_only_log_artifact_detail_does_not_404(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="remote_log_profile")
    wf = WorkflowDefinition.objects.create(name="wf-remote-log", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="remote-log-run",
        working_directory=str(tmp_path / "remote-log-run"),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="log",
        label="orca_out:remote_final",
        path="TDDFT/opt_iter_2/orca.out",
        metadata={
            "source": "remote_reference_only",
            "remote_only": True,
            "retained_on_hpc": True,
            "remote_path": "/work/PROJECT/GROUP/user/reaction_hub_runs/remote-log-run/TDDFT/opt_iter_2/orca.out",
        },
    )

    detail_resp = client.get(f"/results/artifact/{artifact.pk}/")
    content_resp = client.get(f"/results/artifact/{artifact.pk}/content/")
    file_resp = client.get(f"/results/artifact/{artifact.pk}/file/")
    body = detail_resp.content.decode("utf-8", errors="replace")
    content_body = content_resp.content.decode("utf-8", errors="replace")

    assert detail_resp.status_code == 200
    assert "Retained on HPC only" in body
    assert content_resp.status_code == 200
    assert "Retained on HPC only" in content_body
    assert file_resp.status_code == 404


@pytest.mark.django_db
def test_workflow_open_artifact_ase_gui_local_structure(client, monkeypatch, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_gui_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_gui_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-gui-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-gui-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )

    launched: dict[str, object] = {}

    class _DummyProcess:
        def poll(self):
            return None

    def _fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr("workflows.views.subprocess.Popen", _fake_popen)

    resp = client.post(f"/workflows/{run.pk}/artifacts/{artifact.pk}/open-ase-gui/")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "ASE GUI launch command started" in payload["message"]
    assert isinstance(launched.get("command"), list)
    assert str(run_dir / "final.xyz") in " ".join(str(part) for part in launched["command"])


@pytest.mark.django_db
def test_workflow_open_artifact_ase_gui_local_database(client, monkeypatch, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_gui_db_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_gui_db_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-gui-db-run"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "coadsorption_final_for_dft.db"
    db_path.write_bytes(b"sqlite")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-gui-db-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="database",
        label="database:coadsorption_final_for_dft.db",
        path="coadsorption_final_for_dft.db",
    )

    launched: dict[str, object] = {}

    class _DummyProcess:
        def poll(self):
            return None

    def _fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr("workflows.views.subprocess.Popen", _fake_popen)

    resp = client.post(f"/workflows/{run.pk}/artifacts/{artifact.pk}/open-ase-gui/")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "ASE GUI launch command started" in payload["message"]
    assert isinstance(launched.get("command"), list)
    assert str(db_path) in " ".join(str(part) for part in launched["command"])


@pytest.mark.django_db
def test_workflow_open_artifact_ase_gui_local_trajectory(client, monkeypatch, tmp_path: Path):
    import sys
    profile = CalculationSettingsProfile.objects.create(name="workflow_gui_traj_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_gui_traj_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-gui-traj-run"
    run_dir.mkdir(parents=True)
    traj_path = run_dir / "final.traj"
    traj_path.write_bytes(b"dummy-traj-data")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-gui-traj-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:final.traj",
        path="final.traj",
    )

    launched: dict[str, object] = {}

    class _DummyProcess:
        def poll(self):
            return None

    def _fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr("workflows.views.subprocess.Popen", _fake_popen)

    resp = client.post(f"/workflows/{run.pk}/artifacts/{artifact.pk}/open-ase-gui/")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "ASE GUI launch command started" in payload["message"]
    assert isinstance(launched.get("command"), list)
    assert launched["command"][0] == sys.executable
    assert str(traj_path) in " ".join(str(part) for part in launched["command"])


@pytest.mark.django_db
def test_workflow_open_artifact_ase_gui_remote_only_rejected(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_gui_remote_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_gui_remote_wf", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-gui-remote-run",
        working_directory=str(tmp_path / "workflow-gui-remote-run"),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:remote_final.xyz",
        path="remote_final.xyz",
        metadata={"remote_only": True, "remote_path": "/work/user/run/remote_final.xyz"},
    )

    resp = client.post(f"/workflows/{run.pk}/artifacts/{artifact.pk}/open-ase-gui/")
    payload = resp.json()

    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "retained on HPC" in payload["error"]


@pytest.mark.django_db
def test_workflow_delete_image_artifact_removes_db_record_and_local_file(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_delete_image_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_delete_image_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-delete-image-run"
    run_dir.mkdir(parents=True)
    image_path = run_dir / "rendered.svg"
    image_path.write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>', encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-delete-image-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="render:rendered.svg",
        path="rendered.svg",
    )

    resp = client.post(f"/workflows/{run.pk}/artifacts/{artifact.pk}/delete-image/")
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert not Artifact.objects.filter(pk=artifact.pk).exists()
    assert not image_path.exists()


@pytest.mark.django_db
def test_workflow_delete_image_artifact_rejects_non_image_artifact(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_delete_non_image_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_delete_non_image_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-delete-non-image-run"
    run_dir.mkdir(parents=True)
    structure_path = run_dir / "final.xyz"
    structure_path.write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-delete-non-image-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )

    resp = client.post(f"/workflows/{run.pk}/artifacts/{artifact.pk}/delete-image/")
    payload = resp.json()

    assert resp.status_code == 400
    assert payload["ok"] is False
    assert Artifact.objects.filter(pk=artifact.pk).exists()
    assert structure_path.exists()


@pytest.mark.django_db
def test_workflow_detail_renders_xyzrender_action_button(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_button_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_button_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-button-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-button-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )

    resp = client.get(f"/workflows/{run.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/" in body
    assert f'id="workflow-render-rot-{artifact.pk}"' in body
    assert '<option value="x">x</option>' in body
    assert '<option value="y">y</option>' in body
    assert '<option value="z">z</option>' in body
    assert '<option value="xy">xy</option>' in body
    assert '<option value="xz">xz</option>' in body
    assert '<option value="yz">yz</option>' in body
    assert '<option value="xyz">xyz</option>' not in body
    assert "Render SVG" in body
    assert "Render GIF" in body


@pytest.mark.django_db
def test_workflow_detail_does_not_build_mlneb_comparison_on_overview(client, monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="workflow_non_neb_profile")
    run = WorkflowRun.objects.create(
        workflow=WorkflowDefinition.objects.create(name="ordinary optimisation"),
        run_name="ordinary-run",
        working_directory="/work/ordinary-run",
        settings_profile=profile,
    )

    def _unexpected_comparison(*args, **kwargs):
        raise AssertionError("MLNEB comparison should only be built on its own tab")

    monkeypatch.setattr("workflows.views._build_mlneb_compare_context", _unexpected_comparison)

    assert client.get(f"/workflows/{run.pk}/?tab=overview").status_code == 200


@pytest.mark.django_db
def test_workflow_detail_renders_xyzrender_action_button_for_scan_endpoints_db(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_button_db_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_button_db_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-button-db-run"
    run_dir.mkdir(parents=True)
    db_path = run_dir / "scan_endpoints.db"
    db_path.write_bytes(b"")
    artifact = Artifact.objects.create(
        workflow_run=WorkflowRun.objects.create(
            workflow=wf,
            run_name="workflow-render-button-db-run",
            working_directory=str(run_dir),
            status="completed",
            settings_profile=profile,
        ),
        kind="database",
        label="scan_endpoints_db",
        path="scan_endpoints.db",
    )

    resp = client.get(f"/workflows/{artifact.workflow_run.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert f"/workflows/{artifact.workflow_run.pk}/artifacts/{artifact.pk}/render-xyz/" in body
    assert f'id="workflow-render-rot-{artifact.pk}"' in body
    assert f'id="workflow-render-mode-{artifact.pk}"' in body
    assert f'data-mode-select="workflow-render-mode-{artifact.pk}"' in body
    assert '<option value="transition_state">Transition state image</option>' in body
    assert '<option value="first">First endpoint image</option>' in body
    assert '<option value="last">Last endpoint image</option>' in body
    assert '<option value="trajectory">Full scan (ping-pong)</option>' in body
    assert f"/workflows/{artifact.workflow_run.pk}/artifacts/{artifact.pk}/open-ase-gui/" in body
    assert 'method: "POST"' in body
    assert "!response.ok || !data.ok" in body
    assert "gif_rot: rot" in body


@pytest.mark.django_db
def test_workflow_detail_shows_batch_single_opts_link_for_database_artifact(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_batch_db_link_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_batch_db_link_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-batch-db-link-run"
    (run_dir / "scratch").mkdir(parents=True)
    db_path = run_dir / "scratch" / "coadsorption_final_for_dft.db"
    db_path.write_bytes(b"sqlite")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-batch-db-link-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="database",
        label="coadsorption_db:coadsorption_final_for_dft.db",
        path="scratch/coadsorption_final_for_dft.db",
    )

    resp = client.get(f"/workflows/{run.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert (
        f"/jobs/submit/?from_run={run.pk}&source_artifact={artifact.pk}&mode=single_optimisation&batch=1"
        in body
    )
    assert "Batch Single Opts" in body


@pytest.mark.django_db
def test_workflow_detail_renders_xyzrender_image_artifacts(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_gallery_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_gallery_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-gallery-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final_rot_xy.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
        encoding="utf-8",
    )
    (run_dir / "final_rot_xy.gif").write_bytes(b"GIF89a")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-gallery-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="render:final_rot_xy.svg",
        path="final_rot_xy.svg",
    )
    gif_artifact = Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="render:final_rot_xy.gif",
        path="final_rot_xy.gif",
    )

    resp = client.get(f"/workflows/{run.pk}/")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "Rendered Structure Images" in body
    assert "render:final_rot_xy.svg" in body
    assert "render:final_rot_xy.gif" in body
    assert f"/results/artifact/{gif_artifact.pk}/content/" in body
    assert f'href="/results/artifact/{gif_artifact.pk}/content/"' in body
    assert f'href="/results/artifact/{gif_artifact.pk}/"' not in body
    assert f"/workflows/{run.pk}/artifacts/{gif_artifact.pk}/delete-image/" in body


@pytest.mark.django_db
def test_workflow_detail_shows_smooth_resubmit_button_for_completed_scan_run(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_smooth_button_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_smooth_button_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-smooth-button-run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-smooth-button-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "tddft_bond_scan"}},
    )
    system = HPCSystem.objects.create(
        name="workflow_smooth_button_system",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="99040",
        job_name="workflow-smooth-button-run",
        status=JobStatus.COMPLETED,
    )

    resp = client.get(f"/workflows/{run.pk}/?tab=overview")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "TS Smoothing Resubmit" in body
    assert f'action="/jobs/{job.pk}/resubmit-smooth/"' in body


@pytest.mark.django_db
def test_workflow_detail_shows_replot_resubmit_button_for_completed_supported_run(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_replot_button_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_replot_button_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-replot-button-run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-replot-button-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "single_optimisation"}},
    )
    system = HPCSystem.objects.create(
        name="workflow_replot_button_system",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="99041",
        job_name="workflow-replot-button-run",
        status=JobStatus.COMPLETED,
    )

    resp = client.get(f"/workflows/{run.pk}/?tab=overview")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "Resubmit Replot" in body
    assert f'action="/jobs/{job.pk}/resubmit-replot/"' in body


@pytest.mark.django_db
def test_workflow_detail_links_postprocess_run_to_original_workflow(client):
    profile = CalculationSettingsProfile.objects.create(name="workflow_postprocess_source_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_postprocess_source_wf", code_path="x.py")
    source_run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="original-optimisation",
        working_directory="/work/original-optimisation",
        status="completed",
        settings_profile=profile,
    )
    postprocess_run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="ct-postprocess",
        working_directory="/work/ct-postprocess",
        status="completed",
        settings_profile=profile,
        metadata={
            "source_run_id": source_run.pk,
            "submission": {"run_mode": "charge_transfer"},
        },
    )

    body = client.get(f"/workflows/{postprocess_run.pk}/").content.decode("utf-8", errors="replace")

    assert f'href="/workflows/{source_run.pk}/"' in body
    assert "View Original Workflow" in body


@pytest.mark.django_db
def test_workflow_detail_marks_uncached_remote_summary_image_as_hpc_reference(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_remote_summary_badge_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_remote_summary_badge_wf", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-remote-summary-badge-run",
        working_directory="/work/PROJECT/GROUP/user/reaction_hub_runs/workflow-remote-summary-badge-run/12588039",
        status="completed",
        settings_profile=profile,
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="image",
        label="plot:converged_state_wavelengths",
        path="DFT/scan_plots/converged_state_wavelengths.png",
        metadata={"source": "remote_summary"},
    )

    resp = client.get(f"/workflows/{run.pk}/?tab=overview")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "plot:converged_state_wavelengths" in body
    assert "HPC path reference" in body


@pytest.mark.django_db
def test_workflow_detail_orders_and_formats_computed_properties_and_hides_path_column(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_props_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_props_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-props-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-props-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="b_text",
        value_text="zeta",
        unit="",
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="a_text",
        value_text="alpha",
        unit="",
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="b_num",
        value_float=2.7182818,
        unit="eV",
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="a_num",
        value_float=3.1415926,
        unit="eV",
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="wavelength",
        value_float=520.0,
        unit="nm",
    )

    resp = client.get(f"/workflows/{run.pk}/?tab=overview")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert body.index("<h3>Computed Properties</h3>") < body.index("<h3>Artifacts</h3>")
    assert body.index("<td>a_text</td>") < body.index("<td>b_text</td>")
    assert body.index("<td>b_text</td>") < body.index("<td>a_num</td>")
    assert body.index("<td>a_num</td>") < body.index("<td>b_num</td>")
    assert ">3.14<" in body
    assert ">2.72<" in body
    assert "3.1415926" not in body
    assert "2.7182818" not in body
    assert 'class="prop-val" data-raw-val="520.00" data-unit="nm"' in body
    assert 'id="workflow-show-artifact-paths"' in body
    assert 'class="artifact-path-col" style="display:none;"' in body


@pytest.mark.django_db
def test_workflow_detail_shows_adsorption_energy(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_adsorption_energy_profile")
    adsorbate = ReferenceAdsorbate.objects.create(
        name="workflow_adsorption_energy_adsorbate",
        formula="CO",
        settings_profile=profile,
    )
    cluster = ReferenceNanocluster.objects.create(
        name="workflow_adsorption_energy_cluster",
        composition="Cu13",
        settings_profile=profile,
    )
    wf = WorkflowDefinition.objects.create(name="workflow_adsorption_energy_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-adsorption-energy-run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-adsorption-energy-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
        reference_adsorbate=adsorbate,
        reference_nanocluster=cluster,
    )
    AdsorptionEnergyResult.objects.create(
        workflow_run=run,
        adsorbate=adsorbate,
        nanocluster=cluster,
        settings_profile=profile,
        total_energy_eV=Decimal("-10.000000"),
        reference_energy_eV=Decimal("-9.123456"),
        adsorption_energy_eV=Decimal("-0.876544"),
        metadata={"eint_energy_eV": "-1.234567"},
    )

    resp = client.get(f"/workflows/{run.pk}/?tab=overview")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "Adsorption energy" in body
    assert "-0.88 eV" in body
    assert "Interaction energy · Eint" in body
    assert "-1.23 eV" in body


@pytest.mark.django_db
def test_workflow_detail_artifacts_show_source_job_id_and_sort_by_highest_job_id(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_artifact_jobid_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_artifact_jobid_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-artifact-jobid-run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-artifact-jobid-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="workflow_artifact_jobid_system",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    older_job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="88001",
        job_name="older-job",
        status=JobStatus.COMPLETED,
    )
    newer_job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="99009",
        job_name="newer-job",
        status=JobStatus.COMPLETED,
    )
    Artifact.objects.create(
        workflow_run=run,
        job=older_job,
        kind="structure",
        label="artifact-from-older-job",
        path="88001/final.xyz",
    )
    Artifact.objects.create(
        workflow_run=run,
        job=newer_job,
        kind="structure",
        label="artifact-from-newer-job",
        path="99009/final.xyz",
    )
    Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="artifact-without-job-id",
        path="final.xyz",
    )

    resp = client.get(f"/workflows/{run.pk}/?tab=overview")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "<th>Source Job ID</th>" in body
    assert body.index("artifact-from-newer-job") < body.index("artifact-from-older-job")
    assert body.index("artifact-from-older-job") < body.index("artifact-without-job-id")
    assert "<code>99009</code>" in body
    assert "<code>88001</code>" in body

    artifacts = list(resp.context["artifacts"])
    assert [artifact.label for artifact in artifacts] == [
        "artifact-from-newer-job",
        "artifact-from-older-job",
        "artifact-without-job-id",
    ]
    assert str(getattr(artifacts[0], "source_scheduler_job_id", "")) == "99009"
    assert str(getattr(artifacts[1], "source_scheduler_job_id", "")) == "88001"
    assert str(getattr(artifacts[2], "source_scheduler_job_id", "")) == ""


@pytest.mark.django_db
def test_workflow_delete_duplicate_artifacts_view_removes_duplicates_by_label_ingestion_path_and_submission_id(
    client, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_dedupe_artifacts_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_dedupe_artifacts_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-dedupe-artifacts-run"
    run_dir.mkdir(parents=True)
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-dedupe-artifacts-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    system = HPCSystem.objects.create(
        name="workflow_dedupe_artifacts_system",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="12345",
        job_name="dedupe-artifacts-job",
        status=JobStatus.COMPLETED,
    )

    duplicate_a = Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="cache/evaluated_copy_a.traj",
        metadata={"remote_path": "/work/example/reaction_hub_runs/run/12345/DFT/active_learning/evaluated.traj"},
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="cache/evaluated_copy_b.traj",
        metadata={"remote_path": "/work/example/reaction_hub_runs/run/DFT/active_learning/evaluated.traj"},
    )
    Artifact.objects.create(
        workflow_run=run,
        job=job,
        kind="trajectory",
        label="trajectory:other.traj",
        path="cache/other.traj",
        metadata={"remote_path": "/work/example/reaction_hub_runs/run/12345/DFT/active_learning/other.traj"},
    )
    # Different submission id, should not be deduped with the above pair.
    Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="cache/evaluated_other_submission.traj",
        metadata={
            "remote_path": "/work/example/reaction_hub_runs/run/12345/DFT/active_learning/evaluated.traj",
            "scheduler_job_id": "99999",
        },
    )
    # Keep this duplicate by making it referenced from computed properties.
    ComputedProperty.objects.create(
        workflow_run=run,
        source_artifact=duplicate_a,
        name="test_prop",
        value_float=1.0,
        unit="eV",
    )

    resp = client.post(f"/workflows/{run.pk}/artifacts/delete-duplicates/")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["removed_count"] == 1
    assert payload["duplicate_groups"] == 1

    labels_paths = {
        (item.label, item.path)
        for item in Artifact.objects.filter(workflow_run=run).order_by("path")
    }
    assert ("trajectory:evaluated.traj", "cache/evaluated_copy_a.traj") in labels_paths
    assert ("trajectory:evaluated.traj", "cache/evaluated_copy_b.traj") not in labels_paths
    assert ("trajectory:evaluated.traj", "cache/evaluated_other_submission.traj") in labels_paths
    assert ("trajectory:other.traj", "cache/other.traj") in labels_paths


@pytest.mark.django_db
def test_workflow_detail_bond_scan_tab_links_by_formula_or_hash_and_filters_target(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_scan_compare_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_scan_compare_wf", code_path="x.py")

    def _make_run(
        *,
        name: str,
        formula: str,
        structure_hash: str,
        target_nm: float,
        start_e: float,
        peak_e: float,
        barrier_e: float,
    ) -> WorkflowRun:
        run_dir = tmp_path / name
        run_dir.mkdir(parents=True, exist_ok=True)
        run = WorkflowRun.objects.create(
            workflow=wf,
            run_name=name,
            working_directory=str(run_dir),
            status="completed",
            settings_profile=profile,
            metadata={
                "structure": {"formula": formula, "structure_sha256": structure_hash},
                "submission": {"run_mode": "tddft_bond_scan", "target_wavelength_nm": target_nm},
            },
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="bond_scan_start_energy_eV",
            unit="eV",
            value_float=start_e,
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="bond_scan_peak_energy_eV",
            unit="eV",
            value_float=peak_e,
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="bond_scan_activation_barrier_eV",
            unit="eV",
            value_float=barrier_e,
        )
        return run

    current = _make_run(
        name="scan_current",
        formula="CO",
        structure_hash="aaa111",
        target_nm=520.0,
        start_e=-200.0,
        peak_e=-199.2,
        barrier_e=0.8,
    )
    same_hash = _make_run(
        name="scan_same_hash",
        formula="CO",
        structure_hash="aaa111",
        target_nm=520.0,
        start_e=-201.0,
        peak_e=-200.0,
        barrier_e=1.0,
    )
    same_formula = _make_run(
        name="scan_same_formula",
        formula="CO",
        structure_hash="bbb222",
        target_nm=510.0,
        start_e=-198.0,
        peak_e=-197.6,
        barrier_e=0.4,
    )
    other = _make_run(
        name="scan_other",
        formula="H2O",
        structure_hash="ccc333",
        target_nm=520.0,
        start_e=-50.0,
        peak_e=-49.5,
        barrier_e=0.5,
    )

    resp = client.get(f"/workflows/{current.pk}/?tab=bond_scan")
    body = resp.content.decode("utf-8", errors="replace")

    assert resp.status_code == 200
    assert "Bond-Length Scan Comparison" in body
    assert same_hash.run_name in body
    assert same_formula.run_name in body
    assert other.run_name not in body

    resp_filtered = client.get(f"/workflows/{current.pk}/?tab=bond_scan&scan_target_nm=520.0")
    body_filtered = resp_filtered.content.decode("utf-8", errors="replace")
    assert resp_filtered.status_code == 200
    assert same_hash.run_name in body_filtered
    assert same_formula.run_name not in body_filtered


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_creates_render_artifacts(client, monkeypatch, settings, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"
    settings.REACTION_HUB_XYZRENDER_ROT_FRAMES = 200

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text
        assert (
            command[0] == "xyzrender"
            or ("-m" in command and command[command.index("-m") + 1] == "xyzrender")
        )
        assert "avogadro" not in str(env.get("PATH", "")).lower()
        assert "--gif-rot" in command
        assert command[command.index("--gif-rot") + 1] == "xz"
        assert "--rot-frames" in command
        out_dir = Path(str(cwd))
        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )
        if "-m" in command and command[command.index("-m") + 1] == "xyzrender":
            input_name = command[command.index("-m") + 2]
        else:
            input_name = command[1]
        (out_dir / f"{Path(input_name).stem}.gif").write_bytes(b"GIF89a")

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_rot": "xz"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "rotation 'xz'" in payload["message"]
    assert Artifact.objects.filter(
        workflow_run=run,
        path="final_render.svg",
        kind="image",
        label="render:final_render.svg",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="final_rot_xz.gif",
        kind="image",
        label="render:final_rot_xz.gif",
    ).exists()


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_can_render_svg_without_gif(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_svg_only_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_svg_only_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-svg-only-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-svg-only-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text, env
        assert command[0] == "xyzrender"
        assert "--gif-rot" not in command
        assert "--rot-frames" not in command
        assert "-go" not in command
        assert "--gif-trj" not in command
        out_dir = Path(str(cwd))
        input_path = out_dir / command[1]
        rendered_atoms = ase_read(str(input_path))
        rendered_positions = rendered_atoms.get_positions()
        original_second_position = [0.0, 0.0, 1.1]
        assert any(
            abs(float(rendered_positions[1][idx]) - original_second_position[idx]) > 1.0e-6
            for idx in range(3)
        )
        svg_name = command[command.index("-o") + 1]
        assert svg_name == "final_render_rot_xz.svg"
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_rot": "xz", "render_output": "svg"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    svg_artifact = Artifact.objects.get(
        workflow_run=run,
        path="final_render_rot_xz.svg",
        kind="image",
        label="render:final_render_rot_xz.svg",
    )
    assert svg_artifact.metadata.get("render_output") == "svg"
    assert svg_artifact.metadata.get("svg_orientation") == "xz"
    assert not Artifact.objects.filter(
        workflow_run=run,
        path="final_rot_xz.gif",
    ).exists()


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_defaults_rotation_to_x(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_default_rot_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_default_rot_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-default-rot-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-default-rot-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text
        assert (
            command[0] == "xyzrender"
            or ("-m" in command and command[command.index("-m") + 1] == "xyzrender")
        )
        assert "--gif-rot" in command
        assert command[command.index("--gif-rot") + 1] == "x"
        assert "avogadro" not in str(env.get("PATH", "")).lower()

        out_dir = Path(str(cwd))
        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )
        if "-m" in command and command[command.index("-m") + 1] == "xyzrender":
            input_name = command[command.index("-m") + 2]
        else:
            input_name = command[1]
        (out_dir / f"{Path(input_name).stem}.gif").write_bytes(b"GIF89a")

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "rotation 'x'" in payload["message"]
    assert Artifact.objects.filter(
        workflow_run=run,
        path="final_rot_x.gif",
        kind="image",
        label="render:final_rot_x.gif",
    ).exists()


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_uses_last_frame_for_trajectory_artifacts(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_last_frame_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_last_frame_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-last-frame-run"
    run_dir.mkdir(parents=True)

    traj_path = run_dir / "evaluated.traj"
    write(
        traj_path,
        [
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.10)]),
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.25)]),
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.45)]),
        ],
    )
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-last-frame-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="evaluated.traj",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text
        assert (
            command[0] == "xyzrender"
            or ("-m" in command and command[command.index("-m") + 1] == "xyzrender")
        )
        assert "avogadro" not in str(env.get("PATH", "")).lower()
        out_dir = Path(str(cwd))
        if "-m" in command and command[command.index("-m") + 1] == "xyzrender":
            input_name = command[command.index("-m") + 2]
        else:
            input_name = command[1]
        input_path = out_dir / input_name
        frames = ase_read(str(input_path), index=":")
        assert len(frames) == 1
        assert frames[0].get_distance(0, 1) == pytest.approx(1.45)
        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )
        (out_dir / f"{Path(input_name).stem}.gif").write_bytes(b"GIF89a")

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_rot": "xy"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert Artifact.objects.filter(
        workflow_run=run,
        path="evaluated_render.svg",
        kind="image",
        label="render:evaluated_render.svg",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="evaluated_rot_xy.gif",
        kind="image",
        label="render:evaluated_rot_xy.gif",
    ).exists()


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_can_render_full_ping_pong_for_trajectory_artifacts(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_full_traj_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_full_traj_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-full-traj-run"
    run_dir.mkdir(parents=True)

    traj_path = run_dir / "evaluated.traj"
    write(
        traj_path,
        [
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.10)]),
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.25)]),
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, 1.45)]),
        ],
    )
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-full-traj-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="evaluated.traj",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text
        assert command[0] == "xyzrender"
        assert "--gif-trj" in command
        assert "-go" in command
        assert "--gif-ts" not in command
        assert "--gif-rot" not in command
        assert "--rot-frames" not in command
        assert "avogadro" not in str(env.get("PATH", "")).lower()

        out_dir = Path(str(cwd))
        input_path = out_dir / command[1]
        frames = ase_read(str(input_path), index=":")
        assert [frame.get_distance(0, 1) for frame in frames] == pytest.approx([1.10, 1.25, 1.45, 1.25])

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

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_mode": "trajectory", "gif_rot": "xy"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "ping-pong trajectory" in payload["message"]
    assert Artifact.objects.filter(
        workflow_run=run,
        path="evaluated_ping_pong.svg",
        kind="image",
        label="render:evaluated_ping_pong.svg",
    ).exists()
    gif_artifact = Artifact.objects.get(
        workflow_run=run,
        path="evaluated_ping_pong.gif",
        kind="image",
        label="render:evaluated_ping_pong.gif",
    )
    assert gif_artifact.metadata.get("gif_mode") == "trajectory"
    assert gif_artifact.metadata.get("trajectory_pattern") == "ping_pong"
    assert gif_artifact.metadata.get("trajectory_source") == "artifact_frames"


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_rejects_trajectory_mode_for_non_trajectory_artifact(
    client,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_mode_reject_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_mode_reject_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-mode-reject-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-mode-reject-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_mode": "trajectory", "gif_rot": "xy"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "Trajectory GIF mode is only available" in payload["error"]


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_from_scan_endpoints_db_creates_ping_pong_gif(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_scan_db_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_scan_db_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-scan-db-run"
    run_dir.mkdir(parents=True)

    db_path = run_dir / "scan_endpoints.db"
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

    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-scan-db-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="database",
        label="scan_endpoints_db",
        path="scan_endpoints.db",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text
        assert command[0] == "xyzrender"
        assert "--gif-ts" not in command
        assert "--gif-rot" not in command
        assert "--rot-frames" not in command
        assert "avogadro" not in str(env.get("PATH", "")).lower()

        out_dir = Path(str(cwd))
        input_path = out_dir / command[1]
        frames = ase_read(str(input_path), index=":")
        assert len(frames) == 4
        assert [frame.get_distance(0, 1) for frame in frames] == pytest.approx([1.10, 1.20, 1.30, 1.20])
        if "--gif-trj" in command:
            assert "-go" in command
            assert all(frame.get_chemical_symbols()[-8:] == ["H"] * 8 for frame in frames)
        else:
            assert "-go" not in command
            assert all(len(frame) == 2 for frame in frames)

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

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_rot": "xz"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "ping-pong trajectory" in payload["message"]
    assert Artifact.objects.filter(
        workflow_run=run,
        path="scan_endpoints_ping_pong.svg",
        kind="image",
        label="render:scan_endpoints_ping_pong.svg",
    ).exists()
    gif_artifact = Artifact.objects.get(
        workflow_run=run,
        path="scan_endpoints_ping_pong.gif",
        kind="image",
        label="render:scan_endpoints_ping_pong.gif",
    )
    assert gif_artifact.metadata.get("gif_mode") == "trajectory"
    assert gif_artifact.metadata.get("trajectory_pattern") == "ping_pong"
    assert gif_artifact.metadata.get("xyzrender_gif_flag") == "--gif-trj"
    assert gif_artifact.metadata.get("fixed_scale_anchors") is True
    assert gif_artifact.metadata.get("fixed_scale_anchor_symbol") == "H"


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_from_scan_endpoints_db_creates_transition_state_image(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_scan_db_ts_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_scan_db_ts_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-scan-db-ts-run"
    run_dir.mkdir(parents=True)

    db_path = run_dir / "scan_endpoints.db"
    db = ase_connect(str(db_path))
    for iteration, energy, bond_length in (
        (1, -200.0, 1.10),
        (2, -199.2, 1.35),
        (3, -199.6, 1.20),
    ):
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, bond_length)]),
            data={"scan_bond_length": float(bond_length)},
            iteration=int(iteration),
            total_energy_eV=float(energy),
        )

    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-scan-db-ts-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="database",
        label="scan_endpoints_db",
        path="scan_endpoints.db",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text
        assert command[0] == "xyzrender"
        assert "--gif-trj" not in command
        assert "--gif-ts" not in command
        assert "--gif-rot" in command
        assert command[command.index("--gif-rot") + 1] == "yz"
        assert "avogadro" not in str(env.get("PATH", "")).lower()

        out_dir = Path(str(cwd))
        input_path = out_dir / command[1]
        frames = ase_read(str(input_path), index=":")
        assert len(frames) == 1
        assert frames[0].get_distance(0, 1) == pytest.approx(1.35)

        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )
        input_name = command[1]
        (out_dir / f"{Path(input_name).stem}.gif").write_bytes(b"GIF89a")

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_mode": "transition_state", "gif_rot": "yz"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "transition-state image" in payload["message"]
    assert Artifact.objects.filter(
        workflow_run=run,
        path="scan_endpoints_transition_state.svg",
        kind="image",
        label="render:scan_endpoints_transition_state.svg",
    ).exists()
    gif_artifact = Artifact.objects.get(
        workflow_run=run,
        path="scan_endpoints_transition_state_rot_yz.gif",
        kind="image",
        label="render:scan_endpoints_transition_state_rot_yz.gif",
    )
    assert gif_artifact.metadata.get("scan_endpoint_render_mode") == "transition_state"
    assert gif_artifact.metadata.get("trajectory_source") == "scan_endpoints_db"
    assert gif_artifact.metadata.get("gif_rotation") == "yz"


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_from_scan_endpoints_db_creates_first_and_last_images(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_scan_db_endpoints_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_scan_db_endpoints_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-scan-db-endpoints-run"
    run_dir.mkdir(parents=True)

    db_path = run_dir / "scan_endpoints.db"
    db = ase_connect(str(db_path))
    for iteration, energy, bond_length in (
        (1, -200.0, 1.10),
        (2, -199.2, 1.35),
        (3, -199.6, 1.60),
    ):
        db.write(
            Atoms("CO", positions=[(0.0, 0.0, 0.0), (0.0, 0.0, bond_length)]),
            data={"scan_bond_length": float(bond_length)},
            iteration=int(iteration),
            total_energy_eV=float(energy),
        )

    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-scan-db-endpoints-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="database",
        label="scan_endpoints_db",
        path="scan_endpoints.db",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    rendered_distances: list[float] = []

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text, env
        assert command[0] == "xyzrender"
        assert "--gif-trj" not in command
        assert "--gif-rot" not in command
        assert "-go" not in command

        out_dir = Path(str(cwd))
        input_path = out_dir / command[1]
        frames = ase_read(str(input_path), index=":")
        assert len(frames) == 1
        rendered_distances.append(float(frames[0].get_distance(0, 1)))

        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    for mode in ("first", "last"):
        resp = client.post(
            f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
            data=json.dumps({"gif_mode": mode, "gif_rot": "x", "render_output": "svg"}),
            content_type="application/json",
        )
        payload = resp.json()
        assert resp.status_code == 200
        assert payload["ok"] is True
        assert f"{mode} endpoint image" in payload["message"]

    assert rendered_distances == pytest.approx([1.10, 1.60])
    first_artifact = Artifact.objects.get(
        workflow_run=run,
        path="scan_endpoints_first_endpoint_rot_x.svg",
        kind="image",
        label="render:scan_endpoints_first_endpoint_rot_x.svg",
    )
    last_artifact = Artifact.objects.get(
        workflow_run=run,
        path="scan_endpoints_last_endpoint_rot_x.svg",
        kind="image",
        label="render:scan_endpoints_last_endpoint_rot_x.svg",
    )
    assert first_artifact.metadata.get("scan_endpoint_render_mode") == "first"
    assert last_artifact.metadata.get("scan_endpoint_render_mode") == "last"
    assert first_artifact.metadata.get("trajectory_source") == "scan_endpoints_db"
    assert last_artifact.metadata.get("trajectory_source") == "scan_endpoints_db"


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_respects_explicit_command_path(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_explicit_path_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_explicit_path_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-explicit-path-run"
    run_dir.mkdir(parents=True)
    (run_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-explicit-path-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="reaction_hub_final.xyz",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "C:/Avogadro/bin/xyzrender"

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text
        assert command[0] == "C:/Avogadro/bin/xyzrender"
        assert "avogadro" not in str(env.get("PATH", "")).lower()
        out_dir = Path(str(cwd))
        svg_name = command[command.index("-o") + 1]
        (out_dir / svg_name).write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2"></svg>',
            encoding="utf-8",
        )
        input_name = command[1]
        (out_dir / f"{Path(input_name).stem}.gif").write_bytes(b"GIF89a")

        class _Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        return _Completed()

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_rot": "xy"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "Avogadro" not in payload["message"]
    assert Artifact.objects.filter(
        workflow_run=run,
        path="reaction_hub_final_render.svg",
        kind="image",
        label="render:reaction_hub_final_render.svg",
    ).exists()
    assert Artifact.objects.filter(
        workflow_run=run,
        path="reaction_hub_final_rot_xy.gif",
        kind="image",
        label="render:reaction_hub_final_rot_xy.gif",
    ).exists()


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_keeps_svg_when_cairo_missing(
    client,
    monkeypatch,
    settings,
    tmp_path: Path,
):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_cairo_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_cairo_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-cairo-run"
    run_dir.mkdir(parents=True)
    (run_dir / "evaluated.traj").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-cairo-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="trajectory",
        label="trajectory:evaluated.traj",
        path="evaluated.traj",
    )
    settings.REACTION_HUB_XYZRENDER_COMMAND = "xyzrender"

    def _fake_run(command, cwd, check, stdout, stderr, text, env):  # noqa: ANN001
        del check, stdout, stderr, text, env
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

    monkeypatch.setattr("workflows.views.subprocess.run", _fake_run)

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_rot": "xy"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert "GIF generation skipped" in payload["message"]
    assert (
        "configure_venv_gtk.ps1" in payload["message"]
        or "libcairo2" in payload["message"]
    )
    assert Artifact.objects.filter(
        workflow_run=run,
        path="evaluated_render.svg",
        kind="image",
        label="render:evaluated_render.svg",
    ).exists()
    assert not Artifact.objects.filter(
        workflow_run=run,
        path="evaluated_rot_xy.gif",
        kind="image",
        label="render:evaluated_rot_xy.gif",
    ).exists()


@pytest.mark.django_db
def test_workflow_render_artifact_xyz_rejects_invalid_rotation(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="workflow_render_invalid_rot_profile")
    wf = WorkflowDefinition.objects.create(name="workflow_render_invalid_rot_wf", code_path="x.py")
    run_dir = tmp_path / "workflow-render-invalid-rot-run"
    run_dir.mkdir(parents=True)
    (run_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="workflow-render-invalid-rot-run",
        working_directory=str(run_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind="structure",
        label="structure:final.xyz",
        path="final.xyz",
    )

    resp = client.post(
        f"/workflows/{run.pk}/artifacts/{artifact.pk}/render-xyz/",
        data=json.dumps({"gif_rot": "bad-rotation"}),
        content_type="application/json",
    )
    payload = resp.json()

    assert resp.status_code == 400
    assert payload["ok"] is False
    assert "Unsupported rotation" in payload["error"]


@pytest.mark.django_db
def test_job_submit_batch_from_traj_creates_numbered_run_names(client, monkeypatch, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(
        name="batch_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="batch_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="batch_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user_batch",
        is_active=True,
    )

    submitted_names: list[str] = []

    def _fake_submit(**kwargs):
        run_name = str(kwargs["run_name"])
        submitted_names.append(run_name)
        run = WorkflowRun.objects.create(
            workflow=kwargs["workflow"],
            run_name=run_name,
            status="initiated",
            working_directory=f"/work/runs/{run_name}",
            settings_profile=kwargs["settings_profile"],
            metadata={"submission": {}, "structure": kwargs.get("structure_metadata") or {}},
        )
        job = JobRecord.objects.create(
            workflow_run=run,
            hpc_system=kwargs["hpc_system"],
            scheduler_job_id=str(70000 + len(submitted_names)),
            job_name=run_name,
            status=JobStatus.INITIATED,
            remote_workdir=f"/scratch/{run_name}",
            stdout_path=f"/work/runs/{run_name}/workflow.log",
        )
        return SimpleNamespace(
            workflow_run_id=run.pk,
            job_record_id=job.pk,
            scheduler_job_id=job.scheduler_job_id,
            remote_run_dir=f"/work/runs/{run_name}",
        )

    monkeypatch.setattr("jobs.views.submit_slurm_workflow_run", _fake_submit)

    traj_path = tmp_path / "batch_co.traj"
    frames = [
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]),
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]]),
    ]
    write(traj_path, frames)
    upload = SimpleUploadedFile("batch_co.traj", traj_path.read_bytes(), content_type="application/octet-stream")

    resp = client.post(
        "/jobs/submit/",
        data={
            "run_name": "batch_co",
            "run_mode": "single_optimisation",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "structure_file": upload,
            "traj_frame": "-1",
            "submit_all_traj_frames": "on",
            "fmax": "0.01",
            "nodes": "2",
            "tasks_per_node": "32",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "00:20:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "short",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "venv_activate_path": "",
            "pythonpath_entries": "",
        },
    )

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    assert submitted_names == ["batch_co_0", "batch_co_1"]


@pytest.mark.django_db
def test_job_submit_tddft_wavelength_batch_submits_one_job_per_target(client, monkeypatch):
    profile = CalculationSettingsProfile.objects.create(
        name="tddft_batch_submit_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="tddft_batch_submit_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="tddft_batch_submit_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user_tddft_batch",
        is_active=True,
    )

    submitted: list[tuple[str, float | None]] = []

    def _fake_submit(**kwargs):
        run_name = str(kwargs["run_name"])
        target_wavelength_nm = kwargs.get("target_wavelength_nm")
        submitted.append((run_name, None if target_wavelength_nm is None else float(target_wavelength_nm)))
        run = WorkflowRun.objects.create(
            workflow=kwargs["workflow"],
            run_name=run_name,
            status="initiated",
            working_directory=f"/work/runs/{run_name}",
            settings_profile=kwargs["settings_profile"],
            metadata={"submission": {}, "structure": kwargs.get("structure_metadata") or {}},
        )
        job = JobRecord.objects.create(
            workflow_run=run,
            hpc_system=kwargs["hpc_system"],
            scheduler_job_id=str(72000 + len(submitted)),
            job_name=run_name,
            status=JobStatus.INITIATED,
            remote_workdir=f"/scratch/{run_name}",
            stdout_path=f"/work/runs/{run_name}/workflow.log",
        )
        return SimpleNamespace(
            workflow_run_id=run.pk,
            job_record_id=job.pk,
            scheduler_job_id=job.scheduler_job_id,
            remote_run_dir=f"/work/runs/{run_name}",
        )

    monkeypatch.setattr("jobs.views.submit_slurm_workflow_run", _fake_submit)

    upload = SimpleUploadedFile(
        "scan_input.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.2\n",
        content_type="chemical/x-xyz",
    )
    resp = client.post(
        "/jobs/submit/",
        data={
            "run_name": "co_scan_blscan",
            "run_mode": "tddft_bond_scan",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "structure_file": upload,
            "traj_frame": "-1",
            "n_iter": "4",
            "use_tddft_optimisation": "on",
            "fmax": "0.01",
            "active_learning_optimizer": "bfgs",
            "target_wavelength_nm": "520.0",
            "target_wavelengths_nm": "500, 600, 700",
            "tddft_scan_nroots": "50",
            "bond_i": "0",
            "bond_j": "1",
            "increment": "0.05",
            "fix": "0.5",
            "fixed_symbols": "Cu",
            "distance_change_fraction": "1.0",
            "refine_points_each_side": "0",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "04:00:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "standard",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "venv_activate_path": "",
            "pythonpath_entries": "",
        },
    )

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    assert submitted == [
        ("co_scan_blscan_500nm", 500.0),
        ("co_scan_blscan_600nm", 600.0),
        ("co_scan_blscan_700nm", 700.0),
    ]

    run = WorkflowRun.objects.get(run_name="co_scan_blscan_500nm")
    submission_meta = dict(run.metadata.get("submission") or {})
    assert submission_meta.get("wavelength_batch_parent_run_name") == "co_scan_blscan"
    assert submission_meta.get("wavelength_batch_size") == 3
    assert submission_meta.get("wavelength_batch_mode") == "target_wavelengths"
    assert submission_meta.get("target_wavelengths_nm") == [500.0, 600.0, 700.0]


@pytest.mark.django_db
def test_job_submit_batch_from_db_creates_numbered_run_names(client, monkeypatch, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(
        name="batch_db_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="batch_db_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="batch_db_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user_batch_db",
        is_active=True,
    )

    submitted_names: list[str] = []

    def _fake_submit(**kwargs):
        run_name = str(kwargs["run_name"])
        submitted_names.append(run_name)
        run = WorkflowRun.objects.create(
            workflow=kwargs["workflow"],
            run_name=run_name,
            status="initiated",
            working_directory=f"/work/runs/{run_name}",
            settings_profile=kwargs["settings_profile"],
            metadata={"submission": {}, "structure": kwargs.get("structure_metadata") or {}},
        )
        job = JobRecord.objects.create(
            workflow_run=run,
            hpc_system=kwargs["hpc_system"],
            scheduler_job_id=str(71000 + len(submitted_names)),
            job_name=run_name,
            status=JobStatus.INITIATED,
            remote_workdir=f"/scratch/{run_name}",
            stdout_path=f"/work/runs/{run_name}/workflow.log",
        )
        return SimpleNamespace(
            workflow_run_id=run.pk,
            job_record_id=job.pk,
            scheduler_job_id=job.scheduler_job_id,
            remote_run_dir=f"/work/runs/{run_name}",
        )

    monkeypatch.setattr("jobs.views.submit_slurm_workflow_run", _fake_submit)

    db_path = tmp_path / "batch_co.db"
    frames = [
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]),
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]]),
    ]
    with ase_connect(db_path) as db:
        for atoms in frames:
            db.write(atoms)
    upload = SimpleUploadedFile("batch_co.db", db_path.read_bytes(), content_type="application/octet-stream")

    resp = client.post(
        "/jobs/submit/",
        data={
            "run_name": "batch_co_db",
            "run_mode": "single_optimisation",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "upload_file",
            "structure_file": upload,
            "traj_frame": "-1",
            "submit_all_traj_frames": "on",
            "fmax": "0.01",
            "nodes": "2",
            "tasks_per_node": "32",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "00:20:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "short",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "venv_activate_path": "",
            "pythonpath_entries": "",
        },
    )

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    assert submitted_names == ["batch_co_db_0", "batch_co_db_1"]


@pytest.mark.django_db
def test_job_submit_batch_from_existing_db_propagates_coadsorbate_reference_hints(
    client, monkeypatch, tmp_path: Path
):
    profile = CalculationSettingsProfile.objects.create(
        name="batch_existing_db_hints_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="batch_existing_db_hints_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="batch_existing_db_hints_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user_batch_existing_db_hints",
        is_active=True,
    )
    cluster = ReferenceNanocluster.objects.create(
        name="batch_existing_db_hints_cluster",
        composition="Au13",
        settings_profile=profile,
    )

    source_dir = tmp_path / "batch-existing-db-hints-source"
    source_db = source_dir / "scratch" / "coadsorption_final_for_dft.db"
    source_db.parent.mkdir(parents=True, exist_ok=True)
    with ase_connect(source_db) as db:
        db.write(Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]))
        db.write(Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]]))

    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="coadsorbate_source",
        status="completed",
        working_directory=str(source_dir),
        settings_profile=profile,
        metadata={
            "submission": {
                "coadsorbate_config": {
                    "reference_cluster_id": cluster.pk,
                    "adsorbate_metadata": [
                        {"formula": "CO", "source": "ase_molecule", "ase_molecule_formula": "CO"},
                        {"formula": "NO2", "source": "pubchem", "pubchem_query": "nitrogen dioxide"},
                    ],
                }
            }
        },
    )
    source_artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind=ArtifactKind.DATABASE,
        label="coadsorption_db:coadsorption_final_for_dft.db",
        path="scratch/coadsorption_final_for_dft.db",
    )

    captured_structure_meta: list[dict[str, object]] = []

    def _fake_submit(**kwargs):
        structure_meta = kwargs.get("structure_metadata")
        captured_structure_meta.append(dict(structure_meta or {}))
        run_name = str(kwargs["run_name"])
        run = WorkflowRun.objects.create(
            workflow=kwargs["workflow"],
            run_name=run_name,
            status="initiated",
            working_directory=f"/work/runs/{run_name}",
            settings_profile=kwargs["settings_profile"],
            metadata={"submission": {}, "structure": structure_meta or {}},
        )
        job = JobRecord.objects.create(
            workflow_run=run,
            hpc_system=kwargs["hpc_system"],
            scheduler_job_id=str(73000 + len(captured_structure_meta)),
            job_name=run_name,
            status=JobStatus.INITIATED,
            remote_workdir=f"/scratch/{run_name}",
            stdout_path=f"/work/runs/{run_name}/workflow.log",
        )
        return SimpleNamespace(
            workflow_run_id=run.pk,
            job_record_id=job.pk,
            scheduler_job_id=job.scheduler_job_id,
            remote_run_dir=f"/work/runs/{run_name}",
        )

    monkeypatch.setattr("jobs.views.submit_slurm_workflow_run", _fake_submit)

    resp = client.post(
        "/jobs/submit/",
        data={
            "run_name": "coads_batch_followup",
            "run_mode": "single_optimisation",
            "workflow": str(workflow.pk),
            "hpc_system": str(system.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "existing_run",
            "source_run_id": str(source_run.pk),
            "source_artifact_id": str(source_artifact.pk),
            "submit_all_traj_frames": "on",
            "fmax": "0.01",
            "nodes": "1",
            "tasks_per_node": "1",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "00:20:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "short",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "venv_activate_path": "",
            "pythonpath_entries": "",
        },
    )

    assert resp.status_code == 302
    assert resp["Location"] == "/jobs/"
    assert len(captured_structure_meta) == 2
    for structure_meta in captured_structure_meta:
        hints = structure_meta.get("adsorption_reference_hints")
        assert isinstance(hints, dict)
        assert int(hints.get("reference_cluster_id")) == cluster.pk
        assert hints.get("adsorbate_formula_counts") == {"CO": 1, "NO2": 1}
        assert hints.get("adsorbate_formulas") == ["CO", "NO2"]
        assert int(hints.get("source_run_id")) == source_run.pk
        assert structure_meta.get("coadsorbate_provenance") == hints


@pytest.mark.django_db
def test_job_submit_prefills_from_existing_run_query(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(
        name="prefill_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="prefill_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    source_dir = tmp_path / "prefill-source-run"
    source_dir.mkdir(parents=True)
    (source_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_opt",
        working_directory=str(source_dir),
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {
                "nodes": 2,
                "tasks_per_node": 32,
                "cpus_per_task": 1,
                "walltime": "96:00:00",
                "qos": "long",
                "account": "SLURM_ACCOUNT",
                "partition": "standard",
                "module_loads": ["epcc-setup-env", "orca/6.0.0"],
                "pythonpath_entries": ["/work/PROJECT/GROUP/user/ORCA/cplas"],
                "venv_activate_path": "/work/PROJECT/GROUP/user/venv/bin/activate",
                "active_learning_optimizer": "fire2",
                "target_wavelength_nm": 515.0,
                "slurm_preset": "dev_short",
                "scan_config": {
                    "bond_i": 0,
                    "bond_j": 1,
                    "increment": 0.05,
                    "fix": 0.5,
                    "fixed_symbols": ["Cu"],
                    "target_distance": 2.2,
                    "distance_change_fraction": 1.0,
                    "enable_ts_smoothing": False,
                    "refine_points_each_side": 0,
                    "restart_extend_steps": 5,
                    "restart_extend_increment_A": 0.2,
                    "tddft_scan_nroots": 50,
                },
            }
        },
    )
    artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )
    hpc = HPCSystem.objects.create(
        name="prefill_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="prefill_user",
        is_active=True,
    )
    JobRecord.objects.create(
        workflow_run=source_run,
        hpc_system=hpc,
        scheduler_job_id="81234",
        job_name="co_opt",
        status=JobStatus.COMPLETED,
    )

    resp = client.get(f"/jobs/submit/?from_run={source_run.pk}&mode=tddft_bond_scan")

    assert resp.status_code == 200
    form = resp.context["form"]
    assert form.initial["geometry_source"] == "existing_run"
    assert int(form.initial["source_run_id"]) == source_run.pk
    assert int(form.initial["source_artifact_id"]) == artifact.pk
    assert form.initial["run_mode"] == "tddft_bond_scan"
    assert form.initial["workflow"] == workflow.pk
    assert form.initial["settings_profile"] == profile.pk
    assert form.initial["hpc_system"] == hpc.pk
    assert form.initial["run_name"] == "co_opt_blscan"
    assert form.initial["walltime"] == "96:00:00"
    assert form.initial["qos"] == "long"
    assert form.initial["active_learning_optimizer"] == "fire2"
    assert form.initial["target_wavelength_nm"] == 515.0
    assert form.initial["slurm_preset"] == "dev_short"
    assert form.initial["restart_extend_steps"] == 0
    assert form.initial["restart_extend_increment_A"] == ""
    assert "epcc-setup-env" in form.initial["module_loads"]


@pytest.mark.django_db
def test_job_submit_prefills_selected_existing_database_artifact_for_batch_query(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(
        name="prefill_existing_db_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="prefill_existing_db_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    source_dir = tmp_path / "prefill-existing-db-source"
    source_dir.mkdir(parents=True)
    (source_dir / "reaction_hub_final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.1\n", encoding="utf-8")

    db_path = source_dir / "scratch" / "coadsorption_final_for_dft.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with ase_connect(db_path) as db:
        db.write(Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]]))
        db.write(Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.25]]))

    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_opt",
        working_directory=str(source_dir),
        status="completed",
        settings_profile=profile,
    )
    Artifact.objects.create(
        workflow_run=source_run,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )
    db_artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind=ArtifactKind.DATABASE,
        label="coadsorption_db:coadsorption_final_for_dft.db",
        path="scratch/coadsorption_final_for_dft.db",
    )

    resp = client.get(
        f"/jobs/submit/?from_run={source_run.pk}&source_artifact={db_artifact.pk}&batch=1"
    )

    assert resp.status_code == 200
    form = resp.context["form"]
    assert form.initial["geometry_source"] == "existing_run"
    assert int(form.initial["source_run_id"]) == source_run.pk
    assert int(form.initial["source_artifact_id"]) == db_artifact.pk
    assert form.initial["run_mode"] == "single_optimisation"
    assert form.initial["submit_all_traj_frames"] is True
    assert form.initial["run_name"] == "co_opt_batch"


@pytest.mark.django_db
@override_settings(REACTION_HUB_SUBMIT_DEFAULT_MODULES=["epcc-setup-env", "cray-python", "orca/6.0.0"])
def test_job_submit_batch_redirect_uses_default_module_loads_not_gpu_prefill(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(
        name="prefill_existing_db_coadsorbate_profile",
        xc_functional="PBE0",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="prefill_existing_db_coadsorbate_workflow",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    source_dir = tmp_path / "prefill-existing-db-coadsorbate-source"
    source_dir.mkdir(parents=True)

    db_path = source_dir / "scratch" / "coadsorption_final_for_dft.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with ase_connect(db_path) as db:
        db.write(Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]]))
        db.write(Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.25]]))

    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="coadsorbate_source",
        working_directory=str(source_dir),
        status="completed",
        settings_profile=profile,
        metadata={
            "submission": {
                "run_mode": "coadsorbate_placement",
                "module_loads": [
                    "PrgEnv-cray",
                    "rocm",
                    "craype-accel-amd-gfx90a",
                    "craype-x86-milan",
                    "cray-python",
                ],
            }
        },
    )
    db_artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind=ArtifactKind.DATABASE,
        label="coadsorption_db:coadsorption_final_for_dft.db",
        path="scratch/coadsorption_final_for_dft.db",
    )

    resp = client.get(
        f"/jobs/submit/?from_run={source_run.pk}&source_artifact={db_artifact.pk}&batch=1"
    )

    assert resp.status_code == 200
    form = resp.context["form"]
    assert form.initial["run_mode"] == "single_optimisation"
    assert form.initial["submit_all_traj_frames"] is True
    module_loads = str(form.initial.get("module_loads") or "")
    assert "orca/6.0.0" in module_loads
    assert "rocm" not in module_loads
    assert "craype-accel-amd-gfx90a" not in module_loads


@pytest.mark.django_db
def test_job_submit_existing_run_source_posts_with_prepared_xyz(client, monkeypatch, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(
        name="existing_source_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="existing_source_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    hpc = HPCSystem.objects.create(
        name="existing_source_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="existing_source_user",
        is_active=True,
    )

    source_dir = tmp_path / "source-existing-run"
    source_dir.mkdir(parents=True)
    source_file = source_dir / "reaction_hub_final.xyz"
    source_file.write_text("2\nCO\nC 0.0 0.0 0.0\nO 0.0 0.0 1.15\n", encoding="utf-8")
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_opt_for_reuse",
        working_directory=str(source_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind="structure",
        label="structure:reaction_hub_final.xyz",
        path="reaction_hub_final.xyz",
        metadata={"is_final_structure": True},
    )

    captured: dict[str, object] = {}

    def _fake_submit(**kwargs):
        captured.update(kwargs)
        run_name = str(kwargs["run_name"])
        run = WorkflowRun.objects.create(
            workflow=kwargs["workflow"],
            run_name=run_name,
            status="initiated",
            working_directory=f"/work/runs/{run_name}",
            settings_profile=kwargs["settings_profile"],
            metadata={"submission": {}, "structure": kwargs.get("structure_metadata") or {}},
        )
        job = JobRecord.objects.create(
            workflow_run=run,
            hpc_system=kwargs["hpc_system"],
            scheduler_job_id="92001",
            job_name=run_name,
            status=JobStatus.INITIATED,
            remote_workdir=f"/scratch/{run_name}",
            stdout_path=f"/work/runs/{run_name}/workflow.log",
        )
        return SimpleNamespace(
            workflow_run_id=run.pk,
            job_record_id=job.pk,
            scheduler_job_id=job.scheduler_job_id,
            remote_run_dir=f"/work/runs/{run_name}",
        )

    monkeypatch.setattr("jobs.views.submit_slurm_workflow_run", _fake_submit)

    resp = client.post(
        "/jobs/submit/",
        data={
            "run_name": "co_reuse_followup",
            "run_mode": "single_optimisation",
            "workflow": str(workflow.pk),
            "hpc_system": str(hpc.pk),
            "settings_profile": str(profile.pk),
            "reference_intent": "none",
            "geometry_source": "existing_run",
            "source_run_id": str(source_run.pk),
            "source_artifact_id": str(artifact.pk),
            "traj_frame": "-1",
            "fmax": "0.01",
            "active_learning_optimizer": "fire2",
            "fixed_symbols": "Cu,Ti",
            "bond_i": "0",
            "bond_j": "1",
            "target_distance": "1.25",
            "fix": "0.5",
            "nodes": "2",
            "tasks_per_node": "32",
            "cpus_per_task": "1",
            "slurm_preset": "custom",
            "walltime": "00:20:00",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "short",
            "remote_runs_root": "/work/PROJECT/GROUP/{username}/reaction_hub_runs",
            "scratch_base": "/scratch/PROJECT/GROUP",
            "module_loads": "",
            "venv_activate_path": "",
            "pythonpath_entries": "",
        },
    )

    assert resp.status_code == 302
    assert str(resp["Location"]).startswith("/jobs/")
    assert str(captured.get("structure_xyz", "")).startswith("2")
    structure_metadata = captured.get("structure_metadata", {})
    assert isinstance(structure_metadata, dict)
    assert structure_metadata.get("source") == "existing_run"
    assert int(structure_metadata.get("source_run_id")) == source_run.pk
    assert int(structure_metadata.get("source_artifact_id")) == artifact.pk
    assert structure_metadata.get("formula") == "CO"
    assert captured.get("active_learning_optimizer") == "fire2"
    single_constraints = captured.get("single_constraints_config", {})
    assert isinstance(single_constraints, dict)
    assert single_constraints.get("fixed_symbols") == ("Cu", "Ti")
    assert single_constraints.get("bond_pair_indices") == (0, 1)
    assert single_constraints.get("target_distance") == pytest.approx(1.25)
    assert single_constraints.get("fix") == pytest.approx(0.5)


@pytest.mark.django_db
def test_structure_preview_view_supports_existing_run_source(client, tmp_path: Path):
    profile = CalculationSettingsProfile.objects.create(name="preview_existing_profile")
    workflow = WorkflowDefinition.objects.create(name="preview_existing_wf", code_path="x.py")
    source_dir = tmp_path / "preview-existing-run"
    source_dir.mkdir(parents=True)
    (source_dir / "final.xyz").write_text("2\nCO\nC 0 0 0\nO 0 0 1.2\n", encoding="utf-8")
    source_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="preview_source_run",
        working_directory=str(source_dir),
        status="completed",
        settings_profile=profile,
    )
    artifact = Artifact.objects.create(
        workflow_run=source_run,
        kind="structure",
        label="final",
        path="final.xyz",
    )

    resp = client.post(
        "/jobs/submit/structure-preview/",
        data={
            "geometry_source": "existing_run",
            "source_run_id": str(source_run.pk),
            "source_artifact_id": str(artifact.pk),
        },
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert payload["metadata"]["source"] == "existing_run"
    assert int(payload["metadata"]["source_run_id"]) == source_run.pk
    assert int(payload["metadata"]["source_artifact_id"]) == artifact.pk
    assert payload["element_counts"] == {"C": 1, "O": 1}
    assert payload["element_max_z"] == {"C": 0.0, "O": 1.2}
    assert payload["global_max_z"] == 1.2


@pytest.mark.django_db
def test_structure_preview_view_generates_neb_set_bond_distance_image(client):
    upload = SimpleUploadedFile(
        "h2.xyz",
        b"2\nH2\nH 0 0 0\nH 0 0 0.74\n",
        content_type="chemical/x-xyz",
    )

    resp = client.post(
        "/jobs/submit/structure-preview/",
        data={
            "run_mode": "active_learning_neb",
            "geometry_source": "upload_file",
            "structure_file": upload,
            "neb_final_image_strategy": "set_bond_distance_from_initial",
            "neb_break_bond_i": "0",
            "neb_break_bond_j": "1",
            "neb_break_bond_extension_A": "1.50",
            "neb_break_bond_fix": "0.5",
        },
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    metadata = payload["metadata"]
    assert metadata["source"] == "generated_set_bond_distance_preview"
    assert metadata["set_bond_i"] == 0
    assert metadata["set_bond_j"] == 1
    assert metadata["set_bond_distance_A"] == pytest.approx(1.5)
    assert metadata["distance_after_set_A"] == pytest.approx(1.5, abs=1.0e-6)


@pytest.mark.django_db
def test_structure_open_ase_gui_view_uses_extxyz_temp_file_for_periodic_structure(
    client, monkeypatch, tmp_path: Path
):
    launched: dict[str, object] = {}

    class _DummyProcess:
        def poll(self):
            return None

    def _fake_popen(command, **kwargs):
        launched["command"] = command
        launched["kwargs"] = kwargs
        return _DummyProcess()

    monkeypatch.setattr("jobs.views.subprocess.Popen", _fake_popen)

    periodic = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 2.0]],
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 8.0]],
        pbc=[True, True, True],
    )
    cif_path = tmp_path / "periodic_surface.cif"
    write(cif_path, periodic)
    upload = SimpleUploadedFile(
        "periodic_surface.cif",
        cif_path.read_bytes(),
        content_type="chemical/x-cif",
    )

    resp = client.post(
        "/jobs/submit/structure-open-ase-gui/",
        data={
            "geometry_source": "upload_file",
            "structure_file": upload,
        },
    )
    payload = resp.json()

    assert resp.status_code == 200
    assert payload["ok"] is True
    assert isinstance(launched.get("command"), list)
    temp_path = str(launched["command"][-1])
    assert temp_path.endswith(".extxyz")
    temp_text = Path(temp_path).read_text(encoding="utf-8")
    assert "Lattice=" in temp_text


@pytest.mark.django_db
def test_admin_delete_job_deletes_associated_workflow_run():
    profile = CalculationSettingsProfile.objects.create(name="admin_delete_profile")
    workflow = WorkflowDefinition.objects.create(name="admin_delete_workflow", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="admin-delete-run",
        working_directory="/tmp/admin-delete-run",
        status="initiated",
        settings_profile=profile,
    )
    hpc = HPCSystem.objects.create(
        name="admin_delete_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=hpc,
        scheduler_job_id="70001",
        job_name="admin-delete-job",
        status=JobStatus.INITIATED,
    )

    admin_model = JobRecordAdmin(JobRecord, AdminSite())
    admin_model.delete_model(request=None, obj=job)

    assert not WorkflowRun.objects.filter(pk=run.pk).exists()
    assert not JobRecord.objects.filter(pk=job.pk).exists()


@pytest.mark.django_db
def test_admin_bulk_delete_job_queryset_deletes_associated_workflow_runs():
    profile = CalculationSettingsProfile.objects.create(name="admin_bulk_delete_profile")
    workflow = WorkflowDefinition.objects.create(name="admin_bulk_delete_workflow", code_path="x.py")
    hpc = HPCSystem.objects.create(
        name="admin_bulk_delete_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )

    run_one = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="admin-bulk-run-1",
        working_directory="/tmp/admin-bulk-run-1",
        status="initiated",
        settings_profile=profile,
    )
    run_two = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="admin-bulk-run-2",
        working_directory="/tmp/admin-bulk-run-2",
        status="initiated",
        settings_profile=profile,
    )

    selected_job = JobRecord.objects.create(
        workflow_run=run_one,
        hpc_system=hpc,
        scheduler_job_id="71001",
        job_name="admin-bulk-job-selected",
        status=JobStatus.INITIATED,
    )
    sibling_job = JobRecord.objects.create(
        workflow_run=run_one,
        hpc_system=hpc,
        scheduler_job_id="71002",
        job_name="admin-bulk-job-sibling",
        status=JobStatus.INITIATED,
    )
    other_job = JobRecord.objects.create(
        workflow_run=run_two,
        hpc_system=hpc,
        scheduler_job_id="71003",
        job_name="admin-bulk-job-other",
        status=JobStatus.INITIATED,
    )

    admin_model = JobRecordAdmin(JobRecord, AdminSite())
    admin_model.delete_queryset(request=None, queryset=JobRecord.objects.filter(pk=selected_job.pk))

    assert not WorkflowRun.objects.filter(pk=run_one.pk).exists()
    assert not JobRecord.objects.filter(pk=selected_job.pk).exists()
    assert not JobRecord.objects.filter(pk=sibling_job.pk).exists()
    assert WorkflowRun.objects.filter(pk=run_two.pk).exists()
    assert JobRecord.objects.filter(pk=other_job.pk).exists()


@pytest.mark.django_db
def test_admin_jobrecord_actions_hide_default_delete_selected():
    admin_model = JobRecordAdmin(JobRecord, AdminSite())
    request = RequestFactory().get("/admin/jobs/jobrecord/")
    User = get_user_model()
    request.user = User.objects.create_superuser(
        username="admin_actions_user",
        email="admin-actions@example.com",
        password="not-used",
    )

    actions = admin_model.get_actions(request)

    assert "delete_selected" not in actions
    assert "create_reference_adsorbate" in actions
    assert "mark_selected_for_deletion" in actions
    assert "unmark_selected_for_deletion" in actions
    assert "delete_marked_for_deletion" in actions
    assert admin_model.has_delete_permission(request) is False


@pytest.mark.django_db
def test_admin_create_reference_adsorbate_action_tags_selected_job():
    profile = CalculationSettingsProfile.objects.create(name="admin_ref_ads_profile")
    workflow = WorkflowDefinition.objects.create(name="admin_ref_ads_workflow", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="CO_admin_reference",
        working_directory="/tmp/co-admin-reference",
        status="completed",
        settings_profile=profile,
        metadata={"structure": {"formula": "CO"}},
    )
    hpc = HPCSystem.objects.create(
        name="admin_ref_ads_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=hpc,
        scheduler_job_id="71999",
        job_name="co-admin-reference-job",
        status=JobStatus.COMPLETED,
    )

    admin_model = JobRecordAdmin(JobRecord, AdminSite())
    admin_model.message_user = lambda *args, **kwargs: None
    admin_model.create_reference_adsorbate(
        request=None,
        queryset=JobRecord.objects.filter(pk=job.pk),
    )
    run.refresh_from_db()

    adsorbate = ReferenceAdsorbate.objects.get(name="CO_admin_reference")
    assert adsorbate.formula == "CO"
    assert adsorbate.settings_profile == profile
    assert adsorbate.structure_path == "/tmp/co-admin-reference"
    assert run.reference_adsorbate == adsorbate
    assert run.reference_nanocluster is None


@pytest.mark.django_db
def test_admin_mark_selected_for_deletion_action_marks_jobs():
    profile = CalculationSettingsProfile.objects.create(name="admin_mark_delete_profile")
    workflow = WorkflowDefinition.objects.create(name="admin_mark_delete_workflow", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="admin-mark-delete-run",
        working_directory="/tmp/admin-mark-delete-run",
        status="initiated",
        settings_profile=profile,
    )
    hpc = HPCSystem.objects.create(
        name="admin_mark_delete_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=hpc,
        scheduler_job_id="72001",
        job_name="admin-mark-delete-job",
        status=JobStatus.FAILED,
    )

    admin_model = JobRecordAdmin(JobRecord, AdminSite())
    admin_model.message_user = lambda *args, **kwargs: None
    admin_model.mark_selected_for_deletion(
        request=None,
        queryset=JobRecord.objects.filter(pk=job.pk),
    )
    job.refresh_from_db()

    assert job.status == JobStatus.DELETED
    event = JobStatusEvent.objects.get(job=job)
    assert event.previous_status == JobStatus.FAILED
    assert event.payload["action"] == "admin_mark_for_deletion"


@pytest.mark.django_db
def test_admin_unmark_selected_for_deletion_action_restores_previous_status():
    profile = CalculationSettingsProfile.objects.create(name="admin_unmark_delete_profile")
    workflow = WorkflowDefinition.objects.create(name="admin_unmark_delete_workflow", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="admin-unmark-delete-run",
        working_directory="/tmp/admin-unmark-delete-run",
        status="initiated",
        settings_profile=profile,
    )
    hpc = HPCSystem.objects.create(
        name="admin_unmark_delete_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=hpc,
        scheduler_job_id="72002",
        job_name="admin-unmark-delete-job",
        status=JobStatus.FAILED,
    )

    admin_model = JobRecordAdmin(JobRecord, AdminSite())
    admin_model.message_user = lambda *args, **kwargs: None
    admin_model.mark_selected_for_deletion(request=None, queryset=JobRecord.objects.filter(pk=job.pk))
    admin_model.unmark_selected_for_deletion(request=None, queryset=JobRecord.objects.filter(pk=job.pk))
    job.refresh_from_db()

    assert job.status == JobStatus.FAILED
    event = JobStatusEvent.objects.filter(job=job).first()
    assert event.previous_status == JobStatus.DELETED
    assert event.payload["action"] == "admin_unmark_for_deletion"


@pytest.mark.django_db
def test_admin_permanent_delete_skips_jobs_not_marked_for_deletion():
    profile = CalculationSettingsProfile.objects.create(name="admin_safe_delete_profile")
    workflow = WorkflowDefinition.objects.create(name="admin_safe_delete_workflow", code_path="x.py")
    hpc = HPCSystem.objects.create(
        name="admin_safe_delete_hpc",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    marked_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="admin-safe-delete-marked",
        working_directory="/tmp/admin-safe-delete-marked",
        status="initiated",
        settings_profile=profile,
    )
    unmarked_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="admin-safe-delete-unmarked",
        working_directory="/tmp/admin-safe-delete-unmarked",
        status="initiated",
        settings_profile=profile,
    )
    marked_job = JobRecord.objects.create(
        workflow_run=marked_run,
        hpc_system=hpc,
        scheduler_job_id="73001",
        job_name="admin-safe-delete-marked-job",
        status=JobStatus.DELETED,
    )
    unmarked_job = JobRecord.objects.create(
        workflow_run=unmarked_run,
        hpc_system=hpc,
        scheduler_job_id="73002",
        job_name="admin-safe-delete-unmarked-job",
        status=JobStatus.FAILED,
    )

    messages_seen = []
    admin_model = JobRecordAdmin(JobRecord, AdminSite())
    admin_model.message_user = lambda request, message, **kwargs: messages_seen.append((message, kwargs))
    admin_model.delete_marked_for_deletion(
        request=None,
        queryset=JobRecord.objects.filter(pk__in=[marked_job.pk, unmarked_job.pk]),
    )

    assert not WorkflowRun.objects.filter(pk=marked_run.pk).exists()
    assert WorkflowRun.objects.filter(pk=unmarked_run.pk).exists()
    assert JobRecord.objects.filter(pk=unmarked_job.pk).exists()
    assert any("Skipped 1 selected jobs" in message for message, _ in messages_seen)
    assert any("PERMANENTLY deleted 1 workflow runs" in message for message, _ in messages_seen)
