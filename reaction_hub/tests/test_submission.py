from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from ase import Atoms
from ase.db import connect as ase_connect
from ase.io import write
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile


from jobs.geometry import structure_file_to_xyz, structure_file_to_xyz_frames
from jobs.models import HPCSystem, JobRecord, JobStatus
from jobs.tasks import (
    _queue_retry_countdown,
    dispatch_next_queued_gpu_workflow_task,
    restart_cancelled_job_task,
    resubmit_failed_job_task,
    restore_ct_serial_job_task,
    submit_queued_gpu_workflow_task,
)
from jobs.submission import (
    _build_wrapper_script,
    _default_nwchem_cam,
    _link_backfilled_dft_run,
    _normalize_xc_for_backend,
    _reservation_has_capacity,
    _scaled_final_tddft_nroots,
    _stage_remote_run_files_batched,
    _submission_queue_limit,
    _supports_batched_remote_stage,
    resolve_active_reservation,
    submit_slurm_workflow_run,
)


@pytest.mark.django_db
def test_backfilled_dft_run_moves_to_refinement_stage_and_keeps_reaction_step():
    from references.models import CalculationSettingsProfile
    from workflows.models import Campaign, CampaignRun, WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(name="backfill-profile")
    workflow = WorkflowDefinition.objects.create(name="backfill-workflow")
    source = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="GPU_survey_CO_NO",
    )
    refinement = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="DFT_res_Cu20CNO2",
    )
    campaign = Campaign.objects.create(name="Urea campaign", slug="urea-campaign")
    CampaignRun.objects.create(
        campaign=campaign,
        workflow_run=source,
        checkpoint_stage=3,
        reaction_step_key="Step_C",
    )

    _link_backfilled_dft_run(
        campaign_name=campaign.name,
        original_run_id=source.pk,
        new_run=refinement,
    )

    membership = CampaignRun.objects.get(campaign=campaign, workflow_run=refinement)
    assert membership.checkpoint_stage == 4
    assert membership.reaction_step_key == "Step_C"


def test_gpu_submission_queue_is_limited_to_two(settings):
    settings.REACTION_HUB_GPU_QUEUE_MAX_JOBS = 2
    assert _submission_queue_limit("gpu-exc") == 2


def test_taskfarm_submission_queue_is_limited_to_128(settings):
    settings.REACTION_HUB_TASKFARM_QUEUE_MAX_JOBS = 128
    assert _submission_queue_limit("taskfarm") == 128


def test_active_reservation_only_routes_reservation_qos(settings, monkeypatch):
    settings.REACTION_HUB_ACTIVE_RESERVATIONS = [
        {
            "name": "SLURM_ACCOUNT_RES_3",
            "hpc_system": "archer2",
            "account": "SLURM_ACCOUNT",
            "partition": "standard",
            "qos": "reservation",
            "start_utc": "2026-08-27T13:00:00Z",
            "end_utc": "2026-08-31T23:00:00Z",
            "nodes_quota": 90,
        }
    ]
    now = [datetime(2026, 8, 27, 12, 59, 59, tzinfo=timezone.utc)]
    monkeypatch.setattr("jobs.submission.timezone.now", lambda: now[0])

    match = {
        "hpc_key": "archer2",
        "partition": "standard",
        "account": "SLURM_ACCOUNT",
    }
    assert resolve_active_reservation(**match, qos="reservation") == ""
    now[0] = datetime(2026, 8, 27, 13, 0, tzinfo=timezone.utc)
    assert resolve_active_reservation(**match, qos="reservation") == "SLURM_ACCOUNT_RES_3"
    assert resolve_active_reservation(**match, qos="taskfarm") == ""
    assert resolve_active_reservation(**match, qos="standard") == ""
    assert resolve_active_reservation(**match, qos="long") == ""
    now[0] = datetime(2026, 8, 31, 23, 0, tzinfo=timezone.utc)
    assert resolve_active_reservation(**match, qos="reservation") == ""


def test_reservation_capacity_counts_requested_nodes():
    class Client:
        def exec_checked(self, command):
            assert "--reservation SLURM_ACCOUNT_RES_3" in command
            assert "--user user" in command
            return "32\n32\n24\n"

    client = Client()
    assert _reservation_has_capacity(
        client,
        reservation="SLURM_ACCOUNT_RES_3",
        username="user",
        requested_nodes=2,
        node_quota=90,
    )
    assert not _reservation_has_capacity(
        client,
        reservation="SLURM_ACCOUNT_RES_3",
        username="user",
        requested_nodes=3,
        node_quota=90,
    )


def test_queue_waiting_submission_tasks_are_redelivered_after_worker_loss():
    for task in (
        restart_cancelled_job_task,
        resubmit_failed_job_task,
        restore_ct_serial_job_task,
        submit_queued_gpu_workflow_task,
    ):
        assert task.acks_late is True
        assert task.reject_on_worker_lost is True
    assert restore_ct_serial_job_task.max_retries is None
    assert submit_queued_gpu_workflow_task.max_retries is None


@pytest.mark.django_db
def test_failed_resubmission_retries_without_entering_handler_when_queue_is_full(monkeypatch):
    from celery.exceptions import Retry
    from jobs.tasks import _task_request
    from references.models import CalculationSettingsProfile
    from workflows.models import WorkflowDefinition, WorkflowRun
    from jobs.views import _job_resubmit_failed_sync_response, _QueueCapacityUnavailable

    profile = CalculationSettingsProfile.objects.create(name="resubmit-capacity-profile")
    workflow = WorkflowDefinition.objects.create(name="resubmit-capacity-workflow")
    run = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="retry")
    hpc = HPCSystem.objects.create(name="ARCHER2", host="example.org", username="user")
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=hpc,
        scheduler_job_id="123",
        job_name="retry",
        status=JobStatus.FAILED,
        scheduler_payload={"qos": "taskfarm"},
    )
    monkeypatch.setattr("jobs.tasks._queue_has_capacity", lambda _qos: False)
    monkeypatch.setattr(
        "jobs.views._job_resubmit_failed_sync_response",
        lambda *_args, **_kwargs: pytest.fail("handler must not block a full queue"),
    )

    with pytest.raises(Retry):
        resubmit_failed_job_task.run(job.pk)

    monkeypatch.setattr("jobs.tasks._queue_has_capacity", lambda _qos: True)
    monkeypatch.setattr(
        "jobs.views._job_resubmit_failed_sync_response",
        _job_resubmit_failed_sync_response,
    )

    def queue_filled(*_args, **_kwargs):
        raise _QueueCapacityUnavailable("taskfarm")

    monkeypatch.setattr("jobs.views._restart_cancelled_job_in_place", queue_filled)
    with pytest.raises(Retry):
        resubmit_failed_job_task.run(job.pk)

    def ssh_failed(*_args, **_kwargs):
        raise RuntimeError("SSH authentication failed")

    monkeypatch.setattr("jobs.views._restart_cancelled_job_in_place", ssh_failed)
    with pytest.raises(RuntimeError, match="SSH authentication failed"):
        _job_resubmit_failed_sync_response(
            _task_request("/jobs/1/resubmit-failed/"),
            job.pk,
            raise_on_failure=True,
        )


@pytest.mark.django_db
def test_queued_gpu_task_submits_run_and_moves_campaign_membership(monkeypatch):
    from references.models import CalculationSettingsProfile
    from workflows.models import Campaign, CampaignRun, WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(
        name="queued-gpu-profile",
        xc_functional="PBE0",
        basis_set="def2-SVP",
    )
    workflow = WorkflowDefinition.objects.create(name="queued-gpu-workflow")
    HPCSystem.objects.create(
        name="ARCHER2",
        scheduler_type="slurm",
        host="example.org",
        username="user",
    )
    scheduled = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="scheduled-gpu-run",
        metadata={
            "queue_status": "enqueued_celery_gpu",
            "structure_xyz": "1\nCu\nCu 0 0 0\n",
            "coadsorbate_config": {"mace_fmax": 0.05},
            "submission": {"nodes": 2, "tasks_per_node": 4, "cpus_per_task": 8},
        },
    )
    campaign = Campaign.objects.create(name="GPU campaign", slug="gpu-campaign")
    CampaignRun.objects.create(campaign=campaign, workflow_run=scheduled, checkpoint_stage=3)

    def fake_submit(**kwargs):
        submitted = WorkflowRun.objects.create(
            workflow=kwargs["workflow"],
            settings_profile=kwargs["settings_profile"],
            run_name=kwargs["run_name"],
        )
        return SimpleNamespace(workflow_run_id=submitted.pk, scheduler_job_id="gpu-123")

    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run", fake_submit)
    result = submit_queued_gpu_workflow_task.apply(
        args=[scheduled.pk], task_id="queued-gpu-task"
    ).get()

    scheduled.refresh_from_db()
    assert scheduled.metadata["queue_status"] == "submitted_slurm"
    assert scheduled.metadata["scheduler_job_id"] == "gpu-123"
    assert result["submitted_workflow_run_id"] == scheduled.metadata["submitted_workflow_run_id"]
    assert CampaignRun.objects.filter(
        campaign=campaign,
        workflow_run_id=result["submitted_workflow_run_id"],
        checkpoint_stage=3,
    ).exists()
    assert not CampaignRun.objects.filter(campaign=campaign, workflow_run=scheduled).exists()


@pytest.mark.django_db
def test_queued_gpu_task_recovers_structure_from_origin(monkeypatch):
    from references.models import CalculationSettingsProfile
    from workflows.models import WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(name="queued-gpu-origin-profile")
    workflow = WorkflowDefinition.objects.create(name="queued-gpu-origin-workflow")
    HPCSystem.objects.create(name="ARCHER2", host="example.org", username="user")
    origin = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="gpu-origin",
        metadata={"structure_xyz": "1\nCu\nCu 0 0 0\n"},
    )
    queued = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="gpu-retry",
        metadata={
            "queue_status": "enqueued_celery_gpu",
            "structure": {"scheduled_workflow_run_id": origin.pk},
        },
    )
    captured = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        submitted = WorkflowRun.objects.create(
            workflow=workflow,
            settings_profile=profile,
            run_name="gpu-submitted",
        )
        return SimpleNamespace(workflow_run_id=submitted.pk, scheduler_job_id="gpu-origin-123")

    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run", fake_submit)
    submit_queued_gpu_workflow_task.apply(args=[queued.pk], task_id="gpu-origin-task").get()
    assert captured["structure_xyz"] == origin.metadata["structure_xyz"].strip()


@pytest.mark.django_db
def test_gpu_dispatcher_sends_oldest_queued_run_to_trh(monkeypatch):
    from references.models import CalculationSettingsProfile
    from workflows.models import WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(name="dispatcher-profile")
    workflow = WorkflowDefinition.objects.create(name="dispatcher-workflow")
    HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh.example.org",
        username="user",
    )
    queued = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="queued-trh-gpu",
        metadata={"queue_status": "enqueued_celery_gpu"},
    )

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(settings, "REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS", False)
    monkeypatch.setattr("jobs.tasks._default_ssh_client_factory", None, raising=False)
    monkeypatch.setattr("jobs.submission._default_ssh_client_factory", lambda _hpc: lambda: Client())
    monkeypatch.setattr("jobs.submission._active_user_job_count", lambda _client, username: 0)
    monkeypatch.setattr(
        submit_queued_gpu_workflow_task,
        "delay",
        lambda run_id: SimpleNamespace(id=f"task-{run_id}"),
    )

    result = dispatch_next_queued_gpu_workflow_task()

    queued.refresh_from_db()
    assert result == {"submitted": 1, "workflow_run_id": queued.pk, "task_id": f"task-{queued.pk}"}
    assert queued.metadata["queue_status"] == "submitting_celery_gpu"
    assert queued.metadata["submission"]["hpc_name"] == "TRH408"


@pytest.mark.django_db
def test_trh_gpu_dispatcher_returns_paused_reason(monkeypatch):
    from references.models import CalculationSettingsProfile
    from workflows.models import WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(name="paused-trh-profile")
    workflow = WorkflowDefinition.objects.create(name="paused-trh-workflow")
    HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh.example.org",
        username="user",
    )
    WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="queued-trh-gpu-paused",
        metadata={"queue_status": "enqueued_celery_gpu"},
    )
    monkeypatch.setattr(settings, "REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS", True)
    assert dispatch_next_queued_gpu_workflow_task() == {
        "submitted": 0,
        "reason": "TRH408 GPU submissions paused",
    }


@pytest.mark.django_db
def test_trh_submission_redirects_to_falcon_when_paused(monkeypatch):
    from references.models import CalculationSettingsProfile
    from workflows.models import WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(name="redirect-trh-profile")
    workflow = WorkflowDefinition.objects.create(name="redirect-trh-workflow")
    HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="falcon.example.org",
        username="user123",
    )
    queued = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="queued-trh-redirect",
        metadata={
            "queue_status": "enqueued_celery_gpu",
            "submission": {"hpc_name": "TRH408"},
        },
    )

    captured_args = {}

    def mock_submit(**kwargs):
        captured_args.update(kwargs)
        return SimpleNamespace(
            workflow_run_id=queued.pk,
            job_record_id=1,
            scheduler_job_id="12345",
            remote_run_dir="/tmp/test",
        )

    monkeypatch.setattr(settings, "REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS", True)
    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run", mock_submit)

    result = submit_queued_gpu_workflow_task.run(queued.pk)

    assert result["scheduler_job_id"] == "12345"
    assert captured_args["hpc_system"].name == "FALCON"



@pytest.mark.django_db
def test_gpu_dispatcher_skips_runs_in_paused_campaigns(monkeypatch):
    from references.models import CalculationSettingsProfile
    from workflows.models import Campaign, CampaignRun, WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(name="paused-dispatcher-profile")
    workflow = WorkflowDefinition.objects.create(name="paused-dispatcher-workflow")
    HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh.example.org",
        username="user",
    )
    paused = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="paused-alloy-gpu",
        metadata={"queue_status": "enqueued_celery_gpu"},
    )
    campaign = Campaign.objects.create(name="Paused alloy", slug="paused-alloy", status="paused")
    CampaignRun.objects.create(campaign=campaign, workflow_run=paused)

    class Client:
        def close(self):
            pass

    monkeypatch.setattr(settings, "REACTION_HUB_PAUSE_TRH_GPU_SUBMISSIONS", False)
    monkeypatch.setattr("jobs.submission._default_ssh_client_factory", lambda _hpc: lambda: Client())
    monkeypatch.setattr("jobs.submission._active_user_job_count", lambda _client, username: 0)


    assert dispatch_next_queued_gpu_workflow_task() == {
        "submitted": 0,
        "reason": "No queued GPU workflows",
    }


def test_serial_recovery_queue_polling_is_backed_off_and_staggered():
    countdowns = {_queue_retry_countdown(job_id) for job_id in range(2098, 2161)}
    assert min(countdowns) >= 1800
    assert max(countdowns) < 2700
    assert len(countdowns) == 63


def test_default_worker_consumes_durable_submission_queue(settings):
    assert {queue.name for queue in settings.CELERY_TASK_QUEUES} == {"celery", "submissions"}
    assert settings.CELERY_WORKER_CONCURRENCY == 1
    assert settings.CELERY_WORKER_PREFETCH_MULTIPLIER == 1
    assert settings.CELERY_TASK_ROUTES["jobs.tasks.restore_ct_serial_job_task"] == {
        "queue": "submissions"
    }


def test_alloy_growth_wrapper_uses_regular_submission_input():
    wrapper = _build_wrapper_script(
        workflow_module="utilities.similarity.alloy_nanocluster_growth_workflow",
        run_mode="alloy_growth",
        hpc_name="TRH408",
        n_iter=1,
        use_tddft_optimisation=False,
        fmax=0.05,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=520,
        basis_keyword="lanl2dz",
        xc_functional="pbe0",
        solvent=None,
        spin_multiplicity=None,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=8,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        alloy_config={"add_symbols": ["Cu", "Pb"], "n_additions": 13},
    )

    assert 'read("input_structure.extxyz", index=\'-1\')' in wrapper
    assert "grow_alloy_nanocluster_by_atom_addition" in wrapper


def test_alloy_seed_optimisation_wrapper_reuses_growth_cache():
    wrapper = _build_wrapper_script(
        workflow_module="utilities.similarity.alloy_nanocluster_growth_workflow",
        run_mode="alloy_growth",
        hpc_name="ARCHER2",
        n_iter=1,
        use_tddft_optimisation=False,
        fmax=0.05,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=520,
        basis_keyword="lanl2dz",
        xc_functional="pbe0",
        solvent=None,
        spin_multiplicity=None,
        nodes=1,
        tasks_per_node=4,
        cpus_per_task=8,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        alloy_config={"optimize_seed_only": True, "mace_device": "cuda"},
    )

    assert "optimize_alloy_nanocluster" in wrapper
    assert "optimize_seed_only" not in wrapper
    assert "index=':'" in wrapper


def test_dft_mlneb_wrapper_is_valid_python_and_selects_catlearn_backend():
    wrapper = _build_wrapper_script(
        workflow_module="cats.working_directory.active_learning_neb_workflow",
        run_mode="active_learning_neb",
        hpc_name="ARCHER2",
        n_iter=1,
        use_tddft_optimisation=False,
        fmax=0.05,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=520,
        basis_keyword="lanl2dz",
        xc_functional="cam-b3lyp",
        solvent=None,
        spin_multiplicity=None,
        nodes=1,
        tasks_per_node=8,
        cpus_per_task=1,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        neb_config={"backend": "dft", "final_image_strategy": "explicit_final_image"},
    )

    compile(wrapper, "run_workflow_submit.py", "exec")
    assert wrapper.count("backend='dft'") == 1
    assert "run_active_learning_neb_workflow" in wrapper


def test_dft_mlneb_evaluation_only_does_not_require_final_endpoint():
    wrapper = _build_wrapper_script(
        workflow_module="cats.working_directory.active_learning_neb_workflow",
        run_mode="active_learning_neb",
        hpc_name="ARCHER2",
        n_iter=1,
        use_tddft_optimisation=False,
        fmax=0.05,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=520,
        basis_keyword="lanl2dz",
        xc_functional="cam-b3lyp",
        solvent=None,
        spin_multiplicity=None,
        nodes=1,
        tasks_per_node=8,
        cpus_per_task=1,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        neb_config={"backend": "dft", "evaluation_only": True},
    )

    assert 'final = None' in wrapper
    assert 'read("input_structure_final.extxyz")' not in wrapper
    assert "evaluation_only=True" in wrapper


@pytest.mark.parametrize("excited_roots", [[], [18]])
def test_charge_transfer_wrapper_converts_and_retains_replot_data(excited_roots):
    wrapper = _build_wrapper_script(
        workflow_module="cats.working_directory.tddft_active_learning_workflow",
        run_mode="charge_transfer",
        hpc_name="ARCHER2",
        n_iter=1,
        use_tddft_optimisation=False,
        fmax=0.01,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=520,
        basis_keyword="lanl2dz",
        xc_functional="cam-b3lyp",
        solvent=None,
        spin_multiplicity=None,
        nodes=1,
        tasks_per_node=4,
        cpus_per_task=1,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        charge_transfer_config={
            "source_run_id": 42,
            "ground_state_mulliken_only": True,
            "mulliken_excitation_numbers": excited_roots,
            "source_working_directory": "/work/PROJECT/GROUP/user/source_run",
            "orca_basename": "orca",
            "nroots": 400,
            "fragments_0based": {"1": list(range(63)), "2": [63, 64]},
            "fragments_1based": {"1": "1-63", "2": "64-65"},
            "density_analysis": {"enabled": True, "grid_quality": 3},
        },
    )

    assert "get_full_ct_spectrum" in wrapper
    assert "plot_charge_transfer_spectrum_from_folders" not in wrapper
    assert 'source_root / "DFT" / "final_tddft_spectrum"' in wrapper
    assert wrapper.index('source_root / "DFT" / "final_tddft_spectrum"') < wrapper.index("source_root,\n")
    assert 'fragments = {\'1\': \'1-63\', \'2\': \'64-65\'}' in wrapper
    assert "_parse_ct_multiple" in wrapper
    assert "_parse_ifct_files" in wrapper
    assert "mulliken_population_routine" in wrapper
    assert "ground_state_mulliken_only = True" in wrapper
    assert "if ground_state_mulliken_only:" in wrapper
    assert 'source_directory / "initial.gbw"' in wrapper
    assert "if source_output.is_file():" in wrapper
    assert '"ground_state_mulliken": ground_state_mulliken' in wrapper
    assert f"for state in {excited_roots!r}:" in wrapper
    assert '"excited_state_mulliken": excited_state_mulliken' in wrapper
    assert '"18", "13"' in wrapper
    assert 'NO_{state:04d}.mwfn' in wrapper
    assert '["mulliken_population"] if ground_state_mulliken_only' in wrapper
    assert 'Path("orca.gbw").unlink(missing_ok=True)' in wrapper
    assert "generate_excitation_density_analysis" in wrapper
    assert '"density_analysis": density_analysis' in wrapper
    compile(wrapper, "run_workflow_submit.py", "exec")


def test_density_analysis_wrapper_skips_fragment_ct_and_selects_orca_excitations():
    wrapper = _build_wrapper_script(
        workflow_module="cats.working_directory.tddft_active_learning_workflow",
        run_mode="charge_transfer",
        hpc_name="ARCHER2",
        n_iter=1,
        use_tddft_optimisation=False,
        fmax=0.01,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=520,
        basis_keyword="lanl2dz",
        xc_functional="cam-b3lyp",
        solvent=None,
        spin_multiplicity=1,
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        charge_transfer_config={
            "source_working_directory": "/work/source",
            "density_analysis_only": True,
            "density_analysis": {"enabled": True, "grid_quality": 3},
        },
    )

    assert "density_analysis_only = True" in wrapper
    assert "ifct_only = False" in wrapper
    assert "if ifct_only:" in wrapper
    assert 'transitions = extract_uv_transitions("orca.out")' in wrapper
    assert 'from mep_state_tracker_runtime import ContinuousMEPTracker' in wrapper
    assert '"run_type": "density_analysis" if density_analysis_only' in wrapper
    compile(wrapper, "run_workflow_submit.py", "exec")


def test_single_optimisation_wrapper_runs_complete_spectroscopy_bundle():
    wrapper = _build_wrapper_script(
        workflow_module="cats.working_directory.tddft_active_learning_workflow",
        run_mode="single_optimisation",
        hpc_name="ARCHER2",
        n_iter=1,
        use_tddft_optimisation=False,
        fmax=0.01,
        active_learning_optimizer="bfgs",
        qm_backend="orca",
        qm_backend_settings={},
        target_wavelength_nm=520,
        basis_keyword="lanl2dz",
        xc_functional="cam-b3lyp",
        solvent=None,
        spin_multiplicity=None,
        nodes=1,
        tasks_per_node=128,
        cpus_per_task=1,
        al_steps_per_iter=200,
        max_al_steps_per_iter=None,
        single_constraints_config={
            "final_tddft_spectrum": True,
            "snapshot_single_point": True,
            "cluster_tddft_difference_spectrum": True,
            "charge_transfer_spectrum": True,
            "final_tddft_nroots": 450,
            "final_tddft_gradient_root": 2,
        },
        charge_transfer_config={
            "orca_basename": "orca",
            "nroots": 450,
            "fragments_0based": {"1": list(range(63)), "2": [63, 64]},
            "fragments_1based": {"1": "1-63", "2": "64-65"},
        },
    )

    assert "final_tddft_spectrum=True" in wrapper
    assert "system_charge=0" in wrapper
    assert "snapshot_single_point=True" in wrapper
    assert "final_tddft_gradient_root=2" in wrapper
    assert "final_cluster_tddft_difference_spectrum=True" in wrapper
    assert 'ct_directory = workflow_root / "DFT" / "final_tddft_spectrum"' in wrapper
    assert "get_full_ct_spectrum" in wrapper
    assert 'summary["charge_transfer"]' in wrapper
    compile(wrapper, "run_workflow_submit.py", "exec")
from references.models import CalculationSettingsProfile
from results.models import Artifact, ArtifactKind
from workflows.models import WorkflowDefinition, WorkflowRun


@pytest.mark.django_db
def test_serial_recovery_duplicate_task_exits_without_polling():
    workflow = WorkflowDefinition.objects.create(name="serial_restore_owner", code_path="x.py")
    profile = CalculationSettingsProfile.objects.create(name="serial_restore_owner")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="serial_restore_owner",
    )
    job = JobRecord.objects.create(
        workflow_run=run,
        scheduler_job_id="cancelled",
        job_name="serial_restore_owner",
        status=JobStatus.CANCELLED,
        scheduler_payload={"ct_serial_restore_task_id": "owner-task"},
    )

    result = restore_ct_serial_job_task.apply(args=[job.pk], task_id="duplicate-task").get()

    assert result["skipped"] == "duplicate_restore_task"
    assert result["owner_task_id"] == "owner-task"


def test_nwchem_hse06_submission_settings_are_expanded():
    xc = _normalize_xc_for_backend("HSE06", "nwchem")

    assert xc == "xpbe96 1.0 xcampbe96 -0.25 cpbe96 1.0 srhfexch 0.25"
    assert _default_nwchem_cam(xc) == "0.11 cam_alpha 0.0 cam_beta 1.0"


class _DummyChannel:
    def __init__(self, code: int):
        self._code = code

    def recv_exit_status(self) -> int:
        return self._code


class _DummyStream:
    def __init__(self, text: str, code: int):
        self._payload = text.encode("utf-8")
        self.channel = _DummyChannel(code)

    def read(self) -> bytes:
        return self._payload


class _FakeSFTPFile:
    def __init__(self, store: dict[str, str], path: str):
        self._store = store
        self._path = path
        self._buf = []

    def write(self, text: str):
        self._buf.append(text)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._store[self._path] = "".join(self._buf)
        return False


class _FakeSFTP:
    def __init__(self):
        self.files_written: dict[str, str] = {}
        self.mode_changes: dict[str, int] = {}

    def file(self, path: str, mode: str):
        assert mode == "w"
        return _FakeSFTPFile(self.files_written, path)

    def chmod(self, path: str, mode: int):
        self.mode_changes[path] = mode

    def close(self):
        return None


class _FakeSSHClient:
    def __init__(self):
        self.commands: list[str] = []
        self.sftp = _FakeSFTP()
        self.closed = False
        self.squeue_outputs: list[str] = []

    def exec_command(self, command: str):
        self.commands.append(command)
        if "squeue --noheader" in command and self.squeue_outputs:
            out, err, code = self.squeue_outputs.pop(0), "", 0
        elif "sbatch" in command:
            out, err, code = "Submitted batch job 12345\n", "", 0
        else:
            out, err, code = "", "", 0
        return None, _DummyStream(out, code), _DummyStream(err, code)

    def open_sftp(self):
        return self.sftp

    def close(self):
        self.closed = True


class _FakeExecCheckedSSHClient:
    def __init__(self):
        self.commands: list[str] = []
        self.closed = False

    def exec_checked(self, command: str):
        self.commands.append(command)
        if "sbatch" in command:
            return "Submitted batch job 12345\n"
        return ""

    # OpenSSH interface includes upload_text; in batched mode this should not be used.
    def upload_text(self, remote_path: str, text: str):
        raise AssertionError("upload_text should not be called in batched stage mode")

    def close(self):
        self.closed = True


def test_forced_tty_uses_existing_chunked_upload_path():
    client = _FakeExecCheckedSSHClient()
    client.force_tty = True

    assert not _supports_batched_remote_stage(client)


def test_batched_remote_stage_includes_extra_workflow_files():
    client = _FakeExecCheckedSSHClient()

    _stage_remote_run_files_batched(
        ssh_client=client,
        remote_run_dir="/work/user/run",
        wrapper_filename="run.py",
        submit_filename="submit.sh",
        structure_text="initial",
        structure_rel_path="initial.extxyz",
        wrapper_script="wrapper",
        submission_script="submit",
        extra_remote_files={"input_structure_final.extxyz": "final"},
    )

    assert len(client.commands) == 1
    assert "/work/user/run/input_structure_final.extxyz" in client.commands[0]
    assert "printf %s final" in client.commands[0]


@pytest.mark.django_db
def test_submit_charge_transfer_job_preserves_explicit_taskfarm_queue(monkeypatch, settings):
    profile = CalculationSettingsProfile.objects.create(
        name="ct_taskfarm_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
    )
    workflow = WorkflowDefinition.objects.create(
        name="ct_taskfarm_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="ARCHER2",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()
    monkeypatch.setattr("jobs.submission._store_initial_structure_artifact", lambda **_kwargs: None)

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="au20co_taskfarm_ct",
        structure_xyz="22\nAu20CO\n" + "Au 0 0 0\n" * 20 + "C 0 0 1\nO 0 0 2\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="taskfarm",
        walltime="24:00:00",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="charge_transfer",
        charge_transfer_config={
            "source_run_id": 12,
            "source_working_directory": "/work/PROJECT/GROUP/user123/source_run",
            "orca_basename": "orca",
            "nroots": 100,
            "fragments_0based": {"1": list(range(20)), "2": [20, 21]},
            "fragments_1based": {"1": "1-20", "2": "21-22"},
        },
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    assert run.metadata["submission"]["partition"] == "standard"
    assert run.metadata["submission"]["qos"] == "taskfarm"
    assert run.metadata["submission"]["nodes"] == 1
    assert run.metadata["submission"]["tasks_per_node"] == 8
    submit_script = fake_client.sftp.files_written[
        "/work/PROJECT/GROUP/user123/reaction_hub_runs/au20co_taskfarm_ct/submit_workflow.sh"
    ]
    assert "#SBATCH --qos=taskfarm" in submit_script


@pytest.mark.django_db
def test_submit_charge_transfer_job_preserves_explicit_standard_queue(monkeypatch, settings):
    profile = CalculationSettingsProfile.objects.create(
        name="ct_standard_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
    )
    workflow = WorkflowDefinition.objects.create(
        name="ct_standard_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="ARCHER2",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()
    monkeypatch.setattr("jobs.submission._store_initial_structure_artifact", lambda **_kwargs: None)

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="au20co_standard_ct",
        structure_xyz="22\nAu20CO\n" + "Au 0 0 0\n" * 20 + "C 0 0 1\nO 0 0 2\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="standard",
        walltime="24:00:00",
        nodes=1,
        tasks_per_node=128,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="charge_transfer",
        charge_transfer_config={
            "source_run_id": 12,
            "source_working_directory": "/work/PROJECT/GROUP/user123/source_run",
            "orca_basename": "orca",
            "nroots": 100,
            "fragments_0based": {"1": list(range(20)), "2": [20, 21]},
            "fragments_1based": {"1": "1-20", "2": "21-22"},
        },
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    assert run.metadata["submission"]["partition"] == "standard"
    assert run.metadata["submission"]["qos"] == "standard"
    assert run.metadata["submission"]["nodes"] == 1
    assert run.metadata["submission"]["tasks_per_node"] == 128
    submit_script = fake_client.sftp.files_written[
        "/work/PROJECT/GROUP/user123/reaction_hub_runs/au20co_standard_ct/submit_workflow.sh"
    ]
    assert "#SBATCH --qos=standard" in submit_script



@pytest.mark.django_db
def test_mulliken_only_recovery_uses_archer2_short_queue(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(
        name="mulliken_short",
        basis_set="LANL2DZ",
    )
    workflow = WorkflowDefinition.objects.create(name="mulliken_short", code_path="x.py")
    system = HPCSystem.objects.create(
        name="ARCHER2",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()
    monkeypatch.setattr("jobs.submission._store_initial_structure_artifact", lambda **_kwargs: None)

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="mulliken_short",
        structure_xyz="2\nAuO\nAu 0 0 0\nO 0 0 2\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="serial",
        qos="serial",
        walltime="24:00:00",
        nodes=1,
        tasks_per_node=8,
        cpus_per_task=1,
        module_loads=[],
        venv_activate_path="/venv/bin/activate",
        pythonpath_entries=["/cplas"],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="charge_transfer",
        charge_transfer_config={
            "ground_state_mulliken_only": True,
            "source_run_id": 12,
            "source_working_directory": "/work/source",
            "fragments_0based": {"1": [0], "2": [1]},
            "fragments_1based": {"1": "1", "2": "2"},
        },
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    submit_script = fake_client.sftp.files_written[
        "/work/PROJECT/GROUP/user123/reaction_hub_runs/mulliken_short/submit_workflow.sh"
    ]
    assert run.metadata["submission"]["qos"] == "short"
    assert run.metadata["submission"]["workflow_total_cpus"] == 1
    assert "#SBATCH --partition=standard" in submit_script
    assert "#SBATCH --qos=short" in submit_script
    assert "#SBATCH --time=00:20:00" in submit_script
    assert "#SBATCH --tasks-per-node=1" in submit_script


@pytest.mark.django_db
@pytest.mark.parametrize("reuse_wavefunction", [False, True])
def test_submit_combined_spectroscopy_job_is_tagged_and_uses_final_tddft_roots(monkeypatch, reuse_wavefunction):
    profile = CalculationSettingsProfile.objects.create(
        name="combined_submit_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
    )
    workflow = WorkflowDefinition.objects.create(
        name="combined_submit_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="ARCHER2",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()
    monkeypatch.setattr("jobs.submission._store_initial_structure_artifact", lambda **_kwargs: None)

    seed_config = {}
    if reuse_wavefunction:
        from results.models import Artifact
        seed_run = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="seed")
        seed_job = JobRecord.objects.create(workflow_run=seed_run, hpc_system=system, remote_workdir="/work/seed", scheduler_job_id="42")
        seed = Artifact.objects.create(workflow_run=seed_run, job=seed_job, path="TDDFT/scan/orca.gbw", label="seed")
        seed_config = {"snapshot_single_point": True, "initial_wavefunction_artifact_id": seed.pk,
                       "final_tddft_gradient_root": 2}

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="combined_spectroscopy_submit",
        structure_xyz="2\nAuC\nAu 0 0 0\nC 0 0 2\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="standard",
        walltime="24:00:00",
        nodes=1,
        tasks_per_node=128,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        single_constraints_config={
            "cluster_tddft_difference_spectrum": True,
            "final_tddft_nroots": 14 if reuse_wavefunction else 75,
            **seed_config,
        },
        charge_transfer_config={
            "nroots": 400,
            "fragments_0based": {"1": [0], "2": [1]},
            "fragments_1based": {"1": "1", "2": "2"},
        },
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    submission = run.metadata["submission"]
    wrapper = fake_client.sftp.files_written[
        "/work/PROJECT/GROUP/user123/reaction_hub_runs/combined_spectroscopy_submit/run_workflow_submit.py"
    ]
    assert submission["tags"] == ["charge_transfer"]
    assert submission["single_constraints_config"]["final_tddft_spectrum"] is True
    count = 14 if reuse_wavefunction else 75
    assert submission["charge_transfer_config"]["nroots"] == count
    assert f"final_tddft_nroots={count}" in wrapper
    assert f'"nroots": {count}' in wrapper
    if reuse_wavefunction:
        assert submission["single_constraints_config"]["initial_wavefunction_artifact_id"] == seed.pk
        assert any("/work/seed/TDDFT/scan/orca.gbw" in command and "initial.gbw" in command for command in fake_client.commands)


def test_scaled_final_tddft_nroots_matches_reference_sizes():
    assert _scaled_final_tddft_nroots("13\nsmall\n" + "H 0 0 0\n" * 13) == 50
    assert _scaled_final_tddft_nroots("63\nlarge\n" + "Au 0 0 0\n" * 63) == 450


@pytest.mark.django_db
def test_submit_slurm_workflow_run_creates_records_and_remote_files(settings, tmp_path: Path):
    settings.REACTION_HUB_SUBMIT_AL_STEPS_PER_ITER = 200
    settings.REACTION_HUB_SUBMIT_MAX_AL_STEPS_PER_ITER = 200
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((tmp_path / "remote_cache").resolve())
    profile = CalculationSettingsProfile.objects.create(
        name="co_cam_ref",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-submit",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="co_ref_test",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="standard",
        reservation="debug-res",
        walltime="01:00:00",
        nodes=1,
        tasks_per_node=4,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        fmax=0.02,
        target_wavelength_nm=510.0,
        spin_multiplicity=3,
        system_charge=1,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        scan_config=None,
        submitted_by="tester",
        structure_metadata={"source_filename": "co.xyz"},
        ssh_client_factory=lambda: fake_client,
    )

    assert result.scheduler_job_id == "12345"
    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    job = JobRecord.objects.get(pk=result.job_record_id)

    assert run.working_directory.endswith("/reaction_hub_runs/co_ref_test")
    assert run.metadata["submission"]["account"] == "SLURM_ACCOUNT"
    assert run.metadata["submission"]["reservation"] == "debug-res"
    assert run.metadata["submission"]["workflow_cpus_per_node"] == 4
    assert run.metadata["submission"]["workflow_total_cpus"] == 4
    assert run.metadata["submission"]["n_iter"] == 1
    assert run.metadata["submission"]["use_tddft_optimisation"] is False
    assert run.metadata["submission"]["spin_multiplicity"] == 3
    assert run.system_charge == 1
    assert run.metadata["submission"]["system_charge"] == 1
    assert run.metadata["submission"]["max_al_steps_per_iter"] is None
    assert "target_wavelength_nm" not in run.metadata["submission"]
    assert run.metadata["submission"]["active_learning_optimizer"] == "bfgs"
    assert run.metadata["submission"]["qm_backend"] == "orca"
    assert run.metadata["submission"]["module_loads"] == [
        "epcc-setup-env",
        "cray-python",
        "orca/6.0.0",
    ]
    assert run.metadata["structure"]["source_filename"] == "co.xyz"
    assert job.scheduler_job_id == "12345"
    assert job.status == JobStatus.INITIATED
    assert job.remote_workdir.endswith("/user123/12345")
    assert job.stdout_path.endswith("/workflow.12345.log")
    assert job.scheduler_payload["qm_backend"] == "orca"
    assert fake_client.closed
    initial_structure = Artifact.objects.get(workflow_run=run, path="input_structure.extxyz")
    assert initial_structure.kind == ArtifactKind.STRUCTURE
    assert initial_structure.label == "structure:input_structure.extxyz"
    assert initial_structure.job_id == job.pk
    metadata = initial_structure.metadata if isinstance(initial_structure.metadata, dict) else {}
    assert metadata.get("source") == "submission"
    assert metadata.get("is_initial_structure") is True
    assert str(metadata.get("remote_path", "")).endswith("/reaction_hub_runs/co_ref_test/input_structure.extxyz")
    local_cache_path = Path(str(metadata.get("local_cache_path") or ""))
    assert local_cache_path.exists() and local_cache_path.is_file()
    assert "C 0 0 0" in local_cache_path.read_text(encoding="utf-8")

    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/co_ref_test"
    assert f"{remote_dir}/input_structure.extxyz" in fake_client.sftp.files_written
    assert f"{remote_dir}/run_workflow_submit.py" in fake_client.sftp.files_written
    assert f"{remote_dir}/submit_workflow.sh" in fake_client.sftp.files_written
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]
    assert 'workflow_nodes = int(os.environ.get("WORKFLOW_NODES", 1))' in wrapper_script
    assert 'workflow_tasks_per_node = int(os.environ.get("WORKFLOW_TASKS_PER_NODE", 4))' in wrapper_script
    assert 'workflow_cpus_per_task = int(os.environ.get("WORKFLOW_CPUS_PER_TASK", 1))' in wrapper_script
    assert 'atoms = read("input_structure.extxyz")' in wrapper_script
    assert 'atoms = read("input_structure.extxyz", format="xyz")' in wrapper_script
    assert "BondScanSettings(enabled=False)" in wrapper_script
    assert "use_tddft_optimisation=False" in wrapper_script
    assert "ensure_initial_dft_converged=False" in wrapper_script
    assert "al_steps_per_iter=200" in wrapper_script
    assert "max_al_steps_per_iter=None" in wrapper_script
    assert "fmax=0.02" in wrapper_script
    assert "active_learning_optimizer='bfgs'" in wrapper_script
    assert "qm_backend='orca'" in wrapper_script
    assert "target_wavelength_nm=510.0" in wrapper_script
    assert "spin_multiplicity=3" in wrapper_script
    assert "system_charge=1" in wrapper_script
    assert "export WORKFLOW_NODES=1" in submit_script
    assert "export WORKFLOW_TASKS_PER_NODE=4" in submit_script
    assert "export WORKFLOW_CPUS_PER_TASK=1" in submit_script
    assert "export SLURM_CPU_FREQ_REQ=2250000" in submit_script
    assert "#SBATCH --reservation=debug-res" in submit_script
    assert "module load orca/6.0.0" in submit_script
    assert (
        "export PYTHONPATH=/work/PROJECT/GROUP/user123/ORCA/cplas${PYTHONPATH:+:$PYTHONPATH}"
        in submit_script
    )
    assert 'REACTION_HUB_PREVIOUS_JOB_IDS_CSV=""' in submit_script
    assert 'REACTION_HUB_FULL_RSYNC="${REACTION_HUB_FULL_RSYNC:-0}"' in submit_script
    assert "JOB_ARCHIVE_DIRECTORY" not in submit_script
    assert "build_rsync_excludes() {" in submit_script
    assert "sync_minimal_inputs_to_temp() {" in submit_script
    assert "sync_minimal_outputs_to_job_archive" not in submit_script
    assert "sync_minimal_outputs_back_home() {" in submit_script
    assert 'if [ "$REACTION_HUB_FULL_RSYNC" = "1" ]; then' in submit_script
    assert 'sync_minimal_inputs_to_temp' in submit_script
    assert 'sync_minimal_outputs_back_home || true' in submit_script
    assert 'rsync -av "$HOME_DIRECTORY/" "$TEMP_DIRECTORY/" "${RSYNC_EXCLUDES[@]}"' in submit_script
    assert 'rsync -av "$TEMP_DIRECTORY/" "$HOME_DIRECTORY/" "${RSYNC_EXCLUDES[@]}" || true' in submit_script


@pytest.mark.django_db
def test_submit_slurm_workflow_run_supports_user_token_in_scratch_base():
    profile = CalculationSettingsProfile.objects.create(
        name="co_cam_ref_user_token",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_user_token",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="falcon-submit-user-token",
        scheduler_type="slurm",
        host="example.org",
        username="alice",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="co_ref_test_user_token",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/home/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/$USER/",
        account="FALCON_SLURM_ACCOUNT",
        partition="dev",
        qos="",
        walltime="01:00:00",
        nodes=1,
        tasks_per_node=4,
        cpus_per_task=1,
        module_loads=["Python/3.11.5-GCCcore-13.2.0", "ORCA"],
        venv_activate_path="/scratch/PROJECT/shared/venv/mace_venv/bin/activate",
        pythonpath_entries=[
            "/scratch/PROJECT/shared/software/github/cplas",
            "/scratch/PROJECT/shared/software/github/carmm",
            "/scratch/PROJECT/shared/software/github/CatLearn",
        ],
        fmax=0.02,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        scan_config=None,
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    job = JobRecord.objects.get(pk=result.job_record_id)
    remote_dir = "/home/alice/reaction_hub_runs/co_ref_test_user_token"
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]

    assert 'TEMP_DIRECTORY="/scratch/PROJECT/$USER/$SLURM_JOB_ID"' in submit_script
    assert 'TEMP_DIRECTORY="/scratch/PROJECT/$USER/$USER/$SLURM_JOB_ID"' not in submit_script
    assert job.remote_workdir == "/scratch/PROJECT/alice/12345"


@pytest.mark.django_db
def test_submit_slurm_workflow_run_uses_two_calls_for_openssh_style_client():
    profile = CalculationSettingsProfile.objects.create(
        name="co_cam_ref_batched",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_batched",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-submit-batched",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeExecCheckedSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="co_ref_test_batched",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="standard",
        walltime="01:00:00",
        nodes=1,
        tasks_per_node=4,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        fmax=0.02,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        scan_config=None,
        submitted_by="tester",
        structure_metadata={"source_filename": "co.xyz"},
        ssh_client_factory=lambda: fake_client,
    )

    assert result.scheduler_job_id == "12345"
    assert fake_client.closed
    assert len(fake_client.commands) == 3
    assert "Run directory already exists" in fake_client.commands[1]
    assert "printf %s" in fake_client.commands[1]
    assert "/work/PROJECT/GROUP/user123/reaction_hub_runs/co_ref_test_batched/input_structure.extxyz" in fake_client.commands[1]
    assert "sbatch" in fake_client.commands[2]


@pytest.mark.django_db
def test_submit_slurm_workflow_run_writes_single_constraint_wrapper():
    profile = CalculationSettingsProfile.objects.create(
        name="co_cam_single_constraints",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_single_constraints",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-single-constraints",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="co_single_constraints_test",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="standard",
        walltime="01:00:00",
        nodes=1,
        tasks_per_node=4,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        fmax=0.02,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        single_constraints_config={
            "fixed_symbols": ("Cu", "Ti"),
            "bond_pair_indices": (0, 1),
            "target_distance": 1.25,
            "fix": 0.5,
            "cluster_tddft_difference_spectrum": True,
        },
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/co_single_constraints_test"
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    single_constraints = run.metadata["submission"].get("single_constraints_config", {})

    assert "from ase.constraints import FixAtoms, FixBondLength" in wrapper_script
    assert "fixed_symbols = ('Cu', 'Ti')" in wrapper_script
    assert "bond_pair_indices = (0, 1)" in wrapper_script
    assert "target_distance = 1.25" in wrapper_script
    assert "constraints.append(FixAtoms(indices=fixed_indices))" in wrapper_script
    assert "constraints.append(" in wrapper_script
    assert "FixBondLength(" in wrapper_script
    assert single_constraints.get("fixed_symbols") == ["Cu", "Ti"]
    assert single_constraints.get("bond_pair_indices") == [0, 1]
    assert single_constraints.get("target_distance") == 1.25
    assert single_constraints.get("fix") == 0.5
    assert single_constraints.get("cluster_tddft_difference_spectrum") is True
    assert "final_cluster_tddft_difference_spectrum=True" in wrapper_script


@pytest.mark.django_db
def test_submit_slurm_workflow_run_writes_nwchem_single_test_wrapper():
    profile = CalculationSettingsProfile.objects.create(
        name="nwchem_submit_profile",
        xc_functional="PBE0",
        basis_set="def2-SVP",
        solvent="",
        metadata={
            "qm_backend": "nwchem",
            "nwchem": {
                "version": "7.2.2",
                "memory_mb": 4000,
                "molden": True,
            },
        },
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_nwchem",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-nwchem",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="co_nwchem_test",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="standard",
        walltime="01:00:00",
        nodes=1,
        tasks_per_node=4,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        fmax=0.02,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        single_constraints_config={
            "final_tddft_spectrum": True,
            "cluster_tddft_difference_spectrum": True,
            "final_tddft_nroots": 10,
        },
        qm_backend="nwchem",
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    job = JobRecord.objects.get(pk=result.job_record_id)
    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/co_nwchem_test"
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]

    assert run.metadata["submission"]["qm_backend"] == "nwchem"
    assert run.metadata["submission"]["qm_backend_settings"]["memory_mb"] == 4000
    assert run.metadata["submission"]["module_loads"] == [
        "epcc-setup-env",
        "cray-python",
        "other-software",
        "nwchem/7.2.2",
    ]
    assert job.scheduler_payload["qm_backend"] == "nwchem"
    assert "module load orca/6.0.0" not in submit_script
    assert "module load other-software\nmodule load nwchem/7.2.2" in submit_script
    assert "module load nwchem/7.2.2" in submit_script
    assert "from cats.working_directory.nwchem_single_workflow import run_nwchem_single_optimisation" in wrapper_script
    assert 'os.environ.setdefault("OMP_NUM_THREADS", "1")' in wrapper_script
    assert 'os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])' in wrapper_script
    assert 'os.environ.setdefault("MKL_DYNAMIC", "FALSE")' in wrapper_script
    assert "run_nwchem_single_optimisation(" in wrapper_script
    assert "basis_keyword='def2-SVP'" in wrapper_script
    assert "xc_functional='PBE0'" in wrapper_script
    assert "cam=None" in wrapper_script
    assert "final_tddft_spectrum=True" in wrapper_script
    assert "final_cluster_tddft_difference_spectrum=True" in wrapper_script
    assert "final_tddft_nroots=10" in wrapper_script
    assert "def export_final_tddft_spectrum" not in wrapper_script
    assert "class working_directory" not in wrapper_script
    assert "def copy_path_if_exists" not in wrapper_script
    assert "get_nwchem_calculator(" not in wrapper_script
    assert '"tddft energy"' not in wrapper_script
    assert '"input_structure_resolved.extxyz"' in submit_script
    assert '"nwchem_final.extxyz"' in submit_script
    assert "sync_root_nwchem_items_back_home()" in submit_script
    assert '"nwchem"' in submit_script
    assert '"nwchem_property"' in submit_script
    assert '"nwchem_tddft"' in submit_script


@pytest.mark.django_db
def test_submit_slurm_workflow_run_uses_falcon_nwchem_runtime_modules():
    profile = CalculationSettingsProfile.objects.create(
        name="nwchem_falcon_profile",
        xc_functional="PBE0",
        basis_set="def2-SVP",
        metadata={"qm_backend": "nwchem", "nwchem": {"version": "7.2.2", "memory_mb": 4000}},
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_nwchem_falcon",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="falcon",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="falcon_nwchem_test",
        structure_xyz="1\nH\nH 0 0 0\n",
        remote_runs_root="/shared/home1/$USER/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/$USER",
        account="FALCON_SLURM_ACCOUNT",
        partition="compute",
        qos="",
        walltime="01:00:00",
        nodes=1,
        tasks_per_node=192,
        cpus_per_task=1,
        module_loads=["Python/3.11.5-GCCcore-13.2.0", "ORCA"],
        venv_activate_path="/scratch/PROJECT/shared/venv/mace_venv/bin/activate",
        pythonpath_entries=["/scratch/PROJECT/shared/software/github/cplas"],
        fmax=0.02,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        qm_backend="nwchem",
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/shared/home1/user123/reaction_hub_runs/falcon_nwchem_test"
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]

    assert run.metadata["submission"]["module_loads"] == [
        "Python/3.11.5-GCCcore-13.2.0",
        "intel",
    ]
    assert run.metadata["submission"]["slurm_memory"] == "720G"
    assert run.metadata["submission"]["qm_backend_settings"]["memory_mb"] == 3840
    assert "module load ORCA" not in submit_script
    assert "module load nwchem/7.2.2" not in submit_script
    assert "module load intel" in submit_script
    assert "#SBATCH --ntasks=192" in submit_script
    assert "#SBATCH --mem=720G" in submit_script
    assert "SLURM_CPU_FREQ_REQ" not in submit_script
    assert "ulimit -u unlimited" in submit_script
    assert "hpc_name='falcon'" in wrapper_script
    assert "NWCHEM_MPI_RANKS" not in wrapper_script
    assert 'os.environ.setdefault("OMP_NUM_THREADS", "1")' in wrapper_script


@pytest.mark.django_db
def test_submit_slurm_workflow_run_scales_final_tddft_nroots(settings):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str((Path("scratch") / "remote_cache_test").resolve())
    profile = CalculationSettingsProfile.objects.create(
        name="co_cam_scaled",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_scaled_nroots",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="archer2",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()
    structure_xyz = "63\nAu63\n" + "\n".join(f"Au {idx} 0 0" for idx in range(63)) + "\n"

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="au63_scaled_nroots",
        structure_xyz=structure_xyz,
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="standard",
        walltime="01:00:00",
        nodes=1,
        tasks_per_node=4,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        fmax=0.02,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        single_constraints_config={
            "final_tddft_spectrum": True,
            "final_tddft_nroots": 50,
            "final_tddft_nroots_mode": "scaled_by_atoms",
        },
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/au63_scaled_nroots"
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]

    assert "final_tddft_nroots=450" in wrapper_script
    assert run.metadata["submission"]["single_constraints_config"]["final_tddft_nroots"] == 450
    assert run.metadata["submission"]["single_constraints_config"]["final_tddft_nroots_mode"] == "scaled_by_atoms"


@pytest.mark.django_db
def test_submit_slurm_workflow_run_writes_tddft_scan_wrapper():
    profile = CalculationSettingsProfile.objects.create(
        name="co_cam_scan",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_scan",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-scan",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="co_scan_test",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="long",
        walltime="96:00:00",
        nodes=2,
        tasks_per_node=32,
        cpus_per_task=1,
        module_loads=["epcc-setup-env", "cray-python", "orca/6.0.0"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        fmax=0.03,
        active_learning_optimizer="fire2",
        target_wavelength_nm=465.0,
        n_iter=4,
        use_tddft_optimisation=True,
        run_mode="tddft_bond_scan",
        scan_config={
            "bond_i": 0,
            "bond_j": 1,
            "increment": 0.1,
            "fix": 0.5,
            "fixed_symbols": ("Cu", "Au"),
            "target_distance": 2.0,
            "distance_change_fraction": 1.0,
            "enable_ts_smoothing": True,
            "refine_points_each_side": 2,
            "restart_extend_steps": 5,
            "restart_extend_increment_A": 0.2,
            "tddft_scan_nroots": 72,
        },
        submitted_by="tester",
        structure_metadata={"source_filename": "co.xyz"},
        ssh_client_factory=lambda: fake_client,
    )

    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/co_scan_test"
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]
    assert 'atoms = read("input_structure.extxyz")' in wrapper_script
    assert 'atoms = read("input_structure.extxyz", format="xyz")' in wrapper_script
    assert "enabled=True" in wrapper_script
    assert "bond_pair_indices=(0, 1)" in wrapper_script
    assert "target_distance=2.0" in wrapper_script
    assert "enable_ts_smoothing=True" in wrapper_script
    assert "refine_points_each_side=2" in wrapper_script
    assert "restart_extend_steps=5" in wrapper_script
    assert "restart_extend_increment_A=0.2" in wrapper_script
    assert "n_iter=4" in wrapper_script
    assert "use_tddft_optimisation=True" in wrapper_script
    assert "fmax=0.03" in wrapper_script
    assert "active_learning_optimizer='fire2'" in wrapper_script
    assert "qm_backend='orca'" in wrapper_script
    assert "target_wavelength_nm=465.0" in wrapper_script
    assert "tddft_scan_nroots=72" in wrapper_script
    assert 'workflow_tasks_per_node = int(os.environ.get("WORKFLOW_TASKS_PER_NODE", 32))' in wrapper_script
    assert 'workflow_cpus_per_task = int(os.environ.get("WORKFLOW_CPUS_PER_TASK", 1))' in wrapper_script
    assert "export WORKFLOW_NODES=2" in submit_script
    assert "export WORKFLOW_TASKS_PER_NODE=32" in submit_script
    assert "export WORKFLOW_CPUS_PER_TASK=1" in submit_script
    assert "export SLURM_CPU_FREQ_REQ=2250000" in submit_script


@pytest.mark.django_db
def test_submit_slurm_workflow_run_writes_coadsorbate_gpu_wrapper_and_adsorbate_files():
    profile = CalculationSettingsProfile.objects.create(
        name="coadsorbate_profile",
        xc_functional="PBE0",
        basis_set="def2-svp",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_coadsorbate",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-coadsorbate",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="coadsorbate_test",
        structure_xyz="2\nAu\nAu 0 0 0\nAu 0 0 2.8\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="gpu",
        qos="gpu-exc",
        walltime="12:00:00",
        nodes=2,
        tasks_per_node=1,
        cpus_per_task=1,
        module_loads=["cray-python", "orca/6.0.0"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="coadsorbate_placement",
        coadsorbate_config={
            "mace_model_path": "/work/PROJECT/GROUP/user123/models/mace.model",
            "mace_device": "cuda",
            "mace_fmax": 0.05,
            "mace_max_steps": 200,
            "mace_dispersion": True,
            "mace_enable_cueq": False,
            "relax_support": True,
            "generation_site_symbols": ("Ti", "Au", "Cu"),
            "generation_passive_symbols": ("Au", "Ce", "O"),
            "generation_height": 2.0,
            "generation_origin_atom_index": 1,
            "generation_site_cutoff": 3.5,
            "generation_include_below_surface": True,
            "generation_force_triangulated_com_fallback": True,
            "generation_rotation_step_degrees": 45.0,
            "generation_include_flipped_orientations": False,
            "keep_bound_only": True,
            "similarity_support_species": ("Zn", "Au", "Ce", "Ti", "O"),
            "site_probe_symbol": "He",
            "site_probe_similarity_cutoff": 0.97,
            "single_mode_duplicate_cutoff": 0.96,
            "candidate_duplicate_cutoff": 0.95,
            "relaxed_minimum_duplicate_cutoff": 0.94,
            "similarity_batch_size": 100,
            "generation_skip_geometry_screening": True,
            "adsorbates": [
                {"xyz": "2\nCO\nC 0 0 0\nO 0 0 1.1\n", "metadata": {"source": "ase_molecule"}},
                {"xyz": "3\nNO2\nN 0 0 0\nO 0 0 1.2\nO 0 1.0 -0.2\n", "metadata": {"source": "upload_file"}},
            ],
        },
        submitted_by="tester",
        structure_metadata={"source": "reference_cluster"},
        ssh_client_factory=lambda: fake_client,
    )

    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/coadsorbate_test"
    assert result.workflow_run_id is not None
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]
    assert f"{remote_dir}/input_structure.extxyz" in fake_client.sftp.files_written
    assert f"{remote_dir}/adsorbates/adsorbate_001.xyz" in fake_client.sftp.files_written
    assert f"{remote_dir}/adsorbates/adsorbate_002.xyz" in fake_client.sftp.files_written
    assert "run_coadsorbate_similarity_workflow" in wrapper_script
    assert 'structure = read("input_structure.extxyz")' in wrapper_script
    assert "adsorbates = [read(path) for path in adsorbate_paths]" in wrapper_script
    assert "candidate_suggested_cutoff_scale=1.0" in wrapper_script
    assert "site_probe_symbol='He'" in wrapper_script
    assert "site_probe_similarity_cutoff=0.97" in wrapper_script
    assert 'site_probe_report_csv_path=site_probe_report_csv_path' in wrapper_script
    assert 'str(site_probe_report_csv_path)' in wrapper_script
    assert "single_mode_duplicate_cutoff=0.96" in wrapper_script
    assert "candidate_duplicate_cutoff=0.95" in wrapper_script
    assert "relaxed_minimum_duplicate_cutoff=0.94" in wrapper_script
    assert "relaxed_minimum_suggested_cutoff_scale=1.0" in wrapper_script
    assert "similarity_batch_size=100" in wrapper_script
    assert '"skip_geometry_screening": True' in wrapper_script
    assert "relax_support=True" in wrapper_script
    assert "require_converged=True" in wrapper_script
    assert '"origin_atom_index": 1' in wrapper_script
    assert '"site_cutoff": 3.5' in wrapper_script
    assert '"include_below_surface": True' in wrapper_script
    assert '"force_triangulated_com_fallback": True' in wrapper_script
    assert '"rotation_step_degrees": 45.0' in wrapper_script
    assert '"include_flipped_orientations": False' in wrapper_script
    assert "keep_bound_only=True" in wrapper_script
    assert "mace_dispersion=True" in wrapper_script
    assert "mace_enable_cueq=False" in wrapper_script
    assert 'final_db_path = Path("scratch/coadsorption_final_for_dft.db")' in wrapper_script
    assert 'single_a_final_db_path = Path("scratch/coadsorption_single_a_final_for_dft.db")' in wrapper_script
    assert 'single_b_final_db_path = Path("scratch/coadsorption_single_b_final_for_dft.db")' in wrapper_script
    assert 'str(single_a_final_db_path)' in wrapper_script
    assert 'str(single_b_final_db_path)' in wrapper_script
    assert "#SBATCH --nodes=2" in submit_script
    assert "#SBATCH --gpus=8" in submit_script
    assert "#SBATCH --exclusive" in submit_script
    assert "#SBATCH --tasks-per-node=" not in submit_script
    assert "#SBATCH --cpus-per-task=" not in submit_script
    assert "export OMP_NUM_THREADS=8" in submit_script
    assert "export SRUN_CPUS_PER_TASK=8" in submit_script
    assert "unset MPICH_GPU_SUPPORT_ENABLED" in submit_script
    assert (
        "time srun --ntasks-per-node=4 --cpus-per-task=8 "
        '--hint=nomultithread --distribution=block:block "$GPU_BIND_SCRIPT" '
        "python3 -u -m similarity.workflows.mace_mpi_entry"
    ) in submit_script
    gpu_bind = fake_client.sftp.files_written[f"{remote_dir}/gpu-bind.sh"]
    assert 'ROCR_VISIBLE_DEVICES="${gpu_map[$SLURM_LOCALID]}"' in gpu_bind
    assert 'PYTORCH_KERNEL_CACHE_PATH="/tmp/torch-kernels-' in gpu_bind
    assert 'mkdir -p "$PYTORCH_KERNEL_CACHE_PATH"' in gpu_bind
    assert 'exec "$@"' in gpu_bind
    assert "module load rocm" in submit_script
    assert "module load orca/6.0.0" not in submit_script
    assert '        "input_structure.extxyz"' in submit_script
    assert '        "adsorbates"' in submit_script
    assert '        "scratch"' in submit_script


@pytest.mark.django_db
def test_submit_slurm_workflow_run_writes_silicate_pore_coadsorbate_wrapper():
    profile = CalculationSettingsProfile.objects.create(
        name="silicate_pore_profile",
        xc_functional="PBE0",
        basis_set="def2-svp",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_silicate_pore",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-silicate",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="silicate_pore_test",
        structure_xyz=(
            "4\nSiO3\n"
            "Si 0 0 0\n"
            "O -1 -1 0\n"
            "O 1 -1 0\n"
            "O 0 -1 -1\n"
        ),
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="gpu",
        qos="gpu-shd",
        walltime="12:00:00",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=1,
        module_loads=["cray-python"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="coadsorbate_placement",
        coadsorbate_config={
            "placement_mode": "silicate_pore",
            "mace_model_path": "/work/PROJECT/GROUP/user123/models/mace.model",
            "mace_device": "cuda",
            "similarity_support_species": ("Si", "O", "Al", "H"),
            "silicate_origin_atom_index": 0,
            "silicate_pore_triangle_indices": (1, 2, 3),
            "silicate_heights": (1.45, 2.5),
            "silicate_clash_scale": 0.8,
            "silicate_push_step": 0.1,
            "silicate_max_push_steps": 8,
            "adsorbates": [
                {"xyz": "6\nCH3OH\nC 0 0.7 0\nO 0 -0.7 0\nH 0 -1.6 0\nH 1 1 0\nH -1 1 0\nH 0 1 1\n", "metadata": {"source": "ase_molecule"}},
            ],
        },
        submitted_by="tester",
        structure_metadata={"source": "upload"},
        ssh_client_factory=lambda: fake_client,
    )

    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/silicate_pore_test"
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    assert result.scheduler_job_id == "12345"
    assert "run_silicate_pore_adsorbate_workflow" in wrapper_script
    assert 'final_db_path = Path("scratch/silicate_pore_final_for_dft.db")' in wrapper_script
    assert '"placement_mode": "silicate_pore"' in wrapper_script
    assert '"dft_db_paths": [str(final_db_path)] if final_db_path.exists() else []' in wrapper_script
    assert '"pore_triangle_indices": (1, 2, 3)' in wrapper_script
    assert '"clash_scale": 0.8' in wrapper_script
    assert '"push_step": 0.1' in wrapper_script


@pytest.mark.django_db
def test_submit_slurm_workflow_run_trh408_uses_home_run_in_place_and_omits_empty_slurm_fields():
    profile = CalculationSettingsProfile.objects.create(
        name="trh408_profile",
        xc_functional="PBE0",
        basis_set="def2-svp",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_coadsorbate_trh408",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh408.example.invalid",
        username="user",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="coadsorbate_trh408_test",
        structure_xyz="2\nAu\nAu 0 0 0\nAu 0 0 2.8\n",
        remote_runs_root="/home/{username}/reaction_hub_runs",
        scratch_base="",
        account="",
        partition="",
        qos="",
        walltime="12:00:00",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=8,
        module_loads=["orca/6.0.0", "cray-python"],
        venv_activate_path="/srv/research/mace_venv/bin/activate",
        pythonpath_entries=[],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="coadsorbate_placement",
        coadsorbate_config={
            "mace_model_path": "/srv/research/models/mace.model",
            "mace_device": "cuda",
            "adsorbates": [
                {"xyz": "2\nCO\nC 0 0 0\nO 0 0 1.1\n", "metadata": {"source": "ase_molecule"}},
            ],
        },
        submitted_by="tester",
        structure_metadata={"source": "reference_cluster"},
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    job = JobRecord.objects.get(pk=result.job_record_id)
    remote_dir = "/home/user/reaction_hub_runs/coadsorbate_trh408_test"
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]

    assert "#SBATCH --gres=gpu:1" in submit_script
    assert "#SBATCH --job-name=testgpu" in submit_script
    assert "#SBATCH --exclusive" in submit_script
    assert "#SBATCH --tasks-per-node=1" not in submit_script
    assert "#SBATCH --cpus-per-task=8" in submit_script
    assert "#SBATCH --account=" not in submit_script
    assert "#SBATCH --partition=" not in submit_script
    assert "#SBATCH --qos=" not in submit_script
    assert "#SBATCH --reservation=" not in submit_script
    assert 'REACTION_HUB_RUN_IN_PLACE=1' in submit_script
    assert 'TEMP_DIRECTORY="$HOME_DIRECTORY"' in submit_script
    assert "time srun --ntasks=1 --gpus-per-task=1" not in submit_script
    assert 'time python3 -u "$python_input" >> "$HOME_DIRECTORY/$output" 2>&1' in submit_script
    assert "module load " not in submit_script
    assert run.metadata["submission"]["structure_artifact_path"] == "input_structure.extxyz"
    assert run.metadata["submission"]["run_in_place"] is True
    assert run.metadata["submission"]["scratch_base"] == "/tmp"
    assert run.metadata["submission"]["partition"] == ""
    assert run.metadata["submission"]["reservation"] == ""
    assert job.remote_workdir == run.working_directory


@pytest.mark.django_db
def test_submit_slurm_workflow_run_trh408_overrides_generic_remote_root_with_home_default():
    profile = CalculationSettingsProfile.objects.create(
        name="trh408_remote_root_profile",
        xc_functional="PBE0",
        basis_set="def2-svp",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_trh408_remote_root",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh408.example.invalid",
        username="user",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="trh408_generic_root_test",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="",
        account="",
        partition="",
        qos="",
        walltime="00:20:00",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=8,
        module_loads=["cray-python"],
        venv_activate_path="",
        pythonpath_entries=[],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        submitted_by="tester",
        structure_metadata={"source": "upload_file"},
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/home/user/reaction_hub_runs/trh408_generic_root_test"
    assert f"{remote_dir}/run_workflow_submit.py" in fake_client.sftp.files_written
    assert run.working_directory == remote_dir
    assert run.metadata["submission"]["remote_runs_root"] == "/home/user/reaction_hub_runs"


@pytest.mark.django_db
def test_submit_slurm_workflow_run_writes_active_learning_neb_wrapper_and_final_image():
    profile = CalculationSettingsProfile.objects.create(
        name="neb_profile",
        xc_functional="PBE0",
        basis_set="def2-svp",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_neb",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh408.example.invalid",
        username="user",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="neb_trh408_test",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/home/{username}/reaction_hub_runs",
        scratch_base="",
        account="",
        partition="",
        qos="",
        walltime="08:00:00",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=8,
        module_loads=["orca/6.0.0", "cray-python"],
        venv_activate_path="/srv/research/mace_venv/bin/activate",
        pythonpath_entries=[],
        fmax=0.05,
        active_learning_optimizer="fire2",
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="active_learning_neb",
        neb_config={
            "backend": "mace",
            "atom_shift_per_image_A": 0.2,
            "min_images": 7,
            "max_images": 21,
            "spring_constant": 0.1,
            "max_steps": 200,
            "climb": True,
            "split_on_minimum": True,
            "minimum_energy_window_eV": 0.02,
            "fixed_symbols": ("Au", "Cu"),
            "mace_model_path": "/srv/research/models/mace.model",
            "mace_device": "cuda",
            "mace_default_dtype": "float64",
            "mace_dispersion": False,
            "catlearn_min_data": 3,
            "catlearn_unc_convergence": 0.05,
            "catlearn_max_unc": 0.03,
            "catlearn_steps": 150,
            "catlearn_ml_steps": 400,
            "catlearn_dtrust": 2.0,
            "final_structure_extxyz": "2\nfinal\nC 0 0 0\nO 0 0 1.35\n",
            "final_structure_metadata": {"source": "upload_file"},
        },
        submitted_by="tester",
        structure_metadata={"source": "upload_file"},
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/home/user/reaction_hub_runs/neb_trh408_test"
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]

    assert f"{remote_dir}/input_structure.extxyz" in fake_client.sftp.files_written
    assert f"{remote_dir}/input_structure_final.extxyz" in fake_client.sftp.files_written
    assert "run_active_learning_neb_workflow" in wrapper_script
    assert 'initial = read("input_structure.extxyz")' in wrapper_script
    assert 'final = read("input_structure_final.extxyz")' in wrapper_script
    assert "backend='mace'" in wrapper_script
    assert "optimizer='fire2'" in wrapper_script
    assert "fixed_symbols=('Au', 'Cu')" in wrapper_script
    assert "catlearn_steps=150" in wrapper_script
    assert "catlearn_ml_steps=400" in wrapper_script
    assert "#SBATCH --gres=gpu:1" in submit_script
    assert "#SBATCH --partition=" not in submit_script
    assert 'REACTION_HUB_RUN_IN_PLACE=1' in submit_script
    assert run.metadata["submission"]["run_mode"] == "active_learning_neb"
    assert run.metadata["submission"]["partition"] == ""
    assert run.metadata["submission"]["neb_config"]["backend"] == "mace"
    assert run.metadata["submission"]["neb_config"]["final_image_strategy"] == "explicit_final_image"
    assert run.metadata["submission"]["neb_config"]["mace_device"] == "cuda"
    assert run.metadata["submission"]["neb_config"]["final_structure_path"] == "input_structure_final.extxyz"


@pytest.mark.django_db
def test_submit_slurm_workflow_run_writes_active_learning_neb_wrapper_for_bond_extension_mode():
    profile = CalculationSettingsProfile.objects.create(
        name="neb_bond_extension_profile",
        xc_functional="PBE0",
        basis_set="def2-svp",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_neb_bond_extension",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="TRH408",
        scheduler_type="slurm",
        host="trh408.example.invalid",
        username="user",
    )
    fake_client = _FakeSSHClient()

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="neb_trh408_bond_extension_test",
        structure_xyz="3\nH2O\nO 0 0 0\nH 0 0 0.96\nH 0.92 0 0.0\n",
        remote_runs_root="/home/{username}/reaction_hub_runs",
        scratch_base="",
        account="",
        partition="",
        qos="",
        walltime="08:00:00",
        nodes=1,
        tasks_per_node=1,
        cpus_per_task=8,
        module_loads=["orca/6.0.0", "cray-python"],
        venv_activate_path="/srv/research/mace_venv/bin/activate",
        pythonpath_entries=[],
        fmax=0.05,
        active_learning_optimizer="fire2",
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="active_learning_neb",
        neb_config={
            "final_image_strategy": "set_bond_distance_from_initial",
            "optimize_initial_image": True,
            "optimize_final_image": True,
            "endpoint_opt_steps": 120,
            "endpoint_opt_fmax": 0.04,
            "break_bond_i": 0,
            "break_bond_j": 1,
            "set_bond_distance_A": 1.8,
            "break_bond_fix": 0.5,
            "break_assert_broken": True,
            "break_cutoff_scale": 1.2,
            "backend": "mace",
            "atom_shift_per_image_A": 0.2,
            "min_images": 7,
            "max_images": 21,
            "spring_constant": 0.1,
            "max_steps": 200,
            "climb": True,
            "split_on_minimum": True,
            "minimum_energy_window_eV": 0.02,
            "fixed_symbols": ("Au", "Cu"),
            "mace_model_path": "/srv/research/models/mace.model",
            "mace_device": "cuda",
            "mace_default_dtype": "float64",
            "mace_dispersion": False,
            "catlearn_min_data": 3,
            "catlearn_unc_convergence": 0.05,
            "catlearn_max_unc": 0.03,
            "catlearn_steps": 150,
            "catlearn_ml_steps": 400,
            "catlearn_dtrust": 2.0,
            "final_structure_metadata": {"source": "generated_from_initial_set_bond_distance"},
        },
        submitted_by="tester",
        structure_metadata={"source": "upload_file"},
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/home/user/reaction_hub_runs/neb_trh408_bond_extension_test"
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]

    assert f"{remote_dir}/input_structure.extxyz" in fake_client.sftp.files_written
    assert f"{remote_dir}/input_structure_final.extxyz" not in fake_client.sftp.files_written
    assert "final = None" in wrapper_script
    assert "final_image_strategy='set_bond_distance_from_initial'" in wrapper_script
    assert "break_bond_i=0" in wrapper_script
    assert "break_bond_j=1" in wrapper_script
    assert "set_bond_distance_A=1.8" in wrapper_script
    assert run.metadata["submission"]["neb_config"]["final_image_strategy"] == "set_bond_distance_from_initial"
    assert run.metadata["submission"]["neb_config"]["final_structure_path"] == ""


def test_structure_file_to_xyz_supports_xyz_traj_and_db(tmp_path: Path):
    xyz_upload = SimpleUploadedFile(
        "co.xyz",
        b"2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        content_type="chemical/x-xyz",
    )
    xyz_text, xyz_meta = structure_file_to_xyz(xyz_upload, traj_frame=None)
    assert "C" in xyz_text and "O" in xyz_text
    assert xyz_meta["formula"] == "CO"
    assert "cell_A" in xyz_meta
    assert "pbc" in xyz_meta
    assert isinstance(xyz_meta.get("structure_sha256"), str)
    assert len(xyz_meta["structure_sha256"]) == 64

    traj_path = tmp_path / "co.traj"
    frames = [Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])]
    write(traj_path, frames)
    traj_upload = SimpleUploadedFile(
        "co.traj",
        traj_path.read_bytes(),
        content_type="application/octet-stream",
    )
    traj_text, traj_meta = structure_file_to_xyz(traj_upload, traj_frame=-1)
    assert "C" in traj_text and "O" in traj_text
    assert traj_meta["source_suffix"] == ".traj"
    assert "cell_A" in traj_meta
    assert "pbc" in traj_meta
    assert isinstance(traj_meta.get("structure_sha256"), str)
    assert len(traj_meta["structure_sha256"]) == 64

    db_path = tmp_path / "co.db"
    with ase_connect(db_path) as db:
        db.write(
            Atoms(
                "CO",
                positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.2]],
                cell=[[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]],
                pbc=[True, True, True],
            )
        )
    db_upload = SimpleUploadedFile(
        "co.db",
        db_path.read_bytes(),
        content_type="application/octet-stream",
    )
    db_text, db_meta = structure_file_to_xyz(db_upload, traj_frame=-1)
    assert "C" in db_text and "O" in db_text
    assert db_meta["source_suffix"] == ".db"
    assert db_meta["pbc"] == [True, True, True]
    assert db_meta["cell_A"] == [[6.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, 0.0, 12.0]]
    assert isinstance(db_meta.get("structure_sha256"), str)
    assert len(db_meta["structure_sha256"]) == 64


@pytest.mark.parametrize("uploaded_name", ["POSCAR", "contcar", "my_CONTCAR_candidate.dat"])
def test_structure_file_to_xyz_forces_vasp_format_for_poscar_contcar_names(
    tmp_path: Path, uploaded_name: str
):
    vasp_path = tmp_path / "vasp_input"
    atoms = Atoms(
        "Cu2",
        positions=[[0.0, 0.0, 0.0], [1.5, 1.5, 1.5]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    write(vasp_path, atoms, format="vasp")

    upload = SimpleUploadedFile(
        uploaded_name,
        vasp_path.read_bytes(),
        content_type="application/octet-stream",
    )
    xyz_text, metadata = structure_file_to_xyz(upload, traj_frame=None)

    assert "Cu" in xyz_text
    assert metadata["formula"] == "Cu2"
    assert metadata.get("source_read_format") == "vasp"
    assert metadata.get("pbc") == [True, True, True]
    assert metadata.get("cell_A") == [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]


def test_structure_file_to_xyz_frames_reads_all_traj_frames(tmp_path: Path):
    traj_path = tmp_path / "co_scan.traj"
    frames = [
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]),
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]]),
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
    ]
    write(traj_path, frames)
    traj_upload = SimpleUploadedFile(
        "co_scan.traj",
        traj_path.read_bytes(),
        content_type="application/octet-stream",
    )

    payloads = structure_file_to_xyz_frames(traj_upload)

    assert len(payloads) == 3
    assert payloads[0][1]["frame_index"] == 0
    assert payloads[1][1]["frame_index"] == 1
    assert payloads[2][1]["frame_index"] == 2
    assert payloads[2][1]["frame_count"] == 3


def test_structure_file_to_xyz_frames_reads_all_db_rows(tmp_path: Path):
    db_path = tmp_path / "co_scan.db"
    frames = [
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]]),
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]]),
        Atoms("CO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.5]]),
    ]
    with ase_connect(db_path) as db:
        for atoms in frames:
            db.write(atoms)

    db_upload = SimpleUploadedFile(
        "co_scan.db",
        db_path.read_bytes(),
        content_type="application/octet-stream",
    )

    payloads = structure_file_to_xyz_frames(db_upload)

    assert len(payloads) == 3
    assert payloads[0][1]["frame_index"] == 0
    assert payloads[1][1]["frame_index"] == 1
    assert payloads[2][1]["frame_index"] == 2
    assert payloads[2][1]["frame_count"] == 3


@pytest.mark.django_db
def test_submit_tddft_mlneb_stages_manual_geometry_only_band():
    profile = CalculationSettingsProfile.objects.create(
        name="tddft_mlneb_band_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
        solvent="Water",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_tddft_mlneb_band",
        code_path="cats/working_directory/tddft_mlneb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-band",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()
    band = (
        "2\nimage 0\nC 0 0 0\nO 0 0 1.1\n"
        "2\nimage 1\nC 0 0 0\nO 0 0 1.3\n"
        "2\nimage 2\nC 0 0 0\nO 0 0 1.5\n"
    )

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="tddft_mlneb_manual_band_test",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="long",
        walltime="96:00:00",
        nodes=2,
        tasks_per_node=32,
        cpus_per_task=1,
        module_loads=["cray-python", "orca/6.0.0"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        n_iter=1,
        use_tddft_optimisation=True,
        run_mode="active_learning_neb",
        neb_config={
            "backend": "tddft_mlneb",
            "initial_band_extxyz": band,
            "tddft_scan_nroots": 50,
        },
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/tddft_mlneb_manual_band_test"
    wrapper = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]
    assert fake_client.sftp.files_written[f"{remote_dir}/input_band.extxyz"] == band.strip()
    assert f"{remote_dir}/input_structure_final.extxyz" not in fake_client.sftp.files_written
    assert 'initial_band = read("input_band.extxyz", index=":")' in wrapper
    assert 'initial = read("input_structure.extxyz")' not in wrapper
    assert "initial = None" in wrapper
    assert "initial_band_atoms=initial_band" in wrapper
    assert "final = None" in wrapper
    assert run.metadata["submission"]["neb_config"]["initial_band_path"] == "input_band.extxyz"


@pytest.mark.django_db
def test_submit_dft_mlneb_stages_manual_geometry_only_band():
    profile = CalculationSettingsProfile.objects.create(
        name="dft_mlneb_band_profile",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_dft_mlneb_band",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="archer2-dft-band",
        scheduler_type="slurm",
        host="example.org",
        username="user123",
    )
    fake_client = _FakeSSHClient()
    band = (
        "2\nimage 0\nC 0 0 0\nO 0 0 1.1\n"
        "2\nimage 1\nC 0 0 0\nO 0 0 1.3\n"
        "2\nimage 2\nC 0 0 0\nO 0 0 1.5\n"
    )

    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="dft_mlneb_manual_band_test",
        structure_xyz="2\nCO\nC 0 0 0\nO 0 0 1.1\n",
        remote_runs_root="/work/PROJECT/GROUP/{username}/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/GROUP",
        account="SLURM_ACCOUNT",
        partition="standard",
        qos="long",
        walltime="96:00:00",
        nodes=2,
        tasks_per_node=32,
        cpus_per_task=1,
        module_loads=["cray-python", "orca/6.0.0"],
        venv_activate_path="/work/PROJECT/GROUP/user123/venv/bin/activate",
        pythonpath_entries=["/work/PROJECT/GROUP/user123/ORCA/cplas"],
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="active_learning_neb",
        neb_config={
            "backend": "dft",
            "initial_band_extxyz": band,
            "catlearn_guesses_per_cycle": 5,
            "dft_evaluation_timeout_seconds": 43200,
        },
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/work/PROJECT/GROUP/user123/reaction_hub_runs/dft_mlneb_manual_band_test"
    wrapper = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]
    assert fake_client.sftp.files_written[f"{remote_dir}/input_band.extxyz"] == band.strip()
    assert f"{remote_dir}/input_structure_final.extxyz" not in fake_client.sftp.files_written
    assert 'initial_band = read("input_band.extxyz", index=":")' in wrapper
    assert 'initial = read("input_structure.extxyz")' not in wrapper
    assert "initial = None" in wrapper
    assert "run_active_learning_neb_workflow" in wrapper
    assert "initial_band_atoms=initial_band" in wrapper
    assert "backend='dft'" in wrapper
    assert "final = None" in wrapper
    assert run.metadata["submission"]["neb_config"]["initial_band_path"] == "input_band.extxyz"
    assert run.metadata["submission"]["neb_config"]["catlearn_guesses_per_cycle"] == 5
    assert run.metadata["submission"]["neb_config"]["dft_evaluation_timeout_seconds"] == 43200


@pytest.mark.django_db
def test_submit_slurm_workflow_run_falcon_small_system_uses_48_orca_cpus():
    profile = CalculationSettingsProfile.objects.create(
        name="falcon_cam_b3lyp",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_orca_falcon_48",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="falcon",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    # 20 atoms (<= 30 atoms threshold)
    xyz_20 = "20\nAu20 cluster\n" + "\n".join(f"Au {i} 0 0" for i in range(20)) + "\n"
    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="falcon_48_test",
        structure_xyz=xyz_20,
        remote_runs_root="/shared/home1/$USER/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/$USER",
        account="FALCON_SLURM_ACCOUNT",
        partition="htc_genoa",
        qos="",
        walltime="12:00:00",
        nodes=1,
        tasks_per_node=192,
        cpus_per_task=1,
        slurm_memory="720G",
        module_loads=["Python/3.11.5-GCCcore-13.2.0", "ORCA"],
        venv_activate_path="/scratch/PROJECT/shared/venv/mace_venv/bin/activate",
        pythonpath_entries=["/scratch/PROJECT/shared/software/github/cplas"],
        fmax=0.01,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        qm_backend="orca",
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/shared/home1/user123/reaction_hub_runs/falcon_48_test"
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]

    # Slurm reserves full 192-core node with 720G RAM
    assert "#SBATCH --nodes=1" in submit_script
    assert "#SBATCH --tasks-per-node=192" in submit_script
    assert "#SBATCH --mem=720G" in submit_script

    # Workflow environment exports 48 tasks and 70% memory (10500 MB/cpu) for <= 30 atoms
    assert "export WORKFLOW_TASKS_PER_NODE=48" in submit_script
    assert "export WORKFLOW_MEM_PER_CPU=10500" in submit_script
    assert run.metadata["submission"]["workflow_cpus_per_node"] == 48
    assert run.metadata["submission"]["workflow_total_cpus"] == 48
    assert run.metadata["submission"]["qm_backend_settings"]["memory_per_cpu_mb"] == 10500

    # Wrapper script defaults fallback to 48
    assert 'os.environ.get("WORKFLOW_TASKS_PER_NODE", 48)' in wrapper_script


@pytest.mark.django_db
def test_submit_slurm_workflow_run_falcon_large_system_uses_96_orca_cpus():
    profile = CalculationSettingsProfile.objects.create(
        name="falcon_cam_b3lyp_large",
        xc_functional="CAM-B3LYP",
        basis_set="LANL2DZ",
    )
    workflow = WorkflowDefinition.objects.create(
        name="wf_submit_orca_falcon_96",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    system = HPCSystem.objects.create(
        name="FALCON",
        scheduler_type="slurm",
        host="falcon",
        username="user123",
    )
    fake_client = _FakeSSHClient()

    # 55 atoms (> 30 atoms threshold)
    xyz_55 = "55\nAu55 cluster\n" + "\n".join(f"Au {i} 0 0" for i in range(55)) + "\n"
    result = submit_slurm_workflow_run(
        workflow=workflow,
        hpc_system=system,
        settings_profile=profile,
        run_name="falcon_96_test",
        structure_xyz=xyz_55,
        remote_runs_root="/shared/home1/$USER/reaction_hub_runs",
        scratch_base="/scratch/PROJECT/$USER",
        account="FALCON_SLURM_ACCOUNT",
        partition="htc_genoa",
        qos="",
        walltime="12:00:00",
        nodes=1,
        tasks_per_node=192,
        cpus_per_task=1,
        slurm_memory="720G",
        module_loads=["Python/3.11.5-GCCcore-13.2.0", "ORCA"],
        venv_activate_path="/scratch/PROJECT/shared/venv/mace_venv/bin/activate",
        pythonpath_entries=["/scratch/PROJECT/shared/software/github/cplas"],
        fmax=0.01,
        n_iter=1,
        use_tddft_optimisation=False,
        run_mode="single_optimisation",
        qm_backend="orca",
        submitted_by="tester",
        ssh_client_factory=lambda: fake_client,
    )

    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    remote_dir = "/shared/home1/user123/reaction_hub_runs/falcon_96_test"
    submit_script = fake_client.sftp.files_written[f"{remote_dir}/submit_workflow.sh"]
    wrapper_script = fake_client.sftp.files_written[f"{remote_dir}/run_workflow_submit.py"]

    # Slurm reserves full 192-core node with 720G RAM
    assert "#SBATCH --nodes=1" in submit_script
    assert "#SBATCH --tasks-per-node=192" in submit_script
    assert "#SBATCH --mem=720G" in submit_script

    # Workflow environment exports 96 tasks and 70% memory (5250 MB/cpu) for > 30 atoms
    assert "export WORKFLOW_TASKS_PER_NODE=96" in submit_script
    assert "export WORKFLOW_MEM_PER_CPU=5250" in submit_script
    assert run.metadata["submission"]["workflow_cpus_per_node"] == 96
    assert run.metadata["submission"]["workflow_total_cpus"] == 96
    assert run.metadata["submission"]["qm_backend_settings"]["memory_per_cpu_mb"] == 5250

    # Wrapper script defaults fallback to 96
    assert 'os.environ.get("WORKFLOW_TASKS_PER_NODE", 96)' in wrapper_script


