import io
from types import SimpleNamespace

import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from celery.exceptions import Retry
from jobs import tasks
from jobs.models import HPCSystem
from references.models import CalculationSettingsProfile
from results.models import Artifact, ComputedProperty
from workflows.models import WorkflowDefinition, WorkflowRun

@pytest.mark.django_db(transaction=True)
def test_enqueue_once_and_retry_when_short_queue_full(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="queue_mulliken")
    workflow = WorkflowDefinition.objects.create(name="queue_mulliken", code_path="x.py")
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="CuO")
    sent = []
    monkeypatch.setattr(tasks.submit_mulliken_recovery_task, "apply_async", lambda **kwargs: sent.append(kwargs))
    first = tasks.enqueue_mulliken_recovery(source.pk, "/analysis.csv")
    second = tasks.enqueue_mulliken_recovery(source.pk, "/analysis.csv")
    assert first["task_id"] == second["task_id"] and len(sent) == 1
    assert sent[0]["queue"] == "submissions"
    monkeypatch.setattr(tasks, "_queue_has_capacity", lambda qos: False)
    with pytest.raises(Retry):
        tasks.submit_mulliken_recovery_task.run(source.pk)
    source.refresh_from_db()
    assert source.metadata["mulliken_recovery_task"]["reason"] == "short_queue_full"


@pytest.mark.django_db(transaction=True)
def test_broker_failure_does_not_leave_a_false_queued_marker(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="queue_failed")
    workflow = WorkflowDefinition.objects.create(name="queue_failed", code_path="x.py")
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="CuO")
    def fail(**kwargs):
        raise ConnectionError("broker unavailable")
    monkeypatch.setattr(tasks.submit_mulliken_recovery_task, "apply_async", fail)
    with pytest.raises(ConnectionError):
        tasks.enqueue_mulliken_recovery(source.pk, "/analysis.csv")
    source.refresh_from_db()
    assert "mulliken_recovery_task" not in source.metadata


@pytest.mark.django_db
def test_full_ct_workup_is_frozen_dynamic_and_fragmented(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="full_ct")
    workflow = WorkflowDefinition.objects.create(name="full_ct", code_path="x.py")
    source = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="Au13_CO_O",
        working_directory="/work/runs/Au13_CO_O",
        metadata={"result_summary": {"final": {"opt_dir": "DFT/opt_iter_1"}}},
    )
    symbols = ["Au"] * 13 + ["C", "O", "O"]
    positions = [[float(i), 10.0, 0.0] for i in range(13)] + [
        [0.0, 0.0, 0.0],
        [1.1, 0.0, 0.0],
        [5.0, 0.0, 0.0],
    ]
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={"symbols": symbols, "positions_A": positions},
    )
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user")
    captured = []

    def submit(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(workflow_run_id=900, job_record_id=901, scheduler_job_id="902")

    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run", submit)
    tasks._submit_mulliken_recovery_run(source.pk)
    source.metadata["submission"] = {
        "partition": "standard",
        "qos": "reservation",
        "reservation": "test-reservation",
        "walltime": "12:00:00",
    }
    source.metadata["full_ct_resources"] = {"nodes": 2, "tasks_per_node": 64}
    source.save(update_fields=["metadata", "updated_at"])
    tasks._submit_full_ct_workup_run(source.pk)

    assert captured[0]["charge_transfer_config"]["source_working_directory"].endswith(
        "/DFT/opt_iter_1"
    )
    full = captured[1]
    assert full["run_mode"] == "single_optimisation"
    assert full["single_constraints_config"] == {
        "snapshot_single_point": True,
        "final_tddft_spectrum": True,
        "cluster_tddft_difference_spectrum": True,
        "final_tddft_nroots_mode": "scaled_by_atoms",
    }
    assert full["charge_transfer_config"]["fragments_0based"] == {
        "1": list(range(13)),
        "2": [13, 14],
        "3": [15],
    }
    assert full["qos"] == "reservation"
    assert full["reservation"] == "test-reservation"
    assert full["walltime"] == "12:00:00"
    assert (full["nodes"], full["tasks_per_node"]) == (2, 64)

    source.metadata["submission"]["single_constraints_config"] = {
        "final_tddft_gradient_root": 18, "final_tddft_nroots": 130,
    }
    source.save(update_fields=["metadata"])
    tasks._submit_mulliken_recovery_run(source.pk)
    assert captured[-1]["charge_transfer_config"]["mulliken_excitation_numbers"] == [18]
    assert captured[-1]["charge_transfer_config"]["nroots"] == 130
    assert captured[0]["charge_transfer_config"]["mulliken_excitation_numbers"] == []


def test_generic_analysis_request_is_allowlisted_and_routes_density(monkeypatch):
    sent = []
    monkeypatch.setattr(
        tasks.submit_manuscript_density_state_job_task,
        "apply_async",
        lambda **kwargs: sent.append(kwargs),
    )

    result = tasks.submit_analysis_request_task.run(
        "density_analysis",
        {"source_run_id": "46", "ct_run_id": 6543, "excitation_numbers": [8, 2, 8], "density_workers": 8, "nto_max_pairs": 3, "nto_target_coverage": 0.8},
    )

    assert result["status"] == "queued"
    assert result["excitation_numbers"] == [2, 8]
    assert sent[0]["args"] == [46, 6543, [2, 8]]
    assert sent[0]["kwargs"] == {"density_workers": 2, "nto_max_pairs": 3, "nto_target_coverage": 0.8}
    assert sent[0]["queue"] == "submissions"
    with pytest.raises(ValueError, match="Unsupported analysis operation"):
        tasks.submit_analysis_request_task.run("python", {"code": "do_anything()"})


@pytest.mark.django_db
def test_dft_mlneb_routes_guarded_failed_resubmit(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="retry_mlneb")
    workflow = WorkflowDefinition.objects.create(name="retry_mlneb", code_path="x.py")
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="retry_mlneb")
    hpc = HPCSystem.objects.create(name="retry_hpc", host="example.org", username="user")
    from jobs.models import JobRecord
    job = JobRecord.objects.create(
        workflow_run=source, hpc_system=hpc, scheduler_job_id="retry-1", job_name="retry", status="failed"
    )
    calls = []
    monkeypatch.setattr(
        "jobs.views._job_resubmit_failed_sync_response",
        lambda request, job_id, **kwargs: calls.append((job_id, kwargs)) or SimpleNamespace(status_code=302),
    )

    result = tasks._submit_dft_mlneb(source_run_id=source.pk, retry_job_id=job.pk)

    assert result["ok"] is True
    assert calls == [(job.pk, {"wait_for_queue_capacity": False, "raise_on_failure": True})]


@pytest.mark.django_db
def test_dft_mlneb_taskfarm_fans_out_then_reconstructs_training_band(monkeypatch, tmp_path):
    profile = CalculationSettingsProfile.objects.create(name="taskfarm_mlneb")
    workflow = WorkflowDefinition.objects.create(
        name="taskfarm_mlneb", code_path="cats/working_directory/active_learning_neb_workflow.py"
    )
    source = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="initial")
    final = WorkflowRun.objects.create(workflow=workflow, settings_profile=profile, run_name="final")
    for run, shift in ((source, 0.0), (final, 0.4)):
        ComputedProperty.objects.create(
            workflow_run=run,
            name="final_structure_snapshot",
            metadata={
                "symbols": ["Au", "Au", "Au"],
                "positions_A": [[0, 0, 0], [3, 0, 0], [0, 3, shift]],
            },
        )
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user")
    calls = []

    def submit(**kwargs):
        calls.append(kwargs)
        index = len(calls)
        if kwargs["submitted_by"] == "dft-mlneb-taskfarm":
            child = WorkflowRun.objects.create(
                workflow=workflow,
                settings_profile=profile,
                run_name=kwargs["run_name"],
                status="completed",
                submitted_by=kwargs["submitted_by"],
                metadata={"structure": kwargs["structure_metadata"]},
            )
            image = read(io.StringIO(kwargs["structure_xyz"]), format="extxyz")
            image.calc = SinglePointCalculator(
                image, energy=float(index), forces=np.zeros((len(image), 3))
            )
            cache_path = tmp_path / f"evaluated-{index}.traj"
            write(cache_path, image)
            Artifact.objects.create(
                workflow_run=child,
                label="evaluated trajectory",
                path="neb/evaluated.traj",
                metadata={"local_cache_path": str(cache_path)},
            )
            return SimpleNamespace(workflow_run_id=child.pk, job_record_id=200 + index, scheduler_job_id=str(300 + index))
        return SimpleNamespace(workflow_run_id=999, job_record_id=998, scheduler_job_id="997")

    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run", submit)

    result = tasks._submit_dft_mlneb(
        source_run_id=source.pk,
        final_run_id=final.pk,
        taskfarm_initial_band=True,
        taskfarm_images=5,
    )

    assert result["status"] == "training_submitted"
    assert len(calls) == 5
    assert [call["structure_metadata"]["image_index"] for call in calls] == list(range(5))
    assert all(call["nodes"] == 1 for call in calls)
    assert all(call["neb_config"] == {"backend": "dft", "evaluation_only": True} for call in calls)

    resumed = tasks._submit_dft_mlneb(
        source_run_id=source.pk,
        final_run_id=final.pk,
        taskfarm_initial_band=True,
        taskfarm_images=5,
    )

    assert resumed["status"] == "submitted"
    assert calls[-1]["submitted_by"] == "dft-mlneb"
    training = read(io.StringIO(calls[-1]["neb_config"]["initial_band_extxyz"]), index=":", format="extxyz")
    assert [image.get_potential_energy() for image in training] == [1, 2, 3, 4, 5]
    assert len(calls[-1]["structure_metadata"]["training_run_ids"]) == 5


@pytest.mark.django_db
def test_submit_full_ct_workup_run_falcon_small_system_48cpus_360g(monkeypatch):
    from types import SimpleNamespace
    from jobs.models import HPCSystem

    HPCSystem.objects.get_or_create(
        name="FALCON",
        defaults={
            "scheduler_type": "slurm",
            "host": "falconlogin.cf.ac.uk",
            "port": 22,
            "username": "tester",
        },
    )

    profile = CalculationSettingsProfile.objects.create(
        name="full_ct_falcon_prof",
        basis_set="def2-SVP",
        xc_functional="PBE0",
    )
    workflow = WorkflowDefinition.objects.create(
        name="full_ct_falcon_wf",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    xyz_lines = ["23", "test"] + [f"Au 0.0 0.0 {i * 1.5}" for i in range(20)] + [
        "N 0.0 0.0 31.0",
        "O 0.0 1.0 32.0",
        "O 0.0 -1.0 32.0",
    ]
    source = WorkflowRun.objects.create(
        workflow=workflow,
        settings_profile=profile,
        run_name="Au20_NO2_test_falcon",
        status="completed",
        metadata={
            "target_ct_hpc": "FALCON",
            "structure": {"final_structure_xyz": "\n".join(xyz_lines)},
        },
    )
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={
            "symbols": ["Au"] * 20 + ["N", "O", "O"],
            "positions_A": [[0.0, 0.0, float(i) * 1.5] for i in range(20)] + [
                [0.0, 0.0, 31.0],
                [0.0, 1.0, 32.0],
                [0.0, -1.0, 32.0],
            ],
        },
    )

    captured = []

    def fake_submit(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(workflow_run_id=999, job_record_id=1001, scheduler_job_id="810999")

    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run", fake_submit)

    tasks._submit_full_ct_workup_run(source.pk)

    assert len(captured) == 1
    call = captured[0]
    assert call["tasks_per_node"] == 48
    assert call["slurm_memory"] == "360G"
    assert call["workflow_tasks_per_node"] == 48
    assert call["partition"] == "htc_genoa"


@pytest.mark.django_db
def test_submit_full_ct_workup_run_mlneb_source_resolves_tddft_workflow_and_structure(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from jobs.models import HPCSystem

    HPCSystem.objects.get_or_create(
        name="FALCON",
        defaults={
            "scheduler_type": "slurm",
            "host": "falconlogin.cf.ac.uk",
            "port": 22,
            "username": "tester",
        },
    )

    profile = CalculationSettingsProfile.objects.create(
        name="full_ct_mlneb_prof",
        basis_set="def2-SVP",
        xc_functional="PBE0",
    )
    tddft_wf = WorkflowDefinition.objects.create(
        name="TDDFT Active Learning",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    mlneb_wf = WorkflowDefinition.objects.create(
        name="Active Learning NEB (MACE)",
        code_path="cats/working_directory/active_learning_neb_workflow.py",
    )
    source = WorkflowRun.objects.create(
        workflow=mlneb_wf,
        settings_profile=profile,
        run_name="mlneb_cu13_test",
        status="completed",
        metadata={
            "target_ct_hpc": "FALCON",
        },
    )
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={
            "symbols": ["Cu"] * 13 + ["C", "O", "O"],
            "positions_A": [[0.0, 0.0, float(i) * 1.5] for i in range(13)] + [
                [0.0, 0.0, 20.0],
                [0.0, 1.1, 20.0],
                [0.0, -1.1, 20.0],
            ],
        },
    )

    captured = []

    def fake_submit(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(workflow_run_id=888, job_record_id=889, scheduler_job_id="810888")

    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run", fake_submit)

    res = tasks._submit_full_ct_workup_run(source.pk)

    assert res["status"] == "submitted"
    assert len(captured) == 1
    call = captured[0]
    assert call["workflow"] == tddft_wf
    assert call["tasks_per_node"] == 48
    assert call["slurm_memory"] == "360G"
    assert call["workflow_tasks_per_node"] == 48
    assert call["partition"] == "htc_genoa"
    assert "1" in call["charge_transfer_config"]["fragments_0based"]
    assert "2" in call["charge_transfer_config"]["fragments_0based"]
