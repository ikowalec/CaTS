from types import SimpleNamespace
import pytest

from jobs import tasks
from jobs.models import HPCSystem
from references.models import CalculationSettingsProfile
from results.models import ComputedProperty
from workflows.models import WorkflowDefinition, WorkflowRun


@pytest.mark.django_db
def test_tddft_mlneb_submission_with_aligned_endpoints(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="tddft-mlneb-test-profile")
    wf_mlneb = WorkflowDefinition.objects.create(
        name="TDDFT MLNEB",
        code_path="cats/working_directory/tddft_mlneb_workflow.py",
    )
    source = WorkflowRun.objects.create(
        workflow=wf_mlneb,
        settings_profile=profile,
        run_name="Au20_O2_initial",
    )
    final = WorkflowRun.objects.create(
        workflow=wf_mlneb,
        settings_profile=profile,
        run_name="Au20_O_O_final",
    )
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={
            "symbols": ["Au", "Au", "Au", "O", "O"],
            "positions_A": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [2.2, 1.0, 0.0],
            ],
        },
    )
    ComputedProperty.objects.create(
        workflow_run=final,
        name="final_structure_snapshot",
        metadata={
            "symbols": ["Au", "Au", "Au", "O", "O"],
            "positions_A": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
                [4.5, 1.0, 0.0],
            ],
        },
    )
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user", is_active=True)

    captured = []
    monkeypatch.setattr(
        "jobs.submission.submit_slurm_workflow_run",
        lambda **kwargs: captured.append(kwargs)
        or SimpleNamespace(workflow_run_id=99, job_record_id=100, scheduler_job_id="15999999"),
    )

    result = tasks.submit_analysis_request_task.run(
        "tddft_mlneb",
        {
            "source_run_id": source.pk,
            "final_run_id": final.pk,
            "target_wavelength_nm": 611.4,
            "nroots": 20,
            "qos": "long",
        },
    )

    assert result["scheduler_job_id"] == "15999999"
    assert len(captured) == 1
    job = captured[0]
    assert job["workflow"] == wf_mlneb
    assert job["target_wavelength_nm"] == 611.4
    assert job["qos"] == "long"
    assert job["partition"] == "standard"
    assert job["walltime"] == "96:00:00"
    assert job["neb_config"]["backend"] == "tddft_mlneb"
    assert job["neb_config"]["final_image_strategy"] == "explicit_final_image"
    assert job["neb_config"]["tddft_scan_nroots"] == 20
    assert "input_structure_final.extxyz" in job["neb_config"]["final_structure_path"]
    assert job["neb_config"]["final_structure_extxyz"] != ""


def test_enqueue_tddft_mlneb_routes_to_submissions_queue(monkeypatch):
    captured = []
    monkeypatch.setattr(
        tasks.submit_analysis_request_task,
        "apply_async",
        lambda *args, **kwargs: captured.append((args, kwargs))
        or SimpleNamespace(id=kwargs.get("task_id", "celery-task-1234")),
    )

    res = tasks.enqueue_tddft_mlneb(
        source_run_id=3391,
        target_wavelength_nm=611.4,
        final_run_id=3341,
        nroots=20,
        qos="long",
    )

    assert res["status"] == "queued"
    assert res["task_id"]
    assert res["operation"] == "tddft_mlneb"
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert kwargs["queue"] == "submissions"
    assert kwargs["task_id"] == res["task_id"]
    assert kwargs["args"] == (
        "tddft_mlneb",
        {
            "source_run_id": 3391,
            "final_run_id": 3341,
            "target_wavelength_nm": 611.4,
            "nroots": 20,
            "qos": "long",
        },
    )

