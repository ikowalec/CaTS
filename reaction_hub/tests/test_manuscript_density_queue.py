from celery.exceptions import Retry
import pytest

from jobs import tasks
from jobs.models import HPCSystem


def test_density_analysis_accepts_ifct_only():
    clean = tasks._validate_analysis_request(
        "density_analysis", {"source_run_id": 1, "ct_run_id": 2, "ifct_only": True}
    )
    assert clean["ifct_only"] is True


def test_density_reservation_uses_existing_dispatcher_and_bounded_options(monkeypatch):
    from unittest.mock import Mock

    payload = dict(source_run_id=9787, ct_run_id=9787, excitation_numbers=[5],
                   retain_cubes=True, qos="reservation", reservation="current", walltime="01:00:00")
    clean = tasks._validate_analysis_request("density_analysis", payload)
    assert clean["density_workers"] == 1
    for invalid in ["00:00:00", "25:00:00", "01:99:00", "$(command)"]:
        with pytest.raises(ValueError, match="walltime"):
            tasks._validate_analysis_request("density_analysis", {**payload, "walltime": invalid})
    monkeypatch.setattr(tasks, "_existing_density_run", lambda *args: None)
    submit = Mock(return_value={"workflow_run_id": 1})
    monkeypatch.setattr(tasks, "_submit_manuscript_density_run", submit)
    assert tasks._dispatch_analysis_request("density_analysis", payload) == {"workflow_run_id": 1}
    assert submit.call_args.kwargs == clean


def test_ifct_dependency_uses_database_job_and_afterok(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from workflows.models import WorkflowRun

    payload = {"source_run_id": 9757, "ct_run_id": 9759, "ifct_only": True, "dependency_job_id": 8495}
    assert tasks._validate_analysis_request("density_analysis", payload)["dependency_job_id"] == 8495
    for invalid in (0, True, "15627779"):
        with pytest.raises(ValueError, match="database job ID"):
            tasks._validate_analysis_request("density_analysis", {**payload, "dependency_job_id": invalid})
    with pytest.raises(ValueError, match="IFCT-only"):
        tasks._validate_analysis_request("density_analysis", {**payload, "ifct_only": False})

    source = SimpleNamespace(pk=9757, run_uid="snapshot", run_name="step22", workflow="orca",
                             settings_profile="matched", system_charge=0, metadata={})
    hpc = SimpleNamespace(name="FALCON")
    job = SimpleNamespace(pk=8495, status="running", scheduler_job_id="15627779", hpc_system=hpc,
                          remote_workdir="/work/optical")
    ct_run = SimpleNamespace(pk=9759, jobs=Mock(), working_directory="/work/optical", metadata={
        "submission": {"charge_transfer_config": {
            "fragments_0based": {"1": [0], "2": [1]}, "fragments_1based": {"1": "1", "2": "2"}}}})
    ct_run.jobs.get.return_value = job
    monkeypatch.setattr(WorkflowRun.objects, "select_related", lambda *args: SimpleNamespace(get=lambda **kw: source))
    monkeypatch.setattr(WorkflowRun.objects, "get", lambda **kw: ct_run)
    monkeypatch.setattr(HPCSystem.objects, "get", lambda **kw: hpc)
    monkeypatch.setattr(tasks, "_final_structure_xyz", lambda run: "2\n\nAu 0 0 0\nO 0 0 2\n")
    submit = Mock(return_value=SimpleNamespace(workflow_run_id=99, job_record_id=100, scheduler_job_id="123"))
    monkeypatch.setattr("jobs.submission.submit_slurm_workflow_run", submit)
    tasks._submit_manuscript_density_run(**payload)
    ct_run.jobs.get.assert_called_once_with(pk=8495)
    assert submit.call_args.kwargs["dependency_job_id"] == "15627779"
    assert submit.call_args.kwargs["charge_transfer_config"]["ifct_only"] is True
    job.status = "failed"
    with pytest.raises(ValueError, match="failed or been cancelled"):
        tasks._submit_manuscript_density_run(**payload)
    assert submit.call_count == 1


def test_manuscript_density_job_waits_for_a_serial_slot(monkeypatch):
    monkeypatch.setattr(tasks, "_existing_density_run", lambda _source_id, **_kwargs: None)
    monkeypatch.setattr(tasks, "_serial_queue_has_capacity", lambda: False)
    monkeypatch.setattr(
        tasks,
        "_submit_manuscript_density_run",
        lambda *_args: pytest.fail("submission must wait for serial capacity"),
    )

    with pytest.raises(Retry):
        tasks.submit_manuscript_density_job_task.run(2041, 3134)


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("states", "expected"),
    [(["PENDING"] * 9 + ["RUNNING"] * 3, True), (["PENDING"] * 10, False), (["RUNNING"] * 4, False)],
)
def test_serial_limits_are_ten_queued_and_four_running(monkeypatch, states, expected):
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user123")

    class Client:
        def exec_checked(self, _command):
            return "\n".join(f"serial|{state}" for state in states)

        def close(self):
            pass

    monkeypatch.setattr("jobs.submission._default_ssh_client_factory", lambda _hpc: Client)
    assert tasks._serial_queue_has_capacity() is expected


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (["taskfarm|PENDING"] * 96 + ["taskfarm|RUNNING"] * 31, "taskfarm"),
        (["taskfarm|PENDING"] * 96 + ["taskfarm|RUNNING"] * 32, "standard"),
        (
            ["taskfarm|PENDING"] * 128
            + ["standard|PENDING"] * 48
            + ["standard|RUNNING"] * 15,
            "standard",
        ),
        (
            ["taskfarm|PENDING"] * 128
            + ["standard|PENDING"] * 64
            + ["serial|RUNNING"] * 4
            + ["serial|PENDING"] * 9,
            "serial",
        ),
        (
            ["taskfarm|PENDING"] * 128
            + ["standard|PENDING"] * 64
            + ["serial|RUNNING"] * 4
            + ["serial|PENDING"] * 10,
            None,
        ),
    ],
)
def test_density_qos_priority_and_limits(monkeypatch, rows, expected):
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user123")

    class Client:
        def exec_checked(self, _command):
            return "\n".join(rows)

        def close(self):
            pass

    monkeypatch.setattr("jobs.submission._default_ssh_client_factory", lambda _hpc: Client)
    assert tasks._next_density_qos() == expected


def test_resolve_density_source_directory_full_spectrum():
    class MockCTRun:
        working_directory = "/work/persistent/runs/ct_run"
        metadata = {
            "result_summary": {
                "workflow_root": "/mnt/lustre/scratch/14847118",
                "artifacts": [
                    "DFT/final_tddft_spectrum/orca.out",
                    "DFT/final_tddft_spectrum/orca.molden.input",
                ],
            }
        }

    resolved = tasks._resolve_density_source_directory(MockCTRun())
    assert resolved == "/mnt/lustre/scratch/14847118"


def test_resolve_density_source_directory_root_level():
    class MockCTRun:
        working_directory = "/work/persistent/runs/legacy_ct_run"
        metadata = {
            "result_summary": {
                "workflow_root": None,
                "artifacts": ["orca.out", "orca.molden.input"],
            }
        }

    resolved = tasks._resolve_density_source_directory(MockCTRun())
    assert resolved == "/work/persistent/runs/legacy_ct_run"


def test_resolve_density_source_directory_fallback_job_remote_workdir():
    class MockJob:
        remote_workdir = "/mnt/lustre/scratch/fallback_job"

    class MockCTRun:
        working_directory = ""
        metadata = {}

    resolved = tasks._resolve_density_source_directory(MockCTRun(), ct_job=MockJob())
    assert resolved == "/mnt/lustre/scratch/fallback_job"
