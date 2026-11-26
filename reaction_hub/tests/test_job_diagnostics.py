import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs import tasks
from jobs.models import HPCSystem, JobRecord
from references.models import CalculationSettingsProfile
from results.models import Artifact
from workflows.models import WorkflowDefinition, WorkflowRun


def test_job_diagnostics_rejects_commands_and_unbounded_tail():
    with pytest.raises(ValueError, match="unexpected=.*command"):
        tasks._validate_analysis_request("job_diagnostics", {"job_id": 1, "command": "id"})
    with pytest.raises(ValueError, match="between 1 and 500"):
        tasks._validate_analysis_request("job_diagnostics", {"job_id": 1, "tail_lines": 501})


def test_job_diagnostics_enqueue_uses_submissions_queue(monkeypatch):
    calls = []
    monkeypatch.setattr(
        tasks.submit_analysis_request_task,
        "apply_async",
        lambda **kwargs: calls.append(kwargs) or SimpleNamespace(id=kwargs["task_id"]),
    )
    result = tasks.enqueue_analysis_request("job_diagnostics", {"job_id": 7})
    assert result["status"] == "queued"
    assert calls[0]["args"] == ("job_diagnostics", {"job_id": 7})
    assert calls[0]["queue"] == "submissions"


def _job(name):
    profile = CalculationSettingsProfile.objects.create(name=name)
    definition = WorkflowDefinition.objects.create(name=name)
    run = WorkflowRun.objects.create(
        workflow=definition,
        settings_profile=profile,
        run_name="run",
        working_directory="/work/run",
    )
    system = HPCSystem.objects.create(name=name, host="example.org", username="user")
    return JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="12345",
        job_name="test",
        remote_workdir="/work/run/scratch",
        stdout_path="/work/run/slurm.out",
        stderr_path="/work/run/slurm.err",
    )


@pytest.mark.django_db
def test_job_diagnostics_uses_recorded_paths_and_caches_artifact(tmp_path, settings, monkeypatch):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str(tmp_path / "cache")
    job = _job("diagnostics")
    commands = []

    class Client:
        def exec_checked(self, command):
            commands.append(command)
            if command.startswith("realpath"):
                return command.split("-- ", 1)[1].strip("'\"") + "\n"
            if "squeue" in command:
                return "12345|RUNNING|00:01|00:20|node|test\n"
            if "sacct" in command:
                return "12345|RUNNING|00:01|00:20|0:0|None\n"
            return "bounded log\n"

        def close(self):
            pass

    monkeypatch.setattr("jobs.submission._default_ssh_client_factory", lambda _system: Client)
    result = tasks.submit_analysis_request_task.run(
        "job_diagnostics", {"job_id": job.pk, "tail_lines": 25}
    )

    assert result["status"] == "cached"
    assert any("tail -n 25" in command for command in commands)
    assert any("ml_time.txt" in command for command in commands)
    artifact = Artifact.objects.get(pk=result["artifact_id"])
    payload = json.loads(Path(artifact.metadata["local_cache_path"]).read_text(encoding="utf-8"))
    assert payload["job"]["scheduler_job_id"] == "12345"
    assert payload["logs"]["stdout"]["text"] == "bounded log\n"


@pytest.mark.django_db
def test_job_diagnostics_refuses_path_outside_job_directories(tmp_path, settings, monkeypatch):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str(tmp_path / "cache")
    job = _job("diagnostics-path")
    job.stdout_path = "/etc/passwd"
    job.save(update_fields=["stdout_path", "updated_at"])

    class Client:
        def exec_checked(self, command):
            if command.startswith("realpath"):
                return command.split("-- ", 1)[1].strip("'\"") + "\n"
            return ""

        def close(self):
            pass

    monkeypatch.setattr("jobs.submission._default_ssh_client_factory", lambda _system: Client)
    result = tasks.submit_analysis_request_task.run("job_diagnostics", {"job_id": job.pk})
    payload = json.loads(Path(result["local_cache_path"]).read_text(encoding="utf-8"))
    assert payload["logs"]["stdout"] == {
        "available": False,
        "reason": "recorded path is outside job directories",
    }


@pytest.mark.django_db
def test_job_diagnostics_uses_sacct_when_completed_job_is_absent_from_squeue(
    tmp_path, settings, monkeypatch
):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str(tmp_path / "cache")
    job = _job("diagnostics-completed")

    class Client:
        def exec_checked(self, command):
            if command.startswith("realpath"):
                return command.split("-- ", 1)[1].strip("'\"") + "\n"
            if "squeue" in command:
                raise RuntimeError("job no longer in squeue")
            if "sacct" in command:
                return "12345|COMPLETED|00:09:01|00:20:00|0:0|None\n"
            return ""

        def close(self):
            pass

    monkeypatch.setattr("jobs.submission._default_ssh_client_factory", lambda _system: Client)
    result = tasks.submit_analysis_request_task.run("job_diagnostics", {"job_id": job.pk})
    payload = json.loads(Path(result["local_cache_path"]).read_text(encoding="utf-8"))

    assert payload["scheduler"]["squeue"] == ""
    assert "COMPLETED|00:09:01" in payload["scheduler"]["sacct"]
