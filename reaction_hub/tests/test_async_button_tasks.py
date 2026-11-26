from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.django_db
def test_workflow_render_artifact_queues_celery_task(client, monkeypatch):
    calls = []

    def fake_delay(run_id, artifact_id, payload):
        calls.append((run_id, artifact_id, payload))
        return SimpleNamespace(id="render-task-1")

    monkeypatch.setattr("workflows.views.render_artifact_xyz_task.delay", fake_delay)

    resp = client.post(
        "/workflows/10/artifacts/20/render-xyz/",
        data=json.dumps({"gif_rot": "xy", "gif_mode": "trajectory", "render_output": "gif"}),
        content_type="application/json",
    )

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert payload["task_id"] == "render-task-1"
    assert calls == [(10, 20, {"gif_rot": "xy", "gif_mode": "trajectory", "render_output": "gif"})]


@pytest.mark.django_db
def test_job_restart_cancelled_queues_celery_task(client, monkeypatch):
    calls = []

    def fake_delay(job_id, *, requested_by="", next_url=""):
        calls.append((job_id, requested_by, next_url))
        return SimpleNamespace(id="restart-task-1")

    monkeypatch.setattr("jobs.views.restart_cancelled_job_task.delay", fake_delay)

    resp = client.post("/jobs/42/restart-cancelled/", data={"next": "/jobs/"}, follow=True)

    assert resp.status_code == 200
    assert resp.redirect_chain == [("/jobs/", 302)]
    assert calls == [(42, "", "/jobs/")]
    assert "Cancelled-job restart queued as Celery task restart-task-1." in resp.content.decode(
        "utf-8",
        errors="replace",
    )


@pytest.mark.django_db
@pytest.mark.parametrize(
    ("url", "patch_path", "expected_message"),
    [
        (
            "/jobs/43/resubmit-failed/",
            "jobs.views.resubmit_failed_job_task.delay",
            "Failed-job resubmit queued as Celery task task-1.",
        ),
        (
            "/jobs/44/resubmit-smooth/",
            "jobs.views.resubmit_smooth_job_task.delay",
            "Smoothing resubmit queued as Celery task task-1.",
        ),
        (
            "/jobs/45/resubmit-replot/",
            "jobs.views.resubmit_replot_job_task.delay",
            "Replot resubmit queued as Celery task task-1.",
        ),
    ],
)
def test_job_resubmit_buttons_queue_celery_tasks(client, monkeypatch, url, patch_path, expected_message):
    calls = []

    def fake_delay(job_id, *, requested_by="", next_url=""):
        calls.append((job_id, requested_by, next_url))
        return SimpleNamespace(id="task-1")

    monkeypatch.setattr(patch_path, fake_delay)

    resp = client.post(url, data={"next": "/jobs/"}, follow=True)

    assert resp.status_code == 200
    assert resp.redirect_chain == [("/jobs/", 302)]
    assert calls == [(int(url.split("/")[2]), "", "/jobs/")]
    assert expected_message in resp.content.decode("utf-8", errors="replace")


@pytest.mark.django_db
@pytest.mark.parametrize("query", ["", "?sync=1"])
def test_workflow_ingest_artifact_queues_celery_task(client, monkeypatch, query):
    from references.models import CalculationSettingsProfile
    from results.models import Artifact, ArtifactKind
    from workflows.models import WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(name="prof-1")
    definition = WorkflowDefinition.objects.create(name="test-def")
    run = WorkflowRun.objects.create(workflow=definition, settings_profile=profile, run_name="run_001")
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind=ArtifactKind.LOG,
        label="orca.out",
        path="orca.out",
        metadata={"remote_only": True, "remote_path": "/work/orca.out"},
    )
    calls = []

    def fake_apply_async(args, queue):
        calls.append((args, queue))
        return SimpleNamespace(id="ingest-task-1")

    monkeypatch.setattr("workflows.views.ingest_artifact_from_hpc_task.apply_async", fake_apply_async)

    resp = client.post(f"/workflows/{run.pk}/artifacts/{artifact.pk}/ingest/{query}")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert payload["task_id"] == "ingest-task-1"
    assert calls == [([run.pk, artifact.pk], "submissions")]


@pytest.mark.django_db
def test_results_artifact_ingest_delegates(client, monkeypatch):
    from references.models import CalculationSettingsProfile
    from results.models import Artifact, ArtifactKind
    from workflows.models import WorkflowDefinition, WorkflowRun

    profile = CalculationSettingsProfile.objects.create(name="prof-2")
    definition = WorkflowDefinition.objects.create(name="test-def-2")
    run = WorkflowRun.objects.create(workflow=definition, settings_profile=profile, run_name="run_002")
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind=ArtifactKind.LOG,
        label="orca.out",
        path="orca.out",
        metadata={"remote_only": True, "remote_path": "/work/orca.out"},
    )
    calls = []

    def fake_apply_async(args, queue):
        calls.append((args, queue))
        return SimpleNamespace(id="ingest-task-2")

    monkeypatch.setattr("workflows.views.ingest_artifact_from_hpc_task.apply_async", fake_apply_async)

    resp = client.post(f"/results/artifact/{artifact.pk}/ingest/")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["ok"] is True
    assert payload["queued"] is True
    assert payload["task_id"] == "ingest-task-2"
    assert calls == [([run.pk, artifact.pk], "submissions")]


@pytest.mark.django_db
def test_workflow_ingest_artifact_sync_download(client, tmp_path, settings, monkeypatch):
    from jobs.models import HPCSystem
    from references.models import CalculationSettingsProfile
    from results.models import Artifact, ArtifactKind
    from workflows.models import WorkflowDefinition, WorkflowRun
    from workflows.tasks import ingest_artifact_from_hpc_task

    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str(tmp_path / "cache")
    profile = CalculationSettingsProfile.objects.create(name="prof-3")
    definition = WorkflowDefinition.objects.create(name="test-def-3")
    run = WorkflowRun.objects.create(
        workflow=definition,
        settings_profile=profile,
        run_name="run_003",
        working_directory="/work/run3",
    )
    HPCSystem.objects.create(
        name="ARCHER2_TEST",
        host="hpc.example.invalid",
        username="testuser",
        is_active=True,
    )
    artifact = Artifact.objects.create(
        workflow_run=run,
        kind=ArtifactKind.LOG,
        label="orca.out",
        path="orca.out",
        metadata={"remote_only": True, "remote_path": "/work/run3/orca.out"},
    )

    class MockSSH:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

        def exec_checked(self, cmd):
            return "1"

    monkeypatch.setattr("core.openssh.OpenSSHClient", MockSSH)
    monkeypatch.setattr("workflows.views._fetch_remote_artifact_bytes", lambda client, path: b"DUMMY_ORCA_OUTPUT_DATA")

    payload = ingest_artifact_from_hpc_task.run(run.pk, artifact.pk)
    assert payload["status_code"] == 200
    assert payload["ok"] is True
    assert payload["bytes"] == len(b"DUMMY_ORCA_OUTPUT_DATA")

    artifact.refresh_from_db()
    assert artifact.metadata.get("remote_only") is False
    assert artifact.metadata.get("downloaded_from_hpc") is True
    local_path = artifact.metadata.get("local_cache_path")
    assert local_path is not None
    assert Path(local_path).read_bytes() == b"DUMMY_ORCA_OUTPUT_DATA"

    task_res = ingest_artifact_from_hpc_task(run.pk, artifact.pk)
    assert task_res["ok"] is True
    assert task_res["already_cached"] is True
