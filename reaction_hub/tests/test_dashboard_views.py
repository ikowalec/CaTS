from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.mark.django_db
def test_dashboard_quick_access_shows_sync_hpc_jobs_button(client):
    resp = client.get("/")

    assert resp.status_code == 200
    content = resp.content.decode("utf-8", errors="replace")
    assert 'id="sync-hpc-jobs-form"' in content
    assert 'id="sync-hpc-jobs-button"' in content
    assert "Sync HPC Jobs" in content


@pytest.mark.django_db
def test_dashboard_sync_hpc_jobs_executes_sync(client, monkeypatch):
    calls = []

    def fake_sync_hpc_jobs():
        calls.append("sync")
        return {"checked": 10, "updated": 2}

    monkeypatch.setattr("ingest.services.sync_hpc_jobs", fake_sync_hpc_jobs)

    resp = client.post("/sync-hpc-jobs/", follow=True)

    assert resp.status_code == 200
    assert resp.redirect_chain == [("/", 302)]
    assert calls == ["sync"]
    content = resp.content.decode("utf-8", errors="replace")
    assert "HPC jobs synced successfully! Checked 10 jobs, updated 2." in content


@pytest.mark.django_db
def test_dashboard_sync_hpc_jobs_reports_failure(client, monkeypatch):
    calls = []

    def fake_sync_hpc_jobs():
        calls.append("sync")
        raise RuntimeError("sync connection failed")

    monkeypatch.setattr("ingest.services.sync_hpc_jobs", fake_sync_hpc_jobs)

    resp = client.post("/sync-hpc-jobs/", follow=True)

    assert resp.status_code == 200
    assert resp.redirect_chain == [("/", 302)]
    assert calls == ["sync"]
    content = resp.content.decode("utf-8", errors="replace")
    assert "HPC sync error: sync connection failed" in content


@pytest.mark.django_db
def test_dashboard_sync_hpc_jobs_handles_get_request(client, monkeypatch):
    calls = []

    def fake_sync_hpc_jobs():
        calls.append("sync")
        return {"checked": 5, "updated": 1}

    monkeypatch.setattr("ingest.services.sync_hpc_jobs", fake_sync_hpc_jobs)

    resp = client.get("/sync-hpc-jobs/", follow=True)

    assert resp.status_code == 200
    assert resp.redirect_chain == [("/", 302)]
    assert calls == ["sync"]
    content = resp.content.decode("utf-8", errors="replace")
    assert "HPC jobs synced successfully! Checked 5 jobs, updated 1." in content


