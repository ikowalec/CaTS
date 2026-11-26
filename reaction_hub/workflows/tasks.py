from __future__ import annotations

import json

from celery import shared_task
from django.test import RequestFactory


@shared_task
def render_artifact_xyz_task(run_id: int, artifact_id: int, payload: dict | None = None) -> dict:
    from .views import _workflow_render_artifact_xyz_sync_response

    body = json.dumps(payload or {}).encode("utf-8")
    request = RequestFactory().post(
        f"/workflows/{int(run_id)}/artifacts/{int(artifact_id)}/render-xyz/",
        data=body,
        content_type="application/json",
    )
    response = _workflow_render_artifact_xyz_sync_response(request, int(run_id), int(artifact_id))
    content = response.content.decode("utf-8", errors="replace")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"ok": False, "error": content[:500] or "Unexpected render response."}
    data["status_code"] = int(response.status_code)
    return data


@shared_task
def ingest_artifact_from_hpc_task(run_id: int, artifact_id: int) -> dict:
    from .views import _workflow_ingest_artifact_sync_response

    request = RequestFactory().post(f"/workflows/{int(run_id)}/artifacts/{int(artifact_id)}/ingest/")
    response = _workflow_ingest_artifact_sync_response(request, int(run_id), int(artifact_id))
    content = response.content.decode("utf-8", errors="replace")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"ok": False, "error": content[:500] or "Unexpected ingest response."}
    data["status_code"] = int(response.status_code)
    return data


def enqueue_artifact_ingest(run_id: int, artifact_id: int, queue: str = "submissions") -> dict:
    task = ingest_artifact_from_hpc_task.apply_async(args=[int(run_id), int(artifact_id)], queue=queue)
    return {"ok": True, "task_id": task.id, "queued": True}
