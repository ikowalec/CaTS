from __future__ import annotations

from pathlib import Path

from celery import shared_task
from django.conf import settings
from reaction_hub.celery import SingleFlightTask

from .services import ingest_workflow_outputs


@shared_task(base=SingleFlightTask)
def ingest_workflow_outputs_task(workflow_root: str | None = None) -> dict:
    root = Path(workflow_root) if workflow_root else Path(settings.REACTION_HUB_INGEST_ROOT)
    return ingest_workflow_outputs(root)


@shared_task
def backfill_ct_plots_task(charge_transfer_run_id: int) -> dict:
    from results.models import Artifact
    from workflows.models import WorkflowRun

    from .services import _ingest_summary_payload, _parse_json_payload
    from .spectral import write_ct_plot_artifacts

    ct_run = WorkflowRun.objects.get(pk=int(charge_transfer_run_id))
    summary_artifact = Artifact.objects.filter(
        workflow_run=ct_run,
        label="reaction_hub_summary",
    ).first()
    summary = {}
    if summary_artifact is not None:
        local_path = Path(str((summary_artifact.metadata or {}).get("local_cache_path") or ""))
        if local_path.is_file():
            summary = _parse_json_payload(local_path.read_text(encoding="utf-8")) or {}
    if not summary:
        summary = dict((ct_run.metadata or {}).get("result_summary") or {})
    if not isinstance(summary.get("charge_transfer"), dict):
        sub_cfg = (ct_run.metadata or {}).get("submission", {}).get("charge_transfer_config", {})
        if (
            bool(sub_cfg.get("density_analysis_only"))
            or summary.get("run_type") in ("density_analysis", "ts_vibrations_gradient")
        ):
            return {
                "charge_transfer_run_id": ct_run.pk,
                "source_run_id": int(summary.get("source_run_id") or ct_run.pk),
                "artifact_ids": [],
                "skipped": "density_analysis_only",
            }
        raise ValueError(f"Workflow run {ct_run.pk} has no retained charge-transfer summary.")
    if (
        str(summary["charge_transfer"].get("analysis_scope") or "").strip().lower()
        == "ground_state_mulliken"
    ):
        return {
            "charge_transfer_run_id": ct_run.pk,
            "source_run_id": int(summary.get("source_run_id") or ct_run.pk),
            "artifact_ids": [],
            "skipped": "ground_state_mulliken_only",
        }
    source_run_id = int(summary.get("source_run_id") or ct_run.pk)
    source_run = WorkflowRun.objects.get(pk=source_run_id)
    _ingest_summary_payload(run=ct_run, summary=summary, source_artifact=summary_artifact)
    artifacts = write_ct_plot_artifacts(source_run)
    return {
        "charge_transfer_run_id": ct_run.pk,
        "source_run_id": source_run.pk,
        "artifact_ids": [artifact.pk for artifact in artifacts],
    }
