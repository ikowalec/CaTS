from __future__ import annotations

import django_tables2 as tables
from django.utils.formats import date_format

from .models import WorkflowRun


class WorkflowRunTable(tables.Table):
    run_name = tables.LinkColumn("workflow-run-detail", args=[tables.A("pk")])
    settings_profile = tables.Column(verbose_name="Settings Profile", empty_values=())
    started_at = tables.DateTimeColumn(verbose_name="Started", format="Y-m-d H:i:s", empty_values=())
    completed_at = tables.DateTimeColumn(verbose_name="Completed", format="Y-m-d H:i:s", empty_values=())
    created_at = tables.DateTimeColumn(verbose_name="Created", format="Y-m-d H:i:s")

    @staticmethod
    def _fallback_job_timestamp(record: WorkflowRun, attr_name: str):
        jobs = getattr(record, "jobs", None)
        if jobs is None:
            return None
        try:
            iterable = list(jobs.all())
        except Exception:
            return None
        candidates = [getattr(job, attr_name, None) for job in iterable]
        valid = [value for value in candidates if value is not None]
        if not valid:
            return None
        return min(valid) if attr_name == "started_at" else max(valid)

    def render_started_at(self, value, record: WorkflowRun):
        timestamp = value or self._fallback_job_timestamp(record, "started_at")
        return date_format(timestamp, "Y-m-d H:i:s") if timestamp else "-"

    def render_completed_at(self, value, record: WorkflowRun):
        timestamp = value or self._fallback_job_timestamp(record, "completed_at")
        return date_format(timestamp, "Y-m-d H:i:s") if timestamp else "-"

    @staticmethod
    def render_settings_profile(value, record: WorkflowRun):
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        submission = metadata.get("submission") if isinstance(metadata, dict) else None
        if isinstance(submission, dict):
            profile_name = str(submission.get("settings_profile_name") or "").strip()
            if profile_name:
                return profile_name

        profile = getattr(record, "settings_profile", None)
        if profile is None:
            return "-"
        return profile.name

    class Meta:
        model = WorkflowRun
        fields = (
            "run_name",
            "workflow",
            "status",
            "settings_profile",
            "created_at",
            "started_at",
            "completed_at",
        )
        attrs = {"class": "table table-striped table-sm"}
