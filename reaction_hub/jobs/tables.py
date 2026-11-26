from __future__ import annotations

import django_tables2 as tables

from .models import JobRecord


class JobRecordTable(tables.Table):
    scheduler_job_id = tables.LinkColumn("job-detail", args=[tables.A("pk")])
    hpc_system = tables.Column(verbose_name="HPC System")
    campaigns = tables.Column(empty_values=(), orderable=False)
    actions = tables.TemplateColumn(
        template_code="""
{% if record.status == "initiated" or record.status == "pending" or record.status == "running" %}
<form method="post" action="{% url 'job-scancel-delete' record.pk %}" style="margin:0;">
  {% csrf_token %}
  <button
    type="submit"
    onclick="return confirm('Scancel this job and rename remote run folder to *_to_delete?');"
  >
    Scancel + Delete
  </button>
</form>
{% elif record.status == "cancelled" %}
<form method="post" action="{% url 'job-restart-cancelled' record.pk %}" style="margin:0;">
  {% csrf_token %}
  <button
    type="submit"
    onclick="return confirm('Restart this cancelled job from the same run folder and submit a new scheduler job?');"
  >
    Restart In Place
  </button>
</form>
{% elif record.status == "failed" %}
<form method="post" action="{% url 'job-resubmit-failed' record.pk %}" style="margin:0;">
  {% csrf_token %}
  <button
    type="submit"
    onclick="return confirm('Resubmit this failed job from the same run folder and submit a new scheduler job?');"
  >
    Resubmit Failed
  </button>
</form>
{% elif record.status == "completed" and record.supports_replot_resubmit %}
<form method="post" action="{% url 'job-resubmit-replot' record.pk %}" style="margin:0;">
  {% csrf_token %}
  <button
    type="submit"
    onclick="return confirm('Resubmit this completed job with short queue settings (20m) to refresh plots/summary outputs?');"
  >
    Resubmit Replot
  </button>
</form>
{% else %}
<span class="muted">-</span>
{% endif %}
""",
        orderable=False,
        verbose_name="Actions",
    )

    def render_campaigns(self, record):
        return ", ".join(
            membership.campaign.name
            for membership in record.workflow_run.campaign_memberships.all()
        ) or "-"

    class Meta:
        model = JobRecord
        fields = (
            "scheduler_job_id",
            "job_name",
            "status",
            "workflow_run",
            "campaigns",
            "hpc_system",
            "remote_workdir",
            "submitted_at",
            "completed_at",
        )
        sequence = (
            "scheduler_job_id",
            "job_name",
            "status",
            "workflow_run",
            "campaigns",
            "hpc_system",
            "remote_workdir",
            "submitted_at",
            "completed_at",
            "actions",
        )
        attrs = {"class": "table table-striped table-sm"}
