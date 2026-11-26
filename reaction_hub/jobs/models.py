from __future__ import annotations

from django.db import models
from django.utils import timezone

from core.models import TimeStampedModel

_REPLOT_ALLOWED_RUN_MODES = {"", "single_optimisation", "tddft_bond_scan"}


def _run_mode_from_metadata(run_metadata: object) -> str:
    metadata = run_metadata if isinstance(run_metadata, dict) else {}
    submission = metadata.get("submission")
    if not isinstance(submission, dict):
        return ""
    return str(submission.get("run_mode") or "").strip()


def _supports_replot_resubmit_for_run(workflow_run) -> bool:
    run_mode = _run_mode_from_metadata(getattr(workflow_run, "metadata", {}))
    if run_mode not in _REPLOT_ALLOWED_RUN_MODES:
        return False
    if run_mode:
        return True
    # Legacy runs may not carry explicit run_mode metadata.
    code_path = str(getattr(getattr(workflow_run, "workflow", None), "code_path", "") or "").strip().lower()
    return "coadsorbate" not in code_path


class SchedulerType(models.TextChoices):
    SLURM = "slurm", "Slurm"


class JobStatus(models.TextChoices):
    INITIATED = "initiated", "Initiated"
    PENDING = "pending", "Pending"
    RUNNING = "running", "Running"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"
    CANCELLED = "cancelled", "Cancelled"
    DELETED = "deleted", "Deleted"
    UNKNOWN = "unknown", "Unknown"


FINAL_JOB_STATES = {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.DELETED}


class HPCSystem(TimeStampedModel):
    name = models.CharField(max_length=120, unique=True)
    scheduler_type = models.CharField(
        max_length=16,
        choices=SchedulerType.choices,
        default=SchedulerType.SLURM,
    )
    host = models.CharField(max_length=255)
    port = models.PositiveIntegerField(default=22)
    username = models.CharField(max_length=120)
    ssh_key_path = models.CharField(max_length=512, blank=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class JobRecord(TimeStampedModel):
    workflow_run = models.ForeignKey(
        "workflows.WorkflowRun",
        on_delete=models.CASCADE,
        related_name="jobs",
    )
    hpc_system = models.ForeignKey(
        HPCSystem,
        on_delete=models.PROTECT,
        related_name="jobs",
        null=True,
        blank=True,
    )
    scheduler_job_id = models.CharField(max_length=64, db_index=True)
    job_name = models.CharField(max_length=180)
    status = models.CharField(
        max_length=16,
        choices=JobStatus.choices,
        default=JobStatus.INITIATED,
        db_index=True,
    )
    submitted_at = models.DateTimeField(null=True, blank=True)
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    last_seen_at = models.DateTimeField(null=True, blank=True)
    remote_workdir = models.CharField(max_length=512, blank=True)
    stdout_path = models.CharField(max_length=512, blank=True)
    stderr_path = models.CharField(max_length=512, blank=True)
    exit_code = models.IntegerField(null=True, blank=True)
    scheduler_payload = models.JSONField(default=dict, blank=True)
    failure_reason = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.UniqueConstraint(
                fields=["hpc_system", "scheduler_job_id"],
                name="jobs_unique_scheduler_id_per_system",
            )
        ]

    def __str__(self) -> str:
        return f"{self.job_name} ({self.scheduler_job_id})"

    @property
    def is_terminal(self) -> bool:
        return self.status in FINAL_JOB_STATES

    @property
    def run_mode(self) -> str:
        return _run_mode_from_metadata(getattr(self.workflow_run, "metadata", {}))

    @property
    def supports_replot_resubmit(self) -> bool:
        return _supports_replot_resubmit_for_run(self.workflow_run)

    def transition_to(self, new_status: str, *, message: str = "", payload: dict | None = None) -> bool:
        if new_status == self.status:
            return False
        previous_status = self.status
        self.status = new_status
        now = timezone.now()
        self.last_seen_at = now
        if new_status == JobStatus.RUNNING and self.started_at is None:
            self.started_at = now
        if new_status in FINAL_JOB_STATES and self.completed_at is None:
            self.completed_at = now
        self.save(update_fields=["status", "last_seen_at", "started_at", "completed_at", "updated_at"])
        JobStatusEvent.objects.create(
            job=self,
            previous_status=previous_status,
            status=new_status,
            message=message,
            payload=payload or {},
        )
        return True


class JobStatusEvent(TimeStampedModel):
    job = models.ForeignKey(JobRecord, on_delete=models.CASCADE, related_name="events")
    previous_status = models.CharField(max_length=16, blank=True)
    status = models.CharField(max_length=16, choices=JobStatus.choices)
    message = models.CharField(max_length=255, blank=True)
    payload = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Scheduler Status Event"
        verbose_name_plural = "Scheduler Status Events"

    def __str__(self) -> str:
        return f"{self.job.scheduler_job_id}: {self.previous_status} -> {self.status}"
