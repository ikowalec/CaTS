from __future__ import annotations

import uuid

from django.db import models

from core.models import TimeStampedModel


class WorkflowStatus(models.TextChoices):
    INITIATED = "initiated", "Initiated"
    RUNNING = "running", "Running"
    COMPLETED = "completed", "Completed"
    FAILED = "failed", "Failed"


class WorkflowDefinition(TimeStampedModel):
    name = models.CharField(max_length=120, unique=True)
    code_path = models.CharField(max_length=512, blank=True)
    description = models.TextField(blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class WorkflowRun(TimeStampedModel):
    workflow = models.ForeignKey(
        WorkflowDefinition,
        on_delete=models.CASCADE,
        related_name="runs",
    )
    run_uid = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    run_name = models.CharField(max_length=180)
    status = models.CharField(
        max_length=16,
        choices=WorkflowStatus.choices,
        default=WorkflowStatus.INITIATED,
        db_index=True,
    )
    started_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    submitted_by = models.CharField(max_length=128, blank=True)
    working_directory = models.CharField(max_length=512)
    system_charge = models.IntegerField(default=0)
    settings_profile = models.ForeignKey(
        "references.CalculationSettingsProfile",
        on_delete=models.PROTECT,
        related_name="workflow_runs",
    )
    reference_adsorbate = models.ForeignKey(
        "references.ReferenceAdsorbate",
        on_delete=models.PROTECT,
        related_name="workflow_runs",
        null=True,
        blank=True,
    )
    reference_nanocluster = models.ForeignKey(
        "references.ReferenceNanocluster",
        on_delete=models.PROTECT,
        related_name="workflow_runs",
        null=True,
        blank=True,
    )
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]
        indexes = [
            models.Index(fields=["status", "started_at"]),
            models.Index(fields=["status", "completed_at"]),
        ]

    def __str__(self) -> str:
        return f"{self.workflow.name}:{self.run_name}"


class CampaignStatus(models.TextChoices):
    DRAFT = "draft", "Draft"
    ACTIVE = "active", "Active"
    PAUSED = "paused", "Paused"
    COMPLETED = "completed", "Completed"
    ARCHIVED = "archived", "Archived"


class Campaign(TimeStampedModel):
    name = models.CharField(max_length=255, unique=True)
    slug = models.SlugField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    target_clusters = models.CharField(max_length=255, blank=True, help_text="CSV of target cluster formulas e.g. Au20, Cu20")
    reaction_scheme = models.JSONField(default=dict, blank=True, help_text="Dictionary of elementary steps and target reactions")
    status = models.CharField(
        max_length=32,
        choices=CampaignStatus.choices,
        default=CampaignStatus.ACTIVE,
        db_index=True,
    )
    max_nodes_quota = models.IntegerField(default=90, help_text="Maximum CPU nodes allowed on active reservation")
    gpu_concurrency_limit = models.IntegerField(default=1, help_text="Maximum active GPU Slurm jobs allowed")
    checkpoint_ledger = models.JSONField(default=list, blank=True, help_text="Array of checkpoint items with completion status")
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    @property
    def member_runs_count(self) -> int:
        return self.campaign_runs.count()


class CampaignRun(TimeStampedModel):
    campaign = models.ForeignKey(
        Campaign,
        on_delete=models.CASCADE,
        related_name="campaign_runs",
    )
    workflow_run = models.ForeignKey(
        WorkflowRun,
        on_delete=models.CASCADE,
        related_name="campaign_memberships",
    )
    checkpoint_stage = models.IntegerField(default=1, help_text="Checkpoint stage 1 to 6")
    reaction_step_key = models.CharField(max_length=64, blank=True, help_text="e.g. Step_A, Step_B, CO_O, O_O")
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["campaign", "checkpoint_stage", "-created_at"]
        unique_together = ("campaign", "workflow_run")

    def __str__(self) -> str:
        return f"{self.campaign.name} -> Run #{self.workflow_run_id} (Stage {self.checkpoint_stage})"
