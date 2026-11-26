from __future__ import annotations

from decimal import Decimal

from django.core.exceptions import ValidationError
from django.db import models

from core.models import TimeStampedModel


class ArtifactKind(models.TextChoices):
    STRUCTURE = "structure", "Structure"
    TRAJECTORY = "trajectory", "Trajectory"
    LOG = "log", "Log"
    CSV = "csv", "CSV"
    JSON = "json", "JSON"
    IMAGE = "image", "Image"
    DATABASE = "database", "Database"
    OTHER = "other", "Other"


class Artifact(TimeStampedModel):
    workflow_run = models.ForeignKey(
        "workflows.WorkflowRun",
        on_delete=models.CASCADE,
        related_name="artifacts",
    )
    job = models.ForeignKey(
        "jobs.JobRecord",
        on_delete=models.SET_NULL,
        related_name="artifacts",
        null=True,
        blank=True,
    )
    kind = models.CharField(max_length=20, choices=ArtifactKind.choices, default=ArtifactKind.OTHER)
    label = models.CharField(max_length=120)
    path = models.CharField(max_length=512)
    checksum = models.CharField(max_length=128, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["kind", "label"]
        constraints = [
            models.UniqueConstraint(
                fields=["workflow_run", "path"],
                name="results_unique_artifact_path_per_run",
            )
        ]

    def __str__(self) -> str:
        return f"{self.workflow_run.run_name}: {self.label}"


class ComputedProperty(TimeStampedModel):
    workflow_run = models.ForeignKey(
        "workflows.WorkflowRun",
        on_delete=models.CASCADE,
        related_name="computed_properties",
    )
    source_artifact = models.ForeignKey(
        Artifact,
        on_delete=models.SET_NULL,
        related_name="computed_properties",
        null=True,
        blank=True,
    )
    name = models.CharField(max_length=120)
    value_float = models.FloatField(null=True, blank=True)
    value_text = models.CharField(max_length=255, blank=True)
    unit = models.CharField(max_length=32, blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["name"]
        verbose_name = "Computed Property"
        verbose_name_plural = "Computed Properties"
        constraints = [
            models.UniqueConstraint(
                fields=["workflow_run", "name", "unit"],
                name="results_unique_property_name_unit_per_run",
            )
        ]

    def __str__(self) -> str:
        return f"{self.workflow_run.run_name}: {self.name}"


class AdsorptionEnergyResult(TimeStampedModel):
    workflow_run = models.OneToOneField(
        "workflows.WorkflowRun",
        on_delete=models.CASCADE,
        related_name="adsorption_energy",
    )
    adsorbate = models.ForeignKey(
        "references.ReferenceAdsorbate",
        on_delete=models.PROTECT,
        related_name="adsorption_results",
    )
    nanocluster = models.ForeignKey(
        "references.ReferenceNanocluster",
        on_delete=models.PROTECT,
        related_name="adsorption_results",
    )
    settings_profile = models.ForeignKey(
        "references.CalculationSettingsProfile",
        on_delete=models.PROTECT,
        related_name="adsorption_results",
    )
    total_energy_eV = models.DecimalField(max_digits=14, decimal_places=6)
    reference_energy_eV = models.DecimalField(max_digits=14, decimal_places=6)
    adsorption_energy_eV = models.DecimalField(max_digits=14, decimal_places=6)
    method_notes = models.TextField(blank=True)
    metadata = models.JSONField(default=dict, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.workflow_run.run_name}: {self.adsorption_energy_eV} eV"

    def clean(self):
        super().clean()
        run = self.workflow_run
        errors = {}
        if run.settings_profile_id != self.settings_profile_id:
            errors["settings_profile"] = (
                "Adsorption energy settings profile must match the workflow run settings profile."
            )
        if run.reference_adsorbate_id and run.reference_adsorbate_id != self.adsorbate_id:
            errors["adsorbate"] = "Adsorbate must match workflow run reference adsorbate."
        if run.reference_nanocluster_id and run.reference_nanocluster_id != self.nanocluster_id:
            errors["nanocluster"] = "Nanocluster must match workflow run reference nanocluster."
        if errors:
            raise ValidationError(errors)

    def save(self, *args, **kwargs):
        self.full_clean()
        super().save(*args, **kwargs)

    @classmethod
    def from_components(
        cls,
        *,
        workflow_run,
        adsorbate,
        nanocluster,
        settings_profile,
        total_energy_eV: Decimal,
        reference_energy_eV: Decimal,
        method_notes: str = "",
        metadata: dict | None = None,
    ) -> "AdsorptionEnergyResult":
        adsorption_energy_eV = total_energy_eV - reference_energy_eV
        return cls.objects.create(
            workflow_run=workflow_run,
            adsorbate=adsorbate,
            nanocluster=nanocluster,
            settings_profile=settings_profile,
            total_energy_eV=total_energy_eV,
            reference_energy_eV=reference_energy_eV,
            adsorption_energy_eV=adsorption_energy_eV,
            method_notes=method_notes,
            metadata=metadata or {},
        )
