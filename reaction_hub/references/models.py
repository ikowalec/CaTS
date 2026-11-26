from __future__ import annotations

from django.db import models

from core.models import TimeStampedModel


class CalculationSettingsProfile(TimeStampedModel):
    name = models.CharField(max_length=120, unique=True)
    approved = models.BooleanField(default=True)
    xc_functional = models.CharField(max_length=64, default="PBE0")
    basis_set = models.CharField(max_length=128, default="6-31g(d,p)")
    solvent = models.CharField(max_length=64, blank=True, default="Water")
    integration_grid = models.CharField(max_length=64, blank=True, default="DefGrid2")
    scf_convergence = models.CharField(max_length=64, blank=True, default="TightSCF")
    metadata = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class ReferenceAdsorbate(TimeStampedModel):
    name = models.CharField(max_length=120, unique=True)
    formula = models.CharField(max_length=64, blank=True)
    smiles = models.CharField(max_length=256, blank=True)
    charge = models.IntegerField(default=0)
    multiplicity = models.PositiveIntegerField(default=1)
    structure_path = models.CharField(max_length=512, blank=True)
    settings_profile = models.ForeignKey(
        CalculationSettingsProfile,
        on_delete=models.PROTECT,
        related_name="adsorbates",
    )
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name


class ReferenceNanocluster(TimeStampedModel):
    name = models.CharField(max_length=120, unique=True)
    composition = models.CharField(max_length=120)
    atom_count = models.PositiveIntegerField(null=True, blank=True)
    charge = models.IntegerField(default=0)
    multiplicity = models.PositiveIntegerField(default=1)
    structure_path = models.CharField(max_length=512, blank=True)
    settings_profile = models.ForeignKey(
        CalculationSettingsProfile,
        on_delete=models.PROTECT,
        related_name="nanoclusters",
    )
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name
