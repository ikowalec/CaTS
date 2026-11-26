from django.contrib import admin

from .models import AdsorptionEnergyResult, Artifact, ComputedProperty


@admin.register(Artifact)
class ArtifactAdmin(admin.ModelAdmin):
    list_display = ("label", "kind", "workflow_run", "path", "updated_at")
    list_filter = ("kind",)
    search_fields = ("label", "path", "workflow_run__run_name")


@admin.register(ComputedProperty)
class ComputedPropertyAdmin(admin.ModelAdmin):
    list_display = ("name", "value_float", "unit", "workflow_run")
    list_filter = ("name", "unit")
    search_fields = ("name", "workflow_run__run_name")


@admin.register(AdsorptionEnergyResult)
class AdsorptionEnergyResultAdmin(admin.ModelAdmin):
    list_display = (
        "workflow_run",
        "adsorbate",
        "nanocluster",
        "settings_profile",
        "adsorption_energy_eV",
    )
    list_filter = ("settings_profile", "adsorbate", "nanocluster")
    search_fields = ("workflow_run__run_name",)
