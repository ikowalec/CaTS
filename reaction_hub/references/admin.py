from django.contrib import admin

from .models import CalculationSettingsProfile, ReferenceAdsorbate, ReferenceNanocluster


@admin.register(CalculationSettingsProfile)
class CalculationSettingsProfileAdmin(admin.ModelAdmin):
    list_display = ("name", "backend", "approved", "xc_functional", "basis_set", "solvent", "updated_at")
    list_filter = ("approved", "xc_functional", "solvent")
    search_fields = ("name", "basis_set", "notes")

    @admin.display(description="Backend")
    def backend(self, obj):
        metadata = obj.metadata if isinstance(obj.metadata, dict) else {}
        return metadata.get("qm_backend") or metadata.get("calculator") or "orca"


@admin.register(ReferenceAdsorbate)
class ReferenceAdsorbateAdmin(admin.ModelAdmin):
    list_display = ("name", "formula", "charge", "multiplicity", "settings_profile")
    list_filter = ("settings_profile",)
    search_fields = ("name", "formula", "smiles")


@admin.register(ReferenceNanocluster)
class ReferenceNanoclusterAdmin(admin.ModelAdmin):
    list_display = ("name", "composition", "atom_count", "settings_profile")
    list_filter = ("settings_profile",)
    search_fields = ("name", "composition")
