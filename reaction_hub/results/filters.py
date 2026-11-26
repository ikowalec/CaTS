from __future__ import annotations

import django_filters

from .models import AdsorptionEnergyResult


class AdsorptionEnergyResultFilter(django_filters.FilterSet):
    workflow_run__run_name = django_filters.CharFilter(lookup_expr="icontains")
    adsorption_energy_eV__lt = django_filters.NumberFilter(field_name="adsorption_energy_eV", lookup_expr="lt")
    adsorption_energy_eV__gt = django_filters.NumberFilter(field_name="adsorption_energy_eV", lookup_expr="gt")

    class Meta:
        model = AdsorptionEnergyResult
        fields = [
            "adsorbate",
            "nanocluster",
            "settings_profile",
            "workflow_run__run_name",
            "adsorption_energy_eV__lt",
            "adsorption_energy_eV__gt",
        ]
