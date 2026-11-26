from __future__ import annotations

from decimal import Decimal

import django_tables2 as tables

from .models import AdsorptionEnergyResult
from .services import (
    _metadata_adsorbate_display_name,
    _metadata_distortion_energy_ed_eV,
    _metadata_interaction_energy_eint_eV,
)


class AdsorptionEnergyResultTable(tables.Table):
    workflow_run = tables.LinkColumn("result-detail", args=[tables.A("pk")])
    ed_energy_eV = tables.Column(
        verbose_name="Ed (eV)",
        accessor="ed_energy_value",
        orderable=False,
        empty_values=(),
    )
    eint_energy_eV = tables.Column(
        verbose_name="Eint (eV)",
        accessor="eint_energy_value",
        order_by=("eint_energy_order",),
        empty_values=(),
    )

    @staticmethod
    def _format_energy(value):
        if value is None:
            return "-"
        if isinstance(value, str) and value.strip().lower() in {"", "none", "nan"}:
            return "-"
        try:
            number = Decimal(str(value))
        except Exception:
            return value
        return f"{number:.2f}"

    def render_total_energy_eV(self, value):
        return self._format_energy(value)

    def render_reference_energy_eV(self, value):
        return self._format_energy(value)

    def render_adsorption_energy_eV(self, value):
        return self._format_energy(value)

    def render_adsorbate(self, _value, record: AdsorptionEnergyResult):
        return _metadata_adsorbate_display_name(record)

    def render_ed_energy_eV(self, value, record: AdsorptionEnergyResult):
        if value is None:
            value = _metadata_distortion_energy_ed_eV(record)
        return self._format_energy(value)

    def render_eint_energy_eV(self, value, record: AdsorptionEnergyResult):
        if value is None:
            value = _metadata_interaction_energy_eint_eV(record)
        return self._format_energy(value)

    class Meta:
        model = AdsorptionEnergyResult
        fields = (
            "workflow_run",
            "adsorbate",
            "nanocluster",
            "settings_profile",
            "total_energy_eV",
            "reference_energy_eV",
            "adsorption_energy_eV",
            "ed_energy_eV",
            "eint_energy_eV",
        )
        attrs = {"class": "table table-striped table-sm"}
