from __future__ import annotations

from decimal import Decimal

import pytest
from django.core.exceptions import ValidationError

from references.models import CalculationSettingsProfile, ReferenceAdsorbate, ReferenceNanocluster
from results.models import AdsorptionEnergyResult
from workflows.models import WorkflowDefinition, WorkflowRun


@pytest.mark.django_db
def test_adsorption_energy_requires_matching_settings_and_references():
    profile_ok = CalculationSettingsProfile.objects.create(
        name="profile_ok",
        xc_functional="PBE0",
        basis_set="6-31g(d,p)",
        solvent="Water",
    )
    profile_bad = CalculationSettingsProfile.objects.create(
        name="profile_bad",
        xc_functional="CAM-B3LYP",
        basis_set="def2-svp",
        solvent="Methanol",
    )

    ads = ReferenceAdsorbate.objects.create(
        name="ads1",
        formula="O2",
        settings_profile=profile_ok,
    )
    cluster = ReferenceNanocluster.objects.create(
        name="cluster1",
        composition="Cu13",
        settings_profile=profile_ok,
    )

    wf = WorkflowDefinition.objects.create(name="wf", code_path="x.py")
    run = WorkflowRun.objects.create(
        workflow=wf,
        run_name="run-1",
        working_directory="/tmp/run-1",
        status="completed",
        settings_profile=profile_ok,
        reference_adsorbate=ads,
        reference_nanocluster=cluster,
    )

    ok = AdsorptionEnergyResult(
        workflow_run=run,
        adsorbate=ads,
        nanocluster=cluster,
        settings_profile=profile_ok,
        total_energy_eV=Decimal("-100.000000"),
        reference_energy_eV=Decimal("-98.000000"),
        adsorption_energy_eV=Decimal("-2.000000"),
    )
    ok.full_clean()
    ok.save()

    bad = AdsorptionEnergyResult(
        workflow_run=run,
        adsorbate=ads,
        nanocluster=cluster,
        settings_profile=profile_bad,
        total_energy_eV=Decimal("-100.000000"),
        reference_energy_eV=Decimal("-98.000000"),
        adsorption_energy_eV=Decimal("-2.000000"),
    )
    with pytest.raises(ValidationError):
        bad.full_clean()
