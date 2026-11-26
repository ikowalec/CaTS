from __future__ import annotations

from decimal import Decimal

import pytest

from jobs.models import JobRecord
from references.models import CalculationSettingsProfile, ReferenceAdsorbate, ReferenceNanocluster
from results.models import AdsorptionEnergyResult, ComputedProperty
from results.services import (
    build_results_dashboard,
    upsert_adsorption_energy_for_run,
    upsert_atomization_energy_for_run,
    upsert_mixing_energy_for_run,
)
from workflows.models import WorkflowDefinition, WorkflowRun


@pytest.mark.django_db
def test_upsert_mixing_energy_uses_most_stable_same_size_monometallic_references():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_mixing")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_mixing", code_path="x.py")

    run_au = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au3_low_mixing",
        status="completed",
        working_directory="/tmp/au3_low_mixing",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au3"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_au,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-9.0,
    )
    reference_job = JobRecord.objects.create(
        workflow_run=run_au,
        scheduler_job_id="au3-low-job",
        job_name="au3-low-job",
        status="completed",
    )
    run_au_high = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au3_high_mixing",
        status="completed",
        working_directory="/tmp/au3_high_mixing",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au3"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_au_high,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-8.5,
    )

    run_cu = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu3_mixing",
        status="completed",
        working_directory="/tmp/cu3_mixing",
        settings_profile=profile,
        metadata={"structure": {"formula": "Cu3"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_cu,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-6.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au2cu_system_mixing",
        status="completed",
        working_directory="/tmp/au2cu_system_mixing",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au2Cu"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-8.5,
    )

    prop = upsert_mixing_energy_for_run(run_system)

    assert prop is not None
    assert prop.name == "mixing_energy_eV"
    assert prop.unit == "eV"
    assert prop.value_float == pytest.approx(-0.5)
    per_atom = ComputedProperty.objects.get(
        workflow_run=run_system,
        name="mixing_energy_per_atom_eV",
        unit="eV/atom",
    )
    assert per_atom.value_float == pytest.approx(-0.166667)
    assert prop.metadata["reference_energy_eV"] == "-8.000000"
    assert prop.metadata["source"] == "monometallic_nanoparticle_references"
    breakdown = {entry["symbol"]: entry for entry in prop.metadata["reference_breakdown"]}
    assert set(breakdown) == {"Au", "Cu"}
    assert breakdown["Au"]["reference_run_id"] == run_au.pk
    assert breakdown["Au"]["reference_workflow"] == {
        "id": workflow.pk,
        "name": workflow.name,
        "code_path": workflow.code_path,
    }
    assert breakdown["Au"]["reference_jobs"][0]["id"] == reference_job.pk


@pytest.mark.django_db
def test_upsert_atomization_energy_uses_elemental_gas_references():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_atomization")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_atomization", code_path="x.py")
    au_ref = ReferenceAdsorbate.objects.create(
        name="Au_atom_atomization",
        formula="Au",
        settings_profile=profile,
    )
    for symbol, reference, energy in (("Au", au_ref, -2.0), ("Cu", None, -1.0)):
        reference_run = WorkflowRun.objects.create(
            workflow=workflow,
            run_name=f"{symbol}_atom_atomization",
            status="completed",
            working_directory=f"/tmp/{symbol}_atom_atomization",
            settings_profile=profile,
            reference_adsorbate=reference,
            metadata={"structure": {"formula": symbol}},
        )
        ComputedProperty.objects.create(
            workflow_run=reference_run,
            name="final_single_point_energy_eV",
            unit="eV",
            value_float=energy,
        )

    system_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au2cu_atomization",
        status="completed",
        working_directory="/tmp/au2cu_atomization",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au2Cu"}},
    )
    ComputedProperty.objects.create(
        workflow_run=system_run,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-5.5,
    )

    prop = upsert_atomization_energy_for_run(system_run)

    assert prop is not None
    assert prop.value_float == pytest.approx(0.5)
    assert prop.metadata["gas_atom_energy_eV"] == "-5.000000"
    assert prop.metadata["source"] == "elemental_gas_adsorbate_references"
    assert ComputedProperty.objects.get(
        workflow_run=system_run,
        name="atomization_energy_per_atom_eV",
        unit="eV/atom",
    ).value_float == pytest.approx(0.166667)


@pytest.mark.django_db
def test_upsert_mixing_energy_for_monometal_uses_lowest_same_size_nanoparticle():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_mixing_mono")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_mixing_mono", code_path="x.py")
    low = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au3_low",
        status="completed",
        working_directory="/tmp/au3_low",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au3"}},
    )
    ComputedProperty.objects.create(
        workflow_run=low,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-6.0,
    )
    high = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au3_high",
        status="completed",
        working_directory="/tmp/au3_high",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au3"}},
    )
    ComputedProperty.objects.create(
        workflow_run=high,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-5.5,
    )

    low_prop = upsert_mixing_energy_for_run(low)
    high_prop = upsert_mixing_energy_for_run(high)

    assert low_prop is not None
    assert high_prop is not None
    assert low_prop.value_float == pytest.approx(0.0)
    assert high_prop.value_float == pytest.approx(0.5)


@pytest.mark.django_db
def test_upsert_mixing_energy_skips_non_metal_structures():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_mixing_skip")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_mixing_skip", code_path="x.py")

    au_ref = ReferenceNanocluster.objects.create(
        name="Au13_ref_mixing_skip",
        composition="Au13",
        settings_profile=profile,
    )
    run_au = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_reference_mixing_skip",
        status="completed",
        working_directory="/tmp/au13_reference_mixing_skip",
        settings_profile=profile,
        reference_nanocluster=au_ref,
    )
    ComputedProperty.objects.create(
        workflow_run=run_au,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-26.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au2o_system_mixing_skip",
        status="completed",
        working_directory="/tmp/au2o_system_mixing_skip",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au2O"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-5.5,
    )

    prop = upsert_mixing_energy_for_run(run_system)

    assert prop is None
    assert not ComputedProperty.objects.filter(
        workflow_run=run_system,
        name__in=["mixing_energy_eV", "mixing_energy_per_atom_eV"],
    ).exists()


@pytest.mark.django_db
def test_upsert_adsorption_energy_ignores_elemental_metal_references():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_metal_not_adsorbate")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_metal_not_adsorbate", code_path="x.py")
    metal = ReferenceAdsorbate.objects.create(
        name="Au_atom_not_adsorbate", formula="Au", settings_profile=profile
    )
    cluster = ReferenceNanocluster.objects.create(
        name="Au13_not_adsorbate", composition="Au13", settings_profile=profile
    )
    for name, formula, energy, adsorbate, nanocluster in (
        ("au_atom", "Au", -2.0, metal, None),
        ("au13", "Au13", -26.0, None, cluster),
        ("au20", "Au20", -41.0, None, None),
    ):
        run = WorkflowRun.objects.create(
            workflow=workflow,
            run_name=name,
            status="completed",
            working_directory=f"/tmp/{name}",
            settings_profile=profile,
            reference_adsorbate=adsorbate,
            reference_nanocluster=nanocluster,
            metadata={"structure": {"formula": formula}},
        )
        ComputedProperty.objects.create(
            workflow_run=run,
            name="final_single_point_energy_eV",
            unit="eV",
            value_float=energy,
        )

    assert upsert_adsorption_energy_for_run(run) is None


@pytest.mark.django_db
def test_upsert_adsorption_energy_formula_match_cu13_plus_co():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile")
    workflow = WorkflowDefinition.objects.create(name="svc_wf", code_path="x.py")

    adsorbate = ReferenceAdsorbate.objects.create(
        name="CO_ref",
        formula="CO",
        settings_profile=profile,
    )
    nanocluster = ReferenceNanocluster.objects.create(
        name="Cu13_ref",
        composition="Cu13",
        settings_profile=profile,
    )

    run_ads = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_reference",
        status="completed",
        working_directory="/tmp/co_reference",
        settings_profile=profile,
        reference_adsorbate=adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13_reference",
        status="completed",
        working_directory="/tmp/cu13_reference",
        settings_profile=profile,
        reference_nanocluster=nanocluster,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13co_system",
        status="completed",
        working_directory="/tmp/cu13co_system",
        settings_profile=profile,
        metadata={"structure": {"formula": "Cu13CO"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.0,
    )

    result = upsert_adsorption_energy_for_run(run_system)

    assert result is not None
    assert result.nanocluster_id == nanocluster.id
    assert result.adsorbate_id == adsorbate.id
    assert result.total_energy_eV == Decimal("-111.000000")
    assert result.reference_energy_eV == Decimal("-110.000000")
    assert result.adsorption_energy_eV == Decimal("-1.000000")

    dashboard = build_results_dashboard(limit=100)
    target_rows = [row for row in dashboard["rows"] if row["run_id"] == run_system.id]
    assert target_rows
    assert target_rows[0]["status_code"] == "computed"


@pytest.mark.django_db
def test_upsert_adsorption_energy_uses_reference_run_formula_when_reference_record_formula_is_stale():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_stale_formula")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_stale_formula", code_path="x.py")

    # Intentionally stale/mistyped values in reference records.
    adsorbate = ReferenceAdsorbate.objects.create(
        name="CO2_ref_stale",
        formula="CO",
        settings_profile=profile,
    )
    nanocluster = ReferenceNanocluster.objects.create(
        name="Au13_ref_stale",
        composition="Au12",
        settings_profile=profile,
    )

    run_ads = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co2_reference",
        status="completed",
        working_directory="/tmp/co2_reference",
        settings_profile=profile,
        reference_adsorbate=adsorbate,
        metadata={"structure": {"formula": "CO2"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-20.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_reference",
        status="completed",
        working_directory="/tmp/au13_reference",
        settings_profile=profile,
        reference_nanocluster=nanocluster,
        metadata={"structure": {"formula": "Au13"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cau13o2_system",
        status="completed",
        working_directory="/tmp/cau13o2_system",
        settings_profile=profile,
        metadata={"structure": {"formula": "CAu13O2"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-121.0,
    )

    result = upsert_adsorption_energy_for_run(run_system)

    assert result is not None
    assert result.nanocluster_id == nanocluster.id
    assert result.adsorbate_id == adsorbate.id
    assert result.total_energy_eV == Decimal("-121.000000")
    assert result.reference_energy_eV == Decimal("-120.000000")
    assert result.adsorption_energy_eV == Decimal("-1.000000")


@pytest.mark.django_db
def test_upsert_adsorption_energy_supports_half_diatomic_adsorbate_reference():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_half_o2")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_half_o2", code_path="x.py")

    o2_adsorbate = ReferenceAdsorbate.objects.create(
        name="O2_ref",
        formula="O2",
        settings_profile=profile,
    )
    au13_cluster = ReferenceNanocluster.objects.create(
        name="Au13_ref",
        composition="Au13",
        settings_profile=profile,
    )

    run_ads = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="o2_reference",
        status="completed",
        working_directory="/tmp/o2_reference",
        settings_profile=profile,
        reference_adsorbate=o2_adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_reference",
        status="completed",
        working_directory="/tmp/au13_reference_half_o2",
        settings_profile=profile,
        reference_nanocluster=au13_cluster,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13o_system",
        status="completed",
        working_directory="/tmp/au13o_system",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au13O"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-106.0,
    )

    result = upsert_adsorption_energy_for_run(run_system)

    assert result is not None
    assert result.nanocluster_id == au13_cluster.id
    assert result.adsorbate_id == o2_adsorbate.id
    assert result.total_energy_eV == Decimal("-106.000000")
    assert result.reference_energy_eV == Decimal("-105.000000")
    assert result.adsorption_energy_eV == Decimal("-1.000000")
    assert result.metadata.get("adsorbate_stoich_factor") == "1/2"
    assert result.metadata.get("adsorbate_scaled_energy_eV") == "-5.000000"


@pytest.mark.django_db
def test_upsert_adsorption_energy_adds_distortion_corrected_interaction_energy():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_distortion")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_distortion", code_path="x.py")

    adsorbate = ReferenceAdsorbate.objects.create(
        name="CO_ref_distortion",
        formula="CO",
        settings_profile=profile,
    )
    nanocluster = ReferenceNanocluster.objects.create(
        name="Cu13_ref_distortion",
        composition="Cu13",
        settings_profile=profile,
    )

    run_ads = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_reference_distortion",
        status="completed",
        working_directory="/tmp/co_reference_distortion",
        settings_profile=profile,
        reference_adsorbate=adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13_reference_distortion",
        status="completed",
        working_directory="/tmp/cu13_reference_distortion",
        settings_profile=profile,
        reference_nanocluster=nanocluster,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13co_system_distortion",
        status="completed",
        working_directory="/tmp/cu13co_system_distortion",
        settings_profile=profile,
        metadata={
            "structure": {"formula": "Cu13CO"},
            "result_summary": {"final": {"cluster_fragment": {"energy_eV": -99.5}}},
        },
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.0,
    )

    result = upsert_adsorption_energy_for_run(run_system)

    assert result is not None
    assert result.adsorption_energy_eV == Decimal("-1.000000")
    assert result.metadata.get("cluster_distortion_energy_eV") == "0.500000"
    assert result.metadata.get("ed_energy_eV") == "0.500000"
    assert result.metadata.get("eint_energy_eV") == "-1.500000"
    assert result.metadata.get("distortion_corrected_adsorption_energy_eV") == "-1.500000"
    assert result.metadata.get("adsorbate_cluster_interaction_energy_eV") == "-1.500000"


@pytest.mark.django_db
def test_upsert_adsorption_energy_uses_legacy_distortion_sibling():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_legacy_distortion")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_legacy_distortion", code_path="x.py")
    adsorbate = ReferenceAdsorbate.objects.create(
        name="CO_ref_legacy_distortion", formula="CO", settings_profile=profile
    )
    cluster = ReferenceNanocluster.objects.create(
        name="Cu13_ref_legacy_distortion", composition="Cu13", settings_profile=profile
    )
    for name, formula, energy, ads_ref, cluster_ref in (
        ("co_ref", "CO", -10.0, adsorbate, None),
        ("cu13_ref", "Cu13", -100.0, None, cluster),
    ):
        reference_run = WorkflowRun.objects.create(
            workflow=workflow,
            run_name=name,
            status="completed",
            settings_profile=profile,
            reference_adsorbate=ads_ref,
            reference_nanocluster=cluster_ref,
            metadata={"structure": {"formula": formula}},
        )
        ComputedProperty.objects.create(
            workflow_run=reference_run,
            name="final_single_point_energy_eV",
            unit="eV",
            value_float=energy,
        )

    sibling = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13co_distortion",
        status="completed",
        settings_profile=profile,
        metadata={
            "structure": {"formula": "Cu13"},
            "legacy_catalog": {
                "source_tree_id": "tree",
                "working_directory_relative": "family/distortion_energy/Cu13CO_case",
            },
        },
    )
    ComputedProperty.objects.create(
        workflow_run=sibling,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-99.5,
    )
    system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13co_system_legacy_distortion",
        status="completed",
        settings_profile=profile,
        metadata={
            "structure": {"formula": "Cu13CO"},
            "legacy_catalog": {
                "source_tree_id": "tree",
                "working_directory_relative": "family/Cu13CO_case/Cu13CO_case",
            },
        },
    )
    ComputedProperty.objects.create(
        workflow_run=system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.0,
    )

    result = upsert_adsorption_energy_for_run(system)

    assert result is not None
    assert result.metadata["eint_energy_eV"] == "-1.500000"


@pytest.mark.django_db
def test_upsert_adsorption_energy_prefers_homonuclear_diatomic_reference_over_atomic():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_diatomic_prefer")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_diatomic_prefer", code_path="x.py")

    o2_adsorbate = ReferenceAdsorbate.objects.create(
        name="O2_ref_prefer",
        formula="O2",
        settings_profile=profile,
    )
    o_adsorbate = ReferenceAdsorbate.objects.create(
        name="O_ref_atomic",
        formula="O",
        settings_profile=profile,
    )
    au13_cluster = ReferenceNanocluster.objects.create(
        name="Au13_ref_prefer",
        composition="Au13",
        settings_profile=profile,
    )

    run_o2 = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="o2_reference_prefer",
        status="completed",
        working_directory="/tmp/o2_reference_prefer",
        settings_profile=profile,
        reference_adsorbate=o2_adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_o2,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    # Created later on purpose: timestamp alone should not override diatomic preference.
    run_o = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="o_reference_atomic",
        status="completed",
        working_directory="/tmp/o_reference_atomic",
        settings_profile=profile,
        reference_adsorbate=o_adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_o,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-4.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_reference_prefer",
        status="completed",
        working_directory="/tmp/au13_reference_prefer",
        settings_profile=profile,
        reference_nanocluster=au13_cluster,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13o_system_prefer",
        status="completed",
        working_directory="/tmp/au13o_system_prefer",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au13O"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-106.0,
    )

    result = upsert_adsorption_energy_for_run(run_system)

    assert result is not None
    assert result.adsorbate_id == o2_adsorbate.id
    assert result.reference_energy_eV == Decimal("-105.000000")
    assert result.adsorption_energy_eV == Decimal("-1.000000")
    assert result.metadata.get("adsorbate_stoich_factor") == "1/2"


@pytest.mark.django_db
def test_upsert_adsorption_energy_uses_cadsorbate_hints_for_multi_adsorbate_reference():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_coads_hint")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_coads_hint", code_path="x.py")

    co_adsorbate = ReferenceAdsorbate.objects.create(
        name="CO_ref_coads",
        formula="CO",
        settings_profile=profile,
    )
    no_adsorbate = ReferenceAdsorbate.objects.create(
        name="NO_ref_coads",
        formula="NO",
        settings_profile=profile,
    )
    au13_cluster = ReferenceNanocluster.objects.create(
        name="Au13_ref_coads",
        composition="Au13",
        settings_profile=profile,
    )

    run_ads_co = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_reference_coads",
        status="completed",
        working_directory="/tmp/co_reference_coads",
        settings_profile=profile,
        reference_adsorbate=co_adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads_co,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_ads_no = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="no_reference_coads",
        status="completed",
        working_directory="/tmp/no_reference_coads",
        settings_profile=profile,
        reference_adsorbate=no_adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads_no,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-8.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_reference_coads",
        status="completed",
        working_directory="/tmp/au13_reference_coads",
        settings_profile=profile,
        reference_nanocluster=au13_cluster,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_co_no_system",
        status="completed",
        working_directory="/tmp/au13_co_no_system",
        settings_profile=profile,
        metadata={
            "structure": {
                # Intentionally not formula-balanced for strict decomposition; hints should drive matching.
                "formula": "Au13CNO3",
                "adsorption_reference_hints": {
                    "reference_cluster_id": au13_cluster.pk,
                    "adsorbate_formula_counts": {"CO": 1, "NO": 1},
                },
            }
        },
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-119.0,
    )

    result = upsert_adsorption_energy_for_run(run_system)

    assert result is not None
    assert result.nanocluster_id == au13_cluster.id
    assert result.total_energy_eV == Decimal("-119.000000")
    assert result.reference_energy_eV == Decimal("-118.000000")
    assert result.adsorption_energy_eV == Decimal("-1.000000")
    assert result.metadata.get("match_rule") == "coadsorbate_reference_hints"
    component_breakdown = result.metadata.get("component_breakdown")
    assert isinstance(component_breakdown, list)
    assert len(component_breakdown) == 2
    assert {entry.get("adsorbate_formula") for entry in component_breakdown} == {"CO", "NO"}

    dashboard = build_results_dashboard(limit=100)
    row = next(entry for entry in dashboard["rows"] if entry["run_id"] == run_system.pk)
    assert row.get("adsorbate_name") == "CO, NO"


@pytest.mark.django_db
def test_upsert_adsorption_energy_infers_coadsorption_interaction_from_single_adsorption_results():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_coads_interaction")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_coads_interaction", code_path="x.py")

    co_adsorbate = ReferenceAdsorbate.objects.create(
        name="CO_ref_interaction",
        formula="CO",
        settings_profile=profile,
    )
    no_adsorbate = ReferenceAdsorbate.objects.create(
        name="NO_ref_interaction",
        formula="NO",
        settings_profile=profile,
    )
    au13_cluster = ReferenceNanocluster.objects.create(
        name="Au13_ref_interaction",
        composition="Au13",
        settings_profile=profile,
    )

    run_ads_co = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_reference_interaction",
        status="completed",
        working_directory="/tmp/co_reference_interaction",
        settings_profile=profile,
        reference_adsorbate=co_adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads_co,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_ads_no = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="no_reference_interaction",
        status="completed",
        working_directory="/tmp/no_reference_interaction",
        settings_profile=profile,
        reference_adsorbate=no_adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads_no,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-8.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_reference_interaction",
        status="completed",
        working_directory="/tmp/au13_reference_interaction",
        settings_profile=profile,
        reference_nanocluster=au13_cluster,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system_co = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_co_system_interaction",
        status="completed",
        working_directory="/tmp/au13_co_system_interaction",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au13CO"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system_co,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.0,
    )
    result_co = upsert_adsorption_energy_for_run(run_system_co)
    assert result_co is not None
    assert result_co.adsorption_energy_eV == Decimal("-1.000000")

    run_system_no = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_no_system_interaction",
        status="completed",
        working_directory="/tmp/au13_no_system_interaction",
        settings_profile=profile,
        metadata={"structure": {"formula": "Au13NO"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system_no,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-109.0,
    )
    result_no = upsert_adsorption_energy_for_run(run_system_no)
    assert result_no is not None
    assert result_no.adsorption_energy_eV == Decimal("-1.000000")

    run_system_coads = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au13_co_no_system_interaction",
        status="completed",
        working_directory="/tmp/au13_co_no_system_interaction",
        settings_profile=profile,
        metadata={
            "structure": {
                "formula": "Au13CNO3",
                "adsorption_reference_hints": {
                    "reference_cluster_id": au13_cluster.pk,
                    "adsorbate_formula_counts": {"CO": 1, "NO": 1},
                },
            }
        },
    )
    ComputedProperty.objects.create(
        workflow_run=run_system_coads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-119.6,
    )

    coads_result = upsert_adsorption_energy_for_run(run_system_coads)
    assert coads_result is not None
    assert coads_result.adsorption_energy_eV == Decimal("-1.600000")
    assert coads_result.metadata.get("coadsorption_component_adsorption_sum_eV") == "-2.000000"
    assert coads_result.metadata.get("coadsorption_interaction_energy_eV") == "0.400000"


@pytest.mark.django_db
def test_upsert_adsorption_energy_skips_coadsorbate_screening_run_mode():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_skip_coads_mode")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_skip_coads_mode", code_path="x.py")

    adsorbate = ReferenceAdsorbate.objects.create(
        name="CO_ref_skip_coads_mode",
        formula="CO",
        settings_profile=profile,
    )
    nanocluster = ReferenceNanocluster.objects.create(
        name="Cu13_ref_skip_coads_mode",
        composition="Cu13",
        settings_profile=profile,
    )

    run_ads = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_reference_skip_coads_mode",
        status="completed",
        working_directory="/tmp/co_reference_skip_coads_mode",
        settings_profile=profile,
        reference_adsorbate=adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13_reference_skip_coads_mode",
        status="completed",
        working_directory="/tmp/cu13_reference_skip_coads_mode",
        settings_profile=profile,
        reference_nanocluster=nanocluster,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13co_system_skip_coads_mode",
        status="completed",
        working_directory="/tmp/cu13co_system_skip_coads_mode",
        settings_profile=profile,
        metadata={
            "submission": {"run_mode": "coadsorbate_placement"},
            "structure": {"formula": "Cu13CO"},
        },
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.0,
    )

    result = upsert_adsorption_energy_for_run(run_system)

    assert result is None
    assert not AdsorptionEnergyResult.objects.filter(workflow_run=run_system).exists()


@pytest.mark.django_db
def test_build_results_dashboard_excludes_coadsorbate_screening_runs():
    profile = CalculationSettingsProfile.objects.create(name="svc_profile_dashboard_exclude_coads")
    workflow = WorkflowDefinition.objects.create(name="svc_wf_dashboard_exclude_coads", code_path="x.py")
    coadsorbate_workflow = WorkflowDefinition.objects.create(
        name="svc_wf_dashboard_exclude_coads_screening",
        code_path="similarity/workflows/coadsorbate_similarity.py",
    )

    adsorbate = ReferenceAdsorbate.objects.create(
        name="CO_ref_dashboard_exclude_coads",
        formula="CO",
        settings_profile=profile,
    )
    nanocluster = ReferenceNanocluster.objects.create(
        name="Cu13_ref_dashboard_exclude_coads",
        composition="Cu13",
        settings_profile=profile,
    )

    run_ads = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="co_reference_dashboard_exclude_coads",
        status="completed",
        working_directory="/tmp/co_reference_dashboard_exclude_coads",
        settings_profile=profile,
        reference_adsorbate=adsorbate,
    )
    ComputedProperty.objects.create(
        workflow_run=run_ads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-10.0,
    )

    run_cluster = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13_reference_dashboard_exclude_coads",
        status="completed",
        working_directory="/tmp/cu13_reference_dashboard_exclude_coads",
        settings_profile=profile,
        reference_nanocluster=nanocluster,
    )
    ComputedProperty.objects.create(
        workflow_run=run_cluster,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-100.0,
    )

    run_system = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="cu13co_system_dashboard_exclude_coads",
        status="completed",
        working_directory="/tmp/cu13co_system_dashboard_exclude_coads",
        settings_profile=profile,
        metadata={"structure": {"formula": "Cu13CO"}},
    )
    ComputedProperty.objects.create(
        workflow_run=run_system,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.0,
    )

    run_cads = WorkflowRun.objects.create(
        workflow=coadsorbate_workflow,
        run_name="cu13co_system_dashboard_exclude_coads_screening",
        status="completed",
        working_directory="/tmp/cu13co_system_dashboard_exclude_coads_screening",
        settings_profile=profile,
        metadata={
            "submission": {"run_mode": "coadsorbate_placement"},
            "structure": {"formula": "Cu13CO"},
        },
    )
    ComputedProperty.objects.create(
        workflow_run=run_cads,
        name="final_single_point_energy_eV",
        unit="eV",
        value_float=-111.5,
    )

    upsert_adsorption_energy_for_run(run_system)
    dashboard = build_results_dashboard(limit=100)
    row_ids = {row["run_id"] for row in dashboard["rows"]}

    assert run_system.pk in row_ids
    assert run_cads.pk not in row_ids
