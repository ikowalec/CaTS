from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from django.core.management import call_command

from ingest.services import _store_scan_profile, ingest_single_run
from ingest.spectral import (
    load_spectral_dataset,
    spectrum_curve_from_csv,
    transition_data_from_dataset,
)
from references.models import CalculationSettingsProfile
from results.models import ComputedProperty
from workflows.models import WorkflowDefinition, WorkflowRun


def _csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.django_db
def test_modern_spectral_backfill_is_dry_run_safe_and_idempotent():
    profile = CalculationSettingsProfile.objects.create(name="modern_spectral_profile")
    workflow = WorkflowDefinition.objects.create(name="modern_spectral_workflow")
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="modern_spectral_run",
        working_directory="/unavailable/modern_spectral_run",
        settings_profile=profile,
    )
    ComputedProperty.objects.create(
        workflow_run=run,
        name="final_tddft_transitions",
        unit="",
        value_text="transitions",
        metadata={
            "transitions": [{
                "index": 1,
                "wavenumber_cm": 20000.0,
                "wavelength_nm": 500.0,
                "oscillator_strength": 0.3,
            }]
        },
    )

    call_command("backfill_spectral_data", "--run-id", str(run.pk), "--dry-run")
    assert not ComputedProperty.objects.filter(workflow_run=run, name="spectral_dataset").exists()

    call_command("backfill_spectral_data", "--run-id", str(run.pk))
    first = ComputedProperty.objects.get(workflow_run=run, name="spectral_dataset")
    first_checksum = first.metadata["content_sha256"]
    call_command("backfill_spectral_data", "--run-id", str(run.pk))
    first.refresh_from_db()

    assert first.metadata["content_sha256"] == first_checksum
    assert transition_data_from_dataset(run, "final_tddft").wavelengths_nm.tolist() == [500.0]

    _store_scan_profile(
        run=run,
        source_artifact=None,
        profile={
            "start_energy_eV": -10.0,
            "peak_energy_eV": -9.5,
            "activation_barrier_eV": 0.5,
            "points": [{
                "iteration": 1,
                "energy_eV": -10.0,
                "excited_states": [{
                    "state": 1,
                    "wavelength_nm": 500.0,
                    "oscillator_strength": 0.3,
                }],
                "chosen_iroot_data": {"iroot": 1, "ct_e": 0.2, "ct_pct": 20.0},
            }],
        },
    )
    assert load_spectral_dataset(run, "bond_scan")["scan_endpoints"][0][
        "chosen_iroot_data"
    ]["ct_pct"] == pytest.approx(20.0)


@pytest.mark.django_db
def test_modern_ingestion_retains_raw_and_broadened_spectral_data(tmp_path: Path):
    run_dir = tmp_path / "modern_summary"
    run_dir.mkdir()
    (run_dir / "reaction_hub_summary.json").write_text(
        json.dumps({
            "status": "completed",
            "tddft": {
                "wavenumbers_cm": [25000.0],
                "wavelengths_nm": [400.0],
                "oscillator_strengths": [0.25],
            },
        }),
        encoding="utf-8",
    )

    run = ingest_single_run(run_dir)
    dataset = load_spectral_dataset(run, "final_tddft")

    assert dataset["transitions"][0]["wavelength_nm"] == pytest.approx(400.0)
    assert dataset["broadened_spectra"]["final_tddft_spectrum"]["points"]

    difference_csv = tmp_path / "difference.csv"
    _csv(
        difference_csv,
        [{
            "wavelength_nm": 400.0,
            "delta_intensity": 0.1,
            "full_intensity": 0.8,
            "cluster_intensity": 0.7,
        }],
    )
    retained_curve = spectrum_curve_from_csv(difference_csv)
    assert [series["name"] for series in retained_curve["series"]] == [
        "delta_intensity",
        "full_intensity",
        "cluster_intensity",
    ]
