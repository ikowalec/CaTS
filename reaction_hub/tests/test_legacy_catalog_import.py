from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import pytest
from ase import Atoms
from ase.io import write
from django.core.management import call_command

from ingest.legacy_catalog import import_catalog
from ingest.services import _store_scan_profile, ingest_single_run
from ingest.spectral import (
    broadened_series_from_dataset,
    ct_series_from_dataset,
    load_spectral_dataset,
    spectrum_curve_from_csv,
    transition_data_from_dataset,
    update_spectral_dataset,
    write_broadened_plot_artifacts,
    write_ct_plot_artifacts,
)
from jobs.models import JobRecord
from references.models import CalculationSettingsProfile
from results.models import Artifact, ComputedProperty
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
def test_legacy_catalog_import_is_jobless_idempotent_and_replot_ready(tmp_path: Path):
    source = tmp_path / "original_hpc" / "source_group"
    calculation_dir = source / "Au2_CO"
    calculation_dir.mkdir(parents=True)
    (calculation_dir / "orca.out").write_text("ORCA TERMINATED NORMALLY\n", encoding="utf-8")
    (calculation_dir / "orca.molden.input").write_text("[Molden Format]\n", encoding="utf-8")
    (calculation_dir / "orca.gbw").write_bytes(b"wavefunction")
    (calculation_dir / "evaluated.traj").write_text("trajectory", encoding="utf-8")

    catalog_root = tmp_path / "catalogs"
    catalog = catalog_root / "source_group"
    (catalog / "structures").mkdir(parents=True)
    (catalog / "plots").mkdir()
    write(
        catalog / "structures" / "calc1.extxyz",
        Atoms("Au2CO", positions=[[0, 0, 0], [2.5, 0, 0], [1, 1, 0], [1, 2, 0]]),
    )
    (catalog / "plots" / "calc1_uvvis.png").write_bytes(b"png")
    (catalog / "summary.json").write_text(
        json.dumps({"root": str(source), "source_tree_id": "stable-tree-1"}),
        encoding="utf-8",
    )
    _csv(
        catalog / "calculations.csv",
        [{
            "calculation_id": "calc1",
            "calculation_fingerprint": "stable-calculation-1",
            "orca_out": "Au2_CO/orca.out",
            "input_source": "embedded",
            "input_text": "! CAM-B3LYP 6-31G(d,p)\n* xyz 0 1\n*",
            "orca_version": "6.1.0",
            "charge": 0,
            "multiplicity": 1,
            "simple_keywords": '["CAM-B3LYP 6-31G(d,p)"]',
            "structure_file": "structures/calc1.extxyz",
            "formula": "Au2CO",
            "atom_count": 4,
            "energy_hartree": -100.5,
            "normal_termination": True,
            "molden_files": "['Au2_CO/orca.molden.input']",
            "ifct_files": "[]",
            "ct_multiple_files": "[]",
            "fragment_config": "",
        }],
    )
    _csv(catalog / "audit.csv", [{"calculation_id": "calc1", "status": "accepted"}])
    _csv(
        catalog / "atomic_charges.csv",
        [
            {
                "calculation_id": "calc1",
                "method": method,
                "atom_index_0based": index,
                "atom_index_1based": index + 1,
                "symbol": symbol,
                "charge_e": charge,
            }
            for method, charges in (("mulliken", [0.1, 0.1, -0.1, -0.1]), ("loewdin", [0.2, 0.0, -0.1, -0.1]))
            for index, (symbol, charge) in enumerate(zip(["Au", "Au", "C", "O"], charges))
        ],
    )
    _csv(
        catalog / "artifacts.csv",
        [
            {
                "calculation_id": "calc1",
                "path": "Au2_CO/orca.gbw",
                "path_relative_to_calculation": "orca.gbw",
                "kind": "other",
                "size_bytes": 12,
                "mtime_ns": 1,
                "sha256": "a" * 64,
            },
            {
                "calculation_id": "calc1",
                "path": "Au2_CO/orca.out",
                "path_relative_to_calculation": "orca.out",
                "kind": "log",
                "size_bytes": 25,
                "mtime_ns": 2,
                "sha256": "b" * 64,
            },
        ],
    )
    _csv(
        catalog / "transitions.csv",
        [{
            "calculation_id": "calc1",
            "state": 1,
            "wavenumber_cm": 25000.0,
            "wavelength_nm": 400.0,
            "oscillator_strength": 0.25,
        }],
    )
    _csv(
        catalog / "spectra.csv",
        [{"calculation_id": "calc1", "wavelength_nm": 400.0, "intensity": 0.75}],
    )
    _csv(
        catalog / "ct_states.csv",
        [{
            "calculation_id": "calc1",
            "source": "IFCTdata.txt",
            "state": 1,
            "f": 0.25,
            "nm": 400.0,
            "CT(e)": 0.12,
            "CT(%)": 12.0,
        }],
    )
    _csv(
        catalog / "ct_terms.csv",
        [{"calculation_id": "calc1", "source": "IFCTdata.txt", "state": 1, "term": "1->2", "pct": 80.0}],
    )
    _csv(
        catalog / "ct_spectra.csv",
        [{
            "calculation_id": "calc1",
            "source": "CT_multiple/curve.csv",
            "property": "fragment transfer",
            "energy_eV": 3.1,
            "wavelength_nm": 399.95,
            "intensity": 8.5,
        }],
    )
    _csv(
        catalog / "scan_frames.csv",
        [{
            "scan_id": "scan1",
            "trajectory": "Au2_CO/evaluated.traj",
            "related_calculation_ids": "['calc1']",
            "frame_index_0based": 0,
            "frame_index_1based": 1,
            "formula": "Au2CO",
            "symbols": '["Au", "Au", "C", "O"]',
            "positions_A": "[[0, 0, 0], [2.5, 0, 0], [1, 1, 0], [1, 2, 0]]",
            "cell_A": "[[0, 0, 0], [0, 0, 0], [0, 0, 0]]",
            "pbc": "[false, false, false]",
            "energy_eV": -1.0,
            "forces_eV_A": "[[0, 0, 0], [0, 0, 0], [0.01, 0, 0], [0, 0, 0]]",
            "fmax_eV_A": 0.02,
            "info": '{"step": 0}',
            "constraints": "[]",
            "indices_0based": "[]",
            "indices_1based": "[]",
            "bond_pairs_0based": "[]",
            "bond_pairs_1based": "[]",
            "bond_distances_A": "[]",
        }],
    )
    (catalog / "fragments.jsonl").write_text(
        json.dumps({"calculation_id": "calc1", "fragments_1based": [[1, 2], [3, 4]]}) + "\n",
        encoding="utf-8",
    )

    relocated_parent = tmp_path / "relocated_hpc"
    relocated_parent.mkdir()
    relocated_source = relocated_parent / source.name
    shutil.move(source, relocated_source)
    call_command(
        "import_legacy_catalog",
        "--catalog-root",
        str(catalog_root),
        "--source-root",
        str(relocated_parent),
        "--dry-run",
    )
    assert WorkflowRun.objects.count() == 0

    first = import_catalog(catalog, current_source_root=relocated_source)
    moved_parent = tmp_path / "moved_again"
    moved_parent.mkdir()
    moved_source = moved_parent / source.name
    shutil.move(relocated_source, moved_source)
    (catalog / "summary.json").write_text(
        json.dumps({"root": str(moved_source), "source_tree_id": "stable-tree-1"}),
        encoding="utf-8",
    )
    second = import_catalog(catalog, current_source_root=moved_source)

    assert first["created"] == 1
    assert second["updated"] == 1
    assert WorkflowRun.objects.count() == 1
    assert JobRecord.objects.count() == 0
    run = WorkflowRun.objects.get()
    assert run.settings_profile.name == "cam-b3lyp-lanl2dz"
    assert not Artifact.objects.filter(workflow_run=run).exclude(job=None).exists()
    assert run.working_directory == (moved_source / "Au2_CO").as_posix()
    assert run.metadata["legacy_catalog"]["source_tree_id"] == "stable-tree-1"
    assert run.metadata["legacy_catalog"]["source_root_history"] == sorted(
        [source.as_posix(), moved_source.as_posix()]
    )
    assert run.metadata["legacy_catalog"]["calculation"]["orca_version"] == "6.1.0"
    manifest_artifact = Artifact.objects.get(workflow_run=run, path="orca.gbw")
    assert manifest_artifact.metadata["source_relative_path"] == "Au2_CO/orca.gbw"
    assert manifest_artifact.metadata["parsed_sha256"] == "a" * 64
    assert Artifact.objects.get(workflow_run=run, path="orca.out").metadata[
        "parsed_sha256"
    ] == "b" * 64
    assert ComputedProperty.objects.get(
        workflow_run=run, name="final_single_point_energy", unit="Eh"
    ).value_float == pytest.approx(-100.5)
    assert ComputedProperty.objects.get(
        workflow_run=run, name="final_loewdin_charges", unit="e"
    ).metadata["charges"] == pytest.approx([0.2, 0.0, -0.1, -0.1])
    dataset = load_spectral_dataset(run, "final_tddft")
    assert dataset["ct_states"][0]["charge_transfer_pct"] == pytest.approx(12.0)
    assert dataset["scan_frames"][0]["scan_id"] == "scan1"
    assert dataset["scan_frames"][0]["positions_A"][2] == [1, 1, 0]
    assert dataset["scan_frames"][0]["forces_eV_A"][2] == [0.01, 0, 0]

    shutil.rmtree(catalog)
    transitions = transition_data_from_dataset(run, "final_tddft")
    ct_series = ct_series_from_dataset(run, "final_tddft")
    uvvis_series = broadened_series_from_dataset(run, "final_tddft")
    assert transitions.wavelengths_nm.tolist() == [400.0]
    assert transitions.intensities.tolist() == [0.25]
    assert uvvis_series["legacy_uvvis"][0]["intensity"] == pytest.approx(0.75)
    assert ct_series["CT_multiple/curve.csv|fragment transfer"][0]["intensity"] == pytest.approx(8.5)

    update_spectral_dataset(
        run=run,
        dataset_name="final_tddft",
        payload={
            "fragments": {
                "fragment_labels": {"1": "NC", "2": "CO"},
            },
            "broadened_spectra": {
                "final_tddft_spectrum": {
                    "points": [{"x": 300.0, "intensity": 0.1}, {"x": 800.0, "intensity": 0.0}],
                },
            },
            "ct_spectra": [
                {
                    "source": "CT_multiple/total_spectrum.txt",
                    "property": "UV-Vis",
                    "points": [{"wavelength_nm": 400.0, "intensity": 0.25}],
                },
                {
                    "source": "CT_multiple/ET_1to2.txt",
                    "property": "1->2",
                    "points": [{"wavelength_nm": 400.0, "intensity": 0.0625}],
                },
            ],
            "ct_states": [
                {"state": 1, "wavelength_nm": 400.0, "oscillator_strength": 0.25},
                {
                    "source": "CT_spectrum_automatic/IFCTmajor.txt",
                    "state": 1,
                    "wavelength_nm": 400.0,
                    "oscillator_strength": 0.25,
                },
                {"state": 2, "wavelength_nm": 500.0, "oscillator_strength": 0.20},
            ],
            "ct_terms": [
                {"state": 1, "term": "1->2", "percentage": 25.0},
                {
                    "source": "CT_spectrum_automatic/IFCTmajor.txt",
                    "state": 1,
                    "term": "1->2",
                    "percentage": 25.0,
                },
                {"state": 1, "term": "1<-2", "percentage": 0.4},
                {"state": 2, "term": "1<-2", "percentage": 40.0},
            ],
        },
    )
    with pytest.MonkeyPatch.context() as monkeypatch:
        cache_root = tmp_path / "plot_cache"
        monkeypatch.setattr("django.conf.settings.REACTION_HUB_REMOTE_CACHE_ROOT", str(cache_root))
        plots = write_ct_plot_artifacts(run)
        repeated = write_ct_plot_artifacts(run)
        broadened_plots = write_broadened_plot_artifacts(run)
        repeated_broadened = write_broadened_plot_artifacts(run)
    assert len(plots) == 2
    assert repeated[0].pk == plots[0].pk
    plot_path = Path(plots[0].metadata["local_cache_path"])
    assert plot_path.read_bytes().startswith(b"\x89PNG")
    assert plots[0].label == "plot:charge_transfer:final_tddft"
    assert plots[0].metadata["x_unit"] == "nm"
    assert plots[0].metadata["wavelength_range_nm"] == pytest.approx([300.0, 800.0])
    assert plots[0].metadata["minimum_ct_percentage"] == pytest.approx(0.5)
    assert plots[0].metadata["minimum_annotation_signal_fraction"] == pytest.approx(0.1)
    assert plots[0].metadata["gaussian_linewidth_nm"] == pytest.approx(20.0)
    assert plots[0].metadata["direction_color_convention"] == {
        "->": "#dc2626",
        "<-": "#2563eb",
    }
    assert plots[0].metadata["max_net_positive_ct"]["percentage"] == pytest.approx(25.0)
    assert plots[0].metadata["max_net_negative_ct"]["percentage"] == pytest.approx(-40.0)
    assert plots[0].metadata["integrated_positive_ct_area_pct"] == pytest.approx(13.89, abs=0.01)
    assert plots[0].metadata["integrated_negative_ct_area_pct"] == pytest.approx(17.78, abs=0.01)
    assert {peak["state"] for peak in plots[0].metadata["resolved_peak_contributors"]} == {1, 2}
    pair_plot = next(plot for plot in plots if plot.label.endswith(":nc_co"))
    assert pair_plot.metadata["fragment_pair"] == ["NC", "CO"]
    assert set(pair_plot.metadata["directions"]) == {"NC→CO", "CO→NC"}
    assert pair_plot.metadata["excitation_lines_nm"] == pytest.approx([400.0, 500.0])
    uvvis_plot = Artifact.objects.get(workflow_run=run, label="plot:uvvis:final_tddft")
    assert uvvis_plot.metadata["source"] == "retained_ifct_states"
    assert uvvis_plot.metadata["source_state_count"] == 2
    assert uvvis_plot.metadata["ordinary_transition_count"] == 1
    assert Artifact.objects.filter(workflow_run=run, label="legacy:uvvis_original").exists()
    assert len(broadened_plots) == 1
    assert repeated_broadened[0].pk == broadened_plots[0].pk
    assert broadened_plots[0].metadata["line_color"] == "#2563eb"
    assert broadened_plots[0].metadata["wavelength_range_nm"] == [300.0, 800.0]
    assert Path(broadened_plots[0].metadata["local_cache_path"]).read_bytes().startswith(b"\x89PNG")


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
