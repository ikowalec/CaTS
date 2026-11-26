from __future__ import annotations

import json
import importlib.util
import sys
from pathlib import Path

import pytest
from ase import Atoms

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from cats.multiwfn.tools import (  # noqa: E402
    get_mulliken_population_routine_config,
    parse_mulliken_population_report,
)
from ingest.services import _ingest_remote_summary_for_job, _ingest_summary_payload, _store_charges_payload
from ingest.tasks import backfill_ct_plots_task
from jobs.models import HPCSystem, JobRecord
from references.models import CalculationSettingsProfile
from results.models import ComputedProperty
from workflows.models import WorkflowDefinition, WorkflowRun


def test_orca_default_charge_method_is_mulliken(tmp_path: Path):
    path = Path(__file__).resolve().parents[2] / "cats/orca/run/orca_property.py"
    spec = importlib.util.spec_from_file_location("charge_parser", path)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    output = tmp_path / "orca.out"
    output.write_text(
        "MULLIKEN ATOMIC CHARGES\n 0 Au : 0.2\n 1 O : -0.2\n\n"
        "LOEWDIN ATOMIC CHARGES\n 0 Au : 0.1\n 1 O : -0.1\n",
        encoding="utf-8",
    )
    atoms = parser.get_charges_from_orca_output(Atoms("AuO"), output)
    assert atoms.info["charge_method"] == "mulliken"
    assert list(atoms.get_initial_charges()) == [0.2, -0.2]


def test_multiwfn_mulliken_config_and_parser(tmp_path: Path):
    config = get_mulliken_population_routine_config("orca.molden.input")
    assert [entry.split()[0] for entry in config[:4]] == [
        "orca.molden.input",
        "7",
        "5",
        "-1",
    ]

    report = tmp_path / "Mulliken_population.txt"
    report.write_text(
        """Population of atoms:
 Atom     1(Au)    Population: 19.75000000    Net charge:  0.25000000
 Atom     2(O )    Population:  8.25000000    Net charge: -0.25000000
 Total net charge:    0.00000000
""",
        encoding="utf-8",
    )
    assert parse_mulliken_population_report(str(report)) == [
        {"index_1based": 1, "symbol": "Au", "charge_e": 0.25},
        {"index_1based": 2, "symbol": "O", "charge_e": -0.25},
    ]
    assert config[-4].startswith("n ")
    assert [entry.split()[0] for entry in config[-3:]] == ["0", "0", "q"]


def test_multiwfn_mulliken_parser_accepts_spin_table_with_padded_symbols(tmp_path: Path):
    report = tmp_path / "Mulliken_population.txt"
    report.write_text(
        """Population of atoms:
     Atom      Alpha pop.   Beta pop.    Spin pop.     Atomic charge
     1(Au)      9.75000      9.75000      0.00000         0.50000
     2(O )      4.25000      4.25000      0.00000        -0.50000
 Total net charge:  0.00000000      Total spin electrons:  0.00000000
""",
        encoding="utf-8",
    )

    assert parse_mulliken_population_report(str(report)) == [
        {"index_1based": 1, "symbol": "Au", "charge_e": 0.5},
        {"index_1based": 2, "symbol": "O", "charge_e": -0.5},
    ]


@pytest.mark.django_db
def test_ct_ingest_retains_multiwfn_ground_mulliken_on_source_run(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="mulliken_profile")
    workflow = WorkflowDefinition.objects.create(
        name="mulliken_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    source = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au_o_source",
        settings_profile=profile,
    )
    ct_run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="au_o_source_ct",
        settings_profile=profile,
        metadata={"submission": {"run_mode": "charge_transfer"}},
    )
    spectral = ComputedProperty.objects.create(
        workflow_run=source,
        name="spectral_dataset",
        unit="",
        value_text="datasets",
        metadata={"datasets": {"final_tddft": {"ct_terms": [{"term": "1->2", "pct": 12.0}]}}},
    )
    plot_calls = []
    monkeypatch.setattr(
        "ingest.services.write_ct_plot_artifacts", lambda selected: plot_calls.append(selected.pk)
    )

    _ingest_summary_payload(
        run=ct_run,
        summary={
            "source_run_id": source.pk,
            "charge_transfer": {
                "analysis_scope": "ground_state_mulliken",
                "fragment_labels": {"1": "Nanocluster", "2": "O"},
                "ground_state_mulliken": {
                    "program": "Multiwfn",
                    "source_wavefunction": "orca.molden.input",
                    "atoms": [
                        {"index_1based": 1, "symbol": "Au", "charge_e": 0.2},
                        {"index_1based": 2, "symbol": "O", "charge_e": -0.2},
                    ],
                    "fragment_charges_e": {"1": 0.2, "2": -0.2},
                },
            },
        },
    )

    prop = ComputedProperty.objects.get(
        workflow_run=source,
        name="multiwfn_ground_mulliken_charges",
        unit="e",
    )
    assert prop.metadata["charges"] == [0.2, -0.2]
    assert prop.metadata["fragment_charges_e"] == {"1": 0.2, "2": -0.2}
    _store_charges_payload(run=source, charges=[0.1, -0.1], symbols=["Au", "O"],
                          source_artifact=None, charge_method="loewdin")
    default = ComputedProperty.objects.get(workflow_run=source, name="final_charges", unit="e")
    assert default.metadata["charge_method"] == "mulliken"
    assert default.metadata["charges"] == [0.2, -0.2]
    assert ComputedProperty.objects.get(workflow_run=source, name="final_loewdin_charges", unit="e").metadata["charges"] == [0.1, -0.1]
    spectral.refresh_from_db()
    assert spectral.metadata["datasets"]["final_tddft"]["ct_terms"] == [
        {"term": "1->2", "pct": 12.0}
    ]
    assert plot_calls == []


@pytest.mark.django_db
def test_mulliken_only_run_skips_ct_plot_backfill():
    profile = CalculationSettingsProfile.objects.create(name="mulliken_backfill_profile")
    workflow = WorkflowDefinition.objects.create(
        name="mulliken_backfill_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    source = WorkflowRun.objects.create(workflow=workflow, run_name="source", settings_profile=profile)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="mulliken_only",
        settings_profile=profile,
        metadata={
            "result_summary": {
                "source_run_id": source.pk,
                "charge_transfer": {"analysis_scope": "ground_state_mulliken"},
            }
        },
    )

    result = backfill_ct_plots_task.run(run.pk)

    assert result["skipped"] == "ground_state_mulliken_only"
    assert result["artifact_ids"] == []


@pytest.mark.django_db
def test_remote_mulliken_only_ingest_skips_full_output_discovery(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="mulliken_fast_ingest_profile")
    workflow = WorkflowDefinition.objects.create(
        name="mulliken_fast_ingest_workflow",
        code_path="cats/working_directory/tddft_active_learning_workflow.py",
    )
    source = WorkflowRun.objects.create(workflow=workflow, run_name="source", settings_profile=profile)
    run = WorkflowRun.objects.create(
        workflow=workflow,
        run_name="mulliken_only",
        working_directory="/work/user/mulliken_only",
        settings_profile=profile,
        metadata={"submission": {"charge_transfer_config": {"ground_state_mulliken_only": True}}},
    )
    system = HPCSystem.objects.create(name="mulliken_fast_hpc", host="example", username="user")
    job = JobRecord.objects.create(
        workflow_run=run,
        hpc_system=system,
        scheduler_job_id="123",
        job_name="mulliken_only",
    )
    for name in (
        "_download_remote_evaluated_traj_artifacts",
        "_download_remote_scan_endpoints_db_artifacts",
        "_download_remote_scan_plot_artifacts",
        "_retain_spectrum_csv_artifacts",
    ):
        monkeypatch.setattr(
            f"ingest.services.{name}",
            lambda *args, _name=name, **kwargs: (_ for _ in ()).throw(AssertionError(_name)),
        )

    class Client:
        def file_exists(self, path):
            return path == "/work/user/mulliken_only/reaction_hub_summary.json"

        def read_text(self, path):
            return json.dumps(
                {
                    "source_run_id": source.pk,
                    "charge_transfer": {
                        "analysis_scope": "ground_state_mulliken",
                        "fragment_labels": {"1": "NC", "2": "O"},
                        "ground_state_mulliken": {
                            "atoms": [
                                {"index_1based": 1, "symbol": "Au", "charge_e": 0.2},
                                {"index_1based": 2, "symbol": "O", "charge_e": -0.2},
                            ],
                            "fragment_charges_e": {"1": 0.2, "2": -0.2},
                        },
                    },
                }
            )

    assert _ingest_remote_summary_for_job(job, Client()) is True
    assert source.computed_properties.filter(name="multiwfn_ground_mulliken_charges").exists()
