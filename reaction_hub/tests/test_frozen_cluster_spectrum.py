from types import SimpleNamespace

import pytest

from jobs import tasks
from jobs.models import HPCSystem
from references.models import CalculationSettingsProfile
from results.models import ComputedProperty
from workflows.models import WorkflowDefinition, WorkflowRun


@pytest.mark.django_db
def test_frozen_cluster_spectrum_uses_only_source_gold_coordinates(monkeypatch):
    profile = CalculationSettingsProfile.objects.create(name="frozen-spectrum")
    workflow = WorkflowDefinition.objects.create(name="frozen-spectrum", code_path="x.py")
    source = WorkflowRun.objects.create(
        workflow=workflow, settings_profile=profile, run_name="Au20_O2", system_charge=0
    )
    ComputedProperty.objects.create(
        workflow_run=source,
        name="final_structure_snapshot",
        metadata={
            "symbols": ["Au", "Au", "O", "O"],
            "positions_A": [[0, 0, 0], [1, 2, 3], [4, 5, 6], [7, 8, 9]],
        },
    )
    HPCSystem.objects.create(name="FALCON", host="example.org", username="user")
    captured = []
    monkeypatch.setattr(
        "jobs.submission.submit_slurm_workflow_run",
        lambda **kwargs: captured.append(kwargs)
        or SimpleNamespace(workflow_run_id=91, job_record_id=92, scheduler_job_id="93"),
    )

    result = tasks.submit_analysis_request_task.run(
        "frozen_cluster_spectrum", {"source_run_id": source.pk, "system_charge": -1, "nroots": 20}
    )

    assert result["scheduler_job_id"] == "93"
    assert captured[0]["structure_xyz"].splitlines()[0] == "2"
    assert [line.split()[0] for line in captured[0]["structure_xyz"].splitlines()[2:]] == ["Au", "Au"]
    assert captured[0]["spin_multiplicity"] == 2
    assert (captured[0]["partition"], captured[0]["qos"], captured[0]["walltime"]) == (
        "htc_genoa", "", "00:20:00"
    )
    assert captured[0]["use_tddft_optimisation"] is False
    assert captured[0]["single_constraints_config"]["snapshot_single_point"] is True
    assert captured[0]["single_constraints_config"]["final_tddft_nroots"] == 20
    assert captured[0].get("charge_transfer_config") is None


def test_frozen_cluster_spectrum_rejects_unbounded_roots():
    with pytest.raises(ValueError, match="1-40 roots"):
        tasks._validate_analysis_request(
            "frozen_cluster_spectrum", {"source_run_id": 3391, "system_charge": -1, "nroots": 125}
        )
