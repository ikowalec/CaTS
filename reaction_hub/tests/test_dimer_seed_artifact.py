from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from jobs import tasks
from results.models import Artifact
from workflows.models import WorkflowRun


def test_dimer_seed_requires_ingested_artifact_without_fallback(monkeypatch, tmp_path):
    source = SimpleNamespace(pk=9323)
    monkeypatch.setattr(WorkflowRun.objects, "select_related", lambda *args: SimpleNamespace(get=lambda **kw: source))
    get = Mock(return_value=SimpleNamespace(metadata={"local_cache_path": str(tmp_path / "missing.xyz")}))
    monkeypatch.setattr(Artifact.objects, "get", get)
    with pytest.raises(ValueError, match="must be ingested before submission"):
        tasks._submit_dimer_refinement(9323, structure_artifact_id=166436)
    get.assert_called_once_with(pk=166436, workflow_run=source, kind__in=["structure", "trajectory"])


def test_dimer_seed_artifact_payload_is_bounded():
    payload = {"source_run_id": 9323, "structure_artifact_id": 166436}
    assert tasks._validate_analysis_request("dimer_refine", payload)["structure_artifact_id"] == 166436
    for invalid in (0, -1, True, "166436"):
        with pytest.raises(ValueError, match="positive artifact ID"):
            tasks._validate_analysis_request("dimer_refine", {**payload, "structure_artifact_id": invalid})
    with pytest.raises(ValueError, match="not both"):
        tasks._validate_analysis_request("dimer_refine", {**payload, "structure_xyz": "stale input"})

