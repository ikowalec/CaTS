import base64
import json
from pathlib import Path
import pytest
from references.models import CalculationSettingsProfile
from jobs.models import JobRecord, JobStatus, HPCSystem
from workflows.models import WorkflowDefinition, WorkflowRun
from results.models import Artifact
from ingest.services import _ingest_remote_summary_for_job


class _DummySSHClient:
    def __init__(self, remote_files: dict[str, bytes]):
        self.remote_files = remote_files

    def file_exists(self, path: str) -> bool:
        return path in self.remote_files

    def read_text(self, path: str) -> str:
        if path in self.remote_files:
            return self.remote_files[path].decode("utf-8")
        raise FileNotFoundError(path)

    def exec_checked(self, command: str) -> str:
        for path, data in self.remote_files.items():
            if f"base64 < '{path}'" in command or f"base64 < {path}" in command:
                b64 = base64.b64encode(data).decode("ascii")
                return f"__REACTION_HUB_B64_START__\n{b64}\n__REACTION_HUB_B64_END__\n"
        return ""


@pytest.mark.django_db
def test_ingest_retains_pov_and_ini_and_skips_cub(tmp_path: Path, settings, monkeypatch):
    monkeypatch.setattr(settings, "REACTION_HUB_REMOTE_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "REACTION_HUB_INGEST_DOWNLOAD_CUBES", False)

    remote_root = "/work/user/jobs/12345"

    profile = CalculationSettingsProfile.objects.create(name="test_profile")
    hpc = HPCSystem.objects.create(name="TEST_HPC", host="localhost", username="user", is_active=True)
    wf = WorkflowDefinition.objects.create(name="test_wf")
    run = WorkflowRun.objects.create(
        workflow=wf,
        settings_profile=profile,
        run_name="test_density_run",
        working_directory=remote_root,
    )
    job = JobRecord.objects.create(
        hpc_system=hpc,
        workflow_run=run,
        scheduler_job_id="12345",
        status=JobStatus.COMPLETED,
    )

    summary_json = {
        "status": "completed",
        "artifacts": [
            "density_analysis/density_analysis_manifest.json",
            "density_analysis/structure.pov",
            "density_analysis/structure.ini",
            "density_analysis/state_001_cdd.pov",
            "density_analysis/state_001_cdd.ini",
            "density_analysis/state_001_cdd.cub",
        ],
    }

    remote_files = {
        f"{remote_root}/reaction_hub_summary.json": json.dumps(summary_json).encode("utf-8"),
        f"{remote_root}/density_analysis/density_analysis_manifest.json": b"{}",
        f"{remote_root}/density_analysis/structure.pov": b"// pov scene",
        f"{remote_root}/density_analysis/structure.ini": b"Pause_When_Done=False",
        f"{remote_root}/density_analysis/state_001_cdd.pov": b"mesh2 {}",
        f"{remote_root}/density_analysis/state_001_cdd.ini": b"Width=900",
        f"{remote_root}/density_analysis/state_001_cdd.cub": b"BULKY CUBE CONTENT 100MB",
    }

    client = _DummySSHClient(remote_files)
    ingested = _ingest_remote_summary_for_job(job, client)
    assert ingested is True

    # 1. Check Artifact records
    pov_art = Artifact.objects.filter(workflow_run=run, path__endswith=".pov")
    ini_art = Artifact.objects.filter(workflow_run=run, path__endswith=".ini")
    cub_art = Artifact.objects.filter(workflow_run=run, path__endswith=".cub")

    assert pov_art.count() == 2
    assert ini_art.count() == 2
    assert cub_art.count() == 1  # Record exists for lineage

    # 2. Check local file caching: .pov and .ini downloaded, .cub skipped!
    for art in [*pov_art, *ini_art]:
        cache_path_str = art.metadata.get("local_cache_path")
        assert cache_path_str is not None, f"Artifact {art.path} missing local_cache_path"
        cache_path = Path(cache_path_str)
        assert cache_path.is_file(), f"Expected cache file {cache_path} to exist"

    cub_cache_path_str = cub_art.first().metadata.get("local_cache_path")
    assert cub_cache_path_str is None, "Bulky .cub file should NOT have been downloaded or cached!"
