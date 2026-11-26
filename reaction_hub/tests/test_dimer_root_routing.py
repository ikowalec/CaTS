import pytest

from jobs.models import HPCSystem
from jobs.submission import submit_slurm_workflow_run
from references.models import CalculationSettingsProfile
from workflows.models import WorkflowDefinition, WorkflowRun
from .test_submission import _FakeSSHClient


@pytest.mark.django_db
@pytest.mark.parametrize("root", [5, 11, 17, 22])
def test_dimer_root_and_fragments_survive_normalisation_and_wrapper(root, settings, tmp_path):
    settings.REACTION_HUB_REMOTE_CACHE_ROOT = str(tmp_path)
    profile = CalculationSettingsProfile.objects.create(name="dimer-profile", xc_functional="CAM-B3LYP", basis_set="LANL2DZ", solvent="")
    workflow = WorkflowDefinition.objects.create(name="dimer", code_path="cats/working_directory/active_learning_neb_workflow.py")
    hpc = HPCSystem.objects.create(name="ARCHER2", scheduler_type="slurm", host="example.org", username="test")
    client = _FakeSSHClient()
    fragments = {"1": "1-20", "2": "21", "3": "22"}
    result = submit_slurm_workflow_run(
        workflow=workflow, hpc_system=hpc, settings_profile=profile, run_name="root-test",
        structure_xyz="22\n\n" + "\n".join(f"Cu {i} 0 0" for i in range(20)) + "\nO 21 0 0\nO 23 0 0\n",
        remote_runs_root="/work/test/runs", scratch_base="/work/test/scratch", account="SLURM_ACCOUNT",
        partition="standard", qos="standard", walltime="01:00:00", nodes=1, tasks_per_node=128,
        cpus_per_task=1, module_loads=[], venv_activate_path="/work/test/venv/bin/activate",
        pythonpath_entries=["/work/test/repo"], run_mode="active_learning_neb", qm_backend="orca",
        n_iter=1, use_tddft_optimisation=True,
        neb_config={"backend": "tddft", "final_image_strategy": "dimer", "initial_iroot": root,
                    "tddft_scan_nroots": 32, "target_wavelength_nm": 452.4, "fragments": fragments},
        submitted_by="test", ssh_client_factory=lambda: client,
    )
    run = WorkflowRun.objects.get(pk=result.workflow_run_id)
    assert run.metadata["submission"]["neb_config"]["initial_iroot"] == root
    assert run.metadata["submission"]["neb_config"]["fragments"] == fragments
    wrapper = client.sftp.files_written["/work/test/runs/root-test/run_workflow_submit.py"]
    assert f"initial_iroot={root}," in wrapper
    assert "target_wavelength_nm=452.4," in wrapper
    assert f"fragments={fragments!r}," in wrapper
    compile(wrapper, "run_workflow_submit.py", "exec")
