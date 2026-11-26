from unittest.mock import Mock, patch

from django.test import SimpleTestCase
from django.urls import reverse

from .models import JobRecord, JobStatus
from .submission import submit_slurm_workflow_run


class _FakeSSHClient:
    def exec_checked(self, command):
        return "Submitted batch job 12345" if "sbatch " in command else ""

    def upload_text(self, _path, _text):
        pass

    def close(self):
        pass


class MarkJobForDeletionTests(SimpleTestCase):
    @patch.object(JobRecord.objects, "filter")
    def test_marks_completed_job_and_returns_to_workflow_browser(self, filter_jobs):
        job = Mock(pk=7, status=JobStatus.COMPLETED, scheduler_job_id="123")
        filter_jobs.return_value.first.return_value = job

        response = self.client.post(reverse("job-mark-delete", args=[job.pk]), {"next": "/workflows/?page=2"})

        self.assertRedirects(response, "/workflows/?page=2", fetch_redirect_response=False)
        job.transition_to.assert_called_once_with(
            JobStatus.DELETED,
            message="User marked job for deletion from the workflow browser.",
            payload={"requested_by": "", "action": "mark_for_deletion"},
        )


class UntrackedCoadsorbateSubmissionTests(SimpleTestCase):
    @patch("jobs.submission._store_initial_structure_artifact")
    @patch("jobs.submission.JobRecord.objects.create")
    @patch("jobs.submission.WorkflowRun.objects.create")
    def test_submission_creates_no_database_entries(self, create_run, create_job, store_artifact):
        workflow = Mock(code_path="similarity/workflows/coadsorbate_similarity.py")
        hpc = Mock(
            scheduler_type="slurm",
            name="TRH408",
            username="user",
            host="trh",
            port=22,
            ssh_key_path="",
        )
        profile = Mock(pk=1, name="test", xc_functional="pbe0", basis_set="def2-svp", solvent="")

        result = submit_slurm_workflow_run(
            workflow=workflow,
            hpc_system=hpc,
            settings_profile=profile,
            run_name="untracked_mace",
            structure_xyz="1\n\nC 0 0 0\n",
            remote_runs_root="/runs",
            scratch_base="/scratch",
            account="",
            partition="gpu",
            qos="",
            walltime="01:00:00",
            nodes=1,
            tasks_per_node=1,
            cpus_per_task=8,
            module_loads=[],
            venv_activate_path="/venv/bin/activate",
            pythonpath_entries=[],
            n_iter=1,
            use_tddft_optimisation=False,
            run_mode="coadsorbate_placement",
            coadsorbate_config={"adsorbates": [{"xyz": "1\n\nH 0 0 0\n"}]},
            ssh_client_factory=_FakeSSHClient,
        )

        self.assertIsNone(result.workflow_run_id)
        self.assertIsNone(result.job_record_id)
        self.assertEqual(result.scheduler_job_id, "12345")
        create_run.assert_not_called()
        create_job.assert_not_called()
        store_artifact.assert_not_called()
