from types import SimpleNamespace
from unittest.mock import patch, MagicMock
from django.test import SimpleTestCase
from jobs.tasks import _job_cancel, _validate_analysis_request


class JobCancelOperationTests(SimpleTestCase):
    def test_scheduler_id_guard_prevents_remote_action(self):
        job = SimpleNamespace(scheduler_job_id="123", hpc_system=object(), is_terminal=False)
        with patch("jobs.models.JobRecord.objects") as records, patch("jobs.submission._default_ssh_client_factory") as factory:
            records.select_related.return_value.get.return_value = job
            with self.assertRaises(ValueError):
                _job_cancel(job_id=1, expected_scheduler_job_id=124)
            factory.assert_not_called()

    def test_worker_cancellation_preserves_state_until_verified(self):
        job = SimpleNamespace(scheduler_job_id="123", hpc_system=object(), is_terminal=False, status="running")
        client = MagicMock()
        with patch("jobs.models.JobRecord.objects") as records, patch("jobs.models.JobStatusEvent.objects.create") as event, patch("jobs.submission._default_ssh_client_factory", return_value=lambda: client), patch("jobs.submission._exec_checked") as execute:
            records.select_related.return_value.get.return_value = job
            result = _job_cancel(job_id=1, expected_scheduler_job_id=123)
            self.assertEqual(result["status"], "cancellation_requested")
            self.assertEqual(job.status, "running")
            self.assertIn("scancel", execute.call_args.args[1])
            client.close.assert_called_once()
            event.assert_called_once()

    def test_payload_rejects_raw_commands(self):
        with self.assertRaises(ValueError):
            _validate_analysis_request("job_cancel", {"job_id": 1, "expected_scheduler_job_id": 123, "command": "anything"})
