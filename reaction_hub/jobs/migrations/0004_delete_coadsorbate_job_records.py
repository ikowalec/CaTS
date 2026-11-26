from django.db import migrations


def delete_coadsorbate_job_records(apps, schema_editor):
    JobRecord = apps.get_model("jobs", "JobRecord")

    job_ids = []
    for job in JobRecord.objects.select_related("workflow_run__workflow").iterator():
        run = job.workflow_run
        metadata = run.metadata if isinstance(run.metadata, dict) else {}
        submission = metadata.get("submission")
        if not isinstance(submission, dict):
            submission = {}
        run_mode = str(submission.get("run_mode") or "").strip()
        code_path = str(getattr(run.workflow, "code_path", "") or "").strip().lower()
        if run_mode == "coadsorbate_placement" or "coadsorbate" in code_path:
            job_ids.append(job.pk)

    if job_ids:
        JobRecord.objects.filter(pk__in=job_ids).delete()


class Migration(migrations.Migration):

    dependencies = [
        ("jobs", "0003_alter_jobstatusevent_options"),
    ]

    operations = [
        migrations.RunPython(delete_coadsorbate_job_records, migrations.RunPython.noop),
    ]
