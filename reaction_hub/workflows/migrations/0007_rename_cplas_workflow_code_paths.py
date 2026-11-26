from __future__ import annotations

from django.db import migrations


def rename_cplas_workflow_code_paths(apps, schema_editor):
    """Migrate WorkflowDefinition.code_path values from cplas/ to cats/."""
    del schema_editor
    WorkflowDefinition = apps.get_model("workflows", "WorkflowDefinition")
    WorkflowRun = apps.get_model("workflows", "WorkflowRun")

    for workflow in list(WorkflowDefinition.objects.order_by("pk")):
        normalized = str(workflow.code_path or "").strip().replace("\\", "/")
        if not normalized.startswith("cplas/"):
            continue
        target_path = "cats/" + normalized[len("cplas/") :]
        existing = (
            WorkflowDefinition.objects.filter(code_path=target_path)
            .exclude(pk=workflow.pk)
            .order_by("pk")
            .first()
        )
        if existing is not None:
            WorkflowRun.objects.filter(workflow_id=workflow.pk).update(workflow_id=existing.pk)
            workflow.delete()
            continue
        workflow.code_path = target_path
        workflow.save(update_fields=["code_path"])


class Migration(migrations.Migration):
    dependencies = [
        ("workflows", "0006_alter_campaign_status"),
    ]

    operations = [
        migrations.RunPython(
            rename_cplas_workflow_code_paths,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
