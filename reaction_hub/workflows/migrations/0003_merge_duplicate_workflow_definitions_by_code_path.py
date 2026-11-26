from __future__ import annotations

from collections import defaultdict

from django.db import migrations


def _name_priority(name: str) -> tuple[int, str]:
    text = str(name or "").strip()
    if not text:
        return (2, "")
    machine_style = ("_" in text) and all(ch.islower() or ch.isdigit() or ch in {"_", "-"} for ch in text)
    return (1 if machine_style else 0, text.lower())


def merge_duplicate_workflow_definitions_by_code_path(apps, schema_editor):
    del schema_editor
    WorkflowDefinition = apps.get_model("workflows", "WorkflowDefinition")
    WorkflowRun = apps.get_model("workflows", "WorkflowRun")

    grouped: dict[str, list[object]] = defaultdict(list)
    for workflow in WorkflowDefinition.objects.exclude(code_path="").all():
        code_path = str(workflow.code_path or "").strip()
        if not code_path:
            continue
        grouped[code_path.lower()].append(workflow)

    for workflows in grouped.values():
        if len(workflows) < 2:
            continue
        canonical = min(workflows, key=lambda item: (_name_priority(getattr(item, "name", "")), int(item.pk)))
        duplicates = [item for item in workflows if int(item.pk) != int(canonical.pk)]
        for duplicate in duplicates:
            WorkflowRun.objects.filter(workflow_id=duplicate.pk).update(workflow_id=canonical.pk)
            duplicate.delete()


class Migration(migrations.Migration):
    dependencies = [
        ("workflows", "0002_rename_workflows_w_status_0cd270_idx_workflows_w_status_dd17d5_idx_and_more"),
    ]

    operations = [
        migrations.RunPython(
            merge_duplicate_workflow_definitions_by_code_path,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
