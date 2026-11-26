from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("workflows", "0003_merge_duplicate_workflow_definitions_by_code_path"),
    ]

    operations = [
        migrations.AddField(
            model_name="workflowrun",
            name="system_charge",
            field=models.IntegerField(default=0),
        ),
    ]
