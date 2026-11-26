from django.db import migrations, models
import django.db.models.deletion
import uuid


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("core", "0001_initial"),
        ("references", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="WorkflowDefinition",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=120, unique=True)),
                ("code_path", models.CharField(blank=True, max_length=512)),
                ("description", models.TextField(blank=True)),
            ],
            options={"ordering": ["name"]},
        ),
        migrations.CreateModel(
            name="WorkflowRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("run_uid", models.UUIDField(default=uuid.uuid4, editable=False, unique=True)),
                ("run_name", models.CharField(max_length=180)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("initiated", "Initiated"),
                            ("running", "Running"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                        ],
                        db_index=True,
                        default="initiated",
                        max_length=16,
                    ),
                ),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("submitted_by", models.CharField(blank=True, max_length=128)),
                ("working_directory", models.CharField(max_length=512)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "reference_adsorbate",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="workflow_runs",
                        to="references.referenceadsorbate",
                    ),
                ),
                (
                    "reference_nanocluster",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="workflow_runs",
                        to="references.referencenanocluster",
                    ),
                ),
                (
                    "settings_profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="workflow_runs",
                        to="references.calculationsettingsprofile",
                    ),
                ),
                (
                    "workflow",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="runs",
                        to="workflows.workflowdefinition",
                    ),
                ),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.AddIndex(
            model_name="workflowrun",
            index=models.Index(fields=["status", "started_at"], name="workflows_w_status_0cd270_idx"),
        ),
        migrations.AddIndex(
            model_name="workflowrun",
            index=models.Index(fields=["status", "completed_at"], name="workflows_w_status_aed8c4_idx"),
        ),
    ]
