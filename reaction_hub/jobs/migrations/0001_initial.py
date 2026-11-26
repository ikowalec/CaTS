from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("core", "0001_initial"),
        ("workflows", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="HPCSystem",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "name",
                    models.CharField(max_length=120, unique=True),
                ),
                (
                    "scheduler_type",
                    models.CharField(
                        choices=[("slurm", "Slurm")], default="slurm", max_length=16
                    ),
                ),
                ("host", models.CharField(max_length=255)),
                ("port", models.PositiveIntegerField(default=22)),
                ("username", models.CharField(max_length=120)),
                ("ssh_key_path", models.CharField(blank=True, max_length=512)),
                ("is_active", models.BooleanField(default=True)),
            ],
            options={"ordering": ["name"]},
        ),
        migrations.CreateModel(
            name="JobRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("scheduler_job_id", models.CharField(db_index=True, max_length=64)),
                ("job_name", models.CharField(max_length=180)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("initiated", "Initiated"),
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                            ("cancelled", "Cancelled"),
                            ("unknown", "Unknown"),
                        ],
                        db_index=True,
                        default="initiated",
                        max_length=16,
                    ),
                ),
                ("submitted_at", models.DateTimeField(blank=True, null=True)),
                ("started_at", models.DateTimeField(blank=True, null=True)),
                ("completed_at", models.DateTimeField(blank=True, null=True)),
                ("last_seen_at", models.DateTimeField(blank=True, null=True)),
                ("remote_workdir", models.CharField(blank=True, max_length=512)),
                ("stdout_path", models.CharField(blank=True, max_length=512)),
                ("stderr_path", models.CharField(blank=True, max_length=512)),
                ("exit_code", models.IntegerField(blank=True, null=True)),
                ("scheduler_payload", models.JSONField(blank=True, default=dict)),
                ("failure_reason", models.TextField(blank=True)),
                (
                    "hpc_system",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="jobs",
                        to="jobs.hpcsystem",
                    ),
                ),
                (
                    "workflow_run",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="jobs",
                        to="workflows.workflowrun",
                    ),
                ),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="JobStatusEvent",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("previous_status", models.CharField(blank=True, max_length=16)),
                (
                    "status",
                    models.CharField(
                        choices=[
                            ("initiated", "Initiated"),
                            ("pending", "Pending"),
                            ("running", "Running"),
                            ("completed", "Completed"),
                            ("failed", "Failed"),
                            ("cancelled", "Cancelled"),
                            ("unknown", "Unknown"),
                        ],
                        max_length=16,
                    ),
                ),
                ("message", models.CharField(blank=True, max_length=255)),
                ("payload", models.JSONField(blank=True, default=dict)),
                (
                    "job",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="events",
                        to="jobs.jobrecord",
                    ),
                ),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.AddConstraint(
            model_name="jobrecord",
            constraint=models.UniqueConstraint(
                fields=("hpc_system", "scheduler_job_id"),
                name="jobs_unique_scheduler_id_per_system",
            ),
        ),
    ]
