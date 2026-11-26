from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("core", "0001_initial"),
        ("jobs", "0001_initial"),
        ("references", "0001_initial"),
        ("workflows", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="Artifact",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                (
                    "kind",
                    models.CharField(
                        choices=[
                            ("structure", "Structure"),
                            ("trajectory", "Trajectory"),
                            ("log", "Log"),
                            ("csv", "CSV"),
                            ("json", "JSON"),
                            ("image", "Image"),
                            ("database", "Database"),
                            ("other", "Other"),
                        ],
                        default="other",
                        max_length=20,
                    ),
                ),
                ("label", models.CharField(max_length=120)),
                ("path", models.CharField(max_length=512)),
                ("checksum", models.CharField(blank=True, max_length=128)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "job",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="artifacts",
                        to="jobs.jobrecord",
                    ),
                ),
                (
                    "workflow_run",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="artifacts",
                        to="workflows.workflowrun",
                    ),
                ),
            ],
            options={"ordering": ["kind", "label"]},
        ),
        migrations.CreateModel(
            name="ComputedProperty",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=120)),
                ("value_float", models.FloatField(blank=True, null=True)),
                ("value_text", models.CharField(blank=True, max_length=255)),
                ("unit", models.CharField(blank=True, max_length=32)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "source_artifact",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="computed_properties",
                        to="results.artifact",
                    ),
                ),
                (
                    "workflow_run",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="computed_properties",
                        to="workflows.workflowrun",
                    ),
                ),
            ],
            options={"ordering": ["name"]},
        ),
        migrations.CreateModel(
            name="AdsorptionEnergyResult",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("total_energy_eV", models.DecimalField(decimal_places=6, max_digits=14)),
                ("reference_energy_eV", models.DecimalField(decimal_places=6, max_digits=14)),
                ("adsorption_energy_eV", models.DecimalField(decimal_places=6, max_digits=14)),
                ("method_notes", models.TextField(blank=True)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                (
                    "adsorbate",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="adsorption_results",
                        to="references.referenceadsorbate",
                    ),
                ),
                (
                    "nanocluster",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="adsorption_results",
                        to="references.referencenanocluster",
                    ),
                ),
                (
                    "settings_profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="adsorption_results",
                        to="references.calculationsettingsprofile",
                    ),
                ),
                (
                    "workflow_run",
                    models.OneToOneField(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="adsorption_energy",
                        to="workflows.workflowrun",
                    ),
                ),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.AddConstraint(
            model_name="artifact",
            constraint=models.UniqueConstraint(
                fields=("workflow_run", "path"),
                name="results_unique_artifact_path_per_run",
            ),
        ),
        migrations.AddConstraint(
            model_name="computedproperty",
            constraint=models.UniqueConstraint(
                fields=("workflow_run", "name", "unit"),
                name="results_unique_property_name_unit_per_run",
            ),
        ),
    ]
