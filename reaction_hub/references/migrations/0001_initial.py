from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        ("core", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="CalculationSettingsProfile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=120, unique=True)),
                ("approved", models.BooleanField(default=True)),
                ("xc_functional", models.CharField(default="PBE0", max_length=64)),
                ("basis_set", models.CharField(default="6-31g(d,p)", max_length=128)),
                ("solvent", models.CharField(blank=True, default="Water", max_length=64)),
                ("integration_grid", models.CharField(blank=True, default="DefGrid2", max_length=64)),
                ("scf_convergence", models.CharField(blank=True, default="TightSCF", max_length=64)),
                ("metadata", models.JSONField(blank=True, default=dict)),
                ("notes", models.TextField(blank=True)),
            ],
            options={"ordering": ["name"]},
        ),
        migrations.CreateModel(
            name="ReferenceAdsorbate",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=120, unique=True)),
                ("formula", models.CharField(blank=True, max_length=64)),
                ("smiles", models.CharField(blank=True, max_length=256)),
                ("charge", models.IntegerField(default=0)),
                ("multiplicity", models.PositiveIntegerField(default=1)),
                ("structure_path", models.CharField(blank=True, max_length=512)),
                ("notes", models.TextField(blank=True)),
                (
                    "settings_profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="adsorbates",
                        to="references.calculationsettingsprofile",
                    ),
                ),
            ],
            options={"ordering": ["name"]},
        ),
        migrations.CreateModel(
            name="ReferenceNanocluster",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("name", models.CharField(max_length=120, unique=True)),
                ("composition", models.CharField(max_length=120)),
                ("atom_count", models.PositiveIntegerField(blank=True, null=True)),
                ("charge", models.IntegerField(default=0)),
                ("multiplicity", models.PositiveIntegerField(default=1)),
                ("structure_path", models.CharField(blank=True, max_length=512)),
                ("notes", models.TextField(blank=True)),
                (
                    "settings_profile",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.PROTECT,
                        related_name="nanoclusters",
                        to="references.calculationsettingsprofile",
                    ),
                ),
            ],
            options={"ordering": ["name"]},
        ),
    ]
