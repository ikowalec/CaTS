from django.db import migrations


def seed_reference_data(apps, schema_editor):
    CalculationSettingsProfile = apps.get_model("references", "CalculationSettingsProfile")
    ReferenceAdsorbate = apps.get_model("references", "ReferenceAdsorbate")
    ReferenceNanocluster = apps.get_model("references", "ReferenceNanocluster")

    profile, _ = CalculationSettingsProfile.objects.get_or_create(
        name="default_pbe0_water",
        defaults={
            "approved": True,
            "xc_functional": "PBE0",
            "basis_set": "6-31g(d,p)",
            "solvent": "Water",
            "integration_grid": "DefGrid2",
            "scf_convergence": "TightSCF",
            "metadata": {"source": "migration_seed"},
        },
    )
    ReferenceAdsorbate.objects.get_or_create(
        name="O2",
        defaults={
            "formula": "O2",
            "charge": 0,
            "multiplicity": 3,
            "settings_profile": profile,
            "notes": "Seed adsorbate.",
        },
    )
    ReferenceNanocluster.objects.get_or_create(
        name="Cu13",
        defaults={
            "composition": "Cu13",
            "atom_count": 13,
            "charge": 0,
            "multiplicity": 1,
            "settings_profile": profile,
            "notes": "Seed nanocluster.",
        },
    )


def unseed_reference_data(apps, schema_editor):
    CalculationSettingsProfile = apps.get_model("references", "CalculationSettingsProfile")
    ReferenceAdsorbate = apps.get_model("references", "ReferenceAdsorbate")
    ReferenceNanocluster = apps.get_model("references", "ReferenceNanocluster")

    ReferenceAdsorbate.objects.filter(name="O2").delete()
    ReferenceNanocluster.objects.filter(name="Cu13").delete()
    CalculationSettingsProfile.objects.filter(name="default_pbe0_water").delete()


class Migration(migrations.Migration):
    dependencies = [
        ("references", "0001_initial"),
    ]

    operations = [
        migrations.RunPython(seed_reference_data, reverse_code=unseed_reference_data),
    ]
