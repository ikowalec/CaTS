from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("jobs", "0001_initial"),
    ]

    operations = [
        migrations.AlterField(
            model_name="jobrecord",
            name="status",
            field=models.CharField(
                choices=[
                    ("initiated", "Initiated"),
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                    ("cancelled", "Cancelled"),
                    ("deleted", "Deleted"),
                    ("unknown", "Unknown"),
                ],
                db_index=True,
                default="initiated",
                max_length=16,
            ),
        ),
        migrations.AlterField(
            model_name="jobstatusevent",
            name="status",
            field=models.CharField(
                choices=[
                    ("initiated", "Initiated"),
                    ("pending", "Pending"),
                    ("running", "Running"),
                    ("completed", "Completed"),
                    ("failed", "Failed"),
                    ("cancelled", "Cancelled"),
                    ("deleted", "Deleted"),
                    ("unknown", "Unknown"),
                ],
                max_length=16,
            ),
        ),
    ]
