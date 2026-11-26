from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("workflows", "0005_campaign_campaignrun")]

    operations = [
        migrations.AlterField(
            model_name="campaign",
            name="status",
            field=models.CharField(
                choices=[
                    ("draft", "Draft"),
                    ("active", "Active"),
                    ("paused", "Paused"),
                    ("completed", "Completed"),
                    ("archived", "Archived"),
                ],
                db_index=True,
                default="active",
                max_length=32,
            ),
        ),
    ]
