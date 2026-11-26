from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from ingest.services import ingest_workflow_outputs


class Command(BaseCommand):
    help = "Ingest workflow outputs into reaction_hub database."

    def add_arguments(self, parser):
        parser.add_argument(
            "--workflow-root",
            type=str,
            default=settings.REACTION_HUB_INGEST_ROOT,
            help="Root directory to search for workflow runs.",
        )

    def handle(self, *args, **options):
        workflow_root = Path(options["workflow_root"]).resolve()
        summary = ingest_workflow_outputs(workflow_root)
        self.stdout.write(self.style.SUCCESS(f"Ingestion complete: {summary}"))
