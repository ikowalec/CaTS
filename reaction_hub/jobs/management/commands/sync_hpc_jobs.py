from __future__ import annotations

from django.core.management.base import BaseCommand

from ingest.services import sync_hpc_jobs


class Command(BaseCommand):
    help = "Poll configured HPC schedulers and sync job states into the DB."

    def add_arguments(self, parser):
        parser.add_argument(
            "--force-reingestion",
            "--force-reingest",
            dest="force_reingestion",
            action="store_true",
            help=(
                "Re-run ingestion for completed jobs during sync (useful after transient "
                "connection drops or partially ingested runs)."
            ),
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Emit verbose sync diagnostics to stdout.",
        )
        parser.add_argument(
            "--retry-failed-ingestion",
            action="store_true",
            help=(
                "Retry ingestion for failed jobs. Disabled by default to avoid "
                "extra remote ingestion overhead."
            ),
        )

    def handle(self, *args, **options):
        summary = sync_hpc_jobs(
            force_reingestion=bool(options.get("force_reingestion")),
            retry_failed_ingestion=bool(options.get("retry_failed_ingestion")),
            debug=bool(options.get("debug")),
        )
        self.stdout.write(self.style.SUCCESS(f"Sync complete: {summary}"))
