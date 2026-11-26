from pathlib import Path

from django.core.management import call_command
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from ingest.legacy_catalog import import_catalog_tree


class Command(BaseCommand):
    help = "Merge parsed legacy ORCA catalogs into Reaction Hub."

    def add_arguments(self, parser):
        parser.add_argument("--catalog-root", type=Path, required=True)
        parser.add_argument(
            "--source-root",
            type=Path,
            help=(
                "Current root of the legacy source tree. For a catalog tree, each catalog's "
                "relative path is appended to this root."
            ),
        )
        parser.add_argument("--dry-run", action="store_true", help="Validate and roll back all database changes.")

    def handle(self, *args, **options):
        root = options["catalog_root"].resolve()
        source_root = options.get("source_root")
        if not root.is_dir():
            raise CommandError(f"Catalog root is not a directory: {root}")
        try:
            if options["dry_run"]:
                with transaction.atomic():
                    summary = import_catalog_tree(root, current_source_root=source_root)
                    transaction.set_rollback(True)
            else:
                summary = import_catalog_tree(root, current_source_root=source_root)
        except ValueError as exc:
            raise CommandError(str(exc)) from exc
        if not options["dry_run"]:
            call_command("backfill_spectral_data", stdout=self.stdout)
        prefix = "Dry run complete" if options["dry_run"] else "Import complete"
        self.stdout.write(self.style.SUCCESS(f"{prefix}: {summary}"))
