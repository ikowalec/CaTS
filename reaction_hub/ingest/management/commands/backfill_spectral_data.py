from django.core.management.base import BaseCommand
from django.db import transaction

from ingest.services import _retain_spectrum_csv_artifacts
from ingest.spectral import (
    broadened_plot_candidate_count,
    ct_plot_dataset_names,
    normalize_ct_dataset_rows,
    update_spectral_dataset,
    write_broadened_plot_artifacts,
    write_ct_plot_artifacts,
)
from results.models import ComputedProperty
from workflows.models import WorkflowRun


class Command(BaseCommand):
    help = "Retain replot-ready numerical spectral data for existing workflow runs."

    def add_arguments(self, parser):
        parser.add_argument("--run-id", type=int, help="Only backfill one workflow run.")
        parser.add_argument("--dry-run", action="store_true", help="Report changes and roll them back.")

    def handle(self, *args, **options):
        runs = WorkflowRun.objects.order_by("pk")
        if options.get("run_id"):
            runs = runs.filter(pk=options["run_id"])
        changed = 0
        plot_candidates = 0
        plots_available = 0
        broadened_candidates = 0
        broadened_available = 0
        with transaction.atomic():
            for run in runs.iterator():
                before = ComputedProperty.objects.filter(
                    workflow_run=run, name="spectral_dataset", unit=""
                ).values_list("metadata", flat=True).first()
                transitions = ComputedProperty.objects.filter(
                    workflow_run=run, name="final_tddft_transitions", unit=""
                ).first()
                if transitions:
                    rows = list((transitions.metadata or {}).get("transitions") or [])
                    if rows:
                        update_spectral_dataset(
                            run=run,
                            dataset_name="final_tddft",
                            source_artifact=transitions.source_artifact,
                            payload={
                                "provenance": {"source": "final_tddft_transitions_backfill"},
                                "transitions": rows,
                                "transition_units": {
                                    "wavenumber_cm": "cm^-1",
                                    "wavelength_nm": "nm",
                                    "oscillator_strength": "dimensionless",
                                },
                            },
                        )
                normalize_ct_dataset_rows(run)
                _retain_spectrum_csv_artifacts(run)
                after = ComputedProperty.objects.filter(
                    workflow_run=run, name="spectral_dataset", unit=""
                ).values_list("metadata", flat=True).first()
                changed += before != after
                candidates = len(ct_plot_dataset_names(run))
                plot_candidates += candidates
                broadened_candidates += broadened_plot_candidate_count(run)
                if not options["dry_run"] and candidates:
                    plots_available += len(write_ct_plot_artifacts(run))
                if not options["dry_run"]:
                    broadened_available += len(write_broadened_plot_artifacts(run))
            if options["dry_run"]:
                transaction.set_rollback(True)
        mode = "Dry run" if options["dry_run"] else "Backfill"
        plot_summary = (
            f"{plot_candidates} CT and {broadened_candidates} TDDFT/cluster plot(s) would be populated"
            if options["dry_run"]
            else f"{plots_available} CT and {broadened_available} TDDFT/cluster plot(s) available"
        )
        self.stdout.write(
            self.style.SUCCESS(f"{mode} complete: {changed} run(s) changed; {plot_summary}.")
        )
