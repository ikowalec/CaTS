from __future__ import annotations

import django_filters
from django.db.models import OuterRef, Q, Subquery

from results.models import ComputedProperty
from results.services import _formula_counts
from workflows.models import WorkflowRun
from .models import JobRecord


class JobRecordFilter(django_filters.FilterSet):
    scheduler_job_id = django_filters.CharFilter(lookup_expr="icontains")
    job_name = django_filters.CharFilter(lookup_expr="icontains")
    status = django_filters.ChoiceFilter(choices=JobRecord._meta.get_field("status").choices)
    formula = django_filters.CharFilter(method="filter_formula", label="Formula")
    species = django_filters.CharFilter(
        method="filter_species",
        label="Allowed atomic species",
        help_text="Comma- or space-separated; Cu shows structures containing only Cu.",
    )

    @staticmethod
    def _run_formulas(queryset):
        run_ids = queryset.values_list("workflow_run_id", flat=True).distinct()
        return WorkflowRun.objects.filter(pk__in=run_ids).annotate(
            final_formula=Subquery(
                ComputedProperty.objects.filter(
                    workflow_run_id=OuterRef("pk"),
                    name="final_structure_formula",
                )
                .exclude(value_text="")
                .order_by("-updated_at", "-created_at", "-id")
                .values("value_text")[:1]
            )
        )

    def filter_formula(self, queryset, _name, value):
        value = str(value or "").strip()
        if not value:
            return queryset
        runs = self._run_formulas(queryset).filter(
            Q(final_formula__icontains=value)
            | Q(metadata__structure__formula__icontains=value)
            | Q(metadata__result_summary__final__formula__icontains=value)
        )
        return queryset.filter(workflow_run_id__in=runs.values("pk"))

    def filter_species(self, queryset, _name, value):
        allowed = set(str(value or "").replace(",", " ").split())
        if not allowed:
            return queryset
        run_ids = []
        for run in self._run_formulas(queryset):
            metadata = run.metadata if isinstance(run.metadata, dict) else {}
            structure = metadata.get("structure")
            summary = metadata.get("result_summary")
            final = summary.get("final") if isinstance(summary, dict) else None
            formula = run.final_formula
            if not formula and isinstance(structure, dict):
                formula = structure.get("formula", "")
            if not formula and isinstance(final, dict):
                formula = final.get("formula", "")
            counts = _formula_counts(formula)
            if counts and set(counts) <= allowed:
                run_ids.append(run.pk)
        return queryset.filter(workflow_run_id__in=run_ids)

    class Meta:
        model = JobRecord
        fields = ["scheduler_job_id", "job_name", "status", "hpc_system", "formula", "species"]
