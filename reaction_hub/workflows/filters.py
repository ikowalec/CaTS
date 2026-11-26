from __future__ import annotations

import django_filters

from .models import WorkflowRun


class WorkflowRunFilter(django_filters.FilterSet):
    run_name = django_filters.CharFilter(lookup_expr="icontains")
    status = django_filters.ChoiceFilter(choices=WorkflowRun._meta.get_field("status").choices)
    workflow = django_filters.NumberFilter(field_name="workflow_id")

    class Meta:
        model = WorkflowRun
        fields = ["run_name", "status", "workflow"]
