from __future__ import annotations

from rest_framework import serializers, viewsets

from .models import WorkflowDefinition, WorkflowRun


class WorkflowDefinitionSerializer(serializers.ModelSerializer):
    class Meta:
        model = WorkflowDefinition
        fields = "__all__"


class WorkflowRunSerializer(serializers.ModelSerializer):
    workflow_name = serializers.CharField(source="workflow.name", read_only=True)

    class Meta:
        model = WorkflowRun
        fields = "__all__"


class WorkflowDefinitionViewSet(viewsets.ModelViewSet):
    queryset = WorkflowDefinition.objects.all()
    serializer_class = WorkflowDefinitionSerializer
    filterset_fields = ["name"]
    search_fields = ["name", "description"]
    ordering_fields = ["name", "updated_at"]


class WorkflowRunViewSet(viewsets.ModelViewSet):
    queryset = WorkflowRun.objects.select_related("workflow", "settings_profile").all()
    serializer_class = WorkflowRunSerializer
    filterset_fields = ["status", "workflow", "settings_profile"]
    search_fields = ["run_name", "working_directory", "submitted_by"]
    ordering_fields = ["created_at", "started_at", "completed_at"]
