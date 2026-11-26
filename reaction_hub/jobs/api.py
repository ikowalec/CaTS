from __future__ import annotations

from rest_framework import serializers, viewsets

from .models import HPCSystem, JobRecord, JobStatusEvent


class HPCSystemSerializer(serializers.ModelSerializer):
    class Meta:
        model = HPCSystem
        fields = "__all__"


class JobStatusEventSerializer(serializers.ModelSerializer):
    class Meta:
        model = JobStatusEvent
        fields = "__all__"


class JobRecordSerializer(serializers.ModelSerializer):
    events = JobStatusEventSerializer(many=True, read_only=True)

    class Meta:
        model = JobRecord
        fields = "__all__"


class JobRecordViewSet(viewsets.ModelViewSet):
    queryset = JobRecord.objects.select_related("workflow_run", "hpc_system").all()
    serializer_class = JobRecordSerializer
    filterset_fields = ["status", "workflow_run", "hpc_system"]
    search_fields = ["scheduler_job_id", "job_name", "remote_workdir"]
    ordering_fields = ["created_at", "submitted_at", "completed_at"]
