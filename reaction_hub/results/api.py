from __future__ import annotations

from rest_framework import serializers, viewsets

from .models import AdsorptionEnergyResult, Artifact, ComputedProperty


class ArtifactSerializer(serializers.ModelSerializer):
    class Meta:
        model = Artifact
        fields = "__all__"


class ComputedPropertySerializer(serializers.ModelSerializer):
    class Meta:
        model = ComputedProperty
        fields = "__all__"


class AdsorptionEnergyResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = AdsorptionEnergyResult
        fields = "__all__"


class ArtifactViewSet(viewsets.ModelViewSet):
    queryset = Artifact.objects.select_related("workflow_run", "job").all()
    serializer_class = ArtifactSerializer
    filterset_fields = ["kind", "workflow_run", "job"]
    search_fields = ["label", "path"]
    ordering_fields = ["created_at", "kind", "label"]


class ComputedPropertyViewSet(viewsets.ModelViewSet):
    queryset = ComputedProperty.objects.select_related("workflow_run", "source_artifact").all()
    serializer_class = ComputedPropertySerializer
    filterset_fields = ["name", "unit", "workflow_run"]
    search_fields = ["name", "value_text"]
    ordering_fields = ["created_at", "name", "value_float"]


class AdsorptionEnergyResultViewSet(viewsets.ModelViewSet):
    queryset = AdsorptionEnergyResult.objects.select_related(
        "workflow_run",
        "adsorbate",
        "nanocluster",
        "settings_profile",
    ).all()
    serializer_class = AdsorptionEnergyResultSerializer
    filterset_fields = ["adsorbate", "nanocluster", "settings_profile", "workflow_run"]
    search_fields = ["workflow_run__run_name", "method_notes"]
    ordering_fields = ["created_at", "adsorption_energy_eV", "total_energy_eV"]
