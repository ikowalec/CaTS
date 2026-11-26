from __future__ import annotations

from django.contrib import admin
from django.urls import include, path
from rest_framework.routers import DefaultRouter

from jobs.api import JobRecordViewSet
from results.api import AdsorptionEnergyResultViewSet, ArtifactViewSet, ComputedPropertyViewSet
from workflows.api import WorkflowDefinitionViewSet, WorkflowRunViewSet

router = DefaultRouter()
router.register(r"workflows", WorkflowDefinitionViewSet)
router.register(r"runs", WorkflowRunViewSet)
router.register(r"jobs", JobRecordViewSet)
router.register(r"artifacts", ArtifactViewSet)
router.register(r"properties", ComputedPropertyViewSet)
router.register(r"adsorption-energies", AdsorptionEnergyResultViewSet)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("dashboard.urls")),
    path("campaigns/", include("workflows.campaign_urls")),
    path("jobs/", include("jobs.urls")),
    path("workflows/", include("workflows.urls")),
    path("results/", include("results.urls")),
    path("api/", include(router.urls)),
]
