from __future__ import annotations

from django.urls import path

from .views import (
    AdsorptionEnergyResultDetailView,
    AdsorptionEnergyResultListView,
    ArtifactContentView,
    ArtifactDetailView,
    ArtifactFileView,
    artifact_ingest_view,
    EnergyBasketExportCsvView,
    EnergyBasketView,
    RunEnergyListView,
    ResultsDashboardView,
    ResultsDashboardExportCsvView,
    recompute_adsorption_energies_view,
    result_reprocess_view,
)

urlpatterns = [
    path("", AdsorptionEnergyResultListView.as_view(), name="result-list"),
    path("dashboard/", ResultsDashboardView.as_view(), name="result-dashboard"),
    path("export/dashboard.csv", ResultsDashboardExportCsvView.as_view(), name="result-dashboard-export-csv"),
    path("run-energies/", RunEnergyListView.as_view(), name="result-run-energies"),
    path("energy-dataset/", EnergyBasketView.as_view(), name="result-energy-basket"),
    path("export/energy-dataset.csv", EnergyBasketExportCsvView.as_view(), name="result-energy-basket-export-csv"),
    path("recompute-adsorption/", recompute_adsorption_energies_view, name="result-recompute-adsorption"),
    path("energy/<int:pk>/", AdsorptionEnergyResultDetailView.as_view(), name="result-detail"),
    path("energy/<int:pk>/reprocess/", result_reprocess_view, name="result-reprocess"),
    path("artifact/<int:pk>/", ArtifactDetailView.as_view(), name="artifact-detail"),
    path("artifact/<int:pk>/content/", ArtifactContentView.as_view(), name="artifact-content"),
    path("artifact/<int:pk>/file/", ArtifactFileView.as_view(), name="artifact-file"),
    path("artifact/<int:pk>/ingest/", artifact_ingest_view, name="artifact-ingest"),
]
