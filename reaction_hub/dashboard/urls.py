from __future__ import annotations

from django.urls import path

from .views import DashboardView, DocsDetailView, DocsIndexView, sync_hpc_jobs_view

urlpatterns = [
    path("docs/", DocsIndexView.as_view(), name="docs-index"),
    path("docs/<path:doc_path>", DocsDetailView.as_view(), name="docs-detail"),
    path("sync-hpc-jobs/", sync_hpc_jobs_view, name="dashboard-sync-hpc-jobs"),
    path("", DashboardView.as_view(), name="dashboard-home"),
]
