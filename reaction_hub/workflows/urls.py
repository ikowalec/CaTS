from __future__ import annotations

from django.urls import path

from .views import (
    workflow_delete_duplicate_artifacts_view,
    WorkflowRunDetailView,
    WorkflowRunListView,
    workflow_delete_image_artifact_view,
    workflow_ingest_artifact_view,
    workflow_open_artifact_ase_gui_view,
    workflow_render_artifact_xyz_view,
)

urlpatterns = [
    path("", WorkflowRunListView.as_view(), name="workflow-run-list"),
    path("<int:pk>/", WorkflowRunDetailView.as_view(), name="workflow-run-detail"),
    path(
        "<int:pk>/artifacts/<int:artifact_pk>/open-ase-gui/",
        workflow_open_artifact_ase_gui_view,
        name="workflow-open-artifact-ase-gui",
    ),
    path(
        "<int:pk>/artifacts/<int:artifact_pk>/ingest/",
        workflow_ingest_artifact_view,
        name="workflow-ingest-artifact",
    ),
    path(
        "<int:pk>/artifacts/<int:artifact_pk>/render-xyz/",
        workflow_render_artifact_xyz_view,
        name="workflow-render-artifact-xyz",
    ),
    path(
        "<int:pk>/artifacts/<int:artifact_pk>/delete-image/",
        workflow_delete_image_artifact_view,
        name="workflow-delete-image-artifact",
    ),
    path(
        "<int:pk>/artifacts/delete-duplicates/",
        workflow_delete_duplicate_artifacts_view,
        name="workflow-delete-duplicate-artifacts",
    ),
]
