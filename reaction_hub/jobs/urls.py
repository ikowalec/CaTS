from __future__ import annotations

from django.urls import path

from .views import (
    JobRecordDetailView,
    JobRecordListView,
    JobSubmitView,
    job_bulk_failed_action_view,
    job_mark_delete_view,
    job_mark_failed_delete_view,
    job_mark_all_cancelled_deleted_view,
    job_restart_cancelled_view,
    job_resubmit_failed_view,
    job_resubmit_replot_view,
    job_resubmit_smooth_view,
    job_scancel_delete_view,
    structure_open_ase_gui_view,
    structure_preview_view,
)

urlpatterns = [
    path("", JobRecordListView.as_view(), name="job-list"),
    path("submit/", JobSubmitView.as_view(), name="job-submit"),
    path("submit/structure-preview/", structure_preview_view, name="job-structure-preview"),
    path("submit/structure-open-ase-gui/", structure_open_ase_gui_view, name="job-structure-open-ase-gui"),
    path("mark-cancelled-deleted/", job_mark_all_cancelled_deleted_view, name="job-mark-cancelled-deleted"),
    path("bulk-failed-action/", job_bulk_failed_action_view, name="job-bulk-failed-action"),
    path("<int:pk>/scancel-delete/", job_scancel_delete_view, name="job-scancel-delete"),
    path("<int:pk>/mark-delete/", job_mark_delete_view, name="job-mark-delete"),
    path("<int:pk>/mark-failed-delete/", job_mark_failed_delete_view, name="job-mark-failed-delete"),
    path("<int:pk>/restart-cancelled/", job_restart_cancelled_view, name="job-restart-cancelled"),
    path("<int:pk>/resubmit-failed/", job_resubmit_failed_view, name="job-resubmit-failed"),
    path("<int:pk>/resubmit-replot/", job_resubmit_replot_view, name="job-resubmit-replot"),
    path("<int:pk>/resubmit-smooth/", job_resubmit_smooth_view, name="job-resubmit-smooth"),
    path("<int:pk>/", JobRecordDetailView.as_view(), name="job-detail"),
]
