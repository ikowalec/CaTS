from django.contrib import admin

from .models import WorkflowDefinition, WorkflowRun


@admin.register(WorkflowDefinition)
class WorkflowDefinitionAdmin(admin.ModelAdmin):
    list_display = ("name", "code_path", "updated_at")
    search_fields = ("name", "code_path", "description")


@admin.register(WorkflowRun)
class WorkflowRunAdmin(admin.ModelAdmin):
    list_display = (
        "run_name",
        "workflow",
        "status",
        "system_charge",
        "settings_profile",
        "started_at",
        "completed_at",
    )
    list_filter = ("status", "workflow", "settings_profile")
    search_fields = ("run_name", "working_directory", "submitted_by")
