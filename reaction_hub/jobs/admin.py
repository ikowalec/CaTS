from django.contrib import admin, messages

from .models import HPCSystem, JobRecord, JobStatus, JobStatusEvent
from references.models import ReferenceAdsorbate
from workflows.models import WorkflowRun


@admin.register(HPCSystem)
class HPCSystemAdmin(admin.ModelAdmin):
    list_display = ("name", "scheduler_type", "host", "username", "is_active")
    list_filter = ("scheduler_type", "is_active")
    search_fields = ("name", "host", "username")


@admin.register(JobRecord)
class JobRecordAdmin(admin.ModelAdmin):
    list_display = (
        "job_name",
        "scheduler_job_id",
        "workflow_run",
        "status",
        "marked_for_deletion",
        "hpc_system",
        "submitted_at",
        "completed_at",
    )
    list_filter = ("status", "hpc_system")
    search_fields = ("job_name", "scheduler_job_id", "remote_workdir")
    actions = (
        "create_reference_adsorbate",
        "mark_selected_for_deletion",
        "unmark_selected_for_deletion",
        "delete_marked_for_deletion",
    )

    def has_delete_permission(self, request, obj=None):
        # Disable direct admin deletion for JobRecord objects to avoid
        # accidental bulk/manual removals. Use explicit lifecycle actions.
        return False

    def get_actions(self, request):
        actions = super().get_actions(request)
        # Keep only explicit safe actions; hide Django's default bulk delete.
        actions.pop("delete_selected", None)
        return actions

    @staticmethod
    def _associated_workflow_runs(queryset):
        run_ids = queryset.values_list("workflow_run_id", flat=True).distinct()
        return WorkflowRun.objects.filter(pk__in=run_ids)

    @admin.display(boolean=True, description="Marked For Deletion")
    def marked_for_deletion(self, obj: JobRecord) -> bool:
        return obj.status == JobStatus.DELETED

    @staticmethod
    def _run_formula(run: WorkflowRun) -> str:
        prop = (
            run.computed_properties.filter(name="final_structure_formula")
            .order_by("-updated_at", "-created_at", "-id")
            .first()
        )
        if prop is not None and prop.value_text:
            return str(prop.value_text).strip()
        metadata = run.metadata if isinstance(run.metadata, dict) else {}
        structure = metadata.get("structure")
        if isinstance(structure, dict):
            return str(structure.get("formula") or "").strip()
        return ""

    def delete_model(self, request, obj: JobRecord):
        if obj.workflow_run_id:
            WorkflowRun.objects.filter(pk=obj.workflow_run_id).delete()
            return
        super().delete_model(request, obj)

    def delete_queryset(self, request, queryset):
        runs = self._associated_workflow_runs(queryset)
        if runs.exists():
            runs.delete()
            return
        super().delete_queryset(request, queryset)

    @admin.action(description="Create reference adsorbate from selected jobs", permissions=["change"])
    def create_reference_adsorbate(self, request, queryset):
        created_count = 0
        linked_count = 0
        skipped_count = 0
        for job in queryset.select_related("workflow_run", "workflow_run__settings_profile"):
            run = job.workflow_run
            name = str(run.run_name or job.job_name or "").strip()
            if not name or run.settings_profile_id is None:
                skipped_count += 1
                continue
            formula = self._run_formula(run)
            adsorbate, created = ReferenceAdsorbate.objects.get_or_create(
                name=name[:120],
                defaults={
                    "formula": formula[:64],
                    "settings_profile": run.settings_profile,
                    "structure_path": str(run.working_directory or "").strip(),
                },
            )
            if adsorbate.settings_profile_id != run.settings_profile_id:
                skipped_count += 1
                continue
            updates = []
            if formula and not adsorbate.formula:
                adsorbate.formula = formula[:64]
                updates.append("formula")
            if run.working_directory and not adsorbate.structure_path:
                adsorbate.structure_path = str(run.working_directory)[:512]
                updates.append("structure_path")
            if updates:
                adsorbate.save(update_fields=[*updates, "updated_at"])
            run.reference_adsorbate = adsorbate
            run.reference_nanocluster = None
            run.save(update_fields=["reference_adsorbate", "reference_nanocluster", "updated_at"])
            created_count += int(created)
            linked_count += int(not created)
        if skipped_count:
            self.message_user(
                request,
                f"Skipped {skipped_count} job(s) without a usable run/profile or with a name/profile conflict.",
                level=messages.WARNING,
            )
        self.message_user(
            request,
            f"Created {created_count} and linked {linked_count} reference adsorbate(s).",
        )

    @admin.action(description="Mark selected jobs for deletion", permissions=["change"])
    def mark_selected_for_deletion(self, request, queryset):
        marked_count = 0
        for job in queryset.exclude(status=JobStatus.DELETED):
            if job.transition_to(
                JobStatus.DELETED,
                message="Marked for deletion from Django admin.",
                payload={"action": "admin_mark_for_deletion"},
            ):
                marked_count += 1
        self.message_user(request, f"Marked {marked_count} selected jobs for deletion.")

    @admin.action(description="Unmark selected jobs for deletion", permissions=["change"])
    def unmark_selected_for_deletion(self, request, queryset):
        unmarked_count = 0
        for job in queryset.filter(status=JobStatus.DELETED):
            event = job.events.filter(status=JobStatus.DELETED).first()
            restored_status = event.previous_status if event and event.previous_status else JobStatus.UNKNOWN
            if job.transition_to(
                restored_status,
                message="Unmarked for deletion from Django admin.",
                payload={"action": "admin_unmark_for_deletion"},
            ):
                unmarked_count += 1
        self.message_user(request, f"Unmarked {unmarked_count} selected jobs for deletion.")

    @admin.action(description="PERMANENTLY delete jobs already marked for deletion", permissions=["change"])
    def delete_marked_for_deletion(self, request, queryset):
        to_delete = queryset.filter(status=JobStatus.DELETED)
        skipped_count = queryset.exclude(status=JobStatus.DELETED).count()
        runs = self._associated_workflow_runs(to_delete)
        run_count = runs.count()
        deleted_count = 0
        if run_count:
            deleted_count, _ = runs.delete()
        if skipped_count:
            self.message_user(
                request,
                f"Skipped {skipped_count} selected jobs because they were not marked for deletion.",
                level=messages.WARNING,
            )
        self.message_user(
            request,
            f"PERMANENTLY deleted {run_count} workflow runs ({deleted_count} total records) marked for deletion.",
        )


@admin.register(JobStatusEvent)
class JobStatusEventAdmin(admin.ModelAdmin):
    list_display = ("job", "previous_status", "status", "message", "created_at")
    list_filter = ("status",)
    search_fields = ("job__scheduler_job_id", "message")
