from __future__ import annotations

import json
import re
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.db.models import Count, Q
from django.db.models.functions import TruncDate
from django.http import Http404, HttpResponseRedirect
from django.urls import reverse
from django.utils import timezone
from django.utils.safestring import mark_safe
from django.views.decorators.http import require_POST
from django.views.generic import TemplateView

from jobs.models import JobRecord, JobStatus
from jobs.tasks import sync_hpc_jobs_task


def detect_job_tags_and_badges(job_name: str = "", run_name: str = "", mode: str = "", metadata: dict = None) -> list[tuple[str, str]]:
    text = f"{job_name} {run_name} {mode}".lower()
    meta_dict = metadata if isinstance(metadata, dict) else {}
    sub_dict = meta_dict.get("submission") if isinstance(meta_dict.get("submission"), dict) else {}
    single_cfg = meta_dict.get("single_constraints_config") or sub_dict.get("single_constraints_config") or {}
    tags_in_meta = set(meta_dict.get("tags") or [])
    if isinstance(sub_dict.get("tags"), list):
        tags_in_meta.update(sub_dict.get("tags"))
    tags_in_meta = {str(t).lower() for t in tags_in_meta}

    badges = []

    # 1. Ground state DFT Optimisation vs Single Point
    if "opt" in text or "single_optimisation" in mode or "optimisation" in tags_in_meta or "dft_optimisation" in tags_in_meta:
        badges.append(("DFT Optimisation", "#10b981"))
    elif "gas_ref" in text or "single_point" in text or "single_point" in mode:
        badges.append(("DFT Single Point", "#059669"))

    # 2. TDDFT Spectrum
    if single_cfg.get("final_tddft_spectrum") or "tddft" in text or "tddft_spectrum" in tags_in_meta:
        badges.append(("TDDFT Spectrum", "#06b6d4"))

    # 3. CDS Spectrum (Cluster Difference Spectrum)
    if single_cfg.get("cluster_tddft_difference_spectrum") or "cds" in text or "cds_spectrum" in tags_in_meta or "cluster_difference" in text:
        badges.append(("CDS Spectrum", "#8b5cf6"))

    # 4. Charge Transfer
    if (
        single_cfg.get("charge_transfer_spectrum")
        or "charge_transfer" in tags_in_meta
        or re.search(r"(?<![a-z0-9])ct(?![a-z0-9])", text)
    ):
        badges.append(("Charge Transfer", "#ec4899"))

    # 5. GPU Survey
    if "survey" in text or "gpu" in text or "coadsorbate_placement" in mode or "gpu_survey" in tags_in_meta:
        badges.append(("GPU Survey", "#6366f1"))

    # 6. MLNEB Path
    if "mlneb" in text or "mep" in text or "active_learning_neb" in mode:
        badges.append(("MLNEB Path", "#3b82f6"))

    if not badges:
        badges.append(("Workflow Run", "#64748b"))

    return badges

def _detect_mode_badge(job_name: str, run_name: str) -> tuple[str, str]:
    badges = detect_job_tags_and_badges(job_name=job_name, run_name=run_name)
    return badges[0]


class DashboardView(TemplateView):
    template_name = "dashboard/index.html"

    def _status_counts(self, selected_campaign=None) -> dict[str, int]:
        qs = JobRecord.objects.all()
        if selected_campaign:
            from workflows.models import CampaignRun
            member_run_ids = CampaignRun.objects.filter(campaign=selected_campaign).values_list("workflow_run_id", flat=True)
            qs = qs.filter(workflow_run_id__in=member_run_ids)

        counts = qs.aggregate(
            total_jobs=Count("id"),
            initiated=Count("id", filter=Q(status=JobStatus.INITIATED)),
            pending=Count("id", filter=Q(status=JobStatus.PENDING)),
            running=Count("id", filter=Q(status=JobStatus.RUNNING)),
            completed=Count("id", filter=Q(status=JobStatus.COMPLETED)),
            failed=Count("id", filter=Q(status=JobStatus.FAILED)),
            cancelled=Count("id", filter=Q(status=JobStatus.CANCELLED)),
            deleted=Count("id", filter=Q(status=JobStatus.DELETED)),
        )
        return {key: int(value or 0) for key, value in counts.items()}

    def _trend(self, selected_campaign=None, days_count: int = 14) -> dict[str, str]:
        end_date = timezone.now().date()
        date_list = [end_date - timedelta(days=i) for i in range(days_count - 1, -1, -1)]
        date_str_list = [d.isoformat() for d in date_list]

        start_dt = timezone.now() - timedelta(days=days_count)
        qs = JobRecord.objects.filter(created_at__gte=start_dt)
        if selected_campaign:
            from workflows.models import CampaignRun
            member_run_ids = CampaignRun.objects.filter(campaign=selected_campaign).values_list("workflow_run_id", flat=True)
            qs = qs.filter(workflow_run_id__in=member_run_ids)

        trend_qs = (
            qs.annotate(day=TruncDate("created_at"))
            .values("day")
            .annotate(
                initiated=Count("id", filter=Q(status=JobStatus.INITIATED)),
                running=Count("id", filter=Q(status=JobStatus.RUNNING)),
                completed=Count("id", filter=Q(status=JobStatus.COMPLETED)),
                failed=Count(
                    "id",
                    filter=Q(status__in=[JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.DELETED]),
                ),
            )
            .order_by("day")
        )
        counts_by_day = {row["day"].isoformat(): row for row in trend_qs if row["day"]}

        initiated = [counts_by_day.get(d, {}).get("initiated", 0) for d in date_str_list]
        running = [counts_by_day.get(d, {}).get("running", 0) for d in date_str_list]
        completed = [counts_by_day.get(d, {}).get("completed", 0) for d in date_str_list]
        failed = [counts_by_day.get(d, {}).get("failed", 0) for d in date_str_list]

        return {
            "labels_json": json.dumps(date_str_list),
            "initiated_json": json.dumps(initiated),
            "running_json": json.dumps(running),
            "completed_json": json.dumps(completed),
            "failed_json": json.dumps(failed),
        }

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Campaign Management Data
        from workflows.models import Campaign, CampaignRun
        campaigns = Campaign.objects.all().prefetch_related("campaign_runs")
        
        selected_slug = str(self.request.GET.get("campaign") or "").strip()
        selected_campaign = campaigns.filter(slug=selected_slug).first() if selected_slug else None

        context["counts"] = self._status_counts(selected_campaign)
        context["trend"] = self._trend(selected_campaign)
        
        # Calculate active Slurm reservation CPU nodes & GPU jobs
        active_running_jobs = JobRecord.objects.filter(
            Q(status__in=[JobStatus.RUNNING, JobStatus.PENDING, JobStatus.INITIATED, "running", "pending", "R", "PD"])
            | Q(workflow_run__status__in=["running", "queued"])
        ).select_related("hpc_system", "workflow_run")

        if selected_campaign:
            member_run_ids = CampaignRun.objects.filter(campaign=selected_campaign).values_list("workflow_run_id", flat=True)
            active_running_jobs = active_running_jobs.filter(workflow_run_id__in=member_run_ids)

        # Count active nodes ONLY for jobs scheduled to a Slurm reservation
        def _is_reservation_job(job: JobRecord) -> bool:
            run = job.workflow_run
            if not run or not isinstance(run.metadata, dict):
                return False
            sub = run.metadata.get("submission") or {}
            res = (sub.get("reservation") if isinstance(sub, dict) else None) or run.metadata.get("reservation") or getattr(job, "reservation", "")
            return bool(str(res or "").strip())

        def _get_node_count(job: JobRecord) -> int:
            run = job.workflow_run
            if not run or not isinstance(run.metadata, dict):
                return 1
            sub = run.metadata.get("submission")
            val = None
            if isinstance(sub, dict) and "nodes" in sub:
                val = sub["nodes"]
            if val is None:
                val = run.metadata.get("nodes")
            try:
                return int(val) if val is not None else 1
            except (ValueError, TypeError):
                return 1

        active_reservation_jobs = [j for j in active_running_jobs if _is_reservation_job(j)]
        active_nodes_count = sum(_get_node_count(j) for j in active_reservation_jobs)
        def _is_gpu_job(job: JobRecord) -> bool:
            if selected_campaign:
                run = job.workflow_run
                if not run or not selected_campaign.campaign_runs.filter(workflow_run_id=run.pk).exists():
                    return False
            run = job.workflow_run
            metadata = run.metadata if run and isinstance(run.metadata, dict) else {}
            submission = metadata.get("submission") if isinstance(metadata.get("submission"), dict) else {}
            partition = str(submission.get("partition") or "").strip().lower()
            qos = str(submission.get("qos") or "").strip().lower()
            try:
                gpus = int(submission.get("gpus") or 0)
            except (TypeError, ValueError):
                gpus = 0
            return partition == "gpu" or qos in {"gpu", "gpu-exc", "gpu-shd"} or gpus > 0

        active_gpu_jobs = sum(1 for job in active_running_jobs if _is_gpu_job(job))


        campaign_summaries = []
        for c in campaigns:
            m_ids = c.campaign_runs.values_list("workflow_run_id", flat=True)
            j_cnt = JobRecord.objects.filter(workflow_run_id__in=m_ids).count()
            total_stage = len(c.checkpoint_ledger) or 6
            completed_stage = sum(1 for item in (c.checkpoint_ledger or []) if item.get("completed"))
            pct = int((completed_stage / total_stage) * 100) if total_stage > 0 else 0
            campaign_summaries.append({
                "instance": c,
                "completed_stage": completed_stage,
                "total_stage": total_stage,
                "completion_pct": pct,
                "member_count": c.campaign_runs.count(),
                "job_count": j_cnt,
            })

        context["campaigns"] = campaign_summaries
        context["selected_campaign"] = selected_campaign
        reservation = next(
            (
                item
                for item in (getattr(settings, "REACTION_HUB_ACTIVE_RESERVATIONS", []) or [])
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ),
            {},
        )
        allocated_nodes = int(reservation.get("nodes_allocated") or reservation.get("nodes_quota") or 0)
        reserved_nodes = int(reservation.get("nodes_reserved") or 0)
        context["active_nodes_count"] = active_nodes_count
        context["reservation_nodes_allocated"] = allocated_nodes
        context["reservation_nodes_reserved"] = reserved_nodes
        context["max_nodes_quota"] = selected_campaign.max_nodes_quota if selected_campaign else max(0, allocated_nodes - reserved_nodes)
        context["active_gpu_jobs"] = active_gpu_jobs
        context["max_gpu_quota"] = selected_campaign.gpu_concurrency_limit if selected_campaign else 1

        # Filter recent completions / attention needed by selected campaign if specified
        base_completed = JobRecord.objects.filter(status=JobStatus.COMPLETED)
        base_attention = JobRecord.objects.filter(status__in=[JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.DELETED])
        
        if selected_campaign:
            member_run_ids = CampaignRun.objects.filter(campaign=selected_campaign).values_list("workflow_run_id", flat=True)
            base_completed = base_completed.filter(workflow_run_id__in=member_run_ids)
            base_attention = base_attention.filter(workflow_run_id__in=member_run_ids)

        # Annotate recent completions and attention needed items with helper UI properties
        completed_jobs = list(base_completed.select_related("workflow_run", "hpc_system").order_by("-completed_at")[:10])
        for j in completed_jobs:
            r_name = j.workflow_run.run_name if j.workflow_run else ""
            meta = j.workflow_run.metadata if j.workflow_run else {}
            mode = str((meta.get("submission") or {}).get("run_mode") if isinstance(meta, dict) else "").lower()
            badges = detect_job_tags_and_badges(j.job_name, r_name, mode, meta)
            setattr(j, "mode_badges", badges)
            setattr(j, "mode_badge_text", badges[0][0])
            setattr(j, "mode_badge_color", badges[0][1])

        failed_jobs = list(base_attention.select_related("workflow_run", "hpc_system").order_by("-updated_at")[:10])
        for j in failed_jobs:
            r_name = j.workflow_run.run_name if j.workflow_run else ""
            meta = j.workflow_run.metadata if j.workflow_run else {}
            mode = str((meta.get("submission") or {}).get("run_mode") if isinstance(meta, dict) else "").lower()
            badges = detect_job_tags_and_badges(j.job_name, r_name, mode, meta)
            setattr(j, "mode_badges", badges)
            setattr(j, "mode_badge_text", badges[0][0])
            setattr(j, "mode_badge_color", badges[0][1])

        context["recent_completed"] = completed_jobs
        context["attention_needed"] = failed_jobs
        return context


def sync_hpc_jobs_view(request):
    try:
        from ingest.services import sync_hpc_jobs
        summary = sync_hpc_jobs()
        checked = summary.get("checked", 0)
        updated = summary.get("updated", 0)
        messages.success(request, f"HPC jobs synced successfully! Checked {checked} jobs, updated {updated}.")
    except Exception as exc:
        messages.error(request, f"HPC sync error: {exc}")
    return HttpResponseRedirect(reverse("dashboard-home"))




def _docs_root() -> Path:
    return (Path(settings.BASE_DIR) / "docs").resolve()


def _read_doc_title(path: Path) -> str:
    try:
        for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = str(raw_line).strip()
            if line.startswith("#"):
                return line.lstrip("#").strip() or path.name
    except OSError:
        pass
    return path.name


def _resolve_docs_path(doc_path: str) -> Path:
    root = _docs_root()
    candidate_rel = Path(doc_path)
    if candidate_rel.is_absolute():
        raise Http404("Invalid documentation path.")
    if candidate_rel.suffix.lower() != ".md":
        raise Http404("Unsupported documentation file type.")

    resolved = (root / candidate_rel).resolve()
    if root not in resolved.parents:
        raise Http404("Documentation path is outside docs root.")
    if not resolved.exists() or not resolved.is_file():
        raise Http404("Documentation file not found.")
    return resolved


def _render_markdown_html(text: str) -> str | None:
    try:
        import markdown
    except Exception:
        return None
    # Optional markdown rendering when the dependency is present.
    return mark_safe(
        markdown.markdown(
            text,
            extensions=["fenced_code", "tables", "sane_lists", "toc"],
            output_format="html5",
        )
    )


class DocsIndexView(TemplateView):
    template_name = "docs/index.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        root = _docs_root()
        docs: list[dict[str, str]] = []
        if root.exists():
            for path in sorted(root.rglob("*.md")):
                if not path.is_file():
                    continue
                rel = path.relative_to(root).as_posix()
                docs.append(
                    {
                        "name": path.name,
                        "path": rel,
                        "title": _read_doc_title(path),
                    }
                )
        context["docs"] = docs
        context["docs_root"] = str(root)
        return context


class DocsDetailView(TemplateView):
    template_name = "docs/detail.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        doc_path = str(kwargs.get("doc_path") or "").strip()
        if not doc_path:
            raise Http404("Documentation path not provided.")
        abs_path = _resolve_docs_path(doc_path)
        source = abs_path.read_text(encoding="utf-8", errors="replace")
        root = _docs_root()
        relative_path = abs_path.relative_to(root).as_posix()
        doc_html = _render_markdown_html(source)
        context["doc_name"] = abs_path.name
        context["doc_path"] = relative_path
        context["doc_source"] = source
        context["doc_html"] = doc_html
        context["markdown_rendered"] = bool(doc_html)
        return context
