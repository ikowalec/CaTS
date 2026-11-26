from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from django.conf import settings
from django.core.paginator import Paginator
from django.db.models import Count, Q
from django.http import Http404
from django.shortcuts import render
from django.views.generic import DetailView, ListView

from dashboard.views import detect_job_tags_and_badges
from jobs.models import HPCSystem, JobRecord, JobStatus
from results.models import Artifact, ArtifactKind
from workflows.models import Campaign, CampaignRun, WorkflowRun, WorkflowStatus


class CampaignListView(ListView):
    model = Campaign
    template_name = "campaigns/campaign_list.html"
    context_object_name = "campaigns"

    def get_queryset(self):
        return Campaign.objects.all().order_by("id")

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        campaigns_data = []

        total_campaign_runs = 0
        total_completed_runs = 0
        all_clusters = defaultdict(int)

        for c in self.get_queryset():
            member_runs = WorkflowRun.objects.filter(campaign_memberships__campaign=c)
            total_runs = member_runs.count()
            total_campaign_runs += total_runs

            status_counts = member_runs.aggregate(
                completed=Count("id", filter=Q(status=WorkflowStatus.COMPLETED)),
                running=Count("id", filter=Q(status=WorkflowStatus.RUNNING)),
                initiated=Count("id", filter=Q(status=WorkflowStatus.INITIATED)),
                failed=Count("id", filter=Q(status=WorkflowStatus.FAILED)),
                cancelled=Count("id", filter=Q(status="cancelled")),
            )

            comp_cnt = status_counts["completed"] or 0
            total_completed_runs += comp_cnt

            ledger = c.checkpoint_ledger or []
            total_stages = len(ledger) or 1
            completed_stages = sum(1 for item in ledger if item.get("completed"))
            progress_pct = int((completed_stages / total_stages) * 100) if total_stages > 0 else 0

            # Target clusters parsing
            clusters = [cl.strip() for cl in (c.target_clusters or "").split(",") if cl.strip()]
            for cl in clusters:
                all_clusters[cl] += total_runs

            # Jobs count
            member_run_ids = member_runs.values_list("id", flat=True)
            jobs_count = JobRecord.objects.filter(workflow_run_id__in=member_run_ids).count()

            campaigns_data.append({
                "campaign": c,
                "total_runs": total_runs,
                "completed_runs": comp_cnt,
                "running_runs": status_counts["running"] or 0,
                "initiated_runs": status_counts["initiated"] or 0,
                "failed_runs": status_counts["failed"] or 0,
                "cancelled_runs": status_counts["cancelled"] or 0,
                "jobs_count": jobs_count,
                "total_stages": total_stages,
                "completed_stages": completed_stages,
                "progress_pct": progress_pct,
                "clusters": clusters,
                "reaction_steps_count": len(c.reaction_scheme or {}),
            })

        context["campaigns_data"] = campaigns_data
        context["total_campaigns"] = len(campaigns_data)
        context["total_campaign_runs"] = total_campaign_runs
        context["total_completed_runs"] = total_completed_runs
        context["overall_progress_pct"] = (
            int((total_completed_runs / total_campaign_runs) * 100) if total_campaign_runs > 0 else 0
        )
        context["top_clusters"] = sorted(all_clusters.items(), key=lambda x: x[1], reverse=True)[:8]

        # Active Slurm reservation status
        reservation = next(
            (
                item
                for item in (getattr(settings, "REACTION_HUB_ACTIVE_RESERVATIONS", []) or [])
                if isinstance(item, dict) and str(item.get("name") or "").strip()
            ),
            {},
        )
        context["reservation_name"] = reservation.get("name", "SLURM_ACCOUNT_RES_3")
        context["reservation_nodes"] = reservation.get("nodes_allocated", 100)
        return context


class CampaignDetailView(DetailView):
    model = Campaign
    template_name = "campaigns/campaign_detail.html"
    context_object_name = "campaign"
    slug_url_kwarg = "slug"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        campaign = self.object

        # 1. Member workflow runs
        member_runs_qs = WorkflowRun.objects.filter(
            campaign_memberships__campaign=campaign
        ).select_related("workflow").order_by("-id")

        total_runs = member_runs_qs.count()
        status_counts = member_runs_qs.aggregate(
            completed=Count("id", filter=Q(status=WorkflowStatus.COMPLETED)),
            running=Count("id", filter=Q(status=WorkflowStatus.RUNNING)),
            initiated=Count("id", filter=Q(status=WorkflowStatus.INITIATED)),
            failed=Count("id", filter=Q(status=WorkflowStatus.FAILED)),
            cancelled=Count("id", filter=Q(status="cancelled")),
        )

        completed_cnt = status_counts["completed"] or 0
        running_cnt = status_counts["running"] or 0
        initiated_cnt = status_counts["initiated"] or 0
        failed_cnt = status_counts["failed"] or 0
        cancelled_cnt = status_counts["cancelled"] or 0
        success_rate = (
            f"{(completed_cnt / (completed_cnt + failed_cnt + cancelled_cnt) * 100):.1f}%"
            if (completed_cnt + failed_cnt + cancelled_cnt) > 0
            else "N/A"
        )

        # 2. Checkpoint stage ledger analysis
        ledger = campaign.checkpoint_ledger or []
        stage_run_counts = defaultdict(int)
        for cr in campaign.campaign_runs.all():
            stage_run_counts[cr.checkpoint_stage] += 1

        ledger_enriched = []
        for item in ledger:
            st_num = item.get("stage", 1)
            is_comp = bool(item.get("completed"))
            r_count = stage_run_counts.get(st_num, 0)
            status_label = "COMPLETED" if is_comp else ("IN PROGRESS" if r_count > 0 else "PENDING")
            ledger_enriched.append({
                "stage": st_num,
                "title": item.get("title", f"Stage {st_num}"),
                "completed": is_comp,
                "status_label": status_label,
                "runs_count": r_count,
            })

        completed_stages_cnt = sum(1 for item in ledger if item.get("completed"))
        total_stages_cnt = len(ledger) or 1
        progress_pct = int((completed_stages_cnt / total_stages_cnt) * 100)

        # 3. System / Cluster Matrix Breakdown
        system_stats = defaultdict(lambda: {
            "completed": 0, "running": 0, "initiated": 0, "failed": 0, "cancelled": 0, "total": 0
        })

        for r in member_runs_qs:
            meta = r.metadata if isinstance(r.metadata, dict) else {}
            struct = meta.get("structure") if isinstance(meta.get("structure"), dict) else {}
            formula = str(struct.get("formula") or "")
            r_name = str(r.run_name or "")

            if "Cu20" in r_name or formula == "Cu20":
                c_label = "Cu20"
            elif "Au20" in r_name or formula == "Au20":
                c_label = "Au20"
            elif "Au63" in r_name or formula == "Au63":
                c_label = "Au63"
            elif "Au55" in r_name or formula == "Au55":
                c_label = "Au55"
            elif "Au13" in r_name or formula == "Au13":
                c_label = "Au13"
            elif "Cu13" in r_name or formula == "Cu13":
                c_label = "Cu13"
            elif "Cu55" in r_name or formula == "Cu55":
                c_label = "Cu55"
            elif "Cu63" in r_name or formula == "Cu63":
                c_label = "Cu63"
            elif any(k in formula or k in r_name for k in ["Au21Cu34", "Au29Cu26", "Au40Cu15", "Au49Cu15", "AuCu_55"]):
                c_label = "AuxCuy (55-atom)"
            elif "Au" in formula and "Cu" in formula:
                c_label = "Other AuxCuy Alloys"
            elif formula:
                c_label = formula
            else:
                c_label = "Other Clusters"

            st_key = r.status.lower()
            if st_key in system_stats[c_label]:
                system_stats[c_label][st_key] += 1
            system_stats[c_label]["total"] += 1

        system_matrix = []
        for sys_name, counts in sorted(system_stats.items(), key=lambda x: x[1]["total"], reverse=True):
            sys_total = counts["total"]
            sys_comp = counts["completed"]
            sys_rate = f"{(sys_comp / sys_total * 100):.1f}%" if sys_total > 0 else "0.0%"
            system_matrix.append({
                "system": sys_name,
                "completed": sys_comp,
                "running": counts["running"],
                "initiated": counts["initiated"],
                "failed": counts["failed"],
                "cancelled": counts["cancelled"],
                "total": sys_total,
                "success_rate": sys_rate,
            })

        # 4. Reaction Scheme Steps Breakdown
        reaction_steps = []
        for step_k, equation in (campaign.reaction_scheme or {}).items():
            step_runs = campaign.campaign_runs.filter(reaction_step_key=step_k)
            step_runs_cnt = step_runs.count()
            reaction_steps.append({
                "key": step_k,
                "equation": equation,
                "runs_count": step_runs_cnt,
            })

        # 5. Computational Jobs Stats
        member_run_ids = member_runs_qs.values_list("id", flat=True)
        jobs_qs = JobRecord.objects.filter(workflow_run_id__in=member_run_ids).select_related("hpc_system")
        total_jobs = jobs_qs.count()

        hpc_counts = defaultdict(int)
        for j in jobs_qs:
            hpc_name = j.hpc_system.name if j.hpc_system else "Unknown HPC"
            hpc_counts[hpc_name] += 1

        # 6. Key Scientific Artifacts Gallery
        # Find visual artifact plots (MEP gifs, UV-Vis spectra, CT diagrams) from member runs
        key_artifacts = Artifact.objects.filter(
            workflow_run_id__in=member_run_ids[:500],
            kind=ArtifactKind.IMAGE,
        ).select_related("workflow_run").order_by("-id")[:12]

        # 7. Paginated Runs Table with badges
        search_query = self.request.GET.get("q", "").strip()
        status_filter = self.request.GET.get("status", "").strip().lower()
        runs_filtered = member_runs_qs

        if search_query:
            runs_filtered = runs_filtered.filter(
                Q(run_name__icontains=search_query) | Q(id__icontains=search_query)
            )
        if status_filter:
            runs_filtered = runs_filtered.filter(status=status_filter)

        paginator = Paginator(runs_filtered, 25)
        page_num = self.request.GET.get("page", 1)
        runs_page = paginator.get_page(page_num)

        # Enrich page objects with badges
        for r in runs_page:
            r.method_badges = detect_job_tags_and_badges(run_name=r.run_name or "", metadata=r.metadata)
            c_mem = r.campaign_memberships.filter(campaign=campaign).first()
            r.campaign_stage = c_mem.checkpoint_stage if c_mem else 1
            r.reaction_step = c_mem.reaction_step_key if c_mem else ""

        # Context packing
        context["total_runs"] = total_runs
        context["completed_cnt"] = completed_cnt
        context["running_cnt"] = running_cnt
        context["initiated_cnt"] = initiated_cnt
        context["failed_cnt"] = failed_cnt
        context["cancelled_cnt"] = cancelled_cnt
        context["success_rate"] = success_rate
        context["progress_pct"] = progress_pct
        context["completed_stages_cnt"] = completed_stages_cnt
        context["total_stages_cnt"] = total_stages_cnt
        context["ledger_enriched"] = ledger_enriched
        context["system_matrix"] = system_matrix
        context["reaction_steps"] = reaction_steps
        context["total_jobs"] = total_jobs
        context["hpc_counts"] = dict(hpc_counts)
        context["key_artifacts"] = key_artifacts
        context["runs_page"] = runs_page
        context["search_query"] = search_query
        context["status_filter"] = status_filter
        context["clusters"] = [cl.strip() for cl in (campaign.target_clusters or "").split(",") if cl.strip()]

        return context
