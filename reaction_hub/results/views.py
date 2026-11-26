from __future__ import annotations

import csv
import json
import mimetypes
from pathlib import Path
from urllib.parse import urlencode

from django.conf import settings
from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import DecimalField, ExpressionWrapper, F, OuterRef, Q, Subquery
from django.db.models.functions import Cast, Coalesce
from django.http import FileResponse, Http404, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.safestring import mark_safe
from django.views import View
from django.views.decorators.http import require_POST
from django.views.generic import DetailView
from django_filters.views import FilterView
from django_tables2.views import SingleTableMixin

from core.table_pagination import FriendlyTablePaginationMixin
from jobs.models import JobRecord, JobStatus
from references.models import CalculationSettingsProfile
from workflows.identity import canonical_run_path, normalize_run_path
from workflows.models import WorkflowDefinition, WorkflowStatus
from .filters import AdsorptionEnergyResultFilter
from .models import AdsorptionEnergyResult, Artifact, ComputedProperty
from .services import (
    _formula_counts,
    _metadata_adsorbate_display_name,
    _metadata_distortion_energy_ed_eV,
    _metadata_interaction_energy_eint_eV,
    build_results_dashboard,
    recompute_adsorption_energies,
    upsert_adsorption_energy_for_run,
)
from .tables import AdsorptionEnergyResultTable

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
_STRUCTURE_EXTENSIONS = {".xyz", ".traj", ".cif", ".pdb"}
_TEXT_PREVIEW_EXTENSIONS = {
    ".txt",
    ".log",
    ".out",
    ".csv",
    ".json",
    ".xyz",
    ".cif",
    ".pdb",
    ".md",
    ".yaml",
    ".yml",
    ".ini",
    ".toml",
}
_TEXT_PREVIEW_MAX_BYTES = 200_000
_KEY_PLOT_FILES = {
    "stacked_uvvis.png",
    "stacked_ct.png",
    "energy_vs_reaction_coordinate.png",
    "final_tddft_spectrum.png",
}
_RENDER_IMAGE_FILES = {"test.svg", "orca.gif"}
_ENERGY_BASKET_SESSION_KEY = "reaction_hub_energy_basket_run_ids"
_ENERGY_DATA_FILTERS = (
    ("tddft", "TDDFT / spectrum"),
    ("ct", "Charge transfer"),
    ("scan", "Scan trajectory"),
    ("atomization", "Atomisation energy"),
    ("mixing", "Mixing energy"),
    ("adsorption", "Adsorption energy"),
)
_ENERGY_DATASET_COLUMNS = (
    ("run_id", "Run ID"),
    ("run_name", "Run"),
    ("status", "Status"),
    ("formula", "Formula"),
    ("element_a_concentration", "Element A Concentration (%)"),
    ("element_b_concentration", "Element B Concentration (%)"),
    ("workflow", "Workflow"),
    ("settings_profile", "Settings"),
    ("total_energy_eV", "Total Energy (eV)"),
    ("mixing_energy_eV", "Mixing Energy (eV)"),
    ("mixing_energy_per_atom_eV", "Mixing Energy/Atom (eV)"),
    ("atomization_energy_eV", "Atomization Energy (eV)"),
    ("atomization_energy_per_atom_eV", "Atomization Energy/Atom (eV)"),
    ("adsorption_energy_eV", "Adsorption Energy (eV)"),
    ("reference_energy_eV", "Reference Energy (eV)"),
    ("working_directory", "Working Directory"),
)
_DEFAULT_ENERGY_DATASET_COLUMNS = (
    "run_id",
    "run_name",
    "formula",
    "element_a_concentration",
    "element_b_concentration",
    "total_energy_eV",
    "mixing_energy_eV",
    "mixing_energy_per_atom_eV",
    "atomization_energy_eV",
    "atomization_energy_per_atom_eV",
)
_ENERGY_SORTS = {
    "run_id": "workflow_run_id",
    "-run_id": "-workflow_run_id",
    "run": "workflow_run__run_name",
    "-run": "-workflow_run__run_name",
    "run_name": "workflow_run__run_name",
    "-run_name": "-workflow_run__run_name",
    "energy": "value_float",
    "-energy": "-value_float",
    "total_energy_eV": "value_float",
    "-total_energy_eV": "-value_float",
    "status": "workflow_run__status",
    "-status": "-workflow_run__status",
    "formula": "sort_formula",
    "-formula": "-sort_formula",
    "element_a_concentration": "value_float",
    "-element_a_concentration": "-value_float",
    "element_b_concentration": "value_float",
    "-element_b_concentration": "-value_float",
    "workflow": "workflow_run__workflow__name",
    "-workflow": "-workflow_run__workflow__name",
    "settings_profile": "workflow_run__settings_profile__name",
    "-settings_profile": "-workflow_run__settings_profile__name",
    "mixing_energy_eV": "sort_mixing_energy_eV",
    "-mixing_energy_eV": "-sort_mixing_energy_eV",
    "mixing_energy_per_atom_eV": "sort_mixing_energy_per_atom_eV",
    "-mixing_energy_per_atom_eV": "-sort_mixing_energy_per_atom_eV",
    "atomization_energy_eV": "sort_atomization_energy_eV",
    "-atomization_energy_eV": "-sort_atomization_energy_eV",
    "atomization_energy_per_atom_eV": "sort_atomization_energy_per_atom_eV",
    "-atomization_energy_per_atom_eV": "-sort_atomization_energy_per_atom_eV",
    "adsorption_energy_eV": "sort_adsorption_energy_eV",
    "-adsorption_energy_eV": "-sort_adsorption_energy_eV",
    "reference_energy_eV": "sort_reference_energy_eV",
    "-reference_energy_eV": "-sort_reference_energy_eV",
    "working_directory": "workflow_run__working_directory",
    "-working_directory": "-workflow_run__working_directory",
    "created": "workflow_run__created_at",
    "-created": "-workflow_run__created_at",
}


def _energy_basket_run_ids(request) -> list[int]:
    ids = []
    for raw in request.session.get(_ENERGY_BASKET_SESSION_KEY, []):
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(set(ids))


def _set_energy_basket_run_ids(request, run_ids: list[int]) -> None:
    request.session[_ENERGY_BASKET_SESSION_KEY] = sorted(set(int(run_id) for run_id in run_ids))
    request.session.modified = True


def _energy_sort(request, default: str = "run") -> str:
    sort = str(request.GET.get("sort") or default).strip()
    return sort if sort in _ENERGY_SORTS else default


def _run_energy_query_string(**params: str) -> str:
    return urlencode({key: value for key, value in params.items() if value})


def _energy_dataset_columns(request) -> list[str]:
    allowed = {key for key, _label in _ENERGY_DATASET_COLUMNS}
    selected = [key for key in request.GET.getlist("columns") if key in allowed]
    return selected or list(_DEFAULT_ENERGY_DATASET_COLUMNS)


def _energy_species(request) -> list[str]:
    return sorted({item for value in request.GET.getlist("species") for item in value.replace(",", " ").split()})


def _energy_settings_profile_ids(request) -> list[int]:
    ids = []
    for raw in request.GET.getlist("settings_profile") or request.POST.getlist("settings_profile"):
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(set(ids))


def _energy_workflow_ids(request) -> list[int]:
    ids = []
    for raw in request.GET.getlist("workflow") or request.POST.getlist("workflow"):
        try:
            ids.append(int(raw))
        except (TypeError, ValueError):
            continue
    return sorted(set(ids))


def _energy_data_filters(request) -> list[str]:
    allowed = {key for key, _label in _ENERGY_DATA_FILTERS}
    return [key for key in request.GET.getlist("data") if key in allowed]


def _run_ids_with_energy_data(data_type: str):
    if data_type == "tddft":
        return ComputedProperty.objects.filter(
            Q(name__startswith="final_tddft_")
            | Q(
                name="spectral_dataset",
                metadata__datasets__final_tddft__transitions__0__isnull=False,
            )
        ).values("workflow_run_id")
    if data_type == "ct":
        return ComputedProperty.objects.filter(name="spectral_dataset").filter(
            Q(metadata__datasets__final_tddft__ct_states__0__isnull=False)
            | Q(metadata__datasets__final_tddft__ct_spectra__0__isnull=False)
            | Q(metadata__datasets__charge_transfer__ct_states__0__isnull=False)
            | Q(metadata__datasets__charge_transfer__ct_spectra__0__isnull=False)
        ).values("workflow_run_id")
    if data_type == "scan":
        return ComputedProperty.objects.filter(
            name="spectral_dataset",
            metadata__datasets__final_tddft__scan_frames__0__isnull=False,
        ).values("workflow_run_id")
    property_names = {
        "atomization": "atomization_energy_eV",
        "mixing": "mixing_energy_eV",
    }
    if data_type in property_names:
        return ComputedProperty.objects.filter(name=property_names[data_type]).values("workflow_run_id")
    if data_type == "adsorption":
        return AdsorptionEnergyResult.objects.values("workflow_run_id")
    return ComputedProperty.objects.none().values("workflow_run_id")


def _attach_run_energy_availability(rows) -> None:
    run_ids = [row.workflow_run_id for row in rows]
    availability = {run_id: set() for run_id in run_ids}
    for key, _label in _ENERGY_DATA_FILTERS:
        matching = set(_run_ids_with_energy_data(key).filter(workflow_run_id__in=run_ids).values_list(
            "workflow_run_id", flat=True
        ))
        for run_id in matching:
            availability[run_id].add(key)
    labels = dict(_ENERGY_DATA_FILTERS)
    for row in rows:
        row.available_data = [
            {"key": key, "label": labels[key]}
            for key, _label in _ENERGY_DATA_FILTERS
            if key in availability[row.workflow_run_id]
        ]


def _energy_concentration_filters(request) -> dict[str, str]:
    return {
        key: str(request.GET.get(key) or request.POST.get(key) or "").strip()
        for key in (
            "element_a",
            "element_a_min",
            "element_a_max",
            "element_b",
            "element_b_min",
            "element_b_max",
        )
    }


def _energy_dataset_query_string(
    *, sort: str, columns: list[str], filters: dict[str, str] | None = None
) -> str:
    params = [("sort", sort), *((("columns", column) for column in columns))]
    params.extend((key, value) for key, value in (filters or {}).items() if value)
    return urlencode(params)


def _energy_dataset_sort_for_column(column: str, current_sort: str) -> str:
    aliases = {
        "run_name": "run",
        "total_energy_eV": "energy",
    }
    if current_sort == aliases.get(column):
        return f"-{column}"
    if current_sort == f"-{aliases.get(column)}":
        return column
    return f"-{column}" if current_sort == column else column


def _concentration(formula: str, element: str) -> float | None:
    counts = _formula_counts(formula)
    return 100.0 * counts.get(element, 0) / sum(counts.values()) if counts and element else None


def _filter_energy_concentrations(rows, filters: dict[str, str]):
    for prop in rows:
        values = prop.energy_dataset_values
        keep = True
        for slot in ("a", "b"):
            value = _concentration(values["formula"], filters[f"element_{slot}"])
            values[f"element_{slot}_concentration"] = value
            for bound, comparison in (
                ("min", lambda x, y: x >= y),
                ("max", lambda x, y: x <= y),
            ):
                raw = filters[f"element_{slot}_{bound}"]
                try:
                    keep = keep and (not raw or value is not None and comparison(value, float(raw)))
                except ValueError:
                    pass
        if keep:
            yield prop


def _sort_energy_concentrations(rows, sort: str):
    key = sort.removeprefix("-")
    if key not in {"element_a_concentration", "element_b_concentration"}:
        return rows
    descending = sort.startswith("-")
    return sorted(
        rows,
        key=lambda prop: (
            prop.energy_dataset_values[key] is None,
            -prop.energy_dataset_values[key]
            if descending and prop.energy_dataset_values[key] is not None
            else prop.energy_dataset_values[key] or 0.0,
            prop.workflow_run.run_name,
        ),
    )


def _format_dataset_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _run_formula_from_metadata(run) -> str:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    structure = metadata.get("structure")
    if isinstance(structure, dict):
        formula = str(structure.get("formula") or "").strip()
        if formula:
            return formula
    result_summary = metadata.get("result_summary")
    if isinstance(result_summary, dict):
        final_data = result_summary.get("final")
        if isinstance(final_data, dict):
            formula = str(final_data.get("formula") or "").strip()
            if formula:
                return formula
    return ""


def _attach_run_energy_formulas(rows) -> None:
    run_ids = [prop.workflow_run_id for prop in rows]
    formula_by_run: dict[int, str] = {}
    if run_ids:
        formula_rows = (
            ComputedProperty.objects.filter(
                workflow_run_id__in=run_ids,
                name="final_structure_formula",
            )
            .exclude(value_text="")
            .order_by("workflow_run_id", "-updated_at", "-created_at", "-id")
            .values("workflow_run_id", "value_text")
        )
        for row in formula_rows:
            formula_by_run.setdefault(row["workflow_run_id"], str(row["value_text"] or "").strip())

    for prop in rows:
        prop.structure_formula = formula_by_run.get(prop.workflow_run_id) or _run_formula_from_metadata(prop.workflow_run)


def _attach_run_energy_dataset_values(rows) -> None:
    _attach_run_energy_formulas(rows)
    run_ids = [prop.workflow_run_id for prop in rows]
    computed_energies_by_run: dict[int, dict[str, float]] = {}
    if run_ids:
        for row in ComputedProperty.objects.filter(
            workflow_run_id__in=run_ids,
            name__in=[
                "mixing_energy_eV",
                "mixing_energy_per_atom_eV",
                "atomization_energy_eV",
                "atomization_energy_per_atom_eV",
            ],
            value_float__isnull=False,
        ).values("workflow_run_id", "name", "value_float"):
            computed_energies_by_run.setdefault(int(row["workflow_run_id"]), {})[
                str(row["name"])
            ] = float(row["value_float"])
    adsorption_by_run = {
        result.workflow_run_id: result
        for result in AdsorptionEnergyResult.objects.filter(workflow_run_id__in=run_ids)
    }

    for prop in rows:
        run = prop.workflow_run
        adsorption = adsorption_by_run.get(prop.workflow_run_id)
        values = {
            "run_id": run.pk,
            "run_name": run.run_name,
            "status": run.status,
            "formula": getattr(prop, "structure_formula", ""),
            "element_a_concentration": None,
            "element_b_concentration": None,
            "workflow": run.workflow.name,
            "settings_profile": run.settings_profile.name,
            "total_energy_eV": prop.value_float,
            "mixing_energy_eV": computed_energies_by_run.get(prop.workflow_run_id, {}).get(
                "mixing_energy_eV"
            ),
            "mixing_energy_per_atom_eV": computed_energies_by_run.get(prop.workflow_run_id, {}).get(
                "mixing_energy_per_atom_eV"
            ),
            "atomization_energy_eV": computed_energies_by_run.get(prop.workflow_run_id, {}).get(
                "atomization_energy_eV"
            ),
            "atomization_energy_per_atom_eV": computed_energies_by_run.get(
                prop.workflow_run_id, {}
            ).get("atomization_energy_per_atom_eV"),
            "adsorption_energy_eV": (
                float(adsorption.adsorption_energy_eV) if adsorption is not None else None
            ),
            "reference_energy_eV": (
                float(adsorption.reference_energy_eV) if adsorption is not None else None
            ),
            "working_directory": run.working_directory,
        }
        prop.energy_dataset_values = values


def _run_energy_queryset(
    *,
    prefix: str = "",
    formula: str = "",
    settings_profile_ids: list[int] | None = None,
    workflow_ids: list[int] | None = None,
    status: str = "",
    origin: str = "",
    data_filters: list[str] | None = None,
    energy_min: str = "",
    energy_max: str = "",
    run_ids: list[int] | None = None,
    sort: str = "run",
):
    qs = (
        ComputedProperty.objects.select_related(
            "workflow_run",
            "workflow_run__workflow",
            "workflow_run__settings_profile",
        )
        .filter(name="final_single_point_energy_eV", unit="eV", value_float__isnull=False)
    )
    if prefix:
        qs = qs.filter(workflow_run__run_name__startswith=prefix)
    if formula:
        formula_run_ids = ComputedProperty.objects.filter(
            name="final_structure_formula",
            value_text__icontains=formula,
        ).values("workflow_run_id")
        qs = qs.filter(
            Q(workflow_run_id__in=formula_run_ids)
            | Q(workflow_run__metadata__structure__formula__icontains=formula)
            | Q(workflow_run__metadata__result_summary__final__formula__icontains=formula)
        )
    if run_ids is not None:
        qs = qs.filter(workflow_run_id__in=run_ids)
    if settings_profile_ids:
        qs = qs.filter(workflow_run__settings_profile_id__in=settings_profile_ids)
    if workflow_ids:
        qs = qs.filter(workflow_run__workflow_id__in=workflow_ids)
    if status in WorkflowStatus.values:
        qs = qs.filter(workflow_run__status=status)
    if origin == "imported":
        qs = qs.filter(workflow_run__metadata__has_key="legacy_catalog")
    elif origin == "submitted":
        qs = qs.exclude(workflow_run__metadata__has_key="legacy_catalog")
    for data_type in data_filters or []:
        qs = qs.filter(workflow_run_id__in=_run_ids_with_energy_data(data_type))
    try:
        if energy_min:
            qs = qs.filter(value_float__gte=float(energy_min))
        if energy_max:
            qs = qs.filter(value_float__lte=float(energy_max))
    except (TypeError, ValueError):
        pass
    qs = qs.annotate(
        sort_formula=Subquery(
            ComputedProperty.objects.filter(
                workflow_run_id=OuterRef("workflow_run_id"),
                name="final_structure_formula",
            )
            .exclude(value_text="")
            .order_by("-updated_at", "-created_at", "-id")
            .values("value_text")[:1]
        ),
        sort_mixing_energy_eV=Subquery(
            ComputedProperty.objects.filter(
                workflow_run_id=OuterRef("workflow_run_id"),
                name="mixing_energy_eV",
                unit="eV",
                value_float__isnull=False,
            )
            .order_by("-updated_at", "-created_at", "-id")
            .values("value_float")[:1]
        ),
        sort_mixing_energy_per_atom_eV=Subquery(
            ComputedProperty.objects.filter(
                workflow_run_id=OuterRef("workflow_run_id"),
                name="mixing_energy_per_atom_eV",
                unit="eV/atom",
                value_float__isnull=False,
            )
            .order_by("-updated_at", "-created_at", "-id")
            .values("value_float")[:1]
        ),
        sort_atomization_energy_eV=Subquery(
            ComputedProperty.objects.filter(
                workflow_run_id=OuterRef("workflow_run_id"),
                name="atomization_energy_eV",
                unit="eV",
                value_float__isnull=False,
            )
            .order_by("-updated_at", "-created_at", "-id")
            .values("value_float")[:1]
        ),
        sort_atomization_energy_per_atom_eV=Subquery(
            ComputedProperty.objects.filter(
                workflow_run_id=OuterRef("workflow_run_id"),
                name="atomization_energy_per_atom_eV",
                unit="eV/atom",
                value_float__isnull=False,
            )
            .order_by("-updated_at", "-created_at", "-id")
            .values("value_float")[:1]
        ),
        sort_adsorption_energy_eV=Subquery(
            AdsorptionEnergyResult.objects.filter(workflow_run_id=OuterRef("workflow_run_id")).values(
                "adsorption_energy_eV"
            )[:1]
        ),
        sort_reference_energy_eV=Subquery(
            AdsorptionEnergyResult.objects.filter(workflow_run_id=OuterRef("workflow_run_id")).values(
                "reference_energy_eV"
            )[:1]
        ),
    )
    return qs.order_by(_ENERGY_SORTS.get(sort, "workflow_run__run_name"), "workflow_run__run_name")


def _write_run_energy_csv(response: HttpResponse, rows, columns: list[str]) -> HttpResponse:
    writer = csv.writer(response)
    labels = dict(_ENERGY_DATASET_COLUMNS)
    writer.writerow([labels[column] for column in columns])
    for prop in rows:
        values = getattr(prop, "energy_dataset_values", {})
        writer.writerow([_format_dataset_value(values.get(column)) for column in columns])
    return response


def _is_remote_only_artifact(artifact: Artifact) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    return bool(metadata.get("remote_only"))


def _is_remote_sourced_artifact(artifact: Artifact) -> bool:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    source = str(metadata.get("source") or "").strip().lower()
    if source.startswith("remote_"):
        return True
    return bool(
        metadata.get("downloaded_from_hpc")
        or metadata.get("remote_path")
        or metadata.get("remote_only")
    )


def _metadata_local_cache_path(artifact: Artifact) -> Path | None:
    metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
    raw = str(metadata.get("local_cache_path") or "").strip()
    if not raw:
        return None
    return _runtime_path(raw)


def _runtime_path(path_value: str | Path) -> Path:
    normalized = normalize_run_path(path_value)
    if not normalized:
        return Path("")
    direct = Path(normalized)
    if direct.is_absolute():
        return direct
    # Windows drive-prefixed paths are not absolute on Linux; canonicalize.
    canonical = canonical_run_path(normalized)
    canonical_path = Path(canonical)
    if canonical_path.is_absolute():
        return canonical_path
    return direct


def _relative_to_run_root(run_dir_text: str | Path, artifact_path_text: str | Path) -> str | None:
    run_root = canonical_run_path(run_dir_text)
    artifact_path = canonical_run_path(artifact_path_text)
    if not run_root or not artifact_path:
        return None
    if artifact_path == run_root:
        return ""
    prefix = f"{run_root}/"
    if artifact_path.startswith(prefix):
        return artifact_path[len(prefix) :]
    return None


def _safe_cache_relative_path(path_value: str | Path) -> Path:
    normalized = normalize_run_path(path_value).lstrip("/")
    parts = [part for part in Path(normalized).parts if part and part not in {".", ".."}]
    if not parts:
        return Path("artifact.dat")
    return Path(*parts)


def _inferred_local_cache_path(artifact: Artifact) -> Path:
    run = artifact.workflow_run
    run_key = str(getattr(run, "run_uid", "") or f"run_{run.pk}")
    run_root_text = str(run.working_directory or "").strip()
    artifact_path_text = str(artifact.path or "").strip()
    relative_under_run = _relative_to_run_root(run_root_text, artifact_path_text)
    if relative_under_run == "":
        relative_path = _safe_cache_relative_path(Path(artifact_path_text).name or "artifact.dat")
    elif relative_under_run:
        relative_path = _safe_cache_relative_path(relative_under_run)
    else:
        relative_path = _safe_cache_relative_path(artifact_path_text)
    cache_root = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()
    return (cache_root / run_key / relative_path).resolve()


def _artifact_content_available(artifact: Artifact) -> bool:
    if _is_remote_only_artifact(artifact):
        return False
    path = _resolve_artifact_path(artifact)
    return path.exists() and path.is_file()


def _resolve_artifact_path_from_run_root(artifact: Artifact) -> Path:
    run_root_text = str(artifact.workflow_run.working_directory or "").strip()
    run_root = _runtime_path(run_root_text)
    artifact_path_text = str(artifact.path or "").strip()
    relative_under_run = _relative_to_run_root(run_root_text, artifact_path_text)
    if relative_under_run == "":
        return run_root
    if relative_under_run:
        return (run_root / Path(relative_under_run)).resolve()
    artifact_path = _runtime_path(artifact_path_text)
    if artifact_path.is_absolute():
        return artifact_path
    return (run_root / artifact_path).resolve()


def _resolve_artifact_path(artifact: Artifact) -> Path:
    local_cache = _metadata_local_cache_path(artifact)
    remote_sourced = _is_remote_sourced_artifact(artifact)
    if local_cache is not None:
        if local_cache.exists() and local_cache.is_file():
            return local_cache
        if remote_sourced:
            return local_cache

    inferred_cache = _inferred_local_cache_path(artifact)
    if remote_sourced:
        fallback_path = _resolve_artifact_path_from_run_root(artifact)
        if fallback_path.exists() and fallback_path.is_file():
            return fallback_path
        if inferred_cache.exists() and inferred_cache.is_file():
            return inferred_cache
        # Prefer deterministic local cache target over misleading /work -> C:\\work translation.
        return inferred_cache

    if local_cache is not None:
        return local_cache
    return _resolve_artifact_path_from_run_root(artifact)


def _is_image_artifact(artifact: Artifact) -> bool:
    if artifact.kind == "image":
        return True
    return Path(str(artifact.path or "")).suffix.lower() in _IMAGE_EXTENSIONS


def _is_key_plot_artifact(artifact: Artifact) -> bool:
    name = Path(str(artifact.path or "")).name.lower()
    if name in _KEY_PLOT_FILES:
        return True
    return str(artifact.label or "").startswith("plot:")


def _is_render_image_artifact(artifact: Artifact) -> bool:
    name = Path(str(artifact.path or "")).name.lower()
    if name in _RENDER_IMAGE_FILES:
        return True
    return str(artifact.label or "").startswith("render:")


def _is_structure_artifact(artifact: Artifact) -> bool:
    if artifact.kind in {"structure", "trajectory"}:
        return True
    return Path(str(artifact.path or "")).suffix.lower() in _STRUCTURE_EXTENSIONS


def _ase_gui_openable(artifact: Artifact) -> bool:
    if not _is_structure_artifact(artifact):
        return False
    if _is_remote_only_artifact(artifact):
        return False
    return _artifact_content_available(artifact)


def _query_flag_enabled(request, key: str) -> bool:
    raw = str(request.GET.get(key, "")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _resolved_artifact_file_response(artifact: Artifact, *, as_attachment: bool) -> HttpResponse:
    if _is_remote_only_artifact(artifact):
        raise Http404("Artifact is retained on HPC and is not mirrored locally.")

    path = _resolve_artifact_path(artifact)
    if not path.exists() or not path.is_file():
        raise Http404("Artifact file not found")

    suffix = path.suffix.lower()
    if not as_attachment and (suffix in {".traj", ".db"} or artifact.kind == "trajectory"):
        try:
            import io
            from ase.io import read, write

            with path.open("rb") as handle:
                raw_bytes = handle.read()

            idx_ulm = raw_bytes.find(b"- of Ulm")
            idx_af = raw_bytes.find(b"AFFormat")
            if idx_ulm != -1:
                clean_bytes = raw_bytes[idx_ulm:]
            elif idx_af != -1:
                clean_bytes = raw_bytes[idx_af:]
            else:
                clean_bytes = raw_bytes

            try:
                frames = read(io.BytesIO(clean_bytes), format="traj", index=":")
            except Exception:
                frames = read(io.BytesIO(clean_bytes), index=":")

            if frames:
                buf = io.StringIO()
                write(buf, frames, format="xyz")
                return HttpResponse(buf.getvalue(), content_type="text/plain; charset=utf-8")
        except Exception:
            pass

    content_type, _ = mimetypes.guess_type(str(path))
    response = FileResponse(path.open("rb"), content_type=content_type or "application/octet-stream")
    disposition = "attachment" if as_attachment else "inline"
    response["Content-Disposition"] = f'{disposition}; filename="{path.name}"'
    return response


def _is_text_previewable(path: Path, content_type: str | None) -> bool:
    suffix = path.suffix.lower()
    if suffix in _TEXT_PREVIEW_EXTENSIONS:
        return True
    if not content_type:
        return False
    if content_type.startswith("text/"):
        return True
    return content_type in {
        "application/json",
        "application/x-json",
        "application/xml",
        "application/yaml",
        "application/x-yaml",
    }


def _load_text_preview(path: Path) -> tuple[str, bool]:
    size = path.stat().st_size
    truncated = bool(size > _TEXT_PREVIEW_MAX_BYTES)
    if truncated:
        with path.open("rb") as handle:
            payload = handle.read(_TEXT_PREVIEW_MAX_BYTES)
        return payload.decode("utf-8", errors="replace"), True

    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            parsed = json.loads(text)
        except Exception:
            return text, False
        return json.dumps(parsed, indent=2, sort_keys=True), False
    return text, False


class AdsorptionEnergyResultListView(FriendlyTablePaginationMixin, SingleTableMixin, FilterView):
    model = AdsorptionEnergyResult
    table_class = AdsorptionEnergyResultTable
    template_name = "results/result_list.html"
    filterset_class = AdsorptionEnergyResultFilter

    def get_queryset(self):
        base_queryset = (
            super()
            .get_queryset()
            .exclude(workflow_run__metadata__submission__run_mode="coadsorbate_placement")
            .exclude(workflow_run__workflow__code_path__icontains="coadsorbate")
        )

        def _decimal_output():
            return DecimalField(max_digits=18, decimal_places=6)

        ed_from_cluster = ExpressionWrapper(
            Cast(F("metadata__cluster_final_energy_eV"), output_field=_decimal_output())
            - Cast(F("metadata__cluster_initial_energy_eV"), output_field=_decimal_output()),
            output_field=_decimal_output(),
        )
        ed_energy = Coalesce(
            ed_from_cluster,
            Cast(F("metadata__ed_energy_eV"), output_field=_decimal_output()),
            Cast(F("metadata__cluster_distortion_energy_eV"), output_field=_decimal_output()),
            output_field=_decimal_output(),
        )
        eint_energy = Coalesce(
            Cast(F("metadata__eint_energy_eV"), output_field=_decimal_output()),
            Cast(F("metadata__adsorbate_cluster_interaction_energy_eV"), output_field=_decimal_output()),
            ExpressionWrapper(F("adsorption_energy_eV") - ed_energy, output_field=_decimal_output()),
            output_field=_decimal_output(),
        )
        return base_queryset.annotate(
            ed_energy_value=ed_energy,
            eint_energy_value=eint_energy,
            eint_energy_order=eint_energy,
        )


def _parse_dashboard_limit(raw_limit: str | None, *, default: int) -> int:
    try:
        limit = int(str(raw_limit).strip())
    except (TypeError, ValueError):
        return default
    return max(1, min(limit, 20000))


class ResultsDashboardView(View):
    template_name = "results/result_dashboard.html"

    def get(self, request):
        limit = _parse_dashboard_limit(request.GET.get("limit"), default=5000)
        context = {
            "dashboard": build_results_dashboard(limit=limit),
            "dashboard_limit": limit,
        }
        return render(request, self.template_name, context)


class ResultsDashboardExportCsvView(View):
    def get(self, request):
        limit = _parse_dashboard_limit(request.GET.get("limit"), default=5000)

        dashboard = build_results_dashboard(limit=limit)
        rows = dashboard.get("rows", [])

        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=reaction_hub_results_dashboard.csv"
        writer = csv.writer(response)
        writer.writerow(
            [
                "run_id",
                "run_name",
                "status",
                "settings_profile",
                "system_formula",
                "nanocluster",
                "adsorbate",
                "system_energy_eV",
                "reference_energy_eV",
                "adsorption_energy_eV",
                "ed_energy_eV",
                "eint_energy_eV",
                "bond_scan_start_energy_eV",
                "bond_scan_peak_energy_eV",
                "bond_scan_activation_barrier_eV",
            ]
        )

        for row in rows:
            writer.writerow(
                [
                    row.get("run_id"),
                    row.get("run_name"),
                    row.get("status"),
                    row.get("settings_profile"),
                    row.get("system_formula"),
                    row.get("nanocluster_name"),
                    row.get("adsorbate_name"),
                    row.get("system_energy_eV"),
                    row.get("reference_energy_eV"),
                    row.get("adsorption_energy_eV"),
                    row.get("ed_energy_eV"),
                    row.get("eint_energy_eV"),
                    row.get("bond_scan_start_energy_eV"),
                    row.get("bond_scan_peak_energy_eV"),
                    row.get("bond_scan_activation_barrier_eV"),
                ]
            )
        return response


class RunEnergyListView(View):
    template_name = "results/run_energy_list.html"

    def get(self, request):
        prefix = str(request.GET.get("prefix") or "").strip()
        formula = str(request.GET.get("formula") or "").strip()
        species = _energy_species(request)
        settings_profile_ids = _energy_settings_profile_ids(request)
        workflow_ids = _energy_workflow_ids(request)
        status = str(request.GET.get("status") or "").strip()
        origin = str(request.GET.get("origin") or "").strip()
        data_filters = _energy_data_filters(request)
        energy_min = str(request.GET.get("energy_min") or "").strip()
        energy_max = str(request.GET.get("energy_max") or "").strip()
        sort = _energy_sort(request)
        rows = list(
            _run_energy_queryset(
                prefix=prefix,
                formula=formula,
                settings_profile_ids=settings_profile_ids,
                workflow_ids=workflow_ids,
                status=status,
                origin=origin,
                data_filters=data_filters,
                energy_min=energy_min,
                energy_max=energy_max,
                sort=sort,
            )
        )
        _attach_run_energy_formulas(rows)
        if species:
            allowed = set(species)
            rows = [
                row
                for row in rows
                if (counts := _formula_counts(row.structure_formula)) and set(counts) <= allowed
            ]
        result_count = len(rows)
        page_obj = Paginator(rows, 100).get_page(request.GET.get("page"))
        rows = list(page_obj.object_list)
        _attach_run_energy_availability(rows)
        basket_ids = set(_energy_basket_run_ids(request))
        query_without_sort = request.GET.copy()
        query_without_sort.pop("sort", None)
        query_without_sort.pop("page", None)
        query_without_page = request.GET.copy()
        query_without_page.pop("page", None)
        settings_profiles = list(
            CalculationSettingsProfile.objects.filter(
                workflow_runs__computed_properties__name="final_single_point_energy_eV"
            ).distinct().order_by("name")
        )
        workflows = list(
            WorkflowDefinition.objects.filter(
                runs__computed_properties__name="final_single_point_energy_eV"
            ).distinct().order_by("name")
        )
        active_filters = []
        if prefix:
            active_filters.append(f"Run starts with {prefix}")
        if formula:
            active_filters.append(f"Formula contains {formula}")
        if species:
            active_filters.append(f"Species: {', '.join(species)} only")
        if status:
            active_filters.append(f"Status: {status}")
        if origin:
            active_filters.append(f"Origin: {origin}")
        selected_profile_names = [
            profile.name for profile in settings_profiles if profile.pk in settings_profile_ids
        ]
        if selected_profile_names:
            active_filters.append(f"Settings: {', '.join(selected_profile_names)}")
        selected_workflow_names = [
            workflow.name for workflow in workflows if workflow.pk in workflow_ids
        ]
        if selected_workflow_names:
            active_filters.append(f"Workflow: {', '.join(selected_workflow_names)}")
        if energy_min or energy_max:
            active_filters.append(f"Energy: {energy_min or '−∞'} to {energy_max or '∞'} eV")
        active_filters.extend(dict(_ENERGY_DATA_FILTERS)[key] for key in data_filters)
        return render(
            request,
            self.template_name,
            {
                "rows": rows,
                "prefix": prefix,
                "formula": formula,
                "species": species,
                "settings_profile_ids": settings_profile_ids,
                "settings_profiles": settings_profiles,
                "workflow_ids": workflow_ids,
                "workflows": workflows,
                "status": status,
                "status_choices": WorkflowStatus.choices,
                "origin": origin,
                "data_filters": data_filters,
                "data_filter_choices": _ENERGY_DATA_FILTERS,
                "energy_min": energy_min,
                "energy_max": energy_max,
                "sort": sort,
                "basket_ids": basket_ids,
                "basket_count": len(basket_ids),
                "result_count": result_count,
                "page_obj": page_obj,
                "active_filters": active_filters,
                "query_without_sort": query_without_sort.urlencode(),
                "query_without_page": query_without_page.urlencode(),
                "return_query": request.GET.urlencode(),
            },
        )

    def post(self, request):
        selected = []
        for raw in request.POST.getlist("run_ids"):
            try:
                selected.append(int(raw))
            except (TypeError, ValueError):
                continue
        if selected:
            _set_energy_basket_run_ids(request, _energy_basket_run_ids(request) + selected)
            messages.success(request, f"Added {len(set(selected))} run(s) to the energy dataset.")
        else:
            messages.warning(request, "No runs were selected.")
        target = reverse("result-run-energies")
        query = str(request.POST.get("return_query") or "").lstrip("?")
        if not query:
            query = urlencode(
                [
                    *((key, request.POST.get(key)) for key in (
                        "prefix", "formula", "status", "origin", "energy_min", "energy_max", "sort"
                    ) if request.POST.get(key)),
                    *(("species", value) for value in request.POST.getlist("species") if value),
                    *(("settings_profile", value) for value in request.POST.getlist("settings_profile") if value),
                    *(("workflow", value) for value in request.POST.getlist("workflow") if value),
                    *(("data", value) for value in request.POST.getlist("data") if value),
                ]
            )
        return redirect(f"{target}?{query}" if query else target)


class EnergyBasketView(View):
    template_name = "results/energy_basket.html"

    def get(self, request):
        sort = _energy_sort(request, default="energy")
        columns = _energy_dataset_columns(request)
        filters = _energy_concentration_filters(request)
        rows = list(_run_energy_queryset(run_ids=_energy_basket_run_ids(request), sort=sort))
        _attach_run_energy_dataset_values(rows)
        rows = list(_filter_energy_concentrations(rows, filters))
        rows = _sort_energy_concentrations(rows, sort)
        labels = dict(_ENERGY_DATASET_COLUMNS)
        for prop in rows:
            values = getattr(prop, "energy_dataset_values", {})
            prop.energy_dataset_cells = [
                {"key": column, "label": labels[column], "value": _format_dataset_value(values.get(column))}
                for column in columns
            ]
        return render(
            request,
            self.template_name,
            {
                "rows": rows,
                "sort": sort,
                "columns": columns,
                "filters": filters,
                "column_options": [
                    {
                        "key": key,
                        "label": label,
                        "selected": key in columns,
                        "sort": _energy_dataset_sort_for_column(key, sort),
                    }
                    for key, label in _ENERGY_DATASET_COLUMNS
                ],
                "column_query": urlencode([
                    *(("columns", column) for column in columns),
                    *((key, value) for key, value in filters.items() if value),
                ]),
                "dataset_query": _energy_dataset_query_string(
                    sort=sort, columns=columns, filters=filters
                ),
            },
        )

    def post(self, request):
        action = str(request.POST.get("action") or "").strip()
        columns = [key for key in request.POST.getlist("columns") if key in dict(_ENERGY_DATASET_COLUMNS)]
        sort = str(request.POST.get("sort") or "energy").strip()
        if sort not in _ENERGY_SORTS:
            sort = "energy"
        target = reverse("result-energy-basket")
        filters = _energy_concentration_filters(request)
        query = _energy_dataset_query_string(
            sort=sort,
            columns=columns or list(_DEFAULT_ENERGY_DATASET_COLUMNS),
            filters=filters,
        )
        if action == "clear":
            _set_energy_basket_run_ids(request, [])
            messages.success(request, "Cleared the energy dataset.")
            return redirect(f"{target}?{query}")

        remove_ids = []
        for raw in request.POST.getlist("run_ids"):
            try:
                remove_ids.append(int(raw))
            except (TypeError, ValueError):
                continue
        if remove_ids:
            kept = [run_id for run_id in _energy_basket_run_ids(request) if run_id not in set(remove_ids)]
            _set_energy_basket_run_ids(request, kept)
            messages.success(request, f"Removed {len(set(remove_ids))} run(s) from the energy dataset.")
        else:
            messages.warning(request, "No runs were selected.")
        return redirect(f"{target}?{query}")


class EnergyBasketExportCsvView(View):
    def get(self, request):
        sort = _energy_sort(request, default="energy")
        columns = _energy_dataset_columns(request)
        filters = _energy_concentration_filters(request)
        rows = list(_run_energy_queryset(run_ids=_energy_basket_run_ids(request), sort=sort))
        _attach_run_energy_dataset_values(rows)
        rows = list(_filter_energy_concentrations(rows, filters))
        rows = _sort_energy_concentrations(rows, sort)
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = "attachment; filename=reaction_hub_energy_dataset.csv"
        return _write_run_energy_csv(response, rows, columns)


@require_POST
def recompute_adsorption_energies_view(request):
    raw_limit = str(request.POST.get("limit", "")).strip()
    limit: int | None = None
    if raw_limit:
        try:
            parsed = int(raw_limit)
        except ValueError:
            messages.error(request, "Eads recompute: limit must be an integer.")
            return redirect("result-dashboard")
        if parsed <= 0:
            messages.error(request, "Eads recompute: limit must be a positive integer.")
            return redirect("result-dashboard")
        limit = parsed

    summary = recompute_adsorption_energies(limit=limit)
    messages.success(
        request,
        (
            "Eads recompute finished. "
            f"Checked {summary['inspected']} runs; "
            f"created {summary['created']}, updated {summary['updated']}, "
            f"no match {summary['no_match']}, errors {summary['errors']}."
        ),
    )
    return redirect("result-dashboard")


def _job_marked_failed_for_missing_properties(job: JobRecord) -> bool:
    reason_text = str(job.failure_reason or "").strip().lower()
    if "no computed properties were ingested" in reason_text:
        return True
    latest_failed_event = (
        job.events.filter(status=JobStatus.FAILED)
        .order_by("-created_at", "-id")
        .first()
    )
    if latest_failed_event is None:
        return False
    payload = latest_failed_event.payload if isinstance(latest_failed_event.payload, dict) else {}
    return str(payload.get("rule") or "").strip() == "completed_requires_computed_properties"


def _recover_false_failed_completion_status(result: AdsorptionEnergyResult) -> bool:
    run = result.workflow_run
    if not run.computed_properties.exists():
        return False

    job = (
        JobRecord.objects.filter(workflow_run=run)
        .order_by("-created_at", "-id")
        .first()
    )
    if job is None or job.status != JobStatus.FAILED:
        return False
    if not _job_marked_failed_for_missing_properties(job):
        return False

    status_changed = job.transition_to(
        JobStatus.COMPLETED,
        message="Recovered completed status via manual result reprocess.",
        payload={
            "trigger": "result_reprocess",
            "rule": "recover_completed_from_properties",
            "result_id": int(result.pk),
        },
    )
    reason_changed = False
    if job.failure_reason:
        job.failure_reason = ""
        job.save(update_fields=["failure_reason", "updated_at"])
        reason_changed = True

    run_changed = False
    if run.status != WorkflowStatus.COMPLETED:
        run.status = WorkflowStatus.COMPLETED
        run.save(update_fields=["status", "updated_at"])
        run_changed = True

    return status_changed or reason_changed or run_changed


@require_POST
def result_reprocess_view(request, pk: int):
    result = (
        AdsorptionEnergyResult.objects.select_related("workflow_run")
        .filter(pk=pk)
        .first()
    )
    if result is None:
        messages.error(request, "Result was not found.")
        return redirect("result-list")

    run = result.workflow_run
    run_path = Path(run.working_directory)
    ingest_error = ""
    ingested = False
    if run_path.exists():
        try:
            from ingest.services import ingest_single_run

            ingest_single_run(run_path)
            ingested = True
        except Exception as exc:
            ingest_error = str(exc)
    else:
        ingest_error = f"Run directory is not available locally: {run_path}"

    recompute_error = ""
    recomputed = False
    try:
        recomputed = upsert_adsorption_energy_for_run(run) is not None
    except Exception as exc:
        recompute_error = str(exc)

    recovered_status = _recover_false_failed_completion_status(result)

    if recompute_error:
        messages.error(request, f"Reprocess failed while recomputing adsorption energy: {recompute_error}")
    elif ingest_error and not recomputed and not recovered_status:
        messages.warning(request, f"Reprocess did not ingest new data. {ingest_error}")
    elif ingest_error:
        messages.warning(
            request,
            (
                "Reprocess completed using existing ingested data."
                f" Ingestion warning: {ingest_error}"
            ),
        )
    else:
        messages.success(request, "Reprocess completed and result metadata was refreshed.")

    return redirect("result-detail", pk=result.pk)


class AdsorptionEnergyResultDetailView(DetailView):
    model = AdsorptionEnergyResult
    template_name = "results/result_detail.html"
    context_object_name = "result"

    def _build_structure_viewer_html(self, structure_artifact: Artifact) -> str:
        artifact_path = _resolve_artifact_path(structure_artifact)
        if not artifact_path.exists():
            return "<p>Structure file missing on disk.</p>"

        fmt = artifact_path.suffix.lower().lstrip(".")
        if fmt not in {"xyz", "cif", "pdb"}:
            return "<p>Unsupported structure format for viewer.</p>"

        try:
            data = artifact_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"<p>Failed to read structure: {exc}</p>"

        try:
            import py3Dmol  # type: ignore

            viewer = py3Dmol.view(width=700, height=420)
            viewer.addModel(data, fmt)
            viewer.setStyle({"stick": {}})
            viewer.zoomTo()
            return viewer._make_html()  # noqa: SLF001
        except Exception:
            escaped = data[:5000].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f"<pre>{escaped}</pre>"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        run = self.object.workflow_run
        structure_artifact = (
            Artifact.objects.filter(workflow_run=run, kind="structure").order_by("id").first()
        )
        artifacts = list(Artifact.objects.filter(workflow_run=run).order_by("kind", "label"))
        image_artifacts = [
            artifact
            for artifact in artifacts
            if _is_image_artifact(artifact) and _artifact_content_available(artifact)
        ]
        key_plot_artifacts = [artifact for artifact in image_artifacts if _is_key_plot_artifact(artifact)]
        render_image_artifacts = [
            artifact for artifact in image_artifacts if _is_render_image_artifact(artifact)
        ]
        context["artifacts"] = artifacts
        context["image_artifacts"] = image_artifacts
        context["image_artifact_ids"] = [artifact.pk for artifact in image_artifacts]
        context["key_plot_artifacts"] = key_plot_artifacts
        context["render_image_artifacts"] = render_image_artifacts
        context["content_available_artifact_ids"] = [
            artifact.pk for artifact in artifacts if _artifact_content_available(artifact)
        ]
        context["ase_openable_artifact_ids"] = [artifact.pk for artifact in artifacts if _ase_gui_openable(artifact)]
        context["properties"] = run.computed_properties.all().order_by("name")
        context["adsorbate_display_name"] = _metadata_adsorbate_display_name(self.object)
        ed_energy_eV = _metadata_distortion_energy_ed_eV(self.object)
        context["ed_energy_eV"] = ed_energy_eV
        context["has_ed_energy"] = ed_energy_eV is not None
        eint_energy_eV = _metadata_interaction_energy_eint_eV(self.object)
        context["eint_energy_eV"] = eint_energy_eV
        context["has_eint_energy"] = eint_energy_eV is not None
        context["structure_artifact"] = structure_artifact
        if structure_artifact:
            context["structure_viewer_html"] = mark_safe(
                self._build_structure_viewer_html(structure_artifact)
            )
        else:
            context["structure_viewer_html"] = "<p>No structure artifact linked to this run.</p>"
        return context


class ArtifactDetailView(DetailView):
    model = Artifact
    template_name = "results/artifact_detail.html"
    context_object_name = "artifact"

    def get_object(self, queryset=None):
        return super().get_object(queryset=queryset)

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        artifact = self.object
        path = _resolve_artifact_path(artifact)
        remote_only = _is_remote_only_artifact(artifact)
        local_exists = path.exists() and path.is_file()
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        context["artifact_path"] = str(path)
        context["remote_path"] = str(metadata.get("remote_path") or "")
        context["remote_only"] = remote_only
        context["local_exists"] = local_exists
        context["content_available"] = bool(local_exists and not remote_only)
        context["is_image"] = _is_image_artifact(artifact)
        context["content_url"] = (
            reverse("artifact-content", kwargs={"pk": artifact.pk})
            if context["content_available"]
            else ""
        )
        context["file_url"] = (
            reverse("artifact-file", kwargs={"pk": artifact.pk})
            if context["content_available"]
            else ""
        )
        context["download_url"] = (
            f"{context['file_url']}?download=1"
            if context["file_url"]
            else ""
        )
        return context


class ArtifactContentView(View):
    def get(self, request, pk: int):
        try:
            artifact = Artifact.objects.select_related("workflow_run").get(pk=pk)
        except Artifact.DoesNotExist as exc:
            raise Http404("Artifact not found") from exc

        path = _resolve_artifact_path(artifact)
        if _query_flag_enabled(request, "raw") or _query_flag_enabled(request, "download"):
            return _resolved_artifact_file_response(
                artifact,
                as_attachment=_query_flag_enabled(request, "download"),
            )

        remote_only = _is_remote_only_artifact(artifact)
        content_available = bool((not remote_only) and path.exists() and path.is_file())
        content_type, _ = mimetypes.guess_type(str(path))
        file_url = reverse("artifact-file", kwargs={"pk": artifact.pk}) if content_available else ""
        download_url = f"{file_url}?download=1" if file_url else ""
        is_image = bool(content_available and _is_image_artifact(artifact))
        text_preview = ""
        text_preview_truncated = False
        if content_available and (not is_image) and _is_text_previewable(path, content_type):
            try:
                text_preview, text_preview_truncated = _load_text_preview(path)
            except Exception as exc:
                text_preview = f"Failed to read preview text: {exc}"
                text_preview_truncated = False

        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        context = {
            "artifact": artifact,
            "artifact_path": str(path),
            "remote_path": str(metadata.get("remote_path") or ""),
            "remote_only": remote_only,
            "content_available": content_available,
            "content_type": content_type or "application/octet-stream",
            "is_image": is_image,
            "text_preview": text_preview,
            "text_preview_truncated": text_preview_truncated,
            "file_url": file_url,
            "download_url": download_url,
        }
        return render(request, "results/artifact_content.html", context)


class ArtifactFileView(View):
    def get(self, request, pk: int):
        try:
            artifact = Artifact.objects.select_related("workflow_run").get(pk=pk)
        except Artifact.DoesNotExist as exc:
            raise Http404("Artifact not found") from exc
        return _resolved_artifact_file_response(
            artifact,
            as_attachment=_query_flag_enabled(request, "download"),
        )


@require_POST
def artifact_ingest_view(request, pk: int):
    try:
        artifact = Artifact.objects.select_related("workflow_run").get(pk=pk)
    except Artifact.DoesNotExist as exc:
        raise Http404("Artifact not found") from exc
    from workflows.views import workflow_ingest_artifact_view

    return workflow_ingest_artifact_view(
        request,
        pk=artifact.workflow_run.pk,
        artifact_pk=artifact.pk,
    )
