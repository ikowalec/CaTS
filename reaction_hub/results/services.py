from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
import re

from django.db.models import Q

from workflows.models import WorkflowRun, WorkflowStatus

from .models import AdsorptionEnergyResult, ComputedProperty

_EV_PER_HARTREE = Decimal("27.211386245988")
_DECIMAL_QUANT = Decimal("0.000001")
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_STATUS_LABELS = {
    "computed": "Computed",
    "matched": "Matched (not saved)",
    "matched_from_hints": "Matched from co-adsorbate hints",
    "missing_system_formula": "Missing system formula",
    "missing_system_energy": "Missing system energy",
    "missing_reference_candidates": "Missing reference runs",
    "missing_reference_match": "No formula match",
}
_COADSORBATE_RUN_MODE = "coadsorbate_placement"
_MIXING_ENERGY_PROPERTY = "mixing_energy_eV"
_MIXING_ENERGY_PER_ATOM_PROPERTY = "mixing_energy_per_atom_eV"
_ATOMIZATION_ENERGY_PROPERTY = "atomization_energy_eV"
_ATOMIZATION_ENERGY_PER_ATOM_PROPERTY = "atomization_energy_per_atom_eV"
_METAL_SYMBOLS = {
    "Li",
    "Be",
    "Na",
    "Mg",
    "Al",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
}


def _to_decimal(value: float | int | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def _quantize_energy(value: float | int | Decimal) -> Decimal:
    return _to_decimal(value).quantize(_DECIMAL_QUANT, rounding=ROUND_HALF_UP)


def _fraction_to_decimal(value: Fraction) -> Decimal:
    return _to_decimal(value.numerator) / _to_decimal(value.denominator)


def _fraction_to_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _run_mode(run: WorkflowRun) -> str:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    submission = metadata.get("submission")
    if not isinstance(submission, dict):
        return ""
    return str(submission.get("run_mode") or "").strip().lower()


def _is_coadsorbate_screening_run(run: WorkflowRun) -> bool:
    if _run_mode(run) == _COADSORBATE_RUN_MODE:
        return True
    workflow = getattr(run, "workflow", None)
    code_path = str(getattr(workflow, "code_path", "") or "").strip().lower()
    return "coadsorbate" in code_path


def _property_value_float(run: WorkflowRun, *, name: str, unit: str) -> float | None:
    prop = (
        run.computed_properties.filter(name=name, unit=unit)
        .order_by("-updated_at", "-created_at", "-id")
        .first()
    )
    if prop is None or prop.value_float is None:
        return None
    return float(prop.value_float)


def _run_energy_ev(run: WorkflowRun) -> Decimal | None:
    value_eV = _property_value_float(run, name="final_single_point_energy_eV", unit="eV")
    if value_eV is not None:
        return _quantize_energy(value_eV)

    value_h = _property_value_float(run, name="final_single_point_energy", unit="Eh")
    if value_h is not None:
        return _quantize_energy(_to_decimal(value_h) * _EV_PER_HARTREE)

    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    result_summary = metadata.get("result_summary")
    if isinstance(result_summary, dict):
        final_data = result_summary.get("final")
        if isinstance(final_data, dict):
            value = final_data.get("energy_eV")
            if value is not None:
                try:
                    return _quantize_energy(float(value))
                except (TypeError, ValueError):
                    pass
    return None


def _run_cluster_fragment_energy_ev(run: WorkflowRun) -> Decimal | None:
    value_eV = _property_value_float(run, name="final_cluster_single_point_energy_eV", unit="eV")
    if value_eV is not None:
        return _quantize_energy(value_eV)

    value_h = _property_value_float(run, name="final_cluster_single_point_energy", unit="Eh")
    if value_h is not None:
        return _quantize_energy(_to_decimal(value_h) * _EV_PER_HARTREE)

    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    result_summary = metadata.get("result_summary")
    if isinstance(result_summary, dict):
        final_data = result_summary.get("final")
        if isinstance(final_data, dict):
            cluster_fragment = final_data.get("cluster_fragment")
            if isinstance(cluster_fragment, dict):
                value = cluster_fragment.get("energy_eV")
                if value is not None:
                    try:
                        return _quantize_energy(float(value))
                    except (TypeError, ValueError):
                        pass

    legacy = metadata.get("legacy_catalog")
    if isinstance(legacy, dict):
        relative = str(legacy.get("working_directory_relative") or "").replace("\\", "/").strip("/")
        parts = relative.split("/")
        if len(parts) >= 3 and parts[-1] == parts[-2]:
            sibling_relative = "/".join([*parts[:-2], "distortion_energy", parts[-1]])
            sibling = WorkflowRun.objects.filter(
                status=WorkflowStatus.COMPLETED,
                settings_profile_id=run.settings_profile_id,
                metadata__legacy_catalog__source_tree_id=legacy.get("source_tree_id"),
                metadata__legacy_catalog__working_directory_relative=sibling_relative,
            ).first()
            if sibling is not None:
                return _run_energy_ev(sibling)
    return None


def _metadata_distortion_energy_ed_eV(result: AdsorptionEnergyResult) -> Decimal | None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}

    raw_initial = metadata.get("cluster_initial_energy_eV")
    raw_final = metadata.get("cluster_final_energy_eV")
    if raw_initial is not None and raw_final is not None:
        try:
            return _quantize_energy(Decimal(str(raw_final)) - Decimal(str(raw_initial)))
        except Exception:
            pass

    raw_ed = metadata.get("ed_energy_eV")
    if raw_ed is not None:
        try:
            return _quantize_energy(Decimal(str(raw_ed)))
        except Exception:
            pass

    raw_distortion = metadata.get("cluster_distortion_energy_eV")
    if raw_distortion is not None:
        try:
            return _quantize_energy(Decimal(str(raw_distortion)))
        except Exception:
            pass

    return None


def _metadata_interaction_energy_eint_eV(result: AdsorptionEnergyResult) -> Decimal | None:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}

    raw_eint = metadata.get("eint_energy_eV")
    if raw_eint is not None:
        try:
            return _quantize_energy(Decimal(str(raw_eint)))
        except Exception:
            pass

    raw_legacy = metadata.get("adsorbate_cluster_interaction_energy_eV")
    if raw_legacy is not None:
        try:
            return _quantize_energy(Decimal(str(raw_legacy)))
        except Exception:
            pass

    ed_energy = _metadata_distortion_energy_ed_eV(result)
    if ed_energy is None:
        return None
    try:
        return _quantize_energy(_to_decimal(result.adsorption_energy_eV) - ed_energy)
    except Exception:
        return None


def _metadata_adsorbate_display_name(result: AdsorptionEnergyResult) -> str:
    metadata = result.metadata if isinstance(result.metadata, dict) else {}
    component_breakdown = metadata.get("component_breakdown")
    labels: list[str] = []
    if isinstance(component_breakdown, list):
        for entry in component_breakdown:
            if not isinstance(entry, dict):
                continue
            formula = str(entry.get("adsorbate_formula") or "").strip()
            name = str(entry.get("adsorbate_name") or "").strip()
            label = formula or name
            if label and label not in labels:
                labels.append(label)
    if labels:
        return ", ".join(labels)

    raw_formula = str(metadata.get("adsorbate_formula") or "").strip()
    if raw_formula:
        return raw_formula

    adsorbate = getattr(result, "adsorbate", None)
    if adsorbate is None:
        return "-"
    formula = str(getattr(adsorbate, "formula", "") or "").strip()
    if formula:
        return formula
    name = str(getattr(adsorbate, "name", "") or "").strip()
    return name or "-"


def _component_adsorbate_display_name(components: list[_MatchedAdsorbateComponent]) -> str:
    labels: list[str] = []
    for component in components:
        label = str(component.formula or component.candidate.name).strip()
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels) if labels else "-"


def _matched_adsorbate_display_name(match: dict[str, object]) -> str:
    components = match.get("components")
    if isinstance(components, list):
        component_items = [entry for entry in components if isinstance(entry, _MatchedAdsorbateComponent)]
        if component_items:
            return _component_adsorbate_display_name(component_items)

    formula = str(match.get("adsorbate_formula") or "").strip()
    if formula:
        return formula

    adsorbate = match.get("adsorbate")
    if isinstance(adsorbate, _ReferenceCandidate):
        return str(adsorbate.name or "-")
    return "-"


def _result_adsorbate_display_name(
    result: AdsorptionEnergyResult | None, *, match: dict[str, object]
) -> str:
    if result is not None:
        return _metadata_adsorbate_display_name(result)
    return _matched_adsorbate_display_name(match)


def _result_interaction_energy_eint_eV(result: AdsorptionEnergyResult | None) -> Decimal | None:
    if result is not None:
        return _metadata_interaction_energy_eint_eV(result)
    return None


def _run_formula(run: WorkflowRun) -> str:
    formula_prop = (
        run.computed_properties.filter(name="final_structure_formula")
        .order_by("-updated_at", "-created_at", "-id")
        .first()
    )
    if formula_prop is not None and formula_prop.value_text:
        return str(formula_prop.value_text).strip()

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


def _formula_counts(formula: str) -> Counter[str] | None:
    text = str(formula or "").strip()
    if not text:
        return None

    try:
        from ase.formula import Formula

        counts = Formula(text).count()
        parsed = Counter({str(symbol): int(value) for symbol, value in counts.items() if int(value) > 0})
        return parsed if parsed else None
    except Exception:
        parsed_fallback: Counter[str] = Counter()
        cursor = 0
        for match in _FORMULA_TOKEN_RE.finditer(text):
            if match.start() != cursor:
                return None
            symbol = match.group(1)
            raw_count = match.group(2)
            count = int(raw_count) if raw_count else 1
            if count <= 0:
                return None
            parsed_fallback[symbol] += count
            cursor = match.end()
        if cursor != len(text):
            return None
        return parsed_fallback if parsed_fallback else None


def _timestamp_score(value: datetime | None) -> float:
    if value is None:
        return 0.0
    try:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return float(value.timestamp())
    except Exception:
        return 0.0


@dataclass(frozen=True)
class _FormulaVariant:
    formula: str
    counts: Counter[str]


@dataclass(frozen=True)
class _ReferenceCandidate:
    run: WorkflowRun
    name: str
    variants: tuple[_FormulaVariant, ...]
    energy_eV: Decimal


@dataclass(frozen=True)
class _MatchedAdsorbateComponent:
    candidate: _ReferenceCandidate
    formula: str
    stoich_factor: Fraction
    scaled_energy_eV: Decimal


def _counts_key(counts: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(symbol), int(value)) for symbol, value in counts.items() if int(value) > 0))


def _formula_variants(*formulas: str) -> tuple[_FormulaVariant, ...]:
    variants: list[_FormulaVariant] = []
    seen_keys: set[tuple[tuple[str, int], ...]] = set()
    for raw_formula in formulas:
        formula = str(raw_formula or "").strip()
        if not formula:
            continue
        counts = _formula_counts(formula)
        if counts is None:
            continue
        key = _counts_key(counts)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        variants.append(_FormulaVariant(formula=formula, counts=counts))
    return tuple(variants)


def _is_homonuclear_diatomic(counts: Counter[str]) -> bool:
    if len(counts) != 1:
        return False
    (_symbol, count), = counts.items()
    return int(count) == 2


def _is_metal_only_counts(counts: Counter[str]) -> bool:
    return bool(counts) and all(str(symbol) in _METAL_SYMBOLS for symbol in counts)


def _pick_elemental_adsorbate_reference(
    *,
    system_run: WorkflowRun,
    symbol: str,
) -> tuple[_ReferenceCandidate, str] | None:
    best: tuple[tuple[Decimal, float, int], _ReferenceCandidate, str] | None = None
    for candidate in _collect_adsorbate_candidates(system_run):
        matched_formula = ""
        for variant in candidate.variants:
            if variant.counts == Counter({str(symbol): 1}):
                matched_formula = variant.formula
                break
        if not matched_formula:
            continue
        score = (
            candidate.energy_eV,
            -_timestamp_score(candidate.run.completed_at or candidate.run.updated_at),
            -int(candidate.run.pk),
        )
        if best is None or score < best[0]:
            best = (score, candidate, matched_formula)
    if best is None:
        candidates = (
            WorkflowRun.objects.filter(
                status=WorkflowStatus.COMPLETED,
                settings_profile_id=system_run.settings_profile_id,
            )
            .exclude(pk=system_run.pk)
            .filter(
                Q(
                    computed_properties__name="final_structure_formula",
                    computed_properties__value_text=symbol,
                )
                | Q(metadata__structure__formula=symbol)
                | Q(metadata__result_summary__final__formula=symbol)
            )
            .prefetch_related("computed_properties")
            .distinct()
        )
        for run in candidates:
            formula = _run_formula(run)
            energy = _run_energy_ev(run)
            if _formula_counts(formula) != Counter({str(symbol): 1}) or energy is None:
                continue
            candidate = _ReferenceCandidate(
                run=run,
                name=run.run_name,
                variants=_formula_variants(formula),
                energy_eV=energy,
            )
            score = (
                energy,
                -_timestamp_score(run.completed_at or run.updated_at),
                -int(run.pk),
            )
            if best is None or score < best[0]:
                best = (score, candidate, formula)
    if best is None:
        return None
    _score, candidate, matched_formula = best
    return candidate, matched_formula


def _pick_monometallic_nanoparticle_references(
    *,
    system_run: WorkflowRun,
    system_counts: Counter[str],
    atom_count: int,
) -> dict[str, tuple[WorkflowRun, Decimal, str]]:
    best: dict[str, tuple[tuple[Decimal, float, int], WorkflowRun, Decimal, str]] = {}
    target_formulas = [
        f"{symbol}{atom_count if atom_count != 1 else ''}"
        for symbol in system_counts
    ]
    candidates = (
        WorkflowRun.objects.filter(
            status=WorkflowStatus.COMPLETED,
            settings_profile_id=system_run.settings_profile_id,
            reference_adsorbate__isnull=True,
        )
        .filter(
            Q(
                computed_properties__name="final_structure_formula",
                computed_properties__value_text__in=target_formulas,
            )
            | Q(metadata__structure__formula__in=target_formulas)
            | Q(metadata__result_summary__final__formula__in=target_formulas)
        )
        .select_related("workflow", "settings_profile")
        .prefetch_related("computed_properties", "jobs__hpc_system")
        .distinct()
    )
    for run in candidates:
        formula = _run_formula(run)
        counts = _formula_counts(formula)
        energy = _run_energy_ev(run)
        if counts is None or energy is None or len(counts) != 1:
            continue
        symbol, count = next(iter(counts.items()))
        if symbol not in system_counts or int(count) != atom_count:
            continue
        score = (
            energy,
            -_timestamp_score(run.completed_at or run.updated_at),
            -int(run.pk),
        )
        if symbol not in best or score < best[symbol][0]:
            best[symbol] = (score, run, energy, formula)
    return {
        symbol: (run, energy, formula)
        for symbol, (_score, run, energy, formula) in best.items()
    }


def _delete_mixing_energy_properties(run: WorkflowRun) -> None:
    ComputedProperty.objects.filter(
        workflow_run=run,
        name__in=[_MIXING_ENERGY_PROPERTY, _MIXING_ENERGY_PER_ATOM_PROPERTY],
    ).delete()


def _delete_atomization_energy_properties(run: WorkflowRun) -> None:
    ComputedProperty.objects.filter(
        workflow_run=run,
        name__in=[_ATOMIZATION_ENERGY_PROPERTY, _ATOMIZATION_ENERGY_PER_ATOM_PROPERTY],
    ).delete()


def upsert_atomization_energy_for_run(run: WorkflowRun) -> ComputedProperty | None:
    if run.status != WorkflowStatus.COMPLETED or run.reference_adsorbate_id is not None:
        _delete_atomization_energy_properties(run)
        return None

    system_formula = _run_formula(run)
    system_counts = _formula_counts(system_formula)
    system_energy = _run_energy_ev(run)
    if system_counts is None or system_energy is None:
        _delete_atomization_energy_properties(run)
        return None

    atom_count = sum(int(value) for value in system_counts.values())
    if atom_count <= 0:
        _delete_atomization_energy_properties(run)
        return None

    references_by_symbol: dict[str, tuple[_ReferenceCandidate, str]] = {}
    for symbol in sorted(system_counts):
        reference = _pick_elemental_adsorbate_reference(
            system_run=run,
            symbol=str(symbol),
        )
        if reference is None:
            _delete_atomization_energy_properties(run)
            return None
        references_by_symbol[str(symbol)] = reference

    gas_atom_energy = Decimal("0")
    reference_breakdown: list[dict[str, object]] = []
    for symbol, count in sorted(system_counts.items()):
        candidate, reference_formula = references_by_symbol[str(symbol)]
        scaled_energy = _quantize_energy(candidate.energy_eV * _to_decimal(int(count)))
        gas_atom_energy += scaled_energy
        reference_breakdown.append(
            {
                "symbol": str(symbol),
                "count": int(count),
                "reference_run_id": int(candidate.run.pk),
                "reference_name": candidate.name,
                "reference_formula": reference_formula,
                "reference_energy_eV": str(candidate.energy_eV),
                "scaled_reference_energy_eV": str(scaled_energy),
            }
        )

    gas_atom_energy = _quantize_energy(gas_atom_energy)
    atomization_energy = _quantize_energy(gas_atom_energy - system_energy)
    atomization_energy_per_atom = _quantize_energy(
        atomization_energy / _to_decimal(atom_count)
    )
    metadata = {
        "source": "elemental_gas_adsorbate_references",
        "system_formula": system_formula,
        "system_energy_eV": str(system_energy),
        "gas_atom_energy_eV": str(gas_atom_energy),
        "atom_count": int(atom_count),
        "reference_breakdown": reference_breakdown,
        "method": "Eatomization = sum_i(N_i * Egas_atom_i) - Esystem",
    }
    prop, _created = ComputedProperty.objects.update_or_create(
        workflow_run=run,
        name=_ATOMIZATION_ENERGY_PROPERTY,
        unit="eV",
        defaults={
            "value_float": float(atomization_energy),
            "value_text": "",
            "metadata": metadata,
        },
    )
    ComputedProperty.objects.update_or_create(
        workflow_run=run,
        name=_ATOMIZATION_ENERGY_PER_ATOM_PROPERTY,
        unit="eV/atom",
        defaults={
            "value_float": float(atomization_energy_per_atom),
            "value_text": "",
            "metadata": metadata,
        },
    )
    return prop


def upsert_mixing_energy_for_run(run: WorkflowRun) -> ComputedProperty | None:
    if run.status != WorkflowStatus.COMPLETED:
        _delete_mixing_energy_properties(run)
        return None
    if run.reference_adsorbate_id is not None:
        _delete_mixing_energy_properties(run)
        return None

    system_formula = _run_formula(run)
    system_counts = _formula_counts(system_formula)
    system_energy = _run_energy_ev(run)
    if system_counts is None or system_energy is None or not _is_metal_only_counts(system_counts):
        _delete_mixing_energy_properties(run)
        return None

    atom_count = sum(int(value) for value in system_counts.values())
    if atom_count <= 0:
        _delete_mixing_energy_properties(run)
        return None

    references_by_symbol = _pick_monometallic_nanoparticle_references(
        system_run=run,
        system_counts=system_counts,
        atom_count=atom_count,
    )
    if set(references_by_symbol) != set(system_counts):
        _delete_mixing_energy_properties(run)
        return None

    reference_energy = Decimal("0")
    reference_breakdown: list[dict[str, object]] = []
    for symbol, count in sorted(system_counts.items()):
        reference_run, reference_run_energy, reference_formula = references_by_symbol[str(symbol)]
        reference_energy_per_atom = reference_run_energy / _to_decimal(atom_count)
        scaled_energy = _quantize_energy(reference_energy_per_atom * _to_decimal(int(count)))
        reference_energy += scaled_energy
        reference_breakdown.append(
            {
                "symbol": str(symbol),
                "count": int(count),
                "reference_run_id": int(reference_run.pk),
                "reference_run_uid": str(reference_run.run_uid),
                "reference_name": reference_run.run_name,
                "reference_formula": reference_formula,
                "reference_atom_count": atom_count,
                "reference_energy_eV": str(reference_run_energy),
                "reference_energy_per_atom_eV": str(_quantize_energy(reference_energy_per_atom)),
                "scaled_reference_energy_eV": str(scaled_energy),
                "reference_working_directory": reference_run.working_directory,
                "reference_workflow": {
                    "id": int(reference_run.workflow_id),
                    "name": reference_run.workflow.name,
                    "code_path": reference_run.workflow.code_path,
                },
                "reference_settings_profile": {
                    "id": int(reference_run.settings_profile_id),
                    "name": reference_run.settings_profile.name,
                },
                "reference_jobs": [
                    {
                        "id": int(job.pk),
                        "name": job.job_name,
                        "scheduler_job_id": job.scheduler_job_id,
                        "status": job.status,
                        "hpc_system_id": job.hpc_system_id,
                        "hpc_system_name": job.hpc_system.name if job.hpc_system else "",
                    }
                    for job in reference_run.jobs.all()
                ],
            }
        )

    reference_energy = _quantize_energy(reference_energy)
    mixing_energy = _quantize_energy(system_energy - reference_energy)
    mixing_energy_per_atom = _quantize_energy(mixing_energy / _to_decimal(atom_count))
    metadata = {
        "source": "monometallic_nanoparticle_references",
        "system_formula": system_formula,
        "system_energy_eV": str(system_energy),
        "reference_energy_eV": str(reference_energy),
        "atom_count": int(atom_count),
        "reference_breakdown": reference_breakdown,
        "method": "Emix = Esystem - sum_i(N_i * E(M_i,N) / N)",
    }
    prop, _created = ComputedProperty.objects.update_or_create(
        workflow_run=run,
        name=_MIXING_ENERGY_PROPERTY,
        unit="eV",
        defaults={
            "value_float": float(mixing_energy),
            "value_text": "",
            "metadata": metadata,
        },
    )
    ComputedProperty.objects.update_or_create(
        workflow_run=run,
        name=_MIXING_ENERGY_PER_ATOM_PROPERTY,
        unit="eV/atom",
        defaults={
            "value_float": float(mixing_energy_per_atom),
            "value_text": "",
            "metadata": metadata,
        },
    )
    return prop


def _adsorbate_stoich_factor(
    *,
    system_counts: Counter[str],
    cluster_counts: Counter[str],
    adsorbate_counts: Counter[str],
) -> Fraction | None:
    if not system_counts or not cluster_counts or not adsorbate_counts:
        return None

    remainder: dict[str, int] = {}
    symbols = set(system_counts) | set(cluster_counts)
    for symbol in symbols:
        sys_count = int(system_counts.get(symbol, 0))
        cluster_count = int(cluster_counts.get(symbol, 0))
        delta = sys_count - cluster_count
        if delta < 0:
            return None
        if delta > 0:
            remainder[str(symbol)] = delta

    if not remainder:
        return None
    if set(remainder) != set(adsorbate_counts):
        return None

    factor: Fraction | None = None
    for symbol, ads_count_raw in adsorbate_counts.items():
        ads_count = int(ads_count_raw)
        if ads_count <= 0:
            return None
        rem_count = int(remainder.get(str(symbol), 0))
        if rem_count <= 0:
            return None
        candidate = Fraction(rem_count, ads_count)
        if factor is None:
            factor = candidate
            continue
        if candidate != factor:
            return None

    if factor is None or factor <= 0:
        return None
    return factor


def _collect_adsorbate_candidates(system_run: WorkflowRun) -> list[_ReferenceCandidate]:
    qs = (
        WorkflowRun.objects.filter(
            status=WorkflowStatus.COMPLETED,
            settings_profile_id=system_run.settings_profile_id,
            reference_adsorbate__isnull=False,
        )
        .exclude(pk=system_run.pk)
        .select_related("reference_adsorbate")
        .prefetch_related("computed_properties")
        .order_by("-completed_at", "-updated_at", "-id")
    )

    candidates: list[_ReferenceCandidate] = []
    for run in qs:
        variants = [
            variant
            for variant in _formula_variants(
                str(run.reference_adsorbate.formula or "").strip(), _run_formula(run)
            )
            if not _is_metal_only_counts(variant.counts)
        ]
        energy = _run_energy_ev(run)
        if not variants or energy is None:
            continue
        candidates.append(
            _ReferenceCandidate(
                run=run,
                name=str(run.reference_adsorbate.name),
                variants=variants,
                energy_eV=energy,
            )
        )
    return candidates


def _collect_nanocluster_candidates(system_run: WorkflowRun) -> list[_ReferenceCandidate]:
    qs = (
        WorkflowRun.objects.filter(
            status=WorkflowStatus.COMPLETED,
            settings_profile_id=system_run.settings_profile_id,
            reference_nanocluster__isnull=False,
        )
        .exclude(pk=system_run.pk)
        .select_related("reference_nanocluster")
        .prefetch_related("computed_properties")
        .order_by("-completed_at", "-updated_at", "-id")
    )

    candidates: list[_ReferenceCandidate] = []
    for run in qs:
        variants = _formula_variants(str(run.reference_nanocluster.composition or "").strip(), _run_formula(run))
        energy = _run_energy_ev(run)
        if not variants or energy is None:
            continue
        candidates.append(
            _ReferenceCandidate(
                run=run,
                name=str(run.reference_nanocluster.name),
                variants=variants,
                energy_eV=energy,
            )
        )
    return candidates


def _run_structure_metadata(run: WorkflowRun) -> dict[str, object]:
    metadata = run.metadata if isinstance(run.metadata, dict) else {}
    structure = metadata.get("structure")
    if isinstance(structure, dict):
        return structure
    return {}


def _adsorption_reference_hints_from_run(run: WorkflowRun) -> dict[str, object]:
    structure = _run_structure_metadata(run)
    for key in ("adsorption_reference_hints", "coadsorbate_provenance"):
        candidate = structure.get(key)
        if isinstance(candidate, dict):
            return candidate
    return {}


def _hints_adsorbate_formula_counts(hints: dict[str, object]) -> Counter[str]:
    counts: Counter[str] = Counter()
    raw_counts = hints.get("adsorbate_formula_counts")
    if isinstance(raw_counts, dict):
        for raw_formula, raw_count in raw_counts.items():
            formula = str(raw_formula or "").strip()
            if not formula:
                continue
            try:
                count = int(raw_count)
            except (TypeError, ValueError):
                continue
            if count > 0:
                counts[formula] += count
    if counts:
        return counts

    raw_formulas = hints.get("adsorbate_formulas")
    if isinstance(raw_formulas, list):
        for entry in raw_formulas:
            formula = str(entry or "").strip()
            if formula:
                counts[formula] += 1
    if counts:
        return counts

    raw_adsorbates = hints.get("adsorbates")
    if isinstance(raw_adsorbates, list):
        for entry in raw_adsorbates:
            if isinstance(entry, dict):
                formula = str(entry.get("formula") or "").strip()
            else:
                formula = str(entry or "").strip()
            if formula:
                counts[formula] += 1
    return counts


def _pick_component_combo_from_hints(system_run: WorkflowRun) -> dict[str, object]:
    hints = _adsorption_reference_hints_from_run(system_run)
    if not hints:
        return {}

    system_formula = _run_formula(system_run)
    system_energy = _run_energy_ev(system_run)
    if system_energy is None:
        return {"status": "missing_system_energy", "system_formula": system_formula, "system_energy_eV": None}

    raw_cluster_id = hints.get("reference_cluster_id")
    try:
        reference_cluster_id = int(raw_cluster_id) if raw_cluster_id is not None else 0
    except (TypeError, ValueError):
        reference_cluster_id = 0
    if reference_cluster_id <= 0:
        return {}

    adsorbate_formula_counts = _hints_adsorbate_formula_counts(hints)
    if not adsorbate_formula_counts:
        return {}

    cluster_candidates = [
        candidate
        for candidate in _collect_nanocluster_candidates(system_run)
        if candidate.run.reference_nanocluster_id == reference_cluster_id
    ]
    adsorbate_candidates = _collect_adsorbate_candidates(system_run)
    if not cluster_candidates or not adsorbate_candidates:
        return {
            "status": "missing_reference_candidates",
            "system_formula": system_formula,
            "system_energy_eV": system_energy,
        }

    cluster = max(
        cluster_candidates,
        key=lambda candidate: (
            _timestamp_score(candidate.run.completed_at or candidate.run.updated_at),
            int(candidate.run.pk),
        ),
    )

    components: list[_MatchedAdsorbateComponent] = []
    for formula, count in sorted(adsorbate_formula_counts.items()):
        if count <= 0:
            continue
        target_counts = _formula_counts(formula)
        if target_counts is None:
            return {
                "status": "missing_reference_match",
                "system_formula": system_formula,
                "system_energy_eV": system_energy,
            }

        best_adsorbate: tuple[tuple[int, float, int], _ReferenceCandidate, _FormulaVariant] | None = None
        for adsorbate in adsorbate_candidates:
            for variant in adsorbate.variants:
                if variant.counts != target_counts:
                    continue
                score = (
                    1 if _is_homonuclear_diatomic(variant.counts) else 0,
                    _timestamp_score(adsorbate.run.completed_at or adsorbate.run.updated_at),
                    int(adsorbate.run.pk),
                )
                if best_adsorbate is None or score > best_adsorbate[0]:
                    best_adsorbate = (score, adsorbate, variant)

        if best_adsorbate is None:
            return {
                "status": "missing_reference_match",
                "system_formula": system_formula,
                "system_energy_eV": system_energy,
            }

        _score, adsorbate, matched_variant = best_adsorbate
        stoich_factor = Fraction(int(count), 1)
        scaled_energy = _quantize_energy(adsorbate.energy_eV * _fraction_to_decimal(stoich_factor))
        components.append(
            _MatchedAdsorbateComponent(
                candidate=adsorbate,
                formula=matched_variant.formula,
                stoich_factor=stoich_factor,
                scaled_energy_eV=scaled_energy,
            )
        )

    if not components:
        return {
            "status": "missing_reference_match",
            "system_formula": system_formula,
            "system_energy_eV": system_energy,
        }

    total_scaled_adsorbate_energy = _quantize_energy(sum(component.scaled_energy_eV for component in components))
    reference_energy = _quantize_energy(cluster.energy_eV + total_scaled_adsorbate_energy)
    adsorption_energy = _quantize_energy(system_energy - reference_energy)

    primary_component = max(
        components,
        key=lambda component: (
            _fraction_to_decimal(component.stoich_factor),
            _timestamp_score(component.candidate.run.completed_at or component.candidate.run.updated_at),
            int(component.candidate.run.pk),
        ),
    )
    primary_formula = primary_component.formula
    primary_factor = primary_component.stoich_factor

    return {
        "status": "matched",
        "matched_status": "matched_from_hints",
        "match_rule": "coadsorbate_reference_hints",
        "system_formula": system_formula,
        "system_energy_eV": system_energy,
        "reference_energy_eV": reference_energy,
        "adsorption_energy_eV": adsorption_energy,
        "cluster": cluster,
        "adsorbate": primary_component.candidate,
        "cluster_formula": (
            cluster.variants[0].formula if cluster.variants else str(cluster.run.reference_nanocluster.composition or "")
        ),
        "adsorbate_formula": primary_formula,
        "adsorbate_stoich_factor": primary_factor,
        "adsorbate_scaled_energy_eV": primary_component.scaled_energy_eV,
        "components": components,
        "reference_cluster_id": int(reference_cluster_id),
    }


def _pick_component_pair(system_run: WorkflowRun) -> dict[str, object]:
    system_formula = _run_formula(system_run)
    system_counts = _formula_counts(system_formula)
    system_energy = _run_energy_ev(system_run)

    if system_counts is None:
        return {"status": "missing_system_formula", "system_formula": system_formula, "system_energy_eV": system_energy}
    if system_energy is None:
        return {"status": "missing_system_energy", "system_formula": system_formula, "system_energy_eV": None}

    adsorbate_candidates = _collect_adsorbate_candidates(system_run)
    cluster_candidates = _collect_nanocluster_candidates(system_run)
    if not adsorbate_candidates or not cluster_candidates:
        return {
            "status": "missing_reference_candidates",
            "system_formula": system_formula,
            "system_energy_eV": system_energy,
        }

    best: tuple[
        tuple[int, float, float, int, int],
        _ReferenceCandidate,
        _ReferenceCandidate,
        _FormulaVariant,
        _FormulaVariant,
        Fraction,
    ] | None = None
    for cluster in cluster_candidates:
        for adsorbate in adsorbate_candidates:
            for cluster_variant in cluster.variants:
                for adsorbate_variant in adsorbate.variants:
                    adsorbate_factor = _adsorbate_stoich_factor(
                        system_counts=system_counts,
                        cluster_counts=cluster_variant.counts,
                        adsorbate_counts=adsorbate_variant.counts,
                    )
                    if adsorbate_factor is None:
                        continue

                    diatomic_bonus = 1 if _is_homonuclear_diatomic(adsorbate_variant.counts) else 0
                    score = (
                        diatomic_bonus,
                        max(
                            _timestamp_score(cluster.run.completed_at or cluster.run.updated_at),
                            _timestamp_score(adsorbate.run.completed_at or adsorbate.run.updated_at),
                        ),
                        _timestamp_score(cluster.run.updated_at) + _timestamp_score(adsorbate.run.updated_at),
                        int(cluster.run.pk),
                        int(adsorbate.run.pk),
                    )
                    if best is None or score > best[0]:
                        best = (
                            score,
                            cluster,
                            adsorbate,
                            cluster_variant,
                            adsorbate_variant,
                            adsorbate_factor,
                        )

    if best is None:
        return {
            "status": "missing_reference_match",
            "system_formula": system_formula,
            "system_energy_eV": system_energy,
        }

    _score, cluster, adsorbate, cluster_variant, adsorbate_variant, adsorbate_factor = best
    scaled_adsorbate_energy = _quantize_energy(
        adsorbate.energy_eV * _fraction_to_decimal(adsorbate_factor)
    )
    components = [
        _MatchedAdsorbateComponent(
            candidate=adsorbate,
            formula=adsorbate_variant.formula,
            stoich_factor=adsorbate_factor,
            scaled_energy_eV=scaled_adsorbate_energy,
        )
    ]
    reference_energy = _quantize_energy(cluster.energy_eV + scaled_adsorbate_energy)
    adsorption_energy = _quantize_energy(system_energy - reference_energy)
    return {
        "status": "matched",
        "matched_status": "matched",
        "match_rule": "formula_composition_match_with_settings_profile",
        "system_formula": system_formula,
        "system_energy_eV": system_energy,
        "reference_energy_eV": reference_energy,
        "adsorption_energy_eV": adsorption_energy,
        "cluster": cluster,
        "adsorbate": adsorbate,
        "cluster_formula": cluster_variant.formula,
        "adsorbate_formula": adsorbate_variant.formula,
        "adsorbate_stoich_factor": adsorbate_factor,
        "adsorbate_scaled_energy_eV": scaled_adsorbate_energy,
        "components": components,
    }


def _pick_component_match(system_run: WorkflowRun) -> dict[str, object]:
    hinted_match = _pick_component_combo_from_hints(system_run)
    if hinted_match.get("status") == "matched":
        return hinted_match
    return _pick_component_pair(system_run)


def _coadsorption_interaction_payload(
    *,
    run: WorkflowRun,
    cluster: _ReferenceCandidate,
    components: list[_MatchedAdsorbateComponent],
    adsorption_energy_eV: Decimal,
) -> dict[str, object]:
    if len(components) <= 1:
        return {}
    cluster_ref = cluster.run.reference_nanocluster
    if cluster_ref is None:
        return {}

    component_rows: list[tuple[_MatchedAdsorbateComponent, AdsorptionEnergyResult]] = []
    for component in components:
        ads_ref = component.candidate.run.reference_adsorbate
        if ads_ref is None:
            return {}
        result_qs = (
            AdsorptionEnergyResult.objects.filter(
                settings_profile=run.settings_profile,
                nanocluster=cluster_ref,
                adsorbate=ads_ref,
            )
            .exclude(workflow_run=run)
            .order_by("-updated_at", "-created_at", "-id")
        )
        selected: AdsorptionEnergyResult | None = None
        for candidate_result in result_qs:
            candidate_meta = candidate_result.metadata if isinstance(candidate_result.metadata, dict) else {}
            component_breakdown = candidate_meta.get("component_breakdown")
            if isinstance(component_breakdown, list) and len(component_breakdown) > 1:
                continue
            selected = candidate_result
            break
        if selected is None:
            return {}
        component_rows.append((component, selected))

    adsorption_sum = _quantize_energy(
        sum(
            _to_decimal(row.adsorption_energy_eV) * _fraction_to_decimal(component.stoich_factor)
            for component, row in component_rows
        )
    )
    interaction_energy = _quantize_energy(adsorption_energy_eV - adsorption_sum)
    return {
        "coadsorption_interaction_energy_eV": str(interaction_energy),
        "coadsorption_component_adsorption_sum_eV": str(adsorption_sum),
        "coadsorption_component_result_ids": [int(row.pk) for _component, row in component_rows],
    }


def upsert_adsorption_energy_for_run(run: WorkflowRun) -> AdsorptionEnergyResult | None:
    upsert_mixing_energy_for_run(run)
    upsert_atomization_energy_for_run(run)

    if run.status != WorkflowStatus.COMPLETED:
        return None
    if run.reference_adsorbate_id is not None or run.reference_nanocluster_id is not None:
        return None
    if _is_coadsorbate_screening_run(run):
        return None

    match = _pick_component_match(run)
    if match.get("status") != "matched":
        return None

    cluster: _ReferenceCandidate = match["cluster"]  # type: ignore[assignment]
    adsorbate: _ReferenceCandidate = match["adsorbate"]  # type: ignore[assignment]
    components = match.get("components")
    if isinstance(components, list):
        component_items = [entry for entry in components if isinstance(entry, _MatchedAdsorbateComponent)]
    else:
        component_items = []
    adsorbate_factor = match.get("adsorbate_stoich_factor")
    if isinstance(adsorbate_factor, Fraction):
        adsorbate_factor_text = _fraction_to_text(adsorbate_factor)
    else:
        adsorbate_factor_text = "1"
    if len(component_items) > 1:
        component_terms = [
            (
                f"{_fraction_to_text(component.stoich_factor)}*{component.candidate.name}"
                if component.stoich_factor != Fraction(1, 1)
                else component.candidate.name
            )
            for component in component_items
        ]
        method_expr = "Eads = Esystem - (Ecluster + " + " + ".join(component_terms) + ")."
    else:
        method_expr = (
            f"Eads = Esystem - (Ecluster + {adsorbate_factor_text} * Eadsorbate)."
            if adsorbate_factor_text != "1"
            else "Eads = Esystem - (Ecluster + Eadsorbate)."
        )

    cluster_final_energy_eV = _run_cluster_fragment_energy_ev(run)
    cluster_distortion_energy_eV: Decimal | None = None
    distortion_energy_ed_eV: Decimal | None = None
    interaction_energy_eV: Decimal | None = None
    if cluster_final_energy_eV is not None:
        cluster_distortion_energy_eV = _quantize_energy(cluster_final_energy_eV - cluster.energy_eV)
        distortion_energy_ed_eV = _quantize_energy(cluster_final_energy_eV - cluster.energy_eV)
        interaction_energy_eV = _quantize_energy(match["adsorption_energy_eV"] - distortion_energy_ed_eV)
        method_expr += (
            " Distortion correction: "
            "Ed = Ecluster_final - Ecluster_initial. "
            "Einteraction = Eads - Ed."
        )

    component_breakdown: list[dict[str, object]] = []
    for component in component_items:
        component_breakdown.append(
            {
                "adsorbate_name": component.candidate.name,
                "adsorbate_formula": component.formula,
                "adsorbate_run_id": int(component.candidate.run.pk),
                "adsorbate_stoich_factor": _fraction_to_text(component.stoich_factor),
                "adsorbate_energy_eV": str(component.candidate.energy_eV),
                "adsorbate_scaled_energy_eV": str(component.scaled_energy_eV),
            }
        )

    interaction_payload = _coadsorption_interaction_payload(
        run=run,
        cluster=cluster,
        components=component_items,
        adsorption_energy_eV=match["adsorption_energy_eV"],
    )
    match_rule = str(match.get("match_rule") or "formula_composition_match_with_settings_profile")

    result_metadata: dict[str, object] = {
        "match_rule": match_rule,
        "system_formula": match["system_formula"],
        "cluster_formula": str(match["cluster_formula"]),
        "adsorbate_formula": str(match["adsorbate_formula"]),
        "adsorbate_stoich_factor": adsorbate_factor_text,
        "cluster_run_id": cluster.run.pk,
        "adsorbate_run_id": adsorbate.run.pk,
        "cluster_energy_eV": str(cluster.energy_eV),
        "adsorbate_energy_eV": str(adsorbate.energy_eV),
        "adsorbate_scaled_energy_eV": str(match["adsorbate_scaled_energy_eV"]),
        "component_breakdown": component_breakdown or None,
        "coadsorbate_component_count": len(component_breakdown),
        **interaction_payload,
    }
    if (
        cluster_final_energy_eV is not None
        and cluster_distortion_energy_eV is not None
        and distortion_energy_ed_eV is not None
        and interaction_energy_eV is not None
    ):
        result_metadata.update(
            {
                "cluster_initial_energy_eV": str(cluster.energy_eV),
                "cluster_final_energy_eV": str(cluster_final_energy_eV),
                "cluster_distortion_energy_eV": str(cluster_distortion_energy_eV),
                "ed_energy_eV": str(distortion_energy_ed_eV),
                "eint_energy_eV": str(interaction_energy_eV),
                "distortion_corrected_adsorption_energy_eV": str(interaction_energy_eV),
                "adsorbate_cluster_interaction_energy_eV": str(interaction_energy_eV),
            }
        )

    result, _created = AdsorptionEnergyResult.objects.update_or_create(
        workflow_run=run,
        defaults={
            "adsorbate": adsorbate.run.reference_adsorbate,
            "nanocluster": cluster.run.reference_nanocluster,
            "settings_profile": run.settings_profile,
            "total_energy_eV": match["system_energy_eV"],
            "reference_energy_eV": match["reference_energy_eV"],
            "adsorption_energy_eV": match["adsorption_energy_eV"],
            "method_notes": (
                "Auto-derived from completed reference runs with the same settings profile: "
                f"{method_expr}"
            ),
            "metadata": result_metadata,
        },
    )
    return result


def recompute_adsorption_energies(*, limit: int | None = None) -> dict[str, int]:
    qs = WorkflowRun.objects.filter(
        status=WorkflowStatus.COMPLETED,
        reference_adsorbate__isnull=True,
        reference_nanocluster__isnull=True,
    ).exclude(
        workflow__code_path__icontains="coadsorbate",
    ).order_by("-updated_at", "-id")

    if limit is not None:
        parsed_limit = int(limit)
        if parsed_limit <= 0:
            return {"inspected": 0, "created": 0, "updated": 0, "no_match": 0, "errors": 0}
        qs = qs[:parsed_limit]

    summary = {"inspected": 0, "created": 0, "updated": 0, "no_match": 0, "errors": 0}

    for run in qs:
        summary["inspected"] += 1
        had_existing = AdsorptionEnergyResult.objects.filter(workflow_run=run).exists()
        try:
            result = upsert_adsorption_energy_for_run(run)
        except Exception:
            summary["errors"] += 1
            continue

        if result is None:
            summary["no_match"] += 1
            continue

        if had_existing:
            summary["updated"] += 1
        else:
            summary["created"] += 1

    return summary


def build_results_dashboard(*, limit: int = 100) -> dict[str, object]:
    runs = list(
        WorkflowRun.objects.filter(status=WorkflowStatus.COMPLETED)
        .exclude(workflow__code_path__icontains="coadsorbate")
        .select_related("settings_profile", "workflow")
        .prefetch_related("computed_properties")
        .order_by("-created_at", "-id")[: max(1, int(limit))]
    )
    result_map = {
        row.workflow_run_id: row
        for row in AdsorptionEnergyResult.objects.filter(workflow_run__in=runs).select_related(
            "adsorbate", "nanocluster"
        )
    }

    rows: list[dict[str, object]] = []
    total_system_runs = 0
    matched_candidates = 0
    computed_results = 0

    for run in runs:
        if run.reference_adsorbate_id is not None or run.reference_nanocluster_id is not None:
            continue
        total_system_runs += 1

        existing = result_map.get(run.pk)
        match = _pick_component_match(run)
        if match.get("status") == "matched":
            matched_candidates += 1
        if existing is not None:
            computed_results += 1

        cluster = match.get("cluster") if match.get("status") == "matched" else None
        unmatched_status = str(match.get("matched_status") or match.get("status"))
        ed_energy_eV = _metadata_distortion_energy_ed_eV(existing) if existing is not None else None
        eint_energy_eV = _result_interaction_energy_eint_eV(existing)
        adsorbate_name = _result_adsorbate_display_name(existing, match=match)

        rows.append(
            {
                "run_id": run.pk,
                "run_name": run.run_name,
                "settings_profile": run.settings_profile.name if run.settings_profile_id else "-",
                "status_code": "computed" if existing is not None else unmatched_status,
                "status": _STATUS_LABELS.get(
                    "computed" if existing is not None else unmatched_status,
                    str(match.get("status") or "unknown"),
                ),
                "result_id": existing.pk if existing is not None else None,
                "system_formula": match.get("system_formula") or _run_formula(run) or "-",
                "system_energy_eV": (
                    existing.total_energy_eV if existing is not None else match.get("system_energy_eV")
                ),
                "reference_energy_eV": (
                    existing.reference_energy_eV if existing is not None else match.get("reference_energy_eV")
                ),
                "adsorption_energy_eV": (
                    existing.adsorption_energy_eV if existing is not None else match.get("adsorption_energy_eV")
                ),
                "ed_energy_eV": ed_energy_eV,
                "eint_energy_eV": eint_energy_eV,
                "bond_scan_start_energy_eV": _property_value_float(
                    run,
                    name="bond_scan_start_energy_eV",
                    unit="eV",
                ),
                "bond_scan_peak_energy_eV": _property_value_float(
                    run,
                    name="bond_scan_peak_energy_eV",
                    unit="eV",
                ),
                "bond_scan_activation_barrier_eV": _property_value_float(
                    run,
                    name="bond_scan_activation_barrier_eV",
                    unit="eV",
                ),
                "nanocluster_name": (
                    str(existing.nanocluster.name)
                    if existing is not None
                    else (cluster.name if cluster is not None else "-")
                ),
                "adsorbate_name": adsorbate_name,
            }
        )

    return {
        "rows": rows,
        "total_system_runs": total_system_runs,
        "matched_candidates": matched_candidates,
        "computed_results": computed_results,
    }
