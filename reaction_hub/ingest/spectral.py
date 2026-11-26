from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from django.conf import settings

from results.models import Artifact, ArtifactKind, ComputedProperty
from workflows.identity import normalize_run_path


SCHEMA_VERSION = 1
PROPERTY_NAME = "spectral_dataset"
CT_PLOT_VERSION = 15
BROADENED_PLOT_VERSION = 1
_CT_LINESTYLES = ("-", "--", "-.", ":")


def _ct_direction(label: str, fragment_labels: dict[str, str]) -> tuple[str, str, str] | None:
    match = re.search(r"(\d+)\s*(->|<-)\s*(\d+)", str(label))
    if not match:
        return None
    left, arrow, right = match.groups()
    source, target = (left, right) if arrow == "->" else (right, left)
    source_label = fragment_labels.get(source, f"Fragment {source}")
    target_label = fragment_labels.get(target, f"Fragment {target}")
    return source, target, f"{source_label}→{target_label}"


@dataclass(frozen=True)
class RetainedTransitionData:
    wavenumbers_cm: Any
    intensities: Any
    wavelengths_nm: Any = None


def _jsonable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _merge(target: dict, update: dict) -> dict:
    result = dict(target)
    for key, value in update.items():
        if key in ("ct_states", "ct_terms", "ct_spectra") and not value and result.get(key):
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = _jsonable(value)
    return result


def deduplicate_ct_rows(rows: list[dict]) -> list[dict]:
    """Collapse identical CT results emitted from more than one output folder."""
    unique = []
    seen = set()
    for raw_row in rows:
        row = dict(raw_row)
        key = json.dumps(
            {key: value for key, value in row.items() if key not in {"source", "calculation_id"}},
            sort_keys=True,
            separators=(",", ":"),
        )
        if key not in seen:
            seen.add(key)
            unique.append(row)
    return unique


def update_spectral_dataset(
    *,
    run,
    dataset_name: str,
    payload: dict[str, Any],
    source_artifact=None,
) -> ComputedProperty:
    """Merge lossless numerical spectrum data into a versioned run property."""
    payload = dict(payload)
    for key in ("ct_states", "ct_terms"):
        if key in payload:
            payload[key] = deduplicate_ct_rows(list(payload[key] or []))
    existing = ComputedProperty.objects.filter(
        workflow_run=run,
        name=PROPERTY_NAME,
        unit="",
    ).first()
    metadata = dict(existing.metadata or {}) if existing else {}
    datasets = dict(metadata.get("datasets") or {})
    datasets[str(dataset_name)] = _merge(
        dict(datasets.get(str(dataset_name)) or {}),
        payload,
    )
    canonical = {
        "schema_version": SCHEMA_VERSION,
        "datasets": datasets,
    }
    canonical["content_sha256"] = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    prop, _ = ComputedProperty.objects.update_or_create(
        workflow_run=run,
        name=PROPERTY_NAME,
        unit="",
        defaults={
            "value_text": f"schema:{SCHEMA_VERSION}",
            "value_float": None,
            "source_artifact": source_artifact or getattr(existing, "source_artifact", None),
            "metadata": canonical,
        },
    )
    return prop


def load_spectral_dataset(run, dataset_name: str | None = None):
    prop = ComputedProperty.objects.filter(
        workflow_run=run,
        name=PROPERTY_NAME,
        unit="",
    ).first()
    datasets = dict((prop.metadata if prop else {}).get("datasets") or {})
    return datasets.get(dataset_name) if dataset_name is not None else datasets


def transition_data_from_dataset(run, dataset_name: str):
    """Rebuild the plotting TransitionData without consulting source files."""
    import numpy as np

    dataset = load_spectral_dataset(run, dataset_name) or {}
    rows = [
        row
        for row in list(dataset.get("transitions") or [])
        if row.get("oscillator_strength") is not None
        and (row.get("wavenumber_cm") is not None or row.get("wavelength_nm") is not None)
    ]
    wavenumbers = [
        row.get("wavenumber_cm")
        if row.get("wavenumber_cm") is not None
        else 1.0e7 / float(row["wavelength_nm"])
        for row in rows
    ]
    return RetainedTransitionData(
        wavenumbers_cm=np.asarray(wavenumbers, dtype=float),
        intensities=np.asarray([row["oscillator_strength"] for row in rows], dtype=float),
        wavelengths_nm=(
            np.asarray([row["wavelength_nm"] for row in rows], dtype=float)
            if rows and all(row.get("wavelength_nm") is not None for row in rows)
            else None
        ),
    )


def ct_series_from_dataset(run, dataset_name: str) -> dict[str, list[dict]]:
    dataset = load_spectral_dataset(run, dataset_name) or {}
    return {
        "|".join(
            filter(
                None,
                (
                    str(series.get("source") or ""),
                    str(series.get("property") or ""),
                ),
            )
        )
        or str(index): list(series.get("points") or [])
        for index, series in enumerate(dataset.get("ct_spectra") or [], start=1)
    }


def ct_plot_dataset_names(run) -> list[str]:
    """Return retained datasets containing enough CT data to draw a spectrum."""
    names = []
    for name, dataset in (load_spectral_dataset(run) or {}).items():
        spectra = any(series.get("points") for series in dataset.get("ct_spectra") or [])
        states = any(
            row.get("wavelength_nm") is not None
            and row.get("oscillator_strength") is not None
            for row in dataset.get("ct_states") or []
        )
        if spectra or states:
            names.append(str(name))
    return names


def normalize_ct_dataset_rows(run) -> bool:
    changed = False
    for dataset_name, dataset in (load_spectral_dataset(run) or {}).items():
        payload = {}
        for key in ("ct_states", "ct_terms"):
            rows = list(dataset.get(key) or [])
            unique = deduplicate_ct_rows(rows)
            if unique != rows:
                payload[key] = unique
        if payload:
            update_spectral_dataset(run=run, dataset_name=dataset_name, payload=payload)
            changed = True
    return changed


def write_ct_plot_artifacts(run) -> list[Artifact]:
    """Create portable plots from retained CT data and register them as artifacts."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np

    artifacts = []
    datasets = load_spectral_dataset(run) or {}
    cache_root = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()
    run_key = str(getattr(run, "run_uid", "") or f"run_{run.pk}")

    for dataset_name in ct_plot_dataset_names(run):
        dataset = dict(datasets[dataset_name])
        dataset["ct_states"] = deduplicate_ct_rows(list(dataset.get("ct_states") or []))
        dataset["ct_terms"] = deduplicate_ct_rows(list(dataset.get("ct_terms") or []))
        fragment_labels = {
            str(key): str(value)
            for key, value in ((dataset.get("fragments") or {}).get("fragment_labels") or {}).items()
        }
        tddft_points = list(
            ((dataset.get("broadened_spectra") or {}).get("final_tddft_spectrum") or {}).get("points")
            or []
        )
        tddft_wavelengths = [
            float(point["x"])
            for point in tddft_points
            if isinstance(point.get("x"), (int, float)) and math.isfinite(float(point["x"]))
        ]
        plot_window_nm = (
            (min(tddft_wavelengths), max(tddft_wavelengths))
            if len(tddft_wavelengths) >= 2
            else (300.0, 800.0)
        )
        source_hash = hashlib.sha256(
            json.dumps(
                {
                    "plot_version": CT_PLOT_VERSION,
                    "plot_window_nm": plot_window_nm,
                    "ct_spectra": dataset.get("ct_spectra") or [],
                    "ct_states": dataset.get("ct_states") or [],
                    "ct_terms": dataset.get("ct_terms") or [],
                    "fragments": dataset.get("fragments") or {},
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", dataset_name).strip("._") or "spectrum"
        relative_path = f"generated/{safe_name}_ct_spectrum.png"
        output_path = (cache_root / run_key / relative_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
        plotted = False
        pair_styles: dict[tuple[str, str], str] = {}

        def ct_style(label: str) -> tuple[str, str]:
            match = re.search(r"(\d+)\s*(->|<-)\s*(\d+)", label)
            if not match:
                return "#475569", "-"
            left, arrow, right = match.groups()
            pair = tuple(sorted((left, right), key=lambda value: int(value)))
            if pair not in pair_styles:
                pair_styles[pair] = _CT_LINESTYLES[len(pair_styles) % len(_CT_LINESTYLES)]
            return ("#dc2626" if arrow == "->" else "#2563eb"), pair_styles[pair]

        states = []
        for row in dataset.get("ct_states") or []:
            try:
                wavelength = float(row["wavelength_nm"])
                strength = float(row["oscillator_strength"])
                state = int(row["state"])
            except (KeyError, TypeError, ValueError):
                continue
            if wavelength > 0 and strength >= 0:
                states.append((state, 1239.841984 / wavelength, strength))

        for index, series in enumerate([] if states else dataset.get("ct_spectra") or [], start=1):
            label = str(series.get("property") or series.get("source") or f"CT series {index}")
            descriptor = f"{label} {series.get('source') or ''}".lower()
            if any(token in descriptor for token in ("redis", "total", "uv-vis", "uvvis")):
                continue
            points = [point for point in series.get("points") or [] if point.get("intensity") is not None]
            xy = []
            for point in points:
                wavelength = point.get("wavelength_nm")
                if wavelength is None and point.get("energy_eV"):
                    wavelength = 1239.841984 / float(point["energy_eV"])
                if wavelength is not None:
                    xy.append((float(wavelength), float(point["intensity"])))
            if not xy:
                continue
            xy.sort()
            direction = _ct_direction(label, fragment_labels)
            color, linestyle = ct_style(label)
            axis.plot(
                [point[0] for point in xy],
                [point[1] for point in xy],
                marker="o" if len(xy) == 1 else None,
                linewidth=1.8,
                color=color,
                linestyle=linestyle,
                label=direction[2] if direction else label,
            )
            plotted = True

        linewidth_nm = 20.0
        minimum_ct_pct = 0.5
        minimum_annotation_signal_fraction = 0.10
        minimum_peak_prominence_fraction = 0.01
        minimum_peak_distance_nm = 20.0
        component_count = 0
        net_pct_by_state = {}
        max_positive = None
        max_negative = None
        integrated_positive_area_pct = None
        integrated_negative_area_pct = None
        resolved_peak_contributors = []
        x = None
        state_profiles = {}
        profiles = {}
        total_profile = None
        if states:
            x = np.arange(200.0, 1001.0, 1.0)
            for state, energy, strength in states:
                wavelength = 1239.841984 / energy
                state_profiles[state] = strength * np.exp(
                    -4.0 * math.log(2.0) * ((x - wavelength) / linewidth_nm) ** 2
                )
            total_profile = sum(state_profiles.values(), np.zeros_like(x))
        if states and not plotted:
            for series in dataset.get("ct_spectra") or []:
                raw_label = str(series.get("property") or series.get("source") or "")
                match = re.search(r"(\d+)\s*(->|<-)\s*(\d+)", raw_label)
                if not match:
                    continue
                label = "".join(match.groups())
                profile = np.zeros_like(x)
                for point in series.get("points") or []:
                    try:
                        intensity = float(point["intensity"])
                        wavelength = point.get("wavelength_nm")
                        if wavelength is None:
                            wavelength = 1239.841984 / float(point["energy_eV"])
                        wavelength = float(wavelength)
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        continue
                    profile += intensity * np.exp(
                        -4.0 * math.log(2.0) * ((x - wavelength) / linewidth_nm) ** 2
                    )
                if np.any(profile):
                    profiles[label] = profile
            retained_profile_labels = set(profiles)
            for term in dataset.get("ct_terms") or []:
                try:
                    peak = state_profiles[int(term["state"])]
                    percentage = float(term["percentage"])
                except (KeyError, TypeError, ValueError):
                    continue
                label = str(term.get("term") or "unassigned")
                if (
                    "redis" in label.lower()
                    or "total" in label.lower()
                    or abs(percentage) < minimum_ct_pct
                ):
                    continue
                direction = -1.0 if "<-" in label else 1.0
                if label not in retained_profile_labels:
                    profiles[label] = (
                        profiles.get(label, np.zeros_like(x))
                        + peak * abs(percentage) / 100.0
                    )
                state = int(term["state"])
                net_pct_by_state[state] = net_pct_by_state.get(state, 0.0) + abs(percentage) * direction
            for label, profile in sorted(
                profiles.items(), key=lambda item: float(item[1].max()), reverse=True
            ):
                if np.any(profile):
                    direction = _ct_direction(label, fragment_labels)
                    display_label = direction[2] if direction else label.replace("->", "→").replace("<-", "←")
                    color, linestyle = ct_style(label)
                    axis.plot(
                        x,
                        profile,
                        linewidth=1.8,
                        color=color,
                        linestyle=linestyle,
                        label=display_label,
                    )
                    component_count += 1
            plotted = component_count > 0
            state_lookup = {state: 1239.841984 / energy for state, energy, _ in states}
            forward_profile = sum(
                (profile for label, profile in profiles.items() if "->" in label),
                np.zeros_like(x),
            )
            reverse_profile = sum(
                (profile for label, profile in profiles.items() if "<-" in label),
                np.zeros_like(x),
            )
            plot_mask = (x >= plot_window_nm[0]) & (x <= plot_window_nm[1])
            plot_x = x[plot_mask]
            total_area = float(np.trapezoid(total_profile[plot_mask], plot_x))
            if total_area > 0:
                integrated_positive_area_pct = (
                    float(np.trapezoid(forward_profile[plot_mask], plot_x)) / total_area * 100.0
                )
                integrated_negative_area_pct = (
                    float(np.trapezoid(reverse_profile[plot_mask], plot_x)) / total_area * 100.0
                )
            positive = [
                (pct, state, float(np.interp(state_lookup[state], x, forward_profile) / forward_profile.max()))
                for state, pct in net_pct_by_state.items()
                if pct >= minimum_ct_pct
                and plot_window_nm[0] <= state_lookup[state] <= plot_window_nm[1]
                and forward_profile.max() > 0
                and np.interp(state_lookup[state], x, forward_profile)
                >= forward_profile.max() * minimum_annotation_signal_fraction
            ]
            negative = [
                (pct, state, float(np.interp(state_lookup[state], x, reverse_profile) / reverse_profile.max()))
                for state, pct in net_pct_by_state.items()
                if pct <= -minimum_ct_pct
                and plot_window_nm[0] <= state_lookup[state] <= plot_window_nm[1]
                and reverse_profile.max() > 0
                and np.interp(state_lookup[state], x, reverse_profile)
                >= reverse_profile.max() * minimum_annotation_signal_fraction
            ]
            if positive:
                pct, state, relative_signal = max(positive)
                max_positive = {
                    "percentage": pct,
                    "state": state,
                    "wavelength_nm": state_lookup[state],
                    "relative_signal": relative_signal,
                }
            if negative:
                pct, state, relative_signal = min(negative)
                max_negative = {
                    "percentage": pct,
                    "state": state,
                    "wavelength_nm": state_lookup[state],
                    "relative_signal": relative_signal,
                }
            from scipy.signal import find_peaks

            visible_indices = np.flatnonzero(plot_mask)
            visible_total = total_profile[plot_mask]
            peak_indices, peak_properties = find_peaks(
                visible_total,
                prominence=float(visible_total.max()) * minimum_peak_prominence_fraction,
                distance=max(1, round(minimum_peak_distance_nm)),
            )
            state_strengths = {state: strength for state, _, strength in states}
            for peak_index, prominence in zip(
                peak_indices,
                peak_properties["prominences"],
            ):
                full_index = int(visible_indices[peak_index])
                state = max(
                    state_profiles,
                    key=lambda candidate: float(state_profiles[candidate][full_index]),
                )
                excitation_wavelength = float(state_lookup[state])
                peak_height = float(total_profile[full_index])
                contribution = float(state_profiles[state][full_index])
                resolved_peak_contributors.append(
                    {
                        "peak_wavelength_nm": float(x[full_index]),
                        "peak_intensity": peak_height,
                        "prominence": float(prominence),
                        "state": state,
                        "excitation_wavelength_nm": excitation_wavelength,
                        "oscillator_strength": float(state_strengths[state]),
                        "peak_intensity_fraction": contribution / peak_height,
                        "net_ct_percentage": net_pct_by_state.get(state),
                    }
                )

        if not plotted:
            plt.close(fig)
            continue

        axis.set_xlabel("Wavelength (nm)", fontsize=13, fontweight="bold")
        axis.set_ylabel("CT-weighted oscillator strength (a.u.)", fontsize=13, fontweight="bold")
        axis.tick_params(axis="both", which="major", labelsize=11)
        axis.set_xlim(*plot_window_nm)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=11)
        if max_positive:
            axis.text(
                0.015,
                0.97,
                f"Max net +CT: +{max_positive['percentage']:.1f}% · "
                f"{max_positive['wavelength_nm']:.1f} nm · S{max_positive['state']}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="#dc2626",
                fontsize=11.5,
                fontweight="bold",
            )
        if max_negative:
            axis.text(
                0.015,
                0.90,
                f"Max net −CT: {max_negative['percentage']:.1f}% · "
                f"{max_negative['wavelength_nm']:.1f} nm · S{max_negative['state']}",
                transform=axis.transAxes,
                ha="left",
                va="top",
                color="#2563eb",
                fontsize=11.5,
                fontweight="bold",
            )
        if integrated_positive_area_pct is not None:
            axis.text(
                0.985,
                0.86,
                f"Integrated 1→2 area: {integrated_positive_area_pct:.1f}% of total UV–vis",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color="#dc2626",
                fontsize=11,
                fontweight="bold",
            )
        if integrated_negative_area_pct is not None:
            axis.text(
                0.985,
                0.79,
                f"Integrated 1←2 area: {integrated_negative_area_pct:.1f}% of total UV–vis",
                transform=axis.transAxes,
                ha="right",
                va="top",
                color="#2563eb",
                fontsize=11,
                fontweight="bold",
            )
        valid_contributors = [
            c for c in resolved_peak_contributors if c.get("net_ct_percentage") is not None
        ]
        for index, contributor in enumerate(valid_contributors):
            net_ct = contributor["net_ct_percentage"]
            color = "#dc2626" if net_ct > 0 else "#2563eb"
            net_ct_text = f"{net_ct:+.1f}% net CT"
            wavelength = contributor["excitation_wavelength_nm"]
            axis.axvline(wavelength, color=color, linestyle=":", linewidth=1.4, alpha=0.85)
            axis.text(
                wavelength + 3.0,
                0.66 - 0.14 * index,
                f"{contributor['peak_wavelength_nm']:.0f} nm peak: S{contributor['state']} "
                f"at {wavelength:.1f} nm\n"
                f"{net_ct_text} · "
                f"{contributor['peak_intensity_fraction']:.0%} Int.",
                transform=axis.get_xaxis_transform(),
                ha="left",
                va="center",
                color=color,
                fontsize=10.5,
                fontweight="semibold",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
            )
        axis.text(
            0.995,
            0.015,
            "Recreated from retained CT data"
            + (
                f" · Gaussian FWHM {linewidth_nm:g} nm"
                f" · annotations ≥{minimum_annotation_signal_fraction:.0%} of directional peak"
                if states
                else ""
            ),
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            color="#64748b",
            fontsize=9.0,
        )
        fig.savefig(output_path, dpi=200, format="png")
        plt.close(fig)

        transition_count = sum(
            row.get("wavelength_nm") is not None and row.get("oscillator_strength") is not None
            for row in dataset.get("transitions") or []
        )
        if total_profile is not None and len(states) > transition_count:
            uvvis_relative_path = f"generated/{safe_name}_uvvis.png"
            uvvis_output_path = (cache_root / run_key / uvvis_relative_path).resolve()
            uvvis_output_path.parent.mkdir(parents=True, exist_ok=True)
            uvvis_fig, uvvis_axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
            uvvis_axis.plot(x, total_profile, color="#7c3aed", linewidth=2.0)
            uvvis_axis.set(
                xlim=(200.0, 1000.0),
                xlabel="Wavelength (nm)",
                ylabel="Oscillator strength (a.u.)",
            )
            uvvis_axis.grid(alpha=0.2)
            uvvis_axis.text(
                0.985,
                0.96,
                f"Source: retained IFCT states ({len(states)} states; "
                f"ordinary TDDFT table: {transition_count})",
                transform=uvvis_axis.transAxes,
                ha="right",
                va="top",
                color="#475569",
                fontsize=8.5,
            )
            uvvis_axis.text(
                0.985,
                0.02,
                f"Gaussian FWHM {linewidth_nm:g} nm",
                transform=uvvis_axis.transAxes,
                ha="right",
                va="bottom",
                color="#64748b",
                fontsize=7.5,
            )
            uvvis_fig.savefig(uvvis_output_path, dpi=180, format="png")
            plt.close(uvvis_fig)
            Artifact.objects.update_or_create(
                workflow_run=run,
                path=uvvis_relative_path,
                defaults={
                    "kind": ArtifactKind.IMAGE,
                    "label": f"plot:uvvis:{dataset_name}",
                    "checksum": hashlib.sha256(uvvis_output_path.read_bytes()).hexdigest(),
                    "metadata": {
                        "source": "retained_ifct_states",
                        "dataset_name": dataset_name,
                        "local_cache_path": normalize_run_path(uvvis_output_path),
                        "retained_data_sha256": source_hash,
                        "source_state_count": len(states),
                        "ordinary_transition_count": transition_count,
                        "selection_reason": "IFCT contains more states than the ordinary TDDFT table",
                        "x_unit": "nm",
                        "wavelength_range_nm": [200.0, 1000.0],
                        "gaussian_linewidth_nm": linewidth_nm,
                        "gaussian_linewidth_convention": "fwhm",
                    },
                },
            )
            Artifact.objects.filter(
                workflow_run=run,
                label="plot:legacy_uvvis",
            ).update(label="legacy:uvvis_original")

        checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
        artifact, _ = Artifact.objects.update_or_create(
            workflow_run=run,
            path=relative_path,
            defaults={
                "kind": ArtifactKind.IMAGE,
                "label": f"plot:charge_transfer:{dataset_name}",
                "checksum": checksum,
                "metadata": {
                    "source": "retained_spectral_dataset",
                    "dataset_name": dataset_name,
                    "local_cache_path": normalize_run_path(output_path),
                    "retained_data_sha256": source_hash,
                    "state_count": len(states),
                    "component_count": component_count,
                    "fragment_labels": fragment_labels,
                    "direction_color_convention": {
                        "->": "#dc2626",
                        "<-": "#2563eb",
                    },
                    "fragment_pair_linestyles": {
                        "-".join(pair): linestyle for pair, linestyle in pair_styles.items()
                    },
                    "minimum_ct_percentage": (
                        minimum_ct_pct if states else None
                    ),
                    "minimum_annotation_signal_fraction": (
                        minimum_annotation_signal_fraction
                        if states
                        else None
                    ),
                    "max_net_positive_ct": max_positive,
                    "max_net_negative_ct": max_negative,
                    "integrated_positive_ct_area_pct": integrated_positive_area_pct,
                    "integrated_negative_ct_area_pct": integrated_negative_area_pct,
                    "resolved_peak_contributors": resolved_peak_contributors,
                    "minimum_peak_prominence_fraction": minimum_peak_prominence_fraction,
                    "minimum_peak_distance_nm": minimum_peak_distance_nm,
                    "integrated_area_denominator": (
                        "total_oscillator_strength_in_plot_window"
                        if integrated_positive_area_pct is not None
                        else None
                    ),
                    "x_unit": "nm",
                    "wavelength_range_nm": list(plot_window_nm),
                    "gaussian_linewidth_nm": (
                        linewidth_nm if states else None
                    ),
                    "gaussian_linewidth_convention": "fwhm" if states else None,
                },
            },
        )
        artifacts.append(artifact)
        pair_profiles: dict[tuple[str, str], list[tuple[str, Any, str]]] = {}
        for label, profile in profiles.items():
            direction = _ct_direction(label, fragment_labels)
            if direction is None or not np.any(profile):
                continue
            source, target, display_label = direction
            pair = tuple(sorted((source, target), key=lambda value: int(value)))
            color = "#dc2626" if "->" in label else "#2563eb"
            pair_profiles.setdefault(pair, []).append((display_label, profile, color))
        for pair, directions in pair_profiles.items():
            pair_labels = [fragment_labels.get(key, f"Fragment {key}") for key in pair]
            pair_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", "_".join(pair_labels)).strip("._").lower()
            pair_relative_path = f"generated/{safe_name}_ct_{pair_slug}.png"
            pair_output_path = (cache_root / run_key / pair_relative_path).resolve()
            pair_fig, pair_axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
            for display_label, profile, color in directions:
                pair_axis.plot(x, profile, linewidth=1.8, color=color, label=display_label)
            for contributor in resolved_peak_contributors:
                if contributor.get("net_ct_percentage") is None:
                    continue
                pair_axis.axvline(
                    contributor["excitation_wavelength_nm"],
                    color="#64748b",
                    linestyle=":",
                    linewidth=1.2,
                    alpha=0.75,
                )
            pair_axis.set_xlabel("Wavelength (nm)", fontsize=13, fontweight="bold")
            pair_axis.set_ylabel("CT-weighted oscillator strength (a.u.)", fontsize=13, fontweight="bold")
            pair_axis.tick_params(axis="both", which="major", labelsize=11)
            pair_axis.set_xlim(*plot_window_nm)
            pair_axis.grid(alpha=0.25)
            pair_axis.legend(frameon=False, fontsize=11)
            pair_fig.savefig(pair_output_path, dpi=200, format="png")
            plt.close(pair_fig)
            pair_artifact, _ = Artifact.objects.update_or_create(
                workflow_run=run,
                path=pair_relative_path,
                defaults={
                    "kind": ArtifactKind.IMAGE,
                    "label": f"plot:charge_transfer:{dataset_name}:{pair_slug}",
                    "checksum": hashlib.sha256(pair_output_path.read_bytes()).hexdigest(),
                    "metadata": {
                        "source": "retained_spectral_dataset",
                        "dataset_name": dataset_name,
                        "local_cache_path": normalize_run_path(pair_output_path),
                        "retained_data_sha256": source_hash,
                        "fragment_pair": pair_labels,
                        "directions": [label for label, _, _ in directions],
                        "excitation_lines_nm": [
                            row["excitation_wavelength_nm"] for row in resolved_peak_contributors
                        ],
                        "x_unit": "nm",
                        "wavelength_range_nm": list(plot_window_nm),
                        "gaussian_linewidth_nm": linewidth_nm,
                        "gaussian_linewidth_convention": "fwhm",
                    },
                },
            )
            artifacts.append(pair_artifact)
    return artifacts


def broadened_series_from_dataset(run, dataset_name: str) -> dict[str, list[dict]]:
    dataset = load_spectral_dataset(run, dataset_name) or {}
    retained = {}
    for spectrum_name, spectrum in (dataset.get("broadened_spectra") or {}).items():
        series = list(spectrum.get("series") or [])
        if series:
            for item in series:
                retained[f"{spectrum_name}|{item.get('name') or 'intensity'}"] = list(
                    item.get("points") or []
                )
        elif spectrum.get("points"):
            retained[str(spectrum_name)] = list(spectrum["points"])
    return retained


_CANONICAL_BROADENED_PLOTS = {
    "final_tddft_spectrum": (
        "plot:final_tddft_spectrum",
        "Final TDDFT spectrum",
        "Oscillator strength (a.u.)",
    ),
    "cluster_tddft_spectrum": (
        "plot:cluster_tddft_spectrum",
        "Isolated nanocluster TDDFT spectrum",
        "Oscillator strength (a.u.)",
    ),
    "cluster_adsorbate_difference_spectrum": (
        "plot:cluster_adsorbate_difference_spectrum",
        "Adsorbate-induced TDDFT difference spectrum",
        "Common-peak normalized delta intensity",
    ),
}


def broadened_plot_candidate_count(run) -> int:
    return sum(
        spectrum_name in _CANONICAL_BROADENED_PLOTS
        for dataset in (load_spectral_dataset(run) or {}).values()
        for spectrum_name in (dataset.get("broadened_spectra") or {})
    )


def _difference_plot_metadata(run, spectrum: dict[str, Any]) -> dict[str, Any]:
    source_path = str(spectrum.get("source_artifact_path") or "")
    artifact = Artifact.objects.filter(workflow_run=run, path=source_path).first()
    cache_path = Path(str(((artifact.metadata if artifact else {}) or {}).get("local_cache_path") or ""))
    try:
        with cache_path.open(newline="", encoding="utf-8-sig") as handle:
            row = next(csv.DictReader(handle))
        return {
            "plot_left_nm": float(row["plot_left_nm"]),
            "plot_right_nm": float(row["plot_right_nm"]),
            "normalization": str(row.get("normalization_mode") or "common_peak"),
            "normalization_factor": float(row["normalization_factor"]),
        }
    except (OSError, StopIteration, KeyError, TypeError, ValueError):
        return {}


def write_broadened_plot_artifacts(run) -> list[Artifact]:
    """Refresh canonical TDDFT plots from their losslessly retained curves."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    artifacts = []
    cache_root = Path(settings.REACTION_HUB_REMOTE_CACHE_ROOT).resolve()
    run_key = str(getattr(run, "run_uid", "") or f"run_{run.pk}")
    for dataset_name, dataset in (load_spectral_dataset(run) or {}).items():
        for spectrum_name, spectrum in (dataset.get("broadened_spectra") or {}).items():
            spec = _CANONICAL_BROADENED_PLOTS.get(str(spectrum_name))
            if spec is None:
                continue
            label, _, ylabel = spec
            series = list(spectrum.get("series") or [])
            preferred_name = (
                "delta_intensity_common_peak"
                if spectrum_name == "cluster_adsorbate_difference_spectrum"
                else "intensity"
            )
            selected = next(
                (item for item in series if item.get("name") == preferred_name),
                series[0] if series else {"points": spectrum.get("points") or []},
            )
            points = []
            for point in selected.get("points") or spectrum.get("points") or []:
                try:
                    x_value = float(point["x"])
                    intensity = float(point["intensity"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(x_value) and math.isfinite(intensity):
                    points.append((x_value, intensity))
            points.sort()
            if len(points) < 2:
                continue

            extra = (
                _difference_plot_metadata(run, spectrum)
                if spectrum_name == "cluster_adsorbate_difference_spectrum"
                else {}
            )
            plot_window = (
                float(extra.get("plot_left_nm", points[0][0])),
                float(extra.get("plot_right_nm", points[-1][0])),
            )
            visible = [point for point in points if plot_window[0] <= point[0] <= plot_window[1]]
            if visible:
                points = visible
            safe_dataset = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset_name)).strip("._")
            relative_path = f"generated/{safe_dataset or 'spectrum'}_{spectrum_name}.png"
            output_path = (cache_root / run_key / relative_path).resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)

            fig, axis = plt.subplots(figsize=(9.2, 5.4), constrained_layout=True)
            if spectrum_name == "cluster_adsorbate_difference_spectrum":
                axis.axhline(0.0, color="#475569", linewidth=0.9)
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color="#2563eb",
                linewidth=1.8,
            )
            axis.set(
                xlim=plot_window,
                xlabel="Wavelength (nm)",
                ylabel=ylabel,
            )
            axis.grid(alpha=0.2)
            axis.text(
                0.995,
                0.015,
                "Recreated from retained spectral data",
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                color="#64748b",
                fontsize=7.5,
            )
            fig.savefig(output_path, dpi=180, format="png")
            plt.close(fig)

            retained_hash = hashlib.sha256(
                json.dumps(
                    {
                        "plot_version": BROADENED_PLOT_VERSION,
                        "dataset_name": dataset_name,
                        "spectrum_name": spectrum_name,
                        "series": selected.get("name") or "intensity",
                        "points": points,
                        "plot_window_nm": plot_window,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            metadata = {
                "source": "retained_spectral_dataset",
                "dataset_name": dataset_name,
                "spectrum_name": spectrum_name,
                "series_name": selected.get("name") or "intensity",
                "local_cache_path": normalize_run_path(output_path),
                "retained_data_sha256": retained_hash,
                "plot_version": BROADENED_PLOT_VERSION,
                "x_unit": "nm",
                "wavelength_range_nm": list(plot_window),
                "line_color": "#2563eb",
                **extra,
            }
            checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
            existing = list(Artifact.objects.filter(workflow_run=run, label=label).order_by("pk"))
            if existing:
                for artifact in existing:
                    artifact.kind = ArtifactKind.IMAGE
                    artifact.checksum = checksum
                    artifact.metadata = metadata
                    artifact.save(update_fields=["kind", "checksum", "metadata"])
                artifact = existing[0]
            else:
                artifact = Artifact.objects.create(
                    workflow_run=run,
                    path=relative_path,
                    kind=ArtifactKind.IMAGE,
                    label=label,
                    checksum=checksum,
                    metadata=metadata,
                )
            artifacts.append(artifact)
    return artifacts


def spectrum_curve_from_csv(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    if not rows:
        return None
    fieldnames = {str(key).strip().lower(): key for key in rows[0]}
    x_key = next(
        (fieldnames[key] for key in ("wavelength_nm", "energy_ev", "wavenumber_cm") if key in fieldnames),
        None,
    )
    y_keys = [
        original
        for normalized, original in fieldnames.items()
        if "intensity" in normalized or "oscillator_strength" in normalized
    ]
    if x_key is None or not y_keys:
        return None
    series = []
    for y_key in y_keys:
        points = []
        for row in rows:
            try:
                points.append({"x": float(row[x_key]), "intensity": float(row[y_key])})
            except (TypeError, ValueError):
                continue
        if points:
            series.append(
                {
                    "name": str(y_key),
                    "intensity_unit": "a.u.",
                    "points": points,
                }
            )
    if not series:
        return None
    normalized_x = str(x_key).strip().lower()
    result = {
        "x_unit": {"wavelength_nm": "nm", "energy_ev": "eV", "wavenumber_cm": "cm^-1"}[normalized_x],
        "series": series,
    }
    if len(series) == 1:
        result.update(
            {
                "intensity_unit": series[0]["intensity_unit"],
                "points": series[0]["points"],
            }
        )
    return result
