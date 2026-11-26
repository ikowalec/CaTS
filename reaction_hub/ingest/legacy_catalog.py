from __future__ import annotations

import ast
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import uuid

from ase.io import read as ase_read
from django.db import transaction

from references.models import CalculationSettingsProfile
from results.models import Artifact, ArtifactKind
from workflows.identity import normalize_run_path
from workflows.models import WorkflowDefinition, WorkflowRun, WorkflowStatus

from .services import (
    _store_charges_payload,
    _store_structure_payload,
    _store_tddft_summary_payload,
    _upsert_property,
)
from .spectral import update_spectral_dataset


HARTREE_TO_EV = 27.211386245988
CATALOG_TABLES = (
    "calculations.csv",
    "audit.csv",
    "transitions.csv",
    "spectra.csv",
    "atomic_charges.csv",
    "artifacts.csv",
    "ct_states.csv",
    "ct_terms.csv",
    "ct_spectra.csv",
    "scan_frames.csv",
)


def discover_catalogs(root: str | Path) -> list[Path]:
    root = Path(root).resolve()
    if (root / "calculations.csv").is_file():
        return [root]
    return sorted({path.parent for path in root.rglob("calculations.csv")})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _group(rows: list[dict], key: str = "calculation_id") -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            grouped.setdefault(value, []).append(row)
    return grouped


def _number(value, integer: bool = False):
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    try:
        parsed = float(text)
        if not math.isfinite(parsed):
            return None
        return int(parsed) if integer else parsed
    except (OverflowError, ValueError):
        return None


def _structured(value):
    text = str(value or "").strip()
    if not text:
        return []
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
    return text


def _boolean(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes"}


@lru_cache(maxsize=None)
def _checksum_cached(path_text: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with Path(path_text).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checksum(path: Path) -> str:
    if not path.is_file():
        return ""
    stat = path.stat()
    return _checksum_cached(str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))


def _artifact(
    run: WorkflowRun,
    path: Path,
    *,
    kind: str,
    label: str,
    stored_path: str | Path | None = None,
    metadata: dict | None = None,
) -> Artifact:
    exists = path.is_file()
    normalized_path = normalize_run_path(stored_path if stored_path is not None else path)
    existing = Artifact.objects.filter(workflow_run=run, path=normalized_path).first()
    existing_metadata = existing.metadata if existing and isinstance(existing.metadata, dict) else {}
    artifact, _ = Artifact.objects.update_or_create(
        workflow_run=run,
        path=normalized_path,
        defaults={
            "job": None,
            "kind": kind,
            "label": label,
            "checksum": _checksum(path) if exists else "",
            "metadata": {
                **existing_metadata,
                "source": "legacy_orca_catalog",
                "exists_at_import": exists,
                "ingestion_path": normalize_run_path(path),
                **(metadata or {}),
            },
        },
    )
    return artifact


def _source_artifact(
    run: WorkflowRun,
    source_root: Path,
    calculation_directory: Path,
    source_relative_path: str | Path,
    *,
    kind: str,
    label: str,
    stored_path: str | Path | None = None,
    metadata: dict | None = None,
) -> Artifact:
    source_relative_path = Path(str(source_relative_path))
    if stored_path is None:
        try:
            stored_path = source_relative_path.relative_to(calculation_directory)
        except ValueError:
            stored_path = source_relative_path
    return _artifact(
        run,
        source_root / source_relative_path,
        kind=kind,
        label=label,
        stored_path=stored_path,
        metadata={
            "source_relative_path": normalize_run_path(source_relative_path),
            **(metadata or {}),
        },
    )


def _transition_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "index": _number(row.get("state"), integer=True),
            "wavenumber_cm": _number(row.get("wavenumber_cm")),
            "wavelength_nm": _number(row.get("wavelength_nm")),
            "oscillator_strength": _number(row.get("oscillator_strength")),
        }
        for row in rows
    ]


def _ct_state_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "source": row.get("source", ""),
            "state": _number(row.get("state"), integer=True),
            "oscillator_strength": _number(row.get("f")),
            "wavelength_nm": _number(row.get("nm")),
            "transferred_charge_e": _number(row.get("CT(e)")),
            "charge_transfer_pct": _number(row.get("CT(%)")),
        }
        for row in rows
    ]


def _ct_term_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "source": row.get("source", ""),
            "state": _number(row.get("state"), integer=True),
            "term": row.get("term", ""),
            "percentage": _number(row.get("pct")),
        }
        for row in rows
    ]


def _ct_spectral_series(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (str(row.get("source") or ""), str(row.get("property") or ""))
        grouped.setdefault(key, []).append(
            {
                "energy_eV": _number(row.get("energy_eV")),
                "wavelength_nm": _number(row.get("wavelength_nm")),
                "intensity": _number(row.get("intensity")),
            }
        )
    return [
        {
            "source": source,
            "property": property_name,
            "energy_unit": "eV",
            "wavelength_unit": "nm",
            "intensity_unit": "a.u.",
            "points": points,
        }
        for (source, property_name), points in grouped.items()
    ]


def _scan_rows(rows: list[dict]) -> list[dict]:
    result = []
    for row in rows:
        result.append(
            {
                "scan_id": row.get("scan_id", ""),
                "trajectory": row.get("trajectory", ""),
                "related_calculation_ids": _structured(row.get("related_calculation_ids")),
                "frame_index_0based": _number(row.get("frame_index_0based"), integer=True),
                "frame_index_1based": _number(row.get("frame_index_1based"), integer=True),
                "formula": row.get("formula", ""),
                "symbols": _structured(row.get("symbols")),
                "positions_A": _structured(row.get("positions_A")),
                "cell_A": _structured(row.get("cell_A")),
                "pbc": _structured(row.get("pbc")),
                "energy_eV": _number(row.get("energy_eV")),
                "forces_eV_A": _structured(row.get("forces_eV_A")),
                "fmax_eV_A": _number(row.get("fmax_eV_A")),
                "info": _structured(row.get("info")),
                "constraints": _structured(row.get("constraints")),
                "indices_0based": _structured(row.get("indices_0based")),
                "indices_1based": _structured(row.get("indices_1based")),
                "bond_pairs_0based": _structured(row.get("bond_pairs_0based")),
                "bond_pairs_1based": _structured(row.get("bond_pairs_1based")),
                "bond_distances_A": _structured(row.get("bond_distances_A")),
            }
        )
    return result


@transaction.atomic
def import_catalog(
    catalog_dir: str | Path,
    *,
    current_source_root: str | Path | None = None,
) -> dict[str, int]:
    catalog_dir = Path(catalog_dir).resolve()
    calculations = _read_csv(catalog_dir / "calculations.csv")
    if not calculations:
        raise ValueError(f"No calculations found in {catalog_dir / 'calculations.csv'}")
    try:
        catalog_summary = json.loads((catalog_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        catalog_summary = {}
    original_source_root = normalize_run_path(catalog_summary.get("root") or catalog_dir)
    source_tree_id = str(catalog_summary.get("source_tree_id") or original_source_root)
    source_root = Path(current_source_root or original_source_root).resolve()

    raw_tables = {
        name: _read_csv(catalog_dir / name)
        for name in CATALOG_TABLES
        if name not in {"calculations.csv", "scan_frames.csv"}
    }
    tables = {name: _group(rows) for name, rows in raw_tables.items()}
    scan_groups: dict[str, list[dict]] = {}
    for row in _read_csv(catalog_dir / "scan_frames.csv"):
        related = _structured(row.get("related_calculation_ids"))
        related = related if isinstance(related, list) else [related]
        for calculation_id in related:
            if str(calculation_id).strip():
                scan_groups.setdefault(str(calculation_id), []).append(row)
    tables["scan_frames.csv"] = scan_groups
    fragments = {}
    fragment_path = catalog_dir / "fragments.jsonl"
    if fragment_path.is_file():
        for line in fragment_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("calculation_id"):
                fragments[str(row["calculation_id"])] = row

    workflow, _ = WorkflowDefinition.objects.get_or_create(
        name="legacy_orca_cam_b3lyp_631gdp_gas",
        defaults={
            "code_path": "cats/orca/legacy_catalog.py",
            "description": "Imported legacy gas-phase CAM-B3LYP/6-31G(d,p) ORCA calculations.",
        },
    )
    modern_profile, _ = CalculationSettingsProfile.objects.get_or_create(
        name="cam-b3lyp-lanl2dz",
        defaults={
            "approved": True,
            "xc_functional": "CAM-B3LYP",
            "basis_set": "LANL2DZ",
            "solvent": "vacuum",
        },
    )

    created = updated = artifacts = 0
    for calculation in calculations:
        calculation_id = str(calculation.get("calculation_id") or "").strip()
        if not calculation_id:
            raise ValueError(f"Calculation without calculation_id in {catalog_dir}")
        orca_rel = normalize_run_path(calculation.get("orca_out") or "")
        calculation_fingerprint = str(calculation.get("calculation_fingerprint") or "").strip()
        calculation_identity = calculation_fingerprint or f"{source_tree_id}:{calculation_id}"
        calculation_directory = Path(orca_rel).parent
        working_directory = normalize_run_path(
            source_root / calculation_directory
        ).removesuffix("/.")
        run_uid = uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"legacy-orca:{calculation_identity}",
        )
        status = WorkflowStatus.COMPLETED if _boolean(calculation.get("normal_termination")) else WorkflowStatus.FAILED
        run = WorkflowRun.objects.filter(run_uid=run_uid).first()
        was_created = run is None
        if run is None:
            run = WorkflowRun(run_uid=run_uid)
        metadata = dict(run.metadata or {})
        previous_legacy = metadata.get("legacy_catalog") if isinstance(metadata.get("legacy_catalog"), dict) else {}
        source_root_history = {
            str(value)
            for value in [
                *(previous_legacy.get("source_root_history") or []),
                previous_legacy.get("source_root"),
                original_source_root,
            ]
            if value
        }
        metadata["legacy_catalog"] = {
            "schema_version": 2,
            "calculation_id": calculation_id,
            "calculation_fingerprint": calculation_fingerprint,
            "source_tree_id": source_tree_id,
            "catalog_directory": normalize_run_path(catalog_dir),
            "source_root": original_source_root,
            "source_root_history": sorted(source_root_history),
            "current_source_root": normalize_run_path(source_root),
            "working_directory_relative": normalize_run_path(calculation_directory),
            "calculation": calculation,
            "audit": (tables.get("audit.csv") or {}).get(calculation_id, []),
        }
        run.workflow = workflow
        run.run_name = f"{Path(orca_rel).parent.name or 'legacy'}-{calculation_id[:8]}"
        run.status = status
        run.working_directory = working_directory
        run.settings_profile = modern_profile
        run.metadata = metadata
        run.save()
        created += was_created
        updated += not was_created

        for row in (tables.get("artifacts.csv") or {}).get(calculation_id, []):
            relative_path = str(row.get("path") or "").strip()
            if not relative_path:
                continue
            kind = str(row.get("kind") or ArtifactKind.OTHER).strip().lower()
            if kind not in ArtifactKind.values:
                kind = ArtifactKind.OTHER
            _source_artifact(
                run,
                source_root,
                calculation_directory,
                relative_path,
                kind=kind,
                label=f"manifest:{Path(relative_path).name}",
                stored_path=str(row.get("path_relative_to_calculation") or "").strip() or None,
                metadata={
                    "parsed_size_bytes": _number(row.get("size_bytes"), integer=True),
                    "parsed_mtime_ns": _number(row.get("mtime_ns"), integer=True),
                    "parsed_sha256": str(row.get("sha256") or ""),
                },
            )
            artifacts += 1

        raw_log = _source_artifact(
            run,
            source_root,
            calculation_directory,
            orca_rel,
            kind=ArtifactKind.LOG,
            label="orca_out:legacy",
        )
        artifacts += 1
        energy_h = _number(calculation.get("energy_hartree"))
        if energy_h is not None:
            _upsert_property(
                run=run,
                name="final_single_point_energy",
                unit="Eh",
                value_float=energy_h,
                source_artifact=raw_log,
                metadata={"source": "legacy_orca_catalog"},
            )
            _upsert_property(
                run=run,
                name="final_single_point_energy_eV",
                unit="eV",
                value_float=energy_h * HARTREE_TO_EV,
                source_artifact=raw_log,
                metadata={"source": "legacy_orca_catalog"},
            )

        structure_rel = str(calculation.get("structure_file") or "").strip()
        if structure_rel:
            structure_path = catalog_dir / structure_rel
            structure_artifact = _artifact(
                run,
                structure_path,
                kind=ArtifactKind.STRUCTURE,
                label="structure:legacy_final",
                metadata={"is_final_structure": True},
            )
            artifacts += 1
            if structure_path.is_file():
                try:
                    atoms = ase_read(structure_path, index=-1, parallel=False)
                    _store_structure_payload(
                        run=run,
                        formula=atoms.get_chemical_formula(),
                        atom_count=len(atoms),
                        symbols=atoms.get_chemical_symbols(),
                        positions_A=atoms.positions.tolist(),
                        source_artifact=structure_artifact,
                    )
                except Exception:
                    pass

        charge_sets: dict[str, list[dict]] = {}
        for row in (tables.get("atomic_charges.csv") or {}).get(calculation_id, []):
            method = str(row.get("method") or "").strip().lower()
            if method:
                charge_sets.setdefault(method, []).append(row)
        for method in sorted(charge_sets, key=lambda value: value == "loewdin"):
            rows = sorted(
                charge_sets[method],
                key=lambda row: _number(row.get("atom_index_0based"), integer=True) or 0,
            )
            charges = [_number(row.get("charge_e")) for row in rows]
            if charges and all(value is not None for value in charges):
                _store_charges_payload(
                    run=run,
                    charges=charges,
                    symbols=[str(row.get("symbol") or "") for row in rows],
                    source_artifact=raw_log,
                    charge_method=method,
                )

        transition_rows = _transition_rows((tables.get("transitions.csv") or {}).get(calculation_id, []))
        if transition_rows:
            _store_tddft_summary_payload(
                run=run,
                tddft={
                    "wavenumbers_cm": [row["wavenumber_cm"] for row in transition_rows],
                    "wavelengths_nm": [row["wavelength_nm"] for row in transition_rows],
                    "oscillator_strengths": [row["oscillator_strength"] for row in transition_rows],
                },
                source_artifact=raw_log,
            )

        spectrum_rows = (tables.get("spectra.csv") or {}).get(calculation_id, [])
        ct_states = _ct_state_rows((tables.get("ct_states.csv") or {}).get(calculation_id, []))
        ct_terms = _ct_term_rows((tables.get("ct_terms.csv") or {}).get(calculation_id, []))
        ct_spectra = _ct_spectral_series((tables.get("ct_spectra.csv") or {}).get(calculation_id, []))
        scan_frames = _scan_rows((tables.get("scan_frames.csv") or {}).get(calculation_id, []))
        spectral_payload = {
            "provenance": {
                "source": "legacy_orca_catalog",
                "catalog_directory": normalize_run_path(catalog_dir),
                "calculation_id": calculation_id,
            },
            "transition_units": {
                "wavenumber_cm": "cm^-1",
                "wavelength_nm": "nm",
                "oscillator_strength": "dimensionless",
            },
            "ct_state_units": {
                "wavelength_nm": "nm",
                "transferred_charge_e": "e",
                "charge_transfer_pct": "%",
            },
        }
        if transition_rows:
            spectral_payload["transitions"] = transition_rows
        if spectrum_rows:
            spectral_payload["broadened_spectra"] = {
                "legacy_uvvis": {
                    "x_unit": "nm",
                    "intensity_unit": "a.u.",
                    "normalization": "raw",
                    "points": [
                        {
                            "x": _number(row.get("wavelength_nm")),
                            "intensity": _number(row.get("intensity")),
                        }
                        for row in spectrum_rows
                    ],
                }
            }
        if ct_states:
            spectral_payload["ct_states"] = ct_states
        if ct_terms:
            spectral_payload["ct_terms"] = ct_terms
        if ct_spectra:
            spectral_payload["ct_spectra"] = ct_spectra
        if fragments.get(calculation_id):
            spectral_payload["fragments"] = fragments[calculation_id]
        if scan_frames:
            spectral_payload["scan_frames"] = scan_frames
        update_spectral_dataset(
            run=run,
            dataset_name="final_tddft",
            payload=spectral_payload,
            source_artifact=raw_log,
        )

        declared = {
            "molden_files": ArtifactKind.OTHER,
            "ifct_files": ArtifactKind.OTHER,
            "ct_multiple_files": ArtifactKind.CSV,
        }
        for field, kind in declared.items():
            values = _structured(calculation.get(field))
            values = values if isinstance(values, list) else ([values] if values else [])
            for value in values:
                _source_artifact(
                    run,
                    source_root,
                    calculation_directory,
                    str(value),
                    kind=kind,
                    label=f"legacy:{Path(str(value)).name}",
                )
                artifacts += 1
        for series in ct_spectra:
            source = str(series.get("source") or "").strip()
            if source:
                _source_artifact(
                    run,
                    source_root,
                    calculation_directory,
                    source,
                    kind=ArtifactKind.CSV,
                    label=f"legacy:{Path(source).name}",
                )
                artifacts += 1
        config = str(calculation.get("fragment_config") or "").strip()
        if config:
            _source_artifact(
                run,
                source_root,
                calculation_directory,
                calculation_directory / config,
                kind=ArtifactKind.OTHER,
                label="legacy:fragment_config",
            )
            artifacts += 1
        plot = catalog_dir / "plots" / f"{calculation_id}_uvvis.png"
        if plot.is_file():
            _artifact(run, plot, kind=ArtifactKind.IMAGE, label="plot:legacy_uvvis")
            artifacts += 1
        for frame in scan_frames:
            trajectory = str(frame.get("trajectory") or "").strip()
            if trajectory:
                _source_artifact(
                    run,
                    source_root,
                    calculation_directory,
                    trajectory,
                    kind=ArtifactKind.TRAJECTORY,
                    label=f"trajectory:{Path(trajectory).name}",
                )
                artifacts += 1
        for table_name in CATALOG_TABLES:
            if table_name == "calculations.csv" or (tables.get(table_name) or {}).get(calculation_id):
                table_path = catalog_dir / table_name
                if table_path.is_file():
                    _artifact(
                        run,
                        table_path,
                        kind=ArtifactKind.CSV,
                        label=f"catalog:{table_name}",
                        metadata={"row_filter": {"calculation_id": calculation_id}},
                    )
                    artifacts += 1
        summary_path = catalog_dir / "summary.json"
        if summary_path.is_file():
            _artifact(run, summary_path, kind=ArtifactKind.JSON, label="catalog:summary.json")
            artifacts += 1
        if fragment_path.is_file() and calculation_id in fragments:
            _artifact(
                run,
                fragment_path,
                kind=ArtifactKind.JSON,
                label="catalog:fragments.jsonl",
                metadata={"row_filter": {"calculation_id": calculation_id}},
            )
            artifacts += 1

    imported_ids = [
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            "legacy-orca:"
            + (
                str(row.get("calculation_fingerprint") or "").strip()
                or f"{source_tree_id}:{row['calculation_id']}"
            ),
        )
        for row in calculations
        if row.get("calculation_id")
    ]
    missing_artifacts = Artifact.objects.filter(
        workflow_run__run_uid__in=imported_ids,
        metadata__exists_at_import=False,
    ).count()
    return {
        "catalogs": 1,
        "calculations": len(calculations),
        "created": created,
        "updated": updated,
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "rejected": sum(
            str(row.get("status") or "").strip().lower() == "rejected"
            for row in raw_tables.get("audit.csv", [])
        ),
    }


def import_catalog_tree(
    root: str | Path,
    *,
    current_source_root: str | Path | None = None,
) -> dict[str, int]:
    root = Path(root).resolve()
    summary = {
        "catalogs": 0,
        "calculations": 0,
        "created": 0,
        "updated": 0,
        "artifacts": 0,
        "missing_artifacts": 0,
        "rejected": 0,
    }
    catalogs = discover_catalogs(root)
    if not catalogs:
        raise ValueError(f"No legacy catalogs found under {root}")
    for catalog in catalogs:
        mapped_source_root = None
        if current_source_root is not None:
            mapped_source_root = Path(current_source_root).resolve() / catalog.relative_to(root)
        result = import_catalog(catalog, current_source_root=mapped_source_root)
        for key in summary:
            summary[key] += result[key]
    return summary
