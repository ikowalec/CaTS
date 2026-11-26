from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from tqdm.auto import tqdm


_INPUT_LINE_RE = re.compile(r"^\s*\|\s*\d+>\s?(.*)$")
_NUMBER_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+-]?\d+)?"
_COORD_RE = re.compile(
    rf"^\s*(?:\d+\s+)?([A-Z][a-z]?)\s+({_NUMBER_RE})\s+({_NUMBER_RE})\s+({_NUMBER_RE})(?:\s|$)"
)
_ENERGY_RE = re.compile(rf"FINAL SINGLE POINT ENERGY\s+({_NUMBER_RE})", re.IGNORECASE)
_TDDFT_MARKER = "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
CATALOG_SCHEMA_VERSION = 3
_PROGRAM_VERSION_RE = re.compile(r"Program Version\s+([^\r\n]+)", re.IGNORECASE)
_XYZ_INPUT_RE = re.compile(
    r"^\s*\*\s+(?:xyz|xyzfile|int)\s+([+-]?\d+)\s+(\d+)(?:\s|$)",
    re.IGNORECASE | re.MULTILINE,
)

CALCULATION_FIELDS = [
    "calculation_id",
    "calculation_fingerprint",
    "parse_fingerprint",
    "orca_out",
    "input_source",
    "input_file",
    "input_text",
    "orca_version",
    "charge",
    "multiplicity",
    "simple_keywords",
    "structure_file",
    "formula",
    "cluster_family",
    "adsorbate_formula",
    "atom_count",
    "energy_hartree",
    "normal_termination",
    "tddft_status",
    "transition_count",
    "ct_status",
    "fragment_config",
    "molden_files",
    "ifct_files",
    "ct_multiple_files",
    "artifact_count",
    "warnings",
]

ATOMIC_CHARGE_FIELDS = [
    "calculation_id",
    "method",
    "atom_index_0based",
    "atom_index_1based",
    "symbol",
    "charge_e",
]

ARTIFACT_FIELDS = [
    "calculation_id",
    "path",
    "path_relative_to_calculation",
    "kind",
    "size_bytes",
    "mtime_ns",
    "sha256",
]

REUSABLE_TABLES = (
    "transitions.csv",
    "spectra.csv",
    "atomic_charges.csv",
    "ct_states.csv",
    "ct_terms.csv",
    "ct_spectra.csv",
    "scan_frames.csv",
)


def _progress(items: Iterable[Any], *, description: str, verbosity: int, level: int = 0):
    return tqdm(
        items,
        desc=description,
        unit="file",
        disable=verbosity < level,
        dynamic_ncols=True,
    )


def _verbose(message: str, *, verbosity: int, level: int = 1) -> None:
    if verbosity >= level:
        tqdm.write(message)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: json.dumps(_jsonable(row.get(field)), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple, set))
                    else row.get(field, "")
                    for field in fields
                }
            )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _group_by_calculation(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        calculation_id = str(row.get("calculation_id") or "").strip()
        if calculation_id:
            grouped.setdefault(calculation_id, []).append(row)
    return grouped


def _group_scan_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        try:
            related = json.loads(str(row.get("related_calculation_ids") or "[]"))
        except ValueError:
            related = []
        for calculation_id in related if isinstance(related, list) else []:
            grouped.setdefault(str(calculation_id), []).append(row)
    return grouped


def _load_existing_catalog(output: Path) -> dict[str, Any] | None:
    try:
        summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if summary.get("schema_version") not in {2, CATALOG_SCHEMA_VERSION}:
        return None
    if any(not (output / name).is_file() for name in (*REUSABLE_TABLES, "artifacts.csv")):
        return None
    calculations = _read_csv(output / "calculations.csv")
    if not calculations:
        return None
    artifact_rows = _group_by_calculation(_read_csv(output / "artifacts.csv"))
    root_name = Path(str(summary.get("root") or "")).name
    for calculation in calculations:
        if calculation.get("parse_fingerprint"):
            continue
        calculation_id = str(calculation.get("calculation_id") or "")
        _, calculation["parse_fingerprint"] = _calculation_fingerprints(
            root_name,
            str(calculation.get("orca_out") or ""),
            artifact_rows.get(calculation_id, []),
        )
    fragments: dict[str, dict[str, Any]] = {}
    try:
        fragment_lines = (output / "fragments.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        fragment_lines = []
    for line in fragment_lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        calculation_id = str(row.get("calculation_id") or "").strip()
        if calculation_id:
            fragments[calculation_id] = row
    return {
        "calculations": {
            str(row.get("orca_out") or ""): row
            for row in calculations
            if row.get("orca_out")
        },
        "tables": {
            name: (
                _group_scan_rows(_read_csv(output / name))
                if name == "scan_frames.csv"
                else _group_by_calculation(_read_csv(output / name))
            )
            for name in REUSABLE_TABLES
        },
        "artifacts": artifact_rows,
        "fragments": fragments,
    }


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _calculation_fingerprints(
    root_name: str,
    orca_out: str,
    artifacts: list[dict[str, Any]],
) -> tuple[str, str]:
    raw_output_sha256 = next(
        (
            str(row.get("sha256") or "")
            for row in artifacts
            if str(row.get("path_relative_to_calculation") or "").lower() == "orca.out"
        ),
        "",
    )
    calculation_fingerprint = _fingerprint(
        {
            "root_name": root_name,
            "orca_out": orca_out,
            "orca_out_sha256": raw_output_sha256,
        }
    )
    payload = {
        "root_name": root_name,
        "orca_out": orca_out,
        "artifacts": sorted(
            [
                [
                    str(row.get("path_relative_to_calculation") or ""),
                    str(row.get("sha256") or ""),
                ]
                for row in artifacts
            ]
        ),
    }
    return calculation_fingerprint, _fingerprint(payload)


def _can_reuse_calculation(
    calculation: dict[str, Any] | None,
    parse_fingerprint: str,
    output: Path,
    tables: dict[str, dict[str, list[dict[str, Any]]]],
    fragments: dict[str, dict[str, Any]],
) -> bool:
    if not calculation or calculation.get("parse_fingerprint") != parse_fingerprint:
        return False
    calculation_id = str(calculation.get("calculation_id") or "")
    if "parse_error" in {
        str(calculation.get("tddft_status") or ""),
        str(calculation.get("ct_status") or ""),
    }:
        return False
    structure_file = str(calculation.get("structure_file") or "").strip()
    if structure_file and not (output / structure_file).is_file():
        return False
    if calculation.get("tddft_status") == "parsed":
        if not tables["transitions.csv"].get(calculation_id):
            return False
        if not tables["spectra.csv"].get(calculation_id):
            return False
        if not (output / "plots" / f"{calculation_id}_uvvis.png").is_file():
            return False
    if calculation.get("ct_status") == "parsed" and not any(
        tables[name].get(calculation_id)
        for name in ("ct_states.csv", "ct_terms.csv", "ct_spectra.csv")
    ):
        return False
    if calculation.get("fragment_config") and calculation_id not in fragments:
        return False
    return True


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_provenance(input_text: str, output_text: str) -> dict[str, Any]:
    version = _PROGRAM_VERSION_RE.search(output_text)
    charge = multiplicity = None
    xyz = _XYZ_INPUT_RE.search(input_text)
    if xyz:
        charge, multiplicity = int(xyz.group(1)), int(xyz.group(2))
    else:
        charge_match = re.search(r"^\s*Charge\s*(?:=\s*)?([+-]?\d+)", input_text, re.I | re.M)
        mult_match = re.search(r"^\s*Mult\s*(?:=\s*)?(\d+)", input_text, re.I | re.M)
        charge = int(charge_match.group(1)) if charge_match else None
        multiplicity = int(mult_match.group(1)) if mult_match else None
    return {
        "orca_version": version.group(1).strip() if version else "",
        "charge": charge,
        "multiplicity": multiplicity,
        "simple_keywords": [
            line.strip()[1:].strip()
            for line in input_text.splitlines()
            if line.strip().startswith("!")
        ],
    }


def _artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if path.name.lower() in {"orca.out", "orca.log"} or suffix in {".log", ".out"}:
        return "log"
    if suffix in {".xyz", ".extxyz", ".cif", ".pdb"}:
        return "structure"
    if suffix == ".traj":
        return "trajectory"
    if suffix in {".csv", ".tsv"}:
        return "csv"
    if suffix in {".json", ".jsonl"}:
        return "json"
    if suffix in {".png", ".jpg", ".jpeg", ".gif", ".svg"}:
        return "image"
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return "database"
    return "other"


def _artifact_manifest(
    calculation_id: str,
    directory: Path,
    root: Path,
    output: Path,
    calculation_directories: set[Path],
    previous_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    directory = directory.resolve()
    output = output.resolve()
    previous = {
        str(row.get("path_relative_to_calculation") or ""): row
        for row in (previous_rows or [])
    }
    for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
        resolved = path.resolve()
        try:
            resolved.relative_to(output)
            continue
        except ValueError:
            pass
        nearest_calculation = next(
            (
                parent.resolve()
                for parent in (resolved.parent, *resolved.parents)
                if parent.resolve() in calculation_directories
            ),
            None,
        )
        if nearest_calculation != directory:
            continue
        try:
            stat = resolved.stat()
            relative_path = resolved.relative_to(directory).as_posix()
            previous_row = previous.get(relative_path, {})
            previous_matches = (
                str(previous_row.get("size_bytes") or "") == str(int(stat.st_size))
                and str(previous_row.get("mtime_ns") or "") == str(int(stat.st_mtime_ns))
                and bool(previous_row.get("sha256"))
            )
            rows.append(
                {
                    "calculation_id": calculation_id,
                    "path": _relative(resolved, root),
                    "path_relative_to_calculation": relative_path,
                    "kind": _artifact_kind(resolved),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "sha256": previous_row["sha256"] if previous_matches else _sha256(resolved),
                }
            )
        except OSError as exc:
            warnings.append(f"artifact_manifest_error: {_relative(resolved, root)}: {exc}")
    return rows, warnings


def _atomic_charge_rows(calculation_id: str, path: Path, atoms: Atoms | None) -> list[dict[str, Any]]:
    if atoms is None:
        return []
    from cats.orca.run.orca_property import _parse_orca_output_charges

    rows: list[dict[str, Any]] = []
    symbols = atoms.get_chemical_symbols()
    for method in ("mulliken", "loewdin"):
        try:
            charges = _parse_orca_output_charges(path, method=method)
        except (OSError, ValueError):
            continue
        if len(charges) != len(symbols):
            continue
        rows.extend(
            {
                "calculation_id": calculation_id,
                "method": method,
                "atom_index_0based": index,
                "atom_index_1based": index + 1,
                "symbol": symbols[index],
                "charge_e": float(charge),
            }
            for index, charge in enumerate(charges)
        )
    return rows


def extract_embedded_orca_input(text: str) -> str:
    """Return the numbered input lines echoed in ORCA's INPUT FILE section."""
    in_input = False
    collected: list[str] = []
    for line in text.splitlines():
        if line.strip().upper() == "INPUT FILE":
            in_input = True
            continue
        if not in_input:
            continue
        match = _INPUT_LINE_RE.match(line)
        if match:
            collected.append(match.group(1).rstrip())
        elif collected and line.strip().startswith("==="):
            break
    return "\n".join(collected)


def qualify_orca_input(input_text: str) -> list[str]:
    """Return rejection reasons for the requested gas-phase method."""
    upper = input_text.upper()
    reasons: list[str] = []
    if not input_text.strip():
        return ["input_missing"]
    if not re.search(r"(?<![A-Z0-9])CAM-B3LYP(?![A-Z0-9])", upper):
        reasons.append("wrong_functional")
    if not re.search(r"(?<![A-Z0-9])6-31G\s*\(D\s*,\s*P\)(?![A-Z0-9])", upper):
        reasons.append("wrong_basis")
    if re.search(r"(?:%\s*)?CPCM(?:\s*\(|\b)", upper):
        reasons.append("cpcm_present")
    return reasons


def expand_fragment_selector(selector: str) -> list[int]:
    """Expand a Multiwfn one-based selector such as ``1-63,65``."""
    indices: list[int] = []
    for token in re.split(r"[\s,]+", selector.strip()):
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            if len(parts) != 2 or not all(part.strip().isdigit() for part in parts):
                raise ValueError(f"Invalid fragment range: {token!r}")
            start, end = (int(part) for part in parts)
            if start <= 0 or end < start:
                raise ValueError(f"Invalid fragment range: {token!r}")
            indices.extend(range(start, end + 1))
        elif token.isdigit() and int(token) > 0:
            indices.append(int(token))
        else:
            raise ValueError(f"Invalid fragment index: {token!r}")
    if not indices:
        raise ValueError("Fragment selector is empty.")
    return indices


def parse_multiwfn_config(path: Path, atom_count: int | None = None) -> dict[str, Any] | None:
    """Parse the Multiwfn ``18, 16`` interfragment CT configuration."""
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    clean = [line.split("//", 1)[0].strip() for line in lines]
    start = next(
        (index for index in range(len(clean) - 2) if clean[index] == "18" and clean[index + 1] == "16"),
        None,
    )
    if start is None:
        return None
    try:
        fragment_count = int(clean[start + 2])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid fragment count in {path}") from exc
    if fragment_count <= 0 or start + 3 + fragment_count > len(clean):
        raise ValueError(f"Invalid fragment block in {path}")

    selectors = clean[start + 3 : start + 3 + fragment_count]
    fragments_1based = [expand_fragment_selector(selector) for selector in selectors]
    validation_errors: list[str] = []
    flat = [index for fragment in fragments_1based for index in fragment]
    if len(flat) != len(set(flat)):
        validation_errors.append("overlapping_fragment_indices")
    if atom_count is not None:
        if any(index > atom_count for index in flat):
            validation_errors.append("fragment_index_out_of_range")
        missing = sorted(set(range(1, atom_count + 1)) - set(flat))
        if missing:
            validation_errors.append("unassigned_atoms")

    after = start + 3 + fragment_count
    return {
        "config_path": str(path),
        "molden_path": clean[start - 1] if start else "",
        "orca_output": clean[after] if after < len(clean) else "",
        "selectors": selectors,
        "fragments_1based": fragments_1based,
        "fragments_0based": [[index - 1 for index in fragment] for fragment in fragments_1based],
        "validation_errors": validation_errors,
    }


def _read_atoms_from_orca(path: Path, text: str) -> Atoms | None:
    try:
        return ase_read(str(path), format="orca-output")
    except Exception:
        pass

    last_block: list[tuple[str, list[float]]] = []
    lines = text.splitlines()
    for line_index, line in enumerate(lines):
        if "CARTESIAN COORDINATES (ANGSTROEM)" not in line.upper():
            continue
        block: list[tuple[str, list[float]]] = []
        for candidate in lines[line_index + 1 :]:
            match = _COORD_RE.match(candidate)
            if match:
                block.append(
                    (
                        match.group(1),
                        [float(match.group(i).replace("D", "E").replace("d", "e")) for i in range(2, 5)],
                    )
                )
            elif block:
                break
        if block:
            last_block = block
    if not last_block:
        return None
    return Atoms([row[0] for row in last_block], positions=[row[1] for row in last_block])


def _formula_for_indices(atoms: Atoms, indices: Iterable[int]) -> str:
    selected = list(indices)
    return atoms[selected].get_chemical_formula() if selected else ""


def _classify_formula(atoms: Atoms, config: dict[str, Any] | None) -> tuple[str, str]:
    fragments = [] if (config or {}).get("validation_errors") else (config or {}).get("fragments_0based", [])
    valid_fragments = [fragment for fragment in fragments if fragment and max(fragment) < len(atoms)]
    if valid_fragments:
        cluster = max(valid_fragments, key=lambda fragment: sum(atoms[index].symbol == "Au" for index in fragment))
        adsorbate = [index for fragment in valid_fragments if fragment is not cluster for index in fragment]
        au_count = sum(atoms[index].symbol == "Au" for index in cluster)
        return (f"Au{au_count}" if au_count else _formula_for_indices(atoms, cluster), _formula_for_indices(atoms, adsorbate))

    au_indices = [atom.index for atom in atoms if atom.symbol == "Au"]
    other_indices = [atom.index for atom in atoms if atom.symbol != "Au"]
    return (f"Au{len(au_indices)}" if au_indices else "no_Au", _formula_for_indices(atoms, other_indices))


def _find_fragment_config(directory: Path, orca_name: str, atom_count: int | None) -> dict[str, Any] | None:
    candidates = sorted(
        {path for pattern in ("config.txt", "*config*.txt") for path in directory.glob(pattern)}
    )
    fallback = None
    for candidate in candidates:
        try:
            parsed = parse_multiwfn_config(candidate, atom_count=atom_count)
        except (OSError, ValueError):
            continue
        if parsed is None:
            continue
        parsed["config_path"] = candidate.name
        referenced = Path(str(parsed.get("orca_output") or "")).name
        if referenced == orca_name:
            return parsed
        fallback = fallback or parsed
    return fallback


def _parse_ifct_files(
    calculation_id: str, directory: Path, root: Path, verbosity: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], list[str]]:
    from cats.multiwfn.tools import parse_ifct

    state_rows: list[dict[str, Any]] = []
    term_rows: list[dict[str, Any]] = []
    files: list[str] = []
    warnings: list[str] = []
    candidates = sorted(
        {
            *directory.rglob("IFCTdata.txt"),
            *directory.rglob("IFCTmajor.txt"),
        }
    )
    full_term_directories = {path.parent for path in candidates if path.name == "IFCTdata.txt"}
    for path in _progress(candidates, description="IFCT files", verbosity=verbosity, level=2):
        files.append(_relative(path, root))
        try:
            states, terms = parse_ifct(path)
        except Exception as exc:
            warnings.append(f"{_relative(path, root)}: {exc}")
            continue
        if path.name != "IFCTdata.txt":
            for row in states.to_dict("records"):
                state_rows.append({"calculation_id": calculation_id, "source": _relative(path, root), **_jsonable(row)})
        if path.name != "IFCTmajor.txt" or path.parent not in full_term_directories:
            for row in terms.to_dict("records"):
                term_rows.append({"calculation_id": calculation_id, "source": _relative(path, root), **_jsonable(row)})
    return state_rows, term_rows, files, warnings


def _parse_ct_multiple(
    calculation_id: str, directory: Path, root: Path, verbosity: int
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    import pandas as pd

    rows: list[dict[str, Any]] = []
    index_files: list[str] = []
    warnings: list[str] = []
    index_paths = sorted(directory.rglob("CT_multiple.txt"))
    for index_path in _progress(index_paths, description="CT_multiple files", verbosity=verbosity, level=2):
        index_files.append(_relative(index_path, root))
        for line in index_path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.strip().split(maxsplit=1)
            if not parts:
                continue
            rel_source = Path(parts[0].strip("\"'"))
            property_name = parts[1] if len(parts) == 2 else rel_source.stem
            source = next(
                (candidate for candidate in (directory / rel_source, index_path.parent / rel_source, index_path.parent / rel_source.name) if candidate.exists()),
                None,
            )
            if source is None:
                warnings.append(f"Missing CT_multiple member: {parts[0]}")
                continue
            try:
                table = pd.read_csv(source)
                numeric = [pd.to_numeric(table[column], errors="coerce") for column in table.columns]
                numeric = [series for series in numeric if series.notna().any()]
                if len(numeric) < 2:
                    table = pd.read_csv(source, sep=r"\s+", header=None)
                    numeric = [pd.to_numeric(table[column], errors="coerce") for column in table.columns]
                    numeric = [series for series in numeric if series.notna().any()]
                if len(numeric) < 2:
                    raise ValueError("fewer than two numeric columns")
                points = list(zip(numeric[0], numeric[1]))
                if points and float(points[0][0]).is_integer() and int(points[0][0]) == len(points) - 1:
                    points = points[1:]
                for energy, intensity in points:
                    if pd.isna(energy) or pd.isna(intensity):
                        continue
                    energy = float(energy)
                    rows.append(
                        {
                            "calculation_id": calculation_id,
                            "source": _relative(source, root),
                            "property": property_name,
                            "energy_eV": energy,
                            "wavelength_nm": 1239.841984 / energy if energy > 0 else None,
                            "intensity": float(intensity),
                        }
                    )
            except Exception as exc:
                warnings.append(f"{_relative(source, root)}: {exc}")
    return rows, index_files, warnings


def _constraint_data(atoms: Atoms) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    indices: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    for constraint in list(getattr(atoms, "constraints", []) or []):
        try:
            payload = constraint.todict()
        except Exception:
            payload = {"name": constraint.__class__.__name__}
        payloads.append(_jsonable(payload))
        getter = getattr(constraint, "get_indices", None)
        if callable(getter):
            try:
                indices.update(int(index) for index in getter())
            except Exception:
                pass
        constraint_pairs = getattr(constraint, "pairs", None)
        for pair in [] if constraint_pairs is None else constraint_pairs:
            if len(pair) == 2:
                pairs.add((int(pair[0]), int(pair[1])))
                indices.update((int(pair[0]), int(pair[1])))
    sorted_pairs = sorted(pairs)
    return {
        "constraints": payloads,
        "indices_0based": sorted(indices),
        "indices_1based": [index + 1 for index in sorted(indices)],
        "bond_pairs_0based": [list(pair) for pair in sorted_pairs],
        "bond_pairs_1based": [[a + 1, b + 1] for a, b in sorted_pairs],
        "bond_distances_A": [float(atoms.get_distance(*pair)) for pair in sorted_pairs],
    }


def _catalog_trajectories(
    root: Path,
    output: Path,
    calculation_dirs: dict[Path, str],
    verbosity: int,
    reused_calculation_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    trajectories = sorted(
        traj for traj in root.rglob("*.traj") if (traj.parent / "orca.out").is_file()
    )
    for traj in _progress(trajectories, description="ASE trajectories", verbosity=verbosity):
        try:
            traj.resolve().relative_to(output.resolve())
            continue
        except ValueError:
            pass
        scan_id = hashlib.sha256(_relative(traj, root).encode("utf-8")).hexdigest()[:16]
        related = [calc_id for directory, calc_id in calculation_dirs.items() if directory == traj.parent]
        if not related or all(calc_id in (reused_calculation_ids or set()) for calc_id in related):
            continue
        try:
            frames = ase_read(str(traj), index=":")
        except Exception as exc:
            audit.append({"path": _relative(traj, root), "status": "rejected", "reasons": [f"trajectory_parse_error: {exc}"]})
            _verbose(f"Rejected trajectory {_relative(traj, root)}: {exc}", verbosity=verbosity)
            continue
        for frame_index, atoms in enumerate(frames):
            constraint_data = _constraint_data(atoms)
            try:
                energy = float(atoms.get_potential_energy())
            except Exception:
                energy = None
            try:
                forces = atoms.get_forces(apply_constraint=False)
                fmax = max((float(sum(float(value) ** 2 for value in vector) ** 0.5) for vector in forces), default=None)
            except Exception:
                forces = None
                fmax = None
            rows.append(
                {
                    "scan_id": scan_id,
                    "trajectory": _relative(traj, root),
                    "related_calculation_ids": related,
                    "frame_index_0based": frame_index,
                    "frame_index_1based": frame_index + 1,
                    "formula": atoms.get_chemical_formula(),
                    "symbols": atoms.get_chemical_symbols(),
                    "positions_A": atoms.get_positions().tolist(),
                    "cell_A": atoms.cell.array.tolist(),
                    "pbc": atoms.get_pbc().tolist(),
                    "energy_eV": energy,
                    "forces_eV_A": None if forces is None else forces.tolist(),
                    "fmax_eV_A": fmax,
                    "info": _jsonable(atoms.info),
                    **constraint_data,
                }
            )
        _verbose(f"Parsed trajectory {_relative(traj, root)} ({len(frames)} frames)", verbosity=verbosity)
    return rows, audit


def catalog_legacy_orca(
    root: Path | str,
    output: Path | str,
    *,
    verbosity: int = -1,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    output = Path(output).resolve()
    existing = _load_existing_catalog(output) if reuse_existing else None
    output.mkdir(parents=True, exist_ok=True)
    plots_dir = output / "plots"
    plots_dir.mkdir(exist_ok=True)
    structures_dir = output / "structures"
    structures_dir.mkdir(exist_ok=True)

    calculations: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    ct_states: list[dict[str, Any]] = []
    ct_terms: list[dict[str, Any]] = []
    ct_spectra: list[dict[str, Any]] = []
    atomic_charges: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    fragments: list[dict[str, Any]] = []
    calculation_dirs: dict[Path, str] = {}
    reused_calculation_ids: set[str] = set()
    reusable_targets = {
        "transitions.csv": transitions,
        "spectra.csv": spectra,
        "atomic_charges.csv": atomic_charges,
        "ct_states.csv": ct_states,
        "ct_terms.csv": ct_terms,
        "ct_spectra.csv": ct_spectra,
    }

    orca_outputs = sorted(root.rglob("orca.out"))
    calculation_directories = {path.parent.resolve() for path in orca_outputs}
    for orca_out in _progress(orca_outputs, description="ORCA outputs", verbosity=verbosity):
        try:
            orca_out.resolve().relative_to(output)
            continue
        except ValueError:
            pass
        rel_path = _relative(orca_out, root)
        calculation_id = hashlib.sha256(rel_path.encode("utf-8")).hexdigest()[:16]
        try:
            text = orca_out.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            audit.append({"calculation_id": calculation_id, "path": rel_path, "status": "rejected", "reasons": [f"read_error: {exc}"]})
            continue

        embedded_input = extract_embedded_orca_input(text)
        input_source = "embedded"
        input_file = ""
        if not embedded_input and (orca_out.parent / "orca.inp").exists():
            embedded_input = (orca_out.parent / "orca.inp").read_text(encoding="utf-8", errors="replace")
            input_source = "orca.inp"
            input_file = _relative(orca_out.parent / "orca.inp", root)
        input_provenance = _input_provenance(embedded_input, text)
        reasons = qualify_orca_input(embedded_input)
        audit.append(
            {
                "calculation_id": calculation_id,
                "path": rel_path,
                "status": "accepted" if not reasons else "rejected",
                "reasons": reasons,
            }
        )
        if reasons:
            _verbose(f"Rejected {rel_path}: {', '.join(reasons)}", verbosity=verbosity)
            continue

        manifest_rows, manifest_warnings = _artifact_manifest(
            calculation_id,
            orca_out.parent,
            root,
            output,
            calculation_directories,
            (existing or {}).get("artifacts", {}).get(calculation_id, []),
        )
        calculation_fingerprint, parse_fingerprint = _calculation_fingerprints(
            root.name,
            rel_path,
            manifest_rows,
        )
        artifacts.extend(manifest_rows)
        calculation_dirs[orca_out.parent.resolve()] = calculation_id
        cached_calculation = (existing or {}).get("calculations", {}).get(rel_path)
        if not manifest_warnings and existing and _can_reuse_calculation(
            cached_calculation,
            parse_fingerprint,
            output,
            existing["tables"],
            existing["fragments"],
        ):
            calculations.append(cached_calculation)
            for table_name, target in reusable_targets.items():
                target.extend(existing["tables"][table_name].get(calculation_id, []))
            if calculation_id in existing["fragments"]:
                fragments.append(existing["fragments"][calculation_id])
            reused_calculation_ids.add(calculation_id)
            _verbose(f"Reused {rel_path}", verbosity=verbosity)
            continue

        atoms = _read_atoms_from_orca(orca_out, text)
        warnings: list[str] = list(manifest_warnings)
        if atoms is None:
            warnings.append("geometry_missing")
        atom_count = len(atoms) if atoms is not None else None
        config = _find_fragment_config(orca_out.parent, orca_out.name, atom_count)
        if config:
            config = _jsonable(config)
            fragments.append({"calculation_id": calculation_id, **config})
            warnings.extend(config.get("validation_errors", []))
        cluster_family, adsorbate_formula = (
            _classify_formula(atoms, config) if atoms is not None else ("unknown", "")
        )
        structure_file = ""
        if atoms is not None:
            structure_path = structures_dir / f"{calculation_id}.extxyz"
            structure = atoms.copy()
            structure.info.update(
                {
                    "calculation_id": calculation_id,
                    "source_orca_out": rel_path,
                    "cluster_family": cluster_family,
                    "adsorbate_formula": adsorbate_formula,
                }
            )
            try:
                ase_write(structure_path, structure, format="extxyz")
                structure_file = _relative(structure_path, output)
                _verbose(f"Wrote structure {structure_file}", verbosity=verbosity, level=2)
            except Exception as exc:
                warnings.append(f"structure_write_error: {exc}")

        energy_matches = _ENERGY_RE.findall(text)
        energy_hartree = (
            float(energy_matches[-1].replace("D", "E").replace("d", "e")) if energy_matches else None
        )
        tddft_status = "not_present"
        transition_count = 0
        if _TDDFT_MARKER in text:
            try:
                import matplotlib

                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as plt

                from cats.orca.routines.orca_uv_extractor import extract_uv_transitions
                from cats.visualise.uvvis import plot_uv_spectrum_nm

                transition_data = extract_uv_transitions(str(orca_out))
                wavelengths = (
                    transition_data.wavelengths_nm
                    if transition_data.wavelengths_nm is not None
                    else 1.0e7 / transition_data.wavenumbers_cm
                )
                for index, (wavenumber, wavelength, intensity) in enumerate(
                    zip(transition_data.wavenumbers_cm, wavelengths, transition_data.intensities), start=1
                ):
                    transitions.append(
                        {
                            "calculation_id": calculation_id,
                            "state": index,
                            "wavenumber_cm": float(wavenumber),
                            "wavelength_nm": float(wavelength),
                            "oscillator_strength": float(intensity),
                        }
                    )
                fig, ax = plt.subplots()
                _, spectrum = plot_uv_spectrum_nm(
                    transition_data, linewidth_nm=20.0, start_nm=200.0, end_nm=1000.0, step=1.0, ax=ax
                )
                ax.set_title(f"{atoms.get_chemical_formula() if atoms is not None else calculation_id} UV/Vis")
                fig.savefig(plots_dir / f"{calculation_id}_uvvis.png", dpi=200, bbox_inches="tight")
                plt.close(fig)
                spectra.extend(
                    {
                        "calculation_id": calculation_id,
                        "wavelength_nm": float(x_value),
                        "intensity": float(y_value),
                    }
                    for x_value, y_value in zip(spectrum.x, spectrum.y)
                )
                transition_count = len(transition_data.wavenumbers_cm)
                tddft_status = "parsed"
            except Exception as exc:
                tddft_status = "parse_error"
                warnings.append(f"tddft_parse_error: {exc}")

        parsed_states, parsed_terms, ifct_files, ifct_warnings = _parse_ifct_files(
            calculation_id, orca_out.parent, root, verbosity
        )
        parsed_ct_spectra, ct_multiple_files, ct_multiple_warnings = _parse_ct_multiple(
            calculation_id, orca_out.parent, root, verbosity
        )
        parsed_charges = _atomic_charge_rows(calculation_id, orca_out, atoms)
        ct_states.extend(parsed_states)
        ct_terms.extend(parsed_terms)
        ct_spectra.extend(parsed_ct_spectra)
        atomic_charges.extend(parsed_charges)
        warnings.extend(ifct_warnings + ct_multiple_warnings)
        molden_files = sorted(
            _relative(path, root)
            for path in orca_out.parent.iterdir()
            if path.is_file() and "molden" in path.name.lower()
        )
        if parsed_states or parsed_terms or parsed_ct_spectra:
            ct_status = "parsed"
        elif ifct_files or ct_multiple_files:
            ct_status = "parse_error"
        elif config or molden_files:
            ct_status = "inputs_present_output_missing"
        else:
            ct_status = "not_present"
        calculations.append(
            {
                "calculation_id": calculation_id,
                "calculation_fingerprint": calculation_fingerprint,
                "parse_fingerprint": parse_fingerprint,
                "orca_out": rel_path,
                "input_source": input_source,
                "input_file": input_file,
                "input_text": embedded_input,
                **input_provenance,
                "structure_file": structure_file,
                "formula": atoms.get_chemical_formula() if atoms is not None else "",
                "cluster_family": cluster_family,
                "adsorbate_formula": adsorbate_formula,
                "atom_count": atom_count,
                "energy_hartree": energy_hartree,
                "normal_termination": "ORCA TERMINATED NORMALLY" in text.upper(),
                "tddft_status": tddft_status,
                "transition_count": transition_count,
                "ct_status": ct_status,
                "fragment_config": config.get("config_path", "") if config else "",
                "molden_files": molden_files,
                "ifct_files": ifct_files,
                "ct_multiple_files": ct_multiple_files,
                "artifact_count": len(manifest_rows),
                "warnings": warnings,
            }
        )
        _verbose(
            f"Accepted {rel_path}: {calculations[-1]['formula'] or 'unknown formula'}, "
            f"TDDFT={tddft_status}, CT={ct_status}",
            verbosity=verbosity,
        )
        _verbose(
            f"  id={calculation_id}, fragments={len((config or {}).get('fragments_0based', []))}, "
            f"warnings={len(warnings)}",
            verbosity=verbosity,
            level=2,
        )

    scan_frames, trajectory_audit = _catalog_trajectories(
        root,
        output,
        calculation_dirs,
        verbosity,
        reused_calculation_ids,
    )
    if existing:
        for calculation_id in sorted(reused_calculation_ids):
            scan_frames.extend(existing["tables"]["scan_frames.csv"].get(calculation_id, []))
    audit.extend(trajectory_audit)

    _write_csv(output / "calculations.csv", calculations, CALCULATION_FIELDS)
    _write_csv(output / "audit.csv", audit, ["calculation_id", "path", "status", "reasons"])
    _write_csv(
        output / "transitions.csv",
        transitions,
        ["calculation_id", "state", "wavenumber_cm", "wavelength_nm", "oscillator_strength"],
    )
    _write_csv(output / "spectra.csv", spectra, ["calculation_id", "wavelength_nm", "intensity"])
    _write_csv(output / "atomic_charges.csv", atomic_charges, ATOMIC_CHARGE_FIELDS)
    _write_csv(output / "artifacts.csv", artifacts, ARTIFACT_FIELDS)
    _write_csv(
        output / "ct_states.csv",
        ct_states,
        ["calculation_id", "source", "state", "f", "nm", "CT(e)", "CT(%)"],
    )
    _write_csv(output / "ct_terms.csv", ct_terms, ["calculation_id", "source", "state", "term", "pct"])
    _write_csv(
        output / "ct_spectra.csv",
        ct_spectra,
        ["calculation_id", "source", "property", "energy_eV", "wavelength_nm", "intensity"],
    )
    _write_csv(
        output / "scan_frames.csv",
        scan_frames,
        [
            "scan_id",
            "trajectory",
            "related_calculation_ids",
            "frame_index_0based",
            "frame_index_1based",
            "formula",
            "symbols",
            "positions_A",
            "cell_A",
            "pbc",
            "energy_eV",
            "forces_eV_A",
            "fmax_eV_A",
            "info",
            "constraints",
            "indices_0based",
            "indices_1based",
            "bond_pairs_0based",
            "bond_pairs_1based",
            "bond_distances_A",
        ],
    )
    with (output / "fragments.jsonl").open("w", encoding="utf-8") as handle:
        for fragment in fragments:
            handle.write(json.dumps(_jsonable(fragment), sort_keys=True) + "\n")

    raw_output_hashes = {
        row["calculation_id"]: row["sha256"]
        for row in artifacts
        if Path(str(row.get("path") or "")).name.lower() == "orca.out"
    }
    identity_payload = {
        "root_name": root.name,
        "calculations": [
            [row["calculation_id"], raw_output_hashes.get(row["calculation_id"], "")]
            for row in sorted(calculations, key=lambda item: item["calculation_id"])
        ],
    }
    source_tree_id = hashlib.sha256(
        json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "source_tree_id": source_tree_id,
        "root": str(root),
        "output": str(output),
        "orca_outputs": sum(1 for row in audit if "calculation_id" in row),
        "accepted": len(calculations),
        "rejected": sum(row.get("status") == "rejected" and "calculation_id" in row for row in audit),
        "tddft_parsed": sum(row["tddft_status"] == "parsed" for row in calculations),
        "ct_parsed": sum(row["ct_status"] == "parsed" for row in calculations),
        "atomic_charge_rows": len(atomic_charges),
        "artifacts": len(artifacts),
        "reused_calculations": len(reused_calculation_ids),
        "trajectory_frames": len(scan_frames),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Catalogue legacy gas-phase CAM-B3LYP ORCA calculations.")
    parser.add_argument("--root", type=Path, required=True, help="Directory recursively containing orca.out files.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for CSV/JSONL catalogues and plots.")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-q", "--quiet", action="store_true", help="Disable progress and summary output.")
    output_group.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Show per-file details; repeat as -vv for companion-file progress and diagnostics.",
    )
    parser.add_argument(
        "--no-reuse",
        action="store_true",
        help="Ignore an existing schema-compatible catalog and reparse every calculation.",
    )
    args = parser.parse_args(argv)
    verbosity = -1 if args.quiet else int(args.verbose)
    summary = catalog_legacy_orca(
        args.root,
        args.output,
        verbosity=verbosity,
        reuse_existing=not args.no_reuse,
    )
    if not args.quiet:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
