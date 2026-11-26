from __future__ import annotations

import csv
import os
from pathlib import Path
import re
from typing import Optional

from cats.multiwfn.iroot_scoring import IRootCandidate, select_iroot_by_score
from cats.multiwfn.plot_uvvis import get_full_ct_spectrum
from cats.multiwfn.tools import parse_ifct

from .io import validate_orca_output

SCAN_LOG_FIELDNAMES = [
    "iteration",
    "scan_dir",
    "bond_i",
    "bond_j",
    "bond_length",
    "iroot",
    "wavelength_nm",
    "intensity",
    "energy_ev",
    "ct_e",
    "ct_pct",
    "ct_magnitude",
    "score",
    "similarity",
    "spatial_overlap",
    "score_intensity",
    "score_ct",
    "score_wavelength",
    "score_similarity",
    "target_wavelength_nm",
    "ct_net",
    "point_kind",
    "source_iteration",
    "reference_iteration",
    "point_label",
]
_STATE_HEADER_RE = re.compile(r"^\s*STATE\s+(\d+)\s*:", re.IGNORECASE)
_TRANSITION_TERM_RE = re.compile(
    r"(\d+\s*[A-Za-z]?)\s*(->|<-)\s*(\d+\s*[A-Za-z]?)\s*[:=]?\s*"
    r"([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?)"
    r"(?:\s*\(\s*c\s*=\s*([+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?)\s*\))?"
)


def extract_state_transition_vectors(
    orca_out: Path,
    *,
    state_filter: set[int] | list[int] | tuple[int, ...] | None = None,
) -> dict[int, dict[str, float]]:
    vectors: dict[int, dict[str, float]] = {}
    current_state: int | None = None
    filter_set = {int(s) for s in state_filter} if state_filter is not None else None
    try:
        with open(orca_out, "r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                line = raw_line.rstrip("\n")
                state_match = _STATE_HEADER_RE.match(line)
                if state_match is not None:
                    current_state = int(state_match.group(1))
                    if filter_set is None or current_state in filter_set:
                        vectors.setdefault(current_state, {})
                    continue
                if current_state is None:
                    continue
                if filter_set is not None and current_state not in filter_set:
                    continue
                if "->" not in line and "<-" not in line:
                    if (
                        "TD-DFT/TDA-EXCITATION SPECTRA" in line
                        or "ABSORPTION SPECTRUM" in line
                    ):
                        break
                    continue
                for term_match in _TRANSITION_TERM_RE.finditer(line):
                    from_orb = re.sub(r"\s+", "", term_match.group(1)).lower()
                    arrow = term_match.group(2)
                    to_orb = re.sub(r"\s+", "", term_match.group(3)).lower()
                    raw_coeff = (
                        term_match.group(5)
                        if term_match.group(5) is not None
                        else term_match.group(4)
                    )
                    coeff = float(raw_coeff)
                    label = f"{from_orb}{arrow}{to_orb}"
                    state_vector = vectors.setdefault(current_state, {})
                    state_vector[label] = state_vector.get(label, 0.0) + coeff
    except OSError:
        return {}
    return {state: vec for state, vec in vectors.items() if vec}


def get_ct_info(
    scan_dir: Path,
    fragments: dict[str, str],
    *,
    min_oscillator_strength: float = 0.005,
):
    parent_dir = Path.cwd()
    os.chdir(scan_dir)
    os.environ["OMP_STACKSIZE"] = "1500M"
    try:
        if not Path("orca.out").exists():
            raise FileNotFoundError(f"Missing orca.out in {scan_dir}")
        orcadir = os.environ.get("ORCADIR")
        if not orcadir:
            conv = shutil.which("orca_2mkl")
            if conv:
                orcadir = str(Path(conv).parent)
        if not orcadir:
            raise RuntimeError("ORCADIR is not set and orca_2mkl not found on PATH; cannot run orca_2mkl.")
        exit_code = os.system(f"{orcadir}/orca_2mkl orca -molden")
        if exit_code != 0:
            raise RuntimeError(f"orca_2mkl failed with exit code {exit_code}.")
        if not Path("orca.molden.input").exists():
            raise FileNotFoundError(f"Missing orca.molden.input in {scan_dir}")
        get_full_ct_spectrum("orca.out", "orca.molden.input", fragments=fragments)
        if not Path("IFCTmajor.txt").exists():
            raise FileNotFoundError(f"Missing IFCTmajor.txt in {scan_dir}")
        df = parse_ifct("IFCTmajor.txt", wide=True)
    finally:
        if "OMP_STACKSIZE" in os.environ:
            del os.environ["OMP_STACKSIZE"]
        os.chdir(parent_dir)

    threshold = float(min_oscillator_strength)
    df = df[df["f"] > threshold]
    return df


def pick_iroot(
    df,
    *,
    target_wavelength_nm: float,
    previous_choice: Optional[IRootCandidate] = None,
    previous_state: Optional[int] = None,
    score_weights: dict[str, float],
    wavelength_sigma_nm: float,
    similarity_sigma_nm: float,
    similarity_sigma_ct: float,
    min_similarity: float = 0.0,
    min_relative_intensity: float = 0.0,
    min_transition_overlap: float = 0.0,
    min_spatial_overlap: float = 0.0,
    min_ct_similarity: float = 0.0,
    intensity_reference_window_nm: float | None = 200.0,
    require_transition_overlap: bool = False,
    require_spatial_overlap: bool = False,
    state_vectors: Optional[dict[int, dict[str, float]]] = None,
    state_spatial_overlaps: Optional[dict[int, float]] = None,
):
    best, scored = select_iroot_by_score(
        df,
        target_wavelength_nm=target_wavelength_nm,
        previous_choice=previous_choice,
        previous_state=previous_state,
        weights=score_weights,
        wavelength_sigma_nm=wavelength_sigma_nm,
        similarity_sigma_nm=similarity_sigma_nm,
        similarity_sigma_ct=similarity_sigma_ct,
        min_similarity=min_similarity,
        min_relative_intensity=min_relative_intensity,
        min_transition_overlap=min_transition_overlap,
        min_spatial_overlap=min_spatial_overlap,
        min_ct_similarity=min_ct_similarity,
        intensity_reference_window_nm=intensity_reference_window_nm,
        require_transition_overlap=require_transition_overlap,
        require_spatial_overlap=require_spatial_overlap,
        state_vectors=state_vectors,
        state_spatial_overlaps=state_spatial_overlaps,
    )
    return best, scored


def _format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return ""
    return f"{value:.8f}"


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _parse_optional_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_scan_log(path: Path) -> dict[str, dict[str, str]]:
    existing: dict[str, dict[str, str]] = {}
    if not path.exists():
        return existing
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iteration = row.get("iteration")
                if iteration is not None and iteration != "":
                    existing[iteration] = row
    except Exception as exc:
        print(f"Warning: Failed to read scan log {path}: {exc}")
    return existing


def _iter_sort_key(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return 10**9


def _write_scan_log_rows(log_path: Path, existing: dict[str, dict[str, str]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCAN_LOG_FIELDNAMES)
        writer.writeheader()
        for key in sorted(existing, key=_iter_sort_key):
            row = existing[key]
            normalized = {}
            for name in SCAN_LOG_FIELDNAMES:
                value = row.get(name, "")
                normalized[name] = "" if value is None else str(value)
            writer.writerow(normalized)


def record_scan_log_row(
    log_path: Path,
    *,
    iteration: int,
    scan_dir: Path,
    bond_pair: tuple[int, int] | None = None,
    bond_length: Optional[float] = None,
    iroot: int | None = None,
    wavelength_nm: Optional[float] = None,
    intensity: Optional[float] = None,
    energy_ev: Optional[float] = None,
    ct_e: Optional[float] = None,
    ct_pct: Optional[float] = None,
    ct_net: Optional[float] = None,
    ct_magnitude: Optional[float] = None,
    score: Optional[float] = None,
    similarity: Optional[float] = None,
    spatial_overlap: Optional[float] = None,
    score_components: Optional[dict[str, float]] = None,
    target_wavelength_nm: Optional[float] = None,
    point_kind: str = "",
    source_iteration: int | None = None,
    reference_iteration: int | None = None,
    point_label: str = "",
) -> None:
    components = score_components or {}
    row = {
        "iteration": str(iteration),
        "scan_dir": str(scan_dir),
        "bond_i": str(bond_pair[0]) if bond_pair is not None else "",
        "bond_j": str(bond_pair[1]) if bond_pair is not None else "",
        "bond_length": _format_optional_float(bond_length),
        "iroot": "" if iroot is None else str(int(iroot)),
        "wavelength_nm": _format_optional_float(wavelength_nm),
        "intensity": _format_optional_float(intensity),
        "energy_ev": _format_optional_float(energy_ev),
        "ct_e": _format_optional_float(ct_e),
        "ct_pct": _format_optional_float(ct_pct),
        "ct_net": _format_optional_float(ct_net),
        "ct_magnitude": _format_optional_float(ct_magnitude),
        "score": _format_optional_float(score),
        "similarity": _format_optional_float(similarity),
        "spatial_overlap": _format_optional_float(spatial_overlap),
        "score_intensity": _format_optional_float(components.get("intensity")),
        "score_ct": _format_optional_float(components.get("ct")),
        "score_wavelength": _format_optional_float(components.get("wavelength")),
        "score_similarity": _format_optional_float(components.get("similarity")),
        "target_wavelength_nm": _format_optional_float(target_wavelength_nm),
        "point_kind": str(point_kind or ""),
        "source_iteration": "" if source_iteration is None else str(int(source_iteration)),
        "reference_iteration": "" if reference_iteration is None else str(int(reference_iteration)),
        "point_label": str(point_label or ""),
    }

    existing = load_scan_log(log_path)
    existing[str(iteration)] = row
    try:
        _write_scan_log_rows(log_path, existing)
    except Exception as exc:
        print(f"Warning: Failed to write scan log {log_path}: {exc}")


def record_scan_choice(
    log_path: Path,
    *,
    iteration: int,
    scan_dir: Path,
    bond_pair: tuple[int, int] | None,
    bond_length: Optional[float],
    choice: IRootCandidate,
    target_wavelength_nm: float,
    point_kind: str = "",
    point_label: str = "",
):
    record_scan_log_row(
        log_path,
        iteration=iteration,
        scan_dir=scan_dir,
        bond_pair=bond_pair,
        bond_length=bond_length,
        iroot=choice.state,
        wavelength_nm=choice.wavelength_nm,
        intensity=choice.intensity,
        energy_ev=choice.energy_ev,
        ct_e=choice.ct_e,
        ct_pct=choice.ct_pct,
        ct_net=choice.ct_net,
        ct_magnitude=choice.ct_magnitude,
        score=choice.score,
        similarity=choice.similarity,
        spatial_overlap=choice.spatial_overlap,
        score_components=choice.score_components or {},
        target_wavelength_nm=target_wavelength_nm,
        point_kind=point_kind,
        point_label=point_label,
    )


def record_geometry_choice(
    log_path: Path,
    *,
    iteration: int,
    geometry_step: int,
    scan_dir: Path,
    bond_pair: tuple[int, int] | None,
    bond_length: Optional[float],
    choice: IRootCandidate,
    target_wavelength_nm: float,
):
    fieldnames = [
        "iteration",
        "geometry_step",
        "scan_dir",
        "bond_i",
        "bond_j",
        "bond_length",
        "iroot",
        "wavelength_nm",
        "intensity",
        "energy_ev",
        "ct_e",
        "ct_pct",
        "ct_magnitude",
        "score",
        "similarity",
        "spatial_overlap",
        "score_intensity",
        "score_ct",
        "score_wavelength",
        "score_similarity",
        "target_wavelength_nm",
        "ct_net",
    ]

    components = choice.score_components or {}
    row = {
        "iteration": str(iteration),
        "geometry_step": str(geometry_step),
        "scan_dir": str(scan_dir),
        "bond_i": str(bond_pair[0]) if bond_pair is not None else "",
        "bond_j": str(bond_pair[1]) if bond_pair is not None else "",
        "bond_length": _format_optional_float(bond_length),
        "iroot": str(choice.state),
        "wavelength_nm": _format_optional_float(choice.wavelength_nm),
        "intensity": _format_optional_float(choice.intensity),
        "energy_ev": _format_optional_float(choice.energy_ev),
        "ct_e": _format_optional_float(choice.ct_e),
        "ct_pct": _format_optional_float(choice.ct_pct),
        "ct_net": _format_optional_float(choice.ct_net),
        "ct_magnitude": _format_optional_float(choice.ct_magnitude),
        "score": _format_optional_float(choice.score),
        "similarity": _format_optional_float(choice.similarity),
        "spatial_overlap": _format_optional_float(choice.spatial_overlap),
        "score_intensity": _format_optional_float(components.get("intensity")),
        "score_ct": _format_optional_float(components.get("ct")),
        "score_wavelength": _format_optional_float(components.get("wavelength")),
        "score_similarity": _format_optional_float(components.get("similarity")),
        "target_wavelength_nm": _format_optional_float(target_wavelength_nm),
    }

    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
    except Exception as exc:
        print(f"Warning: Failed to append geometry log {log_path}: {exc}")


def load_last_geometry_state(
    log_path: Path, iteration: int
) -> tuple[int, Optional[IRootCandidate], Optional[Path]]:
    if not log_path.exists():
        return 0, None, None
    last_step = 0
    last_row: dict[str, str] | None = None
    try:
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if _parse_optional_int(row.get("iteration")) != iteration:
                    continue
                step = _parse_optional_int(row.get("geometry_step"))
                if step is None:
                    continue
                if step >= last_step:
                    last_step = step
                    last_row = row
    except Exception as exc:
        print(f"Warning: Failed to read geometry log {log_path}: {exc}")
        return 0, None, None

    if last_row is None or last_step == 0:
        return 0, None, None

    iroot = _parse_optional_int(last_row.get("iroot"))
    wavelength = _parse_optional_float(last_row.get("wavelength_nm"))
    intensity = _parse_optional_float(last_row.get("intensity"))
    energy_ev = _parse_optional_float(last_row.get("energy_ev"))
    if iroot is None or wavelength is None or intensity is None:
        choice = None
    else:
        score_components = {}
        for key in ("score_intensity", "score_ct", "score_wavelength", "score_similarity"):
            value = _parse_optional_float(last_row.get(key))
            if value is not None:
                score_components[key.split("_", 1)[1]] = value
        choice = IRootCandidate(
            state=iroot,
            wavelength_nm=wavelength,
            intensity=intensity,
            energy_ev=energy_ev,
            ct_e=_parse_optional_float(last_row.get("ct_e")),
            ct_pct=_parse_optional_float(last_row.get("ct_pct")),
            ct_net=_parse_optional_float(last_row.get("ct_net")),
            score=_parse_optional_float(last_row.get("score")),
            similarity=_parse_optional_float(last_row.get("similarity")),
            spatial_overlap=_parse_optional_float(last_row.get("spatial_overlap")),
            score_components=score_components if score_components else None,
        )

    scan_dir_text = last_row.get("scan_dir") if last_row else None
    scan_dir = Path(scan_dir_text) if scan_dir_text else None
    if scan_dir is not None and not scan_dir.exists():
        scan_dir = None
    return last_step, choice, scan_dir


def recover_scan_choice(
    scan_dir: Path,
    *,
    target_wavelength_nm: float,
    fragments: dict[str, str],
    score_weights: dict[str, float],
    wavelength_sigma_nm: float,
    similarity_sigma_nm: float,
    similarity_sigma_ct: float,
    min_oscillator_strength: float = 0.005,
) -> Optional[IRootCandidate]:
    orca_out = scan_dir / "orca.out"
    if not validate_orca_output(orca_out):
        return None
    try:
        df = get_ct_info(
            scan_dir,
            fragments,
            min_oscillator_strength=min_oscillator_strength,
        )
    except (FileNotFoundError, RuntimeError):
        return None
    best, _ = pick_iroot(
        df,
        target_wavelength_nm=target_wavelength_nm,
        score_weights=score_weights,
        wavelength_sigma_nm=wavelength_sigma_nm,
        similarity_sigma_nm=similarity_sigma_nm,
        similarity_sigma_ct=similarity_sigma_ct,
        state_vectors=extract_state_transition_vectors(scan_dir / "orca.out"),
    )
    return best


def summarize_geometry_log(log_path: Path, summary_path: Optional[Path] = None) -> None:
    if not log_path.exists():
        return
    summary_path = summary_path or (log_path.parent / "iroot_geometry_summary.csv")
    groups: dict[int, list[dict[str, str]]] = {}
    try:
        with open(log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                iteration = _parse_optional_int(row.get("iteration"))
                if iteration is None:
                    continue
                groups.setdefault(iteration, []).append(row)
    except Exception as exc:
        print(f"Warning: Failed to read geometry log {log_path}: {exc}")
        return

    fieldnames = [
        "iteration",
        "geometry_steps",
        "iroot_sequence",
        "iroot_unique",
        "iroot_final",
        "bond_length_final",
        "scan_dir_final",
        "wavelength_nm_final",
        "ct_net_final",
        "intensity_final",
        "score_final",
    ]

    rows: list[dict[str, str]] = []
    for iteration in sorted(groups):
        entries = groups[iteration]
        entries.sort(key=lambda r: _parse_optional_int(r.get("geometry_step")) or 0)
        iroots = [str(r.get("iroot") or "").strip() for r in entries]
        iroots_clean = [i for i in iroots if i]
        unique = sorted({i for i in iroots_clean})
        final = entries[-1]
        rows.append(
            {
                "iteration": str(iteration),
                "geometry_steps": str(len(entries)),
                "iroot_sequence": ",".join(iroots_clean),
                "iroot_unique": ",".join(unique),
                "iroot_final": str(final.get("iroot") or ""),
                "bond_length_final": str(final.get("bond_length") or ""),
                "scan_dir_final": str(final.get("scan_dir") or ""),
                "wavelength_nm_final": str(final.get("wavelength_nm") or ""),
                "ct_net_final": str(final.get("ct_net") or ""),
                "intensity_final": str(final.get("intensity") or ""),
                "score_final": str(final.get("score") or ""),
            }
        )

    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        print(f"Warning: Failed to write geometry summary {summary_path}: {exc}")


def summarize_scan_log(log_path: Path, summary_path: Optional[Path] = None) -> None:
    existing = load_scan_log(log_path)
    if not existing:
        return
    summary_path = summary_path or (log_path.parent / "iroot_scan_summary.csv")
    fieldnames = [
        "iteration",
        "bond_length",
        "iroot",
        "wavelength_nm",
        "ct_net",
        "intensity",
        "score",
        "scan_dir",
    ]

    def _iter_sort_key(value: str) -> int:
        try:
            return int(value)
        except ValueError:
            return 10**9

    rows: list[dict[str, str]] = []
    for key in sorted(existing, key=_iter_sort_key):
        row = existing[key]
        rows.append(
            {
                "iteration": str(row.get("iteration") or key),
                "bond_length": str(row.get("bond_length") or ""),
                "iroot": str(row.get("iroot") or ""),
                "wavelength_nm": str(row.get("wavelength_nm") or ""),
                "ct_net": str(row.get("ct_net") or ""),
                "intensity": str(row.get("intensity") or ""),
                "score": str(row.get("score") or ""),
                "scan_dir": str(row.get("scan_dir") or ""),
            }
        )

    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        print(f"Warning: Failed to write scan summary {summary_path}: {exc}")
