from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path


def _default_output_path_for_dir(path: Path) -> Path:
    return path / "orca.out"


def scan_dir_mtime(
    path: Path,
    *,
    output_path_for_dir: Callable[[Path], Path] = _default_output_path_for_dir,
) -> float:
    try:
        marker = output_path_for_dir(path)
        if marker.exists():
            return float(marker.stat().st_mtime)
        return float(path.stat().st_mtime)
    except OSError:
        return 0.0


def scan_dir_has_valid_output(
    scan_dir: Path,
    *,
    validate_orca_output: Callable[[Path], bool],
    output_path_for_dir: Callable[[Path], Path] = _default_output_path_for_dir,
) -> bool:
    return validate_orca_output(output_path_for_dir(scan_dir))


def resolve_scan_dir_from_hint(
    scan_dir_text: str,
    *,
    scan_dir_has_valid_output: Callable[[Path], bool],
) -> Path | None:
    text = str(scan_dir_text or "").strip()
    if not text:
        return None
    direct = Path(text)
    if scan_dir_has_valid_output(direct):
        return direct

    parts = [part for part in direct.parts if part not in {"", "."}]
    for anchor in ("TDDFT", "DFT", "TDDFT_active_learning_shared"):
        for idx, part in enumerate(parts):
            if part != anchor:
                continue
            candidate = Path(*parts[idx:])
            if scan_dir_has_valid_output(candidate):
                return candidate

    name = direct.name
    if name.startswith("TDDFT_scan_iter_"):
        candidate = Path("TDDFT") / name.removeprefix("TDDFT_")
        if scan_dir_has_valid_output(candidate):
            return candidate
    if name.startswith("DFT_scan_iter_"):
        candidate = Path("DFT") / name.removeprefix("DFT_")
        if scan_dir_has_valid_output(candidate):
            return candidate
    return None


def resolve_scan_dir_for_iteration_for_log_repair(
    iteration: int,
    *,
    use_tddft_optimisation: bool,
    resolve_scan_dir: Callable[[int], Path],
    tddft_scan_geom_dirs: Callable[[int], list[Path]],
    resolve_opt_dir_for_mode: Callable[[int], Path],
    scan_dir_mtime: Callable[[Path], float] = scan_dir_mtime,
    output_path_for_dir: Callable[[Path], Path] = _default_output_path_for_dir,
) -> Path | None:
    candidates: list[Path] = []
    base_scan = resolve_scan_dir(iteration)
    if output_path_for_dir(base_scan).exists():
        candidates.append(base_scan)
    for geom_dir in tddft_scan_geom_dirs(iteration):
        if output_path_for_dir(geom_dir).exists():
            candidates.append(geom_dir)

    opt_dir = resolve_opt_dir_for_mode(
        iteration,
        use_tddft_optimisation=use_tddft_optimisation,
    )
    if output_path_for_dir(opt_dir).exists():
        candidates.append(opt_dir)

    if not candidates:
        return None
    return max(candidates, key=lambda item: (scan_dir_mtime(item), str(item)))


def score_components_from_scan_log_row(
    row: dict[str, str],
    *,
    parse_optional_float: Callable[[object], float | None],
) -> dict[str, float] | None:
    components: dict[str, float] = {}
    for key, label in (
        ("score_intensity", "intensity"),
        ("score_ct", "ct"),
        ("score_wavelength", "wavelength"),
        ("score_similarity", "similarity"),
    ):
        parsed = parse_optional_float(row.get(key))
        if parsed is not None:
            components[label] = float(parsed)
    return components or None


def rewrite_scan_log_row_scan_dir(
    *,
    scan_log_path: Path,
    row: dict[str, str],
    scan_dir: Path,
    parse_optional_int: Callable[[object], int | None],
    parse_optional_float: Callable[[object], float | None],
    score_components_from_scan_log_row: Callable[[dict[str, str]], dict[str, float] | None],
    record_scan_log_row: Callable[..., object],
) -> bool:
    iteration = parse_optional_int(row.get("iteration"))
    if iteration is None:
        return False
    bond_i = parse_optional_int(row.get("bond_i"))
    bond_j = parse_optional_int(row.get("bond_j"))
    bond_pair = (
        (int(bond_i), int(bond_j))
        if bond_i is not None and bond_j is not None
        else None
    )
    record_scan_log_row(
        scan_log_path,
        iteration=int(iteration),
        scan_dir=scan_dir,
        bond_pair=bond_pair,
        bond_length=parse_optional_float(row.get("bond_length")),
        iroot=parse_optional_int(row.get("iroot")),
        wavelength_nm=parse_optional_float(row.get("wavelength_nm")),
        intensity=parse_optional_float(row.get("intensity")),
        energy_ev=parse_optional_float(row.get("energy_ev")),
        ct_e=parse_optional_float(row.get("ct_e")),
        ct_pct=parse_optional_float(row.get("ct_pct")),
        ct_net=parse_optional_float(row.get("ct_net")),
        ct_magnitude=parse_optional_float(row.get("ct_magnitude")),
        score=parse_optional_float(row.get("score")),
        similarity=parse_optional_float(row.get("similarity")),
        score_components=score_components_from_scan_log_row(row),
        target_wavelength_nm=parse_optional_float(row.get("target_wavelength_nm")),
        point_kind=str(row.get("point_kind") or "").strip(),
        source_iteration=parse_optional_int(row.get("source_iteration")),
        reference_iteration=parse_optional_int(row.get("reference_iteration")),
        point_label=str(row.get("point_label") or "").strip(),
    )
    return True


def repair_scan_log_stale_paths(
    *,
    scan_log_path: Path,
    max_iter: int,
    use_tddft_optimisation: bool,
    load_scan_log: Callable[[Path], dict[str, dict[str, str]]],
    parse_optional_int: Callable[[object], int | None],
    scan_dir_has_valid_output: Callable[[Path], bool],
    resolve_scan_dir_from_hint: Callable[[str], Path | None],
    resolve_scan_dir_for_iteration_for_log_repair: Callable[..., Path | None],
    rewrite_scan_log_row_scan_dir: Callable[..., bool],
    summarize_scan_log: Callable[[Path], None],
) -> int:
    if not scan_log_path.exists():
        return 0
    rows = load_scan_log(scan_log_path)
    if not rows:
        return 0

    def _row_sort_key(item: tuple[str, dict[str, str]]) -> int:
        key, row = item
        iteration = parse_optional_int(row.get("iteration")) or parse_optional_int(key)
        if iteration is None:
            return 10**9
        return int(iteration)

    repaired = 0
    for key, row in sorted(rows.items(), key=_row_sort_key):
        iteration = parse_optional_int(row.get("iteration")) or parse_optional_int(key)
        if iteration is None:
            continue
        scan_dir_text = str(row.get("scan_dir") or "").strip()
        if scan_dir_text and scan_dir_has_valid_output(Path(scan_dir_text)):
            continue

        replacement = resolve_scan_dir_from_hint(scan_dir_text)
        if replacement is None and int(iteration) > 0:
            replacement = resolve_scan_dir_for_iteration_for_log_repair(
                int(iteration),
                use_tddft_optimisation=use_tddft_optimisation,
            )
        if replacement is None:
            continue

        point_kind = str(row.get("point_kind") or "").strip()
        if not point_kind and int(iteration) > int(max_iter):
            continue
        if not rewrite_scan_log_row_scan_dir(
            scan_log_path=scan_log_path,
            row=row,
            scan_dir=replacement,
        ):
            continue
        repaired += 1

    if repaired:
        summarize_scan_log(scan_log_path)
    return repaired


def load_last_geometry_log_row_for_iteration(
    geometry_log_path: Path,
    *,
    iteration: int,
    parse_optional_int: Callable[[object], int | None],
) -> dict[str, str] | None:
    if not geometry_log_path.exists():
        return None
    last_step = -1
    last_row: dict[str, str] | None = None
    try:
        with open(geometry_log_path, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if parse_optional_int(row.get("iteration")) != int(iteration):
                    continue
                step = parse_optional_int(row.get("geometry_step"))
                if step is None:
                    continue
                if step >= last_step:
                    last_step = int(step)
                    last_row = row
    except Exception:
        return None
    return last_row


def backfill_missing_primary_scan_log_rows(
    *,
    scan_log_path: Path,
    max_iter: int,
    use_tddft_optimisation: bool,
    bond_pair: tuple[int, int] | None,
    fragments: dict[str, str],
    target_wavelength_nm: float,
    geometry_log_filename: str,
    load_scan_log: Callable[[Path], dict[str, dict[str, str]]],
    parse_optional_int: Callable[[object], int | None],
    parse_optional_float: Callable[[object], float | None],
    validate_orca_output: Callable[[Path], bool],
    resolve_opt_dir_for_mode: Callable[..., Path],
    resolve_scan_dir_from_hint: Callable[[str], Path | None],
    resolve_scan_dir_for_iteration_for_log_repair: Callable[..., Path | None],
    scan_dir_has_valid_output: Callable[[Path], bool],
    record_scan_log_row: Callable[..., object],
    score_components_from_scan_log_row: Callable[[dict[str, str]], dict[str, float] | None],
    recover_scan_choice: Callable[..., object],
    score_weights: dict[str, float],
    wavelength_sigma_nm: float,
    similarity_sigma_nm: float,
    similarity_sigma_ct: float,
    min_oscillator_strength: float,
    bond_length_from_orca: Callable[[Path, tuple[int, int]], float | None],
    record_scan_choice: Callable[..., object],
    get_last_iroot: Callable[[Path], int | None],
    summarize_scan_log: Callable[[Path], None],
    load_last_geometry_log_row_for_iteration: Callable[[Path], dict[str, str] | None],
) -> int:
    if max_iter <= 0:
        return 0
    rows = load_scan_log(scan_log_path)
    existing_primary_iterations: set[int] = set()
    for key, row in rows.items():
        iteration = parse_optional_int(row.get("iteration"))
        if iteration is None:
            iteration = parse_optional_int(key)
        if iteration is None:
            continue
        if int(iteration) < 1 or int(iteration) > int(max_iter):
            continue
        if str(row.get("point_kind") or "").strip():
            continue
        existing_primary_iterations.add(int(iteration))

    backfilled = 0
    for iteration in range(1, int(max_iter) + 1):
        if iteration in existing_primary_iterations:
            continue
        opt_dir = resolve_opt_dir_for_mode(
            iteration,
            use_tddft_optimisation=use_tddft_optimisation,
        )
        opt_out = opt_dir / "orca.out"
        if not validate_orca_output(opt_out):
            continue

        geometry_log_path = opt_dir / "active_learning" / geometry_log_filename
        geometry_row = load_last_geometry_log_row_for_iteration(
            geometry_log_path,
            iteration=iteration,
        )
        scan_dir: Path | None = None
        if geometry_row is not None:
            scan_dir_text = str(geometry_row.get("scan_dir") or "").strip()
            scan_dir = resolve_scan_dir_from_hint(scan_dir_text)
        if scan_dir is None:
            scan_dir = resolve_scan_dir_for_iteration_for_log_repair(
                iteration,
                use_tddft_optimisation=use_tddft_optimisation,
            )
        if scan_dir is None or not scan_dir_has_valid_output(scan_dir):
            continue

        if geometry_row is not None:
            bond_i = parse_optional_int(geometry_row.get("bond_i"))
            bond_j = parse_optional_int(geometry_row.get("bond_j"))
            resolved_bond_pair = (
                (int(bond_i), int(bond_j))
                if bond_i is not None and bond_j is not None
                else bond_pair
            )
            record_scan_log_row(
                scan_log_path,
                iteration=iteration,
                scan_dir=scan_dir,
                bond_pair=resolved_bond_pair,
                bond_length=parse_optional_float(geometry_row.get("bond_length")),
                iroot=parse_optional_int(geometry_row.get("iroot")),
                wavelength_nm=parse_optional_float(geometry_row.get("wavelength_nm")),
                intensity=parse_optional_float(geometry_row.get("intensity")),
                energy_ev=parse_optional_float(geometry_row.get("energy_ev")),
                ct_e=parse_optional_float(geometry_row.get("ct_e")),
                ct_pct=parse_optional_float(geometry_row.get("ct_pct")),
                ct_net=parse_optional_float(geometry_row.get("ct_net")),
                ct_magnitude=parse_optional_float(geometry_row.get("ct_magnitude")),
                score=parse_optional_float(geometry_row.get("score")),
                similarity=parse_optional_float(geometry_row.get("similarity")),
                score_components=score_components_from_scan_log_row(geometry_row),
                target_wavelength_nm=(
                    parse_optional_float(geometry_row.get("target_wavelength_nm"))
                    or float(target_wavelength_nm)
                ),
            )
            backfilled += 1
            continue

        recovered_choice = recover_scan_choice(
            scan_dir,
            target_wavelength_nm=target_wavelength_nm,
            fragments=fragments,
            score_weights=score_weights,
            wavelength_sigma_nm=wavelength_sigma_nm,
            similarity_sigma_nm=similarity_sigma_nm,
            similarity_sigma_ct=similarity_sigma_ct,
            min_oscillator_strength=min_oscillator_strength,
        )
        if recovered_choice is not None:
            recovered_bond_length = (
                bond_length_from_orca(scan_dir, bond_pair)
                if bond_pair is not None
                else None
            )
            record_scan_choice(
                scan_log_path,
                iteration=iteration,
                scan_dir=scan_dir,
                bond_pair=bond_pair,
                bond_length=recovered_bond_length,
                choice=recovered_choice,
                target_wavelength_nm=target_wavelength_nm,
            )
            backfilled += 1
            continue

        record_scan_log_row(
            scan_log_path,
            iteration=iteration,
            scan_dir=scan_dir,
            bond_pair=bond_pair,
            bond_length=(
                bond_length_from_orca(scan_dir, bond_pair)
                if bond_pair is not None
                else None
            ),
            iroot=get_last_iroot(scan_dir / "orca.out"),
            target_wavelength_nm=target_wavelength_nm,
        )
        backfilled += 1

    if backfilled:
        summarize_scan_log(scan_log_path)
    return backfilled
