"""Natural transition orbital (NTO) generation with Multiwfn."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from cats.multiwfn.tools import run_multiwfn, save_routine_config


def get_nto_routine_config(
    path_to_molden: str,
    path_to_orca_out: str,
    excitation_number: int,
    output_mwfn: str,
) -> list[str]:
    """Return the Multiwfn 18-6 routine for one excited state."""

    state = int(excitation_number)
    if state < 1:
        raise ValueError("excitation_number must be at least 1")
    return [
        str(path_to_molden),
        "18",
        "6",
        str(path_to_orca_out),
        f"{state}",
        "3",
        str(output_mwfn),
        "0",
        "q",
    ]


def get_nto_cube_routine_config(
    path_to_nto_mwfn: str,
    orbital: str,
    *,
    grid_quality: int = 2,
) -> list[str]:
    """Export an NTO amplitude, optionally selecting alpha/beta explicitly."""

    selector = str(orbital).lower()
    if not re.fullmatch(r"(?:h|ha|hb)(?:-\d+)?|(?:l|la|lb)(?:\+\d+)?", selector):
        raise ValueError("orbital must be an h/l selector with an optional frontier offset")
    quality = int(grid_quality)
    if quality not in {1, 2, 3}:
        raise ValueError("grid_quality must be 1, 2, or 3")
    return [
        str(path_to_nto_mwfn),
        "5",
        "4",
        selector,
        f"{quality}",
        "2",
        "0",
        "q",
    ]


def parse_nto_eigenvalues(log_path: str | Path) -> list[float]:
    """Parse the reported leading NTO-pair eigenvalues from a Multiwfn log."""

    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"highest\s+\d+\s+eigenvalues of NTO pairs:\s*(.*?)\s*Sum of all eigenvalues:",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        return []
    return [float(value) for value in re.findall(r"[+-]?\d+\.\d+(?:[Ee][+-]?\d+)?", match.group(1))]


def parse_nto_eigenvalue_blocks(log_path: str | Path) -> dict[str, dict[str, object]]:
    """Parse restricted or spin-resolved NTO weights and reported totals."""

    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    blocks = {}
    for spin in ("alpha", "beta"):
        match = re.search(
            rf"highest\s+\d+\s+eigenvalues of {spin} NTO pairs:\s*(.*?)\s*"
            rf"Sum of all {spin} eigenvalues:\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            blocks[spin] = {
                "values": [float(value) for value in re.findall(r"[+-]?\d+\.\d+(?:[Ee][+-]?\d+)?", match.group(1))],
                "total": float(match.group(2)),
            }
    if blocks:
        return blocks
    values = parse_nto_eigenvalues(log_path)
    match = re.search(r"Sum of all eigenvalues:\s*([+-]?\d+\.\d+(?:[Ee][+-]?\d+)?)", text, re.I)
    return {"restricted": {"values": values, "total": float(match.group(1)) if match else sum(values)}} if values else {}


def select_nto_pairs(log_path, hole_selector="h", *, max_pairs=1, target_coverage=0.8, hard_cap=10):
    """Select fewest leading NTO pairs reaching target coverage (minimum 80%).

    If max_pairs is specified but cumulative coverage is still below target_coverage,
    selection continues beyond max_pairs until target_coverage is reached (capped by hard_cap).
    """

    blocks = parse_nto_eigenvalue_blocks(log_path)
    requested_spin = {"ha": "alpha", "hb": "beta"}.get(str(hole_selector).lower())
    allowed = [requested_spin] if requested_spin else list(blocks)
    entries = sorted(
        (
            {"spin": spin, "spin_rank": rank, "weight": float(weight)}
            for spin in allowed if spin in blocks
            for rank, weight in enumerate(blocks[spin]["values"], start=1)
        ),
        key=lambda row: row["weight"],
        reverse=True,
    )
    total = sum(float(blocks[spin]["total"]) for spin in allowed if spin in blocks)
    selected, cumulative = [], 0.0
    cap = max(int(max_pairs) if max_pairs is not None else 1, int(hard_cap))
    for idx, row in enumerate(entries, start=1):
        cumulative += row["weight"]
        coverage = cumulative / total if total else 0.0
        selected.append({**row, "cumulative_coverage": coverage})
        if total and coverage >= float(target_coverage):
            break
        if idx >= cap:
            break
    return selected


def ground_state_orbital_cubes(path_to_molden, *, output_directory="ground_state_orbitals", grid_quality=3, runner=run_multiwfn):
    """Export ground-state alpha/beta HOMOs and LUMOs, without generating NTOs."""
    import hashlib

    source = Path(path_to_molden).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    orbitals = []
    for selector in ("ha", "hb", "la", "lb"):
        config = directory / f"ground_{selector}_config.txt"
        log = directory / f"ground_{selector}_log.txt"
        target = directory / f"ground_{selector}.cub"
        raw = Path("MOvalue.cub")
        raw.unlink(missing_ok=True)
        save_routine_config(get_nto_cube_routine_config(str(source), selector, grid_quality=grid_quality), str(config))
        runner(str(config), str(log))
        if not raw.is_file() or not raw.stat().st_size:
            raise RuntimeError(f"Multiwfn did not export ground-state orbital {selector}")
        raw.replace(target)
        orbitals.append({"selector": selector, "cube": str(target), "log": str(log), "routine": str(config)})
    result = {"wavefunction": str(source), "wavefunction_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "orbital_kind": "ground_state_canonical", "grid_quality": grid_quality, "orbitals": orbitals}
    import json
    (directory / "orbital_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def nto_routine(
    path_to_molden: str,
    path_to_orca_out: str,
    excitation_number: int,
    *,
    output_prefix: str | None = None,
    grid_quality: int = 2,
    overwrite: bool = True,
    hole_selector: str = "h",
    particle_selector: str = "l",
    max_pairs: int = 1,
    target_coverage: float = 0.8,
    runner: Callable[[str, str], object] = run_multiwfn,
) -> dict[str, object]:
    """Generate the NTO wavefunction and dominant hole/particle cube files.

    Files are written in the current directory, matching the existing Multiwfn API.
    The dominant hole and particle NTOs are the HOMO and LUMO, respectively, of
    Multiwfn's exported NTO wavefunction.
    """

    state = int(excitation_number)
    if hole_selector not in {"h", "ha", "hb"} or particle_selector not in {"l", "la", "lb"}:
        raise ValueError("Invalid hole/particle NTO selectors")
    if state < 1:
        raise ValueError("excitation_number must be at least 1")
    prefix = output_prefix or f"NTO_state_{state}"
    nto_path = Path(f"{prefix}.mwfn")
    nto_config = Path(f"{prefix}_config.txt")
    nto_log = Path(f"{prefix}_log.txt")
    cube_paths = {"hole": Path(f"{prefix}_hole.cub"), "particle": Path(f"{prefix}_particle.cub")}

    if overwrite:
        for path in (nto_path, nto_config, nto_log, *cube_paths.values(), Path("MOvalue.cub")):
            path.unlink(missing_ok=True)

    if not nto_path.is_file():
        save_routine_config(
            get_nto_routine_config(path_to_molden, path_to_orca_out, state, str(nto_path)),
            str(nto_config),
        )
        runner(str(nto_config), str(nto_log))
    if not nto_path.is_file():
        raise RuntimeError(f"Multiwfn did not create {nto_path}")

    pairs = (
        select_nto_pairs(nto_log, hole_selector, max_pairs=max_pairs, target_coverage=target_coverage)
        if nto_log.is_file() and int(max_pairs) > 1
        else []
    )
    if not pairs:
        values = parse_nto_eigenvalues(nto_log) if nto_log.is_file() else []
        pairs = [{"spin": "selected", "spin_rank": 1, "weight": values[0] if values else None, "cumulative_coverage": values[0] if values else None}]
    for index, pair in enumerate(pairs, start=1):
        spin = pair["spin"]
        offset = int(pair["spin_rank"]) - 1
        if spin == "selected":
            selectors = {"hole": hole_selector, "particle": particle_selector}
        else:
            suffix = {"alpha": "a", "beta": "b", "restricted": ""}[spin]
            selectors = {
                "hole": f"h{suffix}" + (f"-{offset}" if offset else ""),
                "particle": f"l{suffix}" + (f"+{offset}" if offset else ""),
            }
        for role, selector in selectors.items():
            target = (
                cube_paths[role]
                if len(pairs) == 1
                else Path(f"{prefix}_pair_{index:02d}_{spin}_{role}.cub")
            )
            raw_cube = Path("MOvalue.cub")
            raw_cube.unlink(missing_ok=True)
            config = target.with_name(target.stem + "_config.txt")
            log = target.with_name(target.stem + "_log.txt")
            save_routine_config(
                get_nto_cube_routine_config(str(nto_path), selector, grid_quality=grid_quality),
                str(config),
            )
            runner(str(config), str(log))
            if not raw_cube.is_file():
                raise RuntimeError(f"Multiwfn did not create {raw_cube} for the {role} NTO")
            raw_cube.replace(target)
            pair[f"{role}_selector"] = selector
            pair[f"{role}_cube"] = str(target)

    cube_paths = {role: Path(pairs[0][f"{role}_cube"]) for role in ("hole", "particle")}
    eigenvalues = [pair["weight"] for pair in pairs if pair["weight"] is not None]
    return {
        "excitation_number": state,
        "nto_wavefunction": str(nto_path),
        "hole_cube": str(cube_paths["hole"]),
        "particle_cube": str(cube_paths["particle"]),
        "pair_eigenvalues": eigenvalues,
        "pairs": pairs,
        "coverage_fraction": pairs[-1]["cumulative_coverage"],
        "dominant_pair_weight": eigenvalues[0] if eigenvalues else None,
        "grid_quality": int(grid_quality),
        "hole_selector": hole_selector,
        "particle_selector": particle_selector,
    }
