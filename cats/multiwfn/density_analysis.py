"""Select excited states and export their Multiwfn density cubes."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable

from cats.multiwfn.cdd import charge_density_difference_routine
from cats.multiwfn.nto import nto_routine


DEFAULT_WAVELENGTH_BANDS_NM = ((300.0, 400.0), (400.0, 500.0), (500.0, 600.0), (600.0, 800.0))


def _value(row: dict, *keys: str):
    return next((row[key] for key in keys if row.get(key) is not None), None)


def select_relevant_excitations(
    ct_states: Iterable[dict],
    *,
    excitation_numbers: Iterable[int] = (),
    wavelength_bands_nm: Iterable[tuple[float, float]] = DEFAULT_WAVELENGTH_BANDS_NM,
) -> list[dict]:
    """Select explicit states, or the brightest state in each manuscript wavelength band."""

    states: dict[int, dict] = {}
    for raw in ct_states:
        row = dict(raw)
        state = _value(row, "excitation_number", "state", "index")
        try:
            state = int(state)
        except (TypeError, ValueError):
            continue
        if state < 1:
            continue
        states[state] = {
            **row,
            "excitation_number": state,
            "wavelength_nm": _value(row, "wavelength_nm", "nm"),
            "oscillator_strength": _value(row, "oscillator_strength", "f"),
            "charge_transfer_pct": _value(row, "charge_transfer_pct", "CT(%)"),
            "transferred_charge_e": _value(row, "transferred_charge_e", "CT(e)"),
        }

    requested = [int(value) for value in excitation_numbers]
    if requested:
        return [states.get(state, {"excitation_number": state, "reason": "explicit"}) for state in dict.fromkeys(requested)]

    selected: list[dict] = []
    bands = [(float(low), float(high)) for low, high in wavelength_bands_nm]
    for low, high in bands:
        candidates = []
        for row in states.values():
            try:
                wavelength = float(row["wavelength_nm"])
                strength = float(row["oscillator_strength"])
            except (KeyError, TypeError, ValueError):
                continue
            if low <= wavelength < high:
                candidates.append((strength, row))
        if candidates:
            row = dict(max(candidates, key=lambda item: item[0])[1])
            row["reason"] = f"brightest_{low:g}_{high:g}_nm"
            selected.append(row)
    return selected


def _generate_state_density(
    source_molden: str,
    source_output: str,
    output_directory: str,
    selected_state: dict,
    grid_quality: int,
    include_nto: bool,
    hole_selector: str,
    particle_selector: str,
    nto_max_pairs: int = 1,
    nto_target_coverage: float = 0.8,
) -> dict:
    output = Path(output_directory)
    state = int(selected_state["excitation_number"])
    prefix = f"state_{state:03d}"
    root = Path.cwd()
    with TemporaryDirectory(prefix=f".{prefix}_", dir=output) as temporary:
        try:
            os.chdir(temporary)
            cdd = charge_density_difference_routine(
                source_molden,
                source_output,
                state,
                output_prefix=f"{prefix}_cdd",
                grid_quality=grid_quality,
            )
            nto = (
                nto_routine(
                    source_molden,
                    source_output,
                    state,
                    output_prefix=f"{prefix}_nto",
                    grid_quality=grid_quality,
                    hole_selector=hole_selector,
                    particle_selector=particle_selector,
                    max_pairs=nto_max_pairs,
                    target_coverage=nto_target_coverage,
                )
                if include_nto
                else {}
            )
        finally:
            os.chdir(root)
            for source in Path(temporary).glob(f"{prefix}*"):
                target = output / source.name
                target.unlink(missing_ok=True)
                source.replace(target)
    pairs = [
        {
            **pair,
            "hole_cube": str(Path(output.name) / Path(pair["hole_cube"]).name),
            "particle_cube": str(Path(output.name) / Path(pair["particle_cube"]).name),
        }
        for pair in nto.get("pairs", [])
    ]
    return {
        **selected_state,
        "cdd_cube": str(Path(output.name) / Path(str(cdd["cdd_cube"])).name),
        "nto_hole_cube": (
            str(Path(output.name) / Path(str(nto["hole_cube"])).name) if nto else ""
        ),
        "nto_particle_cube": (
            str(Path(output.name) / Path(str(nto["particle_cube"])).name) if nto else ""
        ),
        "dominant_nto_pair_weight": nto.get("dominant_pair_weight"),
        "nto_pairs": pairs,
        "nto_coverage_fraction": nto.get("coverage_fraction"),
        "hole_selector": hole_selector,
        "particle_selector": particle_selector,
        "grid_quality": int(grid_quality),
        "sign_convention": cdd["sign_convention"],
    }


def generate_excitation_density_analysis(
    path_to_molden: str | Path,
    path_to_orca_out: str | Path,
    ct_states: Iterable[dict],
    *,
    output_directory: str | Path = "density_analysis",
    excitation_numbers: Iterable[int] = (),
    wavelength_bands_nm: Iterable[tuple[float, float]] = DEFAULT_WAVELENGTH_BANDS_NM,
    grid_quality: int = 3,
    include_nto: bool = True,
    export_pov_scenes: bool = True,
    delete_cubes_after_export: bool = True,
    hole_selector: str = "h",
    particle_selector: str = "l",
    max_workers: int = 1,
    nto_max_pairs: int = 1,
    nto_target_coverage: float = 0.8,
) -> dict:
    """Generate CDD and dominant NTO cubes for a bounded set of relevant states."""

    source_molden = Path(path_to_molden).resolve()
    source_output = Path(path_to_orca_out).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    requested = tuple(int(value) for value in excitation_numbers)
    bands = tuple((float(low), float(high)) for low, high in wavelength_bands_nm)
    selected = select_relevant_excitations(
        ct_states,
        excitation_numbers=requested,
        wavelength_bands_nm=bands,
    )
    workers = min(max(1, int(max_workers)), max(1, len(selected)))
    args = [
        (
            str(source_molden),
            str(source_output),
            str(output),
            selected_state,
            int(grid_quality),
            bool(include_nto),
            hole_selector,
            particle_selector,
            int(nto_max_pairs),
            float(nto_target_coverage),
        )
        for selected_state in selected
    ]
    if workers == 1:
        results = [_generate_state_density(*item) for item in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_state_density, *item) for item in args]
            results = [future.result() for future in futures]

    if export_pov_scenes:
        try:
            from cats.multiwfn.nto_povray import export_density_analysis_pov_scenes

            results = export_density_analysis_pov_scenes(
                output,
                results,
                delete_cubes=delete_cubes_after_export,
            )
        except Exception:
            # If POV-Ray scene export fails, preserve results with cubes
            pass

    manifest = {
        "selection_method": "explicit" if requested else "brightest_per_wavelength_band",
        "wavelength_bands_nm": [list(band) for band in bands],
        "grid_quality": int(grid_quality),
        "include_nto": bool(include_nto),
        "max_workers": workers,
        "nto_max_pairs": int(nto_max_pairs),
        "nto_target_coverage": float(nto_target_coverage),
        "pov_scenes_exported": bool(export_pov_scenes),
        "states": results,
    }
    manifest_path = output / "density_analysis_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    manifest["manifest_path"] = str(Path(output.name) / manifest_path.name)
    return manifest
