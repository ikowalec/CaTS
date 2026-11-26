# -*- coding: utf-8 -*-
"""Parser for ORCA UV/Vis transition data."""

from __future__ import annotations

import re
from typing import Optional

import numpy as np

from cats.visualise.uvvis_data import TransitionData

SPECSTRING_START = "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
SPECSTRING_END = "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
_VERSION6_SENTINEL = "Program Version 6"
_VERSION_PREFIX = "Program Version"
_TRANSITION_LINE = re.compile(r"^\s*\d")

__all__ = ["TransitionData", "extract_uv_transitions"]


def extract_uv_transitions(filename: str) -> TransitionData:
    """
    Return the raw transition data extracted from the ORCA output file.

    The returned :class:`TransitionData` object can still be unpacked as
    ``wavenumbers, intensities`` for backwards compatibility.
    """

    return _parse_transitions(filename)


def _parse_transitions(filename: str) -> TransitionData:
    energy_col = 1
    intensity_col = 3
    found_start = False
    wavenumbers = []
    intensities = []
    wavelengths = []

    try:
        with open(filename, "r") as handle:
            for line in handle:
                if _VERSION6_SENTINEL in line:
                    energy_col, intensity_col = 4, 6
                    continue
                if (
                    _VERSION_PREFIX in line
                    and _VERSION6_SENTINEL not in line
                    and "Program Version" in line
                ):
                    energy_col, intensity_col = 1, 3

                if SPECSTRING_START in line:
                    found_start = True
                    for line in handle:
                        if SPECSTRING_END in line:
                            break
                        if not _TRANSITION_LINE.match(line):
                            continue
                        parts = line.strip().split()
                        if len(parts) <= max(energy_col, intensity_col):
                            continue
                        energy = float(parts[energy_col])
                        intensity = float(parts[intensity_col])
                        wavenumbers.append(energy)
                        intensities.append(intensity)

                        raw_wavelength = _extract_wavelength_from_line(
                            parts, energy, energy_col, intensity_col
                        )
                        wavelengths.append(
                            raw_wavelength if raw_wavelength is not None else np.nan
                        )
                    break
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"'{filename}' not found") from exc

    if not found_start:
        raise ValueError(f"'{SPECSTRING_START}' not found in '{filename}'.")
    if not wavenumbers:
        raise ValueError(
            f"No transition data found between the spectrum markers in '{filename}'."
        )

    wavenumbers_arr = np.asarray(wavenumbers, dtype=float)
    intensities_arr = np.asarray(intensities, dtype=float)

    wavelengths_arr: Optional[np.ndarray] = None
    if wavelengths:
        wavelength_values = np.asarray(wavelengths, dtype=float)
        if not np.isnan(wavelength_values).any():
            wavelengths_arr = wavelength_values

    return TransitionData(
        wavenumbers_cm=wavenumbers_arr,
        intensities=intensities_arr,
        wavelengths_nm=wavelengths_arr,
    )


def _extract_wavelength_from_line(
    parts: list[str],
    wavenumber: float,
    energy_index: int,
    intensity_index: int,
) -> Optional[float]:
    """Return the wavelength column from the parsed line when ORCA reports it."""
    if wavenumber == 0.0:
        return None

    target = 1.0e7 / wavenumber
    tolerance = max(0.05, target * 1.0e-4)

    for idx, token in enumerate(parts):
        if idx in {energy_index, intensity_index}:
            continue
        try:
            value = float(token)
        except ValueError:
            continue
        if value <= 0.0:
            continue
        if abs(value - target) <= tolerance:
            return value

    return None
