from __future__ import annotations
from cats.visualise.uvvis_data import TransitionData
import re
from typing import Optional
import numpy as np
_EV_PER_WN = 1.239841984e-4
_FLOAT_RE = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"

_NWCHEM_ROOT_RE = re.compile(
    rf"Root\s+(?P<root>\d+)\b.*?\s+"
    rf"(?P<energy_au>{_FLOAT_RE})\s+a\.u\.\s+"
    rf"(?P<energy_ev>{_FLOAT_RE})\s+eV",
    re.IGNORECASE,
)

_NWCHEM_TOTAL_OSC_RE = re.compile(
    rf"^\s*Total\s+Oscillator\s+Strength\s+(?P<value>{_FLOAT_RE})\b",
    re.IGNORECASE,
)

_NWCHEM_BARE_OSC_RE = re.compile(
    rf"^\s*Oscillator\s+Strength\s+"
    rf"(?:(?P<value>{_FLOAT_RE})|(?P<spin_forbidden>Spin\s+forbidden))\b",
    re.IGNORECASE,
)


def extract_uv_transitions_nwchem(filename: str) -> TransitionData:
    """
    Parse NWChem TDDFT output and return TransitionData compatible with the
    UV/Vis visualization utilities.

    The parser looks for blocks of the form:

        Root 200         a              0.116857103 a.u.                3.1798 eV
        ...
        Total Oscillator Strength                     0.0000271105

    and spin-labelled/spin-forbidden blocks of the form:

        Root   1 triplet a              0.262572727 a.u.                7.1450 eV
        ...
        Oscillator Strength                   Spin forbidden

    For each root it extracts:
        - excitation energy in eV (from the Root line)
        - oscillator strength, using 0.0 for spin-forbidden roots

    It then converts energies to wavenumbers (cm^-1) using _EV_PER_WN and
    computes corresponding wavelengths in nm.
    """

    wavenumbers: list[float] = []
    intensities: list[float] = []
    wavelengths: list[float] = []

    current_energy_ev: Optional[float] = None

    try:
        with open(filename, "r") as handle:
            for line in handle:
                # Detect Root line with energy in a.u. and eV
                m_root = _NWCHEM_ROOT_RE.search(line)
                if m_root:
                    # m_root.group(1) is the root index if you ever need it
                    # energy_au = float(m_root.group("energy_au"))  # unused here
                    energy_ev = float(m_root.group("energy_ev"))
                    current_energy_ev = energy_ev
                    continue

                # If we are inside a root block, use total oscillator strength when
                # present. Bare oscillator strength lines are used for spin-forbidden
                # roots and older NWChem formats, but "Dipole Oscillator Strength" is
                # intentionally ignored.
                if current_energy_ev is not None:
                    m_osc = _NWCHEM_TOTAL_OSC_RE.search(line) or _NWCHEM_BARE_OSC_RE.search(line)
                    if m_osc:
                        if m_osc.group("value") is not None:
                            osc_strength = float(m_osc.group("value"))
                        else:
                            osc_strength = 0.0

                        # Convert eV -> cm^-1 using the same constant as ORCA utilities
                        wn_cm = current_energy_ev / _EV_PER_WN  # cm^-1

                        wavenumbers.append(wn_cm)
                        intensities.append(osc_strength)

                        # Always compute wavelength in nm for NWChem
                        if wn_cm != 0.0:
                            wavelengths.append(1.0e7 / wn_cm)
                        else:
                            wavelengths.append(np.nan)

                        # Close this root block
                        current_energy_ev = None

    except FileNotFoundError as exc:
        raise FileNotFoundError(f"'{filename}' not found") from exc

    if not wavenumbers:
        raise ValueError(
            "No NWChem TDDFT roots with oscillator strengths found in "
            f"'{filename}'."
        )

    wavenumbers_arr = np.asarray(wavenumbers, dtype=float)
    intensities_arr = np.asarray(intensities, dtype=float)

    wavelengths_arr: Optional[np.ndarray] = None
    if wavelengths:
        wavelength_values = np.asarray(wavelengths, dtype=float)
        # For NWChem we always compute wavelengths, but keep the NaN logic consistent
        if not np.isnan(wavelength_values).any():
            wavelengths_arr = wavelength_values
        else:
            # If any NaN slipped in, treat as "no wavelengths"
            if np.isfinite(wavelength_values).any():
                # Only keep finite values if you want partial wavelengths;
                # otherwise comment this block and leave wavelengths_arr=None.
                wavelengths_arr = wavelength_values

    return TransitionData(
        wavenumbers_cm=wavenumbers_arr,
        intensities=intensities_arr,
        wavelengths_nm=wavelengths_arr,
    )
