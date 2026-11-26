"""Extract and plot a broadened UV/Vis spectrum from an NWChem output file.

This file is intentionally self-contained so it can be copied and run directly::

    python nwchem_uvvis_extraction.py nwchem.out

The defaults match the NWChem TDDFT workflow: a 350--800 nm axis sampled every
1 nm and Gaussian broadening with a 20 nm full width at half maximum (FWHM).
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

EV_PER_WAVENUMBER = 1.239841984e-4
DEFAULT_START_NM = 350
DEFAULT_END_NM = 800.0
DEFAULT_LINEWIDTH_NM = 20
DEFAULT_STEP_NM = 1.0

_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?"
_ROOT_RE = re.compile(
    rf"\bRoot\s+(?P<root>\d+)\b.*?"
    rf"(?P<energy_au>{_FLOAT})\s+a\.u\.\s+"
    rf"(?P<energy_ev>{_FLOAT})\s+eV\b",
    re.IGNORECASE,
)
_TOTAL_OSC_RE = re.compile(
    rf"^\s*Total\s+Oscillator\s+Strength\s+(?P<value>{_FLOAT})\b",
    re.IGNORECASE,
)
_BARE_OSC_RE = re.compile(
    rf"^\s*Oscillator\s+Strength\s+"
    rf"(?:(?P<value>{_FLOAT})|(?P<spin_forbidden>Spin\s+forbidden))\b",
    re.IGNORECASE,
)

__all__ = [
    "SpectrumData",
    "TransitionData",
    "extract_uv_transitions_nwchem",
    "build_uv_spectrum_nm",
    "plot_uv_spectrum_nm",
    "write_spectrum_csv",
    "main",
]


@dataclass(frozen=True)
class TransitionData:
    """Electronic transitions extracted from NWChem TDDFT output."""

    wavenumbers_cm: np.ndarray
    intensities: np.ndarray
    wavelengths_nm: np.ndarray
    root_numbers: np.ndarray

    def __iter__(self):
        """Preserve compatibility with code that unpacks energy and intensity."""
        yield self.wavenumbers_cm
        yield self.intensities

    @property
    def energies_ev(self) -> np.ndarray:
        return self.wavenumbers_cm * EV_PER_WAVENUMBER

    @property
    def has_wavelengths(self) -> bool:
        return True


@dataclass(frozen=True)
class SpectrumData:
    """Broadened spectrum and the transitions used to construct it."""

    x: np.ndarray
    y: np.ndarray
    sticks_x: np.ndarray
    sticks_y: np.ndarray
    x_unit: str = "nm"
    sticks_unit: str = "nm"

    def as_xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x, self.y


def _as_float(value: str) -> float:
    """Accept the E and Fortran D exponents found in NWChem output."""
    return float(value.replace("D", "E").replace("d", "e"))


def extract_uv_transitions_nwchem(filename: str | Path) -> TransitionData:
    """Parse TDDFT root energies and oscillator strengths from ``nwchem.out``.

    Spin-forbidden transitions are retained with zero oscillator strength.
    ``Total Oscillator Strength`` is preferred; the unqualified form is also
    accepted for spin-forbidden roots and older NWChem output.
    """

    path = Path(filename)
    roots: list[int] = []
    energies_ev: list[float] = []
    intensities: list[float] = []
    current: tuple[int, float] | None = None

    try:
        lines = path.open("r", encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"NWChem output not found: {path}") from exc

    with lines:
        for line in lines:
            root_match = _ROOT_RE.search(line)
            if root_match:
                current = (
                    int(root_match.group("root")),
                    _as_float(root_match.group("energy_ev")),
                )
                continue

            if current is None:
                continue
            oscillator_match = _TOTAL_OSC_RE.search(line) or _BARE_OSC_RE.search(line)
            if oscillator_match is None:
                continue

            strength = (
                _as_float(oscillator_match.group("value"))
                if oscillator_match.group("value") is not None
                else 0.0
            )
            root, energy_ev = current
            if energy_ev <= 0:
                raise ValueError(f"Root {root} has a non-positive excitation energy: {energy_ev} eV")
            roots.append(root)
            energies_ev.append(energy_ev)
            intensities.append(strength)
            current = None

    if not roots:
        raise ValueError(f"No NWChem TDDFT transitions found in {path}")

    energies = np.asarray(energies_ev, dtype=float)
    wavenumbers = energies / EV_PER_WAVENUMBER
    return TransitionData(
        wavenumbers_cm=wavenumbers,
        intensities=np.asarray(intensities, dtype=float),
        wavelengths_nm=1.0e7 / wavenumbers,
        root_numbers=np.asarray(roots, dtype=int),
    )


def build_uv_spectrum_nm(
    transitions: TransitionData,
    *,
    linewidth_nm: float = DEFAULT_LINEWIDTH_NM,
    start_nm: float = DEFAULT_START_NM,
    end_nm: float = DEFAULT_END_NM,
    step: float = DEFAULT_STEP_NM,
    normalize: bool = False,
) -> SpectrumData:
    """Apply wavelength-domain Gaussian broadening to NWChem transitions.

    ``linewidth_nm`` is the Gaussian full width at half maximum. Raw oscillator
    strengths are used as peak heights, matching the NWChem TDDFT workflow.
    """

    linewidth_nm = float(linewidth_nm)
    start_nm = float(start_nm)
    end_nm = float(end_nm)
    step = float(step)
    if linewidth_nm <= 0:
        raise ValueError("linewidth_nm must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if start_nm < 0 or end_nm <= start_nm:
        raise ValueError("require 0 <= start_nm < end_nm")

    axis = np.arange(start_nm, end_nm + step, step, dtype=float)
    sigma = linewidth_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    offsets = (axis[np.newaxis, :] - transitions.wavelengths_nm[:, np.newaxis]) / sigma
    profile = np.sum(
        transitions.intensities[:, np.newaxis] * np.exp(-0.5 * offsets**2),
        axis=0,
    )
    if normalize and profile.size and float(profile.max()) > 0:
        profile = profile / profile.max()

    return SpectrumData(
        x=axis,
        y=profile,
        sticks_x=transitions.wavelengths_nm,
        sticks_y=transitions.intensities,
    )


def plot_uv_spectrum_nm(
    spectrum: SpectrumData,
    *,
    output: str | Path | None = None,
    title: str = "NWChem TDDFT spectrum",
    show_sticks: bool = False,
    show: bool = False,
    dpi: int = 300,
):
    """Plot a spectrum and optionally save it; return ``(figure, axes)``."""

    fig, ax = plt.subplots()
    ax.plot(spectrum.x, spectrum.y)
    if show_sticks:
        ax.vlines(spectrum.sticks_x, 0.0, spectrum.sticks_y, color="tab:blue", alpha=0.35)
    ax.set(
        xlim=(float(spectrum.x[0]), float(spectrum.x[-1])),
        xlabel="Wavelength / nm",
        ylabel="Broadened oscillator strength",
        title=title,
    )
    fig.tight_layout()
    if output is not None:
        fig.savefig(Path(output), dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def write_spectrum_csv(spectrum: SpectrumData, filename: str | Path) -> None:
    """Write the same two-column format exported by the TDDFT workflows."""

    with Path(filename).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("wavelength_nm", "intensity"))
        writer.writerows(zip(spectrum.x, spectrum.y))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="_nwchem.out", help="NWChem output (default: nwchem.out)")
    parser.add_argument("-o", "--output", type=Path, help="PNG output (default: <input>_uvvis.png)")
    parser.add_argument("--csv", type=Path, help="also write broadened spectrum data")
    parser.add_argument("--start-nm", type=float, default=DEFAULT_START_NM)
    parser.add_argument("--end-nm", type=float, default=DEFAULT_END_NM)
    parser.add_argument("--linewidth-nm", type=float, default=DEFAULT_LINEWIDTH_NM, help="Gaussian FWHM")
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_NM)
    parser.add_argument("--normalize", action="store_true", help="scale the spectrum maximum to 1")
    parser.add_argument("--sticks", action="store_true", help="draw the individual transitions")
    parser.add_argument("--show", action="store_true", help="open an interactive plot window")
    parser.add_argument("--title", default="NWChem TDDFT spectrum")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""

    args = _parser().parse_args(argv)
    input_path = Path(args.input)
    output = args.output or input_path.with_name(f"{input_path.stem}_uvvis.png")
    transitions = extract_uv_transitions_nwchem(input_path)
    spectrum = build_uv_spectrum_nm(
        transitions,
        linewidth_nm=args.linewidth_nm,
        start_nm=args.start_nm,
        end_nm=args.end_nm,
        step=args.step,
        normalize=args.normalize,
    )
    figure, _ = plot_uv_spectrum_nm(
        spectrum,
        output=output,
        title=args.title,
        show_sticks=args.sticks,
        show=args.show,
    )
    if args.csv:
        write_spectrum_csv(spectrum, args.csv)
    if not args.show:
        plt.close(figure)
    print(f"Parsed {len(transitions.root_numbers)} transitions; wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
