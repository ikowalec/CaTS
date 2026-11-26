from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

__all__ = ["SpectrumData", "TransitionData"]


@dataclass(frozen=True)
class SpectrumData:
    """Container for the broadened UV/Vis spectrum and stick data."""

    x: np.ndarray
    y: np.ndarray
    sticks_x: np.ndarray
    sticks_y: np.ndarray
    x_unit: str
    sticks_unit: str

    def as_xy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the broadened spectrum as an (x, y) tuple."""
        return self.x, self.y


@dataclass(frozen=True)
class TransitionData:
    """
    Container for the raw transition data extracted from a UV/Vis spectrum.

    The ``wavelengths_nm`` attribute is ``None`` when the output does not
    report wavelengths explicitly.
    """

    wavenumbers_cm: np.ndarray
    intensities: np.ndarray
    wavelengths_nm: Optional[np.ndarray] = None

    def __iter__(self):
        """Allow unpacking like ``energies, intensities = data`` for compatibility."""
        yield self.wavenumbers_cm
        yield self.intensities

    @property
    def has_wavelengths(self) -> bool:
        """Return ``True`` when wavelength information was present in the output."""
        return self.wavelengths_nm is not None
