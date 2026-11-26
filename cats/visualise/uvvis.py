from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from cats.visualise.uvvis_data import SpectrumData, TransitionData

_GAUSS_FACTOR = float(4.0 * np.log(2.0))
_EV_PER_WN = 1.239841984e-4

__all__ = [
    "SpectrumData",
    "TransitionData",
    "extract_uv_spectrum",
    "plot_uv_spectrum_nm",
    "plot_uv_spectrum_nm_with_ev_peaks",
    "extract_uv_transitions_from_folders",
    "plot_uv_spectra_from_folders",
    "plot_uv_spectra_electronvolt",
    "plot_uv_spectra_stacked",
]


def _format_colorbar_ticks(ticks: np.ndarray, *, axis_label: Optional[str] = None) -> list[str]:
    label = str(axis_label or "").strip().lower()
    is_bond_length_axis = ("bond" in label and "length" in label) and ("angstrom" in label or "angstroem" in label)
    if is_bond_length_axis:
        return [f"{float(tick):.2f}" for tick in ticks]
    return [f"{float(tick):g}" for tick in ticks]


def extract_uv_spectrum(
    filename: str,
    *,
    nm_plot: bool = True,
    units: Optional[str] = None,
    linewidth_nm: float = 20.0,
    linewidth_wn: float = 1000.0,
    startx: Optional[float] = None,
    endx: Optional[float] = None,
    step: float = 1.0,
    x_axis: Optional[Iterable[float]] = None,
    normalize: bool = False,
    normalize_factor: float = 1.0,
) -> SpectrumData:
    """
    Parse an ORCA output file and return the broadened spectrum data.

    Parameters
    ----------
    filename:
        Path to the ORCA output file.
    nm_plot:
        Legacy flag mirroring the behaviour of ``orca-uv.py``. Ignored when
        ``units`` is provided.
    units:
        Select the output axis units: ``"nm"`` for wavelengths or ``"cm-1"``
        for wavenumbers. When omitted the legacy ``nm_plot`` flag is used.
    linewidth_nm / linewidth_wn:
        Full-width at half maximum used for the Gaussian broadening. The
        appropriate value is selected based on the chosen units.
    startx / endx:
        Optional lower and upper bounds of the spectrum x-axis in the selected
        units. ``None`` falls back to the automatic limits used by ``orca-uv``.
    step:
        Step size used to build the spectrum axis when ``x_axis`` is not
        provided.
    x_axis:
        Optional explicit x-axis to sample the broadened spectrum on. When
        provided the ``startx``, ``endx`` and ``step`` arguments are ignored.
    normalize:
        When ``True`` the broadened spectrum is rescaled so that ``max(y)`` is
        equal to ``normalize_factor``.
    normalize_factor:
        Scaling factor applied when ``normalize`` is ``True``.

    Returns
    -------
    SpectrumData
        A dataclass containing the broadened spectrum and the original stick
        data in the selected units.
    """

    from cats.orca.routines.orca_uv_extractor import extract_uv_transitions

    transition_data = extract_uv_transitions(filename)
    energies_cm = transition_data.wavenumbers_cm
    intensities = transition_data.intensities.astype(float, copy=False)

    use_nm, axis_unit = _resolve_units(units, nm_plot)

    if use_nm:
        if transition_data.wavelengths_nm is not None:
            sticks_x = transition_data.wavelengths_nm.astype(float, copy=False)
        else:
            sticks_x = _wn_to_nm(energies_cm)
        width = float(linewidth_nm)
        sticks_unit = "nm"
    else:
        sticks_x = energies_cm.astype(float, copy=False)
        width = float(linewidth_wn)
        sticks_unit = "cm^-1"

    if width <= 0:
        raise ValueError("Line width must be positive.")
    if step <= 0:
        raise ValueError("Step size must be positive.")
    if normalize_factor <= 0:
        raise ValueError("normalize_factor must be positive.")
    if startx is not None and endx is not None and np.isclose(startx, endx):
        raise ValueError("startx and endx must not be equal.")
    if startx is not None and startx < 0:
        raise ValueError("startx must be >= 0.")
    if endx is not None and endx < 0:
        raise ValueError("endx must be >= 0.")

    if x_axis is not None:
        if isinstance(x_axis, np.ndarray):
            axis = x_axis.astype(float, copy=False)
        else:
            axis = np.asarray(list(x_axis), dtype=float)
        if axis.ndim != 1:
            raise ValueError("x_axis must be one-dimensional.")
    else:
        axis = _build_axis(sticks_x, width, startx, endx, step)

    profile = _gaussian_broadening(sticks_x, intensities, axis, width)

    if normalize and profile.size:
        maximum = profile.max()
        if maximum > 0:
            profile = profile / maximum * normalize_factor

    return SpectrumData(
        x=axis,
        y=profile,
        sticks_x=sticks_x,
        sticks_y=intensities,
        x_unit=axis_unit,
        sticks_unit=sticks_unit,
    )


def plot_uv_spectrum_nm(
    transition_data: TransitionData,
    *,
    linewidth_nm: float = 20.0,
    start_nm: Optional[float] = None,
    end_nm: Optional[float] = None,
    step: float = 1.0,
    x_axis: Optional[Iterable[float]] = None,
    normalize: bool = False,
    normalize_factor: float = 1.0,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
) -> Tuple[plt.Axes, SpectrumData]:
    """
    Plot a broadened UV/Vis spectrum in nanometers from generic transition data.

    This helper only depends on :class:`TransitionData` and does not perform any
    parsing itself, making it suitable for spectra originating from different
    quantum chemistry codes as long as the transition information is provided.
    """

    sticks_x = (
        transition_data.wavelengths_nm.astype(float, copy=False)
        if transition_data.wavelengths_nm is not None
        else _wn_to_nm(transition_data.wavenumbers_cm)
    )
    intensities = transition_data.intensities.astype(float, copy=False)

    width = float(linewidth_nm)
    if width <= 0:
        raise ValueError("Line width must be positive.")
    if step <= 0:
        raise ValueError("Step size must be positive.")
    if normalize_factor <= 0:
        raise ValueError("normalize_factor must be positive.")
    if start_nm is not None and end_nm is not None and np.isclose(start_nm, end_nm):
        raise ValueError("start_nm and end_nm must not be equal.")
    if start_nm is not None and start_nm < 0:
        raise ValueError("start_nm must be >= 0.")
    if end_nm is not None and end_nm < 0:
        raise ValueError("end_nm must be >= 0.")

    if x_axis is not None:
        axis = np.asarray(list(x_axis), dtype=float)
        if axis.ndim != 1:
            raise ValueError("x_axis must be one-dimensional.")
    else:
        axis = _build_axis(sticks_x, width, start_nm, end_nm, step)

    profile = _gaussian_broadening(sticks_x, intensities, axis, width)

    if normalize and profile.size:
        peak = profile.max()
        if peak > 0:
            profile = profile / peak * normalize_factor

    spectrum = SpectrumData(
        x=axis,
        y=profile,
        sticks_x=sticks_x,
        sticks_y=intensities,
        x_unit="nm",
        sticks_unit="nm",
    )

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(axis, profile, label=label)
    ax.set_xlabel("Wavelength / nm")
    ax.set_ylabel("Intensity")

    if label is not None:
        ax.legend()

    return ax, spectrum


def plot_uv_spectrum_nm_with_ev_peaks(
    transition_data: TransitionData,
    *,
    linewidth_nm: float = 20.0,
    start_nm: Optional[float] = None,
    end_nm: Optional[float] = None,
    step: float = 1.0,
    x_axis: Optional[Iterable[float]] = None,
    normalize: bool = False,
    normalize_factor: float = 1.0,
    ax: Optional[plt.Axes] = None,
    label: Optional[str] = None,
    peak_relative_height: float = 0.05,
) -> Tuple[plt.Axes, SpectrumData]:
    """
    Plot a wavelength spectrum and label peaks with electronvolt energies.

    Peak positions are determined from the broadened profile and annotated with
    the corresponding energy in eV, computed from the peak wavelength.
    """

    if peak_relative_height <= 0 or peak_relative_height > 1:
        raise ValueError("peak_relative_height must be in the interval (0, 1].")

    ax, spectrum = plot_uv_spectrum_nm(
        transition_data,
        linewidth_nm=linewidth_nm,
        start_nm=start_nm,
        end_nm=end_nm,
        step=step,
        x_axis=x_axis,
        normalize=normalize,
        normalize_factor=normalize_factor,
        ax=ax,
        label=label,
    )

    line = ax.lines[-1]
    line_color = line.get_color()

    axis = spectrum.x
    profile = spectrum.y
    peak_indices = _find_profile_peaks(profile, peak_relative_height)

    for p_idx in peak_indices:
        peak_nm = float(axis[p_idx])
        if peak_nm <= 0:
            continue
        peak_y = float(profile[p_idx])
        peak_ev = float(_wn_to_ev(1.0e7 / peak_nm))

        ax.plot(peak_nm, peak_y, marker="o", color=line_color)
        ax.text(
            peak_nm,
            peak_y,
            f"{peak_ev:.2f} eV",
            fontsize=8,
            ha="center",
            va="bottom",
            color=line_color,
        )

    return ax, spectrum


def extract_uv_transitions_from_folders(
    folders: Iterable[str],
    *,
    output_name: str = "orca.out",
) -> Tuple[TransitionData, ...]:
    """
    Extract transition data for each folder without performing any plotting.

    Parameters
    ----------
    folders:
        Iterable of directories containing ORCA output files.
    output_name:
        Name of the ORCA output file found in each folder.

    Returns
    -------
    tuple[TransitionData, ...]
        Transition data objects in the same order as ``folders``.
    """

    from cats.orca.routines.orca_uv_extractor import extract_uv_transitions

    folder_list = list(folders)
    if not folder_list:
        raise ValueError("folders must contain at least one directory.")

    transitions: list[TransitionData] = []
    for folder in folder_list:
        path = Path(folder) / output_name
        print("Extracting transitions from:", path)
        transitions.append(extract_uv_transitions(str(path)))

    return tuple(transitions)


def plot_uv_spectra_from_folders(
    folders: Iterable[str],
    *,
    output_name: str = "orca.out",
    units: Optional[str] = None,
    linewidth_nm: float = 20.0,
    linewidth_wn: float = 1000.0,
    step: float = 1.0,
    normalize: bool = False,
    normalize_factor: float = 1.0,
    cmap: Union[str, mcolors.Colormap] = "viridis",
    ax: Optional[plt.Axes] = None,
    transitions: Optional[Iterable[TransitionData]] = None,
    labels: Optional[Iterable[Union[str, Real]]] = None,
) -> Tuple[plt.Axes, Tuple[SpectrumData, ...]]:
    """
    Extract and plot UV/Vis spectra from multiple ORCA calculation folders.

    Each folder is expected to contain the ORCA output file specified by
    ``output_name`` (``orca.out`` by default). The spectra are broadened using
    Gaussian profiles and plotted on a shared axis with a color gradient.

    Parameters
    ----------
    folders:
        Iterable of directories containing ORCA output files.
    output_name:
        Name of the ORCA output file found in each folder.
    units:
        Output axis units: ``"nm"`` or ``"cm-1"``. Falls back to the legacy
        ``nm_plot`` behaviour (wavelengths).
    linewidth_nm / linewidth_wn:
        Gaussian FWHM for wavelength or wavenumber spectra respectively.
    step:
        Sampling step applied to the x-axis.
    normalize:
        When ``True`` each spectrum is scaled so that ``max(y)`` equals
        ``normalize_factor``.
    normalize_factor:
        Target peak height when ``normalize`` is enabled.
    cmap:
        Matplotlib colormap name or :class:`matplotlib.colors.Colormap` used for the
        spectrum colors.
    ax:
        Optional matplotlib Axes to draw on. When omitted a new figure and axes
        are created.
    transitions:
        Optional iterable of :class:`TransitionData` objects pre-extracted for each
        folder. When provided the ORCA outputs are not re-read.
    labels:
        Optional iterable matching ``folders``. String labels are used directly;
        numeric labels drive the colormap scaling and produce a colorbar annotated
        with their minimum and maximum values.

    Returns
    -------
    (Axes, tuple[SpectrumData, ...])
        The axes used for plotting and the extracted spectra in plotting order.
    """

    from cats.orca.routines.orca_uv_extractor import extract_uv_transitions

    folder_list = list(folders)
    if not folder_list:
        raise ValueError("folders must contain at least one directory.")

    label_list: Optional[list[str]] = None
    color_values: Optional[np.ndarray] = None
    if labels is not None:
        label_items = list(labels)
        if len(label_items) != len(folder_list):
            raise ValueError("labels must contain the same number of entries as folders.")
        label_list = [str(label) for label in label_items]
        if label_items and all(isinstance(label, Real) for label in label_items):
            numeric_values = np.asarray(label_items, dtype=float)
            if np.isfinite(numeric_values).all():
                color_values = numeric_values.astype(float, copy=False)

    if transitions is not None:
        transition_list = [_ensure_transition_data(item) for item in transitions]
        if len(transition_list) != len(folder_list):
            raise ValueError("transitions must contain the same number of entries as folders.")
    else:
        transition_list = []
        for folder in folder_list:
            path = Path(folder) / output_name
            print("Extracting transitions from:", path)
            transition_list.append(extract_uv_transitions(str(path)))

    use_nm, axis_unit = _resolve_units(units, True)
    width = float(linewidth_nm) if use_nm else float(linewidth_wn)

    if width <= 0:
        raise ValueError("Line width must be positive.")
    if step <= 0:
        raise ValueError("Step size must be positive.")
    if normalize_factor <= 0:
        raise ValueError("normalize_factor must be positive.")

    spectra: list[SpectrumData] = []
    max_stick = 0.0

    stick_store: list[Tuple[np.ndarray, np.ndarray]] = []

    for idx, folder in enumerate(folder_list):
        transition_data = transition_list[idx]
        energies_cm = transition_data.wavenumbers_cm
        intensities = transition_data.intensities.astype(float, copy=False)

        if use_nm:
            if transition_data.wavelengths_nm is not None:
                sticks_x = transition_data.wavelengths_nm.astype(float, copy=False)
            else:
                sticks_x = _wn_to_nm(energies_cm)
            sticks_unit = "nm"
        else:
            sticks_x = energies_cm.astype(float, copy=False)
            sticks_unit = "cm^-1"

        stick_store.append((sticks_x, intensities))
        max_stick = max(max_stick, float(np.max(sticks_x)))

    stop = max_stick + width * 3.0
    if stop <= 0.0:
        stop = width * 3.0

    axis = np.arange(0.0, stop + step, step, dtype=float)

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    elif isinstance(cmap, mcolors.Colormap):
        cmap_obj = cmap
    else:
        raise TypeError("cmap must be a Matplotlib colormap name or Colormap instance.")

    use_colorbar = color_values is not None and color_values.size > 0
    colorbar_norm: Optional[mcolors.Normalize] = None
    colorbar_ticks: Tuple[float, ...] = ()

    if use_colorbar:
        vmin = float(np.min(color_values))
        vmax = float(np.max(color_values))

        if np.isclose(vmin, vmax):
            span = max(abs(vmin), 1.0)
            if span == 0.0:
                span = 1.0
            half_span = span * 0.5
            colorbar_norm = mcolors.Normalize(vmin=vmin - half_span, vmax=vmax + half_span)
            normalized = np.full(color_values.shape, 0.5, dtype=float)
            colorbar_ticks = (vmin, vmax)
        else:
            colorbar_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            normalized = colorbar_norm(color_values)
            colorbar_ticks = (vmin, vmax)

        colors = cmap_obj(normalized)
    else:
        colors = cmap_obj(np.linspace(0.0, 1.0, len(stick_store)))

    if ax is None:
        _, ax = plt.subplots()

    for idx, (sticks_x, intensities) in enumerate(stick_store):
        print("Processing Gaussian broadening for folder:", folder_list[idx])
        profile = _gaussian_broadening(sticks_x, intensities, axis, width)

        if normalize and profile.size:
            peak = profile.max()
            if peak > 0:
                profile = profile / peak * normalize_factor

        spectrum = SpectrumData(
            x=axis,
            y=profile,
            sticks_x=sticks_x,
            sticks_y=intensities,
            x_unit=axis_unit,
            sticks_unit=sticks_unit,
        )
        spectra.append(spectrum)

        if use_colorbar:
            ax.plot(axis, profile, color=colors[idx])
        else:
            if label_list is not None:
                label = label_list[idx]
            else:
                label = Path(folder_list[idx]).name or str(folder_list[idx])
            ax.plot(axis, profile, color=colors[idx], label=label)

    ax.set_xlabel("Wavelength / nm" if use_nm else "Energy / cm^-1")
    ax.set_ylabel("Intensity")

    if use_colorbar and colorbar_norm is not None:
        mappable = plt.cm.ScalarMappable(norm=colorbar_norm, cmap=cmap_obj)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax)
        if colorbar_ticks:
            cbar.set_ticks(colorbar_ticks)
            cbar.set_ticklabels([f"{tick:g}" for tick in colorbar_ticks])
    else:
        ax.legend()

    return ax, tuple(spectra)


def plot_uv_spectra_electronvolt(
    folders: Iterable[str],
    *,
    output_name: str = "orca.out",
    linewidth_ev: float = 0.1,
    step: float = 0.01,
    normalize: bool = False,
    normalize_factor: float = 1.0,
    cmap: Union[str, mcolors.Colormap] = "viridis",
    ax: Optional[plt.Axes] = None,
    transitions: Optional[Iterable[TransitionData]] = None,
    labels: Optional[Iterable[Union[str, Real]]] = None,
    peak_relative_height: float = 0.05,
) -> Tuple[plt.Axes, Tuple[SpectrumData, ...]]:
    """
    Plot broadened spectra in electronvolts using ORCA outputs from multiple folders.

    Parameters
    ----------
    folders:
        Iterable of directories containing ORCA output files.
    output_name:
        Name of the ORCA output file found in each folder.
    linewidth_ev:
        Gaussian FWHM specified directly in electronvolts.
    step:
        Sampling step along the electronvolt axis.
    normalize / normalize_factor:
        Same semantics as :func:`plot_uv_spectra_from_folders`.
    cmap:
        Matplotlib colormap name or instance used for the traces.
    ax:
        Optional matplotlib Axes to draw on.
    transitions:
        Optional iterable of :class:`TransitionData` objects pre-extracted for each
        folder. When provided the ORCA outputs are not re-read.
    labels:
        Optional iterable matching ``folders`` to override legend labels. Numeric
        labels trigger the same colorbar behaviour as
        :func:`plot_uv_spectra_from_folders`.
    peak_relative_height:
        Minimum peak height expressed as a fraction of the maximum intensity within
        each spectrum. Peaks below this relative height are ignored.

    Returns
    -------
    (Axes, tuple[SpectrumData, ...])
        The axes used for plotting and the extracted spectra in plotting order.
    """

    from cats.orca.routines.orca_uv_extractor import extract_uv_transitions

    folder_list = list(folders)
    if not folder_list:
        raise ValueError("folders must contain at least one directory.")

    label_list: Optional[list[str]] = None
    color_values: Optional[np.ndarray] = None
    if labels is not None:
        label_items = list(labels)
        if len(label_items) != len(folder_list):
            raise ValueError("labels must contain the same number of entries as folders.")
        label_list = [str(label) for label in label_items]
        if label_items and all(isinstance(label, Real) for label in label_items):
            numeric_values = np.asarray(label_items, dtype=float)
            if np.isfinite(numeric_values).all():
                color_values = numeric_values.astype(float, copy=False)

    if transitions is not None:
        transition_list = [_ensure_transition_data(item) for item in transitions]
        if len(transition_list) != len(folder_list):
            raise ValueError("transitions must contain the same number of entries as folders.")
    else:
        transition_list = []
        for folder in folder_list:
            path = Path(folder) / output_name
            print("Extracting transitions from:", path)
            transition_list.append(extract_uv_transitions(str(path)))

    width = float(linewidth_ev)
    if width <= 0:
        raise ValueError("Line width must be positive.")
    if step <= 0:
        raise ValueError("Step size must be positive.")
    if normalize_factor <= 0:
        raise ValueError("normalize_factor must be positive.")
    if peak_relative_height <= 0 or peak_relative_height > 1:
        raise ValueError("peak_relative_height must be in the interval (0, 1].")

    spectra: list[SpectrumData] = []
    stick_store: list[Tuple[np.ndarray, np.ndarray]] = []
    min_stick = np.inf
    max_stick = -np.inf

    for idx, folder in enumerate(folder_list):
        transition_data = transition_list[idx]
        energies_cm = transition_data.wavenumbers_cm
        intensities = transition_data.intensities.astype(float, copy=False)

        sticks_x = _wn_to_ev(energies_cm)
        sticks_unit = "eV"

        stick_store.append((sticks_x, intensities))
        min_stick = min(min_stick, float(np.min(sticks_x)))
        max_stick = max(max_stick, float(np.max(sticks_x)))

    if not np.isfinite(min_stick) or not np.isfinite(max_stick):
        raise ValueError("No transition data available for plotting.")

    start = max(0.0, min_stick - width * 3.0)
    stop = max(start + step, max_stick + width * 3.0)
    axis = np.arange(start, stop + step, step, dtype=float)

    if axis.size == 0:
        raise ValueError("Computed axis is empty; adjust step or line width.")

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    elif isinstance(cmap, mcolors.Colormap):
        cmap_obj = cmap
    else:
        raise TypeError("cmap must be a Matplotlib colormap name or Colormap instance.")

    use_colorbar = color_values is not None and color_values.size > 0
    colorbar_norm: Optional[mcolors.Normalize] = None
    colorbar_ticks: Tuple[float, ...] = ()

    if use_colorbar:
        vmin = float(np.min(color_values))
        vmax = float(np.max(color_values))

        if np.isclose(vmin, vmax):
            span = max(abs(vmin), 1.0)
            if span == 0.0:
                span = 1.0
            half_span = span * 0.5
            colorbar_norm = mcolors.Normalize(vmin=vmin - half_span, vmax=vmax + half_span)
            normalized = np.full(color_values.shape, 0.5, dtype=float)
            colorbar_ticks = (vmin, vmax)
        else:
            colorbar_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            normalized = colorbar_norm(color_values)
            colorbar_ticks = (vmin, vmax)

        colors = cmap_obj(normalized)
    else:
        colors = cmap_obj(np.linspace(0.0, 1.0, len(stick_store)))

    if ax is None:
        _, ax = plt.subplots()

    for idx, (sticks_x, intensities) in enumerate(stick_store):
        print("Processing Gaussian broadening for folder:", folder_list[idx])
        profile = _gaussian_broadening(sticks_x, intensities, axis, width)

        if normalize and profile.size:
            peak = profile.max()
            if peak > 0:
                profile = profile / peak * normalize_factor

        spectrum = SpectrumData(
            x=axis,
            y=profile,
            sticks_x=sticks_x,
            sticks_y=intensities,
            x_unit="eV",
            sticks_unit=sticks_unit,
        )
        spectra.append(spectrum)

        if use_colorbar:
            line, = ax.plot(axis, profile, color=colors[idx])
            line_color = colors[idx]
        else:
            if label_list is not None:
                label = label_list[idx]
            else:
                label = Path(folder_list[idx]).name or str(folder_list[idx])
            line, = ax.plot(axis, profile, color=colors[idx], label=label)
            line_color = line.get_color()

        peak_indices = _find_profile_peaks(profile, peak_relative_height)
        for p_idx in peak_indices:
            peak_x = float(axis[p_idx])
            peak_y = float(profile[p_idx])
            ax.plot(peak_x, peak_y, marker="o", color=line_color)
            ax.text(
                peak_x,
                peak_y,
                f"{peak_x:.2f} eV",
                fontsize=8,
                ha="center",
                va="bottom",
                color=line_color,
            )

    ax.set_xlabel("Energy / eV")
    ax.set_ylabel("Intensity")

    if use_colorbar and colorbar_norm is not None:
        mappable = plt.cm.ScalarMappable(norm=colorbar_norm, cmap=cmap_obj)
        mappable.set_array([])
        cbar = plt.colorbar(mappable, ax=ax)
        if colorbar_ticks:
            cbar.set_ticks(colorbar_ticks)
            cbar.set_ticklabels([f"{tick:g}" for tick in colorbar_ticks])
    else:
        ax.legend()

    return ax, tuple(spectra)


def plot_uv_spectra_stacked(
    folders: Iterable[str],
    *,
    output_name: str = "orca.out",
    units: Optional[str] = None,
    linewidth_nm: float = 20.0,
    linewidth_wn: float = 1000.0,
    step: float = 1.0,
    normalize: bool = False,
    normalize_factor: float = 1.0,
    cmap: Union[str, mcolors.Colormap] = "viridis",
    ax: Optional[plt.Axes] = None,
    transitions: Optional[Iterable[TransitionData]] = None,
    labels: Optional[Iterable[Real]] = None,
    offset_scale: float = 1.0,
    stack_axis_label: str = "Stack value",
) -> Tuple[plt.Axes, plt.Axes, Tuple[SpectrumData, ...]]:
    """
    Plot UV/Vis spectra stacked vertically using numeric labels as offsets.

    Parameters
    ----------
    folders:
        Iterable of directories containing ORCA output files.
    output_name:
        Name of the ORCA output file found in each folder.
    units:
        Output axis units: ``"nm"`` or ``"cm-1"``. Falls back to the legacy
        ``nm_plot`` behaviour (wavelengths).
    linewidth_nm / linewidth_wn:
        Gaussian FWHM for wavelength or wavenumber spectra respectively.
    step:
        Sampling step applied to the x-axis.
    normalize:
        When ``True`` each spectrum is scaled so that ``max(y)`` equals
        ``normalize_factor``.
    normalize_factor:
        Target peak height when ``normalize`` is enabled.
    cmap:
        Matplotlib colormap name or :class:`matplotlib.colors.Colormap` used for the
        spectrum colors.
    ax:
        Optional matplotlib Axes to draw on. When omitted a new figure and axes
        are created.
    transitions:
        Optional iterable of :class:`TransitionData` objects pre-extracted for each
        folder. When provided the ORCA outputs are not re-read.
    labels:
        Iterable of numeric values matching ``folders``. These values drive the
        colormap mapping, the stacking offsets and the secondary y-axis tick labels.
    offset_scale:
        Multiplicative factor applied to ``labels`` to control the vertical spacing
        between stacked spectra.
    stack_axis_label:
        Label used for the secondary y-axis and colorbar.

    Returns
    -------
    (Axes, Axes, tuple[SpectrumData, ...])
        The axes used for plotting (primary intensity axis, secondary stack axis) and
        the extracted spectra in plotting order.
    """

    from cats.orca.routines.orca_uv_extractor import extract_uv_transitions

    folder_list = list(folders)
    if not folder_list:
        raise ValueError("folders must contain at least one directory.")
    if labels is None:
        raise ValueError("labels must be provided to create a stacked plot.")

    label_items = list(labels)
    if len(label_items) != len(folder_list):
        raise ValueError("labels must contain the same number of entries as folders.")
    try:
        label_values = np.asarray(label_items, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("labels must be numeric values.") from exc
    if not np.isfinite(label_values).all():
        raise ValueError("labels must contain finite numeric values.")

    offset_scale = float(offset_scale)
    if np.isclose(offset_scale, 0.0):
        raise ValueError("offset_scale must be non-zero.")

    offsets = label_values * offset_scale

    if transitions is not None:
        transition_list = [_ensure_transition_data(item) for item in transitions]
        if len(transition_list) != len(folder_list):
            raise ValueError("transitions must contain the same number of entries as folders.")
    else:
        transition_list = []
        for folder in folder_list:
            path = Path(folder) / output_name
            print("Extracting transitions from:", path)
            transition_list.append(extract_uv_transitions(str(path)))

    use_nm, axis_unit = _resolve_units(units, True)
    width = float(linewidth_nm) if use_nm else float(linewidth_wn)

    if width <= 0:
        raise ValueError("Line width must be positive.")
    if step <= 0:
        raise ValueError("Step size must be positive.")
    if normalize_factor <= 0:
        raise ValueError("normalize_factor must be positive.")

    spectra: list[SpectrumData] = []
    stick_store: list[Tuple[np.ndarray, np.ndarray]] = []
    max_stick = 0.0

    for idx, folder in enumerate(folder_list):
        transition_data = transition_list[idx]
        energies_cm = transition_data.wavenumbers_cm
        intensities = transition_data.intensities.astype(float, copy=False)

        if use_nm:
            if transition_data.wavelengths_nm is not None:
                sticks_x = transition_data.wavelengths_nm.astype(float, copy=False)
            else:
                sticks_x = _wn_to_nm(energies_cm)
            sticks_unit = "nm"
        else:
            sticks_x = energies_cm.astype(float, copy=False)
            sticks_unit = "cm^-1"

        stick_store.append((sticks_x, intensities))
        max_stick = max(max_stick, float(np.max(sticks_x)))

    stop = max_stick + width * 3.0
    if stop <= 0.0:
        stop = width * 3.0

    axis = np.arange(0.0, stop + step, step, dtype=float)

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    elif isinstance(cmap, mcolors.Colormap):
        cmap_obj = cmap
    else:
        raise TypeError("cmap must be a Matplotlib colormap name or Colormap instance.")

    vmin = float(np.min(label_values))
    vmax = float(np.max(label_values))
    if np.isclose(vmin, vmax):
        span = max(abs(vmin), 1.0)
        if span == 0.0:
            span = 1.0
        half_span = span * 0.5
        colorbar_norm = mcolors.Normalize(vmin=vmin - half_span, vmax=vmax + half_span)
    else:
        colorbar_norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    normalized = colorbar_norm(label_values)
    colors = cmap_obj(normalized)

    if ax is None:
        _, ax = plt.subplots()

    min_y = np.inf
    max_y = -np.inf

    for idx, (sticks_x, intensities) in enumerate(stick_store):
        print("Processing Gaussian broadening for folder:", folder_list[idx])
        profile = _gaussian_broadening(sticks_x, intensities, axis, width)

        if normalize and profile.size:
            peak = profile.max()
            if peak > 0:
                profile = profile / peak * normalize_factor

        spectrum = SpectrumData(
            x=axis,
            y=profile,
            sticks_x=sticks_x,
            sticks_y=intensities,
            x_unit=axis_unit,
            sticks_unit=sticks_unit,
        )
        spectra.append(spectrum)

        stacked_profile = profile + offsets[idx]
        min_y = min(min_y, float(np.min(stacked_profile)))
        max_y = max(max_y, float(np.max(stacked_profile)))

        ax.plot(axis, stacked_profile, color=colors[idx])

    if not np.isfinite(min_y) or not np.isfinite(max_y):
        raise ValueError("No data available for plotting.")

    ax.set_xlim(axis[0], axis[-1])
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel("Wavelength / nm" if use_nm else "Energy / cm^-1")
    ax.set_ylabel("Intensity")

    mappable = plt.cm.ScalarMappable(norm=colorbar_norm, cmap=cmap_obj)
    mappable.set_array([])
    cbar = plt.colorbar(mappable, ax=ax)
    tick_values = np.unique(label_values)
    if tick_values.size:
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(_format_colorbar_ticks(tick_values, axis_label=stack_axis_label))
    cbar.set_label(stack_axis_label)

    secondary = ax.twinx()
    secondary.set_yticks([])
    secondary.set_ylabel("")
    secondary.tick_params(right=False, labelright=False)
    secondary.spines["right"].set_visible(False)

    return ax, secondary, tuple(spectra)


def _ensure_transition_data(
    item: Union[
        TransitionData,
        Tuple[Iterable[float], Iterable[float]],
        Tuple[Iterable[float], Iterable[float], Optional[Iterable[float]]],
    ]
) -> TransitionData:
    if isinstance(item, TransitionData):
        return item

    if isinstance(item, (tuple, list)) and len(item) >= 2:
        wavenumbers = np.asarray(item[0], dtype=float)
        intensities = np.asarray(item[1], dtype=float)
        if wavenumbers.shape != intensities.shape:
            raise ValueError("wavenumbers and intensities must have matching shapes.")

        wavelengths_arr: Optional[np.ndarray] = None
        if len(item) > 2 and item[2] is not None:
            wavelengths_arr = np.asarray(item[2], dtype=float)
            if wavelengths_arr.shape != wavenumbers.shape:
                raise ValueError("wavelength data must match the wavenumber shape.")

        return TransitionData(
            wavenumbers_cm=wavenumbers,
            intensities=intensities,
            wavelengths_nm=wavelengths_arr,
        )

    raise TypeError(
        "transitions entries must be TransitionData instances or tuple-like "
        "(wavenumbers, intensities[, wavelengths]) sequences."
    )


def _build_axis(
    sticks_x: np.ndarray,
    width: float,
    startx: Optional[float],
    endx: Optional[float],
    step: float,
) -> np.ndarray:
    start = 0.0 if startx is None else float(startx)
    stop = float(np.max(sticks_x) + width * 3.0) if endx is None else float(endx)

    if stop <= start:
        raise ValueError("endx must be greater than startx.")

    # Include the end point by adding one extra step.
    return np.arange(start, stop + step, step, dtype=float)


def _gaussian_broadening(
    sticks_x: np.ndarray,
    intensities: np.ndarray,
    axis: np.ndarray,
    width: float,
) -> np.ndarray:
    """Compute Gaussian broadening where ``width`` is the full width at half maximum."""

    diff = (sticks_x[:, np.newaxis] - axis[np.newaxis, :]) / width
    gaussians = np.exp(-_GAUSS_FACTOR * diff**2)
    return np.sum(intensities[:, np.newaxis] * gaussians, axis=0)


def _wn_to_nm(energies_cm: np.ndarray) -> np.ndarray:
    if np.any(energies_cm == 0):
        raise ValueError("Encountered zero wavenumber while converting to nm.")
    return (1.0e7 / energies_cm).astype(float)


def _wn_to_ev(energies_cm: np.ndarray) -> np.ndarray:
    return np.asarray(energies_cm, dtype=float) * _EV_PER_WN


def _find_profile_peaks(profile: np.ndarray, rel_height: float) -> np.ndarray:
    if profile.size == 0:
        return np.empty(0, dtype=int)

    max_val = float(np.max(profile))
    if max_val <= 0:
        return np.empty(0, dtype=int)

    threshold = max_val * float(rel_height)

    peaks: list[int] = []
    values = profile
    last_idx = values.size - 1

    if last_idx == 0:
        if values[0] >= threshold:
            return np.asarray([0], dtype=int)
        return np.empty(0, dtype=int)

    if values[0] >= values[1] and values[0] >= threshold:
        peaks.append(0)

    for idx in range(1, last_idx):
        val = values[idx]
        if val < threshold:
            continue
        if val >= values[idx - 1] and val >= values[idx + 1]:
            peaks.append(idx)

    if values[last_idx] >= values[last_idx - 1] and values[last_idx] >= threshold:
        peaks.append(last_idx)

    if not peaks:
        return np.empty(0, dtype=int)

    return np.asarray(peaks, dtype=int)


def _resolve_units(units: Optional[str], legacy_flag: bool) -> Tuple[bool, str]:
    if units is None:
        return legacy_flag, "nm" if legacy_flag else "cm^-1"

    key = units.strip().lower().replace("_", "").replace(" ", "")
    if key in {"nm", "nanometer", "nanometers", "wavelength"}:
        return True, "nm"
    if key in {"cm-1", "cm^-1", "wavenumber", "wavenumbers", "wn"}:
        return False, "cm^-1"

    raise ValueError("units must be either 'nm' or 'cm-1'.")
