import numpy as np


def get_uvvis_plot(
    uvvis_data_curve="spectrum_curve.txt",
    fragments={},
    relative_intensity_threshold=0.1,
    min_wavelength_nm=300.0,
    max_wavelength_nm=800.0,
):
    """Requirements:
    CT_analysis_clean.txt
    orca.molden.input
    orca.out
    spectrum_curve.txt
    spectrum_line.txt

    # TODO: label points of interest automatically
    """

    from cats.multiwfn.tools import get_uvvis_data
    import pandas as pd
    import numpy as np
    from matplotlib import pyplot as plt
    from cats.multiwfn.tools import charge_transfer_routine

    data = get_uvvis_data()
    if min_wavelength_nm is not None:
        data = data[data["wavelength"] >= float(min_wavelength_nm)]
    if max_wavelength_nm is not None:
        data = data[data["wavelength"] <= float(max_wavelength_nm)]
    if data.empty:
        raise ValueError(
            "No UV-vis transitions remain after wavelength filtering. "
            "Adjust min_wavelength_nm/max_wavelength_nm."
        )
    data["intensity_normal"] = data["intensity"] / data["intensity"].max()
    intensity_threshold = (
        float(relative_intensity_threshold) * float(data["intensity"].max())
    )
    excitation_numbers = (
        data.loc[data["intensity"] >= intensity_threshold, "excitation"]
        .astype(int)
        .tolist()
    )
    print(f"Performing analysis for {len(excitation_numbers)} intensity excitations.")

    # plotting
    curve = pd.read_csv(uvvis_data_curve, header=None, index_col=None)

    values = [str(val[0]).split() for val in curve.values]
    x = np.asarray([float(val[0]) for val in values], dtype=float)
    y = np.asarray([float(val[1]) for val in values], dtype=float)
    if min_wavelength_nm is not None:
        mask = x >= float(min_wavelength_nm)
        x = x[mask]
        y = y[mask]
    if max_wavelength_nm is not None:
        mask = x <= float(max_wavelength_nm)
        x = x[mask]
        y = y[mask]
    if x.size == 0 or y.size == 0:
        raise ValueError(
            "No UV-vis curve points remain after wavelength filtering. "
            "Adjust min_wavelength_nm/max_wavelength_nm."
        )
    max_y = float(np.max(y))
    y_normal = y / max_y if max_y > 0.0 else y

    fig, ax1 = plt.subplots()
    fontsize = 18
    # Plot the first set of data
    ax1.plot(x, y_normal, color="black")
    ax1.set_xlabel("wavelength (nm)", fontsize=fontsize)
    ax1.set_ylabel("Absorbance (a.u.)", color="black", fontsize=fontsize)

    # Create a second y-axis sharing the same x-axis
    ax2 = ax1.twinx()
    ax2.bar(data["wavelength"], data["intensity_normal"], color="blue", alpha=0.5)
    ax2.set_ylabel("Oscillator Strength", color="blue", fontsize=fontsize)

    ax1.tick_params(axis="x", which="both", labelsize=fontsize)

    # Remove both y-ticks and y-tick labels
    ax1.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        right=False,
        labelright=False,
    )
    ax2.tick_params(
        axis="y",
        which="both",
        left=False,
        labelleft=False,
        right=False,
        labelright=False,
    )

    left_limit = float(min_wavelength_nm) if min_wavelength_nm is not None else float(np.min(x))
    right_limit = float(max_wavelength_nm) if max_wavelength_nm is not None else float(np.max(x))
    if right_limit > left_limit:
        plt.xlim(left=left_limit, right=right_limit)
    plt.tight_layout()

    if len(fragments) > 1 and excitation_numbers:
        results = charge_transfer_routine(
            fragments, excitation_numbers, overwrite=False
        )
        df = pd.DataFrame.from_dict(results)
        df = df.transpose()
        df.reset_index(inplace=True)
        df.rename(columns={"index": "excitation"}, inplace=True)

        df["excitation"] = df["excitation"].astype("Int64")
        data["excitation"] = data["excitation"].astype("Int64")
        df_joined = pd.merge(data, df, on="excitation")

        # Filter out low intensity peaks
        # df_joined = df_joined[df_joined["intensity_normal"] > 0.25]

        # cast the strings into a number
        df_joined["wavelength"] = df_joined["wavelength"].astype(float)
        df_joined["net_charge"] = df_joined["net_charge"].astype(float)

        df_joined = df_joined.sort_values(by="net_charge", ascending=False)

        # Highlight and annotate peaks with maximum and minimum net charge
        max_net_charge_peak = df_joined.loc[df_joined["net_charge"].idxmax()]
        min_net_charge_peak = df_joined.loc[df_joined["net_charge"].idxmin()]

        # Rounding and muiplying 100x to turn into CT % is required
        # Only plot if CT % mangitude is greater than 0.5%
        if max_net_charge_peak["net_charge"] > 0.005:
            annotation_max = f"{max_net_charge_peak['wavelength'].round(1)} nm\nCT {round(100 * max_net_charge_peak['net_charge'], 1)} %"
            print(f"Annotating: {annotation_max}")
            ax2.scatter(
                max_net_charge_peak["wavelength"],
                max_net_charge_peak["intensity_normal"],
                color="black",
            )
            ax2.annotate(
                annotation_max,
                (
                    max_net_charge_peak["wavelength"],
                    max_net_charge_peak["intensity_normal"],
                ),
                xytext=(0.2, 0.7),
                textcoords="axes fraction",
                arrowprops=dict(facecolor="black", shrink=0.03),
                # textcoords="offset points",
                # xytext=(0,-10),
                ha="center",
                # va="top",
                color="black",
                fontsize=14,
            )

        if min_net_charge_peak["net_charge"] < -0.005:
            annotation_min = f"{min_net_charge_peak['wavelength'].round(1)} nm\nCT {round(100 * min_net_charge_peak['net_charge'], 1)} %"
            print(f"Annotating: {annotation_min}")

            ax2.scatter(
                min_net_charge_peak["wavelength"],
                min_net_charge_peak["intensity_normal"],
                color="black",
            )
            ax2.annotate(
                annotation_min,
                (
                    min_net_charge_peak["wavelength"],
                    min_net_charge_peak["intensity_normal"],
                ),
                xytext=(0.8, 0.7),
                textcoords="axes fraction",
                arrowprops=dict(facecolor="black", shrink=0.03),
                # textcoords="offset points",
                # xytext=(0,-10),
                ha="center",
                # va='top',
                color="black",
                fontsize=14,
            )

    return plt


def gaussian(x, amplitude, mean, fwhm):
    """Gaussian function for curve fitting as per 3.12.1 in Multiwfn manual.
    Parameters:
    x : array-like
        The input values where the Gaussian function is evaluated.
    amplitude : float
        The height of the peak of the Gaussian.
    mean : float
        The position of the center of the peak.
    fwhm : float
        Full width at half maximum of the Gaussian peak.
    Returns:
        y : array-like
        The values of the Gaussian function evaluated at x."""

    # Calculate sigma from FWHM
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))

    # Calculate and return Gaussian values
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * sigma**2))


# Function for the sum of two Gaussians
def calculate_gaussian_area(amplitude, fwhm):
    """Calculates the area under a Gaussian curve.
    Area = amplitude * sigma * sqrt(2*pi)
    sigma = fwhm / (2 * sqrt(2 * log(2)))
    Area = amplitude * fwhm * (sqrt(pi) / (2 * sqrt(log(2))))
    """
    if (
        fwhm <= 0 or amplitude == 0
    ):  # Area is 0 if FWHM is not positive or amplitude is zero
        return 0.0
    return amplitude * fwhm * (np.sqrt(np.pi) / (2 * np.sqrt(np.log(2))))


def double_gaussian(x, amplitude1, mean1, fwhm1, amplitude2, mean2, fwhm2):
    """Sum of two Gaussian functions."""
    return gaussian(x, amplitude1, mean1, fwhm1) + gaussian(x, amplitude2, mean2, fwhm2)


def triple_gaussian(
    x, amplitude1, mean1, fwhm1, amplitude2, mean2, fwhm2, amplitude3, mean3, fwhm3
):
    """Sum of three Gaussian functions."""
    return (
        gaussian(x, amplitude1, mean1, fwhm1)
        + gaussian(x, amplitude2, mean2, fwhm2)
        + gaussian(x, amplitude3, mean3, fwhm3)
    )


def quadruple_gaussian(
    x,
    amplitude1,
    mean1,
    fwhm1,
    amplitude2,
    mean2,
    fwhm2,
    amplitude3,
    mean3,
    fwhm3,
    amplitude4,
    mean4,
    fwhm4,
):
    """Sum of four Gaussian functions."""
    return (
        gaussian(x, amplitude1, mean1, fwhm1)
        + gaussian(x, amplitude2, mean2, fwhm2)
        + gaussian(x, amplitude3, mean3, fwhm3)
        + gaussian(x, amplitude4, mean4, fwhm4)
    )


def plot_charge_transfer_spectrum(
    folder,
    use_eV=False,
    min_wavelength_nm=300.0,
    max_wavelength_nm=800.0,
    min_peak_intensity_fraction=0.25,
    min_peak_distance_points=10,  # Adjusted default
    min_peak_width_points=3,  # New parameter for find_peaks
    min_fwhm_abs=5,  # New parameter for FWHM lower bound (e.g., 5 nm)
    max_gaussians_per_spectrum=4,  # Max Gaussians for dynamic fit
    verbose=False,  # Verbosity for fitting process
    plot_individual_components=True,
    nroots=None,  # number of roots to consider from Multiwfn output
):  # Whether to plot individual fitted Gaussian components
    """Relies on data obtained from Multiwfn using the following steps:
    # TODO:
    # 1. Use Multiwfn to calculate the charge transfer spectrum
    #
    # Parameters:
    # folder : str
    #     The folder where the CT_multiple folder is located.
    # use_eV : bool
    #     If True, the energy will be plotted in eV, otherwise in nm.
    # min_peak_intensity_fraction : float
    #     Minimum height of peaks found by find_peaks, as a fraction of the maximum intensity.
    # min_peak_distance_points : int
    #     Minimum horizontal distance (in data points) between neighboring peaks for find_peaks.
    # min_peak_width_points : int
    #     Minimum width of peaks (in data points) for find_peaks.
    # min_fwhm_abs : float
    #     Absolute minimum FWHM value allowed during curve fitting (in nm or eV).
    # max_gaussians_per_spectrum : int
    #     Maximum number of Gaussians to attempt to fit dynamically for each spectrum.
    # verbose : bool
    #     If True, prints detailed information about the fitting process.
    # plot_individual_components : bool
    #     If True, plots the individual Gaussian components for multi-peak fits."""

    # Helper function for N Gaussians, defined once
    def model_n_gaussians_local_func(x_model, *params_model):
        total_y_model = np.zeros_like(x_model, dtype=float)
        for i_model in range(len(params_model) // 3):
            total_y_model += gaussian(
                x_model,
                params_model[i_model * 3],
                params_model[i_model * 3 + 1],
                params_model[i_model * 3 + 2],
            )
        return total_y_model

    def _extract_energy_intensity_columns(temp_df, header_hint):
        candidate_columns = []
        if header_hint is not None:
            candidate_columns.append(header_hint)
        candidate_columns.extend(list(temp_df.columns))

        seen_columns = set()
        for column_name in candidate_columns:
            if column_name in seen_columns or column_name not in temp_df.columns:
                continue
            seen_columns.add(column_name)
            energies = []
            intensities = []
            for value in temp_df[column_name].values:
                tokens = re.sub(" +", " ", str(value).strip()).split(" ")
                if len(tokens) < 2:
                    continue
                try:
                    energies.append(float(tokens[0]))
                    intensities.append(float(tokens[1]))
                except ValueError:
                    continue
            if energies:
                return np.array(energies, dtype=float), np.array(intensities, dtype=float)
        return None, None

    import re
    from pathlib import Path
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.signal import find_peaks
    import numpy as np
    from matplotlib import colormaps

    # Parse CT_multiple manifest entries robustly
    manifest_file = Path(folder) / "CT_multiple" / "CT_multiple.txt"
    manifest_entries: list[tuple[str, str]] = []
    if manifest_file.is_file():
        raw_text = manifest_file.read_text(encoding="utf-8", errors="replace")
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            match = re.match(r'^["\']?([^"\']+\.txt)["\']?\s+(.+)$', line)
            if match:
                manifest_entries.append((match.group(1).strip(), match.group(2).strip()))
            elif " " in line:
                parts = line.split(" ", 1)
                rel_path = parts[0].strip('"\'')
                if rel_path.endswith(".txt"):
                    manifest_entries.append((rel_path, parts[1].strip()))

    df = pd.DataFrame()

    for i, (rel_path, prop_name) in enumerate(manifest_entries):
        file_path = Path(folder) / rel_path
        if not file_path.is_file():
            file_path = Path(folder) / "CT_multiple" / Path(rel_path).name
        if not file_path.is_file():
            if verbose:
                print(f"Skipping {file_path}: file not found.")
            continue

        temp_df = pd.read_csv(
            file_path,
            header=0,
            index_col=None,
            encoding="utf-8-sig",
            encoding_errors="replace",
        )
        header_hint = (
            f"         {int(nroots)}           1" if nroots is not None else None
        )
        energies, intensities = _extract_energy_intensity_columns(temp_df, header_hint)
        if energies is None or intensities is None:
            if verbose:
                print(
                    f"Skipping {file_path}: no parseable energy/intensity column found."
                )
            continue
        temp_df = pd.DataFrame(
            {
                "energy": energies,
                "intensity": 1000.0 * intensities,  # Scale intensity for better visibility
            }
        )

        if "energy" not in df:
            df["energy"] = temp_df["energy"]

        df[prop_name] = temp_df["intensity"].astype(float)

    if df.empty or "energy" not in df.columns:
        raise ValueError(
            f"No CT spectra could be parsed from {folder}/CT_multiple/ entries."
        )

    # Adjust energy units if necessary
    if not use_eV:
        # Convert energy from eV to nm if needed
        df["energy"] = 1239.84193 / df["energy"].astype(float)

    # Restrict analysis/fitting window.
    if not use_eV:
        if min_wavelength_nm is not None:
            df = df[df["energy"] >= float(min_wavelength_nm)]
        if max_wavelength_nm is not None:
            df = df[df["energy"] <= float(max_wavelength_nm)]
    else:
        min_energy_eV = None
        max_energy_eV = None
        if max_wavelength_nm is not None and float(max_wavelength_nm) > 0.0:
            min_energy_eV = 1239.84193 / float(max_wavelength_nm)
        if min_wavelength_nm is not None and float(min_wavelength_nm) > 0.0:
            max_energy_eV = 1239.84193 / float(min_wavelength_nm)
        if min_energy_eV is not None:
            df = df[df["energy"] >= float(min_energy_eV)]
        if max_energy_eV is not None:
            df = df[df["energy"] <= float(max_energy_eV)]
    if df.empty:
        raise ValueError(
            "No CT data remains after wavelength filtering. "
            "Adjust min_wavelength_nm/max_wavelength_nm."
        )

    # Assuming 'energy' is the first column (index 0)
    x_column = df.columns[0]
    y_columns = [
        name for name in df.columns if "transfer" in name or "UV" in name
    ]  # All columns except the first one

    # --- STAGE 1: FIT ALL SPECTRA AND COLLECT DATA ---
    all_peaks_data_collected = []
    fitted_parameters_per_spectrum = {}

    if verbose:
        print("\n--- Stage 1: Fitting all spectra ---")
    for y_column_name in y_columns:
        # Extract energy and intensity data
        temp_df = df[[x_column, y_column_name]].dropna()
        energy_data = (
            temp_df[x_column].astype(float).values
        )  # Use numpy arrays for scipy
        intensity_data = temp_df[y_column_name].astype(float).values
        if energy_data.size > 1:
            sort_idx = np.argsort(energy_data)
            energy_data = energy_data[sort_idx]
            intensity_data = intensity_data[sort_idx]

        if energy_data.size == 0 or intensity_data.size == 0:
            if verbose:
                print(
                    f"Skipping {y_column_name} due to empty data after filtering/dropna."
                )
            fitted_parameters_per_spectrum[y_column_name] = {
                "popt": None,
                "num_gaussians": 0,
                "peak_details": [],
            }
            continue

        fitted_intensity_plot = np.zeros_like(energy_data)
        num_fitted_gaussians_plot = 0
        peak_details_for_analysis_list = []

        # --- Fitting Logic ---
        # --- Dynamic Multi-Gaussian Fitting (now the only fitting path) ---
        if verbose:
            print(f"Attempting dynamic multi-Gaussian fit for {y_column_name}...")
        y_max_val = np.max(intensity_data) if intensity_data.size > 0 else 1.0

        fp_height_abs = (
            np.mean(intensity_data) * min_peak_intensity_fraction
            if intensity_data.size > 0
            else 0.01
        )
        if intensity_data.size > 0 and fp_height_abs <= 0:
            fp_height_abs = 0.01 * y_max_val
        if fp_height_abs == 0 and y_max_val > 0:
            fp_height_abs = y_max_val * 0.05

        fp_config = {
            "height": fp_height_abs,
            "distance": min_peak_distance_points,
            "width": min_peak_width_points,
            "prominence": fp_height_abs * 0.5,  # Prominence as half of height threshold
        }
        try:
            peak_indices, properties = find_peaks(intensity_data, **fp_config)
        except Exception as e:
            if verbose:
                print(f"Error during find_peaks for {y_column_name}: {e}")
            peak_indices, properties = np.array([]), {}

        initial_guesses_dyn = []
        num_gaussians_to_try = 0

        if len(peak_indices) == 0:
            if verbose:
                print(
                    f"No peaks found by find_peaks for {y_column_name}. Attempting a single generic Gaussian."
                )
            num_gaussians_to_try = 1
            std_dev_e = np.std(energy_data) if energy_data.size > 1 else min_fwhm_abs
            initial_guesses_dyn = [
                y_max_val,
                np.mean(energy_data) if energy_data.size > 0 else 0,
                max(std_dev_e, min_fwhm_abs),
            ]
        else:
            sort_key = (
                "prominences"
                if "prominences" in properties and np.any(properties.get("prominences"))
                else "peak_heights"
            )
            if (
                sort_key in properties
                and properties.get(sort_key) is not None
                and len(properties[sort_key]) == len(peak_indices)
            ):
                sorted_idx = np.argsort(properties[sort_key])[::-1]
                peak_indices = peak_indices[sorted_idx]
                for prop_key in properties:
                    if properties[prop_key] is not None and len(
                        properties[prop_key]
                    ) == len(sorted_idx):
                        properties[prop_key] = properties[prop_key][sorted_idx]

            num_gaussians_to_try = min(len(peak_indices), max_gaussians_per_spectrum)
            if num_gaussians_to_try == 0:
                if verbose:
                    print(
                        f"No suitable peaks to fit for {y_column_name} after filtering."
                    )
            else:
                avg_x_spacing = (
                    np.mean(np.diff(np.sort(energy_data)))
                    if energy_data.size > 1
                    else 1.0
                )
                for i in range(num_gaussians_to_try):
                    amp_g = intensity_data[peak_indices[i]]
                    mean_g = energy_data[peak_indices[i]]
                    width_prop = properties.get("widths", [])
                    fwhm_g_prop = (
                        width_prop[i] * avg_x_spacing
                        if i < len(width_prop)
                        else (
                            np.std(energy_data) / num_gaussians_to_try
                            if energy_data.size > 1
                            else min_fwhm_abs
                        )
                    )
                    fwhm_g = max(fwhm_g_prop, min_fwhm_abs)
                    initial_guesses_dyn.extend([amp_g, mean_g, fwhm_g])
                if verbose:
                    print(
                        f"Attempting to fit {num_gaussians_to_try} Gaussians for {y_column_name} with initial guesses: {initial_guesses_dyn}"
                    )

        if num_gaussians_to_try > 0:
            lower_b, upper_b = [], []
            e_min, e_max = (
                (np.min(energy_data), np.max(energy_data))
                if energy_data.size > 0
                else (0, 1)
            )
            e_range = max(e_max - e_min, min_fwhm_abs * 2)  # Ensure range is sensible
            if e_range <= 0:
                e_range = min_fwhm_abs * 10

            for _ in range(num_gaussians_to_try):
                lower_b.extend([0, e_min, min_fwhm_abs])
                upper_b.extend([y_max_val * 1.5, e_max, e_range])

            try:
                popt_dyn, _ = curve_fit(
                    model_n_gaussians_local_func,
                    energy_data,
                    intensity_data,
                    p0=initial_guesses_dyn,
                    bounds=(lower_b, upper_b),
                    maxfev=3000 * num_gaussians_to_try**2,
                )
                # fitted_intensity_plot = model_n_gaussians_local_func(energy_data, *popt_dyn) # Not needed here, will be reconstructed for plotting
                for i_peak in range(num_gaussians_to_try):
                    amp, mean_val, fwhm_val = popt_dyn[i_peak * 3 : i_peak * 3 + 3]
                    area = calculate_gaussian_area(amp, fwhm_val)
                    peak_details_for_analysis_list.append(
                        {
                            "amplitude": amp,
                            "mean": mean_val,
                            "fwhm": fwhm_val,
                            "area": area,
                            "component_index": i_peak + 1,
                        }
                    )
                num_fitted_gaussians_plot = num_gaussians_to_try
                fitted_parameters_per_spectrum[y_column_name] = {
                    "popt": popt_dyn,
                    "num_gaussians": num_fitted_gaussians_plot,
                    "peak_details": list(peak_details_for_analysis_list),
                }
                if verbose:
                    print(
                        f"Successfully fitted {num_fitted_gaussians_plot} Gaussians for {y_column_name}."
                    )
            except (RuntimeError, ValueError) as e:
                fitted_parameters_per_spectrum[y_column_name] = {
                    "popt": None,
                    "num_gaussians": 0,
                    "peak_details": [],
                }
                if verbose:
                    print(
                        f"Curve_fit failed for {num_gaussians_to_try} Gaussians for {y_column_name}: {e}"
                    )
        else:  # num_gaussians_to_try was 0
            fitted_parameters_per_spectrum[y_column_name] = {
                "popt": None,
                "num_gaussians": 0,
                "peak_details": [],
            }

        # Collect for all_peaks_data_collected (used for analysis table and finding ref UV)
        if num_fitted_gaussians_plot > 0:
            for p_info in peak_details_for_analysis_list:
                all_peaks_data_collected.append(
                    {
                        "column": y_column_name,
                        "type": "UV" if "UV" in y_column_name else "transfer",
                        "mean": p_info["mean"],
                        "amplitude": p_info["amplitude"],
                        "fwhm": p_info["fwhm"],
                        "area": p_info["area"],
                        "peak_component_index": p_info["component_index"],
                    }
                )

    # --- STAGE 2: DETERMINE REFERENCE UV COMPOSITE AREA ---
    if verbose:
        print("\n--- Stage 2: Determining Reference UV Composite Area ---")
    total_ref_uv_composite_area = 0.0
    ref_uv_column_name_for_composite = None

    if all_peaks_data_collected:
        peaks_df_collected = pd.DataFrame(all_peaks_data_collected)
        if not peaks_df_collected.empty and "area" in peaks_df_collected.columns:
            peaks_df_collected["area"] = pd.to_numeric(
                peaks_df_collected["area"], errors="coerce"
            ).fillna(0)

            uv_peaks_collected = peaks_df_collected[
                peaks_df_collected["type"] == "UV"
            ].copy()
            if (
                not uv_peaks_collected.empty
                and not uv_peaks_collected["area"].isnull().all()
                and uv_peaks_collected["area"].max() > 0
            ):
                main_uv_peak_component = uv_peaks_collected.loc[
                    uv_peaks_collected["area"].idxmax()
                ]
                ref_uv_column_name_for_composite = main_uv_peak_component["column"]

                components_of_ref_uv_spec = peaks_df_collected[
                    peaks_df_collected["column"] == ref_uv_column_name_for_composite
                ]
                total_ref_uv_composite_area = components_of_ref_uv_spec["area"].sum()
                if verbose:
                    print(
                        f"Reference UV spectrum for composite area: '{ref_uv_column_name_for_composite}', Total Composite Area: {total_ref_uv_composite_area:.1f}"
                    )
            else:
                if verbose:
                    print(
                        "No suitable UV peaks found to determine reference composite area."
                    )
        else:
            if verbose:
                print("Collected peaks DataFrame is empty or missing 'area' column.")
    else:
        if verbose:
            print("No peak data collected in Stage 1.")

    # --- STAGE 3: PLOT ALL SPECTRA ---
    if verbose:
        print("\n--- Stage 3: Plotting all spectra ---")
    base_fontsize = 20
    plt.figure(figsize=(20, 8))
    if len(y_columns) < 10:
        colors = colormaps.get_cmap("tab10")(
            np.linspace(0, 1, 10)
        )  # Use y_columns for color mapping
    else:
        colors = colormaps.get_cmap("tab20")(
            np.linspace(0, 1, (len(y_columns)))
        )  # Use y_columns for color mapping
    scale_factor = 2.0

    for i_col, y_column_name in enumerate(y_columns):
        temp_df = df[[x_column, y_column_name]].dropna()
        energy_data = temp_df[x_column].astype(float).values
        intensity_data = temp_df[y_column_name].astype(float).values
        if energy_data.size > 1:
            sort_idx = np.argsort(energy_data)
            energy_data = energy_data[sort_idx]
            intensity_data = intensity_data[sort_idx]
        current_color = colors[i_col % len(colors)]
        opacity = 0.1 if "UV" in y_column_name else 0.3

        if energy_data.size == 0:
            continue  # Should have been caught in Stage 1

        # Use a dense plotting grid so fitted curves are rendered smoothly.
        dense_points = max(800, int(energy_data.size) * 6)
        if energy_data.size > 1:
            energy_plot = np.linspace(float(np.min(energy_data)), float(np.max(energy_data)), dense_points)
        else:
            energy_plot = energy_data.copy()

        plt.bar(
            energy_data,
            intensity_data,
            width=1,
            alpha=opacity,
            label=f"{y_column_name} data",
            color=current_color,
        )

        fit_info = fitted_parameters_per_spectrum.get(y_column_name)
        if fit_info and fit_info["num_gaussians"] > 0:
            popt_dyn_plot = fit_info["popt"]
            num_gaussians_for_plot = fit_info["num_gaussians"]
            peak_details_for_plot = fit_info["peak_details"]

            reconstructed_fit_intensity = model_n_gaussians_local_func(
                energy_plot, *popt_dyn_plot
            )
            current_spectrum_total_area = sum(p["area"] for p in peak_details_for_plot)

            composite_label_text = f"{y_column_name} "
            if num_gaussians_for_plot == 1:
                p_single = peak_details_for_plot[0]
                composite_label_text += f'fit ({p_single["mean"]:.0f} nm, FWHM: {p_single["fwhm"]:.1f}, Area: {p_single["area"]:.1f}'
                if "transfer" in y_column_name and total_ref_uv_composite_area > 0:
                    percentage_of_uv = (
                        p_single["area"] / total_ref_uv_composite_area
                    ) * 100.0
                    composite_label_text += f", {percentage_of_uv:.1f}%"
            else:  # Composite of multiple Gaussians
                composite_label_text += (
                    f"composite fit (Area: {current_spectrum_total_area:.1f}"
                )
                if "transfer" in y_column_name and total_ref_uv_composite_area > 0:
                    percentage_of_uv = (
                        current_spectrum_total_area / total_ref_uv_composite_area
                    ) * 100.0
                    composite_label_text += f", {percentage_of_uv:.1f}%"
            composite_label_text += ")"

            plt.plot(
                energy_plot,
                reconstructed_fit_intensity * scale_factor,
                label=composite_label_text,
                color=current_color,
                linewidth=2,
            )

            if plot_individual_components and num_gaussians_for_plot > 1:
                linestyles = ["--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
                for idx, p_info in enumerate(peak_details_for_plot):
                    amp, mean_val, fwhm_val = (
                        p_info["amplitude"],
                        p_info["mean"],
                        p_info["fwhm"],
                    )
                    ind_peak_curve = gaussian(energy_plot, amp, mean_val, fwhm_val)
                    plt.plot(
                        energy_plot,
                        ind_peak_curve * scale_factor,
                        linestyle=linestyles[idx % len(linestyles)],
                        label=f'Comp {p_info["component_index"]} ({mean_val:.0f} nm, Area: {p_info["area"]:.1f})',
                        color=current_color,
                        alpha=0.7,
                        zorder=4,
                    )
        else:  # Fit failed
            fallback_curve = np.interp(energy_plot, energy_data, intensity_data)
            plt.plot(
                energy_plot,
                fallback_curve * scale_factor,
                label=f"{y_column_name} fit (failed)",
                color=current_color,
                linestyle=":",
            )

    if use_eV:
        left_eV = None
        right_eV = None
        if max_wavelength_nm is not None and float(max_wavelength_nm) > 0.0:
            left_eV = 1239.84193 / float(max_wavelength_nm)
        if min_wavelength_nm is not None and float(min_wavelength_nm) > 0.0:
            right_eV = 1239.84193 / float(min_wavelength_nm)
        if left_eV is not None or right_eV is not None:
            plt.xlim(left=left_eV, right=right_eV)
    else:
        left_nm = (
            float(min_wavelength_nm)
            if min_wavelength_nm is not None
            else float(np.min(df[x_column].values))
        )
        right_nm = (
            float(max_wavelength_nm)
            if max_wavelength_nm is not None
            else float(np.max(df[x_column].values))
        )
        if right_nm > left_nm:
            plt.xlim(left=left_nm, right=right_nm)
    if use_eV:
        plt.xlabel("Energy (eV)", fontsize=base_fontsize)
    else:
        plt.xlabel("Wavelength (nm)", fontsize=base_fontsize)

    plt.ylabel(
        "Intensity (arb. unit)", fontsize=base_fontsize
    )  # Or a more general term if appropriate
    plt.title(
        "Charge Transfer Spectrum", fontsize=base_fontsize + 2
    )  # Title slightly larger

    # Apply font size to tick labels
    plt.xticks(fontsize=base_fontsize - 1)
    plt.yticks(fontsize=base_fontsize - 1)

    # Adjust subplot to make room for the legend
    plt.subplots_adjust(right=0.75)  # Adjust this value as needed
    # Place legend to the right of the plot
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=base_fontsize -1)
    plt.legend(loc="upper right", fontsize=base_fontsize - 1)
    plt.grid(True)

    # --- STAGE 4: FINAL ANALYSIS DATAFRAME ---
    if verbose:
        print("\n--- Stage 4: Generating Analysis DataFrame ---")
    analysis_results_df = pd.DataFrame()  # Default empty DataFrame

    if not all_peaks_data_collected:
        print("No peak data was successfully fitted for any spectrum.")
    else:
        peaks_df = pd.DataFrame(all_peaks_data_collected)  # Use the collected data
        uv_peaks = peaks_df[peaks_df["type"] == "UV"].copy()
        transfer_peaks = peaks_df[peaks_df["type"] == "transfer"].copy()

        if uv_peaks.empty:
            print("No UV peaks found/fitted to use as reference.")
        else:
            # Identify the UV peak component with the largest area as the reference
            main_uv_peak_component = uv_peaks.loc[uv_peaks["area"].idxmax()]
            # Use total_ref_uv_composite_area for the table's reference if its column matches
            # For the table, we might still want to report based on the single largest component or the composite.
            # Let's use the single largest component's area for the table as before, unless specified otherwise.
            # If the request implies the table should also use composite area, this part needs adjustment.
            # For now, keeping table reference as single largest UV component area.
            ref_uv_area_for_table = main_uv_peak_component["area"]
            ref_uv_pos_for_table = main_uv_peak_component["mean"]
            ref_uv_col_for_table = main_uv_peak_component["column"]

            if verbose:
                print(
                    f"\nReference for Analysis Table (largest single UV component from '{ref_uv_col_for_table}'):"
                )
                print(
                    f"  Position={main_uv_peak_component['mean']:.2f}, Area={main_uv_peak_component['area']:.2f}, Amplitude={main_uv_peak_component['amplitude']:.3f}, FWHM={main_uv_peak_component['fwhm']:.2f}"
                )
                if ref_uv_column_name_for_composite:
                    print(
                        f"Reference for Plot Labels (composite UV from '{ref_uv_column_name_for_composite}'): Total Area={total_ref_uv_composite_area:.2f}"
                    )

            if transfer_peaks.empty:
                print(
                    "No transfer peaks found/fitted for analysis against UV reference."
                )
            else:
                results_list = []
                for _, t_peak in transfer_peaks.iterrows():
                    fraction = (
                        (t_peak["area"] / ref_uv_area_for_table) * 100.0
                        if ref_uv_area_for_table != 0
                        else 0.0
                    )
                    results_list.append(
                        {
                            "transfer_spectrum_column": t_peak["column"],
                            "transfer_peak_component_index": t_peak[
                                "peak_component_index"
                            ],
                            "transfer_peak_position": t_peak["mean"],
                            "transfer_peak_amplitude": t_peak["amplitude"],
                            "transfer_peak_fwhm": t_peak["fwhm"],
                            "transfer_peak_area": t_peak["area"],
                            "reference_uv_spectrum_column": ref_uv_col_for_table,
                            "reference_uv_peak_position": ref_uv_pos_for_table,
                            "reference_uv_peak_area": ref_uv_area_for_table,
                            "fraction_of_ref_uv_area (%)": fraction,
                        }
                    )
                analysis_results_df = pd.DataFrame(results_list)

    return plt, analysis_results_df


def _autoscale_intensity_axis(axis):
    axis.relim()
    axis.autoscale(enable=True, axis="y")
    axis.margins(y=0.08)
    axis.set_ylim(bottom=0)


def plot_charge_transfer_spectrum_from_folders(preamble, folders, nroots):
    """Convenience function to plot charge transfer spectrum from multiple folders.
    Each folder should contain the necessary data files as per the requirements."""

    assert isinstance(folders, list), "folders should be a list of folder names"

    base_fontsize = 20  # Define base_fontsize here as well

    for folder in folders:
        print(f"\nProcessing folder: {folder}")
        plot_object, results_df = plot_charge_transfer_spectrum(
            rf"{preamble}/{folder}",
            min_peak_intensity_fraction=1,  # Example: peaks must be at least x% mean
            min_peak_distance_points=1,  # Example: peaks must be 15 data points apart
            min_peak_width_points=1,  # Example: peaks must be at least 5 data points wide
            min_fwhm_abs=15,  # Example: minimum FWHM of 10 nm (or eV if use_eV=True)
            max_gaussians_per_spectrum=20,  # Try fitting up to 3 Gaussians
            verbose=False,  # Show fitting details
            plot_individual_components=False,
            nroots=nroots
        )  # Control plotting of individual components

        current_axes = plot_object.gca()
        _autoscale_intensity_axis(current_axes)
        plot_object.savefig(
            rf"{preamble}/{folder}/CT_spectrum_all_peaks.png",
            dpi=300,
            bbox_inches="tight",
        )

        # --- Remove UV-vis plot data for the zoomed graph ---
        # Remove bars associated with UV
        bars_to_remove = []
        for container in current_axes.containers:
            label = container.get_label()
            if label and "UV" in label:
                bars_to_remove.append(container)
        for bar_container in bars_to_remove:
            bar_container.remove()

        # Remove lines associated with UV (e.g., composite fits)
        lines_to_remove = []
        for line_artist in current_axes.get_lines():
            label = line_artist.get_label()
            if label and "UV" in label:
                lines_to_remove.append(line_artist)
        for line_to_remove in lines_to_remove:
            line_to_remove.remove()

        # Regenerate legend after removing artists
        # current_axes.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=base_fontsize -1) # Apply fontsize here too
        current_axes.legend(loc="upper right", fontsize=base_fontsize - 1)

        _autoscale_intensity_axis(current_axes)
        plot_object.savefig(
            rf"{preamble}/{folder}/CT_spectrum_zoom.png", dpi=300, bbox_inches="tight"
        )

        plot_object.xlim(right=2000)  # Modifies the current axes of the plot returned
        plot_object.savefig(
            rf"{preamble}/{folder}/CT_spectrum_zoom_IR.png", dpi=300, bbox_inches="tight"
        )

        if not results_df.empty:
            print(f"\n--- Peak Analysis Results for folder: {folder} ---")
            print(results_df)
            results_df.to_csv(
                rf"{preamble}/{folder}/CT_spectrum_analysis_results.csv", index=False
            )
        else:
            print(f"No analysis results dataframe generated for folder: {folder}")


# Example usage of the plotting function
# This function will plot the charge transfer spectrum from multiple folders.
# It assumes that each folder contains the necessary data files as per the requirements.
# Define the preamble and folders to process
# preamble = r"<research-data-root>\2025\6.June\CT_spectrum\Au55_O2"
# import os
# folders = os.listdir(preamble)
# plot_charge_transfer_spectrum_from_folders(preamble, folders)


def get_orca_uvvis_as_uv_bil(path_to_orca_out):
    """
    Runs Multiwfn to extract UV-vis data from an ORCA output file
    and converts it to a UV Bil format. This is useful for comparing the data
    directly with the std2 results in the SpecDis software.
    Parameters:
    path_to_orca_out : str
        The path to the ORCA output file.

    Returns: None
    """
    from cats.multiwfn.tools import save_routine_config
    from cats.multiwfn.tools import run_multiwfn

    routine_config = []
    routine_config.append(path_to_orca_out)  # path to the ORCA output file
    routine_config.append("11 // Plot spectrum")
    routine_config.append("3 // UV-vis ")
    routine_config.append("-2")  # export to a plain text file
    routine_config.append("q // graceful exit")

    save_routine_config(routine_config, "config.txt")
    run_multiwfn("config.txt", "log.txt")

    from cats.multiwfn.convert import convert_transinfo_to_uv_bil as convert

    convert("transinfo.txt")

    return None


def get_full_ct_spectrum(path_to_orca_out, path_to_molden, fragments):
    """
    Runs Multiwfn to extract full charge transfer spectrum data from an ORCA output file.
    Parameters:
    path_to_orca_out : str
        The path to the ORCA output file.

    Returns: None
    """
    from cats.multiwfn.tools import save_routine_config
    from cats.multiwfn.tools import run_multiwfn

    routine_config = []
    routine_config.append(path_to_molden)
    routine_config.append("18")
    routine_config.append("16")
    routine_config.append(str(len(fragments)))  # number of fragments
    for fragment in fragments:
        routine_config.append(str(fragments[fragment]))  # fragment atomic indices
    routine_config.append(path_to_orca_out)
    routine_config.append("1")  # save the information to text files
    routine_config.append("0")  # return from IFCT to submenu 18
    routine_config.append("0")  # return from submenu 18 to main menu
    routine_config.append("q")  # graceful exit

    save_routine_config(routine_config, "config.txt")
    try:
        run_multiwfn("config.txt", "log.txt")
    except RuntimeError as exc:
        from pathlib import Path

        manifest = Path("CT_multiple") / "CT_multiple.txt"
        transinfo = Path("transinfo.txt")
        if manifest.is_file() or transinfo.is_file():
            print(f"Warning: Multiwfn exit error ignored as output files exist: {exc}")
        else:
            raise

    return None
