def convert_transinfo_to_uv_bil(transinfo_file, output_file="transinfo.uv.bil"):
    """
    Convert transinfo.txt (Multiwfn spectrum data) to a .uv.bil (Spectrum-Dis) file format.
    The transinfo.txt file is obtained from Multiwfn and contains spectral data.
    The procedure requires orca.out-->11-->3-->-2 options.

    The input file is expected to have three columns: energy, strength, and broadening.
    The output file will contain the wavelength in nm and the oscillator strength.
    """

    import pandas as pd

    # Define input and output file paths

    # Read the file, skipping the first line
    with open(transinfo_file, "r") as f:
        lines = f.readlines()[1:]  # Skip line 1

    # Clean and split each line into columns
    data = [line.strip().split() for line in lines]

    # Create a DataFrame assuming 3 columns: xx, yy, zz
    df = pd.DataFrame(data, columns=["energy", "strength", "broadening"])
    ev_to_nm = 1239.84193
    df["energy_nm"] = ev_to_nm / df["energy"].to_numpy().astype(float)

    # normalize the strength column to one
    sum_strength = df["strength"].to_numpy().astype(float).sum()
    df["strength"] = df["strength"].to_numpy().astype(float) / sum_strength
    df["strength"] = df["strength"].to_numpy().astype(float)

    # Select only the first and second columns
    filtered_df = df[["energy_nm", "strength"]]

    # Write to output file with exactly 8 spaces between columns
    with open(output_file, "w") as f:
        f.write(
            "SpecDis bil-file (length formalism) wavelength (nm)       oscillator strength\n"
        )
        for _, row in filtered_df.iterrows():
            f.write(f"{row['energy_nm']:.1f}\t{row['strength']:.6f}\n")

    return None

