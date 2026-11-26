# Author: Igor Kowalec
from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union, Tuple, List
import pandas as pd


_MULLIKEN_ATOM_RE = re.compile(
    r"^\s*(?P<index>\d+)\((?P<symbol>[A-Za-z]{1,3})\s*\)\s+(?P<values>[-+0-9.Ee\s]+)$"
)
_MULLIKEN_NET_CHARGE_RE = re.compile(
    r"^\s*Atom\s+(?P<index>\d+)\((?P<symbol>[A-Za-z]{1,3})\s*\)\s+"
    r"Population:\s*[-+0-9.Ee]+\s+Net charge:\s*(?P<charge>[-+0-9.Ee]+)\s*$"
)


def set_multiwfn_path():
    """Edit the path  in the .env file in your $VIRTUAL_ENV directory
    MULTIWFN_DIR = /path/to/Multiwfn_dir
    Multiwfnpath /path/to/Multiwfn_executable
    """

    import os
    from pathlib import Path

    try:
        from dotenv import load_dotenv
        load_dotenv(f'{os.environ["VIRTUAL_ENV"]}/.env')
    except ImportError:
        venv_env = Path(os.environ.get("VIRTUAL_ENV", "")) / ".env"
        if venv_env.is_file():
            for line in venv_env.read_text(encoding="utf-8", errors="replace").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))


def run_multiwfn(job_config: str, job_output: str):
    import os
    import shutil
    from pathlib import Path

    set_multiwfn_path()
    install_ini = Path(os.environ.get("Multiwfnpath", "")) / "settings.ini"
    local_ini = Path("settings.ini")
    if install_ini.is_file() and not local_ini.exists():
        try:
            shutil.copy2(install_ini, local_ini)
        except Exception:
            pass

    cmd = (
        f'ulimit -s unlimited 2>/dev/null; '
        f'export OMP_STACKSIZE=1500M; '
        f'export Multiwfnpath="{Path.cwd()}"; '
        f'{os.environ["MULTIWFN_DIR"]} < {job_config} > {job_output} 2>&1'
    )
    rc = os.system(cmd)
    if rc != 0:
        raise RuntimeError(f"Multiwfn exited with code {rc}. Check {job_output}")


def save_routine_config(routine_config, config_name):
    """Save the list as a file which can later be read for reproducibility"""

    with open(config_name, "w") as file:
        file.write("\n".join(routine_config) + "\n")


def get_mulliken_population_routine_config(
    path_to_molden: str,
    *,
    report_name: str = "Mulliken_population.txt",
):
    """Return the documented Multiwfn 7-5 Mulliken population export routine."""

    return [
        str(path_to_molden),
        "7 // population analysis and atomic charges",
        "5 // Mulliken population analysis",
        "-1 // all atoms",
        str(report_name),
        "1 // export atom populations and charges",
        "n // do not export a separate coordinates-and-charges file",
        "0 // return from Mulliken population analysis",
        "0 // return from population analysis",
        "q",
    ]


def parse_mulliken_population_report(report_name: str) -> list[dict[str, Any]]:
    """Parse the ``Population of atoms`` table exported by Multiwfn 7-5."""

    atoms: list[dict[str, Any]] = []
    in_population_table = False
    for line in Path(report_name).read_text(encoding="utf-8", errors="replace").splitlines():
        if "Population of atoms:" in line:
            in_population_table = True
            continue
        if not in_population_table:
            continue
        net_charge_match = _MULLIKEN_NET_CHARGE_RE.match(line)
        if net_charge_match is not None:
            atoms.append(
                {
                    "index_1based": int(net_charge_match.group("index")),
                    "symbol": net_charge_match.group("symbol"),
                    "charge_e": float(net_charge_match.group("charge")),
                }
            )
            continue
        match = _MULLIKEN_ATOM_RE.match(line)
        if match is None:
            if atoms and line.strip():
                break
            continue
        values = [float(value) for value in match.group("values").split()]
        if not values:
            continue
        atoms.append(
            {
                "index_1based": int(match.group("index")),
                "symbol": match.group("symbol"),
                "charge_e": values[-1],
            }
        )
    if not atoms:
        raise ValueError(f"No Multiwfn Mulliken atom charges found in {report_name}.")
    return atoms


def mulliken_population_routine(
    path_to_molden: str,
    *,
    fragments_0based: dict[str, list[int]] | None = None,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Calculate ground-state Mulliken charges from the same wavefunction used by IFCT."""

    config_name = "Mulliken_population_config.txt"
    log_name = "Mulliken_population_log.txt"
    report_name = "Mulliken_population.txt"
    if overwrite:
        for filename in (config_name, log_name, report_name):
            Path(filename).unlink(missing_ok=True)
    save_routine_config(
        get_mulliken_population_routine_config(path_to_molden, report_name=report_name),
        config_name,
    )
    if not Path(report_name).is_file():
        run_multiwfn(config_name, log_name)
    atoms = parse_mulliken_population_report(report_name)
    charges = [float(atom["charge_e"]) for atom in atoms]
    fragment_charges = {
        str(label): sum(charges[int(index)] for index in indices)
        for label, indices in (fragments_0based or {}).items()
    }
    return {
        "method": "Mulliken",
        "program": "Multiwfn",
        "source_wavefunction": str(path_to_molden),
        "atoms": atoms,
        "total_charge_e": sum(charges),
        "fragment_charges_e": fragment_charges,
    }


def get_charge_transfer_routine_config(fragments, excitation_numbers):

    routine_config = [
        "orca.molden.input",
        "18 // electron excitation analysis",
        "8 // calculate interfragment charge transfter via IFCT",
        "1 // Mulliken 1 Hirschfeld 2",
        "orca.out",
        f"{excitation_numbers[0]} // choose excited state",
        f"{len(fragments)} // define fragments, should probably consider nearest neighbors separately",
    ]


    for fragment in fragments:
        routine_config.append(f"{fragments[fragment]}")

    routine_config.append("0")

    for excitation in excitation_numbers[1:]:
        routine_config += ["18", "8", "1", f"{excitation}", f"{len(fragments)}"]

        for fragment in fragments:
            routine_config.append(f"{fragments[fragment]}")

        routine_config.append("0")

    graceful_exit = ["0", "q"]
    routine_config += graceful_exit

    return routine_config


def clean_output(output_file_name, keep_old_file=True):
    """Clear the Progress lines in the output of multiwfn"""

    import os

    clean_output = f'{output_file_name.split(".")[0]}_clean.txt'

    file = open(output_file_name, "r")
    lines = file.readlines()

    file_clean = open(clean_output, "w")
    for line in lines:
        if not "Progress" in line:
            file_clean.write(line)
    file.close()
    file_clean.close()

    if not keep_old_file:
        os.remove(output_file_name)

    return clean_output


def parse_ct_output(output_name):

    try:
        file = open(output_name)
    except FileNotFoundError():
        print(f"File not found: {output_name}")

    lines = file.readlines()

    # TODO: adjust for more fragments
    intrinsic_CT = []
    apparent_ct = []
    net_charge = []

    for line in lines:
        if "Apparent charge transfer percentage, CT(%):" in line:
            apparent_ct += [line.split()[-2]]
        if "Intrinsic charge transfer percentage, CT(%):" in line:
            intrinsic_CT += [line.split()[-2]]
        if "Net  " in line:
            net_charge += [line.split()[-1]]

    charges = {
        "intrinsic_CT": intrinsic_CT,
        "apparent_CT": apparent_ct,
        "net_charge": net_charge,
    }

    file.close()

    return charges


def combine_ct_results(charges, fragments, excitation_numbers):
    """Put the parsed data together with initial input for coherent data."""

    results = {}

    for index, excitation in enumerate(excitation_numbers):
        results[str(excitation)] = {
            "fragments": fragments,
            "apparent_CT": charges["apparent_CT"][index],
            "intrinsic_CT": charges["intrinsic_CT"][index],
            "net_charge": charges["net_charge"][index],
        }

    return results


def charge_transfer_routine(fragments, excitation_numbers, overwrite=True):
    """Run the Charge Transfer analysis routine."""
    import os

    output_file = "CT_analysis.txt"
    config_file_name = "CT_config.txt"
    clean_output_name = f'{output_file.split(".")[0]}_clean.txt'

    if overwrite:
        to_clear = [
            "uvvis_input.txt",
            "uvvis_output.txt",
            "spectrum_line.txt",
            "spectrum_curve.txt",
            output_file,
            config_file_name,
            clean_output_name,
        ]
        for file in to_clear:
            if os.path.exists(file):
                os.remove(file)

    routine_config = get_charge_transfer_routine_config(fragments, excitation_numbers)
    save_routine_config(routine_config=routine_config, config_name=config_file_name)

    if not os.path.exists(clean_output_name) or overwrite:
        run_multiwfn(job_config=config_file_name, job_output=output_file)
        clean_output(output_file, keep_old_file=False)

    charges = parse_ct_output(clean_output_name)
    results = combine_ct_results(charges, fragments, excitation_numbers)

    return results


def get_uvvis_data():
    """Get UV-Vis data recovered from orca.out by Multiwfn"""

    import pandas as pd
    import os

    input_file = "uvvis_input.txt"
    output_file = "uvvis_output.txt"
    uvvis_data_lines = "spectrum_line.txt"

    if not os.path.exists(uvvis_data_lines):
        """Write the instructions"""
        prompts = ["orca.out", "11", "3", "2", "q"]
        with open(input_file, "w") as file:
            file.write("\n".join(prompts))
        run_multiwfn(input_file, output_file)

    """Analyse the output"""
    with open(uvvis_data_lines, "r") as file:
        lines = file.readlines()
        energies = []
        intensities = []

        for _, line in enumerate(lines):
            line = line.split()
            if not line:
                continue
            elif float(line[1]) == 0.0:
                continue
            else:
                energies += [line[0]]
                intensities += [line[1]]

        df = pd.DataFrame()
        df["excitation"] = list(range(1, len(energies) + 1))
        df["wavelength"] = energies
        df["wavelength"] = df["wavelength"].astype("float32")
        df["intensity"] = intensities
        df["intensity"] = df["intensity"].astype("float32")

    return df


def get_top_intensity_excitations(relative_intensity_threshold=0.1):
    """Retrieve the top excitations"""

    uvvis_data = get_uvvis_data()
    uvvis_data = uvvis_data.sort_values("intensity", ascending=False)
    uvvis_data["relative_intensity"] = (
        uvvis_data["intensity"] / uvvis_data["intensity"].max()
    )
    uvvis_data = uvvis_data[
        uvvis_data["relative_intensity"] > relative_intensity_threshold
    ]

    excitation_numbers = uvvis_data["excitation"].values

    return excitation_numbers


def test_block():
    from cats.multiwfn.tools import get_top_intensity_excitations
    from cats.multiwfn.tools import charge_transfer_routine
    import pandas as pd

    fragments = {
        "1": "1-55",
        "2": "56",
    }  # Example. To match ASE, one should avoid using 1-55 notation, instead use 1,2,3,4 ...
    excitation_numbers = get_top_intensity_excitations()
    print(excitation_numbers)

    print(f"Performing analysis for{len(excitation_numbers)} intensity excitations.")

    results = charge_transfer_routine(fragments, excitation_numbers, overwrite=False)
    df = pd.DataFrame.from_dict(results)
    print(df.transpose())


def get_fragments_from_formula(atoms, chemical_symbols_in_fragment_one=[]):
    """Get atomic indices corresponding to the atoms with chemical symbol in
    chemical_symbols_in_fragment_one, and the remaining ones, to be used with Multiwfn
    for the purpose of charge transfer analysis. One can specify e.g. metal symbols in
    nanoparticles to be fragment 1, and the adsorbate gets picked up as fragment.
    The indices are adjusted (start from 0 in Python/ASE, start from 1 in Multiwfn)

    Parameters:
    atoms: Atoms object
    chemical_symbols_in_fragment_one: list of strings

    Returns: two arrays of indices in each fragment"""
    import numpy as np

    fragment_1 = [
        atom.index for atom in atoms if atom.symbol in chemical_symbols_in_fragment_one
    ]
    fragment_2 = [
        atom.index
        for atom in atoms
        if not atom.symbol in chemical_symbols_in_fragment_one
    ]

    assert fragment_1 and fragment_2, "The second fragment is empty!"

    """The indexing in multiwfn starts from 1 not 0"""
    fragment_1 = np.array(fragment_1)
    fragment_1 += 1
    fragment_2 = np.array(fragment_2)
    fragment_2 += 1

    return fragment_1, fragment_2


def save_cube_files(exitations, quality):
    pass


def get_cartesian_molden(path_to_orca_gbw):
    """
    Get the Cartesian coordinates from the ORCA GBW file in Molden format.
    Requires the following to be changed in the settings.ini file of Multiwfn:
    - orcapath= path_to_your_orca_directory
    - orca_2mklpath= path_to_your_orca_2mkl_executable
    - iloadasCart= 1
    Multiwfn reads the wavefunction, translates the spherical harmonics to cartesian coordinates,
    and saves the coordinates in a Molden file.

    Such file can be used as input to the simpified TDDFT std2 by Grimme et al.

    The function does not return anything, it just runs Multiwfn with the appropriate settings.
    """

    from cats.multiwfn.tools import run_multiwfn
    from cats.multiwfn.tools import save_routine_config

    routine_config = []
    routine_config.append(path_to_orca_gbw)  # path to the ORCA GBW file
    routine_config.append("100 // other options")
    routine_config.append("2 // export file")
    routine_config.append("6 // as molden")
    routine_config.append("cartesian.molden")
    routine_config.append("0 // Return to main menu")
    routine_config.append("q")
    save_routine_config(routine_config, "config.txt")
    run_multiwfn("config.txt", "log.txt")

    return None


def parse_ifct(
    file: Union[str, Path],
    *,
    wide: bool = False,
    top_n: Optional[int] = None,
    min_pct: Optional[float] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Parse an IFCT contribution listing into pandas DataFrames.

    Parameters
    ----------
    file
        Path to the text file.
    wide
        If False (default), return (states_df, terms_df) where terms_df is long-form:
            terms_df columns = [state, term, pct]
        If True, return a single merged DataFrame:
            states_df merged with wide-form term columns (one column per term).
    top_n
        When wide=True (or when building wide terms), keep only top N terms per state (by pct).
    min_pct
        Filter terms to pct >= min_pct before wide pivot (and also applies to returned terms_df).

    Returns
    -------
    If wide=False:
        (states_df, terms_df)
    If wide=True:
        merged_df
    """
    _NUM_RE = r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
    _ROW_RE = re.compile(
        rf"^\s*(?P<state>\d+)\s+"
        rf"(?P<f>{_NUM_RE})\s+"
        rf"(?P<nm>{_NUM_RE}):\s+"
        rf"(?P<rest>.+?)\s*$"
    )

    _TERM_RE = re.compile(
        r"(?P<label>Redis\(\d+\)|\d+->\d+|\d+<-\d+)\s+"
        r"(?P<pct>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*%"
    )

    _CT_E_RE = re.compile(
        rf"CT\(e\)\s*[:=]?\s*(?P<ct>{_NUM_RE})",
        re.IGNORECASE,
    )
    _CT_PCT_RE = re.compile(
        rf"CT\(%\)\s*[:=]?\s*(?P<ct>{_NUM_RE})",
        re.IGNORECASE,
    )


    file = Path(file)
    lines = file.read_text(encoding="utf-8", errors="replace").splitlines()

    state_rows: List[Dict[str, Any]] = []
    term_rows: List[Dict[str, Any]] = []

    header = next((line.split() for line in lines if line.strip()), [])
    if header and header[0].casefold() == "state" and any(
        column.casefold().startswith("hole(") for column in header
    ):
        term_columns = {
            index: (
                f"Redis({match.group(1)})"
                if (match := re.fullmatch(r"redis\((\d+)\)", column, re.IGNORECASE))
                else column
            )
            for index, column in enumerate(header)
            if re.fullmatch(r"redis\(\d+\)|\d+(?:->|<-)\d+", column, re.IGNORECASE)
        }
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < len(header):
                continue
            try:
                state = int(fields[0])
            except ValueError:
                continue
            state_rows.append({"state": state})
            for index, label in term_columns.items():
                term_rows.append(
                    {"state": state, "term": label, "pct": float(fields[index]) * 100.0}
                )
        states_df = pd.DataFrame(state_rows, columns=["state"])
        terms_df = pd.DataFrame(term_rows, columns=["state", "term", "pct"])
        if min_pct is not None:
            terms_df = terms_df[terms_df["pct"] >= float(min_pct)].reset_index(drop=True)
        if not wide:
            return states_df, terms_df
        wide_terms = terms_df.pivot_table(
            index="state", columns="term", values="pct", aggfunc="max"
        ).reset_index()
        wide_terms.columns.name = None
        return states_df.merge(wide_terms, on="state", how="left")

    for line in lines:
        m = _ROW_RE.match(line)
        if not m:
            continue

        state = int(m.group("state"))
        f = float(m.group("f"))
        nm = float(m.group("nm"))
        rest = m.group("rest")
        ct_e = None
        ct_pct = None

        m_ct_e = _CT_E_RE.search(rest)
        if m_ct_e:
            ct_e = float(m_ct_e.group("ct"))

        m_ct_pct = _CT_PCT_RE.search(rest)
        if m_ct_pct:
            ct_pct = float(m_ct_pct.group("ct"))

        state_rows.append({"state": state, "f": f, "nm": nm, "CT(e)": ct_e, "CT(%)": ct_pct})

        for tm in _TERM_RE.finditer(rest):
            term_rows.append(
                {"state": state, "term": tm.group("label"), "pct": float(tm.group("pct"))}
            )

    states_df = (
        pd.DataFrame(state_rows, columns=["state", "f", "nm", "CT(e)", "CT(%)"])
        .sort_values("state")
        .reset_index(drop=True)
    )
    terms_df = (
        pd.DataFrame(term_rows, columns=["state", "term", "pct"])
        .sort_values(["state", "pct"], ascending=[True, False])
        .reset_index(drop=True)
    )

    if min_pct is not None and not terms_df.empty:
        terms_df = terms_df[terms_df["pct"] >= float(min_pct)].reset_index(drop=True)

    if not wide:
        return states_df, terms_df

    # build wide-form terms
    df = terms_df.copy()
    if top_n is not None and not df.empty:
        df = (
            df.sort_values(["state", "pct"], ascending=[True, False])
              .groupby("state", as_index=False)
              .head(int(top_n))
        )

    wide_terms = (
        df.pivot_table(index="state", columns="term", values="pct", aggfunc="max")
          .reset_index()
    )
    wide_terms.columns.name = None

    merged = states_df.merge(wide_terms, on="state", how="left")
    return merged



def theory():
    """
    Based on the paper in Nature Comms - I should be looking at
    all transitions to the LUMO.

    In RT-TDDFT after the pulse we should look for oscillation.

    Fourier Transform will show two types of oscillations - with 1 peak around 0 eV
    (single-? interaction aka hot electron carriers)
    or 2+ peaks (plasmonic resonance)

    Also orca_2json looks amazing, it may be worth exploring. Apparently
    %Method
    WriteJSONPropertyfile True
    End

    Can then parse JSON files instead of ORCA input!

    """
