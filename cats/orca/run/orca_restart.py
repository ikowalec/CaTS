# Author: Dr Igor Kowalec
import os
import fnmatch
from typing import Tuple, Dict, Any, Optional
import numpy as np
from ase.io import read
import ase


def check_ase_version(min_version: str) -> bool:
    """Return True if installed ASE version >= min_version.

    Compares numeric components of the version strings, padding as needed.
    """

    def parse(v: str):
        parts = v.split(".")
        out = []
        for p in parts:
            num = "".join(ch for ch in p if ch.isdigit())
            out.append(int(num) if num else 0)
        return out

    cur = parse(ase.__version__)
    req = parse(min_version)

    n = max(len(cur), len(req))
    cur += [0] * (n - len(cur))
    req += [0] * (n - len(req))

    return tuple(cur) >= tuple(req)


def _read_restart_geometry(xyz_path: str) -> Optional[ase.Atoms]:
    if not os.path.exists(xyz_path):
        return None

    print("Assuming internal ORCA optimisation - reading ORCA xyz file for restart...")
    try:
        return read(xyz_path, format="xyz")
    except Exception:
        print("Could not read ORCA xyz file for restart - assuming ASE optimisation...")
        return None


def _finished_normally(out_path: str) -> bool:
    try:
        with open(out_path, "r") as stdout:
            lines = stdout.readlines()
    except OSError:
        return False

    return bool(lines) and ("TOTAL RUN TIME" in lines[-1])


def _maybe_promote_ges(directory: str) -> None:
    
    ges_path = os.path.join(directory, "orca.ges")
    if not os.path.exists(ges_path):
        return

    print(
        "ORCA calculation did not finish normally - replacing .gbw file with .ges file for restart..."
    )
    gbw_path = os.path.join(directory, "orca.gbw")
    try:
        if os.path.exists(gbw_path):
            os.remove(gbw_path)
        os.rename(ges_path, gbw_path)
    except OSError:
        pass


def _extract_results(directory: str, out_path: str) -> Tuple[Optional[ase.Atoms], Dict[str, Any]]:
    if not os.path.exists(out_path):
        return None, {}

    if not _finished_normally(out_path):
        _maybe_promote_ges(directory)
        return None, {}

    print("ORCA calculation finished normally - reading .out file results...")
    try:
        if check_ase_version("3.24.0"):
            output = read(out_path, format="orca-output")
            return output, output.calc.results

        # Import lazily to avoid ImportError on newer ASE versions
        from ase.io.orca import read_orca_outputs as _read_orca_outputs

        results = _read_orca_outputs(directory, out_path)
        return None, results
    except Exception:
        return None, {}


def restart_optimisation(atoms, directory, keep_temp: bool = False) -> Tuple[ase.Atoms, Dict[str, Any]]:
    """
    ORCA restart functionality

    TODO: must be updated for use with engrad rather than opt task

    # read_orca_outputs is deprecated as of 3.24.0
    """

    results: Dict[str, Any] = {}
    if not os.path.isdir(directory):
        print("No valid restart files found - using input geometry.")
        return atoms, results

    xyz_path = os.path.join(directory, "orca.xyz")
    out_path = os.path.join(directory, "orca.out")

    output = _read_restart_geometry(xyz_path)
    file_output, file_results = _extract_results(directory, out_path)
    if file_output is not None:
        output = file_output
    if file_results:
        results = file_results

    if not keep_temp:
        clean_temporary_files(directory)

    if output is not None:
        print("Recovered geometry from previous calculation.")
        assert len(atoms) == len(output)
        symbols_1 = sorted(atoms.get_chemical_symbols())
        symbols_2 = sorted(output.get_chemical_symbols())
        assert symbols_1 == symbols_2
    else:
        print("No valid restart files found - using input geometry.")
        output = atoms

    return output, results


def clean_temporary_files(directory):
    """Remove .tmp files from ORCA working directory."""
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, "*.tmp"):
            os.remove(os.path.join(directory, file))
        if fnmatch.fnmatch(file, "*.bas*"):
            os.remove(os.path.join(directory, file))


def check_forces(results: Dict[str, Any], fmax: float = 0.02) -> bool:
    """Check forces array for convergence against criterion."""
    f = results.get("forces")
    if f is None:
        return False
    try:
        arr = np.asarray(f)
        if arr.size == 0:
            return False
        return np.linalg.norm(arr, axis=1).max() <= fmax
    except Exception:
        return False
    
from ase import Atoms
def check_forces_on_atoms(atoms: Atoms, fmax: float = 0.01) -> bool:
    """Check forces array for convergence against criterion."""
    f = atoms.calc.results.get("forces")
    if f is None:
        return False
    try:
        arr = np.asarray(f)
        if arr.size == 0:
            return False
        return np.linalg.norm(arr, axis=1).max() <= fmax
    except Exception:
        return False

    
def test_check_forces_on_atoms():
    from ase.build import molecule
    from ase.calculators.emt import EMT

    # Create a water molecule
    atoms = molecule("H2O")
    atoms.calc = EMT()
    atoms.get_forces()  # To populate forces

    # Check forces with a high fmax (should be True)
    assert check_forces_on_atoms(atoms, fmax=10.0) == True

    # Check forces with a low fmax (should be False)
    assert check_forces_on_atoms(atoms, fmax=0.001) == False

    print("test_check_forces_on_atoms passed.")
