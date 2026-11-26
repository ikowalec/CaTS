# Author: Dr Igor Kowalec
# ORCA example

import io
import os
import shutil
import warnings
from contextlib import redirect_stdout

import ase.io.orca as _ase_io_orca
from ase.calculators.orca import ORCA
# must update ASE to 3.23.0+ for this import to work
from ase.calculators.orca import OrcaProfile

# Suppress spurious ASE ORCA engrad warnings during external AL/CatLearn optimization
warnings.filterwarnings("ignore", message=".*Geometry optimization did not converge.*", category=UserWarning)

_orig_read_orca_outputs = getattr(_ase_io_orca, "read_orca_outputs", None)
if _orig_read_orca_outputs is not None:
    def _quiet_read_orca_outputs(*args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return _orig_read_orca_outputs(*args, **kwargs)

    _ase_io_orca.read_orca_outputs = _quiet_read_orca_outputs

_SUPPORTED_ORCA_VERSIONS = {
    "falcon": {"6_1_0"},
}
_FALCON_ORCA_ROOT = "/shared/apps/easybuild/x86_64/amd/zen4/software/ORCA"


def _validate_orca_version(hpc: str, version: str) -> str:
    hpc_key = hpc.lower()
    if hpc_key not in _SUPPORTED_ORCA_VERSIONS:
        supported = sorted(_SUPPORTED_ORCA_VERSIONS)
        raise ValueError(f"Unsupported HPC '{hpc}'. Select from {supported}.")

    valid = sorted(_SUPPORTED_ORCA_VERSIONS[hpc_key])
    if version not in valid:
        raise ValueError(
            f"Requested ORCA version '{version}' is not supported on {hpc_key}. Select from {valid}."
        )

    return hpc_key


def get_orca_path(hpc="falcon", version="6_1_0"):
    hpc_key = _validate_orca_version(hpc, version)

    # Prefer explicit ORCADIR if available.
    orcadir = os.environ.get("ORCADIR")
    if orcadir:
        return f"{orcadir}/orca"

    # Fall back to whatever ORCA binary is currently loaded on PATH.
    orca_in_path = shutil.which("orca")
    if orca_in_path:
        return orca_in_path

    if hpc_key == "falcon":
        version_num = version.replace("_", ".")
        return f"{_FALCON_ORCA_ROOT}/{version_num}-gompi-2023b/bin/orca"

    raise RuntimeError(
        f"Unable to resolve ORCA executable for {hpc_key}. Set ORCADIR or load ORCA into PATH."
    )


def get_orca_calculator(
    orcasimpleinput, orcablocks, directory=".", hpc="falcon", version="6_1_0", **kwargs
):
    """
    Function returning a properly configured ORCA Calculator object in ASE.

    :param orcasimpleinput: str
    :param orcablocks: str
    :param directory: str
    :param kwargs:

    :return: ORCA Calculator object
    """

    # TODO: need assertion for CPU count in orcablocks and nodes_per_instance

    command = _get_cpu_command(hpc=hpc, version=version)
    
    profile = OrcaProfile(command=command)  # TODO: to be added dynamically

    calc = ORCA(
        profile=profile,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        directory=directory,
        **kwargs,
    )

    return calc


def get_orcablocks(settings, number_of_cpus, memory_per_cpu):
    """

    :param settings:
    :param number_of_cpus:
    :param memory_per_cpu:
    :return:
    """

    settings_blocks = [
        f"%MaxCore {memory_per_cpu}",
        f"%PAL NPROCS {number_of_cpus}",
        "END",
    ]

    for setting in settings:
        settings_blocks.append(setting)

    orcablocks = " \n".join(settings_blocks)

    return orcablocks


def get_orca_input_header(
    xc="CAM-B3LYP",
    basis_set="LANL2DZ",
    integration_grid="DefGrid3",
    scf_convergence_criteria="VeryTightSCF",
    output_level="NormalPrint",
    **kwargs,
):
    """
    This function takes a range of keyword arguments for an ORCA calculation that are normally located in the first line
    of the input file. The keywords are then combined into a single-line input. Useful when screening settings.

    :param xc: str
        Exchange-correlation functional to be used, using ORCA's notation
    :param basis_set: str
        Basis set to be used for the calculation
    :param integration_grid: str
        DFT integration grid to be used for the simulation
    :param scf_convergence_criteria: str
        TightSCF should normally be used for geometry optimisations, VeryTightSCF for electronic accuracy
    :param output_level: str
        Verbosity of the ORCA program
    :param kwargs: dict
        Other keyword parameters that may belong in the input header

    :return: str
        A single line of input to be passed to the ORCA calculator as the orcasimpleinput parameter
    """

    settings = [
        xc,
        basis_set,
        integration_grid,
        scf_convergence_criteria,
        output_level,
    ] + [value for key, value in kwargs.items()]

    orcasimpleinput = " ".join(settings)

    return orcasimpleinput


def _get_cpu_command(hpc, version="6_1_0"):
    """
    Helper function to return appropriate syntax for cpu settings on each HPC.

    Parameters:
        hpc: str
            The HPC infrastructure to be used.
        version: str
            ORCA version to be used.
    """

    executable = get_orca_path(hpc=hpc, version=version)
    command = f"time {executable}"
    # add_mpirun_wrapper_to_path(hpc)

    return command


def add_mpirun_wrapper_to_path(hpc):
    """
    Function to add the mpirun wrapper to the PATH environment variable.
    ORCA uses mpirun internally. To avoid srun/mpirun conflicts we use an mpirun wrapper scripts.

    Based on: https://guido.vonrudorff.de/2018/orca-in-slurm-native-mode/

    Wrapper contents (must be set to executable with "chmod +x file"):

    #$/bin/sh
    echo "MPIWRAPPER IN" "${@}"
    echo "MPIWRAPPER OUT" $(echo "${@}" | sed 's/^-np/-n/')
    exec srun --mpi=openmpi $(echo "${@}" | sed 's/^-np/-n/')
    """
    wrapper_location: dict[str, str] = {}
    hpc_key = hpc.lower()
    if hpc_key not in wrapper_location:
        raise ValueError(f"No mpirun wrapper configured for HPC '{hpc}'.")
    os.environ["PATH"] = f"{wrapper_location[hpc_key]}:{os.environ['PATH']}"
