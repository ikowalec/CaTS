from ase.build import molecule
from ase.optimize.bfgs import BFGS
from cats.orca.run.orca_calculator import get_orca_calculator
from cats.orca.run.orca_calculator import get_orcablocks
from cats.orca.run.orca_calculator import get_orca_input_header
from cats.orca.run.orca_restart import restart_optimisation
from cats.orca.run.orca_restart import check_forces
import pandas as pd
import os
import sys

ORCA_HPC = os.environ.get("CATS_ORCA_HPC") or os.environ.get("CPLAS_ORCA_HPC", "falcon")
ORCA_VERSION = os.environ.get("CATS_ORCA_VERSION") or os.environ.get("CPLAS_ORCA_VERSION", "6_1_0")

# Calculator setup
TOTAL_NODES = 4  # Total number of nodes to use
CPUS_PER_NODE = 128  # Explicitly set number of CPUs #TODO: do it dynamically for HPCs
MEMORY_PER_CPU = 1300  # memory in Mb # TODO: set max dynamically per HPC, although might affect queueing time
CPUS_PER_INSTANCE = int(TOTAL_NODES * CPUS_PER_NODE)  # Number of CPUs per instance
SLURM_TASK_ID = int(sys.argv[1])  # SLURM task ID from job array

orcasimpleinput = get_orca_input_header(
    xc="CAM-B3LYP",
    basis_set="6-31g(d,p)",  # replaced for Au
    integration_grid="DefGrid3",
    scf_convergence_criteria="VeryTightSCF",
    output_level="NormalPrint",
    task="OPT",
    scf_convergence_strategy="SlowConv",
)

blocks_settings = [
    "%GEOM",
    "Convergence Normal",
    "Trust 0.5",
    "END",
    "%BASIS",
    'newGTO 79 "LANL2DZ" end',
    'newECP 79 "HayWadt" end',
    "END",
]

orcablocks = get_orcablocks(blocks_settings, CPUS_PER_INSTANCE, MEMORY_PER_CPU)

# Set up a geometry optimisation
import fnmatch
from ase.io import read

filenames = [
    filename for filename in os.listdir() if fnmatch.fnmatch(filename, "*.xyz")
]
samples = [
    (filename.split(".")[0], read(filename, format="xyz")) for filename in filenames
]


def routine(atoms, directory):
    """QM Job Routine For Execution"""
    # check for restarts
    atoms, results = restart_optimisation(atoms, directory, keep_temp=False)
    converged = check_forces(results, fmax=0.05)

    if not converged:
        atoms.calc = get_orca_calculator(
            version=ORCA_VERSION,
            hpc=ORCA_HPC,
            orcasimpleinput=orcasimpleinput,
            orcablocks=orcablocks,
            charge=0,
            mult=2,
            directory=directory,
        )
        opt = BFGS(atoms)
        opt.run(fmax=0.05)
        e_final = atoms.get_potential_energy()
    else:
        e_final = results["energy"]

    return e_final


def condensed_generator(structure_index):
    """Concurrent Job Generator"""

    atoms = samples[structure_index][1]
    directory = samples[structure_index][0]

    sync_command = f"rsync -av --exclude='*.tmp' --exclude='*.bas' $TEMP_DIRECTORY/{directory} $HOME_DIRECTORY/"

    df1 = pd.DataFrame()
    try:
        e_final = routine(atoms, directory)
    except Exception as e:
        os.system(sync_command)
        print(f"Encountered the following exception while getting the energy: {e}")
        sys.exit()

    # Put the data together in a single DataFrame row
    df1["structure_index"] = [structure_index]
    df1["name"] = [directory]
    df1["energy"] = [e_final]

    # return the files to the original directory
    os.system(sync_command)

    return df1


df = condensed_generator(SLURM_TASK_ID)

if os.path.exists(f"{os.environ['HOME_DIRECTORY']}/configuration_info.csv"):
    df.to_csv(
        f"{os.environ['HOME_DIRECTORY']}/configuration_info.csv",
        mode="a",
        index=False,
        header=False,
    )
else:
    df.to_csv(f"{os.environ['HOME_DIRECTORY']}/configuration_info.csv", index=False)

print(df)
