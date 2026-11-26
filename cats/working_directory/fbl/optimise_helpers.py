import os
import sys
from typing import Iterable, Optional, Sequence, Tuple

import pandas as pd
from ase.atoms import Atoms
from ase.io import read
from ase.optimize import FIRE2
from catlearn.activelearning.local import LocalAL

from cats.orca.run.orca_calculator import (
    get_orca_calculator,
    get_orcablocks,
    get_orca_input_header,
)

# Resource defaults for ORCA jobs
TOTAL_NODES = 2
CPUS_PER_NODE = 128
MEMORY_PER_CPU = 1500
CPUS_PER_INSTANCE = TOTAL_NODES * CPUS_PER_NODE
DEFAULT_TRAJECTORY = "opt.traj"
ORCA_HPC = os.environ.get("CATS_ORCA_HPC") or os.environ.get("CPLAS_ORCA_HPC", "falcon")
ORCA_VERSION = os.environ.get("CATS_ORCA_VERSION") or os.environ.get("CPLAS_ORCA_VERSION", "6_1_0")


def build_orca_inputs() -> Tuple[str, str]:
    """Prepare ORCA input settings and resource blocks."""
    orcasimpleinput = get_orca_input_header(
        xc="CAM-B3LYP",
        basis_set="6-31g(d,p)",
        integration_grid="DefGrid3",
        scf_convergence_criteria="TightSCF",
        output_level="SmallPrint",
        task="ENGRAD",
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
        "%scf",
        "MAXITER 1000",
        "END",
    ]

    orcablocks = get_orcablocks(blocks_settings, CPUS_PER_INSTANCE, MEMORY_PER_CPU)
    return orcasimpleinput, orcablocks


def load_atoms(trajectory_file: str = DEFAULT_TRAJECTORY) -> Atoms:
    """Load the initial atomic configuration."""
    if not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"Trajectory file '{trajectory_file}' is missing.")
    try:
        return read(trajectory_file)
    except Exception as exc:
        raise RuntimeError(f"Failed to read '{trajectory_file}'.") from exc


def create_samples(atoms: Atoms) -> Iterable[Tuple[str, Atoms]]:
    """Return the list of optimisation jobs."""
    return [("relaxed", atoms)]


def create_calculator(directory: str, orcasimpleinput: str, orcablocks: str):
    """Initialise the ORCA calculator for a given directory."""
    return get_orca_calculator(
        version=ORCA_VERSION,
        hpc=ORCA_HPC,
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        charge=0,
        mult=2,
        directory=directory,
    )


def run_local_optimisation(atoms: Atoms, directory: str, calculator) -> float:
    """Run the ML-assisted local optimisation until convergence."""
    os.makedirs(directory, exist_ok=True)

    atoms.set_constraint()

    dyn = LocalAL(
        atoms=atoms,
        ase_calc=calculator,
        unc_convergence=0.05,
        local_opt=FIRE2,
        local_opt_kwargs={"maxstep": 0.1},
        save_memory=False,
        use_restart=True,
        min_data=3,
        restart=False,
        verbose=True,
    )
    dyn.run(
        fmax=0.01,
        max_unc=0.03,
        steps=100,
        ml_steps=500,
        dtrust=2.0,
    )
    return atoms.get_potential_energy()


def optimise_structure(
    structure_index: int,
    sample: Tuple[str, Atoms],
    orca_inputs: Tuple[str, str],
) -> pd.DataFrame:
    """Optimise a single structure and package the results."""
    name, source_atoms = sample
    atoms = source_atoms.copy()
    orcasimpleinput, orcablocks = orca_inputs
    calculator = create_calculator(name, orcasimpleinput, orcablocks)
    energy = run_local_optimisation(atoms, name, calculator)
    return pd.DataFrame(
        {
            "structure_index": [structure_index],
            "name": [name],
            "energy": [energy],
        }
    )


def append_results(dataframe: pd.DataFrame, output_path: str) -> None:
    """Persist the optimisation results to disk."""
    is_new_file = not os.path.exists(output_path)
    mode = "w" if is_new_file else "a"
    dataframe.to_csv(output_path, mode=mode, index=False, header=is_new_file)


def run_samples(
    samples: Sequence[Tuple[str, Atoms]],
    orca_inputs: Tuple[str, str],
    output_path: str,
) -> None:
    """Execute optimisations for all samples."""
    for structure_index, sample in enumerate(samples):
        try:
            dataframe = optimise_structure(structure_index, sample, orca_inputs)
        except Exception as exc:
            print(f"Encountered an exception while getting the energy: {exc}")
            sys.exit(1)
        append_results(dataframe, output_path)


def run_optimisation(
    task_id: int,
    trajectory_file: str = DEFAULT_TRAJECTORY,
    output_path: Optional[str] = None,
) -> str:
    """Complete optimisation workflow for a given SLURM task."""
    atoms = load_atoms(trajectory_file)
    samples = create_samples(atoms)
    orca_inputs = build_orca_inputs()
    final_output_path = output_path or f"configuration_info_{task_id}.csv"
    run_samples(samples, orca_inputs, final_output_path)
    return final_output_path

