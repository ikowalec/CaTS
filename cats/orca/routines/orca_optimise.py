# Author: Dr Igor Kowalec
# ORCA example

from ase.atoms import Atoms
from ase.optimize.bfgs import BFGS
from cats.orca.run.orca_calculator import get_orca_calculator
from cats.orca.run.orca_calculator import get_orcablocks
from cats.orca.run.orca_calculator import get_orca_input_header


# Calculator setup
number_of_cpus = 40  # Explicitly set number of CPUs #TODO: do it dynamically for HPCs
memory_per_cpu = 1000  # memory in Mb # TODO: set max dynamically per HPC, although might affect queueing time

blocks_settings = []

orcasimpleinput = get_orca_input_header(
    xc="CAM-B3LYP",
    basis_set="LANL2DZ",
    integration_grid="DefGrid3",
    scf_convergence_criteria="VeryTightSCF",
    output_level="NormalPrint",
    task="ENGRAD",
)

orcablocks = get_orcablocks(blocks_settings, number_of_cpus, memory_per_cpu)

# This should be called whenever a Calculator object is needed
calc = get_orca_calculator(
    orcasimpleinput=orcasimpleinput, orcablocks=orcablocks, label="orca"
)
print(calc.label)

# Set up a geometry optimisation
water = Atoms("OHH", positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)], calculator=calc)
opt = BFGS(water)
opt.run(fmax=0.05)


print("Final energy is", water.get_potential_energy(), "eV")
