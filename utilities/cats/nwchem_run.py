from ase.optimize import BFGS
from ase.build import molecule
from cats.nwchem.run import get_nwchem_command
from cats.nwchem.run import get_nwchem_calculator
from cats.nwchem.run import apply_default_basis
from cats.nwchem.run import get_nwchem_socket_calculator

# Define your molecule or cluster (example: AuCu)
atoms = molecule("CO")

nodes = 1
cores_per_node = 192
cores = nodes * cores_per_node
memory_per_core = 2000  # mb

calc = get_nwchem_calculator(
    "optimize",
    get_nwchem_command("falcon", nodes, cores_per_node, cores),
    memory_per_core=memory_per_core,
)

calc.parameters.dft["mult"] = (
    1  # should be 1 or 2 by default when even/odd no. electrons
)
calc.parameters["restart_kw"] = "start"

# set the default basis set
atoms.calc = calc
apply_default_basis(atoms)

# set up the i-Pi sockets connectivity
with get_nwchem_socket_calculator(calc, host="127.0.0.1", codata_warning=False)[
    0
] as socket_calc:
    atoms.calc = socket_calc

    # Run the optimization
    opt = BFGS(atoms, trajectory="atoms.traj")

    # Ensure the run command is invoked in production
    # opt.run(fmax=0.01)
