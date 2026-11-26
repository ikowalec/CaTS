import os

from cats.orca.run.orca_calculator import get_orca_calculator
from cats.orca.run.orca_calculator import get_orcablocks
from cats.orca.run.orca_calculator import get_orca_input_header
from catlearn.activelearning.local import LocalAL
from ase.io import read
from ase.optimize import FIRE2

# Calculator setup
number_of_cpus = 32 # Explicitly set number of CPUs #TODO: do it dynamically for HPCs
memory_per_cpu = 1300 # memory in Mb # TODO: set max dynamically per HPC, although might affect queueing time
ORCA_HPC = os.environ.get("CATS_ORCA_HPC") or os.environ.get("CPLAS_ORCA_HPC", "falcon")
ORCA_VERSION = os.environ.get("CATS_ORCA_VERSION") or os.environ.get("CPLAS_ORCA_VERSION", "6_1_0")

def optimize_atoms(atoms, calc, restart=True, previous_calculations=None, local_opt_kwargs={"maxstep":0.01}):

    # Initialize local optimization
    dyn = LocalAL(atoms=atoms,
                        ase_calc=calc,
                        unc_convergence=0.05,
                        local_opt=FIRE2,
                        local_opt_kwargs=local_opt_kwargs,
                        save_memory=False,
                        use_restart=restart,
                        min_data=1,
                        restart=restart,
                        verbose=True,
                        prev_calculations=previous_calculations
            )

    dyn.run(fmax=0.01,
            max_unc=0.03,
            steps=150,
            ml_steps=500,
            dtrust=2.0)

    return atoms


def get_default_orca_calculator(atoms, flavour, cpus, mem, multiplicity="lowest"):
    
    '''This function sets up an ORCA calculator for either DFT or TDDFT geometry optimization.'''
    
    blocks_settings = []
    if "Au" in atoms.get_chemical_symbols():
        BASIS = ['%BASIS',
                'newGTO 79 "LANL2DZ" end',
                'newECP 79 "HayWadt" end',
                'END'
                ]
    else:
        BASIS=['%BASIS',
               'AuxJ "def2/J"',
               'END']
    
    blocks_settings += BASIS
 
    GEOM = ['%GEOM', 
            '',
            'Convergence Normal', 
            'Trust 0.5', 
            'END']
    
    blocks_settings += GEOM
    
    #TODO:  The problem is this should be dynamically selected to match irradiation energy as best as it can.
    # This is not standard practice as the calcualtions are difficult to converge. 
    # I care about my practice and not standard practice!
    #
    # The (suboptimal) way is to calculate TDA spectrum, analyse the peaks around original +- 5, match energy and weigh based on intensity/charge transfer
    # The above can be obtained rather quickly with multiwfn - should be minutes for about 10 signals based on the wavefunction.
    # However it requires TDA to be evaluated on the same structure twice. Also the optimiser would have to be stopped at each iteration - this is doable and the 
    # active learning potential would be retrained with new data at each iteration.
    # currently lowest time is 10-12h per TDA -> this would allow only about 4 structures to be evaluated in 4 days
    # can we cut down TDA time further?
    #
    TDDFT = ["%TDDFT",
        "MaxIter 999",
        "NRoots 5",
        "IRoot 1", # select excited state to optimize  
        "FollowIRoot TRUE",
        "FIRKEEPFIRSTREF TRUE",
        "MaxDim 20",
        "ETol 1e-5",
        "RTol 1e-5",
        "TPrint 1e-8",
        "END"
            ]

    #if constraint:
    #    BASIS[1] = f'Constraints {{ B 0 1 {atoms.get_distance(0, 1)} C }} end',

    orcasimpleinput = get_orca_input_header(xc='CAM-B3LYP',
                                            cosx='RIJCOSX',
                                            basis_set='6-31g(d,p)',
                                            integration_grid='DefGrid3',
                                            scf_convergence_criteria='VeryTightSCF',
                                            output_level='NormalPrint',
                                            task='ENGRAD',
                                            scf_convergence_strategy="SlowConv")
    
    if flavour == 'DFT':
        directory = "DFT"
    elif flavour == 'TDDFT':
        directory="TDDFT"
        blocks_settings += TDDFT
    
    orcablocks = get_orcablocks(blocks_settings, number_of_cpus=cpus, memory_per_cpu=mem)

    # Use lowest multiplicity by default
    if multiplicity == "lowest":
        mult = sum(atoms.get_atomic_numbers()) % 2 + 1
    else:
        mult = multiplicity

    calc = get_orca_calculator(orcasimpleinput=orcasimpleinput,
                            orcablocks=orcablocks,
                            version=ORCA_VERSION,
                            hpc=ORCA_HPC,
                            charge=0,
                            mult=mult,
                            directory=directory)
    
    return calc

# Set up a geometry optimisation
from ase.build import molecule
from ase.constraints import FixBondLength

atoms = molecule("H2O")

os.mkdir("DFT")
os.chdir("DFT")
# DFT optimisation
calc = get_default_orca_calculator(atoms, flavour='DFT', cpus=number_of_cpus, mem=memory_per_cpu)
atoms = optimize_atoms(atoms, calc, restart=False)

e = atoms.get_potential_energy()
print(f"GS O2 E: {e} eV")

os.chdir("..")
os.mkdir("TDDFT")
os.chdir("TDDFT")
# TDDFT optimisation
# set constraint
atoms.set_constraint(FixBondLength(0, 1))
calc = get_default_orca_calculator(atoms, flavour='TDDFT', cpus=number_of_cpus, mem=memory_per_cpu)
atoms.calc = None
atoms = optimize_atoms(atoms, calc, restart=False)

print(f"Final IROOT E is {atoms.get_potential_energy()} eV")
os.chdir("..")
