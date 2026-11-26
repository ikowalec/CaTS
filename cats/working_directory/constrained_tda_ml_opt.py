import os
import shutil
from pathlib import Path

from ase.build import molecule
from ase.constraints import FixAtoms, FixBondLength
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import FIRE2
from catlearn.activelearning.local import LocalAL
from cats.multiwfn.plot_uvvis import plot_charge_transfer_spectrum_from_folders
from cats.multiwfn.plot_uvvis import get_full_ct_spectrum
from cats.multiwfn.iroot_scoring import IRootCandidate, select_iroot_by_score
from cats.multiwfn.tools import charge_transfer_routine
from cats.multiwfn.tools import parse_ifct
from cats.orca.run.orca_calculator import get_orca_calculator
from cats.orca.run.orca_calculator import get_orcablocks
from cats.orca.run.orca_calculator import get_orca_input_header

# Calculator setup
nodes = 2
cpus_per_node = 32
number_of_cpus = int(nodes*cpus_per_node) # Explicitly set number of CPUs #TODO: do it dynamically for HPCs
memory_per_cpu = int(1300*128/cpus_per_node) # memory in Mb # TODO: set max dynamically per HPC, although might affect queueing time
ORCA_HPC = os.environ.get("CATS_ORCA_HPC") or os.environ.get("CPLAS_ORCA_HPC", "falcon")
ORCA_VERSION = os.environ.get("CATS_ORCA_VERSION") or os.environ.get("CPLAS_ORCA_VERSION", "6_1_0")
ROOTS = 35
IROOT = 1
TDDFT_SCAN_ROOTS = 40
TARGET_WAVELENGTH_NM = 520.0
IROOT_SCORE_WEIGHTS = {
    "intensity": 0.3,
    "ct": 0.4,
    "wavelength": 0.3,
    "similarity": 0.1,
}
IROOT_WAVELENGTH_SIGMA_NM = 20.0
IROOT_SIMILARITY_SIGMA_NM = 15.0
IROOT_SIMILARITY_SIGMA_CT = 0.05

# Load initial structure
# atoms = read("orca.out", format='orca-output')
atoms = read("orca.xyz")

def optimize_atoms(atoms, calc, fmax=0.01, active_learn=True, restart=True, previous_calculations=None, local_opt_kwargs={"maxstep":0.005}, traj_name="atoms.traj"):
    directory = calc.directory # retain directory and reinstantiate
    if active_learn:
        # Initialize local optimization
        dyn = LocalAL(atoms=atoms,
                        ase_calc=calc,
                        unc_convergence=0.03,
                        local_opt=FIRE2,
                        local_opt_kwargs=local_opt_kwargs,
                        save_memory=False,
                        use_restart=restart,
                        min_data=3,
                        restart=restart,
                        verbose=True,
                        prev_calculations=previous_calculations,
                        trajectory=f"{directory}/predicted.traj",
                        trainingset=f"{directory}/evaluated.traj",
                        pred_evaluated=f"{directory}/predicted_evaluated.traj",
                        converged_trajectory=f"{directory}/converged.traj",
                        initial_traj=f"{directory}/initial_struc.traj",
                        last_traj=None,
                        tabletxt=f"{directory}/ml_summary.txt",
                        timetxt=f"{directory}/ml_time.txt",
            )
        dyn.run(fmax=fmax,
            max_unc=0.03,
            steps=200,
            ml_steps=1000,
            dtrust=2.0)
        atoms.calc.directory = directory # reset directory
    else:
        atoms.calc = calc
        directory = calc.directory 
        opt = FIRE2(atoms)
        opt.run(fmax=fmax, steps=300)
        atoms = read(f"{calc.directory}/orca.out")
        atoms.calc.directory = directory # reset directory
        trajectory = Trajectory(traj_name, "a")
        trajectory.write(atoms)

    return atoms


def get_last_iroot(path: str | Path) -> int | None:
    last = None
    with open(path, "r", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s.startswith("IROOT"):
                parts = s.split()
                if len(parts) >= 2:
                    try:
                        last = int(parts[1])
                    except ValueError:
                        pass
    return last


def get_default_orca_calculator(
    atoms,
    flavour,
    cpus,
    mem,
    task="engrad",
    multiplicity="lowest",
    constraint=False,
    directory="DFT",
    nroots_override: int | None = None,
):
    
    '''This function sets up an ORCA calculator for either DFT or TDDFT geometry optimization.'''
    

    blocks_settings = []

    ''' Check for existing gbw file for restart '''
    if os.path.exists(f"{directory}/initial.gbw"):
        blocks_settings.append('%MOINP "initial.gbw"')
        restart = 'restart'
    else:
        restart = ''
    
    """
    if "Cu" in atoms.get_chemical_symbols():
        BASIS = ['%BASIS',
                'newGTO 29 "LANL2DZ" end',
                'newECP 29 "HayWadt" end',
                'END'
                ]
        blocks_settings += BASIS
    """
    
 
    GEOM = ['%GEOM']
    
    if constraint:
        O_indices = [atom.index for atom in atoms if atom.symbol == "O"]
        C_indices = [atom.index for atom in atoms if atom.symbol == "C"]
        GEOM.append("Constraints")
        GEOM.append(f'{{ B {C_indices[0]} {O_indices[1]} {atoms.get_distance(C_indices[0], O_indices[1])} C }}')
        Cu_indices = [atom.index for atom in atoms if atom.symbol == "Cu"]
        for Cu in Cu_indices:
            GEOM.append(f'{{ C {Cu} C }}')
        GEOM.append('end')
    
    #GEOM.append('Convergence Normal')
    ev_per_hartree=27.2114
    bohr_per_angstrom=1.88973
    # 0.01 eV/A in Hartree/Bohr
    hartree_per_ev=1/ev_per_hartree
    angstrom_per_bohr=1/bohr_per_angstrom
    tolmaxg = 0.01 * hartree_per_ev * angstrom_per_bohr

    # This is not respected in TDDFT optimizations - see ORCA manual, must be using TightOpt instead
    GEOM.append(f'TolMaxG {tolmaxg:.1e}') # Hartree/Bohr, i.e. below 0.01 eV/A
    GEOM.append('Trust 0.5')
    GEOM.append('END')

    blocks_settings.extend(GEOM)
    
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
    if flavour == 'TDDFT':
        nroots = nroots_override if nroots_override is not None else ROOTS
        TDDFT = ["%TDDFT",
        "MaxIter 999",
        f"NRoots {nroots}",
        "MaxDim 20",
        "ETol 3.6e-4",
        "RTol 3.6e-4",
        "TPrint 1e-8",
        "END"
            ] 
        blocks_settings += TDDFT
    elif flavour == 'TDDFT_opt':
        TDDFT = ["%TDDFT",
        "MaxIter 999",
        f"NRoots {ROOTS}",
        f"IRoot {IROOT}", # select excited state to optimize
        "FollowIRoot TRUE",
        "FIRENTHRESH 0.2 END", # when choosing excited state, allow for 0.2 eV crossing
        # "FIRKEEPFIRSTREF TRUE",
        "MaxDim 20",
        "ETol 3.6e-4",
        "RTol 3.6e-4",
        "TPrint 1e-8",
        "END"
            ]  
        blocks_settings += TDDFT

    restart = ''
    
    orcasimpleinput = get_orca_input_header(xc='cam-b3lyp', # 'CAM-B3LYP' substitute
                                            cosx='RIJCOSX',
                                            basis_set='def2-svp',
                                            integration_grid='DefGrid2',
                                            scf_convergence_criteria='TightSCF',
                                            output_level='NormalPrint',
                                            task=task,
                                            scf_convergence_strategy="SlowConv",
                                            restart=restart
                                            )
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


def copy_initial_gbw(src_dir: str | Path, dest_dir: str | Path, ensure_dir=True, verbose=True):
    '''Copy and ensure formatting of the Path objects'''
    
    src_dir = Path(src_dir) if isinstance(src_dir, str) else src_dir
    dest_dir = Path(dest_dir) if isinstance(dest_dir, str) else dest_dir
    if verbose:
        print(f"Copied orca.gbw from {src_dir} to {dest_dir}")
    if ensure_dir:
        dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_dir / "orca.gbw", dest_dir / "initial.gbw")
    

############### Full workflow ####################
directory = Path("DFT")
calc = get_default_orca_calculator(atoms, 
                                   flavour='DFT', 
                                   cpus=number_of_cpus, 
                                   mem=memory_per_cpu, 
                                   task='Opt',
                                   constraint=False,
                                   directory=directory)
atoms.calc = calc
# atoms = optimize_atoms(atoms, calc, fmax=0.1, active_learn=False, restart=False)

#################################################

# Perform a TDA Scan on the ground state optimized structure
directory = Path("TDDFT")
copy_initial_gbw(atoms.calc.directory, directory)
calc = get_default_orca_calculator(atoms, 
                                   flavour='TDDFT', 
                                   cpus=number_of_cpus, 
                                   mem=memory_per_cpu, 
                                   task='energy',
                                   constraint=False,
                                   directory=directory,
                                   nroots_override=TDDFT_SCAN_ROOTS)
atoms.calc = calc
e_tddft = atoms.get_potential_energy()

def get_ct_info(atoms):

    # Trigger Multiwfn analysis
    """
    # Settings specific to Multiwfn
    export OMP_STACKSIZE=1500M # 1500MB for each thread
    ulimit -s unlimited
    # Set stacksize to unlimited for ORCA
    # ulimit -s unlimited
    
    export output=test.log.$SLURM_JOB_ID

    $ORCADIR/orca_2mkl orca -molden
    """

    os.environ["OMP_STACKSIZE"]="1500M"
    # Require a completed ORCA run before trying to build the molden file.
    if not Path("orca.out").exists():
        raise FileNotFoundError("Missing orca.out; run the TDDFT calculation before CT analysis.")
    if not Path("orca.gbw").exists():
        raise FileNotFoundError("Missing orca.gbw; ORCA run did not produce a GBW file.")

    # Get the orca molden file
    parent_dir = os.getcwd()
    calc_dir = Path(atoms.calc.directory) if atoms.calc is not None else None
    fallback_dirs = []
    if calc_dir is not None:
        fallback_dirs.append(calc_dir)
    # Common TDDFT directory name for this workflow.
    fallback_dirs.append(Path("TDDFT"))
    # Pick the first directory that has ORCA output.
    calc_dir = next((d for d in fallback_dirs if d is not None and (d / "orca.out").exists()), None)
    if calc_dir is None:
        raise RuntimeError("No calculator attached and no TDDFT outputs found; cannot locate ORCA outputs.")
    os.chdir(calc_dir)

    # Ensure the TDDFT single-point has actually produced output.
    if not Path("orca.out").exists():
        atoms.get_potential_energy()
    if not Path("orca.out").exists():
        raise FileNotFoundError(
            f"Missing orca.out in {calc_dir}; ensure the TDDFT calculation completed."
        )
    if not Path("orca.gbw").exists():
        raise FileNotFoundError(
            f"Missing orca.gbw in {calc_dir}; ensure the TDDFT calculation completed."
        )

    os.system(f"$ORCADIR/orca_2mkl orca -molden")
    
    # Analyze the charge transfer spectrum
    fragments = {"1":"1-13", "2":"14, 15, 16"}
    get_full_ct_spectrum("orca.out", 
                         "orca.molden.input", 
                         fragments=fragments)
    
    # states, terms = parse_ifct("IFCTmajor.txt")
    df = parse_ifct("IFCTmajor.txt", wide=True)
    os.chdir(parent_dir)

    # filter the excitations based on intensity:
    df = df[df["f"] > 0.005]
   
    return df

def get_iroot_from_ct(
    atoms,
    *,
    target_wavelength_nm: float,
    previous_choice: IRootCandidate | None = None,
    previous_state: int | None = None,
):
    """Select IROOT using weighted scoring of intensity, CT magnitude, and wavelength proximity."""
    df = get_ct_info(atoms)
    best, scored = select_iroot_by_score(
        df,
        target_wavelength_nm=target_wavelength_nm,
        previous_choice=previous_choice,
        previous_state=previous_state,
        weights=IROOT_SCORE_WEIGHTS,
        wavelength_sigma_nm=IROOT_WAVELENGTH_SIGMA_NM,
        similarity_sigma_nm=IROOT_SIMILARITY_SIGMA_NM,
        similarity_sigma_ct=IROOT_SIMILARITY_SIGMA_CT,
    )
    if best is None:
        fallback = previous_state if previous_state is not None else 1
        return fallback, int(fallback + 5), None, scored
    iroot = int(best.state)
    roots = int(iroot + 5)
    return iroot, roots, best, scored

IROOT, ROOTS, iroot_choice, _ = get_iroot_from_ct(
    atoms, target_wavelength_nm=TARGET_WAVELENGTH_NM
)

""" Now perform constrained TDDFT geometry optimisation on selected IROOT """
# Once ground state established get TDDFT opt
constraints = [FixAtoms(indices=[atom.index for atom in atoms if atom.symbol in ["Cu"]]), 
               FixBondLength(13, 14)]
atoms.set_constraint(constraints)

directory = "TDDFT_opt_iter_19"
copy_initial_gbw(atoms.calc.directory, directory)

calc = get_default_orca_calculator(atoms, 
                                   flavour='TDDFT_opt', 
                                   cpus=number_of_cpus, 
                                   mem=memory_per_cpu, 
                                   task='TightOpt', 
                                   constraint=True,
                                   directory=directory)
atoms.calc = calc
#atoms = optimize_atoms(atoms, calc, fmax=0.1, 
#                                   active_learn=False, 
#                                   restart=False,
#                                   traj_name="atoms_iter_0.traj")
#
# print(f"Final IROOT E is {atoms.get_potential_energy()} eV")
#
# If TDDFT_opt was run, capture CT on the optimized structure for comparison
opt_out = Path(directory) / "orca.out"
if opt_out.exists():
    _, _, opt_choice, _ = get_iroot_from_ct(
        atoms,
        target_wavelength_nm=TARGET_WAVELENGTH_NM,
        previous_choice=iroot_choice,
        previous_state=IROOT,
    )
    previous_choice = opt_choice or iroot_choice
else:
    previous_choice = iroot_choice


#### Bond breaking scan with constrained TDDFT optimizations
atoms_list = [atoms.copy()]

for i in range(20,40):

    # create new directory for iteration
    prev_directory = Path(f"TDDFT_opt_iter_{i-1}")

    tddft_directory = Path(f"TDDFT_scan_iter_{i}")
    opt_directory = Path(f"TDDFT_opt_iter_{i}")

    # Perform a TDDFT single-point on the current structure to extract CT
    copy_initial_gbw(prev_directory, tddft_directory)
    calc = get_default_orca_calculator(
        atoms,
        flavour='TDDFT',
        cpus=number_of_cpus,
        mem=memory_per_cpu,
        task='energy',
        constraint=False,
        directory=tddft_directory,
        nroots_override=TDDFT_SCAN_ROOTS,
    )
    atoms.calc = calc
    e_tddft = atoms.get_potential_energy()

    # Use previous optimization IROOT as similarity anchor
    IROOT_old = get_last_iroot(prev_directory / "orca.out")
    IROOT, ROOTS, iroot_choice, _ = get_iroot_from_ct(
        atoms,
        target_wavelength_nm=TARGET_WAVELENGTH_NM,
        previous_choice=previous_choice,
        previous_state=IROOT_old,
    )
    scan_choice = iroot_choice
    if IROOT_old is not None and IROOT != IROOT_old:
        print(f"Warning: IROOT changed from {IROOT_old} to {IROOT} on iteration {i}.")

    print(f"Iteration {i}: Optimizing for IROOT = {IROOT} with ROOTS = {ROOTS}")

    # Extend the bond length by increment
    atoms.set_constraint(None)  # Remove previous constraints
    O_indices = [atom.index for atom in atoms if atom.symbol == "O"]
    atoms.set_distance(
        O_indices[0],
        O_indices[1],
        atoms.get_distance(O_indices[0], O_indices[1]) + 0.1,
        fix=0.5,
    )

    # reapply constraints
    constraints = [FixAtoms(indices=[atom.index for atom in atoms if atom.symbol in ["Cu"]]), 
                   FixBondLength(13, 15)]
    atoms.set_constraint(constraints)

    # invoke constrained TDDFT optimization with selected IROOT
    copy_initial_gbw(tddft_directory, opt_directory)
    calc = get_default_orca_calculator(
        atoms,
        flavour='TDDFT_opt',
        cpus=number_of_cpus,
        mem=memory_per_cpu,
        task='TightOpt',
        constraint=True,
        directory=opt_directory,
    )
    atoms.calc = calc
    atoms = optimize_atoms(
        atoms,
        calc,
        fmax=0.1,
        active_learn=False,
        restart=False,
        traj_name=f"atoms_iter_{i}.traj",
    )
    # Capture CT on the optimized structure for comparison in the next scan
    opt_out = opt_directory / "orca.out"
    if opt_out.exists():
        _, _, opt_choice, _ = get_iroot_from_ct(
            atoms,
            target_wavelength_nm=TARGET_WAVELENGTH_NM,
            previous_choice=scan_choice,
            previous_state=IROOT,
        )
        previous_choice = opt_choice or scan_choice
    else:
        previous_choice = scan_choice
    atoms_list.append(atoms.copy())

with Trajectory("constrained_tddft_optimization.traj", "w") as traj:
    for atoms in atoms_list:
        traj.write(atoms)
