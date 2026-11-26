#!/usr/bin/env python3
"""
Au nanocluster geometry optimisation with ONETEP via ASE 3.26+

- External calculator: ONETEP (linear-scaling DFT)
- Driven by ASE BFGSLineSearch
- Functional: PBE0 (placeholder; replace ONLY if your ONETEP really has CAM-B3LYP)
"""

from ase.cluster import Octahedron
from ase.optimize import BFGSLineSearch
from ase.calculators.onetep import Onetep, OnetepProfile

# -----------------------
# Build Au nanocluster
# -----------------------
# Example: 55-atom cuboctahedral-like Au cluster
# (change size/shells to taste)
au_cluster = Octahedron('Au', 3, 1)  # 3 shells, c.f. ASE nanocluster docs
au_cluster.center(vacuum=10.0)       # ~10 Å vacuum padding around cluster

# -----------------------
# ONETEP profile
# -----------------------
# command: ONETEP launch command WITHOUT input file name
#         (adapt to your cluster: mpirun, srun, etc.)
# pseudo_path: directory containing your Au pseudopotential(s)
profile = OnetepProfile(
    command="srun --mpi=pmix_v4 onetep.X",  # EXAMPLE for an HPC system
    pseudo_path="/path/to/onetep/pseudos"   # directory with Au .recpot/.usp
)

# -----------------------
# ONETEP parameters
# -----------------------

# ONETEP-level keywords (written verbatim into .dat input)
# Units follow ONETEP docs: eV/bohr/Å, etc.
keywords = {
    # We let ASE handle geometry optimisation, so this remains a single-point task
    "task": "SinglePoint",

    # Output / SCF
    "output_detail": "verbose",
    "elec_energy_tol": 1.0e-6,         # tighten if needed

    # Metallic cluster: modest electronic smearing (EDFT or FOE docs)
    "kbt": "0.1 eV",

    # Properties – useful for later analysis (DOS, etc.)
    "do_properties": True,

    # Real-space cutoffs (psinc and kernel). Values in the ballpark of
    # what is used for Au nanoparticles in ONETEP documentation and masterclasses.
    "cutoff_energy": "900 eV",
    "kernel_cutoff": "900 bohr",
}

# NGWF setup and pseudopotentials for Au.
# Values chosen to be in line with recent ONETEP usage on Au systems
# (e.g. ~9 NGWFs per Au and radii ~10–13 a0). You should still converge
# w.r.t. these for your specific system.
ngwfs_count = {"Au": 9}
ngwfs_radius = {"Au": 10.0}   # bohr; increase if you find basis-set errors

# Example pseudopotential name; must actually exist in pseudo_path.
# Preferably a single Au potential in that directory so ASE’s guessing
# doesn’t get confused.
pseudopotentials = {"Au": "Au_PBE_OTF.usp"}

# -----------------------
# Create ASE–ONETEP calculator
# -----------------------

calc = Onetep(
    profile=profile,
    directory="onetep_au_cluster",  # working directory for this calculation
    append=True,                    # do not overwrite .out on restart

    # XC functional:
    # ONETEP public docs support HF, B3LYP, PBE0 etc., NOT CAM-B3LYP.
    # If your local build truly has CAM-B3LYP exposed via LibXC, then:
    #   1) Find the exact XC_FUNCTIONAL name in your keyword database.
    #   2) Put that exact string here instead of "PBE0".
    xc="PBE0",

    # NGWF settings
    ngwfs_count=ngwfs_count,
    ngwfs_radius=ngwfs_radius,

    # Pseudopotentials
    pseudopotentials=pseudopotentials,

    # Extra ONETEP keywords
    keywords=keywords,
)

au_cluster.calc = calc

# -----------------------
# Geometry optimisation (ASE side)
# -----------------------

# Optimiser runs in Python, ONETEP just supplies energies/forces.
# fmax is in eV/Å.
opt = BFGSLineSearch(au_cluster, logfile="geom_opt.log")
opt.run(fmax=0.05)

# After convergence, you can query final energy, save geometry, etc.
final_energy = au_cluster.get_potential_energy()
print(f"Final optimised energy: {final_energy:.6f} eV")

# Example: write final structure to a file
from ase.io import write
write("au_cluster_optimised.xyz", au_cluster)
