from acat.build.ordering import SymmetricClusterOrderingGenerator as SCOG
from ase.cluster import Octahedron
from ase.io import read
from ase.visualize import view

metals = ['Au', 'Cu']

# Initialize a pure Ni truncated octahedral nanoparticle 
# as the template
atoms = Octahedron('Au', 5, 2) # 55 Octahedron

# Using currently used Au5 nanoparticle as the template 
atoms = read(r"<repo-root>\.data\Au63.xyz")

# Now generate 50 symmetric Au3Cu nanoalloys with 
# the composition of 3:1 and save in a trajectory 
# file 'orderings.traj'
scog = SCOG(atoms, elements=['Au', 'Cu'], 
            symmetry='mirror_circular',
            composition={'Au': 3,'Cu': 1},
            trajectory='orderings_Au3Cu_63.traj')
scog.run(max_gen=20000, mode='systematic', eps=0.005)

# results in 6530 structures
# Visualize the output structures. 
images = read('orderings_Au3Cu_63.traj', index=':')
view(images)

# Calculate the energy of each structure and save in a database
# for image in images:
from mace.calculators import mace_mp
from ase.optimize import BFGS

PARAMS = {'model': "medium",
          'dispersion': True,
          'default_dtype': 'float64',
          'device': 'cpu'}

from ase.io.trajectory import Trajectory
traj = Trajectory('orderings_Au3Cu_energy.traj', 'w')

images = zip(range(len(images)), images) 

def evaluate_structure(atoms, index):
    atoms.calc =mace_mp(**PARAMS)
    opt = BFGS(atoms, trajectory=f"{index}.traj", logfile=f"{index}.log")
    opt.run(fmax=0.05)

from multiprocessing import Pool
with Pool(2) as p:
    p.starmap(evaluate_structure, images)
