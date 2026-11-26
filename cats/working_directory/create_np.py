from ase.cluster import Octahedron
atoms = Octahedron('Cu', 5, 2)

from ase.visualize import view
view(atoms)