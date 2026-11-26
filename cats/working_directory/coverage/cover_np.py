from cover_np_functions import get_bridge_sites
from cover_np_functions import get_unique_sites
from cover_np_functions import preoptimize
from cover_np_functions import find_point_C
from ase import Atom
from ase.io import read
from cover_np_functions import get_outer_shell_indices
from ase.visualize import view

atoms = read(
    r"<research-data-root>\2024\10.October\orca.xyz"
)
outer_indices = get_outer_shell_indices(atoms)

print(f"A total of {len(outer_indices)} surface atoms identified.")

# view([atoms, atoms[outer_indices]])

bridge_sites = get_bridge_sites(atoms)
coordinates_list, site_names = bridge_sites

from carmm.analyse.neighbours import neighbours
from ase.neighborlist import natural_cutoffs

no_subsurface_coordinates = []
no_subsurface_site_names = []

for _, coordinates in enumerate(coordinates_list):
    temp = atoms.copy() + Atom("H", position=coordinates)
    # Choose a cutoff to determine max bond distance
    cutoff = natural_cutoffs(temp, mult=1.15)
    nn, shell_list = neighbours(temp, centre=[-1], shell=1, cutoff=cutoff)

    if len(nn) < 6:
        no_subsurface_coordinates.append(coordinates)
        no_subsurface_site_names.append(site_names[_])

unique_sites, unique_site_names = get_unique_sites(
    atoms,
    no_subsurface_coordinates,
    no_subsurface_site_names,
    tolerance=2,
    use_COM=True,
)

# Visualize the bridging sites
show_all_sites = atoms.copy()
show_no_subsurface = atoms.copy()
show_unique_sites = atoms.copy()

for coordinates in coordinates_list:
    show_all_sites += Atom("H", position=coordinates)
for coordinates in no_subsurface_coordinates:
    show_no_subsurface += Atom("H", position=coordinates)
for coordinates in unique_sites:
    show_unique_sites += Atom("H", position=coordinates)


# view([show_all_sites, show_no_subsurface, show_unique_sites])

# Create nanoparticles with adsorbates
COM = atoms.get_center_of_mass()

atoms_list = []
for coordinates in unique_sites:
    # Create new geometry
    new_atoms = atoms.copy() + Atom("O", position=find_point_C(coordinates, COM, 1.5))
    atoms_list.append(preoptimize(new_atoms, fix_atoms=["Au"]))

view(atoms_list)

"""
# structures_list = add_CO_adsorbate(atoms, unique_sites)
outer_indices = get_outer_shell_indices(atoms)
unique_sites, unique_site_names = get_unique_sites(atoms, atoms[outer_indices].positions, outer_indices)
structures_list = add_CO_adsorbate(atoms, unique_site_names)
view(structures_list)
"""
