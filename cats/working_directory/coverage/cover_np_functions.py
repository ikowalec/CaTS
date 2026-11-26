from ase import Atom
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.calculators.emt import EMT
from tqdm import tqdm
import numpy as np
from mace.calculators import mace_mp
from ase.optimize import BFGS
from ase.constraints import FixAtoms


mace_calculator = mace_mp(size="small")
e_np = -202672.276817
DISTANCE = 2.74 * 0.9  # 90% of sum of radii of Cu and Au


def find_point_C(A, B, DISTANCE):
    # Convert to numpy arrays
    A = np.array(A)
    B = np.array(B)

    # Distance between A and B
    AB_dist = np.linalg.norm(A - B)

    # Solve for C
    # Distance between C and B is "distance"
    # Distance between A and C is AB_dist + "distance"

    # Direction vector from B to A
    direction = (A - B) / AB_dist

    # Calculate point C
    C = B + direction * (AB_dist + DISTANCE)

    return C


def preoptimize(atoms, fix_atoms=["Au"]):
    """Preoptimize structure using the MACE-MP forcefield
    Paramerets:

    atoms: Atoms object
    fix_atoms: list of chemical symbols to constrain during optimisation"""

    if fix_atoms:
        atoms.set_constraint(
            [FixAtoms([atom.index for atom in atoms if atom.symbol in fix_atoms])]
        )

    atoms.calc = mace_calculator
    # Pre-optimize
    opt = BFGS(atoms)
    opt.run(fmax=0.05)

    # Remove force-field information and constraints
    atoms.calc = None
    atoms.set_constraint()

    return atoms


def get_outer_shell_indices(atoms: Atoms):
    """
    Parameters

    atoms: Atoms object

    Returns:
        indices of outer shell atoms
    """

    outer_indices = []
    COM = atoms.get_center_of_mass()
    no_atoms = len(atoms)

    # Loop over all atoms
    for i in range(len(atoms)):
        # Place a Cu atom on an Au atom in the direction away from centre-of-mass (COM)
        C = find_point_C(atoms[i].position, COM, DISTANCE)
        atoms_new = atoms.copy() + Atom("Cu", position=C)

        # Find distance matrix between Cu atom and all other Au atoms
        distance_matrix = atoms_new.get_distances(
            no_atoms, [atom.index for atom in atoms_new if atom.symbol == "Au"]
        )
        distance_matrix = sorted(distance_matrix)

        # Check if Cu nearest-neighbor-distance (NND) is around "distance"
        # If yes - expect the atom to be on the surface
        if (round(distance_matrix[0], 4) - DISTANCE) ** 2 < 0.5:
            outer_indices.append(i)

    return outer_indices


def get_unique_sites(atoms, coordinates_list, site_names, tolerance: int, use_COM=True):
    """
    Takes specified coordinates, places a Cu atom at DISTANCE away, and checks energies of the resulting structures.
    Then, a list of unique coordinates is returned.

    Parameters:

    atoms: Atoms
    coordinates: list of floats
    notations: list of integers

    returns: list[list[float, float, float]]"""
    COM = atoms.get_center_of_mass()
    unique_sites = []
    unique_site_names = []
    explored_energies = []

    dataset = zip(site_names, coordinates_list)

    # Loop over all atoms
    for notation, xyz in tqdm(dataset):
        # Place a Cu atom on an Au atom in the direction away from centre-of-mass (COM)
        if use_COM:
            C = find_point_C(xyz, COM, DISTANCE)
        else:
            C = xyz  # or use the supplied coordinate

        atoms_new = atoms.copy() + Atom("Cu", position=C)
        atoms_new.calc = mace_calculator
        e = round(atoms_new.get_potential_energy(), tolerance)

        # Filter based on unique energies of the prepared structures to obtain unique surface atoms
        if e not in explored_energies:
            explored_energies.append(e)
            unique_sites.append(xyz)
            unique_site_names.append(notation)

    print(explored_energies)

    return unique_sites, unique_site_names


def add_CO_adsorbate(atoms: Atoms, coordinates_list: list[float]):
    """
    Parameters

    atoms: Atoms object
        Nanoparticle geometry.
    indices: [int, int, ...]
        List of atomic indices where the CO adsorbate should be placed

    returns:
        list[Atoms]
    """
    COM = atoms.get_center_of_mass()
    structures_list = []
    for coordinates in coordinates_list:
        # put C 2 Angstrom away from Au, then O 1.12 away from Cu
        coordinate_C = find_point_C(coordinates, COM, 2)
        coordinate_O = find_point_C(coordinate_C, COM, 1.12)
        atoms_new = (
            atoms.copy()
            + Atom("O", position=coordinate_O)
            + Atom("C", position=coordinate_C)
        )
        structures_list.append(atoms_new)

    return structures_list


def get_bridge_sites(atoms):
    import numpy as np

    outer_indices = get_outer_shell_indices(atoms)

    # atoms_new = atoms.copy()[outer_indices]

    # get distance between all atoms excluding itself
    distances = [
        np.linalg.norm(atoms[outer_indices[0]].position - atoms[atom_index].position)
        for atom_index in outer_indices
        if atom_index != outer_indices[0]
    ]
    lowest_distance = np.amin(distances)

    # get all distances, filter close to lowest, save name as the two indices
    accepted_sites = []

    for current_atom in outer_indices:

        distances = [
            np.linalg.norm(atoms[current_atom].position - atoms[atom_index].position)
            for atom_index in outer_indices
            if atom_index != outer_indices[0]
        ]

        names = [
            (current_atom, atom_index)
            for atom_index in outer_indices
            if atom_index != outer_indices[0]
        ]

        dataset = zip(distances, names)

        for distance, name in dataset:
            if (distance - lowest_distance) ** 2 < 0.5 * lowest_distance:
                accepted_sites.append((distance, name))

    # is this correct? Their environment should be more important, no?
    """
    # then filter sites based on unique distances
    accepted_sites_filtered = []
    for site in tqdm(accepted_sites):
        # check if distance is already present in the list of sites
        if not accepted_sites_filtered:
            accepted_sites_filtered.append(site)
        elif not site[0] in list(zip(*accepted_sites_filtered))[0]:
            accepted_sites_filtered.append(site)
    
    site_coordinates = []
    for site in accepted_sites_filtered:
        # save site coordinate and notation
        site_coordinates.append(((atoms[site[1][0]].position + atoms[site[1][1]].position)/2, site[1]))
    """

    site_coordinates = []
    site_names = []

    for distance, site in accepted_sites:
        # save site coordinate and notation
        coordinate = (atoms[site[0]].position + atoms[site[1]].position) / 2
        if not any(
            np.array_equal(coordinate, array) for array in site_coordinates
        ):  # any(np.array_equal(array_to_check, arr) for arr in arrays)
            site_coordinates.append(coordinate)
            site_names.append(f"{site[0]}-{site[1]}_{distance}")

    # filtered_sites = zip(site_coordinates, site_names)

    return site_coordinates, site_names
