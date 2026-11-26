# Authors: Igor Kowalec, Oliver Morris

from ase.io import read
from ase.visualize import view
from carmm.analyse.neighbours import neighbours
import numpy as np
from ase import Atom
from scipy.spatial import distance_matrix


def get_o_si_pairs(qm):
    qm = qm.copy()
    o_si_pairs = []
    for atom in qm:
        if atom.symbol == 'O':
            nn = neighbours(qm, [atom.index], shell=1)[0]
            nn = [n for n in nn if n != atom.index]  # remove self from neighbours

            if len(nn) == 1:
                o_si_pairs.append((atom.index, nn[0]))
    return o_si_pairs


def place_hcaps(atoms, o_si_pairs, cluster=None):
    atoms = atoms.copy()  # Create a copy of the original atoms object to avoid modifying it directly

    # Check if indices exceed local QM atom count (e.g. cluster indices were passed)
    has_cluster_indices = any(max(a, b) >= len(atoms) for a, b in o_si_pairs)
    if has_cluster_indices:
        if cluster is not None:
            # Auto-map global cluster indices to local QM indices
            cluster_indices = identify_indices(atoms, cluster)
            cluster_to_qm = {c_idx: qm_idx for qm_idx, c_idx in enumerate(cluster_indices)}
            o_si_pairs = [
                (cluster_to_qm[a], cluster_to_qm[b])
                for a, b in o_si_pairs
                if a in cluster_to_qm and b in cluster_to_qm
            ]
        else:
            invalid_pair = next((a, b) for a, b in o_si_pairs if max(a, b) >= len(atoms))
            raise IndexError(
                f"Pair indices {invalid_pair} exceed local atoms length ({len(atoms)}). "
                f"If these are global cluster indices, pass cluster=cluster to place_hcaps."
            )

    for idx_A, idx_B in o_si_pairs:
        # Extract their positions as numpy arrays
        pos_A = atoms[idx_A].position
        pos_B = atoms[idx_B].position

        # Calculate the vector from A to B
        vec_AB = pos_B - pos_A

        # Normalize the vector to get the direction (unit vector)
        unit_vec = vec_AB / np.linalg.norm(vec_AB)

        # Calculate the position of Atom C (0.95 Å away from A)
        distance = 0.95
        pos_C = pos_A - (distance * unit_vec)

        # Create Atom C (Hydrogen atom) and append it to the system
        atom_C = Atom('H', position=pos_C)
        atoms.append(atom_C)

    return atoms


def identify_hcap_clashes(qm, o_si_pairs):
    atoms = qm.copy()
    atoms = place_hcaps(atoms, o_si_pairs)
    
    # Track where the newly added H atoms start
    h_start_idx = len(qm)
    
    clashing_o_indices = []
    # Match each H atom index to its parent O-Si pair
    for i in range(len(o_si_pairs)):
        h1_idx = h_start_idx + i
        for j in range(i + 1, len(o_si_pairs)):
            h2_idx = h_start_idx + j
            
            # If H caps clash, grab their parent Oxygen atoms from the original QM region
            if atoms.get_distance(h1_idx, h2_idx) < 2.0:  # Assuming a clash threshold of 2 Å
                o1_idx = o_si_pairs[i][0]
                o2_idx = o_si_pairs[j][0]
                clashing_o_indices.append((o1_idx, o2_idx))
                
    return clashing_o_indices


def identify_Si_next_to_clash(qm, cluster, unique_clashes):
    if not unique_clashes:
        return []
    
    # Map QM atom indices to Cluster atom indices
    qm_to_cluster_map = identify_indices(qm, cluster)
    qm_cluster_set = set(qm_to_cluster_map)
    
    # Get the QM indices of the clashing boundary Oxygens
    o1_qm_idx, o2_qm_idx = unique_clashes[0]
    
    # Translate to Cluster indices
    o1_cluster_idx = qm_to_cluster_map[o1_qm_idx]
    o2_cluster_idx = qm_to_cluster_map[o2_qm_idx]
    
    nearest_Si = []
    # Find the T-site (Si or Al) atoms connected to these Oxygens in the full cluster
    # that are NOT yet in the QM region
    for o_idx in [o1_cluster_idx, o2_cluster_idx]:
        nn = neighbours(cluster, [o_idx], shell=1)[0]
        for n in nn:
            if n not in qm_cluster_set and cluster[n].symbol in ('Si', 'Al'):
                nearest_Si.append(n)
                
    return nearest_Si


def identify_Si_neighbours(cluster, nearest_Si):
    cluster = cluster.copy()
    Si_neighbours = []
    for Si_index in nearest_Si:
        nn = neighbours(cluster, [Si_index], shell=1)[0]
        nn = [n for n in nn if n != Si_index]  # remove self from neighbours
        Si_neighbours.append((Si_index, nn))
    return Si_neighbours


def identify_indices(qm, cluster, tolerance=0.01):
    qm = qm.copy()
    cluster = cluster.copy()
    # Get all positions as numpy arrays
    qm_pos = qm.get_positions()
    cluster_pos = cluster.get_positions()
    
    # Calculate all distances at once (vectorised)
    dist_matrix = distance_matrix(qm_pos, cluster_pos)
    
    # Find the closest cluster index for each QM atom
    cluster_indices = np.argmin(dist_matrix, axis=1)
    
    # Safety Check: Ensure the closest atom is actually a match, not just "nearest"
    min_distances = np.min(dist_matrix, axis=1)
    if np.any(min_distances > tolerance):
        bad_atoms = np.where(min_distances > tolerance)[0]
        raise ValueError(f"QM atom indices {bad_atoms} did not match any cluster atom within tolerance!")
        
    # Convert numpy int64 elements to standard Python integers
    return [int(idx) for idx in cluster_indices]


def expand_qm_to_avoid_clash(qm, cluster, Si_neighbours):
    qm = qm.copy()
    cluster = cluster.copy()

    new_qm_indices = set(identify_indices(qm, cluster))  # Start with the original QM indices
    for Si_index, neighbours_list in Si_neighbours:
        new_qm_indices.add(Si_index)
        for neighbour in neighbours_list:
            new_qm_indices.add(neighbour)  # Add the neighbour indices to the set

    return cluster[sorted(new_qm_indices)]  # Return a new Atoms object with deterministically ordered expanded QM region


def expand_until_no_clashes(qm, cluster):
    qm = qm.copy()
    cluster = cluster.copy()
    
    o_si_pairs = get_o_si_pairs(qm)
    unique_clashes = identify_hcap_clashes(qm, o_si_pairs)
    
    # Stop condition: if no clashes are found, return the QM region as-is
    if not unique_clashes:
        return qm
        
    nearest_Si = identify_Si_next_to_clash(qm, cluster, unique_clashes)
    if not nearest_Si:
        print("Warning: Clashes detected but no unincluded neighboring Si/Al found in cluster.")
        return qm

    Si_neighbours = identify_Si_neighbours(cluster, nearest_Si)
    expanded_qm = expand_qm_to_avoid_clash(qm, cluster, Si_neighbours)

    # view(expanded_qm)  # Visualize the expanded QM region

    return expanded_qm


if __name__ == '__main__':
    # EDIT HERE TO MATCH YOUR STRUCTURES
    # Read structures
    qm = read('.data/zeolites_260821/pristine_qm.xyz')
    cluster = read('.data/zeolites_260821/corrected.xyz')

    assert (qm[0].position == cluster[0].position).all(), "The first atom in both structures should be the same (usually the origin)."

    # Set variables
    expanded_qm = qm.copy()
    previous_size = -1
    new_size = len(expanded_qm)

    # Keep expanding until there are no clashes left
    while new_size != previous_size:
        previous_size = new_size
        expanded_qm = expand_until_no_clashes(expanded_qm, cluster)
        new_size = len(expanded_qm)
        print("Expanded QM size:", new_size)
        # view(expanded_qm)

    expanded_qm_indices = identify_indices(expanded_qm, cluster)
    print(f"Expanded QM indices 0-indexed: {[int(index) for index in expanded_qm_indices]}")
    print(f"Expanded QM indices 1-indexed: {[int(index) + 1 for index in expanded_qm_indices]}")

    # view(cluster[expanded_qm_indices])

    cluster[expanded_qm_indices].write('.data/zeolites_260821/expanded_qm.xyz')
